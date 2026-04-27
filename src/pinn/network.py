import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class Sine(nn.Module):
    """Sine activation for periodic features (SIREN-style)."""

    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sine": Sine,
}


class FullyConnectedBlock(nn.Module):
    """Fully connected block with optional normalization and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "silu",
        use_norm: bool = False,
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_norm:
            layers.append(nn.LayerNorm(out_features))
        if activation != "none":
            layers.append(ACTIVATIONS[activation]())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiDomainPINN(nn.Module):
    """
    Multi-domain Physics-Informed Neural Network for metal battery P2D model.

    Architecture:
    - Shared encoder processes (t, x, r, params)
    - Three domain heads for negative electrode / separator / positive electrode
    - Global head for terminal voltage and SEI thickness

    The domain encoding is a learnable embedding that tells the network
    which region of the cell a point belongs to.

    Input:
        t: (B, 1) normalized time [0, 1]
        x: (B, 1) normalized through-cell position [0, 1]
        r: (B, 1) normalized radial position in particle [0, 1]
        params: (B, N_params) normalized parameter vector
        domain: (B,) integer domain label (0=neg, 1=sep, 2=pos)

    Output:
        c_e: (B, 1) electrolyte concentration
        phi_s: (B, 1) solid potential (neg/pos domains only)
        phi_e: (B, 1) electrolyte potential
        c_s: (B, 1) solid concentration (pos domain only)
        V: (B, 1) terminal voltage (global)
        L_sei: (B, 1) SEI thickness (global)
    """

    DOMAIN_NEG = 0
    DOMAIN_SEP = 1
    DOMAIN_POS = 2

    def __init__(
        self,
        num_params: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 6,
        activation: str = "silu",
        num_domains: int = 3,
        domain_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_params = num_params
        self.num_domains = num_domains
        self.domain_embed_dim = domain_embed_dim

        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)

        input_dim = 3 + num_params + domain_embed_dim  # t, x, r, params, domain_embed

        layers = []
        layers.append(FullyConnectedBlock(input_dim, hidden_dim, activation))
        for _ in range(num_layers - 1):
            layers.append(FullyConnectedBlock(hidden_dim, hidden_dim, activation))
        self.encoder = nn.Sequential(*layers)

        self.head_neg = nn.Linear(hidden_dim, 3)  # c_e, phi_s, phi_e
        self.head_sep = nn.Linear(hidden_dim, 2)   # c_e, phi_e
        self.head_pos = nn.Linear(hidden_dim, 4)   # c_e, phi_s, phi_e, c_s
        self.head_global = nn.Linear(hidden_dim, 2) # V, L_sei

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        params: torch.Tensor,
        domain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            t: (B, 1) normalized time
            x: (B, 1) normalized spatial position
            r: (B, 1) normalized radial position
            params: (B, N) normalized parameters
            domain: (B,) domain indices {0, 1, 2}

        Returns:
            dict with keys depending on domain:
            - 'neg': c_e, phi_s, phi_e
            - 'sep': c_e, phi_e
            - 'pos': c_e, phi_s, phi_e, c_s
            - 'global': V, L_sei
        """
        domain_emb = self.domain_embedding(domain)
        inp = torch.cat([t, x, r, params, domain_emb], dim=-1)

        features = self.encoder(inp)

        result = {}
        neg_mask = domain == self.DOMAIN_NEG
        sep_mask = domain == self.DOMAIN_SEP
        pos_mask = domain == self.DOMAIN_POS

        if neg_mask.any():
            out = self.head_neg(features[neg_mask])
            result["neg_c_e"] = out[:, 0:1]
            result["neg_phi_s"] = out[:, 1:2]
            result["neg_phi_e"] = out[:, 2:3]

        if sep_mask.any():
            out = self.head_sep(features[sep_mask])
            result["sep_c_e"] = out[:, 0:1]
            result["sep_phi_e"] = out[:, 1:2]

        if pos_mask.any():
            out = self.head_pos(features[pos_mask])
            result["pos_c_e"] = out[:, 0:1]
            result["pos_phi_s"] = out[:, 1:2]
            result["pos_phi_e"] = out[:, 2:3]
            result["pos_c_s"] = out[:, 3:4]

        global_out = self.head_global(features)
        result["V"] = global_out[:, 0:1]
        result["L_sei"] = global_out[:, 1:2]

        return result

    def forward_voltage_only(
        self,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified forward pass for voltage-only prediction.

        Uses a fixed spatial sampling (midpoints of each domain) and
        returns only the terminal voltage.
        """
        B = t.shape[0]
        device = t.device

        n_neg = B // 3
        n_sep = B // 3
        n_pos = B - n_neg - n_sep

        x_neg = torch.full((n_neg, 1), 0.25, device=device)
        x_sep = torch.full((n_sep, 1), 0.5, device=device)
        x_pos = torch.full((n_pos, 1), 0.75, device=device)
        x = torch.cat([x_neg, x_sep, x_pos], dim=0)

        r = torch.full((B, 1), 0.5, device=device)

        domain = torch.cat([
            torch.full((n_neg,), self.DOMAIN_NEG, dtype=torch.long, device=device),
            torch.full((n_sep,), self.DOMAIN_SEP, dtype=torch.long, device=device),
            torch.full((n_pos,), self.DOMAIN_POS, dtype=torch.long, device=device),
        ])

        if params.shape[0] == 1:
            params = params.expand(B, -1)

        result = self.forward(t, x, r, params, domain)
        return result["V"]


class VoltageMLP(nn.Module):
    """
    Simple MLP for voltage prediction: (t, params) -> V(t).

    Used in Phase 0 for pure data fitting baseline.
    Much simpler and faster to train than the full MultiDomainPINN.
    """

    def __init__(
        self,
        num_params: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "silu",
    ):
        super().__init__()
        self.num_params = num_params

        input_dim = 1 + num_params  # t + params
        layers = [FullyConnectedBlock(input_dim, hidden_dim, activation)]
        for _ in range(num_layers - 1):
            layers.append(FullyConnectedBlock(hidden_dim, hidden_dim, activation))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, 1) normalized time
            params: (B, N) normalized parameters
        Returns:
            v: (B, 1) predicted voltage (normalized)
        """
        inp = torch.cat([t, params], dim=-1)
        features = self.encoder(inp)
        return self.head(features)


class VoltagePredictor(nn.Module):
    """
    Two-stage voltage predictor for surrogate modeling.

    Factorizes V(t, params) into:
      V(t, params) = V_shape(t, params) * scale(params) + offset(params)

    - V_shape: normalized voltage curve shape, trained with per-sim normalization
    - offset(params): predicts per-simulation mean voltage
    - scale(params): predicts per-simulation voltage spread
    """

    def __init__(
        self,
        num_params: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "silu",
    ):
        super().__init__()
        self.num_params = num_params

        input_dim = 1 + num_params
        layers = [FullyConnectedBlock(input_dim, hidden_dim, activation)]
        for _ in range(num_layers - 1):
            layers.append(FullyConnectedBlock(hidden_dim, hidden_dim, activation))
        self.shape_encoder = nn.Sequential(*layers)
        self.shape_head = nn.Linear(hidden_dim, 1)

        param_input = num_params
        self.offset_net = nn.Sequential(
            FullyConnectedBlock(param_input, hidden_dim // 2, activation),
            FullyConnectedBlock(hidden_dim // 2, hidden_dim // 4, activation),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.scale_net = nn.Sequential(
            FullyConnectedBlock(param_input, hidden_dim // 2, activation),
            FullyConnectedBlock(hidden_dim // 2, hidden_dim // 4, activation),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_shape(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([t, params], dim=-1)
        return self.shape_head(self.shape_encoder(inp))

    def forward_offset(self, params: torch.Tensor) -> torch.Tensor:
        return self.offset_net(params)

    def forward_scale(self, params: torch.Tensor) -> torch.Tensor:
        return self.scale_net(params)

    def forward(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Full prediction: V(t, params) = shape * scale + offset"""
        v_shape = self.forward_shape(t, params)
        params_unique = params[:, :self.num_params].mean(dim=0, keepdim=True).expand(params.shape[0], -1)
        scale = self.forward_scale(params_unique)
        offset = self.forward_offset(params_unique)
        return v_shape * scale + offset


class InversePINN(nn.Module):
    """
    PINN for inverse problems: identifies physical parameters from data.

    Wraps MultiDomainPINN and makes selected parameters learnable.
    """

    def __init__(
        self,
        base_model: MultiDomainPINN,
        learnable_param_names: list[str],
        init_values: dict[str, float],
        bounds: Optional[dict[str, tuple[float, float]]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.learnable_param_names = learnable_param_names
        self.bounds = bounds or {}

        self.learnable_params_raw = nn.ParameterDict()
        self._param_transforms = {}

        for name in learnable_param_names:
            init_val = init_values.get(name, 1.0)
            lo, hi = self.bounds.get(name, (init_val * 0.01, init_val * 100))

            log_init = torch.tensor(
                self._to_log_space(init_val, lo, hi), dtype=torch.float32
            )
            self.learnable_params_raw[name.replace(".", "_")] = nn.Parameter(
                log_init.unsqueeze(0)
            )
            self._param_transforms[name] = (lo, hi)

    @staticmethod
    def _to_log_space(value: float, lo: float, hi: float) -> float:
        """Map a physical value to [0, 1] via log-space transform."""
        if lo <= 0 or hi <= 0:
            return value
        return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))

    @staticmethod
    def _from_log_space(logit: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """Map [0, 1] back to physical space via log-space transform."""
        return torch.exp(logit * (np.log(hi) - np.log(lo)) + np.log(lo))

    def get_param(self, name: str) -> torch.Tensor:
        """Get the current physical value of a learnable parameter."""
        key = name.replace(".", "_")
        lo, hi = self._param_transforms[name]
        raw = torch.sigmoid(self.learnable_params_raw[key])
        return self._from_log_space(raw, lo, hi)

    def get_all_params(self) -> dict[str, torch.Tensor]:
        """Get all learnable parameters in physical units."""
        return {name: self.get_param(name) for name in self.learnable_param_names}

    def build_param_vector(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Build a normalized parameter vector for the base model input."""
        import numpy as np

        params = []
        for name in self.learnable_param_names:
            val = self.get_param(name)
            params.append(val.expand(batch_size, 1))
        return torch.cat(params, dim=-1)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        domain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass using current learnable parameter values."""
        B = t.shape[0]
        params = self.build_param_vector(B, t.device)
        return self.base_model(t, x, r, params, domain)
