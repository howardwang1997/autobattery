"""Physics-Informed Battery Foundation Model (Hybrid).

Architecture: Transformer backbone (same as vanilla) + Physics auxiliary heads.
The model predicts V(t) directly for accuracy, but also predicts a structured
decomposition (OCV, eta_act, eta_ohm, eta_conc) as auxiliary outputs.

During training, physics-informed losses encourage:
1. Monotonicity: V(t) should decrease during discharge
2. C-rate equivariance: overpotentials should scale correctly with current
3. Decomposition consistency: V ≈ OCV - eta_act - eta_ohm - eta_conc
4. Sign constraints: overpotentials should be non-negative
5. Parameter-group sensitivity: eta_act mostly depends on kinetics params, etc.

This hybrid approach ensures:
- Full accuracy (Transformer backbone matches vanilla)
- Physical interpretability (decomposition is learned alongside)
- Differentiable physics (can extract degradation signatures)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierBasis(nn.Module):
    def __init__(self, n_basis=32, n_time=200):
        super().__init__()
        t = torch.linspace(0, 1, n_time)
        parts = [torch.ones(n_time, 1)]
        for k in range(1, n_basis // 2 + 1):
            parts.append(torch.sin(2 * math.pi * k * t).unsqueeze(1))
            parts.append(torch.cos(2 * math.pi * k * t).unsqueeze(1))
        basis = torch.cat(parts, dim=1)[:, :n_basis]
        self.register_buffer("basis", basis)

    def forward(self, coeffs):
        return coeffs @ self.basis.T


class PhysicsInformedBatteryModel(nn.Module):
    """Hybrid physics-informed battery foundation model.

    Main output: V_pred(t) — direct voltage prediction (high accuracy)
    Auxiliary outputs: OCV, eta_act, eta_ohm, eta_conc (interpretability)

    The decomposition is encouraged but not enforced, so the model can
    achieve vanilla-level accuracy while still learning interpretable structure.
    """

    def __init__(
        self,
        n_chemistries=6,
        n_params=8,
        n_time=200,
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        dropout=0.1,
        n_decomp_basis=32,
    ):
        super().__init__()
        self.n_time = n_time
        self.d_model = d_model

        # === Main Transformer backbone (same as vanilla) ===
        self.chem_embed = nn.Embedding(n_chemistries, d_model)
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        prefix_dim = d_model * 3
        self.prefix_proj = nn.Sequential(
            nn.Linear(prefix_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cond_film = FiLMLayer(d_model, d_model)

        # === Main voltage output head ===
        self.voltage_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, 1),
        )

        # === Physics decomposition heads (auxiliary) ===
        self.ocv_head = FourierBasis(n_decomp_basis, n_time)
        self.ocv_coeff_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_decomp_basis),
        )

        self.eta_act_coeff_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_decomp_basis),
        )
        self.eta_ohm_coeff_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, max(n_decomp_basis // 2, 8)),
        )
        self.eta_conc_coeff_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_decomp_basis),
        )

        self.decomp_fourier_act = FourierBasis(n_decomp_basis, n_time)
        self.decomp_fourier_ohm = FourierBasis(max(n_decomp_basis // 2, 8), n_time)
        self.decomp_fourier_conc = FourierBasis(n_decomp_basis, n_time)

        # Aggregate token for decomposition
        self.decomp_pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _run_transformer(self, chem_ids, params, conditions, targets=None):
        B = chem_ids.shape[0]
        device = chem_ids.device

        z_chem = self.chem_embed(chem_ids)
        z_param = self.param_encoder(params)
        z_cond = self.cond_encoder(conditions)

        prefix = self.prefix_proj(torch.cat([z_chem, z_param, z_cond], dim=-1))
        prefix = prefix.unsqueeze(1)

        t_norm = torch.linspace(0, 1, self.n_time, device=device)
        z_time = self.time_embed(t_norm.unsqueeze(0).expand(B, -1).unsqueeze(-1))

        if targets is not None:
            v_tokens = torch.zeros(B, self.n_time, self.d_model, device=device)
            v_tokens = v_tokens + z_time
        else:
            v_tokens = torch.zeros(B, self.n_time, self.d_model, device=device)
            v_tokens = v_tokens + z_time

        z_cond_exp = z_cond.unsqueeze(1).expand(-1, self.n_time, -1)
        tokens = self.cond_film(v_tokens, z_cond_exp)

        seq = torch.cat([prefix, tokens], dim=1)
        n_total = 1 + self.n_time
        causal_mask = torch.triu(
            torch.ones(n_total, n_total, device=device), diagonal=1
        ).bool()

        out = self.transformer(seq, mask=causal_mask)
        return out[:, 1:, :], z_param

    def forward(self, chem_ids, params, conditions, targets=None):
        token_out, z_param = self._run_transformer(
            chem_ids, params, conditions, targets
        )

        # Main voltage prediction
        V_pred = self.voltage_head(token_out).squeeze(-1)

        # Decomposition from aggregated token representation
        pooled = token_out.mean(dim=1)

        ocv_coeffs = self.ocv_coeff_net(pooled)
        ocv = torch.sigmoid(self.ocv_head(ocv_coeffs))

        act_coeffs = self.eta_act_coeff_net(pooled)
        eta_act = torch.relu(self.decomp_fourier_act(act_coeffs))

        ohm_coeffs = self.eta_ohm_coeff_net(pooled)
        eta_ohm = torch.relu(self.decomp_fourier_ohm(ohm_coeffs))

        conc_coeffs = self.eta_conc_coeff_net(pooled)
        eta_conc = torch.relu(self.decomp_fourier_conc(conc_coeffs))

        components = {
            "ocv": ocv,
            "eta_activation": eta_act,
            "eta_ohmic": eta_ohm,
            "eta_concentration": eta_conc,
        }
        return V_pred, components

    @torch.no_grad()
    def generate(self, chem_ids, params, conditions):
        V_pred, components = self.forward(chem_ids, params, conditions)
        return V_pred, components


class FiLMLayer(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        return self.gamma(cond) * x + self.beta(cond)


class PhysicsInformedBatteryModelSmall(PhysicsInformedBatteryModel):
    def __init__(self, **kwargs):
        defaults = dict(d_model=128, n_heads=4, n_layers=4, d_ff=512, n_decomp_basis=24)
        defaults.update(kwargs)
        super().__init__(**defaults)


class PhysicsInformedBatteryModelLarge(PhysicsInformedBatteryModel):
    def __init__(self, **kwargs):
        defaults = dict(d_model=512, n_heads=16, n_layers=12, d_ff=2048, n_decomp_basis=48)
        defaults.update(kwargs)
        super().__init__(**defaults)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, cls in [
        ("Small", PhysicsInformedBatteryModelSmall),
        ("Base", PhysicsInformedBatteryModel),
        ("Large", PhysicsInformedBatteryModelLarge),
    ]:
        m = cls()
        n = count_parameters(m)
        B = 4
        chem = torch.randint(0, 6, (B,))
        params = torch.randn(B, 8)
        cond = torch.randn(B, 2)
        V_pred, comp = m(chem, params, cond)
        print(f"{name:8s}: {n/1e6:.1f}M params, V_pred={V_pred.shape}")
        for k, v in comp.items():
            print(f"  {k}: range=[{v.min().item():.4f}, {v.max().item():.4f}]")
