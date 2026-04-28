"""Battery Foundation Transformer.

Architecture: Causal Transformer decoder that autoregressively generates
voltage curves V(t) conditioned on chemistry type, parameters, and operating conditions.

Input:
  - chemistry_id: int (0-6)
  - params: (N_params,) normalized parameter vector
  - conditions: (C_rate, Temperature)
  
Output:
  - V(t): voltage curve at N_TIME uniform time points
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChemistryEmbedding(nn.Module):
    def __init__(self, n_chemistries=6, dim=64):
        super().__init__()
        self.embed = nn.Embedding(n_chemistries, dim)

    def forward(self, chem_ids):
        return self.embed(chem_ids)


class ParameterEncoder(nn.Module):
    def __init__(self, n_params=8, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, params):
        return self.net(params)


class ConditionEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, conditions):
        return self.net(conditions)


class FiLMLayer(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        return self.gamma(cond) * x + self.beta(cond)


class BatteryTransformer(nn.Module):
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
    ):
        super().__init__()
        self.n_time = n_time
        self.d_model = d_model

        self.chem_embed = ChemistryEmbedding(n_chemistries, d_model)
        self.param_encoder = ParameterEncoder(n_params, d_model)
        self.cond_encoder = ConditionEncoder(d_model)

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.voltage_embed = nn.Linear(1, d_model)

        prefix_dim = d_model * 3
        self.prefix_proj = nn.Sequential(
            nn.Linear(prefix_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.cond_film = FiLMLayer(d_model, d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, chem_ids, params, conditions, targets=None):
        """
        Args:
            chem_ids: (B,) int tensor
            params: (B, N_params) float tensor (normalized)
            conditions: (B, 2) float tensor [C_rate, Temperature]
            targets: (B, N_time) float tensor or None (for inference use V_ref)
        
        Returns:
            V_pred: (B, N_time) predicted voltage curve
        """
        B = chem_ids.shape[0]
        device = chem_ids.device

        z_chem = self.chem_embed(chem_ids)
        z_param = self.param_encoder(params)
        z_cond = self.cond_encoder(conditions)

        prefix = self.prefix_proj(torch.cat([z_chem, z_param, z_cond], dim=-1))
        prefix = prefix.unsqueeze(1)

        t_normalized = torch.linspace(0, 1, self.n_time, device=device).unsqueeze(0).expand(B, -1)
        z_time = self.time_embed(t_normalized.unsqueeze(-1))

        if targets is not None:
            v_tokens = self.voltage_embed(targets.unsqueeze(-1))
        else:
            v_tokens = torch.zeros(B, self.n_time, self.d_model, device=device)

        tokens = v_tokens + z_time

        z_cond_expanded = z_cond.unsqueeze(1).expand(-1, self.n_time, -1)
        z_cond_all = torch.cat([z_cond.unsqueeze(1), z_cond_expanded], dim=1)
        tokens = self.cond_film(tokens, z_cond_expanded)

        seq = torch.cat([prefix, tokens], dim=1)
        n_total = 1 + self.n_time

        causal_mask = torch.triu(
            torch.ones(n_total, n_total, device=device), diagonal=1
        ).bool()

        out = self.transformer(seq, mask=causal_mask)

        v_tokens_out = out[:, 1:, :]
        v_pred = self.output_head(v_tokens_out).squeeze(-1)

        return v_pred

    @torch.no_grad()
    def generate(self, chem_ids, params, conditions):
        return self.forward(chem_ids, params, conditions, targets=None)


class BatteryTransformerSmall(BatteryTransformer):
    def __init__(self, **kwargs):
        defaults = dict(d_model=128, n_heads=4, n_layers=4, d_ff=512)
        defaults.update(kwargs)
        super().__init__(**defaults)


class BatteryTransformerLarge(BatteryTransformer):
    def __init__(self, **kwargs):
        defaults = dict(d_model=512, n_heads=16, n_layers=12, d_ff=2048)
        defaults.update(kwargs)
        super().__init__(**defaults)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, cls in [("Small", BatteryTransformerSmall),
                       ("Base", BatteryTransformer),
                       ("Large", BatteryTransformerLarge)]:
        m = cls()
        n = count_parameters(m)
        B = 4
        chem = torch.randint(0, 6, (B,))
        params = torch.randn(B, 8)
        cond = torch.randn(B, 2)
        targets = torch.randn(B, 200)
        out = m(chem, params, cond, targets)
        gen = m.generate(chem, params, cond)
        print(f"{name:8s}: {n/1e6:.1f}M params, out={out.shape}, gen={gen.shape}")
