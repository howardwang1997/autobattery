"""Physics-informed PDE residual computation for battery electrolyte equations.

Implements conservation laws from the Doyle-Fuller-Newman (DFN) model:
1. Electrolyte mass conservation: ε ∂c_e/∂t = ∂/∂x(D_e ∂c_e/∂x) + source
2. Electrolyte charge conservation: ∂/∂x(κ ∂φ_e/∂x) + ∂/∂x(κ_D ∂ln c_e/∂x) + j = 0

Focus on separator region where source terms vanish (no electrochemical reactions).
"""

import torch
import torch.nn as nn
import numpy as np


def electrolyte_diffusivity(c_e, T=298.15):
    """D_e(c_e) from Nyman2008. c_e in mol/m³."""
    c = c_e / 1000.0
    D_e = 8.794e-11 * c ** 2 - 3.972e-10 * c + 4.862e-10
    return torch.clamp(D_e, min=1e-12)


def electrolyte_conductivity(c_e, T=298.15):
    """κ(c_e) from Nyman2008. c_e in mol/m³."""
    c = c_e / 1000.0
    kappa = 0.1297 * c ** 3 - 2.51 * c ** 1.5 + 3.329 * c
    return torch.clamp(kappa, min=0.01)


class BatteryPhysics:
    """PDE residual computation on non-uniform PyBaMM mesh."""

    X_NODES = np.array([
        2.13e-06, 6.39e-06, 1.065e-05, 1.491e-05, 1.917e-05, 2.343e-05,
        2.769e-05, 3.195e-05, 3.621e-05, 4.047e-05, 4.473e-05, 4.899e-05,
        5.325e-05, 5.751e-05, 6.177e-05, 6.603e-05, 7.029e-05, 7.455e-05,
        7.881e-05, 8.307e-05, 8.55e-05, 8.61e-05, 8.67e-05, 8.73e-05,
        8.79e-05, 8.85e-05, 8.91e-05, 8.97e-05, 9.03e-05, 9.09e-05,
        9.15e-05, 9.21e-05, 9.27e-05, 9.33e-05, 9.39e-05, 9.45e-05,
        9.51e-05, 9.57e-05, 9.63e-05, 9.69e-05, 9.909e-05, 1.0287e-04,
        1.0665e-04, 1.1043e-04, 1.1421e-04, 1.1799e-04, 1.2177e-04,
        1.2555e-04, 1.2933e-04, 1.3311e-04, 1.3689e-04, 1.4067e-04,
        1.4445e-04, 1.4823e-04, 1.5201e-04, 1.5579e-04, 1.5957e-04,
        1.6335e-04, 1.6713e-04, 1.7091e-04,
    ], dtype=np.float32)

    F_CONST = 96485.33
    R_CONST = 8.31446
    T_REF = 298.15

    NX_NEG = 20
    NX_SEP = 20
    NX_POS = 20
    NX = 60

    EPS_NEG = 0.25
    EPS_SEP = 0.47
    EPS_POS = 0.335

    L_NEG = 8.52e-05
    L_SEP = 1.20e-05
    L_POS = 7.56e-05

    def __init__(self, device="cpu", c_e_ref=1000.0, t_discharge=3600.0):
        self.device = device
        self.c_e_ref = c_e_ref
        self.t_discharge = t_discharge

        x = torch.tensor(self.X_NODES, dtype=torch.float32, device=device)
        self.x = x

        dx = torch.diff(x)
        self.dx = dx

        self.dx_neg = dx[: self.NX_NEG - 1]
        self.dx_sep = dx[self.NX_NEG : self.NX_NEG + self.NX_SEP - 1]
        self.dx_pos = dx[self.NX_NEG + self.NX_SEP :]

        self.eps = torch.cat([
            torch.full((self.NX_NEG,), self.EPS_NEG, device=device),
            torch.full((self.NX_SEP,), self.EPS_SEP, device=device),
            torch.full((self.NX_POS,), self.EPS_POS, device=device),
        ])

    def set_discharge_time(self, c_rate):
        """Set discharge time based on C-rate."""
        self.t_discharge = 3600.0 / max(c_rate, 0.01)

    def d_dt_central(self, f, dt):
        """∂f/∂t via central difference. f: (..., H, W), returns (..., H, W-2)."""
        return (f[..., 2:] - f[..., :-2]) / (2.0 * dt)

    def d_dx_central(self, f, dx):
        """∂f/∂x via central difference on non-uniform grid. f: (..., H, W), dx: (H-1,)."""
        dx_plus = dx[1:]
        dx_minus = dx[:-1]
        denom = dx_plus + dx_minus
        return (dx_minus[..., None] * f[..., 2:, :] + dx_plus[..., None] * f[..., :-2, :]) / (denom[..., None] * (f[..., 2:, :] - f[..., :-2, :]) + 1e-30) * 0

    def d2_dx2_nonuniform(self, f, dx):
        """∂²f/∂x² on non-uniform grid. f: (..., H, W), dx: (H-1,).
        Returns (..., H-2, W)."""
        dx_plus = dx[1:]
        dx_minus = dx[:-1]
        denom = 0.5 * (dx_plus + dx_minus)
        return (f[..., 2:, :] - 2 * f[..., 1:-1, :] + f[..., :-2, :]) / (denom[..., None] ** 2)

    def separator_mass_residual(self, c_e, dt_physical):
        """
        Separator mass conservation: ε_sep ∂c_e/∂t = ∂/∂x(D_e(c_e) ∂c_e/∂x)

        No reaction in separator (j=0). Implemented in divergence form for
        concentration-dependent D_e.

        Args:
            c_e: (B, H, W) electrolyte concentration [mol/m³]
            dt_physical: time step in seconds
        Returns:
            residual: (B, H_sep-2, W-2) dimensionless PDE residual
        """
        c_e_sep = c_e[:, self.NX_NEG : self.NX_NEG + self.NX_SEP, :]
        dx_sep = self.dx_sep

        dc_dt = self.d_dt_central(c_e_sep, dt_physical)

        D_e_face = electrolyte_diffusivity(c_e_sep[:, :-1, :] * 0.5 + c_e_sep[:, 1:, :] * 0.5)
        flux = D_e_face * (c_e_sep[:, 1:, :] - c_e_sep[:, :-1, :]) / dx_sep[..., None]
        div_flux = self._ddx_flux(flux, dx_sep)

        H = min(dc_dt.shape[1], div_flux.shape[1])
        W = min(dc_dt.shape[2], div_flux.shape[2])

        lhs = self.EPS_SEP * dc_dt[:, :H, :W]
        rhs = div_flux[:, :H, :W]

        scale = lhs.abs().mean() + 1e-10
        return (lhs - rhs) / scale

    def separator_charge_residual(self, c_e, phi_e, t_plus, dt_physical):
        """
        Separator charge conservation (algebraic):
        ∂i_e/∂x = 0, where i_e = -κ∂φ_e/∂x + κ_D∂ln(c_e)/∂x

        Residual: ∂/∂x(κ ∂φ_e/∂x) - ∂/∂x(κ_D ∂ln(c_e)/∂x) = 0
        where κ_D = 2RTκ(1-t⁺)/F.

        Implemented in divergence form for variable κ(c_e).
        t⁺ appears linearly — wrong t⁺ gives nonzero residual.

        Returns:
            residual: (B, H_sep-2, W) dimensionless PDE residual
        """
        c_e_sep = c_e[:, self.NX_NEG : self.NX_NEG + self.NX_SEP, :]
        phi_e_sep = phi_e[:, self.NX_NEG : self.NX_NEG + self.NX_SEP, :]

        c_e_face = 0.5 * (c_e_sep[:, :-1, :] + c_e_sep[:, 1:, :])
        kappa_face = electrolyte_conductivity(c_e_face)
        kappa_D_face = (
            2.0 * self.R_CONST * self.T_REF * kappa_face * (1.0 - t_plus[:, None, None])
            / self.F_CONST
        )

        ln_c_e = torch.log(torch.clamp(c_e_sep, min=1.0))

        flux_phi = kappa_face * self._ddx(phi_e_sep, self.dx_sep)
        flux_ln_c = kappa_D_face * self._ddx(ln_c_e, self.dx_sep)

        div_flux_phi = self._ddx_flux(flux_phi, self.dx_sep)
        div_flux_ln_c = self._ddx_flux(flux_ln_c, self.dx_sep)

        residual = div_flux_phi - div_flux_ln_c
        scale = residual.abs().mean() + 1e-10
        return residual / scale

    def _ddx(self, f, dx):
        """∂f/∂x on non-uniform grid. f: (B, H, W), dx: (H-1,). Returns (B, H-1, W)."""
        dx_inv = 1.0 / dx
        return (f[:, 1:, :] - f[:, :-1, :]) * dx_inv[..., None]

    def _ddx_flux(self, flux, dx):
        """∂(flux)/∂x for face-centered flux. flux: (B, H-1, W), dx: (H-1,). Returns (B, H-2, W)."""
        dx_avg = 0.5 * (dx[:-1] + dx[1:])
        return (flux[:, 1:, :] - flux[:, :-1, :]) / dx_avg[..., None]

    def cross_consistency_residual(self, c_e, phi_e, t_plus, dt_physical, region="pos"):
        """
        Cross-consistency of mass and charge conservation (eliminates j):
        
        From mass: a·j = F/(1-t⁺) × [ε ∂c_e/∂t - ∂/∂x(D_e ∂c_e/∂x)]
        From charge: a·j = -∂/∂x(κ ∂φ_e/∂x) - ∂/∂x(κ_D ∂ln(c_e)/∂x)
        
        Eliminate j → R = F/(1-t⁺) × [ε ∂c_e/∂t - div(D_e grad c_e)]
                         + div(κ grad φ_e) + div(κ_D grad ln c_e) = 0
        
        t⁺ appears in both the mass source (1/(1-t⁺)) and κ_D (proportional to 1-t⁺),
        creating strong t⁺ sensitivity.

        Args:
            c_e: (B, H, W) full-domain concentration [mol/m³]
            phi_e: (B, H, W) full-domain potential [V]
            t_plus: (B,) cation transference number
            dt_physical: time step in seconds
            region: 'pos', 'neg', or 'full'
        Returns:
            residual: (B, H'-2, W'-2) dimensionless residual
        """
        if region == "pos":
            c_e_r = c_e[:, self.NX_NEG + self.NX_SEP :, :]
            phi_e_r = phi_e[:, self.NX_NEG + self.NX_SEP :, :]
            dx_r = self.dx_pos
            eps = self.EPS_POS
        elif region == "neg":
            c_e_r = c_e[:, : self.NX_NEG, :]
            phi_e_r = phi_e[:, : self.NX_NEG, :]
            dx_r = self.dx_neg
            eps = self.EPS_NEG
        else:
            c_e_r = c_e
            phi_e_r = phi_e
            dx_r = self.dx
            eps = self.eps

        # Mass equation: ε ∂c_e/∂t - ∂/∂x(D_e ∂c_e/∂x) = (1-t⁺)/F × a·j
        dc_dt = self.d_dt_central(c_e_r, dt_physical)
        c_e_face = 0.5 * (c_e_r[:, :-1, :] + c_e_r[:, 1:, :])
        D_e_face = electrolyte_diffusivity(c_e_face)
        flux_c = D_e_face * (c_e_r[:, 1:, :] - c_e_r[:, :-1, :]) / dx_r[..., None]
        div_flux_c = self._ddx_flux(flux_c, dx_r)

        H = min(dc_dt.shape[1], div_flux_c.shape[1])
        W = min(dc_dt.shape[2], div_flux_c.shape[2])
        mass_source = eps * dc_dt[:, :H, :W] - div_flux_c[:, :H, :W]

        # Charge equation: ∂/∂x(κ ∂φ_e/∂x) + ∂/∂x(κ_D ∂ln(c_e)/∂x) = -a·j
        c_e_face_phi = 0.5 * (phi_e_r[:, :-1, :] + phi_e_r[:, 1:, :])
        c_e_face2 = 0.5 * (c_e_r[:, :-1, :] + c_e_r[:, 1:, :])
        kappa_face = electrolyte_conductivity(c_e_face2)
        kappa_D_face = (
            2.0 * self.R_CONST * self.T_REF * kappa_face * (1.0 - t_plus[:, None, None])
            / self.F_CONST
        )

        flux_phi = kappa_face * (phi_e_r[:, 1:, :] - phi_e_r[:, :-1, :]) / dx_r[..., None]
        ln_c_e = torch.log(torch.clamp(c_e_r, min=1.0))
        flux_ln_c = kappa_D_face * (ln_c_e[:, 1:, :] - ln_c_e[:, :-1, :]) / dx_r[..., None]

        div_flux_phi = self._ddx_flux(flux_phi, dx_r)
        div_flux_ln_c = self._ddx_flux(flux_ln_c, dx_r)

        H2 = min(div_flux_phi.shape[1], div_flux_ln_c.shape[1])
        W2 = min(W, min(div_flux_phi.shape[2], div_flux_ln_c.shape[2]))
        H = min(H, H2)
        W = min(W, W2)

        charge_source = div_flux_phi[:, :H, :W] + div_flux_ln_c[:, :H, :W]

        mass_source = mass_source[:, :H, :W]

        # Cross-consistency: F/(1-t⁺) × mass_source + charge_source = 0
        factor = self.F_CONST / (1.0 - t_plus[:, None, None])
        residual = factor * mass_source + charge_source

        scale = residual.abs().mean() + 1e-10
        return residual / scale


class PDELoss(nn.Module):
    """Combined PDE loss for Physics-Informed Neural Operator training."""

    def __init__(
        self,
        c_e_stats,
        phi_e_stats,
        param_stats,
        device="cpu",
        lambda_mass=1.0,
        lambda_charge=1.0,
        lambda_consistency=0.5,
    ):
        super().__init__()
        self.physics = BatteryPhysics(device=device)
        self.lambda_mass = lambda_mass
        self.lambda_charge = lambda_charge
        self.lambda_consistency = lambda_consistency

        self.c_e_mean = c_e_stats["mean"]
        self.c_e_std = c_e_stats["std"]
        self.phi_e_mean = phi_e_stats["mean"]
        self.phi_e_std = phi_e_stats["std"]

        self.param_mean = torch.tensor(param_stats["mean"], device=device)
        self.param_std = torch.tensor(param_stats["std"], device=device)

    def denormalize_fields(self, fields_norm):
        """Convert normalized FNO output to physical units."""
        c_e = fields_norm[:, 0] * self.c_e_std + self.c_e_mean
        phi_e = fields_norm[:, 1] * self.phi_e_std + self.phi_e_mean
        return c_e, phi_e

    def denormalize_params(self, params_norm):
        """Convert normalized parameters to physical units."""
        return params_norm * self.param_std + self.param_mean

    def forward(self, fields_norm, params_norm, c_rate):
        """
        Compute total PDE residual loss.

        Args:
            fields_norm: (B, 2, H, W) normalized [c_e, phi_e]
            params_norm: (B, 7) normalized parameters
            c_rate: (B, 1) C-rate values
        Returns:
            loss: scalar PDE loss
            breakdown: dict with individual residual norms
        """
        c_e, phi_e = self.denormalize_fields(fields_norm)
        params_phys = self.denormalize_params(params_norm)
        t_plus = params_phys[:, 2]

        c_rate_scalar = c_rate.squeeze(-1).abs()
        c_rate_scalar = torch.clamp(c_rate_scalar, min=0.01)

        B, H, W = c_e.shape

        dt_physical = 3600.0 / c_rate_scalar

        total_loss = torch.tensor(0.0, device=fields_norm.device)
        breakdown = {}

        for i in range(B):
            c_e_i = c_e[i : i + 1]
            phi_e_i = phi_e[i : i + 1]
            t_plus_i = t_plus[i : i + 1]
            dt_i = dt_physical[i].item() / (W - 1)

            r_mass = self.physics.separator_mass_residual(c_e_i, dt_i)
            mass_loss = (r_mass ** 2).mean()
            total_loss = total_loss + self.lambda_mass * mass_loss

            r_charge = self.physics.separator_charge_residual(
                c_e_i, phi_e_i, t_plus_i, dt_i
            )
            charge_loss = (r_charge ** 2).mean()
            total_loss = total_loss + self.lambda_charge * charge_loss

        total_loss = total_loss / B

        breakdown["mass_residual"] = mass_loss.item()
        breakdown["charge_residual"] = charge_loss.item()

        return total_loss, breakdown
