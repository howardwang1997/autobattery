import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class MetalBatteryPDE:
    """
    PDE residual definitions for the metal battery P2D model.

    Encodes the following physics:
    1. Electrolyte mass transport (concentrated solution theory)
    2. Metal plating/stripping kinetics (Butler-Volmer)
    3. SEI growth (reaction-limited or diffusion-limited)
    4. Cathode intercalation diffusion (Fick's law in spherical coordinates)
    5. Charge conservation (solid + electrolyte)

    All quantities are normalized:
    - time: t_norm = t / t_end in [0, 1]
    - space: x_norm = x / L_total in [0, 1]
    - radial: r_norm = r / R_particle in [0, 1]
    """

    def __init__(
        self,
        F: float = 96485.3329,
        R: float = 8.314462618,
        T_ref: float = 298.15,
        L_neg: float = 20e-6,
        L_sep: float = 25e-6,
        L_pos: float = 70e-6,
        t_end: float = 3600.0,
    ):
        self.F = F
        self.R = R
        self.T_ref = T_ref
        self.L_total = L_neg + L_sep + L_pos
        self.L_neg = L_neg
        self.L_sep = L_sep
        self.L_pos = L_pos
        self.t_end = t_end
        self.L_neg_frac = L_neg / self.L_total
        self.L_sep_frac = L_sep / self.L_total

    def _denorm_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        return t_norm * self.t_end

    def _denorm_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.L_total

    def _denorm_r(self, r_norm: torch.Tensor) -> torch.Tensor:
        return r_norm * 1e-6

    def electrolyte_diffusion_residual(
        self,
        c_e: torch.Tensor,
        t_norm: torch.Tensor,
        x_norm: torch.Tensor,
        D_e: torch.Tensor,
        eps: float = 0.4,
        bruggeman: float = 1.5,
        t_plus: float = 0.38,
        a_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Residual of electrolyte mass transport PDE:
        eps * dc_e/dt - d/dx[D_e * eps^b * dc_e/dx] - (1-t_plus)*a*j/F = 0

        Uses finite differences on the spatial gradient (appropriate for PINN).
        """
        c_e_t = torch.autograd.grad(
            c_e, t_norm,
            grad_outputs=torch.ones_like(c_e),
            create_graph=True,
            allow_unused=True,
        )[0]
        if c_e_t is None:
            c_e_t = torch.zeros_like(c_e)

        c_e_x = torch.autograd.grad(
            c_e, x_norm,
            grad_outputs=torch.ones_like(c_e),
            create_graph=True,
            allow_unused=True,
        )[0]
        if c_e_x is None:
            c_e_x = torch.zeros_like(c_e)

        c_e_xx = torch.autograd.grad(
            c_e_x, x_norm,
            grad_outputs=torch.ones_like(c_e_x),
            create_graph=True,
            allow_unused=True,
        )[0]
        if c_e_xx is None:
            c_e_xx = torch.zeros_like(c_e_x)

        D_e_eff = D_e * (eps ** bruggeman)

        diffusion = D_e_eff * c_e_xx / (self.L_total ** 2)

        transient = eps * c_e_t / self.t_end

        source = torch.zeros_like(c_e)
        if a_j is not None:
            source = (1 - t_plus) * a_j / self.F

        residual = transient - diffusion - source
        return residual

    def metal_plating_kinetics(
        self,
        c_e: torch.Tensor,
        phi_s: torch.Tensor,
        phi_e: torch.Tensor,
        j0_metal: torch.Tensor,
        alpha_a: float = 0.5,
        alpha_c: float = 0.5,
    ) -> torch.Tensor:
        """
        Butler-Volmer kinetics for metal plating/stripping:
        j = j0 * [exp(alpha_a * F * eta / (R*T)) - exp(-alpha_c * F * eta / (R*T))]

        eta = phi_s - phi_e (no OCP for metal anode, U_metal = 0V vs. M/M+)

        Returns the local current density j (A/m^2).
        """
        eta = phi_s - phi_e

        exp_a = torch.exp(alpha_a * self.F * eta / (self.R * self.T_ref))
        exp_c = torch.exp(-alpha_c * self.F * eta / (self.R * self.T_ref))

        j = j0_metal * (exp_a - exp_c)
        return j

    def sei_growth_residual(
        self,
        L_sei: torch.Tensor,
        t_norm: torch.Tensor,
        j_side: torch.Tensor,
        k_sei: torch.Tensor,
        Ea_sei: float = 5e4,
    ) -> torch.Tensor:
        """
        SEI growth kinetic model (reaction-limited):
        dL_sei/dt = k_sei * exp(-Ea/(R*T)) * j_side / F

        Returns PDE residual.
        """
        L_sei_t = torch.autograd.grad(
            L_sei, t_norm,
            grad_outputs=torch.ones_like(L_sei),
            create_graph=True,
            allow_unused=True,
        )[0]
        if L_sei_t is None:
            L_sei_t = torch.zeros_like(L_sei)

        growth_rate = k_sei * torch.exp(torch.tensor(-Ea_sei / (self.R * self.T_ref)))
        source = growth_rate * j_side / self.F

        residual = L_sei_t / self.t_end - source
        return residual

    def cathode_diffusion_residual(
        self,
        c_s: torch.Tensor,
        t_norm: torch.Tensor,
        r_norm: torch.Tensor,
        D_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fick's second law in spherical coordinates (normalized):
        dc_s/dt = (1/r^2) * d/dr[r^2 * D_s * dc_s/dr]

        With normalization r_norm = r / R_particle, the equation becomes:
        dc_s/dt' = (D_s / R_p^2) * (d^2c_s/dr'^2 + 2/r' * dc_s/dr')
        """
        c_s_t = torch.autograd.grad(
            c_s, t_norm,
            grad_outputs=torch.ones_like(c_s),
            create_graph=True,
            allow_unused=True,
        )[0]
        if c_s_t is None:
            c_s_t = torch.zeros_like(c_s)

        c_s_r = torch.autograd.grad(
            c_s, r_norm,
            grad_outputs=torch.ones_like(c_s),
            create_graph=True,
            allow_unused=True,
        )[0]
        if c_s_r is None:
            c_s_r = torch.zeros_like(c_s)

        c_s_rr = torch.autograd.grad(
            c_s_r, r_norm,
            grad_outputs=torch.ones_like(c_s_r),
            create_graph=True,
            allow_unused=True,
        )[0]
        if c_s_rr is None:
            c_s_rr = torch.zeros_like(c_s_r)

        r_safe = torch.clamp(r_norm, min=1e-6)

        R_p = 1e-6
        diffusion = D_s / (R_p ** 2) * (c_s_rr + 2.0 / r_safe * c_s_r)

        residual = c_s_t / self.t_end - diffusion
        return residual

    def charge_conservation_solid_residual(
        self,
        phi_s: torch.Tensor,
        x_norm: torch.Tensor,
        sigma_eff: torch.Tensor,
        j: torch.Tensor,
        a: float = 1e5,
    ) -> torch.Tensor:
        """
        Solid-phase charge conservation:
        d/dx[sigma_eff * dphi_s/dx] = a * j
        """
        phi_s_x = torch.autograd.grad(
            phi_s, x_norm,
            grad_outputs=torch.ones_like(phi_s),
            create_graph=True,
            allow_unused=True,
        )[0]
        if phi_s_x is None:
            phi_s_x = torch.zeros_like(phi_s)

        phi_s_xx = torch.autograd.grad(
            phi_s_x, x_norm,
            grad_outputs=torch.ones_like(phi_s_x),
            create_graph=True,
            allow_unused=True,
        )[0]
        if phi_s_xx is None:
            phi_s_xx = torch.zeros_like(phi_s_x)

        laplacian = sigma_eff * phi_s_xx / (self.L_total ** 2)

        residual = laplacian - a * j
        return residual

    def butler_volmer_cathode(
        self,
        c_e: torch.Tensor,
        c_s_surf: torch.Tensor,
        c_s_max: float,
        phi_s: torch.Tensor,
        phi_e: torch.Tensor,
        U_ocp: torch.Tensor,
        k0: torch.Tensor,
        alpha_a: float = 0.5,
        alpha_c: float = 0.5,
    ) -> torch.Tensor:
        """
        Butler-Volmer kinetics for cathode intercalation:
        j = j0 * [exp(alpha_a*F*eta/RT) - exp(-alpha_c*F*eta/RT)]

        j0 = F * k0 * (c_e/c_e0)^0.5 * (c_s_surf)^0.5 * (c_s_max - c_s_surf)^0.5

        eta = phi_s - phi_e - U_ocp(c_s_surf)
        """
        c_s_safe = torch.clamp(c_s_surf, min=1e-10)
        c_s_diff = torch.clamp(c_s_max - c_s_surf, min=1e-10)

        j0 = self.F * k0 * torch.sqrt(c_e * c_s_safe * c_s_diff)

        eta = phi_s - phi_e - U_ocp

        exp_a = torch.exp(alpha_a * self.F * eta / (self.R * self.T_ref))
        exp_c = torch.exp(-alpha_c * self.F * eta / (self.R * self.T_ref))

        j = j0 * (exp_a - exp_c)
        return j

    def total_residual(
        self,
        outputs: dict[str, torch.Tensor],
        t_norm: torch.Tensor,
        x_norm: torch.Tensor,
        r_norm: torch.Tensor,
        domain: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute all PDE residuals.

        ``params`` may carry either a scalar tensor per physics quantity
        (broadcast to every collocation point) or a per-collocation-point
        tensor of shape ``(N, 1)``. In the second case we slice it with
        the same domain mask used to gather the outputs, so the residual
        operator and its physics parameters are dimension-aligned.

        Returns dict of named residuals for weighted loss computation.
        """
        residuals = {}

        neg_mask = domain == 0
        pos_mask = domain == 2
        N = domain.shape[0]

        def _slice(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Return ``tensor[mask]`` if it's per-point, else broadcast it."""
            if tensor.dim() == 0 or tensor.shape[0] == 1:
                return tensor
            if tensor.shape[0] == N:
                return tensor[mask]
            return tensor   # leave caller's broadcast logic alone

        if neg_mask.any():
            residuals["metal_kinetics"] = self.metal_plating_kinetics(
                c_e=outputs["neg_c_e"],
                phi_s=outputs["neg_phi_s"],
                phi_e=outputs["neg_phi_e"],
                j0_metal=_slice(params["j0_metal"], neg_mask),
            )

        if pos_mask.any():
            residuals["cathode_diffusion"] = self.cathode_diffusion_residual(
                c_s=outputs["pos_c_s"],
                t_norm=t_norm[pos_mask],
                r_norm=r_norm[pos_mask],
                D_s=_slice(params["D_s"], pos_mask),
            )

        residuals["sei_growth"] = self.sei_growth_residual(
            L_sei=outputs["L_sei"],
            t_norm=t_norm,
            j_side=params.get("j_side", torch.zeros_like(t_norm)),
            k_sei=params["k_sei"],
        )

        return residuals
