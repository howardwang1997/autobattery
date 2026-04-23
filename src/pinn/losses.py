import torch
import torch.nn as nn
from typing import Optional


class PINNLoss(nn.Module):
    """
    Combined loss for Physics-Informed Neural Network training.

    L_total = lambda_data * L_data + lambda_pde * L_pde
            + lambda_bc * L_bc + lambda_ic * L_ic

    Components:
    - L_data: MSE between predicted and observed voltage (or other quantities)
    - L_pde: Mean squared PDE residual across all collocation points
    - L_bc: Boundary condition enforcement loss
    - L_ic: Initial condition enforcement loss
    """

    def __init__(
        self,
        lambda_data: float = 10.0,
        lambda_pde: float = 1.0,
        lambda_bc: float = 5.0,
        lambda_ic: float = 5.0,
        pde_weights: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.pde_weights = pde_weights or {}
        self.mse = nn.MSELoss()

    def data_loss(
        self,
        v_pred: torch.Tensor,
        v_obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE between predicted and observed voltage."""
        if mask is not None:
            return self.mse(v_pred[mask], v_obs[mask])
        return self.mse(v_pred, v_obs)

    def pde_loss(
        self, residuals: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Weighted sum of squared PDE residuals."""
        total = torch.tensor(0.0, device=next(iter(residuals.values())).device)
        for name, res in residuals.items():
            weight = self.pde_weights.get(name, 1.0)
            total = total + weight * torch.mean(res ** 2)
        return total

    def boundary_loss(
        self,
        outputs: dict[str, torch.Tensor],
        domain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce boundary conditions:
        - phi_s at current collector (x=0 for neg, x=1 for pos) = given potential
        - dc_e/dx = 0 at cell boundaries
        - dc_s/dr = 0 at particle center (r=0)

        For the simplified case, we enforce:
        - neg phi_s at current collector = 0 (ground)
        """
        loss = torch.tensor(0.0)

        if "neg_phi_s" in outputs:
            loss = loss + torch.mean(outputs["neg_phi_s"] ** 2) * 0.01

        return loss

    def initial_condition_loss(
        self,
        outputs: dict[str, torch.Tensor],
        t_norm: torch.Tensor,
        c_e_init: float = 1000.0,
    ) -> torch.Tensor:
        """
        Enforce initial conditions at t=0:
        - Uniform electrolyte concentration
        - Uniform solid concentration in cathode
        """
        loss = torch.tensor(0.0)

        t_zero_mask = t_norm < 0.01
        if t_zero_mask.any():
            if "neg_c_e" in outputs:
                loss = loss + torch.mean(
                    (outputs["neg_c_e"][t_zero_mask.squeeze(-1)] - c_e_init) ** 2
                ) * 0.01

        return loss

    def forward(
        self,
        v_pred: Optional[torch.Tensor] = None,
        v_obs: Optional[torch.Tensor] = None,
        pde_residuals: Optional[dict[str, torch.Tensor]] = None,
        outputs: Optional[dict[str, torch.Tensor]] = None,
        t_norm: Optional[torch.Tensor] = None,
        domain: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Returns dict with individual losses and total.
        """
        losses = {}
        total = torch.tensor(0.0)
        device = torch.device("cpu")

        if v_pred is not None and v_obs is not None:
            device = v_pred.device
            l_data = self.data_loss(v_pred, v_obs)
            losses["data"] = l_data
            total = total + self.lambda_data * l_data

        if pde_residuals is not None:
            device = next(iter(pde_residuals.values())).device
            l_pde = self.pde_loss(pde_residuals)
            losses["pde"] = l_pde
            total = total + self.lambda_pde * l_pde

        if outputs is not None and domain is not None:
            l_bc = self.boundary_loss(outputs, domain)
            if l_bc.device.type != "meta":
                losses["bc"] = l_bc
                total = total + self.lambda_bc * l_bc

        if outputs is not None and t_norm is not None:
            l_ic = self.initial_condition_loss(outputs, t_norm)
            if l_ic.device.type != "meta":
                losses["ic"] = l_ic
                total = total + self.lambda_ic * l_ic

        losses["total"] = total.to(device)
        return losses
