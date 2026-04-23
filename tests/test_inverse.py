import torch
import numpy as np
from src.pinn.network import MultiDomainPINN, InversePINN
from src.pinn.pdes import MetalBatteryPDE
from src.pinn.losses import PINNLoss


def test_loss_computation():
    """Test that PINNLoss produces valid scalar loss."""
    loss_fn = PINNLoss(lambda_data=10.0, lambda_pde=1.0)

    v_pred = torch.randn(16, requires_grad=True)
    v_obs = torch.randn(16)

    losses = loss_fn(v_pred=v_pred, v_obs=v_obs)
    assert "total" in losses
    assert "data" in losses
    assert losses["total"].shape == ()
    assert losses["total"].requires_grad


def test_pde_loss_from_residuals():
    """Test PDE loss from named residuals."""
    loss_fn = PINNLoss(lambda_pde=2.0)
    residuals = {
        "metal_kinetics": torch.randn(10, 1),
        "cathode_diffusion": torch.randn(10, 1),
    }

    losses = loss_fn(pde_residuals=residuals)
    assert "pde" in losses
    assert losses["pde"].item() > 0


def test_inverse_training_one_step():
    """Test that one training step of inverse PINN reduces loss."""
    torch.manual_seed(42)

    base = MultiDomainPINN(num_params=2, hidden_dim=32, num_layers=2)
    learnable = ["D_e", "k_sei"]
    init_values = {"D_e": 1e-10, "k_sei": 1e-10}

    inv_model = InversePINN(base, learnable, init_values)

    t = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
    v_obs = torch.sin(t.detach() * 3.14) * 3.7

    loss_fn = PINNLoss(lambda_data=10.0)

    optimizer = torch.optim.Adam(inv_model.parameters(), lr=1e-3)

    x = torch.full((50, 1), 0.75).requires_grad_(True)
    r = torch.full((50, 1), 0.5).requires_grad_(True)
    domain = torch.full((50,), 2, dtype=torch.long)

    outputs = inv_model(t, x, r, domain)
    v_pred = outputs["V"]
    losses = loss_fn(v_pred=v_pred, v_obs=v_obs)
    loss_before = losses["total"].item()

    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    outputs = inv_model(t, x, r, domain)
    v_pred = outputs["V"]
    losses = loss_fn(v_pred=v_pred, v_obs=v_obs)
    loss_after = losses["total"].item()

    assert loss_after < loss_before, f"Loss did not decrease: {loss_before} -> {loss_after}"
