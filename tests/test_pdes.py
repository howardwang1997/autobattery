import torch
import numpy as np
from src.pinn.pdes import MetalBatteryPDE
from src.pinn.network import MultiDomainPINN


def _get_network_outputs(B=8, num_params=4, domain_val=0):
    """Helper: run network and return connected tensors."""
    model = MultiDomainPINN(num_params=num_params, hidden_dim=32, num_layers=2)
    t = torch.rand(B, 1, requires_grad=True)
    x = torch.rand(B, 1, requires_grad=True)
    r = torch.rand(B, 1, requires_grad=True)
    params = torch.randn(B, num_params)
    domain = torch.full((B,), domain_val, dtype=torch.long)
    outputs = model(t, x, r, params, domain)
    return outputs, t, x, r


def test_electrolyte_diffusion_with_network():
    """Electrolyte diffusion residual with network-connected tensors."""
    pde = MetalBatteryPDE()
    outputs, t, x, r = _get_network_outputs(B=8, domain_val=0)
    res = pde.electrolyte_diffusion_residual(
        outputs["neg_c_e"], t, x, D_e=torch.tensor(1e-10)
    )
    assert res.shape == (8, 1)
    assert res.requires_grad


def test_metal_plating_kinetics_zero_overpotential():
    """Zero overpotential should give zero current."""
    pde = MetalBatteryPDE()
    phi_s = torch.ones(5, 1) * 0.5
    phi_e = torch.ones(5, 1) * 0.5
    j0 = torch.tensor(0.1)

    j = pde.metal_plating_kinetics(torch.ones(5, 1), phi_s, phi_e, j0)
    assert torch.allclose(j, torch.zeros_like(j), atol=1e-6)


def test_metal_plating_kinetics_positive_overpotential():
    """Positive overpotential should give positive current."""
    pde = MetalBatteryPDE()
    phi_s = torch.ones(5, 1) * 0.1
    phi_e = torch.ones(5, 1) * 0.0
    j0 = torch.tensor(0.1)

    j = pde.metal_plating_kinetics(torch.ones(5, 1), phi_s, phi_e, j0)
    assert (j > 0).all()


def test_cathode_diffusion_with_network():
    """Cathode diffusion residual with network-connected tensors."""
    pde = MetalBatteryPDE()
    outputs, t, x, r = _get_network_outputs(B=8, domain_val=2)
    res = pde.cathode_diffusion_residual(
        outputs["pos_c_s"], t, r, D_s=torch.tensor(1e-13)
    )
    assert res.shape == (8, 1)
    assert res.requires_grad


def test_sei_growth_residual_with_network():
    """SEI growth residual with network-connected tensors."""
    pde = MetalBatteryPDE()
    outputs, t, x, r = _get_network_outputs(B=8)
    res = pde.sei_growth_residual(
        outputs["L_sei"], t, j_side=torch.ones(8, 1) * 0.01, k_sei=torch.tensor(1e-10)
    )
    assert res.shape == (8, 1)
    assert res.requires_grad


def test_charge_conservation_with_network():
    """Charge conservation residual with network-connected tensors."""
    pde = MetalBatteryPDE()
    outputs, t, x, r = _get_network_outputs(B=8, domain_val=0)
    res = pde.charge_conservation_solid_residual(
        outputs["neg_phi_s"], x,
        sigma_eff=torch.tensor(10.0),
        j=torch.ones(8, 1) * 0.01,
    )
    assert res.shape == (8, 1)


def test_butler_volmer_cathode():
    """Test cathode Butler-Volmer kinetics."""
    pde = MetalBatteryPDE()
    j = pde.butler_volmer_cathode(
        c_e=torch.ones(5, 1) * 1000,
        c_s_surf=torch.ones(5, 1) * 0.5,
        c_s_max=15000.0,
        phi_s=torch.ones(5, 1) * 3.5,
        phi_e=torch.ones(5, 1) * 3.0,
        U_ocp=torch.ones(5, 1) * 3.3,
        k0=torch.tensor(1e-11),
    )
    assert j.shape == (5, 1)
    assert not torch.isnan(j).any()
