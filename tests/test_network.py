import torch
from src.pinn.network import MultiDomainPINN, InversePINN


def test_multi_domain_pinn_output_shapes():
    """Test that MultiDomainPINN produces correct output shapes."""
    B = 16
    num_params = 7
    model = MultiDomainPINN(num_params=num_params, hidden_dim=64, num_layers=3)

    t = torch.randn(B, 1)
    x = torch.randn(B, 1)
    r = torch.randn(B, 1)
    params = torch.randn(B, num_params)
    domain = torch.randint(0, 3, (B,))

    result = model(t, x, r, params, domain)

    assert "V" in result
    assert "L_sei" in result
    assert result["V"].shape == (B, 1)
    assert result["L_sei"].shape == (B, 1)


def test_multi_domain_pinn_domain_routing():
    """Test that domain-specific heads are activated correctly."""
    model = MultiDomainPINN(num_params=4, hidden_dim=32, num_layers=2)

    t = torch.randn(3, 1)
    x = torch.randn(3, 1)
    r = torch.randn(3, 1)
    params = torch.randn(3, 4)
    domain = torch.tensor([0, 1, 2])

    result = model(t, x, r, params, domain)

    assert "neg_c_e" in result
    assert "sep_c_e" in result
    assert "pos_c_e" in result
    assert "pos_c_s" in result


def test_voltage_only_forward():
    """Test simplified voltage-only forward pass."""
    model = MultiDomainPINN(num_params=4, hidden_dim=32, num_layers=2)

    t = torch.randn(10, 1)
    params = torch.randn(1, 4)

    v = model.forward_voltage_only(t, params)
    assert v.shape[0] == 10
    assert v.shape[1] == 1


def test_inverse_pinn_parameter_transform():
    """Test that InversePINN correctly transforms parameters."""
    base = MultiDomainPINN(num_params=4, hidden_dim=32, num_layers=2)

    learnable = ["D_e", "D_s"]
    init_values = {"D_e": 1e-10, "D_s": 1e-13}
    bounds = {"D_e": (1e-12, 1e-8), "D_s": (1e-15, 1e-10)}

    inv_model = InversePINN(base, learnable, init_values, bounds)

    params = inv_model.get_all_params()
    assert "D_e" in params
    assert "D_s" in params

    D_e = params["D_e"]
    assert D_e.item() > 0
    assert 1e-12 <= D_e.item() <= 1e-8


def test_inverse_pinn_forward():
    """Test forward pass through InversePINN."""
    base = MultiDomainPINN(num_params=4, hidden_dim=32, num_layers=2)
    learnable = ["D_e", "k_sei", "t_plus", "j0"]
    init_values = {"D_e": 1e-10, "k_sei": 1e-10, "t_plus": 0.4, "j0": 0.1}

    inv_model = InversePINN(base, learnable, init_values)

    B = 8
    t = torch.randn(B, 1).requires_grad_(True)
    x = torch.randn(B, 1).requires_grad_(True)
    r = torch.randn(B, 1).requires_grad_(True)
    domain = torch.randint(0, 3, (B,))

    result = inv_model(t, x, r, domain)
    assert "V" in result
