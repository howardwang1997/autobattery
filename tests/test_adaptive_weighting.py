"""Tests for src.pinn.adaptive_weighting (SoftAdapt + HardBoundaryVoltage)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.pinn.adaptive_weighting import (
    HardBoundaryVoltage,
    SoftAdapt,
    SoftAdaptConfig,
)


# ---------------------------------------------------------------------------
# SoftAdapt
# ---------------------------------------------------------------------------


def test_softadapt_warmup_returns_uniform():
    sa = SoftAdapt(["data", "pde"], SoftAdaptConfig(warmup_steps=10))
    for _ in range(10):
        sa.update({"data": 1.0, "pde": 1.0})
    w = sa.weights()
    assert math.isclose(w["data"], 0.5, abs_tol=1e-6)
    assert math.isclose(w["pde"], 0.5, abs_tol=1e-6)


def test_softadapt_weights_sum_to_one():
    sa = SoftAdapt(["a", "b", "c"], SoftAdaptConfig(warmup_steps=0))
    sa.update({"a": 1.0, "b": 0.5, "c": 2.0})
    w = sa.weights()
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)


def test_softadapt_increases_weight_for_growing_loss():
    """If loss A keeps shrinking and loss B grows, the adaptive weight
    should shift mass toward B (the term that's getting harder)."""
    sa = SoftAdapt(["a", "b"], SoftAdaptConfig(warmup_steps=0, beta=2.0,
                                                normalised=True, ema_alpha=0.5))
    a, b = 1.0, 1.0
    for step in range(40):
        a *= 0.9     # shrinks
        b *= 1.1     # grows
        sa.update({"a": a, "b": b})
    w = sa.weights()
    assert w["b"] > w["a"], f"Expected weight on growing loss to be larger: {w}"


def test_softadapt_unknown_term_raises():
    sa = SoftAdapt(["data"])
    with pytest.raises(KeyError):
        sa.update({"data": 1.0, "ghost": 0.0})


def test_softadapt_constant_losses_stay_uniform():
    sa = SoftAdapt(["a", "b"], SoftAdaptConfig(warmup_steps=0, beta=1.0,
                                                normalised=True))
    for _ in range(50):
        sa.update({"a": 1.0, "b": 1.0})
    w = sa.weights()
    # With normalised relative deltas of 0, softmax over zeros is uniform.
    assert math.isclose(w["a"], 0.5, abs_tol=1e-6)


def test_softadapt_holds_no_tensors():
    """The adapter must never store an autograd tensor; this is what
    guarantees it can't leak a backward graph."""
    sa = SoftAdapt(["data", "pde"])
    loss = torch.tensor(1.0, requires_grad=True)
    sa.update({"data": float(loss.detach()), "pde": 0.5})
    for v in sa._prev.values():
        assert isinstance(v, float)
    for v in sa._weights.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# HardBoundaryVoltage
# ---------------------------------------------------------------------------


class _ConstantOne(nn.Module):
    def __init__(self, num_params=2):
        super().__init__()
        self.num_params = num_params

    def forward(self, t, params):
        return torch.ones_like(t)


def test_hard_boundary_endpoints_match_inits_exactly():
    wrap = HardBoundaryVoltage(_ConstantOne())
    t = torch.tensor([[0.0], [1.0], [0.5]])
    params = torch.zeros((3, 2))
    v_init = torch.tensor([[4.2], [4.2], [4.2]])
    v_final = torch.tensor([[3.0], [3.0], [3.0]])
    out = wrap(t, params, v_init, v_final)
    assert out.shape == (3, 1)
    assert torch.allclose(out[0], v_init[0])
    assert torch.allclose(out[1], v_final[1])
    # Mid-point is mean of endpoints + bump * NN(...) = 3.6 + 0.25 * 1.0
    assert torch.allclose(out[2], torch.tensor([[3.85]]), atol=1e-6)


def test_hard_boundary_handles_1d_inputs():
    wrap = HardBoundaryVoltage(_ConstantOne())
    t = torch.tensor([0.0, 1.0])
    params = torch.zeros((2, 2))
    v_init = torch.tensor([4.2, 4.2])
    v_final = torch.tensor([3.0, 3.0])
    out = wrap(t, params, v_init, v_final)
    assert out.shape == (2, 1)
    assert torch.allclose(out[0], torch.tensor([[4.2]]))
    assert torch.allclose(out[1], torch.tensor([[3.0]]))
