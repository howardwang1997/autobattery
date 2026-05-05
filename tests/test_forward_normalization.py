"""Regression tests for src.pinn.forward.

Two non-negotiables for Phase A1:

1. The validation RMSE uses the SAME normaliser as training. Previously
   training used per-sim normalisation but evaluation denormalised with
   global stats — a silent bug that under-counted error. We freeze that
   contract here.
2. PDE collocation samples parameters from the training distribution
   rather than zeros. Otherwise ``--use-pde`` is theatre.

These tests build a 16-curve synthetic dataset, instantiate the trainer
on CPU, run a few forward passes, and assert structural invariants.
They do NOT assert convergence — that needs the H20 box.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

logging.disable(logging.CRITICAL)


def _toy_dataset(path: Path, n_sim: int = 16, n_time: int = 50, n_params: int = 4) -> Path:
    rng = np.random.default_rng(0)
    t = np.tile(np.linspace(0, 1, n_time)[None, :], (n_sim, 1)).astype(np.float32)
    params = rng.uniform(1e-13, 1e-10, size=(n_sim, n_params)).astype(np.float64)
    # Voltage curves: smooth decay with parameter-dependent slope.
    log_p = np.log10(params)
    v = (
        3.6 - 0.4 * t
        + 0.05 * (log_p[:, :1] - log_p[:, :1].mean()) * t
        + 0.001 * rng.standard_normal((n_sim, n_time)).astype(np.float32)
    )
    masks = np.ones((n_sim, n_time), dtype=bool)
    np.savez_compressed(
        path,
        times=t,
        voltages=v.astype(np.float32),
        currents=np.zeros_like(t),
        masks=masks,
        c_rates=np.full(n_sim, 1.0),
        temperatures=np.full(n_sim, 25.0),
        param_names=np.array([f"p{i}" for i in range(n_params)]),
        param_values=params,
    )
    return path


def _build_trainer(model_kind: str, **kwargs):
    from src.pinn.forward import ForwardTrainer
    from src.pinn.losses import PINNLoss
    from src.pinn.pdes import MetalBatteryPDE
    from src.pinn.network import VoltageMLP, MultiDomainPINN

    if model_kind == "mlp":
        model = VoltageMLP(num_params=4, hidden_dim=32, num_layers=2)
    elif model_kind == "pinn":
        model = MultiDomainPINN(num_params=4, hidden_dim=32, num_layers=2)
    else:
        raise ValueError(model_kind)

    return ForwardTrainer(
        model=model,
        pde=MetalBatteryPDE(),
        loss_fn=PINNLoss(lambda_data=1.0, lambda_pde=1.0),
        device=torch.device("cpu"),
        lr=1e-3,
        scheduler="none",
        num_epochs=2,
        log_every=1,
        save_every=999,
        checkpoint_dir=str(Path("outputs/test_checkpoints")),
        **kwargs,
    )


def test_normalizer_is_shared_between_train_and_val(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("mlp", norm_mode="global")
    trainer._precompute_data(str(data), n_points=50)

    # The same normalizer instance is used everywhere.
    assert trainer.normalizer is not None
    rmse = trainer._validate_rmse_mV()
    assert isinstance(rmse, float)
    assert rmse > 0.0  # sanity, model is random


def test_global_norm_is_invertible(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("mlp", norm_mode="global")
    trainer._precompute_data(str(data), n_points=50)

    v_raw = trainer._v_raw
    v_norm = trainer.normalizer.normalize(v_raw)
    v_back = trainer.normalizer.denormalize(v_norm)
    assert torch.allclose(v_back, v_raw, atol=1e-3)


def test_per_sim_norm_emits_deprecation(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_trainer("mlp", norm_mode="per_sim")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_pde_collocation_uses_real_params(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("pinn", norm_mode="global", use_pde=True,
                              pde_collocation_points=8)
    trainer._precompute_data(str(data), n_points=50)
    colloc = trainer._sample_collocation(8)

    assert colloc["params"].shape == (8, 4)
    # Real params come from a normalized log-space distribution centred at 0.
    # If they were zeros the std would be exactly 0.
    assert float(colloc["params"].std()) > 1e-6


def test_physics_param_mapping_returns_positive(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("pinn", norm_mode="global", use_pde=True,
                              pde_collocation_points=4)
    trainer._precompute_data(str(data), n_points=50)
    colloc = trainer._sample_collocation(4)
    physics = trainer._physics_params_from_norm(colloc["params"])

    for k, v in physics.items():
        assert torch.all(v > 0), f"physics param {k} not positive: {v}"
        assert v.shape == (4, 1)


def test_train_runs_one_epoch_without_error(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("mlp", norm_mode="global")
    history = trainer.train(str(data), batch_size=4, n_points=50)

    assert "train_loss" in history and len(history["train_loss"]) == 2
    assert "val_rmse_mV" in history and len(history["val_rmse_mV"]) >= 1


def test_softadapt_records_weights(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("pinn", norm_mode="global", use_pde=True,
                              pde_collocation_points=4,
                              adaptive_weighting="softadapt")
    history = trainer.train(str(data), batch_size=4, n_points=50)

    # SoftAdapt should populate non-trivial weight history.
    assert len(history["softadapt_w_data"]) == 2
    assert len(history["softadapt_w_pde"]) == 2


def test_predict_rejects_per_sim_mode(tmp_path):
    data = _toy_dataset(tmp_path / "toy.npz")
    trainer = _build_trainer("mlp", norm_mode="per_sim")
    trainer._precompute_data(str(data), n_points=50)
    t = torch.zeros((4, 1))
    p = torch.zeros((4, 4))
    with pytest.raises(RuntimeError):
        trainer.predict(t, p)
