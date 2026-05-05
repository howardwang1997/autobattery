"""Tests for the rank-robustness math in scripts/99_rank_robustness.py.

The script's role is to back the paper's "rank=3 is structural, not an
artefact" claim. These tests cover the load-bearing pieces:

  * eta_rank correctly counts eigenvalues above relative threshold
  * signature_matrix recovers a known Jacobian on synthetic data
  * fisher_eigenvalues returns a sorted-descending non-negative spectrum
  * transform_params is a bijection on the data manifold (so different
    parameterisations should report the *same* numerical rank when
    eigenvalues are well above the noise floor)
  * bootstrap variance is finite and stable

These run on CPU in <1s and require no PyBaMM / GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def rr():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "rank_robustness",
        repo_root / "scripts" / "99_rank_robustness.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rank_robustness"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# eta_rank
# ---------------------------------------------------------------------------


def test_eta_rank_zero_when_all_eigs_zero(rr):
    assert rr.eta_rank(np.array([0.0, 0.0, 0.0]), 1e-6) == 0


def test_eta_rank_full_when_eigs_uniform(rr):
    assert rr.eta_rank(np.array([1.0, 1.0, 1.0, 1.0]), 1e-6) == 4


def test_eta_rank_geometric_decay(rr):
    eigs = np.array([1.0, 1e-1, 1e-3, 1e-7, 1e-12])
    assert rr.eta_rank(eigs, 1e-2) == 2     # eigs > 0.01: λ=1, λ=0.1
    assert rr.eta_rank(eigs, 1e-6) == 3     # + λ=1e-3
    assert rr.eta_rank(eigs, 1e-9) == 4     # + λ=1e-7
    assert rr.eta_rank(eigs, 1e-15) == 5    # all five


# ---------------------------------------------------------------------------
# Signature recovery
# ---------------------------------------------------------------------------


def _synthetic_dataset(seed=0, n_sim=400, n_time=80, n_params=5):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_time)
    # Smooth orthogonal-ish signature basis.
    true_J = np.stack([
        np.sin((j + 1) * np.pi * t) * (0.05 + 0.02 * j) for j in range(n_params)
    ], axis=0)
    P = rng.uniform(-1.0, 1.0, size=(n_sim, n_params))
    V_baseline = 3.6 - 0.4 * t
    V = V_baseline[None, :] + P @ true_J + 0.001 * rng.standard_normal((n_sim, n_time))
    return V, P, true_J


def test_signature_matrix_recovers_J_on_synthetic(rr):
    V, P, true_J = _synthetic_dataset()
    J = rr.signature_matrix(V, P, alpha=1e-6)
    err = np.linalg.norm(J - true_J) / np.linalg.norm(true_J)
    assert err < 0.05


def test_signature_matrix_demeans_intercept(rr):
    """A constant V offset must not appear in the recovered Jacobian."""
    V, P, true_J = _synthetic_dataset()
    V_shift = V + 100.0     # huge constant offset
    J = rr.signature_matrix(V_shift, P, alpha=1e-6)
    err = np.linalg.norm(J - true_J) / np.linalg.norm(true_J)
    assert err < 0.05, "constant V offset leaked into signature"


# ---------------------------------------------------------------------------
# Fisher eigenvalue spectrum
# ---------------------------------------------------------------------------


def test_fisher_eigenvalues_sorted_descending(rr):
    V, P, _ = _synthetic_dataset()
    J = rr.signature_matrix(V, P)
    eigs = rr.fisher_eigenvalues(J)
    diffs = np.diff(eigs)
    assert (diffs <= 1e-12).all(), "eigenvalues are not non-increasing"


def test_fisher_eigenvalues_non_negative(rr):
    V, P, _ = _synthetic_dataset()
    J = rr.signature_matrix(V, P)
    eigs = rr.fisher_eigenvalues(J)
    assert (eigs >= -1e-9).all()


def test_fisher_eigenvalues_match_singular_values_squared(rr):
    """λ_i(JJᵀ) == σ_i(J)². Standard linear algebra fact, but it's the
    bridge between SVD-based intuition and FIM-based computation."""
    V, P, _ = _synthetic_dataset()
    J = rr.signature_matrix(V, P, alpha=1e-9)
    eigs = rr.fisher_eigenvalues(J)
    sigmas = np.linalg.svd(J, compute_uv=False)
    n = min(len(eigs), len(sigmas))
    np.testing.assert_allclose(eigs[:n], sigmas[:n] ** 2, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Parameterisation invariance (above noise floor)
# ---------------------------------------------------------------------------


def test_log_standardised_invariant_to_log_scale_shift(rr):
    """Same data, two parameterisations both built from log10. Both should
    give identical eigenvalue ratios, even if absolute scales differ."""
    rng = np.random.default_rng(2)
    n_sim, n_params, n_time = 300, 5, 60
    P_pos = 10 ** rng.uniform(-12, -8, size=(n_sim, n_params))
    t = np.linspace(0, 1, n_time)
    log_p = np.log10(P_pos)
    log_p_centered = log_p - log_p.mean(axis=0)
    true_J = np.stack([
        np.sin((j + 1) * np.pi * t) * 0.1 for j in range(n_params)
    ], axis=0)
    V = 3.6 - 0.4 * t + log_p_centered @ true_J + 0.0001 * rng.standard_normal((n_sim, n_time))

    J1 = rr.signature_matrix(V, rr.transform_params(P_pos, "log"), alpha=1e-9)
    J2 = rr.signature_matrix(V, rr.transform_params(P_pos, "log_standardised"), alpha=1e-9)
    eigs1 = rr.fisher_eigenvalues(J1) / rr.fisher_eigenvalues(J1).max()
    eigs2 = rr.fisher_eigenvalues(J2) / rr.fisher_eigenvalues(J2).max()
    # Standardisation rescales each col → eigenvalue *ratios* shift, but the
    # *rank at any reasonable η* should be identical.
    for eta in (1e-3, 1e-6, 1e-9):
        assert rr.eta_rank(eigs1, eta) == rr.eta_rank(eigs2, eta), (
            f"rank disagrees at eta={eta}: {eigs1=} vs {eigs2=}"
        )


def test_pca_whitened_preserves_eta_rank_above_noise(rr):
    """PCA whitening of the parameter columns is an invertible affine map.
    The η-rank of the Fisher matrix is invariant under invertible affine
    transforms (modulo numerical noise). This is the theorem that lets us
    compare ranks across parameterisations at all — it's worth pinning
    down with a test."""
    rng = np.random.default_rng(3)
    n_sim, n_params, n_time = 600, 5, 80
    # Truly low-rank V→θ relationship: V depends only on the first 2
    # PCA directions of θ.
    log_p = rng.uniform(-1, 1, size=(n_sim, n_params))
    P = 10.0 ** log_p
    t = np.linspace(0, 1, n_time)
    pcs = log_p[:, :2]
    sig1 = np.sin(np.pi * t) * 0.1
    sig2 = np.cos(2 * np.pi * t) * 0.05
    V = 3.6 - 0.4 * t + pcs[:, :1] * sig1 + pcs[:, 1:2] * sig2 \
        + 1e-4 * rng.standard_normal((n_sim, n_time))

    P_log = rr.transform_params(P, "log")
    P_pca = rr.transform_params(P, "pca_whitened")

    eigs_log = rr.fisher_eigenvalues(rr.signature_matrix(V, P_log, alpha=1e-9))
    eigs_pca = rr.fisher_eigenvalues(rr.signature_matrix(V, P_pca, alpha=1e-9))

    # At η levels safely above the 1e-4 noise floor, both parameterisations
    # must report the same rank — that's the "rank is parameterisation-
    # invariant above noise" claim that the paper rests on.
    for eta in (1e-2, 1e-3):
        assert rr.eta_rank(eigs_log, eta) == rr.eta_rank(eigs_pca, eta) == 2


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_eigenvalues_returns_correct_shape(rr):
    V, P, _ = _synthetic_dataset()
    eigs = rr.bootstrap_eigenvalues(V, P, n_boot=10, alpha=1e-6, seed=0)
    assert eigs.shape == (10, P.shape[1])


def test_bootstrap_variance_is_finite(rr):
    V, P, _ = _synthetic_dataset()
    eigs = rr.bootstrap_eigenvalues(V, P, n_boot=30, alpha=1e-6, seed=0)
    # Each column (per-eigenvalue distribution) should have finite std.
    stds = eigs.std(axis=0)
    assert np.all(np.isfinite(stds))


# ---------------------------------------------------------------------------
# Multi-rate joint Fisher
# ---------------------------------------------------------------------------


def test_multi_rate_joint_rank_at_least_max_single_rate(rr):
    """Joint Fisher = sum of per-rate Fisher matrices; its eigenspectrum
    dominates each individual Fisher's. Therefore the η-rank of the joint
    Fisher must be ≥ the max single-rate η-rank for any η. This is the
    Sulzer-2021 argument compressed into one inequality."""
    rng = np.random.default_rng(4)
    n_sim_per_rate, n_params, n_time = 200, 5, 60
    rates = [0.1, 0.5, 1.0]
    t = np.linspace(0, 1, n_time)
    Vs, Ps, crs = [], [], []
    for cr in rates:
        log_p = rng.uniform(-1, 1, size=(n_sim_per_rate, n_params))
        # Each rate excites a different mode strongly.
        amplitudes = np.array([cr if (i + int(cr * 10)) % 3 == 0 else 0.05
                                for i in range(n_params)])
        J_cr = np.stack([
            np.sin((j + 1) * np.pi * t) * amplitudes[j] for j in range(n_params)
        ], axis=0)
        V = 3.6 - 0.4 * t + log_p @ J_cr + 1e-5 * rng.standard_normal((n_sim_per_rate, n_time))
        Vs.append(V); Ps.append(10 ** log_p)
        crs.extend([cr] * n_sim_per_rate)

    V_all = np.vstack(Vs)
    P_all = np.vstack(Ps)
    cr_all = np.array(crs)
    P_t = rr.transform_params(P_all, "log_standardised")

    F_joint, used = rr.joint_multi_rate_fisher(
        V_all, P_t, cr_all, rates, alpha=1e-9,
    )
    eigs_joint = np.sort(np.abs(np.linalg.eigvalsh(F_joint)))[::-1]

    single_ranks = []
    for cr in rates:
        mask = np.abs(cr_all - cr) < 0.01
        J_cr = rr.signature_matrix(V_all[mask], P_t[mask], alpha=1e-9)
        single_ranks.append(rr.eta_rank(rr.fisher_eigenvalues(J_cr), 1e-6))
    joint_rank = rr.eta_rank(eigs_joint, 1e-6)
    assert joint_rank >= max(single_ranks)
    assert len(used) == len(rates)


# ---------------------------------------------------------------------------
# Transform validation
# ---------------------------------------------------------------------------


def test_transform_unknown_mode_raises(rr):
    with pytest.raises(ValueError):
        rr.transform_params(np.array([[1.0, 2.0]]), "ramen")


def test_transform_log_handles_zero_safely(rr):
    P = np.array([[0.0, 1.0], [1e-30, 2.0]])
    out = rr.transform_params(P, "log")
    assert np.all(np.isfinite(out))
