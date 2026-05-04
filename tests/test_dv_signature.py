"""Tests for the differential-voltage signature module.

These are pure-numpy tests so they run quickly without a GPU; the
PyBaMM-dependent end-to-end test lives in
``tests/test_lithium_plating_model.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.diagnosis import (
    DegradationDiagnosis,
    SignatureLibrary,
    build_signature_library,
)


# ---------------------------------------------------------------------------
# Synthetic generator: V(t) ≈ V_baseline(t) + Σ_j a_j(t) * (p_j - p_j*)
# ---------------------------------------------------------------------------


def _synthetic_dataset(n_sim=400, n_time=80, n_params=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_time)

    # Ground-truth signatures: smooth basis functions (sinusoids of different freq).
    true_signatures = np.stack(
        [np.sin((j + 1) * np.pi * t) * (0.05 + 0.02 * j) for j in range(n_params)],
        axis=0,
    )
    p_ref = np.zeros(n_params)

    # Sample params uniformly around p_ref.
    P = rng.uniform(-1.0, 1.0, size=(n_sim, n_params))
    V_baseline = 3.6 - 0.4 * t  # arbitrary smooth discharge shape
    V = V_baseline[None, :] + P @ true_signatures + 0.001 * rng.standard_normal(
        (n_sim, n_time)
    )
    return V, P, true_signatures, t


def test_build_signature_library_recovers_signatures():
    V, P, true_sig, t = _synthetic_dataset()
    library = build_signature_library(
        V_sim=V,
        P_sim=P,
        param_names=("p0", "p1", "p2", "p3"),
        log_scale_params=(),
        ridge_alpha=1e-6,   # near-OLS
        time_grid=t,
    )

    assert library.signatures.shape == true_sig.shape
    # Ridge fits in the *normalised* parameter space, so the recovered
    # signatures absorb param_scale. Re-scale before comparing.
    recovered = library.signatures / library.param_scale[:, None]
    err = np.linalg.norm(recovered - true_sig) / np.linalg.norm(true_sig)
    assert err < 0.05, f"Signature recovery error too large: {err:.3f}"


def test_signature_library_save_load_roundtrip(tmp_path):
    V, P, _, t = _synthetic_dataset()
    library = build_signature_library(
        V_sim=V, P_sim=P,
        param_names=("p0", "p1", "p2", "p3"),
        time_grid=t, ridge_alpha=0.1,
        bootstrap_samples=8,
    )
    path = library.save(tmp_path / "lib.npz")
    loaded = SignatureLibrary.load(path)

    assert loaded.param_names == library.param_names
    np.testing.assert_allclose(loaded.signatures, library.signatures)
    np.testing.assert_allclose(loaded.param_scale, library.param_scale)
    assert loaded.bootstrap_signatures is not None
    assert loaded.bootstrap_signatures.shape[0] == 8


def test_diagnosis_recovers_known_coefficients():
    V, P, true_sig, t = _synthetic_dataset(n_sim=600)
    library = build_signature_library(
        V_sim=V, P_sim=P,
        param_names=("p0", "p1", "p2", "p3"),
        time_grid=t, ridge_alpha=1e-4,
    )

    # Build a synthetic experimental cell whose ΔV exactly = a known
    # combination of the true signatures.
    coeffs_true = np.array([0.6, -0.3, 0.1, 0.0])
    n_time = library.n_time
    v_ref = 3.6 - 0.4 * t
    v_exp = v_ref + coeffs_true @ true_sig

    # ΔV = coeffs_true · true_sig.  Library signatures encode
    #   library.signatures[j] = scale[j] · true_sig[j]
    # so the diagnosis-space coefficient is coeffs_true / scale.
    diag = DegradationDiagnosis(library, regressor="ridge", alpha=1e-6)
    res = diag.diagnose_cycle(
        voltage=v_exp, v_ref=v_ref, cycle=1, smooth_sigma=0.0,
    )
    expected = coeffs_true / library.param_scale
    err = np.abs(res.coeffs - expected) / (np.abs(expected) + 1e-3)
    assert err.max() < 0.1, f"Recovered coeffs deviate: {res.coeffs} vs {expected}"
    assert res.dV_rmse_mV < 5.0  # mV


def test_bootstrap_yields_confidence_intervals():
    V, P, true_sig, t = _synthetic_dataset()
    library = build_signature_library(
        V_sim=V, P_sim=P,
        param_names=("p0", "p1", "p2", "p3"),
        time_grid=t, ridge_alpha=1e-3,
    )
    v_ref = 3.6 - 0.4 * t
    v_exp = v_ref + 0.4 * true_sig[0] + 0.001 * np.random.default_rng(1).standard_normal(t.shape)

    diag = DegradationDiagnosis(library)
    res = diag.diagnose_cycle(
        voltage=v_exp, v_ref=v_ref, cycle=1, smooth_sigma=0.0,
        bootstrap_samples=50, rng=np.random.default_rng(2),
    )
    assert res.coeffs_ci_low is not None and res.coeffs_ci_high is not None
    assert (res.coeffs_ci_high >= res.coeffs_ci_low).all()


def test_leave_one_out_ablation_increases_rmse_for_useful_signature():
    V, P, true_sig, t = _synthetic_dataset()
    library = build_signature_library(
        V_sim=V, P_sim=P,
        param_names=("p0", "p1", "p2", "p3"),
        time_grid=t, ridge_alpha=1e-3,
    )
    v_ref = 3.6 - 0.4 * t
    # Cell whose signal is dominated by p0.
    curves = [
        {"voltage": v_ref + 0.5 * true_sig[0] + 0.0005 * np.random.default_rng(i).standard_normal(t.shape),
         "cycle": i, "capacity": 1.0}
        for i in range(10)
    ]

    diag = DegradationDiagnosis(library, regressor="ridge")
    ablation = diag.leave_one_out_ablation(curves, n_ref_cycles=3, smooth_sigma=0.0)
    assert ablation["p0"]["delta_mV"] > ablation["p3"]["delta_mV"], (
        "Dropping a useful signature should hurt RMSE more than dropping a useless one"
    )


def test_dimension_validation():
    with pytest.raises(ValueError):
        build_signature_library(
            V_sim=np.zeros((10, 50)),
            P_sim=np.zeros((11, 3)),  # mismatched first dim
            param_names=("a", "b", "c"),
        )
    with pytest.raises(ValueError):
        build_signature_library(
            V_sim=np.zeros((10, 50)),
            P_sim=np.zeros((10, 3)),
            param_names=("a", "b"),  # mismatched columns
        )
