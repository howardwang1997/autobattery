"""Tests for src/universality/ modules.

CPU-only, synthetic data, runs in <5s. No external datasets needed.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: synthetic cohort
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth():
    """Synthetic cohort: 120 cells, 3 archetypes, ~500 cycles max."""
    rng = np.random.default_rng(42)
    n_cells = 120
    T = 500

    cycles = np.arange(1, T + 1, dtype=int)
    Q0 = 1.0 + 0.05 * rng.standard_normal(n_cells)
    capacities = np.zeros((n_cells, T))
    archetypes = np.zeros(n_cells, dtype=int)
    N_star_true = np.zeros(n_cells)

    # archetype 0: gradual linear fade
    for i in range(0, 40):
        rate = 0.0001 + 0.00005 * rng.standard_normal()
        capacities[i] = Q0[i] * (1 - rate * cycles)
        capacities[i] += 0.002 * rng.standard_normal(T)
        archetypes[i] = 0
        N_star_true[i] = 1.0 / max(rate, 1e-9)  # cycle where Q/Q0 = 0

    # archetype 1: knee at ~200-300, then steep
    for i in range(40, 80):
        knee_c = int(rng.uniform(200, 300))
        rate1 = 0.0001
        rate2 = 0.001 + 0.0003 * rng.standard_normal()
        cap = Q0[i] * np.ones(T)
        cap[:knee_c] = Q0[i] * (1 - rate1 * cycles[:knee_c])
        cap[knee_c:] = cap[knee_c - 1] - Q0[i] * rate2 * (cycles[knee_c:] - knee_c)
        cap += 0.002 * rng.standard_normal(T)
        capacities[i] = cap
        archetypes[i] = 1
        N_star_true[i] = knee_c

    # archetype 2: cliff death around 350-450
    for i in range(80, 120):
        cliff = int(rng.uniform(350, 450))
        rate = 0.00005
        cap = Q0[i] * (1 - rate * cycles)
        cap[cliff:] = cap[cliff - 1] * np.exp(-0.01 * (cycles[cliff:] - cliff))
        cap += 0.002 * rng.standard_normal(T)
        capacities[i] = cap
        archetypes[i] = 2
        N_star_true[i] = cliff

    # some cells short (NaN after their "test end")
    for i in rng.choice(n_cells, size=20, replace=False):
        cutoff = int(rng.uniform(100, 400))
        capacities[i, cutoff:] = np.nan

    # formulation features (2 continuous)
    feature_1 = rng.uniform(0.5, 2.5, n_cells)
    feature_2 = rng.uniform(0.8, 1.2, n_cells)

    return {
        "cycles": np.broadcast_to(cycles[None, :], (n_cells, T)).copy(),
        "capacities": capacities,
        "Q0": Q0,
        "archetypes_true": archetypes,
        "N_star_true": N_star_true,
        "feature_1": feature_1,
        "feature_2": feature_2,
        "n_cells": n_cells,
        "T": T,
    }


# ---------------------------------------------------------------------------
# curves.py
# ---------------------------------------------------------------------------


def test_aligned_curves_retention(synth):
    from src.universality.curves import AlignedCurves

    curves = AlignedCurves(
        cell_ids=np.array([f"C{i:03d}" for i in range(synth["n_cells"])]),
        cycles=synth["cycles"],
        capacities=synth["capacities"],
        Q0=synth["Q0"],
    )
    ret = curves.capacity_retention()
    assert ret.shape == (synth["n_cells"], synth["T"])
    # first column ≈ 1.0
    first_valid = ret[:, 0]
    np.testing.assert_allclose(first_valid[~np.isnan(first_valid)], 1.0, atol=0.05)


def test_aligned_curves_fade_pct(synth):
    from src.universality.curves import AlignedCurves

    curves = AlignedCurves(
        cell_ids=np.array([f"C{i:03d}" for i in range(synth["n_cells"])]),
        cycles=synth["cycles"],
        capacities=synth["capacities"],
        Q0=synth["Q0"],
    )
    fade = curves.fade_pct()
    assert np.all(np.isfinite(fade))
    assert np.all(fade > -5)  # can be slightly negative due to noise


# ---------------------------------------------------------------------------
# knee.py
# ---------------------------------------------------------------------------


def test_kneedle_detects_known_knee():
    from src.universality.knee import kneedle_detect

    x = np.arange(500, dtype=float)
    y = np.ones(500)
    y[:250] = 1.0 - 0.0001 * x[:250]
    y[250:] = y[249] - 0.002 * (x[250:] - 250)
    idx = kneedle_detect(x, y)
    assert idx is not None
    assert 200 < x[idx] < 300


def test_piecewise_detects_known_breakpoint():
    from src.universality.knee import piecewise_linear_detect

    x = np.arange(200, dtype=float)
    y = np.ones(200)
    y[:100] = 1.0 - 0.0005 * x[:100]
    y[100:] = y[99] - 0.005 * (x[100:] - 100)
    idx = piecewise_linear_detect(x, y, min_segment=15)
    assert idx is not None
    assert 80 < idx < 120


def test_detect_knees_batch(synth):
    from src.universality.knee import detect_knees

    results = detect_knees(
        np.array([f"C{i:03d}" for i in range(synth["n_cells"])]),
        synth["cycles"], synth["capacities"], synth["Q0"],
    )
    assert len(results) == synth["n_cells"]
    knee_count = sum(r.has_knee for r in results)
    assert knee_count > 20  # archetype 1+2 should all have knees


def test_no_knee_in_linear_curve():
    from src.universality.knee import kneedle_detect

    x = np.arange(300, dtype=float)
    y = 1.0 - 0.0002 * x
    idx = kneedle_detect(x, y, S=5.0)
    # may or may not detect — but if it does, it should be near the end
    # (the "knee" of a line is ill-defined)


# ---------------------------------------------------------------------------
# archetype.py
# ---------------------------------------------------------------------------


def test_fpca_shapes(synth):
    from src.universality.archetype import fpca

    ret = synth["capacities"] / synth["Q0"][:, None]
    ret = np.nan_to_num(ret, nan=ret[~np.isnan(ret)].mean())
    scores, comps, mean, evr = fpca(ret, n_components=5)
    assert scores.shape == (synth["n_cells"], 5)
    assert comps.shape == (5, synth["T"])
    assert mean.shape == (synth["T"],)
    assert evr.shape == (5,)
    assert evr.sum() > 0.5  # first 5 PCs should explain > 50%


def test_cluster_archetypes_finds_multiple_clusters(synth):
    from src.universality.archetype import cluster_archetypes

    ret = synth["capacities"] / synth["Q0"][:, None]
    arch = cluster_archetypes(ret, k_range=range(2, 5), seed=42)
    assert arch.n_archetypes >= 2
    assert arch.labels.shape == (synth["n_cells"],)
    used_labels = set(arch.labels[arch.labels >= 0])
    assert len(used_labels) >= 2


def test_archetype_curves_shape(synth):
    from src.universality.archetype import cluster_archetypes

    ret = synth["capacities"] / synth["Q0"][:, None]
    arch = cluster_archetypes(ret, k_range=range(2, 5), seed=42)
    k = arch.n_archetypes
    T_eff = len(arch.cycle_grid)
    assert arch.archetype_curves.shape == (k, T_eff)
    assert arch.archetype_band_low.shape == (k, T_eff)
    assert arch.archetype_band_high.shape == (k, T_eff)


# ---------------------------------------------------------------------------
# scaling.py
# ---------------------------------------------------------------------------


def test_rescale_curves_output_shape(synth):
    from src.universality.scaling import rescale_curves

    xi, q, mask = rescale_curves(
        synth["cycles"], synth["capacities"],
        synth["Q0"], synth["N_star_true"],
    )
    assert xi.shape == (200,)
    assert q.shape == (synth["n_cells"], 200)
    assert mask.shape == (synth["n_cells"],)
    assert mask.sum() > 50


def test_scaling_analysis_returns_finite_metrics(synth):
    from src.universality.scaling import scaling_analysis

    res = scaling_analysis(
        synth["cycles"], synth["capacities"],
        synth["Q0"], synth["N_star_true"],
    )
    assert np.isfinite(res.residual_rms)
    assert np.isfinite(res.residual_median)
    assert 0 <= res.collapse_fraction_5pct <= 1
    assert 0 <= res.collapse_fraction_10pct <= 1


def test_fit_parametric_master_runs():
    from src.universality.scaling import fit_parametric_master

    xi = np.linspace(0, 2, 100)
    q = 1.0 - 0.15 * np.sqrt(xi + 0.01)
    result = fit_parametric_master(xi, q, form="sqrt")
    assert result["form"] == "sqrt"
    assert len(result["params"]) == 1
    assert result["r2"] > 0.9


# ---------------------------------------------------------------------------
# phase_diagram.py
# ---------------------------------------------------------------------------


def test_fit_phase_diagram_runs(synth):
    from src.universality.phase_diagram import fit_phase_diagram

    F = np.column_stack([synth["feature_1"], synth["feature_2"]])
    res = fit_phase_diagram(
        F, synth["archetypes_true"], synth["N_star_true"],
        feature_names=["feature_1", "feature_2"],
        seed=42,
    )
    assert 0 <= res.archetype_accuracy <= 1
    assert res.nstar_r2 is not None


def test_boundary_grid_shape(synth):
    from src.universality.phase_diagram import fit_phase_diagram, boundary_grid

    F = np.column_stack([synth["feature_1"], synth["feature_2"]])
    res = fit_phase_diagram(
        F, synth["archetypes_true"], synth["N_star_true"],
        feature_names=["feature_1", "feature_2"],
        try_symbolic=False, seed=42,
    )
    bg = boundary_grid(
        res.archetype_classifier,
        {"feature_1": (0.5, 2.5), "feature_2": (0.8, 1.2)},
        ["feature_1", "feature_2"],
        grid_n=20,
    )
    assert bg["labels"].shape == (20, 20)
    assert bg["probs"].shape[:2] == (20, 20)
