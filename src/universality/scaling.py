"""Scaling-collapse analysis for battery capacity-retention curves.

Theory: within a degradation archetype, the capacity trajectory can be
characterised by a single timescale N★ such that

    Q(N) / Q₀ ≈ Φ(N / N★)

where Φ is a *master curve* shared by all cells of that archetype.

This module:
  1. Accepts N★ per cell (typically the knee-point cycle from ``knee.py``).
  2. Rescales all curves to ξ = N/N★ , q = Q/Q₀.
  3. Fits the master curve Φ(ξ) as a smoothed median (or parametric form).
  4. Quantifies the collapse quality:
     - **collapse residual**: median absolute deviation from Φ after rescaling
     - **collapse fraction**: % of cells within ±5% of Φ

If collapse is good (fraction ≥ 90 %), N★ = f(formulation) is the *only*
parameter needed to predict the full trajectory — a genuine universality
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CollapseResult:
    xi_grid: np.ndarray              # (M,) rescaled cycle axis [0, xi_max]
    master_curve: np.ndarray         # (M,) Φ(ξ)
    band_lo: np.ndarray              # 5th percentile
    band_hi: np.ndarray              # 95th percentile
    residual_rms: float              # RMS deviation from master (in Q/Q0 units)
    residual_median: float           # median absolute deviation
    collapse_fraction_5pct: float    # fraction of points within ±5% of Φ
    collapse_fraction_10pct: float   # fraction of points within ±10% of Φ
    n_cells_used: int


def rescale_curves(
    cycles: np.ndarray,
    capacities: np.ndarray,
    Q0: np.ndarray,
    N_star: np.ndarray,
    xi_max: float = 2.0,
    n_xi: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rescale Q(N) curves and interpolate onto a common ξ = N/N★ grid.

    Returns (xi_grid, q_rescaled, valid_mask).
    q_rescaled is (n_cells, n_xi); NaN where the original curve didn't reach.
    valid_mask is (n_cells,) bool — cells with valid N★ and enough data.
    """
    n = len(Q0)
    xi_grid = np.linspace(0, xi_max, n_xi)
    q_rescaled = np.full((n, n_xi), np.nan)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(N_star[i]) or N_star[i] <= 0:
            continue
        cap = capacities[i]
        cyc = cycles[i] if cycles.ndim == 2 else cycles
        not_nan = ~np.isnan(cap)
        if not_nan.sum() < 5:
            continue
        xi_orig = cyc[not_nan].astype(float) / N_star[i]
        q_orig = cap[not_nan] / Q0[i]
        q_rescaled[i] = np.interp(xi_grid, xi_orig, q_orig, left=np.nan, right=np.nan)
        valid_mask[i] = True

    return xi_grid, q_rescaled, valid_mask


def fit_master_curve(
    xi_grid: np.ndarray,
    q_rescaled: np.ndarray,
    valid_mask: np.ndarray,
) -> CollapseResult:
    """Compute master curve Φ(ξ) as the per-ξ median of the rescaled curves.

    Returns collapse quality metrics.
    """
    Q = q_rescaled[valid_mask]
    n_used = int(valid_mask.sum())

    if n_used < 3:
        return CollapseResult(
            xi_grid=xi_grid,
            master_curve=np.full_like(xi_grid, np.nan),
            band_lo=np.full_like(xi_grid, np.nan),
            band_hi=np.full_like(xi_grid, np.nan),
            residual_rms=np.nan,
            residual_median=np.nan,
            collapse_fraction_5pct=0.0,
            collapse_fraction_10pct=0.0,
            n_cells_used=n_used,
        )

    with np.errstate(all="ignore"):
        master = np.nanmedian(Q, axis=0)
        lo = np.nanquantile(Q, 0.05, axis=0)
        hi = np.nanquantile(Q, 0.95, axis=0)

    # residuals
    deviations = Q - master[None, :]
    finite = np.isfinite(deviations)
    if finite.sum() == 0:
        rms, med, f5, f10 = np.nan, np.nan, 0.0, 0.0
    else:
        flat = deviations[finite]
        rms = float(np.sqrt(np.mean(flat ** 2)))
        med = float(np.median(np.abs(flat)))
        f5 = float(np.mean(np.abs(flat) < 0.05))
        f10 = float(np.mean(np.abs(flat) < 0.10))

    return CollapseResult(
        xi_grid=xi_grid,
        master_curve=master,
        band_lo=lo,
        band_hi=hi,
        residual_rms=rms,
        residual_median=med,
        collapse_fraction_5pct=f5,
        collapse_fraction_10pct=f10,
        n_cells_used=n_used,
    )


# ---------------------------------------------------------------------------
# Parametric master-curve fit (optional refinement)
# ---------------------------------------------------------------------------


def fit_parametric_master(
    xi_grid: np.ndarray,
    master_curve: np.ndarray,
    form: str = "power_exp",
) -> dict:
    """Fit a simple parametric model to the master curve.

    Forms:
      - 'linear': q = 1 - a·ξ
      - 'sqrt':   q = 1 - a·√ξ
      - 'power_exp': q = 1 - a·ξ^b · exp(-c·ξ)
    """
    from scipy.optimize import curve_fit

    valid = np.isfinite(master_curve)
    xi = xi_grid[valid]
    q = master_curve[valid]

    if form == "linear":
        def f(x, a):
            return 1.0 - a * x
        p0 = [0.1]
    elif form == "sqrt":
        def f(x, a):
            return 1.0 - a * np.sqrt(x + 1e-9)
        p0 = [0.1]
    elif form == "power_exp":
        def f(x, a, b, c):
            return 1.0 - a * (x + 1e-9) ** b * np.exp(-c * x)
        p0 = [0.1, 0.5, 0.01]
    else:
        raise ValueError(f"unknown form {form!r}")

    try:
        popt, pcov = curve_fit(f, xi, q, p0=p0, maxfev=5000)
        q_pred = f(xi, *popt)
        r2 = 1.0 - np.sum((q - q_pred) ** 2) / max(np.sum((q - q.mean()) ** 2), 1e-15)
    except Exception:
        popt, r2 = p0, np.nan

    return {"form": form, "params": [float(p) for p in popt], "r2": float(r2)}


# ---------------------------------------------------------------------------
# Convenience: full scaling analysis
# ---------------------------------------------------------------------------


def scaling_analysis(
    cycles: np.ndarray,
    capacities: np.ndarray,
    Q0: np.ndarray,
    N_star: np.ndarray,
    xi_max: float = 2.0,
    n_xi: int = 200,
) -> CollapseResult:
    """One-call wrapper: rescale + fit master + metrics."""
    xi_grid, q_rs, mask = rescale_curves(
        cycles, capacities, Q0, N_star, xi_max=xi_max, n_xi=n_xi,
    )
    return fit_master_curve(xi_grid, q_rs, mask)
