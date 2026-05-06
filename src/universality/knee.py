"""Knee-point detection for battery Q(N) curves.

Two detectors provided:

* **kneedle** — wraps the Kneedle algorithm (Satopää et al. 2011) for
  capacity-retention curves.  Fast, stateless, good default.
* **piecewise_linear** — fits a two-segment piecewise-linear model via
  brute-force search over breakpoints.  More robust to noisy curves;
  returns the *first* significant breakpoint.

Both return a :class:`KneeResult` per cell.

References
----------
* Fermín-Cueto et al. 2020, *Nat. Energy* 5, 273-280  — first systematic
  knee detection + early-prediction in batteries.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KneeResult:
    cell_id: str
    N_knee: Optional[float]     # cycle number at knee (None if no knee detected)
    Q_knee: Optional[float]     # capacity at the knee cycle
    has_knee: bool
    method: str                 # "kneedle" or "piecewise"
    confidence: float           # 0-1 heuristic score


# ---------------------------------------------------------------------------
# Kneedle detector
# ---------------------------------------------------------------------------


def kneedle_detect(
    cycles: np.ndarray,
    capacities: np.ndarray,
    S: float = 1.0,
    curve: str = "concave",
    direction: str = "decreasing",
    online: bool = False,
) -> Optional[int]:
    """Return the index of the knee in (cycles, capacities).

    Wraps the ``kneed`` package when available; otherwise a pure-numpy
    fallback that implements the Satopää core idea: knee = point of
    maximum signed curvature of the (x, y_normalised) curve.

    Parameters
    ----------
    S        : sensitivity; higher → more knees detected (→ pick first)
    curve    : "concave" (typical for capacity fade)
    direction: "decreasing"
    """
    # try the package first
    try:
        from kneed import KneeLocator

        kl = KneeLocator(
            cycles, capacities,
            S=S, curve=curve, direction=direction, online=online,
        )
        if kl.knee is None:
            return None
        return int(np.argmin(np.abs(cycles - kl.knee)))
    except ImportError:
        pass

    # pure-numpy fallback
    x = cycles.astype(float)
    y = capacities.astype(float)
    n = len(x)
    if n < 5:
        return None

    # normalise to [0, 1] x [0, 1]
    x_n = (x - x[0]) / max(x[-1] - x[0], 1e-9)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1e-9)

    # signed curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}
    dx = np.gradient(x_n)
    dy = np.gradient(y_n)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-15
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    # ignore endpoints (numerically noisy)
    kappa[:3] = 0
    kappa[-3:] = 0
    idx = int(np.argmax(kappa))
    return idx if kappa[idx] > 0.01 else None


# ---------------------------------------------------------------------------
# Piecewise-linear detector
# ---------------------------------------------------------------------------


def piecewise_linear_detect(
    cycles: np.ndarray,
    capacities: np.ndarray,
    min_segment: int = 20,
) -> Optional[int]:
    """Two-segment piecewise-linear fit; return the breakpoint index.

    Brute-force over all feasible breakpoints; pick the one that minimises
    total SSE.  The knee is the point where the *slope changes most*.
    """
    n = len(cycles)
    if n < 2 * min_segment:
        return None

    x = cycles.astype(float)
    y = capacities.astype(float)

    best_sse = np.inf
    best_bp = None

    for bp in range(min_segment, n - min_segment):
        # fit y = a·x + b on each segment (closed-form OLS)
        def _sse(x_s, y_s):
            if len(x_s) < 2:
                return np.inf
            A = np.vstack([x_s, np.ones_like(x_s)]).T
            try:
                coeff, _, _, _ = np.linalg.lstsq(A, y_s, rcond=None)
            except np.linalg.LinAlgError:
                return np.inf
            return float(np.sum((y_s - A @ coeff) ** 2))

        sse = _sse(x[:bp], y[:bp]) + _sse(x[bp:], y[bp:])
        if sse < best_sse:
            best_sse = sse
            best_bp = bp

    if best_bp is None:
        return None

    # sanity: slope must change by at least 10 %
    def _slope(seg_x, seg_y):
        if len(seg_x) < 2:
            return 0.0
        A = np.vstack([seg_x, np.ones_like(seg_x)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, seg_y, rcond=None)
        return float(coeff[0])

    s1 = _slope(x[:best_bp], y[:best_bp])
    s2 = _slope(x[best_bp:], y[best_bp:])
    if abs(s2) < abs(s1) * 1.1 and abs(s1) > 1e-9:
        # slope did not steepen → probably no real knee
        return None

    return best_bp


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def detect_knees(
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    capacities: np.ndarray,
    Q0: np.ndarray,
    method: str = "kneedle",
    eol_threshold: float = 0.8,
    S: float = 1.0,
) -> list[KneeResult]:
    """Detect knee-points for an entire AlignedCurves-like batch.

    Parameters
    ----------
    cell_ids, cycles, capacities : from AlignedCurves
    Q0                           : initial capacity per cell
    method                       : "kneedle" or "piecewise"
    eol_threshold                : the retention level at which a knee is
                                   physically meaningful
    S                            : Kneedle sensitivity
    """
    n_cells = len(cell_ids)
    results: list[KneeResult] = []

    for i in range(n_cells):
        cid = str(cell_ids[i])
        valid = ~np.isnan(capacities[i]) & ~np.isnan(cycles[i])
        cyc = cycles[i, valid].astype(float)
        cap = capacities[i, valid]

        if len(cyc) < 10:
            results.append(KneeResult(cid, None, None, False, method, 0.0))
            continue

        if method == "kneedle":
            idx = kneedle_detect(cyc, cap, S=S)
        elif method == "piecewise":
            idx = piecewise_linear_detect(cyc, cap)
        else:
            raise ValueError(f"unknown method {method!r}")

        if idx is None:
            results.append(KneeResult(cid, None, None, False, method, 0.0))
            continue

        N_knee = float(cyc[idx])
        Q_knee = float(cap[idx])
        # confidence: how much the slope changes at the knee
        def _slope(seg_x, seg_y):
            if len(seg_x) < 2:
                return 0.0
            A = np.vstack([seg_x, np.ones_like(seg_x)]).T
            coeff, _, _, _ = np.linalg.lstsq(A, seg_y, rcond=None)
            return float(coeff[0])

        n_seg = max(len(cyc) // 4, 5)
        s_before = _slope(cyc[max(0, idx - n_seg):idx + 1],
                          cap[max(0, idx - n_seg):idx + 1])
        s_after = _slope(cyc[idx:min(len(cyc), idx + n_seg)],
                         cap[idx:min(len(cyc), idx + n_seg)])

        if abs(s_before) > 1e-9:
            conf = min(abs(s_after / s_before) / 3.0, 1.0)
        else:
            conf = 0.5

        results.append(KneeResult(cid, N_knee, Q_knee, True, method, conf))

    return results
