"""Functional PCA + GMM clustering of capacity-retention curves.

Implements the "archetype taxonomy" stage of the universality pipeline.

Pipeline
--------
1. Drop cells with too-short trajectories.
2. Impute missing tail (post-test) by NaN-aware extension or truncation.
3. Centre each curve at its initial value (Q/Q0).
4. FPCA via SVD on the (n_cells, T) retention matrix.
5. GMM on the leading PCA scores; pick *k* by BIC.
6. Return per-cell archetype labels + cluster mean curves + within-cluster
   spread.

The archetype mean curves are the visual deliverable for paper Fig 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ArchetypeResult:
    labels: np.ndarray                     # (n,) int cluster id
    n_archetypes: int
    bic: dict[int, float]                  # k -> BIC
    pca_scores: np.ndarray                 # (n, n_components)
    pca_components: np.ndarray             # (n_components, T)
    pca_mean: np.ndarray                   # (T,)
    explained_variance_ratio: np.ndarray   # (n_components,)
    archetype_curves: np.ndarray           # (k, T) cluster-mean retention curves
    archetype_band_low: np.ndarray         # (k, T) 5th percentile
    archetype_band_high: np.ndarray        # (k, T) 95th percentile
    cycle_grid: np.ndarray                 # (T,) shared cycle axis
    used_indices: np.ndarray               # (n_used,) indices into the input that survived filtering


def _nan_aware_truncate(retention: np.ndarray, min_valid_frac: float) -> tuple[np.ndarray, int]:
    """Find the longest cycle prefix where at least *min_valid_frac* of cells
    have non-NaN data, and truncate to that length."""
    n, T = retention.shape
    valid_per_t = (~np.isnan(retention)).sum(axis=0) / n
    good = np.where(valid_per_t >= min_valid_frac)[0]
    if len(good) == 0:
        return retention[:, :1], 1
    T_eff = int(good.max()) + 1
    return retention[:, :T_eff], T_eff


def _impute_tail(retention: np.ndarray) -> np.ndarray:
    """Carry the last valid value forward to fill NaN tails."""
    out = retention.copy()
    n, T = out.shape
    for i in range(n):
        valid = ~np.isnan(out[i])
        if not valid.any():
            continue
        last = np.where(valid)[0][-1]
        out[i, last + 1:] = out[i, last]
        # also fill any internal NaNs by interpolation
        if (~valid[: last + 1]).any():
            x_v = np.where(valid)[0]
            out[i, : last + 1] = np.interp(np.arange(last + 1), x_v, out[i, x_v])
    return out


def fpca(
    retention: np.ndarray,
    n_components: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based FPCA on a (n_cells, T) retention matrix.

    Returns (scores, components, mean, evr).
    """
    mean = retention.mean(axis=0)
    X = retention - mean
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, len(s))
    scores = (U[:, :k] * s[:k])             # (n, k)
    components = Vt[:k]                     # (k, T)
    var = (s ** 2) / max(retention.shape[0] - 1, 1)
    evr = (var / var.sum())[:k] if var.sum() > 0 else np.zeros(k)
    return scores, components, mean, evr


def _gmm_fit(scores: np.ndarray, k: int, seed: int = 0):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=seed, n_init=5, max_iter=300, reg_covar=1e-6,
    )
    gmm.fit(scores)
    return gmm


def _select_k_by_bic(scores: np.ndarray, k_range: range, seed: int = 0) -> tuple[int, dict[int, float]]:
    bics: dict[int, float] = {}
    for k in k_range:
        try:
            gmm = _gmm_fit(scores, k, seed=seed)
            bics[k] = float(gmm.bic(scores))
        except Exception:
            bics[k] = float("inf")
    best_k = min(bics, key=bics.get)
    return best_k, bics


def cluster_archetypes(
    retention: np.ndarray,
    cycle_grid: Optional[np.ndarray] = None,
    n_components: int = 5,
    k_range: range = range(2, 7),
    min_valid_frac: float = 0.5,
    seed: int = 0,
) -> ArchetypeResult:
    """End-to-end FPCA + GMM clustering on a retention matrix.

    Parameters
    ----------
    retention      : (n, T) Q(N)/Q0; NaN allowed (will be truncated/imputed)
    cycle_grid     : (T,) corresponding cycle axis; defaults to 1..T
    n_components   : how many FPC scores to feed the GMM
    k_range        : cluster counts to scan over for BIC
    min_valid_frac : fraction of cells that must have data at a cycle for
                     that cycle to be kept
    """
    if cycle_grid is None:
        cycle_grid = np.arange(1, retention.shape[1] + 1)

    # filter cells with almost no data
    valid_cells = (~np.isnan(retention)).sum(axis=1) >= max(10, int(retention.shape[1] * 0.1))
    used_idx = np.where(valid_cells)[0]
    R = retention[valid_cells]

    R_trunc, T_eff = _nan_aware_truncate(R, min_valid_frac)
    R_imp = _impute_tail(R_trunc)
    grid_eff = cycle_grid[:T_eff]

    scores, comps, mean, evr = fpca(R_imp, n_components=n_components)
    best_k, bics = _select_k_by_bic(scores, k_range, seed=seed)
    gmm = _gmm_fit(scores, best_k, seed=seed)
    labels_used = gmm.predict(scores)

    # per-cluster mean and 5/95-percentile bands
    arch_mean = np.zeros((best_k, T_eff))
    band_lo = np.zeros((best_k, T_eff))
    band_hi = np.zeros((best_k, T_eff))
    for c in range(best_k):
        m = labels_used == c
        if m.sum() == 0:
            continue
        arch_mean[c] = R_imp[m].mean(axis=0)
        band_lo[c] = np.quantile(R_imp[m], 0.05, axis=0)
        band_hi[c] = np.quantile(R_imp[m], 0.95, axis=0)

    # canonical labelling: order clusters by mean fade at end (slowest = 0)
    order = np.argsort(arch_mean[:, -1])[::-1]
    relabel = {old: new for new, old in enumerate(order)}
    labels_used = np.array([relabel[int(l)] for l in labels_used])
    arch_mean = arch_mean[order]
    band_lo = band_lo[order]
    band_hi = band_hi[order]

    # broadcast labels back into the input length (filtered cells -> -1)
    labels_full = np.full(retention.shape[0], -1, dtype=int)
    labels_full[used_idx] = labels_used

    return ArchetypeResult(
        labels=labels_full,
        n_archetypes=best_k,
        bic=bics,
        pca_scores=scores,
        pca_components=comps,
        pca_mean=mean,
        explained_variance_ratio=evr,
        archetype_curves=arch_mean,
        archetype_band_low=band_lo,
        archetype_band_high=band_hi,
        cycle_grid=grid_eff,
        used_indices=used_idx,
    )
