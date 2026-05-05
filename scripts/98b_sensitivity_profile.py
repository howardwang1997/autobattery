#!/usr/bin/env python3
"""
Sensitivity-based Profile Analysis (no surrogate needed).

For each parameter, compute the "sensitivity landscape":
  - Sort samples by θ_i value
  - For each sample, compute residual vs reference
  - Plot residual vs θ_i to see profile shape

Also compute local sensitivity (∂V/∂θ_i) directly from finite differences
in the database, which gives Fisher-like information without a forward model.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/profile_likelihood")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")

PARAM_NAMES = ["SEI", "LAM_neg", "LAM_pos", "D_n", "D_p", "t+", "R_mult"]
SEED = 42


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
    return params, V


def compute_sensitivity_ranking(params, V):
    """Compute sensitivity of V to each parameter using variance decomposition.

    For each parameter θ_i, fit a univariate smooth (binned average) of
    V(t) vs θ_i, and measure how much variance it explains.
    """
    n_samples, n_time = V.shape
    n_params = params.shape[1]
    V_mean = V.mean(axis=0)
    V_total_var = np.sum((V - V_mean) ** 2)

    results = {}
    for pi, pname in enumerate(PARAM_NAMES):
        log_p = np.log10(np.clip(params[:, pi], 1e-30, None))

        n_bins = 30
        bin_edges = np.linspace(log_p.min(), log_p.max(), n_bins + 1)

        bin_means = []
        bin_centers = []
        for b in range(n_bins):
            mask = (log_p >= bin_edges[b]) & (log_p < bin_edges[b + 1])
            if mask.sum() < 2:
                continue
            bin_means.append(V[mask].mean(axis=0))
            bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)

        if len(bin_means) < 3:
            results[pname] = {
                "explained_variance_ratio": 0.0,
                "sensitivity_rank": n_params,
            }
            continue

        bin_means = np.array(bin_means)

        explained = np.zeros(n_samples)
        for si in range(n_samples):
            closest_bin = np.argmin(np.abs(np.array(bin_centers) - log_p[si]))
            explained[si] = np.sum((bin_means[closest_bin] - V_mean) ** 2)

        evr = np.mean(explained) / V_total_var if V_total_var > 0 else 0

        sensitivities = np.diff(bin_means, axis=0)
        mean_sensitivity = np.sqrt(np.mean(sensitivities ** 2))

        results[pname] = {
            "explained_variance_ratio": float(evr),
            "mean_sensitivity": float(mean_sensitivity),
            "n_bins_used": len(bin_means),
        }

        logger.info(
            "  %s: EVR=%.4f, sensitivity=%.4f",
            pname, evr, mean_sensitivity,
        )

    ranked = sorted(results.items(), key=lambda x: x[1].get("explained_variance_ratio", 0), reverse=True)
    for rank, (pname, r) in enumerate(ranked):
        r["sensitivity_rank"] = rank + 1

    return results, ranked


def compute_conditional_profile(params, V, test_idx, param_idx, n_grid=50):
    """Conditional profile: for each θ_i value, find the sample with
    minimum V(t) distance to the reference, allowing other params to vary.
    """
    V_ref = V[test_idx]
    log_p = np.log10(np.clip(params[:, param_idx], 1e-30, None))
    log_ref = log_p[test_idx]

    log_grid = np.linspace(log_p.min(), log_p.max(), n_grid)
    profile = np.zeros(n_grid)

    for gi, log_val in enumerate(log_grid):
        dists = np.abs(log_p - log_val)
        mask = dists < (log_p.max() - log_p.min()) / n_grid * 1.5
        if mask.sum() == 0:
            mask = dists < (log_p.max() - log_p.min()) / n_grid * 3

        if mask.sum() == 0:
            profile[gi] = np.inf
            continue

        residuals = np.sum((V[mask] - V_ref) ** 2, axis=1)
        profile[gi] = residuals.min()

    return log_grid, profile


def compute_marginal_profile(params, V, test_idx, param_idx, n_grid=50):
    """Marginal profile: sweep θ_i, keep other params at true values.
    This shows intrinsic V sensitivity to each parameter.
    """
    V_ref = V[test_idx]
    theta_true = params[test_idx]
    log_true = np.log10(np.clip(theta_true[param_idx], 1e-30, None))

    log_p_range = np.log10(np.clip(params[:, param_idx], 1e-30, None))
    log_grid = np.linspace(log_p_range.min(), log_p_range.max(), n_grid)

    profile = np.zeros(n_grid)
    for gi, log_val in enumerate(log_grid):
        target_val = 10.0 ** log_val
        dists = np.abs(np.log10(np.clip(params[:, param_idx], 1e-30, None)) - log_val)
        closest = np.argmin(dists)
        profile[gi] = np.sum((V[closest] - V_ref) ** 2)

    return log_grid, profile


def main():
    params, V = load_data()
    logger.info("Data: %d samples, %d params, %d V-points", *params.shape)

    logger.info("=" * 60)
    logger.info("PART 1: Sensitivity Ranking (Variance Decomposition)")
    logger.info("=" * 60)
    sens_results, sens_ranked = compute_sensitivity_ranking(params, V)

    logger.info("\nRanked by explained variance:")
    for rank, (pname, r) in enumerate(sens_ranked):
        logger.info("  #%d %s: EVR=%.4f", rank + 1, pname, r["explained_variance_ratio"])

    np.random.seed(SEED)
    test_indices = np.random.choice(len(params), 5, replace=False)

    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Conditional & Marginal Profiles (5 test cases)")
    logger.info("=" * 60)

    profile_data = {}
    for ci, ti in enumerate(test_indices):
        profile_data[f"case_{ci}"] = {"test_idx": int(ti)}

        for pi, pname in enumerate(PARAM_NAMES):
            log_grid_c, profile_c = compute_conditional_profile(params, V, ti, pi)
            log_grid_m, profile_m = compute_marginal_profile(params, V, ti, pi)

            valid_c = profile_c[profile_c < np.inf]
            valid_m = profile_m[profile_m < np.inf]

            dynamic_range_c = (valid_c.max() / (valid_c.min() + 1e-30)) if len(valid_c) > 0 else 1
            dynamic_range_m = (valid_m.max() / (valid_m.min() + 1e-30)) if len(valid_m) > 0 else 1

            profile_data[f"case_{ci}"][pname] = {
                "conditional_dynamic_range": float(dynamic_range_c),
                "marginal_dynamic_range": float(dynamic_range_m),
                "conditional_min": float(valid_c.min()) if len(valid_c) > 0 else None,
                "conditional_max": float(valid_c.max()) if len(valid_c) > 0 else None,
            }

    param_profiles = {}
    for pi, pname in enumerate(PARAM_NAMES):
        dyn_ranges_c = []
        dyn_ranges_m = []
        for ci in range(5):
            d = profile_data[f"case_{ci}"][pname]
            dyn_ranges_c.append(d["conditional_dynamic_range"])
            dyn_ranges_m.append(d["marginal_dynamic_range"])

        param_profiles[pname] = {
            "mean_conditional_dynrange": float(np.mean(dyn_ranges_c)),
            "mean_marginal_dynrange": float(np.mean(dyn_ranges_m)),
            "log10_mean_conditional": float(np.log10(np.mean(dyn_ranges_c) + 1e-30)),
            "log10_mean_marginal": float(np.log10(np.mean(dyn_ranges_m) + 1e-30)),
        }

    ranked_by_profile = sorted(
        param_profiles.items(),
        key=lambda x: x[1]["log10_mean_conditional"],
        reverse=True,
    )

    logger.info("\nRanked by conditional profile dynamic range:")
    for rank, (pname, r) in enumerate(ranked_by_profile):
        logger.info(
            "  #%d %s: log10(dyn_range)=%.2f",
            rank + 1, pname, r["log10_mean_conditional"],
        )

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Method Comparison")
    logger.info("=" * 60)
    logger.info(
        "%-12s %10s %15s %15s",
        "Parameter", "Fisher ID?", "Sens. Rank", "Profile Rank",
    )
    fisher_id = {"D_n", "SEI", "LAM_neg"}
    for pname in PARAM_NAMES:
        s_rank = next(r[pname]["sensitivity_rank"] for r in [dict(sens_ranked)] if pname in r)
        if isinstance(s_rank, dict):
            s_rank = s_rank.get("sensitivity_rank", "?")
        p_rank = next(i + 1 for i, (p, _) in enumerate(ranked_by_profile) if p == pname)
        logger.info(
            "%-12s %10s %15s %15s",
            pname,
            "ID" if pname in fisher_id else "UN",
            s_rank,
            p_rank,
        )

    output = {
        "sensitivity_analysis": sens_results,
        "sensitivity_ranking": [(p, r) for p, r in sens_ranked],
        "profile_analysis": param_profiles,
        "profile_ranking": [(p, r) for p, r in ranked_by_profile],
        "profile_cases": profile_data,
    }

    out_path = OUTPUT_DIR / "sensitivity_profile_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
