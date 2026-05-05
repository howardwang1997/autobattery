#!/usr/bin/env python3
"""
Empirical Profile Likelihood Analysis (no surrogate model).

Uses the simulation database directly: for each test case and each parameter,
sweeps θ_i and finds the minimum residual ||V(θ) - V_ref||² over the database.

This gives ground-truth profile likelihood shapes without surrogate approximation errors.
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
N_TEST_CASES = 10
N_GRID = 30
np.random.seed(SEED)


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
    return params, V


def empirical_profile(params, V, test_idx, param_idx, n_grid=N_GRID):
    """Compute empirical profile likelihood for parameter param_idx
    using test sample test_idx as reference.
    """
    V_ref = V[test_idx]
    theta_ref = params[test_idx]

    log_params = np.log10(np.clip(params, 1e-30, None))
    log_ref = np.log10(np.clip(theta_ref[param_idx], 1e-30, None))

    log_grid = np.linspace(log_ref - 1.5, log_ref + 1.5, n_grid)
    profile = np.zeros(n_grid)

    for gi, log_val in enumerate(log_grid):
        dists = np.abs(log_params[:, param_idx] - log_val)
        mask = dists < 0.2
        if mask.sum() == 0:
            mask = dists < 0.5
        if mask.sum() == 0:
            mask = dists < 1.0

        if mask.sum() == 0:
            profile[gi] = np.inf
            continue

        residuals = np.sum((V[mask] - V_ref) ** 2, axis=1)
        profile[gi] = residuals.min()

    valid = profile < np.inf
    if valid.sum() < 3:
        return log_grid, profile, np.zeros(n_grid), {}

    min_loss = profile[valid].min()
    chi2 = np.full(n_grid, np.inf)
    if min_loss > 0:
        chi2[valid] = (profile[valid] - min_loss) / min_loss * 1000
    else:
        chi2[valid] = 0

    threshold = 3.84
    below = chi2 <= threshold
    if below.any():
        indices = np.where(below)[0]
        ci_low = 10.0 ** log_grid[indices[0]]
        ci_high = 10.0 ** log_grid[indices[-1]]
        ci_width = log_grid[indices[-1]] - log_grid[indices[0]]
    else:
        ci_low = ci_high = None
        ci_width = float("inf")

    flatness = 0
    if valid.sum() > 2:
        valid_losses = profile[valid]
        flatness = (valid_losses.max() - valid_losses.min()) / (valid_losses.min() + 1e-30)

    return log_grid, profile, chi2, {
        "ci_low": float(ci_low) if ci_low is not None else None,
        "ci_high": float(ci_high) if ci_high is not None else None,
        "ci_width_log10": float(ci_width),
        "flatness_ratio": float(flatness),
        "n_valid_grid_points": int(valid.sum()),
        "is_bounded": ci_low is not None,
    }


def main():
    params, V = load_data()
    logger.info("Data: %d samples, %d params", *params.shape)

    test_indices = np.random.choice(len(params), N_TEST_CASES, replace=False)

    all_results = {}
    param_summaries = {}

    for pi, pname in enumerate(PARAM_NAMES):
        widths = []
        flatnesses = []
        bounded = 0

        for ci, ti in enumerate(test_indices):
            log_grid, profile, chi2, info = empirical_profile(params, V, ti, pi)

            case_key = f"case_{ci}"
            if case_key not in all_results:
                all_results[case_key] = {
                    "test_idx": int(ti),
                    "theta_true": params[ti].tolist(),
                    "profiles": {},
                }
            all_results[case_key]["profiles"][pname] = {
                "log_grid": log_grid.tolist(),
                "losses": [float(x) if np.isfinite(x) else None for x in profile],
                "chi2_scaled": [float(x) if np.isfinite(x) else None for x in chi2],
                "info": info,
            }

            if info["is_bounded"]:
                widths.append(info["ci_width_log10"])
                bounded += 1
            flatnesses.append(info["flatness_ratio"])

        param_summaries[pname] = {
            "mean_ci_width": float(np.mean(widths)) if widths else float("inf"),
            "median_ci_width": float(np.median(widths)) if widths else float("inf"),
            "mean_flatness": float(np.mean(flatnesses)),
            "median_flatness": float(np.median(flatnesses)),
            "bounded_fraction": float(bounded / N_TEST_CASES),
        }

    logger.info("=" * 60)
    logger.info("EMPIRICAL PROFILE LIKELIHOOD SUMMARY")
    logger.info("=" * 60)
    logger.info("%-12s %10s %10s %10s %10s",
                "Parameter", "CI width", "Flatness", "Bounded", "Rank")
    logger.info("-" * 52)

    ranked = sorted(param_summaries.items(), key=lambda x: x[1]["median_flatness"])
    for rank, (pname, s) in enumerate(ranked):
        logger.info("%-12s %10.2f %10.2f %9.0f/%d %10d",
                     pname,
                     s["median_ci_width"],
                     s["median_flatness"],
                     s["bounded_fraction"] * N_TEST_CASES,
                     N_TEST_CASES,
                     rank + 1)

    output = {
        "param_summaries": param_summaries,
        "cases": all_results,
        "rank_by_flatness": [p[0] for p in ranked],
        "config": {"n_test_cases": N_TEST_CASES, "n_grid": N_GRID},
    }

    out_path = OUTPUT_DIR / "empirical_profile_likelihood_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
