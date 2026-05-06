#!/usr/bin/env python3
"""Profile Likelihood using GP surrogate on fullfield simulation data.

Adapts the profile likelihood framework (Raue et al., 2009) to work
directly with the simulation database, bypassing the need for PyBaMM
forward solves or experimental data.

Uses Gaussian Process regression as the surrogate forward model,
which provides:
  - Better interpolation than MLP for small datasets (1200 samples)
  - Built-in uncertainty quantification
  - Exact fit at training points

Algorithm:
  1. Train GP surrogate: θ → V(t) (via PCA compression)
  2. For each test sample, find global MAP (already known = true params)
  3. For each parameter, sweep θ_i and optimize others via L-BFGS
  4. Profile: Δχ²(θ_i) = min_{θ_{-i}} ||V_pred - V_true||² / σ² - χ²_min
  5. Flat profile = unidentifiable; sharp V = identifiable
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from itertools import product
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/profile_likelihood")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")

PARAM_NAMES = ["SEI", "LAM_neg", "LAM_pos", "D_n", "D_p", "t+", "R_mult"]
CONFIDENCE_95 = 3.84

SEED = 42
N_TEST = 3
N_GRID = 25
N_PCA = 10
SIGMA_V = 0.005  # 5 mV noise assumption

np.random.seed(SEED)


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
    return params, V


class GPSurrogate:
    """GP surrogate via PCA compression of V(t)."""

    def __init__(self, n_pca=N_PCA):
        self.n_pca = n_pca
        self.pca = None
        self.scaler_p = None
        self.gp = None
        self.v_mean = None
        self.v_std = None

    def fit(self, params, V):
        self.v_mean = V.mean(axis=0)
        self.v_std = V.std(axis=0) + 1e-8
        V_norm = (V - self.v_mean) / self.v_std

        self.pca = PCA(n_components=self.n_pca)
        V_pca = self.pca.fit_transform(V_norm)

        log_p = np.log10(np.clip(params, 1e-30, None))
        self.scaler_p = StandardScaler()
        P_scaled = self.scaler_p.fit_transform(log_p)

        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(P_scaled.shape[1]),
                                                nu=2.5) + WhiteKernel(0.01)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, random_state=SEED,
        )
        self.gp.fit(P_scaled, V_pca)

        V_pca_pred = self.gp.predict(P_scaled)
        V_norm_pred = self.pca.inverse_transform(V_pca_pred)
        V_pred = V_norm_pred * self.v_std + self.v_mean
        r2 = 1 - np.sum((V - V_pred) ** 2) / np.sum((V - self.v_mean) ** 2)

        logger.info("GP surrogate R²=%.4f (PCA %d components, explained var=%.3f)",
                     r2, self.n_pca, self.pca.explained_variance_ratio_.sum())
        return r2

    def predict(self, theta):
        log_p = np.log10(np.clip(theta, 1e-30, None)).reshape(1, -1)
        P_scaled = self.scaler_p.transform(log_p)
        V_pca_pred = self.gp.predict(P_scaled)
        V_norm_pred = self.pca.inverse_transform(V_pca_pred.reshape(1, -1))
        return (V_norm_pred * self.v_std + self.v_mean).flatten()


def profile_one_param(surrogate, theta_true, V_true, param_idx):
    log_true = np.log10(np.clip(theta_true, 1e-30, None))
    log_val = log_true[param_idx]

    log_grid = np.linspace(log_val - 1.5, log_val + 1.5, N_GRID)
    rmses = np.zeros(N_GRID)

    free_idx = [i for i in range(len(theta_true)) if i != param_idx]

    for gi, log_fixed in enumerate(log_grid):
        best_rmse = float("inf")

        for trial in range(3):
            log_start = log_true.copy()
            if trial > 0:
                log_start[free_idx] += np.random.randn(len(free_idx)) * 0.2

            def objective(log_free):
                log_full = log_start.copy()
                log_full[param_idx] = log_fixed
                log_full[free_idx] = log_free
                theta = 10.0 ** log_full
                V_pred = surrogate.predict(theta)
                return np.sum((V_pred - V_true) ** 2)

            x0 = log_start[free_idx]
            try:
                res = minimize(objective, x0, method="L-BFGS-B",
                               options={"maxiter": 100, "ftol": 1e-10})
                rmse = np.sqrt(res.fun / len(V_true))
                if rmse < best_rmse:
                    best_rmse = rmse
            except Exception:
                pass

        rmses[gi] = best_rmse

    chi2 = (rmses ** 2) / (SIGMA_V ** 2) * len(V_true)
    chi2_min = chi2.min()
    delta_chi2 = chi2 - chi2_min

    below_95 = delta_chi2 <= CONFIDENCE_95
    if below_95.any():
        idx = np.where(below_95)[0]
        ci_low = 10.0 ** log_grid[idx[0]]
        ci_high = 10.0 ** log_grid[idx[-1]]
        ci_width_decades = log_grid[idx[-1]] - log_grid[idx[0]]
    else:
        ci_low = ci_high = None
        ci_width_decades = float("inf")

    flat = bool(delta_chi2.max() < CONFIDENCE_95)

    return {
        "log_grid": log_grid.tolist(),
        "rmses_V": rmses.tolist(),
        "delta_chi2": delta_chi2.tolist(),
        "delta_chi2_max": float(delta_chi2.max()),
        "ci_low": float(ci_low) if ci_low is not None else None,
        "ci_high": float(ci_high) if ci_high is not None else None,
        "ci_width_decades": float(ci_width_decades),
        "flat_at_95pct": flat,
        "identifiable": not flat,
    }


def main():
    params, V = load_data()
    logger.info("Data: %d samples, %d params, %d V-points", *params.shape)

    surrogate = GPSurrogate(n_pca=N_PCA)
    surrogate_r2 = surrogate.fit(params, V)

    if surrogate_r2 < 0.95:
        logger.warning("Surrogate R²=%.3f < 0.95, results may be unreliable", surrogate_r2)

    test_indices = np.random.choice(len(params), N_TEST, replace=False)

    all_profiles = {}
    summary = {}

    for ci, ti in enumerate(test_indices):
        theta_true = params[ti]
        V_true = V[ti]
        case_id = f"case_{ci}"

        logger.info("Test case %d (sample %d)", ci, ti)
        all_profiles[case_id] = {"theta_true": theta_true.tolist(), "profiles": {}}

        for pi, pname in enumerate(PARAM_NAMES):
            logger.info("  Profiling %s ...", pname)
            prof = profile_one_param(surrogate, theta_true, V_true, pi)
            all_profiles[case_id]["profiles"][pname] = prof

            logger.info("    %s: Δχ²_max=%.2f, CI=%.1f dec, flat=%s",
                         pname, prof["delta_chi2_max"], prof["ci_width_decades"],
                         prof["flat_at_95pct"])

    for pname in PARAM_NAMES:
        widths = []
        deltas = []
        n_id = 0
        for ci in range(N_TEST):
            p = all_profiles[f"case_{ci}"]["profiles"][pname]
            if p["ci_width_decades"] < float("inf"):
                widths.append(p["ci_width_decades"])
            deltas.append(p["delta_chi2_max"])
            if p["identifiable"]:
                n_id += 1

        summary[pname] = {
            "mean_ci_width": float(np.mean(widths)) if widths else float("inf"),
            "mean_delta_chi2_max": float(np.mean(deltas)),
            "identifiable_fraction": float(n_id / N_TEST),
            "surrogate_r2": float(surrogate_r2),
        }

    logger.info("=" * 60)
    logger.info("PROFILE LIKELIHOOD SUMMARY (GP surrogate R²=%.3f)", surrogate_r2)
    logger.info("=" * 60)
    logger.info("%-12s %10s %12s %10s %10s",
                "Parameter", "Δχ²_max", "CI (dec)", "% ID", "Verdict")
    logger.info("-" * 56)

    fisher_id = {"D_n", "SEI", "LAM_neg"}
    for pname in PARAM_NAMES:
        s = summary[pname]
        verdict = "ID" if s["identifiable_fraction"] > 0.5 else "UN"
        fisher = "Fisher-ID" if pname in fisher_id else "Fisher-UN"
        logger.info("%-12s %10.2f %12.1f %9.0f%% %10s  (%s)",
                     pname, s["mean_delta_chi2_max"], s["mean_ci_width"],
                     s["identifiable_fraction"] * 100, verdict, fisher)

    output = {
        "surrogate_r2": float(surrogate_r2),
        "n_test_cases": N_TEST,
        "n_grid_points": N_GRID,
        "sigma_v": SIGMA_V,
        "summary": summary,
        "cases": all_profiles,
    }

    out_path = OUTPUT_DIR / "gp_profile_likelihood_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", out_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()[:7]
        for pi, pname in enumerate(PARAM_NAMES):
            ax = axes[pi]
            for ci in range(N_TEST):
                prof = all_profiles[f"case_{ci}"]["profiles"][pname]
                grid = np.array(prof["log_grid"])
                dchi2 = np.array(prof["delta_chi2"])
                ax.semilogy(10.0 ** grid, dchi2 + 1e-10, alpha=0.7)
            ax.axhline(CONFIDENCE_95, color="red", ls="--", lw=1, label="95% CI")
            ax.set_xlabel(pname)
            ax.set_ylabel("Δχ²")
            ax.set_title(pname)
            ax.set_xscale("log")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        axes[-1].axis("off") if len(axes) > 7 else None
        fig.suptitle(f"Profile Likelihood (GP surrogate R²={surrogate_r2:.3f})")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "gp_profile_likelihood.png", dpi=150)
        plt.close(fig)
        logger.info("Figure saved to %s", OUTPUT_DIR / "gp_profile_likelihood.png")
    except Exception as e:
        logger.warning("Plotting failed: %s", e)


if __name__ == "__main__":
    main()
