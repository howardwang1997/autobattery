#!/usr/bin/env python3
"""
Optimization-based parameter recovery with GP forward model.

Direct test: given V_obs (known theta), minimize ||V_obs - GP(theta)||
to recover theta. Compare ID vs UN recovery quality.

Uses GP trained on fullfield LFP data as forward model.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from scipy.optimize import differential_evolution, minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/parameter_recovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
FISHER_ID = {"SEI", "D_n", "LAM_neg"}
FISHER_UN = {"D_p", "t+", "LAM_pos", "R_mult"}

SEED = 42
np.random.seed(SEED)


def main():
    logger.info("Optimization-based Parameter Recovery (GP forward model)")
    logger.info("=" * 60)

    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)

    n_sims, n_time = V.shape
    n_params = params.shape[1]
    logger.info(f"Loaded {n_sims} sims, V{n_sims}x{n_time}, params {params.shape}")

    n_test = 50
    idx = np.random.permutation(n_sims)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]
    params_train = params[idx_train]
    V_train = V[idx_train]
    params_test = params[idx_test]
    V_test = V[idx_test]

    # Log-transform params for GP
    log_params_train = np.log10(np.abs(params_train) + 1e-20)
    log_params_test = np.log10(np.abs(params_test) + 1e-20)

    # PCA compress V
    n_pca = 10
    pca = PCA(n_components=n_pca)
    Z_train = pca.fit_transform(V_train)
    logger.info(f"PCA({n_pca}) explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Train GP: theta -> PCA(V)
    scaler = StandardScaler()
    log_params_scaled = scaler.fit_transform(log_params_train)

    logger.info("Training GP surrogate on %d samples...", len(log_params_scaled))
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_params)) + WhiteKernel(0.01)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6, random_state=SEED)
    gp.fit(log_params_scaled, Z_train)
    logger.info(f"GP R² = {gp.score(log_params_scaled, Z_train):.4f}")

    param_min = log_params_train.min(axis=0)
    param_max = log_params_train.max(axis=0)
    bounds = list(zip(param_min, param_max))

    results = {}

    for noise_mV in [0, 5]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Noise σ = {noise_mV} mV")
        logger.info(f"{'='*60}")

        noise_std = noise_mV / 1000.0
        np.random.seed(SEED + noise_mV)

        recovery = {pn: {"rel_errors": [], "abs_errors": []} for pn in PARAM_NAMES}
        v_rmses = []

        for ti in range(n_test):
            V_target = V_test[ti] + np.random.randn(n_time) * noise_std
            Z_target = pca.transform(V_target.reshape(1, -1))[0]
            theta_true = params_test[ti]
            log_theta_true = log_params_test[ti]

            def objective(log_theta):
                lt_scaled = scaler.transform(log_theta.reshape(1, -1))
                Z_pred = gp.predict(lt_scaled)[0]
                return np.sum((Z_target - Z_pred) ** 2)

            try:
                res = differential_evolution(
                    objective, bounds, seed=SEED + ti,
                    maxiter=300, tol=1e-10, popsize=20,
                    mutation=(0.5, 1.5), recombination=0.8, polish=True
                )
                theta_opt = 10.0 ** res.x
                v_rmse = np.sqrt(res.fun / n_pca)
            except Exception as e:
                logger.warning(f"  Test {ti}: optimization failed: {e}")
                continue

            for j, pn in enumerate(PARAM_NAMES):
                if abs(theta_true[j]) > 1e-20:
                    rel_err = abs(theta_opt[j] - theta_true[j]) / abs(theta_true[j])
                else:
                    rel_err = abs(theta_opt[j] - theta_true[j])
                recovery[pn]["rel_errors"].append(float(rel_err))
                recovery[pn]["abs_errors"].append(float(abs(theta_opt[j] - theta_true[j])))
            v_rmses.append(float(v_rmse))

            if (ti + 1) % 10 == 0:
                logger.info(f"  {ti+1}/{n_test} done")

        logger.info(f"\n  Recovery results ({len(v_rmses)} converged):")
        summary = {}
        id_errs = []
        un_errs = []
        for j, pn in enumerate(PARAM_NAMES):
            errs = recovery[pn]["rel_errors"]
            if len(errs) > 0:
                med = float(np.median(errs))
                mn = float(np.mean(errs))
                group = "ID" if pn in FISHER_ID else "UN"
                logger.info(f"    {pn:10s}: median_rel_err={med:.4f}, mean={mn:.4f} [{group}]")
                summary[pn] = {"median_rel_error": med, "mean_rel_error": mn,
                               "group": group, "n": len(errs)}
                if group == "ID":
                    id_errs.append(mn)
                else:
                    un_errs.append(mn)

        sep = np.mean(id_errs) / (np.mean(un_errs) + 1e-10) if id_errs and un_errs else 0
        logger.info(f"\n  ID mean rel_err = {np.mean(id_errs):.4f}")
        logger.info(f"  UN mean rel_err = {np.mean(un_errs):.4f}")
        logger.info(f"  UN/ID ratio = {sep:.2f}x")
        logger.info(f"  V RMSE = {np.mean(v_rmses)*1000:.2f} mV")

        results[f"noise_{noise_mV}mV"] = {
            "noise_mV": noise_mV,
            "per_param": summary,
            "id_mean_err": float(np.mean(id_errs)),
            "un_mean_err": float(np.mean(un_errs)),
            "un_id_ratio": float(sep),
            "mean_v_rmse_mV": float(np.mean(v_rmses) * 1000),
            "n_converged": len(v_rmses),
        }

    with open(OUTPUT_DIR / "gp_recovery_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (key, title) in enumerate([(f"noise_0mV", "σ=0 mV"), (f"noise_5mV", "σ=5 mV")]):
        if key not in results:
            continue
        ax = axes[ax_idx]
        s = results[key]["per_param"]
        names = [pn for pn in PARAM_NAMES if pn in s]
        medians = [s[pn]["median_rel_error"] for pn in names]
        colors = ["tab:blue" if s[pn]["group"] == "ID" else "tab:red" for pn in names]
        ax.barh(names, medians, color=colors)
        ax.set_xlabel("Median relative error |θ̂-θ|/|θ|")
        ax.set_title(f"GP optimization recovery ({title})\nblue=Fisher-ID, red=Fisher-UN")
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gp_recovery.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved to %s", OUTPUT_DIR / "gp_recovery.png")

    logger.info("\nDONE")


if __name__ == "__main__":
    main()
