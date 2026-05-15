#!/usr/bin/env python3
"""
Per-PCA-component GP recovery.
Each PCA component gets its own GP (like 96c), achieving R²=0.999 per component.
Then optimization-based parameter recovery.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/parameter_recovery")
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")
PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
FISHER_ID = {"SEI", "D_n", "LAM_neg"}
FISHER_UN = {"D_p", "t+", "LAM_pos", "R_mult"}
SEED = 42
np.random.seed(SEED)


def main():
    logger.info("Per-PCA GP Recovery")
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)

    n_sims, n_time = V.shape
    n_params = params.shape[1]
    n_test = 50
    idx = np.random.permutation(n_sims)
    idx_test, idx_train = idx[:n_test], idx[n_test:]
    params_train, V_train = params[idx_train], V[idx_train]
    params_test, V_test = params[idx_test], V[idx_test]

    log_params_train = np.log10(np.abs(params_train) + 1e-20)
    log_params_test = np.log10(np.abs(params_test) + 1e-20)

    n_pca = 10
    pca = PCA(n_components=n_pca, random_state=SEED)
    Z_train = pca.fit_transform(V_train)
    cumvar = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA({n_pca}) cumvar={cumvar:.4f}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(log_params_train)

    gps = []
    for k in range(n_pca):
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-4),
            n_restarts_optimizer=2, alpha=1e-8, random_state=SEED
        )
        gp.fit(X_train, Z_train[:, k])
        r2 = gp.score(X_train, Z_train[:, k])
        logger.info(f"  GP[{k}] R²={r2:.4f}")
        gps.append(gp)

    def predict_gp(log_theta):
        x = scaler.transform(log_theta.reshape(1, -1))
        z = np.array([gp.predict(x)[0] for gp in gps])
        return z

    bounds = list(zip(log_params_train.min(axis=0), log_params_train.max(axis=0)))

    results = {}
    for noise_mV in [0, 5]:
        logger.info(f"\n--- σ={noise_mV} mV ---")
        np.random.seed(SEED + noise_mV)
        noise_std = noise_mV / 1000.0
        recovery = {pn: [] for pn in PARAM_NAMES}

        for ti in range(n_test):
            V_target = V_test[ti] + np.random.randn(n_time) * noise_std
            Z_target = pca.transform(V_target.reshape(1, -1))[0]

            def objective(log_theta):
                z_pred = predict_gp(log_theta)
                return np.sum((Z_target - z_pred) ** 2)

            res = differential_evolution(objective, bounds, seed=SEED + ti,
                                          maxiter=500, tol=1e-12, popsize=25, polish=True)
            theta_opt = 10.0 ** res.x
            theta_true = params_test[ti]

            for j, pn in enumerate(PARAM_NAMES):
                rel = abs(theta_opt[j] - theta_true[j]) / (abs(theta_true[j]) + 1e-20)
                recovery[pn].append(float(rel))
            if (ti + 1) % 10 == 0:
                logger.info(f"  {ti+1}/{n_test}")

        summary = {}
        id_errs, un_errs = [], []
        for pn in PARAM_NAMES:
            med = float(np.median(recovery[pn]))
            mn = float(np.mean(recovery[pn]))
            g = "ID" if pn in FISHER_ID else "UN"
            summary[pn] = {"median": med, "mean": mn, "group": g}
            (id_errs if g == "ID" else un_errs).append(mn)
            logger.info(f"  {pn:10s}: med={med:.4f} mean={mn:.4f} [{g}]")

        sep = np.mean(un_errs) / (np.mean(id_errs) + 1e-10)
        logger.info(f"  UN/ID ratio={sep:.2f}x")
        results[f"noise_{noise_mV}mV"] = {"per_param": summary,
            "id_mean": float(np.mean(id_errs)), "un_mean": float(np.mean(un_errs)),
            "ratio": float(sep)}

    with open(OUTPUT_DIR / "per_pca_gp_recovery.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    key = "noise_0mV"
    names = PARAM_NAMES
    meds = [results[key]["per_param"][pn]["median"] for pn in names]
    colors = ["tab:blue" if pn in FISHER_ID else "tab:red" for pn in names]
    ax.barh(names, meds, color=colors)
    ax.set_xlabel("Median relative error")
    ax.set_title("Per-PCA GP optimization recovery (σ=0)\nblue=ID, red=UN")
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "per_pca_gp_recovery.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
