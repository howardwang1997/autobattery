#!/usr/bin/env python3
"""
Noise breakdown: systematic sweep of noise levels for parameter recovery.
Find the identifiability breakpoint.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/parameter_recovery")
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")
PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
FISHER_ID = {"SEI", "D_n", "LAM_neg"}
SEED = 42
np.random.seed(SEED)


def main():
    logger.info("Noise Breakdown Sweep")
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)

    n_sims = len(V)
    idx = np.random.permutation(n_sims)
    n_test = 200
    idx_test, idx_train = idx[:n_test], idx[n_test:]
    V_train, V_test = V[idx_train], V[idx_test]
    P_train, P_test = params[idx_train], params[idx_test]

    noise_levels = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
    results = {}

    for noise_mV in noise_levels:
        np.random.seed(SEED + int(noise_mV * 10))
        noise_std = noise_mV / 1000.0

        V_tr = V_train + np.random.randn(*V_train.shape) * noise_std
        V_te = V_test + np.random.randn(*V_test.shape) * noise_std

        model = Ridge(alpha=1.0)
        model.fit(V_tr, P_train)
        P_pred = model.predict(V_te)

        per_param = {}
        id_r2s, un_r2s = [], []
        for j, pn in enumerate(PARAM_NAMES):
            r2 = float(r2_score(P_test[:, j], P_pred[:, j]))
            per_param[pn] = r2
            (id_r2s if pn in FISHER_ID else un_r2s).append(r2)

        id_mean = float(np.mean(id_r2s))
        un_mean = float(np.mean(un_r2s))
        sep = id_mean / (abs(un_mean) + 1e-10)

        results[f"noise_{noise_mV}mV"] = {
            "per_param_r2": per_param,
            "id_mean_r2": id_mean,
            "un_mean_r2": un_mean,
            "separation": sep,
        }
        logger.info(f"  σ={noise_mV:5.1f} mV: ID R²={id_mean:+.4f}, UN R²={un_mean:+.4f}, sep={sep:.2f}x")

    with open(OUTPUT_DIR / "noise_breakdown_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    noises = noise_levels
    for pn in PARAM_NAMES:
        r2s = [results[f"noise_{n}mV"]["per_param_r2"][pn] for n in noises]
        c = "tab:blue" if pn in FISHER_ID else "tab:red"
        axes[0].plot(noises, r2s, "o-", label=pn, color=c, alpha=0.8)
    axes[0].set_xlabel("Noise σ (mV)"); axes[0].set_ylabel("R²")
    axes[0].set_title("Per-param recovery vs noise"); axes[0].legend(fontsize=7, ncol=2)
    axes[0].axhline(0, color="grey", ls=":", lw=0.5); axes[0].grid(alpha=0.3)

    id_means = [results[f"noise_{n}mV"]["id_mean_r2"] for n in noises]
    un_means = [results[f"noise_{n}mV"]["un_mean_r2"] for n in noises]
    seps = [results[f"noise_{n}mV"]["separation"] for n in noises]
    axes[1].plot(noises, id_means, "o-", label="ID mean", color="tab:blue", lw=2)
    axes[1].plot(noises, un_means, "o-", label="UN mean", color="tab:red", lw=2)
    ax2 = axes[1].twinx()
    ax2.plot(noises, seps, "s--", label="Separation ratio", color="tab:green", lw=2)
    ax2.set_ylabel("Separation (×)", color="tab:green")
    axes[1].set_xlabel("Noise σ (mV)"); axes[1].set_ylabel("Mean R²")
    axes[1].set_title("ID/UN separation vs noise"); axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right"); axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "noise_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
