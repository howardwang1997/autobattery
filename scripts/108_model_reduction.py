#!/usr/bin/env python3
"""
Model reduction V(t) prediction test.
Compare: full 7-param model vs ID-only 3-param model vs UN-only 4-param model.
Question: does dropping UN params degrade V(t) prediction?
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/parameter_recovery")
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")
PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
FISHER_ID_IDX = [0, 3, 4]  # D_n, SEI, LAM_neg
FISHER_UN_IDX = [1, 2, 5, 6]  # D_p, t+, LAM_pos, R_mult
SEED = 42
np.random.seed(SEED)


def main():
    logger.info("Model Reduction V(t) Prediction")
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)

    n_sims = len(V)
    idx = np.random.permutation(n_sims)
    n_test = 200
    idx_test, idx_train = idx[:n_test], idx[n_test:]
    V_train, V_test = V[idx_train], V[idx_test]
    P_train, P_test = params[idx_train], params[idx_test]

    configs = {
        "full_7param": list(range(7)),
        "fisher_id_3param": FISHER_ID_IDX,
        "fisher_un_4param": FISHER_UN_IDX,
        "top3_datadriven": [3, 1, 6],  # SEI, D_p, R_mult from forward selection
        "random_3param_a": [0, 2, 5],
        "random_3param_b": [1, 4, 6],
    }

    results = {}
    for name, param_idx in configs.items():
        pnames = [PARAM_NAMES[i] for i in param_idx]
        P_tr = P_train[:, param_idx]
        P_te = P_test[:, param_idx]

        model = Ridge(alpha=1.0)
        model.fit(P_tr, V_train)
        V_pred = model.predict(P_te)

        rmse = np.sqrt(mean_squared_error(V_test, V_pred))
        r2 = r2_score(V_test.flatten(), V_pred.flatten())
        per_time_rmse = np.sqrt(np.mean((V_test - V_pred) ** 2, axis=0)).mean() * 1000

        results[name] = {
            "params": pnames,
            "n_params": len(param_idx),
            "v_rmse_mV": float(rmse * 1000),
            "v_r2": float(r2),
            "per_time_rmse_mV": float(per_time_rmse),
        }
        logger.info(f"  {name:25s} ({len(param_idx)} params): R²={r2:.4f}, RMSE={rmse*1000:.2f} mV")

    # Also: predict V from params, then predict params back
    logger.info("\n--- Inverse: predict params from V ---")
    inverse_results = {}
    for name, param_idx in configs.items():
        P_tr = P_train[:, param_idx]
        P_te = P_test[:, param_idx]
        model_inv = Ridge(alpha=1.0)
        model_inv.fit(V_train, P_tr)
        P_pred = model_inv.predict(V_test)

        r2s = []
        for j in range(len(param_idx)):
            r2s.append(float(r2_score(P_te[:, j], P_pred[:, j])))
        inverse_results[name] = {
            "params": [PARAM_NAMES[i] for i in param_idx],
            "mean_r2": float(np.mean(r2s)),
            "per_param_r2": dict(zip([PARAM_NAMES[i] for i in param_idx], r2s)),
        }
        logger.info(f"  {name:25s}: mean param R²={np.mean(r2s):.4f}")

    all_results = {"forward_v_prediction": results, "inverse_param_recovery": inverse_results}
    with open(OUTPUT_DIR / "model_reduction_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = list(results.keys())
    r2s = [results[n]["v_r2"] for n in names]
    nps = [results[n]["n_params"] for n in names]
    short = [n.replace("_", "\n") for n in names]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "grey", "grey"]
    axes[0].bar(range(len(names)), r2s, color=colors[:len(names)])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(short, fontsize=8)
    axes[0].set_ylabel("V(t) prediction R²")
    axes[0].set_title("Forward: θ → V(t)")
    axes[0].grid(alpha=0.3, axis="y")

    inv_names = list(inverse_results.keys())
    inv_r2s = [inverse_results[n]["mean_r2"] for n in inv_names]
    axes[1].bar(range(len(inv_names)), inv_r2s, color=colors[:len(inv_names)])
    axes[1].set_xticks(range(len(inv_names)))
    axes[1].set_xticklabels([n.replace("_", "\n") for n in inv_names], fontsize=8)
    axes[1].set_ylabel("Mean param recovery R²")
    axes[1].set_title("Inverse: V(t) → θ")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "model_reduction.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
