#!/usr/bin/env python3
"""
Multi-rate parameter recovery experiment.

Show that using multiple C-rates improves parameter recovery,
validating the Fisher multi-rate rank gain (3->4).
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

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
    logger.info("Multi-Rate Parameter Recovery")
    logger.info("=" * 60)

    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
        c_rates = f["c_rates"][:].astype(np.float64)

    unique_rates = np.unique(c_rates)
    logger.info(f"C-rates: {unique_rates}")
    logger.info(f"Total: {len(V)} sims")

    n_sims = len(V)
    n_params = params.shape[1]
    n_test = min(200, n_sims // 5)
    idx = np.random.permutation(n_sims)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]

    params_train = params[idx_train]
    params_test = params[idx_test]

    param_range = params_train.max(axis=0) - params_train.min(axis=0)
    param_range[param_range < 1e-20] = 1.0

    results = {}

    # Strategy 1: Single rate (pooled)
    logger.info("\n--- Single rate (all pooled) ---")
    V_train_all = V[idx_train]
    V_test_all = V[idx_test]

    model = Ridge(alpha=1.0)
    model.fit(V_train_all, params_train)
    theta_pred = model.predict(V_test_all)
    norm_err = np.abs(theta_pred - params_test) / param_range

    single_rate = {}
    for j, pn in enumerate(PARAM_NAMES):
        r2 = float(r2_score(params_test[:, j], theta_pred[:, j]))
        single_rate[pn] = {"r2": r2, "norm_err": float(np.mean(norm_err[:, j]))}
        logger.info(f"  {pn:10s}: R²={r2:+.4f}, err={np.mean(norm_err[:, j]):.4f}")

    results["single_rate_pooled"] = single_rate

    # Strategy 2: Per-rate separate models, average predictions
    logger.info("\n--- Per-rate models (average predictions) ---")
    multi_rate_avg = {pn: {"predictions": []} for pn in PARAM_NAMES}

    for rate in unique_rates:
        mask_train = c_rates[idx_train] == rate
        mask_test = c_rates[idx_test] == rate

        if mask_train.sum() < 10 or mask_test.sum() < 5:
            logger.info(f"  C={rate}: train={mask_train.sum()}, test={mask_test.sum()} (skip)")
            continue

        logger.info(f"  C={rate}: train={mask_train.sum()}, test={mask_test.sum()}")

        V_tr = V[idx_train][mask_train]
        P_tr = params[idx_train][mask_train]
        V_te = V[idx_test][mask_test]

        model_r = Ridge(alpha=1.0)
        model_r.fit(V_tr, P_tr)
        theta_pred_r = model_r.predict(V_te)

        for j, pn in enumerate(PARAM_NAMES):
            for i_pred, i_test in enumerate(np.where(mask_test)[0]):
                multi_rate_avg[pn]["predictions"].append(
                    {"pred": float(theta_pred_r[i_pred, j]),
                     "true": float(params_test[i_test, j]),
                     "rate": float(rate)}
                )

    # Compute per-rate-averaged recovery
    multi_rate_results = {}
    for pn in PARAM_NAMES:
        preds = multi_rate_avg[pn]["predictions"]
        if not preds:
            continue
        # Average predictions across rates for same test sample
        from collections import defaultdict
        by_sample = defaultdict(list)
        for p in preds:
            by_sample[p["true"]].append(p["pred"])
        avg_preds = {true: np.mean(preds_list) for true, preds_list in by_sample.items()}
        trues = list(avg_preds.keys())
        preds_avg = list(avg_preds.values())
        r2 = r2_score(trues, preds_avg)
        multi_rate_results[pn] = {"r2": float(r2)}
        logger.info(f"  {pn:10s}: multi-rate R²={r2:+.4f}")

    results["multi_rate_averaged"] = multi_rate_results

    # Strategy 3: Concatenated multi-rate features
    logger.info("\n--- Concatenated multi-rate features ---")
    concat_results = {}

    for rate in unique_rates:
        mask_train = c_rates[idx_train] == rate
        mask_test = c_rates[idx_test] == rate
        if mask_train.sum() < 10 or mask_test.sum() < 5:
            continue

        test_indices_at_rate = np.where(mask_test)[0]
        train_indices_at_rate = np.where(mask_train)[0]

        # Find test sims that have ALL rates available
        break

    # Better approach: group sims by param_set_id
    # Each param_set has multiple C-rates
    with h5py.File(DATA_PATH, "r") as f:
        param_set_ids = f["param_set_ids"][:]

    unique_param_sets = np.unique(param_set_ids)
    logger.info(f"  {len(unique_param_sets)} unique parameter sets")

    # For each param_set, collect V at all rates -> concatenate
    train_sets = set(param_set_ids[idx_train].tolist())
    test_sets = set(param_set_ids[idx_test].tolist())

    V_concat_train = []
    P_concat_train = []
    V_concat_test = []
    P_concat_test = []

    for ps_id in train_sets:
        mask = (param_set_ids == ps_id)
        if mask.sum() == len(unique_rates):
            order = np.argsort(c_rates[mask])
            V_cat = V[mask][order].flatten()
            V_concat_train.append(V_cat)
            P_concat_train.append(params[mask][order][0])

    for ps_id in test_sets:
        mask = (param_set_ids == ps_id)
        if mask.sum() == len(unique_rates):
            order = np.argsort(c_rates[mask])
            V_cat = V[mask][order].flatten()
            V_concat_test.append(V_cat)
            P_concat_test.append(params[mask][order][0])

    if V_concat_train and V_concat_test:
        V_concat_train = np.array(V_concat_train)
        P_concat_train = np.array(P_concat_train)
        V_concat_test = np.array(V_concat_test)
        P_concat_test = np.array(P_concat_test)

        logger.info(f"  Concatenated: train={len(V_concat_train)}, test={len(V_concat_test)}, "
                     f"feature_dim={V_concat_train.shape[1]}")

        model_concat = Ridge(alpha=1.0)
        model_concat.fit(V_concat_train, P_concat_train)
        theta_pred_concat = model_concat.predict(V_concat_test)

        norm_err_concat = np.abs(theta_pred_concat - P_concat_test) / param_range

        for j, pn in enumerate(PARAM_NAMES):
            r2 = float(r2_score(P_concat_test[:, j], theta_pred_concat[:, j]))
            concat_results[pn] = {"r2": r2, "norm_err": float(np.mean(norm_err_concat[:, j]))}
            single_r2 = single_rate.get(pn, {}).get("r2", 0)
            delta = r2 - single_r2
            logger.info(f"  {pn:10s}: concat R²={r2:+.4f} (single={single_r2:+.4f}, Δ={delta:+.4f})")

    results["multi_rate_concatenated"] = concat_results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-RATE vs SINGLE-RATE COMPARISON")
    logger.info("=" * 60)
    for pn in PARAM_NAMES:
        s_r2 = single_rate.get(pn, {}).get("r2", 0)
        c_r2 = concat_results.get(pn, {}).get("r2", 0)
        delta = c_r2 - s_r2
        group = "ID" if pn in FISHER_ID else "UN"
        logger.info(f"  {pn:10s}: single={s_r2:+.4f} -> concat={c_r2:+.4f} (Δ={delta:+.4f}) [{group}]")

    id_delta = np.mean([concat_results.get(pn, {}).get("r2", 0) - single_rate.get(pn, {}).get("r2", 0) for pn in FISHER_ID])
    un_delta = np.mean([concat_results.get(pn, {}).get("r2", 0) - single_rate.get(pn, {}).get("r2", 0) for pn in FISHER_UN])
    logger.info(f"\n  ID mean ΔR² = {id_delta:+.4f}")
    logger.info(f"  UN mean ΔR² = {un_delta:+.4f}")

    with open(OUTPUT_DIR / "multi_rate_recovery_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(PARAM_NAMES))
    w = 0.35
    single_r2s = [single_rate.get(pn, {}).get("r2", 0) for pn in PARAM_NAMES]
    concat_r2s = [concat_results.get(pn, {}).get("r2", 0) for pn in PARAM_NAMES]
    colors_single = ["tab:blue" if pn in FISHER_ID else "tab:red" for pn in PARAM_NAMES]

    ax.bar(x - w/2, single_r2s, w, label="Single-rate", color=[c+"80" for c in ["#1f77b4" if pn in FISHER_ID else "#d62728" for pn in PARAM_NAMES]])
    ax.bar(x + w/2, concat_r2s, w, label="Multi-rate (concat)", color=["#1f77b4" if pn in FISHER_ID else "#d62728" for pn in PARAM_NAMES])
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel("R² (parameter recovery)")
    ax.set_title("Multi-rate vs Single-rate Recovery\nblue=Fisher-ID, red=Fisher-UN")
    ax.legend()
    ax.axhline(0, color="grey", ls=":", lw=0.5)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_rate_recovery.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved")


if __name__ == "__main__":
    main()
