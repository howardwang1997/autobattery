#!/usr/bin/env python3
"""
LIB signature ablation — cross-chemistry validation.
Same as 97 but on LIB fullfield data.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/signature_ablation")
DATA_PATH = Path("data/fullfield/fullfield_lib.h5")
SEED = 42
np.random.seed(SEED)


def main():
    logger.info("LIB Signature Ablation")
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
        param_names_file = list(f.attrs.get("param_names", [f"p{i}" for i in range(params.shape[1])]))

    PARAM_NAMES = [n.split("[")[0].strip()[:20] for n in param_names_file]
    n_params = params.shape[1]
    n_sims = len(V)
    logger.info(f"Loaded {n_sims} sims, {n_params} params: {PARAM_NAMES}")

    # Build signatures
    theta_median = np.median(params, axis=0)
    signatures = np.zeros((n_params, V.shape[1]))
    for j in range(n_params):
        lo = params[:, j] < theta_median[j]
        hi = ~lo
        if lo.sum() > 0 and hi.sum() > 0:
            dtheta = params[hi, j].mean() - params[lo, j].mean()
            dV = V[hi].mean(axis=0) - V[lo].mean(axis=0)
            signatures[j] = dV / (dtheta + 1e-20)
        signatures[j] /= (np.linalg.norm(signatures[j]) + 1e-20)

    # Test: predict params from V via signatures
    n_test = min(300, n_sims // 4)
    idx = np.random.permutation(n_sims)
    idx_test, idx_train = idx[:n_test], idx[n_test:]

    results = {}

    # Full model
    model_full = Ridge(alpha=1.0)
    model_full.fit(params[idx_train], V[idx_train])
    V_pred = model_full.predict(params[idx_test])
    r2_full = float(r2_score(V[idx_test].flatten(), V_pred.flatten()))
    results["all_7param"] = {"r2": r2_full, "n_params": n_params}
    logger.info(f"Full 7-param R²={r2_full:.4f}")

    # Forward selection
    forward = []
    selected = []
    remaining = list(range(n_params))
    for step in range(n_params):
        best_r2, best_j = -999, None
        for j in remaining:
            trial = selected + [j]
            model = Ridge(alpha=1.0)
            model.fit(params[idx_train][:, trial], V[idx_train])
            V_p = model.predict(params[idx_test][:, trial])
            r2 = r2_score(V[idx_test].flatten(), V_p.flatten())
            if r2 > best_r2:
                best_r2, best_j = r2, j
        selected.append(best_j)
        remaining.remove(best_j)
        forward.append({"step": step + 1, "added": PARAM_NAMES[best_j], "r2": float(best_r2)})
        logger.info(f"  Step {step+1}: added {PARAM_NAMES[best_j]:20s} R²={best_r2:.4f}")
    results["forward_selection"] = forward

    # Leave-one-out
    loo = {}
    for j in range(n_params):
        keep = [i for i in range(n_params) if i != j]
        model = Ridge(alpha=1.0)
        model.fit(params[idx_train][:, keep], V[idx_train])
        V_p = model.predict(params[idx_test][:, keep])
        r2 = r2_score(V[idx_test].flatten(), V_p.flatten())
        loo[PARAM_NAMES[j]] = {"delta_r2": float(r2_full - r2)}
        logger.info(f"  LOO {PARAM_NAMES[j]:20s}: ΔR²={r2_full - r2:.4f}")
    results["leave_one_out"] = loo

    # Subset sweep
    sweep = {}
    for k in range(1, n_params + 1):
        best_r2, best_set = -999, None
        for combo in combinations(range(n_params), k):
            model = Ridge(alpha=1.0)
            model.fit(params[idx_train][:, list(combo)], V[idx_train])
            V_p = model.predict(params[idx_test][:, list(combo)])
            r2 = r2_score(V[idx_test].flatten(), V_p.flatten())
            if r2 > best_r2:
                best_r2, best_set = r2, combo
        names_k = [PARAM_NAMES[i] for i in best_set]
        sweep[f"k{k}"] = {"r2": float(best_r2), "params": names_k}
        logger.info(f"  k={k}: best R²={best_r2:.4f} params={names_k}")
    results["subset_sweep"] = sweep

    with open(OUTPUT_DIR / "lib_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Done")


if __name__ == "__main__":
    main()
