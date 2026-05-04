#!/usr/bin/env python3
"""
A2a: XGBoost baseline — V(t) -> 7 degradation parameters.
Compare per-parameter R² and RMSE for identifiable vs unidentifiable params.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
IDENT_IDX = [0, 3, 4]

CHEM_FILES = {
    "LFP": "data/fullfield/fullfield_lfp_degradation.h5",
    "NMC811": "data/fullfield/fullfield_nmc811_degradation.h5",
    "NCA": "data/fullfield/fullfield_nca_degradation.h5",
    "LCO": "data/fullfield/fullfield_lco_degradation.h5",
    "LFP_v2": "data/fullfield/fullfield_lfp_v2_degradation.h5",
}


def load_data(h5_path):
    with h5py.File(h5_path, "r") as f:
        V = f["V"][:].astype(np.float32)
        params = f["params"][:].astype(np.float32)
    params_log = params.copy()
    for i in LOG_PARAMS:
        params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))
    return V, params_log


def run_baselines(name, h5_path):
    logger.info("Processing %s", name)
    V, params = load_data(h5_path)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(V))
    nt = int(len(V) * 0.8)
    train_idx, test_idx = idx[:nt], idx[nt:]

    X_train, X_test = V[train_idx], V[test_idx]
    y_train, y_test = params[train_idx], params[test_idx]

    results = {}

    for model_name, ModelClass, kwargs in [
        ("XGBoost", xgb.XGBRegressor, {
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
            "objective": "reg:squarederror", "tree_method": "hist",
        }),
        ("RandomForest", RandomForestRegressor, {
            "n_estimators": 500, "max_depth": 15, "n_jobs": -1,
        }),
    ]:
        per_param = {}
        for pi, pname in enumerate(PARAM_NAMES):
            model = ModelClass(**kwargs)
            model.fit(X_train, y_train[:, pi])
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test[:, pi], y_pred)
            mae = mean_absolute_error(y_test[:, pi], y_pred)
            rel_mae = mae / (y_test[:, pi].std() + 1e-10)
            per_param[pname] = {"r2": float(r2), "mae": float(mae), "rel_mae": float(rel_mae)}

        id_r2 = np.mean([per_param[PARAM_NAMES[i]]["r2"] for i in IDENT_IDX])
        un_r2 = np.mean([per_param[PARAM_NAMES[i]]["r2"] for i in range(7) if i not in IDENT_IDX])
        results[model_name] = {
            "per_param": per_param,
            "id_r2_avg": float(id_r2),
            "un_r2_avg": float(un_r2),
            "r2_ratio": float(un_r2 / max(id_r2, 1e-6)),
        }
        logger.info(
            "  %s %s: ID R²=%.3f, UN R²=%.3f, ratio=%.2f",
            name, model_name, id_r2, un_r2, un_r2 / max(id_r2, 1e-6),
        )

    return results


def main():
    output_dir = Path("outputs/baselines/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for chem_name, h5_path in CHEM_FILES.items():
        if not Path(h5_path).exists():
            continue
        all_results[chem_name] = run_baselines(chem_name, h5_path)

    # Print summary
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON: XGBoost & RandomForest")
    print("=" * 90)

    for model_name in ["XGBoost", "RandomForest"]:
        print(f"\n--- {model_name}: Per-Parameter R² ---")
        hdr = "{:10s}".format("Chem")
        for pname in PARAM_NAMES:
            hdr += " {:>8s}".format(pname[:6])
        hdr += " {:>8s} {:>8s} {:>6s}".format("ID_avg", "UN_avg", "Ratio")
        print(hdr)
        print("-" * len(hdr))

        for chem_name in sorted(all_results.keys()):
            if model_name not in all_results[chem_name]:
                continue
            r = all_results[chem_name][model_name]
            row = "{:10s}".format(chem_name)
            for pname in PARAM_NAMES:
                row += " {:8.3f}".format(r["per_param"][pname]["r2"])
            row += " {:8.3f} {:8.3f} {:5.2f}x".format(
                r["id_r2_avg"], r["un_r2_avg"], r["r2_ratio"]
            )
            print(row)

    with open(output_dir / "results.json", "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
