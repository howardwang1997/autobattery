"""Severson-style early-life prediction baseline.

Reviewers will compare any new LMB cycle-life model against the
Severson et al. (Nature Energy 2019) feature-engineered baseline.
This script computes those features per cell from V/I cycling data and
trains a Ridge / Random Forest / XGBoost regressor on the resulting
feature matrix to predict 80%-of-life cycle (or any user-supplied EOL
threshold).

Inputs:  a directory of per-cell data in either Neware xlsx or numpy
         (each file = one cell).
Outputs: under ``outputs/baselines/severson/``
         ├── features.csv     — per-cell features
         ├── cv_metrics.json  — leave-one-cell-out RMSE, MAPE
         └── predictions.csv  — fold-by-fold predictions

Run ``python scripts/31_baseline_severson_features.py --help`` for
options. CPU-only; sized to run on a workstation but trivially
parallelisable via ``scripts/h20/06_severson_baseline.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("severson_baseline")


# Severson feature set: features computed from cycle 10 / cycle 100
# (or cycle 2 / cycle k chosen by --feature-cycles).
FEATURE_KEYS = [
    "log_var_dQ100_10",     # log10(Var(ΔQ_100−10))   ← Severson's headline feature
    "min_dQ100_10",         # min over voltage of ΔQ_100−10
    "skew_dQ100_10",
    "kurtosis_dQ100_10",
    "slope_q_2_to_100",     # linear slope of capacity 2..100
    "intercept_q_2_to_100",
    "discharge_q_2",
    "max_discharge_q",
    "min_charge_t_2_to_100",
    "internal_resistance_2_min_100",
    "ce_avg_first_100",     # coulombic efficiency
]


@dataclass
class CellRecord:
    cell_id: str
    cycles: np.ndarray            # int, sorted
    capacity_Ah: np.ndarray       # per-cycle discharge capacity
    coulombic_efficiency: np.ndarray
    voltage_curves: list[np.ndarray]   # per-cycle resampled discharge voltage
    discharge_time_s: np.ndarray       # per-cycle
    eol_cycle: Optional[int] = None    # cached


def _resample(v: np.ndarray, n: int = 200) -> np.ndarray:
    if len(v) < 2:
        return np.full(n, np.nan)
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, np.linspace(0.0, 1.0, len(v)), v)


def _per_cycle_summary(
    raw: dict[str, np.ndarray], v_grid_n: int = 200,
) -> CellRecord | None:
    if "cycle" not in raw:
        return None
    cycles = raw["cycle"].astype(np.int64)
    unique = np.unique(cycles[cycles > 0])

    cyc_list, cap_list, ce_list, v_list, dur_list = [], [], [], [], []
    for c in unique:
        # discharge: current < 0
        d_mask = (cycles == c) & (raw["current"] < -0.005)
        # charge:    current > 0
        c_mask = (cycles == c) & (raw["current"] > 0.005)
        if d_mask.sum() < 10 or c_mask.sum() < 5:
            continue
        t_d, v_d, i_d = raw["time"][d_mask], raw["voltage"][d_mask], raw["current"][d_mask]
        t_c, v_c, i_c = raw["time"][c_mask], raw["voltage"][c_mask], raw["current"][c_mask]
        valid_d = ~np.isnan(v_d)
        t_d, v_d, i_d = t_d[valid_d], v_d[valid_d], i_d[valid_d]
        if len(t_d) < 10:
            continue
        dt_d = np.diff(t_d)
        cap_d = -np.sum(i_d[:-1] * dt_d) / 3600.0
        if cap_d < 1e-4:
            continue
        dt_c = np.diff(t_c)
        cap_c = np.sum(i_c[:-1] * dt_c) / 3600.0
        ce = cap_d / max(cap_c, 1e-9)

        cyc_list.append(int(c))
        cap_list.append(float(cap_d))
        ce_list.append(float(ce))
        v_list.append(_resample(v_d, v_grid_n))
        dur_list.append(float(t_d[-1] - t_d[0]))

    if len(cyc_list) == 0:
        return None
    order = np.argsort(cyc_list)
    return CellRecord(
        cell_id="",
        cycles=np.array(cyc_list)[order],
        capacity_Ah=np.array(cap_list)[order],
        coulombic_efficiency=np.array(ce_list)[order],
        voltage_curves=[v_list[i] for i in order],
        discharge_time_s=np.array(dur_list)[order],
    )


def _features(
    record: CellRecord, cycle_a: int = 10, cycle_b: int = 100,
) -> Optional[dict[str, float]]:
    cyc = record.cycles
    idx_a = np.where(cyc == cycle_a)[0]
    idx_b = np.where(cyc == cycle_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None
    v_a = record.voltage_curves[idx_a[0]]
    v_b = record.voltage_curves[idx_b[0]]
    if np.any(np.isnan(v_a)) or np.any(np.isnan(v_b)):
        return None

    # Severson uses ΔQ(V) — discharge capacity as a function of voltage —
    # but with V/I-only data we approximate with ΔV(t).
    dQ = v_b - v_a
    log_var = float(np.log10(np.var(dQ) + 1e-12))
    minimum = float(dQ.min())
    skew = float(((dQ - dQ.mean()) ** 3).mean() / (dQ.std() ** 3 + 1e-12))
    kurt = float(((dQ - dQ.mean()) ** 4).mean() / (dQ.std() ** 4 + 1e-12))

    # Capacity slope/intercept over cycles 2..100.
    win = (cyc >= 2) & (cyc <= cycle_b)
    if win.sum() < 5:
        return None
    cyc_w = cyc[win].astype(np.float64)
    cap_w = record.capacity_Ah[win]
    slope, intercept = np.polyfit(cyc_w, cap_w, 1)

    # Discharge capacity at cycle 2.
    idx_2 = np.where(cyc == 2)[0]
    discharge_q_2 = float(record.capacity_Ah[idx_2[0]]) if len(idx_2) > 0 else float("nan")
    max_q = float(record.capacity_Ah.max())

    min_dur = float(record.discharge_time_s[win].min())
    # Crude internal-resistance proxy: voltage drop at start of discharge.
    ir2 = float(record.voltage_curves[idx_2[0]][0] - record.voltage_curves[idx_2[0]][-1]) \
        if len(idx_2) > 0 else float("nan")
    ir100 = float(v_b[0] - v_b[-1])
    ce_avg = float(record.coulombic_efficiency[win].mean())

    return {
        "log_var_dQ100_10": log_var,
        "min_dQ100_10": minimum,
        "skew_dQ100_10": skew,
        "kurtosis_dQ100_10": kurt,
        "slope_q_2_to_100": float(slope),
        "intercept_q_2_to_100": float(intercept),
        "discharge_q_2": discharge_q_2,
        "max_discharge_q": max_q,
        "min_charge_t_2_to_100": min_dur,
        "internal_resistance_2_min_100": ir2 - ir100,
        "ce_avg_first_100": ce_avg,
    }


def _eol_cycle(record: CellRecord, retention: float = 0.8) -> Optional[int]:
    if len(record.capacity_Ah) == 0:
        return None
    q0 = record.capacity_Ah[:5].mean()
    if q0 <= 0:
        return None
    threshold = retention * q0
    below = np.where(record.capacity_Ah < threshold)[0]
    if len(below) == 0:
        return int(record.cycles[-1])  # right-censored
    return int(record.cycles[below[0]])


def _fit_and_eval(
    X: np.ndarray, y: np.ndarray, cell_ids: Sequence[str], model: str,
) -> dict[str, Any]:
    from src.data.cv_split import leave_one_cell_out

    folds = leave_one_cell_out(cell_ids, val_fraction=0.0)
    preds = np.zeros_like(y, dtype=np.float64)
    truths = np.zeros_like(y, dtype=np.float64)

    for fold in folds:
        if model == "ridge":
            from sklearn.linear_model import RidgeCV
            est = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        elif model == "rf":
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(n_estimators=200, random_state=0)
        elif model == "xgb":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("xgboost is required for --model xgb") from exc
            est = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                               random_state=0, verbosity=0)
        else:
            raise ValueError(model)
        est.fit(X[fold.train_idx], y[fold.train_idx])
        preds[fold.test_idx] = est.predict(X[fold.test_idx])
        truths[fold.test_idx] = y[fold.test_idx]

    err = preds - truths
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float(np.mean(np.abs(err) / np.maximum(truths, 1)) * 100)
    return {
        "rmse_cycles": rmse,
        "mape_pct": mape,
        "predictions": preds.tolist(),
        "truths": truths.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Directory of per-cell xlsx/csv files")
    parser.add_argument("--feature-cycles", nargs=2, type=int, default=(10, 100),
                        metavar=("CYCLE_A", "CYCLE_B"))
    parser.add_argument("--retention", type=float, default=0.8)
    parser.add_argument("--model", choices=["ridge", "rf", "xgb"], default="rf")
    parser.add_argument("--output", default="outputs/baselines/severson")
    args = parser.parse_args()

    from src.data.loader import ExperimentalDataLoader
    loader = ExperimentalDataLoader()

    cells: list[CellRecord] = []
    feat_rows: list[dict[str, float]] = []
    eol_cycles: list[int] = []

    for path in sorted(Path(args.data_dir).iterdir()):
        if path.suffix not in (".xlsx", ".csv"):
            continue
        cell_id = path.stem
        logger.info("Loading %s", path)
        try:
            raw = loader.load_neware_xlsx(path) if path.suffix == ".xlsx" \
                else loader.load_csv(path)
        except Exception as exc:
            logger.warning("Skipping %s (%s)", path, exc)
            continue

        rec = _per_cycle_summary(raw)
        if rec is None:
            logger.warning("No valid cycles in %s", path)
            continue
        rec.cell_id = cell_id

        feats = _features(rec, *args.feature_cycles)
        eol = _eol_cycle(rec, retention=args.retention)
        if feats is None or eol is None:
            logger.warning("Insufficient data for features in %s", cell_id)
            continue

        cells.append(rec)
        feat_rows.append({"cell_id": cell_id, **feats})
        eol_cycles.append(eol)

    if len(cells) < 3:
        raise SystemExit(
            f"Need at least 3 cells with valid features; got {len(cells)}. "
            "Check --feature-cycles or supply more data."
        )

    X = np.array([[r[k] for k in FEATURE_KEYS] for r in feat_rows])
    y = np.array(eol_cycles, dtype=np.float64)
    cell_ids = [r["cell_id"] for r in feat_rows]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _fit_and_eval(X, y, cell_ids, args.model)
    logger.info("Model %s | leave-one-cell-out RMSE = %.1f cycles, MAPE = %.1f%%",
                args.model, metrics["rmse_cycles"], metrics["mape_pct"])

    with (out_dir / "cv_metrics.json").open("w") as f:
        json.dump({
            "model": args.model,
            "n_cells": len(cells),
            "feature_cycles": list(args.feature_cycles),
            "retention": args.retention,
            "rmse_cycles": metrics["rmse_cycles"],
            "mape_pct": metrics["mape_pct"],
        }, f, indent=2)

    import csv
    with (out_dir / "features.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", *FEATURE_KEYS, "eol_cycle"])
        for row, eol in zip(feat_rows, eol_cycles):
            w.writerow([row["cell_id"], *(row[k] for k in FEATURE_KEYS), eol])
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "true_eol", "pred_eol"])
        for cid, true_v, pred_v in zip(cell_ids, metrics["truths"], metrics["predictions"]):
            w.writerow([cid, int(true_v), float(pred_v)])

    logger.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
