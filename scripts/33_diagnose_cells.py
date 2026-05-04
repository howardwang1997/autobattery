"""Run end-to-end LMB degradation diagnosis on one or more cells.

Loads a precomputed signature library (built by
``scripts/32_signature_library_lmb.py``), iterates over experimental
cells in a directory, and writes per-cell mode trajectories + an
ablation report.

Usage::

    python scripts/33_diagnose_cells.py \
        --library outputs/diagnosis/signature_library_lmb.npz \
        --data-dir data/raw/lmb_long_cycle \
        --output outputs/diagnosis/cells \
        --bootstrap 100 \
        --ablation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagnose_cells")


def _extract_curves(raw: dict[str, np.ndarray], min_pts: int = 15) -> list[dict]:
    cycles = raw["cycle"].astype(np.int64)
    out: list[dict] = []
    for c in np.unique(cycles[cycles > 0]):
        m = (cycles == c) & (raw["current"] < -0.005)
        if m.sum() < min_pts:
            continue
        t = raw["time"][m]
        v = raw["voltage"][m]
        i = raw["current"][m]
        valid = ~np.isnan(v) & ~np.isnan(t) & ~np.isnan(i)
        t, v, i = t[valid], v[valid], i[valid]
        if len(v) < min_pts:
            continue
        dt = np.diff(t)
        if dt.sum() < 10:
            continue
        cap = -np.sum(i[:-1] * dt) / 3600.0
        if cap < 1e-4:
            continue
        out.append({"cycle": int(c), "voltage": v, "capacity": cap})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="signature library npz")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/diagnosis/cells")
    parser.add_argument("--regressor", choices=["ridge", "nnls", "elastic_net"],
                        default="ridge")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-ref-cycles", type=int, default=5)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ablation", action="store_true",
                        help="Run leave-one-signature-out ablation per cell")
    args = parser.parse_args()

    from src.data.loader import ExperimentalDataLoader
    from src.diagnosis import SignatureLibrary, DegradationDiagnosis

    library = SignatureLibrary.load(args.library)
    diag = DegradationDiagnosis(library, regressor=args.regressor, alpha=args.alpha)
    loader = ExperimentalDataLoader()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for path in sorted(Path(args.data_dir).iterdir()):
        if path.suffix not in (".xlsx", ".csv"):
            continue
        cell_id = path.stem
        logger.info("Diagnosing %s", cell_id)
        raw = loader.load_neware_xlsx(path) if path.suffix == ".xlsx" \
            else loader.load_csv(path)
        curves = _extract_curves(raw)
        if len(curves) < args.n_ref_cycles + 5:
            logger.warning("Skipping %s (only %d valid cycles)", cell_id, len(curves))
            continue

        results = diag.diagnose(
            curves,
            n_ref_cycles=args.n_ref_cycles,
            smooth_sigma=args.smooth_sigma,
            bootstrap_samples=args.bootstrap,
        )

        cell_dir = out_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        cycles = np.array([r.cycle for r in results])
        coeffs = np.array([r.coeffs for r in results])
        rmse = np.array([r.dV_rmse_mV for r in results])
        caps = np.array([r.capacity_Ah if r.capacity_Ah is not None else np.nan
                         for r in results])
        np.savez_compressed(
            cell_dir / "trajectory.npz",
            cycles=cycles, coeffs=coeffs, rmse_mV=rmse, capacity_Ah=caps,
            param_names=np.array(library.param_names),
        )
        with (cell_dir / "diagnosis.json").open("w") as f:
            json.dump(
                {"cell_id": cell_id, "results": [r.to_dict() for r in results]},
                f, indent=2,
            )
        summary[cell_id] = {
            "n_cycles": int(len(results)),
            "mean_rmse_mV": float(np.mean(rmse)),
            "first_cycle": int(cycles.min()),
            "last_cycle": int(cycles.max()),
        }

        if args.ablation:
            ablation = diag.leave_one_out_ablation(
                curves, n_ref_cycles=args.n_ref_cycles,
                smooth_sigma=args.smooth_sigma,
            )
            with (cell_dir / "ablation.json").open("w") as f:
                json.dump(ablation, f, indent=2)
            for name, stats in ablation.items():
                logger.info(
                    "  drop %-30s | full=%.1f mV  drop=%.1f mV  Δ=%.1f mV",
                    name, stats["rmse_full_mV"], stats["rmse_dropped_mV"],
                    stats["delta_mV"],
                )

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to %s/summary.json (n_cells=%d)", out_dir, len(summary))


if __name__ == "__main__":
    main()
