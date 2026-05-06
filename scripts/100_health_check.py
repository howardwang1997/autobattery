"""Pre-flight health check for a (cycling, formulation) dataset.

Runs the four checks documented in ``docs/plan_universality_paper.md`` and
prints a verdict on which downstream methods are viable.

Usage:
    python scripts/100_health_check.py \\
        --cycling data/cycling.csv \\
        --metadata data/cell_metadata.csv

Or for the Severson dataset:
    python scripts/100_health_check.py --severson data/external/severson/severson_lfp.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.universality.curves import load_csvs, load_severson_h5


def health_check(curves) -> dict:
    n_cells = curves.n_cells
    cycles_per_cell = (~np.isnan(curves.capacities)).sum(axis=1)
    median_cycles = float(np.median(cycles_per_cell))

    cyc_to_80 = curves.cycles_to_first_below(0.8)
    pct_to_eol = float(np.mean(~np.isnan(cyc_to_80)) * 100)

    # formulation features
    feature_names = sorted(curves.meta.keys())
    n_features = len(feature_names)
    if n_features:
        F = curves.feature_matrix(feature_names)
        unique_configs = len(np.unique(np.round(F, 6), axis=0))
    else:
        unique_configs = 0

    # cells per config (approximate via unique row hashing)
    cells_per_config_median = np.nan
    if unique_configs > 0:
        F = curves.feature_matrix(feature_names)
        _, counts = np.unique(np.round(F, 6), axis=0, return_counts=True)
        cells_per_config_median = float(np.median(counts))

    checks = {
        "n_cells": {
            "value": int(n_cells),
            "pass_floor": n_cells >= 500,
            "pass_recommended": n_cells >= 2000,
        },
        "median_cycles_per_cell": {
            "value": median_cycles,
            "pass_floor": median_cycles >= 100,
            "pass_recommended": median_cycles >= 300,
        },
        "pct_reaching_80pct_SOH": {
            "value": pct_to_eol,
            "pass_floor": pct_to_eol >= 30,
            "pass_recommended": pct_to_eol >= 50,
        },
        "n_formulation_features": {
            "value": n_features,
            "pass_floor": n_features >= 1,
            "pass_recommended": n_features >= 3,
        },
        "n_unique_configs": {
            "value": int(unique_configs),
            "pass_floor": unique_configs >= 50,
        },
        "median_cells_per_config": {
            "value": cells_per_config_median,
            "pass_floor": cells_per_config_median >= 3 if not np.isnan(cells_per_config_median) else False,
        },
    }

    # method-level verdicts
    verdicts = {
        "archetype_clustering": checks["n_cells"]["pass_floor"]
            and checks["median_cycles_per_cell"]["pass_floor"],
        "scaling_collapse": checks["n_cells"]["pass_floor"]
            and checks["median_cycles_per_cell"]["pass_recommended"]
            and checks["pct_reaching_80pct_SOH"]["pass_floor"],
        "phase_diagram": checks["n_cells"]["pass_floor"]
            and checks["n_formulation_features"]["pass_floor"]
            and checks["n_unique_configs"]["pass_floor"],
        "symbolic_law": checks["n_cells"]["pass_recommended"]
            and checks["n_formulation_features"]["pass_recommended"]
            and checks["pct_reaching_80pct_SOH"]["pass_floor"],
    }

    if all(verdicts.values()):
        ceiling = "Joule / Nat. Comm. (full pipeline viable)"
    elif verdicts["archetype_clustering"] and verdicts["phase_diagram"]:
        ceiling = "ESM / npj (no clean scaling collapse)"
    elif verdicts["archetype_clustering"]:
        ceiling = "JPS (Severson-only methods paper)"
    else:
        ceiling = "below paper threshold"

    return {
        "checks": checks,
        "method_verdicts": verdicts,
        "ceiling_estimate": ceiling,
        "feature_names": feature_names,
    }


def _print_report(rep: dict) -> None:
    print("\n=== Universality dataset health check ===\n")
    for name, c in rep["checks"].items():
        v = c["value"]
        marks = []
        if "pass_floor" in c:
            marks.append("FLOOR-OK" if c["pass_floor"] else "FLOOR-FAIL")
        if "pass_recommended" in c:
            marks.append("REC-OK" if c["pass_recommended"] else "REC-FAIL")
        print(f"  {name:35s} = {v}    [{', '.join(marks)}]")
    print("\n=== Method viability ===\n")
    for m, ok in rep["method_verdicts"].items():
        print(f"  {m:25s} : {'YES' if ok else 'NO'}")
    print(f"\nEstimated publication ceiling: {rep['ceiling_estimate']}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycling", help="cycling CSV (cell_id, cycle_number, discharge_capacity)")
    p.add_argument("--metadata", help="cell metadata CSV (cell_id, feature_*)")
    p.add_argument("--severson", help="Severson HDF5 (alternative input)")
    p.add_argument("--output", default="outputs/universality/health_check.json")
    p.add_argument("--max-cycles", type=int, default=2000)
    args = p.parse_args()

    if args.severson:
        curves = load_severson_h5(Path(args.severson))
    elif args.cycling and args.metadata:
        curves = load_csvs(Path(args.cycling), Path(args.metadata), max_cycles=args.max_cycles)
    else:
        raise SystemExit("provide either --severson or both --cycling and --metadata")

    rep = health_check(curves)
    _print_report(rep)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"saved report → {out}")


if __name__ == "__main__":
    main()
