"""Full universality pipeline for the proprietary 10K-cell dataset.

End-to-end run:
    1. Load cycling + metadata CSV
    2. Health check (gate further analysis)
    3. Knee detection → N★ per cell
    4. FPCA + GMM → archetype labels
    5. Scaling collapse globally and within each archetype
    6. Phase diagram + N★ symbolic regression in formulation space
    7. Persist all artefacts under outputs/universality/<run_name>/

Outputs are anonymised by default — formulation features keep their numeric
ranges but feature names are replaced with feature_1, feature_2, … unless
``--keep-names`` is set.

Usage:
    python scripts/102_universality_pipeline.py \\
        --cycling   data/proprietary/cycling.csv \\
        --metadata  data/proprietary/cell_metadata.csv \\
        --output    outputs/universality/run01

Run on H20.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("universality")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycling", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="outputs/universality/run01")
    p.add_argument("--knee-method", default="kneedle", choices=["kneedle", "piecewise"])
    p.add_argument("--max-cycles", type=int, default=2000)
    p.add_argument("--keep-names", action="store_true",
                   help="keep original feature names in output (skip anonymisation)")
    p.add_argument("--no-symbolic", action="store_true",
                   help="skip symbolic regression (saves time)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from src.universality.curves import load_csvs
    from src.universality.knee import detect_knees
    from src.universality.archetype import cluster_archetypes
    from src.universality.scaling import (
        scaling_analysis, fit_parametric_master, rescale_curves, fit_master_curve,
    )
    from src.universality.phase_diagram import fit_phase_diagram, boundary_grid
    from scripts.health_check_inline import health_check  # see below

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load -------------------------------------------------------------
    logger.info("loading cycling=%s, metadata=%s", args.cycling, args.metadata)
    curves = load_csvs(Path(args.cycling), Path(args.metadata),
                       max_cycles=args.max_cycles)

    # anonymise features if requested
    feature_names = sorted(curves.meta.keys())
    if not args.keep_names:
        rename = {n: f"feature_{i+1}" for i, n in enumerate(feature_names)}
        for old, new in rename.items():
            curves.meta[new] = curves.meta.pop(old)
        feature_names = [rename[n] for n in feature_names]
        with (out / "feature_rename.json").open("w") as f:
            json.dump(rename, f, indent=2)

    # Health check (gate) ---------------------------------------------
    rep = health_check(curves)
    with (out / "health_check.json").open("w") as f:
        json.dump(rep, f, indent=2, default=str)

    if not rep["method_verdicts"]["archetype_clustering"]:
        logger.error("dataset fails minimum requirements; aborting")
        return

    # Knees ------------------------------------------------------------
    logger.info("detecting knees (method=%s)", args.knee_method)
    knees = detect_knees(
        curves.cell_ids, curves.cycles, curves.capacities,
        curves.Q0, method=args.knee_method,
    )
    N_star = np.array([k.N_knee if k.has_knee else np.nan for k in knees])
    cyc_80 = curves.cycles_to_first_below(0.8)
    N_star[np.isnan(N_star) & ~np.isnan(cyc_80)] = cyc_80[np.isnan(N_star) & ~np.isnan(cyc_80)]

    # Archetypes -------------------------------------------------------
    logger.info("clustering archetypes")
    retention = curves.capacity_retention()
    arch = cluster_archetypes(retention, cycle_grid=curves.cycles[0], seed=args.seed)
    logger.info("selected k=%d (BIC=%s)", arch.n_archetypes,
                {str(k): round(v, 1) for k, v in arch.bic.items()})

    # Scaling collapse: global + per-archetype -------------------------
    collapse_global = scaling_analysis(
        curves.cycles, curves.capacities, curves.Q0, N_star,
    )
    parametric_global = fit_parametric_master(
        collapse_global.xi_grid, collapse_global.master_curve,
    )

    per_archetype_collapse = {}
    for c in range(arch.n_archetypes):
        mask = arch.labels == c
        if mask.sum() < 10:
            continue
        xi_g, q_rs, valid_m = rescale_curves(
            curves.cycles[mask], curves.capacities[mask],
            curves.Q0[mask], N_star[mask],
        )
        cr = fit_master_curve(xi_g, q_rs, valid_m)
        per_archetype_collapse[c] = {
            "n_cells": int(mask.sum()),
            "n_used": cr.n_cells_used,
            "residual_rms": cr.residual_rms,
            "collapse_fraction_5pct": cr.collapse_fraction_5pct,
            "collapse_fraction_10pct": cr.collapse_fraction_10pct,
            "master_curve": cr.master_curve.tolist(),
            "xi_grid": cr.xi_grid.tolist(),
        }

    # Phase diagram ---------------------------------------------------
    F = curves.feature_matrix(feature_names)
    logger.info("fitting phase diagram on %d features", F.shape[1])
    pd_res = fit_phase_diagram(
        F, arch.labels, N_star, feature_names,
        seed=args.seed, try_symbolic=not args.no_symbolic,
    )

    feature_ranges = {
        n: (float(F[:, j].min()), float(F[:, j].max()))
        for j, n in enumerate(feature_names) if F.shape[0] > 0
    }
    boundary = boundary_grid(
        pd_res.archetype_classifier, feature_ranges, feature_names,
    ) if len(feature_ranges) >= 2 else None

    # Persist ----------------------------------------------------------
    summary = {
        "n_cells": int(curves.n_cells),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "knee_method": args.knee_method,
        "knee_recall": float(np.mean([k.has_knee for k in knees])),
        "n_archetypes": arch.n_archetypes,
        "bic": {str(k): float(v) for k, v in arch.bic.items()},
        "evr": arch.explained_variance_ratio.tolist(),
        "global_collapse": {
            "residual_rms": collapse_global.residual_rms,
            "fraction_5pct": collapse_global.collapse_fraction_5pct,
            "fraction_10pct": collapse_global.collapse_fraction_10pct,
            "parametric": parametric_global,
        },
        "per_archetype_collapse": per_archetype_collapse,
        "phase_diagram": {
            "archetype_accuracy": pd_res.archetype_accuracy,
            "archetype_auc_macro": pd_res.archetype_auc_macro,
            "nstar_r2": pd_res.nstar_r2,
            "nstar_mae": pd_res.nstar_mae,
            "coefficients": pd_res.coefficients,
            "symbolic_law": pd_res.symbolic_law,
        },
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    np.savez_compressed(
        out / "artefacts.npz",
        labels=arch.labels,
        N_star=N_star,
        feature_matrix=F,
        archetype_curves=arch.archetype_curves,
        archetype_band_low=arch.archetype_band_low,
        archetype_band_high=arch.archetype_band_high,
        cycle_grid=arch.cycle_grid,
        master_curve=collapse_global.master_curve,
        xi_grid=collapse_global.xi_grid,
        boundary_labels=(boundary["labels"] if boundary else np.array([])),
    )

    md = _summary_md(summary, has_boundary=boundary is not None)
    (out / "summary.md").write_text(md)
    logger.info("done. summary at %s", out / "summary.md")


def _summary_md(s: dict, has_boundary: bool) -> str:
    L = ["# Universality pipeline summary", ""]
    L.append(f"- Cells: {s['n_cells']}, features: {s['n_features']}")
    L.append(f"- Knee recall: {s['knee_recall']:.1%} ({s['knee_method']})")
    L.append(f"- Archetypes (BIC-selected k): {s['n_archetypes']}")
    L.append("")
    L.append("## Global scaling collapse")
    g = s["global_collapse"]
    L.append(f"- RMS residual: {g['residual_rms']:.4f}")
    L.append(f"- Within ±5%: {g['fraction_5pct']:.1%}")
    L.append(f"- Parametric: {g['parametric']}")
    L.append("")
    L.append("## Per-archetype collapse")
    for c, d in s["per_archetype_collapse"].items():
        L.append(f"  - archetype {c}: n={d['n_cells']}, "
                 f"RMS={d['residual_rms']:.4f}, "
                 f"±5%={d['collapse_fraction_5pct']:.1%}")
    L.append("")
    L.append("## Phase diagram")
    p = s["phase_diagram"]
    L.append(f"- Archetype classifier accuracy: {p['archetype_accuracy']:.3f}")
    L.append(f"- Archetype classifier AUC (macro): {p['archetype_auc_macro']}")
    L.append(f"- N★ regression R²: {p['nstar_r2']}")
    L.append(f"- N★ regression MAE (log10): {p['nstar_mae']}")
    if p["symbolic_law"]:
        L.append(f"- Symbolic law: {p['symbolic_law']}")
    L.append("")
    L.append("## Verdict")
    ok_global = (g["residual_rms"] is not None and g["residual_rms"] < 0.05)
    ok_phase = p["archetype_auc_macro"] is not None and p["archetype_auc_macro"] > 0.8
    ok_nstar = p["nstar_r2"] is not None and p["nstar_r2"] > 0.7
    if ok_global and ok_phase and ok_nstar:
        L.append("**Joule / Nat. Comm. target hit.**")
    elif ok_phase and ok_nstar:
        L.append("**ESM / npj target — global collapse is loose; rely on "
                 "per-archetype collapse instead.**")
    else:
        L.append("**Mixed result — see per-archetype rows; may need more "
                 "data or different feature engineering.**")
    return "\n".join(L)


# we need health_check here; rather than re-import the script, inline it
class _HealthCheckShim:
    """Light shim so this script doesn't need scripts/100 to be a package."""

    @staticmethod
    def __call__(curves):
        from importlib import util
        spec = util.spec_from_file_location(
            "_hc", Path(__file__).parent / "100_health_check.py",
        )
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.health_check(curves)


# inject the shim under the import path used above
import sys, types
mod = types.ModuleType("scripts.health_check_inline")
mod.health_check = _HealthCheckShim()
sys.modules["scripts.health_check_inline"] = mod


if __name__ == "__main__":
    main()
