"""Severson-138 dry run for the universality pipeline.

Validates the archetype clustering + scaling-collapse code on public data
*before* the proprietary 10K-cell run.  Phase diagram is a no-op here
(Severson is single-formulation LFP).

Outputs (under ``outputs/universality/severson/``):
    archetype.json     cluster labels, BIC table, evr
    archetype.png      cluster mean curves with bands
    knees.json         per-cell knee detections
    scaling.json       collapse residual + master curve
    scaling.png        rescaled curves overlaid with master + bands
    summary.md         pass/fail against the dry-run success criteria
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.universality.curves import load_severson_h5
from src.universality.knee import detect_knees
from src.universality.archetype import cluster_archetypes
from src.universality.scaling import scaling_analysis, fit_parametric_master


SUCCESS = {
    "min_archetypes": 2,
    "max_collapse_rms": 0.05,
    "min_knee_recall": 0.80,
}


def _save_archetype_plot(out, arch_res, retention_used, cycle_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")
    for c in range(arch_res.n_archetypes):
        ax.fill_between(
            cycle_grid,
            arch_res.archetype_band_low[c],
            arch_res.archetype_band_high[c],
            color=cmap(c), alpha=0.15,
        )
        ax.plot(cycle_grid, arch_res.archetype_curves[c],
                color=cmap(c), lw=2.0, label=f"archetype {c}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("Q(N) / Q₀")
    ax.set_ylim(0.6, 1.05)
    ax.set_title(f"Severson archetypes (k={arch_res.n_archetypes}, BIC-selected)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _save_scaling_plot(out, xi_grid, q_rescaled, valid_mask, collapse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    Q = q_rescaled[valid_mask]
    for row in Q[:: max(1, len(Q) // 80)]:
        ax.plot(xi_grid, row, color="grey", alpha=0.2, lw=0.5)
    ax.fill_between(
        collapse.xi_grid, collapse.band_lo, collapse.band_hi,
        color="tab:blue", alpha=0.25, label="5–95% band",
    )
    ax.plot(collapse.xi_grid, collapse.master_curve,
            color="tab:blue", lw=2.5, label="master curve Φ(ξ)")
    ax.set_xlabel("ξ = N / N★")
    ax.set_ylabel("q = Q / Q₀")
    ax.set_ylim(0.6, 1.05)
    ax.set_title(
        f"Scaling collapse (n={collapse.n_cells_used} cells, "
        f"RMS={collapse.residual_rms:.3f}, "
        f"{collapse.collapse_fraction_5pct*100:.0f}% within ±5%)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/external/severson/severson_lfp.h5")
    p.add_argument("--output", default="outputs/universality/severson")
    p.add_argument("--knee-method", default="kneedle", choices=["kneedle", "piecewise"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    curves = load_severson_h5(Path(args.data))
    print(f"loaded {curves.n_cells} cells × {curves.n_cycles} cycles")

    # Knee detection ---------------------------------------------------
    knees = detect_knees(
        curves.cell_ids, curves.cycles, curves.capacities,
        curves.Q0, method=args.knee_method,
    )
    knee_recall = float(np.mean([k.has_knee for k in knees]))
    N_star = np.array([
        k.N_knee if k.has_knee else np.nan for k in knees
    ])
    # cells with no detected knee → fall back to the cycle they hit 80% SOH
    cyc_80 = curves.cycles_to_first_below(0.8)
    fb = np.isnan(N_star) & ~np.isnan(cyc_80)
    N_star[fb] = cyc_80[fb]

    with (out / "knees.json").open("w") as f:
        json.dump({
            "method": args.knee_method,
            "n_with_knee": int(sum(k.has_knee for k in knees)),
            "knee_recall": knee_recall,
            "per_cell": [
                {"cell_id": k.cell_id, "N_knee": k.N_knee,
                 "has_knee": k.has_knee, "confidence": k.confidence}
                for k in knees
            ],
        }, f, indent=2, default=str)

    # Archetype clustering ---------------------------------------------
    retention = curves.capacity_retention()
    arch = cluster_archetypes(retention, cycle_grid=curves.cycles[0], seed=args.seed)
    print(f"selected k={arch.n_archetypes} archetypes (BIC: {arch.bic})")

    with (out / "archetype.json").open("w") as f:
        json.dump({
            "n_archetypes": arch.n_archetypes,
            "bic": {str(k): v for k, v in arch.bic.items()},
            "explained_variance_ratio": arch.explained_variance_ratio.tolist(),
            "labels": arch.labels.tolist(),
            "cell_ids": curves.cell_ids.tolist(),
        }, f, indent=2, default=str)
    _save_archetype_plot(out / "archetype.png", arch, retention, arch.cycle_grid)

    # Scaling collapse -------------------------------------------------
    collapse = scaling_analysis(
        curves.cycles, curves.capacities, curves.Q0, N_star,
    )
    parametric = fit_parametric_master(collapse.xi_grid, collapse.master_curve)
    xi_grid, q_rs, mask = (collapse.xi_grid,) + _rescale_again(curves, N_star)

    with (out / "scaling.json").open("w") as f:
        json.dump({
            "n_cells_used": collapse.n_cells_used,
            "residual_rms": collapse.residual_rms,
            "residual_median": collapse.residual_median,
            "collapse_fraction_5pct": collapse.collapse_fraction_5pct,
            "collapse_fraction_10pct": collapse.collapse_fraction_10pct,
            "parametric_master": parametric,
        }, f, indent=2, default=str)
    _save_scaling_plot(out / "scaling.png", collapse.xi_grid, q_rs, mask, collapse)

    # Summary ----------------------------------------------------------
    pass_arch = arch.n_archetypes >= SUCCESS["min_archetypes"]
    pass_scaling = (
        collapse.residual_rms is not None
        and not np.isnan(collapse.residual_rms)
        and collapse.residual_rms <= SUCCESS["max_collapse_rms"]
    )
    pass_knee = knee_recall >= SUCCESS["min_knee_recall"]

    md = [
        "# Severson dry-run summary", "",
        f"- Cells: {curves.n_cells}",
        f"- Knee recall ({args.knee_method}): {knee_recall:.2%}  "
            f"[{'PASS' if pass_knee else 'FAIL'}]",
        f"- Archetypes (BIC): k={arch.n_archetypes}  "
            f"[{'PASS' if pass_arch else 'FAIL'}]",
        f"- Scaling-collapse RMS: {collapse.residual_rms:.4f}  "
            f"[{'PASS' if pass_scaling else 'FAIL'}]",
        f"- Within ±5%: {collapse.collapse_fraction_5pct:.1%}",
        f"- Within ±10%: {collapse.collapse_fraction_10pct:.1%}",
        "",
        f"Parametric master fit ({parametric['form']}): "
            f"params={parametric['params']}, R²={parametric['r2']:.3f}",
        "",
        "## Verdict",
        "",
        ("**Pipeline ready for 10K data**"
         if (pass_arch and pass_scaling and pass_knee)
         else "**Needs investigation before 10K run** — see failed metrics above"),
    ]
    (out / "summary.md").write_text("\n".join(md))
    print((out / "summary.md").read_text())


def _rescale_again(curves, N_star):
    """Helper: rerun rescale just to capture (q_rs, mask) for plotting.
    The full ``scaling_analysis`` only returns the CollapseResult."""
    from src.universality.scaling import rescale_curves
    _, q_rs, mask = rescale_curves(
        curves.cycles, curves.capacities, curves.Q0, N_star,
    )
    return q_rs, mask


if __name__ == "__main__":
    main()
