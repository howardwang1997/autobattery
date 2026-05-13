#!/usr/bin/env python3
"""Paper figure generation for Route A (Identifiability) and Route C (Universality).

Reads from existing output JSON files. Generates publication-quality figures.

Route A figures:
  Fig 1: Fisher eigenvalue spectrum across 5 chemistries
  Fig 2: Cross-method parameter ranking comparison
  Fig 3: MIT Severson V(t) manifold effective rank
  Fig 4: Signature ablation (Fisher-guided vs random vs full)
  Fig 5: Profile likelihood (GP surrogate)
  Fig 6: Rank robustness η-sweep

Route C figures:
  Fig 7: Severson archetypes
  Fig 8: Scaling collapse + master curve
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

BASE = Path("/AI4S/Users/howardwang/h204/autobattery")
OUT = BASE / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]


def fig1_rank_robustness():
    """Fig 1: Eigenvalue spectrum + rank vs η for LFP and LIB."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, subpath) in zip(axes, [("LFP", "rank_robustness"), ("LIB", "rank_robustness/lib")]):
        path = BASE / "outputs" / subpath / "rank_table.json"
        if not path.exists():
            logger.warning("Missing %s", path)
            continue
        with open(path) as f:
            data = json.load(f)

        colors = {"raw": "#1f77b4", "log": "#ff7f0e",
                  "log_standardised": "#2ca02c", "pca_whitened": "#d62728"}
        for mode, eigs in data["spectra"].items():
            e = np.array(eigs)
            e_norm = e / e[0]
            ax.semilogy(range(1, len(e) + 1), e_norm, "o-",
                         label=mode.replace("_", " "), color=colors.get(mode, "grey"))

        for eta in [1e-2, 1e-3]:
            ax.axhline(eta, color="grey", ls=":", lw=0.8)
            ax.text(7.1, eta * 1.2, f"η={eta:.0e}", fontsize=7, color="grey")

        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("λᵢ / λ₁")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 1: Fisher Information Spectrum")
    fig.savefig(OUT / "fig1_spectrum.png")
    plt.close(fig)
    logger.info("Fig 1 saved")


def fig2_severson_rank():
    """Fig 2: Severson within-cell effective rank distribution."""
    path = BASE / "outputs/severson_validation/severson_validation_results.json"
    if not path.exists():
        logger.warning("Missing %s", path)
        return
    with open(path) as f:
        data = json.load(f)

    ranks = data["within_cell_rank"]["per_cell_eff_ranks"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(ranks, bins=30, color="tab:blue", edgecolor="white", alpha=0.8)
    axes[0].axvline(3, color="red", ls="--", lw=2, label="Fisher rank=3")
    axes[0].axvline(np.median(ranks), color="orange", ls="-", lw=2,
                     label=f"Median={np.median(ranks):.2f}")
    axes[0].set_xlabel("Effective rank")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Within-cell V(t) manifold rank")
    axes[0].legend()

    pca_data = data["pca_manifold"]["delta_V"]
    cumvar = np.array(pca_data["cumulative_variance"])
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, "o-", color="tab:blue")
    axes[1].axhline(0.95, color="grey", ls="--", lw=1, label="95%")
    axes[1].axhline(0.99, color="grey", ls=":", lw=1, label="99%")
    axes[1].set_xlabel("Number of PCA components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title("ΔV PCA manifold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Fig 2: MIT Severson Low-Rank Validation")
    fig.savefig(OUT / "fig2_severson_rank.png")
    plt.close(fig)
    logger.info("Fig 2 saved")


def fig3_bnn_prediction():
    """Fig 3: BNN prediction results (fade% and cycle_life)."""
    path = BASE / "outputs/severson_validation/severson_validation_results.json"
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    bnn = data["bnn_prediction"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tasks = [
        ("fade_pct|delta_V", "Fade % from ΔV"),
        ("cycle_life_log|delta_V", "log(Cycle Life) from ΔV"),
    ]

    for ax, (key, title) in zip(axes, tasks):
        r = bnn[key]
        ax.barh(["R²", "MAE"], [r["r2"], r["mae"] / max(bnn["cycle_life_log|delta_V"]["mae"], 1)],
                color=["tab:blue", "tab:orange"])
        ax.set_title(title)
        ax.text(0.5, 0.95, f"R²={r['r2']:.3f}, MAE={r['mae']:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)

    fig.suptitle("Fig 3: MC-Dropout BNN Predictions (Severson)")
    fig.savefig(OUT / "fig3_bnn.png")
    plt.close(fig)
    logger.info("Fig 3 saved")


def fig4_signature_ablation():
    """Fig 4: Signature ablation — Fisher-guided vs random vs full."""
    path = BASE / "outputs/signature_ablation/signature_ablation_results.json"
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    forward = data["experiments"]["forward_selection"]
    steps = [f["step"] for f in forward]
    r2s = [f["r2"] for f in forward]
    labels = [f["added_param"] for f in forward]

    axes[0].plot(steps, r2s, "o-", color="tab:blue", lw=2)
    for s, r2, l in zip(steps, r2s, labels):
        axes[0].annotate(l, (s, r2), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=8)
    axes[0].axhline(data["experiments"]["all_signatures"]["r2_total"],
                     color="grey", ls="--", label="All 7 sigs")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Forward selection")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    loo = data["experiments"]["leave_one_out"]
    params = list(loo.keys())
    deltas = [loo[p]["delta_r2"] for p in params]
    colors = ["tab:blue" if loo[p]["is_identifiable"] else "tab:red" for p in params]

    axes[1].barh(params, deltas, color=colors)
    axes[1].set_xlabel("ΔR² when removed")
    axes[1].set_title("Leave-one-out importance")
    axes[1].axvline(0, color="black", lw=0.5)

    fig.suptitle("Fig 4: Signature Library Ablation")
    fig.savefig(OUT / "fig4_ablation.png")
    plt.close(fig)
    logger.info("Fig 4 saved")


def fig5_profile_likelihood():
    """Fig 5: Profile likelihood curves from GP surrogate."""
    path = BASE / "outputs/profile_likelihood/gp_profile_likelihood_results.json"
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 7, figsize=(20, 4), sharey=True)
    fisher_id = {"D_n", "SEI", "LAM_neg"}

    for pi, pname in enumerate(PARAM_NAMES):
        ax = axes[pi]
        for ci in range(data["n_test_cases"]):
            prof = data["cases"][f"case_{ci}"]["profiles"][pname]
            grid = np.array(prof["log_grid"])
            dchi2 = np.array(prof["delta_chi2"])
            ax.semilogy(10.0 ** grid, dchi2 + 1e-10, alpha=0.7)

        ax.axhline(3.84, color="red", ls="--", lw=1)
        ax.set_xlabel(pname)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

        fisher_label = "Fisher-ID" if pname in fisher_id else "Fisher-UN"
        flat = data["summary"][pname]["identifiable_fraction"] == 0
        status = "FLAT (UN)" if flat else "SHARP (ID)"
        ax.set_title(f"{pname}\n{status} / {fisher_label}", fontsize=9)

    axes[0].set_ylabel("Δχ²")
    fig.suptitle(f"Fig 5: Profile Likelihood (GP surrogate R²={data['surrogate_r2']:.3f})",
                 fontsize=13)
    fig.savefig(OUT / "fig5_profile_likelihood.png")
    plt.close(fig)
    logger.info("Fig 5 saved")


def fig6_rank_vs_eta():
    """Fig 6: Rank vs η threshold with bootstrap CI."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, subpath) in zip(axes, [("LFP", "rank_robustness"), ("LIB", "rank_robustness/lib")]):
        path = BASE / "outputs" / subpath / "rank_table.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        etas_str = list(list(list(data["rank_table"].values())[0].values())[0].keys())
        etas = sorted([float(e) for e in etas_str], reverse=True)

        modes = ["raw", "log", "log_standardised", "pca_whitened"]
        for mode in modes:
            rt = data["rank_table"].get(mode, {})
            cond = rt.get("all_data_singleC_pooled", {})
            if not cond:
                continue
            ranks = [cond[f"{e:g}"]["rank_median"] for e in etas]
            lo = [cond[f"{e:g}"]["rank_q05"] for e in etas]
            hi = [cond[f"{e:g}"]["rank_q95"] for e in etas]

            ax.errorbar(etas, ranks,
                         yerr=[np.array(ranks) - np.array(lo), np.array(hi) - np.array(ranks)],
                         marker="o", capsize=3, label=mode.replace("_", " "))

        ax.set_xscale("log")
        ax.set_xlabel("η (relative eigenvalue threshold)")
        ax.set_ylabel("η-rank")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(3, color="grey", ls="--", lw=1, alpha=0.5)
        ax.text(etas[-1], 3.2, "rank=3", fontsize=7, color="grey")

    fig.suptitle("Fig 6: Rank Robustness vs η")
    fig.savefig(OUT / "fig6_rank_vs_eta.png")
    plt.close(fig)
    logger.info("Fig 6 saved")


def fig7_severson_archetypes():
    """Fig 7: Severson archetype clustering."""
    arch_path = BASE / "outputs/universality/severson/archetype.png"
    if arch_path.exists():
        import shutil
        shutil.copy(arch_path, OUT / "fig7_archetypes.png")
        logger.info("Fig 7 copied from existing archetype.png")
    else:
        logger.warning("Missing archetype.png")


def fig8_scaling_collapse():
    """Fig 8: Scaling collapse + master curve."""
    scale_path = BASE / "outputs/universality/severson/scaling.png"
    if scale_path.exists():
        import shutil
        shutil.copy(scale_path, OUT / "fig8_scaling.png")
        logger.info("Fig 8 copied from existing scaling.png")
    else:
        logger.warning("Missing scaling.png")


def main():
    logger.info("Generating paper figures ...")
    fig1_rank_robustness()
    fig2_severson_rank()
    fig3_bnn_prediction()
    fig4_signature_ablation()
    fig5_profile_likelihood()
    fig6_rank_vs_eta()
    fig7_severson_archetypes()
    fig8_scaling_collapse()
    logger.info("All figures saved to %s", OUT)


if __name__ == "__main__":
    main()
