#!/usr/bin/env python3
"""Generate 5 new publication figures for Route A manuscript Sections 3.7-3.11.

Fig 9:  Jacobian correlation heatmap (degenerate triplet)
Fig 10: Model reduction R² comparison (ID-only vs full vs UN-only)
Fig 11: Noise robustness sweep (ID vs UN separation vs σ)
Fig 12: Multi-rate parameter recovery (single vs concatenated)
Fig 13: Cross-chemistry ablation (LFP vs LIB LOO ΔR²)
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
from matplotlib.colors import LinearSegmentedColormap

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
OUT_FIGS = BASE / "papers" / "route_a_identifiability" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
ID_PARAMS = ["SEI", "D_n", "LAM_neg"]
UN_PARAMS = ["D_p", "t+", "LAM_pos", "R_mult"]

COLORS_ID = "#2166ac"
COLORS_UN = "#b2182b"
COLORS_ID_LIGHT = "#92c5de"
COLORS_UN_LIGHT = "#f4a582"


def fig9_correlation_heatmap():
    corr_data = json.load(open(BASE / "outputs/parameter_recovery/correlation_structure.json"))
    corr = corr_data["jacobian_correlation"]
    params = PARAM_NAMES

    mat = np.array([[corr[p1][p2] for p2 in params] for p1 in params])

    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = LinearSegmentedColormap.from_list("corr", ["#f7f7f7", "#2166ac"], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(len(params)):
        for j in range(len(params)):
            val = mat[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=45, ha="right")
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_title("Jacobian Column Correlation Matrix")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Correlation|")

    deg_pairs = corr_data["high_correlation_pairs"]
    if deg_pairs:
        note = "Degenerate triplet: " + ", ".join(
            f"{p['p1']}/{p['p2']}={p['corr']:.3f}" for p in deg_pairs[:3]
        )
        ax.text(0.5, -0.15, note, transform=ax.transAxes, ha="center", fontsize=8,
                style="italic", color="#555555")

    fig.savefig(OUT_FIGS / "fig9_correlation.png")
    plt.close(fig)
    logger.info("Fig 9: correlation heatmap saved")


def fig10_model_reduction():
    data = json.load(open(BASE / "outputs/parameter_recovery/model_reduction_results.json"))
    fwd = data["forward_v_prediction"]

    models = ["full_7param", "fisher_id_3param", "fisher_un_4param",
              "data_driven_best3", "svd_k3"]
    labels = ["Full 7-param", "Fisher ID\n(SEI,D$_n$,LAM$_{neg}$)",
              "Fisher UN\n(D$_p$,t$^+$,LAM$_{pos}$,R$_{mult}$)",
              "Data-driven\nbest 3", "SVD k=3"]
    colors = ["#888888", COLORS_ID, COLORS_UN, "#4dac26", "#7b3294"]

    r2_vals = []
    for m in models:
        if m in fwd:
            r2_vals.append(fwd[m]["v_r2"])
        else:
            r2_vals.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), r2_vals, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, r2_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("V(t) prediction R²")
    ax.set_ylim(0, max(r2_vals) * 1.1)
    ax.axhline(r2_vals[0], color="grey", ls="--", lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Model Reduction: Voltage Prediction Accuracy")

    fig.savefig(OUT_FIGS / "fig10_model_reduction.png")
    plt.close(fig)
    logger.info("Fig 10: model reduction saved")


def fig11_noise_robustness():
    data = json.load(open(BASE / "outputs/parameter_recovery/noise_breakdown_results.json"))

    noise_levels = []
    id_means = []
    un_means = []
    separations = []
    for key in sorted(data.keys()):
        sigma = float(key.replace("noise_", "").replace("mV", ""))
        noise_levels.append(sigma)
        id_means.append(data[key]["id_mean_r2"])
        un_means.append(data[key]["un_mean_r2"])
        separations.append(data[key]["separation"])

    noise_levels = np.array(noise_levels)
    id_means = np.array(id_means)
    un_means = np.array(un_means)
    separations = np.array(separations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(noise_levels, id_means, "o-", color=COLORS_ID, lw=2, ms=6, label="ID group (SEI, D$_n$, LAM$_{neg}$)")
    ax1.plot(noise_levels, un_means, "s-", color=COLORS_UN, lw=2, ms=6, label="UN group (D$_p$, t$^+$, LAM$_{pos}$, R$_{mult}$)")
    ax1.axvspan(0, 5, alpha=0.1, color="green", label="Typical tester noise")
    ax1.set_xlabel("Measurement noise σ (mV)")
    ax1.set_ylabel("Mean R² (parameter recovery)")
    ax1.set_title("ID vs UN Recovery vs Noise")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.semilogy(noise_levels, separations, "D-", color="#333333", lw=2, ms=6)
    ax2.axhline(1, color="red", ls="--", lw=1, label="No separation")
    ax2.axhline(10, color="orange", ls=":", lw=1, label="10× threshold")
    ax2.axvspan(0, 5, alpha=0.1, color="green")
    ax2.set_xlabel("Measurement noise σ (mV)")
    ax2.set_ylabel("Separation ratio (ID/UN)")
    ax2.set_title("Identifiability Separation Ratio")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.savefig(OUT_FIGS / "fig11_noise.png")
    plt.close(fig)
    logger.info("Fig 11: noise robustness saved")


def fig12_multirate_recovery():
    data = json.load(open(BASE / "outputs/parameter_recovery/multi_rate_recovery_results.json"))

    single = data["single_rate_pooled"]
    concat = data["multi_rate_concatenated"]

    params = PARAM_NAMES
    r2_single = [single[p]["r2"] for p in params]
    r2_concat = [concat[p]["r2"] for p in params]

    colors_bar = [COLORS_ID if p in ID_PARAMS else COLORS_UN for p in params]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(params))
    width = 0.35

    bars1 = ax.bar(x - width / 2, r2_single, width, label="Single rate (1C pooled)",
                   color=[c + "99" for c in colors_bar], edgecolor=colors_bar, linewidth=1.5)
    bars2 = ax.bar(x + width / 2, r2_concat, width, label="Multi-rate (0.5C+1C+2C concat)",
                   color=colors_bar, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=10)
    ax.set_ylabel("R² (parameter recovery)")
    ax.legend(fontsize=9)
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Multi-Rate Protocol Improves ID Parameter Recovery")

    id_params_idx = [i for i, p in enumerate(params) if p in ID_PARAMS]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([np.mean(id_params_idx)])
    ax2.set_xticklabels(["← ID group"], fontsize=8, color=COLORS_ID)

    fig.savefig(OUT_FIGS / "fig12_multirate.png")
    plt.close(fig)
    logger.info("Fig 12: multi-rate recovery saved")


def fig13_cross_chemistry_ablation():
    lfp_data = json.load(open(BASE / "outputs/signature_ablation/signature_ablation_results.json"))
    lib_data = json.load(open(BASE / "outputs/signature_ablation/lib_ablation_results.json"))

    lfp_loo = lfp_data["experiments"]["leave_one_out"]
    lib_loo = lib_data["leave_one_out"]

    lfp_params = list(lfp_loo.keys())
    lib_entries = list(lib_loo.keys())

    lfp_dr2 = [abs(lfp_loo[p]["delta_r2"]) for p in lfp_params]
    lfp_is_id = [lfp_loo[p]["is_identifiable"] for p in lfp_params]

    lib_short = {
        "Negative particle di": "D$_n$",
        "Positive particle di": "D$_p$",
        "Cation transference ": "t$^+$",
        "Positive electrode e": "k$_{pos}$",
        "Negative electrode e": "k$_{neg}$",
        "Positive electrode c": "j$_{0,pos}$",
        "Negative electrode c": "j$_{0,neg}$",
    }
    lib_labels = [lib_short.get(e, e[:10]) for e in lib_entries]
    lib_dr2 = [abs(lib_loo[e]["delta_r2"]) for e in lib_entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    colors_lfp = [COLORS_ID if is_id else COLORS_UN for is_id in lfp_is_id]
    ax1.barh(range(len(lfp_params)), lfp_dr2, color=colors_lfp, edgecolor="white")
    ax1.set_yticks(range(len(lfp_params)))
    ax1.set_yticklabels(lfp_params)
    ax1.set_xlabel("|ΔR²| when removed")
    ax1.set_title("LFP (Prada2013)")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS_ID, label="Fisher ID"),
                       Patch(facecolor=COLORS_UN, label="Fisher UN")]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")

    ax2.barh(range(len(lib_entries)), lib_dr2, color="#666666", edgecolor="white")
    ax2.set_yticks(range(len(lib_entries)))
    ax2.set_yticklabels(lib_labels)
    ax2.set_xlabel("|ΔR²| when removed")
    ax2.set_title("LIB (Chen2020 NMC811)")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Cross-Chemistry Parameter Importance (Leave-One-Out ΔR²)", fontsize=13)
    fig.savefig(OUT_FIGS / "fig13_cross_chemistry.png")
    plt.close(fig)
    logger.info("Fig 13: cross-chemistry ablation saved")


if __name__ == "__main__":
    fig9_correlation_heatmap()
    fig10_model_reduction()
    fig11_noise_robustness()
    fig12_multirate_recovery()
    fig13_cross_chemistry_ablation()
    logger.info("All 5 new figures generated in %s", OUT_FIGS)
