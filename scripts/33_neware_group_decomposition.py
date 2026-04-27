"""Apply group decomposition to real NEWAREA data with bootstrap uncertainty.

Uses the validated hierarchical approach:
1. Group-level attribution (Resistance / LAM / SEI / Diffusion)
2. Per-mode NNLS within detected groups
3. Bootstrap confidence intervals
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
import sys

sys.path.insert(0, ".")
from src.data.loader import ExperimentalDataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation/neware_group")
OUT.mkdir(parents=True, exist_ok=True)

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_TIME = 100

GROUPS = {
    "Resistance": [0, 2, 6],
    "LAM": [4, 5],
    "SEI": [3],
    "Diffusion": [1],
}
GROUP_NAMES = list(GROUPS.keys())
GROUP_COLORS = {"Resistance": "#d62728", "LAM": "#1f77b4", "SEI": "#2ca02c", "Diffusion": "#ff7f0e"}


def compute_signatures():
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    mask = np.abs(cr - 1.0) < 0.01
    V1, P1 = V[mask], P[mask]

    P_reg = P1.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P1[:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_norm = (P_reg - p_med)
    P_std = P_norm.std(axis=0) + 1e-12
    P_norm /= P_std

    sigs = np.zeros((7, N_TIME))
    for t in range(N_TIME):
        m = Ridge(alpha=0.1)
        m.fit(P_norm, V1[:, t])
        sigs[:, t] = m.coef_

    group_sigs = {}
    for gname, idxs in GROUPS.items():
        group_sigs[gname] = np.sum(sigs[idxs], axis=0)

    return sigs, group_sigs


def extract_curves(data, min_pts=15):
    current, voltage, time_arr, cycles = data["current"], data["voltage"], data["time"], data["cycle"]
    unique = np.unique(cycles)
    curves = []
    for cyc in unique:
        mask = (cycles == cyc) & (current < -0.005)
        idx = np.where(mask)[0]
        if len(idx) < min_pts:
            continue
        v_s, i_s, t_s = voltage[idx], current[idx], time_arr[idx]
        valid = ~np.isnan(v_s)
        v_s, i_s, t_s = v_s[valid], i_s[valid], t_s[valid]
        if len(v_s) < min_pts:
            continue
        dt = np.diff(t_s)
        if dt.sum() < 10:
            continue
        cap = -np.sum(i_s[:-1] * dt) / 3600
        if cap < 0.001:
            continue
        curves.append({"cycle": int(cyc), "voltage": v_s, "capacity": cap})
    return curves


def decompose_nnls(dV, sigs, reg=0.01):
    n = sigs.shape[0]
    A = sigs.T
    A_aug = np.vstack([A, np.eye(n) * reg])
    b_aug = np.concatenate([dV, np.zeros(n)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def decompose_group_nnls(dV, group_sigs, reg=0.01):
    gnames = list(group_sigs.keys())
    A = np.column_stack([group_sigs[g] for g in gnames])
    A_aug = np.vstack([A, np.eye(len(gnames)) * reg])
    b_aug = np.concatenate([dV, np.zeros(len(gnames))])
    coeffs, _ = nnls(A_aug, b_aug)
    return dict(zip(gnames, coeffs))


def bootstrap_decompose(dV, sigs, n_boot=200, noise_mV=3.0, reg=0.01):
    n = sigs.shape[0]
    A = sigs.T
    rng = np.random.default_rng(42)
    all_c = np.zeros((n_boot, n))
    for b in range(n_boot):
        dV_n = dV + rng.normal(0, noise_mV / 1000.0, dV.shape)
        A_aug = np.vstack([A, np.eye(n) * reg])
        b_aug = np.concatenate([dV_n, np.zeros(n)])
        c, _ = nnls(A_aug, b_aug)
        all_c[b] = c
    return {
        "mean": np.mean(all_c, axis=0),
        "p5": np.percentile(all_c, 5, axis=0),
        "p95": np.percentile(all_c, 95, axis=0),
        "std": np.std(all_c, axis=0),
    }


def main():
    logger.info("Computing signatures...")
    sigs, group_sigs = compute_signatures()

    logger.info("Loading NEWAREA data...")
    loader = ExperimentalDataLoader()
    data = loader.load_neware_xlsx("/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")
    curves = extract_curves(data)
    logger.info(f"Extracted {len(curves)} curves")

    ref_curves = curves[:5]
    v_refs = []
    for cc in ref_curves:
        v_r = np.interp(np.linspace(0, 1, N_TIME), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        v_refs.append(v_r)
    V_ref = np.mean(v_refs, axis=0)

    # Decompose all cycles
    all_coeffs = []
    all_group_coeffs = []
    all_rmse = []
    all_caps = []

    for cc in curves:
        v_exp = np.interp(np.linspace(0, 1, N_TIME), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        dV = v_exp - V_ref
        dV_smooth = gaussian_filter1d(dV, sigma=3)

        coeffs = decompose_nnls(dV_smooth, sigs)
        group_coeffs = decompose_group_nnls(dV_smooth, group_sigs)

        dV_recon = sigs.T @ coeffs
        rmse = np.sqrt(np.mean((dV_recon - dV_smooth) ** 2)) * 1000

        all_coeffs.append(coeffs)
        all_group_coeffs.append(group_coeffs)
        all_rmse.append(rmse)
        all_caps.append(cc["capacity"])

    cycles = np.array([c["cycle"] for c in curves])
    coeffs = np.array(all_coeffs)
    group_coeffs = all_group_coeffs
    rmses = np.array(all_rmse)
    caps = np.array(all_caps)

    # Bootstrap on selected cycles
    bootstrap_cycles = [1, len(curves) // 4, len(curves) // 2, 3 * len(curves) // 4, len(curves) - 1]
    bootstrap_results = {}
    for bi in bootstrap_cycles:
        cc = curves[bi]
        v_exp = np.interp(np.linspace(0, 1, N_TIME), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        dV = gaussian_filter1d(v_exp - V_ref, sigma=3)
        bs = bootstrap_decompose(dV, sigs)
        bootstrap_results[bi] = bs

    # Print summary
    logger.info(f"\nMean fit RMSE: {rmses.mean():.1f} mV")
    logger.info(f"Capacity: {caps.max()*1000:.1f} → {caps.min()*1000:.1f} mAh ({caps.min()/caps.max()*100:.1f}%)")

    # Group attribution
    logger.info("\nGroup attribution (final cycle):")
    final_g = group_coeffs[-1]
    total_g = sum(final_g.values()) + 1e-10
    for gname in GROUP_NAMES:
        pct = final_g[gname] / total_g * 100
        logger.info(f"  {gname:15s}: {pct:5.1f}%")

    # Mode attribution
    logger.info("\nMode attribution (final cycle):")
    final = coeffs[-1]
    total = final.sum() + 1e-10
    for j in np.argsort(final)[::-1]:
        pct = final[j] / total * 100
        if pct > 1:
            logger.info(f"  {PNAMES[j]:12s}: {pct:5.1f}%")

    # Correlations with capacity
    logger.info("\nCorrelation with capacity fade:")
    for j in range(7):
        c_smooth = gaussian_filter1d(coeffs[:, j], sigma=5)
        r = np.corrcoef(c_smooth, caps)[0, 1]
        logger.info(f"  {PNAMES[j]:12s}: r = {r:+.3f}")

    # Group correlations
    logger.info("\nGroup correlations with capacity:")
    for gname in GROUP_NAMES:
        vals = np.array([gc[gname] for gc in group_coeffs])
        vals_smooth = gaussian_filter1d(vals, sigma=5)
        r = np.corrcoef(vals_smooth, caps)[0, 1]
        logger.info(f"  {gname:15s}: r = {r:+.3f}")

    # Generate figures
    plot_group_decomposition(cycles, group_coeffs, caps, rmses)
    plot_mode_decomposition(cycles, coeffs, caps, rmses)
    plot_bootstrap_panel(curves, bootstrap_cycles, bootstrap_results)
    plot_group_attribution_stacked(cycles, group_coeffs)
    plot_dV_fit_quality(cycles, curves, V_ref, sigs, coeffs)

    logger.info(f"\nOutputs: {OUT}/")


def plot_group_decomposition(cycles, group_coeffs, caps, rmses):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(cycles, caps / caps[0] * 100, "b-", linewidth=2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity retention (%)")
    ax.set_title("NEWAREA Capacity Fade (434 cycles)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cycles, rmses, "r-", linewidth=1, alpha=0.5)
    ax.plot(cycles, gaussian_filter1d(rmses, sigma=5), "r-", linewidth=2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fit RMSE (mV)")
    ax.set_title(f"Reconstruction Error (mean={rmses.mean():.1f} mV)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for gname in GROUP_NAMES:
        vals = np.array([gc[gname] for gc in group_coeffs])
        vals_smooth = gaussian_filter1d(vals, sigma=5)
        ax.plot(cycles, vals_smooth, color=GROUP_COLORS[gname], linewidth=2, label=gname)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Group coefficient")
    ax.set_title("Group Decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    total_per_cycle = np.array([sum(gc.values()) for gc in group_coeffs])
    for gname in GROUP_NAMES:
        vals = np.array([gc[gname] for gc in group_coeffs])
        pcts = vals / (total_per_cycle + 1e-10) * 100
        pcts_smooth = gaussian_filter1d(pcts, sigma=5)
        ax.plot(cycles, pcts_smooth, color=GROUP_COLORS[gname], linewidth=2, label=gname)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Group attribution (%)")
    ax.set_title("Group Attribution Over Cycling")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.suptitle("NEWAREA: Group-Level Degradation Decomposition", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "group_decomposition_newarea.png", dpi=150)
    plt.close(fig)


def plot_mode_decomposition(cycles, coeffs, caps, rmses):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(cycles, caps * 1000, "b-", linewidth=2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (mAh)")
    ax.set_title("Capacity Fade")
    ax.grid(True, alpha=0.3)

    for j in range(7):
        row, col = (j + 1) // 4, (j + 1) % 4
        ax = axes[row, col]
        c_smooth = gaussian_filter1d(coeffs[:, j], sigma=5)
        ax.plot(cycles, c_smooth, color=colors[j], linewidth=1.5)
        r = np.corrcoef(c_smooth, caps)[0, 1]
        ax.set_xlabel("Cycle")
        ax.set_ylabel(f"{PNAMES[j]} coefficient")
        ax.set_title(f"{PNAMES[j]} (r={r:+.3f})")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(cycles, caps / caps[0] * 100, "b--", alpha=0.15, linewidth=1)
        ax2.set_ylabel("Cap (%)", color="b", alpha=0.3)

    fig.suptitle("NEWAREA: Per-Mode Decomposition", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "mode_decomposition_newarea.png", dpi=150)
    plt.close(fig)


def plot_bootstrap_panel(curves, bootstrap_cycles, bootstrap_results):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    fig, axes = plt.subplots(1, len(bootstrap_cycles), figsize=(4 * len(bootstrap_cycles), 5))

    for i, bi in enumerate(bootstrap_cycles):
        ax = axes[i]
        bs = bootstrap_results[bi]
        means = bs["mean"]
        p5 = bs["p5"]
        p95 = bs["p95"]

        yerr_low = np.maximum(means - p5, 0)
        yerr_high = np.maximum(p95 - means, 0)
        ax.bar(range(7), means, yerr=np.column_stack([yerr_low, yerr_high]).T,
               color=colors, alpha=0.7, capsize=3)
        ax.set_xticks(range(7))
        ax.set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Cycle {curves[bi]['cycle']}", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Bootstrap Confidence Intervals (3mV noise, 200 trials)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_newarea.png", dpi=150)
    plt.close(fig)


def plot_group_attribution_stacked(cycles, group_coeffs):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    total = np.array([sum(gc.values()) for gc in group_coeffs])
    bottoms = np.zeros(len(cycles))
    for gname in GROUP_NAMES:
        vals = np.array([gc[gname] for gc in group_coeffs])
        pcts = vals / (total + 1e-10) * 100
        pcts_smooth = gaussian_filter1d(pcts, sigma=5)
        ax.fill_between(cycles, bottoms, bottoms + pcts_smooth,
                        color=GROUP_COLORS[gname], alpha=0.7, label=gname)
        bottoms += pcts_smooth

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Attribution (%)")
    ax.set_title("NEWAREA: Degradation Group Attribution (stacked)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "group_attribution_stacked.png", dpi=150)
    plt.close(fig)


def plot_dV_fit_quality(cycles, curves, V_ref, sigs, coeffs, n_show=6):
    indices = np.linspace(0, len(curves) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        cc = curves[idx]
        v_exp = np.interp(np.linspace(0, 1, N_TIME), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        dV = gaussian_filter1d(v_exp - V_ref, sigma=3)
        dV_recon = sigs.T @ coeffs[idx]

        t_ax = np.linspace(0, 100, N_TIME)
        ax = axes[i]
        ax.plot(t_ax, dV * 1000, "b-", linewidth=2, label="Experimental ΔV")
        ax.plot(t_ax, dV_recon * 1000, "r--", linewidth=1.5, label="Reconstructed")
        rmse = np.sqrt(np.mean((dV_recon - dV) ** 2)) * 1000
        ax.set_title(f"Cycle {cc['cycle']} (RMSE={rmse:.1f}mV)", fontsize=10)
        ax.set_xlabel("Discharge (%)")
        ax.set_ylabel("ΔV (mV)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("NEWAREA: ΔV Fit Quality", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "dV_fit_quality.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
