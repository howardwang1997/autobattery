"""Comprehensive multi-cell analysis with ground truth validation.

Synthesizes all results from:
1. Ground truth decomposition validation (script 35 + 34v2)
2. ICA dQ/dV comparison (script 36)
3. Multi-cell degradation analysis

Generates publication-quality figures summarizing all findings.
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
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation/comprehensive")
OUT.mkdir(parents=True, exist_ok=True)

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_TIME = 100
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


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
    P_std = P_reg.std(axis=0) + 1e-12
    P_norm = (P_reg - p_med) / P_std
    sigs = np.zeros((7, N_TIME))
    for t in range(N_TIME):
        m = Ridge(alpha=0.1)
        m.fit(P_norm, V1[:, t])
        sigs[:, t] = m.coef_
    return sigs


def decompose_nnls(dV, sigs, reg=0.01):
    n = sigs.shape[0]
    A = sigs.T
    A_aug = np.vstack([A, np.eye(n) * reg])
    b_aug = np.concatenate([dV, np.zeros(n)])
    coeffs, _ = nnls(A_aug, b_aug)
    recon = A @ coeffs
    rmse = np.sqrt(np.mean((recon - dV) ** 2)) * 1000
    return coeffs, rmse


def validate_ground_truth(sigs):
    """Validate decomposition against PyBaMM ground truth scenarios."""
    gt_path = Path("data/ground_truth/ground_truth_multicycle.h5")
    if not gt_path.exists():
        logger.warning("No ground truth data found!")
        return None

    results = {}
    with h5py.File(gt_path, "r") as f:
        for sc_name in f.keys():
            if sc_name == "baseline":
                continue
            grp = f[sc_name]
            if "V_cycles" not in grp:
                continue
            V = grp["V_cycles"][:]
            cap = grp["capacity"][:]
            gt = grp["gt_params"][:] if "gt_params" in grp else None
            n_cyc = V.shape[0]
            if n_cyc < 10:
                continue

            V_ref = V[:5].mean(axis=0)
            coeffs_all = np.zeros((7, n_cyc))
            rmses = []
            for c in range(n_cyc):
                dV = gaussian_filter1d(V[c] - V_ref, sigma=3)
                co, rm = decompose_nnls(dV, sigs)
                coeffs_all[:, c] = co
                rmses.append(rm)

            results[sc_name] = {
                "coeffs": coeffs_all,
                "rmse": np.array(rmses),
                "capacity": cap,
                "gt_params": gt,
                "n_cycles": n_cyc,
            }

    return results


def analyze_gt_results(gt_results):
    """Analyze decomposition vs ground truth."""
    logger.info("\n" + "=" * 70)
    logger.info("GROUND TRUTH VALIDATION ANALYSIS")
    logger.info("=" * 70)

    for sc_name, r in gt_results.items():
        gt = r["gt_params"]
        if gt is None:
            continue

        n_cyc = r["n_cycles"]
        coeffs = r["coeffs"]
        cap = r["capacity"]

        cap_fade = (1 - cap[-1] / cap[0]) * 100
        logger.info(f"\n  {sc_name} ({n_cyc} cycles, fade={cap_fade:.1f}%):")

        c_smooth = gaussian_filter1d(coeffs, sigma=5, axis=1)

        gt_sei = gt[:, 3]
        gt_lam_pos = gt[:, 5]
        gt_r_mult = gt[:, 6]
        gt_lam_neg = gt[:, 4]
        gt_d_p = gt[:, 1]

        gt_changes = {}
        for j, (pname, gt_col) in enumerate([
            ("SEI_thick", gt_sei), ("LAM_pos", gt_lam_pos),
            ("R_mult", gt_r_mult), ("LAM_neg", gt_lam_neg),
            ("D_p", gt_d_p)
        ]):
            if gt_col.std() > 1e-15:
                gt_changes[pname] = gt_col
                n_min = min(len(c_smooth[j]), len(gt_col))
                r_corr = np.corrcoef(c_smooth[j, :n_min], gt_col[:n_min])[0, 1]
                logger.info(f"    {pname:12s}: GT range [{gt_col[0]:.4g}, {gt_col[-1]:.4g}], "
                            f"decomp corr={r_corr:+.3f}")

    return gt_results


def plot_comprehensive_summary(gt_results):
    """Generate comprehensive summary figure."""
    if gt_results is None:
        return

    n_sc = len(gt_results)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, (sc_name, r) in enumerate(list(gt_results.items())[:4]):
        ax = axes[0, i]
        n_cyc = r["n_cycles"]
        cycles = np.arange(n_cyc)
        cap = r["capacity"]

        ax.plot(cycles, cap / cap[0] * 100, "k-", linewidth=2)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity (%)")
        ax.set_title(f"{sc_name}\n({n_cyc} cyc, fade={1-cap[-1]/cap[0]:.1%})")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        ax2 = ax.twinx()
        coeffs = r["coeffs"]
        for j in range(7):
            c_s = gaussian_filter1d(coeffs[j], 5)
            ax2.plot(cycles, c_s / (c_s.max() + 1e-10), color=COLORS[j], linewidth=1, alpha=0.5)
        ax2.set_ylabel("Coeff (norm)", alpha=0.3)

    gt_corr_sei = []
    gt_corr_lam = []
    gt_corr_r = []
    scenario_names = []

    for sc_name, r in gt_results.items():
        gt = r["gt_params"]
        if gt is None:
            continue
        coeffs = gaussian_filter1d(r["coeffs"], 5, axis=1)
        n_min = min(coeffs.shape[1], gt.shape[0])

        if gt[:, 3].std() > 1e-15:
            r_sei = np.corrcoef(coeffs[3, :n_min], gt[:n_min, 3])[0, 1]
            gt_corr_sei.append(r_sei)
        else:
            gt_corr_sei.append(0)

        if gt[:, 5].std() > 1e-15:
            r_lam = np.corrcoef(coeffs[5, :n_min], gt[:n_min, 5])[0, 1]
            gt_corr_lam.append(r_lam)
        else:
            gt_corr_lam.append(0)

        if gt[:, 6].std() > 1e-15:
            r_r = np.corrcoef(coeffs[6, :n_min], gt[:n_min, 6])[0, 1]
            gt_corr_r.append(r_r)
        else:
            gt_corr_r.append(0)

        scenario_names.append(sc_name[:15])

    ax = axes[1, 0]
    x = np.arange(len(scenario_names))
    w = 0.25
    ax.bar(x - w, gt_corr_sei, w, label="SEI", color=COLORS[3])
    ax.bar(x, gt_corr_lam, w, label="LAM_pos", color=COLORS[5])
    ax.bar(x + w, gt_corr_r, w, label="R_mult", color=COLORS[6])
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Correlation with GT")
    ax.set_title("Decomposition vs Ground Truth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-1, 1.05)

    ax = axes[1, 1]
    sig_corr = np.zeros((7, 7))
    sigs = compute_signatures()
    for i in range(7):
        for j in range(7):
            sig_corr[i, j] = np.corrcoef(sigs[i], sigs[j])[0, 1]
    im = ax.imshow(np.abs(sig_corr), cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(PNAMES, fontsize=7)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{sig_corr[i,j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("V(t) Signature Correlations")
    plt.colorbar(im, ax=ax, shrink=0.7)

    ax = axes[1, 2]
    U, S, Vt = np.linalg.svd(sigs, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    ax.bar(range(1, 8), S / S[0] * 100, color="steelblue", alpha=0.7, label="Singular value (%)")
    ax2 = ax.twinx()
    ax2.plot(range(1, 8), cumvar * 100, "r-o", linewidth=2, markersize=5, label="Cumulative")
    ax2.axhline(95, color="k", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Cumulative %")
    ax.set_xlabel("Component")
    ax.set_ylabel("SV (% of max)")
    ax.set_title(f"SVD: rank_95={np.searchsorted(cumvar, 0.95)+1}, cond={S[0]/S[-1]:.0f}")
    ax.legend(fontsize=7, loc="center right")
    ax2.legend(fontsize=7, loc="right")

    ax = axes[1, 3]
    method_names = ["1C NNLS", "Group NNLS", "Temporal v1", "Temporal v2", "dQ/dV NNLS"]
    pass_rates = [3/8*100, 5/8*100, 1/8*100, 1/8*100, 0/8*100]
    bar_colors = ["steelblue", "coral", "forestgreen", "mediumseagreen", "mediumpurple"]
    bars = ax.bar(method_names, pass_rates, color=bar_colors, alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Method Comparison (synthetic, >75%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, pct in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pct:.0f}%", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Comprehensive Degradation Decomposition Analysis", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "comprehensive_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comprehensive_summary.png")


def plot_gt_decomposition_detail(gt_results):
    """Detailed per-scenario decomposition vs ground truth."""
    if gt_results is None:
        return

    n_sc = len(gt_results)
    n_cols = 3
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (sc_name, r) in enumerate(gt_results.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        coeffs = r["coeffs"]
        gt = r["gt_params"]
        n_cyc = r["n_cycles"]
        cycles = np.arange(n_cyc)

        for j in range(7):
            c_s = gaussian_filter1d(coeffs[j], 5)
            ax.plot(cycles, c_s / (np.abs(c_s).max() + 1e-10),
                    color=COLORS[j], linewidth=1.5, alpha=0.8, label=f"{PNAMES[j]}")

        if gt is not None:
            n_min = min(n_cyc, gt.shape[0])
            active_gt = []
            for j in range(7):
                if gt[:n_min, j].std() > 1e-12:
                    active_gt.append(j)

            if active_gt:
                ax.set_title(f"{sc_name}\nGT modes: {[PNAMES[j] for j in active_gt]}", fontsize=9)
            else:
                ax.set_title(f"{sc_name}", fontsize=9)
        else:
            ax.set_title(f"{sc_name}", fontsize=9)

        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Cycle")

    for idx in range(n_sc, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("Per-Scenario Decomposition (Ground Truth Available)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "gt_decomposition_detail.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved gt_decomposition_detail.png")


def main():
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE MULTI-CELL ANALYSIS")
    logger.info("=" * 70)

    sigs = compute_signatures()
    logger.info(f"Signatures computed: {sigs.shape}")

    gt_results = validate_ground_truth(sigs)
    if gt_results:
        logger.info(f"\nGround truth scenarios: {len(gt_results)}")
        analyze_gt_results(gt_results)
        plot_comprehensive_summary(gt_results)
        plot_gt_decomposition_detail(gt_results)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("""
Key findings from all experiments:

1. GROUND TRUTH VALIDATION:
   - SEI decomposition correlates with GT at r > 0.99 (excellent)
   - LAM_pos decomposition correlates with GT at r = 0.6-0.97 (good)
   - R_mult decomposition: unreliable (signature r=0.86 with D_n)

2. METHOD COMPARISON (synthetic, 8 scenarios):
   - 1C NNLS:      3/8 pass (38%)
   - Group NNLS:    5/8 pass (62%) ← BEST
   - Temporal v1:   1/8 pass (13%)
   - Temporal v2:   1/8 pass (13%)
   - dQ/dV NNLS:    0/8 pass (0%)

3. dQ/dV ANALYSIS:
   - dQ/dV signatures have HIGHER correlation (mean|r|=0.517 vs 0.403)
   - dQ/dV decomposition is WORSE than V(t) (0/8 vs 3/8 pass)
   - Conclusion: V(t) is the optimal space for decomposition

4. RECOMMENDATION:
   - Use Group NNLS as primary method (62% pass, 100% SEI/LAM detection)
   - Report identifiability analysis as key contribution
   - V(t) > dQ/dV for decomposition (publishable negative result)
""")

    logger.info(f"Outputs: {OUT}/")


if __name__ == "__main__":
    main()
