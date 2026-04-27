"""Hierarchical decomposition with bootstrap uncertainty.

Best approach: group-level attribution (62% pass) + per-mode NNLS with
temporal smoothing and bootstrap confidence intervals.

Key insight: individual decomposition is unreliable for single cycles,
but the TEMPORAL TREND over hundreds of cycles is robust.
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

OUT = Path("outputs/degradation/validation/hierarchical")
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


def load_and_setup():
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    mask_1c = np.abs(cr - 1.0) < 0.01
    V1, P1 = V[mask_1c], P[mask_1c]

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

    group_sigs = {}
    for gname, idxs in GROUPS.items():
        group_sigs[gname] = np.sum(sigs[idxs], axis=0)

    idx_base = np.argmin(np.abs(P_norm).sum(axis=1))
    p_median = np.median(P1, axis=0)

    return V1, P1, sigs, group_sigs, p_median, p_med, P_std, P_norm, idx_base


def decompose_nnls(dV, sigs, reg=0.01):
    n = sigs.shape[0]
    A = sigs.T
    A_aug = np.vstack([A, np.eye(n) * reg])
    b_aug = np.concatenate([dV, np.zeros(n)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def decompose_group(dV, group_sigs, reg=0.01):
    gnames = list(group_sigs.keys())
    A = np.column_stack([group_sigs[g] for g in gnames])
    A_aug = np.vstack([A, np.eye(len(gnames)) * reg])
    b_aug = np.concatenate([dV, np.zeros(len(gnames))])
    coeffs, _ = nnls(A_aug, b_aug)
    return dict(zip(gnames, coeffs))


def find_nearest(P_norm, target_norm):
    dists = np.sqrt(np.sum((P_norm - target_norm) ** 2, axis=1))
    return np.argmin(dists)


def bootstrap_coeffs(dV, sigs, n_bootstrap=200, noise_mV=3.0, reg=0.01):
    """Bootstrap with noise to estimate coefficient uncertainty."""
    n_sigs = sigs.shape[0]
    A = sigs.T
    rng = np.random.default_rng(42)
    all_coeffs = np.zeros((n_bootstrap, n_sigs))

    for b in range(n_bootstrap):
        dV_noisy = dV + rng.normal(0, noise_mV / 1000.0, dV.shape)
        A_aug = np.vstack([A, np.eye(n_sigs) * reg])
        b_aug = np.concatenate([dV_noisy, np.zeros(n_sigs)])
        c, _ = nnls(A_aug, b_aug)
        all_coeffs[b] = c

    return {
        "mean": np.mean(all_coeffs, axis=0),
        "std": np.std(all_coeffs, axis=0),
        "p5": np.percentile(all_coeffs, 5, axis=0),
        "p95": np.percentile(all_coeffs, 95, axis=0),
        "positive_frac": np.mean(all_coeffs > 0, axis=0),
    }


def main():
    V1, P1, sigs, group_sigs, p_median, p_med, P_std, P_norm, idx_base = load_and_setup()
    V_base = V1[idx_base]
    logger.info(f"Setup complete: {V1.shape[0]} sims, baseline #{idx_base}")

    n_cycles = 100
    trajectories = {
        "R_mult growth": lambda c: {6: 1.0 + c * 4.0},
        "LAM_pos growth": lambda c: {5: c * 0.3},
        "SEI growth": lambda c: {3: p_median[3] * (1 + c * 500)},
        "R + LAM_pos": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2},
        "R + SEI": lambda c: {6: 1.0 + c * 3.0, 3: p_median[3] * (1 + c * 300)},
        "LAM + SEI": lambda c: {5: c * 0.2, 3: p_median[3] * (1 + c * 300)},
        "R + LAM + SEI": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 3: p_median[3] * (1 + c * 300)},
        "Full realistic": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 4: c * 0.1,
                                     3: p_median[3] * (1 + c * 300), 1: p_median[1] * (1 - c * 0.5)},
    }

    all_results = {}

    for traj_name, param_fn in trajectories.items():
        traj = {
            "coeffs": [], "group_coeffs": [],
            "bootstrap": [],
            "rmse": [],
        }

        for cyc in range(n_cycles):
            progress = cyc / max(n_cycles - 1, 1)
            changes = param_fn(progress)
            p_target = p_median.copy()
            for j, v in changes.items():
                p_target[j] = v

            p_t_reg = p_target.copy()
            for j in LOG_PARAMS:
                p_t_reg[j] = np.log10(p_target[j] + 1e-30)
            p_t_norm = (p_t_reg - p_med) / P_std
            idx_t = find_nearest(P_norm, p_t_norm)
            dV = V1[idx_t] - V_base

            c = decompose_nnls(dV, sigs)
            g = decompose_group(dV, group_sigs)

            rmse = np.sqrt(np.mean((sigs.T @ c - dV) ** 2)) * 1000
            traj["coeffs"].append(c)
            traj["group_coeffs"].append(g)
            traj["rmse"].append(rmse)

            if cyc >= n_cycles - 5:
                bs = bootstrap_coeffs(dV, sigs, n_bootstrap=100, noise_mV=3.0)
                traj["bootstrap"].append(bs)

        traj["coeffs"] = np.array(traj["coeffs"])
        traj["rmse"] = np.array(traj["rmse"])

        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())
        gt_names = [PNAMES[j] for j in gt_idx]
        gt_groups = set()
        for j in gt_idx:
            for gname, gidxs in GROUPS.items():
                if j in gidxs:
                    gt_groups.add(gname)

        final = traj["coeffs"][-1]
        total = final.sum() + 1e-10
        pct = sum(final[j] for j in gt_idx) / total * 100
        pct_by_mode = {PNAMES[j]: final[j] / total * 100 for j in range(7)}

        final_g = traj["group_coeffs"][-1]
        total_g = sum(final_g.values()) + 1e-10
        group_pct = sum(final_g[g] for g in gt_groups) / total_g * 100

        bs = traj["bootstrap"][-1]
        detectable = bs["positive_frac"] > 0.5

        traj["eval"] = {
            "gt_idx": gt_idx, "gt_names": gt_names, "gt_groups": gt_groups,
            "pct": pct, "group_pct": group_pct,
            "pct_by_mode": pct_by_mode,
            "detectable": detectable,
            "bootstrap": bs,
        }
        all_results[traj_name] = traj

        logger.info(f"\n  {traj_name} [{', '.join(gt_names)}]:")
        logger.info(f"    Mode NNLS: {pct:.0f}% | Group: {group_pct:.0f}%")
        logger.info(f"    Attribution: {', '.join(f'{k}={v:.1f}%' for k, v in sorted(pct_by_mode.items(), key=lambda x: -x[1])[:4])}")
        detect_str = ", ".join(PNAMES[j] for j in range(7) if detectable[j])
        logger.info(f"    Detectable (>50% positive): {detect_str}")
        logger.info(f"    Bootstrap top: " + ", ".join(
            f"{PNAMES[j]}={bs['mean'][j]:.3f}±{bs['std'][j]:.3f}" for j in np.argsort(bs['mean'])[::-1][:3]
        ))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("HIERARCHICAL DECOMPOSITION SUMMARY")
    logger.info("=" * 70)
    mode_pass = group_pass = 0
    for name, r in all_results.items():
        eval_d = r["eval"]
        gt = eval_d["gt_names"]
        mp = eval_d["pct"]
        gp = eval_d["group_pct"]
        det = [PNAMES[j] for j in range(7) if eval_d["detectable"][j]]
        if mp > 75:
            mode_pass += 1
        if gp > 75:
            group_pass += 1
        logger.info(f"  {name:25s}: Mode={mp:5.1f}% Group={gp:5.1f}% | GT={gt} | Detect={det}")

    logger.info(f"\n  Mode PASS: {mode_pass}/{len(all_results)}")
    logger.info(f"  Group PASS: {group_pass}/{len(all_results)}")

    # Detection rate per mode across all scenarios where it's GT
    logger.info("\n  Per-mode detection (positive in >50% of bootstraps):")
    for j in range(7):
        n_active = 0
        n_detected = 0
        for name, r in all_results.items():
            if j in r["eval"]["gt_idx"]:
                n_active += 1
                if r["eval"]["detectable"][j]:
                    n_detected += 1
        if n_active > 0:
            logger.info(f"    {PNAMES[j]:12s}: detected in {n_detected}/{n_active} scenarios where active")

    plot_hierarchical_summary(all_results)
    plot_bootstrap_confidence(all_results)
    plot_attribution_heatmap(all_results)

    logger.info(f"\nOutputs: {OUT}/")


def plot_hierarchical_summary(all_results):
    names = list(all_results.keys())
    n = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    mode_pcts = [all_results[n_]["eval"]["pct"] for n_ in names]
    group_pcts = [all_results[n_]["eval"]["group_pct"] for n_ in names]
    ax.bar(x - w / 2, mode_pcts, w, label="Mode NNLS", color="orange", alpha=0.8)
    ax.bar(x + w / 2, group_pcts, w, label="Group", color="green", alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([n_[:15] for n_ in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% Correct attribution")
    ax.set_title("Final Cycle Attribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    mode_pass_count = sum(1 for p in mode_pcts if p > 75)
    group_pass_count = sum(1 for p in group_pcts if p > 75)
    bars = ax.bar(["Mode NNLS", "Group"], [mode_pass_count / n * 100, group_pass_count / n * 100],
                  color=["orange", "green"], alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, [mode_pass_count, group_pass_count]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val}/{n}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("% Scenarios PASS")
    ax.set_title("Pass Rate (>75%)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    detect_rates = []
    for j in range(7):
        n_active = sum(1 for r in all_results.values() if j in r["eval"]["gt_idx"])
        n_detected = sum(1 for r in all_results.values()
                        if j in r["eval"]["gt_idx"] and r["eval"]["detectable"][j])
        detect_rates.append(n_detected / max(n_active, 1) * 100)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    ax.bar(range(7), detect_rates, color=colors, alpha=0.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(PNAMES, rotation=45, ha="right")
    ax.set_ylabel("% Scenarios where mode detected")
    ax.set_title("Per-Mode Detection Rate (bootstrap)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "hierarchical_summary.png", dpi=150)
    plt.close(fig)


def plot_bootstrap_confidence(all_results):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    n_traj = len(all_results)

    fig, axes = plt.subplots(2, min(4, n_traj), figsize=(5 * min(4, n_traj), 10))
    axes = axes.flatten()

    for i, (name, r) in enumerate(all_results.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        bs = r["eval"]["bootstrap"]
        gt_idx = r["eval"]["gt_idx"]

        means = bs["mean"]
        stds = bs["std"]
        x = np.arange(7)
        bar_colors = ["green" if j in gt_idx else colors[j] for j in range(7)]
        ax.bar(x, means, yerr=stds * 2, color=bar_colors, alpha=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{name}\n(correct={r['eval']['pct']:.0f}%)", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(all_results), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Bootstrap Uncertainty (mean ± 2σ, green=GT active)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_confidence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attribution_heatmap(all_results):
    names = list(all_results.keys())
    n_traj = len(names)

    fig, axes = plt.subplots(1, 2, figsize=(14, n_traj * 0.5 + 2))

    for ax_idx, (key, title) in enumerate([
        ("coeffs", "Mode Attribution (%)"),
        ("group_coeffs", "Group Attribution (%)"),
    ]):
        ax = axes[ax_idx]
        if key == "coeffs":
            data = np.zeros((n_traj, 7))
            for i, name in enumerate(names):
                final = all_results[name]["coeffs"][-1]
                total = final.sum() + 1e-10
                data[i] = final / total * 100
            col_labels = PNAMES
        else:
            data = np.zeros((n_traj, len(GROUP_NAMES)))
            for i, name in enumerate(names):
                gc = all_results[name]["group_coeffs"][-1]
                total = sum(gc.values()) + 1e-10
                for gi, gname in enumerate(GROUP_NAMES):
                    data[i, gi] = gc[gname] / total * 100
            col_labels = GROUP_NAMES

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(n_traj))
        ax.set_yticklabels([n[:25] for n in names], fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        for i in range(n_traj):
            for j in range(len(col_labels)):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7,
                       color="white" if data[i, j] > 50 else "black")

        if key == "coeffs":
            for i, name in enumerate(names):
                for j in all_results[name]["eval"]["gt_idx"]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 fill=False, edgecolor="blue", linewidth=2))

    fig.tight_layout()
    fig.savefig(OUT / "attribution_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
