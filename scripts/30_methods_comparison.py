"""Comprehensive comparison of degradation decomposition methods.

Methods tested:
1. Baseline NNLS (current approach)
2. Group decomposition (cluster correlated modes)
3. Constrained decomposition (monotonicity + kinetic priors)
4. Orthogonalized signatures (PCA projection)
5. Elastic Net (L1+L2 regularization)
6. dQ/dV-based decomposition (ICA features)

Validated on PyBaMM simulation data with known ground truth.
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
from scipy.optimize import nnls, minimize
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation/validation/methods_comparison")
OUT.mkdir(parents=True, exist_ok=True)

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_TIME = 100

GROUPS = {
    "Resistance": [0, 2, 6],    # D_n, t+, R_mult — all affect overpotential
    "LAM": [4, 5],              # LAM_neg, LAM_pos — capacity loss
    "SEI": [3],                  # SEI — unique mechanism
    "Diffusion": [1],            # D_p — somewhat independent
}
GROUP_NAMES = list(GROUPS.keys())
GROUP_COLORS = {"Resistance": "#d62728", "LAM": "#1f77b4", "SEI": "#2ca02c", "Diffusion": "#ff7f0e"}


def load_data():
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    mask_1c = np.abs(cr - 1.0) < 0.01
    V1, P1 = V[mask_1c], P[mask_1c]
    mask_05c = np.abs(cr - 0.5) < 0.01
    V05, P05 = V[mask_05c], P[mask_05c]
    mask_2c = np.abs(cr - 2.0) < 0.01
    V2, P2 = V[mask_2c], P[mask_2c]
    return (V1, P1), (V05, P05), (V2, P2)


def compute_ridge_signatures(V, P):
    n_time = V.shape[1]
    P_reg = P.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P[:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_norm = (P_reg - p_med)
    P_std = P_norm.std(axis=0) + 1e-12
    P_norm /= P_std
    sigs = np.zeros((len(PNAMES), n_time))
    for t in range(n_time):
        m = Ridge(alpha=0.1)
        m.fit(P_norm, V[:, t])
        sigs[:, t] = m.coef_
    return sigs, p_med, P_std, P_norm, V


def compute_group_signatures(signatures):
    """Sum signatures within each group."""
    group_sigs = {}
    for gname, idxs in GROUPS.items():
        group_sigs[gname] = np.sum(signatures[idxs], axis=0)
    return group_sigs


def compute_orthogonal_signatures(signatures, n_components=None):
    """PCA-project signatures to decorrelate them."""
    if n_components is None:
        n_components = signatures.shape[0]
    pca = PCA(n_components=n_components)
    sig_flat = signatures
    projected = pca.fit_transform(sig_flat.T).T
    return projected, pca


def compute_dqdV_signatures(V, P):
    """Compute signatures in dQ/dV space (ICA features)."""
    n_sims, n_time = V.shape
    P_reg = P.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P[:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_norm = (P_reg - p_med)
    P_std = P_norm.std(axis=0) + 1e-12
    P_norm /= P_std

    n_ica = n_time - 1
    ica_sigs = np.zeros((len(PNAMES), n_ica))
    for t in range(n_ica):
        dV = np.diff(V, axis=1)[:, t]
        mask = np.abs(dV) > 1e-6
        if mask.sum() < 10:
            continue
        m = Ridge(alpha=0.1)
        m.fit(P_norm[mask], dV[mask])
        ica_sigs[:, t] = m.coef_
    return ica_sigs, p_med, P_std


# --- Decomposition methods ---

def method_nnls(dV, signatures, reg=0.01):
    n_sigs = signatures.shape[0]
    A = signatures.T
    A_aug = np.vstack([A, np.eye(n_sigs) * reg])
    b_aug = np.concatenate([dV, np.zeros(n_sigs)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def method_elastic_net(dV, signatures, alpha=0.01, l1_ratio=0.5):
    A = signatures.T
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, max_iter=5000)
    model.fit(A, dV)
    return model.coef_


def method_constrained_nnls(dV, signatures, prev_coeffs=None, monotonic_weight=0.1):
    """NNLS with optional monotonicity constraint."""
    n_sigs = signatures.shape[0]
    A = signatures.T

    if prev_coeffs is None:
        return method_nnls(dV, signatures)

    def objective(c):
        recon_err = np.sum((A @ c - dV) ** 2)
        mono_penalty = monotonic_weight * np.sum(np.maximum(0, prev_coeffs - c) ** 2)
        return recon_err + mono_penalty

    c0 = np.maximum(prev_coeffs, 1e-6)
    bounds = [(0, None)] * n_sigs
    result = minimize(objective, c0, bounds=bounds, method='L-BFGS-B')
    return np.maximum(result.x, 0)


def method_group_nnls(dV, group_sigs, reg=0.01):
    """Decompose into mode groups instead of individual modes."""
    gnames = list(group_sigs.keys())
    n_groups = len(gnames)
    A = np.column_stack([group_sigs[g] for g in gnames])
    A_aug = np.vstack([A, np.eye(n_groups) * reg])
    b_aug = np.concatenate([dV, np.zeros(n_groups)])
    coeffs, _ = nnls(A_aug, b_aug)
    return dict(zip(gnames, coeffs))


def method_pca_nnls(dV, signatures_proj, pca, reg=0.01):
    """Decompose using PCA-projected signatures."""
    n_sigs = signatures_proj.shape[0]
    A = signatures_proj.T
    A_aug = np.vstack([A, np.eye(n_sigs) * reg])
    b_aug = np.concatenate([dV, np.zeros(n_sigs)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def method_multicr_nnls(dVs, signatures_list, reg=0.01):
    """Joint decomposition across multiple C-rates."""
    n_sigs = signatures_list[0].shape[0]
    A = np.vstack([s.T for s in signatures_list])
    b = np.concatenate(dVs)
    A_aug = np.vstack([A, np.eye(n_sigs) * reg])
    b_aug = np.concatenate([b, np.zeros(n_sigs)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def main():
    logger.info("Loading data and computing signatures...")
    (V1, P1), (V05, P05), (V2, P2) = load_data()

    sigs_1c, p_med, P_std, P_norm_1c, V1_data = compute_ridge_signatures(V1, P1)
    sigs_05c, _, _, _, _ = compute_ridge_signatures(V05, P05)
    sigs_2c, _, _, _, _ = compute_ridge_signatures(V2, P2)
    ica_sigs, _, _ = compute_dqdV_signatures(V1, P1)
    group_sigs = compute_group_signatures(sigs_1c)

    p_median_raw = np.median(P1, axis=0)
    P_reg_all = P1.copy()
    for j in LOG_PARAMS:
        P_reg_all[:, j] = np.log10(P1[:, j] + 1e-30)
    P_norm_all = (P_reg_all - p_med) / P_std

    idx_base = np.argmin(np.abs(P_norm_all).sum(axis=1))
    V_base = V1[idx_base]
    V_base_05 = V05[idx_base]
    V_base_2 = V2[idx_base]

    logger.info(f"Baseline sim #{idx_base}, {V1.shape[0]} total sims")

    # --- Trajectory-based validation ---
    n_cycles = 100
    trajectories = {
        "R_mult growth": lambda c: {6: 1.0 + c * 4.0},
        "LAM_pos growth": lambda c: {5: c * 0.3},
        "SEI growth": lambda c: {3: p_median_raw[3] * (1 + c * 500)},
        "R + LAM_pos": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2},
        "R + SEI": lambda c: {6: 1.0 + c * 3.0, 3: p_median_raw[3] * (1 + c * 300)},
        "LAM_pos + SEI": lambda c: {5: c * 0.2, 3: p_median_raw[3] * (1 + c * 300)},
        "R + LAM + SEI": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 3: p_median_raw[3] * (1 + c * 300)},
        "Full realistic": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 4: c * 0.1,
                                     3: p_median_raw[3] * (1 + c * 300), 1: p_median_raw[1] * (1 - c * 0.5)},
    }

    all_results = {}

    for traj_name, param_fn in trajectories.items():
        logger.info(f"\n--- {traj_name} ---")
        traj_data = {"nnls": [], "enet": [], "constrained": [],
                     "group": [], "multicr": [], "cycle_rmse": []}

        prev_c = None
        for cyc in range(n_cycles):
            progress = cyc / max(n_cycles - 1, 1)
            changes = param_fn(progress)
            p_target = p_median_raw.copy()
            for j, v in changes.items():
                p_target[j] = v

            p_t_reg = p_target.copy()
            for j in LOG_PARAMS:
                p_t_reg[j] = np.log10(p_target[j] + 1e-30)
            p_t_norm = (p_t_reg - p_med) / P_std
            dists = np.sqrt(np.sum((P_norm_all - p_t_norm) ** 2, axis=1))
            idx_t = np.argmin(dists)

            dV = V1[idx_t] - V_base
            dV_05 = V05[idx_t] - V_base_05
            dV_2 = V2[idx_t] - V_base_2

            c_nnls = method_nnls(dV, sigs_1c)
            c_enet = method_elastic_net(dV, sigs_1c)
            c_const = method_constrained_nnls(dV, sigs_1c, prev_c)
            prev_c = c_const.copy()
            g_nnls = method_group_nnls(dV, group_sigs)
            c_multi = method_multicr_nnls([dV, dV_05, dV_2], [sigs_1c, sigs_05c, sigs_2c])

            rmse = np.sqrt(np.mean((sigs_1c.T @ c_nnls - dV) ** 2)) * 1000

            traj_data["nnls"].append(c_nnls)
            traj_data["enet"].append(c_enet)
            traj_data["constrained"].append(c_const)
            traj_data["group"].append(g_nnls)
            traj_data["multicr"].append(c_multi)
            traj_data["cycle_rmse"].append(rmse)

        for key in ["nnls", "enet", "constrained", "multicr"]:
            traj_data[key] = np.array(traj_data[key])

        # Evaluate each method
        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())
        gt_names = [PNAMES[j] for j in gt_idx]

        # Map gt to groups
        gt_groups = set()
        for j in gt_idx:
            for gname, gidxs in GROUPS.items():
                if j in gidxs:
                    gt_groups.add(gname)

        final_nnls = traj_data["nnls"][-1]
        final_enet = traj_data["enet"][-1]
        final_const = traj_data["constrained"][-1]
        final_multi = traj_data["multicr"][-1]
        final_group = traj_data["group"][-1]

        def pct_correct(c, gt):
            return sum(c[j] for j in gt) / (c.sum() + 1e-10) * 100

        def group_pct_correct(g_dict, gt_groups):
            active = sum(g_dict[g] for g in gt_groups)
            total = sum(g_dict.values()) + 1e-10
            return active / total * 100

        nnls_pct = pct_correct(final_nnls, gt_idx)
        enet_pct = pct_correct(final_enet, gt_idx)
        const_pct = pct_correct(final_const, gt_idx)
        multi_pct = pct_correct(final_multi, gt_idx)
        group_pct = group_pct_correct(final_group, gt_groups)

        traj_data["eval"] = {
            "gt_idx": gt_idx, "gt_names": gt_names, "gt_groups": gt_groups,
            "nnls_pct": nnls_pct, "enet_pct": enet_pct, "const_pct": const_pct,
            "multi_pct": multi_pct, "group_pct": group_pct,
        }

        all_results[traj_name] = traj_data

        logger.info(f"  GT: {gt_names} | Groups: {gt_groups}")
        logger.info(f"  NNLS:       {nnls_pct:5.1f}%  top={PNAMES[np.argmax(final_nnls)]}")
        logger.info(f"  ElasticNet: {enet_pct:5.1f}%  top={PNAMES[np.argmax(final_enet)]}")
        logger.info(f"  Constrained:{const_pct:5.1f}%  top={PNAMES[np.argmax(final_const)]}")
        logger.info(f"  Multi-CR:   {multi_pct:5.1f}%  top={PNAMES[np.argmax(final_multi)]}")
        logger.info(f"  Group:      {group_pct:5.1f}%")

    # --- Summary comparison ---
    logger.info("\n" + "=" * 70)
    logger.info("METHOD COMPARISON SUMMARY")
    logger.info("=" * 70)

    methods = ["nnls_pct", "enet_pct", "const_pct", "multi_pct", "group_pct"]
    method_labels = ["NNLS", "ElasticNet", "Constrained", "Multi-CR", "Group"]
    pass_counts = {m: 0 for m in methods}

    for name, r in all_results.items():
        eval_data = r["eval"]
        line = f"  {name:25s}"
        for m, lbl in zip(methods, method_labels):
            pct = eval_data[m]
            status = "PASS" if pct > 75 else "MARG" if pct > 50 else "FAIL"
            if pct > 75:
                pass_counts[m] += 1
            line += f" | {lbl}: {pct:5.1f}%"
        logger.info(line)

    logger.info(f"\n  Pass rates (>75%):")
    for m, lbl in zip(methods, method_labels):
        logger.info(f"    {lbl:15s}: {pass_counts[m]}/{len(all_results)} ({pass_counts[m]/len(all_results)*100:.0f}%)")

    plot_method_comparison(all_results)
    plot_trajectory_panels(all_results)
    plot_group_method(all_results)

    logger.info(f"\nOutputs: {OUT}/")


def plot_method_comparison(all_results):
    methods = ["nnls_pct", "enet_pct", "const_pct", "multi_pct", "group_pct"]
    labels = ["NNLS", "ElasticNet", "Constr.", "Multi-CR", "Group"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    names = list(all_results.keys())
    x = np.arange(len(names))
    w = 0.15

    for i, (m, lbl) in enumerate(zip(methods, labels)):
        pcts = [all_results[n]["eval"][m] for n in names]
        ax.bar(x + i * w, pcts, w, label=lbl, alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels([n[:15] for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% Correct attribution")
    ax.set_title("Method Comparison (final cycle)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    pass_rates = []
    for m, lbl in zip(methods, labels):
        n_pass = sum(1 for r in all_results.values() if r["eval"][m] > 75)
        pass_rates.append(n_pass / len(all_results) * 100)
    bars = ax.bar(labels, pass_rates, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("% Scenarios PASS")
    ax.set_title("Pass Rate (>75% correct)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "method_comparison_summary.png", dpi=150)
    plt.close(fig)


def plot_trajectory_panels(all_results):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    n_traj = len(all_results)
    fig, axes = plt.subplots(n_traj, 4, figsize=(20, 3 * n_traj))
    if n_traj == 1:
        axes = axes.reshape(1, -1)

    for i, (name, r) in enumerate(all_results.items()):
        gt_idx = r["eval"]["gt_idx"]

        for col, (method_key, method_label) in enumerate([
            ("nnls", "NNLS"), ("constrained", "Constr."),
            ("multicr", "Multi-CR"), ("enet", "ElasticNet")
        ]):
            ax = axes[i, col]
            coeffs = r[method_key]
            for j in range(7):
                c_smooth = gaussian_filter1d(coeffs[:, j], sigma=3)
                lw = 2 if j in gt_idx else 1
                alpha = 1.0 if j in gt_idx else 0.4
                ls = "-" if j in gt_idx else "--"
                ax.plot(range(len(c_smooth)), c_smooth, color=colors[j],
                       linewidth=lw, alpha=alpha, linestyle=ls, label=PNAMES[j])

            pct_key = f"{method_key}_pct"
            pct = r["eval"].get(pct_key, r["eval"].get("group_pct", 0))
            color = "green" if pct > 75 else "orange" if pct > 50 else "red"
            ax.set_title(f"{method_label} | {pct:.0f}%", fontsize=9, color=color)
            if i == 0:
                ax.legend(fontsize=6, ncol=4)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(name, fontsize=9)

    fig.suptitle("Trajectory Decomposition: All Methods (solid=GT active, dashed=inactive)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_method(all_results):
    gcolors = [GROUP_COLORS[g] for g in GROUP_NAMES]
    n_traj = len(all_results)
    fig, axes = plt.subplots(n_traj, 2, figsize=(12, 3 * n_traj))
    if n_traj == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(all_results.keys()):
        r = all_results[name]
        gt_groups = r["eval"]["gt_groups"]

        ax = axes[i, 0]
        group_coeffs = r["group"]
        for g_i, gname in enumerate(GROUP_NAMES):
            vals = [gc[gname] for gc in group_coeffs]
            lw = 2 if gname in gt_groups else 1
            alpha = 1.0 if gname in gt_groups else 0.4
            ax.plot(range(len(vals)), vals, color=gcolors[g_i],
                   linewidth=lw, alpha=alpha, label=gname)
        ax.set_title(f"Group decomposition ({r['eval']['group_pct']:.0f}%)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == n_traj - 1:
            ax.set_xlabel("Cycle")

        ax = axes[i, 1]
        gt_names = r["eval"]["gt_names"]
        final_nnls = r["nnls"][-1]
        total = final_nnls.sum() + 1e-10
        gt_sum = sum(final_nnls[j] for j in r["eval"]["gt_idx"])

        bar_colors = ["green" if PNAMES[j] in gt_names else "gray" for j in range(7)]
        ax.bar(range(7), final_nnls / total * 100, color=bar_colors, alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Attribution (%)")
        ax.set_title(f"NNLS final: {r['eval']['nnls_pct']:.0f}% correct", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Group Decomposition vs NNLS", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "group_vs_nnls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
