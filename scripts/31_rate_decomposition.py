"""Physics-informed rate-dependent decomposition.

Key insight: different degradation modes have different C-rate dependencies.
- R_mult: ΔV ∝ I ∝ C (Ohmic, linear in current)
- D_n/D_p: ΔV ∝ √I ∝ √C (diffusion overpotential, square-root)
- LAM: ΔV independent of rate (capacity loss, kinetically controlled)
- SEI: ΔV ∝ log(I) (Butler-Volmer at low overpotential)

Method:
1. For each time point, compute ΔV at multiple C-rates
2. Fit the C-rate dependence to extract mode contributions
3. This breaks the correlation between modes with similar voltage signatures
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import nnls, minimize
from sklearn.linear_model import Ridge
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation/validation/rate_decomposition")
OUT.mkdir(parents=True, exist_ok=True)

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_TIME = 100
C_RATES = [0.5, 1.0, 2.0]


def load_data():
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    data = {}
    for c in C_RATES:
        mask = np.abs(cr - c) < 0.01
        data[c] = (V[mask], P[mask])
    return data


def compute_ridge_signatures(V, P):
    n_time = V.shape[1]
    P_reg = P.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P[:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_norm = (P_reg - p_med)
    P_std = P_norm.std(axis=0) + 1e-12
    P_norm /= P_std
    sigs = np.zeros((7, n_time))
    for t in range(n_time):
        m = Ridge(alpha=0.1)
        m.fit(P_norm, V[:, t])
        sigs[:, t] = m.coef_
    return sigs


def compute_rate_signatures(data):
    """Compute signatures at each C-rate and their C-rate dependence."""
    sigs_per_cr = {}
    for c in C_RATES:
        V, P = data[c]
        sigs_per_cr[c] = compute_ridge_signatures(V, P)

    # Build augmented signature matrix: [sig(0.5C), sig(1C), sig(2C)]
    # Each mode has 3*N_TIME columns
    n_time = N_TIME
    augmented_sigs = np.zeros((7, 3 * n_time))
    for i, c in enumerate(C_RATES):
        augmented_sigs[:, i * n_time:(i + 1) * n_time] = sigs_per_cr[c]

    return sigs_per_cr, augmented_sigs


def compute_rate_feature_signatures(data):
    """Decompose rate-dependent signatures into physically motivated features.

    For each mode j, fit ΔV_j(C, t) to:
    ΔV = a_j(t) * C + b_j(t) * √C + c_j(t) * 1 + d_j(t) * log(C)

    This gives 4 feature signatures per mode:
    - a_j: Ohmic (rate-linear) component
    - b_j: Diffusion (rate-sqrt) component
    - c_j: Rate-independent component
    - d_j: Kinetic (rate-log) component
    """
    sigs_per_cr = {}
    for c in C_RATES:
        V, P = data[c]
        sigs_per_cr[c] = compute_ridge_signatures(V, P)

    n_time = N_TIME
    C_arr = np.array(C_RATES)

    # Feature basis functions in C-rate
    features = {
        "ohmic": C_arr,       # linear in C
        "diffusion": np.sqrt(C_arr),  # sqrt(C)
        "independent": np.ones(3),  # rate-independent
    }
    n_features = len(features)
    F = np.column_stack(list(features.values()))

    # For each mode, project C-rate signatures onto feature basis
    # sigs_per_cr[c][j, t] ≈ Σ_f coeff_jf(t) * feature_f(c)
    # At each time point: [s(0.5C), s(1C), s(2C)] = F @ [a, b, c]

    # F is 3x3, invertible
    F_inv = np.linalg.inv(F)

    n_modes = 7
    feature_sigs = np.zeros((n_modes * n_features, n_time))
    for j in range(n_modes):
        S_cr = np.array([sigs_per_cr[c][j] for c in C_RATES])  # (3, n_time)
        coeffs = F_inv @ S_cr  # (3, n_time)
        for f_idx in range(n_features):
            feature_sigs[j * n_features + f_idx] = coeffs[f_idx]

    return feature_sigs, features, n_features


def decompose_nnls(dV, sigs, reg=0.01):
    n_sigs = sigs.shape[0]
    A = sigs.T
    A_aug = np.vstack([A, np.eye(n_sigs) * reg])
    b_aug = np.concatenate([dV, np.zeros(n_sigs)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def decompose_rate_features(dVs, feature_sigs, n_features, n_modes, reg=0.01):
    """Decompose multi-C-rate ΔV using rate feature signatures."""
    n_time = dVs[0].shape[0]
    A_blocks = []
    for i, c in enumerate(C_RATES):
        block = np.zeros((n_time, n_modes))
        for j in range(n_modes):
            for f_idx in range(n_features):
                feat_name = ["ohmic", "diffusion", "independent"][f_idx]
                feat_vals = {"ohmic": c, "diffusion": np.sqrt(c), "independent": 1.0}
                block[:, j] += feature_sigs[j * n_features + f_idx] * feat_vals[feat_name]
        A_blocks.append(block)

    A = np.vstack(A_blocks)
    b = np.concatenate(dVs)
    A_aug = np.vstack([A, np.eye(n_modes) * reg])
    b_aug = np.concatenate([b, np.zeros(n_modes)])
    coeffs, _ = nnls(A_aug, b_aug)
    return coeffs


def find_nearest(P, target_raw, p_med, P_std):
    P_reg = P.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P[:, j] + 1e-30)
    target_reg = target_raw.copy()
    for j in LOG_PARAMS:
        target_reg[j] = np.log10(target_raw[j] + 1e-30)
    P_norm = (P_reg - p_med) / P_std
    t_norm = (target_reg - p_med) / P_std
    dists = np.sqrt(np.sum((P_norm - t_norm) ** 2, axis=1))
    return np.argmin(dists)


def eval_pct(coeffs, gt_idx):
    return sum(coeffs[j] for j in gt_idx) / (coeffs.sum() + 1e-10) * 100


def main():
    data = load_data()
    p_median = np.median(data[1.0][1], axis=0)
    P_reg = data[1.0][1].copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(data[1.0][1][:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_std = P_reg.std(axis=0) + 1e-12

    # Compute signatures
    sigs_1c = compute_ridge_signatures(*data[1.0])
    sigs_per_cr, aug_sigs = compute_rate_signatures(data)
    feat_sigs, feat_defs, n_feat = compute_rate_feature_signatures(data)

    # Baseline
    idx_base = np.argmin(np.abs((P_reg - p_med) / P_std).sum(axis=1))
    V_base = {c: data[c][0][idx_base] for c in C_RATES}

    logger.info(f"Baseline #{idx_base}, features: ohmic=C, diffusion=√C, independent=1")

    # Trajectories
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
        results = {"1c_nnls": [], "aug_nnls": [], "rate_feat": [],
                   "rmse_1c": [], "rmse_aug": [], "rmse_feat": []}

        for cyc in range(n_cycles):
            progress = cyc / max(n_cycles - 1, 1)
            changes = param_fn(progress)
            p_target = p_median.copy()
            for j, v in changes.items():
                p_target[j] = v

            dVs = {}
            for c in C_RATES:
                V_c, P_c = data[c]
                idx_t = find_nearest(P_c, p_target, p_med, P_std)
                dVs[c] = V_c[idx_t] - V_base[c]

            # Method 1: 1C NNLS
            c_1c = decompose_nnls(dVs[1.0], sigs_1c)

            # Method 2: Augmented multi-C-rate NNLS
            c_aug = decompose_nnls(np.concatenate([dVs[c] for c in C_RATES]), aug_sigs)

            # Method 3: Rate-feature decomposition
            c_feat = decompose_rate_features(
                [dVs[c] for c in C_RATES], feat_sigs, n_feat, 7
            )

            results["1c_nnls"].append(c_1c)
            results["aug_nnls"].append(c_aug)
            results["rate_feat"].append(c_feat)

            for key, c, sigs in [
                ("1c", c_1c, sigs_1c),
                ("aug", c_aug, aug_sigs),
            ]:
                if key == "1c":
                    A = sigs.T
                    b = dVs[1.0]
                else:
                    A = sigs.T
                    b = np.concatenate([dVs[c] for c in C_RATES])
                rmse = np.sqrt(np.mean((A @ c - b) ** 2)) * 1000
                results[f"rmse_{key}"].append(rmse)

        for key in ["1c_nnls", "aug_nnls", "rate_feat"]:
            results[key] = np.array(results[key])

        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())

        final_1c = results["1c_nnls"][-1]
        final_aug = results["aug_nnls"][-1]
        final_feat = results["rate_feat"][-1]

        pcts = {
            "1C NNLS": eval_pct(final_1c, gt_idx),
            "Multi-CR NNLS": eval_pct(final_aug, gt_idx),
            "Rate-Feature": eval_pct(final_feat, gt_idx),
        }
        method_keys = {"1C NNLS": "1c_nnls", "Multi-CR NNLS": "aug_nnls", "Rate-Feature": "rate_feat"}

        results["eval"] = {
            "gt_idx": gt_idx,
            "gt_names": [PNAMES[j] for j in gt_idx],
            "pcts": pcts,
        }
        all_results[traj_name] = results

        gt_str = ", ".join(PNAMES[j] for j in gt_idx)
        logger.info(f"  {traj_name:25s} [{gt_str}]:")
        for method, pct in pcts.items():
            final = results[method_keys[method]][-1]
            top = PNAMES[np.argmax(final)]
            logger.info(f"    {method:20s}: {pct:5.1f}% top={top}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RATE-DEPENDENT DECOMPOSITION SUMMARY")
    logger.info("=" * 70)
    methods = ["1C NNLS", "Multi-CR NNLS", "Rate-Feature"]
    pass_counts = {m: 0 for m in methods}
    for name, r in all_results.items():
        line = f"  {name:25s}"
        for m in methods:
            pct = r["eval"]["pcts"][m]
            if pct > 75:
                pass_counts[m] += 1
            line += f" | {m}: {pct:5.1f}%"
        logger.info(line)

    logger.info(f"\n  Pass rates (>75%):")
    for m in methods:
        logger.info(f"    {m:20s}: {pass_counts[m]}/{len(all_results)} ({pass_counts[m]/len(all_results)*100:.0f}%)")

    plot_rate_method_comparison(all_results)
    plot_rate_trajectories(all_results)
    plot_r_mult_focus(all_results)

    logger.info(f"\nOutputs: {OUT}/")


def plot_rate_method_comparison(all_results):
    methods = ["1C NNLS", "Multi-CR NNLS", "Rate-Feature"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    names = list(all_results.keys())
    x = np.arange(len(names))
    w = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (m, c) in enumerate(zip(methods, colors)):
        pcts = [all_results[n]["eval"]["pcts"][m] for n in names]
        ax.bar(x + i * w, pcts, w, label=m, color=c, alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x + w)
    ax.set_xticklabels([n[:15] for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% Correct attribution")
    ax.set_title("Method Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    pass_rates = []
    for m in methods:
        n_pass = sum(1 for r in all_results.values() if r["eval"]["pcts"][m] > 75)
        pass_rates.append(n_pass / len(all_results) * 100)
    bars = ax.bar(methods, pass_rates, color=colors)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("% Scenarios PASS")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "rate_method_comparison.png", dpi=150)
    plt.close(fig)


def plot_rate_trajectories(all_results):
    colors_7 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    n_traj = len(all_results)
    fig, axes = plt.subplots(n_traj, 3, figsize=(15, 3 * n_traj))
    if n_traj == 1:
        axes = axes.reshape(1, -1)

    methods = [("1c_nnls", "1C NNLS"), ("aug_nnls", "Multi-CR"), ("rate_feat", "Rate-Feature")]

    for i, (name, r) in enumerate(all_results.items()):
        gt_idx = r["eval"]["gt_idx"]
        for col, (method_key, method_label) in enumerate(methods):
            ax = axes[i, col]
            coeffs = r[method_key]
            for j in range(7):
                c_smooth = gaussian_filter1d(coeffs[:, j], sigma=3)
                lw = 2 if j in gt_idx else 0.8
                alpha = 1.0 if j in gt_idx else 0.3
                ax.plot(range(len(c_smooth)), c_smooth, color=colors_7[j],
                       linewidth=lw, alpha=alpha, label=PNAMES[j] if i == 0 else None)
            pct = r["eval"]["pcts"][method_label]
            color = "green" if pct > 75 else "orange" if pct > 50 else "red"
            ax.set_title(f"{method_label}: {pct:.0f}%", fontsize=9, color=color)
            ax.grid(True, alpha=0.3)
            if i == 0 and col == 0:
                ax.legend(fontsize=6, ncol=4)
            if col == 0:
                ax.set_ylabel(name[:20], fontsize=8)

    fig.suptitle("Rate-Dependent Decomposition (solid=GT active)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "rate_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r_mult_focus(all_results):
    """Focus on R_mult identification improvement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_7 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    r_scenarios = {k: v for k, v in all_results.items() if 6 in v["eval"]["gt_idx"]}

    for si, (name, r) in enumerate(r_scenarios.items()):
        ax = axes[0] if si < 3 else axes[1]
        gt_idx = r["eval"]["gt_idx"]
        for method_key, method_label, ls in [
            ("1c_nnls", "1C", "-"), ("rate_feat", "Rate", "--")
        ]:
            coeffs = r[method_key]
            r_mult_smooth = gaussian_filter1d(coeffs[:, 6], sigma=3)
            label = f"{name[:15]} ({method_label})"
            ax.plot(range(len(r_mult_smooth)), r_mult_smooth, ls=ls,
                   linewidth=2, label=label)

    for ax in axes:
        ax.axhline(0, color="k", alpha=0.3)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("R_mult coefficient")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("R_mult coefficient: 1C NNLS")
    axes[1].set_title("R_mult coefficient: Rate-Feature")

    fig.suptitle("R_mult Identification Focus", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "r_mult_focus.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
