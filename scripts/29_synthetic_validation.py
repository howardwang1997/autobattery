"""Synthetic ground truth validation — v4: PyBaMM data, trajectory-based.

Use actual PyBaMM simulations with known parameters as test cases.
Simulate realistic degradation trajectories by interpolating between
parameter sets in the dataset.
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

OUT = Path("outputs/degradation/validation")
OUT.mkdir(parents=True, exist_ok=True)

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_TIME = 100


def load_data(c_rate=1.0):
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    mask = np.abs(cr - c_rate) < 0.01
    return V[mask], P[mask]


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
    return sigs, p_med, P_std


def find_sim(P, target_raw):
    P_reg = P.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P[:, j] + 1e-30)
    target_reg = target_raw.copy()
    for j in LOG_PARAMS:
        target_reg[j] = np.log10(target_raw[j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_std = P_reg.std(axis=0) + 1e-12
    P_norm = (P_reg - p_med) / P_std
    t_norm = (target_reg - p_med) / P_std
    dists = np.sqrt(np.sum((P_norm - t_norm) ** 2, axis=1))
    return np.argmin(dists)


def decompose(dV, signatures, reg=0.01):
    n_sigs = signatures.shape[0]
    A = signatures.T
    A_aug = np.vstack([A, np.eye(n_sigs) * reg])
    b_aug = np.concatenate([dV, np.zeros(n_sigs)])
    coeffs, _ = nnls(A_aug, b_aug)
    dV_recon = A @ coeffs
    rmse = np.sqrt(np.mean((dV_recon - dV) ** 2)) * 1000
    return coeffs, rmse


def eval_accuracy(coeffs, gt_indices):
    active = sum(coeffs[j] for j in gt_indices)
    total = coeffs.sum() + 1e-10
    return active / total * 100


def main():
    V1, P1 = load_data(1.0)
    sigs, p_med_log, P_std_log = compute_ridge_signatures(V1, P1)
    p_median_raw = np.median(P1, axis=0)
    logger.info(f"Data: {V1.shape[0]} sims, signatures computed")

    corr = np.corrcoef(sigs)
    logger.info("Signature correlation at 1C (Ridge):")
    pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    pairs.sort(key=lambda ij: -abs(corr[ij[0], ij[1]]))
    for i, j in pairs[:5]:
        logger.info(f"  {PNAMES[i]:12s} ↔ {PNAMES[j]:12s}: r={corr[i,j]:.3f}")

    idx_base = find_sim(P1, p_median_raw)
    V_base = V1[idx_base]
    logger.info(f"Baseline sim #{idx_base}")

    n_sims = V1.shape[0]
    rng = np.random.default_rng(42)

    n_test = 200
    test_idx = rng.choice(n_sims, size=min(n_test, n_sims), replace=False)
    test_idx = test_idx[test_idx != idx_base]

    P_reg_all = P1.copy()
    for j in LOG_PARAMS:
        P_reg_all[:, j] = np.log10(P1[:, j] + 1e-30)
    P_norm_all = (P_reg_all - p_med_log) / P_std_log

    dominant_param = np.argmax(np.abs(P_norm_all), axis=1)

    per_param_results = {j: {"correct_pct": [], "rmse": [], "top_match": []} for j in range(7)}

    for ti in test_idx:
        dV = V1[ti] - V_base
        coeffs, rmse = decompose(dV, sigs)

        gt_param = dominant_param[ti]
        top_recovered = np.argmax(coeffs)

        pct = eval_accuracy(coeffs, [gt_param])
        per_param_results[gt_param]["correct_pct"].append(pct)
        per_param_results[gt_param]["rmse"].append(rmse)
        per_param_results[gt_param]["top_match"].append(1 if top_recovered == gt_param else 0)

    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER-LEVEL RECOVERY (200 random test cases)")
    logger.info("=" * 60)
    for j in range(7):
        r = per_param_results[j]
        n = len(r["correct_pct"])
        if n == 0:
            continue
        avg_pct = np.mean(r["correct_pct"])
        top_match_rate = np.mean(r["top_match"])
        avg_rmse = np.mean(r["rmse"])
        logger.info(f"  {PNAMES[j]:12s}: {n:3d} cases, "
                     f"correct={avg_pct:.0f}%, top-match={top_match_rate:.0f}%, "
                     f"RMSE={avg_rmse:.1f}mV")

    scenarios = {
        "R_mult only": np.where((np.abs(P_norm_all[:, 6]) > 1.5) & (np.abs(P_norm_all[:, :6]).max(axis=1) < 0.5))[0],
        "LAM_pos only": np.where((np.abs(P_norm_all[:, 5]) > 1.5) & (np.abs(P_norm_all[:, :5].tolist() + [P_norm_all[:, 6]]).max(axis=1) if False else np.abs(np.delete(P_norm_all, 5, axis=1)).max(axis=1) < 0.5))[0],
    }

    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO: Simulations with ONE dominant parameter")
    logger.info("=" * 60)

    for j in range(7):
        mask_dominant = (dominant_param == j) & (np.abs(P_norm_all).max(axis=1) > 1.0)
        idx_dom = np.where(mask_dominant)[0]
        idx_dom = idx_dom[idx_dom != idx_base]
        if len(idx_dom) < 5:
            logger.info(f"  {PNAMES[j]:12s}: too few cases ({len(idx_dom)})")
            continue

        top_matches = []
        correct_pcts = []
        rmses = []
        for ti in idx_dom[:50]:
            dV = V1[ti] - V_base
            coeffs, rmse = decompose(dV, sigs)
            top_matches.append(np.argmax(coeffs) == j)
            correct_pcts.append(eval_accuracy(coeffs, [j]))
            rmses.append(rmse)

        logger.info(f"  {PNAMES[j]:12s}: {len(idx_dom):3d} sims, "
                     f"top-match={np.mean(top_matches)*100:.0f}%, "
                     f"correct={np.mean(correct_pcts):.0f}%, "
                     f"RMSE={np.mean(rmses):.1f}mV")

    logger.info("\n" + "=" * 60)
    logger.info("TRAJECTORY SIMULATION: Gradual degradation")
    logger.info("=" * 60)

    n_cycles = 100
    traj_results = {}
    trajectories = {
        "R_mult growth": lambda c: {6: 1.0 + c * 4.0},
        "LAM_pos growth": lambda c: {5: c * 0.3},
        "SEI growth": lambda c: {3: p_median_raw[3] * (1 + c * 500)},
        "R_mult + LAM_pos": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2},
        "Realistic (R+LAM+SEI)": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 3: p_median_raw[3] * (1 + c * 300)},
    }

    for traj_name, param_fn in trajectories.items():
        cycle_coeffs = []
        cycle_rmses = []
        for cyc in range(n_cycles):
            progress = cyc / (n_cycles - 1)
            changes = param_fn(progress)
            p_target = p_median_raw.copy()
            for j, v in changes.items():
                p_target[j] = v

            idx_t = find_sim(P1, p_target)
            dV = V1[idx_t] - V_base
            coeffs, rmse = decompose(dV, sigs)
            cycle_coeffs.append(coeffs)
            cycle_rmses.append(rmse)

        cycle_coeffs = np.array(cycle_coeffs)
        cycle_rmses = np.array(cycle_rmses)
        traj_results[traj_name] = {
            "coeffs": cycle_coeffs,
            "rmses": cycle_rmses,
            "changes": param_fn,
        }

        final = cycle_coeffs[-1]
        total = final.sum() + 1e-10
        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())
        gt_names = [PNAMES[j] for j in gt_idx]
        active_pct = sum(final[j] for j in gt_idx) / total * 100
        top3 = [PNAMES[j] for j in np.argsort(final)[::-1][:3]]

        logger.info(f"  {traj_name}: final correct={active_pct:.0f}%, "
                     f"RMSE={cycle_rmses[-1]:.1f}mV, top3={top3}, GT={gt_names}")

    plot_trajectory_results(traj_results, n_cycles)
    plot_param_level_results(per_param_results)

    logger.info(f"\nOutputs: {OUT}/")


def plot_trajectory_results(traj_results, n_cycles):
    n_traj = len(traj_results)
    fig, axes = plt.subplots(n_traj, 2, figsize=(14, 3 * n_traj))
    if n_traj == 1:
        axes = axes.reshape(1, -1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for i, (name, r) in enumerate(traj_results.items()):
        coeffs = r["coeffs"]
        rmses = r["rmses"]

        ax = axes[i, 0]
        for j in range(7):
            c_smooth = gaussian_filter1d(coeffs[:, j], sigma=3)
            if np.max(np.abs(c_smooth)) > 0.001 * np.max(np.abs(coeffs)):
                ax.plot(range(n_cycles), c_smooth, color=colors[j],
                       linewidth=1.5, label=PNAMES[j])
        ax.set_xlabel("Cycle (normalized)")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"{name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(range(n_cycles), rmses, "r-", linewidth=1.5)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RMSE (mV)")
        ax.set_title(f"Fit error")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Trajectory-based Validation (PyBaMM data)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_level_results(per_param):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    names = []
    top_rates = []
    correct_pcts = []
    for j in range(7):
        r = per_param[j]
        if len(r["top_match"]) == 0:
            continue
        names.append(PNAMES[j])
        top_rates.append(np.mean(r["top_match"]) * 100)
        correct_pcts.append(np.mean(r["correct_pct"]))

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, top_rates, w, label="Top-1 match rate (%)", color="green", alpha=0.7)
    ax.bar(x + w / 2, correct_pcts, w, label="Correct attribution (%)", color="blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.axhline(50, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("%")
    ax.set_title("Per-Parameter Recovery (PyBaMM data, 200 random test cases)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "param_level_recovery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
