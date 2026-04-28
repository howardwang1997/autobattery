"""Fixed temporal-constrained degradation decomposition.

Two-stage approach:
1. Stage 1: Per-cycle NNLS to identify which modes are active
2. Stage 2: Apply monotonicity + smoothness only to consistently active modes

This fixes the issue where the original temporal method forced ALL detected 
modes to grow monotonically, amplifying errors from spurious detections.
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
from scipy.optimize import minimize, nnls
from sklearn.linear_model import Ridge
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation/temporal_v2")
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


def extract_dVs(data, n_ref=5):
    current, voltage, time_arr, cycles = data["current"], data["voltage"], data["time"], data["cycle"]
    unique = np.unique(cycles)
    curves = []
    for cyc in unique:
        mask = (cycles == cyc) & (current < -0.005)
        idx = np.where(mask)[0]
        if len(idx) < 15:
            continue
        v_s, i_s, t_s = voltage[idx], current[idx], time_arr[idx]
        valid = ~np.isnan(v_s)
        v_s, i_s, t_s = v_s[valid], i_s[valid], t_s[valid]
        if len(v_s) < 15:
            continue
        dt = np.diff(t_s)
        if dt.sum() < 10:
            continue
        cap = -np.sum(i_s[:-1] * dt) / 3600
        if cap < 0.001:
            continue
        curves.append({"cycle": int(cyc), "voltage": v_s, "capacity": cap})

    ref_curves = curves[:n_ref]
    v_refs = []
    for cc in ref_curves:
        v_r = np.interp(np.linspace(0, 1, N_TIME), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        v_refs.append(v_r)
    V_ref = np.mean(v_refs, axis=0)

    dVs = []
    result_curves = []
    for cc in curves:
        v_exp = np.interp(np.linspace(0, 1, N_TIME), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        dV = gaussian_filter1d(v_exp - V_ref, sigma=3)
        dVs.append(dV)
        result_curves.append(cc)

    return np.array(dVs), result_curves, V_ref


def solve_per_cycle_nnls(dVs, signatures, reg=0.01):
    """Per-cycle NNLS baseline."""
    n_modes = signatures.shape[0]
    n_cycles = dVs.shape[0]
    A = signatures.T
    coeffs = np.zeros((n_modes, n_cycles))
    rmses = []
    for cyc in range(n_cycles):
        A_aug = np.vstack([A, np.eye(n_modes) * reg])
        b_aug = np.concatenate([dVs[cyc], np.zeros(n_modes)])
        c, _ = nnls(A_aug, b_aug)
        coeffs[:, cyc] = c
        recon = A @ c
        rmses.append(np.sqrt(np.mean((recon - dVs[cyc]) ** 2)) * 1000)
    return coeffs, np.array(rmses)


def solve_two_stage_temporal(dVs, signatures, lambda_smooth=0.1, lambda_mono=0.5,
                             activity_threshold=0.3):
    """Two-stage temporal constrained decomposition.

    Stage 1: Per-cycle NNLS to identify active modes
    Stage 2: Joint optimization with monotonicity only on active modes
    """
    n_modes = signatures.shape[0]
    n_cycles = dVs.shape[0]
    A = signatures.T

    c_stage1, rmses1 = solve_per_cycle_nnls(dVs, signatures)

    activity = np.zeros(n_modes)
    for j in range(n_modes):
        n_positive = np.sum(c_stage1[j] > 1e-6)
        activity[j] = n_positive / n_cycles

    active_modes = activity > activity_threshold
    logger.info(f"  Active modes (>{activity_threshold*100:.0f}% cycles): "
                f"{[PNAMES[j] for j in range(7) if active_modes[j]]}")
    logger.info(f"  Inactive modes: {[PNAMES[j] for j in range(7) if not active_modes[j]]}")
    logger.info(f"  Activity: {', '.join(f'{PNAMES[j]}={activity[j]:.2f}' for j in range(7))}")

    n_vars = n_modes * n_cycles

    def objective(x):
        C = x.reshape(n_modes, n_cycles)
        residual = 0.0
        for cyc in range(n_cycles):
            recon = A @ C[:, cyc]
            residual += np.sum((recon - dVs[cyc]) ** 2)

        smoothness = 0.0
        for j in range(n_modes):
            if active_modes[j]:
                diff = np.diff(C[j])
                smoothness += np.sum(diff ** 2)

        mono_penalty = 0.0
        for j in range(n_modes):
            if active_modes[j]:
                violations = np.maximum(0, -(np.diff(C[j])))
                mono_penalty += np.sum(violations ** 2)

        return residual + lambda_smooth * smoothness + lambda_mono * mono_penalty

    def objective_grad(x):
        C = x.reshape(n_modes, n_cycles)
        grad = np.zeros_like(C)
        for cyc in range(n_cycles):
            recon = A @ C[:, cyc]
            err = recon - dVs[cyc]
            grad[:, cyc] += 2 * A.T @ err

        for j in range(n_modes):
            if active_modes[j]:
                diff = np.diff(C[j])
                grad[j, 1:] += 2 * lambda_smooth * diff
                grad[j, :-1] -= 2 * lambda_smooth * diff

                violations = np.maximum(0, -(diff))
                grad[j, 1:] -= 2 * lambda_mono * violations
                grad[j, :-1] += 2 * lambda_mono * violations

        return grad.flatten()

    x0 = np.zeros(n_vars)
    for j in range(n_modes):
        if active_modes[j]:
            cummax = np.maximum.accumulate(c_stage1[j])
            x0[j * n_cycles:(j + 1) * n_cycles] = cummax
        else:
            x0[j * n_cycles:(j + 1) * n_cycles] = 0.0

    bounds = [(0, None)] * n_vars

    logger.info(f"  Optimizing {n_vars} vars (active={active_modes.sum()})...")
    result = minimize(objective, x0, jac=objective_grad, bounds=bounds,
                      method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-14})

    C_opt = result.x.reshape(n_modes, n_cycles)
    logger.info(f"  Converged: {result.success}, nit={result.nit}")

    rmses = []
    for cyc in range(n_cycles):
        recon = A @ C_opt[:, cyc]
        rmses.append(np.sqrt(np.mean((recon - dVs[cyc]) ** 2)) * 1000)

    return C_opt, np.array(rmses), active_modes


def synthetic_validate(sigs):
    """Validate two-stage temporal on synthetic data."""
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

    p_median = np.median(P1, axis=0)
    idx_base = np.argmin(np.abs(P_norm).sum(axis=1))
    V_base = V1[idx_base]

    n_synth_cycles = 60
    trajectories = {
        "R_mult only": lambda c: {6: 1.0 + c * 4.0},
        "LAM_pos only": lambda c: {5: c * 0.3},
        "SEI only": lambda c: {3: p_median[3] * (1 + c * 500)},
        "R + LAM_pos": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2},
        "R + SEI": lambda c: {6: 1.0 + c * 3.0, 3: p_median[3] * (1 + c * 300)},
        "LAM + SEI": lambda c: {5: c * 0.2, 3: p_median[3] * (1 + c * 300)},
        "R + LAM + SEI": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 3: p_median[3] * (1 + c * 300)},
        "Full realistic": lambda c: {6: 1.0 + c * 3.0, 5: c * 0.2, 4: c * 0.1,
                                     3: p_median[3] * (1 + c * 300), 1: p_median[1] * (1 - c * 0.5)},
    }

    all_results = {}

    for traj_name, param_fn in trajectories.items():
        dVs = []
        gt_coeffs = np.zeros((7, n_synth_cycles))

        for cyc in range(n_synth_cycles):
            progress = cyc / max(n_synth_cycles - 1, 1)
            changes = param_fn(progress)
            p_target = p_median.copy()
            for j, v in changes.items():
                p_target[j] = v

            p_t_reg = p_target.copy()
            for j in LOG_PARAMS:
                p_t_reg[j] = np.log10(p_target[j] + 1e-30)
            p_t_norm = (p_t_reg - p_med) / P_std
            dists = np.sqrt(np.sum((P_norm - p_t_norm) ** 2, axis=1))
            idx_t = np.argmin(dists)
            dVs.append(V1[idx_t] - V_base)

            for j, v in changes.items():
                gt_coeffs[j, cyc] = v

        dVs = np.array(dVs)

        c_baseline, rmse_base = solve_per_cycle_nnls(dVs, sigs)
        c_twostage, rmse_ts, active = solve_two_stage_temporal(
            dVs, sigs, lambda_smooth=0.1, lambda_mono=0.5, activity_threshold=0.3
        )

        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())

        def eval_pct(coeffs_mat, cycle_idx=-1):
            c = coeffs_mat[:, cycle_idx]
            return sum(c[j] for j in gt_idx) / (c.sum() + 1e-10) * 100

        base_pct = eval_pct(c_baseline)
        ts_pct = eval_pct(c_twostage)

        all_results[traj_name] = {
            "baseline": c_baseline,
            "twostage": c_twostage,
            "rmse_base": rmse_base,
            "rmse_ts": rmse_ts,
            "gt_idx": gt_idx,
            "base_pct": base_pct,
            "ts_pct": ts_pct,
            "active": active,
        }

        logger.info(f"  {traj_name:25s}: Base={base_pct:5.1f}% 2Stage={ts_pct:5.1f}% "
                    f"(active={[PNAMES[j] for j in range(7) if active[j]]})")

    return all_results


def plot_synthetic_results(synth):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(synth.keys())
    x = np.arange(len(names))
    base_pcts = [synth[k]["base_pct"] for k in names]
    ts_pcts = [synth[k]["ts_pct"] for k in names]

    ax.bar(x - 0.2, base_pcts, 0.35, label="Baseline NNLS", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, ts_pcts, 0.35, label="Two-Stage Temporal", color="coral", alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:18] for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% Correct Attribution")
    ax.set_title("Two-Stage Temporal vs Baseline (Synthetic)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_validation.png", dpi=150)
    plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("TWO-STAGE TEMPORAL DECOMPOSITION (FIXED)")
    logger.info("=" * 60)

    logger.info("\nComputing signatures...")
    sigs = compute_signatures()

    logger.info("\n" + "=" * 60)
    logger.info("SYNTHETIC VALIDATION")
    logger.info("=" * 60)
    synth = synthetic_validate(sigs)

    base_pass = sum(1 for r in synth.values() if r["base_pct"] > 75)
    ts_pass = sum(1 for r in synth.values() if r["ts_pct"] > 75)
    logger.info(f"\n  Baseline NNLS:     {base_pass}/{len(synth)} PASS")
    logger.info(f"  Two-Stage Temporal: {ts_pass}/{len(synth)} PASS")

    plot_synthetic_results(synth)

    logger.info("\n" + "=" * 60)
    logger.info("GROUND TRUTH VALIDATION (if available)")
    logger.info("=" * 60)

    gt_path = Path("data/ground_truth/ground_truth_multicycle.h5")
    if gt_path.exists():
        logger.info("Found ground truth data, validating...")
        validate_on_ground_truth(sigs, gt_path)
    else:
        logger.info("No ground truth data found. Run script 35 first.")

    logger.info(f"\nOutputs saved to {OUT}/")


def validate_on_ground_truth(sigs, gt_path):
    """Validate decomposition against PyBaMM ground truth."""
    with h5py.File(gt_path, "r") as f:
        scenarios = list(f.keys())
        logger.info(f"Ground truth scenarios: {scenarios}")

        all_results = {}
        param_names_h5 = list(f.attrs.get("param_names", [f"p{j}" for j in range(7)]))
        skip_scenarios = ["baseline"]
        
        for sc_name in scenarios:
            if sc_name in skip_scenarios:
                continue
            grp = f[sc_name]
            if "V_cycles" not in grp or "capacity" not in grp:
                continue

            V = grp["V_cycles"][:]
            cap = grp["capacity"][:]
            gt_params = grp["gt_params"][:] if "gt_params" in grp else None
            n_cycles = V.shape[0]

            if n_cycles < 10:
                logger.info(f"  {sc_name}: too few cycles ({n_cycles}), skipping")
                continue

            V_ref = V[:5].mean(axis=0)
            dVs = np.array([gaussian_filter1d(V[c] - V_ref, sigma=3) for c in range(n_cycles)])

            c_base, rmse_base = solve_per_cycle_nnls(dVs, sigs)
            c_ts, rmse_ts, active = solve_two_stage_temporal(dVs, sigs)

            c_smooth_base = gaussian_filter1d(c_base, sigma=3, axis=1)
            c_smooth_ts = gaussian_filter1d(c_ts, sigma=3, axis=1)

            gt_sei = gt_params[:, 3] if gt_params is not None else None
            gt_lam_pos = gt_params[:, 5] if gt_params is not None else None

            r_sei_base = np.corrcoef(c_smooth_base[3], cap)[0, 1] if c_smooth_base[3].std() > 1e-10 else 0
            r_lam_base = np.corrcoef(c_smooth_base[5], cap)[0, 1] if c_smooth_base[5].std() > 1e-10 else 0
            r_sei_ts = np.corrcoef(c_smooth_ts[3], cap)[0, 1] if c_smooth_ts[3].std() > 1e-10 else 0
            r_lam_ts = np.corrcoef(c_smooth_ts[5], cap)[0, 1] if c_smooth_ts[5].std() > 1e-10 else 0

            logger.info(f"\n  {sc_name} ({n_cycles} cycles, cap fade={1-cap[-1]/cap[0]:.1%}):")
            logger.info(f"    RMSE: base={rmse_base.mean():.1f} mV, 2stage={rmse_ts.mean():.1f} mV")
            logger.info(f"    SEI corr w/ cap: base={r_sei_base:+.3f}, 2stage={r_sei_ts:+.3f}")
            logger.info(f"    LAM_pos corr w/ cap: base={r_lam_base:+.3f}, 2stage={r_lam_ts:+.3f}")
            logger.info(f"    Active modes: {[PNAMES[j] for j in range(7) if active[j]]}")

            if gt_sei is not None and gt_sei.std() > 1e-12:
                r_sei_gt_base = np.corrcoef(c_smooth_base[3][:len(gt_sei)], gt_sei[:len(c_smooth_base[3])])[0, 1]
                r_sei_gt_ts = np.corrcoef(c_smooth_ts[3][:len(gt_sei)], gt_sei[:len(c_smooth_ts[3])])[0, 1]
                logger.info(f"    SEI corr w/ GT: base={r_sei_gt_base:+.3f}, 2stage={r_sei_gt_ts:+.3f}")

            if gt_lam_pos is not None and gt_lam_pos.std() > 1e-12:
                r_lam_gt_base = np.corrcoef(c_smooth_base[5][:len(gt_lam_pos)], gt_lam_pos[:len(c_smooth_base[5])])[0, 1]
                r_lam_gt_ts = np.corrcoef(c_smooth_ts[5][:len(gt_lam_pos)], gt_lam_pos[:len(c_smooth_ts[5])])[0, 1]
                logger.info(f"    LAM_pos corr w/ GT: base={r_lam_gt_base:+.3f}, 2stage={r_lam_gt_ts:+.3f}")

            all_results[sc_name] = {
                "c_base": c_base, "c_ts": c_ts,
                "rmse_base": rmse_base, "rmse_ts": rmse_ts,
                "cap": cap,
                "gt_sei": gt_sei, "gt_lam_pos": gt_lam_pos,
                "r_sei_base": r_sei_base, "r_sei_ts": r_sei_ts,
                "r_lam_base": r_lam_base, "r_lam_ts": r_lam_ts,
            }

        if all_results:
            plot_ground_truth_validation(all_results)

    return all_results


def plot_ground_truth_validation(results):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    n_sc = len(results)
    fig, axes = plt.subplots(n_sc, 3, figsize=(18, 4 * n_sc))
    if n_sc == 1:
        axes = axes.reshape(1, -1)

    for i, (sc_name, r) in enumerate(results.items()):
        n_cyc = len(r["cap"])
        cycles = np.arange(n_cyc)

        ax = axes[i, 0]
        ax.plot(cycles, r["cap"] / r["cap"][0] * 100, "k-", linewidth=2)
        ax.set_ylabel("Cap (%)")
        ax.set_title(f"{sc_name}: Capacity")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        for j in range(7):
            c_s = gaussian_filter1d(r["c_base"][j], 3)
            ax.plot(cycles, c_s, color=colors[j], linewidth=1.5, label=PNAMES[j], alpha=0.8)
        ax.set_ylabel("Coeff")
        ax.set_title(f"{sc_name}: Baseline NNLS")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        for j in range(7):
            c_s = gaussian_filter1d(r["c_ts"][j], 3)
            ax.plot(cycles, c_s, color=colors[j], linewidth=1.5, label=PNAMES[j], alpha=0.8)
        ax.set_ylabel("Coeff")
        ax.set_title(f"{sc_name}: Two-Stage Temporal")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ground Truth Validation: Baseline vs Two-Stage Temporal", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ground_truth_validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
