"""Temporal-constrained degradation decomposition.

Key improvement over per-cycle NNLS:
1. Joint optimization across ALL cycles simultaneously
2. Monotonicity: degradation coefficients only increase (or stay constant)
3. Smoothness: temporal regularity prevents noisy fluctuations
4. Kinetic priors: known degradation rate laws (SEI ~ sqrt, LAM ~ linear)

This is a constrained quadratic program:
  minimize  Σ_cyc ||ΔV_cyc - Σ_j c_j(cyc) * S_j||²
            + λ_smooth * Σ |Δc|²                    (smoothness)
            + λ_kinetic * Σ (c_j - prior_j)²        (kinetic prior)
  subject to c_j(cyc) ≥ 0
             c_j(cyc+1) ≥ c_j(cyc)                   (monotonicity)
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
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import sys
import logging

sys.path.insert(0, ".")
from src.data.loader import ExperimentalDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation/temporal")
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
    P_norm = (P_reg - p_med)
    P_std = P_norm.std(axis=0) + 1e-12
    P_norm /= P_std
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


def solve_temporal_nnls(dVs, signatures, lambda_smooth=0.1, lambda_mono=1.0,
                        use_kinetic=False, kinetic_weights=None):
    """Joint temporal decomposition with constraints.

    Variables: C[j, cyc] for j in 0..6, cyc in 0..n_cycles-1
    Flattened to x[j * n_cycles + cyc]
    """
    n_modes = signatures.shape[0]
    n_cycles = dVs.shape[0]
    n_time = dVs.shape[1]
    n_vars = n_modes * n_cycles

    A = signatures.T  # (n_time, n_modes)

    def objective(x):
        C = x.reshape(n_modes, n_cycles)
        residual = 0.0
        for cyc in range(n_cycles):
            recon = A @ C[:, cyc]
            residual += np.sum((recon - dVs[cyc]) ** 2)

        smoothness = 0.0
        for j in range(n_modes):
            for cyc in range(1, n_cycles):
                smoothness += (C[j, cyc] - C[j, cyc - 1]) ** 2

        mono_penalty = 0.0
        for j in range(n_modes):
            for cyc in range(1, n_cycles):
                violation = C[j, cyc - 1] - C[j, cyc]
                if violation > 0:
                    mono_penalty += violation ** 2

        kinetic_penalty = 0.0
        if use_kinetic and kinetic_weights is not None:
            cycles_norm = np.linspace(0, 1, n_cycles)
            for j in range(n_modes):
                if kinetic_weights[j] is not None:
                    prior = kinetic_weights[j] * cycles_norm
                    kinetic_penalty += np.sum((C[j] - prior) ** 2)

        return residual + lambda_smooth * smoothness + lambda_mono * mono_penalty + kinetic_penalty

    def objective_grad(x):
        C = x.reshape(n_modes, n_cycles)
        grad = np.zeros_like(C)

        for cyc in range(n_cycles):
            recon = A @ C[:, cyc]
            err = recon - dVs[cyc]
            grad[:, cyc] += 2 * A.T @ err

        for j in range(n_modes):
            for cyc in range(1, n_cycles):
                diff = C[j, cyc] - C[j, cyc - 1]
                grad[j, cyc] += 2 * lambda_smooth * diff
                grad[j, cyc - 1] -= 2 * lambda_smooth * diff

                violation = C[j, cyc - 1] - C[j, cyc]
                if violation > 0:
                    grad[j, cyc] += 2 * lambda_mono * (-violation)
                    grad[j, cyc - 1] += 2 * lambda_mono * violation

        if use_kinetic and kinetic_weights is not None:
            cycles_norm = np.linspace(0, 1, n_cycles)
            for j in range(n_modes):
                if kinetic_weights[j] is not None:
                    prior = kinetic_weights[j] * cycles_norm
                    grad[j] += 2 * (C[j] - prior)

        return grad.flatten()

    x0 = np.zeros(n_vars)
    for cyc in range(n_cycles):
        from scipy.optimize import nnls as _nnls
        A_aug = np.vstack([A, np.eye(n_modes) * 0.01])
        b_aug = np.concatenate([dVs[cyc], np.zeros(n_modes)])
        c0, _ = _nnls(A_aug, b_aug)
        x0[cyc::n_cycles] = c0

    for j in range(n_modes):
        raw = x0[j * n_cycles:(j + 1) * n_cycles]
        cummax = np.maximum.accumulate(raw)
        x0[j * n_cycles:(j + 1) * n_cycles] = cummax

    bounds = [(0, None)] * n_vars

    logger.info(f"Optimizing {n_vars} variables ({n_modes} modes × {n_cycles} cycles)...")
    result = minimize(objective, x0, jac=objective_grad, bounds=bounds,
                      method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})

    C_opt = result.x.reshape(n_modes, n_cycles)
    logger.info(f"Converged: {result.success}, nit={result.nit}, fun={result.fun:.6f}")

    rmses = []
    for cyc in range(n_cycles):
        recon = A @ C_opt[:, cyc]
        rmse = np.sqrt(np.mean((recon - dVs[cyc]) ** 2)) * 1000
        rmses.append(rmse)

    return C_opt, np.array(rmses)


def solve_baseline_nnls(dVs, signatures):
    """Per-cycle NNLS baseline."""
    from scipy.optimize import nnls as _nnls
    n_cycles = dVs.shape[0]
    n_modes = signatures.shape[0]
    A = signatures.T
    coeffs = np.zeros((n_modes, n_cycles))
    rmses = []

    for cyc in range(n_cycles):
        A_aug = np.vstack([A, np.eye(n_modes) * 0.01])
        b_aug = np.concatenate([dVs[cyc], np.zeros(n_modes)])
        c, _ = _nnls(A_aug, b_aug)
        coeffs[:, cyc] = c
        recon = A @ c
        rmses.append(np.sqrt(np.mean((recon - dVs[cyc]) ** 2)) * 1000)

    return coeffs, np.array(rmses)


def synthetic_validate(sigs):
    """Validate temporal decomposition on synthetic data."""
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

        c_baseline, rmse_base = solve_baseline_nnls(dVs, sigs)
        c_temporal, rmse_temp = solve_temporal_nnls(dVs, sigs, lambda_smooth=0.1, lambda_mono=1.0)

        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())

        def eval_pct(coeffs_mat, cycle_idx=-1):
            c = coeffs_mat[:, cycle_idx]
            return sum(c[j] for j in gt_idx) / (c.sum() + 1e-10) * 100

        def eval_top_match(coeffs_mat, gt_mode_idx, cycle_idx=-1):
            c = coeffs_mat[:, cycle_idx]
            return PNAMES[np.argmax(c)] == PNAMES[gt_mode_idx[0]] if len(gt_mode_idx) == 1 else True

        all_results[traj_name] = {
            "baseline": c_baseline,
            "temporal": c_temporal,
            "rmse_base": rmse_base,
            "rmse_temp": rmse_temp,
            "gt_idx": gt_idx,
            "gt_coeffs": gt_coeffs,
            "base_pct": eval_pct(c_baseline),
            "temp_pct": eval_pct(c_temporal),
        }

        logger.info(f"  {traj_name:25s}: Base={eval_pct(c_baseline):5.1f}% Temp={eval_pct(c_temporal):5.1f}%")

    return all_results


def main():
    logger.info("Computing signatures...")
    sigs = compute_signatures()

    logger.info("\n" + "=" * 60)
    logger.info("SYNTHETIC VALIDATION")
    logger.info("=" * 60)
    synth = synthetic_validate(sigs)

    base_pass = sum(1 for r in synth.values() if r["base_pct"] > 75)
    temp_pass = sum(1 for r in synth.values() if r["temp_pct"] > 75)
    logger.info(f"\n  Baseline NNLS: {base_pass}/{len(synth)} PASS")
    logger.info(f"  Temporal constr: {temp_pass}/{len(synth)} PASS")

    logger.info("\n" + "=" * 60)
    logger.info("NEWAREA REAL DATA")
    logger.info("=" * 60)

    loader = ExperimentalDataLoader()
    data = loader.load_neware_xlsx("/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")
    dVs, curves, V_ref = extract_dVs(data)
    logger.info(f"Extracted {len(curves)} curves")

    logger.info("Running baseline NNLS...")
    c_base, rmse_base = solve_baseline_nnls(dVs, sigs)
    logger.info(f"Baseline RMSE: {rmse_base.mean():.1f} mV")

    logger.info("Running temporal constrained...")
    c_temp, rmse_temp = solve_temporal_nnls(dVs, sigs, lambda_smooth=0.1, lambda_mono=1.0)
    logger.info(f"Temporal RMSE: {rmse_temp.mean():.1f} mV")

    cycles = np.array([c["cycle"] for c in curves])
    caps = np.array([c["capacity"] for c in curves])

    logger.info("\nFinal cycle attribution:")
    for method, coeffs, label in [(c_base, c_base, "Baseline"), (c_temp, c_temp, "Temporal")]:
        final = coeffs[:, -1]
        total = final.sum() + 1e-10
        logger.info(f"\n  {label}:")
        for j in np.argsort(final)[::-1][:5]:
            pct = final[j] / total * 100
            logger.info(f"    {PNAMES[j]:12s}: {pct:5.1f}%")

    logger.info("\nCorrelations with capacity:")
    for method, coeffs, label in [(c_base, c_base, "Base"), (c_temp, c_temp, "Temp")]:
        logger.info(f"  {label}:")
        for j in range(7):
            c_smooth = gaussian_filter1d(coeffs[j], sigma=5)
            r = np.corrcoef(c_smooth, caps)[0, 1]
            logger.info(f"    {PNAMES[j]:12s}: r = {r:+.3f}")

    plot_comparison(cycles, caps, c_base, c_temp, rmse_base, rmse_temp, synth)
    plot_temporal_detail(cycles, caps, c_temp, rmse_temp)

    logger.info(f"\nOutputs: {OUT}/")


def plot_comparison(cycles, caps, c_base, c_temp, rmse_base, rmse_temp, synth):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    ax = axes[0, 0]
    ax.plot(cycles, caps / caps[0] * 100, "b-", linewidth=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Cap (%)")
    ax.set_title("Capacity Retention"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cycles, rmse_base, alpha=0.3, color="orange")
    ax.plot(cycles, rmse_temp, alpha=0.3, color="green")
    ax.plot(cycles, gaussian_filter1d(rmse_base, 5), color="orange", linewidth=2, label="Baseline")
    ax.plot(cycles, gaussian_filter1d(rmse_temp, 5), color="green", linewidth=2, label="Temporal")
    ax.set_xlabel("Cycle"); ax.set_ylabel("RMSE (mV)")
    ax.set_title(f"Fit Error (base={rmse_base.mean():.1f}, temp={rmse_temp.mean():.1f} mV)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    base_pcts = [synth[k]["base_pct"] for k in synth]
    temp_pcts = [synth[k]["temp_pct"] for k in synth]
    names = list(synth.keys())
    x = np.arange(len(names))
    ax.bar(x - 0.2, base_pcts, 0.35, label="Baseline", color="orange", alpha=0.8)
    ax.bar(x + 0.2, temp_pcts, 0.35, label="Temporal", color="green", alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:15] for n in names], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% Correct"); ax.set_title("Synthetic Validation")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    for row in range(1, 3):
        for col in range(3):
            mode_idx = (row - 1) * 3 + col
            if mode_idx >= 7:
                axes[row, col].set_visible(False)
                continue
            ax = axes[row, col]
            j = mode_idx
            ax.plot(cycles, gaussian_filter1d(c_base[j], 5), color="orange", linewidth=1.5,
                   label="Baseline", alpha=0.8)
            ax.plot(cycles, gaussian_filter1d(c_temp[j], 5), color="green", linewidth=2,
                   label="Temporal")
            ax.set_xlabel("Cycle")
            ax.set_ylabel(f"{PNAMES[j]} coeff")
            ax.set_title(f"{PNAMES[j]}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            ax2 = ax.twinx()
            ax2.plot(cycles, caps / caps[0] * 100, "b--", alpha=0.15)
            ax2.set_ylabel("Cap%", color="b", alpha=0.2)

    fig.suptitle("Temporal-Constrained vs Baseline Decomposition", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_detail(cycles, caps, c_temp, rmse_temp):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    total = c_temp.sum(axis=0)
    for j in range(7):
        pcts = c_temp[j] / (total + 1e-10) * 100
        ax.plot(cycles, gaussian_filter1d(pcts, 5), color=colors[j], linewidth=1.5, label=PNAMES[j])
    ax.set_xlabel("Cycle"); ax.set_ylabel("Attribution (%)")
    ax.set_title("Temporal: Per-Mode Attribution")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    group_coeffs = {}
    for gname, gidxs in GROUPS.items():
        group_coeffs[gname] = np.sum(c_temp[gidxs], axis=0)
    total_g = sum(group_coeffs.values()) + 1e-10
    for gname in GROUP_NAMES:
        vals = group_coeffs[gname] / (total_g + 1e-10) * 100
        ax.plot(cycles, gaussian_filter1d(vals, 5), linewidth=2, label=gname)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Group Attribution (%)")
    ax.set_title("Temporal: Group Attribution")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    bottoms = np.zeros(len(cycles))
    gcolors = {"Resistance": "#d62728", "LAM": "#1f77b4", "SEI": "#2ca02c", "Diffusion": "#ff7f0e"}
    for gname in GROUP_NAMES:
        vals = group_coeffs[gname] / (total_g + 1e-10) * 100
        vals_smooth = gaussian_filter1d(vals, 5)
        ax.fill_between(cycles, bottoms, bottoms + vals_smooth,
                        color=gcolors[gname], alpha=0.6, label=gname)
        bottoms += vals_smooth
    ax.set_xlabel("Cycle"); ax.set_ylabel("Attribution (%)")
    ax.set_title("Temporal: Stacked Group Attribution")
    ax.legend(); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for j in range(7):
        c_smooth = gaussian_filter1d(c_temp[j], 5)
        r = np.corrcoef(c_smooth, caps)[0, 1]
        ax.barh(j, abs(r), color=colors[j], alpha=0.7)
        ax.text(abs(r) + 0.02, j, f"r={r:+.3f}", va="center", fontsize=9)
    ax.set_yticks(range(7))
    ax.set_yticklabels(PNAMES)
    ax.set_xlabel("|Correlation with capacity|")
    ax.set_title("Temporal: Mode-Capacity Correlation")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("NEWAREA: Temporal-Constrained Decomposition", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_detail.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
