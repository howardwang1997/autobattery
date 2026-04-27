"""Degradation mode separation via differential voltage signature matching.

Key insight: match ΔV(t) (change from first cycle), not absolute V(t).
This avoids the sim-to-real gap because we match degradation trends,
not absolute voltage values.

Method:
1. From simulation: compute degradation signatures dV/d(degradation_param)
2. From experiment: compute ΔV(t) = V_cycle(t) - V_ref(t)
3. Fit ΔV as linear combination of degradation signatures → identify modes
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.loader import ExperimentalDataLoader
from scipy.ndimage import gaussian_filter1d

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation")
OUT.mkdir(parents=True, exist_ok=True)


def extract_curves(data, min_pts=15):
    current, voltage, time_arr, cycles = data["current"], data["voltage"], data["time"], data["cycle"]
    unique = np.unique(cycles)
    curves = []
    for cyc in unique:
        mask = (cycles == cyc) & (current < -0.005)
        idx = np.where(mask)[0]
        if len(idx) < min_pts: continue
        v_s, i_s, t_s = voltage[idx], current[idx], time_arr[idx]
        valid = ~np.isnan(v_s)
        v_s, i_s, t_s = v_s[valid], i_s[valid], t_s[valid]
        if len(v_s) < min_pts: continue
        dt = np.diff(t_s)
        if dt.sum() < 10: continue
        cap = -np.sum(i_s[:-1] * dt) / 3600
        if cap < 0.001: continue
        curves.append({"cycle": int(cyc), "voltage": v_s, "capacity": cap, "i_mean": abs(i_s.mean())})
    return curves


def compute_sim_signatures():
    """Compute degradation signatures from LFP simulation data."""
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        params = f["params"][:]
        cr = f["c_rates"][:]

    n_time = V.shape[1]
    mask_1c = np.abs(cr - 1.0) < 0.01
    V1 = V[mask_1c]
    P1 = params[mask_1c]

    # Baseline: median parameter values
    p_median = np.median(P1, axis=0)

    # Compute dV/d(param_j) using linear regression at each time point
    # ΔV(t) ≈ Σ_j (∂V/∂p_j) * Δp_j
    pnames = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]

    # Use log-space for parameters that span orders of magnitude
    log_params = [0, 1, 3]  # D_n, D_p, SEI_thick use log scale

    P_reg = P1.copy()
    for j in log_params:
        P_reg[:, j] = np.log10(P1[:, j] + 1e-30)
    p_median_reg = np.median(P_reg, axis=0)

    # Normalize
    P_norm = (P_reg - p_median_reg)
    P_std = P_norm.std(axis=0) + 1e-12
    P_norm /= P_std

    # Linear regression at each time point: V(t) = V0(t) + Σ_j w_j(t) * P_norm_j
    # This gives the degradation signature w_j(t)
    from sklearn.linear_model import Ridge
    signatures = np.zeros((len(pnames), n_time))
    for t in range(n_time):
        model = Ridge(alpha=0.1)
        model.fit(P_norm, V1[:, t])
        signatures[:, t] = model.coef_

    return signatures, pnames, V1, P1, p_median, p_median_reg, P_std


def fit_degradation_modes(curves, signatures, pnames, n_time=100):
    """Fit experimental ΔV(t) as combination of simulation signatures."""
    # Reference: first 5 cycles average
    ref_curves = curves[:5]
    v_refs = []
    for cc in ref_curves:
        v_r = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        v_refs.append(v_r)
    V_ref = np.mean(v_refs, axis=0)

    n_sigs = len(pnames)
    results = []

    for cc in curves:
        v_exp = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(cc["voltage"])), cc["voltage"])
        dV = v_exp - V_ref
        dV_smooth = gaussian_filter1d(dV, sigma=3)

        # Fit: dV(t) = Σ_j c_j * signature_j(t)
        # Non-negative least squares (degradation only increases)
        from scipy.optimize import nnls
        A = signatures.T  # (n_time, n_sigs)
        b = dV_smooth

        # Allow both positive and negative contributions
        A_aug = np.vstack([A, np.eye(n_sigs) * 0.01])
        b_aug = np.concatenate([b, np.zeros(n_sigs)])
        coeffs_full, residual = nnls(A_aug, b_aug)
        c_pos = coeffs_full[:n_sigs]
        c_net = c_pos.copy()

        # Reconstruction quality
        dV_recon = A @ c_net
        rmse = np.sqrt(np.mean((dV_recon - dV_smooth) ** 2)) * 1000

        results.append({
            "cycle": cc["cycle"],
            "capacity": cc["capacity"],
            "dV_rmse_mV": rmse,
            "coeffs": c_net.copy(),
        })

    return results


def main():
    logger.info("Computing simulation degradation signatures...")
    signatures, pnames, V1, P1, p_median, p_median_reg, P_std = compute_sim_signatures()

    # Plot signatures
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    t_ax = np.linspace(0, 1, signatures.shape[1])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for j, (ax, name) in enumerate(zip(axes.flat, pnames)):
        ax.plot(t_ax * 100, signatures[j], color=colors[j], linewidth=2)
        ax.set_xlabel("Normalized discharge time (%)")
        ax.set_ylabel("dV/d(param)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Simulation Degradation Signatures (∂V/∂param)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "degradation_signatures.png", dpi=150)
    plt.close(fig)

    # Load experimental data
    logger.info("Loading experimental data...")
    loader = ExperimentalDataLoader()
    data = loader.load_neware_xlsx("/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")
    curves = extract_curves(data)
    logger.info(f"Extracted {len(curves)} curves")

    # Fit degradation modes
    logger.info("Fitting degradation modes...")
    results = fit_degradation_modes(curves, signatures, pnames)

    # Plot results
    cycles = np.array([r["cycle"] for r in results])
    caps = np.array([r["capacity"] for r in results])
    coeffs = np.array([r["coeffs"] for r in results])
    rmse = np.array([r["dV_rmse_mV"] for r in results])

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # Capacity
    ax = axes[0, 0]
    ax.plot(cycles, caps * 1000, "b-", linewidth=1.5)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Capacity (mAh)")
    ax.set_title("Capacity Fade"); ax.grid(True, alpha=0.3)

    # Each degradation mode coefficient over cycles
    for j in range(min(7, len(pnames))):
        row, col = (j + 1) // 4, (j + 1) % 4
        ax = axes[row, col]
        c_smooth = gaussian_filter1d(coeffs[:, j], sigma=5)
        ax.plot(cycles, c_smooth, color=colors[j], linewidth=1.5)
        ax.set_xlabel("Cycle"); ax.set_ylabel(f"{pnames[j]} coefficient")
        # Correlation with capacity
        r = np.corrcoef(c_smooth[len(c_smooth)//10:], caps[len(caps)//10:])[0, 1] if len(c_smooth) > 10 else 0
        ax.set_title(f"{pnames[j]} (r={r:.3f})")
        ax.grid(True, alpha=0.3)

        # Twin axis: capacity
        ax2 = ax.twinx()
        ax2.plot(cycles, caps / caps[0] * 100, "b--", alpha=0.2, linewidth=1)
        ax2.set_ylabel("Cap (%)", color="b", alpha=0.3)

    fig.suptitle("Degradation Mode Decomposition — NEWAREA (434 cycles)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "degradation_mode_decomposition.png", dpi=150)
    plt.close(fig)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("DEGRADATION MODE DECOMPOSITION")
    logger.info(f"{'='*60}")
    logger.info(f"Mean fit RMSE: {rmse.mean():.1f} mV")
    logger.info(f"\nCorrelation of each mode with capacity fade:")
    for j in range(len(pnames)):
        c_smooth = gaussian_filter1d(coeffs[:, j], sigma=5)
        r = np.corrcoef(c_smooth, caps)[0, 1]
        # Trend: compare first vs last 20%
        n = len(c_smooth)
        early = c_smooth[:max(1, n//5)].mean()
        late = c_smooth[-max(1, n//5):].mean()
        trend = "↑" if late > early else "↓"
        logger.info(f"  {pnames[j]:12s}: r={r:+.3f}, trend={trend} ({early:.3f} → {late:.3f})")

    # Identify dominant mode
    abs_coeffs = np.abs(coeffs)
    total_activity = abs_coeffs.sum(axis=0)
    total_activity /= total_activity.sum() + 1e-10
    dominant = np.argmax(total_activity)
    logger.info(f"\nDominant degradation mode: {pnames[dominant]} ({total_activity[dominant]*100:.1f}%)")
    logger.info(f"Mode contributions:")
    for j in np.argsort(total_activity)[::-1]:
        logger.info(f"  {pnames[j]:12s}: {total_activity[j]*100:.1f}%")

    logger.info(f"\nOutputs: {OUT}/degradation_mode_decomposition.png")
    logger.info("Degradation mode decomposition complete!")


if __name__ == "__main__":
    main()
