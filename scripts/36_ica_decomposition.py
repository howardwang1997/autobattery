"""ICA (Incremental Capacity Analysis) dQ/dV feature decomposition.

Key idea: Decompose degradation in dQ/dV space instead of raw V(t) space.
Physics suggests dQ/dV signatures should be more distinguishable:
- LAM → peak position shift (lost capacity changes thermodynamic window)
- R growth → peak height reduction (overpotential doesn't change thermodynamics)
- SEI → peak area reduction (active lithium loss)

Method:
1. Convert voltage curves V(t) to dQ/dV curves
2. Compute dQ/dV degradation signatures from simulation
3. Check if dQ/dV signatures have lower correlation than V(t) signatures
4. If better → use dQ/dV decomposition
5. If same → use as comparison baseline for paper
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

OUT = Path("outputs/degradation/ica")
OUT.mkdir(parents=True, exist_ok=True)

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_TIME = 100


def voltage_to_dqdv(V, Q, sigma=2):
    """Convert V-Q curve to dQ/dV.
    
    Returns dQ/dV interpolated onto a uniform voltage grid.
    """
    valid = np.isfinite(V) & np.isfinite(Q)
    valid[:-1] &= np.diff(V) != 0
    if valid.sum() < 10:
        return None, None
    
    V_sorted = V.copy()
    Q_sorted = Q.copy()
    
    sort_idx = np.argsort(V_sorted)
    V_sorted = V_sorted[sort_idx]
    Q_sorted = Q_sorted[sort_idx]
    
    mask_dup = np.concatenate([[True], np.diff(V_sorted) > 1e-6])
    V_sorted = V_sorted[mask_dup]
    Q_sorted = Q_sorted[mask_dup]
    
    if len(V_sorted) < 10:
        return None, None
    
    dQdV = np.gradient(Q_sorted, V_sorted)
    dQdV_smooth = gaussian_filter1d(dQdV, sigma=sigma)
    
    V_uniform = np.linspace(V_sorted[0] + 0.01, V_sorted[-1] - 0.01, N_TIME)
    dQdV_interp = np.interp(V_uniform, V_sorted, dQdV_smooth)
    
    return V_uniform, dQdV_interp


def compute_v_signatures():
    """Compute voltage-space degradation signatures (existing method)."""
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

    return sigs, P1, p_med, P_std


def compute_dqdv_signatures():
    """Compute dQ/dV-space degradation signatures."""
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    mask = np.abs(cr - 1.0) < 0.01
    V1, P1 = V[mask], P[mask]

    cap_nominal = 2.5
    Q1 = np.cumsum(np.ones_like(V1) * cap_nominal / N_TIME, axis=1)

    dQdV_all = []
    valid_idx = []
    for i in range(V1.shape[0]):
        V_uniform, dqdv = voltage_to_dqdv(V1[i], Q1[i], sigma=2)
        if dqdv is not None and np.all(np.isfinite(dqdv)):
            dQdV_all.append(dqdv)
            valid_idx.append(i)

    dQdV_all = np.array(dQdV_all)
    valid_idx = np.array(valid_idx)
    P_valid = P1[valid_idx]

    logger.info(f"dQ/dV computed for {len(valid_idx)}/{V1.shape[0]} sims")

    P_reg = P_valid.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P_valid[:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_std = P_reg.std(axis=0) + 1e-12
    P_norm = (P_reg - p_med) / P_std

    sigs = np.zeros((7, N_TIME))
    for t in range(N_TIME):
        m = Ridge(alpha=0.1)
        m.fit(P_norm, dQdV_all[:, t])
        sigs[:, t] = m.coef_

    return sigs, dQdV_all, P_valid, p_med, P_std


def analyze_signature_correlations(sigs, label):
    """Analyze signature correlation matrix."""
    n = sigs.shape[0]
    corr = np.corrcoef(sigs)
    
    high_corr_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            high_corr_pairs.append((PNAMES[i], PNAMES[j], abs(corr[i, j])))
    high_corr_pairs.sort(key=lambda x: -x[2])
    
    logger.info(f"\n{label} signature correlations (sorted by |r|):")
    for n1, n2, r in high_corr_pairs:
        flag = " *** HIGH" if r > 0.7 else ""
        logger.info(f"  {n1:12s} <-> {n2:12s}: r = {r:.3f}{flag}")
    
    abs_corr = np.abs(corr[np.triu_indices(n, k=1)])
    logger.info(f"  Mean |r| = {abs_corr.mean():.3f}, Max |r| = {abs_corr.max():.3f}")
    
    U, S, Vt = np.linalg.svd(sigs, full_matrices=False)
    total_var = (S**2).sum()
    cumvar = np.cumsum(S**2) / total_var
    n_95 = np.searchsorted(cumvar, 0.95) + 1
    cond = S[0] / S[-1]
    logger.info(f"  SVD: rank_95={n_95}, condition={cond:.0f}")
    
    return corr, S, cumvar


def decompose_dqdv_nnls(dQdV_target, signatures, reg=0.01):
    """NNLS decomposition in dQ/dV space."""
    n = signatures.shape[0]
    A = signatures.T
    A_aug = np.vstack([A, np.eye(n) * reg])
    b_aug = np.concatenate([dQdV_target, np.zeros(n)])
    coeffs, _ = nnls(A_aug, b_aug)
    recon = A @ coeffs
    rmse = np.sqrt(np.mean((recon - dQdV_target) ** 2))
    return coeffs, rmse


def synthetic_validation(v_sigs, dqdv_sigs):
    """Test decomposition accuracy on synthetic degradation trajectories."""
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        V = f["V"][:]
        P = f["params"][:]
        cr = f["c_rates"][:]
    mask = np.abs(cr - 1.0) < 0.01
    V1, P1 = V[mask], P[mask]

    cap_nominal = 2.5
    Q1 = np.cumsum(np.ones_like(V1) * cap_nominal / N_TIME, axis=1)

    P_reg = P1.copy()
    for j in LOG_PARAMS:
        P_reg[:, j] = np.log10(P1[:, j] + 1e-30)
    p_med = np.median(P_reg, axis=0)
    P_std = P_reg.std(axis=0) + 1e-12
    P_norm = (P_reg - p_med) / P_std

    p_median = np.median(P1, axis=0)
    idx_base = np.argmin(np.abs(P_norm).sum(axis=1))
    V_base = V1[idx_base]
    Q_base = Q1[idx_base]
    _, dqdv_base = voltage_to_dqdv(V_base, Q_base, sigma=2)

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

    results = {}
    n_synth_cycles = 60

    for traj_name, param_fn in trajectories.items():
        dV_list = []
        dqdv_list = []
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

            dV = V1[idx_t] - V_base
            dV_list.append(dV)

            _, dqdv_t = voltage_to_dqdv(V1[idx_t], Q1[idx_t], sigma=2)
            if dqdv_t is not None:
                dqdv_list.append(dqdv_t - dqdv_base)
            else:
                dqdv_list.append(np.zeros(N_TIME))

            for j, v in changes.items():
                gt_coeffs[j, cyc] = v

        dV_arr = np.array(dV_list)
        dqdv_arr = np.array(dqdv_list)

        changes_final = param_fn(1.0)
        gt_idx = list(changes_final.keys())

        def eval_pct(coeffs_mat, cycle_idx=-1):
            c = coeffs_mat[:, cycle_idx]
            return sum(c[j] for j in gt_idx) / (c.sum() + 1e-10) * 100

        c_v = np.zeros((7, n_synth_cycles))
        c_dqdv = np.zeros((7, n_synth_cycles))
        for cyc in range(n_synth_cycles):
            from scipy.optimize import nnls as _nnls
            A_v = v_sigs.T
            A_aug = np.vstack([A_v, np.eye(7) * 0.01])
            b_aug = np.concatenate([dV_arr[cyc], np.zeros(7)])
            cv, _ = _nnls(A_aug, b_aug)
            c_v[:, cyc] = cv

            A_dq = dqdv_sigs.T
            A_aug2 = np.vstack([A_dq, np.eye(7) * 0.01])
            b_aug2 = np.concatenate([dqdv_arr[cyc], np.zeros(7)])
            cdq, _ = _nnls(A_aug2, b_aug2)
            c_dqdv[:, cyc] = cdq

        pct_v = eval_pct(c_v)
        pct_dqdv = eval_pct(c_dqdv)

        results[traj_name] = {
            "v_pct": pct_v,
            "dqdv_pct": pct_dqdv,
            "gt_idx": gt_idx,
            "c_v": c_v,
            "c_dqdv": c_dqdv,
        }

        logger.info(
            f"  {traj_name:25s}: V={pct_v:5.1f}% | dQ/dV={pct_dqdv:5.1f}%"
        )

    v_pass = sum(1 for r in results.values() if r["v_pct"] > 75)
    dqdv_pass = sum(1 for r in results.values() if r["dqdv_pct"] > 75)
    logger.info(f"\n  V(t) NNLS:    {v_pass}/{len(results)} PASS")
    logger.info(f"  dQ/dV NNLS:   {dqdv_pass}/{len(results)} PASS")

    return results


def plot_signatures(v_sigs, dqdv_sigs, corr_v, corr_dqdv):
    """Compare V and dQ/dV signatures."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    t_ax = np.linspace(0, 1, N_TIME)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for j in range(7):
        ax = axes[j // 4, j % 4]
        ax.plot(t_ax * 100, v_sigs[j] / (np.abs(v_sigs[j]).max() + 1e-10),
                color=colors[j], linewidth=2, label="V(t)")
        ax.plot(t_ax * 100, dqdv_sigs[j] / (np.abs(dqdv_sigs[j]).max() + 1e-10),
                color=colors[j], linewidth=2, linestyle="--", label="dQ/dV")
        ax.set_title(PNAMES[j])
        ax.set_xlabel("Normalized time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[1, 3].set_visible(False)
    fig.suptitle("Degradation Signatures: V(t) (solid) vs dQ/dV (dashed)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "signature_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, corr, label in [(axes[0], corr_v, "V(t)"), (axes[1], corr_dqdv, "dQ/dV")]:
        im = ax.imshow(np.abs(corr), cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(PNAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(PNAMES, fontsize=8)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{label} Signature Correlations")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6)
    fig.suptitle("Signature Correlation: V(t) vs dQ/dV", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "correlation_comparison.png", dpi=150)
    plt.close(fig)


def plot_validation_comparison(results):
    """Bar chart comparing V vs dQ/dV decomposition accuracy."""
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(results.keys())
    x = np.arange(len(names))
    v_pcts = [results[k]["v_pct"] for k in names]
    dqdv_pcts = [results[k]["dqdv_pct"] for k in names]

    ax.bar(x - 0.2, v_pcts, 0.35, label="V(t) NNLS", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, dqdv_pcts, 0.35, label="dQ/dV NNLS", color="coral", alpha=0.8)
    ax.axhline(75, color="k", linestyle="--", alpha=0.3, label="75% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:18] for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% Correct Attribution")
    ax.set_title("Decomposition Accuracy: V(t) vs dQ/dV")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "validation_comparison.png", dpi=150)
    plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("ICA dQ/dV DECOMPOSITION ANALYSIS")
    logger.info("=" * 60)

    logger.info("\nComputing V(t) signatures...")
    v_sigs, P1, p_med, P_std = compute_v_signatures()
    corr_v, S_v, cumvar_v = analyze_signature_correlations(v_sigs, "V(t)")

    logger.info("\nComputing dQ/dV signatures...")
    dqdv_sigs, dQdV_all, P_valid, p_med_dq, P_std_dq = compute_dqdv_signatures()
    corr_dqdv, S_dqdv, cumvar_dqdv = analyze_signature_correlations(dqdv_sigs, "dQ/dV")

    plot_signatures(v_sigs, dqdv_sigs, corr_v, corr_dqdv)

    logger.info("\n" + "=" * 60)
    logger.info("SYNTHETIC VALIDATION")
    logger.info("=" * 60)
    results = synthetic_validation(v_sigs, dqdv_sigs)
    plot_validation_comparison(results)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    v_pass = sum(1 for r in results.values() if r["v_pct"] > 75)
    dqdv_pass = sum(1 for r in results.values() if r["dqdv_pct"] > 75)

    logger.info(f"  V(t) method:   {v_pass}/{len(results)} scenarios pass (>75%)")
    logger.info(f"  dQ/dV method:  {dqdv_pass}/{len(results)} scenarios pass (>75%)")

    if dqdv_pass > v_pass:
        logger.info("  → dQ/dV decomposition is BETTER!")
    elif dqdv_pass < v_pass:
        logger.info("  → dQ/dV decomposition is WORSE (but useful as comparison)")
    else:
        logger.info("  → dQ/dV is comparable to V(t) (useful as comparison baseline)")

    logger.info(f"\nOutputs saved to {OUT}/")


if __name__ == "__main__":
    main()
