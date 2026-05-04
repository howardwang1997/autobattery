#!/usr/bin/env python3
"""
B3: Mutual information estimation with PCA dimensionality reduction.
1. I(V(t); θ_i) for each parameter — which params does V carry info about?
2. I(dQ/dV; θ_i) — does dQ/dV transform increase or decrease information?
3. I(V_first_half; θ_i) — what does measurement length add?
4. DPI numerical verification: I(V; θ) >= I(f(V); θ) for f = dQ/dV.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
IDENT_IDX = [0, 3, 4]
N_PCA = 20


def compute_dqdV(V, n_time=100):
    dV = np.diff(V, axis=1)
    mid_Q = np.linspace(0.5 / n_time, 1 - 0.5 / n_time, n_time - 1)
    dQ = np.diff(mid_Q)[0]
    dQdV = dV / dQ
    return dQdV


def mi_pca(X, y, n_components=N_PCA, n_neighbors=10, n_repeat=3):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)
    explained = pca.explained_variance_ratio_.sum()
    estimates = []
    for seed in range(n_repeat):
        mi = mutual_info_regression(
            Xp, y, n_neighbors=n_neighbors, random_state=seed * 42
        )
        estimates.append(float(mi[0]) if hasattr(mi, "__len__") else float(mi))
    return float(np.mean(estimates)), explained


def main():
    output_dir = Path("outputs/mutual_information")
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = "data/fullfield/fullfield_lfp_degradation.h5"
    with h5py.File(h5_path, "r") as f:
        V = f["V"][:].astype(np.float32)
        params = f["params"][:].astype(np.float32)

    params_log = params.copy()
    for i in LOG_PARAMS:
        params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

    logger.info("Loaded %d samples, V shape %s", len(V), V.shape)

    # Part 1: I(V; θ_i) — raw voltage
    logger.info("Computing I(V(t); theta_i) with PCA(%d)...", N_PCA)
    mi_V = {}
    for pi, pname in enumerate(PARAM_NAMES):
        mi_val, explained = mi_pca(V, params_log[:, pi])
        mi_V[pname] = mi_val
        logger.info("  I(V; %s) = %.4f  (PCA explained %.1f%%)", pname, mi_val, explained * 100)

    # Part 2: I(dQ/dV; θ_i)
    logger.info("Computing I(dQ/dV; theta_i) ...")
    dQdV = compute_dqdV(V, n_time=V.shape[1])
    finite_mask = np.all(np.isfinite(dQdV), axis=1)
    dQdV_clean = dQdV[finite_mask]
    params_clean = params_log[finite_mask]
    logger.info("  dQ/dV: %d/%d finite samples", finite_mask.sum(), len(finite_mask))

    mi_dQdV = {}
    for pi, pname in enumerate(PARAM_NAMES):
        mi_val, explained = mi_pca(dQdV_clean, params_clean[:, pi])
        mi_dQdV[pname] = mi_val
        logger.info("  I(dQ/dV; %s) = %.4f", pname, mi_val)

    # Part 3: I(V_first_half; θ_i)
    mid = V.shape[1] // 2
    V_half = V[:, :mid]
    logger.info("Computing I(V_half; theta_i) ...")
    mi_V_half = {}
    for pi, pname in enumerate(PARAM_NAMES):
        mi_val, _ = mi_pca(V_half, params_log[:, pi])
        mi_V_half[pname] = mi_val

    # DPI verification: I(V; θ) >= I(dQ/dV; θ)?
    dpi_results = {}
    for pname in PARAM_NAMES:
        iv = mi_V[pname]
        idqv = mi_dQdV[pname]
        dpi_results[pname] = {
            "I_V": iv,
            "I_dQdV": idqv,
            "DPI_holds": iv >= idqv * 0.95,
            "information_loss_pct": (1 - idqv / max(iv, 1e-10)) * 100,
        }

    results = {
        "I_V": mi_V,
        "I_dQdV": mi_dQdV,
        "I_V_half": mi_V_half,
        "DPI_verification": dpi_results,
        "n_samples": int(len(V)),
        "n_pca": N_PCA,
    }

    print("\n" + "=" * 80)
    print("MUTUAL INFORMATION ANALYSIS (LFP fullfield, n={}, PCA={})".format(len(V), N_PCA))
    print("=" * 80)

    print("\n--- I(V(t); θ_i) — Information in raw voltage (PCA-estimated) ---")
    print("{:10s} {:>10s} {:>10s} {:>10s} {:>20s}".format(
        "Param", "I(V;θ)", "I(dQ/dV;θ)", "I(V/2;θ)", "DPI holds?"
    ))
    print("-" * 65)
    for pi, pname in enumerate(PARAM_NAMES):
        st = "ID" if pi in IDENT_IDX else "UN"
        dpi = dpi_results[pname]
        dpi_str = "YES" if dpi["DPI_holds"] else "NO"
        loss = dpi["information_loss_pct"]
        print("{:10s} {:10.4f} {:10.4f} {:10.4f} {:>12s} ({:.0f}% loss) [{}]".format(
            pname, mi_V[pname], mi_dQdV[pname], mi_V_half[pname],
            dpi_str, loss, st
        ))

    id_mi_V = np.mean([mi_V[PARAM_NAMES[i]] for i in IDENT_IDX])
    un_mi_V = np.mean([mi_V[PARAM_NAMES[i]] for i in range(7) if i not in IDENT_IDX])
    print(f"\nID avg I(V;θ) = {id_mi_V:.4f}, UN avg = {un_mi_V:.4f}, ratio = {id_mi_V / max(un_mi_V, 1e-10):.1f}x")

    with open(output_dir / "results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
