#!/usr/bin/env python3
"""
Correlation structure analysis.
Which parameters produce correlated V(t) changes? This explains WHY rank is low.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from sklearn.decomposition import PCA

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/parameter_recovery")
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")
PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
SEED = 42
np.random.seed(SEED)


def main():
    logger.info("Correlation Structure Analysis")
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)

    n_sims, n_time = V.shape
    n_params = params.shape[1]

    # 1. Jacobian via finite differences at median
    log_p = np.log10(np.abs(params) + 1e-20)
    log_med = np.median(log_p, axis=0)
    eps = 0.01
    J = np.zeros((n_time, n_params))
    for j in range(n_params):
        log_p_plus = log_med.copy()
        log_p_plus[j] += eps
        log_p_minus = log_med.copy()
        log_p_minus[j] -= eps

        p_plus = 10.0 ** log_p_plus
        p_minus = 10.0 ** log_p_minus

        # Find nearest neighbors
        dists_plus = np.sqrt(np.sum(((log_p - log_p_plus) / (log_p.std(axis=0) + 1e-10)) ** 2, axis=1))
        dists_minus = np.sqrt(np.sum(((log_p - log_p_minus) / (log_p.std(axis=0) + 1e-10)) ** 2, axis=1))
        nn_plus = V[np.argmin(dists_plus)]
        nn_minus = V[np.argmin(dists_minus)]
        J[:, j] = (nn_plus - nn_minus) / (2 * eps)

    # Correlation matrix of Jacobian columns
    J_norm = J / (np.linalg.norm(J, axis=0, keepdims=True) + 1e-20)
    corr = J_norm.T @ J_norm

    logger.info("Jacobian column correlation matrix:")
    corr_dict = {}
    for i in range(n_params):
        corr_dict[PARAM_NAMES[i]] = {PARAM_NAMES[j]: float(corr[i, j]) for j in range(n_params)}
        row = " ".join([f"{corr[i,j]:+.3f}" for j in range(n_params)])
        logger.info(f"  {PARAM_NAMES[i]:10s}: {row}")

    # Find high-correlation pairs
    high_corr = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            if abs(corr[i, j]) > 0.9:
                high_corr.append({"p1": PARAM_NAMES[i], "p2": PARAM_NAMES[j],
                                   "corr": float(corr[i, j])})
                logger.info(f"  HIGH CORR: {PARAM_NAMES[i]} - {PARAM_NAMES[j]} = {corr[i,j]:+.4f}")

    # 2. Sensitivity ranking (Jacobian norm per param)
    sens = np.linalg.norm(J, axis=0)
    sens_ranking = sorted(zip(PARAM_NAMES, sens), key=lambda x: -x[1])
    logger.info("\nSensitivity ranking (||J_j||):")
    for pn, s in sens_ranking:
        logger.info(f"  {pn:10s}: {s:.6f}")

    # 3. FIM eigenvalue decomposition
    FIM = J.T @ J
    eigenvalues, eigenvectors = np.linalg.eigh(FIM)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    logger.info("\nFIM eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        logger.info(f"  λ_{i+1} = {ev:.4e} ({ev/eigenvalues[0]*100:.2f}%)")

    # 4. Eigenvalue composition (which params contribute to each eigenvector)
    logger.info("\nEigenvector composition (|v_ij| for top eigenvectors):")
    eigen_comp = {}
    for i in range(min(4, n_params)):
        v = eigenvectors[:, i]
        comp = {PARAM_NAMES[j]: float(abs(v[j])) for j in range(n_params)}
        eigen_comp[f"eigvec_{i+1}"] = comp
        sorted_comp = sorted(comp.items(), key=lambda x: -x[1])
        top3 = ", ".join([f"{n}={v:.3f}" for n, v in sorted_comp[:3]])
        logger.info(f"  Eigvec {i+1} (λ={eigenvalues[i]:.2e}): {top3}")

    # 5. Null space analysis
    effective_rank = np.sum(eigenvalues) / eigenvalues[0]
    threshold = eigenvalues[0] * 1e-3
    numerical_rank = np.sum(eigenvalues > threshold)
    logger.info(f"\nEffective rank = {effective_rank:.2f}")
    logger.info(f"Numerical rank (η=1e-3) = {numerical_rank}")
    logger.info(f"Null space dim = {n_params - numerical_rank}")

    results = {
        "jacobian_correlation": corr_dict,
        "high_correlation_pairs": high_corr,
        "sensitivity_ranking": {pn: float(s) for pn, s in sens_ranking},
        "eigenvalues": [float(e) for e in eigenvalues],
        "eigenvector_composition": eigen_comp,
        "effective_rank": float(effective_rank),
        "numerical_rank_eta1e3": int(numerical_rank),
        "null_space_dim": int(n_params - numerical_rank),
    }

    with open(OUTPUT_DIR / "correlation_structure.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_params)); ax.set_xticklabels(PARAM_NAMES, rotation=45, fontsize=8)
    ax.set_yticks(range(n_params)); ax.set_yticklabels(PARAM_NAMES, fontsize=8)
    for i in range(n_params):
        for j in range(n_params):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Jacobian column correlation")

    # Eigenvector composition
    ax = axes[1]
    n_show = min(4, n_params)
    x = np.arange(n_params)
    w = 0.2
    for i in range(n_show):
        v = np.abs(eigenvectors[:, i])
        ax.bar(x + i * w, v / v.sum(), w, label=f"λ_{i+1}={eigenvalues[i]:.1e}")
    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(PARAM_NAMES, fontsize=8)
    ax.set_ylabel("|v_ij| (normalized)")
    ax.set_title("Eigenvector composition")
    ax.legend(fontsize=7)

    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "correlation_structure.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
