#!/usr/bin/env python3
"""
Synthetic Parameter Recovery Experiment v2.

Direct test: if two voltage curves are similar, are their parameters similar?
If V(t) is rank-r, then only r linear combinations of θ are determined by V.

Design:
1. For each test sim, find k-nearest-neighbors in V(t) space
2. Average the θ of neighbors → predicted θ
3. Compare predicted vs true θ per dimension
4. Also: PCA-project V to r dims, reconstruct, and check which θ components survive
5. Compare at noise levels σ = 0, 1, 5, 10 mV
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("outputs/parameter_recovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
FISHER_ID = {"SEI", "D_n", "LAM_neg"}
FISHER_UN = {"D_p", "t+", "LAM_pos", "R_mult"}

SEED = 42
np.random.seed(SEED)


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
    logger.info(f"Loaded {V.shape[0]} sims, V{V.shape}, θ{params.shape}")
    return params, V


def knn_recovery(params_train, V_train, params_test, V_test, k=5):
    from scipy.spatial import cKDTree
    tree = cKDTree(V_train)
    _, idx = tree.query(V_test, k=k)
    if k == 1:
        idx = idx[:, None]
    theta_pred = np.array([params_train[idx[i]].mean(axis=0) for i in range(len(V_test))])
    return theta_pred


def pca_subspace_recovery(params_train, V_train, params_test, V_test, n_components):
    """Project V to PCA(n), reconstruct, predict θ via Ridge on PCA scores."""
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    Z_train = pca.fit_transform(V_train)
    Z_test = pca.transform(V_test)
    
    model = Ridge(alpha=1.0)
    model.fit(Z_train, params_train)
    theta_pred = model.predict(Z_test)
    
    return theta_pred, pca.explained_variance_ratio_.sum()


def run_experiment():
    params, V = load_data()
    n_sims = len(params)
    n_params = params.shape[1]
    
    n_test = 200
    idx = np.random.permutation(n_sims)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]
    
    params_train = params[idx_train]
    V_train_clean = V[idx_train]
    params_test = params[idx_test]
    V_test_clean = V[idx_test]
    
    param_range = params_train.max(axis=0) - params_train.min(axis=0)
    param_range[param_range < 1e-10] = 1.0
    
    results = {}
    
    for noise_mV in [0, 1, 5, 10]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Noise σ = {noise_mV} mV")
        logger.info(f"{'='*60}")
        
        noise_std = noise_mV / 1000.0
        np.random.seed(SEED + noise_mV)
        V_train = V_train_clean + np.random.randn(*V_train_clean.shape) * noise_std
        V_test = V_test_clean + np.random.randn(*V_test_clean.shape) * noise_std
        
        noise_results = {"noise_mV": noise_mV, "experiments": {}}
        
        # --- Experiment 1: KNN recovery ---
        for k in [1, 5, 20]:
            theta_pred = knn_recovery(params_train, V_train, params_test, V_test, k=k)
            rel_err = np.abs(theta_pred - params_test) / (np.abs(params_test) + 1e-10)
            norm_err = np.abs(theta_pred - params_test) / param_range
            
            per_param = {}
            id_errs = []
            un_errs = []
            logger.info(f"\n  KNN (k={k}) normalized error (|Δθ|/range):")
            for j, pn in enumerate(PARAM_NAMES):
                med = float(np.median(norm_err[:, j]))
                mn = float(np.mean(norm_err[:, j]))
                r2 = float(r2_score(params_test[:, j], theta_pred[:, j]))
                per_param[pn] = {"median_norm_err": med, "mean_norm_err": mn, "r2": r2}
                group = "ID" if pn in FISHER_ID else "UN"
                logger.info(f"    {pn:10s}: med={med:.4f}  mean={mn:.4f}  R²={r2:+.4f}  [{group}]")
                if pn in FISHER_ID:
                    id_errs.append(mn)
                else:
                    un_errs.append(mn)
            
            logger.info(f"    ID mean_err = {np.mean(id_errs):.4f}, UN mean_err = {np.mean(un_errs):.4f}, "
                         f"ratio = {np.mean(un_errs)/(np.mean(id_errs)+1e-10):.2f}x")
            
            noise_results["experiments"][f"knn_k{k}"] = {
                "per_param": per_param,
                "id_mean_err": float(np.mean(id_errs)),
                "un_mean_err": float(np.mean(un_errs)),
                "id_un_ratio": float(np.mean(un_errs) / (np.mean(id_errs) + 1e-10)),
            }
        
        # --- Experiment 2: PCA subspace sweep ---
        logger.info(f"\n  PCA subspace recovery (V→θ via Ridge on PCA scores):")
        pca_results = {}
        for nc in [1, 2, 3, 4, 5, 7, 10, 20, 50]:
            theta_pred, cumvar = pca_subspace_recovery(
                params_train, V_train, params_test, V_test, n_components=min(nc, V_train.shape[1]))
            
            norm_err = np.abs(theta_pred - params_test) / param_range
            mean_errs = [float(np.mean(norm_err[:, j])) for j in range(n_params)]
            id_err = np.mean([mean_errs[j] for j, pn in enumerate(PARAM_NAMES) if pn in FISHER_ID])
            un_err = np.mean([mean_errs[j] for j, pn in enumerate(PARAM_NAMES) if pn in FISHER_UN])
            
            r2_all = float(r2_score(params_test, theta_pred))
            
            logger.info(f"    PCA({nc:2d}): cumvar={cumvar:.3f}, "
                         f"ID_err={id_err:.4f}, UN_err={un_err:.4f}, "
                         f"UN/ID={un_err/(id_err+1e-10):.2f}x, overall_R²={r2_all:+.4f}")
            
            pca_results[f"pca_{nc}"] = {
                "n_components": nc,
                "cumvar": float(cumvar),
                "id_mean_err": float(id_err),
                "un_mean_err": float(un_err),
                "un_id_ratio": float(un_err / (id_err + 1e-10)),
                "overall_r2": float(r2_all),
                "per_param_err": {pn: float(mean_errs[j]) for j, pn in enumerate(PARAM_NAMES)},
            }
        
        noise_results["experiments"]["pca_sweep"] = pca_results
        
        # --- Experiment 3: Per-parameter identifiability score ---
        # Compute mutual information between V-PCA scores and each θ_j
        logger.info(f"\n  Per-parameter identifiability score (R² from V→θ_j Ridge):")
        from sklearn.linear_model import Ridge
        scores = {}
        for j, pn in enumerate(PARAM_NAMES):
            model = Ridge(alpha=1.0)
            model.fit(V_train, params_train[:, j])
            r2 = r2_score(params_test[:, j], model.predict(V_test))
            scores[pn] = float(r2)
            group = "ID" if pn in FISHER_ID else "UN"
            logger.info(f"    {pn:10s}: R² = {r2:+.4f}  [{group}]")
        
        id_r2 = [scores[p] for p in FISHER_ID]
        un_r2 = [scores[p] for p in FISHER_UN]
        logger.info(f"    ID mean R² = {np.mean(id_r2):.4f}, UN mean R² = {np.mean(un_r2):.4f}")
        
        noise_results["experiments"]["ridge_per_param"] = {
            "per_param_r2": scores,
            "id_mean_r2": float(np.mean(id_r2)),
            "un_mean_r2": float(np.mean(un_r2)),
            "separation_ratio": float(np.mean(id_r2) / (np.mean(np.abs(un_r2)) + 1e-10)),
        }
        
        # --- Experiment 4: Noise amplification ---
        # Perturb θ slightly → measure change in V → amplification factor
        logger.info(f"\n  Sensitivity (noise amplification from θ to V):")
        eps = 0.01
        sensitivity = {}
        for j, pn in enumerate(PARAM_NAMES):
            theta_p = params_test.copy()
            theta_p[:, j] *= (1 + eps)
            dV = np.zeros_like(V_test_clean)
            for i in range(len(theta_p)):
                diffs = theta_p[i] - params_train
                dists = np.sqrt(np.sum((diffs / param_range) ** 2, axis=1))
                w = np.exp(-dists * 10)
                w /= w.sum() + 1e-20
                V_pert = (w[:, None] * V_train_clean).sum(axis=0)
                
                diffs0 = params_test[i] - params_train
                dists0 = np.sqrt(np.sum((diffs0 / param_range) ** 2, axis=1))
                w0 = np.exp(-dists0 * 10)
                w0 /= w0.sum() + 1e-20
                V_base = (w0[:, None] * V_train_clean).sum(axis=0)
                
                dV[i] = V_pert - V_base
            
            dV_rms = float(np.sqrt(np.mean(dV ** 2)))
            sensitivity[pn] = dV_rms
            logger.info(f"    {pn:10s}: δV_RMS = {dV_rms*1000:.4f} mV (for 1% θ perturbation)")
        
        noise_results["experiments"]["sensitivity"] = sensitivity
        
        results[f"noise_{noise_mV}mV"] = noise_results
    
    return results


def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    noise_levels = sorted([int(k.replace("noise_", "").replace("mV", "")) for k in results.keys()])
    
    # Panel 1: KNN recovery per param (σ=5mV)
    ax = axes[0, 0]
    key = "noise_5mV"
    if key in results:
        knn = results[key]["experiments"]["knn_k5"]["per_param"]
        names = list(knn.keys())
        errs = [knn[p]["mean_norm_err"] for p in names]
        colors = ["tab:blue" if p in FISHER_ID else "tab:red" for p in names]
        ax.barh(names, errs, color=colors)
        ax.set_xlabel("Mean normalized error |Δθ|/range")
        ax.set_title(f"KNN recovery (k=5, σ=5mV)\nblue=Fisher-ID, red=Fisher-UN")
    
    # Panel 2: PCA subspace sweep
    ax = axes[0, 1]
    for key_name, label, ls in [("noise_0mV", "σ=0", "-"), ("noise_5mV", "σ=5mV", "--"), ("noise_10mV", "σ=10mV", ":")]:
        if key_name in results:
            pca = results[key_name]["experiments"]["pca_sweep"]
            ncs = [pca[k]["n_components"] for k in sorted(pca.keys())]
            overall_r2 = [pca[k]["overall_r2"] for k in sorted(pca.keys())]
            ax.plot(ncs, overall_r2, ls, label=label, lw=2)
    ax.axvline(3, color="red", ls="--", alpha=0.5, label="Fisher rank=3")
    ax.set_xlabel("PCA components used for V→θ")
    ax.set_ylabel("Overall R²")
    ax.set_title("PCA subspace recovery")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Per-param R² from Ridge (noise sweep)
    ax = axes[1, 0]
    for pn in PARAM_NAMES:
        r2s = []
        for nl in noise_levels:
            key = f"noise_{nl}mV"
            r2 = results[key]["experiments"]["ridge_per_param"]["per_param_r2"][pn]
            r2s.append(r2)
        color = "tab:blue" if pn in FISHER_ID else "tab:red"
        ax.plot(noise_levels, r2s, "o-", label=pn, color=color, alpha=0.8)
    ax.set_xlabel("Noise σ (mV)")
    ax.set_ylabel("R² (V→θ_j Ridge)")
    ax.set_title("Per-parameter recoverability")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="grey", ls=":", lw=0.5)
    
    # Panel 4: Sensitivity (V response to θ perturbation)
    ax = axes[1, 1]
    if "noise_0mV" in results:
        sens = results["noise_0mV"]["experiments"]["sensitivity"]
        names = list(sens.keys())
        vals = [sens[p] * 1000 for p in names]
        colors = ["tab:blue" if p in FISHER_ID else "tab:red" for p in names]
        ax.barh(names, vals, color=colors)
        ax.set_xlabel("δV_RMS (mV) for 1% θ perturbation")
        ax.set_title("Parameter sensitivity\n(lower = harder to identify)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_recovery_v2.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved")


def main():
    logger.info("Parameter Recovery Experiment v2")
    logger.info("=" * 60)
    
    results = run_experiment()
    
    with open(OUTPUT_DIR / "parameter_recovery_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    plot_results(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("KEY SUMMARY")
    logger.info("=" * 60)
    for key in sorted(results.keys()):
        r = results[key]
        logger.info(f"\n--- {key} ---")
        ridge = r["experiments"]["ridge_per_param"]
        logger.info(f"  ID mean R² = {ridge['id_mean_r2']:.4f}")
        logger.info(f"  UN mean R² = {ridge['un_mean_r2']:.4f}")
        logger.info(f"  Separation = {ridge['separation_ratio']:.2f}x")
        
        knn = r["experiments"]["knn_k5"]
        logger.info(f"  KNN ID err = {knn['id_mean_err']:.4f}, UN err = {knn['un_mean_err']:.4f}, "
                     f"ratio = {knn['id_un_ratio']:.2f}x")
    
    # Compare with Fisher prediction
    logger.info("\n--- Fisher ID/UN agreement ---")
    r0 = results["noise_0mV"]["experiments"]["ridge_per_param"]["per_param_r2"]
    for pn in PARAM_NAMES:
        group = "ID" if pn in FISHER_ID else "UN"
        r2 = r0[pn]
        agree = (r2 > 0 and group == "ID") or (r2 <= 0 and group == "UN")
        logger.info(f"  {pn:10s}: R²={r2:+.4f} [{group}] {'✓' if agree else '✗ DISAGREE'}")


if __name__ == "__main__":
    main()
