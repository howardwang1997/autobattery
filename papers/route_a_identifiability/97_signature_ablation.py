#!/usr/bin/env python3
"""
Signature Library + Fisher Rank Ablation Study.

Core claim: Using Fisher identifiability rank to select the number of
degradation signatures improves prediction accuracy vs using all or
random subsets.

Ablation design:
1. Build signature library from fullfield simulation data (∂V/∂θ per param)
2. Train linear decomposition model: ΔV = Σ α_i × S_i for different signature subsets
3. Compare:
   a. All 7 signatures (baseline)
   b. Fisher-rank-guided subset (top-3 identifiable params)
   c. Random subsets of same size (repeated)
   d. Leave-one-out ablation
   e. Forward selection (greedy)
   f. Full subset sweep (1 to 7 signatures)

Metrics: decomposition RMSE, capacity prediction R², coefficient stability
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

OUTPUT_DIR = Path("outputs/signature_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]

FISHER_IDENTIFIABLE = ["SEI", "D_n", "LAM_neg"]
FISHER_UNIDENTIFIABLE = ["D_p", "t+", "LAM_pos", "R_mult"]

SEED = 42
np.random.seed(SEED)


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
    return params, V


def build_signatures(params, V):
    """Build ∂V/∂θ signatures via numerical differentiation.

    For each parameter, compute the Jacobian column:
      S_i = ∂V/∂θ_i ≈ (V(θ + δe_i) - V(θ - δe_i)) / 2δ

    Since we have discrete samples, use finite differences in log-parameter space.
    Use the median V as reference, then compute SVD-based signatures.
    """
    n_samples, n_time = V.shape
    n_params = params.shape[1]

    log_params = np.log10(np.clip(params, 1e-30, None))

    V_ref = np.median(V, axis=0)
    P_ref = np.median(log_params, axis=0)

    delta_V = V - V_ref

    U, S, Vt = np.linalg.svd(delta_V, full_matrices=False)

    signatures = np.zeros((n_params, n_time))
    for pi in range(n_params):
        log_p = log_params[:, pi]
        p_sorted_idx = np.argsort(log_p)

        n_bins = 20
        bin_edges = np.linspace(0, len(p_sorted_idx), n_bins + 1, dtype=int)

        slopes = []
        for b in range(n_bins):
            i_start, i_end = bin_edges[b], bin_edges[b + 1]
            if i_end - i_start < 2:
                continue
            idx = p_sorted_idx[i_start:i_end]
            x = log_p[idx]
            y = delta_V[idx]
            coef = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)[0]
            slopes.append(coef.squeeze())

        if slopes:
            signatures[pi] = np.median(slopes, axis=0)

    for pi in range(n_params):
        s = signatures[pi]
        if np.linalg.norm(s) > 1e-10:
            signatures[pi] /= np.linalg.norm(s)

    return signatures, V_ref, delta_V


def build_signatures_svd(delta_V, n_components=None):
    """Build SVD-based signatures (model-agnostic basis)."""
    U, S, Vt = np.linalg.svd(delta_V, full_matrices=False)
    if n_components is not None:
        Vt = Vt[:n_components]
    return Vt, S


def fit_linear_decomposition(delta_V, signatures, param_indices):
    """Fit ΔV = Σ α_i × S_i using Ridge regression.

    delta_V: (n_samples, n_time)
    signatures: (n_params, n_time)
    param_indices: which params to include

    Returns: coefficients (n_samples, len(param_indices)), predictions, rmse
    """
    n_samples, n_time = delta_V.shape
    S = signatures[param_indices]  # (n_selected, n_time)
    n_sel = len(param_indices)

    best_alpha = 0.1
    best_score = -np.inf
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        scores = []
        for fold in range(3):
            rng_fold = np.random.RandomState(fold)
            perm = rng_fold.permutation(n_time)
            n_val = n_time // 3
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            A_train = S[:, train_idx].T  # (n_train, n_sel)
            A_val = S[:, val_idx].T      # (n_val, n_sel)

            coefs_train = np.zeros((n_samples, n_sel))
            for si in range(n_samples):
                coefs_train[si], _, _, _ = np.linalg.lstsq(
                    A_train, delta_V[si, train_idx], rcond=None
                )

            preds_val = coefs_train @ S  # (n_samples, n_time)
            mse = np.mean((preds_val[:, val_idx] - delta_V[:, val_idx]) ** 2)
            scores.append(-mse)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    A_full = S.T  # (n_time, n_sel)
    coefs = np.zeros((n_samples, n_sel))
    for si in range(n_samples):
        if best_alpha > 0:
            ridge_A = A_full.T @ A_full + best_alpha * np.eye(n_sel)
            ridge_b = A_full.T @ delta_V[si]
            coefs[si] = np.linalg.solve(ridge_A, ridge_b)
        else:
            coefs[si], _, _, _ = np.linalg.lstsq(A_full, delta_V[si], rcond=None)

    preds = coefs @ S  # (n_samples, n_time)
    rmse = np.sqrt(np.mean((delta_V - preds) ** 2, axis=1))

    return coefs, preds, rmse


def predict_capacity_from_coefs(coefs, param_indices, params, V_ref):
    """Use decomposition coefficients to predict capacity-related features.

    Since we don't have direct capacity from the surrogate, use the
    coefficient magnitude and correlation with true parameters as proxy.
    """
    results = {}
    for ci, pi in enumerate(param_indices):
        true_vals = params[:, pi]
        coef_vals = coefs[:, ci]
        if np.std(coef_vals) > 1e-10 and np.std(true_vals) > 1e-10:
            r2 = r2_score(true_vals, coef_vals)
        else:
            r2 = -1.0
        results[PARAM_NAMES[pi]] = float(r2)
    return results


def run_ablation(params, V, signatures, delta_V):
    """Run the full ablation study."""
    n_params = len(PARAM_NAMES)
    all_indices = list(range(n_params))

    fisher_id_indices = [PARAM_NAMES.index(p) for p in FISHER_IDENTIFIABLE]
    fisher_un_indices = [PARAM_NAMES.index(p) for p in FISHER_UNIDENTIFIABLE]

    results = {
        "param_names": PARAM_NAMES,
        "fisher_identifiable": FISHER_IDENTIFIABLE,
        "fisher_unidentifiable": FISHER_UNIDENTIFIABLE,
        "experiments": {},
    }

    # --- Experiment 1: All signatures ---
    logger.info("Experiment 1: All %d signatures", n_params)
    coefs_all, preds_all, rmse_all = fit_linear_decomposition(
        delta_V, signatures, all_indices
    )
    r2_all = 1 - np.sum((delta_V - preds_all) ** 2) / np.sum(delta_V ** 2)
    param_r2_all = predict_capacity_from_coefs(coefs_all, all_indices, params, None)
    results["experiments"]["all_signatures"] = {
        "n_signatures": n_params,
        "signature_set": PARAM_NAMES,
        "mean_rmse": float(rmse_all.mean()),
        "median_rmse": float(np.median(rmse_all)),
        "r2_total": float(r2_all),
        "param_r2": param_r2_all,
    }
    logger.info("  R²=%.4f, mean_rmse=%.4f", r2_all, rmse_all.mean())

    # --- Experiment 2: Fisher-guided subset (top-3) ---
    logger.info("Experiment 2: Fisher-guided (%s)", FISHER_IDENTIFIABLE)
    coefs_fish, preds_fish, rmse_fish = fit_linear_decomposition(
        delta_V, signatures, fisher_id_indices
    )
    r2_fish = 1 - np.sum((delta_V - preds_fish) ** 2) / np.sum(delta_V ** 2)
    param_r2_fish = predict_capacity_from_coefs(coefs_fish, fisher_id_indices, params, None)
    results["experiments"]["fisher_guided"] = {
        "n_signatures": len(FISHER_IDENTIFIABLE),
        "signature_set": FISHER_IDENTIFIABLE,
        "mean_rmse": float(rmse_fish.mean()),
        "median_rmse": float(np.median(rmse_fish)),
        "r2_total": float(r2_fish),
        "param_r2": param_r2_fish,
    }
    logger.info("  R²=%.4f, mean_rmse=%.4f", r2_fish, rmse_fish.mean())

    # --- Experiment 3: Random subsets of size 3 ---
    logger.info("Experiment 3: Random subsets (size=%d, n_trials=30)", len(FISHER_IDENTIFIABLE))
    random_r2s = []
    random_rmses = []
    rng = np.random.RandomState(SEED)
    for trial in range(30):
        rand_idx = sorted(rng.choice(n_params, len(FISHER_IDENTIFIABLE), replace=False).tolist())
        _, preds_rand, rmse_rand = fit_linear_decomposition(
            delta_V, signatures, rand_idx
        )
        r2_rand = 1 - np.sum((delta_V - preds_rand) ** 2) / np.sum(delta_V ** 2)
        random_r2s.append(r2_rand)
        random_rmses.append(rmse_rand.mean())

    results["experiments"]["random_subsets"] = {
        "n_signatures": len(FISHER_IDENTIFIABLE),
        "n_trials": 30,
        "r2_mean": float(np.mean(random_r2s)),
        "r2_std": float(np.std(random_r2s)),
        "r2_best": float(np.max(random_r2s)),
        "r2_worst": float(np.min(random_r2s)),
        "rmse_mean": float(np.mean(random_rmses)),
        "fisher_vs_random_sigma": float(
            (r2_fish - np.mean(random_r2s)) / (np.std(random_r2s) + 1e-10)
        ),
    }
    logger.info(
        "  Random R²: mean=%.4f±%.4f, Fisher is %.1fσ above mean",
        np.mean(random_r2s), np.std(random_r2s),
        (r2_fish - np.mean(random_r2s)) / (np.std(random_r2s) + 1e-10),
    )

    # --- Experiment 4: Forward selection ---
    logger.info("Experiment 4: Forward selection (greedy)")
    selected = []
    remaining = list(range(n_params))
    forward_results = []

    for step in range(n_params):
        best_r2 = -np.inf
        best_idx = None
        for idx in remaining:
            trial_set = selected + [idx]
            _, preds_t, _ = fit_linear_decomposition(delta_V, signatures, trial_set)
            r2_t = 1 - np.sum((delta_V - preds_t) ** 2) / np.sum(delta_V ** 2)
            if r2_t > best_r2:
                best_r2 = r2_t
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        forward_results.append({
            "step": step + 1,
            "added_param": PARAM_NAMES[best_idx],
            "selected": [PARAM_NAMES[i] for i in selected],
            "r2": float(best_r2),
        })
        logger.info("  Step %d: added %s → R²=%.4f", step + 1, PARAM_NAMES[best_idx], best_r2)

    results["experiments"]["forward_selection"] = forward_results

    # --- Experiment 5: Subset size sweep ---
    logger.info("Experiment 5: Subset size sweep (1 to %d)", n_params)
    sweep_results = []
    for k in range(1, n_params + 1):
        best_r2_k = -np.inf
        best_combo_k = None
        for combo in combinations(range(n_params), k):
            _, preds_k, _ = fit_linear_decomposition(delta_V, signatures, list(combo))
            r2_k = 1 - np.sum((delta_V - preds_k) ** 2) / np.sum(delta_V ** 2)
            if r2_k > best_r2_k:
                best_r2_k = r2_k
                best_combo_k = combo

        sweep_results.append({
            "k": k,
            "best_r2": float(best_r2_k),
            "best_params": [PARAM_NAMES[i] for i in best_combo_k],
            "fisher_guided_r2": float(r2_fish) if k == len(FISHER_IDENTIFIABLE) else None,
        })
        fisher_note = " ← Fisher-guided" if k == len(FISHER_IDENTIFIABLE) else ""
        logger.info("  k=%d: best R²=%.4f, params=%s%s",
                     k, best_r2_k, [PARAM_NAMES[i] for i in best_combo_k], fisher_note)

    results["experiments"]["subset_sweep"] = sweep_results

    # --- Experiment 6: Leave-one-out ---
    logger.info("Experiment 6: Leave-one-signature-out")
    loo_results = {}
    full_r2 = r2_all
    for pi, pname in enumerate(PARAM_NAMES):
        keep = [i for i in range(n_params) if i != pi]
        _, preds_loo, rmse_loo = fit_linear_decomposition(delta_V, signatures, keep)
        r2_loo = 1 - np.sum((delta_V - preds_loo) ** 2) / np.sum(delta_V ** 2)
        delta_r2 = full_r2 - r2_loo
        loo_results[pname] = {
            "r2_without": float(r2_loo),
            "delta_r2": float(delta_r2),
            "is_identifiable": pname in FISHER_IDENTIFIABLE,
        }
        logger.info("  Without %s: R²=%.4f (ΔR²=%.4f)", pname, r2_loo, delta_r2)

    results["experiments"]["leave_one_out"] = loo_results

    # --- Experiment 7: SVD-based signatures (model-agnostic) ---
    logger.info("Experiment 7: SVD-based signatures")
    svd_signatures, svd_vals = build_signatures_svd(delta_V)
    for k in [3, 5, 7]:
        svd_sig_k = svd_signatures[:k]
        coefs_svd = delta_V @ svd_sig_k.T @ np.linalg.pinv(svd_sig_k @ svd_sig_k.T)
        preds_svd = coefs_svd @ svd_sig_k
        r2_svd = 1 - np.sum((delta_V - preds_svd) ** 2) / np.sum(delta_V ** 2)
        logger.info("  SVD k=%d: R²=%.4f", k, r2_svd)
        results["experiments"][f"svd_k{k}"] = {
            "n_components": k,
            "r2": float(r2_svd),
        }

    # --- Summary ---
    results["summary"] = {
        "all_r2": float(r2_all),
        "fisher_r2": float(r2_fish),
        "fisher_n_params": len(FISHER_IDENTIFIABLE),
        "random_mean_r2": float(np.mean(random_r2s)),
        "fisher_advantage_vs_random_sigma": float(
            (r2_fish - np.mean(random_r2s)) / (np.std(random_r2s) + 1e-10)
        ),
        "forward_selection_order": [PARAM_NAMES[i] for i in selected],
        "best_3_params": forward_results[2]["selected"] if len(forward_results) >= 3 else None,
        "fisher_matches_best_3": set(FISHER_IDENTIFIABLE) == set(forward_results[2]["selected"]) if len(forward_results) >= 3 else False,
    }

    return results


def main():
    logger.info("Loading data ...")
    params, V = load_data()
    logger.info("Data: %d samples, %d params, %d V-points", *params.shape, V.shape[1])

    logger.info("Building signatures ...")
    signatures, V_ref, delta_V = build_signatures(params, V)
    logger.info("Signatures shape: %s", signatures.shape)

    for pi, pn in enumerate(PARAM_NAMES):
        logger.info("  %s signature norm: %.4f", pn, np.linalg.norm(signatures[pi]))

    results = run_ablation(params, V, signatures, delta_V)

    out_path = OUTPUT_DIR / "signature_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    s = results["summary"]
    logger.info("  All sigs R²:     %.4f", s["all_r2"])
    logger.info("  Fisher R²:       %.4f (n=%d)", s["fisher_r2"], s["fisher_n_params"])
    logger.info("  Random mean R²:  %.4f", s["random_mean_r2"])
    logger.info("  Fisher vs random: %.1fσ", s["fisher_advantage_vs_random_sigma"])
    logger.info("  Forward selection: %s", s["forward_selection_order"])
    logger.info("  Fisher matches best-3: %s", s["fisher_matches_best_3"])
    logger.info("  Saved to %s", out_path)


if __name__ == "__main__":
    main()
