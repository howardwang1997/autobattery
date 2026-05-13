#!/usr/bin/env python3
"""
Profile Likelihood Analysis for Battery Degradation Parameter Identifiability.

For each parameter θ_i:
  - Fix θ_i at a grid of values
  - Optimize all other parameters θ_{-i} to minimize ||V_model - V_data||²
  - Plot the profile: L_p(θ_i) = min_{θ_{-i}} ||V_model(θ_i, θ_{-i}) - V_data||²

Identifiable params: sharp V-shaped profile with narrow confidence interval
Unidentifiable params: flat profile with wide/infinite confidence interval

Uses an MLP surrogate trained on fullfield simulation data for speed.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = Path("outputs/profile_likelihood")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/fullfield/fullfield_lfp_degradation.h5")

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]

SEED = 42
N_SURROGATE_EPOCHS = 2000
SURROGATE_LR = 1e-3
SURROGATE_HIDDEN = 256
N_PROFILE_POINTS = 25
N_TEST_CASES = 3
N_RESTARTS = 1

torch.manual_seed(SEED)
np.random.seed(SEED)


class SurrogateModel(nn.Module):
    def __init__(self, n_params=7, n_outputs=100, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    """Load fullfield simulation data."""
    with h5py.File(DATA_PATH, "r") as f:
        V = f["V"][:].astype(np.float64)
        params = f["params"][:].astype(np.float64)
    return params, V


def train_surrogate(params, V):
    """Train MLP surrogate: θ → V(t)."""
    logger.info("Training MLP surrogate ...")

    log_params = np.log10(np.clip(params, 1e-30, None))
    V_norm = V.copy()

    p_mean, p_std = log_params.mean(axis=0), log_params.std(axis=0) + 1e-8
    v_mean, v_std = V_norm.mean(axis=0), V_norm.std(axis=0) + 1e-8

    p_scaled = (log_params - p_mean) / p_std
    v_scaled = (V_norm - v_mean) / v_std

    X = torch.tensor(p_scaled, dtype=torch.float32)
    Y = torch.tensor(v_scaled, dtype=torch.float32)

    n = len(X)
    n_train = int(0.9 * n)
    ds = TensorDataset(X[:n_train], Y[:n_train])
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = SurrogateModel(hidden=SURROGATE_HIDDEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SURROGATE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200)

    best_loss = float("inf")
    best_state = None
    patience = 500
    patience_counter = 0

    for epoch in range(N_SURROGATE_EPOCHS):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X[n_train:])
                val_loss = nn.functional.mse_loss(val_pred, Y[n_train:]).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 50
            logger.info(
                "  Epoch %d: val_loss=%.6f (best=%.6f)",
                epoch + 1, val_loss, best_loss,
            )
            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        all_pred = model(X).numpy()
        r2 = 1 - np.sum((V_norm - (all_pred * v_std + v_mean)) ** 2) / np.sum(
            (V_norm - V_norm.mean()) ** 2
        )
    logger.info("  Surrogate R² on all data: %.4f", r2)

    return model, p_mean, p_std, v_mean, v_std


def surrogate_predict(model, theta, p_mean, p_std, v_mean, v_std):
    """Predict V(t) from θ using surrogate. theta can be 1D or 2D."""
    theta = np.atleast_2d(theta)
    log_theta = np.log10(np.clip(theta, 1e-30, None))
    p_scaled = (log_theta - p_mean) / p_std
    with torch.no_grad():
        x = torch.tensor(p_scaled, dtype=torch.float32)
        v_scaled = model(x).numpy()
    V = v_scaled * v_std + v_mean
    if V.shape[0] == 1:
        return V.squeeze(0)
    return V


def compute_profile_likelihood(model, theta_true, V_true, p_mean, p_std, v_mean, v_std, param_idx):
    """Compute profile likelihood for one parameter.

    For each fixed value of θ_i on a grid:
      min_{θ_{-i}} ||V(θ_i, θ_{-i}) - V_true||²
    """
    log_true = np.log10(np.clip(theta_true, 1e-30, None))
    log_val = log_true[param_idx]

    log_range_low = log_val - 1.0
    log_range_high = log_val + 1.0
    log_grid = np.linspace(log_range_low, log_range_high, N_PROFILE_POINTS)

    profile_losses = []

    for log_fixed in log_grid:
        best_loss = float("inf")

        for restart in range(N_RESTARTS):
            log_other = log_true.copy()
            if restart > 0:
                noise = np.random.randn(len(log_other)) * 0.3
                noise[param_idx] = 0
                log_other = log_other + noise

            def objective(log_free):
                log_full = log_other.copy()
                free_mask = np.ones(len(log_full), dtype=bool)
                free_mask[param_idx] = False
                log_full[free_mask] = log_free

                theta_full = 10.0 ** log_full
                V_pred = surrogate_predict(model, theta_full, p_mean, p_std, v_mean, v_std)
                return np.sum((V_pred - V_true) ** 2)

            from scipy.optimize import minimize as sp_minimize
            free_mask = np.ones(len(log_true), dtype=bool)
            free_mask[param_idx] = False
            x0 = log_true[free_mask]

            result = sp_minimize(
                objective, x0, method="L-BFGS-B",
                options={"maxiter": 100, "ftol": 1e-10},
            )
            if result.fun < best_loss:
                best_loss = result.fun

        profile_losses.append(best_loss)

    profile_losses = np.array(profile_losses)

    theta_opt_loss = profile_losses[N_PROFILE_POINTS // 2]

    if theta_opt_loss > 0:
        profile_chi2 = (profile_losses - profile_losses.min()) / theta_opt_loss
    else:
        profile_chi2 = np.zeros_like(profile_losses)

    return log_grid, profile_losses, profile_chi2


def compute_confidence_interval(log_grid, profile_chi2, threshold=3.84):
    """Compute 95% confidence interval from profile likelihood (χ²(1) threshold=3.84)."""
    below = profile_chi2 <= threshold
    if not below.any():
        return None, None

    indices = np.where(below)[0]
    ci_low = 10.0 ** log_grid[indices[0]]
    ci_high = 10.0 ** log_grid[indices[-1]]

    return ci_low, ci_high


def compute_empirical_profile(params, V, theta_true, V_true, param_idx):
    """Empirical profile: for each θ_i bin, find min residual across all samples.
    No optimization needed — uses the database directly.
    """
    log_params = np.log10(np.clip(params, 1e-30, None))
    log_true_val = np.log10(np.clip(theta_true[param_idx], 1e-30, None))

    log_grid = np.linspace(log_true_val - 1.0, log_true_val + 1.0, N_PROFILE_POINTS)

    profile_losses = []
    for log_fixed in log_grid:
        dists = np.abs(log_params[:, param_idx] - log_fixed)
        mask = dists < 0.15
        if mask.sum() == 0:
            mask = dists < 0.3
        if mask.sum() == 0:
            profile_losses.append(float("inf"))
            continue

        residuals = np.sum((V[mask] - V_true) ** 2, axis=1)
        profile_losses.append(float(residuals.min()))

    profile_losses = np.array(profile_losses)
    min_loss = profile_losses[profile_losses < float("inf")].min()
    valid = profile_losses < float("inf")

    profile_chi2 = np.full_like(profile_losses, float("inf"))
    if min_loss > 0:
        profile_chi2[valid] = (profile_losses[valid] - min_loss) / min_loss

    return log_grid, profile_losses, profile_chi2


def main():
    params, V = load_data()
    logger.info("Data: %d samples, %d params, %d V-points", *params.shape, V.shape[1])

    model, p_mean, p_std, v_mean, v_std = train_surrogate(params, V)

    np.random.seed(SEED)
    test_indices = np.random.choice(len(params), N_TEST_CASES, replace=False)

    all_results = {}

    for case_idx, test_i in enumerate(test_indices):
        theta_true = params[test_i]
        V_true = V[test_i]

        case_id = f"case_{case_idx}"
        logger.info("Test case %d (sample %d): %s", case_idx, test_i,
                     ", ".join(f"{n}={v:.4e}" for n, v in zip(PARAM_NAMES, theta_true)))

        case_results = {
            "theta_true": theta_true.tolist(),
            "param_names": PARAM_NAMES,
            "profiles": {},
            "empirical_profiles": {},
            "confidence_intervals": {},
        }

        for pi, pname in enumerate(PARAM_NAMES):
            logger.info("  Profiling %s ...", pname)

            log_grid_e, losses_e, chi2_e = compute_empirical_profile(
                params, V, theta_true, V_true, pi
            )
            case_results["empirical_profiles"][pname] = {
                "log_grid": log_grid_e.tolist(),
                "losses": [float(x) if np.isfinite(x) else None for x in losses_e],
                "chi2": [float(x) if np.isfinite(x) else None for x in chi2_e],
            }

            log_grid, losses, chi2 = compute_profile_likelihood(
                model, theta_true, V_true, p_mean, p_std, v_mean, v_std, pi
            )

            ci_low, ci_high = compute_confidence_interval(log_grid, chi2)
            true_val = theta_true[pi]

            is_bounded = ci_low is not None and ci_high is not None
            if is_bounded:
                ci_width_decades = np.log10(ci_high) - np.log10(ci_low)
                rel_width = (ci_high - ci_low) / true_val if true_val > 0 else float("inf")
            else:
                ci_width_decades = float("inf")
                rel_width = float("inf")

            case_results["profiles"][pname] = {
                "log_grid": log_grid.tolist(),
                "losses": losses.tolist(),
                "chi2": chi2.tolist(),
            }
            case_results["confidence_intervals"][pname] = {
                "ci_low": float(ci_low) if ci_low is not None else None,
                "ci_high": float(ci_high) if ci_high is not None else None,
                "ci_width_decades": float(ci_width_decades),
                "relative_width": float(rel_width),
                "true_value": float(true_val),
                "is_bounded": is_bounded,
            }

            logger.info(
                "    %s: ci_width=%.1f decades, rel=%.2f, bounded=%s",
                pname, ci_width_decades, rel_width, is_bounded,
            )

        all_results[case_id] = case_results

    summary = {"param_names": PARAM_NAMES, "cases": {}}
    for pname in PARAM_NAMES:
        widths = []
        bounded_count = 0
        for case_id in all_results:
            ci = all_results[case_id]["confidence_intervals"][pname]
            if ci["is_bounded"]:
                widths.append(ci["ci_width_decades"])
                bounded_count += 1
        summary["cases"][pname] = {
            "mean_ci_width_decades": float(np.mean(widths)) if widths else float("inf"),
            "median_ci_width_decades": float(np.median(widths)) if widths else float("inf"),
            "bounded_fraction": float(bounded_count / N_TEST_CASES),
        }

    logger.info("=" * 60)
    logger.info("PROFILE LIKELIHOOD SUMMARY")
    logger.info("=" * 60)
    logger.info("%-12s %12s %12s %10s", "Parameter", "Mean CI (dec)", "Med CI (dec)", "Bounded")
    for pname in PARAM_NAMES:
        s = summary["cases"][pname]
        logger.info(
            "%-12s %12.2f %12.2f %9.0f/%d",
            pname, s["mean_ci_width_decades"], s["median_ci_width_decades"],
            s["bounded_fraction"] * N_TEST_CASES, N_TEST_CASES,
        )

    output = {
        "summary": summary,
        "cases": all_results,
        "config": {
            "n_test_cases": N_TEST_CASES,
            "n_profile_points": N_PROFILE_POINTS,
            "n_restarts": N_RESTARTS,
            "chi2_threshold_95": 3.84,
        },
    }

    out_path = OUTPUT_DIR / "profile_likelihood_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
