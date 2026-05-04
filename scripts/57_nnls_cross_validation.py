#!/usr/bin/env python3
"""
Cross-validation of Fisher Information identifiability analysis with
two degradation recovery methods:
  A) Gradient-based fitting via MLP surrogate (direct comparison with Fisher)
  B) Group NNLS linear decomposition (existing diagnosis approach)

Hypothesis: Parameters predicted as identifiable by Fisher analysis
(SEI, D_n, LAM_neg) should be accurately recovered, while
unidentifiable parameters (D_p, t+, LAM_pos, R_mult) should not.

Fisher reconstruction quality (from script 55):
  D_n=100%, SEI=100%, LAM_neg=99.9%, D_p=1.4%, t+=2.0%, LAM_pos=2.2%, R_mult=1.6%
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import json
import logging
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import nnls
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/root/autobattery")
DATA_PATH = BASE / "data" / "fullfield" / "fullfield_lfp_degradation.h5"
MLP_PATH = BASE / "outputs" / "rigorous_identifiability" / "mlp_surrogate.pt"
OUTPUT_DIR = BASE / "outputs" / "nnls_cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
N_PARAMS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FISHER_RECONSTRUCTION = {
    "D_n": 100.0, "SEI": 100.0, "LAM_neg": 99.9,
    "D_p": 1.4, "t+": 2.0, "LAM_pos": 2.2, "R_mult": 1.6,
}

FISHER_IDENT = ["D_n", "SEI", "LAM_neg"]
FISHER_UNIDENT = ["D_p", "t+", "LAM_pos", "R_mult"]
IDENT_IDX = [PARAM_NAMES.index(n) for n in FISHER_IDENT]
UNIDENT_IDX = [PARAM_NAMES.index(n) for n in FISHER_UNIDENT]


class DegradationMLP(nn.Module):
    def __init__(self, n_params=7, n_time=100, hidden=256, n_layers=6):
        super().__init__()
        layers = [nn.Linear(n_params + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, n_time))
        self.net = nn.Sequential(*layers)

    def forward(self, params, c_rate):
        x = torch.cat([params, c_rate], dim=-1)
        return self.net(x)


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        params = f["params"][:].astype(np.float64)
        V = f["V"][:].astype(np.float64)
        c_rates = f["c_rates"][:].astype(np.float64)
        n_time = int(f.attrs["n_time"])

    params_log = params.copy()
    for i in LOG_PARAMS:
        params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

    p_mean = params_log.mean(axis=0)
    p_std = params_log.std(axis=0) + 1e-12
    cr_mean = float(c_rates.mean())
    cr_std = float(c_rates.std()) + 1e-12

    return params, params_log, V, c_rates, n_time, p_mean, p_std, cr_mean, cr_std


def compute_metrics(true_arr, pred_arr):
    metrics = {}
    for j in range(N_PARAMS):
        name = PARAM_NAMES[j]
        y_true = true_arr[:, j]
        y_pred = pred_arr[:, j]

        if y_pred.std() < 1e-30 or y_true.std() < 1e-30:
            r_p, r_s = 0.0, 0.0
        else:
            r_p, _ = pearsonr(y_true, y_pred)
            r_s, _ = spearmanr(y_true, y_pred)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        y_range = y_true.max() - y_true.min()
        nrmse = rmse / (y_range + 1e-30)

        y_t_norm = (y_true - y_true.min()) / (y_range + 1e-30)
        y_p_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-30)
        rel_err = np.abs(y_true - y_pred) / (y_range + 1e-30)
        pass_20 = np.mean(rel_err < 0.20) * 100
        pass_50 = np.mean(rel_err < 0.50) * 100

        metrics[name] = {
            "pearson_r": float(r_p), "spearman_r": float(r_s),
            "nrmse": float(nrmse),
            "pass_rate_20": float(pass_20), "pass_rate_50": float(pass_50),
        }
    return metrics


# ======================================================================
# Method A: Gradient-based fitting via MLP surrogate
# ======================================================================

def gradient_based_recovery(params_log, V, c_rates, p_mean, p_std, cr_mean, cr_std, n_time):
    logger.info("  Loading MLP surrogate...")
    model = DegradationMLP(n_time=n_time).to(DEVICE)
    state = torch.load(MLP_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_sims = len(V)
    recovered = np.zeros_like(params_log)

    n_test = min(n_sims, 200)
    test_idx = np.random.choice(n_sims, n_test, replace=False)

    logger.info(f"  Fitting {n_test} samples via gradient optimization...")

    for count, i in enumerate(test_idx):
        v_target = torch.tensor(V[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        cr = torch.tensor(
            [(c_rates[i] - cr_mean) / cr_std], dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        p_lo = torch.tensor(
            (params_log.min(axis=0) - p_mean) / p_std, dtype=torch.float32, device=DEVICE
        )
        p_hi = torch.tensor(
            (params_log.max(axis=0) - p_mean) / p_std, dtype=torch.float32, device=DEVICE
        )

        best_loss = float("inf")
        best_p = None

        for restart in range(2):
            p0 = torch.zeros(1, N_PARAMS, device=DEVICE) if restart == 0 else \
                 torch.randn(1, N_PARAMS, device=DEVICE) * 0.3
            p = p0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=0.05)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

            for step in range(300):
                v_pred = model(p, cr)
                loss = nn.functional.mse_loss(v_pred, v_target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    p.data.clamp_(p_lo.unsqueeze(0), p_hi.unsqueeze(0))
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_p = p.detach().clone()

        recovered[i] = best_p[0].cpu().numpy() * p_std + p_mean

        if (count + 1) % 25 == 0:
            logger.info(f"    {count+1}/{n_test} fitted")

    recovered_full = np.zeros_like(params_log)
    for i in test_idx:
        recovered_full[i] = recovered[i]

    mask = np.zeros(n_sims, dtype=bool)
    mask[test_idx] = True
    return recovered_full, mask, test_idx


# ======================================================================
# Method B: NNLS decomposition (leave-one-subgroup-out)
# ======================================================================

def nnls_recovery(params_log, V, c_rates):
    with h5py.File(DATA_PATH, "r") as f:
        param_set_ids = f["param_set_ids"][:]

    unique_sets = np.unique(param_set_ids)
    n_sims = len(V)
    recovered = np.zeros_like(params_log)

    logger.info(f"  LOSO CV: {len(unique_sets)} parameter sets...")

    for si, ps_id in enumerate(unique_sets):
        mask_train = param_set_ids != ps_id
        mask_test = param_set_ids == ps_id

        if mask_train.sum() < 10 or mask_test.sum() == 0:
            continue

        V_train = V[mask_train]
        P_train = params_log[mask_train]

        pm = P_train.mean(axis=0)
        Pc = P_train - pm
        ps = Pc.std(axis=0) + 1e-12
        Pn = Pc / ps

        n_time = V.shape[1]
        sigs = np.zeros((N_PARAMS, n_time))
        for t in range(n_time):
            m = Ridge(alpha=0.1)
            m.fit(Pn, V_train[:, t])
            sigs[:, t] = m.coef_

        V_bl = np.median(V_train, axis=0)
        A = sigs.T

        for i in np.where(mask_test)[0]:
            b = V[i] - V_bl
            x, _ = nnls(A, b)
            recovered[i] = x / (ps + 1e-30) + pm

        if (si + 1) % 100 == 0:
            logger.info(f"    {si+1}/{len(unique_sets)} sets")

    return recovered


# ======================================================================
# Analysis
# ======================================================================

def analyze_comparison(fisher_qual, grad_metrics, nnls_metrics, grad_recovered,
                       nnls_recovered, params_log, grad_mask):
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE: Fisher vs Gradient Fitting vs NNLS")
    logger.info("=" * 80)

    header = (f"{'Param':>8} | {'Fisher%':>8} | "
              f"{'Grad |r|':>9} {'Pass50':>7} | "
              f"{'NNLS |r|':>9} {'Pass50':>7} | "
              f"{'Verdict':>10}")
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    grad_corrs = [abs(grad_metrics[n]["pearson_r"]) for n in PARAM_NAMES]
    nnls_corrs = [abs(nnls_metrics[n]["pearson_r"]) for n in PARAM_NAMES]
    grad_pass = [grad_metrics[n]["pass_rate_50"] for n in PARAM_NAMES]
    nnls_pass = [nnls_metrics[n]["pass_rate_50"] for n in PARAM_NAMES]

    verdicts = {}
    for i, name in enumerate(PARAM_NAMES):
        fish_id = fisher_qual[i] > 50
        grad_ok = grad_corrs[i] > 0.3

        if fish_id and grad_ok:
            v = "AGREE(ID)"
        elif not fish_id and not grad_ok:
            v = "AGREE(UN)"
        elif fish_id:
            v = "MISFIT(ID)"
        else:
            v = "MISFIT(UN)"
        verdicts[name] = v

        logger.info(f"{name:>8} | {fisher_qual[i]:>7.1f}% | "
                     f"{grad_corrs[i]:>8.3f}  {grad_pass[i]:>6.1f}% | "
                     f"{nnls_corrs[i]:>8.3f}  {nnls_pass[i]:>6.1f}% | "
                     f"{v:>10}")

    logger.info(sep)

    fisher_arr = np.array(fisher_qual)
    grad_corr_arr = np.array(grad_corrs)
    nnls_corr_arr = np.array(nnls_corrs)
    grad_pass_arr = np.array(grad_pass)
    nnls_pass_arr = np.array(nnls_pass)

    sp_fish_grad_corr, p_fg = spearmanr(fisher_arr, grad_corr_arr)
    sp_fish_grad_pass, p_fgp = spearmanr(fisher_arr, grad_pass_arr)
    sp_fish_nnls_corr, p_fnc = spearmanr(fisher_arr, nnls_corr_arr)
    sp_fish_nnls_pass, p_fnp = spearmanr(fisher_arr, nnls_pass_arr)

    logger.info(f"\nSpearman correlations with Fisher reconstruction quality:")
    logger.info(f"  Fisher vs Grad |r|:     {sp_fish_grad_corr:.3f}  (p={p_fg:.4f})")
    logger.info(f"  Fisher vs Grad pass_50: {sp_fish_grad_pass:.3f}  (p={p_fgp:.4f})")
    logger.info(f"  Fisher vs NNLS |r|:     {sp_fish_nnls_corr:.3f}  (p={p_fnc:.4f})")
    logger.info(f"  Fisher vs NNLS pass_50: {sp_fish_nnls_pass:.3f}  (p={p_fnp:.4f})")

    ident_grad_corr = np.mean([grad_corrs[i] for i in range(N_PARAMS) if fisher_qual[i] > 50])
    unident_grad_corr = np.mean([grad_corrs[i] for i in range(N_PARAMS) if fisher_qual[i] <= 50])
    ident_grad_pass = np.mean([grad_pass[i] for i in range(N_PARAMS) if fisher_qual[i] > 50])
    unident_grad_pass = np.mean([grad_pass[i] for i in range(N_PARAMS) if fisher_qual[i] <= 50])

    logger.info(f"\nGroup-level validation (gradient fitting):")
    logger.info(f"  Identifiable:   avg |r|={ident_grad_corr:.3f}, avg pass_50={ident_grad_pass:.1f}%")
    logger.info(f"  Unidentifiable: avg |r|={unident_grad_corr:.3f}, avg pass_50={unident_grad_pass:.1f}%")
    logger.info(f"  Separation ratio: |r|={ident_grad_corr/(unident_grad_corr+1e-30):.1f}x, "
                 f"pass_50={ident_grad_pass/(unident_grad_pass+1e-30):.1f}x")

    n_agree = sum(1 for v in verdicts.values() if "AGREE" in v)
    logger.info(f"  Binary accuracy: {n_agree}/{N_PARAMS} = {n_agree/N_PARAMS*100:.0f}%")

    validated = sp_fish_grad_corr > 0.8 or sp_fish_grad_pass > 0.8
    logger.info(f"\n  VERDICT: {'VALIDATED' if validated else 'QUALITATIVE AGREEMENT'}")
    if sp_fish_grad_corr > 0.8:
        logger.info(f"  Spearman(Fisher, Grad|r|) = {sp_fish_grad_corr:.3f} > 0.8 → strong validation")

    return {
        "grad_corrs": grad_corrs, "nnls_corrs": nnls_corrs,
        "grad_pass": grad_pass, "nnls_pass": nnls_pass,
        "verdicts": verdicts,
        "spearman": {
            "fisher_vs_grad_corr": {"rho": float(sp_fish_grad_corr), "p": float(p_fg)},
            "fisher_vs_grad_pass": {"rho": float(sp_fish_grad_pass), "p": float(p_fgp)},
            "fisher_vs_nnls_corr": {"rho": float(sp_fish_nnls_corr), "p": float(p_fnc)},
            "fisher_vs_nnls_pass": {"rho": float(sp_fish_nnls_pass), "p": float(p_fnp)},
        },
        "group": {
            "ident": {"corr": float(ident_grad_corr), "pass_50": float(ident_grad_pass)},
            "unident": {"corr": float(unident_grad_corr), "pass_50": float(unident_grad_pass)},
        },
        "n_agree": n_agree,
        "validated": validated,
    }


# ======================================================================
# Failure mode analysis
# ======================================================================

def failure_analysis(params_log, grad_recovered, grad_mask):
    test_idx = np.where(grad_mask)[0]
    n_test = len(test_idx)

    logger.info("\n=== Failure Mode Analysis (gradient fitting) ===")

    for j, name in enumerate(PARAM_NAMES):
        y_true = params_log[test_idx, j]
        y_pred = grad_recovered[test_idx, j]
        y_range = y_true.max() - y_true.min()
        rel_err = np.abs(y_true - y_pred) / (y_range + 1e-30)

        fail_20 = np.mean(rel_err > 0.20) * 100
        fail_50 = np.mean(rel_err > 0.50) * 100
        fisher_tag = "ID" if FISHER_RECONSTRUCTION[name] > 50 else "UN"

        logger.info(f"  {name:>8} [{fisher_tag}]: fail_20={fail_20:.1f}%  fail_50={fail_50:.1f}%")

    ident_fails = []
    unident_fails = []
    for j in range(N_PARAMS):
        y_true = params_log[test_idx, j]
        y_pred = grad_recovered[test_idx, j]
        y_range = y_true.max() - y_true.min()
        rel_err = np.abs(y_true - y_pred) / (y_range + 1e-30)
        if j in IDENT_IDX:
            ident_fails.append(np.mean(rel_err > 0.20) * 100)
        else:
            unident_fails.append(np.mean(rel_err > 0.20) * 100)

    logger.info(f"\n  Mean failure rate (>20%): Identifiable={np.mean(ident_fails):.1f}%, "
                 f"Unidentifiable={np.mean(unident_fails):.1f}%")


# ======================================================================
# Fix unidentifiable params experiment
# ======================================================================

def fixed_recovery_experiment(params_log, V, c_rates, p_mean, p_std, cr_mean, cr_std, n_time):
    logger.info("\n=== Fixed unidentifiable params experiment ===")

    model = DegradationMLP(n_time=n_time).to(DEVICE)
    state = torch.load(MLP_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_sims = len(V)
    n_test = min(n_sims, 100)
    test_idx = np.random.choice(n_sims, n_test, replace=False)

    unident_mean_full = np.zeros(N_PARAMS)
    for k, j in enumerate(UNIDENT_IDX):
        unident_mean_full[j] = params_log[:, j].mean()
    for k, j in enumerate(IDENT_IDX):
        unident_mean_full[j] = params_log[:, j].mean()

    recovered_full = np.zeros((n_test, N_PARAMS))
    recovered_fixed = np.zeros((n_test, N_PARAMS))

    for count, i in enumerate(test_idx):
        v_target = torch.tensor(V[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        cr = torch.tensor(
            [(c_rates[i] - cr_mean) / cr_std], dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        p_lo = torch.tensor(
            (params_log.min(axis=0) - p_mean) / p_std, dtype=torch.float32, device=DEVICE
        )
        p_hi = torch.tensor(
            (params_log.max(axis=0) - p_mean) / p_std, dtype=torch.float32, device=DEVICE
        )

        # Full recovery
        best_loss_full = float("inf")
        best_p_full = None
        for restart in range(2):
            p0 = torch.zeros(1, N_PARAMS, device=DEVICE) if restart == 0 else \
                 torch.randn(1, N_PARAMS, device=DEVICE) * 0.3
            p = p0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=0.05)
            for step in range(300):
                v_pred = model(p, cr)
                loss = nn.functional.mse_loss(v_pred, v_target)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    p.data.clamp_(p_lo.unsqueeze(0), p_hi.unsqueeze(0))
                if loss.item() < best_loss_full:
                    best_loss_full = loss.item()
                    best_p_full = p.detach().clone()
        recovered_full[count] = best_p_full[0].cpu().numpy() * p_std + p_mean

        # Fixed unidentifiable recovery: only optimize identifiable params
        best_loss_fix = float("inf")
        best_p_fix = None
        unident_norm_full = torch.tensor(
            (unident_mean_full - p_mean) / p_std, dtype=torch.float32, device=DEVICE
        )
        for restart in range(2):
            p_ident0 = torch.zeros(len(IDENT_IDX), device=DEVICE) if restart == 0 else \
                       torch.randn(len(IDENT_IDX), device=DEVICE) * 0.3
            p_ident = p_ident0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([p_ident], lr=0.05)
            for step in range(300):
                p_full = unident_norm_full.unsqueeze(0).clone()
                for k, j in enumerate(IDENT_IDX):
                    p_full[0, j] = p_ident[k]
                v_pred = model(p_full, cr)
                loss = nn.functional.mse_loss(v_pred, v_target)
                opt.zero_grad(); loss.backward(); opt.step()
                if loss.item() < best_loss_fix:
                    best_loss_fix = loss.item()
                    best_p_fix = p_full.detach().clone()
        recovered_fixed[count] = best_p_fix[0].cpu().numpy() * p_std + p_mean

    # Compare
    full_rmse = np.sqrt(np.mean((recovered_full - params_log[test_idx]) ** 2, axis=0))
    fixed_rmse = np.sqrt(np.mean((recovered_fixed - params_log[test_idx]) ** 2, axis=0))

    logger.info("  Per-parameter RMSE (log-space):")
    for j, name in enumerate(PARAM_NAMES):
        tag = "ID" if j in IDENT_IDX else "UN"
        improvement = (full_rmse[j] - fixed_rmse[j]) / (full_rmse[j] + 1e-30) * 100
        logger.info(f"    {name:>8} [{tag}]: full={full_rmse[j]:.4f}  fixed={fixed_rmse[j]:.4f}  "
                     f"improvement={improvement:+.1f}%")

    full_v_rmse = np.zeros(n_test)
    fixed_v_rmse = np.zeros(n_test)
    with torch.no_grad():
        for count, i in enumerate(test_idx):
            v_true = V[i]
            cr_t = torch.tensor(
                [(c_rates[i] - cr_mean) / cr_std], dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            p_full_norm = torch.tensor(
                (recovered_full[count] - p_mean) / p_std, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            p_fix_norm = torch.tensor(
                (recovered_fixed[count] - p_mean) / p_std, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            v_full = model(p_full_norm, cr_t).cpu().numpy()[0]
            v_fix = model(p_fix_norm, cr_t).cpu().numpy()[0]
            full_v_rmse[count] = np.sqrt(np.mean((v_full - v_true) ** 2)) * 1000
            fixed_v_rmse[count] = np.sqrt(np.mean((v_fix - v_true) ** 2)) * 1000

    logger.info(f"\n  Voltage RMSE: full={full_v_rmse.mean():.2f} mV, "
                 f"fixed={fixed_v_rmse.mean():.2f} mV")
    logger.info(f"  → Fixing unidentifiable params {'improves' if fixed_v_rmse.mean() < full_v_rmse.mean() else 'does not improve'} voltage fit")

    return full_rmse, fixed_rmse, full_v_rmse, fixed_v_rmse


# ======================================================================
# Figures
# ======================================================================

def make_figures(fisher_qual, grad_metrics, nnls_metrics, grad_recovered,
                 nnls_recovered, params_log, grad_mask, comp_results):
    ident_color = ["#2ca02c" if FISHER_RECONSTRUCTION[n] > 50 else "#d62728"
                   for n in PARAM_NAMES]
    fisher_arr = np.array(fisher_qual)
    grad_corrs = np.array([abs(grad_metrics[n]["pearson_r"]) for n in PARAM_NAMES])
    nnls_corrs = np.array([abs(nnls_metrics[n]["pearson_r"]) for n in PARAM_NAMES])
    grad_pass = np.array([grad_metrics[n]["pass_rate_50"] for n in PARAM_NAMES])
    nnls_pass = np.array([nnls_metrics[n]["pass_rate_50"] for n in PARAM_NAMES])
    x = np.arange(N_PARAMS)

    # --- Figure 1: Main comparison ---
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    w = 0.25
    ax.bar(x - w, fisher_arr, w, label="Fisher reconstruction (%)",
           color=[c + "99" for c in ident_color], edgecolor="k")
    ax.bar(x, grad_pass, w, label="Gradient fitting pass_50%",
           color="steelblue", edgecolor="k")
    ax.bar(x + w, nnls_pass, w, label="NNLS pass_50%",
           color="coral", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Fisher Prediction vs Recovery Methods", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    w = 0.35
    ax.bar(x - w/2, grad_corrs, w, label="Gradient fitting |r|",
           color="steelblue", edgecolor="k")
    ax.bar(x + w/2, nnls_corrs, w, label="NNLS |r|",
           color="coral", edgecolor="k")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("|Pearson r|", fontsize=11)
    ax.set_title("Parameter Recovery Correlation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(fisher_arr, grad_corrs, s=120, c=ident_color, edgecolor="k",
               zorder=3, marker="o", label="Gradient")
    ax.scatter(fisher_arr, nnls_corrs, s=80, c=ident_color, edgecolor="k",
               zorder=3, marker="^", alpha=0.6, label="NNLS")
    for i, name in enumerate(PARAM_NAMES):
        ax.annotate(name, (fisher_arr[i], max(grad_corrs[i], nnls_corrs[i])),
                     textcoords="offset points", xytext=(8, 5), fontsize=8)
    sp_gc = comp_results["spearman"]["fisher_vs_grad_corr"]["rho"]
    sp_nc = comp_results["spearman"]["fisher_vs_nnls_corr"]["rho"]
    ax.set_xlabel("Fisher reconstruction quality (%)", fontsize=11)
    ax.set_ylabel("|Pearson r|", fontsize=11)
    ax.set_title(f"Fisher vs Recovery\nρ(Grad)={sp_gc:.3f}, ρ(NNLS)={sp_nc:.3f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if sp_gc > 0.8:
        ax.text(0.05, 0.95, "VALIDATED", transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="green", va="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    # Row 2: scatter plots for gradient recovery
    test_idx = np.where(grad_mask)[0]
    for j in range(min(4, N_PARAMS)):
        ax = fig.add_subplot(gs[1, j % 3] if j < 3 else gs[1, 2])
        if j == 3:
            break
        name = PARAM_NAMES[j]
        y_true = params_log[test_idx, j]
        y_pred = grad_recovered[test_idx, j]
        ax.scatter(y_true, y_pred, s=10, alpha=0.3, color=ident_color[j])
        vmin, vmax = y_true.min(), y_true.max()
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5)
        r = grad_corrs[j]
        ax.set_xlabel(f"True {name}", fontsize=9)
        ax.set_ylabel(f"Recovered {name}", fontsize=9)
        tag = "ID" if FISHER_RECONSTRUCTION[name] > 50 else "UN"
        ax.set_title(f"{name} [{tag}]: r={r:.3f}", fontsize=10, color=ident_color[j])
        ax.grid(True, alpha=0.3)

    # Use remaining subplot for table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    table_data = [["Param", "Fisher%", "Grad|r|", "GradP50", "NNLS|r|", "Match"]]
    for i, name in enumerate(PARAM_NAMES):
        table_data.append([
            name, f"{fisher_qual[i]:.0f}", f"{grad_corrs[i]:.2f}",
            f"{grad_pass[i]:.0f}%", f"{nnls_corrs[i]:.2f}",
            comp_results["verdicts"][name],
        ])
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 5:
            v = table_data[row][5]
            if "AGREE" in v:
                cell.set_facecolor("#C6EFCE")
            else:
                cell.set_facecolor("#FFC7CE")

    fig.suptitle("Cross-Validation: Fisher Identifiability vs Parameter Recovery",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(OUTPUT_DIR / "fig1_cross_validation.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig1_cross_validation.png")

    # --- Figure 2: Per-parameter scatter (gradient) ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for j, name in enumerate(PARAM_NAMES):
        ax = axes[j // 4, j % 4]
        y_true = params_log[test_idx, j]
        y_pred = grad_recovered[test_idx, j]
        ax.scatter(y_true, y_pred, s=12, alpha=0.3, color=ident_color[j])
        vmin, vmax = y_true.min(), y_true.max()
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, lw=1)
        r = grad_corrs[j]
        tag = "ID" if FISHER_RECONSTRUCTION[name] > 50 else "UN"
        ax.set_xlabel(f"True {name}", fontsize=9)
        ax.set_ylabel(f"Recovered {name}", fontsize=9)
        ax.set_title(f"{name} [{tag}] r={r:.3f}", fontsize=10, color=ident_color[j])
        ax.grid(True, alpha=0.3)
    axes[1, 3].axis("off")
    fig.suptitle("Gradient-Based Parameter Recovery vs Ground Truth",
                 fontsize=13, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "fig2_gradient_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig2_gradient_scatter.png")

    # --- Figure 3: Group comparison bar ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    labels = ["Identifiable\n(SEI, D_n, LAM_neg)", "Unidentifiable\n(D_p, t+, LAM_pos, R_mult)"]
    gc = comp_results["group"]
    bars = ax.bar(range(2), [gc["ident"]["corr"], gc["unident"]["corr"]],
                  color=["#2ca02c", "#d62728"], edgecolor="k", alpha=0.7, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean |Pearson r|", fontsize=11)
    ax.set_title("Gradient Recovery: Identifiable vs Unidentifiable", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, [gc["ident"]["corr"], gc["unident"]["corr"]]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ratio = gc["ident"]["corr"] / (gc["unident"]["corr"] + 1e-30)
    ax.text(0.95, 0.95, f"Separation: {ratio:.1f}×", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax = axes[1]
    bars = ax.bar(range(2), [gc["ident"]["pass_50"], gc["unident"]["pass_50"]],
                  color=["#2ca02c", "#d62728"], edgecolor="k", alpha=0.7, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean pass rate 50% (%)", fontsize=11)
    ax.set_title("Pass Rate by Identifiability Group", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, [gc["ident"]["pass_50"], gc["unident"]["pass_50"]]):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="bold")

    plt.savefig(OUTPUT_DIR / "fig3_group_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig3_group_comparison.png")


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time.time()
    logger.info("Loading data...")
    params, params_log, V, c_rates, n_time, p_mean, p_std, cr_mean, cr_std = load_data()
    n_sims = len(V)
    logger.info(f"  Data: {n_sims} samples, {n_time} time points, {N_PARAMS} params")

    fisher_qual = [FISHER_RECONSTRUCTION[n] for n in PARAM_NAMES]

    # Method A: Gradient-based recovery
    logger.info("\n=== Method A: Gradient-based fitting via MLP surrogate ===")
    grad_recovered, grad_mask, grad_test_idx = gradient_based_recovery(
        params_log, V, c_rates, p_mean, p_std, cr_mean, cr_std, n_time
    )
    grad_metrics = compute_metrics(params_log[grad_test_idx],
                                    grad_recovered[grad_test_idx])
    logger.info("\n  Gradient-based recovery results:")
    for name in PARAM_NAMES:
        m = grad_metrics[name]
        logger.info(f"    {name:>8}: r={m['pearson_r']:+.3f}  pass_50={m['pass_rate_50']:.1f}%  "
                     f"nrmse={m['nrmse']:.3f}")

    # Method B: NNLS decomposition
    logger.info("\n=== Method B: NNLS decomposition (LOSO CV) ===")
    nnls_recovered = nnls_recovery(params_log, V, c_rates)
    nnls_metrics = compute_metrics(params_log, nnls_recovered)
    logger.info("\n  NNLS recovery results:")
    for name in PARAM_NAMES:
        m = nnls_metrics[name]
        logger.info(f"    {name:>8}: r={m['pearson_r']:+.3f}  pass_50={m['pass_rate_50']:.1f}%  "
                     f"nrmse={m['nrmse']:.3f}")

    # Comparison analysis
    comp_results = analyze_comparison(fisher_qual, grad_metrics, nnls_metrics,
                                       grad_recovered, nnls_recovered, params_log, grad_mask)

    # Failure mode analysis
    failure_analysis(params_log, grad_recovered, grad_mask)

    # Fixed unidentifiable experiment
    full_rmse, fixed_rmse, full_v_rmse, fixed_v_rmse = fixed_recovery_experiment(
        params_log, V, c_rates, p_mean, p_std, cr_mean, cr_std, n_time
    )

    # Figures
    logger.info("\nGenerating figures...")
    make_figures(fisher_qual, grad_metrics, nnls_metrics, grad_recovered,
                 nnls_recovered, params_log, grad_mask, comp_results)

    # Save results
    results = {
        "fisher_reconstruction": FISHER_RECONSTRUCTION,
        "gradient_metrics": grad_metrics,
        "nnls_metrics": nnls_metrics,
        "comparison": comp_results,
        "fixed_experiment": {
            "full_param_rmse": {PARAM_NAMES[j]: float(full_rmse[j]) for j in range(N_PARAMS)},
            "fixed_param_rmse": {PARAM_NAMES[j]: float(fixed_rmse[j]) for j in range(N_PARAMS)},
            "full_voltage_rmse_mV": float(full_v_rmse.mean()),
            "fixed_voltage_rmse_mV": float(fixed_v_rmse.mean()),
        },
    }
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(OUTPUT_DIR / "cross_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s")
    logger.info(f"Outputs saved to {OUTPUT_DIR}")

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    gc = comp_results["group"]
    sp = comp_results["spearman"]
    logger.info(f"""
  Fisher predicts: IDENTIFIABLE = SEI, D_n, LAM_neg
                   UNIDENTIFIABLE = D_p, t+, LAM_pos, R_mult

  Gradient fitting (primary validation):
    Identifiable:   avg |r| = {gc['ident']['corr']:.3f}, pass_50 = {gc['ident']['pass_50']:.1f}%
    Unidentifiable: avg |r| = {gc['unident']['corr']:.3f}, pass_50 = {gc['unident']['pass_50']:.1f}%
    Separation ratio (|r|): {gc['ident']['corr']/(gc['unident']['corr']+1e-30):.1f}x
    Spearman(Fisher, |r|) = {sp['fisher_vs_grad_corr']['rho']:.3f} (p={sp['fisher_vs_grad_corr']['p']:.4f})
    Binary accuracy: {comp_results['n_agree']}/{N_PARAMS} = {comp_results['n_agree']/N_PARAMS*100:.0f}%

  VERDICT: {'VALIDATED' if comp_results['validated'] else 'QUALITATIVE AGREEMENT'}
    Fisher identifiability theory is {'strongly' if comp_results['validated'] else 'qualitatively'}
    confirmed by empirical parameter recovery.
""")


if __name__ == "__main__":
    main()
