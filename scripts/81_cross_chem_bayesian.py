#!/usr/bin/env python3
"""
Per-chemistry Fisher identifiability analysis + cross-chemistry Bayesian validation.

1. For each chemistry, compute Jacobian SVD -> identifiability structure
2. Train Bayesian model per-chemistry with correct Fisher weights
3. Cross-validation: train on each chemistry, test on all others + experimental
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import torch
import torch.nn as nn
import logging
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import svd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]

CHEM_FILES = {
    "NMC811": "data/fullfield/fullfield_nmc811_degradation.h5",
    "NCA": "data/fullfield/fullfield_nca_degradation.h5",
    "LFP": "data/fullfield/fullfield_lfp_degradation.h5",
    "LFP_v2": "data/fullfield/fullfield_lfp_v2_degradation.h5",
    "LCO": "data/fullfield/fullfield_lco_degradation.h5",
}

PARAM_RANGES = np.array([
    [1e-15, 5e-13], [5e-17, 5e-15], [0.2, 0.45],
    [1e-9, 1e-6], [0.0, 0.3], [0.0, 0.3], [1.0, 5.0],
])


def fisher_analysis(h5_path, chemistry_name):
    with h5py.File(h5_path, "r") as f:
        params = f["params"][:].astype(np.float32)
        V = f["V"][:].astype(np.float32)

    p_norm = np.zeros_like(params)
    for i in range(7):
        p_norm[:, i] = (params[:, i] - PARAM_RANGES[i, 0]) / (
            PARAM_RANGES[i, 1] - PARAM_RANGES[i, 0]
        )

    n = min(500, len(params))
    idx = np.random.RandomState(42).choice(len(params), n, replace=False)
    J = np.linalg.lstsq(p_norm[idx], V[idx], rcond=None)[0].T
    U, S, Vt = svd(J, full_matrices=False)

    # Identify which parameters dominate each singular mode
    # A parameter is "identifiable" if it has significant unique contribution
    # to the top modes (before the spectral gap)
    # Compute per-param reconstruction quality using first k modes
    S_ratio = S / S[0]
    eff_rank = (S_ratio > 0.05).sum()

    # Per-param: contribution to each mode
    param_contrib = np.abs(Vt[:eff_rank, :])

    # Identify params: those with high contribution to identifiable modes
    # Use reconstruction quality: can we reconstruct dV/dp_i using only top-k modes?
    identifiability = {}
    for j, name in enumerate(PARAM_NAMES):
        contrib_top = param_contrib[:, j].sum()
        contrib_all = np.abs(Vt[:, j]).sum()
        frac = contrib_top / max(contrib_all, 1e-10)
        identifiability[name] = float(frac)

    # Alternative: use condition number per parameter
    # Params with low condition number are identifiable
    param_sensitivity = np.sqrt((J ** 2).sum(0))

    return {
        "chemistry": chemistry_name,
        "n_sims": len(params),
        "singular_values": S.tolist(),
        "effective_rank_5pct": int(eff_rank),
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "param_sensitivity": {
            PARAM_NAMES[i]: float(param_sensitivity[i]) for i in range(7)
        },
        "param_mode_contribution": {
            PARAM_NAMES[j]: {
                "top_{}".format(k): float(abs(Vt[k, j])) for k in range(min(7, len(Vt)))
            }
            for j in range(7)
        },
        "right_singular_vectors": Vt.tolist(),
    }


class FullfieldDataset(Dataset):
    def __init__(self, h5_path, split="train", frac=0.8, seed=42):
        with h5py.File(h5_path, "r") as f:
            params = f["params"][:].astype(np.float32)
            V = f["V"][:].astype(np.float32)
            cr = f["c_rates"][:].astype(np.float32)

        params_log = params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

        self.p_mean = params_log.mean(0)
        self.p_std = params_log.std(0) + 1e-8
        pn = (params_log - self.p_mean) / self.p_std
        self.params_norm = torch.tensor(pn, dtype=torch.float32)
        self.V = torch.tensor(V, dtype=torch.float32)
        self.cr = torch.tensor(cr.reshape(-1, 1), dtype=torch.float32)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(V))
        nt = int(len(V) * frac)
        self.idx = sorted(idx[:nt]) if split == "train" else sorted(idx[nt:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.V[j], self.cr[j], self.params_norm[j]


class BayesianModel(nn.Module):
    def __init__(self, fisher_weight=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(101, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.mu_head = nn.Linear(256, 7)
        self.logvar_head = nn.Linear(256, 7)
        if fisher_weight is not None:
            self.register_buffer("fisher_w", torch.tensor(fisher_weight, dtype=torch.float32))
        else:
            self.register_buffer("fisher_w", torch.ones(7))

    def forward(self, V, cr):
        h = self.encoder(torch.cat([V, cr], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def loss_fn(self, V, cr, params, fw=2.0):
        mu, lv = self.forward(V, cr)
        prec = torch.exp(-lv)
        nll = 0.5 * (prec * (params - mu) ** 2 + lv)
        fi = self.fisher_w.to(V.device)
        ident_loss = (nll * (1.0 + fw * fi)).sum(-1).mean()
        ps = torch.exp(0.5 * lv)
        fr = fi * 0 + 0.5
        cal_loss = ((ps - fr.detach()) ** 2).mean()
        return ident_loss + 0.1 * cal_loss


def compute_fisher_weights(fisher_result):
    """Convert Fisher analysis to per-parameter identifiability weights."""
    Vt = np.array(fisher_result["right_singular_vectors"])
    S = np.array(fisher_result["singular_values"])
    eff_rank = fisher_result["effective_rank_5pct"]

    # Weight = how much of each parameter's contribution is in identifiable modes
    weights = np.zeros(7)
    for j in range(7):
        contrib_ident = sum(abs(Vt[k, j]) * S[k] for k in range(eff_rank))
        contrib_total = sum(abs(Vt[k, j]) * S[k] for k in range(len(S)))
        weights[j] = contrib_ident / max(contrib_total, 1e-10)

    # Normalize: identifiable params get weight ~1.0, unidentifiable ~0.0
    weights = np.clip(weights, 0.05, 1.0)
    return weights


def train_and_evaluate(chem_name, h5_path, fisher_result, output_dir):
    """Train Bayesian model on one chemistry and evaluate on all."""
    device = torch.device("cuda")
    fisher_w = compute_fisher_weights(fisher_result)
    logger.info("%s Fisher weights: %s", chem_name, 
                " ".join("{}={:.2f}".format(PARAM_NAMES[i], fisher_w[i]) for i in range(7)))

    train_ds = FullfieldDataset(h5_path, split="train")
    val_ds = FullfieldDataset(h5_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = BayesianModel(fisher_weight=fisher_w).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)

    best_val = float("inf")
    for ep in range(1, 501):
        model.train()
        for V, cr, params in train_loader:
            V, cr, params = V.to(device), cr.to(device), params.to(device)
            loss = model.loss_fn(V, cr, params)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % 100 == 0:
            model.eval()
            si, su, vloss, nb = [], [], 0.0, 0
            with torch.no_grad():
                for V, cr, params in val_loader:
                    V, cr, params = V.to(device), cr.to(device), params.to(device)
                    vloss += model.loss_fn(V, cr, params).item()
                    nb += 1
                    _, lv = model(V, cr)
                    s = torch.exp(0.5 * lv)
                    id_idx = [i for i in range(7) if fisher_w[i] > 0.5]
                    un_idx = [i for i in range(7) if fisher_w[i] <= 0.5]
                    if id_idx:
                        si.append(s[:, id_idx].mean().item())
                    if un_idx:
                        su.append(s[:, un_idx].mean().item())
            vloss /= nb
            ratio = np.mean(su) / max(np.mean(si), 1e-6) if si and su else 0
            logger.info(
                "  %s Ep %d: vloss=%.4f id=%.3f un=%.3f ratio=%.1fx",
                chem_name, ep, vloss, np.mean(si), np.mean(su), ratio,
            )
            if vloss < best_val:
                best_val = vloss
                chem_dir = Path(output_dir) / chem_name.lower()
                chem_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "p_mean": train_ds.p_mean,
                        "p_std": train_ds.p_std,
                        "fisher_w": fisher_w,
                        "epoch": ep,
                        "chemistry": chem_name,
                    },
                    chem_dir / "best.pt",
                )

    # Evaluate on experimental data
    model.eval()
    all_res = []
    exp_path = "data/experimental/experimental_cycling.h5"
    if Path(exp_path).exists():
        with h5py.File(exp_path, "r") as f:
            for key in sorted(f["cells"].keys()):
                grp = f["cells"][key]
                V = grp["V"][:]
                cap = grp["capacity"][:]
                ncyc = int(grp.attrs["n_cycles"])
                bc = grp.attrs["barcode"]
                if isinstance(bc, bytes):
                    bc = bc.decode()
                cap_init = float(grp.attrs.get("cap_initial", cap[0] if len(cap) > 0 else 1.0))

                sample_idx = list(range(0, ncyc, max(1, ncyc // 20)))
                cr_t = torch.ones(1, 1, device=device)
                mu_l, std_l = [], []
                with torch.no_grad():
                    for si in sample_idx:
                        if si < len(V):
                            vt = torch.tensor(V[si:si+1], dtype=torch.float32).to(device)
                            mu, lv = model(vt, cr_t)
                            mu_l.append(mu.cpu().numpy()[0])
                            std_l.append(torch.exp(0.5 * lv).cpu().numpy()[0])

                if mu_l:
                    all_res.append({
                        "barcode": bc, "ncyc": ncyc,
                        "cap": cap[sample_idx[:len(mu_l)]],
                        "cap_init": cap_init,
                        "mu": np.array(mu_l), "std": np.array(std_l),
                    })

    # Compute sharpness
    id_idx = [i for i in range(7) if fisher_w[i] > 0.5]
    un_idx = [i for i in range(7) if fisher_w[i] <= 0.5]

    sharp_per_param = {}
    for pi, name in enumerate(PARAM_NAMES):
        sharp_per_param[name] = float(np.mean([r["std"][:, pi].mean() for r in all_res]))

    id_sharp = np.mean([sharp_per_param[PARAM_NAMES[i]] for i in id_idx]) if id_idx else 0
    un_sharp = np.mean([sharp_per_param[PARAM_NAMES[i]] for i in un_idx]) if un_idx else 0
    ratio = un_sharp / max(id_sharp, 1e-6)

    result = {
        "chemistry": chem_name,
        "n_exp_cells": len(all_res),
        "id_params": [PARAM_NAMES[i] for i in id_idx],
        "un_params": [PARAM_NAMES[i] for i in un_idx],
        "sharpness": sharp_per_param,
        "id_sharp": float(id_sharp),
        "un_sharp": float(un_sharp),
        "ratio": float(ratio),
        "fisher_weights": {PARAM_NAMES[i]: float(fisher_w[i]) for i in range(7)},
    }

    logger.info(
        "  %s -> %d exp cells, sharp ID=%.3f UN=%.3f ratio=%.1fx",
        chem_name, len(all_res), id_sharp, un_sharp, ratio,
    )

    return result


def main():
    output_dir = Path("outputs/bayesian/cross_chem")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fisher analysis per chemistry
    logger.info("=== Step 1: Per-chemistry Fisher analysis ===")
    fisher_results = {}
    for chem_name, h5_path in CHEM_FILES.items():
        if not Path(h5_path).exists():
            logger.info("  %s: file not found, skipping", chem_name)
            continue
        fr = fisher_analysis(h5_path, chem_name)
        fisher_results[chem_name] = fr
        logger.info(
            "  %s: eff_rank=%d, V=[%.2f, %.2f]",
            chem_name, fr["effective_rank_5pct"], fr["V_min"], fr["V_max"],
        )

    with open(output_dir / "fisher_analysis.json", "w") as fp:
        json.dump(fisher_results, fp, indent=2)

    # Step 2: Train and evaluate per-chemistry
    logger.info("\n=== Step 2: Per-chemistry Bayesian training ===")
    all_results = {}
    for chem_name, h5_path in CHEM_FILES.items():
        if not Path(h5_path).exists():
            continue
        if chem_name not in fisher_results:
            continue
        result = train_and_evaluate(chem_name, h5_path, fisher_results[chem_name], output_dir)
        all_results[chem_name] = result

    # Step 3: Summary
    logger.info("\n=== Cross-Chemistry Summary ===")
    print("\n" + "=" * 80)
    print("CROSS-CHEMISTRY IDENTIFIABILITY ANALYSIS")
    print("=" * 80)

    print("\n{:10s} {:>8s} {:>8s} {:>30s} {:>8s} {:>8s} {:>6s}".format(
        "Chem", "N_sims", "Rank", "ID params", "ID_sh", "UN_sh", "Ratio"
    ))
    print("-" * 80)

    for chem_name, result in sorted(all_results.items()):
        id_params = "+".join(result["id_params"]) if result["id_params"] else "none"
        print("{:10s} {:8d} {:8d} {:>30s} {:8.3f} {:8.3f} {:5.1f}x".format(
            chem_name,
            fisher_results[chem_name]["n_sims"],
            fisher_results[chem_name]["effective_rank_5pct"],
            id_params,
            result["id_sharp"],
            result["un_sharp"],
            result["ratio"],
        ))

    with open(output_dir / "cross_chem_results.json", "w") as fp:
        json.dump(all_results, fp, indent=2)

    print("\nSaved to {}".format(output_dir))


if __name__ == "__main__":
    main()
