#!/usr/bin/env python3
"""
Cross-chemistry Bayesian validation with UNIFORM Fisher weights.
Let the model learn chemistry-specific identifiability from data.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
IDENT_IDX_LFP = [0, 3, 4]

CHEM_FILES = {
    "NMC811": "data/fullfield/fullfield_nmc811_degradation.h5",
    "NCA": "data/fullfield/fullfield_nca_degradation.h5",
    "LFP": "data/fullfield/fullfield_lfp_degradation.h5",
    "LFP_v2": "data/fullfield/fullfield_lfp_v2_degradation.h5",
    "LCO": "data/fullfield/fullfield_lco_degradation.h5",
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
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(101, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.mu_head = nn.Linear(256, 7)
        self.logvar_head = nn.Linear(256, 7)

    def forward(self, V, cr):
        h = self.encoder(torch.cat([V, cr], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def loss_fn(self, V, cr, params):
        mu, lv = self.forward(V, cr)
        prec = torch.exp(-lv)
        nll = 0.5 * (prec * (params - mu) ** 2 + lv)
        return nll.sum(-1).mean()


def train_model(h5_path, chem_name):
    device = torch.device("cuda")
    train_ds = FullfieldDataset(h5_path, split="train")
    val_ds = FullfieldDataset(h5_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = BayesianModel().to(device)
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

        if ep == 500:
            model.eval()
            sharp = []
            with torch.no_grad():
                for V, cr, _ in val_loader:
                    V, cr = V.to(device), cr.to(device)
                    _, lv = model(V, cr)
                    sharp.append(torch.exp(0.5 * lv).cpu().numpy())
            sharp = np.concatenate(sharp, axis=0)
            avg_sharp = sharp.mean(0)

            vloss = 0.0
            nb = 0
            with torch.no_grad():
                for V, cr, params in val_loader:
                    V, cr, params = V.to(device), cr.to(device), params.to(device)
                    vloss += model.loss_fn(V, cr, params).item()
                    nb += 1
            vloss /= nb

            if vloss < best_val:
                best_val = vloss

    return model, train_ds, avg_sharp


def evaluate_experimental(model, p_mean, p_std):
    device = next(model.parameters()).device
    model.eval()
    all_res = []

    with h5py.File("data/experimental/experimental_cycling.h5", "r") as f:
        for key in sorted(f["cells"].keys()):
            grp = f["cells"][key]
            V = grp["V"][:]
            cap = grp["capacity"][:]
            ncyc = int(grp.attrs["n_cycles"])
            bc = grp.attrs["barcode"]
            if isinstance(bc, bytes):
                bc = bc.decode()
            cap_init = float(grp.attrs.get("cap_initial", 1.0))

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
                    "cap": cap[sample_idx[:len(mu_l)]],
                    "cap_init": cap_init,
                    "mu": np.array(mu_l),
                    "std": np.array(std_l),
                })

    sharp_per_param = {}
    for pi, name in enumerate(PARAM_NAMES):
        sharp_per_param[name] = float(np.mean([r["std"][:, pi].mean() for r in all_res]))

    return sharp_per_param, len(all_res)


def main():
    output_dir = Path("outputs/bayesian/cross_chem_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train on each chemistry
    logger.info("=== Training per-chemistry models ===")
    results = {}

    for chem_name, h5_path in CHEM_FILES.items():
        if not Path(h5_path).exists():
            logger.info("  %s: not found, skipping", chem_name)
            continue

        logger.info("  Training on %s ...", chem_name)
        model, train_ds, val_sharp = train_model(h5_path, chem_name)

        exp_sharp, n_exp = evaluate_experimental(
            model, train_ds.p_mean, train_ds.p_std
        )

        results[chem_name] = {
            "val_sharpness": {PARAM_NAMES[i]: float(val_sharp[i]) for i in range(7)},
            "exp_sharpness": exp_sharp,
            "n_exp_cells": n_exp,
        }

        # Save model
        chem_dir = output_dir / chem_name.lower()
        chem_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "p_mean": train_ds.p_mean,
                "p_std": train_ds.p_std,
                "chemistry": chem_name,
            },
            chem_dir / "best.pt",
        )

        logger.info(
            "  %s done: val_sharp range [%.3f, %.3f], exp_sharp range [%.3f, %.3f]",
            chem_name,
            min(val_sharp), max(val_sharp),
            min(exp_sharp.values()), max(exp_sharp.values()),
        )

    # Now train on ALL chemistries combined
    logger.info("\n=== Training on ALL chemistries combined ===")
    all_V, all_params, all_cr = [], [], []
    combined_p_mean = None
    combined_p_std = None

    for chem_name, h5_path in CHEM_FILES.items():
        if not Path(h5_path).exists():
            continue
        with h5py.File(h5_path, "r") as f:
            all_V.append(f["V"][:].astype(np.float32))
            all_params.append(f["params"][:].astype(np.float32))
            all_cr.append(f["c_rates"][:].astype(np.float32))

    all_V = np.concatenate(all_V)
    all_params = np.concatenate(all_params)
    all_cr = np.concatenate(all_cr)

    params_log = all_params.copy()
    for i in LOG_PARAMS:
        params_log[:, i] = np.log10(np.maximum(all_params[:, i], 1e-30))

    combined_p_mean = params_log.mean(0)
    combined_p_std = params_log.std(0) + 1e-8
    pn = (params_log - combined_p_mean) / combined_p_std

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(all_V))
    nt = int(len(all_V) * 0.8)
    train_idx = sorted(idx[:nt])
    val_idx = sorted(idx[nt:])

    device = torch.device("cuda")
    model = BayesianModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)

    V_t = torch.tensor(all_V, dtype=torch.float32)
    cr_t = torch.tensor(all_cr.reshape(-1, 1), dtype=torch.float32)
    pn_t = torch.tensor(pn, dtype=torch.float32)

    for ep in range(1, 501):
        model.train()
        batch_idx = torch.from_numpy(
            rng.choice(train_idx, 128, replace=False)
        )
        V_b = V_t[batch_idx].to(device)
        cr_b = cr_t[batch_idx].to(device)
        pn_b = pn_t[batch_idx].to(device)
        loss = model.loss_fn(V_b, cr_b, pn_b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    # Evaluate combined model
    model.eval()
    exp_sharp_combined, n_exp = evaluate_experimental(
        model, combined_p_mean, combined_p_std
    )
    results["COMBINED"] = {
        "exp_sharpness": exp_sharp_combined,
        "n_exp_cells": n_exp,
    }

    torch.save(
        {
            "model": model.state_dict(),
            "p_mean": combined_p_mean,
            "p_std": combined_p_std,
            "chemistry": "combined",
        },
        output_dir / "combined" / "best.pt",
    )

    # Print summary
    print("\n" + "=" * 90)
    print("CROSS-CHEMISTRY BAYESIAN ANALYSIS")
    print("=" * 90)

    print("\n--- Experimental Data Sharpness (lower = more identifiable) ---")
    hdr = "{:10s}".format("Trained on")
    for name in PARAM_NAMES:
        hdr += " {:>8s}".format(name[:5])
    print(hdr)
    print("-" * 90)

    for train_chem in list(CHEM_FILES.keys()) + ["COMBINED"]:
        if train_chem not in results:
            continue
        exp_sh = results[train_chem].get("exp_sharpness", {})
        row = "{:10s}".format(train_chem)
        for name in PARAM_NAMES:
            row += " {:8.3f}".format(exp_sh.get(name, 0))
        print(row)

    # Identify: which params are consistently sharp (identifiable) across ALL models?
    print("\n--- Parameter Sharpness Consistency ---")
    for name in PARAM_NAMES:
        sharps = [results[c]["exp_sharpness"].get(name, 0)
                  for c in results if "exp_sharpness" in results[c]]
        if sharps:
            print(
                "  {:10s}: min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
                    name, min(sharps), max(sharps), np.mean(sharps), np.std(sharps)
                )
            )

    with open(output_dir / "results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nSaved to {}".format(output_dir))


if __name__ == "__main__":
    main()
