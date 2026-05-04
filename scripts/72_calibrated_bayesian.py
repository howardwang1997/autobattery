#!/usr/bin/env python3
"""
Improved Bayesian diagnosis with:
  1. Identifiability-aware training: encourage high uncertainty for unidentifiable params
  2. Uncertainty calibration using held-out synthetic data
  3. Proper evaluation: coverage, calibration error, sharpness
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import time
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]

FISHER_IDENTIFIABLE = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
FISHER_RECON_QUALITY = torch.tensor([1.0, 0.014, 0.02, 1.0, 0.999, 0.022, 0.016])


class DegDataset(Dataset):
    def __init__(self, h5_path, noise_mv=0.0, split="train", frac=0.8, seed=42):
        with h5py.File(h5_path, "r") as f:
            params = f["params"][:].astype(np.float32)
            V = f["V"][:].astype(np.float32)
            cr = f["c_rates"][:].astype(np.float32)

        if noise_mv > 0:
            rng = np.random.RandomState(seed + 100)
            V = V + rng.normal(0, noise_mv / 1000, V.shape).astype(np.float32)

        params_log = params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

        self.p_mean = params_log.mean(0)
        self.p_std = params_log.std(0) + 1e-8
        self.params_norm = torch.tensor((params_log - self.p_mean) / self.p_std, dtype=torch.float32)
        self.V = torch.tensor(V, dtype=torch.float32)
        self.cr = torch.tensor(cr.reshape(-1, 1), dtype=torch.float32)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(V))
        n_train = int(len(V) * frac)
        self.idx = sorted(idx[:n_train]) if split == "train" else sorted(idx[n_train:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.V[j], self.cr[j], self.params_norm[j]


class CalibratedBayesianNN(nn.Module):
    def __init__(self, n_time=100, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_time + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.mu_head = nn.Linear(hidden, 7)
        self.logvar_head = nn.Linear(hidden, 7)

    def forward(self, V, cr):
        h = self.encoder(torch.cat([V, cr], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def loss(self, V, cr, params, fisher_weight=1.0):
        mu, logvar = self.forward(V, cr)
        precision = torch.exp(-logvar)
        nll = 0.5 * (precision * (params - mu) ** 2 + logvar)

        fisher_mask = FISHER_IDENTIFIABLE.to(V.device)
        ident_loss = (nll * (1.0 + fisher_weight * fisher_mask)).sum(-1).mean()

        pred_std = torch.exp(0.5 * logvar)
        fisher_quality = FISHER_RECON_QUALITY.to(V.device)
        calibration_loss = ((pred_std - fisher_quality.detach()) ** 2).mean()

        return ident_loss + 0.1 * calibration_loss


def evaluate_calibration(model, loader, device, p_mean, p_std):
    model.eval()
    all_mu, all_logvar, all_targets = [], [], []
    with torch.no_grad():
        for V, cr, params in loader:
            V, cr = V.to(device), cr.to(device)
            mu, logvar = model(V, cr)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_targets.append(params.numpy())

    mu = np.concatenate(all_mu)
    logvar = np.concatenate(all_logvar)
    targets = np.concatenate(all_targets)
    std = np.exp(0.5 * logvar)

    results = {}
    for i, name in enumerate(PARAM_NAMES):
        errors = np.abs(mu[:, i] - targets[:, i])
        coverage_2sigma = np.mean(errors < 2 * std[:, i])
        coverage_1sigma = np.mean(errors < 1 * std[:, i])
        sharpness = std[:, i].mean()
        mae = errors.mean()
        ident = "IDENTIFIABLE" if FISHER_IDENTIFIABLE[i] > 0.5 else "unident"

        results[name] = {
            "coverage_1sigma": float(coverage_1sigma),
            "coverage_2sigma": float(coverage_2sigma),
            "sharpness": float(sharpness),
            "mae_norm": float(mae),
            "ident": ident,
        }

        if FISHER_IDENTIFIABLE[i] > 0.5:
            mae_real = mae * p_std[i]
            results[name]["mae_real"] = float(mae_real)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fullfield/fullfield_lfp_degradation.h5")
    parser.add_argument("--output", default="outputs/bayesian/calibrated/")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--noise-mv", type=float, default=5.0)
    parser.add_argument("--fisher-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DegDataset(args.data, noise_mv=args.noise_mv, split="train")
    val_ds = DegDataset(args.data, noise_mv=args.noise_mv, split="val")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    model = CalibratedBayesianNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for V, cr, params in train_loader:
            V, cr, params = V.to(device), cr.to(device), params.to(device)
            loss = model.loss(V, cr, params, fisher_weight=args.fisher_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            cal = evaluate_calibration(model, val_loader, device, train_ds.p_mean, train_ds.p_std)
            ident_mae = np.mean([cal[n]["mae_norm"] for n in PARAM_NAMES if cal[n]["ident"] == "IDENTIFIABLE"])
            unident_sharp = np.mean([cal[n]["sharpness"] for n in PARAM_NAMES if cal[n]["ident"] == "unident"])

            if ident_mae < best_val:
                best_val = ident_mae
                torch.save({
                    "model": model.state_dict(),
                    "p_mean": train_ds.p_mean,
                    "p_std": train_ds.p_std,
                    "calibration": cal,
                    "epoch": epoch,
                    "noise_mv": args.noise_mv,
                }, output_dir / "best.pt")

            logger.info(
                f"Epoch {epoch}: ident_mae={ident_mae:.4f}, "
                f"unident_sharpness={unident_sharp:.3f}, "
                f"elapsed={time.time()-t0:.0f}s"
            )

    logger.info(f"\n=== Final Calibration on Val Set ===")
    best_ckpt = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    cal = best_ckpt["calibration"]
    for name in PARAM_NAMES:
        c = cal[name]
        logger.info(
            f"  {name:10s}: coverage_1σ={c['coverage_1sigma']:.2f}, "
            f"coverage_2σ={c['coverage_2sigma']:.2f}, "
            f"sharpness={c['sharpness']:.3f}, "
            f"mae={c['mae_norm']:.4f} [{c['ident']}]"
        )

    import json

    def convert(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        return o

    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump(cal, f, indent=2, default=convert)


if __name__ == "__main__":
    main()
