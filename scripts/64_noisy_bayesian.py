#!/usr/bin/env python3
"""
Train Bayesian diagnosis models on noisy data.
Test identifiability-aware diagnosis under realistic noise.
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

LOG_PARAMS = [0, 1, 3]
PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
IDENTIFIABLE = ["D_n", "SEI", "LAM_neg"]


class NoisyDegDataset(Dataset):
    def __init__(self, h5_path, noise_mv=5.0, temp_std=2.0, split="train", frac=0.8, seed=42):
        with h5py.File(h5_path, "r") as f:
            params = f["params"][:].astype(np.float32)
            V = f["V"][:].astype(np.float32)
            c_rates = f["c_rates"][:].astype(np.float32)

        rng = np.random.RandomState(seed if split == "train" else seed + 1)
        V_noisy = V + rng.normal(0, noise_mv / 1000.0, V.shape).astype(np.float32)
        V_noisy += rng.normal(0, temp_std * 0.001, V.shape).astype(np.float32)

        params_log = params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

        self.p_mean = params_log.mean(0)
        self.p_std = params_log.std(0) + 1e-8
        self.params_norm = (params_log - self.p_mean) / self.p_std
        self.V = V_noisy
        self.c_rates = c_rates.reshape(-1, 1)

        idx = rng.permutation(len(V))
        n_train = int(len(V) * frac)
        self.idx = sorted(idx[:n_train]) if split == "train" else sorted(idx[n_train:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.tensor(self.V[j], dtype=torch.float32),
            torch.tensor(self.c_rates[j], dtype=torch.float32),
            torch.tensor(self.params_norm[j], dtype=torch.float32),
        )


class BayesianNN(nn.Module):
    def __init__(self, n_time=100, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_time + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 7),
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 7),
        )

    def forward(self, V, c_rate):
        h = self.shared(torch.cat([V, c_rate], dim=-1))
        return self.mu(h), self.logvar(h)

    def loss(self, V, c_rate, params, identifiable_weight=2.0):
        mu, logvar = self.forward(V, c_rate)
        precision = torch.exp(-logvar)
        nll = 0.5 * (precision * (params - mu) ** 2 + logvar)

        ident_mask = torch.zeros(7, device=V.device)
        for name in IDENTIFIABLE:
            idx = PARAM_NAMES.index(name)
            ident_mask[idx] = identifiable_weight - 1.0
        ident_mask += 1.0

        return (nll * ident_mask).sum(-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/fullfield/fullfield_lfp_degradation.h5")
    parser.add_argument("--output", type=str, default="outputs/bayesian/noisy/")
    parser.add_argument("--noise-levels", type=str, default="1,5,10,50")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    results = {}

    for noise_mv in noise_levels:
        logger.info(f"\n=== Training with noise = {noise_mv} mV ===")

        train_ds = NoisyDegDataset(args.data, noise_mv=noise_mv, split="train")
        val_ds = NoisyDegDataset(args.data, noise_mv=noise_mv, split="val")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = BayesianNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            for V, cr, params in train_loader:
                V, cr, params = V.to(device), cr.to(device), params.to(device)
                loss = model.loss(V, cr, params)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            if epoch % 50 == 0:
                model.eval()
                preds_mu, preds_std, targets = [], [], []
                with torch.no_grad():
                    for V, cr, params in val_loader:
                        V, cr = V.to(device), cr.to(device)
                        mu, logvar = model(V, cr)
                        preds_mu.append(mu.cpu().numpy())
                        preds_std.append(torch.exp(0.5 * logvar).cpu().numpy())
                        targets.append(params.numpy())

                preds_mu = np.concatenate(preds_mu)
                preds_std = np.concatenate(preds_std)
                targets = np.concatenate(targets)

                errors = np.abs(preds_mu - targets) * train_ds.p_std + train_ds.p_mean
                errors = errors[:, [0, 3, 4]]
                ident_mae = errors.mean()

                logger.info(f"  Noise {noise_mv}mV, Epoch {epoch}: ident MAE={ident_mae:.4f}")

        torch.save({
            "model": model.state_dict(),
            "p_mean": train_ds.p_mean,
            "p_std": train_ds.p_std,
            "noise_mv": noise_mv,
        }, output_dir / f"noisy_{noise_mv}mv.pt")

        results[noise_mv] = {"ident_mae": float(ident_mae)}

    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
