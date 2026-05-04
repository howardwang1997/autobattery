#!/usr/bin/env python3
"""
Train Conditional Normalizing Flow for Bayesian degradation diagnosis.

Maps V(t) → p(θ | V) where θ are degradation parameters.
Uses identifiability-aware projection: only estimate identifiable subspace.
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import torch
import torch.nn as nn
import logging
import time
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
IDENTIFIABLE_IDX = [0, 3, 4]
N_TIME = 100


class DegradationDataset(Dataset):
    def __init__(self, h5_path, split="train", frac=0.8, seed=42):
        with h5py.File(h5_path, "r") as f:
            params = f["params"][:].astype(np.float32)
            V = f["V"][:].astype(np.float32)
            c_rates = f["c_rates"][:].astype(np.float32)

        params_log = params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

        p_mean = params_log.mean(0)
        p_std = params_log.std(0) + 1e-8

        self.params_norm = (params_log - p_mean) / p_std
        self.V = V
        self.c_rates = c_rates.reshape(-1, 1)
        self.p_mean = p_mean
        self.p_std = p_std

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(V))
        n_train = int(len(V) * frac)
        if split == "train":
            self.idx = sorted(idx[:n_train])
        else:
            self.idx = sorted(idx[n_train:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.tensor(self.V[j], dtype=torch.float32),
            torch.tensor(self.c_rates[j], dtype=torch.float32),
            torch.tensor(self.params_norm[j], dtype=torch.float32),
        )


class ConditionalFlow(nn.Module):
    def __init__(self, n_params=7, n_time=100, hidden=256, n_flows=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_time + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        try:
            import zuko
            self.flow = zuko.flows.NSF(
                features=n_params,
                context=hidden,
                transforms=n_flows,
                bins=8,
                hidden_features=[hidden, hidden],
            )
            self.use_zuko = True
        except ImportError:
            self.use_zuko = False
            self.mu_head = nn.Linear(hidden, n_params)
            self.logvar_head = nn.Linear(hidden, n_params)

    def forward(self, V, c_rate):
        ctx = self.encoder(torch.cat([V, c_rate], dim=-1))
        if self.use_zuko:
            dist = self.flow(ctx)
            return dist
        else:
            mu = self.mu_head(ctx)
            logvar = self.logvar_head(ctx)
            return mu, logvar

    def loss(self, V, c_rate, params):
        if self.use_zuko:
            dist = self.forward(V, c_rate)
            return -dist.log_prob(params).mean()
        else:
            mu, logvar = self.forward(V, c_rate)
            return (
                0.5 * (logvar.exp() * (params - mu) ** 2 + logvar)
            ).sum(-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-flows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DegradationDataset(args.data, split="train")
    val_ds = DegradationDataset(args.data, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ConditionalFlow(
        n_params=7, n_time=N_TIME, hidden=args.hidden, n_flows=args.n_flows
    ).to(device)
    n_params_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params_count:,}, device: {device}, use_zuko: {model.use_zuko}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        n = 0
        for V, cr, params in train_loader:
            V, cr, params = V.to(device), cr.to(device), params.to(device)
            loss = model.loss(V, cr, params)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(V)
            n += len(V)
        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for V, cr, params in val_loader:
                V, cr, params = V.to(device), cr.to(device), params.to(device)
                loss = model.loss(V, cr, params)
                val_loss += loss.item() * len(V)
                n += len(V)
        val_loss /= n

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                    "p_mean": train_ds.p_mean,
                    "p_std": train_ds.p_std,
                    "use_zuko": model.use_zuko,
                },
                args.output,
            )

        if epoch % 20 == 0 or is_best:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{args.epochs}: train={train_loss:.4f} val={val_loss:.4f} "
                f"best={best_loss:.4f} {elapsed:.0f}s{' *' if is_best else ''}"
            )

    logger.info(f"Done. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
