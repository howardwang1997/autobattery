#!/usr/bin/env python3
"""
Train ensemble of neural networks for Bayesian degradation diagnosis baseline.

Each member: V(t) + c_rate → params (deterministic).
Uncertainty from ensemble spread.
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

LOG_PARAMS = [0, 1, 3]


class DegradationDataset(Dataset):
    def __init__(self, h5_path, split="train", frac=0.8, seed=42):
        with h5py.File(h5_path, "r") as f:
            params = f["params"][:].astype(np.float32)
            V = f["V"][:].astype(np.float32)
            c_rates = f["c_rates"][:].astype(np.float32)

        params_log = params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

        self.p_mean = params_log.mean(0)
        self.p_std = params_log.std(0) + 1e-8
        self.params_norm = (params_log - self.p_mean) / self.p_std
        self.V = V
        self.c_rates = c_rates.reshape(-1, 1)

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


class EnsembleMember(nn.Module):
    def __init__(self, n_time=100, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_time + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 7),
        )

    def forward(self, V, c_rate):
        return self.net(torch.cat([V, c_rate], dim=-1))


def train_member(member_id, args, train_ds, val_ds, device):
    torch.manual_seed(args.seed + member_id)
    np.random.seed(args.seed + member_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = EnsembleMember().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for V, cr, params in train_loader:
            V, cr, params = V.to(device), cr.to(device), params.to(device)
            pred = model(V, cr)
            loss = criterion(pred, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            model.eval()
            val_loss = 0
            n = 0
            with torch.no_grad():
                for V, cr, params in val_loader:
                    V, cr, params = V.to(device), cr.to(device), params.to(device)
                    pred = model(V, cr)
                    val_loss += criterion(pred, params).item() * len(V)
                    n += len(V)
            val_loss /= n
            if val_loss < best_val:
                best_val = val_loss

    return model.state_dict(), best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-members", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DegradationDataset(args.data, split="train")
    val_ds = DegradationDataset(args.data, split="val")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {args.n_members} ensemble members on {device}")

    t0 = time.time()
    for m_id in range(args.n_members):
        state_dict, val_loss = train_member(m_id, args, train_ds, val_ds, device)
        torch.save(
            {"member_id": m_id, "state_dict": state_dict, "val_loss": val_loss},
            output_dir / f"member_{m_id:03d}.pt",
        )
        if (m_id + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  {m_id+1}/{args.n_members} members done, val_loss={val_loss:.4f} ({elapsed:.0f}s)"
            )

    np.save(output_dir / "p_mean.npy", train_ds.p_mean)
    np.save(output_dir / "p_std.npy", train_ds.p_std)
    logger.info(f"Ensemble saved to {output_dir}")


if __name__ == "__main__":
    main()
