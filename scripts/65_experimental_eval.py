#!/usr/bin/env python3
"""
Evaluate early-cycle life prediction on experimental data.

Compare: CNN trained on synthetic → tested on experimental
         CNN trained on experimental → tested on experimental
         Transfer: synthetic pretrain + experimental fine-tune
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

N_TIME = 100


class ExpCyclingDataset(Dataset):
    def __init__(self, h5_path, n_early=10, eol_threshold=0.80,
                 split="train", frac=0.7, seed=42, max_cells=None,
                 normalize_voltage=True):
        self.n_early = n_early
        self.normalize_voltage = normalize_voltage

        cells = []
        with h5py.File(h5_path, "r") as f:
            for key in sorted(f["cells"].keys()):
                grp = f["cells"][key]
                V = grp["V"][:]
                cap = grp["capacity"][:]
                ncyc = int(grp.attrs["n_cycles"])
                cap_init = float(grp.attrs["cap_initial"])

                if ncyc < n_early + 5 or cap_init <= 0:
                    continue
                if cap_init < 10:  # skip small cells for now
                    continue

                cap_norm = cap / cap[0] if cap[0] > 0 else cap
                below = np.where(cap_norm < eol_threshold)[0]
                eol = int(below[0]) if len(below) > 0 else ncyc

                if eol < n_early + 3:
                    continue

                cells.append({"V": V, "cap": cap, "eol": eol, "ncyc": ncyc})

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(cells))
        n_train = int(len(cells) * frac)
        if split == "train":
            idx = sorted(idx[:n_train])
        elif split == "val":
            idx = sorted(idx[n_train:])

        if max_cells:
            idx = idx[:max_cells]

        self.cells = [cells[i] for i in idx]
        logger.info(f"{split}: {len(self.cells)} experimental cells, n_early={n_early}")

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i):
        cell = self.cells[i]
        V = cell["V"][: self.n_early].copy()

        if self.normalize_voltage:
            vmin, vmax = V.min(), V.max()
            if vmax > vmin:
                V = (V - vmin) / (vmax - vmin)

        return (
            torch.tensor(V, dtype=torch.float32),
            torch.tensor(cell["eol"], dtype=torch.float32),
        )


class EarlyCycleCNN(nn.Module):
    def __init__(self, n_time=100, n_early=10, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(),
        )
        self.delta_enc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, V_early):
        B, K, T = V_early.shape
        x = V_early.view(B * K, 1, T)
        feats = self.encoder(x).view(B, K, -1)
        f_mean = feats.mean(dim=1)
        f_diff = feats[:, -1] - feats[:, 0]

        V_diff = V_early[:, -1] - V_early[:, 0]
        rms = torch.sqrt((V_diff ** 2).mean(dim=1, keepdim=True))
        d_feat = self.delta_enc(rms)

        combined = torch.cat([f_mean, d_feat], dim=-1)
        return self.regressor(combined).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    n = 0
    for V, eol in loader:
        V, eol = V.to(device), eol.to(device)
        pred = model(V)
        loss = criterion(pred, eol)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(eol)
        n += len(eol)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for V, eol in loader:
        V = V.to(device)
        pred = model(V)
        preds.append(pred.cpu().numpy())
        targets.append(eol.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-data", type=str, required=True)
    parser.add_argument("--exp-data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-early", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Experiment 1: Train on experimental only
    logger.info("=== Exp 1: Train on experimental data only ===")
    train_ds = ExpCyclingDataset(args.exp_data, n_early=args.n_early, split="train")
    val_ds = ExpCyclingDataset(args.exp_data, n_early=args.n_early, split="val")

    if len(train_ds) > 5:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = EarlyCycleCNN(n_time=N_TIME, n_early=args.n_early).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()

        best_mae = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, val_rmse = evaluate(model, val_loader, device)
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), output_dir / "exp_only.pt")
            if epoch % 20 == 0:
                logger.info(f"  Epoch {epoch}: mae={val_mae:.1f} best={best_mae:.1f}")

        results["exp_only"] = {"mae": best_mae, "n_train": len(train_ds), "n_val": len(val_ds)}
    else:
        logger.warning(f"Too few training cells: {len(train_ds)}")

    # Experiment 2: Multiple K values
    logger.info("=== Exp 2: Sweep early cycles K ===")
    for K in [3, 5, 10, 20]:
        train_ds_k = ExpCyclingDataset(args.exp_data, n_early=K, split="train")
        val_ds_k = ExpCyclingDataset(args.exp_data, n_early=K, split="val")
        if len(train_ds_k) < 5:
            continue

        train_loader_k = DataLoader(train_ds_k, batch_size=args.batch_size, shuffle=True)
        val_loader_k = DataLoader(val_ds_k, batch_size=args.batch_size, shuffle=False)

        model_k = EarlyCycleCNN(n_time=N_TIME, n_early=K).to(device)
        opt_k = torch.optim.AdamW(model_k.parameters(), lr=1e-3, weight_decay=1e-4)

        best_k = float("inf")
        for ep in range(1, args.epochs + 1):
            train_epoch(model_k, train_loader_k, opt_k, criterion, device)
            mae_k, _ = evaluate(model_k, val_loader_k, device)
            best_k = min(best_k, mae_k)

        results[f"exp_K{K}"] = {"mae": best_k}
        logger.info(f"  K={K}: MAE={best_k:.1f} cycles")

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj

    results = {k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v) for k, v in results.items()}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_dir / 'results.json'}")


import json

if __name__ == "__main__":
    main()
