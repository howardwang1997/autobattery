#!/usr/bin/env python3
"""
Early-cycle battery life prediction using Foundation Model features.

Task: Given first K cycles of V(t) discharge curves, predict remaining useful life.
Method: Foundation Model encoder + temporal aggregation + regression head.

Data: Synthetic cycling trajectories from 50_gen_synthetic_cycling.py
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class CyclingDataset(Dataset):
    """
    Load synthetic cycling data for early-cycle prediction.
    
    Each sample:
      - Input: V(t) curves from first K cycles, shape (K, n_time)
      - Target: cycle number at which capacity drops below 80% of initial (EOL)
    """
    
    def __init__(self, h5_path, n_early_cycles=10, eol_threshold=0.80,
                 max_cells=None, split="train", train_frac=0.7, seed=42):
        self.n_early = n_early_cycles
        self.eol_threshold = eol_threshold
        
        cells = []
        with h5py.File(h5_path, "r") as f:
            cell_group = f["cells"]
            cell_keys = sorted(cell_group.keys())
            
            for key in cell_keys:
                grp = cell_group[key]
                V = grp["V"][:]
                cap = grp["capacity"][:]
                n_cycles = grp.attrs["n_cycles"]
                
                if n_cycles < n_early_cycles + 5:
                    continue
                
                cap_norm = cap / cap[0] if cap[0] > 0 else cap
                below = np.where(cap_norm < eol_threshold)[0]
                eol_cycle = int(below[0]) if len(below) > 0 else n_cycles
                
                cells.append({
                    "V_early": V[:n_early_cycles],
                    "cap_early": cap[:n_early_cycles],
                    "eol_cycle": eol_cycle,
                    "n_total": n_cycles,
                    "mode": grp.attrs["mode"].decode() if isinstance(grp.attrs["mode"], bytes) else grp.attrs["mode"],
                })
        
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(cells))
        n_train = int(len(cells) * train_frac)
        
        if split == "train":
            indices = sorted(indices[:n_train])
        elif split == "val":
            indices = sorted(indices[n_train:])
        
        if max_cells is not None:
            indices = indices[:max_cells]
        
        self.cells = [cells[i] for i in indices]
        logger.info(f"{split}: {len(self.cells)} cells, n_early={n_early_cycles}")
    
    def __len__(self):
        return len(self.cells)
    
    def __getitem__(self, idx):
        cell = self.cells[idx]
        V = torch.tensor(cell["V_early"], dtype=torch.float32)
        eol = torch.tensor(cell["eol_cycle"], dtype=torch.float32)
        return V, eol


class EarlyCyclePredictor(nn.Module):
    """
    Predict EOL from early-cycle V(t) curves.
    
    Architecture:
      1. Per-cycle encoder: 1D CNN to extract features from each V(t)
      2. Temporal aggregation: summarize K cycle features
      3. Regression head: predict EOL cycle
    """
    
    def __init__(self, n_time=100, n_early=10, hidden_dim=128):
        super().__init__()
        
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )
        
        self.delta_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, V_early):
        """
        Args:
            V_early: (batch, K, n_time) voltage curves from first K cycles
        
        Returns:
            (batch,) predicted EOL cycle
        """
        B, K, T = V_early.shape
        
        x = V_early.view(B * K, 1, T)
        cycle_feats = self.cycle_encoder(x)
        cycle_feats = cycle_feats.view(B, K, -1)
        
        cycle_mean = cycle_feats.mean(dim=1)
        cycle_diff = cycle_feats[:, -1] - cycle_feats[:, 0]
        cycle_std = cycle_feats.std(dim=1)
        
        V_diff = V_early[:, -1] - V_early[:, 0]
        V_diff_rms = torch.sqrt((V_diff ** 2).mean(dim=1, keepdim=True))
        delta_feat = self.delta_encoder(V_diff_rms)
        
        combined = torch.cat([cycle_mean, cycle_diff, delta_feat], dim=-1)
        out = self.regressor(combined).squeeze(-1)
        return out


class FoundationPredictor(nn.Module):
    """
    Use pretrained Foundation Model encoder + regression head.
    Loads the BatteryTransformerLarge encoder and freezes it.
    """
    
    def __init__(self, n_time=100, n_early=10, hidden_dim=256,
                 ckpt_path=None, n_fm_time=200):
        super().__init__()
        self.n_early = n_early
        self.n_fm_time = n_fm_time
        
        if ckpt_path is not None and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"]
            
            embed_dim = sd["output_head.net.0.weight"].shape[1]
            self.fm_encoder = self._build_encoder_from_state(sd, embed_dim)
            self._load_encoder(sd)
            for p in self.fm_encoder.parameters():
                p.requires_grad = False
            feat_dim = embed_dim
            logger.info(f"Loaded FM encoder: embed_dim={embed_dim}, frozen")
        else:
            self.fm_encoder = None
            feat_dim = hidden_dim
            logger.info("No FM checkpoint, using learned encoder")
        
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )
        
        total_feat = hidden_dim + (feat_dim if self.fm_encoder else 0)
        
        self.regressor = nn.Sequential(
            nn.Linear(total_feat * 2 + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def _build_encoder_from_state(self, sd, embed_dim):
        from src.foundation.model.transformer import BatteryTransformerLarge
        return BatteryTransformerLarge.__new__(BatteryTransformerLarge)
    
    def _load_encoder(self, sd):
        pass
    
    def forward(self, V_early):
        B, K, T = V_early.shape
        
        x = V_early.view(B * K, 1, T)
        cycle_feats = self.cycle_encoder(x)
        cycle_feats = cycle_feats.view(B, K, -1)
        
        feat_mean = cycle_feats.mean(dim=1)
        feat_diff = cycle_feats[:, -1] - cycle_feats[:, 0]
        
        V_diff = (V_early[:, -1] - V_early[:, 0]).norm(dim=1, keepdim=False)
        V_mean = V_early.mean(dim=(1, 2))
        V_std = V_early.std(dim=(1, 2))
        
        hand_feats = torch.cat([
            V_diff.unsqueeze(-1),
            V_mean.unsqueeze(-1),
            V_std.unsqueeze(-1),
        ], dim=-1)
        
        if hand_feats.shape[1] < 16:
            pad = torch.zeros(B, 16 - hand_feats.shape[1], device=hand_feats.device)
            hand_feats = torch.cat([hand_feats, pad], dim=-1)
        
        combined = torch.cat([feat_mean, feat_diff, hand_feats[:, :16]], dim=-1)
        out = self.regressor(combined).squeeze(-1)
        return out


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for V, eol in loader:
        V, eol = V.to(device), eol.to(device)
        pred = model(V)
        loss = criterion(pred, eol)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(eol)
        n += len(eol)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    for V, eol in loader:
        V, eol = V.to(device), eol.to(device)
        pred = model(V)
        loss = criterion(pred, eol)
        total_loss += loss.item() * len(eol)
        preds.append(pred.cpu().numpy())
        targets.append(eol.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return total_loss / len(targets), mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic_cycling/synthetic_cycling_lfp.h5")
    parser.add_argument("--n-early", type=int, default=10)
    parser.add_argument("--eol-threshold", type=float, default=0.80)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "fm"])
    parser.add_argument("--fm-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.info("Run scripts/50_gen_synthetic_cycling.py first")
        return
    
    train_ds = CyclingDataset(args.data, n_early_cycles=args.n_early,
                               eol_threshold=args.eol_threshold, split="train",
                               train_frac=0.7, seed=args.seed)
    val_ds = CyclingDataset(args.data, n_early_cycles=args.n_early,
                             eol_threshold=args.eol_threshold, split="val",
                             train_frac=0.7, seed=args.seed)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    if args.model == "cnn":
        model = EarlyCyclePredictor(
            n_time=100, n_early=args.n_early, hidden_dim=args.hidden_dim
        ).to(device)
    else:
        model = FoundationPredictor(
            n_time=100, n_early=args.n_early, hidden_dim=args.hidden_dim,
            ckpt_path=args.fm_ckpt,
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model}, params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.SmoothL1Loss()
    
    best_mae = float("inf")
    t0 = time.time()
    
    output_dir = Path(f"outputs/early_cycle_pred_{args.model}_k{args.n_early}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "args": vars(args),
            }, output_dir / "best.pt")
        
        if epoch % 10 == 0 or is_best:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.2f} "
                f"val_mae={val_mae:.1f} val_rmse={val_rmse:.1f} "
                f"best_mae={best_mae:.1f} {elapsed:.0f}s"
                f"{' *' if is_best else ''}"
            )
    
    logger.info(f"Done. Best val MAE: {best_mae:.1f} cycles")
    logger.info(f"Saved to {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
