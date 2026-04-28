"""Pretrain Battery Foundation Model on multi-chemistry data.

Usage (single GPU):
    python scripts/41_pretrain_foundation.py

Usage (2-GPU DDP):
    torchrun --nproc_per_node=2 scripts/41_pretrain_foundation.py
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import time
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_ddp(ddp):
    if ddp:
        dist.destroy_process_group()


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, step):
        if step < self.warmup_steps:
            lr = self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def train(args):
    rank, world_size, local_rank, ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Device: {device}, DDP: {ddp}, World size: {world_size}")
        logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")

    import sys
    sys.path.insert(0, ".")
    from src.foundation.data.dataset import MultiChemDataset
    from src.foundation.model.transformer import (
        BatteryTransformer, BatteryTransformerSmall, BatteryTransformerLarge, count_parameters,
    )

    model_cls = {
        "small": BatteryTransformerSmall,
        "base": BatteryTransformer,
        "large": BatteryTransformerLarge,
    }[args.model_size]
    model = model_cls(n_chemistries=6, n_params=8, n_time=200)
    n_params = count_parameters(model)

    if rank == 0:
        logger.info(f"Model: {args.model_size}, {n_params/1e6:.1f}M params")

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    train_ds = MultiChemDataset(args.data_path, split="train")
    val_ds = MultiChemDataset(args.data_path, split="val")

    if ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                   num_workers=0, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    if rank == 0:
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, "
                     f"Batches/epoch: {len(train_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=min(1000, total_steps // 5),
                                        total_steps=total_steps)

    loss_fn = nn.SmoothL1Loss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            V = batch["V"].to(device)
            chem = batch["chem_id"].to(device)
            params = batch["params"].to(device)
            cond = batch["conditions"].to(device)

            c_rate = cond[:, 0:1]
            cond_norm = torch.cat([c_rate / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1)

            V_pred = model.module if ddp else model
            V_pred = (model.module if ddp else model)(chem, params, cond_norm, targets=V)

            loss = loss_fn(V_pred, V)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            global_step = epoch * len(train_loader) + n_batches
            scheduler.step(global_step)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0

        if rank == 0:
            val_loss = 0.0
            n_val = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    V = batch["V"].to(device)
                    chem = batch["chem_id"].to(device)
                    params = batch["params"].to(device)
                    cond = batch["conditions"].to(device)
                    cond_norm = torch.cat([cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1)

                    V_pred = (model.module if ddp else model)(chem, params, cond_norm, targets=V)
                    val_loss += loss_fn(V_pred, V).item()
                    n_val += 1

            val_loss /= max(n_val, 1)

            v_range = 4.2 - 2.5
            train_rmse_mV = np.sqrt(avg_loss) * v_range * 1000
            val_rmse_mV = np.sqrt(val_loss) * v_range * 1000

            logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                        f"train_loss={avg_loss:.5f} ({train_rmse_mV:.1f}mV) "
                        f"val_loss={val_loss:.5f} ({val_rmse_mV:.1f}mV) "
                        f"time={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.2e}")

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "train_rmse_mV": train_rmse_mV,
                "val_rmse_mV": val_rmse_mV,
                "time": elapsed,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rmse_mV": val_rmse_mV,
                    "n_params": n_params,
                    "model_size": args.model_size,
                }, output_dir / "bfom_best.pt")
                logger.info(f"  Saved best model (val_rmse={val_rmse_mV:.1f}mV)")

            if (epoch + 1) % args.save_every == 0:
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "val_loss": val_loss,
                }, output_dir / f"bfom_epoch{epoch+1}.pt")

    if rank == 0:
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training complete. Best val loss: {best_val_loss:.5f}")
        logger.info(f"Saved to {output_dir}")

    cleanup_ddp(ddp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/foundation/multichem_train.h5")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints_bfom")
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()
    train(args)
