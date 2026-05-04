"""Train Physics-Structured Battery Foundation Model.

Usage (2-GPU DDP):
    torchrun --nproc_per_node=2 scripts/45_train_physics_fom.py

Single GPU:
    python scripts/45_train_physics_fom.py
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
import torch.nn.functional as F
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


class DecompositionLoss(nn.Module):
    """Combined loss with optional decomposition regularization.

    Components:
      1. Main reconstruction loss (SmoothL1 on V_pred vs V_target)
      2. Monotonicity loss: V should decrease over time (discharge)
      3. Overpotential sign loss: all eta should be >= 0
      4. OCV magnitude loss: OCV should be the dominant component
    """

    def __init__(self, lambda_mono=0.1, lambda_sign=0.05, lambda_ocv=0.01, lambda_residual=0.01):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_sign = lambda_sign
        self.lambda_ocv = lambda_ocv
        self.lambda_residual = lambda_residual
        self.recon_loss = nn.SmoothL1Loss()

    def forward(self, V_pred, V_target, components):
        loss = self.recon_loss(V_pred, V_target)
        breakdown = {"recon": loss.item()}

        if self.lambda_mono > 0:
            diffs = V_pred[:, 1:] - V_pred[:, :-1]
            mono_loss = torch.relu(diffs).pow(2).mean()
            loss = loss + self.lambda_mono * mono_loss
            breakdown["monotonicity"] = mono_loss.item()

        if self.lambda_sign > 0:
            sign_loss = torch.tensor(0.0, device=V_pred.device)
            for key in ["eta_activation", "eta_ohmic", "eta_concentration"]:
                if key in components:
                    sign_loss = sign_loss + torch.relu(-components[key]).pow(2).mean()
            loss = loss + self.lambda_sign * sign_loss
            breakdown["overpotential_sign"] = sign_loss.item()

        if "ocv" in components and "eta_activation" in components:
            decomp_V = components["ocv"] - components["eta_activation"] - components.get("eta_ohmic", 0) - components.get("eta_concentration", 0)
            decomp_loss = F.mse_loss(decomp_V, V_pred.detach())
            loss = loss + self.lambda_ocv * decomp_loss
            breakdown["decomp_consistency"] = decomp_loss.item()

        if self.lambda_residual > 0 and "residual" in components:
            res_loss = components["residual"].pow(2).mean()
            loss = loss + self.lambda_residual * res_loss
            breakdown["residual_l2"] = res_loss.item()

        return loss, breakdown


def train(args):
    rank, world_size, local_rank, ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Device: {device}, DDP: {ddp}, World size: {world_size}")
        logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")

    from src.foundation.data.dataset import MultiChemDataset
    from src.foundation.model.physics_structured import (
        PhysicsInformedBatteryModel,
        PhysicsInformedBatteryModelSmall,
        PhysicsInformedBatteryModelLarge,
        count_parameters,
    )

    model_cls = {
        "small": PhysicsInformedBatteryModelSmall,
        "base": PhysicsInformedBatteryModel,
        "large": PhysicsInformedBatteryModelLarge,
    }[args.model_size]
    model = model_cls(n_chemistries=6, n_params=8, n_time=200)
    n_params = count_parameters(model)

    if rank == 0:
        logger.info(f"Model: PhysicsStructured {args.model_size}, {n_params/1e6:.1f}M params")

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    train_ds = MultiChemDataset(args.data_path, split="train")
    val_ds = MultiChemDataset(args.data_path, split="val")

    bs = args.batch_size
    if ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=train_sampler,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    if rank == 0:
        logger.info(
            f"Train: {len(train_ds)}, Val: {len(val_ds)}, "
            f"Batches/epoch: {len(train_loader)}, BS: {bs}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    warmup = min(2000, total_steps // 10)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup, total_steps=total_steps)

    loss_fn = DecompositionLoss(
        lambda_mono=args.lambda_mono,
        lambda_sign=args.lambda_sign,
        lambda_ocv=args.lambda_ocv,
        lambda_residual=args.lambda_residual,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    if rank == 0:
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup}, LR: {args.lr}")

    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_breakdown = {}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            V = batch["V"].to(device, non_blocking=True)
            chem = batch["chem_id"].to(device, non_blocking=True)
            params = batch["params"].to(device, non_blocking=True)
            cond = batch["conditions"].to(device, non_blocking=True)

            cond_norm = torch.cat(
                [cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1
            )

            m = model.module if ddp else model
            V_pred, components = m(chem, params, cond_norm, targets=V)

            loss, breakdown = loss_fn(V_pred, V, components)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step = epoch * len(train_loader) + n_batches
            scheduler.step(global_step)

            epoch_loss += breakdown["recon"]
            for k, v in breakdown.items():
                epoch_breakdown[k] = epoch_breakdown.get(k, 0.0) + v
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        for k in epoch_breakdown:
            epoch_breakdown[k] /= n_batches
        elapsed = time.time() - t0

        if rank == 0:
            val_loss = 0.0
            val_breakdown = {}
            n_val = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    V = batch["V"].to(device, non_blocking=True)
                    chem = batch["chem_id"].to(device, non_blocking=True)
                    params = batch["params"].to(device, non_blocking=True)
                    cond = batch["conditions"].to(device, non_blocking=True)
                    cond_norm = torch.cat(
                        [cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1
                    )

                    m = model.module if ddp else model
                    V_pred, components = m(chem, params, cond_norm, targets=V)
                    vloss, vb = loss_fn(V_pred, V, components)
                    val_loss += vb["recon"]
                    for k, v in vb.items():
                        val_breakdown[k] = val_breakdown.get(k, 0.0) + v
                    n_val += 1

            val_loss /= max(n_val, 1)
            for k in val_breakdown:
                val_breakdown[k] /= max(n_val, 1)

            v_range = 4.2 - 2.5
            train_rmse_mV = np.sqrt(avg_loss) * v_range * 1000
            val_rmse_mV = np.sqrt(val_loss) * v_range * 1000

            samples_per_sec = len(train_loader.dataset) / elapsed
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train={train_rmse_mV:.1f}mV val={val_rmse_mV:.1f}mV "
                f"{elapsed:.1f}s {samples_per_sec:.0f}sa/s lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            breakdown_parts = " ".join(
                f"{k}={v:.5f}" for k, v in epoch_breakdown.items() if k != "recon"
            )
            if breakdown_parts:
                logger.info(f"  breakdown: {breakdown_parts}")

            comp_summary = {}
            for k, v in components.items():
                comp_summary[k] = {
                    "mean": f"{v.mean().item():.4f}",
                    "std": f"{v.std().item():.4f}",
                    "min": f"{v.min().item():.4f}",
                    "max": f"{v.max().item():.4f}",
                }
            if epoch % 10 == 0 or epoch < 5:
                for k, s in comp_summary.items():
                    logger.info(
                        f"  {k}: mean={s['mean']} std={s['std']} "
                        f"range=[{s['min']}, {s['max']}]"
                    )

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "train_rmse_mV": train_rmse_mV,
                "val_rmse_mV": val_rmse_mV,
                "time": elapsed,
                "breakdown": epoch_breakdown,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_rmse_mV": val_rmse_mV,
                        "n_params": n_params,
                        "model_size": args.model_size,
                        "model_type": "physics_structured",
                        "data_path": args.data_path,
                    },
                    output_dir / "physics_fom_best.pt",
                )
                logger.info(f"  -> New best: {val_rmse_mV:.1f}mV")

            if (epoch + 1) % args.save_every == 0:
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "val_loss": val_loss,
                        "val_rmse_mV": val_rmse_mV,
                    },
                    output_dir / f"physics_fom_epoch{epoch+1}.pt",
                )

    if rank == 0:
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Done. Best val: {best_val_loss:.5f}")
        logger.info(f"Saved to {output_dir}")

    cleanup_ddp(ddp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/foundation/multichem_v2.h5")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints_physics_fom")
    parser.add_argument("--model-size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--lambda-mono", type=float, default=0.1)
    parser.add_argument("--lambda-sign", type=float, default=0.05)
    parser.add_argument("--lambda-ocv", type=float, default=0.01)
    parser.add_argument("--lambda-residual", type=float, default=0.01)
    args = parser.parse_args()
    train(args)
