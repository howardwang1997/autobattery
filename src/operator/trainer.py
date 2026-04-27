import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
import json
from pathlib import Path

from .fno import FNO2d
from .dataset import FullFieldDataset

logger = logging.getLogger(__name__)


class FNOTrainer:
    """Trainer for Fourier Neural Operator on full-field battery data."""

    def __init__(
        self,
        model: FNO2d,
        dataset: FullFieldDataset,
        device: torch.device = torch.device("cuda"),
        lr: float = 1e-3,
        num_epochs: int = 200,
        batch_size: int = 16,
        lambda_field: float = 1.0,
        lambda_voltage: float = 10.0,
        log_every: int = 10,
        save_every: int = 50,
        checkpoint_dir: str = "outputs/checkpoints",
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lambda_field = lambda_field
        self.lambda_voltage = lambda_voltage
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        train_subset = torch.utils.data.Subset(dataset, dataset.train_idx.tolist())
        val_subset = torch.utils.data.Subset(dataset, dataset.val_idx.tolist())

        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
        )
        self.val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
        )

        self.history = {"train_loss": [], "val_loss": [], "field_loss": [], "voltage_loss": []}

    def train(self) -> dict:
        logger.info(
            f"FNO training: {self.num_epochs} epochs, "
            f"{len(self.train_loader)} batches/epoch, "
            f"{self.dataset.n_sims} sims"
        )

        t_start = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_field_loss = 0.0
            epoch_v_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                coord = batch["coord"].to(self.device)
                fields_target = batch["fields"].to(self.device)
                params = batch["params"].to(self.device)
                c_rate = batch["c_rate"].to(self.device)
                v_target = batch["voltage"].to(self.device)

                fields_pred, v_pred = self.model(coord, params, c_rate)

                field_loss = nn.functional.mse_loss(fields_pred, fields_target)
                v_loss = nn.functional.mse_loss(v_pred, v_target)
                loss = self.lambda_field * field_loss + self.lambda_voltage * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_field_loss += field_loss.item()
                epoch_v_loss += v_loss.item()
                n_batches += 1

            self.scheduler.step()

            avg_field = epoch_field_loss / max(n_batches, 1)
            avg_v = epoch_v_loss / max(n_batches, 1)

            if (epoch + 1) % self.log_every == 0:
                val_loss = self._validate()
                elapsed = time.time() - t_start
                lr = self.optimizer.param_groups[0]["lr"]

                v_stats = self.dataset._stats.get("V", {"std": 1.0})
                v_rmse_mV = np.sqrt(avg_v) * v_stats["std"] * 1000

                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Field: {avg_field:.6f} | V: {avg_v:.6f} ({v_rmse_mV:.1f}mV) | "
                    f"Val: {val_loss:.6f} | LR: {lr:.2e} | {elapsed:.0f}s"
                )

                self.history["val_loss"].append(val_loss)

            self.history["train_loss"].append(avg_field + avg_v)
            self.history["field_loss"].append(avg_field)
            self.history["voltage_loss"].append(avg_v)

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.num_epochs, final=True)
        elapsed = time.time() - t_start
        logger.info(f"FNO training complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        return self.history

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            coord = batch["coord"].to(self.device)
            fields_target = batch["fields"].to(self.device)
            params = batch["params"].to(self.device)
            c_rate = batch["c_rate"].to(self.device)
            v_target = batch["voltage"].to(self.device)

            fields_pred, v_pred = self.model(coord, params, c_rate)
            loss = nn.functional.mse_loss(fields_pred, fields_target) + \
                   self.lambda_voltage * nn.functional.mse_loss(v_pred, v_target)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def _save_checkpoint(self, epoch, final=False):
        suffix = "final" if final else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"fno_{suffix}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")


import numpy as np
