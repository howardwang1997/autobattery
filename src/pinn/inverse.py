import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Optional
from tqdm import tqdm

from .network import InversePINN, MultiDomainPINN
from .pdes import MetalBatteryPDE
from .losses import PINNLoss

logger = logging.getLogger(__name__)


class InverseTrainer:
    """
    Trainer for the inverse PINN problem: identify electrochemical parameters
    from experimental cycling data.

    Two-phase optimization:
    1. Phase 1 (Adam): Fast exploration of parameter space
    2. Phase 2 (L-BFGS): Precise convergence near the optimum
    """

    def __init__(
        self,
        model: InversePINN,
        pde: MetalBatteryPDE,
        loss_fn: PINNLoss,
        device: torch.device = torch.device("cuda"),
        lr: float = 1e-3,
        param_lr_scale: float = 0.1,
        phase1_epochs: int = 5000,
        phase2_epochs: int = 5000,
        log_every: int = 50,
        save_every: int = 500,
        checkpoint_dir: str = "outputs/checkpoints",
    ):
        self.model = model.to(device)
        self.pde = pde
        self.loss_fn = loss_fn
        self.device = device
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        base_model_params = list(model.base_model.parameters())
        learnable_params_list = list(model.learnable_params_raw.parameters())

        self.optimizer = torch.optim.Adam(
            [
                {"params": base_model_params, "lr": lr},
                {"params": learnable_params_list, "lr": lr * param_lr_scale},
            ],
            lr=lr,
        )

        self.history = {
            "train_loss": [],
            "data_loss": [],
            "pde_loss": [],
            "params": {name: [] for name in model.learnable_param_names},
        }

    def train(
        self,
        t_colloc: torch.Tensor,
        v_obs: torch.Tensor,
        i_input: torch.Tensor,
        val_split: float = 0.2,
    ) -> dict:
        """
        Train inverse PINN to identify parameters from experimental data.

        Args:
            t_colloc: (N, 1) normalized time collocation points
            v_obs: (N, 1) observed voltage
            i_input: (N, 1) input current
            val_split: fraction of data to hold out for validation

        Returns:
            training history and identified parameters
        """
        t_colloc = t_colloc.to(self.device).requires_grad_(True)
        v_obs = v_obs.to(self.device)
        i_input = i_input.to(self.device)

        n = t_colloc.shape[0]
        n_val = int(n * val_split)
        n_train = n - n_val

        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        t_train = t_colloc[train_idx]
        v_train = v_obs[train_idx]

        logger.info(f"Training inverse PINN: {n_train} train, {n_val} val points")
        logger.info(f"Initial parameters:")
        for name, val in self.model.get_all_params().items():
            logger.info(f"  {name} = {val.item():.6e}")

        logger.info("Phase 1: Adam optimization")
        self._train_adam(t_train, v_train, self.phase1_epochs)

        logger.info("Phase 2: L-BFGS fine-tuning")
        self._train_lbfgs(t_train, v_train, self.phase2_epochs)

        final_params = self.model.get_all_params()
        logger.info("Identified parameters:")
        for name, val in final_params.items():
            logger.info(f"  {name} = {val.item():.6e}")

        self._save_checkpoint("final")

        return {
            "history": self.history,
            "params": {k: v.item() for k, v in final_params.items()},
        }

    def _train_adam(self, t: torch.Tensor, v_obs: torch.Tensor, num_epochs: int):
        """Phase 1: Adam optimizer."""
        self.model.train()
        B = t.shape[0]

        x_neg = torch.full((B, 1), 0.25, device=self.device).requires_grad_(True)
        x_sep = torch.full((B, 1), 0.5, device=self.device).requires_grad_(True)
        x_pos = torch.full((B, 1), 0.75, device=self.device).requires_grad_(True)
        r = torch.full((B, 1), 0.5, device=self.device).requires_grad_(True)

        domain = torch.cat([
            torch.full((B // 3,), 0, dtype=torch.long, device=self.device),
            torch.full((B // 3,), 1, dtype=torch.long, device=self.device),
            torch.full((B - 2 * (B // 3),), 2, dtype=torch.long, device=self.device),
        ])

        for epoch in range(num_epochs):
            x_cat = torch.cat([x_neg, x_sep, x_pos], dim=0)
            r_cat = r.expand(x_cat.shape[0], -1)

            outputs = self.model(t.expand(x_cat.shape[0], -1), x_cat, r_cat, domain)

            v_pred = outputs["V"][:B]

            loss_dict = self.loss_fn(v_pred=v_pred, v_obs=v_obs)

            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self._record(epoch, loss_dict, num_epochs)

    def _train_lbfgs(self, t: torch.Tensor, v_obs: torch.Tensor, num_epochs: int):
        """Phase 2: L-BFGS optimizer for fine-tuning."""
        self.model.train()
        B = t.shape[0]

        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=20,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )

        x_neg = torch.full((B, 1), 0.25, device=self.device).requires_grad_(True)
        x_sep = torch.full((B, 1), 0.5, device=self.device).requires_grad_(True)
        x_pos = torch.full((B, 1), 0.75, device=self.device).requires_grad_(True)
        r = torch.full((B, 1), 0.5, device=self.device).requires_grad_(True)
        domain = torch.cat([
            torch.full((B // 3,), 0, dtype=torch.long, device=self.device),
            torch.full((B // 3,), 1, dtype=torch.long, device=self.device),
            torch.full((B - 2 * (B // 3),), 2, dtype=torch.long, device=self.device),
        ])

        for epoch in range(num_epochs):
            def closure():
                optimizer.zero_grad()
                x_cat = torch.cat([x_neg, x_sep, x_pos], dim=0)
                r_cat = r.expand(x_cat.shape[0], -1)
                outputs = self.model(
                    t.expand(x_cat.shape[0], -1), x_cat, r_cat, domain
                )
                v_pred = outputs["V"][:B]
                loss_dict = self.loss_fn(v_pred=v_pred, v_obs=v_obs)
                loss = loss_dict["total"]
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                x_cat = torch.cat([x_neg, x_sep, x_pos], dim=0)
                r_cat = r.expand(x_cat.shape[0], -1)
                outputs = self.model(
                    t.expand(x_cat.shape[0], -1), x_cat, r_cat, domain
                )
                v_pred = outputs["V"][:B]
                loss_dict = self.loss_fn(v_pred=v_pred, v_obs=v_obs)

            self._record(epoch, loss_dict, num_epochs, phase="lbfgs")

    def _record(self, epoch, loss_dict, num_epochs, phase="adam"):
        self.history["train_loss"].append(loss_dict["total"].item())
        if "data" in loss_dict:
            self.history["data_loss"].append(loss_dict["data"].item())
        if "pde" in loss_dict:
            self.history["pde_loss"].append(loss_dict["pde"].item())

        for name in self.model.learnable_param_names:
            val = self.model.get_param(name).item()
            self.history["params"][name].append(val)

        if (epoch + 1) % self.log_every == 0:
            params_str = " | ".join(
                f"{k}={v.item():.4e}"
                for k, v in self.model.get_all_params().items()
            )
            logger.info(
                f"[{phase}] Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {loss_dict['total'].item():.6f} | {params_str}"
            )

    def _save_checkpoint(self, suffix: str = "final"):
        path = self.checkpoint_dir / f"inverse_pinn_{suffix}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "learnable_params": {
                k: v.item() for k, v in self.model.get_all_params().items()
            },
            "history": self.history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")
