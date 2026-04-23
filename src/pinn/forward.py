import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Optional
from tqdm import tqdm

from .network import MultiDomainPINN
from .pdes import MetalBatteryPDE
from .losses import PINNLoss

logger = logging.getLogger(__name__)


class ForwardTrainer:
    """
    Trainer for the forward PINN problem: learn a fast P2D solver.

    Given PyBaMM simulation data (params → voltage curves), trains a PINN
    that respects the PDE physics while fitting the data.
    """

    def __init__(
        self,
        model: MultiDomainPINN,
        pde: MetalBatteryPDE,
        loss_fn: PINNLoss,
        device: torch.device = torch.device("cuda"),
        lr: float = 1e-3,
        scheduler: str = "cosine",
        num_epochs: int = 5000,
        log_every: int = 50,
        save_every: int = 500,
        checkpoint_dir: str = "outputs/checkpoints",
    ):
        self.model = model.to(device)
        self.pde = pde
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
            )
        elif scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=num_epochs // 3, gamma=0.1
            )
        else:
            self.scheduler = None

        self.history = {"train_loss": [], "data_loss": [], "pde_loss": []}

    def train(
        self,
        train_loader: DataLoader,
        collocation_fn=None,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """
        Train the forward PINN.

        Args:
            train_loader: DataLoader providing (x, voltage, current) tuples
            collocation_fn: callable(batch_size) → collocation points for PDE
            val_loader: optional validation data

        Returns:
            training history dict
        """
        self.model.train()
        global_step = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_pde_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                x, v_target, i_input = batch
                x = x.to(self.device)
                v_target = v_target.to(self.device)

                t = x[:, :, 0:1]
                params = x[:, :, 1:]
                B = t.shape[0]
                n_pts = t.shape[1]

                x_pos = torch.full((B * n_pts, 1), 0.75, device=self.device)
                r_mid = torch.full((B * n_pts, 1), 0.5, device=self.device)
                domain = torch.full(
                    (B * n_pts,), MultiDomainPINN.DOMAIN_POS,
                    dtype=torch.long, device=self.device,
                )

                t_flat = t.reshape(-1, 1).requires_grad_(True)
                x_flat = x_pos.requires_grad_(True)
                r_flat = r_mid.requires_grad_(True)
                params_flat = params.reshape(-1, params.shape[-1])

                outputs = self.model(t_flat, x_flat, r_flat, params_flat, domain)
                v_pred = outputs["V"].reshape(B, n_pts)

                loss_dict = self.loss_fn(
                    v_pred=v_pred,
                    v_obs=v_target,
                )

                loss = loss_dict["total"]

                if collocation_fn is not None:
                    colloc_pts = collocation_fn(1024).to(self.device)
                    colloc_t = colloc_pts[:, 0:1].requires_grad_(True)
                    colloc_x = colloc_pts[:, 1:2].requires_grad_(True)
                    colloc_r = colloc_pts[:, 2:3].requires_grad_(True)
                    colloc_params = torch.zeros(
                        colloc_pts.shape[0], self.model.num_params, device=self.device
                    )
                    colloc_domain = torch.randint(
                        0, 3, (colloc_pts.shape[0],), device=self.device
                    )

                    colloc_outputs = self.model(
                        colloc_t, colloc_x, colloc_r, colloc_params, colloc_domain
                    )
                    pde_residuals = self.pde.total_residual(
                        colloc_outputs, colloc_t, colloc_x, colloc_r,
                        colloc_domain, {"D_s": torch.tensor(1e-13), "k_sei": torch.tensor(1e-10), "j0_metal": torch.tensor(0.1)}
                    )
                    pde_loss_dict = self.loss_fn(pde_residuals=pde_residuals)
                    loss = loss + pde_loss_dict["pde"]

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_data_loss += loss_dict.get("data", torch.tensor(0)).item()
                epoch_pde_loss += loss_dict.get("pde", torch.tensor(0)).item()
                num_batches += 1
                global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_data = epoch_data_loss / max(num_batches, 1)
            avg_pde = epoch_pde_loss / max(num_batches, 1)

            self.history["train_loss"].append(avg_loss)
            self.history["data_loss"].append(avg_data)
            self.history["pde_loss"].append(avg_pde)

            if (epoch + 1) % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Loss: {avg_loss:.6f} | Data: {avg_data:.6f} | "
                    f"PDE: {avg_pde:.6f} | LR: {lr:.2e}"
                )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.num_epochs, final=True)
        return self.history

    def _save_checkpoint(self, epoch: int, final: bool = False):
        suffix = "final" if final else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"forward_pinn_{suffix}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def predict(
        self,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Predict voltage for given time and parameters."""
        self.model.eval()
        with torch.no_grad():
            v = self.model.forward_voltage_only(t.to(self.device), params.to(self.device))
        return v
