import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import time
from typing import Optional

from .network import MultiDomainPINN, VoltageMLP, VoltagePredictor
from .pdes import MetalBatteryPDE
from .losses import PINNLoss

logger = logging.getLogger(__name__)


class ForwardTrainer:
    """
    Trainer for the forward PINN problem: learn a fast P2D solver.

    Given PyBaMM simulation data (params -> voltage curves), trains a PINN
    that respects the PDE physics while fitting the data.

    Phase 0: Pure data fitting (no PDE loss) to establish baseline.
    """

    def __init__(
        self,
        model,
        pde: MetalBatteryPDE,
        loss_fn: PINNLoss,
        device: torch.device = torch.device("cuda"),
        lr: float = 1e-3,
        scheduler: str = "cosine",
        num_epochs: int = 5000,
        log_every: int = 10,
        save_every: int = 100,
        checkpoint_dir: str = "outputs/checkpoints",
        use_pde: bool = False,
        pde_collocation_points: int = 256,
        grad_accumulation_steps: int = 1,
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
        self.use_pde = use_pde
        self.pde_collocation_points = pde_collocation_points
        self.grad_accumulation_steps = grad_accumulation_steps

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

        self.history = {"train_loss": [], "data_loss": [], "pde_loss": [], "val_loss": []}

    def _precompute_data(self, data_path: str, n_points: int = 200):
        """Load all data into GPU tensors at once. Batch load from npz."""
        import numpy as np
        from scipy.interpolate import interp1d

        data = np.load(data_path, allow_pickle=True)
        times = data["times"]
        voltages = data["voltages"]
        masks = data["masks"]
        param_values = data["param_values"].astype(np.float64)
        param_names = list(data["param_names"])
        num_sims = len(times)

        param_mean = param_values.mean(axis=0)
        param_std = param_values.std(axis=0) + 1e-12

        log_params = np.log10(np.clip(param_values, 1e-30, None))
        log_p_mean = log_params.mean(axis=0)
        log_p_std = log_params.std(axis=0) + 1e-12
        param_norm_values = (log_params - log_p_mean) / log_p_std

        all_t = np.empty((num_sims, n_points), dtype=np.float32)
        all_v = np.empty((num_sims, n_points), dtype=np.float32)
        all_params = np.empty((num_sims, n_points, len(param_names)), dtype=np.float32)

        for i in range(num_sims):
            mask = masks[i]
            t = times[i][mask]
            v = voltages[i][mask]

            t_end = t[-1] if t[-1] > 0 else 1.0
            t_norm = t / t_end

            t_uniform = np.linspace(0, 1, n_points)
            v_interp = np.interp(t_uniform, t_norm, v)

            all_t[i] = t_uniform.astype(np.float32)
            all_v[i] = v_interp.astype(np.float32)

            all_params[i] = np.tile(param_norm_values[i], (n_points, 1)).astype(np.float32)

        v_mean_global = float(all_v.mean())
        v_std_global = float(all_v.std()) + 1e-8

        v_per_sim_mean_np = all_v.mean(axis=1, keepdims=True).astype(np.float32)
        v_per_sim_std_np = all_v.std(axis=1, keepdims=True).astype(np.float32) + 1e-3
        all_v_norm = ((all_v - v_per_sim_mean_np) / v_per_sim_std_np).astype(np.float32)

        self._v_mean = v_mean_global
        self._v_std = v_std_global
        self._v_per_sim_mean = torch.from_numpy(v_per_sim_mean_np).to(self.device)
        self._v_per_sim_std = torch.from_numpy(v_per_sim_std_np).to(self.device)

        self._t_all = torch.from_numpy(all_t).to(self.device)
        self._v_all = torch.from_numpy(all_v_norm).to(self.device)
        self._v_raw = torch.from_numpy(all_v).to(self.device)
        self._params_all = torch.from_numpy(all_params).to(self.device)
        self._num_samples = num_sims
        self._n_points = n_points

        self._param_mean = param_mean
        self._param_std = param_std
        self._param_names = param_names

        perm = torch.randperm(self._num_samples, device=self.device)
        n_val = max(1, self._num_samples // 10)
        self._val_idx = perm[:n_val]
        self._train_idx = perm[n_val:]

        logger.info(
            f"Precomputed {self._num_samples} simulations "
            f"({len(self._train_idx)} train, {len(self._val_idx)} val) "
            f"on {self.device}"
        )

    def _get_batch(self, batch_size: int) -> dict:
        """Sample a random batch of training data from precomputed tensors."""
        idx = self._train_idx[torch.randint(0, len(self._train_idx), (batch_size,), device=self.device)]
        return {
            "t": self._t_all[idx],
            "v": self._v_all[idx],
            "params": self._params_all[idx],
        }

    def _forward_data_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        t = batch["t"]
        v_target = batch["v"]
        params = batch["params"]

        B, n_pts = t.shape[0], t.shape[1]

        if isinstance(self.model, VoltagePredictor):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            params_mean = params[:, 0, :]

            v_shape_pred = self.model.forward_shape(t_flat, params_flat).reshape(B, n_pts)
            v_offset_pred = self.model.forward_offset(params_mean)
            v_scale_pred = self.model.forward_scale(params_mean)

            v_pred = v_shape_pred * v_scale_pred + v_offset_pred

            v_target_denorm = v_target * self._v_per_sim_std[:B] + self._v_per_sim_mean[:B]

            mse_pred = torch.mean((v_pred - v_target_denorm) ** 2)

            v_target_shape = (v_target_denorm - v_offset_pred) / (v_scale_pred + 1e-6)
            mse_shape = torch.mean((v_shape_pred - v_target) ** 2)

            loss = mse_pred + 0.1 * mse_shape
            return loss * self.loss_fn.lambda_data, {"data": loss, "total": loss * self.loss_fn.lambda_data}
        elif isinstance(self.model, VoltageMLP):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            v_pred = self.model(t_flat, params_flat).reshape(B, n_pts)
        else:
            N = B * n_pts
            t_flat = t.reshape(-1, 1).requires_grad_(True)
            x_pos = torch.full((N, 1), 0.75, device=self.device)
            r_mid = torch.full((N, 1), 0.5, device=self.device)
            domain = torch.full(
                (N,), MultiDomainPINN.DOMAIN_POS,
                dtype=torch.long, device=self.device,
            )
            params_flat = params.reshape(-1, params.shape[-1])
            outputs = self.model(t_flat, x_pos, r_mid, params_flat, domain)
            v_pred = outputs["V"].reshape(B, n_pts)

        loss_dict = self.loss_fn(v_pred=v_pred, v_obs=v_target)
        return loss_dict["total"], loss_dict

    def _forward_pde_loss(self) -> torch.Tensor:
        """Compute PDE residual loss at collocation points."""
        n_colloc = self.pde_collocation_points

        colloc_t = torch.rand(n_colloc, 1, device=self.device, requires_grad=True)
        colloc_x = torch.rand(n_colloc, 1, device=self.device, requires_grad=True)
        colloc_r = torch.rand(n_colloc, 1, device=self.device, requires_grad=True)
        colloc_params = torch.zeros(n_colloc, self.model.num_params, device=self.device)
        colloc_domain = torch.randint(0, 3, (n_colloc,), device=self.device)

        colloc_outputs = self.model(
            colloc_t, colloc_x, colloc_r, colloc_params, colloc_domain
        )

        pde_residuals = self.pde.total_residual(
            colloc_outputs, colloc_t, colloc_x, colloc_r,
            colloc_domain,
            {
                "D_s": torch.tensor(1e-13, device=self.device),
                "k_sei": torch.tensor(1e-10, device=self.device),
                "j0_metal": torch.tensor(0.1, device=self.device),
            }
        )
        return self.loss_fn(pde_residuals=pde_residuals)["pde"]

    @torch.no_grad()
    def _validate(self) -> float:
        batch = {
            "t": self._t_all[self._val_idx],
            "v": self._v_all[self._val_idx],
            "params": self._params_all[self._val_idx],
        }
        loss, _ = self._forward_data_loss(batch)
        return loss.item()

    @torch.no_grad()
    def _validate_rmse_mV(self) -> float:
        batch = {
            "t": self._t_all[self._val_idx],
            "v": self._v_all[self._val_idx],
            "params": self._params_all[self._val_idx],
        }
        t = batch["t"]
        params = batch["params"]
        B, n_pts = t.shape[0], t.shape[1]

        if isinstance(self.model, VoltagePredictor):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            v_pred = self.model(t_flat, params_flat).reshape(B, n_pts)
        elif isinstance(self.model, VoltageMLP):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            v_pred_norm = self.model(t_flat, params_flat).reshape(B, n_pts)
            v_pred = v_pred_norm * self._v_std + self._v_mean
        else:
            N = B * n_pts
            t_flat = t.reshape(-1, 1).to(self.device)
            x_pos = torch.full((N, 1), 0.75, device=self.device)
            r_mid = torch.full((N, 1), 0.5, device=self.device)
            domain = torch.full((N,), MultiDomainPINN.DOMAIN_POS, dtype=torch.long, device=self.device)
            params_flat = params.reshape(-1, params.shape[-1])
            outputs = self.model(t_flat, x_pos, r_mid, params_flat, domain)
            v_pred = outputs["V"].reshape(B, n_pts) * self._v_std + self._v_mean

        v_true = self._v_raw[self._val_idx]
        rmse_V = torch.sqrt(torch.mean((v_pred - v_true) ** 2))
        return rmse_V.item() * 1000

    def train(
        self,
        data_path: str,
        batch_size: int = 256,
        n_points: int = 200,
    ) -> dict:
        """
        Train the forward PINN.

        Precomputes all data onto GPU, then runs training loop.
        """
        self._precompute_data(data_path, n_points)
        steps_per_epoch = max(1, len(self._train_idx) // batch_size)

        logger.info(f"Starting training: {self.num_epochs} epochs, {steps_per_epoch} steps/epoch")
        logger.info(f"PDE loss: {'ON' if self.use_pde else 'OFF'}")

        t_start = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_pde_loss = 0.0

            for step in range(steps_per_epoch):
                batch = self._get_batch(batch_size)
                total_loss, loss_dict = self._forward_data_loss(batch)
                data_loss_val = loss_dict.get("data", torch.tensor(0.0)).item()

                pde_loss_val = 0.0
                if self.use_pde and (step % self.grad_accumulation_steps == 0):
                    pde_loss = self._forward_pde_loss()
                    total_loss = total_loss + self.loss_fn.lambda_pde * pde_loss
                    pde_loss_val = pde_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += total_loss.item()
                epoch_data_loss += data_loss_val
                epoch_pde_loss += pde_loss_val

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            avg_data = epoch_data_loss / max(steps_per_epoch, 1)
            avg_pde = epoch_pde_loss / max(steps_per_epoch, 1)

            self.history["train_loss"].append(avg_loss)
            self.history["data_loss"].append(avg_data)
            self.history["pde_loss"].append(avg_pde)

            val_loss = 0.0
            if (epoch + 1) % self.log_every == 0:
                val_loss = self._validate()
                self.history["val_loss"].append(val_loss)

                val_rmse_mV = self._validate_rmse_mV()

                elapsed = time.time() - t_start
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Loss: {avg_loss:.6f} | Data: {avg_data:.6f} | "
                    f"PDE: {avg_pde:.6f} | Val: {val_loss:.6f} | "
                    f"RMSE: {val_rmse_mV:.1f}mV | "
                    f"LR: {lr:.2e} | {elapsed:.0f}s"
                )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.num_epochs, final=True)
        elapsed = time.time() - t_start
        logger.info(f"Training complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        return self.history

    def _save_checkpoint(self, epoch: int, final: bool = False):
        suffix = "final" if final else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"forward_pinn_{suffix}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "param_mean": self._param_mean,
            "param_std": self._param_std,
            "param_names": self._param_names,
            "v_mean": self._v_mean,
            "v_std": self._v_std,
            "log_p_mean": self._param_mean,
            "log_p_std": self._param_std,
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
