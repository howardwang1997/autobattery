"""Forward PINN trainer.

Phase A1 rewrite (see ``docs/plan_publication_roadmap.md``). The
previous implementation had three load-bearing bugs:

1. **per-simulation voltage normalisation.** Targets were normalised by
   the per-curve mean/std, but the validation RMSE denormalised with
   *global* statistics — so reported numbers under-counted error and
   the model only worked because it leaked the target's first/second
   moments. Fixed: a single ``--norm-mode`` switch defaulting to
   ``global`` (the only honest option). The legacy ``per_sim`` mode is
   kept behind a deprecation warning so prior runs are reproducible.
2. **fake PDE collocation.** ``--use-pde`` sampled collocation
   parameters from a tensor of zeros, making the residual essentially a
   constant. Fixed: collocation parameters are sampled from the same
   distribution as the training batch.
3. **fixed loss weights.** Hardcoded ``lambda_data=10``, ``lambda_pde=1``
   guarantees the PDE term is ignored. Fixed: optional SoftAdapt
   integration via ``--adaptive-weighting``.

The trainer still supports three model classes (``VoltageMLP``,
``VoltagePredictor``, ``MultiDomainPINN``) so existing checkpoints load.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .adaptive_weighting import SoftAdapt, SoftAdaptConfig
from .losses import PINNLoss
from .network import MultiDomainPINN, VoltageMLP, VoltagePredictor
from .pdes import MetalBatteryPDE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@dataclass
class VoltageNormalizer:
    """Single source of truth for voltage normalisation.

    Used by *both* the training loop and validation/inference, so the
    bug where train and val normalisations disagreed is structurally
    impossible.
    """

    mode: str                          # "global" | "per_sim" (legacy)
    v_mean_global: float
    v_std_global: float
    v_per_sim_mean: Optional[torch.Tensor] = None    # (n_sim, 1)
    v_per_sim_std: Optional[torch.Tensor] = None     # (n_sim, 1)

    def normalize(
        self,
        v_raw: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "global":
            return (v_raw - self.v_mean_global) / self.v_std_global
        if self.mode == "per_sim":
            if idx is None:
                raise ValueError("per_sim normalisation needs the simulation index")
            mean = self.v_per_sim_mean[idx]
            std = self.v_per_sim_std[idx]
            return (v_raw - mean) / std
        raise ValueError(self.mode)

    def denormalize(
        self,
        v_norm: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "global":
            return v_norm * self.v_std_global + self.v_mean_global
        if self.mode == "per_sim":
            if idx is None:
                raise ValueError("per_sim denormalisation needs the simulation index")
            mean = self.v_per_sim_mean[idx]
            std = self.v_per_sim_std[idx]
            return v_norm * std + mean
        raise ValueError(self.mode)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ForwardTrainer:
    """Forward PINN trainer with optional PDE residual loss.

    Args:
        model: ``VoltageMLP``, ``VoltagePredictor`` or ``MultiDomainPINN``.
        pde: PDE residual evaluator (only used when ``use_pde=True`` and
            the model is a ``MultiDomainPINN``).
        loss_fn: weighted loss aggregator. Static weights are honoured
            unless ``adaptive_weighting`` is enabled.
        norm_mode: ``"global"`` (default, honest) or ``"per_sim"`` (legacy
            with data leakage; kept for reproducibility only).
        use_pde: whether to add the PDE residual term to the total loss.
            Only meaningful with ``MultiDomainPINN``.
        adaptive_weighting: ``"none"`` or ``"softadapt"``. Default
            ``"none"`` for backward compatibility.
        pde_collocation_points: per-step collocation count.
        pde_warmup_epochs: number of epochs during which only the data
            term is optimised. Avoids the well-known PINN failure where
            a strongly weighted PDE residual against random init pulls
            the model into a local minimum that fits neither.
    """

    def __init__(
        self,
        model: nn.Module,
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
        pde_warmup_epochs: int = 0,
        norm_mode: str = "global",
        adaptive_weighting: str = "none",
        adaptive_config: Optional[SoftAdaptConfig] = None,
        grad_accumulation_steps: int = 1,
    ):
        if norm_mode not in ("global", "per_sim"):
            raise ValueError(f"norm_mode must be 'global' or 'per_sim', got {norm_mode!r}")
        if norm_mode == "per_sim":
            warnings.warn(
                "per_sim voltage normalisation leaks per-curve statistics into "
                "training; reported RMSE is over-optimistic. Use --norm-mode global.",
                DeprecationWarning,
                stacklevel=2,
            )
        if adaptive_weighting not in ("none", "softadapt"):
            raise ValueError(f"adaptive_weighting={adaptive_weighting!r}")

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
        self.pde_warmup_epochs = pde_warmup_epochs
        self.norm_mode = norm_mode
        self.grad_accumulation_steps = grad_accumulation_steps

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=lr * 0.01,
            )
        elif scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, num_epochs // 3), gamma=0.1,
            )
        else:
            self.scheduler = None

        loss_terms = ["data"]
        if use_pde:
            loss_terms.append("pde")
        self.adaptive_weighting = adaptive_weighting
        self.softadapt: Optional[SoftAdapt] = (
            SoftAdapt(loss_terms, config=adaptive_config)
            if adaptive_weighting == "softadapt" else None
        )

        self.history = {
            "train_loss": [], "data_loss": [], "pde_loss": [],
            "val_loss": [], "val_rmse_mV": [],
            "softadapt_w_data": [], "softadapt_w_pde": [],
        }

        # Set during _precompute_data
        self.normalizer: Optional[VoltageNormalizer] = None

    # ---- Data loading -------------------------------------------------

    def _precompute_data(self, data_path: str, n_points: int = 200) -> None:
        data = np.load(data_path, allow_pickle=False)
        times = data["times"]
        voltages = data["voltages"]
        masks = data["masks"]
        param_values = data["param_values"].astype(np.float64)
        param_names = list(data["param_names"])
        num_sims = len(times)

        log_params = np.log10(np.clip(param_values, 1e-30, None))
        log_p_mean = log_params.mean(axis=0)
        log_p_std = log_params.std(axis=0) + 1e-12
        param_norm_values = ((log_params - log_p_mean) / log_p_std).astype(np.float32)

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

        self.normalizer = VoltageNormalizer(
            mode=self.norm_mode,
            v_mean_global=v_mean_global,
            v_std_global=v_std_global,
            v_per_sim_mean=torch.from_numpy(v_per_sim_mean_np).to(self.device),
            v_per_sim_std=torch.from_numpy(v_per_sim_std_np).to(self.device),
        )

        self._t_all = torch.from_numpy(all_t).to(self.device)
        self._v_raw = torch.from_numpy(all_v).to(self.device)
        self._params_all = torch.from_numpy(all_params).to(self.device)
        self._num_samples = num_sims
        self._n_points = n_points

        # Pre-compute normalised voltage. For global mode this is just one
        # tensor; for per_sim we still build it here so the train loop is
        # uniform.
        if self.norm_mode == "global":
            self._v_all = (self._v_raw - v_mean_global) / v_std_global
        else:
            self._v_all = (
                (self._v_raw - self.normalizer.v_per_sim_mean)
                / self.normalizer.v_per_sim_std
            )

        self._param_mean = log_p_mean
        self._param_std = log_p_std
        self._param_names = param_names

        rng = np.random.default_rng(0)
        perm = rng.permutation(self._num_samples)
        n_val = max(1, self._num_samples // 10)
        self._val_idx = torch.from_numpy(perm[:n_val].copy()).to(self.device)
        self._train_idx = torch.from_numpy(perm[n_val:].copy()).to(self.device)

        logger.info(
            "Precomputed %d simulations (%d train, %d val) on %s | norm_mode=%s",
            self._num_samples, len(self._train_idx), len(self._val_idx),
            self.device, self.norm_mode,
        )

    # ---- Batching ------------------------------------------------------

    def _get_batch(self, batch_size: int) -> dict:
        sel = torch.randint(0, len(self._train_idx), (batch_size,), device=self.device)
        idx = self._train_idx[sel]
        return {
            "idx": idx,
            "t": self._t_all[idx],
            "v": self._v_all[idx],
            "v_raw": self._v_raw[idx],
            "params": self._params_all[idx],
        }

    def _sample_collocation(self, n_colloc: int, domain_id: int = 2) -> dict:
        """Sample collocation points with parameters drawn from the
        training distribution (NOT zeros).

        Critically, ``domain_id`` is held constant across the batch.
        ``MultiDomainPINN`` masks its outputs by domain index, which
        breaks the autograd path between the masked output and a
        masked-after-the-fact spatial input (the residual would attempt
        ``grad(c_s_pos[pos_mask], r[pos_mask])`` against tensors with no
        shared graph). Single-domain batches sidestep the issue
        entirely.

        Use one call per residual term:
          * ``domain_id=0`` (neg) → metal-plating kinetics, SEI growth
          * ``domain_id=2`` (pos) → cathode diffusion
        """
        sel = torch.randint(0, len(self._train_idx), (n_colloc,), device=self.device)
        idx = self._train_idx[sel]
        t = torch.rand(n_colloc, 1, device=self.device, requires_grad=True)
        x_lo = {0: 0.0, 1: self.pde.L_neg_frac, 2: self.pde.L_neg_frac + self.pde.L_sep_frac}[domain_id]
        x_hi = {0: self.pde.L_neg_frac, 1: self.pde.L_neg_frac + self.pde.L_sep_frac, 2: 1.0}[domain_id]
        x = (x_lo + (x_hi - x_lo) * torch.rand(n_colloc, 1, device=self.device)).requires_grad_(True)
        r = torch.rand(n_colloc, 1, device=self.device, requires_grad=True)
        params = self._params_all[idx, 0, :]
        domain = torch.full((n_colloc,), domain_id, dtype=torch.long, device=self.device)
        return {"t": t, "x": x, "r": r, "params": params, "domain": domain, "idx": idx}

    # ---- Loss components ---------------------------------------------

    def _data_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        t = batch["t"]
        v_target = batch["v"]
        params = batch["params"]
        idx = batch["idx"]
        B, n_pts = t.shape[0], t.shape[1]

        if isinstance(self.model, VoltagePredictor):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            params_mean = params[:, 0, :]
            v_shape_pred = self.model.forward_shape(t_flat, params_flat).reshape(B, n_pts)
            v_offset_pred = self.model.forward_offset(params_mean)
            v_scale_pred = self.model.forward_scale(params_mean)
            v_pred = v_shape_pred * v_scale_pred + v_offset_pred
            v_target_denorm = self.normalizer.denormalize(v_target, idx=idx)
            l_data = nn.functional.mse_loss(v_pred, v_target_denorm)
            return l_data, {"data": l_data}

        if isinstance(self.model, VoltageMLP):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            v_pred = self.model(t_flat, params_flat).reshape(B, n_pts)
        else:  # MultiDomainPINN
            N = B * n_pts
            t_flat = t.reshape(-1, 1)
            x_pos = torch.full((N, 1), 0.75, device=self.device)
            r_mid = torch.full((N, 1), 0.5, device=self.device)
            domain = torch.full(
                (N,), MultiDomainPINN.DOMAIN_POS, dtype=torch.long, device=self.device,
            )
            params_flat = params.reshape(-1, params.shape[-1])
            outputs = self.model(t_flat, x_pos, r_mid, params_flat, domain)
            v_pred = outputs["V"].reshape(B, n_pts)

        l_data = nn.functional.mse_loss(v_pred, v_target)
        return l_data, {"data": l_data}

    def _pde_loss(self) -> torch.Tensor:
        """Compute PDE residual loss with real (non-zero) parameters.

        Runs one forward pass per domain so that the autograd path
        between ``outputs[domain_*]`` and the spatial inputs stays
        intact (the grad would otherwise be cut by the per-domain
        output masking inside ``MultiDomainPINN``).
        """
        if not isinstance(self.model, MultiDomainPINN):
            return torch.tensor(0.0, device=self.device)

        n_per_domain = max(1, self.pde_collocation_points // 2)

        residuals: dict[str, torch.Tensor] = {}

        # Negative electrode → metal plating kinetics + SEI growth.
        neg = self._sample_collocation(n_per_domain, domain_id=MultiDomainPINN.DOMAIN_NEG)
        out_neg = self.model(neg["t"], neg["x"], neg["r"], neg["params"], neg["domain"])
        phys_neg = self._physics_params_from_norm(neg["params"])
        residuals["metal_kinetics"] = self.pde.metal_plating_kinetics(
            c_e=out_neg["neg_c_e"],
            phi_s=out_neg["neg_phi_s"],
            phi_e=out_neg["neg_phi_e"],
            j0_metal=phys_neg["j0_metal"],
        )
        residuals["sei_growth"] = self.pde.sei_growth_residual(
            L_sei=out_neg["L_sei"],
            t_norm=neg["t"],
            j_side=torch.zeros_like(neg["t"]),
            k_sei=phys_neg["k_sei"],
        )

        # Positive electrode → cathode solid-phase diffusion.
        pos = self._sample_collocation(n_per_domain, domain_id=MultiDomainPINN.DOMAIN_POS)
        out_pos = self.model(pos["t"], pos["x"], pos["r"], pos["params"], pos["domain"])
        phys_pos = self._physics_params_from_norm(pos["params"])
        residuals["cathode_diffusion"] = self.pde.cathode_diffusion_residual(
            c_s=out_pos["pos_c_s"],
            t_norm=pos["t"],
            r_norm=pos["r"],
            D_s=phys_pos["D_s"],
        )

        return self.loss_fn.pde_loss(residuals)

    def _physics_params_from_norm(self, params_norm: torch.Tensor) -> dict:
        """Map normalised network-input params back to physical PDE
        parameters. Falls back to scalar defaults when a name is absent
        from the sweep — those become broadcast tensors so the residual
        graph stays connected.
        """
        device = params_norm.device

        # Index lookups (returns -1 if missing).
        def col(name: str) -> int:
            try:
                return self._param_names.index(name)
            except ValueError:
                return -1

        # Column → physical value via inverse log-space transform.
        def physical(name: str, default: float) -> torch.Tensor:
            j = col(name)
            if j < 0:
                return torch.full(
                    (params_norm.shape[0], 1), default, device=device,
                )
            mean = float(self._param_mean[j])
            std = float(self._param_std[j])
            log_val = params_norm[:, j:j + 1] * std + mean
            return torch.pow(10.0, log_val)

        return {
            "D_s": physical("Positive particle diffusivity [m2.s-1]", 1e-13),
            "k_sei": physical("SEI kinetic rate constant [m.s-1]", 1e-12),
            "j0_metal": physical(
                "Lithium plating kinetic rate constant [m.s-1]", 1e-9,
            ),
        }

    # ---- Validation ---------------------------------------------------

    @torch.no_grad()
    def _validate_rmse_mV(self) -> float:
        """Single source of truth for validation RMSE in mV.

        Critically: predictions are denormalised with the *same*
        :class:`VoltageNormalizer` used in training, so the train/val
        bug from the previous implementation cannot recur.
        """
        idx = self._val_idx
        t = self._t_all[idx]
        params = self._params_all[idx]
        v_raw = self._v_raw[idx]
        B, n_pts = t.shape[0], t.shape[1]

        if isinstance(self.model, VoltagePredictor):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            v_pred = self.model(t_flat, params_flat).reshape(B, n_pts)
        elif isinstance(self.model, VoltageMLP):
            t_flat = t.reshape(-1, 1)
            params_flat = params.reshape(-1, params.shape[-1])
            v_pred_norm = self.model(t_flat, params_flat).reshape(B, n_pts)
            v_pred = self.normalizer.denormalize(v_pred_norm, idx=idx)
        else:
            N = B * n_pts
            t_flat = t.reshape(-1, 1)
            x_pos = torch.full((N, 1), 0.75, device=self.device)
            r_mid = torch.full((N, 1), 0.5, device=self.device)
            domain = torch.full(
                (N,), MultiDomainPINN.DOMAIN_POS, dtype=torch.long, device=self.device,
            )
            params_flat = params.reshape(-1, params.shape[-1])
            outputs = self.model(t_flat, x_pos, r_mid, params_flat, domain)
            v_pred_norm = outputs["V"].reshape(B, n_pts)
            v_pred = self.normalizer.denormalize(v_pred_norm, idx=idx)

        return float(torch.sqrt(torch.mean((v_pred - v_raw) ** 2))) * 1000.0

    # ---- Train --------------------------------------------------------

    def _current_weights(self) -> dict[str, float]:
        if self.softadapt is None:
            w = {"data": float(self.loss_fn.lambda_data)}
            if self.use_pde:
                w["pde"] = float(self.loss_fn.lambda_pde)
            return w
        return self.softadapt.weights()

    def train(
        self,
        data_path: str,
        batch_size: int = 256,
        n_points: int = 200,
    ) -> dict:
        self._precompute_data(data_path, n_points)
        steps_per_epoch = max(1, len(self._train_idx) // batch_size)

        logger.info(
            "Starting training: %d epochs, %d steps/epoch | use_pde=%s | adaptive=%s",
            self.num_epochs, steps_per_epoch, self.use_pde, self.adaptive_weighting,
        )
        t_start = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            ep_total = ep_data = ep_pde = 0.0

            for step in range(steps_per_epoch):
                batch = self._get_batch(batch_size)
                l_data, _ = self._data_loss(batch)

                weights = self._current_weights()
                loss = weights["data"] * l_data
                pde_val = 0.0

                if self.use_pde and epoch >= self.pde_warmup_epochs:
                    l_pde = self._pde_loss()
                    loss = loss + weights.get("pde", 1.0) * l_pde
                    pde_val = float(l_pde.detach())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                ep_total += float(loss.detach())
                ep_data += float(l_data.detach())
                ep_pde += pde_val

                if self.softadapt is not None:
                    losses_for_softadapt = {"data": float(l_data.detach())}
                    if self.use_pde and epoch >= self.pde_warmup_epochs:
                        losses_for_softadapt["pde"] = pde_val
                    self.softadapt.update(losses_for_softadapt)

            if self.scheduler is not None:
                self.scheduler.step()

            avg_total = ep_total / steps_per_epoch
            avg_data = ep_data / steps_per_epoch
            avg_pde = ep_pde / steps_per_epoch
            self.history["train_loss"].append(avg_total)
            self.history["data_loss"].append(avg_data)
            self.history["pde_loss"].append(avg_pde)

            sa_w = self._current_weights()
            self.history["softadapt_w_data"].append(sa_w.get("data", 1.0))
            self.history["softadapt_w_pde"].append(sa_w.get("pde", 0.0))

            if (epoch + 1) % self.log_every == 0:
                val_rmse = self._validate_rmse_mV()
                self.history["val_rmse_mV"].append(val_rmse)
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                logger.info(
                    "Epoch %4d/%d | total=%.4e data=%.4e pde=%.4e | "
                    "RMSE=%.1fmV | w_data=%.3f w_pde=%.3f | lr=%.1e | %.0fs",
                    epoch + 1, self.num_epochs,
                    avg_total, avg_data, avg_pde, val_rmse,
                    sa_w.get("data", 0.0), sa_w.get("pde", 0.0),
                    lr, elapsed,
                )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.num_epochs, final=True)
        elapsed = time.time() - t_start
        logger.info("Training complete in %.0fs (%.1fh)", elapsed, elapsed / 3600.0)
        return self.history

    # ---- Persistence --------------------------------------------------

    def _save_checkpoint(self, epoch: int, final: bool = False) -> Path:
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
            "norm_mode": self.norm_mode,
            "v_mean_global": self.normalizer.v_mean_global,
            "v_std_global": self.normalizer.v_std_global,
            "use_pde": self.use_pde,
            "adaptive_weighting": self.adaptive_weighting,
        }, path)
        logger.info("Checkpoint saved to %s", path)
        return path

    def predict(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Predict (denormalised) voltage. Only valid for ``global`` norm
        mode — per-sim mode requires the curve's own mean/std which is
        not available at inference time.
        """
        if self.norm_mode != "global":
            raise RuntimeError(
                "predict() requires norm_mode='global' (per_sim is not "
                "well-defined at inference time)."
            )
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, (VoltageMLP, VoltagePredictor)):
                v_norm = self.model(t.to(self.device), params.to(self.device))
                if isinstance(self.model, VoltagePredictor):
                    return v_norm
                return v_norm * self.normalizer.v_std_global + self.normalizer.v_mean_global
            v_norm = self.model.forward_voltage_only(t.to(self.device), params.to(self.device))
            return v_norm * self.normalizer.v_std_global + self.normalizer.v_mean_global
