"""Train FNO on LFP degradation data (voltage-only)."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import numpy as np
import h5py
import logging
import time
from pathlib import Path
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class LFPDegDataset(Dataset):
    def __init__(self, h5_path, normalize=True):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.n_sims = int(f.attrs["n_sims"])
            self.n_time = int(f.attrs["n_time"])
            self.param_names = [p.decode() if isinstance(p, bytes) else p for p in f.attrs["param_names"]]
            self.params = f["params"][:].astype(np.float32)
            self.c_rates = f["c_rates"][:].astype(np.float32)
            self.V = f["V"][:].astype(np.float32)

        self.n_params = self.params.shape[1]
        self._stats = None
        if normalize:
            self._stats = self._compute_stats()

        perm = torch.randperm(self.n_sims)
        n_val = max(1, self.n_sims // 10)
        self.train_idx = perm[n_val:]
        self.val_idx = perm[:n_val]

    def _compute_stats(self):
        return {
            "params": {"mean": self.params.mean(0), "std": self.params.std(0) + 1e-12},
            "V": {"mean": float(self.V.mean()), "std": float(self.V.std()) + 1e-8},
            "c_rates": {"mean": float(self.c_rates.mean()), "std": float(self.c_rates.std()) + 1e-8},
        }

    def __len__(self):
        return self.n_sims

    def __getitem__(self, idx):
        params = self.params[idx].copy()
        c_rate = self.c_rates[idx]
        V = self.V[idx].copy()

        if self._stats:
            ps = self._stats["params"]
            params = (params - ps["mean"]) / ps["std"]
            V = (V - self._stats["V"]["mean"]) / self._stats["V"]["std"]

        t_grid = np.linspace(0, 1, self.n_time, dtype=np.float32)
        x_grid = np.array([0.5], dtype=np.float32)
        T, X = np.meshgrid(t_grid, x_grid)
        coord = np.stack([X, T], axis=0)

        # Voltage as 1x1xT field
        V_field = V.reshape(1, 1, self.n_time)

        return {
            "coord": torch.from_numpy(coord),
            "fields": torch.from_numpy(V_field),
            "params": torch.from_numpy(params),
            "c_rate": torch.tensor([c_rate], dtype=torch.float32),
            "voltage": torch.from_numpy(V),
        }


class VoltageFNO(torch.nn.Module):
    """Simple 1D FNO for voltage prediction from parameters."""

    def __init__(self, n_params, n_time, mid_channels=64, n_layers=4, n_modes=16):
        super().__init__()
        self.n_time = n_time
        self.mid_channels = mid_channels

        self.param_embed = torch.nn.Linear(n_params + 1, mid_channels)
        self.lifting = torch.nn.Conv1d(1, mid_channels, 1)

        self.fno_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.fno_layers.append(torch.nn.ModuleDict({
                "spectral": torch.nn.ModuleDict({
                    "real": torch.nn.Linear(n_modes, n_modes, bias=False),
                    "imag": torch.nn.Linear(n_modes, n_modes, bias=False),
                }),
                "local": torch.nn.Conv1d(mid_channels, mid_channels, 1),
                "norm": torch.nn.LayerNorm(mid_channels),
            }))

        self.projection = torch.nn.Sequential(
            torch.nn.Conv1d(mid_channels, mid_channels * 2, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(mid_channels * 2, 1, 1),
        )
        self.n_modes = n_modes

    def forward(self, coord, params, c_rate):
        B = params.shape[0]
        cond = torch.cat([params, c_rate], dim=-1)
        embed = self.param_embed(cond).unsqueeze(-1).expand(-1, -1, self.n_time)

        x = self.lifting(coord[:, 0:1, 0:1, :].reshape(B, 1, self.n_time))
        x = x + embed

        for layer in self.fno_layers:
            residual = x
            x_ft = torch.fft.rfft(x, dim=-1)
            modes = min(self.n_modes, x_ft.shape[-1])
            x_ft_cut = x_ft[:, :, :modes]
            real = layer["spectral"]["real"](x_ft_cut.real)
            imag = layer["spectral"]["imag"](x_ft_cut.imag)
            x_ft_out = torch.zeros_like(x_ft)
            x_ft_out[:, :, :modes] = torch.complex(real, imag)
            x_spectral = torch.fft.irfft(x_ft_out, n=self.n_time, dim=-1)

            x_local = layer["local"](x)
            x = x_spectral + x_local + residual
            x = x.permute(0, 2, 1)
            x = layer["norm"](x)
            x = x.permute(0, 2, 1)
            x = torch.nn.functional.gelu(x)

        V_out = self.projection(x).squeeze(1)
        return None, V_out


def train():
    device = torch.device("cuda:0")

    dataset = LFPDegDataset("data/fullfield/fullfield_lfp_degradation.h5")
    stats = dataset._stats
    n_params = dataset.n_params
    n_time = dataset.n_time
    logger.info(f"Dataset: {len(dataset)} sims, {n_params} params, {n_time} time pts")
    logger.info(f"Params: {dataset.param_names}")

    model = VoltageFNO(n_params, n_time, mid_channels=64, n_layers=4, n_modes=16).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_p:,} params")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, dataset.train_idx),
        batch_size=32, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, dataset.val_idx),
        batch_size=64, shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    out_dir = Path("outputs/checkpoints_lfp")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(300):
        model.train()
        train_loss = 0
        for batch in train_loader:
            coord = batch["coord"].to(device)
            params = batch["params"].to(device)
            c_rate = batch["c_rate"].to(device)
            V_target = batch["voltage"].to(device)

            _, V_pred = model(coord, params, c_rate)
            loss = torch.nn.functional.mse_loss(V_pred, V_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    _, V_pred = model(batch["coord"].to(device), batch["params"].to(device), batch["c_rate"].to(device))
                    val_loss += torch.nn.functional.mse_loss(V_pred, batch["voltage"].to(device)).item()
            val_loss /= len(val_loader)
            val_rmse = np.sqrt(val_loss) * stats["V"]["std"] * 1000

            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}, RMSE={val_rmse:.1f}mV")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rmse_mV": val_rmse,
                }, out_dir / "fno_best.pt")

    # Save final
    torch.save({
        "epoch": 300,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val,
    }, out_dir / "fno_final.pt")
    logger.info(f"Training complete. Best val RMSE saved to {out_dir}/fno_best.pt")

    # Final evaluation
    model.eval()
    all_err = []
    with torch.no_grad():
        for batch in val_loader:
            _, V_pred = model(batch["coord"].to(device), batch["params"].to(device), batch["c_rate"].to(device))
            V_p = V_pred.cpu().numpy() * stats["V"]["std"] + stats["V"]["mean"]
            V_t = batch["voltage"].numpy() * stats["V"]["std"] + stats["V"]["mean"]
            all_err.append(np.sqrt(np.mean((V_p - V_t) ** 2, axis=1)) * 1000)
    all_err = np.concatenate(all_err)
    logger.info(f"Final: RMSE = {all_err.mean():.1f} ± {all_err.std():.1f} mV")


if __name__ == "__main__":
    train()
