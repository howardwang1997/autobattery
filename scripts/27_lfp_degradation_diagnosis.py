"""Degradation diagnosis on NEWAREA experimental data using LFP FNO.

Fits LFP FNO to each discharge cycle, extracts degradation state parameters,
and tracks their evolution over 434 cycles.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import numpy as np
import h5py
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.loader import ExperimentalDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation")
OUT.mkdir(parents=True, exist_ok=True)


def extract_discharge_curves(data, min_pts=15):
    current = data["current"]
    voltage = data["voltage"]
    time_arr = data["time"]
    cycles = data["cycle"]
    unique_cycles = np.unique(cycles)
    curves = []

    for cyc in unique_cycles:
        mask = (cycles == cyc) & (current < -0.005)
        idx = np.where(mask)[0]
        if len(idx) < min_pts:
            continue
        v_seg, i_seg, t_seg = voltage[idx], current[idx], time_arr[idx]
        valid = ~np.isnan(v_seg)
        v_seg, i_seg, t_seg = v_seg[valid], i_seg[valid], t_seg[valid]
        if len(v_seg) < min_pts:
            continue
        dt = np.diff(t_seg)
        if dt.sum() < 10:
            continue
        cap = -np.sum(i_seg[:-1] * dt) / 3600
        if cap < 0.001:
            continue
        curves.append({"cycle": int(cyc), "voltage": v_seg, "capacity": cap, "i_mean": abs(i_seg.mean())})

    return curves


def main():
    device = torch.device("cuda:0")

    # Load LFP FNO
    # Load dataset for stats
    with h5py.File("data/fullfield/fullfield_lfp_degradation.h5", "r") as f:
        all_params = f["params"][:]
        all_V = f["V"][:]
        all_cr = f["c_rates"][:]
        param_names = [p.decode() if isinstance(p, bytes) else str(p) for p in f.attrs["param_names"]]
        n_time = int(f.attrs["n_time"])

    param_mean = all_params.mean(axis=0)
    param_std = all_params.std(axis=0) + 1e-12
    V_mean = float(all_V.mean())
    V_std = float(all_V.std()) + 1e-8
    n_params = len(param_names)

    # Build model manually
    class VoltageFNO(torch.nn.Module):
        def __init__(self, n_params, n_time, mid_channels=64, n_layers=4, n_modes=16):
            super().__init__()
            self.n_time = n_time
            self.mid_channels = mid_channels
            self.n_modes = n_modes
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

        def forward(self, params, c_rate):
            B = params.shape[0]
            c_r = c_rate.view(B, 1)
            cond = torch.cat([params, c_r], dim=-1)
            embed = self.param_embed(cond).unsqueeze(-1).expand(-1, -1, self.n_time)
            x = torch.zeros(B, self.mid_channels, self.n_time, device=params.device)
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
            return V_out

    model = VoltageFNO(n_params, n_time).to(device)
    ckpt = torch.load("outputs/checkpoints_lfp/fno_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded LFP FNO (epoch {ckpt['epoch']}, RMSE={ckpt.get('val_rmse_mV', '?')}mV)")

    # Load experimental data
    logger.info("Loading NEWAREA experimental data...")
    loader = ExperimentalDataLoader()
    data = loader.load_neware_xlsx("/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")
    curves = extract_discharge_curves(data)
    logger.info(f"Extracted {len(curves)} discharge curves")

    # Sample cycles: every 5th + key milestones
    sample = sorted(set(
        list(range(0, len(curves), 5)) +
        [0, 1, 2, len(curves) // 4, len(curves) // 2, 3 * len(curves) // 4, len(curves) - 2, len(curves) - 1]
    ))
    sample = [i for i in sample if i < len(curves)]
    logger.info(f"Fitting {len(sample)} cycles...")

    p_min = all_params.min(axis=0)
    p_max = all_params.max(axis=0)

    results = []
    for i, ci in enumerate(sample):
        cc = curves[ci]
        v_exp = cc["voltage"]
        c_rate = cc["i_mean"] / cc["capacity"] if cc["capacity"] > 0.001 else 1.0

        v_resamp = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(v_exp)), v_exp)
        v_target = torch.tensor((v_resamp - V_mean) / V_std, dtype=torch.float32, device=device).unsqueeze(0)

        best_loss = float("inf")
        best_params = None
        best_v = None
        c_rate_t = torch.tensor([[c_rate]], dtype=torch.float32, device=device)

        for restart in range(3):
            p0 = torch.randn(1, n_params, device=device) * 0.3 if restart > 0 else torch.zeros(1, n_params, device=device)
            p = p0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=0.02)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

            for step in range(400):
                V_pred = model(p, c_rate_t)
                loss = torch.nn.functional.mse_loss(V_pred, v_target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()

                # Clamp to bounds
                with torch.no_grad():
                    dp = p[0:1] * torch.tensor(param_std, device=device) + torch.tensor(param_mean, device=device)
                    for j in range(n_params):
                        dp[0, j].clamp_(p_min[j], p_max[j])
                    p.data = (dp - torch.tensor(param_mean, device=device)) / torch.tensor(param_std, device=device)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = p.detach().clone()
                    with torch.no_grad():
                        best_v = model(best_params, c_rate_t).cpu().numpy()[0].copy()

        # Convert to physical
        params_phys = best_params[0].cpu().numpy() * param_std + param_mean
        v_pred_phys = best_v * V_std + V_mean
        rmse = np.sqrt(np.mean((v_pred_phys - v_resamp) ** 2)) * 1000

        results.append({
            "cycle": cc["cycle"],
            "capacity": cc["capacity"],
            "c_rate": c_rate,
            "params": params_phys.copy(),
            "rmse_mV": rmse,
        })

        if (i + 1) % 10 == 0 or i < 5:
            short_names = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]
            vals = " ".join(f"{s}={params_phys[j]:.2e}" for j, s in enumerate(short_names))
            logger.info(f"  Cycle {cc['cycle']:3d}: RMSE={rmse:.1f}mV, {vals}")

    # Save and plot
    results.sort(key=lambda r: r["cycle"])
    cycles_arr = np.array([r["cycle"] for r in results])
    caps_arr = np.array([r["capacity"] for r in results])
    params_arr = np.array([r["params"] for r in results])
    rmse_arr = np.array([r["rmse_mV"] for r in results])

    np.savez(OUT / "lfp_degradation_trajectories.npz",
             cycles=cycles_arr, capacity=caps_arr, params=params_arr,
             rmse=rmse_arr, param_names=param_names)

    short_names = ["D_n", "D_p", "t⁺", "SEI thickness", "LAM (neg)", "LAM (pos)", "R multiplier"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for j in range(min(7, n_params)):
        ax = axes[j // 3, j % 3]
        vals = params_arr[:, j]
        if np.all(vals > 0) and vals.max() / vals.min() > 10:
            ax.semilogy(cycles_arr, vals, "o-", markersize=2)
        else:
            ax.plot(cycles_arr, vals, "o-", markersize=2)
        ax.set_xlabel("Cycle")
        ax.set_ylabel(short_names[j])

        # Correlation with capacity
        cap_norm = caps_arr / caps_arr[0]
        param_norm = vals / (vals[0] + 1e-30)
        r = np.corrcoef(param_norm, cap_norm)[0, 1]
        ax.set_title(f"{short_names[j]} (r={r:.3f})")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(cycles_arr, cap_norm * 100, "b--", alpha=0.3, linewidth=1)
        ax2.set_ylabel("Capacity (%)", color="b", alpha=0.3)

    # RMSE plot
    ax = axes[2, 1]
    ax.plot(cycles_arr, rmse_arr, "r-o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("RMSE (mV)")
    ax.set_title(f"Fitting Quality (mean={rmse_arr.mean():.1f}mV)")
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[2, 2]
    ax.axis("off")
    text = "DEGRADATION DIAGNOSIS\n" + "=" * 30 + "\n\n"
    text += f"Capacity: {caps_arr[0]*1000:.1f} → {caps_arr[-1]*1000:.1f} mAh\n"
    text += f"({caps_arr[-1]/caps_arr[0]*100:.1f}% retention)\n\n"
    for j in range(n_params):
        ratio = params_arr[-1, j] / (params_arr[0, j] + 1e-30)
        text += f"{short_names[j]}: {params_arr[0,j]:.2e} → {params_arr[-1,j]:.2e}\n  ({ratio:.2f}×)\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8, verticalalignment="top", fontfamily="monospace")

    fig.suptitle("LFP FNO Degradation Diagnosis — NEWAREA (434 cycles)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "lfp_degradation_diagnosis.png", dpi=150)
    plt.close(fig)

    logger.info(f"\n{'='*60}")
    logger.info("LFP DEGRADATION DIAGNOSIS RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Mean RMSE: {rmse_arr.mean():.1f} ± {rmse_arr.std():.1f} mV")
    logger.info(f"Capacity: {caps_arr[0]*1000:.1f} → {caps_arr[-1]*1000:.1f} mAh ({caps_arr[-1]/caps_arr[0]*100:.1f}%)")
    logger.info("\nParameter evolution:")
    for j in range(n_params):
        ratio = params_arr[-1, j] / (params_arr[0, j] + 1e-30)
        corr = np.corrcoef(params_arr[:, j] / (params_arr[0, j] + 1e-30), caps_arr / caps_arr[0])[0, 1]
        logger.info(f"  {short_names[j]:12s}: {params_arr[0,j]:.3e} → {params_arr[-1,j]:.3e} ({ratio:.2f}×, r={corr:.3f})")

    logger.info(f"\nOutputs: {OUT}/lfp_degradation_diagnosis.png")


if __name__ == "__main__":
    main()
