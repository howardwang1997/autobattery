"""Cycle-by-cycle degradation diagnosis using differentiable FNO.

For each discharge cycle in experimental data:
1. Extract V(t) curve
2. Fit LIB FNO parameters via gradient optimization
3. Track D_n, D_p, t⁺ evolution over cycling → degradation mechanisms
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
from src.operator.fno import FNO2d
from src.operator.dataset import FullFieldDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def extract_all_discharge_curves(data, min_pts=15):
    current = data["current"]
    voltage = data["voltage"]
    time_arr = data["time"]
    cycles = data["cycle"]

    unique_cycles = np.unique(cycles)
    curves = []

    for cyc in unique_cycles:
        mask = cycles == cyc
        idx = np.where(mask)[0]
        if len(idx) < min_pts:
            continue

        c_seg = current[idx]
        v_seg = voltage[idx]
        t_seg = time_arr[idx]

        is_discharge = c_seg < -0.005 * np.abs(c_seg).max()
        if not is_discharge.any():
            continue

        changes = np.where(np.diff(is_discharge.astype(int)))[0] + 1
        segments = np.split(np.arange(len(c_seg)), changes)

        longest_seg = None
        longest_len = 0
        for seg in segments:
            if len(seg) > longest_len and is_discharge[seg[0]]:
                longest_seg = seg
                longest_len = len(seg)

        if longest_seg is None or longest_len < min_pts:
            continue

        v_s = v_seg[longest_seg]
        i_s = c_seg[longest_seg]
        t_s = t_seg[longest_seg]

        valid = ~np.isnan(v_s) & ~np.isnan(i_s)
        v_s, i_s, t_s = v_s[valid], i_s[valid], t_s[valid]
        if len(v_s) < min_pts:
            continue

        dt = np.diff(t_s)
        if dt.sum() < 10:
            continue
        cap = -np.sum(i_s[:-1] * dt) / 3600

        curves.append({
            "cycle": int(cyc),
            "voltage": v_s.copy(),
            "current": i_s.copy(),
            "time": t_s.copy(),
            "capacity_Ah": cap,
            "i_mean": np.abs(i_s.mean()),
        })

    return curves


def fit_cycle(model, coord_t, v_target, stats, param_mean, param_std,
              param_bounds, c_rate, device, n_restarts=3, steps=800):
    best_loss = float("inf")
    best_params = None
    best_v_pred = None

    c_rate_t = torch.tensor([[c_rate]], dtype=torch.float32, device=device)

    for restart in range(n_restarts):
        if restart == 0:
            p = torch.zeros(1, param_mean.shape[0], device=device, requires_grad=True)
        else:
            p = torch.randn(1, param_mean.shape[0], device=device) * 0.5
            p.requires_grad_(True)

        opt = torch.optim.Adam([p], lr=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        for step in range(steps):
            _, v_pred = model(coord_t, p, c_rate_t)
            loss = torch.nn.functional.mse_loss(v_pred, v_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = p.detach().clone()
                with torch.no_grad():
                    _, v_best = model(coord_t, best_params, c_rate_t)
                    best_v_pred = v_best.cpu().numpy()[0].copy()

    if best_params is not None:
        params_phys = best_params[0].cpu().numpy() * param_std + param_mean
        for j in range(len(params_phys)):
            params_phys[j] = np.clip(params_phys[j], param_bounds[0][j], param_bounds[1][j])
    else:
        params_phys = param_mean.copy()

    v_target_np = v_target.cpu().numpy()[0] * stats["V"]["std"] + stats["V"]["mean"]
    if best_v_pred is not None:
        v_pred_phys = best_v_pred * stats["V"]["std"] + stats["V"]["mean"]
        rmse = np.sqrt(np.mean((v_pred_phys - v_target_np) ** 2)) * 1000
    else:
        rmse = float("inf")

    return params_phys, rmse, best_loss


def main():
    device = torch.device("cuda:0")
    out_dir = Path("outputs/degradation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LIB FNO
    logger.info("Loading LIB FNO...")
    dataset = FullFieldDataset("data/fullfield/fullfield_lib.h5", fields=["c_e", "phi_e"], normalize=True)
    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]
    param_names = dataset.param_names
    n_params = len(param_names)
    n_time = dataset.n_time
    nx = dataset.nx_full

    with h5py.File("data/fullfield/fullfield_lib.h5", "r") as f:
        all_params = f["params"][:]
    p_min = all_params.min(axis=0)
    p_max = all_params.max(axis=0)

    model = FNO2d(
        num_params=n_params, in_channels=2, out_channels=2,
        mid_channels=64, num_layers=4,
        modes1=min(16, nx), modes2=min(32, n_time),
    ).to(device)
    ckpt = torch.load("outputs/checkpoints_lib/fno_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = dataset[0]
    coord_t = batch["coord"].unsqueeze(0).to(device)

    # Load experimental data
    logger.info("Loading NEWAREA data...")
    loader = ExperimentalDataLoader()
    data = loader.load_neware_xlsx("/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")

    logger.info("Extracting discharge curves...")
    curves = extract_all_discharge_curves(data)
    logger.info(f"Extracted {len(curves)} discharge curves")

    # Sample key cycles for speed
    n_curves = len(curves)
    sample_indices = sorted(set(
        list(range(0, n_curves, max(1, n_curves // 30))) + 
        [0, 1, 2, n_curves // 4, n_curves // 2, 3 * n_curves // 4, n_curves - 2, n_curves - 1]
    ))
    sample_indices = [i for i in sample_indices if i < n_curves]

    logger.info(f"Fitting {len(sample_indices)} sampled cycles...")

    results = []
    for i, ci in enumerate(sample_indices):
        cc = curves[ci]
        v_exp = cc["voltage"]
        c_rate_exp = cc["i_mean"] / cc["capacity_Ah"] if cc["capacity_Ah"] > 0.001 else 1.0

        v_resamp = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(v_exp)), v_exp)
        v_target = torch.tensor(
            (v_resamp - stats["V"]["mean"]) / stats["V"]["std"],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

        params, rmse, loss = fit_cycle(
            model, coord_t, v_target, stats,
            param_mean, param_std, (p_min, p_max),
            c_rate=c_rate_exp, device=device,
            n_restarts=1, steps=300,
        )

        results.append({
            "cycle": cc["cycle"],
            "capacity_Ah": cc["capacity_Ah"],
            "c_rate": c_rate_exp,
            "params": params,
            "rmse_mV": rmse,
            "loss": loss,
        })

        if (i + 1) % 10 == 0 or i < 3:
            logger.info(
                f"  Cycle {cc['cycle']:3d}: Q={cc['capacity_Ah']:.4f} Ah, "
                f"C={c_rate_exp:.2f}, RMSE={rmse:.1f} mV, "
                f"D_n={params[0]:.2e}, D_p={params[1]:.2e}, t+={params[2]:.4f}"
            )

    results.sort(key=lambda r: r["cycle"])

    # Save results
    cycles_arr = np.array([r["cycle"] for r in results])
    caps_arr = np.array([r["capacity_Ah"] for r in results])
    params_arr = np.array([r["params"] for r in results])
    rmse_arr = np.array([r["rmse_mV"] for r in results])

    np.savez(
        out_dir / "degradation_trajectories.npz",
        cycles=cycles_arr, capacity=caps_arr, params=params_arr,
        rmse=rmse_arr, param_names=param_names,
    )
    logger.info(f"Saved {out_dir}/degradation_trajectories.npz")

    # Plot degradation trajectories
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 1: Capacity fade + RMSE
    ax = axes[0, 0]
    ax.plot(cycles_arr, caps_arr / caps_arr[0] * 100, "b-o", markersize=3)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity retention (%)")
    ax.set_title("Capacity Fade")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cycles_arr, rmse_arr, "r-o", markersize=3)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RMSE (mV)")
    ax.set_title("Fitting Quality")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(cycles_arr, np.array([r["c_rate"] for r in results]), "g-o", markersize=3)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("C-rate")
    ax.set_title("Discharge C-rate")
    ax.grid(True, alpha=0.3)

    # Row 2-3: Parameter trajectories (normalized to initial value)
    identifiable = [0, 1, 2]  # D_n, D_p, t+
    for row in range(1, 3):
        for col in range(3):
            idx = (row - 1) * 3 + col
            ax = axes[row, col]
            if idx < n_params:
                vals = params_arr[:, idx]
                short_name = param_names[idx].split("[")[0].strip()[:25]
                ax.semilogy(cycles_arr, np.abs(vals) + 1e-30, "o-", markersize=3)
                ax.set_xlabel("Cycle")
                ax.set_ylabel(short_name)
                ax.set_title(short_name)
                ax.grid(True, alpha=0.3)

                # Add capacity on twin axis
                ax2 = ax.twinx()
                ax2.plot(cycles_arr, caps_arr / caps_arr[0] * 100, "b--", alpha=0.3, linewidth=1)
                ax2.set_ylabel("Capacity (%)", color="b", alpha=0.3)

    fig.suptitle("Battery Degradation Diagnosis via Differentiable FNO\nNEWAREA Cell (434 cycles, 62.8% fade)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_trajectories.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_dir}/degradation_trajectories.png")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DEGRADATION DIAGNOSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total cycles analyzed: {len(results)}")
    logger.info(f"Capacity: {caps_arr[0]:.4f} → {caps_arr[-1]:.4f} Ah ({caps_arr[-1]/caps_arr[0]*100:.1f}%)")
    logger.info(f"Mean RMSE: {rmse_arr.mean():.1f} ± {rmse_arr.std():.1f} mV")
    logger.info("\nParameter evolution (first → last):")
    for j in range(min(7, n_params)):
        short = param_names[j].split("[")[0].strip()[:35]
        ratio = params_arr[-1, j] / (params_arr[0, j] + 1e-30)
        logger.info(f"  {short}: {params_arr[0,j]:.3e} → {params_arr[-1,j]:.3e} ({ratio:.2f}x)")

    # Correlation with capacity
    logger.info("\nCorrelation with capacity fade:")
    cap_norm = caps_arr / caps_arr[0]
    for j in range(min(7, n_params)):
        short = param_names[j].split("[")[0].strip()[:30]
        param_norm = params_arr[:, j] / (params_arr[0, j] + 1e-30)
        corr = np.corrcoef(param_norm, cap_norm)[0, 1]
        logger.info(f"  {short}: r = {corr:.3f}")

    logger.info("\nDegradation diagnosis complete!")


if __name__ == "__main__":
    main()
