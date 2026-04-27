"""Pipeline 4: Experimental validation on NEWARE cycling data.

Uses ExperimentalDataLoader.load_neware_xlsx() which correctly reads
the 'record' sheet with Chinese column headers.

Runs on GPU 3.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import numpy as np
import time
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [P4] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/root/data/raw/exapmles")


def load_all_neware():
    from src.data.loader import ExperimentalDataLoader

    loader = ExperimentalDataLoader()
    datasets = {}

    for fpath in sorted(DATA_DIR.glob("*.xlsx")):
        logger.info(f"Loading {fpath.name}...")
        try:
            data = loader.load_neware_xlsx(fpath)
            n = len(data["time"])
            logger.info(f"  {n} pts, V=[{data['voltage'].min():.3f}, {data['voltage'].max():.3f}]V, "
                       f"I=[{data['current'].min():.4f}, {data['current'].max():.4f}]A, "
                       f"cycles={int(data['cycle'].max())}")
            datasets[fpath.stem] = data
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    return datasets


def extract_discharge_curves(data, min_pts=20):
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

        is_discharge = c_seg < -0.01 * np.abs(c_seg).max()
        segments = np.split(np.arange(len(c_seg)), np.where(np.diff(is_discharge.astype(int)))[0] + 1)

        for seg in segments:
            if len(seg) < min_pts:
                continue
            if not is_discharge[seg[0]]:
                continue

            v_s = v_seg[seg]
            i_s = c_seg[seg]
            t_s = t_seg[seg]

            valid = ~np.isnan(v_s)
            v_s, i_s, t_s = v_s[valid], i_s[valid], t_s[valid]
            if len(v_s) < min_pts:
                continue

            curves.append({
                "cycle": int(cyc),
                "voltage": v_s,
                "current": i_s,
                "time": t_s,
                "i_mean": np.abs(i_s.mean()),
            })

    return curves


def fit_with_fno(curves_by_file):
    from src.operator.fno import FNO2d
    from src.operator.dataset import FullFieldDataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = FullFieldDataset("data/fullfield/fullfield_lmb_v2.h5", fields=["c_e", "phi_e"], normalize=True)
    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]
    param_names = dataset.param_names
    n_time = dataset.n_time
    nx = dataset.nx_full

    model = FNO2d(
        num_params=len(param_names), in_channels=2, out_channels=2,
        mid_channels=64, num_layers=4,
        modes1=min(16, nx), modes2=min(32, n_time),
    ).to(device)
    ckpt = torch.load("outputs/checkpoints/fno_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t_grid = np.linspace(0, 1, n_time, dtype=np.float32)
    x_grid = np.linspace(0, 1, nx, dtype=np.float32)
    T, X = np.meshgrid(t_grid, x_grid)
    coord_t = torch.tensor(np.stack([X, T], axis=0), dtype=torch.float32, device=device).unsqueeze(0)

    out_dir = Path("outputs/experimental")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for fname, curves in curves_by_file.items():
        logger.info(f"\nFitting {fname} ({len(curves)} curves)...")

        for ci, cc in enumerate(curves[:5]):
            v_exp = cc["voltage"]
            if len(v_exp) < 20:
                continue

            v_resamp = np.interp(np.linspace(0, 1, n_time), np.linspace(0, 1, len(v_exp)), v_exp)
            v_target = torch.tensor(
                (v_resamp - stats["V"]["mean"]) / stats["V"]["std"],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)

            best_fit = None
            best_rmse = np.inf

            for c_rate_guess in [0.1, 0.2, 0.5, 1.0, 2.0]:
                c_rate_t = torch.tensor([[c_rate_guess]], dtype=torch.float32, device=device)
                pred_p = torch.zeros(1, len(param_names), device=device, requires_grad=True)
                opt = torch.optim.Adam([pred_p], lr=0.01)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
                best_l, best_pp = float("inf"), pred_p.detach().clone()

                for step in range(1000):
                    _, vp = model(coord_t, pred_p, c_rate_t)
                    loss = torch.nn.functional.mse_loss(vp, v_target)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    if loss.item() < best_l:
                        best_l = loss.item()
                        best_pp = pred_p.detach().clone()

                with torch.no_grad():
                    _, v_pred = model(coord_t, best_pp, c_rate_t)
                v_pred_d = v_pred.cpu().numpy()[0] * stats["V"]["std"] + stats["V"]["mean"]
                rmse = np.sqrt(np.mean((v_pred_d - v_resamp) ** 2)) * 1000

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_fit = {
                        "v_pred": v_pred_d, "v_exp": v_resamp,
                        "params": best_pp.cpu().numpy()[0] * param_std + param_mean,
                        "c_rate": c_rate_guess, "rmse_mV": rmse,
                    }

            if best_fit:
                logger.info(f"  Cycle {cc['cycle']}: C={best_fit['c_rate']:.1f}, RMSE={best_fit['rmse_mV']:.1f}mV")
                all_results.append(best_fit)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(np.linspace(0, 1, n_time), best_fit["v_exp"], "b-", label="Experimental", linewidth=2)
                ax.plot(np.linspace(0, 1, n_time), best_fit["v_pred"], "r--", label="FNO", linewidth=2)
                ax.set_xlabel("Normalized time")
                ax.set_ylabel("Voltage (V)")
                ax.set_title(f"{fname} Cycle {cc['cycle']}: RMSE={best_fit['rmse_mV']:.1f}mV")
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.savefig(out_dir / f"{fname}_cycle{cc['cycle']}.png", dpi=150)
                plt.close(fig)

    if all_results:
        rmses = [r["rmse_mV"] for r in all_results]
        logger.info(f"\nOverall: {len(all_results)} curves fitted, RMSE={np.mean(rmses):.1f}±{np.std(rmses):.1f}mV")


def main():
    logger.info("=" * 60)
    logger.info("Pipeline 4: Experimental Validation (NEWARE)")
    logger.info("=" * 60)

    datasets = load_all_neware()
    if not datasets:
        logger.error("No data loaded!")
        return

    curves_by_file = {}
    for name, data in datasets.items():
        curves = extract_discharge_curves(data)
        if curves:
            curves_by_file[name] = curves
            logger.info(f"{name}: {len(curves)} discharge curves")
            for cc in curves[:3]:
                logger.info(f"  Cycle {cc['cycle']}: {len(cc['voltage'])} pts, V=[{cc['voltage'].min():.3f}, {cc['voltage'].max():.3f}]")

    if not curves_by_file:
        logger.warning("No discharge curves extracted!")
        return

    fit_with_fno(curves_by_file)
    logger.info("Pipeline 4 COMPLETE!")


if __name__ == "__main__":
    main()
