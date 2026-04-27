"""Battery design optimization using existing LMB FNO as forward model.

Treats material parameters as design variables and finds combinations
that maximize energy density (voltage × capacity proxy).
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda:0")

    from src.operator.fno import FNO2d
    from src.operator.dataset import FullFieldDataset

    out_dir = Path("outputs/design")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FullFieldDataset("data/fullfield/fullfield_lmb_v2.h5", fields=["c_e", "phi_e"], normalize=True)
    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]
    param_names = dataset.param_names
    n_params = len(param_names)
    n_time = dataset.n_time
    nx = dataset.nx_full

    model = FNO2d(
        num_params=n_params, in_channels=2, out_channels=2,
        mid_channels=64, num_layers=4,
        modes1=min(16, nx), modes2=min(32, n_time),
    ).to(device)
    ckpt = torch.load("outputs/checkpoints/fno_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = dataset[0]
    coord = batch["coord"].unsqueeze(0).to(device)

    with h5py.File("data/fullfield/fullfield_lmb_v2.h5", "r") as f:
        all_params = f["params"][:]
        all_V = f["V"][:]
        all_cr = f["c_rates"][:]
    p_min = all_params.min(axis=0)
    p_max = all_params.max(axis=0)

    results = []
    c_rate_1c = torch.tensor([[1.0]], device=device)
    c_rate_05c = torch.tensor([[0.5]], device=device)

    for trial in range(50):
        design_norm = torch.randn(1, n_params, device=device) * 0.3
        design_norm.requires_grad_(True)
        opt = torch.optim.Adam([design_norm], lr=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

        for step in range(200):
            _, v_05c = model(coord, design_norm, c_rate_05c)
            _, v_1c = model(coord, design_norm, c_rate_1c)
            loss = -(v_05c.mean() + 0.5 * v_1c.mean())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            with torch.no_grad():
                dp = design_norm * torch.tensor(param_std, device=device) + torch.tensor(param_mean, device=device)
                for j in range(n_params):
                    dp[0, j].clamp_(p_min[j], p_max[j])
                design_norm.data = (dp - torch.tensor(param_mean, device=device)) / torch.tensor(param_std, device=device)

        with torch.no_grad():
            _, v_final = model(coord, design_norm, c_rate_1c)
            dp_final = design_norm[0].cpu().numpy() * param_std + param_mean

        results.append({
            "design": dp_final.copy(),
            "energy_proxy": v_final.mean().item(),
        })

        if (trial + 1) % 10 == 0:
            best = max(r["energy_proxy"] for r in results)
            logger.info(f"Trial {trial+1}/50, best energy_proxy={best:.4f}")

    results.sort(key=lambda r: r["energy_proxy"], reverse=True)

    logger.info("\n=== Top 10 parameter combinations (highest energy) ===")
    for i, r in enumerate(results[:10]):
        logger.info(f"#{i+1}: energy_proxy={r['energy_proxy']:.4f}")
        for j, pn in enumerate(param_names):
            short = pn.split("[")[0].strip()[:30]
            logger.info(f"  {short}: {r['design'][j]:.4e}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    for j in range(min(7, n_params)):
        ax = axes[j]
        vals = [r["design"][j] for r in results]
        energies = [r["energy_proxy"] for r in results]
        sc = ax.scatter(vals, energies, alpha=0.4, s=15, c=energies, cmap="viridis")
        ax.set_xlabel(param_names[j].split("[")[0].strip()[:25], fontsize=8)
        ax.set_ylabel("Energy proxy")
        ax.set_title(param_names[j].split("[")[0].strip()[:25], fontsize=9)
        plt.colorbar(sc, ax=ax, shrink=0.6)
    axes[7].hist([r["energy_proxy"] for r in results], bins=30, alpha=0.7)
    axes[7].set_xlabel("Energy proxy")
    axes[7].set_title("Distribution")
    fig.suptitle("Battery Design Optimization via FNO", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "optimization_results.png", dpi=150)
    plt.close(fig)

    np.savez(out_dir / "optimization_results.npz", results=results, param_names=param_names, allow_pickle=True)
    logger.info(f"Saved to {out_dir}/optimization_results.png and optimization_results.npz")
    logger.info("Design optimization complete!")


if __name__ == "__main__":
    main()
