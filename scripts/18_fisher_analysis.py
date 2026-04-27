"""Standalone Fisher information analysis script.

Can be called standalone or from pipeline1.
"""

import os
import sys
import torch
import numpy as np
import time
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Fisher] %(message)s")
logger = logging.getLogger(__name__)


def compute_fisher_matrix(model, dataset, device, n_samples=200):
    """Compute Fisher Information Matrix via finite-difference Jacobian."""
    stats = dataset._stats
    n_params = len(dataset.param_names)
    eps = 0.01

    fisher_V = torch.zeros(n_params, n_params, device=device)
    fisher_ce = torch.zeros(n_params, n_params, device=device)
    fisher_phie = torch.zeros(n_params, n_params, device=device)

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for si, idx in enumerate(indices):
        batch = dataset[idx]
        coord = batch["coord"].unsqueeze(0).to(device)
        params_base = batch["params"].unsqueeze(0).to(device)
        c_rate = batch["c_rate"].unsqueeze(0).to(device)

        with torch.no_grad():
            J_V, J_ce, J_phie = [], [], []

            for j in range(n_params):
                p_p = params_base.clone(); p_p[0, j] += eps
                p_m = params_base.clone(); p_m[0, j] -= eps

                fp, vp = model(coord, p_p, c_rate)
                fm, vm = model(coord, p_m, c_rate)

                J_V.append(((vp - vm) / (2 * eps)).flatten())
                J_ce.append(((fp[0, 0] - fm[0, 0]) / (2 * eps)).flatten())
                J_phie.append(((fp[0, 1] - fm[0, 1]) / (2 * eps)).flatten())

            J_V = torch.stack(J_V)
            J_ce = torch.stack(J_ce)
            J_phie = torch.stack(J_phie)

            fisher_V += J_V @ J_V.T
            fisher_ce += J_ce @ J_ce.T
            fisher_phie += J_phie @ J_phie.T

        if (si + 1) % 50 == 0:
            logger.info(f"  {si+1}/{len(indices)}")

    N = len(indices)
    return {"V": (fisher_V / N).cpu().numpy(), "c_e": (fisher_ce / N).cpu().numpy(), "phi_e": (fisher_phie / N).cpu().numpy()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fullfield/fullfield_lmb_v2.h5")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fno_final.pt")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.operator.fno import FNO2d
    from src.operator.dataset import FullFieldDataset

    dataset = FullFieldDataset(args.data, fields=["c_e", "phi_e"], normalize=True)
    param_names = dataset.param_names

    model = FNO2d(
        num_params=len(param_names), in_channels=2, out_channels=2,
        mid_channels=64, num_layers=4,
        modes1=min(16, dataset.nx_full), modes2=min(32, dataset.n_time),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"Computing Fisher information ({args.n_samples} samples)...")
    t0 = time.time()
    fisher = compute_fisher_matrix(model, dataset, device, args.n_samples)
    logger.info(f"Took {time.time()-t0:.0f}s")

    out_dir = Path("outputs/fisher")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Analysis
    for obs_name, fim in fisher.items():
        diag = np.diag(fim)
        diag_norm = diag / (diag.max() + 1e-20)
        eigvals = np.linalg.eigvalsh(fim)
        eigvals = eigvals[eigvals > 0]
        cond = eigvals.max() / (eigvals.min() + 1e-20) if len(eigvals) > 0 else np.inf

        logger.info(f"\n=== {obs_name} === condition={cond:.2e}")
        for j, pn in enumerate(param_names):
            logger.info(f"  {pn[:35]:<36}: {diag_norm[j]:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (obs_name, fim) in enumerate(fisher.items()):
        diag = np.diag(fim)
        diag_norm = diag / (diag.max() + 1e-20)
        axes[i].bar(range(len(param_names)), diag_norm)
        axes[i].set_xticks(range(len(param_names)))
        axes[i].set_xticklabels([n[:12] for n in param_names], rotation=45, ha='right', fontsize=7)
        axes[i].set_title(f"Fisher Info: {obs_name}")
        axes[i].set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_dir / "fisher_analysis.png", dpi=150)
    plt.close(fig)

    np.savez(out_dir / "fisher_matrices.npz", **fisher, param_names=param_names, allow_pickle=True)
    logger.info("Fisher analysis complete!")


if __name__ == "__main__":
    main()
