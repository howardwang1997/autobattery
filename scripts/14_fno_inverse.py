"""Parameter identification using trained FNO as forward model."""

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.operator.fno import FNO2d
from src.operator.dataset import FullFieldDataset
from src.simulation.solver import PybammSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fno_final.pt")
    parser.add_argument("--data", type=str, default="data/fullfield/fullfield_lmb.h5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-tests", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/fno_inverse")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FullFieldDataset(args.data, fields=["c_e", "phi_e"], normalize=True)
    stats = dataset._stats

    model = FNO2d(
        num_params=7, in_channels=2, out_channels=2, mid_channels=64,
        num_layers=4, modes1=min(16, dataset.nx_full), modes2=min(32, dataset.n_time),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded FNO from epoch {ckpt['epoch']}")

    param_names = dataset.param_names
    print(f"Parameters: {param_names}")

    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]

    results = []
    n_tests = min(args.n_tests, len(dataset.val_idx))

    for test_i in range(n_tests):
        idx = dataset.val_idx[test_i]
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        batch = dataset[idx]

        gt_params_norm = batch["params"].numpy()
        gt_params = gt_params_norm * param_std + param_mean
        coord = batch["coord"].unsqueeze(0).to(device)
        v_target = batch["voltage"].unsqueeze(0).to(device)
        c_rate = batch["c_rate"].unsqueeze(0).to(device)
        fields_target = batch["fields"].unsqueeze(0).to(device)

        pred_params_norm = torch.randn(1, 7, device=device) * 0.1
        pred_params_norm.requires_grad_(True)

        optimizer = torch.optim.Adam([pred_params_norm], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

        best_loss = float("inf")
        best_params = pred_params_norm.detach().clone()

        t0 = time.time()
        for step in range(args.steps):
            fields_pred, v_pred = model(coord, pred_params_norm, c_rate)

            v_loss = torch.nn.functional.mse_loss(v_pred, v_target)
            field_loss = torch.nn.functional.mse_loss(fields_pred, fields_target)
            loss = v_loss + 0.1 * field_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = pred_params_norm.detach().clone()

        elapsed = time.time() - t0
        recovered_norm = best_params.cpu().numpy()[0]
        recovered = recovered_norm * param_std + param_mean

        rel_err = np.abs(recovered - gt_params) / (np.abs(gt_params) + 1e-10) * 100
        mean_rel_err = rel_err.mean()

        result = {
            "gt_params": gt_params,
            "recovered": recovered,
            "rel_err": rel_err,
            "mean_rel_err": mean_rel_err,
            "best_loss": best_loss,
            "time": elapsed,
            "c_rate": batch["c_rate"].item(),
        }
        results.append(result)

        short_names = [n[:12] for n in param_names]
        err_str = "  ".join([f"{n}={e:.1f}%" for n, e in zip(short_names, rel_err)])
        print(f"Test {test_i}: C={result['c_rate']:.2f}, mean_err={mean_rel_err:.1f}%, time={elapsed:.2f}s | {err_str}")

    mean_errs = np.array([r["rel_err"] for r in results])
    print(f"\n--- Summary ({n_tests} tests) ---")
    print(f"Mean parameter error: {mean_errs.mean():.1f}%")
    for j, name in enumerate(param_names):
        print(f"  {name:30s}: {mean_errs[:, j].mean():.1f} ± {mean_errs[:, j].std():.1f}%")
    print(f"Avg time: {np.mean([r['time'] for r in results]):.2f}s")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for j, name in enumerate(param_names):
        ax = axes[j]
        gt_vals = [r["gt_params"][j] for r in results]
        pred_vals = [r["recovered"][j] for r in results]
        ax.scatter(gt_vals, pred_vals, alpha=0.7, s=60)
        lo = min(min(gt_vals), min(pred_vals))
        hi = max(max(gt_vals), max(pred_vals))
        margin = (hi - lo) * 0.1 + 1e-10
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", alpha=0.5)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Recovered")
        ax.set_title(f"{name}\n(err: {mean_errs[:, j].mean():.1f}%)")
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle(f"FNO-based Parameter Identification ({n_tests} tests, mean err: {mean_errs.mean():.1f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "param_identification.png", dpi=150)
    print(f"\nSaved to {out_dir / 'param_identification.png'}")
    plt.close(fig)

    np.savez(out_dir / "inverse_results.npz", results=results, param_names=param_names)


if __name__ == "__main__":
    main()
