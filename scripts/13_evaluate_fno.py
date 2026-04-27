"""Evaluate trained FNO: visualize predictions, compute errors, benchmark speed."""

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


def denorm(x, mean, std):
    return x * std + mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fno_final.pt")
    parser.add_argument("--data", type=str, default="data/fullfield/fullfield_lmb.h5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/fno_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FullFieldDataset(args.data, fields=["c_e", "phi_e"], normalize=True)

    model = FNO2d(
        num_params=7,
        in_channels=2,
        out_channels=2,
        mid_channels=64,
        num_layers=4,
        modes1=min(16, dataset.nx_full),
        modes2=min(32, dataset.n_time),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    val_idx = dataset.val_idx[:args.n_samples]
    stats = dataset._stats

    fig_v, axes_v = plt.subplots(args.n_samples, 1, figsize=(12, 3 * args.n_samples))
    if args.n_samples == 1:
        axes_v = [axes_v]

    field_names = dataset.fields
    fig_fields = {}
    axes_fields = {}

    for sample_i, idx in enumerate(val_idx):
        batch = dataset[idx.item() if isinstance(idx, torch.Tensor) else idx]
        coord = batch["coord"].unsqueeze(0).to(device)
        params = batch["params"].unsqueeze(0).to(device)
        c_rate = batch["c_rate"].unsqueeze(0).to(device)
        fields_gt = batch["fields"].numpy()
        v_gt = batch["voltage"].numpy()

        with torch.no_grad():
            fields_pred, v_pred = model(coord, params, c_rate)

        fields_pred = fields_pred.cpu().numpy()[0]
        v_pred = v_pred.cpu().numpy()[0]

        v_mean = stats["V"]["mean"]
        v_std = stats["V"]["std"]
        v_pred_denorm = denorm(v_pred, v_mean, v_std)
        v_gt_denorm = denorm(v_gt, v_mean, v_std)
        v_rmse = np.sqrt(np.mean((v_pred_denorm - v_gt_denorm) ** 2)) * 1000

        c_rate_val = batch["c_rate"].item()

        ax = axes_v[sample_i]
        t = np.linspace(0, 1, dataset.n_time)
        ax.plot(t, v_gt_denorm, "b-", label="PyBaMM", linewidth=1.5)
        ax.plot(t, v_pred_denorm, "r--", label="FNO", linewidth=1.5)
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Sim {sample_i}: C-rate={c_rate_val:.2f}, RMSE={v_rmse:.1f}mV")
        ax.legend()
        ax.grid(True, alpha=0.3)

        for fi, fname in enumerate(field_names):
            if fname not in fig_fields:
                fig_fields[fname], axes_fields[fname] = plt.subplots(
                    args.n_samples, 3, figsize=(15, 3 * args.n_samples)
                )
                if args.n_samples == 1:
                    axes_fields[fname] = axes_fields[fname].reshape(1, -1)

            gt_2d = fields_gt[fi]
            pred_2d = fields_pred[fi]
            f_stats = stats.get(fname, {"mean": 0, "std": 1})
            gt_denorm = denorm(gt_2d, f_stats["mean"], f_stats["std"])
            pred_denorm = denorm(pred_2d, f_stats["mean"], f_stats["std"])
            err = np.abs(gt_denorm - pred_denorm)

            vmin = min(gt_denorm.min(), pred_denorm.min())
            vmax = max(gt_denorm.max(), pred_denorm.max())

            for col, (data, title) in enumerate(
                [(gt_denorm, "Ground Truth"), (pred_denorm, "FNO Pred"), (err, "Abs Error")]
            ):
                ax_f = axes_fields[fname][sample_i, col]
                if col < 2:
                    im = ax_f.imshow(data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
                else:
                    im = ax_f.imshow(data, aspect="auto", origin="lower", cmap="hot")
                ax_f.set_title(f"{fname} - {title}" if sample_i == 0 else title)
                plt.colorbar(im, ax=ax_f, fraction=0.046)
                if col == 0:
                    ax_f.set_ylabel(f"Sim {sample_i}")

    fig_v.tight_layout()
    fig_v.savefig(out_dir / "voltage_comparison.png", dpi=150)
    print(f"Saved voltage comparison to {out_dir / 'voltage_comparison.png'}")

    for fname in field_names:
        fig_fields[fname].tight_layout()
        fig_fields[fname].savefig(out_dir / f"{fname}_comparison.png", dpi=150)
        print(f"Saved {fname} comparison to {out_dir / f'{fname}_comparison.png'}")
        plt.close(fig_fields[fname])

    plt.close("all")

    print("\n--- Full validation set error ---")
    all_v_rmse = []
    all_field_rmse = {fname: [] for fname in field_names}
    n_eval = min(100, len(dataset.val_idx))

    for i in range(n_eval):
        idx = dataset.val_idx[i]
        batch = dataset[idx.item() if isinstance(idx, torch.Tensor) else idx]
        coord = batch["coord"].unsqueeze(0).to(device)
        params = batch["params"].unsqueeze(0).to(device)
        c_rate = batch["c_rate"].unsqueeze(0).to(device)
        fields_gt = batch["fields"].numpy()
        v_gt = batch["voltage"].numpy()

        with torch.no_grad():
            fields_pred, v_pred = model(coord, params, c_rate)

        fields_pred = fields_pred.cpu().numpy()[0]
        v_pred = v_pred.cpu().numpy()[0]

        v_gt_denorm = denorm(v_gt, stats["V"]["mean"], stats["V"]["std"])
        v_pred_denorm = denorm(v_pred, stats["V"]["mean"], stats["V"]["std"])
        all_v_rmse.append(np.sqrt(np.mean((v_pred_denorm - v_gt_denorm) ** 2)) * 1000)

        for fi, fname in enumerate(field_names):
            f_stats = stats.get(fname, {"mean": 0, "std": 1})
            gt = denorm(fields_gt[fi], f_stats["mean"], f_stats["std"])
            pred = denorm(fields_pred[fi], f_stats["mean"], f_stats["std"])
            rel_err = np.sqrt(np.mean((gt - pred) ** 2)) / (np.abs(gt).mean() + 1e-8)
            all_field_rmse[fname].append(rel_err * 100)

    print(f"Voltage RMSE: {np.mean(all_v_rmse):.1f} ± {np.std(all_v_rmse):.1f} mV")
    for fname in field_names:
        errs = all_field_rmse[fname]
        print(f"{fname} rel error: {np.mean(errs):.2f} ± {np.std(errs):.2f}%")

    print("\n--- Speed benchmark ---")
    batch = dataset[0]
    coord = batch["coord"].unsqueeze(0).to(device)
    params = batch["params"].unsqueeze(0).to(device)
    c_rate = batch["c_rate"].unsqueeze(0).to(device)

    for _ in range(20):
        with torch.no_grad():
            model(coord, params, c_rate)
    torch.cuda.synchronize()

    t0 = time.time()
    n_runs = 1000
    for _ in range(n_runs):
        with torch.no_grad():
            model(coord, params, c_rate)
    torch.cuda.synchronize()
    t_fno = (time.time() - t0) / n_runs * 1000

    print(f"FNO inference: {t_fno:.3f} ms/sample")
    print(f"PyBaMM solve: ~3000 ms/sample")
    print(f"Speedup: {3000/t_fno:.0f}x")

    batch_sizes = [1, 16, 64, 256]
    print("\nBatch throughput:")
    for bs in batch_sizes:
        coord_b = batch["coord"].unsqueeze(0).repeat(bs, 1, 1, 1).to(device)
        params_b = batch["params"].unsqueeze(0).repeat(bs, 1).to(device)
        c_rate_b = batch["c_rate"].unsqueeze(0).repeat(bs, 1).to(device)
        for _ in range(10):
            with torch.no_grad():
                model(coord_b, params_b, c_rate_b)
        torch.cuda.synchronize()
        t0 = time.time()
        n_bruns = 100
        for _ in range(n_bruns):
            with torch.no_grad():
                model(coord_b, params_b, c_rate_b)
        torch.cuda.synchronize()
        t_batch = (time.time() - t0) / n_bruns * 1000
        throughput = bs / (t_batch / 1000)
        print(f"  batch={bs:3d}: {t_batch:.2f} ms total, {t_batch/bs:.3f} ms/sample, {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
