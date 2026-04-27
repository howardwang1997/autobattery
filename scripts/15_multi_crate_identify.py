"""Multi-C-rate joint parameter identification using trained FNO.

Key insight: Different C-rates activate different physics:
- Low C-rate (0.1C): equilibrium thermodynamics dominates → constrains diffusivity, OCP
- Medium C-rate (0.5C): ohmic + transport → constrains conductivity, transference number
- High C-rate (1.0C): kinetics + diffusion limitation → constrains exchange current, SEI

Joint identification across multiple C-rates provides complementary information,
improving parameter recovery for parameters that are poorly identified at any single C-rate.
"""

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


def find_sim_groups(dataset):
    """Group simulations by parameter set (same params, different C-rates)."""
    params = dataset.params
    c_rates = dataset.c_rates
    n_sims = len(params)

    param_set_map = {}
    for i in range(n_sims):
        key = tuple(np.round(params[i], 8))
        if key not in param_set_map:
            param_set_map[key] = []
        param_set_map[key].append((i, c_rates[i]))

    return param_set_map


def multi_crate_identify(
    model, dataset, device, param_set_indices, steps=2000, lr=0.01, use_fields=True
):
    """
    Joint parameter identification using multiple C-rates simultaneously.

    Loss = Σ_{c_rate} [MSE(V_pred, V_obs) + α × MSE(fields_pred, fields_obs)]

    Args:
        model: trained FNO
        dataset: FullFieldDataset
        device: torch device
        param_set_indices: list of (sim_index, c_rate) tuples for one parameter set
        steps: optimization steps
        lr: learning rate
        use_fields: whether to include field loss
    Returns:
        dict with identification results
    """
    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]

    gt_params_norm = dataset.params[param_set_indices[0][0]].copy()
    gt_params = gt_params_norm * param_std + param_mean

    target_data = []
    for sim_idx, c_rate in param_set_indices:
        batch = dataset[sim_idx]
        target_data.append({
            "coord": batch["coord"].unsqueeze(0).to(device),
            "fields": batch["fields"].unsqueeze(0).to(device),
            "voltage": batch["voltage"].unsqueeze(0).to(device),
            "c_rate": batch["c_rate"].unsqueeze(0).to(device),
        })

    pred_params_norm = torch.zeros(1, 7, device=device)
    pred_params_norm.requires_grad_(True)

    optimizer = torch.optim.Adam([pred_params_norm], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_loss = float("inf")
    best_params = pred_params_norm.detach().clone()

    for step in range(steps):
        total_loss = torch.tensor(0.0, device=device)

        for td in target_data:
            fields_pred, v_pred = model(td["coord"], pred_params_norm, td["c_rate"])

            v_loss = torch.nn.functional.mse_loss(v_pred, td["voltage"])
            total_loss = total_loss + v_loss

            if use_fields:
                f_loss = torch.nn.functional.mse_loss(fields_pred, td["fields"])
                total_loss = total_loss + 0.1 * f_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = pred_params_norm.detach().clone()

    recovered_norm = best_params.cpu().numpy()[0]
    recovered = recovered_norm * param_std + param_mean
    rel_err = np.abs(recovered - gt_params) / (np.abs(gt_params) + 1e-10) * 100

    return {
        "gt_params": gt_params,
        "recovered": recovered,
        "rel_err": rel_err,
        "mean_rel_err": rel_err.mean(),
        "best_loss": best_loss,
        "n_crates": len(target_data),
    }


def single_crate_identify(model, dataset, device, sim_idx, steps=2000, lr=0.01):
    """Single C-rate identification for comparison."""
    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]

    batch = dataset[sim_idx]
    gt_params_norm = batch["params"].numpy().copy()
    gt_params = gt_params_norm * param_std + param_mean

    coord = batch["coord"].unsqueeze(0).to(device)
    v_target = batch["voltage"].unsqueeze(0).to(device)
    fields_target = batch["fields"].unsqueeze(0).to(device)
    c_rate = batch["c_rate"].unsqueeze(0).to(device)

    pred_params_norm = torch.zeros(1, 7, device=device)
    pred_params_norm.requires_grad_(True)

    optimizer = torch.optim.Adam([pred_params_norm], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_loss = float("inf")
    best_params = pred_params_norm.detach().clone()

    for step in range(steps):
        fields_pred, v_pred = model(coord, pred_params_norm, c_rate)
        loss = (torch.nn.functional.mse_loss(v_pred, v_target) +
                0.1 * torch.nn.functional.mse_loss(fields_pred, fields_target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = pred_params_norm.detach().clone()

    recovered_norm = best_params.cpu().numpy()[0]
    recovered = recovered_norm * param_std + param_mean
    rel_err = np.abs(recovered - gt_params) / (np.abs(gt_params) + 1e-10) * 100

    return {
        "gt_params": gt_params,
        "recovered": recovered,
        "rel_err": rel_err,
        "mean_rel_err": rel_err.mean(),
        "best_loss": best_loss,
        "c_rate": batch["c_rate"].item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fno_final.pt")
    parser.add_argument("--data", type=str, default="data/fullfield/fullfield_lmb.h5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-tests", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/multi_crate")
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

    # Group simulations by parameter set
    param_set_map = find_sim_groups(dataset)
    print(f"Found {len(param_set_map)} unique parameter sets")

    # Filter to sets with multiple C-rates
    multi_crate_sets = {k: v for k, v in param_set_map.items() if len(v) >= 3}
    print(f"Parameter sets with 3+ C-rates: {len(multi_crate_sets)}")

    # --- Experiment 1: Single C-rate vs Multi-C-rate identification ---
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Single C-rate vs Multi-C-rate parameter identification")
    print("=" * 80)

    n_tests = min(args.n_tests, len(multi_crate_sets))
    test_sets = list(multi_crate_sets.items())[:n_tests]

    single_results = {cr: [] for cr in [0.1, 0.2, 0.5, 1.0, 2.0]}
    multi_results = []

    for test_i, (param_key, sim_list) in enumerate(test_sets):
        indices = [(s[0], s[1]) for s in sim_list]
        c_rates_available = [s[1] for s in sim_list]

        # Multi-C-rate identification
        t0 = time.time()
        multi_res = multi_crate_identify(
            model, dataset, device, indices, steps=args.steps
        )
        multi_res["time"] = time.time() - t0
        multi_results.append(multi_res)

        # Single C-rate identification for each available C-rate
        for sim_idx, c_rate in indices:
            cr_rounded = round(c_rate * 10) / 10
            if cr_rounded in single_results:
                single_res = single_crate_identify(
                    model, dataset, device, sim_idx, steps=args.steps
                )
                single_results[cr_rounded].append(single_res)

        short_names = [n[:15] for n in param_names]
        err_str = "  ".join([f"{n}={e:.1f}%" for n, e in zip(short_names, multi_res["rel_err"])])
        print(f"Test {test_i}: {len(indices)} C-rates, mean_err={multi_res['mean_rel_err']:.1f}% | {err_str}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Parameter recovery error (%)")
    print("=" * 80)
    print(f"{'Method':<20}", end="")
    for name in param_names:
        print(f"{name[:15]:>16}", end="")
    print(f"{'Mean':>8}")
    print("-" * (20 + 16 * 7 + 8))

    # Single C-rate results
    for cr in sorted(single_results.keys()):
        results = single_results[cr]
        if not results:
            continue
        errs = np.array([r["rel_err"] for r in results])
        label = f"Single {cr:.1f}C"
        print(f"{label:<20}", end="")
        for j in range(7):
            print(f"{errs[:, j].mean():>15.1f}%", end="")
        print(f"{errs.mean():>7.1f}%")

    # Multi C-rate results
    if multi_results:
        errs = np.array([r["rel_err"] for r in multi_results])
        label = f"Multi ({len(test_sets[0][1])} C-rates)"
        print(f"{label:<20}", end="")
        for j in range(7):
            print(f"{errs[:, j].mean():>15.1f}%", end="")
        print(f"{errs.mean():>7.1f}%")

    # --- Experiment 2: Incremental C-rate analysis ---
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: How many C-rates are needed?")
    print("=" * 80)

    if n_tests > 0:
        for n_use in [1, 2, 3, 4]:
            inc_results = []
            for param_key, sim_list in test_sets:
                indices = sim_list[:n_use]
                indices = [(s[0], s[1]) for s in indices]
                res = multi_crate_identify(
                    model, dataset, device, indices, steps=args.steps
                )
                inc_results.append(res)

            errs = np.array([r["rel_err"] for r in inc_results])
            c_rates_used = [s[1] for s in test_sets[0][1][:n_use]]
            print(f"\n{n_use} C-rate(s) {c_rates_used}:")
            print(f"  Mean error: {errs.mean():.1f}%")
            for j, name in enumerate(param_names):
                print(f"  {name[:35]:<36}: {errs[:, j].mean():.1f} ± {errs[:, j].std():.1f}%")

    # --- Visualization ---
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    if multi_results:
        multi_errs = np.array([r["rel_err"] for r in multi_results])

        for j, name in enumerate(param_names):
            ax = axes[j]

            # Plot single C-rate results
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
            for ci, cr in enumerate(sorted(single_results.keys())):
                results = single_results[cr]
                if not results:
                    continue
                errs = np.array([r["rel_err"] for r in results])
                ax.barh(ci, errs[:, j].mean(), color=colors[ci], alpha=0.7,
                        label=f"Single {cr:.1f}C", height=0.6)

            # Multi C-rate
            ax.barh(len(single_results), multi_errs[:, j].mean(),
                    color='red', alpha=0.9, label='Multi C-rate', height=0.6)

            ax.set_xlabel("Error (%)")
            ax.set_title(f"{name[:30]}\n(multi: {multi_errs[:, j].mean():.1f}%)")
            ax.set_yscale("linear")
            ax.set_xlim(0, max(50, multi_errs[:, j].mean() * 1.5))

    axes[-1].axis("off")
    fig.suptitle("Parameter Identification: Single vs Multi C-rate", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "single_vs_multi_crate.png", dpi=150)
    print(f"Saved to {out_dir / 'single_vs_multi_crate.png'}")
    plt.close(fig)

    # Incremental C-rate plot
    if n_tests > 0:
        fig2, axes2 = plt.subplots(1, 7, figsize=(28, 5))
        for n_use in [1, 2, 3, 4]:
            inc_results = []
            for param_key, sim_list in test_sets:
                indices = [(s[0], s[1]) for s in sim_list[:n_use]]
                res = multi_crate_identify(
                    model, dataset, device, indices, steps=args.steps
                )
                inc_results.append(res)
            errs = np.array([r["rel_err"] for r in inc_results])
            for j in range(7):
                axes2[j].bar(n_use, errs[:, j].mean(), alpha=0.7, label=f"{n_use} C-rates")

        for j, name in enumerate(param_names):
            axes2[j].set_title(name[:25])
            axes2[j].set_xlabel("# C-rates")
            axes2[j].set_ylabel("Error (%)")
            axes2[j].set_yscale("log")

        fig2.suptitle("Parameter Error vs Number of C-rates Used")
        fig2.tight_layout()
        fig2.savefig(out_dir / "incremental_crates.png", dpi=150)
        print(f"Saved to {out_dir / 'incremental_crates.png'}")
        plt.close(fig2)

    np.savez(out_dir / "multi_crate_results.npz",
             multi_results=multi_results, param_names=param_names, allow_pickle=True)


if __name__ == "__main__":
    main()
