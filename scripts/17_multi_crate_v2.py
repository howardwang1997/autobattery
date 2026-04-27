"""Multi-C-rate joint parameter identification using trained FNO.

Uses v2 data where each parameter set has 4 C-rates (0.1, 0.2, 0.5, 1.0).
"""

import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.operator.fno import FNO2d
from src.operator.dataset import FullFieldDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/multi_crate")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FullFieldDataset(
        "data/fullfield/fullfield_lmb_v2.h5", fields=["c_e", "phi_e"], normalize=True
    )
    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]
    param_names = dataset.param_names

    model = FNO2d(
        num_params=7, in_channels=2, out_channels=2, mid_channels=64,
        num_layers=4, modes1=min(16, dataset.nx_full), modes2=min(32, dataset.n_time),
    ).to(device)
    ckpt = torch.load("outputs/checkpoints/fno_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded FNO from epoch {ckpt['epoch']}")

    # Load param_set_ids from HDF5
    import h5py
    with h5py.File("data/fullfield/fullfield_lmb_v2.h5", "r") as f:
        ps_ids = f["param_set_ids"][:]
        c_rates_all = f["c_rates"][:]

    # Select test param sets from validation indices
    val_indices = dataset.val_idx.numpy() if isinstance(dataset.val_idx, torch.Tensor) else dataset.val_idx
    test_ps_ids = np.unique(ps_ids[val_indices])
    np.random.seed(42)
    np.random.shuffle(test_ps_ids)
    n_tests = min(15, len(test_ps_ids))
    test_ps_ids = sorted(test_ps_ids[:n_tests])

    steps = 2000

    results = {"single": {}, "multi": {}}
    for n_crates in [1, 2, 3, 4]:
        results[f"multi_{n_crates}"] = []

    for test_i, ps_id in enumerate(test_ps_ids):
        mask = ps_ids == ps_id
        sim_indices = np.where(mask)[0]

        if len(sim_indices) < 4:
            continue

        c_rates_this = c_rates_all[sim_indices]
        gt_params_norm = dataset.params[sim_indices[0]].copy()
        gt_params = gt_params_norm * param_std + param_mean

        # --- Single C-rate identification ---
        for si in sim_indices:
            cr = c_rates_all[si]
            cr_key = round(float(cr) * 10) / 10
            if cr_key not in results["single"]:
                results["single"][cr_key] = []

            batch = dataset[si]
            coord = batch["coord"].unsqueeze(0).to(device)
            v_target = batch["voltage"].unsqueeze(0).to(device)
            fields_target = batch["fields"].unsqueeze(0).to(device)
            c_rate_t = batch["c_rate"].unsqueeze(0).to(device)

            pred_p = torch.zeros(1, 7, device=device, requires_grad=True)
            opt = torch.optim.Adam([pred_p], lr=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
            best_loss, best_p = float("inf"), pred_p.detach().clone()

            for _ in range(steps):
                fp, vp = model(coord, pred_p, c_rate_t)
                loss = torch.nn.functional.mse_loss(vp, v_target) + \
                       0.1 * torch.nn.functional.mse_loss(fp, fields_target)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_p = pred_p.detach().clone()

            rec = best_p.cpu().numpy()[0] * param_std + param_mean
            err = np.abs(rec - gt_params) / (np.abs(gt_params) + 1e-10) * 100
            results["single"][cr_key].append(err)

        # --- Multi C-rate identification ---
        for n_use in [1, 2, 3, 4]:
            target_data = []
            for si in sim_indices[:n_use]:
                batch = dataset[si]
                target_data.append({
                    "coord": batch["coord"].unsqueeze(0).to(device),
                    "fields": batch["fields"].unsqueeze(0).to(device),
                    "voltage": batch["voltage"].unsqueeze(0).to(device),
                    "c_rate": batch["c_rate"].unsqueeze(0).to(device),
                })

            pred_p = torch.zeros(1, 7, device=device, requires_grad=True)
            opt = torch.optim.Adam([pred_p], lr=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
            best_loss, best_p = float("inf"), pred_p.detach().clone()

            for _ in range(steps):
                total_loss = torch.tensor(0.0, device=device)
                for td in target_data:
                    fp, vp = model(td["coord"], pred_p, td["c_rate"])
                    total_loss = total_loss + torch.nn.functional.mse_loss(vp, td["voltage"])
                    total_loss = total_loss + 0.1 * torch.nn.functional.mse_loss(fp, td["fields"])
                opt.zero_grad(); total_loss.backward(); opt.step(); sched.step()
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_p = pred_p.detach().clone()

            rec = best_p.cpu().numpy()[0] * param_std + param_mean
            err = np.abs(rec - gt_params) / (np.abs(gt_params) + 1e-10) * 100
            results[f"multi_{n_use}"].append(err)

        sn = [n[:12] for n in param_names]
        multi4 = results["multi_4"][-1]
        print(f"Test {test_i}: C-rates={[f'{c_rates_all[si]:.1f}' for si in sim_indices[:4]]}, "
              f"multi4 mean={multi4.mean():.1f}% | " +
              "  ".join(f"{n}={e:.1f}%" for n, e in zip(sn, multi4)))

    # --- Print summary ---
    print("\n" + "=" * 100)
    print("PARAMETER RECOVERY ERROR (%) — Mean ± Std")
    print("=" * 100)

    header = f"{'Method':<25}"
    for name in param_names:
        header += f"  {name[:18]:>18}"
    header += f"  {'Overall':>10}"
    print(header)
    print("-" * 100)

    def print_row(label, errs_list):
        errs = np.array(errs_list)
        row = f"{label:<25}"
        for j in range(7):
            row += f"  {errs[:, j].mean():>15.1f}%"
        row += f"  {errs.mean():>8.1f}%"
        print(row)

    # Single C-rate
    for cr_key in sorted(results["single"].keys()):
        errs_list = results["single"][cr_key]
        if errs_list:
            print_row(f"Single {cr_key:.1f}C", errs_list)

    # Multi C-rate
    for n_use in [1, 2, 3, 4]:
        errs_list = results[f"multi_{n_use}"]
        if errs_list:
            print_row(f"Multi ({n_use} C-rates)", errs_list)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 7, figsize=(28, 5))

    methods = []
    for cr_key in sorted(results["single"].keys()):
        if results["single"][cr_key]:
            methods.append((f"Single {cr_key:.1f}C", np.array(results["single"][cr_key])))
    for n_use in [1, 2, 3, 4]:
        if results[f"multi_{n_use}"]:
            methods.append((f"Multi ({n_use} C-rates)", np.array(results[f"multi_{n_use}"])))

    for j, name in enumerate(param_names):
        ax = axes[j]
        x_pos = 0
        colors_single = plt.cm.Blues(np.linspace(0.3, 0.7, 4))
        colors_multi = plt.cm.Reds(np.linspace(0.3, 0.9, 4))

        for mi, (label, errs) in enumerate(methods):
            if "Single" in label:
                ci = int(float(label.split()[1].replace("C", "")) * 10) - 1
                color = colors_single[min(ci, 3)]
            else:
                n = int(label.split("(")[1].split(" ")[0])
                color = colors_multi[min(n - 1, 3)]

            ax.bar(x_pos, errs[:, j].mean(), color=color, alpha=0.8, width=0.7)
            ax.errorbar(x_pos, errs[:, j].mean(), yerr=errs[:, j].std(),
                       color='black', fmt='none', capsize=3)
            x_pos += 1

        ax.set_title(name[:25], fontsize=9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[0] for m in methods], rotation=45, ha='right', fontsize=6)
        ax.set_ylabel("Error (%)")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1e4)
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.3, label='10% threshold')
        ax.axhline(y=1, color='green', linestyle='-', alpha=0.3, label='1% threshold')

    fig.suptitle("Parameter Identification: Effect of Number of C-rates", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_crate_identification.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_dir / 'multi_crate_identification.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
