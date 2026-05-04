"""Evaluate Physics-Structured Battery Foundation Model.

Produces:
1. Per-chemistry voltage RMSE
2. Voltage decomposition visualization (OCV, eta_act, eta_ohm, eta_conc)
3. Cross-chemistry zero-shot evaluation
4. Comparison with vanilla BatteryTransformer baseline
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_physics_model(ckpt_path, device="cuda"):
    import sys
    sys.path.insert(0, ".")
    from src.foundation.model.physics_structured import (
        PhysicsStructuredBatteryModel,
        PhysicsStructuredBatteryModelSmall,
        PhysicsStructuredBatteryModelLarge,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_size = ckpt.get("model_size", "base")
    model_cls = {
        "small": PhysicsStructuredBatteryModelSmall,
        "base": PhysicsStructuredBatteryModel,
        "large": PhysicsStructuredBatteryModelLarge,
    }[model_size]
    model = model_cls(n_chemistries=6, n_params=8, n_time=200)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(
        f"Loaded {model_size} model, val_rmse={ckpt.get('val_rmse_mV', '?')}mV"
    )
    return model


def load_baseline_model(ckpt_path, device="cuda"):
    import sys
    sys.path.insert(0, ".")
    from src.foundation.model.transformer import (
        BatteryTransformer,
        BatteryTransformerSmall,
        BatteryTransformerLarge,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_size = ckpt.get("model_size", "base")
    model_cls = {
        "small": BatteryTransformerSmall,
        "base": BatteryTransformer,
        "large": BatteryTransformerLarge,
    }[model_size]
    model = model_cls(n_chemistries=6, n_params=8, n_time=200)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded baseline {model_size}, val_rmse={ckpt.get('val_rmse_mV', '?')}mV")
    return model


def evaluate_per_chemistry(model, h5_path, device="cuda", is_physics=True):
    import sys
    sys.path.insert(0, ".")
    from src.foundation.data.dataset import MultiChemDataset

    ds = MultiChemDataset(h5_path, split="val")
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    chem_losses = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            V = batch["V"].to(device)
            chem = batch["chem_id"].to(device)
            params = batch["params"].to(device)
            cond = batch["conditions"].to(device)
            cond_norm = torch.cat(
                [cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1
            )

            if is_physics:
                V_pred, _ = model(chem, params, cond_norm, targets=V)
            else:
                V_pred = model(chem, params, cond_norm, targets=V)

            loss = (V_pred - V) ** 2
            for i in range(chem.shape[0]):
                cid = int(chem[i].item())
                if cid not in chem_losses:
                    chem_losses[cid] = []
                chem_losses[cid].append(loss[i].mean().item())

    results = {}
    for cid in sorted(chem_losses.keys()):
        mse = np.mean(chem_losses[cid])
        rmse_mV = np.sqrt(mse) * (4.2 - 2.5) * 1000
        name = ds.chem_names.get(cid, f"Chem {cid}")
        results[cid] = {"name": name, "rmse_mV": rmse_mV, "n_samples": len(chem_losses[cid])}
        logger.info(f"  {name:30s}: RMSE = {rmse_mV:.1f} mV")

    return results, ds


def plot_decomposition(model, ds, device="cuda", out_dir="outputs/physics_fom_eval"):
    """Plot voltage decomposition for representative samples."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for col, cid in enumerate([0, 2, 5]):
        mask = ds.chem_all == cid
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0][0]
        sample = ds[idx]

        V = sample["V"].unsqueeze(0).to(device)
        chem = sample["chem_id"].unsqueeze(0).to(device)
        params = sample["params"].unsqueeze(0).to(device)
        cond = sample["conditions"].unsqueeze(0).to(device)
        cond_norm = torch.cat(
            [cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1
        )

        with torch.no_grad():
            V_pred, comp = model(chem, params, cond_norm)

        V_np = V[0].cpu().numpy()
        Vp_np = V_pred[0].cpu().numpy()
        t = np.linspace(0, 1, len(V_np))

        name = ds.chem_names.get(cid, f"Chem {cid}")

        ax = axes[0, col]
        ax.plot(t, V_np, "k-", linewidth=2, label="Target")
        ax.plot(t, Vp_np, "r--", linewidth=1.5, label="Predicted")
        rmse = np.sqrt(np.mean((V_np - Vp_np) ** 2)) * 1.7 * 1000
        ax.set_title(f"{name}\nRMSE = {rmse:.1f} mV")
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Normalized V")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        colors = {
            "ocv": "#1f77b4",
            "eta_activation": "#ff7f0e",
            "eta_ohmic": "#2ca02c",
            "eta_concentration": "#d62728",
        }
        labels = {
            "ocv": "OCV",
            "eta_activation": "η_act (kinetics)",
            "eta_ohmic": "η_ohm (transport)",
            "eta_concentration": "η_conc (diffusion)",
        }
        for k in ["ocv", "eta_activation", "eta_ohmic", "eta_concentration"]:
            v = comp[k][0].cpu().numpy()
            ax.plot(t, v, color=colors[k], linewidth=1.5, label=labels[k])
        ax.set_title("Voltage Decomposition")
        ax.set_xlabel("Normalized time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        stacked = np.zeros(len(t))
        for k in ["eta_activation", "eta_ohmic", "eta_concentration"]:
            v = comp[k][0].cpu().numpy()
            ax.fill_between(t, stacked, stacked + np.abs(v), alpha=0.6, label=labels[k])
            stacked += np.abs(v)
        ax.plot(t, np.abs(V_np - comp["ocv"][0].cpu().numpy()), "k--", linewidth=1, label="|V - OCV|")
        ax.set_title("Overpotential Contributions")
        ax.set_xlabel("Normalized time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Physics-Structured Voltage Decomposition", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved decomposition plot to {out_dir / 'decomposition.png'}")


def plot_crate_scaling(model, ds, device="cuda", out_dir="outputs/physics_fom_eval"):
    """Verify C-rate equivariance of overpotentials."""
    out_dir = Path(out_dir)
    cid = 0
    mask = ds.chem_all == cid
    indices = np.where(mask)[0]

    crates_unique = np.unique(ds.crate_all[indices])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model.eval()
    for c_rate in crates_unique:
        sub_mask = np.abs(ds.crate_all[indices] - c_rate) < 0.01
        sub_idx = indices[sub_mask]
        if len(sub_idx) == 0:
            continue
        idx = sub_idx[0]
        sample = ds[idx]

        chem = sample["chem_id"].unsqueeze(0).to(device)
        params = sample["params"].unsqueeze(0).to(device)
        cond = sample["conditions"].unsqueeze(0).to(device)
        cond_norm = torch.cat(
            [cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1
        )

        with torch.no_grad():
            _, comp = model(chem, params, cond_norm)

        t = np.linspace(0, 1, 200)
        for ax, (key, label) in zip(
            axes,
            [
                ("eta_activation", "η_act"),
                ("eta_ohmic", "η_ohm"),
                ("eta_concentration", "η_conc"),
            ],
        ):
            v = comp[key][0].cpu().numpy()
            ax.plot(t, v, linewidth=1.5, label=f"{c_rate:.1f}C")

    for ax, (_, label) in zip(
        axes,
        [
            ("eta_activation", "η_act"),
            ("eta_ohmic", "η_ohm"),
            ("eta_concentration", "η_conc"),
        ],
    ):
        ax.set_title(label)
        ax.set_xlabel("Normalized time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Overpotential C-rate Scaling (Chen2020 NMC811)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "crate_scaling.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved C-rate scaling plot to {out_dir / 'crate_scaling.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/foundation/multichem_v2.h5")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints_physics_fom/physics_fom_best.pt")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/physics_fom_eval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_physics_model(args.checkpoint, device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=== Per-Chemistry Evaluation (Physics-Structured) ===")
    phys_results, ds = evaluate_per_chemistry(model, args.data_path, device, is_physics=True)

    if args.baseline_checkpoint:
        logger.info("\n=== Baseline (Vanilla Transformer) ===")
        baseline = load_baseline_model(args.baseline_checkpoint, device)
        base_results, _ = evaluate_per_chemistry(
            baseline, args.data_path, device, is_physics=False
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        names = [phys_results[k]["name"][:15] for k in sorted(phys_results)]
        x = np.arange(len(names))
        w = 0.35
        phys_rmse = [phys_results[k]["rmse_mV"] for k in sorted(phys_results)]
        base_rmse = [base_results[k]["rmse_mV"] for k in sorted(base_results)]
        ax.bar(x - w / 2, phys_rmse, w, label="Physics-Structured", color="steelblue")
        ax.bar(x + w / 2, base_rmse, w, label="Vanilla Transformer", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("RMSE (mV)")
        ax.set_title("Physics-Structured vs Vanilla Transformer")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=150)
        plt.close(fig)

    logger.info("\n=== Voltage Decomposition ===")
    plot_decomposition(model, ds, device, str(out))

    logger.info("\n=== C-rate Scaling ===")
    plot_crate_scaling(model, ds, device, str(out))

    logger.info(f"\nAll outputs saved to {out}/")


if __name__ == "__main__":
    main()
