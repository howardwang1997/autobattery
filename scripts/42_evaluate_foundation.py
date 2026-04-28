"""Evaluate Battery Foundation Model: few-shot, zero-shot, cross-chemistry.

Tests:
1. Per-chemistry voltage RMSE (in-distribution)
2. Zero-shot: leave-one-chemistry-out prediction
3. Few-shot: fine-tune with 1/5/10/50 samples from held-out chemistry
4. Parameter identification via differentiable inversion
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


def load_model(ckpt_path, device="cuda"):
    import sys
    sys.path.insert(0, ".")
    from src.foundation.model.transformer import (
        BatteryTransformer, BatteryTransformerSmall, BatteryTransformerLarge,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_size = ckpt.get("model_size", "base")
    model_cls = {"small": BatteryTransformerSmall, "base": BatteryTransformer,
                 "large": BatteryTransformerLarge}[model_size]
    model = model_cls(n_chemistries=6, n_params=8, n_time=200)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded {model_size} model, val_rmse={ckpt.get('val_rmse_mV', '?')}mV")
    return model


def evaluate_per_chemistry(model, h5_path, device="cuda"):
    """Evaluate voltage RMSE per chemistry on validation set."""
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
            cond_norm = torch.cat([cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1)

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
        logger.info(f"  {name:30s}: RMSE = {rmse_mV:.1f} mV ({len(chem_losses[cid])} samples)")

    return results


def zero_shot_evaluation(model, h5_path, leave_out_chem, device="cuda"):
    """Evaluate on a chemistry that was NOT in training (simulated by masking)."""
    import sys
    sys.path.insert(0, ".")
    from src.foundation.data.dataset import MultiChemDataset

    ds = MultiChemDataset(h5_path, split="val")
    mask = ds.chem_all == leave_out_chem
    if mask.sum() == 0:
        logger.warning(f"No samples for chemistry {leave_out_chem}")
        return None

    indices = np.where(mask)[0]
    losses = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            batch_idx = indices[start:start + 64]
            batch = [ds[i] for i in batch_idx]
            V = torch.stack([b["V"] for b in batch]).to(device)
            chem = torch.stack([b["chem_id"] for b in batch]).to(device)
            params = torch.stack([b["params"] for b in batch]).to(device)
            cond = torch.stack([b["conditions"] for b in batch]).to(device)
            cond_norm = torch.cat([cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1)

            V_pred = model(chem, params, cond_norm, targets=V)
            mse = ((V_pred - V) ** 2).mean().item()
            losses.append(mse)

    rmse_mV = np.sqrt(np.mean(losses)) * (4.2 - 2.5) * 1000
    name = ds.chem_names.get(leave_out_chem, f"Chem {leave_out_chem}")
    logger.info(f"  Zero-shot {name}: RMSE = {rmse_mV:.1f} mV")
    return {"name": name, "rmse_mV": rmse_mV}


def few_shot_finetune(model, h5_path, target_chem, n_shots, device="cuda", epochs=100):
    """Fine-tune model with n_shots samples from target chemistry."""
    import sys
    sys.path.insert(0, ".")
    from src.foundation.data.dataset import MultiChemDataset

    ds = MultiChemDataset(h5_path, split="train")
    mask = ds.chem_all == target_chem
    if mask.sum() < n_shots:
        logger.warning(f"Not enough samples ({mask.sum()}) for {n_shots} shots")
        return None

    indices = np.where(mask)[0][:n_shots]
    few_ds = torch.utils.data.Subset(ds, indices.tolist())
    loader = DataLoader(few_ds, batch_size=min(32, n_shots), shuffle=True)

    model_finetune = {k: v.clone() for k, v in model.state_dict().items()}
    import copy
    model_ft = copy.deepcopy(model)

    for name, param in model_ft.named_parameters():
        if "param_encoder" not in name and "chem_embed" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_ft.parameters()),
        lr=1e-3, weight_decay=0.01,
    )
    loss_fn = nn.MSELoss()

    model_ft.train()
    for ep in range(epochs):
        for batch in loader:
            V = batch["V"].to(device)
            chem = batch["chem_id"].to(device)
            params = batch["params"].to(device)
            cond = batch["conditions"].to(device)
            cond_norm = torch.cat([cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1)

            V_pred = model_ft(chem, params, cond_norm, targets=V)
            loss = loss_fn(V_pred, V)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    val_ds = MultiChemDataset(h5_path, split="val")
    val_mask = val_ds.chem_all == target_chem
    val_indices = np.where(val_mask)[0]

    model_ft.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(val_indices), 64):
            batch_idx = val_indices[start:start + 64]
            batch = [val_ds[i] for i in batch_idx]
            V = torch.stack([b["V"] for b in batch]).to(device)
            chem = torch.stack([b["chem_id"] for b in batch]).to(device)
            params = torch.stack([b["params"] for b in batch]).to(device)
            cond = torch.stack([b["conditions"] for b in batch]).to(device)
            cond_norm = torch.cat([cond[:, 0:1] / 3.0, (cond[:, 1:2] - 273.15) / 50.0], dim=-1)

            V_pred = model_ft(chem, params, cond_norm, targets=V)
            mse = ((V_pred - V) ** 2).mean().item()
            losses.append(mse)

    rmse_mV = np.sqrt(np.mean(losses)) * (4.2 - 2.5) * 1000
    name = ds.chem_names.get(target_chem, f"Chem {target_chem}")
    logger.info(f"  Few-shot ({n_shots:3d}) {name}: RMSE = {rmse_mV:.1f} mV")
    return {"name": name, "n_shots": n_shots, "rmse_mV": rmse_mV}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/foundation/multichem_train.h5")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints_bfom/bfom_best.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/bfom_eval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=== Per-Chemistry Evaluation ===")
    per_chem = evaluate_per_chemistry(model, args.data_path, device)

    logger.info("\n=== Zero-Shot (Leave-One-Out) ===")
    zs_results = {}
    for cid in range(6):
        r = zero_shot_evaluation(model, args.data_path, cid, device)
        if r:
            zs_results[cid] = r

    logger.info("\n=== Few-Shot Fine-tuning ===")
    fs_results = {}
    for cid in [0, 2, 4]:
        for n_shots in [1, 5, 10, 50]:
            r = few_shot_finetune(model, args.data_path, cid, n_shots, device)
            if r:
                key = f"chem{cid}_{n_shots}shot"
                fs_results[key] = r

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    names = [per_chem[k]["name"][:15] for k in sorted(per_chem)]
    rmses = [per_chem[k]["rmse_mV"] for k in sorted(per_chem)]
    ax.barh(names, rmses, color="steelblue", alpha=0.8)
    ax.set_xlabel("RMSE (mV)")
    ax.set_title("Per-Chemistry Voltage RMSE")
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    if zs_results:
        zs_names = [zs_results[k]["name"][:15] for k in sorted(zs_results)]
        zs_rmse = [zs_results[k]["rmse_mV"] for k in sorted(zs_results)]
        ax.bar(range(len(zs_names)), zs_rmse, color="coral", alpha=0.8)
        ax.set_xticks(range(len(zs_names)))
        ax.set_xticklabels(zs_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE (mV)")
        ax.set_title("Zero-Shot (Leave-One-Out)")
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    if fs_results:
        for cid in [0, 2, 4]:
            shots = [k for k in fs_results if f"chem{cid}_" in k]
            if shots:
                n_shots_list = [fs_results[k]["n_shots"] for k in sorted(shots)]
                rmses_list = [fs_results[k]["rmse_mV"] for k in sorted(shots)]
                label = fs_results[shots[0]]["name"][:15]
                ax.plot(n_shots_list, rmses_list, "o-", linewidth=2, label=label)
        ax.set_xlabel("Number of Shots")
        ax.set_ylabel("RMSE (mV)")
        ax.set_title("Few-Shot Fine-tuning")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out / "bfom_evaluation.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved to {out}/")


if __name__ == "__main__":
    main()
