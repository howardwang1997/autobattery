#!/usr/bin/env python3
"""
Noise robustness evaluation for early-cycle EOL predictor.

1. Add realistic noise to LFP cycling data (measurement, temperature,
   manufacturing variability, current noise, combined).
2. Evaluate pre-trained CNN model on noisy data.
3. Train new models with noise augmentation and compare.
4. Evaluate cross-chemistry generalization on multi-chem data.
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import time
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = Path("/root/autobattery")
OUTPUT_DIR = BASE / "outputs" / "noise_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EARLY = 5
EOL_THRESHOLD = 0.80
BATCH_SIZE = 64
N_TIME = 100
NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# Model (must match 51_early_cycle_predict.py exactly)
# ---------------------------------------------------------------------------

class EarlyCyclePredictor(nn.Module):
    def __init__(self, n_time=100, n_early=5, hidden_dim=128):
        super().__init__()
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, V_early):
        B, K, T = V_early.shape
        x = V_early.view(B * K, 1, T)
        cycle_feats = self.cycle_encoder(x).view(B, K, -1)
        cycle_mean = cycle_feats.mean(dim=1)
        cycle_diff = cycle_feats[:, -1] - cycle_feats[:, 0]
        V_diff = V_early[:, -1] - V_early[:, 0]
        V_diff_rms = torch.sqrt((V_diff ** 2).mean(dim=1, keepdim=True))
        delta_feat = self.delta_encoder(V_diff_rms)
        combined = torch.cat([cycle_mean, cycle_diff, delta_feat], dim=-1)
        return self.regressor(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_cells(h5_path):
    """Return list of dicts with V_early, eol_cycle, chem_name."""
    cells = []
    with h5py.File(h5_path, "r") as f:
        cell_group = f["cells"]
        for key in sorted(cell_group.keys()):
            grp = cell_group[key]
            V = grp["V"][:]
            cap = grp["capacity"][:]
            n_cycles = int(grp.attrs["n_cycles"])
            if n_cycles < N_EARLY + 5:
                continue
            cap_norm = cap / cap[0] if cap[0] > 0 else cap
            below = np.where(cap_norm < EOL_THRESHOLD)[0]
            eol_cycle = int(below[0]) if len(below) > 0 else n_cycles
            chem = ""
            if "chem_name" in grp.attrs:
                chem = grp.attrs["chem_name"]
                if isinstance(chem, bytes):
                    chem = chem.decode()
            cells.append({
                "V_early": V[:N_EARLY].copy(),
                "eol_cycle": eol_cycle,
                "chem": chem,
            })
    return cells


def split_cells(cells, train_frac=0.7, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(cells))
    n_train = int(len(cells) * train_frac)
    train = [cells[i] for i in sorted(idx[:n_train])]
    val = [cells[i] for i in sorted(idx[n_train:])]
    return train, val


class CellDataset(Dataset):
    def __init__(self, cells):
        self.cells = cells

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        return torch.tensor(c["V_early"], dtype=torch.float32), \
               torch.tensor(c["eol_cycle"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def add_measurement_noise(V, sigma_mv, rng):
    """Gaussian noise on voltage, sigma in millivolts."""
    sigma_v = sigma_mv / 1000.0
    return V + rng.normal(0, sigma_v, size=V.shape).astype(V.dtype)


def add_temperature_noise(V, alpha_mv_per_C=-1.0, delta_T_std=2.0, rng=None):
    """Shift voltage by dV = alpha * dT, with dT ~ N(0, delta_T_std) per cycle."""
    n_cycles = V.shape[0]
    dT = rng.normal(0, delta_T_std, size=(n_cycles, 1))
    dV = (alpha_mv_per_C / 1000.0) * dT
    return V + dV.astype(V.dtype)


def add_manufacturing_noise(V, rng):
    """Approximate manufacturing variability as per-cell voltage perturbation.

    D_n, D_p log-scale +-20% → diffusion overpotential change ~ ±5 mV.
    t+ +-5% → voltage offset ~ ±3 mV.
    Modeled as a per-cell constant offset + shape-dependent perturbation.
    """
    n_cycles, n_time = V.shape
    V_mean = V.mean(axis=1, keepdims=True)
    V_dev = V - V_mean

    log_Dn_factor = rng.normal(0, 0.20)
    log_Dp_factor = rng.normal(0, 0.20)
    t_plus_factor = rng.uniform(-0.05, 0.05)

    diffusion_shift = (log_Dn_factor + log_Dp_factor) * 0.005 * np.sign(V_dev)
    conc_shift = t_plus_factor * 0.06

    return V + diffusion_shift.astype(V.dtype) + conc_shift


def add_current_noise(V, rate_std=0.02, rng=None):
    """±rate_std C-rate variation per cycle → time-axis rescale of V(t).

    Approximated as: V_noisy = V_nominal + dV/dr * dt, where dt ~ N(0, rate_std*T).
    We approximate dV/dt via finite differences and scale.
    """
    n_cycles, n_time = V.shape
    V_noisy = V.copy()
    for k in range(n_cycles):
        rate_factor = 1.0 + rng.normal(0, rate_std)
        t_nominal = np.linspace(0, 1, n_time)
        t_shifted = t_nominal * rate_factor
        V_noisy[k] = np.interp(t_nominal, t_shifted, V[k])
    return V_noisy


def apply_noise(cells, noise_fn, rng_seed=0):
    """Apply a noise function to a copy of the cells list."""
    rng = np.random.RandomState(rng_seed)
    noisy = []
    for c in cells:
        V_noisy = noise_fn(c["V_early"].copy(), rng=rng)
        noisy.append({"V_early": V_noisy, "eol_cycle": c["eol_cycle"], "chem": c.get("chem", "")})
    return noisy


def apply_combined_noise(V, rng):
    """All noise sources together: measurement(5mV) + temperature + manufacturing + current."""
    V = add_measurement_noise(V, sigma_mv=5.0, rng=rng)
    V = add_temperature_noise(V, rng=rng)
    V = add_manufacturing_noise(V, rng)
    V = add_current_noise(V, rng=rng)
    return V


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds, targets = [], []
    for V, eol in loader:
        V = V.to(DEVICE)
        pred = model(V)
        preds.append(pred.cpu().numpy())
        targets.append(eol.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return mae, rmse


def load_pretrained_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EarlyCyclePredictor(n_time=N_TIME, n_early=N_EARLY, hidden_dim=128)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    logger.info(f"Loaded model from {ckpt_path}, val_mae={ckpt['val_mae']:.2f}")
    return model


def train_model(train_cells, epochs=150, lr=1e-3, augment_fn=None, seed=42):
    """Train a new model, optionally with noise augmentation each epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EarlyCyclePredictor(n_time=N_TIME, n_early=N_EARLY, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.SmoothL1Loss()

    train_ds = CellDataset(train_cells)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)

    best_loss = float("inf")
    best_sd = None
    for epoch in range(1, epochs + 1):
        if augment_fn is not None:
            aug_cells = apply_noise(train_cells, augment_fn, rng_seed=epoch)
            train_ds = CellDataset(aug_cells)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
num_workers=NUM_WORKERS, pin_memory=False)

        model.train()
        total_loss, n = 0.0, 0
        for V, eol in train_loader:
            V, eol = V.to(DEVICE), eol.to(DEVICE)
            pred = model(V)
            loss = criterion(pred, eol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(eol)
            n += len(eol)
        scheduler.step()

        if total_loss / n < best_loss:
            best_loss = total_loss / n
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    return model


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

def experiment_noise_evaluation():
    """Part 1: Evaluate pretrained model on noisy data."""
    logger.info("=" * 60)
    logger.info("Part 1: Noise evaluation on pretrained model")
    logger.info("=" * 60)

    model = load_pretrained_model(BASE / "outputs/early_cycle_pred_cnn_k5/best.pt")
    cells = load_cells(BASE / "data/synthetic_cycling/synthetic_cycling_lfp.h5")
    _, val_cells = split_cells(cells, seed=42)

    clean_ds = CellDataset(val_cells)
    clean_mae, clean_rmse = evaluate(model, clean_ds)
    logger.info(f"Clean baseline: MAE={clean_mae:.2f}, RMSE={clean_rmse:.2f}")

    results = []

    noise_configs = [
        ("measurement_1mV",  lambda V, rng: add_measurement_noise(V, 1.0, rng)),
        ("measurement_5mV",  lambda V, rng: add_measurement_noise(V, 5.0, rng)),
        ("measurement_10mV", lambda V, rng: add_measurement_noise(V, 10.0, rng)),
        ("measurement_50mV", lambda V, rng: add_measurement_noise(V, 50.0, rng)),
        ("temperature",      lambda V, rng: add_temperature_noise(V, rng=rng)),
        ("manufacturing",    add_manufacturing_noise),
        ("current_2pct",     lambda V, rng: add_current_noise(V, rate_std=0.02, rng=rng)),
        ("combined",         apply_combined_noise),
    ]

    for name, noise_fn in noise_configs:
        maes, rmses = [], []
        for seed in range(5):
            noisy_val = apply_noise(val_cells, noise_fn, rng_seed=seed * 100)
            ds = CellDataset(noisy_val)
            mae, rmse = evaluate(model, ds)
            maes.append(mae)
            rmses.append(rmse)
        mean_mae = np.mean(maes)
        mean_rmse = np.mean(rmses)
        std_mae = np.std(maes)
        degrad = ((mean_mae - clean_mae) / clean_mae) * 100
        results.append({
            "name": name, "type": "pretrained_on_noisy",
            "MAE": mean_mae, "MAE_std": std_mae, "RMSE": mean_rmse,
            "degradation_%": degrad,
        })
        logger.info(f"  {name:25s}: MAE={mean_mae:.2f}±{std_mae:.2f}  "
                     f"RMSE={mean_rmse:.2f}  degradation={degrad:+.1f}%")

    results.insert(0, {
        "name": "clean_baseline", "type": "pretrained",
        "MAE": clean_mae, "MAE_std": 0.0, "RMSE": clean_rmse,
        "degradation_%": 0.0,
    })
    return results


def experiment_augmented_training():
    """Part 2: Train with noise augmentation vs clean training, test on noisy data."""
    logger.info("=" * 60)
    logger.info("Part 2: Noise-augmented training comparison")
    logger.info("=" * 60)

    cells = load_cells(BASE / "data/synthetic_cycling/synthetic_cycling_lfp.h5")
    train_cells, val_cells = split_cells(cells, seed=42)

    noise_configs = [
        ("clean", None),
        ("aug_meas_5mV",  lambda V, rng: add_measurement_noise(V, 5.0, rng)),
        ("aug_meas_10mV", lambda V, rng: add_measurement_noise(V, 10.0, rng)),
        ("aug_combined",  apply_combined_noise),
    ]

    test_noise_fns = {
        "clean": None,
        "meas_5mV": lambda V, rng: add_measurement_noise(V, 5.0, rng),
        "meas_10mV": lambda V, rng: add_measurement_noise(V, 10.0, rng),
        "meas_50mV": lambda V, rng: add_measurement_noise(V, 50.0, rng),
        "combined": apply_combined_noise,
    }

    results = []
    trained_models = {}

    for train_name, aug_fn in noise_configs:
        logger.info(f"Training: {train_name}")
        model = train_model(train_cells, epochs=150, augment_fn=aug_fn, seed=42)
        trained_models[train_name] = model

        for test_name, test_fn in test_noise_fns.items():
            if test_fn is None:
                ds = CellDataset(val_cells)
            else:
                noisy_val = apply_noise(val_cells, test_fn, rng_seed=0)
                ds = CellDataset(noisy_val)
            mae, rmse = evaluate(model, ds)
            results.append({
                "name": f"train={train_name}, test={test_name}",
                "type": "augmented_training",
                "train_mode": train_name, "test_mode": test_name,
                "MAE": mae, "RMSE": rmse,
            })
            logger.info(f"  train={train_name:20s} test={test_name:12s}: "
                         f"MAE={mae:.2f}  RMSE={rmse:.2f}")

    return results, trained_models


def experiment_multi_chemistry():
    """Part 3: Cross-chemistry evaluation."""
    logger.info("=" * 60)
    logger.info("Part 3: Multi-chemistry generalization")
    logger.info("=" * 60)

    multi_path = BASE / "data/synthetic_cycling/multi_chem_cycling.h5"
    all_cells = load_cells(multi_path)

    chem_cells = defaultdict(list)
    for c in all_cells:
        chem_cells[c["chem"]].append(c)

    chemistries = sorted(chem_cells.keys())
    logger.info(f"Chemistries: {chemistries}, counts: "
                + ", ".join(f"{ch}={len(chem_cells[ch])}" for ch in chemistries))

    results = []

    # 3a: Pretrained LFP model → zero-shot on other chemistries
    model = load_pretrained_model(BASE / "outputs/early_cycle_pred_cnn_k5/best.pt")
    logger.info("\nZero-shot (pretrained LFP-only model):")
    for chem in chemistries:
        ds = CellDataset(chem_cells[chem])
        mae, rmse = evaluate(model, ds)
        results.append({
            "name": f"zeroshot_{chem}", "type": "zero_shot",
            "chemistry": chem, "MAE": mae, "RMSE": rmse,
        })
        logger.info(f"  {chem:10s}: MAE={mae:.2f}  RMSE={rmse:.2f}")

    # 3b: Leave-one-out cross-chemistry training
    logger.info("\nLeave-one-out cross-chemistry:")
    for held_out in chemistries:
        train_list = []
        for ch in chemistries:
            if ch != held_out:
                train_list.extend(chem_cells[ch])
        if len(train_list) < 20:
            logger.info(f"  Skipping {held_out}: too few training cells")
            continue

        loo_model = train_model(train_list, epochs=150, seed=42)
        test_ds = CellDataset(chem_cells[held_out])
        mae, rmse = evaluate(loo_model, test_ds)

        all_ds = CellDataset(train_list)
        train_mae, _ = evaluate(loo_model, all_ds)
        results.append({
            "name": f"loo_heldout_{held_out}", "type": "leave_one_out",
            "held_out": held_out, "n_train": len(train_list),
            "MAE": mae, "RMSE": rmse, "train_MAE": train_mae,
        })
        logger.info(f"  Held-out={held_out:10s}: test MAE={mae:.2f}  RMSE={rmse:.2f}  "
                     f"(train MAE={train_mae:.2f}, n_train={len(train_list)})")

    # 3c: Train on all chemistries, test per chemistry
    logger.info("\nTrain-all, test per chemistry:")
    all_model = train_model(all_cells, epochs=150, seed=42)
    for chem in chemistries:
        ds = CellDataset(chem_cells[chem])
        mae, rmse = evaluate(all_model, ds)
        results.append({
            "name": f"trainall_{chem}", "type": "train_all",
            "chemistry": chem, "MAE": mae, "RMSE": rmse,
        })
        logger.info(f"  {chem:10s}: MAE={mae:.2f}  RMSE={rmse:.2f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(noise_results, aug_results, multi_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Noise degradation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["name"] for r in noise_results]
    maes = [r["MAE"] for r in noise_results]
    stds = [r["MAE_std"] for r in noise_results]
    colors = ["#2ecc71"] + ["#3498db"] * (len(names) - 2) + ["#e74c3c"]
    bars = ax.barh(names, maes, xerr=stds, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(maes[0], color="red", linestyle="--", alpha=0.7, label="Clean baseline")
    ax.set_xlabel("MAE (cycles)")
    ax.set_title("Pretrained Model: Noise Robustness Evaluation")
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{mae:.2f}", va="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "noise_robustness_bar.png", dpi=150)
    plt.close(fig)

    # Plot 2: Augmented training heatmap
    aug_by_train = defaultdict(dict)
    for r in aug_results:
        aug_by_train[r["train_mode"]][r["test_mode"]] = r["MAE"]

    train_keys = ["clean", "aug_meas_5mV", "aug_meas_10mV", "aug_combined"]
    test_keys = ["clean", "meas_5mV", "meas_10mV", "meas_50mV", "combined"]

    matrix = np.zeros((len(train_keys), len(test_keys)))
    for i, tk in enumerate(train_keys):
        for j, ttk in enumerate(test_keys):
            matrix[i, j] = aug_by_train.get(tk, {}).get(ttk, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(test_keys)))
    ax.set_xticklabels(test_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(train_keys)))
    ax.set_yticklabels(train_keys)
    ax.set_xlabel("Test noise")
    ax.set_ylabel("Train mode")
    ax.set_title("MAE (cycles): Train Clean/Augmented vs Test Noise")
    for i in range(len(train_keys)):
        for j in range(len(test_keys)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "augmented_training_heatmap.png", dpi=150)
    plt.close(fig)

    # Plot 3: Multi-chemistry comparison
    zs = [(r["chemistry"], r["MAE"]) for r in multi_results if r["type"] == "zero_shot"]
    loo = [(r["held_out"], r["MAE"]) for r in multi_results if r["type"] == "leave_one_out"]
    ta = [(r["chemistry"], r["MAE"]) for r in multi_results if r["type"] == "train_all"]

    chemistries = sorted(set(c for c, _ in zs))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(chemistries))
    w = 0.25

    zs_dict = dict(zs)
    loo_dict = dict(loo)
    ta_dict = dict(ta)

    ax.bar(x - w, [zs_dict.get(c, 0) for c in chemistries], w, label="Zero-shot (LFP-trained)",
           color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x, [loo_dict.get(c, 0) for c in chemistries], w, label="Leave-one-out",
           color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, [ta_dict.get(c, 0) for c in chemistries], w, label="Train-all",
           color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(chemistries)
    ax.set_ylabel("MAE (cycles)")
    ax.set_title("Cross-Chemistry Generalization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multi_chemistry_comparison.png", dpi=150)
    plt.close(fig)

    logger.info(f"Plots saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(noise_results, aug_results, multi_results):
    print("\n" + "=" * 90)
    print("NOISE ROBUSTNESS SUMMARY")
    print("=" * 90)

    print("\n--- Part 1: Pretrained Model on Noisy Data ---")
    print(f"{'Noise Type':<30s} {'MAE':>8s} {'±std':>8s} {'RMSE':>8s} {'Degrad%':>10s}")
    print("-" * 70)
    for r in noise_results:
        print(f"{r['name']:<30s} {r['MAE']:8.2f} {r['MAE_std']:8.2f} "
              f"{r['RMSE']:8.2f} {r['degradation_%']:+9.1f}%")

    print("\n--- Part 2: Augmented Training Comparison ---")
    print(f"{'Train Mode':<20s} {'Test Noise':<15s} {'MAE':>8s} {'RMSE':>8s}")
    print("-" * 55)
    for r in aug_results:
        print(f"{r['train_mode']:<20s} {r['test_mode']:<15s} "
              f"{r['MAE']:8.2f} {r['RMSE']:8.2f}")

    print("\n--- Part 3: Multi-Chemistry Generalization ---")
    print(f"{'Method':<15s} {'Chemistry':<10s} {'MAE':>8s} {'RMSE':>8s}")
    print("-" * 45)
    for r in multi_results:
        chem = r.get("chemistry", r.get("held_out", ""))
        print(f"{r['type']:<15s} {chem:<10s} {r['MAE']:8.2f} {r['RMSE']:8.2f}")

    # Analysis
    print("\n--- Key Findings ---")
    worst_noise = max(noise_results[1:], key=lambda r: r["degradation_%"])
    print(f"Most harmful noise: {worst_noise['name']} "
          f"(+{worst_noise['degradation_%']:.1f}% degradation)")

    sig_degrade = [r for r in noise_results[1:] if r["degradation_%"] > 20]
    if sig_degrade:
        print(f"Significant degradation (>20%) at: "
              + ", ".join(r["name"] for r in sig_degrade))
    else:
        print("No noise type caused >20% degradation — model is robust")

    aug_clean = [r for r in aug_results if r["train_mode"] == "clean" and r["test_mode"] == "combined"]
    aug_comb = [r for r in aug_results if r["train_mode"] == "aug_combined" and r["test_mode"] == "combined"]
    if aug_clean and aug_comb:
        improvement = aug_clean[0]["MAE"] - aug_comb[0]["MAE"]
        pct = improvement / aug_clean[0]["MAE"] * 100
        print(f"Noise augmentation on combined noise: "
              f"{aug_clean[0]['MAE']:.2f} → {aug_comb[0]['MAE']:.2f} "
              f"({pct:+.1f}% improvement)")

    zs_best = min((r for r in multi_results if r["type"] == "zero_shot"),
                  key=lambda r: r["MAE"], default=None)
    zs_worst = max((r for r in multi_results if r["type"] == "zero_shot"),
                   key=lambda r: r["MAE"], default=None)
    if zs_best and zs_worst:
        print(f"Zero-shot range: {zs_best['chemistry']}={zs_best['MAE']:.2f} "
              f"to {zs_worst['chemistry']}={zs_worst['MAE']:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    noise_results = experiment_noise_evaluation()
    aug_results, _ = experiment_augmented_training()
    multi_results = experiment_multi_chemistry()

    plot_results(noise_results, aug_results, multi_results)
    print_summary(noise_results, aug_results, multi_results)

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
