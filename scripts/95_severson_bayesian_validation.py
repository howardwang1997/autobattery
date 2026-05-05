#!/usr/bin/env python3
"""
External validation of identifiability theory on MIT/Severson LFP dataset.

Validates Fisher rank=3 finding:
1. PCA of V(t) manifold → dimensionality ≈ 3
2. MC-dropout BNN → uncertainty structure separates informative vs uninformative features
3. Per-timepoint information map → non-uniform information content (mirrors Fisher Jacobian)

Data: 138 LFP/graphite 18650 cells (A123), cycle lives 148-1935
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

OUTPUT_DIR = Path("outputs/severson_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/external/severson/severson_lfp.h5")

MC_SAMPLES = 50
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 32
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_severson():
    """Load parsed Severson data. Returns cells list."""
    cells = []
    with h5py.File(DATA_PATH, "r") as f:
        for cid in sorted(f.keys()):
            g = f[cid]
            cell = {
                "cell_id": cid,
                "cycle_life": int(g.attrs["cycle_life"]),
                "n_cycles": int(g.attrs["n_cycles"]),
                "cap_initial": float(g.attrs["cap_initial_Ah"]),
                "cap_final": float(g.attrs["cap_final_Ah"]),
                "fade_pct": float(g.attrs["fade_pct"]),
                "batch": int(g.attrs["batch"]),
                "V": np.array(g["V"]),
                "capacity": np.array(g["capacity"]),
                "IR": np.array(g["IR"]),
                "temperature": np.array(g["temperature"]),
            }
            cells.append(cell)
    return cells


def prepare_v_evolution_features(cells):
    """Extract V(t) evolution features: ΔV = V_late - V_early for each cell.
    Also return early-cycle V and late-cycle V separately.
    """
    early_curves = []
    late_curves = []
    delta_v = []
    fade_pcts = []
    cycle_lives = []
    cell_ids = []

    for c in cells:
        n = c["n_cycles"]
        n_early = min(10, n // 4)
        n_late = min(10, n // 4)

        V_early = np.mean(c["V"][:n_early], axis=0)
        V_late = np.mean(c["V"][-n_late:], axis=0)
        dV = V_late - V_early

        early_curves.append(V_early)
        late_curves.append(V_late)
        delta_v.append(dV)
        fade_pcts.append(c["fade_pct"])
        cycle_lives.append(c["cycle_life"])
        cell_ids.append(c["cell_id"])

    return {
        "V_early": np.array(early_curves),
        "V_late": np.array(late_curves),
        "delta_V": np.array(delta_v),
        "fade_pct": np.array(fade_pcts),
        "cycle_life": np.array(cycle_lives),
        "cell_ids": cell_ids,
    }


def analyze_pca_manifold(data):
    """PCA analysis of V(t) evolution manifold. Test if dimensionality ≈ 3."""
    logger.info("=" * 60)
    logger.info("PART 1: PCA Manifold Analysis")
    logger.info("=" * 60)

    results = {}

    for feat_name in ["V_early", "V_late", "delta_V"]:
        X = data[feat_name]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        n_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        n_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        n_99 = int(np.searchsorted(cumvar, 0.99) + 1)

        results[feat_name] = {
            "explained_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),
            "cumulative_variance": cumvar[:10].tolist(),
            "n_components_90": n_90,
            "n_components_95": n_95,
            "n_components_99": n_99,
            "singular_values": pca.singular_values_[:10].tolist(),
        }

        logger.info(
            "  %s: top-5 var=[%s], n@90%%=%d, n@95%%=%d, n@99%%=%d",
            feat_name,
            ", ".join("%.4f" % v for v in pca.explained_variance_ratio_[:5]),
            n_90, n_95, n_99,
        )

    return results


def compute_information_map(data):
    """Per-timepoint mutual information I(V_t; fade%) using binning.
    This mirrors the Fisher Jacobian sensitivity analysis.
    """
    logger.info("=" * 60)
    logger.info("PART 2: Per-timepoint Information Map")
    logger.info("=" * 60)

    delta_V = data["delta_V"]
    fade = data["fade_pct"]
    n_points = delta_V.shape[1]

    mi_map = np.zeros(n_points)

    fade_bins = np.quantile(fade, [0, 0.25, 0.5, 0.75, 1.0])
    fade_digit = np.digitize(fade, fade_bins[1:-1])

    for t in range(n_points):
        vt = delta_V[:, t]
        vt_bins = np.quantile(vt, np.linspace(0, 1, 11))
        vt_digit = np.digitize(vt, vt_bins[1:-1])

        n = len(fade_digit)
        joint = np.zeros((4, 10))
        for i in range(n):
            r = min(fade_digit[i], 3)
            c = min(vt_digit[i], 9)
            joint[r, c] += 1

        joint /= joint.sum()
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        mi = 0.0
        for i in range(4):
            for j in range(10):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

        mi_map[t] = mi

    mi_ranked = np.argsort(mi_map)[::-1]
    top5_idx = mi_ranked[:5].tolist()
    bot5_idx = mi_ranked[-5:].tolist()

    logger.info(
        "  MI range: [%.4f, %.4f], mean=%.4f",
        mi_map.min(), mi_map.max(), mi_map.mean(),
    )
    logger.info(
        "  Top-5 timepoints: %s (MI=%s)",
        top5_idx,
        [round(mi_map[i], 4) for i in top5_idx],
    )
    logger.info(
        "  Bot-5 timepoints: %s (MI=%s)",
        bot5_idx,
        [round(mi_map[i], 4) for i in bot5_idx],
    )

    mi_90pct = np.percentile(mi_map, 90)
    info_fraction = (mi_map > mi_90pct).sum()
    total_info = mi_map.sum()
    top_info = mi_map[mi_map > mi_90pct].sum()

    logger.info(
        "  Top 10%% timepoints (%d/%d) carry %.1f%% of total MI",
        info_fraction, n_points, 100 * top_info / total_info if total_info > 0 else 0,
    )

    return {
        "mi_per_timepoint": mi_map.tolist(),
        "top5_timepoints": top5_idx,
        "bottom5_timepoints": bot5_idx,
        "top10pct_info_fraction": float(top_info / total_info) if total_info > 0 else 0,
        "mi_range": [float(mi_map.min()), float(mi_map.max())],
    }


class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_bnn_predictor(data):
    """Train MC-dropout BNN to predict fade% from V(t) features.
    Analyze uncertainty structure and feature importance.
    """
    logger.info("=" * 60)
    logger.info("PART 3: Bayesian NN (MC-Dropout) Prediction")
    logger.info("=" * 60)

    results = {}

    for task_name, target_name in [("fade_pct", "fade_pct"), ("cycle_life_log", "cycle_life")]:
        logger.info("  Task: %s", task_name)

        if target_name == "cycle_life":
            y = np.log(data[target_name]).astype(np.float32)
        else:
            y = data[target_name].astype(np.float32)

        for feat_name in ["delta_V", "V_early"]:
            logger.info("    Features: %s", feat_name)
            X = data[feat_name].astype(np.float32)

            X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std
            y_mean, y_std = y.mean(), y.std() + 1e-8
            y_norm = (y - y_mean) / y_std

            X_t = torch.tensor(X_norm)
            y_t = torch.tensor(y_norm).unsqueeze(1)

            n = len(X_t)
            n_train = int(0.8 * n)
            n_val = n - n_train

            train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
            val_ds = TensorDataset(X_t[n_train:], y_t[n_train:])
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

            best_val_loss = float("inf")
            best_state = None

            for dropout_p in [0.05, 0.1, 0.2]:
                model = MCDropoutNet(X.shape[1], hidden_dim=64, dropout=dropout_p)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

                for epoch in range(EPOCHS):
                    model.train()
                    for xb, yb in train_loader:
                        pred = model(xb)
                        loss = nn.functional.mse_loss(pred, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(X_t[n_train:])
                    val_loss = nn.functional.mse_loss(val_pred, y_t[n_train:]).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            model = MCDropoutNet(X.shape[1], hidden_dim=64, dropout=0.1)
            model.load_state_dict(best_state)

            model.train()
            mc_preds = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    mc_preds.append(model(X_t).numpy())
            mc_preds = np.array(mc_preds).squeeze()

            pred_mean = mc_preds.mean(axis=0) * y_std + y_mean
            pred_std = mc_preds.std(axis=0) * y_std

            y_true = y

            if target_name == "cycle_life":
                pred_mean_orig = np.exp(pred_mean)
                y_true_orig = data["cycle_life"]
                r2 = r2_score(y_true_orig, pred_mean_orig)
                mae = mean_absolute_error(y_true_orig, pred_mean_orig)
            else:
                r2 = r2_score(y_true, pred_mean)
                mae = mean_absolute_error(y_true, pred_mean)

            uncertainty_corr = np.corrcoef(np.abs(y_true - pred_mean), pred_std)[0, 1]

            logger.info(
                "      R²=%.4f, MAE=%.4f, mean_σ=%.4f, uncertainty_corr=%.3f",
                r2, mae, pred_std.mean(), uncertainty_corr,
            )

            key = f"{task_name}|{feat_name}"
            results[key] = {
                "r2": float(r2),
                "mae": float(mae),
                "mean_uncertainty": float(pred_std.mean()),
                "max_uncertainty": float(pred_std.max()),
                "uncertainty_corr_with_error": float(uncertainty_corr),
                "mc_samples": MC_SAMPLES,
            }

    return results


def analyze_v_manifold_rank(data):
    """Directly test: does V(t) evolution live on a rank-3 manifold?
    Uses subspace angle analysis and effective rank.
    """
    logger.info("=" * 60)
    logger.info("PART 4: V(t) Manifold Effective Rank")
    logger.info("=" * 60)

    results = {}

    for feat_name in ["V_early", "V_late", "delta_V"]:
        X = data[feat_name].astype(np.float64)
        X_centered = X - X.mean(axis=0)

        svd_vals = np.linalg.svd(X_centered, compute_uv=False)
        svd_sq = svd_vals ** 2
        total = svd_sq.sum()

        p = svd_sq / total
        entropy = -np.sum(p * np.log(p + 1e-30))
        max_entropy = np.log(len(svd_sq))
        effective_rank = np.exp(entropy)

        participation_ratio = (svd_sq.sum()) ** 2 / (svd_sq ** 2).sum()

        cumvar = np.cumsum(svd_sq / total)
        n_95 = int(np.searchsorted(cumvar, 0.95) + 1)

        results[feat_name] = {
            "effective_rank_entropy": float(effective_rank),
            "participation_ratio": float(participation_ratio),
            "n_components_95": n_95,
            "top10_singular_values": svd_vals[:10].tolist(),
            "svd_ratio_1_to_rest": float(svd_vals[0] / (svd_vals[1:].sum() + 1e-30)),
        }

        logger.info(
            "  %s: eff_rank=%.2f, PR=%.2f, n@95%%=%d, svd[0]/rest=%.2f",
            feat_name, effective_rank, participation_ratio, n_95,
            svd_vals[0] / (svd_vals[1:].sum() + 1e-30),
        )

    return results


def analyze_within_cell_rank(cells):
    """Per-cell temporal V(t) evolution rank analysis.
    For each cell, compute the effective rank of V(t) across cycles.
    This tests: is within-cell V(t) evolution also low-rank?
    """
    logger.info("=" * 60)
    logger.info("PART 5: Within-Cell Temporal Rank Analysis")
    logger.info("=" * 60)

    eff_ranks = []
    prs = []
    n95s = []
    cycle_counts = []

    for c in cells:
        V = c["V"]
        if V.shape[0] < 20:
            continue

        stride = max(1, V.shape[0] // 200)
        V_sub = V[::stride]
        V_centered = V_sub - V_sub.mean(axis=0)

        svd_vals = np.linalg.svd(V_centered, compute_uv=False)
        svd_sq = svd_vals ** 2
        total = svd_sq.sum()

        p = svd_sq / total
        entropy = -np.sum(p * np.log(p + 1e-30))
        eff_rank = np.exp(entropy)
        pr = (svd_sq.sum()) ** 2 / (svd_sq ** 2).sum()

        cumvar = np.cumsum(svd_sq / total)
        n95 = int(np.searchsorted(cumvar, 0.95) + 1)

        eff_ranks.append(eff_rank)
        prs.append(pr)
        n95s.append(n95)
        cycle_counts.append(V.shape[0])

    eff_ranks = np.array(eff_ranks)
    prs = np.array(prs)
    n95s = np.array(n95s)

    logger.info(
        "  Effective rank: mean=%.2f, median=%.2f, range=[%.2f, %.2f]",
        eff_ranks.mean(), np.median(eff_ranks), eff_ranks.min(), eff_ranks.max(),
    )
    logger.info(
        "  Participation ratio: mean=%.2f, median=%.2f",
        prs.mean(), np.median(prs),
    )
    logger.info(
        "  n@95%%: mean=%.1f, median=%.1f, range=[%d, %d]",
        n95s.mean(), np.median(n95s), n95s.min(), n95s.max(),
    )
    logger.info(
        "  Cells with eff_rank <= 3: %d/%d (%.1f%%)",
        (eff_ranks <= 3).sum(), len(eff_ranks), 100 * (eff_ranks <= 3).sum() / len(eff_ranks),
    )
    logger.info(
        "  Cells with eff_rank <= 2: %d/%d (%.1f%%)",
        (eff_ranks <= 2).sum(), len(eff_ranks), 100 * (eff_ranks <= 2).sum() / len(eff_ranks),
    )

    return {
        "eff_rank_mean": float(eff_ranks.mean()),
        "eff_rank_median": float(np.median(eff_ranks)),
        "eff_rank_std": float(eff_ranks.std()),
        "eff_rank_range": [float(eff_ranks.min()), float(eff_ranks.max())],
        "pr_mean": float(prs.mean()),
        "pr_median": float(np.median(prs)),
        "n95_mean": float(n95s.mean()),
        "n95_median": float(np.median(n95s)),
        "frac_effrank_leq3": float((eff_ranks <= 3).sum() / len(eff_ranks)),
        "frac_effrank_leq2": float((eff_ranks <= 2).sum() / len(eff_ranks)),
        "per_cell_eff_ranks": eff_ranks.tolist(),
        "per_cell_prs": prs.tolist(),
    }


def main():
    logger.info("Loading Severson LFP dataset ...")
    cells = load_severson()
    logger.info("Loaded %d cells", len(cells))

    logger.info("Preparing V(t) evolution features ...")
    data = prepare_v_evolution_features(cells)
    logger.info(
        "  V shapes: %s, fade: [%.1f, %.1f]%%, life: [%d, %d]",
        data["V_early"].shape,
        data["fade_pct"].min(), data["fade_pct"].max(),
        data["cycle_life"].min(), data["cycle_life"].max(),
    )

    pca_results = analyze_pca_manifold(data)
    info_results = compute_information_map(data)
    bnn_results = train_bnn_predictor(data)
    rank_results = analyze_v_manifold_rank(data)
    within_cell_results = analyze_within_cell_rank(cells)

    all_results = {
        "dataset": {
            "n_cells": len(cells),
            "cycle_life_range": [int(data["cycle_life"].min()), int(data["cycle_life"].max())],
            "cycle_life_median": int(np.median(data["cycle_life"])),
            "fade_pct_range": [float(data["fade_pct"].min()), float(data["fade_pct"].max())],
            "fade_pct_mean": float(data["fade_pct"].mean()),
            "capacity_range_Ah": [
                float(min(c["cap_initial"] for c in cells)),
                float(max(c["cap_initial"] for c in cells)),
            ],
        },
        "pca_manifold": pca_results,
        "information_map": info_results,
        "bnn_prediction": bnn_results,
        "effective_rank": rank_results,
        "within_cell_rank": within_cell_results,
        "validation_summary": {
            "fisher_rank_predicted": 3,
            "pca_delta_V_n95": pca_results["delta_V"]["n_components_95"],
            "eff_rank_delta_V": rank_results["delta_V"]["effective_rank_entropy"],
            "participation_ratio_delta_V": rank_results["delta_V"]["participation_ratio"],
            "within_cell_eff_rank_median": within_cell_results["eff_rank_median"],
            "within_cell_frac_leq3": within_cell_results["frac_effrank_leq3"],
        },
    }

    out_path = OUTPUT_DIR / "severson_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    vs = all_results["validation_summary"]
    logger.info("  Fisher rank (theory):        %d", vs["fisher_rank_predicted"])
    logger.info("  PCA n@95%% (delta_V):          %d", vs["pca_delta_V_n95"])
    logger.info("  Effective rank (delta_V):    %.2f", vs["eff_rank_delta_V"])
    logger.info("  Participation ratio (dV):    %.2f", vs["participation_ratio_delta_V"])
    logger.info("  Within-cell eff_rank median: %.2f", vs["within_cell_eff_rank_median"])
    logger.info("  Within-cell eff_rank <= 3:   %.1f%%", 100 * vs["within_cell_frac_leq3"])
    logger.info("  Results saved to %s", out_path)


if __name__ == "__main__":
    main()
