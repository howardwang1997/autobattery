"""Comprehensive battery degradation analysis from cycling data.

Combines:
1. Feature extraction from V(t) curves (capacity, IR drop, voltage statistics)
2. Incremental Capacity Analysis (ICA) — dQ/dV peak tracking
3. FNO-based physics-informed feature extraction (when applicable)
4. Correlation with degradation modes
5. Early-cycle life prediction
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import logging

from src.data.loader import ExperimentalDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("outputs/degradation")
OUT.mkdir(parents=True, exist_ok=True)


def extract_all_curves(data, min_pts=20):
    current = data["current"]
    voltage = data["voltage"]
    time_arr = data["time"]
    cycles = data["cycle"]
    unique_cycles = np.unique(cycles)
    results = []

    for cyc in unique_cycles:
        mask = cycles == cyc
        idx = np.where(mask)[0]
        if len(idx) < min_pts:
            continue

        c_seg = current[idx]
        v_seg = voltage[idx]
        t_seg = time_arr[idx]

        is_discharge = c_seg < -0.005 * np.abs(c_seg).max()
        if not is_discharge.any():
            continue

        changes = np.where(np.diff(is_discharge.astype(int)))[0] + 1
        segments = np.split(np.arange(len(c_seg)), changes)
        longest, longest_len = None, 0
        for seg in segments:
            if len(seg) > longest_len and is_discharge[seg[0]]:
                longest, longest_len = seg, len(seg)

        if longest is None or longest_len < min_pts:
            continue

        v_s, i_s, t_s = v_seg[longest], c_seg[longest], t_seg[longest]
        valid = ~np.isnan(v_s) & ~np.isnan(i_s)
        v_s, i_s, t_s = v_s[valid], i_s[valid], t_s[valid]
        if len(v_s) < min_pts or np.diff(t_s).sum() < 10:
            continue

        dt = np.diff(t_s)
        cap = -np.sum(i_s[:-1] * dt) / 3600

        # Compute ICA: dQ/dV
        Q = np.cumsum(np.concatenate([[0], -i_s[:-1] * dt / 3600]))
        Q_smooth = gaussian_filter1d(Q, sigma=2)
        V_smooth = gaussian_filter1d(v_s, sigma=2)

        dV = np.diff(V_smooth)
        dQ = np.diff(Q_smooth)
        valid_ica = np.abs(dV) > 1e-5
        if valid_ica.sum() > 10:
            V_ica = (V_smooth[:-1] + V_smooth[1:])[valid_ica] / 2
            dQdV = dQ[valid_ica] / dV[valid_ica]
            dQdV_smooth = gaussian_filter1d(dQdV, sigma=3)
        else:
            V_ica, dQdV_smooth = np.array([]), np.array([])

        # Features
        i_mean = np.abs(i_s.mean())
        features = {
            "cycle": int(cyc),
            "capacity_Ah": cap,
            "i_mean": i_mean,
            "v_mean": float(v_s.mean()),
            "v_min": float(v_s.min()),
            "v_max": float(v_s.max()),
            "v_range": float(v_s.max() - v_s.min()),
            "v_std": float(v_s.std()),
            "v_end": float(v_s[-1]),
            "duration_s": float(t_s[-1] - t_s[0]),
            "V_ica": V_ica,
            "dQdV": dQdV_smooth,
        }

        # IR drop: first few points voltage drop
        if len(v_s) > 5:
            features["ir_drop"] = float(v_s[0] - v_s[5])
        else:
            features["ir_drop"] = 0.0

        # ICA peak features
        if len(dQdV_smooth) > 10:
            peak_idx = np.argmax(dQdV_smooth)
            features["ica_peak_V"] = float(V_ica[peak_idx])
            features["ica_peak_height"] = float(dQdV_smooth[peak_idx])
            features["ica_peak_width"] = float(len(dQdV_smooth[dQdV_smooth > dQdV_smooth.max() * 0.5]))
        else:
            features["ica_peak_V"] = 0
            features["ica_peak_height"] = 0
            features["ica_peak_width"] = 0

        results.append(features)

    return results


def plot_degradation_analysis(features, title, prefix):
    cycles = np.array([f["cycle"] for f in features])
    caps = np.array([f["capacity_Ah"] for f in features])
    cap_ret = caps / caps.max() * 100

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # 1. Capacity fade
    ax = axes[0, 0]
    ax.plot(cycles, cap_ret, "b-o", markersize=2)
    ax.axhline(80, color="r", linestyle="--", alpha=0.5, label="80% EOL")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Capacity retention (%)")
    ax.set_title("Capacity Fade"); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Duration
    ax = axes[0, 1]
    ax.plot(cycles, np.array([f["duration_s"] for f in features]) / 60, "g-o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Duration (min)")
    ax.set_title("Discharge Duration"); ax.grid(True, alpha=0.3)

    # 3. Mean voltage
    ax = axes[0, 2]
    ax.plot(cycles, [f["v_mean"] for f in features], "r-o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Mean voltage (V)")
    ax.set_title("Mean Discharge Voltage"); ax.grid(True, alpha=0.3)

    # 4. Voltage range
    ax = axes[1, 0]
    ax.plot(cycles, [f["v_range"] for f in features], "m-o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Voltage range (V)")
    ax.set_title("Voltage Range"); ax.grid(True, alpha=0.3)

    # 5. IR drop
    ax = axes[1, 1]
    ir = np.array([f["ir_drop"] for f in features])
    ax.plot(cycles, ir * 1000, "c-o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("IR drop (mV)")
    ax.set_title("IR Drop (first 5 pts)"); ax.grid(True, alpha=0.3)

    # 6. ICA peak position
    ax = axes[1, 2]
    ax.plot(cycles, [f["ica_peak_V"] for f in features], "orange", marker="o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("ICA peak V (V)")
    ax.set_title("ICA Peak Position"); ax.grid(True, alpha=0.3)

    # 7. ICA peak height
    ax = axes[2, 0]
    ax.plot(cycles, [f["ica_peak_height"] for f in features], "brown", marker="o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("ICA peak height")
    ax.set_title("ICA Peak Height (dQ/dV)"); ax.grid(True, alpha=0.3)

    # 8. End voltage
    ax = axes[2, 1]
    ax.plot(cycles, [f["v_end"] for f in features], "purple", marker="o", markersize=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("End voltage (V)")
    ax.set_title("End-of-discharge Voltage"); ax.grid(True, alpha=0.3)

    # 9. Correlation matrix
    ax = axes[2, 2]
    feat_names = ["capacity", "v_mean", "v_range", "v_std", "ir_drop", "ica_peak_V", "ica_peak_h", "duration"]
    feat_keys = ["capacity_Ah", "v_mean", "v_range", "v_std", "ir_drop", "ica_peak_V", "ica_peak_height", "duration_s"]
    feat_matrix = np.array([[f[k] for k in feat_keys] for f in features])
    corr = np.corrcoef(feat_matrix.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=7)
    ax.set_title("Feature Correlation")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_degradation_features.png", dpi=150)
    plt.close(fig)

    # ICA waterfall plot
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    n_show = min(30, len(features))
    step = max(1, len(features) // n_show)
    indices = list(range(0, len(features), step))[:n_show]
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    for i, idx in enumerate(indices):
        f = features[idx]
        if len(f["V_ica"]) > 5:
            offset = i * 0.5
            ax2.plot(f["V_ica"], f["dQdV"] + offset, color=colors[i], linewidth=0.8, label=f"Cycle {f['cycle']}")
    ax2.set_xlabel("Voltage (V)")
    ax2.set_ylabel("dQ/dV (offset)")
    ax2.set_title(f"ICA Waterfall — {title}")
    ax2.legend(fontsize=6, ncol=3, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(OUT / f"{prefix}_ica_waterfall.png", dpi=150)
    plt.close(fig2)

    # Feature vs capacity scatter
    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 9))
    for i, (key, name) in enumerate(zip(feat_keys, feat_names)):
        ax = axes3[i // 4, i % 4]
        vals = np.array([f[key] for f in features])
        ax.scatter(vals, caps, s=8, alpha=0.6)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("Capacity (Ah)", fontsize=8)
        r = np.corrcoef(vals, caps)[0, 1]
        ax.set_title(f"r = {r:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig3.suptitle(f"Feature-Capacity Correlation — {title}", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(OUT / f"{prefix}_feature_capacity.png", dpi=150)
    plt.close(fig3)

    return cap_ret


def early_life_prediction(features, title, prefix, early_pct=0.15):
    caps = np.array([f["capacity_Ah"] for f in features])
    cycles = np.array([f["cycle"] for f in features])
    n_early = max(5, int(len(features) * early_pct))

    # Features for prediction
    feat_keys = ["capacity_Ah", "v_mean", "v_range", "v_std", "ir_drop", "ica_peak_V", "ica_peak_height", "duration_s"]
    X = np.array([[f[k] for k in feat_keys] for f in features])
    y = caps

    # Use first `early_pct` cycles to fit linear model, predict rest
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X_early = X[:n_early]
    y_early = y[:n_early]
    X_all = X

    scaler = StandardScaler()
    X_early_s = scaler.fit_transform(X_early)
    X_all_s = scaler.transform(X_all)

    model = Ridge(alpha=1.0)
    model.fit(X_early_s, y_early)
    y_pred = model.predict(X_all_s)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cycles, caps, "b-o", markersize=3, label="Actual")
    ax.plot(cycles, y_pred, "r--", label=f"Predicted (from first {n_early} cycles)")
    ax.axvline(cycles[n_early - 1], color="gray", linestyle=":", alpha=0.5, label=f"Training boundary (cycle {cycles[n_early-1]})")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Capacity (Ah)")
    ax.set_title(f"Early-Life Prediction — {title}")
    ax.legend(); ax.grid(True, alpha=0.3)

    rmse_all = np.sqrt(np.mean((y_pred - caps) ** 2)) * 1000
    rmse_late = np.sqrt(np.mean((y_pred[n_early:] - caps[n_early:]) ** 2)) * 1000
    ax.text(0.05, 0.05, f"RMSE (all): {rmse_all:.1f} mAh\nRMSE (late): {rmse_late:.1f} mAh",
            transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_life_prediction.png", dpi=150)
    plt.close(fig)

    logger.info(f"  Early-life prediction: RMSE(all)={rmse_all:.1f} mAh, RMSE(late)={rmse_late:.1f} mAh")
    return rmse_all, rmse_late


def main():
    loader = ExperimentalDataLoader()

    # NEWAREA: 434 cycles, heavy degradation
    logger.info("=" * 60)
    logger.info("Analyzing NEWAREA (434 cycles, 62.8% fade)")
    logger.info("=" * 60)
    data = loader.load_neware_xlsx("/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")
    feat_a = extract_all_curves(data)
    logger.info(f"Extracted {len(feat_a)} discharge curves")
    plot_degradation_analysis(feat_a, "NEWAREA_1205XXL01006 (434 cycles)", "newarea")
    early_life_prediction(feat_a, "NEWAREA_1205XXL01006", "newarea")

    # Print summary
    caps = np.array([f["capacity_Ah"] for f in feat_a])
    logger.info(f"\nNEWAREA Summary:")
    logger.info(f"  Cycles: {feat_a[0]['cycle']} → {feat_a[-1]['cycle']}")
    logger.info(f"  Capacity: {caps.max():.4f} → {caps.min():.4f} Ah ({caps.min()/caps.max()*100:.1f}%)")
    eol = np.where(caps / caps.max() < 0.8)[0]
    if len(eol) > 0:
        logger.info(f"  80% SOH reached at cycle: {feat_a[eol[0]]['cycle']}")
    else:
        logger.info("  80% SOH: Not reached")

    logger.info("\nDegradation analysis complete!")
    logger.info(f"Outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
