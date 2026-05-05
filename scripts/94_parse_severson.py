#!/usr/bin/env python3
"""
Parse MIT/Stanford battery degradation dataset (Severson et al., Nature Energy 2019).
3 batches of LFP/graphite 18650 cells (A123), 124 total.
Extracts per-cycle discharge V(t), capacity, IR, temperature, cycle life.

Data source: https://www.kaggle.com/datasets/rickandjoe/mit-battery-degradation-dataset
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_FILES = [
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
]

N_TIME = 100


def extract_discharge(V, I, Qc, Qd, t):
    """Extract discharge portion from a cycle's V, I, Qc, Qd, t arrays.
    Discharge: I < 0 (depleting current convention) or the second half.
    Severson convention: charge first (I>0), then discharge (I<0).
    """
    V = np.asarray(V, dtype=np.float64).flatten()
    I = np.asarray(I, dtype=np.float64).flatten()

    if len(V) < 10:
        return None

    discharge_mask = I < -0.01
    if discharge_mask.sum() < 10:
        mid = len(V) // 2
        V = V[mid:]
        I = I[mid:]
    else:
        V = V[discharge_mask]
        I = I[discharge_mask]

    if len(V) < 5:
        return None

    return V


def resample_curve(V_raw, n_points=N_TIME, v_min=2.0, v_max=3.6):
    """Resample V(t) to fixed number of time points."""
    if V_raw is None or len(V_raw) < 5:
        return None
    t_raw = np.linspace(0, 1, len(V_raw))
    t_target = np.linspace(0, 1, n_points)
    V_resampled = np.interp(t_target, t_raw, V_raw)
    return V_resampled.astype(np.float32)


def parse_batch(filepath):
    """Parse one .mat batch file. Returns list of cell dicts."""
    import warnings
    warnings.filterwarnings("ignore")
    import mat73

    logger.info("Loading %s ..." % Path(filepath).name)
    data = mat73.loadmat(filepath)
    batch = data["batch"]

    n_cells = len(batch["cycle_life"])
    logger.info("  %d cells in batch", n_cells)

    cells = []
    for ci in range(n_cells):
        cycle_life = float(batch["cycle_life"][ci])
        if np.isnan(cycle_life):
            logger.warning("  cell %d: NaN cycle_life, skipping", ci)
            continue

        summ = batch["summary"][ci]
        QDischarge = np.array(summ["QDischarge"]).flatten()
        IR = np.array(summ["IR"]).flatten()
        Tavg = np.array(summ["Tavg"]).flatten()
        cycle_nums = np.array(summ["cycle"]).flatten()

        cyc = batch["cycles"][ci]
        V_all = cyc["V"]
        I_all = cyc["I"]

        n_cycles_raw = len(V_all)

        V_curves = []
        capacities = []
        internal_resistance = []
        temperatures = []
        cycle_indices = []

        for cyc_idx in range(n_cycles_raw):
            if V_all[cyc_idx] is None or I_all[cyc_idx] is None:
                continue

            V_raw = extract_discharge(
                V_all[cyc_idx], I_all[cyc_idx],
                cyc.get("Qc", [None] * n_cycles_raw)[cyc_idx],
                cyc.get("Qd", [None] * n_cycles_raw)[cyc_idx],
                cyc.get("t", [None] * n_cycles_raw)[cyc_idx],
            )
            V_resampled = resample_curve(V_raw)
            if V_resampled is None:
                continue

            cap = QDischarge[cyc_idx] if cyc_idx < len(QDischarge) else 0
            ir = IR[cyc_idx] if cyc_idx < len(IR) else 0
            temp = Tavg[cyc_idx] if cyc_idx < len(Tavg) else 0
            cyc_num = cycle_nums[cyc_idx] if cyc_idx < len(cycle_nums) else cyc_idx

            if cap <= 0:
                continue

            V_curves.append(V_resampled)
            capacities.append(float(cap))
            internal_resistance.append(float(ir))
            temperatures.append(float(temp))
            cycle_indices.append(int(cyc_num))

        if len(V_curves) < 10:
            logger.warning("  cell %d: only %d valid cycles, skipping", ci, len(V_curves))
            continue

        V_arr = np.array(V_curves, dtype=np.float32)
        cap_arr = np.array(capacities, dtype=np.float32)
        ir_arr = np.array(internal_resistance, dtype=np.float32)

        cap_initial = np.median(cap_arr[:5])
        cap_final = cap_arr[-1]
        fade_pct = (1 - cap_final / cap_initial) * 100

        cell = {
            "cell_id": "mit_b%d_c%03d" % (BATCH_FILES.index(Path(filepath).name), ci),
            "batch": int(BATCH_FILES.index(Path(filepath).name)) + 1,
            "batch_cell_idx": ci,
            "cycle_life": int(cycle_life),
            "n_cycles": len(V_curves),
            "cap_initial_Ah": float(cap_initial),
            "cap_final_Ah": float(cap_final),
            "fade_pct": float(fade_pct),
            "chemistry": "LFP",
            "form_factor": "18650",
            "V_min": float(V_arr.min()),
            "V_max": float(V_arr.max()),
            "V": V_arr,
            "capacity": cap_arr,
            "IR": ir_arr,
            "temperature": np.array(temperatures, dtype=np.float32),
            "cycle_numbers": np.array(cycle_indices, dtype=np.int32),
        }
        cells.append(cell)

        logger.info(
            "  cell %d: %d cycles, cap=%.3f->%.3f Ah, fade=%.1f%%, V=[%.3f,%.3f], life=%d",
            ci, len(V_curves), cap_initial, cap_final, fade_pct,
            V_arr.min(), V_arr.max(), int(cycle_life),
        )

    return cells


def main():
    output_dir = Path("data/external/severson")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("/root/.cache/kagglehub/datasets/rickandjoe/mit-battery-degradation-dataset/versions/1")

    all_cells = []
    for bf in BATCH_FILES:
        fpath = data_dir / bf
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue
        cells = parse_batch(str(fpath))
        all_cells.extend(cells)

    logger.info("Total: %d cells parsed", len(all_cells))

    out_path = output_dir / "severson_lfp.h5"
    with h5py.File(out_path, "w") as f:
        for cell in all_cells:
            cid = cell["cell_id"]
            g = f.create_group(cid)
            g.attrs["cycle_life"] = cell["cycle_life"]
            g.attrs["n_cycles"] = cell["n_cycles"]
            g.attrs["cap_initial_Ah"] = cell["cap_initial_Ah"]
            g.attrs["cap_final_Ah"] = cell["cap_final_Ah"]
            g.attrs["fade_pct"] = cell["fade_pct"]
            g.attrs["chemistry"] = cell["chemistry"]
            g.attrs["form_factor"] = cell["form_factor"]
            g.attrs["batch"] = cell["batch"]
            g.attrs["batch_cell_idx"] = cell["batch_cell_idx"]
            g.attrs["V_min"] = cell["V_min"]
            g.attrs["V_max"] = cell["V_max"]
            g.create_dataset("V", data=cell["V"], compression="gzip")
            g.create_dataset("capacity", data=cell["capacity"], compression="gzip")
            g.create_dataset("IR", data=cell["IR"], compression="gzip")
            g.create_dataset("temperature", data=cell["temperature"], compression="gzip")
            g.create_dataset("cycle_numbers", data=cell["cycle_numbers"], compression="gzip")

    print("\n" + "=" * 80)
    print("SEVERSON LFP DATASET PARSED")
    print("=" * 80)
    print("Total cells: %d" % len(all_cells))
    cycle_lives = [c["cycle_life"] for c in all_cells]
    fades = [c["fade_pct"] for c in all_cells]
    caps = [c["cap_initial_Ah"] for c in all_cells]
    print("Cycle life: median=%d, range=[%d, %d]" % (np.median(cycle_lives), min(cycle_lives), max(cycle_lives)))
    print("Capacity: mean=%.3f Ah, range=[%.3f, %.3f]" % (np.mean(caps), min(caps), max(caps)))
    print("Fade: mean=%.1f%%, range=[%.1f, %.1f]" % (np.mean(fades), min(fades), max(fades)))
    print("Saved to %s" % out_path)


if __name__ == "__main__":
    main()
