#!/usr/bin/env python3
"""
Parse experimental cycling data from CSV files into unified HDF5 format.

Input: /AI4S/Users/howardwang/h204/cycle_data/*.csv
Output: data/experimental/experimental_cycling.h5

Each cell contains:
  - discharge_V: (n_cycles, n_time) normalized discharge curves
  - capacity: (n_cycles,) discharge capacity per cycle
  - cycle_indices: which cycles were extracted
  - metadata: barcode, n_cycles, voltage range, capacity range
"""

import numpy as np
import pandas as pd
import h5py
import os
import logging
import time
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "/AI4S/Users/howardwang/h204/cycle_data/"
OUTPUT_PATH = "/root/autobattery/data/experimental/experimental_cycling.h5"
N_TIME = 100
MIN_CYCLES = 10


def parse_cell(filepath):
    try:
        df = pd.read_csv(filepath, usecols=[
            'barcode', 'cycle_index', 'step_name', 'voltage',
            'elecurrent', 'capacity', 'discharge_capacity', 'temp1'
        ])
    except Exception:
        return None

    barcode = str(df['barcode'].iloc[0])
    ncyc = df['cycle_index'].nunique()
    if ncyc < MIN_CYCLES:
        return None

    discharge = df[df['step_name'].str.contains('放电', na=False)].copy()
    if len(discharge) < 10:
        return None

    cycles = sorted(discharge['cycle_index'].unique())
    if len(cycles) < MIN_CYCLES:
        return None

    V_curves = []
    caps = []
    valid_cycles = []

    for cyc in cycles:
        cyc_data = discharge[discharge['cycle_index'] == cyc].copy()
        if len(cyc_data) < 5:
            continue

        V = cyc_data['voltage'].values
        if np.any(np.isnan(V)) or V.max() < 1.0:
            continue

        t = np.linspace(0, 1, len(V))
        t_interp = np.linspace(0, 1, N_TIME)
        V_interp = np.interp(t_interp, t, V)

        cap = cyc_data['capacity'].max()
        if cap <= 0:
            cap = cyc_data['discharge_capacity'].max()

        V_curves.append(V_interp.astype(np.float32))
        caps.append(float(cap))
        valid_cycles.append(int(cyc))

    if len(valid_cycles) < MIN_CYCLES:
        return None

    return {
        "barcode": barcode,
        "V": np.array(V_curves, dtype=np.float32),
        "capacity": np.array(caps, dtype=np.float32),
        "cycle_indices": np.array(valid_cycles, dtype=np.int32),
        "n_cycles": len(valid_cycles),
        "v_min": float(np.min([v.min() for v in V_curves])),
        "v_max": float(np.max([v.max() for v in V_curves])),
        "cap_initial": float(caps[0]) if caps[0] > 0 else float(caps[min(5, len(caps)-1)]),
    }


def main():
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and os.path.getsize(os.path.join(DATA_DIR, f)) > 1000
    ])
    logger.info(f"Found {len(files)} non-empty CSV files")

    t0 = time.time()
    results = []
    with Pool(16) as pool:
        for i, res in enumerate(pool.imap_unordered(parse_cell, files)):
            if res is not None:
                results.append(res)
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(files)} parsed, {len(results)} valid ({time.time()-t0:.0f}s)")

    logger.info(f"Parsed {len(results)} valid cells with >= {MIN_CYCLES} cycles in {time.time()-t0:.0f}s")

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.attrs["n_cells"] = len(results)
        f.attrs["n_time"] = N_TIME
        f.attrs["min_cycles"] = MIN_CYCLES
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        grp = f.create_group("cells")
        for i, res in enumerate(results):
            cell_grp = grp.create_group(f"cell_{i:04d}")
            cell_grp.create_dataset("V", data=res["V"], compression="gzip")
            cell_grp.create_dataset("capacity", data=res["capacity"])
            cell_grp.create_dataset("cycle_indices", data=res["cycle_indices"])
            cell_grp.attrs["barcode"] = np.bytes_(res["barcode"])
            cell_grp.attrs["n_cycles"] = res["n_cycles"]
            cell_grp.attrs["v_min"] = res["v_min"]
            cell_grp.attrs["v_max"] = res["v_max"]
            cell_grp.attrs["cap_initial"] = res["cap_initial"]

    caps = [r["cap_initial"] for r in results if r["cap_initial"] > 0]
    ncyc = [r["n_cycles"] for r in results]
    logger.info(f"Saved {len(results)} cells to {output_path}")
    logger.info(f"  Cycles: [{min(ncyc)}, {max(ncyc)}], mean={np.mean(ncyc):.0f}")
    logger.info(f"  Capacity: [{min(caps):.1f}, {max(caps):.1f}] Ah")

    fade_info = []
    for r in results:
        if r["cap_initial"] > 0 and len(r["capacity"]) > 10:
            fade = (1 - r["capacity"][-1] / r["capacity"][0]) * 100
            fade_info.append(fade)
    fade_info = np.array(fade_info)
    logger.info(f"  Capacity fade: [{fade_info.min():.1f}%, {fade_info.max():.1f}%], mean={fade_info.mean():.1f}%")


if __name__ == "__main__":
    main()
