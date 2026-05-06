#!/usr/bin/env python3
"""Severson archetype deep-dive analysis.

Per-archetype statistics:
  - N★ distribution (median, IQR, range)
  - Scaling quality within each archetype
  - Capacity fade characteristics
  - Cross-archetype comparison
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/universality/severson")
DATA_PATH = Path("data/external/severson/severson_lfp.h5")

SEED = 42
np.random.seed(SEED)


def load_severson():
    cells = []
    with h5py.File(DATA_PATH, "r") as f:
        for cid in sorted(f.keys()):
            g = f[cid]
            cells.append({
                "cell_id": cid,
                "cycle_life": int(g.attrs["cycle_life"]),
                "n_cycles": int(g.attrs["n_cycles"]),
                "cap_initial": float(g.attrs["cap_initial_Ah"]),
                "cap_final": float(g.attrs["cap_final_Ah"]),
                "fade_pct": float(g.attrs["fade_pct"]),
                "batch": int(g.attrs["batch"]),
                "capacity": np.array(g["capacity"]),
            })
    return cells


def detect_knee_simple(capacity, threshold=0.80):
    Q0 = np.median(capacity[:5])
    for i in range(len(capacity)):
        if capacity[i] < threshold * Q0:
            return i
    return None


def main():
    logger.info("Loading Severson data ...")
    cells = load_severson()
    logger.info("%d cells loaded", len(cells))

    arch_path = OUTPUT_DIR / "archetype.json"
    with open(arch_path) as f:
        arch_data = json.load(f)

    labels = np.array(arch_data["labels"])
    cell_ids = arch_data["cell_ids"]
    n_arch = arch_data["n_archetypes"]
    logger.info("Archetypes: k=%d", n_arch)

    arch_results = {}
    for k in range(n_arch):
        mask = labels == k
        idx = np.where(mask)[0]
        n_cells = len(idx)

        cycle_lives = np.array([cells[i]["cycle_life"] for i in idx])
        fade_pcts = np.array([cells[i]["fade_pct"] for i in idx])
        cap_init = np.array([cells[i]["cap_initial"] for i in idx])

        N_stars = []
        for i in idx:
            knee = detect_knee_simple(cells[i]["capacity"])
            N_stars.append(knee if knee is not None else cells[i]["n_cycles"])
        N_stars = np.array(N_stars, dtype=float)

        arch_results[f"archetype_{k}"] = {
            "n_cells": n_cells,
            "frac_total": float(n_cells / len(cells)),
            "cycle_life": {
                "median": float(np.median(cycle_lives)),
                "mean": float(np.mean(cycle_lives)),
                "std": float(np.std(cycle_lives)),
                "iqr": float(np.diff(np.percentile(cycle_lives, [25, 75]))[0]),
                "range": [float(cycle_lives.min()), float(cycle_lives.max())],
            },
            "fade_pct": {
                "median": float(np.median(fade_pcts)),
                "mean": float(np.mean(fade_pcts)),
                "range": [float(fade_pcts.min()), float(fade_pcts.max())],
            },
            "N_star": {
                "median": float(np.median(N_stars)),
                "mean": float(np.mean(N_stars)),
                "iqr": float(np.diff(np.percentile(N_stars, [25, 75]))[0]),
                "range": [float(N_stars.min()), float(N_stars.max())],
            },
        }

        logger.info("  Archetype %d: n=%d (%.0f%%), cycle_life=%.0f±%.0f, fade=%.1f%%, N★=%.0f",
                     k, n_cells, 100 * n_cells / len(cells),
                     np.median(cycle_lives), np.std(cycle_lives),
                     np.median(fade_pcts), np.median(N_stars))

    cross_arch = {}
    cl_medians = [arch_results[f"archetype_{k}"]["cycle_life"]["median"] for k in range(n_arch)]
    fade_medians = [arch_results[f"archetype_{k}"]["fade_pct"]["median"] for k in range(n_arch)]
    nstar_medians = [arch_results[f"archetype_{k}"]["N_star"]["median"] for k in range(n_arch)]

    cross_arch["cycle_life_range"] = [float(min(cl_medians)), float(max(cl_medians))]
    cross_arch["fade_range"] = [float(min(fade_medians)), float(max(fade_medians))]
    cross_arch["N_star_range"] = [float(min(nstar_medians)), float(max(nstar_medians))]
    cross_arch["cycle_life_archetype_ratio"] = float(max(cl_medians) / (min(cl_medians) + 1))
    cross_arch["separation_cl_fade_corr"] = float(
        np.corrcoef(cl_medians, fade_medians)[0, 1]
    )

    logger.info("\nCross-archetype:")
    logger.info("  Cycle life: %.0f to %.0f (ratio %.1f×)",
                 min(cl_medians), max(cl_medians),
                 max(cl_medians) / (min(cl_medians) + 1))
    logger.info("  Fade: %.1f%% to %.1f%%", min(fade_medians), max(fade_medians))
    logger.info("  N★: %.0f to %.0f", min(nstar_medians), max(nstar_medians))

    output = {
        "n_archetypes": n_arch,
        "per_archetype": arch_results,
        "cross_archetype": cross_arch,
    }

    out_path = OUTPUT_DIR / "archetype_deepdive.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
