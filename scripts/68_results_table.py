#!/usr/bin/env python3
"""
Generate final comparison table for all methods and data.

Collects results from:
  - Bayesian CNF
  - Ensemble
  - Point estimate (NN)
  - Experimental evaluation
  - Noise robustness
  - Fisher Information predictions
"""

import os
import json
import numpy as np
import h5py
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "/root/autobattery"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}


def main():
    output_path = os.path.join(PROJECT, "outputs", "final_results.json")
    results = {}

    # 1. Fisher Information predictions
    fisher_path = os.path.join(PROJECT, "outputs", "symbolic_rank_analysis", "results.json")
    fisher = load_json(fisher_path)
    if fisher:
        results["fisher"] = {
            "spm_rank": fisher.get("part3_collinearity", {}).get("effective_rank_1pct", "N/A"),
            "dfn_rank": fisher.get("part4_spm_vs_dfn", {}).get("dfn_effective_rank", "N/A"),
            "tplus_zero": True,
            "key_collinear": "D_n-SEI-LAM_neg r>0.99",
        }

    # 2. Bayesian CNF
    cnf_path = os.path.join(PROJECT, "outputs", "bayesian", "cnf_degradation.pt")
    if os.path.exists(cnf_path):
        import torch
        ckpt = torch.load(cnf_path, map_location="cpu", weights_only=False)
        results["cnf"] = {"best_val_loss": float(ckpt.get("val_loss", 0)), "use_zuko": ckpt.get("use_zuko", False)}

    # 3. Ensemble
    ensemble_dir = os.path.join(PROJECT, "outputs", "bayesian", "ensemble")
    if os.path.isdir(ensemble_dir):
        members = [f for f in os.listdir(ensemble_dir) if f.startswith("member_") and f.endswith(".pt")]
        results["ensemble"] = {"n_members": len(members)}

    # 4. Experimental evaluation
    exp_results = load_json(os.path.join(PROJECT, "outputs", "experimental_eval", "results.json"))
    if exp_results:
        results["experimental"] = exp_results

    # 5. Noise robustness
    noise_results = load_json(os.path.join(PROJECT, "outputs", "bayesian", "noisy", "results.json"))
    if noise_results:
        results["noise_robustness"] = noise_results

    # 6. NNLS cross-validation
    nnls_path = os.path.join(PROJECT, "outputs", "nnls_cross_validation")
    if os.path.isdir(nnls_path):
        results["nnls_cross_validation"] = {"path": nnls_path}

    # 7. Data summary
    data_summary = {}
    for name, path in [
        ("fullfield_degradation", os.path.join(PROJECT, "data", "fullfield", "fullfield_lfp_degradation.h5")),
        ("synthetic_cycling_lfp", os.path.join(PROJECT, "data", "synthetic_cycling", "synthetic_cycling_lfp.h5")),
        ("multi_chem_cycling", os.path.join(PROJECT, "data", "synthetic_cycling", "multi_chem_cycling.h5")),
        ("experimental", os.path.join(PROJECT, "data", "experimental", "experimental_cycling.h5")),
    ]:
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                if "cells" in f:
                    n = len(f["cells"])
                    data_summary[name] = {"n_cells": n}
                elif "params" in f:
                    data_summary[name] = {"n_sims": f["params"].shape[0]}
                else:
                    data_summary[name] = {"keys": list(f.keys())[:5]}

    results["data"] = data_summary

    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    logger.info(f"Final results saved to {output_path}")
    logger.info(f"Sections: {list(results.keys())}")

    for section, data in results.items():
        if isinstance(data, dict):
            logger.info(f"  {section}: {list(data.keys())[:10]}")


if __name__ == "__main__":
    main()
