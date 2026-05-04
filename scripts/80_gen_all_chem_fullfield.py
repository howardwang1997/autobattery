#!/usr/bin/env python3
"""
Generate fullfield degradation data for multiple chemistries in parallel.
Each chemistry: ~1200 simulations with Latin Hypercube Sampling in 7D parameter space.
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import pybamm
import time
import logging
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from scipy.stats import qmc as _qmc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_TIME = 100
N_SIMS = 1200
SEED = 42

PARAM_NAMES = [
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Cation transference number",
    "Initial SEI thickness [m]",
    "Negative electrode LAM fraction",
    "Positive electrode LAM fraction",
    "Resistance multiplier",
]
SHORT_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]

PARAM_RANGES = {
    "D_n": (1e-15, 5e-13),
    "D_p": (5e-17, 5e-15),
    "t+": (0.2, 0.45),
    "SEI": (1e-9, 1e-6),
    "LAM_neg": (0.0, 0.3),
    "LAM_pos": (0.0, 0.3),
    "R_mult": (1.0, 5.0),
}

CHEMISTRIES = [
    {"name": "NMC811", "param_set": "Chen2020", "id": 0},
    {"name": "NCA", "param_set": "Marquis2019", "id": 1},
    {"name": "LFP", "param_set": "Prada2013", "id": 2},
    {"name": "LFP_v2", "param_set": "Ai2020", "id": 3},
    {"name": "LCO", "param_set": "Ramadass2004", "id": 4},
]


def generate_params(n_sims, seed=SEED):
    sampler = _qmc.LatinHypercube(d=7, seed=seed)
    sample = sampler.random(n=n_sims)
    params = np.zeros((n_sims, 7))
    for i, name in enumerate(SHORT_NAMES):
        lo, hi = PARAM_RANGES[name]
        params[:, i] = _qmc.scale(sample[:, i : i + 1], lo, hi).flatten()
    return params


def run_single_sim(args):
    idx, param_vec, chem = args
    try:
        model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
        params = pybamm.ParameterValues(chem["param_set"])

        for i, key in enumerate(PARAM_NAMES):
            if key == "Resistance multiplier":
                continue
            try:
                params[key] = param_vec[i]
            except (KeyError, AttributeError):
                pass

        R_mult = param_vec[6]
        for rp_key in [
            "Negative electrode exchange-current density [A.m-2]",
            "Positive electrode exchange-current density [A.m-2]",
            "Electrolyte conductivity [S.m-1]",
            "Negative electrode conductivity [S.m-1]",
            "Positive electrode conductivity [S.m-1]",
        ]:
            try:
                orig = params[rp_key]
                if isinstance(orig, (int, float)):
                    params[rp_key] = orig / R_mult
            except (KeyError, TypeError):
                pass

        params["Current function [A]"] = params["Nominal cell capacity [A.h]"] * 1.0

        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-4, atol=1e-6)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
        sol = sim.solve([0, 3600])

        V = sol["Terminal voltage [V]"].entries
        t = sol["Time [s]"].entries

        if len(V) < 10 or np.any(np.isnan(V)):
            return None

        t_interp = np.linspace(0, t[-1], N_TIME)
        V_interp = np.interp(t_interp, t, V)

        return {"idx": idx, "V": V_interp.astype(np.float32), "chem_id": chem["id"]}
    except Exception:
        return None


def generate_chemistry(chem, params_all, output_dir):
    logger.info("Generating %s (%s) ...", chem["name"], chem["param_set"])
    output_path = Path(output_dir) / "fullfield_{}_degradation.h5".format(
        chem["name"].lower().replace(" ", "_")
    )

    if output_path.exists():
        logger.info("  Already exists, skipping")
        return

    args_list = [(i, params_all[i], chem) for i in range(N_SIMS)]
    t0 = time.time()
    results = []

    with Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_sim, args_list)):
            if res is not None:
                results.append(res)
            if (i + 1) % 200 == 0:
                logger.info(
                    "  %s: %d/%d done (%.0fs)", chem["name"], i + 1, N_SIMS, time.time() - t0
                )

    results.sort(key=lambda x: x["idx"])
    V_all = np.array([r["V"] for r in results], dtype=np.float32)
    idx_valid = np.array([r["idx"] for r in results])
    params_valid = params_all[idx_valid]
    cr = np.ones(len(results), dtype=np.float32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("V", data=V_all, compression="gzip")
        f.create_dataset("params", data=params_valid)
        f.create_dataset("c_rates", data=cr)
        f.create_dataset("chem_ids", data=np.full(len(results), chem["id"], dtype=np.int32))
        f.attrs["chemistry"] = chem["name"]
        f.attrs["param_set"] = chem["param_set"]
        f.attrs["n_time"] = N_TIME
        f.attrs["n_sims"] = len(results)

    logger.info(
        "  %s: %d/%d valid, V=[%.3f, %.3f], saved to %s",
        chem["name"],
        len(results),
        N_SIMS,
        V_all.min(),
        V_all.max(),
        output_path,
    )


def main():
    output_dir = Path("data/fullfield")
    output_dir.mkdir(parents=True, exist_ok=True)

    params = generate_params(N_SIMS, seed=SEED)

    t0 = time.time()
    for chem in CHEMISTRIES:
        generate_chemistry(chem, params, output_dir)
    logger.info("All chemistries done in %.0fs", time.time() - t0)

    # Summary
    print("\n=== Multi-Chemistry Fullfield Data Summary ===")
    print("{:10s} {:>8s} {:>12s} {:>12s}".format("Chem", "N_sims", "V_min", "V_max"))
    print("-" * 45)
    for chem in CHEMISTRIES:
        path = output_dir / "fullfield_{}_degradation.h5".format(
            chem["name"].lower().replace(" ", "_")
        )
        if path.exists():
            with h5py.File(path, "r") as f:
                V = f["V"][:]
                print(
                    "{:10s} {:8d} {:12.3f} {:12.3f}".format(
                        chem["name"], len(V), V.min(), V.max()
                    )
                )


if __name__ == "__main__":
    main()
