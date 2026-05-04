#!/usr/bin/env python3
"""
Generate fullfield NMC811 degradation data for Bayesian identifiability analysis.

Like fullfield_lfp_degradation.h5 but using Chen2020 (NMC811) parameters.
1200 simulations with Latin Hypercube Sampling in 7D degradation space.
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

RESISTANCE_PARAMS = [
    "Negative electrode exchange-current density [A.m-2]",
    "Positive electrode exchange-current density [A.m-2]",
    "Electrolyte conductivity [S.m-1]",
    "Negative electrode conductivity [S.m-1]",
    "Positive electrode conductivity [S.m-1]",
    "SEI kinetic rate constant [m.s-1]",
    "SEI resistivity [Ohm.m]",
]


def generate_params(n_sims, seed=SEED):
    from scipy.stats import qmc as _qmc
    sampler = _qmc.LatinHypercube(d=7, seed=seed)
    sample = sampler.random(n=n_sims)
    params = np.zeros((n_sims, 7))
    for i, name in enumerate(SHORT_NAMES):
        lo, hi = PARAM_RANGES[name]
        params[:, i] = _qmc.scale(sample[:, i:i+1], lo, hi).flatten()
    return params


def run_single_sim(args):
    idx, param_vec = args
    try:
        model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
        params = pybamm.ParameterValues("Chen2020")

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

        return {"idx": idx, "V": V_interp.astype(np.float32)}

    except Exception as e:
        return None


def main():
    output_path = Path("data/fullfield/fullfield_nmc_degradation.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d NMC811 fullfield simulations", N_SIMS)
    params = generate_params(N_SIMS)

    t0 = time.time()
    results = []
    args_list = [(i, params[i]) for i in range(N_SIMS)]

    with Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_sim, args_list)):
            if res is not None:
                results.append(res)
            if (i + 1) % 100 == 0:
                logger.info(
                    "  %d/%d done, %d valid (%.0fs)",
                    i + 1, N_SIMS, len(results), time.time() - t0,
                )

    logger.info("Completed: %d/%d valid in %.0fs", len(results), N_SIMS, time.time() - t0)

    results.sort(key=lambda x: x["idx"])
    V_all = np.array([r["V"] for r in results], dtype=np.float32)
    idx_valid = np.array([r["idx"] for r in results])
    params_valid = params[idx_valid]
    cr = np.ones(len(results), dtype=np.float32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("V", data=V_all, compression="gzip")
        f.create_dataset("params", data=params_valid)
        f.create_dataset("c_rates", data=cr)
        f.create_dataset("param_set_ids", data=np.zeros(len(results), dtype=np.int32))
        f.attrs["chemistry"] = "NMC811 (Chen2020)"
        f.attrs["model"] = "SPM + SEI reaction limited"
        f.attrs["n_time"] = N_TIME
        f.attrs["n_sims"] = len(results)

    logger.info("Saved %d sims to %s", len(results), output_path)
    logger.info("V range: [%.3f, %.3f]", V_all.min(), V_all.max())


if __name__ == "__main__":
    main()
