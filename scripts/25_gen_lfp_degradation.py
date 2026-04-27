"""Generate LFP training data with degradation state parameters.

Instead of running multi-cycle experiments (slow), parameterize the degradation
state directly and run single discharges. This is faster and cleaner.

Parameters varied:
1. Material: D_n, D_p, t+
2. Degradation state: SEI thickness, negative/positive LAM fraction, resistance multiplier

Each sim = single discharge at given C-rate.
"""

import numpy as np
import h5py
import pybamm
import time
import sys
import logging
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_RANGES = {
    "Negative particle diffusivity [m2.s-1]": (1e-15, 5e-13, "log_uniform"),
    "Positive particle diffusivity [m2.s-1]": (5e-17, 5e-15, "log_uniform"),
    "Cation transference number": (0.2, 0.45, "uniform"),
    "Initial SEI thickness [m]": (1e-9, 1e-6, "log_uniform"),
    "Negative electrode LAM fraction": (0.0, 0.3, "uniform"),
    "Positive electrode LAM fraction": (0.0, 0.3, "uniform"),
    "Resistance multiplier": (1.0, 5.0, "uniform"),
}
PARAM_NAMES = list(PARAM_RANGES.keys())
C_RATES = [0.5, 1.0, 2.0]
N_PARAM_SETS = 400
N_TIME = 100

SEI_DEFAULTS = {
    "SEI growth activation energy [J.mol-1]": 0.0,
    "SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI open-circuit potential [V]": 0.4,
    "SEI reaction exchange current density [A.m-2]": 1.5e-07,
    "Ratio of lithium moles to SEI moles": 2.0,
    "SEI resistivity [Ohm.m]": 200000.0,
}


def _run_lfp_deg_sim(args):
    idx, pv, c_rate = args
    try:
        model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
        params = pybamm.ParameterValues("Prada2013")

        for k, v in SEI_DEFAULTS.items():
            params[k] = v

        # Set material + degradation state parameters
        params["Negative particle diffusivity [m2.s-1]"] = pv[0]
        params["Positive particle diffusivity [m2.s-1]"] = pv[1]
        params["Cation transference number"] = pv[2]
        params["Initial SEI thickness [m]"] = pv[3]

        # LAM: reduce electrode capacity
        lam_neg = pv[4]
        lam_pos = pv[5]
        res_mult = pv[6]

        if lam_neg > 0.001:
            orig_neg_thick = params["Negative electrode thickness [m]"]
            params["Negative electrode thickness [m]"] = orig_neg_thick * (1 - lam_neg)
        if lam_pos > 0.001:
            orig_pos_thick = params["Positive electrode thickness [m]"]
            params["Positive electrode thickness [m]"] = orig_pos_thick * (1 - lam_pos)
        if res_mult > 1.01:
            orig_elec_cond = params["Electrolyte conductivity [S.m-1]"]
            if callable(orig_elec_cond):
                pass
            else:
                params["Electrolyte conductivity [S.m-1]"] = orig_elec_cond / res_mult

        cap = params["Nominal cell capacity [A.h]"]
        params["Current function [A]"] = c_rate * cap

        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-4, atol=1e-6)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)

        t_end = 3600.0 / max(c_rate, 0.01) * 1.5
        sol = sim.solve([0, min(t_end, 5400)])

        V_var = sol["Voltage [V]"]

        t_resamp = np.linspace(sol.t[0], sol.t[-1], N_TIME)
        V_arr = np.array([float(V_var(t)) for t in t_resamp], dtype=np.float32)

        if np.any(np.isnan(V_arr)) or np.any(V_arr < 1.0):
            return {"idx": idx, "ok": False}

        return {
            "idx": idx,
            "c_rate": float(c_rate),
            "V": V_arr,
            "param_vector": pv.astype(np.float32),
            "ok": True,
        }
    except Exception:
        return {"idx": idx, "ok": False}


def main():
    output_path = "data/fullfield/fullfield_lfp_degradation.h5"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(789)
    total = N_PARAM_SETS * len(C_RATES)

    pv = np.zeros((total, len(PARAM_NAMES)), dtype=np.float64)
    for j, (name, (lo, hi, dist)) in enumerate(PARAM_RANGES.items()):
        if dist == "log_uniform":
            pv[:, j] = 10 ** rng.uniform(np.log10(lo), np.log10(hi), total)
        else:
            pv[:, j] = rng.uniform(lo, hi, total)

    tasks = []
    idx = 0
    for pi in range(N_PARAM_SETS):
        for c_rate in C_RATES:
            tasks.append((idx, pv[idx], c_rate))
            idx += 1

    logger.info(f"Running {len(tasks)} LFP+degradation sims ({N_PARAM_SETS} × {len(C_RATES)})...")
    t0 = time.time()
    results = []
    failed = 0

    with Pool(processes=20) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_lfp_deg_sim, tasks, chunksize=2)):
            if r["ok"]:
                results.append(r)
            else:
                failed += 1
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(tasks)} ({failed} failed), {time.time()-t0:.0f}s, {results[-1]['idx'] if results else '?'}")

    logger.info(f"Done: {len(results)} ok, {failed} failed, {time.time()-t0:.0f}s")

    if not results:
        logger.error("No results!")
        sys.exit(1)

    results.sort(key=lambda r: r["idx"])

    with h5py.File(output_path, "w") as f:
        f.create_dataset("params", data=np.array([r["param_vector"] for r in results]), compression="gzip")
        f.create_dataset("c_rates", data=np.array([r["c_rate"] for r in results], dtype=np.float32), compression="gzip")
        f.create_dataset("V", data=np.array([r["V"] for r in results]), compression="gzip")
        f.create_dataset("param_set_ids", data=np.array([r["idx"] // len(C_RATES) for r in results], dtype=np.int32), compression="gzip")
        f.attrs["n_sims"] = len(results)
        f.attrs["n_time"] = N_TIME
        f.attrs["n_param_sets"] = N_PARAM_SETS
        f.attrs["nx_full"] = 1
        f.attrs["c_rates"] = C_RATES
        f.attrs["param_names"] = PARAM_NAMES
        f.attrs["chemistry"] = "LFP (Prada2013 + degradation state)"
        f.attrs["nx_neg"] = 1
        f.attrs["nx_pos"] = 1
        f.attrs["nr_pos"] = 1

    logger.info(f"Saved {output_path}: {len(results)} sims, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
