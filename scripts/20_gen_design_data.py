"""Generate design variation data for battery optimization.

Vary electrode DESIGN parameters (thickness, porosity, particle size)
while keeping material properties fixed.
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

DESIGN_PARAMS = {
    "Positive electrode thickness [m]": (55e-6, 95e-6, "uniform"),
    "Positive electrode porosity": (0.25, 0.42, "uniform"),
    "Separator thickness [m]": (10e-6, 15e-6, "uniform"),
    "Positive particle radius [m]": (3.5e-6, 7e-6, "uniform"),
    "Negative electrode thickness [m]": (65e-6, 105e-6, "uniform"),
    "Negative electrode porosity": (0.20, 0.30, "uniform"),
}
PARAM_NAMES = list(DESIGN_PARAMS.keys())
C_RATES = [0.5, 1.0]
N_DESIGNS = 200
N_TIME = 100


def _run_design_sim(args):
    idx, dv, c_rate = args
    try:
        model = pybamm.lithium_ion.DFN()
        params = pybamm.ParameterValues("Chen2020")
        for name, value in zip(PARAM_NAMES, dv):
            params[name] = value
        cap = params["Nominal cell capacity [A.h]"]
        params["Current function [A]"] = c_rate * cap

        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-5, atol=1e-7, max_step_decrease_count=10)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)

        t_end = 3600.0 / max(c_rate, 0.01) * 1.3
        sol = sim.solve([0, min(t_end, 7200)])

        c_e_var = sol["Electrolyte concentration [mol.m-3]"]
        V_var = sol["Voltage [V]"]
        nx = len(c_e_var.mesh.nodes)
        t_resamp = np.linspace(sol.t[0], sol.t[-1], N_TIME)

        c_e_data = np.zeros((nx, N_TIME), dtype=np.float32)
        V_data = np.zeros(N_TIME, dtype=np.float32)
        for ti, t in enumerate(t_resamp):
            c_e_data[:, ti] = np.array(c_e_var(t)).flatten()[:nx]
            V_data[ti] = float(V_var(t))

        V_all = np.array([float(V_var(t)) for t in sol.t])
        t_all = sol.t
        cutoff_mask = V_all > 2.5
        if cutoff_mask.sum() > 2:
            t_cut = t_all[cutoff_mask][-1]
            energy = np.trapz(V_all[cutoff_mask], t_all[cutoff_mask]) * c_rate * cap
        else:
            t_cut = 0
            energy = 0

        return {
            "idx": idx, "c_rate": float(c_rate),
            "c_e": c_e_data, "V": V_data,
            "nx": nx, "design_vector": dv.astype(np.float32),
            "energy_Wh": float(energy), "t_cutoff": float(t_cut),
            "ok": True,
        }
    except:
        return {"idx": idx, "ok": False}


def main():
    output_path = "data/fullfield/fullfield_design.h5"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(456)
    total = N_DESIGNS * len(C_RATES)

    dv = np.zeros((total, len(PARAM_NAMES)), dtype=np.float64)
    for j, (name, (lo, hi, dist)) in enumerate(DESIGN_PARAMS.items()):
        dv[:, j] = rng.uniform(lo, hi, total)

    tasks = [(idx, dv[idx], C_RATES[idx % len(C_RATES)]) for idx in range(total)]

    logger.info(f"Running {total} design sims ({N_DESIGNS} designs × {len(C_RATES)} C-rates)...")
    t0 = time.time()
    results = []
    failed = 0

    with Pool(processes=24) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_design_sim, tasks, chunksize=4)):
            if r["ok"]:
                results.append(r)
            else:
                failed += 1
            if (i + 1) % 200 == 0:
                logger.info(f"  {i+1}/{total} ({failed} failed)")

    logger.info(f"Done: {len(results)} ok, {failed} failed, {time.time()-t0:.0f}s")

    if not results:
        logger.error("No results!")
        sys.exit(1)

    results.sort(key=lambda r: r["idx"])
    n_sims = len(results)
    nx = results[0]["nx"]

    with h5py.File(output_path, "w") as f:
        f.create_dataset("design_params", data=np.array([r["design_vector"] for r in results]), compression="gzip")
        f.create_dataset("c_rates", data=np.array([r["c_rate"] for r in results], dtype=np.float32), compression="gzip")
        f.create_dataset("V", data=np.array([r["V"] for r in results]), compression="gzip")
        f.create_dataset("c_e", data=np.array([r["c_e"] for r in results]), compression="gzip")
        f.create_dataset("energy_Wh", data=np.array([r["energy_Wh"] for r in results]), compression="gzip")
        f.create_dataset("param_set_ids", data=np.array([r["idx"] // len(C_RATES) for r in results], dtype=np.int32), compression="gzip")
        f.attrs["n_sims"] = n_sims
        f.attrs["n_time"] = N_TIME
        f.attrs["nx_full"] = nx
        f.attrs["n_designs"] = N_DESIGNS
        f.attrs["param_names"] = PARAM_NAMES
        f.attrs["c_rates"] = C_RATES

    logger.info(f"Saved {output_path}: {n_sims} sims")


if __name__ == "__main__":
    main()
