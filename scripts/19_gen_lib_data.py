"""Generate full-field data for standard Li-ion battery (graphite anode, Chen2020).

LIB has more complex physics than LMB:
- Graphite negative electrode with solid-state diffusion
- Stronger concentration gradients
- More industrially relevant
"""

import numpy as np
import h5py
import pybamm
import time
import logging
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_RANGES = {
    "Negative particle diffusivity [m2.s-1]": (1e-16, 5e-14, "log_uniform"),
    "Positive particle diffusivity [m2.s-1]": (1e-16, 5e-14, "log_uniform"),
    "Cation transference number": (0.15, 0.55, "uniform"),
    "Positive electrode conductivity [S.m-1]": (0.1, 100.0, "log_uniform"),
    "Negative electrode conductivity [S.m-1]": (1.0, 5000.0, "log_uniform"),
    "Positive electrode exchange-current density [A.m-2]": (5e-4, 5.0, "log_uniform"),
    "Negative electrode exchange-current density [A.m-2]": (5e-4, 50.0, "log_uniform"),
}
PARAM_NAMES = list(PARAM_RANGES.keys())
C_RATES = [0.1, 0.5, 1.0, 2.0]
N_PARAM_SETS = 500
N_TIME = 100


def _run_lib_sim(args):
    idx, pv, c_rate = args
    try:
        model = pybamm.lithium_ion.DFN()
        params = pybamm.ParameterValues("Chen2020")
        for name, value in zip(PARAM_NAMES, pv):
            params[name] = value
        cap = params["Nominal cell capacity [A.h]"]
        params["Current function [A]"] = c_rate * cap

        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-8, max_step_decrease_count=5)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)

        t_end = 3600.0 / max(c_rate, 0.01) * 1.3
        sol = sim.solve([0, min(t_end, 7200)])

        c_e_var = sol["Electrolyte concentration [mol.m-3]"]
        phi_e_var = sol["Electrolyte potential [V]"]
        V_var = sol["Voltage [V]"]
        c_s_n_surf = sol["Negative particle surface concentration [mol.m-3]"]
        c_s_p_surf = sol["Positive particle surface concentration [mol.m-3]"]

        nx = len(c_e_var.mesh.nodes)
        t_resamp = np.linspace(sol.t[0], sol.t[-1], N_TIME)

        c_e_data = np.zeros((nx, N_TIME), dtype=np.float32)
        phi_e_data = np.zeros((nx, N_TIME), dtype=np.float32)
        V_data = np.zeros(N_TIME, dtype=np.float32)

        csn_nodes = c_s_n_surf.mesh.nodes
        csp_nodes = c_s_p_surf.mesh.nodes
        nxn = len(csn_nodes)
        nxp = len(csp_nodes)
        csn_data = np.zeros((nxn, N_TIME), dtype=np.float32)
        csp_data = np.zeros((nxp, N_TIME), dtype=np.float32)

        for ti, t in enumerate(t_resamp):
            c_e_data[:, ti] = np.array(c_e_var(t)).flatten()[:nx]
            phi_e_data[:, ti] = np.array(phi_e_var(t)).flatten()[:nx]
            V_data[ti] = float(V_var(t))
            csn_data[:, ti] = np.array(c_s_n_surf(t)).flatten()[:nxn]
            csp_data[:, ti] = np.array(c_s_p_surf(t)).flatten()[:nxp]

        return {
            "idx": idx, "c_rate": float(c_rate),
            "c_e": c_e_data, "phi_e": phi_e_data, "V": V_data,
            "c_s_n_surf": csn_data, "c_s_p_surf": csp_data,
            "nx": nx, "nxn": nxn, "nxp": nxp,
            "param_vector": pv.astype(np.float32),
            "ok": True,
        }
    except:
        return {"idx": idx, "ok": False}


def main():
    output_path = "data/fullfield/fullfield_lib.h5"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
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

    logger.info(f"Running {len(tasks)} LIB sims ({N_PARAM_SETS} params × {len(C_RATES)} C-rates)...")
    t0 = time.time()
    results = []
    failed = 0

    with Pool(processes=24) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_lib_sim, tasks, chunksize=4)):
            if r["ok"]:
                results.append(r)
            else:
                failed += 1
            if (i + 1) % 200 == 0:
                logger.info(f"  {i+1}/{len(tasks)} ({failed} failed), {(i+1)/(time.time()-t0):.1f} sims/s")

    logger.info(f"Done: {len(results)} ok, {failed} failed, {time.time()-t0:.0f}s")

    if not results:
        logger.error("No results!")
        return

    results.sort(key=lambda r: r["idx"])
    n_sims = len(results)
    nx = results[0]["nx"]
    nxn = results[0]["nxn"]
    nxp = results[0]["nxp"]

    params_arr = np.array([r["param_vector"] for r in results])
    c_rates_arr = np.array([r["c_rate"] for r in results], dtype=np.float32)
    V_arr = np.array([r["V"] for r in results])
    c_e_arr = np.array([r["c_e"] for r in results])
    phi_e_arr = np.array([r["phi_e"] for r in results])
    csn_arr = np.array([r["c_s_n_surf"] for r in results])
    csp_arr = np.array([r["c_s_p_surf"] for r in results])
    ps_ids = np.array([r["idx"] // len(C_RATES) for r in results], dtype=np.int32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("params", data=params_arr, compression="gzip")
        f.create_dataset("c_rates", data=c_rates_arr, compression="gzip")
        f.create_dataset("V", data=V_arr, compression="gzip")
        f.create_dataset("c_e", data=c_e_arr, compression="gzip")
        f.create_dataset("phi_e", data=phi_e_arr, compression="gzip")
        f.create_dataset("c_s_n_surf", data=csn_arr, compression="gzip")
        f.create_dataset("c_s_p_surf", data=csp_arr, compression="gzip")
        f.create_dataset("param_set_ids", data=ps_ids, compression="gzip")
        f.attrs["n_sims"] = n_sims
        f.attrs["n_time"] = N_TIME
        f.attrs["nx_full"] = nx
        f.attrs["nx_neg"] = nxn
        f.attrs["nx_pos"] = nxp
        f.attrs["n_param_sets"] = N_PARAM_SETS
        f.attrs["c_rates"] = C_RATES
        f.attrs["param_names"] = PARAM_NAMES
        f.attrs["chemistry"] = "LIB (Chen2020 graphite NMC)"

    logger.info(f"Saved {output_path}: {n_sims} sims, {n_param_sets} × {len(C_RATES)}")


if __name__ == "__main__":
    main()
