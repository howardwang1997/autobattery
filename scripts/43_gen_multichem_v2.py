"""Generate expanded multi-chemistry data (v2) for scale-up training.

Changes from v1:
- Prada2013: use SPM without SEI (fixes all-failed issue)
- Marquis2019: narrower parameter ranges (fixes 97% failure)
- All chemistries: 5K-8K param sets (was 2K-3K)
- Temperature dimension added: [25°C, 35°C, 45°C]
- Total target: ~200K sims
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
sys.path.insert(0, ".")

import numpy as np
import h5py
import pybamm
import time
import logging
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_TIME = 200

CHEMISTRY_CONFIGS = {
    0: {
        "name": "Chen2020_NMC811",
        "param_set": "Chen2020",
        "model_opts": {"SEI": "reaction limited"},
        "sei_params": {
            "SEI growth activation energy [J.mol-1]": 0.0,
            "SEI partial molar volume [m3.mol-1]": 9.585e-05,
            "SEI open-circuit potential [V]": 0.4,
            "SEI reaction exchange current density [A.m-2]": 1.5e-07,
            "Ratio of lithium moles to SEI moles": 2.0,
            "SEI resistivity [Ohm.m]": 200000.0,
        },
        "c_rates": [0.2, 0.5, 1.0, 2.0],
        "temperatures": [298.15, 308.15, 318.15],
        "n_sets": 5000,
        "sweep_params": {
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Negative electrode conductivity [S.m-1]": (10, 500, "log"),
            "Positive electrode conductivity [S.m-1]": (0.1, 50, "log"),
        },
    },
    1: {
        "name": "Marquis2019_NCA",
        "param_set": "Marquis2019",
        "model_opts": {},
        "sei_params": {},
        "c_rates": [0.2, 0.5, 1.0, 2.0],
        "temperatures": [298.15, 308.15],
        "n_sets": 5000,
        "sweep_params": {
            "Negative particle diffusivity [m2.s-1]": (5e-14, 5e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.25, 0.45, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.8, 1.5, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (5e-5, 5e-4, "log"),
            "Positive electrode exchange-current density [A.m-2]": (5e-5, 5e-4, "log"),
        },
    },
    2: {
        "name": "Prada2013_LFP",
        "param_set": "Prada2013",
        "model_opts": {},
        "sei_params": {},
        "c_rates": [0.5, 1.0, 2.0, 3.0],
        "temperatures": [298.15, 308.15, 318.15],
        "n_sets": 5000,
        "sweep_params": {
            "Negative particle diffusivity [m2.s-1]": (1e-15, 5e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (5e-17, 5e-15, "log"),
            "Cation transference number": (0.2, 0.45, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
    },
    3: {
        "name": "Ai2020_LFP_v2",
        "param_set": "Ai2020",
        "model_opts": {},
        "sei_params": {},
        "c_rates": [0.2, 0.5, 1.0, 2.0],
        "temperatures": [298.15, 308.15, 318.15],
        "n_sets": 5000,
        "sweep_params": {
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
    },
    4: {
        "name": "OKane2022_NMC_deg",
        "param_set": "OKane2022",
        "model_opts": {"SEI": "reaction limited", "SEI porosity change": "true"},
        "sei_params": {},
        "c_rates": [0.2, 0.5, 1.0],
        "temperatures": [298.15, 308.15],
        "n_sets": 5000,
        "sweep_params": {
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Negative electrode conductivity [S.m-1]": (10, 500, "log"),
            "Positive electrode conductivity [S.m-1]": (0.1, 50, "log"),
        },
    },
    5: {
        "name": "Ramadass2004_LCO",
        "param_set": "Ramadass2004",
        "model_opts": {"SEI": "reaction limited"},
        "sei_params": {},
        "c_rates": [0.2, 0.5, 1.0, 2.0],
        "temperatures": [298.15, 308.15],
        "n_sets": 5000,
        "sweep_params": {
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
    },
}


def _run_single(args):
    chem_id, cfg, param_vals, sweep_keys, c_rate, temperature = args
    try:
        model = pybamm.lithium_ion.SPM(cfg["model_opts"])
        params = pybamm.ParameterValues(cfg["param_set"])

        for k, v in cfg.get("sei_params", {}).items():
            params[k] = v

        for j, key in enumerate(sweep_keys):
            params[key] = param_vals[j]

        cap = params.get("Nominal cell capacity [A.h]", 1.0)
        if cap < 0.01:
            cap = 1.0
        params["Current function [A]"] = c_rate * cap
        params["Ambient temperature [K]"] = temperature

        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-4, atol=1e-6, max_step_decrease_count=5)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
        t_end = 3600.0 / max(c_rate, 0.01) * 1.5
        sol = sim.solve([0, min(t_end, 7200)])

        V_var = sol["Voltage [V]"]
        t_resamp = np.linspace(sol.t[0], sol.t[-1], N_TIME)
        V_arr = np.array([float(V_var(t)) for t in t_resamp], dtype=np.float32)

        if np.any(np.isnan(V_arr)) or np.any(V_arr < 1.5):
            return {"ok": False}

        return {
            "ok": True,
            "chem_id": chem_id,
            "c_rate": c_rate,
            "temperature": temperature,
            "V": V_arr,
            "params": param_vals.astype(np.float32),
        }
    except Exception:
        return {"ok": False}


def main():
    output_path = "data/foundation/multichem_v2.h5"
    rng = np.random.default_rng(42)

    tasks = []
    for chem_id, cfg in CHEMISTRY_CONFIGS.items():
        sweep_keys = list(cfg["sweep_params"].keys())
        n = cfg["n_sets"]
        param_vals = np.zeros((n, len(sweep_keys)), dtype=np.float64)
        for j, (key, (lo, hi, dist)) in enumerate(cfg["sweep_params"].items()):
            if dist == "log":
                param_vals[:, j] = 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
            else:
                param_vals[:, j] = rng.uniform(lo, hi, n)

        for pi in range(n):
            for c_rate in cfg["c_rates"]:
                for temp in cfg["temperatures"]:
                    tasks.append((chem_id, cfg, param_vals[pi], sweep_keys, c_rate, temp))

    logger.info(f"Total tasks: {len(tasks)}")
    for chem_id, cfg in CHEMISTRY_CONFIGS.items():
        n = cfg["n_sets"] * len(cfg["c_rates"]) * len(cfg["temperatures"])
        logger.info(f"  Chem {chem_id} ({cfg['name']}): {n} tasks")

    t0 = time.time()
    results = []
    failed = 0
    chem_failed = {i: 0 for i in CHEMISTRY_CONFIGS}

    with Pool(processes=30) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single, tasks, chunksize=4)):
            if r["ok"]:
                results.append(r)
            else:
                failed += 1
                chem_failed[r.get("chem_id", -1)] = chem_failed.get(r.get("chem_id", -1), 0) + 1
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                logger.info(f"  {i+1}/{len(tasks)} ({failed} failed), "
                            f"{elapsed:.0f}s, {rate:.1f}/s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(results)} ok, {failed} failed, {elapsed:.0f}s")
    for cid in sorted(chem_failed):
        if chem_failed[cid] > 0:
            logger.info(f"  Chem {cid}: {chem_failed[cid]} failed")

    if not results:
        logger.error("No results!")
        return

    max_params = max(len(r["params"]) for r in results)
    chem_ids = np.array([r["chem_id"] for r in results], dtype=np.int32)
    c_rates = np.array([r["c_rate"] for r in results], dtype=np.float32)
    temps = np.array([r["temperature"] for r in results], dtype=np.float32)
    V_arr = np.array([r["V"] for r in results], dtype=np.float32)
    params_padded = np.zeros((len(results), max_params), dtype=np.float32)
    for i, r in enumerate(results):
        params_padded[i, :len(r["params"])] = r["params"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("V", data=V_arr, compression="gzip")
        f.create_dataset("chem_ids", data=chem_ids, compression="gzip")
        f.create_dataset("c_rates", data=c_rates, compression="gzip")
        f.create_dataset("temperatures", data=temps, compression="gzip")
        f.create_dataset("params", data=params_padded, compression="gzip")
        f.attrs["n_sims"] = len(results)
        f.attrs["n_time"] = N_TIME
        f.attrs["max_n_params"] = max_params
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        for chem_id, cfg in CHEMISTRY_CONFIGS.items():
            grp = f.create_group(f"chemistry_{chem_id}")
            grp.attrs["name"] = cfg["name"]
            grp.attrs["param_set"] = cfg["param_set"]
            grp.attrs["n_params"] = len(cfg["sweep_params"])
            grp.attrs["c_rates"] = cfg["c_rates"]
            grp.attrs["temperatures"] = cfg["temperatures"]
            grp.attrs["param_names"] = list(cfg["sweep_params"].keys())
            mask = chem_ids == chem_id
            grp.attrs["n_sims"] = int(mask.sum())
            if mask.sum() > 0:
                grp.attrs["V_min"] = float(V_arr[mask].min())
                grp.attrs["V_max"] = float(V_arr[mask].max())

    sz = Path(output_path).stat().st_size
    logger.info(f"Saved {output_path}: {len(results)} sims, {sz/1e6:.1f} MB, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
