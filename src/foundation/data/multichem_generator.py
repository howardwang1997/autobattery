"""Multi-chemistry data generator for Battery Foundation Model.

Generates voltage curves across 6+ battery chemistries by sweeping
electrochemical parameters with PyBaMM.

Chemistries:
  0: Chen2020      (NMC811/graphite, 5Ah)
  1: Marquis2019   (NCA/graphite, 0.68Ah)
  2: Prada2013     (LFP/graphite, 2.3Ah)
  3: Ai2020        (LFP/graphite v2, 2.28Ah)
  4: OKane2022     (NMC532/graphite+SiOx, 5Ah, built-in degradation)
  5: Ramadass2004  (LiCoO2/graphite, 1Ah)
"""

import numpy as np
import h5py
import pybamm
import time
import logging
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_TIME = 200


@dataclass
class ChemistryConfig:
    name: str
    param_set: str
    model_type: str
    c_rates: list
    n_param_sets: int
    sweep_params: dict
    sei_params: dict


def _build_chemistry_configs():
    cfgs = {}

    cfgs[0] = ChemistryConfig(
        name="Chen2020_NMC811",
        param_set="Chen2020",
        model_type="lithium_ion.SPM",
        c_rates=[0.2, 0.5, 1.0, 2.0],
        n_param_sets=3000,
        sweep_params={
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Negative electrode conductivity [S.m-1]": (10, 500, "log"),
            "Positive electrode conductivity [S.m-1]": (0.1, 50, "log"),
        },
        sei_params={
            "SEI growth activation energy [J.mol-1]": 0.0,
            "SEI partial molar volume [m3.mol-1]": 9.585e-05,
            "SEI open-circuit potential [V]": 0.4,
            "SEI reaction exchange current density [A.m-2]": 1.5e-07,
            "Ratio of lithium moles to SEI moles": 2.0,
            "SEI resistivity [Ohm.m]": 200000.0,
        },
    )

    cfgs[1] = ChemistryConfig(
        name="Marquis2019_NCA",
        param_set="Marquis2019",
        model_type="lithium_ion.SPM",
        c_rates=[0.2, 0.5, 1.0, 2.0],
        n_param_sets=2000,
        sweep_params={
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
        sei_params={},
    )

    cfgs[2] = ChemistryConfig(
        name="Prada2013_LFP",
        param_set="Prada2013",
        model_type="lithium_ion.SPM",
        c_rates=[0.5, 1.0, 2.0, 3.0],
        n_param_sets=2000,
        sweep_params={
            "Negative particle diffusivity [m2.s-1]": (1e-15, 5e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (5e-17, 5e-15, "log"),
            "Cation transference number": (0.2, 0.45, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
        sei_params={
            "SEI growth activation energy [J.mol-1]": 0.0,
            "SEI partial molar volume [m3.mol-1]": 9.585e-05,
            "SEI open-circuit potential [V]": 0.4,
            "SEI reaction exchange current density [A.m-2]": 1.5e-07,
            "Ratio of lithium moles to SEI moles": 2.0,
            "SEI resistivity [Ohm.m]": 200000.0,
        },
    )

    cfgs[3] = ChemistryConfig(
        name="Ai2020_LFP_v2",
        param_set="Ai2020",
        model_type="lithium_ion.SPM",
        c_rates=[0.2, 0.5, 1.0, 2.0],
        n_param_sets=2000,
        sweep_params={
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
        sei_params={},
    )

    cfgs[4] = ChemistryConfig(
        name="OKane2022_NMC_degradation",
        param_set="OKane2022",
        model_type="lithium_ion.DFN",
        c_rates=[0.2, 0.5, 1.0],
        n_param_sets=2000,
        sweep_params={
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Negative electrode conductivity [S.m-1]": (10, 500, "log"),
            "Positive electrode conductivity [S.m-1]": (0.1, 50, "log"),
        },
        sei_params={},
    )

    cfgs[5] = ChemistryConfig(
        name="Ramadass2004_LCO",
        param_set="Ramadass2004",
        model_type="lithium_ion.SPM",
        c_rates=[0.2, 0.5, 1.0, 2.0],
        n_param_sets=2000,
        sweep_params={
            "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-13, "log"),
            "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-12, "log"),
            "Cation transference number": (0.2, 0.5, "uniform"),
            "Electrolyte conductivity [S.m-1]": (0.5, 2.0, "uniform"),
            "Negative electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
            "Positive electrode exchange-current density [A.m-2]": (1e-5, 1e-3, "log"),
        },
        sei_params={},
    )

    return cfgs


CHEMISTRIES = _build_chemistry_configs()


def _sample_params(sweep_params, n, rng):
    keys = list(sweep_params.keys())
    vals = np.zeros((n, len(keys)), dtype=np.float64)
    for j, (key, (lo, hi, dist)) in enumerate(sweep_params.items()):
        if dist == "log":
            vals[:, j] = 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
        else:
            vals[:, j] = rng.uniform(lo, hi, n)
    return keys, vals


def _run_single_sim(args):
    chem_id, cfg, param_idx, param_vals, c_rate = args
    try:
        if cfg.model_type == "lithium_ion.DFN":
            model = pybamm.lithium_ion.DFN({"SEI": "reaction limited", "SEI porosity change": "true"})
        elif cfg.model_type == "lithium_ion.SPM":
            if cfg.sei_params:
                model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
            else:
                model = pybamm.lithium_ion.SPM()
        else:
            model = pybamm.lithium_ion.SPM()

        params = pybamm.ParameterValues(cfg.param_set)

        for k, v in cfg.sei_params.items():
            params[k] = v

        sweep_keys = list(cfg.sweep_params.keys())
        for j, key in enumerate(sweep_keys):
            params[key] = param_vals[j]

        cap = params.get("Nominal cell capacity [A.h]", 1.0)
        if cap < 0.01:
            cap = 1.0
        params["Current function [A]"] = c_rate * cap

        solver = pybamm.CasadiSolver(
            mode="safe", rtol=1e-4, atol=1e-6, max_step_decrease_count=5
        )
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
            "param_idx": param_idx,
            "c_rate": c_rate,
            "V": V_arr,
            "V_min": float(V_arr.min()),
            "V_max": float(V_arr.max()),
            "V_mean": float(V_arr.mean()),
            "V_std": float(V_arr.std()),
            "params": param_vals.astype(np.float32),
        }
    except Exception:
        return {"ok": False}


def generate_multichem_data(output_path, chemistry_ids=None, processes=20):
    if chemistry_ids is None:
        chemistry_ids = list(CHEMISTRIES.keys())

    rng = np.random.default_rng(42)
    tasks = []

    for chem_id in chemistry_ids:
        cfg = CHEMISTRIES[chem_id]
        keys, vals = _sample_params(cfg.sweep_params, cfg.n_param_sets, rng)
        for pi in range(cfg.n_param_sets):
            for c_rate in cfg.c_rates:
                tasks.append((chem_id, cfg, pi, vals[pi], c_rate))

    logger.info(f"Total tasks: {len(tasks)}")
    for chem_id in chemistry_ids:
        cfg = CHEMISTRIES[chem_id]
        n = cfg.n_param_sets * len(cfg.c_rates)
        logger.info(f"  Chem {chem_id} ({cfg.name}): {n} tasks")

    t0 = time.time()
    results = []
    failed = 0

    with Pool(processes=processes) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_single_sim, tasks, chunksize=4)):
            if r["ok"]:
                results.append(r)
            else:
                failed += 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                logger.info(f"  {i+1}/{len(tasks)} ({failed} failed), "
                            f"{elapsed:.0f}s, {rate:.1f}/s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(results)} ok, {failed} failed, {elapsed:.0f}s")

    if not results:
        logger.error("No results!")
        return

    results.sort(key=lambda r: (r["chem_id"], r["param_idx"], r["c_rate"]))

    chem_ids_arr = np.array([r["chem_id"] for r in results], dtype=np.int32)
    c_rates_arr = np.array([r["c_rate"] for r in results], dtype=np.float32)
    V_arr = np.array([r["V"] for r in results], dtype=np.float32)
    max_params = max(len(r["params"]) for r in results)
    params_padded = np.zeros((len(results), max_params), dtype=np.float32)
    for i, r in enumerate(results):
        params_padded[i, :len(r["params"])] = r["params"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("V", data=V_arr, compression="gzip")
        f.create_dataset("chem_ids", data=chem_ids_arr, compression="gzip")
        f.create_dataset("c_rates", data=c_rates_arr, compression="gzip")
        f.create_dataset("params", data=params_padded, compression="gzip")

        param_set_ids = np.zeros(len(results), dtype=np.int32)
        for i, r in enumerate(results):
            cfg = CHEMISTRIES[r["chem_id"]]
            if r["c_rate"] in cfg.c_rates:
                param_set_ids[i] = r["param_idx"] * len(cfg.c_rates) + cfg.c_rates.index(r["c_rate"])
        f.create_dataset("param_set_ids", data=param_set_ids, compression="gzip")

        f.attrs["n_sims"] = len(results)
        f.attrs["n_time"] = N_TIME
        f.attrs["max_n_params"] = max_params
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        for chem_id in chemistry_ids:
            cfg = CHEMISTRIES[chem_id]
            grp = f.create_group(f"chemistry_{chem_id}")
            grp.attrs["name"] = cfg.name
            grp.attrs["param_set"] = cfg.param_set
            grp.attrs["n_params"] = len(cfg.sweep_params)
            grp.attrs["c_rates"] = cfg.c_rates
            grp.attrs["param_names"] = list(cfg.sweep_params.keys())
            mask = chem_ids_arr == chem_id
            n_ok = int(mask.sum())
            grp.attrs["n_sims"] = n_ok
            if n_ok > 0:
                grp.attrs["V_min_global"] = float(V_arr[mask].min())
                grp.attrs["V_max_global"] = float(V_arr[mask].max())
            else:
                grp.attrs["V_min_global"] = 0.0
                grp.attrs["V_max_global"] = 1.0
                logger.warning(f"  Chemistry {chem_id} ({cfg.name}): ALL SIMS FAILED!")

    logger.info(f"Saved {output_path}: {len(results)} sims, {elapsed:.0f}s")
    return results
