"""PyBaMM multi-cycle degradation ground truth simulation (parameterized).

Instead of using PyBaMM's built-in degradation models (which require missing params),
we explicitly parameterize degradation state and run single discharges.
This gives us TRUE ground truth since we know the exact parameter values.

Approach: For each scenario, define a parameter trajectory over N "cycles",
run PyBaMM single-discharge at each state, record V(t) + known ground truth.

Scenarios:
1. SEI growth only
2. Positive electrode LAM
3. Resistance growth  
4. SEI + LAM + R (realistic)
5. All modes active (full)
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import pybamm
import time
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/ground_truth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TIME = 100
C_RATE = 1.0

PARAM_RANGES = {
    "Negative particle diffusivity [m2.s-1]": (1e-15, 5e-13),
    "Positive particle diffusivity [m2.s-1]": (5e-17, 5e-15),
    "Cation transference number": (0.2, 0.45),
    "Initial SEI thickness [m]": (1e-9, 1e-6),
    "Negative electrode LAM fraction": (0.0, 0.3),
    "Positive electrode LAM fraction": (0.0, 0.3),
    "Resistance multiplier": (1.0, 5.0),
}

SEI_DEFAULTS = {
    "SEI growth activation energy [J.mol-1]": 0.0,
    "SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI open-circuit potential [V]": 0.4,
    "SEI reaction exchange current density [A.m-2]": 1.5e-07,
    "Ratio of lithium moles to SEI moles": 2.0,
    "SEI resistivity [Ohm.m]": 200000.0,
}

PNAMES = ["D_n", "D_p", "t+", "SEI_thick", "LAM_neg", "LAM_pos", "R_mult"]


def run_discharge(params_overrides, c_rate=C_RATE):
    """Run a single discharge with given parameter overrides."""
    try:
        model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
        params = pybamm.ParameterValues("Prada2013")

        for k, v in SEI_DEFAULTS.items():
            params[k] = v

        for k, v in params_overrides.items():
            if k == "Negative electrode LAM fraction":
                lam_neg = v
                if lam_neg > 0.001:
                    orig = params["Negative electrode thickness [m]"]
                    params["Negative electrode thickness [m]"] = orig * (1 - lam_neg)
            elif k == "Positive electrode LAM fraction":
                lam_pos = v
                if lam_pos > 0.001:
                    orig = params["Positive electrode thickness [m]"]
                    params["Positive electrode thickness [m]"] = orig * (1 - lam_pos)
            elif k == "Resistance multiplier":
                res_mult = v
                if res_mult > 1.01:
                    cond = params["Electrolyte conductivity [S.m-1]"]
                    if not callable(cond):
                        params["Electrolyte conductivity [S.m-1]"] = cond / res_mult
            else:
                params[k] = v

        cap = params["Nominal cell capacity [A.h]"]
        params["Current function [A]"] = c_rate * cap

        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-4, atol=1e-6)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
        t_end = 3600.0 / max(c_rate, 0.01) * 1.5
        sol = sim.solve([0, min(t_end, 5400)])

        V_var = sol["Voltage [V]"]
        t_resamp = np.linspace(sol.t[0], sol.t[-1], N_TIME)
        V_arr = np.array([float(V_var(t)) for t in t_resamp], dtype=np.float32)

        if np.any(np.isnan(V_arr)) or np.any(V_arr < 1.5):
            return None

        dt = np.diff(sol.t)
        I_arr = sol["Current [A]"].data
        q_disc = -np.sum(I_arr[:-1] * dt) / 3600

        return {"V": V_arr, "capacity": float(q_disc)}
    except Exception:
        return None


def get_baseline_params():
    """Get median parameter values as baseline."""
    return {
        "Negative particle diffusivity [m2.s-1]": 1e-14,
        "Positive particle diffusivity [m2.s-1]": 5e-16,
        "Cation transference number": 0.35,
        "Initial SEI thickness [m]": 5e-8,
        "Negative electrode LAM fraction": 0.0,
        "Positive electrode LAM fraction": 0.0,
        "Resistance multiplier": 1.0,
    }


def build_param_trajectory(baseline, scenario_fn, progress):
    """Build parameter dict for a given degradation progress (0..1)."""
    changes = scenario_fn(progress)
    params = dict(baseline)
    for key, value in changes.items():
        params[key] = value
    return params, changes


SCENARIOS = {
    "SEI_only": lambda p: {
        "Initial SEI thickness [m]": 5e-8 * (1 + p * 200),
    },
    "SEI_fast": lambda p: {
        "Initial SEI thickness [m]": 5e-8 * (1 + p * 2000),
    },
    "LAM_pos_mild": lambda p: {
        "Positive electrode LAM fraction": p * 0.15,
    },
    "LAM_pos_severe": lambda p: {
        "Positive electrode LAM fraction": p * 0.30,
    },
    "R_growth": lambda p: {
        "Resistance multiplier": 1.0 + p * 4.0,
    },
    "SEI_plus_LAM": lambda p: {
        "Initial SEI thickness [m]": 5e-8 * (1 + p * 500),
        "Positive electrode LAM fraction": p * 0.20,
    },
    "SEI_plus_R": lambda p: {
        "Initial SEI thickness [m]": 5e-8 * (1 + p * 300),
        "Resistance multiplier": 1.0 + p * 3.0,
    },
    "R_plus_LAM": lambda p: {
        "Resistance multiplier": 1.0 + p * 3.0,
        "Positive electrode LAM fraction": p * 0.20,
    },
    "full_realistic": lambda p: {
        "Initial SEI thickness [m]": 5e-8 * (1 + p * 300),
        "Positive electrode LAM fraction": p * 0.15,
        "Negative electrode LAM fraction": p * 0.10,
        "Resistance multiplier": 1.0 + p * 2.5,
    },
    "full_aggressive": lambda p: {
        "Initial SEI thickness [m]": 5e-8 * (1 + p * 1000),
        "Positive electrode LAM fraction": p * 0.25,
        "Negative electrode LAM fraction": p * 0.15,
        "Resistance multiplier": 1.0 + p * 4.0,
        "Positive particle diffusivity [m2.s-1]": 5e-16 * (1 - p * 0.5),
    },
}


def run_scenario(scenario_name, scenario_fn, n_cycles=200):
    """Run a single degradation scenario."""
    baseline = get_baseline_params()

    base_result = run_discharge(baseline)
    if base_result is None:
        return {"scenario": scenario_name, "ok": False, "reason": "baseline failed"}

    V_ref = base_result["V"]

    voltages = [V_ref]
    capacities = [base_result["capacity"]]
    gt_params = [np.array([baseline[k] for k in PARAM_RANGES], dtype=np.float64)]

    for cyc in range(1, n_cycles + 1):
        progress = cyc / n_cycles
        params, changes = build_param_trajectory(baseline, scenario_fn, progress)

        result = run_discharge(params)
        if result is None:
            logger.debug(f"  Cycle {cyc} failed")
            continue

        voltages.append(result["V"])
        capacities.append(result["capacity"])

        pvec = np.array([params[k] for k in PARAM_RANGES], dtype=np.float64)
        gt_params.append(pvec)

        if cyc % 50 == 0:
            cap_fade = (1 - result["capacity"] / capacities[0]) * 100
            logger.info(f"    Cycle {cyc}/{n_cycles}, fade={cap_fade:.1f}%")

    return {
        "scenario": scenario_name,
        "ok": True,
        "V_cycles": np.array(voltages, dtype=np.float32),
        "capacities": np.array(capacities, dtype=np.float32),
        "gt_params": np.array(gt_params, dtype=np.float64),
        "n_cycles_completed": len(voltages),
    }


def main():
    n_cycles = 200

    baseline = get_baseline_params()
    logger.info("Running baseline discharge...")
    base = run_discharge(baseline)
    if base is None:
        logger.error("Baseline failed!")
        sys.exit(1)
    logger.info(f"Baseline: V range [{base['V'].min():.3f}, {base['V'].max():.3f}], "
                f"cap={base['capacity']*1000:.1f} mAh")

    output_path = OUT_DIR / "ground_truth_multicycle.h5"

    with h5py.File(str(output_path), "w") as f:
        f.attrs["n_time"] = N_TIME
        f.attrs["n_cycles"] = n_cycles
        f.attrs["c_rate"] = C_RATE
        f.attrs["param_names"] = PNAMES
        f.attrs["param_keys"] = list(PARAM_RANGES.keys())
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        base_grp = f.create_group("baseline")
        base_grp.create_dataset("V", data=base["V"], compression="gzip")
        base_grp.create_dataset("capacity", data=base["capacity"])
        base_grp.create_dataset("params", data=np.array([baseline[k] for k in PARAM_RANGES]))

    t_total = time.time()
    successful = 0

    for i, (sc_name, sc_fn) in enumerate(SCENARIOS.items()):
        logger.info(f"\n[{i+1}/{len(SCENARIOS)}] {sc_name}...")
        t0 = time.time()

        result = run_scenario(sc_name, sc_fn, n_cycles)

        if result["ok"]:
            with h5py.File(str(output_path), "a") as f:
                grp = f.create_group(sc_name)
                grp.attrs["n_cycles"] = result["n_cycles_completed"]
                grp.create_dataset("V_cycles", data=result["V_cycles"], compression="gzip")
                grp.create_dataset("capacity", data=result["capacities"], compression="gzip")
                grp.create_dataset("gt_params", data=result["gt_params"], compression="gzip")

            cap_fade = (1 - result["capacities"][-1] / result["capacities"][0]) * 100
            logger.info(f"  OK: {result['n_cycles_completed']} cycles in {time.time()-t0:.0f}s, "
                        f"fade={cap_fade:.1f}%")
            successful += 1
        else:
            logger.info(f"  FAILED: {result.get('reason', 'unknown')}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Done: {successful}/{len(SCENARIOS)} scenarios in {time.time()-t_total:.0f}s")
    logger.info(f"Saved to {output_path}")

    with h5py.File(str(output_path), "r") as f:
        logger.info(f"\nSummary:")
        for sc_name in SCENARIOS:
            if sc_name in f:
                grp = f[sc_name]
                caps = grp["capacity"][:]
                fade = (1 - caps[-1] / caps[0]) * 100
                gt = grp["gt_params"][:]
                logger.info(f"  {sc_name:25s}: {len(caps):3d} cycles, fade={fade:5.1f}%")
                logger.info(f"    SEI: {gt[0,3]*1e9:.1f} → {gt[-1,3]*1e9:.1f} nm")
                logger.info(f"    LAM_pos: {gt[0,5]:.3f} → {gt[-1,5]:.3f}")
                logger.info(f"    R_mult: {gt[0,6]:.2f} → {gt[-1,6]:.2f}")


if __name__ == "__main__":
    main()
