#!/usr/bin/env python3
"""
Generate synthetic cycling trajectories for early-cycle life prediction.

Each "cell" has:
  - A degradation mode (SEI, LAM, R, or combinations)
  - Random degradation rates
  - N_cycles of degradation, observed as V(t) discharge curves

Uses PyBaMM SPM with degradation state parameterization.
"""

import numpy as np
import h5py
import pybamm
import time
import logging
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_TIME = 100
C_RATE = 1.0
N_CYCLES = 200

DEGRADATION_MODES = [
    "SEI_only",
    "LAM_pos_only",
    "R_only",
    "SEI_plus_LAM",
    "SEI_plus_R",
    "R_plus_LAM",
    "full_realistic",
]

PARAM_KEYS = [
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Cation transference number",
    "Initial SEI thickness [m]",
    "Negative electrode LAM fraction",
    "Positive electrode LAM fraction",
    "Resistance multiplier",
]
PARAM_SHORT = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]


def _generate_degradation_trajectory(mode, n_cycles, rng):
    """Generate parameter trajectory for a single cell."""
    params = np.zeros((n_cycles, 7))
    
    base = np.array([1e-14, 5e-16, 0.35, 5e-8, 0.0, 0.0, 1.0])
    params[0] = base.copy()
    
    if mode == "SEI_only":
        rate = rng.uniform(1e-8, 1e-6)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 3] = 5e-8 + rate * k
    
    elif mode == "LAM_pos_only":
        rate = rng.uniform(1e-4, 2e-3)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 5] = rate * k
    
    elif mode == "R_only":
        rate = rng.uniform(0.005, 0.04)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 6] = 1.0 + rate * k
    
    elif mode == "SEI_plus_LAM":
        sei_rate = rng.uniform(1e-8, 5e-7)
        lam_rate = rng.uniform(5e-5, 1e-3)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 3] = 5e-8 + sei_rate * k
            params[k, 5] = lam_rate * k
    
    elif mode == "SEI_plus_R":
        sei_rate = rng.uniform(1e-8, 5e-7)
        r_rate = rng.uniform(0.003, 0.02)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 3] = 5e-8 + sei_rate * k
            params[k, 6] = 1.0 + r_rate * k
    
    elif mode == "R_plus_LAM":
        r_rate = rng.uniform(0.003, 0.02)
        lam_rate = rng.uniform(5e-5, 1e-3)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 5] = lam_rate * k
            params[k, 6] = 1.0 + r_rate * k
    
    elif mode == "full_realistic":
        sei_rate = rng.uniform(1e-8, 3e-7)
        lam_neg_rate = rng.uniform(1e-5, 5e-4)
        lam_pos_rate = rng.uniform(1e-5, 5e-4)
        r_rate = rng.uniform(0.002, 0.015)
        for k in range(1, n_cycles):
            params[k] = base.copy()
            params[k, 3] = 5e-8 + sei_rate * k
            params[k, 4] = lam_neg_rate * k
            params[k, 5] = lam_pos_rate * k
            params[k, 6] = 1.0 + r_rate * k
    
    return params


def _run_single_discharge(param_vec, c_rate=1.0):
    """Run a single PyBaMM discharge with given parameters."""
    try:
        model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
        params = pybamm.ParameterValues("Prada2013")
        
        sei_defaults = {
            "SEI growth activation energy [J.mol-1]": 0.0,
            "SEI partial molar volume [m3.mol-1]": 9.585e-05,
            "SEI open-circuit potential [V]": 0.4,
            "SEI reaction exchange current density [A.m-2]": 1.5e-07,
            "Ratio of lithium moles to SEI moles": 2.0,
            "SEI resistivity [Ohm.m]": 200000.0,
        }
        for k, v in sei_defaults.items():
            params[k] = v
        
        for i, key in enumerate(PARAM_KEYS):
            if key == "Negative electrode LAM fraction":
                if param_vec[i] > 0.001:
                    orig = params["Negative electrode thickness [m]"]
                    params["Negative electrode thickness [m]"] = orig * (1 - param_vec[i])
            elif key == "Positive electrode LAM fraction":
                if param_vec[i] > 0.001:
                    orig = params["Positive electrode thickness [m]"]
                    params["Positive electrode thickness [m]"] = orig * (1 - param_vec[i])
            elif key == "Resistance multiplier":
                res_mult = param_vec[i]
                if res_mult > 1.01:
                    orig_cond = params["Electrolyte conductivity [S.m-1]"]
                    if not callable(orig_cond):
                        params["Electrolyte conductivity [S.m-1]"] = orig_cond / res_mult
            else:
                params[key] = param_vec[i]
        
        cap_nom = params["Nominal cell capacity [A.h]"]
        params["Current function [A]"] = c_rate * cap_nom
        
        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-4, atol=1e-6)
        sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
        
        t_end = 3600.0 / max(c_rate, 0.01) * 1.5
        sol = sim.solve([0, min(t_end, 5400)])
        
        V_var = sol["Voltage [V]"]
        t_resamp = np.linspace(sol.t[0], sol.t[-1], N_TIME)
        V_interp = np.array([float(V_var(t)) for t in t_resamp], dtype=np.float32)
        
        cap = float(cap_nom)
        
        if np.any(np.isnan(V_interp)) or np.any(V_interp < 1.0):
            return np.full(N_TIME, np.nan), 0.0, False
        
        return V_interp, cap, True
    except Exception as e:
        return np.full(N_TIME, np.nan), 0.0, False


def _simulate_cell(args):
    """Simulate a full degradation trajectory for one cell."""
    cell_idx, mode, n_cycles, seed = args
    rng = np.random.RandomState(seed)
    
    param_traj = _generate_degradation_trajectory(mode, n_cycles, rng)
    
    V_all = np.zeros((n_cycles, N_TIME))
    cap_all = np.zeros(n_cycles)
    valid = np.zeros(n_cycles, dtype=bool)
    
    for k in range(n_cycles):
        V, cap, ok = _run_single_discharge(param_traj[k], C_RATE)
        V_all[k] = V
        cap_all[k] = cap
        valid[k] = ok
        if not ok:
            V_all[k:] = np.nan
            cap_all[k:] = 0.0
            valid[k:] = False
            break
    
    n_valid = valid.sum()
    if n_valid < 5:
        return None
    
    return {
        "cell_idx": cell_idx,
        "mode": mode,
        "V": V_all[:n_valid],
        "cap": cap_all[:n_valid],
        "gt_params": param_traj[:n_valid],
        "n_valid": n_valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--n-cycles", type=int, default=N_CYCLES)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="data/synthetic_cycling/synthetic_cycling_lfp.h5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    rng = np.random.RandomState(args.seed)
    
    modes = []
    for _ in range(args.n_cells):
        modes.append(rng.choice(DEGRADATION_MODES))
    
    task_args = [
        (i, modes[i], args.n_cycles, rng.randint(0, 2**31))
        for i in range(args.n_cells)
    ]
    
    logger.info(f"Generating {args.n_cells} cells with {args.n_workers} workers...")
    t0 = time.time()
    
    results = []
    with Pool(args.n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_simulate_cell, task_args)):
            if res is not None:
                results.append(res)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (args.n_cells - i - 1) / rate
                logger.info(
                    f"  {i+1}/{args.n_cells} cells done "
                    f"({len(results)} valid), {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )
    
    elapsed = time.time() - t0
    logger.info(f"Generated {len(results)} valid cells in {elapsed:.0f}s")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        f.attrs["n_cells"] = len(results)
        f.attrs["n_time"] = N_TIME
        f.attrs["c_rate"] = C_RATE
        f.attrs["max_cycles"] = args.n_cycles
        f.attrs["param_names"] = PARAM_SHORT
        f.attrs["degradation_modes"] = DEGRADATION_MODES
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        grp = f.create_group("cells")
        for i, res in enumerate(results):
            cell_grp = grp.create_group(f"cell_{i:04d}")
            cell_grp.create_dataset("V", data=res["V"])
            cell_grp.create_dataset("capacity", data=res["cap"])
            cell_grp.create_dataset("gt_params", data=res["gt_params"])
            cell_grp.attrs["mode"] = np.bytes_(res["mode"])
            cell_grp.attrs["n_cycles"] = res["n_valid"]
    
    caps = [r["cap"] for r in results]
    lens = [len(c) for c in caps]
    logger.info(f"Saved to {output_path}")
    logger.info(f"  Cell cycle counts: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.0f}")


if __name__ == "__main__":
    main()
