"""Direct PyBaMM parameter fitting baseline.

Reviewers for NE / Joule will demand a non-PINN baseline that fits the
same experimental discharge curves with the same physics. This script
provides one: it runs PyBaMM with the LMB plating + SEI model and uses
SciPy to minimise voltage RMSE versus a single experimental cycle.

Outputs (under ``outputs/baselines/pybamm_fit/<cell>/cycle_<N>/``):
    fit_params.json         — recovered parameter values
    fit_curve.npz           — predicted vs observed voltage trajectories
    fit_summary.json        — RMSE (mV), n_evals, wall-time (s)

Usage::

    python scripts/30_baseline_pybamm_fit.py \
        --data data/raw/cellA.xlsx \
        --cycle 100 \
        --c-rate 1.0 \
        --maxiter 60 \
        --output outputs/baselines/pybamm_fit/cellA

This is intentionally CPU-only (PyBaMM solves on CPU), so it can run on
any worker node. Each fit takes a few minutes; on H20 you can dispatch
many cells in parallel via ``scripts/h20/05_pybamm_baseline.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("baseline_pybamm_fit")


# Parameters fitted (must exist in OKane2022). Bounds match configs/lmb.yaml.
FIT_PARAMS = {
    "Electrolyte diffusivity [m2.s-1]":                       (1.0e-11, 1.0e-9, "log"),
    "Cation transference number":                              (0.18,    0.45,   "lin"),
    "Positive particle diffusivity [m2.s-1]":                  (1.0e-16, 1.0e-13,"log"),
    "Positive electrode exchange-current density [A.m-2]":     (1.0e-4,  1.0e0,  "log"),
    "Lithium plating kinetic rate constant [m.s-1]":           (1.0e-12, 1.0e-7, "log"),
    "SEI kinetic rate constant [m.s-1]":                       (1.0e-14, 1.0e-10,"log"),
}


def _to_unit(values: np.ndarray) -> dict[str, float]:
    """Map [0, 1] vector → real parameter values respecting log/lin scaling."""
    out: dict[str, float] = {}
    for v, (name, (lo, hi, scale)) in zip(values, FIT_PARAMS.items()):
        v = float(np.clip(v, 0.0, 1.0))
        if scale == "log":
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            out[name] = float(10 ** (log_lo + v * (log_hi - log_lo)))
        else:
            out[name] = float(lo + v * (hi - lo))
    return out


def _from_unit(real: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(FIT_PARAMS))
    for i, (name, (lo, hi, scale)) in enumerate(FIT_PARAMS.items()):
        v = real.get(name, (lo + hi) / 2)
        if scale == "log":
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            out[i] = (np.log10(v) - log_lo) / (log_hi - log_lo)
        else:
            out[i] = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _extract_cycle(
    raw: dict[str, np.ndarray], cycle: int, min_pts: int = 15,
) -> Optional[dict[str, np.ndarray]]:
    """Pull a single discharge curve (negative current) out of a Neware export."""
    if "cycle" not in raw:
        raise ValueError("loader did not return a 'cycle' column")
    mask = (raw["cycle"] == cycle) & (raw["current"] < -0.005)
    if mask.sum() < min_pts:
        return None
    t = raw["time"][mask]
    v = raw["voltage"][mask]
    i = raw["current"][mask]
    valid = ~np.isnan(v) & ~np.isnan(t) & ~np.isnan(i)
    t, v, i = t[valid], v[valid], i[valid]
    if len(t) < min_pts:
        return None
    t = t - t[0]
    return {"time": t, "voltage": v, "current": i}


def _resample(t: np.ndarray, v: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.interp(grid, t_norm, v)


def _objective(
    params_unit: np.ndarray,
    target_v: np.ndarray,
    target_t: np.ndarray,
    c_rate: float,
    n_points: int,
    *,
    chemistry: str,
    mode: str,
    parameter_set: str,
) -> float:
    """RMSE in volts between PyBaMM solve and target curve."""
    from src.simulation.models import MetalBatteryDFN
    from src.simulation.solver import PybammSolver

    overrides = _to_unit(params_unit)
    battery = MetalBatteryDFN(chemistry=chemistry, mode=mode, parameter_set=parameter_set)
    solver = PybammSolver(battery)
    pyparams = battery.build_parameter_set(overrides)
    t_end = float(target_t[-1] - target_t[0]) if len(target_t) > 1 else 3600.0
    res = solver.solve(pyparams, c_rate=c_rate, t_end=t_end, n_points=n_points)
    if res is None:
        return 10.0           # large penalty on solver failure
    grid, v_pred = _resample(res["time"], res["voltage"], n_points)
    rmse = float(np.sqrt(np.mean((v_pred - target_v) ** 2)))
    return rmse


def fit_cell(
    raw: dict[str, np.ndarray],
    cycle: int,
    c_rate: float,
    n_points: int = 200,
    maxiter: int = 60,
    chemistry: str = "lmb",
    mode: str = "plating_dominant",
    parameter_set: str = "OKane2022",
    seed: int = 0,
) -> dict[str, Any]:
    cyc = _extract_cycle(raw, cycle)
    if cyc is None:
        raise ValueError(f"Cycle {cycle} is missing or has too few discharge points")
    target_t, target_v = _resample(cyc["time"], cyc["voltage"], n_points)

    from scipy.optimize import differential_evolution
    bounds = [(0.0, 1.0)] * len(FIT_PARAMS)
    t0 = time.time()
    n_evals = {"n": 0}

    def f(x):
        n_evals["n"] += 1
        return _objective(
            x, target_v, target_t, c_rate, n_points,
            chemistry=chemistry, mode=mode, parameter_set=parameter_set,
        )

    result = differential_evolution(
        f, bounds, maxiter=maxiter, popsize=8, polish=True,
        seed=seed, tol=1e-3, workers=1, updating="immediate",
    )
    elapsed = time.time() - t0

    best_real = _to_unit(result.x)
    rmse_mV = float(result.fun * 1000.0)

    # Rerun once at optimum to capture the trajectory.
    from src.simulation.models import MetalBatteryDFN
    from src.simulation.solver import PybammSolver
    battery = MetalBatteryDFN(chemistry=chemistry, mode=mode, parameter_set=parameter_set)
    solver = PybammSolver(battery)
    pyparams = battery.build_parameter_set(best_real)
    res_opt = solver.solve(
        pyparams, c_rate=c_rate,
        t_end=float(cyc["time"][-1] - cyc["time"][0]), n_points=n_points,
    )

    return {
        "fit_params": best_real,
        "rmse_mV": rmse_mV,
        "wall_time_s": elapsed,
        "n_evals": n_evals["n"],
        "optimum_x_unit": result.x.tolist(),
        "trajectory": {
            "time_s": res_opt["time"].tolist() if res_opt is not None else [],
            "voltage_pred": res_opt["voltage"].tolist() if res_opt is not None else [],
            "voltage_obs_resampled": target_v.tolist(),
            "time_norm": target_t.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Neware xlsx or CSV path")
    parser.add_argument("--cycle", type=int, default=10)
    parser.add_argument("--c-rate", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--mode", choices=["plating_dominant", "intercalation"],
                        default="plating_dominant")
    parser.add_argument("--parameter-set", default="OKane2022")
    parser.add_argument("--output", default="outputs/baselines/pybamm_fit/default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from src.data.loader import ExperimentalDataLoader

    loader = ExperimentalDataLoader()
    path = Path(args.data)
    if path.suffix == ".xlsx":
        raw = loader.load_neware_xlsx(path)
    elif path.suffix == ".csv":
        raw = loader.load_csv(path)
    else:
        raise ValueError(f"Unsupported data extension: {path.suffix}")

    logger.info("Loaded %d points, cycles %d..%d",
                len(raw["time"]),
                int(raw["cycle"].min()) if "cycle" in raw else -1,
                int(raw["cycle"].max()) if "cycle" in raw else -1)

    out_dir = Path(args.output) / f"cycle_{args.cycle}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = fit_cell(
        raw,
        cycle=args.cycle,
        c_rate=args.c_rate,
        n_points=args.n_points,
        maxiter=args.maxiter,
        mode=args.mode,
        parameter_set=args.parameter_set,
        seed=args.seed,
    )

    with (out_dir / "fit_params.json").open("w") as f:
        json.dump(result["fit_params"], f, indent=2)
    with (out_dir / "fit_summary.json").open("w") as f:
        json.dump({
            "rmse_mV": result["rmse_mV"],
            "wall_time_s": result["wall_time_s"],
            "n_evals": result["n_evals"],
            "cycle": args.cycle,
            "c_rate": args.c_rate,
            "mode": args.mode,
            "parameter_set": args.parameter_set,
        }, f, indent=2)
    np.savez_compressed(
        out_dir / "fit_curve.npz",
        time_pred=np.asarray(result["trajectory"]["time_s"]),
        voltage_pred=np.asarray(result["trajectory"]["voltage_pred"]),
        time_norm=np.asarray(result["trajectory"]["time_norm"]),
        voltage_obs=np.asarray(result["trajectory"]["voltage_obs_resampled"]),
    )

    logger.info("DONE  cycle=%d  RMSE=%.1f mV  evals=%d  wall=%.1fs",
                args.cycle, result["rmse_mV"], result["n_evals"],
                result["wall_time_s"])


if __name__ == "__main__":
    main()
