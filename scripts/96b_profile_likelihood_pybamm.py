"""Profile-likelihood identifiability analysis with PyBaMM forward.

Profile likelihood (Raue et al., 2009 *Bioinformatics*; applied to
batteries by Sulzer et al., 2021 *J. Electrochem. Soc.*) is the
gold-standard tool for *practical* parameter identifiability — much
more rigorous than Fisher information, which is only a local linear
approximation around a point estimate.

The script's predecessor (``scripts/96_profile_likelihood.py``) was
blocked: it used a low-accuracy MLP surrogate, so every PL profile came
out flat regardless of true identifiability. We fix this by going back
to the PyBaMM forward solver — slower but rigorous — and by using a
**warm-start L-BFGS** inner optimisation seeded from the global MAP, so
every grid point converges in 5–30 PyBaMM evaluations rather than
hundreds.

Algorithm
---------
For each parameter θ_i we want to profile:

  1. Find the global MAP θ★ via differential evolution + L-BFGS polish.
  2. Build a logarithmic grid for θ_i around θ★_i (default 9 points
     spanning ±2 decades).
  3. At each grid value θ_i = g_k:
        • fix θ_i = g_k
        • re-optimise the remaining parameters with L-BFGS, starting
          from θ★. (Warm-starting matters — random restarts cost 10×.)
        • record the resulting RMSE → χ²(g_k).
  4. The profile likelihood curve is χ²(g_k) − χ²(θ★) versus g_k.
     A flat profile (Δχ² < threshold across the whole grid) means θ_i
     is *practically non-identifiable*; a clear minimum at θ★_i means
     θ_i is identifiable; a one-sided ridge means it has a confidence
     bound on only one side (common with kinetic params at very high
     C-rate).

Confidence threshold
--------------------
For a single parameter the 95% PL confidence interval is bounded by
χ²_min + Δ where Δ = χ²_{1, 0.95} = 3.84. Lines drawn at Δχ² ∈ {3.84,
6.63} on every plot.

Usage
-----
    conda activate autobattery
    python scripts/96b_profile_likelihood_pybamm.py \
        --data data/raw/lmb_long_cycle/cell_001.xlsx \
        --cycle 50 \
        --c-rate 1.0 \
        --params D_e t_plus D_p k_sei k_plating \
        --grid-points 9 \
        --output outputs/profile_likelihood/cell_001 \
        --workers 4

Wall-time budget: ≈ 30–60 minutes per parameter on a single CPU; the
``--workers`` flag profiles N parameters in parallel via multiprocessing.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("profile_likelihood")


# Parameters available for profiling — bounds match configs/lmb.yaml.
PROFILE_PARAMS: dict[str, tuple[float, float]] = {
    "D_e":             (1.0e-11, 1.0e-9),
    "t_plus":          (0.18,    0.45),
    "D_p":             (1.0e-16, 1.0e-13),
    "j0_p":            (1.0e-4,  1.0e0),
    "k_plating":       (1.0e-12, 1.0e-7),
    "k_sei":           (1.0e-14, 1.0e-10),
    "L_sei_0":         (1.0e-10, 1.0e-7),
}

PYBAMM_KEY_MAP = {
    "D_e":       "Electrolyte diffusivity [m2.s-1]",
    "t_plus":    "Cation transference number",
    "D_p":       "Positive particle diffusivity [m2.s-1]",
    "j0_p":      "Positive electrode exchange-current density [A.m-2]",
    "k_plating": "Lithium plating kinetic rate constant [m.s-1]",
    "k_sei":     "SEI kinetic rate constant [m.s-1]",
    "L_sei_0":   "Initial inner SEI thickness [m]",
}

LOG_SCALE_PARAMS = {"D_e", "D_p", "j0_p", "k_plating", "k_sei", "L_sei_0"}

CONFIDENCE_THRESHOLDS = {
    "95%": 3.84,    # χ²_{1, 0.95}
    "99%": 6.63,    # χ²_{1, 0.99}
}


# ---------------------------------------------------------------------------
# Forward model wrapper
# ---------------------------------------------------------------------------


def _to_real(value_unit: float, name: str) -> float:
    """Map [0, 1] interval coordinate to the real parameter value."""
    lo, hi = PROFILE_PARAMS[name]
    v = float(np.clip(value_unit, 0.0, 1.0))
    if name in LOG_SCALE_PARAMS:
        return float(10 ** (np.log10(lo) + v * (np.log10(hi) - np.log10(lo))))
    return float(lo + v * (hi - lo))


def _to_unit(real_value: float, name: str) -> float:
    lo, hi = PROFILE_PARAMS[name]
    if name in LOG_SCALE_PARAMS:
        return (np.log10(real_value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return (real_value - lo) / (hi - lo)


def _resample(t: np.ndarray, v: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(t) < 2:
        return np.linspace(0, 1, n), np.full(n, v.mean() if len(v) else np.nan)
    t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.interp(grid, t_norm, v)


def _extract_target_curve(raw: dict[str, np.ndarray], cycle: int, n_points: int):
    if "cycle" not in raw:
        raise ValueError("loader did not return a 'cycle' column")
    mask = (raw["cycle"] == cycle) & (raw["current"] < -0.005)
    if mask.sum() < 15:
        raise ValueError(f"Cycle {cycle} has too few discharge points ({mask.sum()})")
    t, v, i = raw["time"][mask], raw["voltage"][mask], raw["current"][mask]
    valid = ~np.isnan(v) & ~np.isnan(t) & ~np.isnan(i)
    t, v = t[valid], v[valid]
    if len(t) < 15:
        raise ValueError("not enough valid points after NaN filter")
    return _resample(t - t[0], v, n_points)


def _objective(
    params_unit: np.ndarray,
    param_names: list[str],
    target_v: np.ndarray,
    c_rate: float,
    n_points: int,
    chemistry: str,
    mode: str,
    parameter_set: str,
) -> float:
    """RMSE in volts between PyBaMM solve and target curve."""
    from src.simulation.models import MetalBatteryDFN
    from src.simulation.solver import PybammSolver

    overrides = {
        PYBAMM_KEY_MAP[name]: _to_real(params_unit[k], name)
        for k, name in enumerate(param_names)
    }
    battery = MetalBatteryDFN(chemistry=chemistry, mode=mode, parameter_set=parameter_set)
    solver = PybammSolver(battery)
    pyparams = battery.build_parameter_set(overrides)
    res = solver.solve(pyparams, c_rate=c_rate, t_end=3600.0, n_points=n_points)
    if res is None:
        return 5.0
    _, v_pred = _resample(res["time"], res["voltage"], n_points)
    return float(np.sqrt(np.mean((v_pred - target_v) ** 2)))


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


@dataclass
class FitContext:
    target_v: np.ndarray
    c_rate: float
    n_points: int
    chemistry: str
    mode: str
    parameter_set: str
    param_names: list[str]


def _make_objective(ctx: FitContext):
    def f(x):
        return _objective(
            np.asarray(x), ctx.param_names, ctx.target_v, ctx.c_rate,
            ctx.n_points, ctx.chemistry, ctx.mode, ctx.parameter_set,
        )
    return f


def find_map(ctx: FitContext, maxiter: int = 50, seed: int = 0) -> tuple[np.ndarray, float]:
    """Stage 1 of the workflow: global MAP via differential evolution."""
    from scipy.optimize import differential_evolution

    bounds = [(0.0, 1.0)] * len(ctx.param_names)
    f = _make_objective(ctx)

    t0 = time.time()
    result = differential_evolution(
        f, bounds, maxiter=maxiter, popsize=10, polish=True,
        seed=seed, tol=1e-3, workers=1, updating="immediate",
    )
    elapsed = time.time() - t0
    logger.info("MAP found in %.0fs: RMSE=%.4f V (n_evals=%d)",
                elapsed, result.fun, result.nfev)
    return result.x, float(result.fun)


def profile_one_parameter(
    ctx: FitContext,
    map_x: np.ndarray,
    target_param: str,
    grid_unit: np.ndarray,
    inner_maxiter: int = 30,
) -> dict[str, Any]:
    """Stage 2: sweep ``target_param`` across grid; refit other params."""
    from scipy.optimize import minimize

    j = ctx.param_names.index(target_param)
    free_idx = [k for k in range(len(ctx.param_names)) if k != j]
    bounds_free = [(0.0, 1.0)] * len(free_idx)

    f_full = _make_objective(ctx)

    def f_free(x_free, fixed_unit):
        x = map_x.copy()
        x[j] = fixed_unit
        for k, idx in enumerate(free_idx):
            x[idx] = x_free[k]
        return f_full(x)

    grid_real = np.array([_to_real(g, target_param) for g in grid_unit])
    rmses = np.zeros_like(grid_unit)
    fitted_xs = np.zeros((len(grid_unit), len(map_x)))
    n_evals_total = 0

    for k, g in enumerate(grid_unit):
        # Warm-start from MAP (drop the j-th coordinate).
        x0 = map_x[free_idx].copy()
        try:
            r = minimize(
                f_free, x0,
                args=(g,),
                method="L-BFGS-B",
                bounds=bounds_free,
                options={"maxiter": inner_maxiter, "ftol": 1e-6},
            )
            rmses[k] = float(r.fun)
            x_full = map_x.copy()
            x_full[j] = g
            for kk, idx in enumerate(free_idx):
                x_full[idx] = r.x[kk]
            fitted_xs[k] = x_full
            n_evals_total += int(r.nfev)
        except Exception as exc:        # noqa: BLE001
            logger.warning("L-BFGS at %s grid %d failed: %s", target_param, k, exc)
            rmses[k] = np.nan
            fitted_xs[k] = map_x

    # χ²-style residual: assume noise σ_v ≈ 5 mV.
    sigma_v = 5e-3
    chi2 = (rmses ** 2) / (sigma_v ** 2)
    chi2_min = float(np.nanmin(chi2)) if np.any(np.isfinite(chi2)) else np.nan
    delta_chi2 = chi2 - chi2_min

    finite = np.isfinite(delta_chi2)
    flat = bool(finite.all() and np.nanmax(delta_chi2) < CONFIDENCE_THRESHOLDS["95%"])

    return {
        "param": target_param,
        "grid_unit": grid_unit.tolist(),
        "grid_real": grid_real.tolist(),
        "rmses_V": rmses.tolist(),
        "chi2": chi2.tolist(),
        "delta_chi2": delta_chi2.tolist(),
        "n_evals": int(n_evals_total),
        "flat_profile_at_95pct": flat,
        "fitted_x_per_grid": fitted_xs.tolist(),
    }


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------


def _profile_worker(args):
    ctx, map_x, target_param, grid, inner_maxiter = args
    t0 = time.time()
    result = profile_one_parameter(ctx, map_x, target_param, grid, inner_maxiter)
    elapsed = time.time() - t0
    result["wall_time_s"] = elapsed
    logger.info("[%s] done in %.0fs (%d evals)",
                target_param, elapsed, result["n_evals"])
    return result


def _make_grid(name: str, map_real: float, n: int, span_decades: float = 2.0):
    """Build a logarithmic / linear grid centred on the MAP value."""
    lo, hi = PROFILE_PARAMS[name]
    if name in LOG_SCALE_PARAMS:
        log_lo = max(np.log10(lo), np.log10(map_real) - span_decades)
        log_hi = min(np.log10(hi), np.log10(map_real) + span_decades)
        grid_real = np.logspace(log_lo, log_hi, n)
    else:
        delta = (hi - lo) * 0.4
        grid_real = np.linspace(
            max(lo, map_real - delta), min(hi, map_real + delta), n,
        )
    return np.array([_to_unit(g, name) for g in grid_real])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pl_curves(
    profiles: list[dict[str, Any]], out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(profiles)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows),
                              squeeze=False)
    for ax, p in zip(axes.flat, profiles):
        grid_real = np.array(p["grid_real"])
        delta = np.array(p["delta_chi2"])
        if p["param"] in LOG_SCALE_PARAMS:
            ax.set_xscale("log")
        ax.plot(grid_real, delta, "o-", color="tab:blue")
        for label, thr in CONFIDENCE_THRESHOLDS.items():
            ax.axhline(thr, color="grey", ls="--", lw=0.8,
                       label=f"χ²_1, {label}={thr:.2f}")
        ax.set_xlabel(p["param"])
        ax.set_ylabel("Δχ²")
        flat = "flat (UN)" if p["flat_profile_at_95pct"] else "well (ID)"
        ax.set_title(f"{p['param']}  [{flat}]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(profiles):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_likelihood.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Neware xlsx or CSV")
    parser.add_argument("--cycle", type=int, default=10)
    parser.add_argument("--c-rate", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--params", nargs="+", default=list(PROFILE_PARAMS.keys()),
                        help="parameters to profile")
    parser.add_argument("--grid-points", type=int, default=9)
    parser.add_argument("--span-decades", type=float, default=2.0,
                        help="profile grid span in log decades around MAP")
    parser.add_argument("--map-maxiter", type=int, default=50,
                        help="DE iterations for MAP search")
    parser.add_argument("--inner-maxiter", type=int, default=30,
                        help="L-BFGS iterations per grid point")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel workers (one per profiled parameter)")
    parser.add_argument("--mode", choices=["plating_dominant", "intercalation"],
                        default="plating_dominant")
    parser.add_argument("--parameter-set", default="OKane2022")
    parser.add_argument("--chemistry", default="lmb")
    parser.add_argument("--output", default="outputs/profile_likelihood/default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bad = [p for p in args.params if p not in PROFILE_PARAMS]
    if bad:
        raise SystemExit(f"unknown profile params: {bad}; "
                         f"choose from {list(PROFILE_PARAMS.keys())}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load + preprocess --------------------------------------------------
    from src.data.loader import ExperimentalDataLoader
    loader = ExperimentalDataLoader()
    path = Path(args.data)
    if path.suffix.lower() == ".xlsx":
        raw = loader.load_neware_xlsx(path)
    elif path.suffix.lower() == ".csv":
        raw = loader.load_csv(path)
    else:
        raise SystemExit(f"unsupported extension {path.suffix}")
    _, target_v = _extract_target_curve(raw, args.cycle, args.n_points)
    logger.info("loaded cycle %d: V range %.3f .. %.3f",
                args.cycle, target_v.min(), target_v.max())

    # Build context across both stages.
    ctx = FitContext(
        target_v=target_v,
        c_rate=args.c_rate,
        n_points=args.n_points,
        chemistry=args.chemistry,
        mode=args.mode,
        parameter_set=args.parameter_set,
        param_names=list(args.params),
    )

    # Stage 1: MAP -------------------------------------------------------
    logger.info("stage 1: finding global MAP via DE (%d iters)", args.map_maxiter)
    map_x, map_rmse = find_map(ctx, maxiter=args.map_maxiter, seed=args.seed)
    map_real = {
        name: _to_real(map_x[k], name) for k, name in enumerate(args.params)
    }

    # Stage 2: profile each parameter -----------------------------------
    grids = {
        name: _make_grid(name, map_real[name], args.grid_points, args.span_decades)
        for name in args.params
    }

    work = [
        (ctx, map_x, name, grids[name], args.inner_maxiter)
        for name in args.params
    ]

    profiles: list[dict[str, Any]] = []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for r in pool.imap_unordered(_profile_worker, work):
                profiles.append(r)
    else:
        for w in work:
            profiles.append(_profile_worker(w))

    profiles.sort(key=lambda p: args.params.index(p["param"]))

    # Save --------------------------------------------------------------
    summary = {
        "data_path": str(path),
        "cycle": args.cycle,
        "c_rate": args.c_rate,
        "chemistry": args.chemistry,
        "mode": args.mode,
        "parameter_set": args.parameter_set,
        "map_x_unit": map_x.tolist(),
        "map_real": map_real,
        "map_rmse_V": map_rmse,
        "params_profiled": list(args.params),
        "grid_points": args.grid_points,
        "span_decades": args.span_decades,
        "profiles": profiles,
        "verdicts": {
            p["param"]: ("non-identifiable (flat profile)"
                         if p["flat_profile_at_95pct"]
                         else "identifiable (clear minimum)")
            for p in profiles
        },
    }
    with (out / "profile_results.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    plot_pl_curves(profiles, out)

    # Markdown summary --------------------------------------------------
    md = [
        f"# Profile-likelihood analysis — cycle {args.cycle}, C={args.c_rate:g}",
        "",
        f"Data: `{path}`",
        f"PyBaMM mode: `{args.mode}`, parameter set: `{args.parameter_set}`",
        f"Global MAP RMSE: **{map_rmse * 1000:.1f} mV**",
        "",
        "## Per-parameter verdicts",
        "",
        "| param | MAP value | Δχ²(max) | flat? | verdict |",
        "|---|---|---|---|---|",
    ]
    for p in profiles:
        delta_max = float(np.nanmax(np.array(p["delta_chi2"])))
        md.append(
            f"| {p['param']} | {map_real[p['param']]:.3e} | "
            f"{delta_max:.2f} | {p['flat_profile_at_95pct']} | "
            f"{summary['verdicts'][p['param']]} |"
        )
    md += [
        "",
        "## Reading the table",
        "",
        "- `Δχ²(max) < 3.84` → **flat** profile inside the grid → *practically",
        "  non-identifiable* at 95% confidence over the explored range.",
        "- `Δχ²(max) > 3.84` → at least one grid point lies outside the 95% CI",
        "  → parameter is *identifiable* at this resolution.",
        "- See `profile_likelihood.png` for the curves.",
    ]
    with (out / "summary.md").open("w") as f:
        f.write("\n".join(md))

    logger.info("done. summary at %s/summary.md", out)


if __name__ == "__main__":
    main()
