#!/usr/bin/env python3
"""
A2b: PyBaMM parameter fitting baseline (scipy.optimize MLE).
For 50 synthetic test samples, fit 7 parameters via MLE.
Record per-parameter MAE, confidence interval width, convergence status.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pybamm
import json
import logging
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
PYBAMM_KEYS = [
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Cation transference number",
    "Initial SEI thickness [m]",
    "Negative electrode LAM fraction",
    "Positive electrode LAM fraction",
    "Resistance multiplier",
]
IDENT_IDX = [0, 3, 4]

PARAM_RANGES = np.array([
    [1e-15, 5e-13], [5e-17, 5e-15], [0.2, 0.45],
    [1e-9, 1e-6], [0.0, 0.3], [0.0, 0.3], [1.0, 5.0],
])

N_TIME = 64
CHEM_PARAM_SET = "Prada2013"


def solve_pybamm(param_values):
    model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
    params = pybamm.ParameterValues(CHEM_PARAM_SET)
    for i, key in enumerate(PYBAMM_KEYS):
        if key == "Resistance multiplier":
            continue
        try:
            params[key] = param_values[i]
        except (KeyError, AttributeError):
            pass
    R_mult = param_values[6]
    for rp_key in [
        "Negative electrode exchange-current density [A.m-2]",
        "Positive electrode exchange-current density [A.m-2]",
        "Electrolyte conductivity [S.m-1]",
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
    try:
        sol = sim.solve([0, 3600])
        V = sol["Terminal voltage [V]"].entries
        t = sol["Time [s]"].entries
        t_interp = np.linspace(0, t[-1], N_TIME)
        return np.interp(t_interp, t, V)
    except Exception:
        return None


def objective(log_params, V_target):
    param_values = np.zeros(7)
    param_values[0] = 10 ** log_params[0]
    param_values[1] = 10 ** log_params[1]
    param_values[2] = log_params[2]
    param_values[3] = 10 ** log_params[3]
    param_values[4] = log_params[4]
    param_values[5] = log_params[5]
    param_values[6] = log_params[6]
    V_pred = solve_pybamm(param_values)
    if V_pred is None or len(V_pred) != N_TIME:
        return 1e6
    return np.mean((V_pred - V_target) ** 2)


def param_to_log(p):
    lp = np.zeros(7)
    lp[0] = np.log10(max(p[0], 1e-30))
    lp[1] = np.log10(max(p[1], 1e-30))
    lp[2] = p[2]
    lp[3] = np.log10(max(p[3], 1e-30))
    lp[4] = p[4]
    lp[5] = p[5]
    lp[6] = p[6]
    return lp


def log_to_param(lp):
    p = np.zeros(7)
    p[0] = 10 ** lp[0]
    p[1] = 10 ** lp[1]
    p[2] = lp[2]
    p[3] = 10 ** lp[3]
    p[4] = lp[4]
    p[5] = lp[5]
    p[6] = lp[6]
    return p


def fit_single(V_target, true_params, idx):
    lp_true = param_to_log(true_params)
    lp_bounds = [
        (np.log10(1e-15), np.log10(5e-13)),
        (np.log10(5e-17), np.log10(5e-15)),
        (0.2, 0.45),
        (np.log10(1e-9), np.log10(1e-6)),
        (0.0, 0.3),
        (0.0, 0.3),
        (1.0, 5.0),
    ]

    try:
        result = differential_evolution(
            objective, lp_bounds, args=(V_target,),
            seed=42, maxiter=200, tol=1e-8, polish=True,
            popsize=15, mutation=(0.5, 1.0), recombination=0.7,
        )
        fitted_lp = result.x
        fitted_params = log_to_param(fitted_lp)
        converged = result.success
        final_loss = result.fun
    except Exception as e:
        logger.warning("  Fit %d failed: %s", idx, str(e)[:60])
        return None

    # Compute relative error per parameter
    rel_errors = {}
    for pi, pname in enumerate(PARAM_NAMES):
        if true_params[pi] > 0:
            rel_errors[pname] = float(abs(fitted_params[pi] - true_params[pi]) / true_params[pi])
        else:
            rel_errors[pname] = float(abs(fitted_params[pi] - true_params[pi]))

    return {
        "idx": int(idx),
        "converged": bool(converged),
        "final_loss": float(final_loss),
        "true_params": [float(x) for x in true_params],
        "fitted_params": [float(x) for x in fitted_params],
        "rel_errors": rel_errors,
    }


def compute_confidence_widths(V_target, best_lp, n_bootstrap=20):
    lp_bounds = [
        (np.log10(1e-15), np.log10(5e-13)),
        (np.log10(5e-17), np.log10(5e-15)),
        (0.2, 0.45),
        (np.log10(1e-9), np.log10(1e-6)),
        (0.0, 0.3),
        (0.0, 0.3),
        (1.0, 5.0),
    ]
    rng = np.random.RandomState(42)
    all_fitted = []
    for _ in range(n_bootstrap):
        V_noisy = V_target + rng.normal(0, 0.005, V_target.shape)
        try:
            result = minimize(
                objective, best_lp, args=(V_noisy,),
                method="L-BFGS-B", bounds=lp_bounds,
                options={"maxiter": 100},
            )
            all_fitted.append(result.x)
        except Exception:
            continue

    if len(all_fitted) < 5:
        return {pn: float("inf") for pn in PARAM_NAMES}

    all_fitted = np.array(all_fitted)
    widths = all_fitted.std(axis=0)
    return {PARAM_NAMES[i]: float(widths[i]) for i in range(7)}


def main():
    output_dir = Path("outputs/baselines/pybamm_fit")
    output_dir.mkdir(parents=True, exist_ok=True)

    import h5py
    h5_path = "data/fullfield/fullfield_lfp_degradation.h5"
    with h5py.File(h5_path, "r") as f:
        all_V = f["V"][:].astype(np.float32)
        all_params = f["params"][:].astype(np.float32)

    rng = np.random.RandomState(42)
    n_test = 30
    test_idx = rng.choice(len(all_V), n_test, replace=False)

    results = []
    for i, idx in enumerate(test_idx):
        V_target = all_V[idx]
        true_params = all_params[idx]
        # Add 5mV noise
        V_noisy = V_target + rng.normal(0, 0.005, V_target.shape)
        logger.info("Fitting sample %d/%d (idx=%d)", i + 1, n_test, idx)
        result = fit_single(V_noisy, true_params, idx)
        if result is not None:
            # Bootstrap CI
            best_lp = param_to_log(np.array(result["fitted_params"]))
            ci_widths = compute_confidence_widths(V_noisy, best_lp, n_bootstrap=15)
            result["ci_widths"] = ci_widths
            results.append(result)

    logger.info("Completed %d/%d fits", len(results), n_test)

    # Aggregate
    print("\n" + "=" * 80)
    print("PyBaMM PARAMETER FITTING BASELINE (differential evolution MLE)")
    print("=" * 80)

    print("\n--- Per-Parameter Relative Error ---")
    hdr = "{:10s} {:>10s} {:>10s} {:>12s} {:>12s} {:>8s}".format(
        "Param", "RelErr%", "CI_width", "Converge%", "Fisher_pred", "Status"
    )
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for pi, pname in enumerate(PARAM_NAMES):
        rel_errs = [r["rel_errors"][pname] * 100 for r in results]
        ci_ws = [r["ci_widths"].get(pname, float("inf")) for r in results]
        ci_vals = [w for w in ci_ws if np.isfinite(w)]
        converge_pct = sum(1 for r in results if r["converged"]) / max(len(results), 1) * 100
        avg_re = np.mean(rel_errs) if rel_errs else 0
        avg_ci = np.mean(ci_vals) if ci_vals else float("inf")
        st = "ID" if pi in IDENT_IDX else "UN"
        fisher_pred = "Low err/narrow CI" if pi in IDENT_IDX else "High err/wide CI"
        print("{:10s} {:10.1f} {:10.4f} {:12.1f} {:>12s} {:>8s}".format(
            pname, avg_re, avg_ci, converge_pct, fisher_pred, st
        ))
        summary[pname] = {
            "avg_rel_error_pct": float(avg_re),
            "avg_ci_width": float(avg_ci) if np.isfinite(avg_ci) else None,
            "converge_pct": float(converge_pct),
            "status": st,
        }

    with open(output_dir / "results.json", "w") as fp:
        json.dump({"per_sample": results, "summary": summary}, fp, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
