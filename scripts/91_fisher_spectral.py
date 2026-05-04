#!/usr/bin/env python3
"""
B1: Fisher spectral analysis — per-point Jacobian via PyBaMM finite differences.
Compute FIM, eigenvalue spectrum, CRLB for each chemistry at multiple operating points.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pybamm
import h5py
import json
import logging
from pathlib import Path
from scipy.linalg import svd, eigvalsh

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

PARAM_RANGES = {
    "D_n": (1e-15, 5e-13), "D_p": (5e-17, 5e-15), "t+": (0.2, 0.45),
    "SEI": (1e-9, 1e-6), "LAM_neg": (0.0, 0.3), "LAM_pos": (0.0, 0.3),
    "R_mult": (1.0, 5.0),
}

CHEMISTRIES = [
    ("LFP", "Prada2013"),
    ("NMC811", "Chen2020"),
    ("NCA", "Marquis2019"),
    ("LCO", "Ramadass2004"),
    ("LFP_v2", "Ai2020"),
]

N_TIME = 64


SEI_DEFAULTS = {
    "SEI partial molar volume [m3.mol-1]": 1e-5,
    "SEI growth activation energy [J.mol-1]": 3e4,
    "SEI reaction exchange current density [A.m-2]": 1e-6,
    "SEI resistivity [Ohm.m]": 2e5,
    "SEI open-circuit potential [V]": 0.1,
    "SEI electron conductivity [S.m-1]": 1e-10,
    "SEI lithium interstitial diffusivity [m2.s-1]": 1e-15,
    "Initial inner SEI thickness [m]": 1e-9,
    "Initial outer SEI thickness [m]": 1e-9,
    "EC diffusivity [m2.s-1]": 1e-12,
    "EC initial concentration [mol.m-3]": 1000,
    "SEI kinetic rate constant [m.s-1]": 1e-10,
    "Lithium interstitial reference concentration [mol.m-3]": 1.0,
    "Ratio of lithium moles to SEI moles": 1.0,
    "Inner SEI partial molar volume [m3.mol-1]": 1e-5,
    "Outer SEI partial molar volume [m3.mol-1]": 1e-5,
    "Inner SEI electron conductivity [S.m-1]": 1e-10,
    "Outer SEI electron conductivity [S.m-1]": 1e-10,
    "Bulk SEI reaction exchange current density [A.m-2]": 1e-6,
    "SEI growth transfer coefficient": 0.5,
}


def solve_once(chem_param_set, param_values):
    model = pybamm.lithium_ion.SPM({"SEI": "reaction limited"})
    params = pybamm.ParameterValues(chem_param_set)
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
    for k, v in SEI_DEFAULTS.items():
        if k not in params:
            params[k] = v
    solver = pybamm.CasadiSolver(mode="safe", rtol=1e-4, atol=1e-6)
    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
    sol = sim.solve([0, 3600])
    V = sol["Terminal voltage [V]"].entries
    t = sol["Time [s]"].entries
    t_interp = np.linspace(0, t[-1], N_TIME)
    return np.interp(t_interp, t, V)


def compute_jacobian(chem_param_set, param_values, eps=1e-5):
    V_base = solve_once(chem_param_set, param_values)
    J = np.zeros((N_TIME, 7))
    for i in range(7):
        p_plus = param_values.copy()
        delta = param_values[i] * eps
        if delta == 0:
            delta = eps
        p_plus[i] = param_values[i] + delta
        V_plus = solve_once(chem_param_set, p_plus)
        J[:, i] = (V_plus - V_base) / delta
    return J, V_base


def fisher_at_point(chem_param_set, param_values, sigma=0.005):
    J, V = compute_jacobian(chem_param_set, param_values)
    FIM = J.T @ J / (sigma ** 2)
    eigenvalues = np.linalg.eigvalsh(FIM)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-30)
    crlb = np.diag(np.linalg.pinv(FIM))
    crlb = np.maximum(crlb, 0)
    return {
        "J_shape": J.shape,
        "J_rank": int(np.linalg.matrix_rank(J, tol=1e-8)),
        "FIM_eigenvalues": eigenvalues.tolist(),
        "FIM_condition_number": float(eigenvalues[0] / max(eigenvalues[-1], 1e-30)),
        "crlb": {PARAM_NAMES[i]: float(crlb[i]) for i in range(7)},
        "crlb_normalized": None,
        "V_range": [float(V.min()), float(V.max())],
    }


def sample_operating_points(n_points=20, seed=42):
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(n_points):
        pv = np.zeros(7)
        pv[0] = 10 ** rng.uniform(np.log10(1e-15), np.log10(5e-13))
        pv[1] = 10 ** rng.uniform(np.log10(5e-17), np.log10(5e-15))
        pv[2] = rng.uniform(0.2, 0.45)
        pv[3] = 10 ** rng.uniform(np.log10(1e-9), np.log10(1e-6))
        pv[4] = rng.uniform(0.0, 0.3)
        pv[5] = rng.uniform(0.0, 0.3)
        pv[6] = rng.uniform(1.0, 5.0)
        pv[3] = max(pv[3], 1e-9)
        pv[4] = min(pv[4], 0.3)
        pv[5] = min(pv[5], 0.3)
        points.append(pv)
    return points


def main():
    output_dir = Path("outputs/fisher_spectral")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = sample_operating_points(n_points=20)
    all_results = {}

    for chem_name, param_set in CHEMISTRIES:
        logger.info("Computing Fisher for %s (%s)", chem_name, param_set)
        point_results = []
        for pi, pv in enumerate(points):
            try:
                fr = fisher_at_point(param_set, pv)
                fr["point_idx"] = pi
                point_results.append(fr)
                if (pi + 1) % 5 == 0:
                    logger.info("  %s: %d/%d done", chem_name, pi + 1, len(points))
            except Exception as e:
                logger.warning("  %s point %d failed: %s", chem_name, pi, str(e)[:80])
                continue

        if not point_results:
            continue

        avg_eigenvalues = np.mean([p["FIM_eigenvalues"] for p in point_results], axis=0)
        ranks = [p["J_rank"] for p in point_results]
        avg_crlb = {}
        for pname in PARAM_NAMES:
            vals = [p["crlb"][pname] for p in point_results if np.isfinite(p["crlb"][pname])]
            avg_crlb[pname] = float(np.mean(vals)) if vals else float("inf")

        all_results[chem_name] = {
            "param_set": param_set,
            "n_points": len(point_results),
            "ranks": ranks,
            "avg_rank": float(np.mean(ranks)),
            "avg_eigenvalues": avg_eigenvalues.tolist(),
            "avg_crlb": avg_crlb,
        }

        logger.info(
            "  %s: avg_rank=%.1f, eigenvalues=%s",
            chem_name, np.mean(ranks),
            ["{:.2e}".format(e) for e in avg_eigenvalues],
        )

    # Summary
    print("\n" + "=" * 90)
    print("FISHER INFORMATION SPECTRAL ANALYSIS (20 operating points per chemistry)")
    print("=" * 90)

    print("\n--- Average FIM Eigenvalue Spectrum ---")
    hdr = "{:10s}".format("Chem")
    for i in range(7):
        hdr += " {:>12s}".format(f"λ_{i+1}")
    hdr += " {:>6s} {:>6s}".format("Rank", "Cond")
    print(hdr)
    print("-" * len(hdr))

    for chem_name in ["LFP", "NMC811", "NCA", "LCO", "LFP_v2"]:
        if chem_name not in all_results:
            continue
        r = all_results[chem_name]
        ev = r["avg_eigenvalues"]
        row = "{:10s}".format(chem_name)
        for e in ev:
            if e > 0:
                row += " {:12.2e}".format(e)
            else:
                row += " {:>12s}".format("~0")
        cond = ev[0] / max(ev[-1], 1e-30)
        row += " {:6.1f} {:6.0f}".format(r["avg_rank"], cond)
        print(row)

    print("\n--- Cramér-Rao Lower Bound (σ=5mV, normalized) ---")
    hdr = "{:10s}".format("Chem")
    for pname in PARAM_NAMES:
        hdr += " {:>10s}".format(pname[:6])
    print(hdr)
    print("-" * len(hdr))

    for chem_name in ["LFP", "NMC811", "NCA", "LCO", "LFP_v2"]:
        if chem_name not in all_results:
            continue
        r = all_results[chem_name]
        row = "{:10s}".format(chem_name)
        for pname in PARAM_NAMES:
            v = r["avg_crlb"][pname]
            if np.isfinite(v) and v > 0:
                row += " {:10.2e}".format(np.sqrt(v))
            else:
                row += " {:>10s}".format("inf")
        print(row)

    with open(output_dir / "results.json", "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
