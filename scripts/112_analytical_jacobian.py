#!/usr/bin/env python3
"""Analytical Jacobian via numerical differentiation (finite differences).

Computes ∂V/∂θ by solving the forward model at perturbed parameter values.
No special solver sensitivity support required.

Output:
  outputs/analytical_jacobian/{spm,dfn}_jacobian_results.json
  outputs/analytical_jacobian/fig_jacobian_structure.png
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

import pybamm

BASE = Path("/AI4S/Users/howardwang/h204/autobattery")
OUT = BASE / "outputs" / "analytical_jacobian"
OUT.mkdir(parents=True, exist_ok=True)

PARAM_DEFS = [
    ("D_n", "Negative particle diffusivity [m2.s-1]", 3.9e-14, 1.0),
    ("D_p", "Positive particle diffusivity [m2.s-1]", 1.0e-13, 1.0),
    ("t+", "Cation transference number", 0.38, 0.01),
    ("SEI", "Initial SEI thickness [m]", 2.5e-9, 1.0),
    ("LAM_neg", "Negative electrode LAM fraction", 0.0, 0.01),
    ("LAM_pos", "Positive electrode LAM fraction", 0.0, 0.01),
    ("R_mult", "Resistance multiplier", 1.0, 0.01),
]

PARAM_NAMES = [p[0] for p in PARAM_DEFS]


def solve_model(model, param_values, chemistry="Prada2013"):
    param = pybamm.ParameterValues(chemistry)

    param_overrides = {
        "Current function [A]": param["Nominal cell capacity [A.h]"],
    }
    for k, v in param_values.items():
        param_overrides[k] = v
    for k, v in param_overrides.items():
        if k in param:
            param[k] = v

    experiment = pybamm.Experiment([
        "Discharge at 1C until 2.5V",
    ])

    solver = pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-8)
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment, solver=solver)
    sol = sim.solve()

    V = sol["Terminal voltage [V]"]
    t = sol.t
    return t, V(t), sol


def compute_jacobian_fd(model_name="SPM", n_timepoints=200, eps=1e-4):
    if model_name == "SPM":
        model = pybamm.lithium_ion.SPM()
    else:
        model = pybamm.lithium_ion.DFN()

    logger.info("Solving %s at base parameters...", model_name)

    base_params = {p[1]: p[2] for p in PARAM_DEFS}
    t_base, V_base, sol_base = solve_model(model, base_params)

    t_eval = np.linspace(sol_base.t[0], sol_base.t[-1], n_timepoints)
    V_base_interp = np.interp(t_eval, t_base, V_base)

    sens_data = {}
    for short_name, long_name, base_val, perturb_scale in PARAM_DEFS:
        if base_val == 0:
            delta = perturb_scale * eps
        else:
            delta = base_val * eps

        perturbed = dict(base_params)
        perturbed[long_name] = base_val + delta

        logger.info("  Perturbing %s: %s → %s (δ=%s)", short_name, base_val, base_val + delta, delta)

        try:
            t_pert, V_pert, _ = solve_model(model, perturbed)
            V_pert_interp = np.interp(t_eval, t_pert, V_pert)
            dV_dp = (V_pert_interp - V_base_interp) / delta
        except Exception as e:
            logger.warning("  Failed for %s: %s", short_name, e)
            dV_dp = np.zeros(n_timepoints)

        sens_data[short_name] = dV_dp.tolist()
        norm = np.linalg.norm(dV_dp)
        maxval = np.max(np.abs(dV_dp))
        logger.info("  %s: ||∂V/∂θ||=%.4e, max|∂V/∂θ|=%.4e", short_name, norm, maxval)

    J = np.column_stack([sens_data[p] for p in PARAM_NAMES])

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_norm = S / S[0] if S[0] > 0 else S

    corr = np.corrcoef(J.T)
    np.fill_diagonal(corr, 1.0)

    eta_ranks = {}
    for k in range(1, 7):
        eta = 10**(-k)
        eta_ranks[f"1e-{k}"] = int(np.sum(S_norm > eta))

    results = {
        "model": model_name,
        "n_timepoints": n_timepoints,
        "eps": eps,
        "sensitivities": sens_data,
        "singular_values": S.tolist(),
        "singular_values_normalized": S_norm.tolist(),
        "jacobian_rank": int(np.sum(S > 1e-10 * S[0])),
        "eta_rank": eta_ranks,
        "correlation_matrix": {
            PARAM_NAMES[i]: {
                PARAM_NAMES[j]: float(corr[i, j]) for j in range(len(PARAM_NAMES))
            }
            for i in range(len(PARAM_NAMES))
        },
        "jacobian_norms": {
            p: float(np.linalg.norm(np.array(sens_data[p]))) for p in PARAM_NAMES
        },
    }

    return results


def plot_jacobian_structure(spm_results, dfn_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, (label, results) in enumerate([("SPM", spm_results), ("DFN", dfn_results)]):
        if results is None:
            continue

        sens = results["sensitivities"]
        params = PARAM_NAMES

        ax = axes[row, 0]
        J = np.column_stack([sens[p] for p in params])
        J_absmax = np.max(np.abs(J)) + 1e-30
        J_norm = J / J_absmax
        im = ax.imshow(J_norm.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_ylabel("Parameter")
        ax.set_xlabel("Time index")
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_title(f"{label}: ∂V/∂θ (normalized)")
        plt.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[row, 1]
        corr = np.array([
            [results["correlation_matrix"][p1][p2] for p2 in params]
            for p1 in params
        ])
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(params)):
            for j in range(len(params)):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right")
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_title(f"{label}: Column Correlation")
        plt.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[row, 2]
        S_norm = results["singular_values_normalized"]
        ax.semilogy(range(1, len(S_norm) + 1), S_norm, "ko-", ms=8, lw=2)
        for eta_exp in [2, 3, 6]:
            ax.axhline(10**(-eta_exp), color="grey", ls=":", lw=0.8)
            ax.text(len(S_norm) + 0.3, 10**(-eta_exp), f"η=1e-{eta_exp}", fontsize=7, color="grey")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("σᵢ/σ₁")
        ax.set_title(f"{label}: SVD Spectrum (rank={results['jacobian_rank']})")
        ax.grid(alpha=0.3)

    fig.suptitle("Analytical Jacobian Structure: SPM vs DFN", fontsize=14)
    fig.savefig(OUT / "fig_jacobian_structure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved")


def main():
    spm_results = compute_jacobian_fd("SPM", n_timepoints=200, eps=1e-4)
    with open(OUT / "spm_jacobian_results.json", "w") as f:
        json.dump(spm_results, f, indent=2)
    logger.info("SPM: rank=%s, η-rank=%s", spm_results["jacobian_rank"], spm_results["eta_rank"])
    logger.info("SPM norms: %s", {k: f"{v:.4e}" for k, v in spm_results["jacobian_norms"].items()})

    dfn_results = compute_jacobian_fd("DFN", n_timepoints=200, eps=1e-4)
    with open(OUT / "dfn_jacobian_results.json", "w") as f:
        json.dump(dfn_results, f, indent=2)
    logger.info("DFN: rank=%s, η-rank=%s", dfn_results["jacobian_rank"], dfn_results["eta_rank"])
    logger.info("DFN norms: %s", {k: f"{v:.4e}" for k, v in dfn_results["jacobian_norms"].items()})

    plot_jacobian_structure(spm_results, dfn_results)
    logger.info("Done. Results in %s", OUT)


if __name__ == "__main__":
    main()
