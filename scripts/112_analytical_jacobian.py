#!/usr/bin/env python3
"""Analytical Jacobian via log-space finite differences.

Key improvements over v1:
1. Log-space perturbation: all sensitivities in comparable units (V per decade)
2. Both pristine and degraded states
3. SPM and DFN comparison

Output:
  outputs/analytical_jacobian/{spm,dfn}_jacobian_*.json
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
    ("D_n", "Negative particle diffusivity [m2.s-1]", 3.9e-14),
    ("D_p", "Positive particle diffusivity [m2.s-1]", 1.0e-13),
    ("t+", "Cation transference number", 0.38),
    ("SEI", "Initial SEI thickness [m]", 2.5e-9),
    ("LAM_neg", "Negative electrode LAM fraction", 0.0),
    ("LAM_pos", "Positive electrode LAM fraction", 0.0),
    ("R_mult", "Resistance multiplier", 1.0),
]

PARAM_NAMES = [p[0] for p in PARAM_DEFS]

STATES = {
    "pristine": {
        "D_n": 3.9e-14, "D_p": 1.0e-13, "t+": 0.38,
        "SEI": 2.5e-9, "LAM_neg": 0.0, "LAM_pos": 0.0, "R_mult": 1.0,
    },
    "mild_degradation": {
        "D_n": 3.9e-14, "D_p": 1.0e-13, "t+": 0.38,
        "SEI": 1e-7, "LAM_neg": 0.05, "LAM_pos": 0.05, "R_mult": 1.5,
    },
    "heavy_degradation": {
        "D_n": 1.95e-14, "D_p": 5e-14, "t+": 0.36,
        "SEI": 5e-7, "LAM_neg": 0.1, "LAM_pos": 0.1, "R_mult": 2.0,
    },
}


def solve_model(model, param_overrides, chemistry="Prada2013"):
    param = pybamm.ParameterValues(chemistry)
    for k, v in param_overrides.items():
        if k in param:
            param[k] = v
    param["Current function [A]"] = param["Nominal cell capacity [A.h]"]
    experiment = pybamm.Experiment(["Discharge at 1C until 2.5V"])
    solver = pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-8)
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment, solver=solver)
    sol = sim.solve()
    return sol


def compute_log_jacobian(model, state_name, state_params, n_timepoints=200, log_eps=1e-4):
    long_name_map = {p[0]: p[1] for p in PARAM_DEFS}

    base_pv = {long_name_map[k]: v for k, v in state_params.items()}

    logger.info("Solving %s at %s state...", model.name, state_name)
    sol_base = solve_model(model, base_pv)
    t_eval = np.linspace(sol_base.t[0], sol_base.t[-1], n_timepoints)
    V_base = np.interp(t_eval, sol_base.t, sol_base["Terminal voltage [V]"](sol_base.t))

    sens_data = {}
    for pn in PARAM_NAMES:
        lng = long_name_map[pn]
        base_val = state_params[pn]

        if base_val <= 0:
            delta = 1e-6
            perturbed_val = delta
            scale = "additive"
        else:
            perturbed_val = base_val * (1 + log_eps)
            delta = base_val * log_eps
            scale = "log"

        pv = dict(base_pv)
        pv[lng] = perturbed_val

        try:
            sol_pert = solve_model(model, pv)
            V_pert = np.interp(t_eval, sol_pert.t, sol_pert["Terminal voltage [V]"](sol_pert.t))
            dV = (V_pert - V_base) / delta if scale == "additive" else (V_pert - V_base) / delta
            dV_log = dV * base_val if base_val > 0 else dV
        except Exception as e:
            logger.warning("  Failed for %s: %s", pn, e)
            dV_log = np.zeros(n_timepoints)

        sens_data[pn] = dV_log.tolist()
        norm = float(np.linalg.norm(dV_log))
        logger.info("  %s: ||∂V/∂log(θ)||=%.4e V/decade", pn, norm)

    J = np.column_stack([sens_data[p] for p in PARAM_NAMES])
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_norm = S / S[0] if S[0] > 0 else S

    J_clean = J.copy()
    J_clean[np.isnan(J_clean)] = 0
    corr = np.corrcoef(J_clean.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)

    eta_ranks = {}
    for k in range(1, 7):
        eta_ranks[f"1e-{k}"] = int(np.sum(S_norm > 10**(-k)))

    return {
        "model": model.name,
        "state": state_name,
        "state_params": {k: str(v) for k, v in state_params.items()},
        "n_timepoints": n_timepoints,
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
        "log_jacobian_norms": {
            p: float(np.linalg.norm(np.array(sens_data[p]))) for p in PARAM_NAMES
        },
    }


def plot_results(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_results = len(all_results)
    fig, axes = plt.subplots(n_results, 3, figsize=(18, 5 * n_results))
    if n_results == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(all_results):
        label = f"{res['model']} ({res['state']})"
        sens = res["sensitivities"]
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
        ax.set_title(f"{label}: ∂V/∂log(θ)")
        plt.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[row, 1]
        corr = np.array([
            [res["correlation_matrix"][p1][p2] for p2 in params]
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
        ax.set_title(f"{label}: Correlation")
        plt.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[row, 2]
        S_norm = res["singular_values_normalized"]
        ax.semilogy(range(1, len(S_norm) + 1), S_norm, "ko-", ms=8, lw=2)
        for eta_exp in [2, 3, 6]:
            ax.axhline(10**(-eta_exp), color="grey", ls=":", lw=0.8)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("σᵢ/σ₁")
        ax.set_title(f"{label}: rank={res['jacobian_rank']}, η=1e-3 → {res['eta_rank'].get('1e-3', '?')}")
        ax.grid(alpha=0.3)

    fig.suptitle("Log-Space Jacobian Structure Across Models and Degradation States", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig_jacobian_structure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved")


def main():
    all_results = []

    for model_cls, model_label in [(pybamm.lithium_ion.SPM, "SPM"), (pybamm.lithium_ion.DFN, "DFN")]:
        model = model_cls()
        for state_name, state_params in STATES.items():
            res = compute_log_jacobian(model, state_name, state_params)
            fname = f"{model_label.lower()}_jacobian_{state_name}.json"
            with open(OUT / fname, "w") as f:
                json.dump(res, f, indent=2)
            logger.info("%s %s: rank=%s, η-rank=%s, norms=%s",
                        model_label, state_name, res["jacobian_rank"],
                        res["eta_rank"],
                        {k: f"{v:.4e}" for k, v in res["log_jacobian_norms"].items()})
            all_results.append(res)

    plot_results(all_results)
    logger.info("Done. Results in %s", OUT)


if __name__ == "__main__":
    main()
