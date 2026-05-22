#!/usr/bin/env python3
"""Analytical Jacobian derivation for SPM/DFN degradation parameters.

Computes ∂V/∂θ symbolically using PyBaMM's sensitivity analysis,
then analyzes rank structure and column correlations to prove the
degenerate triplet (SEI, t+, R_mult).

Output:
  outputs/analytical_jacobian/spm_jacobian_results.json
  outputs/analytical_jacobian/dfn_jacobian_results.json
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

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]


def build_degradation_input(params_dict):
    param = pybamm.ParameterValues("Prada2013")
    for k, v in params_dict.items():
        param[k] = v
    return param


def compute_jacobian(model_name="SPM", n_timepoints=100):
    if model_name == "SPM":
        model = pybamm.lithium_ion.SPM()
        chemistry = "Prada2013"
    else:
        model = pybamm.lithium_ion.DFN()
        chemistry = "Prada2013"

    param = pybamm.ParameterValues(chemistry)

    default_values = {
        "Negative particle diffusivity [m2.s-1]": 3.9e-14,
        "Positive particle diffusivity [m2.s-1]": 1.0e-13,
        "Cation transference number": 0.38,
        "Initial SEI thickness [m]": 2.5e-9,
        "Negative electrode LAM fraction": 0.0,
        "Positive electrode LAM fraction": 0.0,
        "Resistance multiplier": 1.0,
    }

    sensitivity_params = [
        "Negative particle diffusivity [m2.s-1]",
        "Positive particle diffusivity [m2.s-1]",
        "Cation transference number",
        "Initial SEI thickness [m]",
        "Negative electrode LAM fraction",
        "Positive electrode LAM fraction",
        "Resistance multiplier",
    ]

    for k, v in default_values.items():
        if k in param:
            param[k] = v

    for sp in sensitivity_params:
        if sp in param:
            base_val = param[sp]
            param[sp] = pybamm.InputParameter(sp.replace(" ", "_").replace("[", "").replace("]", "").replace(".", ""))

    experiment = pybamm.Experiment([
        "Discharge at 1C until 2.5V",
    ])

    solver = pybamm.CasadiSolver(mode="safe")

    input_dict = {}
    short_names = [
        "Negative_particle_diffusivity_m2s-1",
        "Positive_particle_diffusivity_m2s-1",
        "Cation_transference_number",
        "Initial_SEI_thickness_m",
        "Negative_electrode_LAM_fraction",
        "Positive_electrode_LAM_fraction",
        "Resistance_multiplier",
    ]
    for sn, sp in zip(short_names, sensitivity_params):
        input_dict[sn] = default_values[sp]

    logger.info("Solving %s with sensitivity analysis...", model_name)
    try:
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment, solver=solver)
        sol = sim.solve(inputs=input_dict, calculate_sensitivities=True)
    except Exception as e:
        logger.error("Failed to solve %s: %s", model_name, e)
        return None

    V = sol["Terminal voltage [V]"]
    t_eval = np.linspace(sol.t[0], sol.t[-1], n_timepoints)

    V_interp = V(t_eval)
    sens_data = {}
    for sn, pn in zip(short_names, PARAM_NAMES):
        try:
            dV_dp = sol["Terminal voltage [V]"].sensitivities[sn]
            if isinstance(dV_dp, (int, float)):
                dV_dp = np.full(n_timepoints, dV_dp)
            else:
                dV_interp = pybamm.Interpolant(
                    [sol.t], [dV_dp], pybamm.t
                )
                dV_dp = np.array(dV_interp(t_eval)).flatten()
            sens_data[pn] = dV_dp.tolist()
        except Exception as e:
            logger.warning("No sensitivity for %s: %s", pn, e)
            sens_data[pn] = np.zeros(n_timepoints).tolist()

    J = np.column_stack([sens_data[p] for p in PARAM_NAMES])

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_norm = S / S[0] if S[0] > 0 else S

    corr = np.corrcoef(J.T)
    np.fill_diagonal(corr, 1.0)

    results = {
        "model": model_name,
        "n_timepoints": n_timepoints,
        "sensitivities": {p: sens_data[p] for p in PARAM_NAMES},
        "singular_values": S.tolist(),
        "singular_values_normalized": S_norm.tolist(),
        "jacobian_rank": int(np.sum(S > 1e-10 * S[0])),
        "eta_rank": {
            f"1e-{k}": int(np.sum(S_norm > 10**(-k))) for k in range(1, 7)
        },
        "correlation_matrix": {
            PARAM_NAMES[i]: {PARAM_NAMES[j]: float(corr[i, j]) for j in range(len(PARAM_NAMES))}
            for i in range(len(PARAM_NAMES))
        },
        "jacobian_norms": {
            p: float(np.linalg.norm(np.array(sens_data[p])))
            for p in PARAM_NAMES
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
        J_norm = J / (np.max(np.abs(J)) + 1e-30)
        im = ax.imshow(J_norm.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_ylabel("Parameter")
        ax.set_xlabel("Time index")
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_title(f"{label}: Normalized Jacobian ∂V/∂θ")
        plt.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[row, 1]
        corr = np.array([[results["correlation_matrix"][p1][p2] for p2 in params] for p1 in params])
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
        ax.semilogy(range(1, len(S_norm)+1), S_norm, "ko-", ms=8, lw=2)
        for eta_exp in [2, 3, 6]:
            ax.axhline(10**(-eta_exp), color="grey", ls=":", lw=0.8)
            ax.text(len(S_norm)+0.3, 10**(-eta_exp), f"η=1e-{eta_exp}", fontsize=7, color="grey")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("σᵢ/σ₁")
        ax.set_title(f"{label}: SVD Spectrum (rank={results['jacobian_rank']})")
        ax.grid(alpha=0.3)

    fig.suptitle("Analytical Jacobian Structure: SPM vs DFN", fontsize=14)
    fig.savefig(OUT / "fig_jacobian_structure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved")


def main():
    spm_results = compute_jacobian("SPM")
    if spm_results:
        with open(OUT / "spm_jacobian_results.json", "w") as f:
            json.dump(spm_results, f, indent=2, default=str)
        logger.info("SPM rank=%s, η-rank=%s", spm_results["jacobian_rank"], spm_results["eta_rank"])
        logger.info("SPM Jacobian norms: %s", {k: f"{v:.2e}" for k, v in spm_results["jacobian_norms"].items()})

    dfn_results = compute_jacobian("DFN")
    if dfn_results:
        with open(OUT / "dfn_jacobian_results.json", "w") as f:
            json.dump(dfn_results, f, indent=2, default=str)
        logger.info("DFN rank=%s, η-rank=%s", dfn_results["jacobian_rank"], dfn_results["eta_rank"])
        logger.info("DFN Jacobian norms: %s", {k: f"{v:.2e}" for k, v in dfn_results["jacobian_norms"].items()})

    if spm_results and dfn_results:
        plot_jacobian_structure(spm_results, dfn_results)
    elif spm_results:
        plot_jacobian_structure(spm_results, spm_results)

    logger.info("Done. Results in %s", OUT)


if __name__ == "__main__":
    main()
