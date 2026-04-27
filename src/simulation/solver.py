import pybamm
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from .models import MetalBatteryDFN
from .parameters import parse_sweep_params, load_config

logger = logging.getLogger(__name__)

FULL_FIELD_VARIABLES = {
    "c_e": "Electrolyte concentration [mol.m-3]",
    "phi_e": "Electrolyte potential [V]",
    "phi_s_neg": "Negative electrode potential [V]",
    "phi_s_pos": "Positive electrode potential [V]",
    "c_s_pos": "Positive particle concentration [mol.m-3]",
    "j_pos": "Positive electrode interfacial current density [A.m-2]",
    "j_neg": "Negative electrode interfacial current density [A.m-2]",
    "L_sei": "Negative SEI thickness [m]",
    "V": "Voltage [V]",
    "I": "Current [A]",
}


class PybammSolver:
    """
    Wrapper around PyBaMM solver for metal battery simulations.

    Handles experiment definition, solver configuration, and solution extraction.
    """

    def __init__(
        self,
        model: MetalBatteryDFN,
        solver: Optional[pybamm.BaseSolver] = None,
    ):
        self.battery_model = model
        self.solver = solver or pybamm.CasadiSolver(
            mode="safe",
            rtol=1e-6,
            atol=1e-8,
            max_step_decrease_count=5,
        )

    def solve(
        self,
        params: pybamm.ParameterValues,
        c_rate: float = 1.0,
        temperature: float = 25.0,
        t_end: float = 3600.0,
        n_points: int = 200,
    ) -> dict:
        """
        Run a single charge/discharge simulation.

        Args:
            params: PyBaMM parameter values.
            c_rate: C-rate for the simulation.
            temperature: Temperature in Celsius.
            t_end: End time in seconds.
            n_points: Number of time points for output.

        Returns:
            dict with keys: "time", "voltage", "current", "capacity",
                            "c_e", "phi_s", "phi_e", "c_s_surf", "params_used"
        """
        model = self.battery_model.model

        capacity = params["Nominal cell capacity [A.h]"]
        current_A = c_rate * capacity

        params = params.copy()
        params["Current function [A]"] = current_A

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            solver=self.solver,
        )

        try:
            sol = sim.solve(
                t_eval=np.linspace(0, t_end, n_points),
            )
        except pybamm.SolverError as e:
            logger.warning(f"Solver failed for C-rate={c_rate}, T={temperature}: {e}")
            return None

        t = sol["Time [s]"].data
        v = sol["Voltage [V]"].data
        i = sol["Current [A]"].data

        result = {
            "time": t,
            "voltage": v,
            "current": i,
            "capacity": capacity,
            "c_rate": c_rate,
            "temperature": temperature,
            "params_used": {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in params.items()
                if isinstance(v, (int, float, np.floating, np.integer))
            },
        }

        try:
            result["c_e"] = sol["Electrolyte concentration [mol.m-3]"].data
        except KeyError:
            pass
        try:
            result["phi_s"] = sol["Negative electrode potential [V]"].data
        except KeyError:
            pass
        try:
            result["c_s_surf"] = sol[
                "Positive electrode surface concentration [mol.m-3]"
            ].data
        except KeyError:
            pass

        return result

    def solve_full_field(
        self,
        params: pybamm.ParameterValues,
        c_rate: float = 1.0,
        temperature: float = 25.0,
        t_end: float = 3600.0,
        n_time: int = 100,
    ) -> Optional[dict]:
        """
        Run simulation and extract all spatial fields for FNO training.

        Returns:
            dict with 2D/3D field arrays shaped (Nt, ...) or (Nx, Nt).
            Fields: c_e, phi_e, phi_s_neg, phi_s_pos, c_s_pos, j_pos, j_neg, L_sei, V, I
            Plus metadata: time, c_rate, params_used, spatial_info
        """
        model = self.battery_model.model
        capacity = params["Nominal cell capacity [A.h]"]
        current_A = c_rate * capacity
        params = params.copy()
        params["Current function [A]"] = current_A

        sim = pybamm.Simulation(
            model, parameter_values=params, solver=self.solver,
        )

        try:
            sol = sim.solve(t_eval=np.linspace(0, t_end, n_time))
        except pybamm.SolverError as e:
            logger.warning(f"Full-field solve failed: C={c_rate}, T={temperature}: {e}")
            return None

        result = {
            "c_rate": c_rate,
            "temperature": temperature,
            "capacity_Ah": float(capacity),
        }

        for key, pybamm_name in FULL_FIELD_VARIABLES.items():
            try:
                result[key] = sol[pybamm_name].data
            except (KeyError, TypeError):
                pass

        if "c_e" in result:
            result["spatial_info"] = {
                "c_e_shape": result["c_e"].shape,
                "phi_s_neg_shape": result.get("phi_s_neg", np.empty(0)).shape,
                "phi_s_pos_shape": result.get("phi_s_pos", np.empty(0)).shape,
                "c_s_pos_shape": result.get("c_s_pos", np.empty(0)).shape,
            }

        return result

    def solve_rest(
        self,
        params: pybamm.ParameterValues,
        initial_soc: float = 1.0,
        rest_time: float = 3600.0,
        temperature: float = 25.0,
    ) -> dict:
        """Run a voltage relaxation simulation after reaching a given SOC."""
        model = self.battery_model.model
        params = params.copy()
        params["Initial State of Charge"] = initial_soc

        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            solver=self.solver,
        )

        experiment = pybamm.Experiment(
            [f"Rest for {rest_time:.1f} seconds"],
            temperature=f"{temperature} oC",
        )

        try:
            sol = sim.solve(experiment)
        except pybamm.SolverError as e:
            logger.warning(f"Rest simulation failed: {e}")
            return None

        return {
            "time": sol["Time [s]"].data,
            "voltage": sol["Voltage [V]"].data,
            "current": np.zeros_like(sol["Time [s]"].data),
            "initial_soc": initial_soc,
            "temperature": temperature,
        }
