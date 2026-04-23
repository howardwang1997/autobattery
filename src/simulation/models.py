import pybamm
import numpy as np
from typing import Optional


class MetalBatteryDFN:
    """
    Pseudo-2-Dimensional (Doyle-Fuller-Newman) model for metal anode batteries.

    Supports both Lithium Metal Battery (LMB) and Sodium Metal Battery (NMB)
    chemistries. The metal anode replaces the conventional intercalation anode,
    removing solid-phase diffusion in the negative electrode and adding
    metal plating/stripping kinetics.

    The model structure:
    - Negative electrode: metal plating/stripping (no particle diffusion)
    - Separator: electrolyte transport only
    - Positive electrode: intercalation with solid-phase diffusion
    - SEI sub-model on metal anode surface
    """

    SUPPORTED_CHEMISTRIES = ("lmb", "nmb")

    def __init__(self, chemistry: str = "lmb", options: Optional[dict] = None):
        if chemistry not in self.SUPPORTED_CHEMISTRIES:
            raise ValueError(
                f"Unsupported chemistry '{chemistry}'. "
                f"Choose from {self.SUPPORTED_CHEMISTRIES}"
            )
        self.chemistry = chemistry

        default_options = {
            "anode": "metal",
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
        }
        if options is not None:
            default_options.update(options)

        if chemistry == "lmb":
            default_options["working ion"] = "Li"
        else:
            default_options["working ion"] = "Na"

        self._options = default_options
        self.model = self._build_model()
        self._default_params = self._load_default_params()

    def _build_model(self) -> pybamm.BaseModel:
        """Build the DFN model with metal anode configuration."""
        options = self._options
        pybamm_options = {
            "SEI": options.get("SEI", "ec reaction limited"),
            "SEI porosity change": options.get("SEI porosity change", "true"),
        }

        if self.chemistry == "nmb":
            model = pybamm.sodium_ion.BasicDFN()
        else:
            model = pybamm.lithium_ion.DFN(pybamm_options)

        return model

    def _load_default_params(self) -> pybamm.ParameterValues:
        """Load default parameter values for the chosen chemistry."""
        if self.chemistry == "lmb":
            try:
                params = pybamm.ParameterValues("Chen2020")
            except KeyError:
                params = pybamm.ParameterValues("Marquis2019")
        else:
            params = pybamm.ParameterValues("Chayambuka2022")

        return params

    def get_default_parameters(self) -> pybamm.ParameterValues:
        """Return a copy of the default parameter values."""
        return self._default_params.copy()

    def build_parameter_set(self, overrides: dict) -> pybamm.ParameterValues:
        """
        Build a parameter set with user-provided overrides.

        Args:
            overrides: dict mapping PyBaMM parameter names to values.
                       Values can be scalars, arrays, or pybamm functions.

        Returns:
            pybamm.ParameterValues ready for simulation.
        """
        params = self.get_default_parameters()
        for key, value in overrides.items():
            if key in params:
                params[key] = value
            else:
                params[key] = value
        return params

    @staticmethod
    def list_available_parameters(params: pybamm.ParameterValues, filter_str: str = ""):
        """List parameter keys, optionally filtered by substring."""
        keys = sorted(params.keys())
        if filter_str:
            keys = [k for k in keys if filter_str.lower() in k.lower()]
        return keys
