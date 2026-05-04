"""PyBaMM model wrappers for metal-anode batteries.

Phase A2 fix: this previously declared `"anode": "metal"` in default_options
but only forwarded `SEI` flags to PyBaMM, so the model fell back to a
standard graphite-anode DFN (Chen2020). For LMB work we now wire up
PyBaMM's lithium plating sub-model and use OKane2022 — the parameter set
designed for plating + SEI co-evolution (O'Kane et al., 2022,
Phys. Chem. Chem. Phys.).

Two operating modes are exposed:

  * ``"plating_dominant"`` — keeps a thin graphite host but drives the
    cell with `lithium plating="partially reversible"` so plating /
    stripping current dominates the negative interface. This is the
    most common LMB-in-PyBaMM idealisation in the literature
    (Hu et al., 2024; O'Kane et al., 2022).
  * ``"intercalation"`` — falls back to vanilla DFN, used for
    Li-ion baselines and ablation studies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pybamm

logger = logging.getLogger(__name__)


_LMB_OPTIONS_PLATING = {
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",
    "SEI": "ec reaction limited",
    "SEI porosity change": "true",
    "loss of active material": ("none", "stress-driven"),
    "particle phases": ("1", "1"),
    "open-circuit potential": ("single", "single"),
}

_LMB_OPTIONS_INTERCALATION = {
    "SEI": "ec reaction limited",
    "SEI porosity change": "true",
}


class MetalBatteryDFN:
    """DFN wrapper for metal-anode batteries with optional lithium plating.

    Args:
        chemistry: ``"lmb"`` (lithium metal) or ``"nmb"`` (sodium metal).
        mode: ``"plating_dominant"`` (default for LMB) or ``"intercalation"``.
            Ignored for ``nmb``.
        parameter_set: PyBaMM parameter set name. Defaults to ``OKane2022``
            for LMB (has plating + SEI parameters) and ``Chayambuka2022``
            for NMB. Override only for ablation.
        options: extra PyBaMM model options merged on top of the chemistry
            defaults.
    """

    SUPPORTED_CHEMISTRIES = ("lmb", "nmb")
    SUPPORTED_MODES = ("plating_dominant", "intercalation")

    def __init__(
        self,
        chemistry: str = "lmb",
        mode: str = "plating_dominant",
        parameter_set: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        if chemistry not in self.SUPPORTED_CHEMISTRIES:
            raise ValueError(
                f"Unsupported chemistry '{chemistry}'. "
                f"Choose from {self.SUPPORTED_CHEMISTRIES}"
            )
        if chemistry == "lmb" and mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose from {self.SUPPORTED_MODES}"
            )
        self.chemistry = chemistry
        self.mode = mode if chemistry == "lmb" else "intercalation"
        self.parameter_set = parameter_set or self._default_parameter_set()

        merged = self._chemistry_default_options().copy()
        if options:
            merged.update(options)
        self._options = merged

        self.model = self._build_model()
        self._default_params = self._load_default_params()

    def _default_parameter_set(self) -> str:
        if self.chemistry == "lmb":
            return "OKane2022"
        return "Chayambuka2022"

    def _chemistry_default_options(self) -> dict[str, Any]:
        if self.chemistry != "lmb":
            return {}
        if self.mode == "plating_dominant":
            return _LMB_OPTIONS_PLATING.copy()
        return _LMB_OPTIONS_INTERCALATION.copy()

    def _build_model(self) -> pybamm.BaseModel:
        if self.chemistry == "nmb":
            return pybamm.sodium_ion.BasicDFN()
        return pybamm.lithium_ion.DFN(self._options)

    def _load_default_params(self) -> pybamm.ParameterValues:
        try:
            params = pybamm.ParameterValues(self.parameter_set)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Parameter set %s not available (%s); falling back",
                self.parameter_set, exc,
            )
            fallback = "Chen2020" if self.chemistry == "lmb" else "Chayambuka2022"
            params = pybamm.ParameterValues(fallback)
            self.parameter_set = fallback
        return params

    def get_default_parameters(self) -> pybamm.ParameterValues:
        return self._default_params.copy()

    def build_parameter_set(self, overrides: dict[str, Any]) -> pybamm.ParameterValues:
        """Return a parameter set with overrides applied.

        Unknown keys are inserted (PyBaMM ignores keys the model does not
        consume). Use ``list_available_parameters`` to sanity-check names
        before sweeping over them.
        """
        params = self.get_default_parameters()
        for key, value in overrides.items():
            params.update({key: value}, check_already_exists=False)
        return params

    @staticmethod
    def list_available_parameters(
        params: pybamm.ParameterValues,
        filter_str: str = "",
    ) -> list[str]:
        keys = sorted(params.keys())
        if filter_str:
            f = filter_str.lower()
            keys = [k for k in keys if f in k.lower()]
        return keys

    def lithium_plating_keys(self) -> list[str]:
        """Convenience: return all parameter keys related to plating / SEI.

        Useful when sweeping for synthetic LMB data generation.
        """
        params = self._default_params
        keys = []
        for k in sorted(params.keys()):
            kl = k.lower()
            if any(tok in kl for tok in ("plating", "sei", "dead lithium", "lithium metal")):
                keys.append(k)
        return keys

    def summary(self) -> dict[str, Any]:
        """Diagnostic summary used by tests and the H20 runbook."""
        return {
            "chemistry": self.chemistry,
            "mode": self.mode,
            "parameter_set": self.parameter_set,
            "options": dict(self._options),
            "n_plating_keys": len(self.lithium_plating_keys()),
        }


def quick_lmb_smoke_test(
    c_rate: float = 0.5,
    t_end: float = 3600.0,
    n_points: int = 101,
) -> dict[str, np.ndarray]:
    """Run a tiny PyBaMM solve so callers can confirm the plating model
    actually integrates on the current machine. Used in tests + H20 setup.
    """
    from .solver import PybammSolver

    battery = MetalBatteryDFN(chemistry="lmb", mode="plating_dominant")
    solver = PybammSolver(battery)
    params = battery.get_default_parameters()
    result = solver.solve(params, c_rate=c_rate, t_end=t_end, n_points=n_points)
    if result is None:
        raise RuntimeError(
            "Smoke test failed: PyBaMM lithium plating solve returned None. "
            "Check your PyBaMM install (>= 24.5) and that OKane2022 is available."
        )
    return result
