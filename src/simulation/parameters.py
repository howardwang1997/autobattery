import yaml
import numpy as np
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParameterRange:
    """Defines a sweepable parameter range."""
    name: str
    min: float
    max: float
    distribution: str = "uniform"  # "uniform" or "log_uniform"
    init: Optional[float] = None
    bounds: Optional[tuple] = None

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == "log_uniform":
            log_min, log_max = np.log10(self.min), np.log10(self.max)
            return float(10 ** rng.uniform(log_min, log_max))
        return float(rng.uniform(self.min, self.max))

    @property
    def initial_value(self) -> float:
        return self.init if self.init is not None else (self.min + self.max) / 2


@dataclass
class ChemistryConfig:
    """Configuration for a specific battery chemistry."""
    name: str
    learnable_params: dict[str, ParameterRange] = field(default_factory=dict)
    pybamm_overrides: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file with inheritance from base.yaml."""
    config_path = Path(config_path)
    configs_dir = config_path.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_path = configs_dir / "base.yaml"
    if base_path.exists() and config_path.name != "base.yaml":
        with open(base_path) as f:
            base = yaml.safe_load(f)
        config = _deep_merge(base, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_sweep_params(config: dict) -> list[ParameterRange]:
    """Parse parameter sweep configuration into ParameterRange objects."""
    sweep = config.get("simulation", {}).get("parameter_sweep", {})
    params = []
    for name, spec in sweep.items():
        params.append(ParameterRange(
            name=name,
            min=spec["min"],
            max=spec["max"],
            distribution=spec.get("distribution", "uniform"),
        ))
    return params


def parse_learnable_params(config: dict) -> dict[str, ParameterRange]:
    """Parse learnable parameters for inverse problem from config."""
    inv_cfg = config.get("training", {}).get("inverse", {}).get("learnable_params", {})
    params = {}
    for name, spec in inv_cfg.items():
        params[name] = ParameterRange(
            name=name,
            min=spec.get("bounds", [spec["init"] * 0.1, spec["init"] * 10])[0],
            max=spec.get("bounds", [spec["init"] * 0.1, spec["init"] * 10])[1],
            init=spec["init"],
            bounds=tuple(spec.get("bounds", [])),
        )
    return params


def get_physics_constants(config: dict) -> dict[str, float]:
    """Extract physics constants from config."""
    return config.get("physics", {})
