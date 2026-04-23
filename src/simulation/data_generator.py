import numpy as np
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .models import MetalBatteryDFN
from .solver import PybammSolver
from .parameters import parse_sweep_params, load_config

logger = logging.getLogger(__name__)


def _run_single_simulation(
    param_dict: dict,
    battery_model: MetalBatteryDFN,
    c_rate: float,
    temperature: float,
    t_end: float,
    n_points: int,
) -> dict:
    """Run a single simulation with given parameter overrides."""
    params = battery_model.build_parameter_set(param_dict)
    solver = PybammSolver(battery_model)
    return solver.solve(params, c_rate, temperature, t_end, n_points)


class SyntheticDataGenerator:
    """
    Generate synthetic training data by running PyBaMM simulations
    across a parameter grid.

    Output is saved as compressed numpy arrays (.npz) for efficient loading.
    """

    def __init__(self, config_path: str, output_dir: str = "data/synthetic"):
        self.config = load_config(config_path)
        self.chemistry = self.config["model"]["chemistry"]
        self.battery_model = MetalBatteryDFN(chemistry=self.chemistry)
        self.sweep_params = parse_sweep_params(self.config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sim_cfg = self.config.get("simulation", {})
        self.c_rates = sim_cfg.get("c_rates", [0.5, 1.0, 2.0])
        self.temperatures = sim_cfg.get("temperatures", [25.0])
        self.t_end = sim_cfg.get("t_end", 3600)
        self.n_points = sim_cfg.get("num_time_points", 200)
        self.num_samples = sim_cfg.get("num_samples", 10000)

    def generate(self, num_workers: int = 1, seed: int = 42) -> Path:
        """
        Generate synthetic dataset.

        Args:
            num_workers: Number of parallel processes (1 = sequential).
            seed: Random seed for reproducibility.

        Returns:
            Path to the saved .npz file.
        """
        rng = np.random.default_rng(seed)

        param_samples = []
        for _ in range(self.num_samples):
            sample = {}
            for p in self.sweep_params:
                sample[p.name] = p.sample(rng)
            param_samples.append(sample)

        all_results = []
        total = len(param_samples) * len(self.c_rates) * len(self.temperatures)
        logger.info(f"Generating {total} simulations...")

        run_fn = partial(
            _run_single_simulation,
            battery_model=self.battery_model,
            t_end=self.t_end,
            n_points=self.n_points,
        )

        count = 0
        for i, param_dict in enumerate(param_samples):
            for c_rate in self.c_rates:
                for temp in self.temperatures:
                    result = run_fn(
                        param_dict=param_dict,
                        c_rate=c_rate,
                        temperature=temp,
                    )
                    if result is not None:
                        all_results.append({
                            "index": count,
                            "param_sample_idx": i,
                            "c_rate": c_rate,
                            "temperature": temp,
                            "time": result["time"],
                            "voltage": result["voltage"],
                            "current": result["current"],
                            "params": param_dict,
                        })
                        count += 1

                    if (count + 1) % 100 == 0:
                        logger.info(f"  Completed {count + 1}/{total}")

        self._save(all_results)
        output_path = self.output_dir / f"synthetic_{self.chemistry}.npz"
        logger.info(f"Saved {len(all_results)} simulations to {output_path}")
        return output_path

    def _save(self, results: list[dict]):
        """Save results to compressed npz file."""
        output_path = self.output_dir / f"synthetic_{self.chemistry}.npz"

        max_len = max(len(r["time"]) for r in results)

        times = np.zeros((len(results), max_len))
        voltages = np.zeros((len(results), max_len))
        currents = np.zeros((len(results), max_len))
        masks = np.zeros((len(results), max_len), dtype=bool)
        c_rates = np.zeros(len(results))
        temperatures = np.zeros(len(results))
        param_names = [p.name for p in self.sweep_params]
        param_values = np.zeros((len(results), len(param_names)))

        for i, r in enumerate(results):
            n = len(r["time"])
            times[i, :n] = r["time"]
            voltages[i, :n] = r["voltage"]
            currents[i, :n] = r["current"]
            masks[i, :n] = True
            c_rates[i] = r["c_rate"]
            temperatures[i] = r["temperature"]
            for j, pname in enumerate(param_names):
                param_values[i, j] = r["params"][pname]

        np.savez_compressed(
            output_path,
            times=times,
            voltages=voltages,
            currents=currents,
            masks=masks,
            c_rates=c_rates,
            temperatures=temperatures,
            param_names=np.array(param_names),
            param_values=param_values,
        )
