import numpy as np
from scipy.interpolate import interp1d
from typing import Optional


class Preprocessor:
    """
    Preprocess battery cycling data for PINN training.

    Operations:
    - Resample to uniform time grid
    - Normalize quantities to [0, 1] or standard score
    - Split by cycle, C-rate, temperature
    - Compute derived quantities (SOC, dV/dt, etc.)
    """

    def __init__(
        self,
        time_normalize: str = "t_end",
        voltage_range: Optional[tuple[float, float]] = None,
        current_range: Optional[tuple[float, float]] = None,
    ):
        self.time_normalize = time_normalize
        self.voltage_range = voltage_range
        self.current_range = current_range

        self._stats = {}

    def resample(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample to uniform time grid using linear interpolation."""
        t_min, t_max = time[0], time[-1]
        t_uniform = np.linspace(t_min, t_max, n_points)

        v_interp = interp1d(time, voltage, kind="linear", fill_value="extrapolate")
        i_interp = interp1d(time, current, kind="linear", fill_value="extrapolate")

        return t_uniform, v_interp(t_uniform), i_interp(t_uniform)

    def normalize_time(
        self, time: np.ndarray, t_end: Optional[float] = None
    ) -> np.ndarray:
        """Normalize time to [0, 1]."""
        if t_end is None:
            t_end = time[-1]
        return time / t_end

    def normalize_voltage(self, voltage: np.ndarray) -> np.ndarray:
        """Normalize voltage to [0, 1] using configured range or data range."""
        if self.voltage_range is not None:
            v_min, v_max = self.voltage_range
        else:
            v_min, v_max = voltage.min(), voltage.max()

        self._stats["v_min"] = v_min
        self._stats["v_max"] = v_max
        return (voltage - v_min) / (v_max - v_min + 1e-12)

    def normalize_current(
        self, current: np.ndarray, capacity_ah: float
    ) -> np.ndarray:
        """Normalize current by C-rate (current / (capacity))."""
        return current / (capacity_ah + 1e-12)

    def denormalize_voltage(self, v_norm: np.ndarray) -> np.ndarray:
        """Denormalize voltage back to physical units."""
        v_min = self._stats.get("v_min", 0)
        v_max = self._stats.get("v_max", 1)
        return v_norm * (v_max - v_min) + v_min

    def compute_soc(
        self,
        current: np.ndarray,
        time: np.ndarray,
        capacity_ah: float,
        initial_soc: float = 1.0,
    ) -> np.ndarray:
        """Compute state of charge from current and time via Coulomb counting."""
        dt = np.diff(time, prepend=time[0])
        dt[0] = dt[1]
        charge_ah = np.cumsum(current * dt) / 3600.0
        soc = initial_soc - charge_ah / capacity_ah
        return np.clip(soc, 0, 1)

    def split_by_cycle(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        cycle_indices: np.ndarray,
    ) -> list[dict]:
        """Split continuous data into individual cycles."""
        unique_cycles = np.unique(cycle_indices)
        cycles = []
        for c in unique_cycles:
            mask = cycle_indices == c
            cycles.append({
                "cycle": int(c),
                "time": time[mask],
                "voltage": voltage[mask],
                "current": current[mask],
            })
        return cycles

    def prepare_forward_training(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        params: dict[str, float],
        n_points: int = 200,
    ) -> dict:
        """
        Prepare a single simulation for forward PINN training.

        Returns dict with normalized quantities and metadata.
        """
        t_resampled, v_resampled, i_resampled = self.resample(
            time, voltage, current, n_points
        )

        t_norm = self.normalize_time(t_resampled)

        return {
            "t": t_norm.astype(np.float32),
            "v": v_resampled.astype(np.float32),
            "i": i_resampled.astype(np.float32),
            "t_physical": t_resampled.astype(np.float32),
            "params": params,
        }

    def prepare_inverse_training(
        self,
        time: np.ndarray,
        voltage: np.ndarray,
        current: np.ndarray,
        n_points: int = 200,
    ) -> dict:
        """
        Prepare experimental data for inverse PINN training.

        Returns normalized collocation points and observation data.
        """
        t_resampled, v_resampled, i_resampled = self.resample(
            time, voltage, current, n_points
        )

        t_norm = self.normalize_time(t_resampled)

        return {
            "t_colloc": t_norm.astype(np.float32),
            "v_obs": v_resampled.astype(np.float32),
            "i_input": i_resampled.astype(np.float32),
            "t_physical": t_resampled.astype(np.float32),
        }
