import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union


class ExperimentalDataLoader:
    """
    Load battery cycling data from various formats.

    Supported formats:
    - CSV (time, voltage, current, temperature columns)
    - Neware .npy exports
    - Arbin .csv exports
    - Battery Archive format
    """

    COLUMN_ALIASES = {
        "time": [
            "time", "time_s", "time [s]", "test_time", "elapsed_time",
            "Time(s)", "t", "Total Time",
        ],
        "voltage": [
            "voltage", "voltage_v", "v", "V", "potential",
            "Voltage(V)", "Ecell/V", "cell_voltage",
        ],
        "current": [
            "current", "current_a", "i", "I", "I(A)",
            "Current(A)", "A", "current_ma",
        ],
        "temperature": [
            "temperature", "temp", "t_c", "temperature_c",
            "Temperature(℃)", "temp_c",
        ],
        "capacity": [
            "capacity", "q", "charge", "discharge_capacity",
            "Capacity(Ah)", "Q",
        ],
        "cycle": [
            "cycle", "cycle_number", "cycle_index", "Cycle Index",
        ],
    }

    def load_csv(
        self,
        path: Union[str, Path],
        time_col: Optional[str] = None,
        voltage_col: Optional[str] = None,
        current_col: Optional[str] = None,
        temperature_col: Optional[str] = None,
        capacity_col: Optional[str] = None,
        cycle_col: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """
        Load cycling data from CSV file.

        If column names are not specified, auto-detects from COLUMN_ALIASES.

        Returns:
            dict with keys: 'time', 'voltage', 'current', 'temperature' (if available)
        """
        path = Path(path)
        df = pd.read_csv(path)

        col_map = {
            "time": time_col,
            "voltage": voltage_col,
            "current": current_col,
            "temperature": temperature_col,
            "capacity": capacity_col,
            "cycle": cycle_col,
        }

        result = {}
        for key, specified_col in col_map.items():
            if specified_col is not None and specified_col in df.columns:
                result[key] = df[specified_col].values.astype(np.float64)
            else:
                col = self._find_column(key, df.columns)
                if col is not None:
                    result[key] = df[col].values.astype(np.float64)

        self._validate(result)
        return result

    def load_neware(self, path: Union[str, Path]) -> dict[str, np.ndarray]:
        """Load data exported from Neware battery testing system."""
        path = Path(path)
        if path.suffix == ".npy":
            data = np.load(path, allow_pickle=True).item()
            return self._normalize_dict(data)
        return self.load_csv(path)

    def load_arbin(self, path: Union[str, Path]) -> dict[str, np.ndarray]:
        """Load data exported from Arbin battery testing system."""
        df = pd.read_csv(path, encoding="utf-8-sig")
        rename_map = {
            "Test Time (s)": "time",
            "Potential (V)": "voltage",
            "Current (A)": "current",
            "Temperature (C)": "temperature",
            "Charge Capacity (Ah)": "charge_capacity",
            "Discharge Capacity (Ah)": "discharge_capacity",
            "Cycle Index": "cycle",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        result = {}
        for key in ["time", "voltage", "current", "temperature", "cycle"]:
            if key in df.columns:
                result[key] = df[key].values.astype(np.float64)

        if "charge_capacity" in df.columns and "discharge_capacity" in df.columns:
            result["capacity"] = df["charge_capacity"].values + df["discharge_capacity"].values

        self._validate(result)
        return result

    def _find_column(self, key: str, columns) -> Optional[str]:
        """Auto-detect column name from aliases."""
        aliases = self.COLUMN_ALIASES.get(key, [])
        for alias in aliases:
            if alias in columns:
                return alias
        return None

    def _normalize_dict(self, data: dict) -> dict[str, np.ndarray]:
        """Normalize a dict of arrays to standard key names."""
        result = {}
        for key in ["time", "voltage", "current", "temperature", "capacity", "cycle"]:
            col = self._find_column(key, data.keys())
            if col is not None:
                arr = np.asarray(data[col], dtype=np.float64)
                result[key] = arr
        return result

    @staticmethod
    def _validate(result: dict):
        """Basic validation of loaded data."""
        if "time" not in result:
            raise ValueError("Could not find time column in data")
        if "voltage" not in result:
            raise ValueError("Could not find voltage column in data")
        n = len(result["time"])
        for key, arr in result.items():
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch: time has {n} points, "
                    f"'{key}' has {len(arr)} points"
                )


class SyntheticDataLoader:
    """Load synthetic data generated by SyntheticDataGenerator."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.data = np.load(path, allow_pickle=True)

    def get_simulation(self, index: int) -> dict:
        """Get a single simulation by index."""
        mask = self.data["masks"][index]
        return {
            "time": self.data["times"][index][mask],
            "voltage": self.data["voltages"][index][mask],
            "current": self.data["currents"][index][mask],
            "c_rate": float(self.data["c_rates"][index]),
            "temperature": float(self.data["temperatures"][index]),
            "params": {
                name: float(self.data["param_values"][index][i])
                for i, name in enumerate(self.data["param_names"])
            },
        }

    def get_all_params(self) -> np.ndarray:
        """Return all parameter vectors, shape (N, num_params)."""
        return self.data["param_values"]

    @property
    def num_simulations(self) -> int:
        return len(self.data["times"])

    @property
    def param_names(self) -> list[str]:
        return list(self.data["param_names"])
