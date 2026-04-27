import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


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

    def load_neware_xlsx(
        self,
        path: Union[str, Path],
        sheet_name: str = "record",
    ) -> dict[str, np.ndarray]:
        """
        Load data from Neware Excel (.xlsx) format.
        
        Expected columns in 'record' sheet:
        - 数据序号 (data index)
        - 循环号 (cycle number)
        - 工步号 (step number)
        - 工步类型 (step type: 搁置/充电/放电)
        - 时间 (step time)
        - 总时间 (total time)
        - 电流(A) (current)
        - 电压(V) (voltage)
        - 容量(Ah) (capacity)
        - 能量(Wh) (energy)
        
        Returns:
            dict with keys: 'time', 'voltage', 'current', 'capacity', 'cycle', 'step_type'
        """
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required to load xlsx files. Run: pip install openpyxl")

        path = Path(path)
        logger.info(f"Loading Neware xlsx from {path}, sheet '{sheet_name}'")
        
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb[sheet_name]
        
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            raise ValueError(f"Sheet '{sheet_name}' has insufficient data")
        
        header = [str(v) if v is not None else "" for v in rows[0]]
        logger.info(f"Columns: {header}")
        
        NEWARE_COLUMN_MAP = {
            "数据序号": "data_idx",
            "循环号": "cycle",
            "工步号": "step",
            "工步类型": "step_type",
            "时间": "step_time",
            "总时间": "time",
            "电流(A)": "current",
            "电压(V)": "voltage",
            "容量(Ah)": "capacity",
            "能量(Wh)": "energy",
        }
        
        col_idx = {}
        for col_name, key in NEWARE_COLUMN_MAP.items():
            try:
                idx = header.index(col_name)
                col_idx[key] = idx
            except ValueError:
                pass
        
        required_cols = ["time", "voltage", "current"]
        for col in required_cols:
            if col not in col_idx:
                raise ValueError(f"Required column '{col}' not found in sheet")
        
        data = {
            "time": [],
            "voltage": [],
            "current": [],
            "cycle": [],
            "step": [],
            "step_type": [],
        }
        
        for row in rows[1:]:
            if row is None:
                continue
            try:
                time_val = row[col_idx["time"]]
                voltage_val = row[col_idx["voltage"]]
                current_val = row[col_idx["current"]]
                
                if time_val is None or voltage_val is None or current_val is None:
                    continue
                
                data["time"].append(self._parse_time_to_seconds(time_val))
                data["voltage"].append(float(voltage_val))
                data["current"].append(float(current_val))
                
                if "cycle" in col_idx:
                    data["cycle"].append(int(row[col_idx["cycle"]]) if row[col_idx["cycle"]] is not None else 0)
                else:
                    data["cycle"].append(0)
                    
                if "step" in col_idx:
                    data["step"].append(int(row[col_idx["step"]]) if row[col_idx["step"]] is not None else 0)
                else:
                    data["step"].append(0)
                    
                if "step_type" in col_idx:
                    data["step_type"].append(str(row[col_idx["step_type"]]) if row[col_idx["step_type"]] is not None else "")
                else:
                    data["step_type"].append("")
                    
            except (ValueError, TypeError, IndexError):
                continue
        
        result = {k: np.array(v, dtype=np.float64) for k, v in data.items() if k in ["time", "voltage", "current", "cycle", "step"]}
        
        if "step_type" in data:
            result["step_type"] = data["step_type"]
        
        wb.close()
        
        logger.info(f"Loaded {len(result['time'])} data points")
        
        self._validate(result)
        return result

    @staticmethod
    def _parse_time_to_seconds(time_str) -> float:
        """Parse time string to seconds. Supports: 'HH:MM:SS', seconds float."""
        if time_str is None:
            return 0.0
        
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        time_str = str(time_str).strip()
        
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
        
        try:
            return float(time_str)
        except ValueError:
            return 0.0

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
