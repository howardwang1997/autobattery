import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FullFieldDataset(Dataset):
    """Load full-field simulation data from HDF5 for FNO training."""

    FIELD_KEYS_2D = ["c_e", "phi_e", "phi_s_neg", "phi_s_pos", "j_pos", "j_neg", "L_sei"]
    FIELD_KEY_3D = "c_s_pos"

    def __init__(self, h5_path: str, fields: list = None, normalize: bool = True):
        self.h5_path = str(h5_path)
        self.fields = fields or ["c_e", "phi_e"]
        self.normalize = normalize

        with h5py.File(self.h5_path, "r") as f:
            self.n_sims = int(f.attrs["n_sims"])
            self.n_time = int(f.attrs["n_time"])
            self.nx_full = int(f.attrs["nx_full"])
            self.nx_neg = int(f.attrs.get("nx_neg", self.nx_full // 3))
            self.nx_pos = int(f.attrs.get("nx_pos", self.nx_full // 3))
            self.nr_pos = int(f.attrs.get("nr_pos", self.nx_pos))
            self.param_names = [p.decode() if isinstance(p, bytes) else p for p in f.attrs["param_names"]]

            logger.info(f"Loading {self.n_sims} simulations from {h5_path} into memory...")
            self.params = f["params"][:].astype(np.float32)
            self.c_rates = f["c_rates"][:].astype(np.float32)
            self.V = f["V"][:].astype(np.float32)
            self.c_e = f["c_e"][:].astype(np.float32)
            self.phi_e = f["phi_e"][:].astype(np.float32)

            self.fields_data = {}
            for key in self.fields:
                if key in f:
                    self.fields_data[key] = f[key][:].astype(np.float32)
            logger.info(f"Data loaded. Fields: {list(self.fields_data.keys())}")

        self._stats = None
        if normalize:
            self._stats = self._compute_stats()

        perm = torch.randperm(self.n_sims)
        n_val = max(1, self.n_sims // 10)
        self.train_idx = perm[n_val:]
        self.val_idx = perm[:n_val]

    def _compute_stats(self):
        stats = {}
        for key, data in self.fields_data.items():
            stats[key] = {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)) + 1e-8,
            }
        stats["params"] = {
            "mean": self.params.mean(axis=0),
            "std": self.params.std(axis=0) + 1e-12,
        }
        stats["V"] = {
            "mean": float(np.nanmean(self.V)),
            "std": float(np.nanstd(self.V)) + 1e-8,
        }
        return stats

    def __len__(self):
        return self.n_sims

    def __getitem__(self, idx):
        params = self.params[idx].copy()
        c_rate = self.c_rates[idx]
        V = self.V[idx].copy()

        fields = []
        for key in self.fields:
            if key in self.fields_data:
                fields.append(self.fields_data[key][idx].copy())

        if self._stats:
            p_stats = self._stats["params"]
            params = (params - p_stats["mean"]) / p_stats["std"]
            V = (V - self._stats["V"]["mean"]) / self._stats["V"]["std"]
            normalized_fields = []
            for fld, key in zip(fields, self.fields):
                s = self._stats.get(key, {"mean": 0, "std": 1})
                normalized_fields.append((fld - s["mean"]) / s["std"])
            fields = normalized_fields

        if fields:
            field_tensor = np.stack(fields, axis=0)
        else:
            field_tensor = np.zeros(1, self.nx_full, self.n_time, dtype=np.float32)

        t_grid = np.linspace(0, 1, self.n_time, dtype=np.float32)
        x_grid = np.linspace(0, 1, self.nx_full, dtype=np.float32)
        T, X = np.meshgrid(t_grid, x_grid)
        coord = np.stack([X, T], axis=0)

        return {
            "coord": torch.from_numpy(coord),
            "fields": torch.from_numpy(field_tensor),
            "params": torch.from_numpy(params),
            "c_rate": torch.tensor([c_rate], dtype=torch.float32),
            "voltage": torch.from_numpy(V),
        }
