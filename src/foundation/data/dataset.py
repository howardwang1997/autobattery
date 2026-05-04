"""Multi-chemistry dataset for Battery Foundation Model training."""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MultiChemDataset(Dataset):
    """Loads multi-chemistry data from HDF5 and normalizes for training."""

    def __init__(self, h5_path, split="train", train_ratio=0.9, seed=42):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            V = f["V"][:]
            chem_ids = f["chem_ids"][:]
            c_rates = f["c_rates"][:]
            params = f["params"][:]
            self.n_time = int(f.attrs["n_time"])
            self.max_n_params = int(f.attrs["max_n_params"])

            if "temperatures" in f:
                temperatures = f["temperatures"][:]
            else:
                temperatures = np.full(len(V), 298.15, dtype=np.float32)

            self.chem_names = {}
            self.chem_param_names = {}
            self.chem_v_stats = {}
            self.chem_p_stats = {}
            self.chem_n_params = {}

            for key in f.keys():
                if key.startswith("chemistry_"):
                    grp = f[key]
                    cid = int(key.split("_")[1])
                    self.chem_names[cid] = grp.attrs["name"]
                    self.chem_param_names[cid] = grp.attrs.get("param_names", [])
                    self.chem_n_params[cid] = int(grp.attrs.get("n_params", self.max_n_params))
                    mask = chem_ids == cid
                    if mask.sum() > 0:
                        self.chem_v_stats[cid] = {
                            "min": float(V[mask].min()),
                            "max": float(V[mask].max()),
                            "mean": float(V[mask].mean()),
                            "std": float(V[mask].std()),
                        }

        n_total = len(V)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        n_train = int(n_total * train_ratio)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train:]
        else:
            self.indices = indices

        self.V_all = V[self.indices]
        self.chem_all = chem_ids[self.indices]
        self.crate_all = c_rates[self.indices]
        self.params_all = params[self.indices]
        self.temp_all = temperatures[self.indices]

        p_flat = self.params_all[self.params_all > 0]
        self.p_global_min = np.percentile(p_flat, 1)
        self.p_global_max = np.percentile(p_flat, 99)

        logger.info(f"MultiChemDataset({split}): {len(self)} samples, "
                     f"{len(self.chem_names)} chemistries")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        V = self.V_all[idx].copy()
        chem_id = int(self.chem_all[idx])
        c_rate = float(self.crate_all[idx])
        temperature = float(self.temp_all[idx])
        pvec = self.params_all[idx].copy()

        v_stats = self.chem_v_stats.get(chem_id, {"min": 2.5, "max": 4.2})
        V_norm = (V - v_stats["min"]) / (v_stats["max"] - v_stats["min"] + 1e-8)
        V_norm = np.clip(V_norm, 0, 1).astype(np.float32)

        n_p = self.chem_n_params.get(chem_id, self.max_n_params)
        p_active = pvec[:n_p]
        p_norm = np.zeros(self.max_n_params, dtype=np.float32)
        pos = p_active > 0
        if pos.any():
            p_log = np.log10(np.abs(p_active))
            p_norm[:n_p] = p_log / 5.0
        p_norm[n_p:] = 0

        return {
            "V": torch.tensor(V_norm),
            "chem_id": torch.tensor(chem_id, dtype=torch.long),
            "params": torch.tensor(p_norm),
            "conditions": torch.tensor([c_rate, temperature], dtype=torch.float32),
        }


def get_dataloaders(h5_path, batch_size=64, num_workers=0):
    train_ds = MultiChemDataset(h5_path, split="train")
    val_ds = MultiChemDataset(h5_path, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
