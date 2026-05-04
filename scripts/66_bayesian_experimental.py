#!/usr/bin/env python3
"""
Bayesian diagnosis on experimental cycling data.

Load a trained Bayesian model and apply it to experimental discharge curves.
Report: per-parameter uncertainty, identifiable vs unidentifiable grouping.
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
IDENTIFIABLE = ["D_n", "SEI", "LAM_neg"]


class BayesianNN(nn.Module):
    def __init__(self, n_time=100, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_time + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 7))
        self.logvar = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 7))

    def forward(self, V, c_rate):
        h = self.shared(torch.cat([V, c_rate], dim=-1))
        return self.mu(h), self.logvar(h)


class ExpDataset(Dataset):
    def __init__(self, h5_path, n_early=10):
        self.samples = []
        with h5py.File(h5_path, "r") as f:
            for key in sorted(f["cells"].keys()):
                grp = f["cells"][key]
                V = grp["V"][:]
                ncyc = int(grp.attrs["n_cycles"])
                if ncyc < n_early:
                    continue
                for cyc_idx in [0, n_early - 1, min(n_early * 2, ncyc - 1)]:
                    if cyc_idx < ncyc:
                        self.samples.append(V[cyc_idx])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.tensor(self.samples[i], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-data", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="outputs/bayesian/noisy/noisy_5mv.pt")
    parser.add_argument("--output", type=str, default="outputs/bayesian/experimental/")
    parser.add_argument("--n-early", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_data = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BayesianNN().to(device)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    p_mean = ckpt_data["p_mean"]
    p_std = ckpt_data["p_std"]

    dataset = ExpDataset(args.exp_data, n_early=args.n_early)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    logger.info(f"Experimental samples: {len(dataset)}")

    all_mu, all_std = [], []
    c_rate = torch.ones(1, 1, device=device)

    with torch.no_grad():
        for V in loader:
            V = V.to(device)
            cr = c_rate.expand(V.shape[0], 1)
            mu, logvar = model(V, cr)
            all_mu.append(mu.cpu().numpy())
            all_std.append(torch.exp(0.5 * logvar).cpu().numpy())

    all_mu = np.concatenate(all_mu)
    all_std = np.concatenate(all_std)

    params_real = all_mu * p_std + p_mean

    logger.info("\n=== Per-parameter uncertainty on experimental data ===")
    for i, name in enumerate(PARAM_NAMES):
        mean_val = params_real[:, i].mean()
        std_val = all_std[:, i].mean() * p_std[i]
        snr = np.abs(mean_val) / (std_val + 1e-10)
        ident = "IDENTIFIABLE" if name in IDENTIFIABLE else "unidentifiable"
        logger.info(f"  {name:10s}: mean={mean_val:10.3e}, uncertainty={std_val:10.3e}, SNR={snr:6.1f} [{ident}]")

    np.savez(
        output_dir / "diagnosis_results.npz",
        mu=all_mu,
        std=all_std,
        params_real=params_real,
        p_mean=p_mean,
        p_std=p_std,
    )
    logger.info(f"Saved to {output_dir / 'diagnosis_results.npz'}")


if __name__ == "__main__":
    main()
