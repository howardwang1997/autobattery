"""Hyperparameter sweep for PINN training."""

import argparse
import logging
import itertools
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lmb.yaml")
    parser.add_argument("--data", type=str, default="data/synthetic/synthetic_lmb.npz")
    args = parser.parse_args()

    sweep_grid = {
        "hidden_dim": [64, 128, 256],
        "num_layers": [4, 6, 8],
        "activation": ["silu", "tanh"],
        "lambda_data": [1.0, 10.0, 50.0],
        "lambda_pde": [0.1, 1.0, 10.0],
        "learning_rate": [1e-4, 1e-3, 1e-2],
    }

    keys = list(sweep_grid.keys())
    values = list(sweep_grid.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f"Total combinations: {total}")

    print("This is a template script. Implement the sweep logic based on your cluster setup.")
    print("Recommended: use Weights & Biases sweeps or Ray Tune for distributed hyperparameter search.")


if __name__ == "__main__":
    main()
