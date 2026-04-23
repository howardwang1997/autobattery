# autobattery

Physics-Informed Neural Network for Metal Battery Electrochemical Modeling.

## Overview

This project uses PINNs to identify electrochemical parameters and build fast surrogate models for **Lithium Metal Batteries (LMB)** and **Sodium Metal Batteries (NMB)**.

**Key capabilities:**
- **Forward problem:** Fast P2D simulation (10-100x speedup over PyBaMM FEM)
- **Inverse problem:** Identify D_e, D_s, k_SEI, j₀, η_CE from experimental cycling data
- **Multi-physics:** Metal plating/stripping + SEI growth + electrolyte transport + cathode intercalation

## Quick Start

### Installation

```bash
conda env create -n autobattery python=3.11
conda activate autobattery
pip install torch
pip install numpy scipy pandas pybamm pyyaml matplotlib tqdm pytest
pip install -e .
```

For ML force field support (optional):
```bash
pip install -e ".[mlff]"
```

### 1. Generate Synthetic Data

```bash
python scripts/01_generate_synthetic.py --config configs/lmb.yaml
```

### 2. Train Forward PINN

```bash
python scripts/02_train_forward.py --config configs/lmb.yaml --data data/synthetic/synthetic_lmb.npz
```

### 3. Train Inverse PINN (Parameter Identification)

```bash
python scripts/03_train_inverse.py --config configs/lmb.yaml --data-dir data/raw
```

### 4. Validate and Visualize

```bash
python scripts/04_validate.py --checkpoint outputs/checkpoints/forward_pinn_final.pt --data-dir data/raw
```

## Project Structure

```
src/
├── simulation/    PyBaMM P2D model, parameter management, data generation
├── data/          Data loading, preprocessing, PyTorch datasets
├── pinn/          PINN architectures, PDE residuals, losses, trainers
├── utils/         Physics constants, visualization
└── mlff/          ML force field integration (UMA/MACE)
configs/           YAML configuration files (base, lmb, nmb)
scripts/           Training and validation scripts
tests/             Unit tests
docs/              Architecture docs, work log
```

## Configuration

Edit `configs/lmb.yaml` for LMB or `configs/nmb.yaml` for NMB. Key sections:

- `model` — Network architecture (hidden_dim, num_layers, activation)
- `training.forward` — Forward PINN training hyperparameters
- `training.inverse` — Inverse PINN: learnable parameters with initial values and bounds
- `simulation` — PyBaMM simulation settings (C-rates, temperatures, parameter sweep ranges)

## Documentation

- [Architecture Design](docs/architecture.md)
- [Work Log](docs/work-log.md)
- [Research Survey](RESEARCH_SURVEY.md)
- [Na-ion Digital Twin Research](NA_ION_DIGITAL_TWIN_RESEARCH.md)

## Hardware

Designed for NVIDIA H20 GPUs (96GB HBM3). 4 GPUs sufficient for all tasks.

## Current Progress & TODO

**Completed:**
- [x] Research survey (6 directions analyzed, Na-ion digital twin chosen)
- [x] Project architecture & code (38 files, 15 unit tests passing)
- [x] PyBaMM baseline verified: Li-ion (0.2-2.0C) + Na-ion (0.1-0.5C)
- [x] conda environment set up

**Next Steps (see `docs/work-log.md` for details):**
- [ ] Load LMB experimental data into `data/raw/`
- [ ] Generate synthetic training data via parameter sweep
- [ ] Train forward PINN (fast P2D solver)
- [ ] Train inverse PINN (parameter identification from experimental data)
- [ ] Validate results and prepare paper figures

## License

MIT
