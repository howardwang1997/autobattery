# Architecture Design Document

## Project: autobattery

**Physics-Informed Neural Network for Metal Battery Electrochemical Modeling**

---

## 1. Overview

This project implements a PINN-based framework for identifying electrochemical parameters and building fast surrogate models for Lithium Metal Batteries (LMB) and Sodium Metal Batteries (NMB).

### Core Pipeline

```
Experimental/Synthetic Data
        │
        ▼
┌──────────────────┐     ┌───────────────────┐
│  PyBaMM P2D Model│     │  Data Preprocessing│
│  (simulation/)   │     │  (data/)           │
│                  │     │                    │
│  Generate 10k+   │────▶│  Normalize, align, │
│  synthetic data  │     │  resample          │
└──────────────────┘     └────────┬──────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │     PINN Framework       │
                    │       (pinn/)            │
                    │                          │
                    │  ┌────────────────────┐  │
                    │  │ Forward Problem     │  │
                    │  │ params → V(t)       │  │
                    │  │ Fast P2D solver     │  │
                    │  └────────────────────┘  │
                    │  ┌────────────────────┐  │
                    │  │ Inverse Problem     │  │
                    │  │ V(t) → params       │  │
                    │  │ Parameter ID        │  │
                    │  └────────────────────┘  │
                    └─────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
                    ▼                            ▼
          Voltage prediction          Parameter identification
          SOC/SOH estimation          D_e, D_s, k_SEI, j₀, η_CE
```

---

## 2. Module Architecture

### 2.1 `src/simulation/` — PyBaMM Simulation Layer

**Purpose:** Generate synthetic training data using physics-based P2D simulations.

| File | Class/Function | Description |
|------|---------------|-------------|
| `models.py` | `MetalBatteryDFN` | Wraps PyBaMM DFN model for LMB/NMB |
| `parameters.py` | `ParameterRange`, `ChemistryConfig` | Parameter management, sweep configs |
| `solver.py` | `PybammSolver` | Single simulation execution |
| `data_generator.py` | `SyntheticDataGenerator` | Batch generation with parameter sweeps |

**Key Design Decisions:**
- Metal anode is modeled by removing solid-phase diffusion in negative electrode
- SEI growth sub-model is included (reaction-limited)
- Both LMB and NMB use the same DFN structure with different parameters
- Output format: compressed `.npz` with padded arrays + masks

### 2.2 `src/data/` — Data Processing Layer

**Purpose:** Load, preprocess, and serve data for PINN training.

| File | Class/Function | Description |
|------|---------------|-------------|
| `loader.py` | `ExperimentalDataLoader` | Load CSV/NPY data from various battery testers |
| `loader.py` | `SyntheticDataLoader` | Load generated `.npz` simulation data |
| `preprocessor.py` | `Preprocessor` | Resample, normalize, compute derived quantities |
| `dataset.py` | `SyntheticDataset` | PyTorch Dataset for forward training |
| `dataset.py` | `ExperimentalDataset` | PyTorch Dataset for inverse training |
| `dataset.py` | `CollocationDataset` | Random PDE collocation point generator |

**Normalization Strategy:**
- Time: t_norm = t / t_end ∈ [0, 1]
- Voltage: V_norm = (V - V_min) / (V_max - V_min)
- Parameters: standardized (mean=0, std=1) across sweep range
- Physical constants kept in SI units within PDE residuals

### 2.3 `src/pinn/` — PINN Core Framework

**Purpose:** Neural network architectures, PDE definitions, loss functions, and training loops.

#### `network.py` — Neural Network Architectures

**`MultiDomainPINN`:**
- Shared encoder: `(t, x, r, params, domain_embed)` → hidden features
- Three domain heads:
  - Negative electrode: c_e, φ_s, φ_e (3 outputs)
  - Separator: c_e, φ_e (2 outputs)
  - Positive electrode: c_e, φ_s, φ_e, c_s (4 outputs)
- Global head: V, L_SEI (2 outputs)
- Domain encoded via learnable embedding (dim=16)
- Default: 6 hidden layers × 128 neurons, SiLU activation

**`InversePINN`:**
- Wraps MultiDomainPINN
- Makes selected parameters learnable via `nn.ParameterDict`
- Parameter bounds enforced via sigmoid + log-space transform
- Supports multi-data joint optimization

#### `pdes.py` — PDE Residual Definitions

**`MetalBatteryPDE`** implements residuals for:

1. **Electrolyte mass transport:**
   ε ∂c_e/∂t = ∂/∂x[ε^b D_e ∂c_e/∂x] + (1-t⁺)aj/F

2. **Metal plating/stripping (Butler-Volmer):**
   j = j₀[exp(αₐFη/RT) - exp(-α_cFη/RT)]
   η = φ_s - φ_e (no OCP for metal anode)

3. **SEI growth (reaction-limited):**
   ∂L_SEI/∂t = k_SEI · exp(-Ea/RT) · j_side/F

4. **Cathode intercalation (Fick's in spherical):**
   ∂c_s/∂t = D_s/R_p² · (∂²c_s/∂r'² + 2/r' · ∂c_s/∂r')

5. **Charge conservation:**
   ∂/∂x[σ_eff ∂φ_s/∂x] = a·j

**All derivatives computed via `torch.autograd`.**

#### `losses.py` — Loss Function

`PINNLoss` combines:
- L_data: MSE(V_pred, V_obs)
- L_pde: Σ w_i · mean(R_i²) for each PDE residual
- L_bc: Boundary condition enforcement
- L_ic: Initial condition enforcement

Configurable weights: λ_data, λ_pde, λ_bc, λ_ic

#### `forward.py` — Forward Problem Training

`ForwardTrainer`:
- Input: PyBaMM simulation data (params → V(t) curves)
- Output: Fast P2D surrogate (10-100x speedup)
- Supports PDE-regularized training with collocation points
- Adam optimizer with cosine annealing scheduler
- Gradient clipping (max_norm=1.0)

#### `inverse.py` — Inverse Problem Training

`InverseTrainer`:
- Input: Experimental voltage curves
- Output: Identified electrochemical parameters
- Two-phase optimization:
  - Phase 1 (Adam): Fast exploration, 5000 epochs
  - Phase 2 (L-BFGS): Precise convergence, 5000 epochs
- Separate learning rates for network weights vs. physical parameters
- Parameter bounds enforced via log-space transform

### 2.4 `src/utils/` — Utilities

| File | Description |
|------|-------------|
| `physics.py` | Physical constants (F, R, T_ref, ion properties) |
| `visualization.py` | Plotting functions (voltage comparison, training history, multi-C-rate) |

### 2.5 `src/mlff/` — ML Force Field Integration (Optional)

| File | Description |
|------|-------------|
| `diffusion.py` | Load UMA/MACE as ASE calculators, compute diffusion coefficients from MD, compute OCV from energies |

---

## 3. Configuration System

YAML-based configs with inheritance:

```
configs/
├── base.yaml      # Default parameters, training settings, physics constants
├── lmb.yaml       # LMB-specific overrides (learnable params, PyBaMM params)
└── nmb.yaml       # NMB-specific overrides
```

`load_config()` automatically merges `base.yaml` with chemistry-specific config.

---

## 4. Data Flow

### Forward Training Data Flow
```
PyBaMM simulation (10k runs)
    → SyntheticDataGenerator
    → .npz file (times, voltages, currents, masks, params)
    → SyntheticDataset (PyTorch)
    → DataLoader (batched)
    → ForwardTrainer
```

### Inverse Training Data Flow
```
Experimental CSV files (V, I, T vs t)
    → ExperimentalDataLoader
    → Preprocessor (resample, normalize)
    → t_colloc, v_obs tensors
    → InverseTrainer
    → Identified parameters + voltage predictions
```

---

## 5. Hardware Requirements

| Task | GPU Memory | Training Time (4×H20) |
|------|-----------|----------------------|
| PyBaMM data generation | CPU only | ~1 day |
| Forward PINN training | 4-8 GB | ~2-3 days |
| Inverse PINN (Adam phase) | 4-8 GB | ~1 day |
| Inverse PINN (L-BFGS phase) | 4-8 GB | ~1 day |
| MLFF MD simulation | 8-32 GB | ~1-4 hours per material |

---

## 6. Key Design Principles

1. **Separation of physics and ML:** PDE residuals are pure math in `pdes.py`, independent of network architecture
2. **Config-driven:** All hyperparameters in YAML, no hardcoded values
3. **Reproducible:** Fixed seeds, saved checkpoints, logged history
4. **Extensible:** New chemistries = new config file; new physics = new PDE residual method
5. **Testable:** Unit tests for PDE residuals, network shapes, inverse convergence
