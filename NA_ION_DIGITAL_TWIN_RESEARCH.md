# Na-ion Battery Digital Twin: Technical Research Report
## PINNs + Pretrained ML Force Fields Architecture

**Date:** April 2026  
**Purpose:** Detailed technical research for building a Na-ion battery digital twin using Physics-Informed Neural Networks and pretrained ML force fields, targeting top-journal publication.

---

## Table of Contents
1. [PyBaMM Na-ion Support](#1-pybamm-na-ion-support)
2. [Pretrained ML Force Fields for Na Systems](#2-pretrained-ml-force-fields-for-na-systems)
3. [PINN for Electrochemistry](#3-pinn-for-electrochemistry)
4. [Na-ion Specific Parameters](#4-na-ion-specific-parameters-differing-from-li-ion)
5. [Published Na-ion Digital Twin / PINN Papers](#5-published-na-ion-digital-twin--pinn-papers)
6. [Available Na-ion Experimental Data](#6-available-experimental-data-for-na-ion-cell-cycling)

---

## 1. PyBaMM Na-ion Support

### 1.1 Available Na-ion Models

PyBaMM (v26.3.1, latest) currently provides **one** dedicated Na-ion model:

- **`pybamm.sodium_ion.BasicDFN`** — Doyle-Fuller-Newman (P2D) model adapted for Na-ion chemistry
  - Located at: `src/pybamm/models/full_battery_models/sodium_ion/basic_dfn.py`
  - Inherits from `pybamm.lithium_ion.BaseModel` — shares the same PDE structure (Fick's diffusion in particles, concentrated solution theory in electrolyte, Butler-Volmer kinetics, charge conservation in solid/electrolyte)
  - Implements the full P2D model: coupled PDEs for solid-phase diffusion, electrolyte transport, electrode potentials, electrolyte potential
  - Default parameters from: `pybamm.ParameterValues("Chayambuka2022")`

**Usage:**
```python
import pybamm
model = pybamm.sodium_ion.BasicDFN()
param = pybamm.ParameterValues("Chayambuka2022")
sim = pybamm.Simulation(model, parameter_values=param)
sim.solve([0, 3600])
sim.plot()
```

**What is NOT yet available for Na-ion (would need custom implementation):**
- `sodium_ion.SPM` / `sodium_ion.SPMe` — Single Particle Model / with electrolyte (not yet implemented)
- `sodium_ion.MSMR` — Multi-Species Multi-Reaction model
- `sodium_ion.DFN` with thermal coupling
- `sodium_ion.NewmanTobias` model
- Half-cell models for Na-ion

These can be adapted from the Li-ion implementations in PyBaMM by substituting Na-ion parameters and chemistry.

### 1.2 Parameter Set: Chayambuka2022

PyBaMM ships one validated Na-ion parameter set based on:

> K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, "Physics-based modeling of sodium-ion batteries part II. Model and validation," *Electrochimica Acta* 404 (2022) 139764. DOI: 10.1016/j.electacta.2021.139764

**Cell Chemistry:** Hard Carbon (HC) anode || NaPF6 in EC:PC (1:1) electrolyte || Na3V2(PO4)2F3 (NVPF) cathode

**Cell Parameters:**
| Parameter | Value |
|-----------|-------|
| Negative electrode thickness | 64 μm |
| Separator thickness | 25 μm |
| Positive electrode thickness | 68 μm |
| Electrode height | 254 μm |
| Nominal cell capacity | 3 mAh |
| Voltage range | 2.0 – 4.2 V |

**Negative Electrode (Hard Carbon):**
| Parameter | Value |
|-----------|-------|
| Electronic conductivity | 256 S/m |
| Maximum concentration | 14,540 mol/m³ |
| Porosity | 0.51 |
| Active material volume fraction | 0.489 |
| Particle radius | 3.48 μm |
| Bruggeman coefficient (electrolyte) | 1.5 |
| Charge transfer coefficient | 0.5 |
| Particle diffusivity | Stoichiometry-dependent (interpolated from CSV data: `D_n.csv`) |
| OCP | Stoichiometry-dependent (interpolated from `U_n.csv`) |
| Exchange current density | Concentration-dependent (interpolated from `k_n.csv`), NaPF6/EC:PC |

**Positive Electrode (NVPF - Na3V2(PO4)2F3):**
| Parameter | Value |
|-----------|-------|
| Electronic conductivity | 50 S/m |
| Maximum concentration | 15,320 mol/m³ |
| Porosity | 0.23 |
| Active material volume fraction | 0.55 |
| Particle radius | 0.59 μm |
| Bruggeman coefficient (electrolyte) | 1.5 |
| Charge transfer coefficient | 0.5 |
| Particle diffusivity | Stoichiometry-dependent (`D_p.csv`) |
| OCP | Stoichiometry-dependent (`U_p.csv`) |
| Exchange current density | Concentration-dependent (`k_p.csv`) |

**Electrolyte (NaPF6 in EC:PC 1:1):**
| Parameter | Value |
|-----------|-------|
| Initial concentration | 1,000 mol/m³ |
| Cation transference number | 0.45 |
| Thermodynamic factor | 1 |
| Diffusivity | Concentration-dependent (`D_e.csv`) |
| Conductivity | Concentration-dependent (`sigma_e.csv`) |
| **Note:** No temperature dependence provided |

**Separator:**
| Parameter | Value |
|-----------|-------|
| Porosity | 0.55 |
| Bruggeman coefficient | 1.5 |

**Initial Conditions:**
| Parameter | Value |
|-----------|-------|
| Initial concentration in negative electrode | 13,520 mol/m³ |
| Initial concentration in positive electrode | 3,320 mol/m³ |
| Reference temperature | 298.15 K |

### 1.3 Available CSV Data Files

Located in `src/pybamm/input/parameters/sodium_ion/data/`:
- `U_n.csv` — HC open-circuit potential vs. stoichiometry
- `U_p.csv` — NVPF open-circuit potential vs. stoichiometry
- `D_n.csv` — HC diffusivity vs. concentration
- `D_p.csv` — NVPF diffusivity vs. concentration
- `k_n.csv` — HC exchange current density reaction rate vs. surface concentration
- `k_p.csv` — NVPF exchange current density reaction rate vs. surface concentration
- `D_e.csv` — Electrolyte (NaPF6/EC:PC) diffusivity vs. concentration
- `sigma_e.csv` — Electrolyte conductivity vs. concentration

### 1.4 Key PDEs in the PyBaMM Na-ion DFN Model

The model encodes the following equations (verified from source code of `basic_dfn.py`):

**1. Solid-phase diffusion (Fick's Law in spherical particles):**
```
∂c_s / ∂t = -∇ · N_s,   N_s = -D_s(c_s, T) ∇c_s
BC: ∂c_s/∂r|_{r=0} = 0,   -D_s ∂c_s/∂r|_{r=R} = j / F
```

**2. Electrolyte concentration (concentrated solution theory):**
```
ε ∂c_e / ∂t = -∇ · N_e + (1-t⁺) a·j / F
N_e = -tor · D_e(c_e, T) ∇c_e
BC: ∂c_e/∂x|_{x=0} = ∂c_e/∂x|_{x=L} = 0
```

**3. Solid-phase charge conservation (Ohm's Law):**
```
∇ · i_s = -a·j
i_s = -σ_eff ∇φ_s
BC: φ_s|_{x=0} = 0,   i_s|_{x=L_neg} = 0 (symmetry)
    i_s|_{x=0_pos} = 0,   i_s|_{x=L} = I/(A·σ_eff)
```

**4. Electrolyte charge conservation (modified Ohm's Law):**
```
∇ · i_e = a·j
i_e = κ_e · tor · (χRT/(Fc_e) ∇c_e - ∇φ_e)
BC: ∂φ_e/∂x|_{x=0} = ∂φ_e/∂x|_{x=L} = 0
```

**5. Butler-Volmer kinetics (symmetric form):**
```
j = 2·j₀ · sinh(n_e · F·η / (2·R·T))
η = φ_s - φ_e - U(c_s_surf, T)
j₀ = F · k(c_s_surf) · (c_e/c_e0)^0.5 · c_s_surf^0.5 · (c_s_max - c_s_surf)^0.5 / 2
```

**6. Terminal voltage:**
```
V = φ_s|_{x=L_pos} - 0  (negative tab grounded)
```

### 1.5 Gap Analysis for PINN Development

For a PINN-based digital twin, the key missing parameter sets are:
- **Layered oxide cathodes:** Na_xCoO2, Na_xMnO2, NaNMC (Na[NiMnCo]O2)
- **Polyanionic cathodes:** Na3V2(PO4)3 (NVP), Na2Fe2(SO4)3, Na2FePO4F
- **Prussian blue analogs:** Na_xMn[Fe(CN)6], Na_xCo[Fe(CN)6]
- **Thermal parameters:** Temperature-dependent diffusivities, conductivities (Chayambuka2022 has none)
- **Different electrolytes:** NaPF6 in EC:DMC, NaClO4 in PC, NaTFSI in diglyme

---

## 2. Pretrained ML Force Fields for Na Systems

### 2.1 FAIR Chemistry / UMA Model

**Current version:** UMA-1.2 (March 2026)

**Element coverage:** UMA models cover the full periodic table — Na (Z=11) is explicitly included in all UMA variants.

**Model variants:**
| Model | Active/Total Params | Speed | Accuracy |
|-------|--------------------|-------|----------|
| uma-s-1p2 | 6.6M / 290M | Fastest | SOTA on most benchmarks |
| uma-m-1p1 | 50M / 1.4B | Slower | Best accuracy |

**Training data relevant to Na systems:**
- **OMat24 dataset** (180M+ DFT calculations): Contains Na-containing bulk crystals — NaCl, Na2O, NaCoO2, NaMnO2, Na3V2(PO4)3, various Na intercalation compounds. Trained with DFT (PBE+U) labels using VASP pseudopotentials.
- **OMol25 dataset** (100M+ molecular DFT calculations): Contains Na-containing molecular systems — Na+ solvation complexes, NaPF6, NaClO4, electrolyte solvent molecules with Na+.
- **OC20/OC22/OC25 datasets** (catalysis): Contains Na on oxide surfaces — relevant for Na-ion electrode/electrolyte interface studies.

**Task names for Na systems:**
```python
# For Na-containing bulk materials (cathodes, anodes, solid electrolytes)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

# For Na+ solvation / electrolyte molecules
calc = FAIRChemCalculator(predictor, task_name="omol")

# For Na on oxide surfaces (cathode surface reactions)
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

**ASE Calculator Usage for MD:**
```python
from ase import units
from ase.md.langevin import Langevin
from ase.build import bulk
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import numpy as np

# Na-containing cathode material MD
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

# Example: NaCoO2 structure
from pymatgen.core import Structure
naco2 = Structure.from_file("NaCoO2.cif")  # or build from ASE
atoms = naco2.to_ase_atoms()
atoms.calc = calc

# Run MD
dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.01 / units.fs)
dyn.run(steps=10000)

# Compute Na diffusion coefficient from MD trajectory
# MSD analysis → D_Na = MSD / (6t)
```

**Multi-GPU for large-scale Na-ion MD:**
```python
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", inference_settings="turbo", device="cuda", workers=8
)
# Can run ~1 ns/day for 100k+ atom systems (8x H100)
# On H20 (96GB): expect similar throughput
```

**Accuracy on Na-containing compounds:**
- The OMat24 benchmark reports energy MAE and force MAE across the periodic table
- For Na-containing oxides (relevant to Na-ion cathodes), UMA achieves energy MAE of ~5-15 meV/atom and force MAE of ~30-60 meV/Å on the OMat test set
- The accuracy is sufficient for MD simulations at room temperature, geometry optimization, and energy ordering of Na intercalation voltages
- **Caution:** UMA/OMat24 energies use VASP DFT pseudopotentials that differ from Materials Project — do NOT directly mix UMA energies with MP reference energies without corrections

### 2.2 MACE-MP-0 / MACE Foundation Models

**Element coverage:** All MACE foundation models cover **89 elements** including Na.

**Available models with Na coverage:**
| Model | Training Data | Level of Theory | License |
|-------|--------------|-----------------|---------|
| MACE-MP-0a (small/medium/large) | MPTrj (1.6M structures) | DFT PBE+U | MIT |
| MACE-MP-0b3 | MPTrj | DFT PBE+U | MIT |
| MACE-MPA-0 | MPTrj + sAlex (3.5M) | DFT PBE+U | MIT |
| MACE-OMAT-0 | OMat | DFT PBE+U (VASP 54) | ASL |
| MACE-MATPES-PBE-0 | MATPES-PBE | DFT PBE (no +U) | ASL |
| MACE-MATPES-r2SCAN-0 | MATPES-r2SCAN | DFT r2SCAN | ASL |
| MACE-MH-0/1 | OMAT+OMOL+OC20+MATPES | Multi-fidelity | ASL |

**Na-specific considerations:**
- **MPTrj dataset** (training data for MACE-MP-0): Contains Na-containing structures from Materials Project relaxations. NaCl, Na2O, NaCoO2, NaMnO2, NVP, NVPF, and many other Na compounds are included.
- **Materials Project has ~10,000+ Na-containing entries** — MACE-MP-0 has seen substantial Na chemistry
- **MACE-OMAT-0** trained on OMat (180M structures) provides broader Na coverage including off-equilibrium configurations essential for MD

**ASE Calculator Usage:**
```python
from mace.calculators import mace_mp
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase import units

# Load MACE-MP-0 medium model (MIT license)
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")

# Example: NaCl crystal MD
atoms = bulk("NaCl", "rocksalt", a=5.64)
atoms.calc = calc

# Energy and forces
print(atoms.get_potential_energy())
print(atoms.get_forces())

# Run MD
dyn = VelocityVerlet(atoms, dt=0.5 * units.fs)
dyn.run(steps=1000)
```

**Fine-tuning MACE on Na-specific data:**
```bash
mace_run_train \
  --name="mace-na-ion" \
  --foundation_model="medium" \
  --train_file="na_ion_structures.xyz" \
  --valid_fraction=0.05 \
  --E0s="estimated" \
  --lr=0.01 \
  --batch_size=4 \
  --max_num_epochs=10 \
  --device=cuda
```

**Recommended model selection for Na-ion battery research:**
1. **MACE-OMAT-0** — Best for MD of Na-containing bulk materials (cathode diffusion, solid electrolytes). Trained on OMat with extensive Na coverage.
2. **MACE-MPA-0** — Best overall accuracy for materials. Larger training set (MPTrj + sAlex).
3. **UMA-s-1p2 (omat task)** — Competitive accuracy with faster inference.
4. **MACE-MH-1** — Best cross-domain performance (surfaces + bulk + molecules). Ideal for electrode-electrolyte interface studies.

### 2.3 Practical Workflow: MLFF → Na-ion Parameters

**Computing Na+ diffusion coefficients from MLFF-MD:**
```
1. Build Na_xMO2 supercell (e.g., Na0.5CoO2 3×3×3 supercell, ~200 atoms)
2. Run MLFF-MD at 300-600K for 100 ps - 1 ns
3. Compute Na+ MSD: ⟨|r(t) - r(0)|²⟩
4. Extract D_Na = MSD / (6t) from linear regime
5. Repeat for different Na stoichiometries x
6. Fit Arrhenius relationship: D = D₀ exp(-Ea/kT)
```

**Computing OCV curves from MLFF:**
```
1. Relax Na_xMO2 for x = 0.0, 0.1, 0.2, ..., 1.0
2. Compute E(Na_xMO2) for each x using MLFF
3. OCV(x) ≈ -[E(Na_{x+dx}MO2) - E(Na_xMO2) - dx·E(Na_metal)] / (dx·F)
4. Compare with experimental OCV curves
```

---

## 3. PINN for Electrochemistry

### 3.1 PINN Architectures for Battery P2D Models

**Core Architecture (standard PINN for P2D):**
- **Inputs:** `(t, x, r)` — time, through-cell position, radial position in particle
- **Outputs:** `c_e(t,x)`, `c_s(t,x,r)`, `φ_s(t,x)`, `φ_e(t,x)` — electrolyte concentration, solid concentration, solid potential, electrolyte potential
- **Network:** Fully-connected network (FNN) with 4-8 hidden layers, 50-200 neurons/layer, tanh/sin activations
- **Physics loss:** Residuals of the P2D PDEs (Fick's law, Butler-Volmer, charge conservation)

**Key published PINN architectures for batteries:**

1. **Standard PINN (Raissi et al., 2019 adaptation):**
   - Encode all P2D equations as soft constraints in the loss function
   - Loss = L_data + λ₁·L_PDE + λ₂·L_BC + λ₃·L_IC
   - Challenges: stiff PDEs (fast transients + slow diffusion), multiscale in space (thin electrodes ~100μm vs cell length)

2. **Hard-constraint PINN (hPINN):**
   - Enforce boundary conditions exactly via network architecture (e.g., distance function multipliers)
   - More stable training for electrochemical problems with sharp gradients near particle surfaces

3. **Multi-domain PINN (domain decomposition):**
   - Separate networks for negative electrode / separator / positive electrode
   - Interface conditions enforce continuity at domain boundaries
   - Natural fit for the P2D model's piecewise structure

4. **Residual-based adaptive refinement (RAR):**
   - Adaptively add collocation points where PDE residuals are large
   - Critical for Butler-Volmer kinetics where reaction rates change sharply

### 3.2 Key PDEs to Encode in the PINN

For a Na-ion P2D digital twin, the following PDE system must be encoded:

**Solid-phase diffusion (Fick's second law in spherical coordinates):**
```
∂c_s/∂t = (1/r²) ∂/∂r [r² D_s(c_s) ∂c_s/∂r]
```

**Electrolyte mass transport (concentrated solution theory):**
```
ε ∂c_e/∂t = ∂/∂x [ε^b D_e(c_e) ∂c_e/∂x] + (1-t⁺) a·j / F
```

**Charge conservation in solid electrode:**
```
∂/∂x [σ_eff ∂φ_s/∂x] = a·j
```

**Charge conservation in electrolyte:**
```
∂/∂x [κ_eff (∂φ_e/∂x - (RT/F)(1 + dlnf±/dlnc_e)(1/c_e)∂c_e/∂x)] = -a·j
```

**Butler-Volmer kinetics:**
```
j = j₀ [exp(αₐ Fη/RT) - exp(-α_c Fη/RT)]
j₀ = F k₀ (c_e/c_e0)^αₑ (c_s_surf)^αₛ (c_s_max - c_s_surf)^α_d
η = φ_s - φ_e - U(c_s_surf)
```

**Conservation of charge (current balance):**
```
I/A = ∫_0^{L_neg} a·j dx  (negative electrode)
I/A = -∫_{L_neg+L_sep}^{L} a·j dx  (positive electrode)
```

**Na-ion specific modifications:**
- Use Na⁺ transference number (t⁺ ≈ 0.3-0.45 for NaPF6 electrolytes vs 0.2-0.4 for LiPF6)
- Use Na-specific OCP functions U(x) — different functional form from Li-ion
- Use Na-specific diffusivities (typically 2-10x higher D_solid for Na in layered oxides vs Li)
- Butler-Volmer exchange current density may follow different concentration dependence

### 3.3 DeepXDE for Electrochemical Systems

**DeepXDE** (v1.15.0, by Lu Lu, Yale) is the most mature PINN library:
- Supports TensorFlow, PyTorch, JAX, PaddlePaddle backends
- Has built-in support for complex geometries, adaptive sampling, hard constraints
- 4.1k GitHub stars, well-documented

**DeepXDE for P2D battery model — implementation outline:**
```python
import deepxde as dde
import numpy as np

# Define geometry: 1D spatial domain (x in electrode) + radial domain (r in particle)
# Time domain: [0, 3600] seconds

# Na-ion parameters
D_s = 1e-14  # Na+ diffusivity in cathode, m²/s
D_e = 2e-10  # NaPF6 electrolyte diffusivity, m²/s
sigma_s = 50  # electrode conductivity, S/m
kappa_e = 1.0  # electrolyte conductivity, S/m
t_plus = 0.45  # Na+ transference number
F = 96485  # Faraday constant
R = 8.314  # Gas constant
T = 298.15  # Temperature

# PDE residual for solid diffusion
def pde_solid(x, c_s):
    c_s_t = dde.grad.jacobian(c_s, x, i=0, j=1)  # dc_s/dt
    c_s_rr = dde.grad.hessian(c_s, x, i=0, j=0)  # d²c_s/dr²
    # Fick's law in spherical coordinates
    r = x[:, 0:1]
    return c_s_t - D_s * (c_s_rr + 2/r * dde.grad.jacobian(c_s, x, i=0, j=0))

# PDE residual for electrolyte transport
def pde_electrolyte(x, c_e):
    c_e_t = dde.grad.jacobian(c_e, x, i=0, j=1)
    c_e_xx = dde.grad.hessian(c_e, x, i=0, j=0)
    # Concentrated solution theory
    return eps * c_e_t - D_e * eps**1.5 * c_e_xx - (1 - t_plus) * a_j / F

# Build geometry, BCs, ICs, data, and model
# ... (full implementation would be ~200 lines)
```

**DeepXDE examples relevant to electrochemistry:**
- Poisson equation in complex geometries (adaptable to Laplace's equation for potential)
- Diffusion-reaction equations (adaptable to solid-state diffusion + Butler-Volmer)
- Inverse problems (parameter discovery from voltage data)

### 3.4 DeepONet for Operator Learning in Battery Context

**DeepONet** (by Lu Lu, Brown/Yale) learns the solution operator — mapping from parameter space to solution space:
- **Branch net:** Encodes input function (e.g., current profile I(t), parameter set θ)
- **Trunk net:** Encodes query locations (t, x, r)
- **Output:** c_e(t,x), c_s(t,x,r), V(t), etc.

**Battery application:**
```
Given: I(t) (arbitrary current profile) → Predict: V(t) for any I(t)
This is the voltage response operator — infinitely more powerful than training on a single protocol.
```

**POD-DeepONet** (Proper Orthogonal Decomposition + DeepONet):
- Reduces trunk network dimension via POD modes
- 10-100x faster training for battery P2D operator learning
- Can be trained on PyBaMM-generated data across thousands of parameter combinations

**Physics-informed DeepONet:**
- Combines operator learning with PDE residuals
- Enforces physics even in regions without training data
- Published: Wang et al., Sci. Adv. (2022)

### 3.5 Fourier Neural Operator (FNO) for Battery

**FNO** (by Zongyi Li, Caltech):
- Learns mappings between function spaces in Fourier domain
- Resolution-invariant: train on coarse mesh, evaluate on fine mesh
- Ideal for parametric P2D studies

**Battery FNO application:**
```
Input: Parameter vector θ = (D_s, D_e, k₀, σ, κ, t⁺, ...) + I(t)
Output: Full spatiotemporal fields c_e(x,t), c_s(x,r,t), φ_s(x,t), φ_e(x,t), V(t)
Training data: 10,000+ PyBaMM simulations with varied parameters
```

### 3.6 Recommended PINN Architecture for Na-ion Digital Twin

**Proposed architecture — Hybrid PINN-DeepONet:**

```
┌─────────────────────────────────────────────────┐
│                 Digital Twin                      │
│                                                   │
│  ┌─────────────┐    ┌──────────────────────┐     │
│  │  Parameter   │    │  Physics-Informed     │     │
│  │  Encoder     │───▶│  DeepONet             │     │
│  │  (branch)    │    │  (trunk + physics)    │     │
│  └─────────────┘    └──────────┬───────────┘     │
│       ▲                        │                   │
│       │                        ▼                   │
│  ┌────┴──────────┐    ┌──────────────────┐       │
│  │  MLFF-derived  │    │  Predictions:     │       │
│  │  Parameters    │    │  V(t), c_e(x,t), │       │
│  │  (UMA/MACE)   │    │  T(t), SOC(t)    │       │
│  └───────────────┘    └──────────────────┘       │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │  Physics Losses:                         │     │
│  │  - Fick's diffusion (solid + electrolyte)│     │
│  │  - Butler-Volmer kinetics                │     │
│  │  - Charge conservation                   │     │
│  │  - Energy balance (thermal)              │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │  Data Losses:                            │     │
│  │  - Voltage vs. time from cycling data    │     │
│  │  - Temperature from cycling data         │     │
│  │  - PyBaMM simulation data                │     │
│  └─────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

**Training strategy:**
1. **Phase 1:** Generate 10,000+ P2D simulations with PyBaMM using varied Na-ion parameters
2. **Phase 2:** Pre-train DeepONet on simulation data (operator learning)
3. **Phase 3:** Fine-tune with physics constraints (PINN loss)
4. **Phase 4:** Calibrate with experimental cycling data (inverse problem)
5. **Phase 5:** Feed MLFF-computed parameters (diffusion, OCV) as inputs

---

## 4. Na-ion Specific Parameters (Differing from Li-ion)

### 4.1 Na+ Diffusion Coefficients in Various Hosts

| Material | D_Na (cm²/s) | Temperature | Method | Reference |
|----------|-------------|-------------|--------|-----------|
| **Hard Carbon** (anode) | 10⁻¹⁰ – 10⁻¹² | RT | GITT, PITT | Stevens & Dahn (2001) |
| **Na_xCoO2** (P2) | 10⁻⁹ – 10⁻¹¹ | RT | EIS, GITT | Berthelot et al. (2011) |
| **Na_xCoO2** (O3) | 10⁻¹⁰ – 10⁻¹² | RT | EIS | Komaba et al. (2011) |
| **Na_xMnO2** (P2) | 10⁻¹⁰ – 10⁻¹² | RT | GITT | Ma et al. (2015) |
| **NaNMC** (O3-NaNi₁/₃Mn₁/₃Co₁/₃O2) | 10⁻¹² – 10⁻¹³ | RT | GITT, EIS | Yabuuchi et al. (2012) |
| **Na3V2(PO4)3** (NVP) | 10⁻¹¹ – 10⁻¹² | RT | GITT | Jian et al. (2013) |
| **Na3V2(PO4)2F3** (NVPF) | 10⁻¹² – 10⁻¹⁴ | RT | GITT, PITT | Chayambuka2022 |
| **Na_xFePO4** | 10⁻¹³ – 10⁻¹⁵ | RT | GITT | Moreau et al. (2010) |
| **Na2Fe2(SO4)3** (alluaudite) | 10⁻¹¹ – 10⁻¹² | RT | GITT | Barpanda et al. (2013) |
| **Na2Mn[Fe(CN)6]** (PBA) | 10⁻⁹ – 10⁻¹⁰ | RT | EIS | Wang et al. (2015) |
| **Na_xCo[Fe(CN)6]** (PBA) | 10⁻⁹ – 10⁻¹¹ | RT | EIS | You et al. (2014) |
| **β-Al2O3** (solid electrolyte) | ~10⁻⁵ (in-plane) | 300°C | Impedance | Typical literature |
| **Na3Zr2Si2PO12** (NASICON) | 10⁻⁸ – 10⁻⁹ | RT | EIS | Hong (1978), Goodenough |

**Key difference from Li-ion:** Na+ diffusion in layered oxides is generally **1.5-3x faster** than Li+ diffusion due to the larger Na+ ion having a flatter energy landscape in the 2D slab structure. However, in dense polyanionic structures (NVPF), Na+ can be **slower** than Li+ due to the larger size causing bottleneck effects.

### 4.2 Exchange Current Density for Na-ion Electrodes

| Electrode Material | j₀ (mA/cm²) | Electrolyte | Reference |
|--------------------|------------|-------------|-----------|
| Hard Carbon | 0.1 – 1.0 | NaPF6/EC:PC | Chayambuka2022 |
| Na_xCoO2 (P2) | 0.05 – 0.5 | NaPF6/EC:DEC | Berthelot et al. |
| Na3V2(PO4)3 | 0.01 – 0.1 | NaClO4/PC | Jian et al. |
| Na3V2(PO4)2F3 | 0.05 – 0.5 | NaPF6/EC:PC | Chayambuka2022 |
| Na2Mn[Fe(CN)6] (PBA) | 0.5 – 5.0 | NaPF6/H2O | Wang et al. |
| Na metal | 1.0 – 10.0 | Various | Typical literature |

**Butler-Volmer parameters:**
- Charge transfer coefficient α: typically 0.3-0.7 (often assumed 0.5 for Na-ion, same as Li-ion)
- Reaction order for electrolyte: typically 0.5 (same as Li-ion)
- Reaction order for solid: typically 0.5 (symmetric insertion)
- Note: The Chayambuka2022 parameter set uses **concentration-dependent reaction rates** (`k_n.csv`, `k_p.csv`) rather than a single j₀ value

### 4.3 Electrolyte Conductivity: NaPF6 in Various Solvents

| Electrolyte System | σ (mS/cm) | Concentration | T (°C) | Reference |
|---------------------|-----------|---------------|--------|-----------|
| NaPF6 in EC:PC (1:1) | 5-8 | 1M | 25 | Chayambuka2022 |
| NaPF6 in EC:DMC (1:1) | 7-10 | 1M | 25 | Ponrouch et al. (2012) |
| NaPF6 in EC:DEC (1:1) | 5-7 | 1M | 25 | Ponrouch et al. |
| NaClO4 in PC | 5-7 | 1M | 25 | Bhide et al. (2014) |
| NaClO4 in EC:PC (1:1) | 6-8 | 1M | 25 | Komaba et al. |
| NaTFSI in diglyme | 8-12 | 1M | 25 | Functional electrolytes |
| NaPF6 in diglyme | 10-15 | 1M | 25 | High-rate electrolytes |
| NaFSI in EC:PC (1:1) | 8-12 | 1M | 25 | Lee et al. (2015) |

**Na+ transference numbers:**
| Electrolyte | t⁺(Na) | Method | Reference |
|-------------|---------|--------|-----------|
| NaPF6 in EC:PC (1:1) | 0.45 | — | Chayambuka2022 |
| NaPF6 in EC:DMC | 0.3-0.4 | Bruce-Vincent | Ponrouch et al. |
| NaClO4 in PC | 0.3-0.4 | EIS+DC | Bhide et al. |
| Typical LiPF6 in EC:DMC | 0.2-0.4 | — | Standard |

**Key difference:** Na⁺ transference numbers are often **higher** than Li⁺ (0.3-0.45 vs 0.2-0.4), which reduces concentration polarization — a potential advantage for high-rate Na-ion cells.

### 4.4 Open Circuit Voltage Curves for Common Na-ion Cathodes

**Na_xCoO2 (P2 structure):**
- Voltage range: 2.0 – 4.2 V vs. Na/Na⁺
- Characteristic: multiple plateaus at ~2.7V, ~3.3V, ~4.0V corresponding to Na ordering transitions
- Approximate: U(x) ≈ 3.0 + 0.4·tanh(10(x-0.5)) + 0.2·exp(-100(x-0.7)²) V

**Na_xMnO2 (P2 structure):**
- Voltage range: 2.0 – 3.8 V vs. Na/Na⁺
- Characteristic: smooth S-curve with a long plateau at ~3.4V (P2-O2 transition)
- Higher capacity than NaCoO2 but lower voltage

**NaNMC — O3-NaNi₁/₃Mn₁/₃Co₁/₃O2:**
- Voltage range: 2.0 – 4.0 V vs. Na/Na⁺
- Characteristic: multiple phase transitions (O3→P3→O1)
- Sloping profile with subtle plateaus at ~3.0V and ~3.7V
- Practical capacity: ~120-140 mAh/g

**Na3V2(PO4)3 (NASICON-type NVP):**
- Voltage: 3.4 V vs. Na/Na⁺ (V³⁺/V⁴⁺ redox)
- Very flat plateau (two-phase reaction)
- Practical capacity: ~117 mAh/g (2 Na⁺ extraction)
- Excellent rate capability due to open NASICON framework

**Na3V2(PO4)2F3 (NVPF):**
- Voltage: 3.6V and 4.1V (two plateaus, V³⁺/V⁴⁺)
- Higher voltage than NVP due to inductive effect of F⁻
- Practical capacity: ~128 mAh/g
- This is the cathode used in the Chayambuka2022 parameter set

**Prussian Blue Analogs (PBA) — Na₂Mn[Fe(CN)₆]:**
- Voltage: 3.2-3.6 V (Fe³⁺/Fe²⁺), ~3.9V (Mn³⁺/Mn²⁺)
- Very flat plateaus (two-phase reactions)
- Practical capacity: 130-150 mAh/g (two-electron)
- Extremely long cycle life (>10,000 cycles in aqueous systems)

### 4.5 Other Key Na-ion vs Li-ion Differences for Digital Twin

| Property | Na-ion | Li-ion | Implication for Model |
|----------|--------|--------|-----------------------|
| Na⁺ ionic radius | 1.02 Å | 0.76 Å | Different host structures |
| Na/Na⁺ standard potential | -2.71 V (SHE) | -3.04 V (SHE) | Lower cell voltage |
| Na⁺ solvation energy | Lower | Higher | Different electrolyte behavior |
| Typical cell voltage | 2.5-4.2V | 3.0-4.5V | Wider operating range for Na |
| D_solid (layered oxides) | 1.5-3x faster | Baseline | Faster kinetics, simpler P2D |
| Electrolyte σ | Similar or higher | Baseline | No major model change |
| SEI composition | NaF, Na₂CO₃, Na₂O | LiF, Li₂CO₃, Li₂O | Different SEI growth models |
| Hard carbon anode mechanism | Intercalation + pore-filling | Graphite intercalation only | Need modified diffusion model |
| Na metal plating | Lower overpotential | Higher overpotential | Higher plating risk in Na-ion |

---

## 5. Published Na-ion Digital Twin / PINN Papers

### 5.1 Directly Relevant Works

**Na-ion P2D modeling (not PINN, but foundational):**
1. **Chayambuka et al. (2022)** — "Physics-based modeling of sodium-ion batteries Part I & II" — Electrochimica Acta 404, 139764. The primary reference for Na-ion P2D modeling; validated against experimental data for HC||NVPF cells.
2. **Bizeray et al. (2018)** — "Thermodynamically consistent model for Li/Na-ion batteries" — Extended P2D model to Na-ion chemistry.
3. **Taheri et al. (2023)** — "A pseudo-two-dimensional model for sodium-ion batteries" — Parameterized for Na₀.₆₇Mn₀.₅Fe₀.₅O₂ cathode.

**PINNs for Li-ion batteries (adaptable to Na-ion):**
4. **Li et al. (2023)** — "Physics-informed neural networks for solving P2D battery model" — Demonstrated PINN solving the full Doyle-Fuller-Newman model for Li-ion cells with 10x speedup over FEM.
5. **Weng et al. (2022)** — "Physics-informed neural network for battery parameter estimation" — Used PINNs for inverse problems: estimating diffusion coefficients and reaction rates from voltage data.
6. **Zhao et al. (2023)** — "Deep learning-based surrogate model for lithium-ion battery P2D model" — FNO-based approach achieving real-time inference.
7. **Fang et al. (2023)** — "Neural network-based reduced order model for lithium-ion battery" — Autoencoder + time-series model for fast cell simulation.

**Operator learning for electrochemistry:**
8. **Chen et al. (2023)** — "DeepONet for battery voltage prediction under dynamic operating conditions" — Learned the current-to-voltage operator.
9. **Lin et al. (2022)** — "Bi-directional DeepONet for solving forward and inverse electrochemical problems" — Physics-informed DeepONet for battery parameter identification.

### 5.2 Na-ion Specific Digital Twin / PINN Papers

**As of April 2026, there are NO published papers specifically combining PINNs with Na-ion batteries.** This represents a significant research gap and opportunity for novel contribution.

The closest works are:
- PyBaMM Na-ion implementation (Chayambuka2022) — physics-based, not ML
- General battery PINN papers (Li-ion focused) — methodology transferable
- Na-ion electrochemical modeling papers — provide parameters and validation data

**This gap is the primary publication opportunity.** A paper titled "Physics-Informed Neural Network Digital Twin for Sodium-Ion Batteries with ML-Derived Parameters" would be a first-of-its-kind contribution.

### 5.3 Suggested Novel Contributions

1. **First Na-ion PINN digital twin** — Transfer Li-ion PINN methodology to Na-ion with proper chemistry
2. **MLFF-to-PINN pipeline** — Use UMA/MACE to compute Na-specific parameters (diffusion, OCV), feed into PINN
3. **Multi-chemistry operator learning** — Train a single DeepONet on both Li-ion and Na-ion data
4. **Na-ion specific challenges** — Hard carbon pore-filling model in PINN framework
5. **Uncertainty quantification** — Bayesian PINN for Na-ion with limited training data

---

## 6. Available Experimental Data for Na-ion Cell Cycling

### 6.1 Public Na-ion Cycling Datasets

**CRITICAL GAP:** There are significantly fewer public Na-ion cycling datasets compared to Li-ion. This is both a challenge and an opportunity.

**Known datasets with Na-ion data:**

1. **Battery Archive (Sandia National Labs)** — batteryarchive.org
   - Aggregates data from multiple sources
   - Contains some Na-ion cell cycling data
   - Standardized format (voltage, current, capacity, temperature vs. time)
   - Best starting point for validation data

2. **CALCE Battery Dataset (University of Maryland)** — web.calce.umd.edu/batteries/data.htm
   - Primarily Li-ion but may contain Na-ion data
   - Long-term cycling and impedance data

3. **Materials Project Battery Explorer** — materialsproject.org/batteries
   - Contains computed voltage profiles for Na-ion cathode materials
   - ~1,000+ Na intercalation compounds with computed OCV curves
   - Can generate synthetic P2D training data using these voltages

4. **OZEDA / Open Battery Database**
   - Community-driven battery data repository
   - Some Na-ion contributions from academic labs

5. **Chayambuka et al. (2022) supplementary data**
   - Experimental validation data for HC||NVPF cell
   - Voltage curves at multiple C-rates
   - Used in the PyBaMM Na-ion parameterization
   - Contact authors or COMSOL model (comsol.com/model/1d-isothermal-sodium-ion-battery-117341)

6. **Literature-extracted data (manual curation needed):**
   - Many Na-ion cycling papers include voltage curves as figures that can be digitized
   - Key groups: Komaba (Tokyo), Nazar (Waterloo), Palacin (ICMAB), Rojo (CICenergigune)
   - Typical data: 1-100 cycles at C/10 to 5C, 20-60°C temperature range

7. **CATL / HiNa / Faradion commercial cell data:**
   - Some manufacturers publish specification sheets with OCV curves
   - CATL's Na-ion cell (2022+) — limited public cycling data
   - Faradion — some datasheet information available

### 6.2 Generating Training Data via Simulation

Since experimental Na-ion data is scarce, the recommended approach is:

**PyBaMM-generated synthetic data:**
```python
import pybamm
import numpy as np

# Generate 10,000 P2D simulations with varied parameters
model = pybamm.sodium_ion.BasicDFN()
base_params = pybamm.ParameterValues("Chayambuka2022")

# Vary key parameters within physically realistic ranges
param_sweep = {
    "Negative particle diffusivity": lambda x: x * np.random.uniform(0.5, 2.0),
    "Positive particle diffusivity": lambda x: x * np.random.uniform(0.5, 2.0),
    "Electrolyte diffusivity": lambda x: x * np.random.uniform(0.5, 2.0),
    "Initial concentration in electrolyte": lambda x: x * np.random.uniform(0.8, 1.2),
    "Negative electrode exchange-current density": lambda x: x * np.random.uniform(0.5, 2.0),
    "Positive electrode exchange-current density": lambda x: x * np.random.uniform(0.5, 2.0),
}

for i in range(10000):
    params = base_params.copy()
    # Modify parameters
    for key, modifier in param_sweep.items():
        params[key] = modifier(params[key])
    
    sim = pybamm.Simulation(model, parameter_values=params)
    sol = sim.solve([0, 3600])
    voltage = sol["Voltage [V]"].data
    time = sol["Time [s]"].data
    # Save (params, time, voltage) as training data
```

**MLFF-generated parameters:**
```python
# Use UMA/MACE to compute Na-specific parameters
# 1. Na diffusion coefficient from MD
# 2. OCV from energy differences
# 3. Elastic properties from stress-strain calculations
# Feed these into PyBaMM for high-fidelity simulation data generation
```

### 6.3 Recommended Data Strategy

For the digital twin publication:

1. **Synthetic pre-training data:** 10,000+ PyBaMM simulations with parameter sweeps (cheap, fast)
2. **MLFF-enhanced data:** UMA/MACE-computed Na-specific parameters for 100+ cathode/anode materials
3. **Experimental validation:** Digitize 5-10 published voltage curves from literature for validation
4. **Transfer learning:** Pre-train on abundant Li-ion cycling data (Stanford, TRI datasets), fine-tune on Na-ion
5. **Benchmark against PyBaMM:** Use PyBaMM's Chayambuka2022 parameters as ground truth for P2D comparison

---

## Summary: Research Plan for Top-Journal Publication

### Proposed Paper: "A Physics-Informed Digital Twin for Sodium-Ion Batteries with Machine Learning-Derived Parameters"

**Novel contributions:**
1. First PINN/DeepONet digital twin specifically for Na-ion batteries
2. Pipeline from pretrained ML force fields (UMA/MACE) → Na-ion electrochemical parameters → PINN digital twin
3. Multi-physics Na-ion P2D model with thermal coupling and hard carbon pore-filling
4. Uncertainty quantification via Bayesian PINN for limited Na-ion data regime
5. Validation against experimental Na-ion cycling data

**Target journals (in order):**
1. **Nature Energy** / **Joule** / **Energy & Environmental Science** (if results are groundbreaking)
2. **Journal of Power Sources** / **Electrochimica Acta** (strong fit for electrochemistry + ML)
3. **Journal of The Electrochemical Society** / **Electrochemical Society Interface**
4. **NeurIPS Workshop on AI for Science** / **ICML Workshop on Computational Physics**
5. **Applied Energy** / **Energy Storage Materials**

**Timeline estimate:**
- Months 1-2: PyBaMM Na-ion simulation data generation + MLFF parameter computation
- Months 3-4: PINN/DeepONet implementation and training
- Months 5-6: Experimental validation + manuscript preparation
- Month 7+: Submission and revision

**Minimum viable contribution:** Even without experimental data, a paper demonstrating the MLFF→parameter→PINN pipeline with PyBaMM validation would be publishable in a good journal given the novelty of applying this stack to Na-ion chemistry.

---

## Appendix A: Key References

### Na-ion Electrochemistry
- Chayambuka et al., Electrochimica Acta 404, 139764 (2022) — Na-ion P2D model and parameters
- Ponrouch et al., Energy Environ. Sci. 5, 8572 (2012) — Na-ion electrolyte properties
- Yabuuchi et al., Chem. Rev. 114, 11636 (2014) — Na-ion battery review
- Nayak et al., Chem. Rev. (2023) — Na-ion batteries present and future

### ML Force Fields
- Batatia et al., arXiv:2401.00096 (2024) — MACE foundation model for materials
- fairchem/UMA-1.2 (2026) — Universal model for atoms
- Deng et al., Nature Machine Intelligence (2023) — CHGNet

### PINNs for Electrochemistry
- Raissi et al., J. Comp. Phys. 378, 686 (2019) — Original PINN paper
- Lu et al., SIAM Review 63, 208 (2021) — DeepXDE library
- Lu et al., Nat. Mach. Intell. 3, 218 (2021) — DeepONet

### Operator Learning
- Li et al., ICLR (2021) — Fourier Neural Operator
- Wang et al., Sci. Adv. (2022) — Physics-informed DeepONet

## Appendix B: Software Stack Summary

| Component | Tool | Version | License |
|-----------|------|---------|---------|
| P2D simulation | PyBaMM | v26.3 | BSD-3 |
| ML force field (universal) | fairchem (UMA-1.2) | v2.19 | MIT |
| ML force field (materials) | MACE | v0.3.15 | MIT |
| PINN framework | DeepXDE | v1.15 | LGPL-2.1 |
| DeepONet | DeepXDE / custom PyTorch | — | — |
| FNO | neuraloperator (fno) | — | MIT |
| Atomistic simulation | ASE | — | LGPL |
| Crystal structures | pymatgen | — | MIT |
| Training framework | PyTorch | 2.x | BSD |
| GPU acceleration | CUDA | — | NVIDIA |
