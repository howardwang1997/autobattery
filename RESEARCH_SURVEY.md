# AI/ML Research Directions for Li-ion & Na-ion Battery Research
## Feasibility Analysis for NVIDIA H20 GPU Cluster (4-12x H20, 96GB HBM3 each)

**Date:** April 2026  
**Hardware:** NVIDIA H20 GPUs (96GB HBM3), 4-12 cards (384GB-1152GB total GPU memory)

---

## Table of Contents
1. [Battery Material Discovery using ML/AI](#1-battery-material-discovery-using-mlai)
2. [Battery Electrolyte Design](#2-battery-electrolyte-design)
3. [Battery Lifetime/Degradation Prediction](#3-battery-lifetimedegradation-prediction)
4. [Battery Digital Twins / Physics-Informed Neural Networks](#4-battery-digital-twins--physics-informed-neural-networks)
5. [Domain-Specific LLMs for Battery Science](#5-domain-specific-llms-for-battery-science)
6. [Sodium-Ion Battery Specific Research](#6-sodium-ion-battery-specific-research)
7. [Open-Source Battery AI Projects and Datasets](#7-open-source-battery-ai-projects-and-datasets)
8. [Recommendations & Priority Matrix](#8-recommendations--priority-matrix)

---

## 1. Battery Material Discovery using ML/AI

### State of the Art (2024-2026)

**Graph Neural Networks (GNNs) for Crystal Structure Prediction:**
- **FAIR Chemistry / UMA Model (Meta, 2025-2026):** The Universal Model for Atoms (UMA) from Meta's FAIR Chemistry team is the current SOTA. UMA-1.2 (March 2026) supports multiple chemistry domains (catalysis, bulk materials, molecules, MOFs, molecular crystals) with a single model. The small variant (uma-s-1p2) has 6.6M active / 290M total params; the medium (uma-m-1p1) has 50M active / 1.4B total params. Trained on OMat24 (180M+ DFT calculations), OMol25, OCx datasets. Supports energy/force prediction, molecular dynamics, geometry optimization, spin gap calculations.
  - **GitHub:** `facebookresearch/fairchem` (2.1k stars, MIT license)
  - **Key capability:** Direct ASE calculator integration for battery material screening

- **DeePMD-kit v3 (DeepModeling, 2025):** Multi-backend framework (TF, PyTorch, JAX, Paddle) for machine learning potentials. Implements Deep Potential series (DPA-2, DPA-3) models. Used extensively for molecular dynamics of battery electrode/electrolyte interfaces.
  - **GitHub:** `deepmodeling/deepmd-kit` (1.9k stars, LGPL-3.0)
  - **Hardware:** Scales from single GPU to multi-node HPC; 4x H20 would be excellent for training medium-sized models

- **MACE / NequIP / Allegro:** Equivariant GNN architectures for interatomic potentials. MACE-MP-0 (2024) trained on Materials Project provides a universal foundation model for materials. These have been applied to Li-ion diffusion pathways in solid electrolytes.

- **CHGNet (Materials Project, 2024):** Universal graph neural network potential for inorganic materials, trained on Materials Project data. Directly applicable to battery electrode material screening. Handles magnetic moments, making it suitable for transition metal oxide cathodes (NMC, LFP, etc.).

- **Crystal Diffusion Variational Autoencoder (CDVAE) / CrystalLLM (2024-2025):** Generative models for novel crystal structure generation. CrystalLLM uses LLM tokenization of crystal structures for inverse design of battery materials.

**Electrode Material Screening Pipelines:**
- High-throughput screening of cathode materials (LiCoO2 variants, NMC, NCA, LFP, LNMO) using GNN-predicted formation energies, voltage profiles, and stability metrics from Materials Project / OQMD / AFLOW databases.
- Anode material discovery: silicon anodes, hard carbon (for Na-ion), lithium metal, graphite intercalation. ML models predict specific capacity, volume expansion, diffusion barriers.
- Solid electrolyte screening: Li-ion conductors (LLZO, LiPON, LISICON types) predicted via ML-guided ionic conductivity estimation.

### Hardware Requirements
| Task | Typical GPU Memory | Training Time | H20 Feasibility |
|------|-------------------|---------------|-----------------|
| GNN inference (UMA-s) | ~4-8 GB | Minutes | Excellent (single GPU) |
| GNN training (medium model) | ~20-40 GB | Hours-days | Excellent (1-2 H20) |
| Large equivariant GNN | ~40-80 GB | Days | Very Good (2-4 H20) |
| Generative crystal design | ~16-32 GB | Hours | Excellent (1 H20) |
| High-throughput screening (100k+ materials) | ~8 GB inference | Hours | Excellent (multi-GPU parallel) |

### Feasibility with 4-12x H20: **EXCELLENT**
Material discovery is the most hardware-compatible direction. Even 4x H20 provides 384GB total VRAM, enough for:
- Training large equivariant GNNs (MACE, NequIP) with full models
- Running UMA-medium for high-accuracy predictions
- Parallel high-throughput screening across hundreds of thousands of candidate materials
- Generative model training for crystal structure design

### Practical Impact / Novelty: **VERY HIGH**
- Direct pathway to discovering new cathode/anode/solid electrolyte materials
- Can publish in Nature/Science family journals with compelling results
- Industry relevance: battery manufacturers (CATL, BYD, Samsung SDI) actively seek ML-discovered materials
- Na-ion specific: enormous white space for screening Na-based compounds

### Key Datasets
- **Materials Project** (~500k inorganic crystals with DFT properties): materialsproject.org
- **OMat24** (180M+ DFT calculations, Meta): huggingface.co/datasets/facebook/OMAT24
- **OQMD** (Open Quantum Materials Database): oqmd.org
- **AFLOW** (Automatic FLOW for Materials Discovery): aflow.org
- **NIST JARVIS**: jarvis.nist.gov
- **MPContribs** (battery-specific data from Materials Project)

### Key Codebases
- `facebookresearch/fairchem` — UMA model, OCP models, universal MLIPs
- `deepmodeling/deepmd-kit` — DeePMD/DPA models, LAMMPS integration
- `ACEsuit/mace` — MACE equivariant interatomic potential
- `mir-group/nequip` — NequIP/Allegro equivariant GNN
- `materialsproject/pymatgen` — Materials analysis toolkit (1.9k stars)
- `materialsproject/chgnet` — Universal GNN potential
- `lingtengqiu/DiffCSP` — Crystal structure prediction via diffusion
- `txie-93/cdvae` — Crystal diffusion variational autoencoder

---

## 2. Battery Electrolyte Design

### State of the Art (2024-2026)

**ML-Accelerated Molecular Simulation:**
- **Machine Learning Force Fields (MLFFs):** Training neural network potentials (DeePMD, MACE, GAP) on DFT data for liquid electrolyte MD simulations. 100-1000x speedup over DFT-MD while maintaining chemical accuracy.
  - Applications: LiPF6 in EC/DMC electrolytes, solid polymer electrolytes (PEO-based), ionic liquid electrolytes, gel polymer electrolytes
  - Key property predictions: ionic conductivity, Li+ transference number, viscosity, diffusion coefficients, solvation structure

- **SchNet / PaiNN / Equiformer for Molecular Property Prediction:** GNNs predicting quantum chemical properties (HOMO/LUMO, dipole moment, polarizability, partial charges) of electrolyte molecules. Used for screening electrolyte additives and solvents.
  - **QM9 / PCQ / OMol25 datasets** for training molecular GNNs

- **Electrolyte Formulation Optimization:**
  - Bayesian optimization + ML surrogate models for multi-component electrolyte formulation
  - Multi-objective optimization: conductivity vs. stability window vs. viscosity vs. cost
  - Transfer learning from Li-ion to Na-ion electrolyte systems

- **Redox Potential Prediction:** ML models predicting oxidation/reduction stability of electrolyte components. Critical for high-voltage cathode compatibility.

- **SEI/CEI Layer Modeling:** ML-driven simulation of solid electrolyte interphase (SEI) formation and growth. This is one of the most challenging and impactful problems:
  - Atomistic simulation of SEI decomposition products (LiF, Li2CO3, Li2O, organic species)
  - Kinetic Monte Carlo + ML for SEI growth modeling
  - Data-driven SEI property prediction (ionic conductivity, mechanical properties)

### Hardware Requirements
| Task | Typical GPU Memory | Training Time | H20 Feasibility |
|------|-------------------|---------------|-----------------|
| MLFF training (liquid electrolyte, 100-500 atoms) | ~8-32 GB | Hours-days | Excellent |
| MLFF training (large system, 1000+ atoms) | ~32-64 GB | Days | Very Good (2-4 H20) |
| MD simulation with MLFF | ~4-16 GB | Hours-days | Excellent |
| Molecular GNN training | ~4-16 GB | Hours | Excellent |
| Bayesian optimization | ~2-8 GB | Minutes-hours | Excellent |
| SEI simulation (multi-scale) | ~16-64 GB | Days-weeks | Good (4-8 H20) |

### Feasibility with 4-12x H20: **EXCELLENT**
- Most electrolyte ML tasks are model-size-limited rather than data-limited
- 96GB per H20 is generous for molecular simulations
- Multi-GPU parallel MD runs are well-supported (LAMMPS + DeePMD-kit)

### Practical Impact / Novelty: **VERY HIGH**
- Electrolyte optimization is a bottleneck in battery development
- New electrolyte formulations can enable higher energy density, faster charging
- Na-ion electrolyte optimization is underexplored — high novelty potential
- SEI modeling with ML is a frontier topic with high publication impact

### Key Datasets
- **OMol25** (Meta): 100M+ molecular DFT calculations
- **SPICE** (Open Force Field): Small-molecule DFT dataset for force field training
- **ANI-1x / ANI-2x**: DFT datasets for organic molecules (relevant to electrolyte solvents)
- **QH9**: Quantum chemistry dataset for organic molecules
- **Electrolyte Genome (Battery Data Genome)**: From Sandia/Argonne national labs

### Key Codebases
- `deepmodeling/deepmd-kit` — ML force fields for electrolyte MD
- `ACEsuit/mace` — MACE for molecular force fields
- `mir-group/nequip` — NequIP for molecular dynamics
- `openforcefield/openff-toolkit` — Force field development toolkit
- `rocel2718/ML-for-electrolyte-design` — ML electrolyte design examples
- `MihailBogojeski/pyg-molecule` — Molecular GNN framework

---

## 3. Battery Lifetime/Degradation Prediction

### State of the Art (2024-2026)

**Key Datasets:**
- **Stanford/SLAC Battery Dataset (Severson et al., 2019; Attia et al., 2020):**
  - 124 commercial LFP cells cycled under fast-charging conditions
  - 1000+ cycles per cell with full electrochemical impedance spectroscopy (EIS) data
  - Ground truth cycle life (150-2300 cycles)
  - **Benchmark task:** Predict cycle life from first 100 cycles
  - **Published in Nature (2019), Nature Energy (2020)**

- **Toyota Research Institute (TRI) Battery Dataset:**
  - Thousands of cells with varied chemistries and cycling protocols
  - Features: voltage curves, temperature, C-rate, capacity fade
  - One of the largest publicly available battery degradation datasets
  - Available at data.matr.io

- **CALCE Battery Dataset (University of Maryland):**
  - Long-term cycling data for Li-ion cells (LCO, NMC, LFP chemistries)
  - Includes temperature stress, C-rate variation studies
  - web.calce.umd.edu/batteries/data.htm

- **Battery Archive (Sandia National Labs):**
  - batteryarchive.org — aggregated dataset from multiple sources
  - Standardized data format for cycle life, impedance, degradation

- **NASA Battery Dataset:**
  - NASA Ames Prognostics Center of Excellence
  - Li-ion cells under various load profiles and temperatures
  - Traditional benchmark for prognostics/health management

- **HNEI Dataset (Hawaiian Natural Energy Institute):**
  - Long-term calendar and cycle aging data
  - Multiple cell formats (18650, pouch) and chemistries

**Model Architectures (2024-2026 SOTA):**
- **Transformer-based models:** Time-series transformers for voltage curve prediction and degradation forecasting. Outperform LSTMs on large-scale datasets.
- **Gaussian Process Regression (GPR):** Physics-informed GPR for uncertainty-aware SOH prediction. Still competitive with uncertainty quantification advantage.
- **Physics-constrained neural networks:** Incorporating degradation models (SEI growth, lithium plating, particle cracking) as constraints.
- **Meta-learning / Few-shot approaches:** Predicting lifetime for new cell chemistries with limited data by transferring knowledge from existing datasets.
- **Multi-task learning:** Jointly predicting SOH, remaining useful life (RUL), and capacity fade using shared representations.
- **Variational autoencoders (VAEs):** Learning latent representations of degradation trajectories for clustering and prediction.

**Recent Key Papers (2024-2026):**
- "Foundation models for battery degradation prediction" — using pre-trained time-series models fine-tuned on battery data
- "Graph-based battery degradation modeling" — representing cell cycling data as graphs
- "Bayesian deep learning for battery lifetime prediction with uncertainty quantification"
- "Transfer learning across battery chemistries for degradation prediction"

### Hardware Requirements
| Task | Typical GPU Memory | Training Time | H20 Feasibility |
|------|-------------------|---------------|-----------------|
| LSTM/GRU training (tabular data) | ~2-8 GB | Minutes | Trivial |
| Transformer training (time-series) | ~4-16 GB | Hours | Excellent |
| Physics-constrained NN | ~4-16 GB | Hours | Excellent |
| VAE/GAN for degradation simulation | ~8-32 GB | Hours-days | Excellent |
| Foundation model fine-tuning | ~16-64 GB | Hours-days | Very Good |

### Feasibility with 4-12x H20: **EXCELLENT (Overpowered)**
- Battery degradation prediction is primarily a data-limited, not compute-limited problem
- Even a single H20 (96GB) is massive overkill for most degradation prediction tasks
- The bottleneck is data availability and quality, not compute

### Practical Impact / Novelty: **HIGH (but crowded)**
- Very active field with many published approaches
- Novel contributions needed: better transfer learning across chemistries, foundation models for battery data
- Industry adoption is rapid (Tesla, BYD, Samsung SDI all use ML for BMS)
- Highest commercial value direction for EV companies

### Key Codebases
- `dsr-18/long-live-the-battery` — Stanford battery life prediction (94 stars)
- `DariusRoman/Machine-learning-pipeline-for-battery-state-of-health-estimation` (75 stars)
- `sautee/battery-state-of-charge-estimation` — SOC estimation with Streamlit (86 stars)
- `wanbin-song/BatteryMachineLearning` — MATLAB-based capacity estimation (81 stars)
- `petermbenjamin/battery-data-toolkit` — Battery data processing tools

---

## 4. Battery Digital Twins / Physics-Informed Neural Networks (PINNs)

### State of the Art (2024-2026)

**Physics-Informed Neural Networks for Electrochemistry:**
- **PINNs for PDE-based battery models:** Replacing finite element / finite volume solvers for the pseudo-2D (P2D) Newman model with neural networks
  - Input: time, spatial position → Output: concentration, potential, current density
  - Physics constraints: Butler-Volmer kinetics, Fick's diffusion law, Ohm's law
  - Key advantage: 10-100x speedup over traditional PDE solvers, enabling real-time digital twin

- **Neural ODE / SINDy approaches:** Discovering governing equations from battery cycling data
  - Learning degradation ODEs from capacity fade data
  - Identifying electrochemical parameters (diffusion coefficients, reaction rates) from EIS/voltage data

- **Reduced-Order Models (ROMs) with ML:**
  - Autoencoder-based dimensionality reduction of high-fidelity battery models
  - Proper Orthogonal Decomposition (POD) + neural network surrogates
  - Real-time capable models for BMS integration

- **Multi-Scale Digital Twins:**
  - Atomistic → electrode → cell → pack → system hierarchy
  - ML bridges between scales (e.g., ML-predicted diffusion coefficients from atomistic sim feed into cell model)
  - Bayesian calibration for model parameters from manufacturing data

- **Electrochemical Impedance Spectroscopy (EIS) Analysis:**
  - Deep learning for equivalent circuit model parameter extraction from EIS data
  - CNN/transformer-based classification of EIS spectra for degradation mode identification

**Recent Developments (2024-2026):**
- **Differentiable physics simulators:** JAX-based battery model implementations enabling gradient-based optimization of electrochemical parameters
- **Operator learning (DeepONet, FNO):** Learning the solution operator of battery PDEs for instant inference across parameter spaces
- **Hybrid PINN-ML models:** Combining first-principles PDE structure with data-driven corrections
- **Uncertainty quantification:** Bayesian PINNs for confidence-bounded predictions in digital twin applications

### Hardware Requirements
| Task | Typical GPU Memory | Training Time | H20 Feasibility |
|------|-------------------|---------------|-----------------|
| PINN training (1D P2D model) | ~2-8 GB | Minutes-hours | Trivial |
| PINN training (3D cell model) | ~8-32 GB | Hours | Excellent |
| DeepONet / FNO training | ~8-32 GB | Hours-days | Excellent |
| Neural ODE training | ~4-16 GB | Hours | Excellent |
| Multi-scale digital twin | ~16-64 GB | Days | Very Good |
| Bayesian PINN (MCMC sampling) | ~4-16 GB | Hours-days | Excellent |

### Feasibility with 4-12x H20: **EXCELLENT**
- PINNs and operator learning models are generally small-to-medium in size
- 96GB H20 is more than sufficient for any battery PINN application
- Multi-GPU setup enables parameter sweeps and ensemble models for uncertainty quantification

### Practical Impact / Novelty: **HIGH**
- Digital twin is a "holy grail" for battery management systems
- Strong industry demand (EV OEMs, grid storage operators)
- Academic novelty in combining PINNs with battery electrochemistry is still high
- Na-ion specific: P2D model parameterization for Na-ion cells is largely unexplored

### Key Codebases
- `maziarraissi/PINNs` — Original PINN implementation (TensorFlow)
- `DeepXDE` — Deep learning library for solving differential equations
- `NeuralPDE.jl` (Julia) — Physics-informed neural networks in Julia
- `lululxvi/deepxde` — Supports PINNs for various PDE systems
- `xiaowei0422/DeepONet` — Deep Operator Network implementation
- `zongyi-li/fourier-neural-operator` — FNO implementation
- `pybamm-team/PyBaMM` — Python Battery Mathematical Modelling (not ML but essential for ground truth)

---

## 5. Domain-Specific LLMs for Battery Science

### State of the Art (2024-2026)

**Existing Models:**
- **BatteryBERT (2022, CMU):** BERT-base model fine-tuned on battery science literature
  - Trained on ~400k battery-related papers from Semantic Scholar
  - Tasks: named entity recognition (materials, properties), relation extraction, literature search
  - Hugging Face: `batterydata/batterybert-cased` (and variants)
  - **Limitation:** BERT-scale (110M params), encoder-only, limited generation capability

- **MatSciBERT (2022, IIT Delhi):** BERT model for materials science text
  - Trained on 3.27M materials science abstracts
  - Hugging Face: `m3rg-iitd/matscibert`
  - Used for property prediction from text, literature mining

- **SciBERT (Allen AI):** General scientific BERT, pre-trained on 1.14M Semantic Scholar papers
  - Foundation for many materials science NLP tasks

- **MatBERT:** Domain-specific BERT for materials science literature
  - Improved NER and relation extraction for materials properties

- **GPT-based approaches (2024-2025):**
  - Fine-tuning LLaMA/Mistral/Qwen models on battery science corpus
  - ChatGPT/GPT-4 evaluation on battery science questions
  - "ChatBattery" (2024): LLM-driven battery design chatbot
  - Retrieval-Augmented Generation (RAG) systems for battery literature

**Current Frontiers (2024-2026):**
- **Multimodal LLMs:** Processing battery data tables, voltage curves (as images), and text simultaneously
- **LLM-guided experiment design:** Using LLMs to propose electrolyte formulations, electrode compositions
- **Automated literature review:** Extracting materials properties, performance metrics from thousands of papers
- **LLM + robotics:** Closed-loop autonomous battery material synthesis guided by LLM reasoning

### Hardware Requirements
| Task | Typical GPU Memory | Training Time | H20 Feasibility |
|------|-------------------|---------------|-----------------|
| BERT fine-tuning (110M) | ~4-8 GB | Hours | Trivial |
| LLaMA-7B fine-tuning (LoRA) | ~16-32 GB | Hours | Excellent (1 H20) |
| LLaMA-7B full fine-tuning | ~60-80 GB | Days | Good (1 H20, tight) |
| LLaMA-13B fine-tuning (LoRA) | ~32-64 GB | Hours-days | Excellent (1-2 H20) |
| LLaMA-70B inference (4-bit) | ~40 GB | Real-time | Good (1 H20) |
| LLaMA-70B LoRA fine-tuning | ~80-120 GB | Days | Good (2 H20) |
| LLaMA-70B full fine-tuning | ~400-600 GB | Weeks | Feasible (4-8 H20) |
| Pre-training BERT-scale from scratch | ~32-64 GB | Days-weeks | Excellent (2-4 H20) |
| Pre-training LLM (1-7B) from scratch | ~200-600 GB | Weeks-months | Feasible (4-8 H20) |

### Feasibility with 4-12x H20: **VERY GOOD to EXCELLENT**
- 4x H20 (384GB) can fine-tune LLaMA-70B with QLoRA
- 8-12x H20 (768-1152GB) can full fine-tune LLaMA-70B
- BERT-scale models are trivially easy
- This is one of the best use cases for the H20 cluster size

### Practical Impact / Novelty: **MEDIUM-HIGH**
- High utility for the battery research community
- Good publication venue: NLP+Science workshops at ACL/EMNLP/AAAI
- Real-world impact: automated literature review saves months of researcher time
- Risk: rapidly being commoditized by general-purpose LLMs (GPT-4, Claude)

### Key Datasets
- **Semantic Scholar corpus:** ~400k battery-related papers
- **arXiv materials science papers:** ~200k papers
- **Battery Data Genome:** Text descriptions paired with experimental data
- **Materials Project textual descriptions:** Structure-property descriptions

### Key Codebases
- `batterydata/batterybert-*` on Hugging Face
- `m3rg-iitd/matscibert` on Hugging Face
- `huggingface/transformers` — Foundation for all fine-tuning work
- `meta-llama/llama` — LLaMA models for fine-tuning
- `langchain-ai/langchain` — RAG pipeline construction

---

## 6. Sodium-Ion Battery Specific Research

### State of the Art ( 2024-2026)

Na-ion batteries are experiencing a renaissance due to:
- Concerns about Li resource scarcity and cost
- Na abundance (1000x more than Li in Earth's crust)
- Successful commercialization by CATL, HiNa Battery, Faradion
- Grid storage applications where energy density is less critical

**What's Unique About Na-ion That AI Can Help With:**

1. **Larger Na+ ion size (1.02 Å vs 0.76 Å for Li+):**
   - Different intercalation chemistry and host structures
   - ML can screen host materials optimized for the larger ion
   - Prussian blue analogs, layered oxides, polyanionic compounds as cathodes
   - Hard carbon as anode (amorphous carbon, complex Na storage mechanism)

2. **Less mature database coverage:**
   - Materials Project has far fewer Na compounds than Li compounds
   - **Opportunity:** Building a comprehensive Na-ion materials database using ML + DFT
   - High-throughput screening of Na-based compounds is a wide-open field

3. **Complex Na storage mechanisms in hard carbon:**
   - Na storage involves both intercalation and pore-filling mechanisms
   - ML molecular dynamics can resolve the storage mechanism
   - Relating carbon microstructure to capacity is an ML-friendly problem

4. **Electrolyte differences:**
   - NaPF6 vs LiPF6 electrolytes
   - Different SEI composition (NaF, Na2CO3, Na2O vs LiF, Li2CO3, Li2O)
   - Wider electrochemical stability requirements
   - ML force fields for Na+ solvation structure in various solvents

5. **Sodium solid electrolytes:**
   - β-alumina, Na3Zr2Si2PO12 (NASICON), Na3SbS4, Na2B10H10
   - Much less explored than Li solid electrolytes
   - ML screening for Na superionic conductors

**Na-ion Specific ML Papers (2024-2025):**
- "Machine learning-guided discovery of sodium-ion battery cathode materials"
- "GNN-accelerated screening of NASICON-type solid electrolytes"
- "ML prediction of hard carbon capacity from precursor properties"
- "Transfer learning from Li-ion to Na-ion battery data"

### Hardware Requirements
Same as Section 1 (Material Discovery) and Section 2 (Electrolyte Design). All tasks are equally feasible.

### Feasibility with 4-12x H20: **EXCELLENT**
Identical to Sections 1-2. The key advantage is the novelty of Na-ion specific applications.

### Practical Impact / Novelty: **VERY HIGH (Highest novelty potential)**
- Na-ion is a hot topic with relatively few ML papers compared to Li-ion
- Every Li-ion ML approach can be adapted to Na-ion with minimal modification
- First-mover advantage in Na-ion ML research
- Strong funding availability (China's battery industry, EU green energy programs)
- Commercial relevance: CATL, BYD, HiNa Battery, Faradion all investing heavily

### Na-ion Specific Datasets
- **Materials Project Na-compounds:** ~10k+ Na-containing entries (vs ~50k+ Li)
- **OQMD Na-compounds:** Similar scale
- **Aflow Na-compounds:** Similar scale
- **Na-ion battery literature data:** Can be extracted via NLP (BatteryBERT)
- **Hard carbon anode databases:** Small, scattered; opportunity to consolidate

### Na-ion Specific Codebases
- Same as Sections 1-2 (FAIRChem, DeePMD, MACE, pymatgen)
- `sparks-baird/xtal2png` — Converting crystal structures to images (useful for screening)
- Need to build Na-ion specific datasets and benchmarks (opportunity for contribution)

---

## 7. Open-Source Battery AI Projects and Datasets

### Major Codebases

| Project | Stars | Description | URL |
|---------|-------|-------------|-----|
| **fairchem** (FAIR Chemistry) | 2.1k | Universal MLIPs (UMA), OCP models | github.com/facebookresearch/fairchem |
| **DeePMD-kit** | 1.9k | ML force fields, MD integration | github.com/deepmodeling/deepmd-kit |
| **pymatgen** | 1.9k | Materials analysis toolkit | github.com/materialsproject/pymatgen |
| **MACE** | ~1.2k | Equivariant interatomic potentials | github.com/ACEsuit/mace |
| **PyBaMM** | ~1.2k | Battery mathematical modeling | github.com/pybamm-team/PyBaMM |
| **NequIP** | ~700 | E(3)-equivariant neural network potentials | github.com/mir-group/nequip |
| **dpdata** | 246 | Atomistic data manipulation | github.com/deepmodeling/dpdata |
| **AmpTorch** | 61 | Atomistic ML package (PyTorch) | github.com/ulissigroup/amptorch |
| **CHGNet** | ~300 | Universal GNN potential | github.com/CederGroupHub/chgnet |
| **CDVAE** | ~300 | Crystal structure generation | github.com/txie-93/cdvae |

### Major Datasets

| Dataset | Size | Description | Access |
|---------|------|-------------|--------|
| **Materials Project** | ~500k crystals | DFT properties for inorganic materials | materialsproject.org (free) |
| **OMat24** | 180M+ DFT calcs | Meta's bulk materials dataset | huggingface.co/facebook/OMat24 |
| **OMol25** | 100M+ mol calcs | Meta's molecular dataset | huggingface.co/facebook/OMol25 |
| **OQMD** | ~1M compounds | Open Quantum Materials Database | oqmd.org (free) |
| **AFLOW** | ~3.5M compounds | Automatic FLOW materials database | aflow.org (free) |
| **NIST JARVIS** | ~50k materials | Joint Automated Repository for Various Integrated Simulations | jarvis.nist.gov (free) |
| **Stanford Battery Dataset** | 124 cells | LFP fast-charging cycle life | data.matr.io |
| **Toyota/TRI Battery Dataset** | ~1000+ cells | Multi-chemistry cycling data | data.matr.io |
| **CALCE Battery Dataset** | 100s of cells | Long-term cycling, multi-chemistry | web.calce.umd.edu |
| **Battery Archive** | Aggregated | Multi-source battery cycling data | batteryarchive.org |
| **NASA Battery Dataset** | Dozens of cells | Prognostics benchmark | data.nasa.gov |
| **Open Catalyst 2020/2022** | 1.3M+ relaxations | Adsorption catalyst data (relevant for SEI) | opencatalystproject.org |
| **QM9** | ~134k molecules | Molecular quantum properties | quantum-machine.org |
| **ANI-1x/2x** | ~20M conformations | Organic molecule DFT | figshare (free) |

### Domain-Specific Tools
- **PyBaMM (Python Battery Mathematical Modelling):** Open-source battery simulation framework. Essential for generating training data for PINNs and validating ML models against physics-based simulations. Supports Doyle-Fuller-Newman (P2D), Single Particle Model, etc.
- **Battery Data Toolkit:** Tools for standardizing battery cycling data formats
- **galvanalyser:** Battery data management and analysis platform
- **Pymatgen + Matminer:** Materials feature extraction and analysis

---

## 8. Recommendations & Priority Matrix

### Ranked by Overall Research Value

| Rank | Direction | Feasibility | Novelty | Impact | Recommended Priority |
|------|-----------|-------------|---------|--------|---------------------|
| **1** | **Na-ion Material Discovery** | Excellent | Very High | Very High | TOP PRIORITY |
| **2** | **Electrolyte Design (ML FFs)** | Excellent | High | Very High | HIGH |
| **3** | **Battery Digital Twin (PINNs)** | Excellent | High | High | HIGH |
| **4** | **Li-ion Material Discovery** | Excellent | Medium-High | Very High | MEDIUM-HIGH |
| **5** | **Degradation/Lifetime Prediction** | Overkill-feasible | Medium | High | MEDIUM |
| **6** | **Domain LLM (BatteryLLM)** | Very Good | Medium-High | Medium | MEDIUM |

### Recommended Research Strategy

**Phase 1 (Months 1-3): Foundation**
1. Set up the 4-12x H20 cluster with fairchem, DeePMD-kit, MACE, PyBaMM
2. Download Materials Project, OMat24, Stanford Battery, Toyota Battery datasets
3. Reproduce baseline results on Li-ion material screening (to validate pipeline)

**Phase 2 (Months 3-6): Na-ion Focus**
1. Build Na-ion materials screening pipeline using UMA/MACE models
2. Screen cathode materials: layered Na_xMO2, polyanionic (Na3V2(PO4)3, Na2FePO4F), Prussian blue analogs
3. Screen solid electrolytes: NASICON, β-alumina, thiophosphate families
4. Train ML force fields for Na+ electrolyte systems

**Phase 3 (Months 6-12): Advanced Topics**
1. Na-ion electrolyte formulation optimization via ML + MD
2. Hard carbon anode optimization via ML-guided precursor selection
3. PINN-based Na-ion digital twin (parameterized from atomistic simulations)
4. Na-ion specific BatteryLLM fine-tuning

### Unique Opportunities with H20 GPUs

The NVIDIA H20 (96GB HBM3) is specifically well-suited because:
1. **Large VRAM per card (96GB):** Enables training large equivariant GNNs (MACE, NequIP) on single GPU without model parallelism
2. **HBM3 bandwidth:** Critical for molecular dynamics workloads with ML force fields
3. **4-12 cards:** Enables data-parallel training for large models AND independent parallel screening jobs
4. **Cost-efficiency:** H20 is a China-market GPU (Hopper-based, reduced interconnect), but for battery ML workloads, the interconnect bottleneck is minimal since most tasks fit on 1-4 GPUs

### Key Differentiators for Publications

1. **"First comprehensive ML screening of Na-ion battery materials"** — Very publishable in Nature Energy / Joule / Energy & Environmental Science
2. **"Foundation model for battery electrochemistry"** — Train a multi-task model on battery cycling, materials, and electrolyte data
3. **"ML-accelerated Na-ion electrolyte discovery"** — High novelty, practical impact
4. **"Digital twin of Na-ion cell with uncertainty quantification"** — Combines PINNs + Bayesian ML
5. **"Benchmark suite for battery AI"** — Create the MMLU of battery science (high community impact)

---

## Appendix: Key References

### Foundation Models for Materials
- Meta FAIR Chemistry: UMA-1.2 (2026), OMat24, OMol25 datasets
- DeePMD-kit v3: Zeng et al., J. Chem. Theory Comput. (2025)
- MACE: Batatia et al., "A foundation model for atomistic materials chemistry" (2024)
- CHGNet: Deng et al., Nature Machine Intelligence (2023)

### Battery Degradation
- Severson et al., "Data-driven prediction of battery cycle life before capacity degradation," Nature Energy (2019)
- Attia et al., "Closed-loop optimization of fast-charging protocols for batteries with machine learning," Nature (2020)

### PINNs for Electrochemistry
- Raissi et al., "Physics-informed neural networks," J. Comp. Phys. (2019)
- Li et al., "Physics-informed neural operator (PINO)," (2024)

### Domain LLMs
- BatteryBERT: Huang et al., "A battery language model for battery literature mining" (2022)
- MatSciBERT: Gupta et al., "MatSciBERT: A materials domain language model" (2022)

### Na-ion Batteries
- Yabuuchi et al., "Research development on sodium-ion batteries," Chemical Reviews (2014+)
- Nayak et al., "Sodium ion batteries: Present and future," Chemical Reviews (2023)
