# AGENTS.md — Project Memory & Context

## §1 Project Overview

**Goal**: Build a rigorous framework for battery degradation parameter identifiability analysis (Fisher Information theory + Bayesian diagnosis + cross-chemistry validation), targeting JPS/Applied Energy publication. Secondary track: three-stage Pretrain-Finetune-LoRA framework for NE/Joule.

**Remote server**: `10.239.68.24` (2× NVIDIA H20, conda env `autobattery`)
**Working directory**: `/AI4S/Users/howardwang/h204/autobattery`
**Current branch**: `main` (at `840c576`)

## §2 Environment & Constraints

- PyBaMM 26.4.1, PyTorch 2.5.1+cu124, mat73 installed
- Server cannot reach `data.matr.io` (Toyota TRI)
- `kagglehub` installed and working (used for MIT dataset)
- No `torchbnn` package — use MC-dropout for Bayesian NN
- Data safety: experimental data must use HKRI Agent's minimax-m2.7 model; simulation data unrestricted
- `mat73` gives string-type warnings for MATLAB v7.3 .mat files — non-fatal

## §3 Data Assets

| Dataset | Location | Size | Notes |
|---------|----------|------|-------|
| LFP fullfield | `data/fullfield/fullfield_lfp_degradation.h5` | 1200 sims, 7 params | 4 params vary significantly (LAM_pos, D_p, t+, R_mult); SEI/LAM_neg/D_n tiny |
| LIB fullfield | `data/fullfield/fullfield_lib.h5` | 1888 sims, 7 params | Different chemistry, wider param range |
| LMB fullfield | `data/fullfield/fullfield_lmb.h5` | multiple outputs | Lithium metal battery data |
| Experimental | `data/experimental/experimental_cycling.h5` | 164 cells | V(t)+capacity only, no mechanism data |
| MIT Severson | `data/external/severson/severson_lfp.h5` | 138 LFP cells | Cycle lives 148-1935, fade 3.6-23.8% |
| MIT raw | `/root/.cache/kagglehub/datasets/rickandjoe/.../versions/1/` | 7.8GB | 3 batch .mat files |

**Param order** (all fullfield datasets): `["SEI", "LAM_neg", "LAM_pos", "D_n", "D_p", "t+", "R_mult"]`

## §4 Phase A Results (Completed)

### A1: XGBoost/RF Baseline (`scripts/90_xgboost_baseline.py`)
- SEI R²=0.85-1.00, D_p R²=0.93-0.99 (except LFP)
- t+/R_mult always negative R² (unidentifiable)
- LAM only identifiable in LFP

### A2: Fisher Spectral (`scripts/91_fisher_spectral.py`)
- **rank=3 for ALL 5 chemistries** (not 4)
- CRLB: D_p and SEI finite+tight, t+/LAM/R_mult all inf
- LFP required full SEI_DEFAULTS dict for Prada2013
- FIM eigenvalues: λ₁ >> λ₂ >> λ₃ >> λ₄≈0

### A3: Mutual Information (`scripts/92_mutual_information.py`)
- PCA(20) + kNN MI estimation
- **ID avg I(V;θ)=0.20, UN avg=0.007 → 28.7× separation ratio**
- SEI carries most info (I=0.39), t+/R_mult zero info
- DPI violations are estimation artifacts

### A4: PyBaMM MLE Fit (`scripts/93_pybamm_fit_baseline.py`)
- differential_evolution 100% converges but to wrong values (SEI=164%, LAM_neg=135%)
- Validates that identifiability is structural, not optimization-related
- Fixed int64 JSON serialization bug

### A5: Rigorous Identifiability (`outputs/rigorous_identifiability/`)
- **Fisher ID group**: D_n, SEI, LAM_neg
- **Fisher UN group**: D_p, t+, LAM_pos, R_mult
- Effective rank: 2.79, null space dim: 4
- Reconstruction quality: full RMSE=76, fixed RMSE=83

## §5 Phase B Results (This Session)

### B1: MIT Severson Parser (`scripts/94_parse_severson.py`)
- **138 cells** parsed (2 NaN cycle_life skipped in batch 3)
- 3 batches: 46 + 48 + 44 cells
- Output: `data/external/severson/severson_lfp.h5`

### B2: Bayesian NN External Validation (`scripts/95_severson_bayesian_validation.py`)
**PCA Manifold Analysis:**
| Feature | Top-1 var | Top-3 cumvar | n@95% | Eff rank |
|---------|-----------|-------------|-------|----------|
| V_early | 0.638 | 0.828 | 10 | 1.78 |
| V_late  | 0.673 | 0.804 | 13 | 1.31 |
| delta_V | 0.637 | 0.796 | 13 | 1.45 |

**Within-Cell Temporal Rank:**
- Median effective rank: **1.22** (range 1.08-3.02)
- **99.3% of cells** have eff_rank ≤ 3
- **97.1% of cells** have eff_rank ≤ 2
- Real LFP data is even MORE degenerate than Fisher rank=3 predicts

**MC-Dropout BNN:**
| Task | Features | R² | MAE | Uncertainty corr |
|------|----------|-----|-----|-----------------|
| fade% | delta_V | **0.968** | 0.283 | 0.201 |
| fade% | V_early | 0.675 | 0.422 | 0.856 |
| log(cycle_life) | delta_V | **0.827** | 56.3 | 0.361 |
| log(cycle_life) | V_early | 0.576 | 96.1 | 0.556 |

**Information Map:**
- Peak MI at t=57-61 (late discharge) → MI=0.85
- Minimum MI at t=0,1,90,91,96 → MI≈0.17
- Top 10% of timepoints carry 15% of total MI

**Key insight**: Real data eff_rank ~1.2-1.5 < Fisher rank=3 → degradation is dominated by a single mode (likely SEI/LLI), validating and strengthening the low-rank conclusion.

### B3: Signature Library Ablation (`scripts/97_signature_ablation.py`)
**Experiment results:**
| Experiment | R² | # Params | Notes |
|-----------|-----|---------|-------|
| All 7 signatures | 0.990 | 7 | Baseline |
| **Fisher-guided (D_n, SEI, LAM_neg)** | **0.985** | 3 | **99.4% of full, Fisher-correct set** |
| Random subsets (size 3) | 0.861±0.115 | 3 | Fisher is +1.1σ above mean |
| Best data-driven k=3 | 0.985 | 3 | LAM_neg, D_n, R_mult |
| SVD k=3 | 0.998 | 3 | Model-agnostic upper bound |

**Forward selection order**: D_n → LAM_neg → R_mult → SEI → LAM_pos → t+ → D_p
- First 2 (D_n, LAM_neg) match Fisher ID group
- Fisher-guided 3 params capture 99.4% of full 7-param performance

**Leave-one-out importance (ΔR²):**
| Param | ΔR² when removed | Category |
|-------|-----------------|----------|
| D_n | **0.0437** | ID (critical) |
| LAM_neg | **0.0066** | ID |
| R_mult | 0.0023 | Mixed |
| SEI | 0.0021 | ID |
| t+ | 0.0006 | UN |
| LAM_pos | 0.0005 | UN |
| D_p | 0.0002 | UN |

### B4: Sensitivity Profile Analysis (`scripts/98b_sensitivity_profile.py`)
**Variance decomposition sensitivity ranking:**
1. D_n (EVR=0.0002)
2. D_p (EVR=0.0001)
3. SEI (EVR=0.0001)
4-7. LAM_neg, t+, LAM_pos, R_mult (EVR≈0)

**Conditional profile dynamic range ranking:**
1. D_p (log10=30.97)
2. t+ (30.42)
3. D_n (30.35)
4-7. R_mult, LAM_pos, LAM_neg, SEI

**Key finding**: Different methods give different parameter rankings — identifiability is method-dependent, requiring multi-method validation.

### B5: Profile Likelihood (attempted, `scripts/96_profile_likelihood.py`)
- Surrogate-based (MLP R²=0.87): all params show identical flat profiles → surrogate too inaccurate
- Empirical (database binning): too sparse in 7D, all flat
- **Blocker**: needs PyBaMM forward model for proper profile likelihood (est. 2-4 hours compute)

## §6 Cross-Cutting Conclusions

1. **Fisher rank=3 is universal** across 5 chemistries (NMC811, NCA, LFP, LFP_v2, LCO)
2. **Real-world data confirms** low-rank structure (MIT Severson: within-cell eff_rank median=1.22, 99.3% ≤ 3)
3. **Fisher-guided signature selection** achieves 99.4% of full model accuracy with 3/7 params
4. **Which params are identifiable is chemistry- and method-dependent**: Fisher says D_n/SEI/LAM_neg; forward selection says D_n/LAM_neg/R_mult; sensitivity says D_n/D_p/SEI
5. **dQ/dV does NOT universally improve identifiability**
6. **PyBaMM MLE fails structurally** — optimization converges but to wrong values
7. **Real LFP degradation is rank ~1** (below Fisher rank=3) — dominated by single mechanism
8. **V(t) evolution strongly encodes degradation state** (delta_V → fade% R²=0.97)
9. **Late discharge region** (t=57-61 in normalized time) carries most information about fade

## §7 Publication Roadmap

### Route A: Identifiability Theory (JPS/Applied Energy, 4-6 weeks)
**Title**: "Fisher Information Theory Reveals Structural Non-Identifiability in Battery Degradation Diagnosis: A Cross-Chemistry Validation Study"

**Paper structure:**
1. Introduction: why parameter identifiability matters
2. Theory: Fisher Information Matrix, rank analysis, CRLB
3. Methods: 5-chemistry fullfield simulation + MIT Severson validation
4. Results:
   - Table 1: FIM rank, eigenvalues, CRLB per chemistry (from Phase A)
   - Fig 1: Jacobian structure heatmap (from rigorous_identifiability)
   - Fig 2: Cross-chemistry ID/UN separation (MI analysis, 28.7× ratio)
   - Fig 3: MIT Severson manifold analysis (eff_rank 1.22, PCA, info map)
   - Fig 4: Signature ablation (Fisher-guided 99.4% of full)
   - Fig 5: Sensitivity comparison across methods
5. Discussion: implications for degradation modeling, model reduction
6. Conclusion: rank=3 universal, Fisher-guided selection near-optimal

**Remaining work:**
- [ ] Profile Likelihood with PyBaMM forward model (for Fig 3)
- [ ] Publication-quality figures
- [ ] Cross-chemistry NMC/NCA fullfield generation (if time)
- [ ] Paper writing

### Route B: LMB Cryo-EM Story (NE/Joule, 6-12 months)
See `docs/plan_publication_roadmap.md` on main branch.

## §8 Key Scripts (on remote server)

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/94_parse_severson.py` | MIT/Severson parser (mat73) | Done |
| `scripts/95_severson_bayesian_validation.py` | BNN + PCA + info map validation | Done |
| `scripts/96_profile_likelihood.py` | Profile likelihood (surrogate-based) | Blocked |
| `scripts/97_signature_ablation.py` | Signature library ablation | Done |
| `scripts/98b_sensitivity_profile.py` | Sensitivity ranking + profile | Done |
| `scripts/11_generate_fullfield.py` | Generate fullfield simulation data | Existing |
| `scripts/30_baseline_pybamm_fit.py` | PyBaMM MLE baseline | Existing |
| `scripts/18_fisher_analysis.py` | Fisher analysis | Existing |

## §9 Output Files

| Path | Content |
|------|---------|
| `outputs/rigorous_identifiability/` | Phase A Jacobian, subspace, practical results |
| `outputs/severson_validation/severson_validation_results.json` | MIT BNN/PCA/rank validation |
| `outputs/signature_ablation/signature_ablation_results.json` | Ablation study results |
| `outputs/profile_likelihood/sensitivity_profile_results.json` | Sensitivity + profile analysis |
| `outputs/profile_likelihood/empirical_profile_likelihood_results.json` | Empirical PL (limited) |
| `data/external/severson/severson_lfp.h5` | Parsed Severson dataset (138 cells) |

## §10 Conventions

- All scripts use `os.environ["MKL_THREADING_LAYER"] = "GNU"` for PyTorch compat
- Remote commands: `ssh 10.239.68.24 "source /root/miniconda3/etc/profile.d/conda.sh && conda activate autobattery && cd /AI4S/Users/howardwang/h204/autobattery && ..."`
- Lint/typecheck: no standard commands defined yet — check before running
- Commit style: conventional commits (feat:, fix:, docs:)
