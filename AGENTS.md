# AGENTS.md — Project Memory & Context

## §1 Project Overview

**Goal**: Two parallel publication tracks:
- **Route A**: Battery degradation parameter identifiability analysis (Fisher Information theory + Bayesian diagnosis + cross-chemistry validation), targeting JPS/Applied Energy.
- **Route C**: Universal scaling and degradation archetypes across 10⁴ lithium-ion cells, targeting Joule / Nat. Comm.

Secondary track: **Route B** LMB Cryo-EM story (NE/Joule, 6-12 months, see `docs/plan_publication_roadmap.md`).

**Remote server**: `10.239.68.24` (2× NVIDIA H20, conda env `autobattery`)
**Working directory**: `/AI4S/Users/howardwang/h204/autobattery`
**Current branch**: `main` (at `da2dedf`)

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

### B5: Profile Likelihood (completed, `scripts/96c_profile_likelihood_gp.py`)
- MLP surrogate (R²=0.87): all flat → insufficient
- **GP surrogate (R²=0.999)**: clean separation achieved
- **R_mult is the only unidentifiable parameter** (Δχ²_max=0.86, flat profile)
- All other params show sharp profiles (Δχ²_max > 1000)
- Cross-method disagreement: Fisher says D_p/t+/LAM_pos also UN; PL says they're ID
- Figure: `outputs/profile_likelihood/gp_profile_likelihood.png`

### B5b: Rank Robustness — Cross-Chemistry (`scripts/99_rank_robustness.py`)
**LFP (1200 sims):** η=1e-3 → rank=3 [3–3], joint multi-rate → rank=4
**LIB (1888 sims):** η=1e-3 → rank=3 [2–3], joint multi-rate → rank=4
- rank=3 confirmed across BOTH chemistries at η=1e-3
- Multi-rate Fisher gains +1 rank in both cases
- Raw parameterisation gives lower rank (scale issues)
- Figures: `outputs/rank_robustness/`, `outputs/rank_robustness/lib/`

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

### B6: Rank Robustness (`scripts/99_rank_robustness.py`)
**η-rank table (log_standardised, all data):**
| η | rank median | 95% CI |
|---|------------|--------|
| 1e-2 | 3 | [2–3] |
| 1e-3 | **3** | [3–3] |
| 1e-6 | 7 | [7–7] |

- rank=3 robust at η=1e-3 across all 4 parameterisations (raw/log/log_std/pca_white)
- Multi-rate joint Fisher: single-rate rank=3 → joint rank=4 (η=1e-3)
- Raw parameterisation gives lower rank (2-5) due to extreme scale differences
- Bootstrap 200× confirms stability
- Figures: `outputs/rank_robustness/spectrum.png`, `rank_vs_eta.png`, `rank_gain_multirate.png`

### B7: Universality Pipeline — Severson Dryrun (`scripts/101_severson_dryrun.py`)
- **Archetypes**: k=6 (BIC-selected), ≥2 required → PASS
- **Scaling collapse**: RMS=3.78%, ≤5% required → PASS
- **Knee recall**: 100%, ≥80% required → PASS
- **Master curve fit**: R²=0.967 (power_exp form), 91.1% curves within ±5%
- **Verdict**: Pipeline ready for 10K data

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
   - Fig 5: Rank robustness (η-sweep, parameterisation, multi-rate)
   - Fig 6: Sensitivity comparison across methods
5. Discussion: implications for degradation modeling, model reduction
6. Conclusion: rank=3 universal, Fisher-guided selection near-optimal

**Completed:**
- [x] GP Profile Likelihood using fullfield simulation V(t) as target (`scripts/96c_profile_likelihood_gp.py`)
- [x] Cross-chemistry rank robustness on LIB fullfield (`scripts/99_rank_robustness.py`)
- [x] Publication-quality figures (`scripts/103_paper_figures.py`) — 8 figures in `outputs/paper_figures/`

**Remaining work:**
- [ ] Paper writing (all experimental results available)

### Route C: Universality & Scaling (Joule / Nat. Comm.)
**Title**: "Universal Scaling and Degradation Archetypes Across 10⁴ Lithium-Ion Cells"
**Plan**: `docs/plan_universality_paper.md`

**Pipeline**: `src/universality/` (curves, knee, archetype, scaling, phase_diagram)
**Scripts**: `100_health_check.py`, `101_severson_dryrun.py`, `102_universality_pipeline.py`, `103_paper_figures.py`

**Severson dryrun**: ALL PASS (archetypes k=6, collapse RMS=3.78%, knee recall=100%)
**Status**: Waiting for proprietary 10K-cell dataset

**What to do before 10K data:**
- [x] `scripts/103_paper_figures.py` — 8 figures generated in `outputs/paper_figures/`
- [x] Severson archetype internal analysis (within-archetype scaling, N★ distribution)
- [ ] Synthetic 10K-cell benchmark (generate from simulation to validate pipeline at scale)
- [ ] Paper methods section draft (pipeline description independent of data)

### Route B: LMB Cryo-EM Story (NE/Joule, 6-12 months)
See `docs/plan_publication_roadmap.md` on main branch.

## §8 Next Action Plan (Priority Order)

### Immediate (this week)
1. ~~**Run GP Profile Likelihood** (`96c`) using fullfield simulation V(t)~~ ✅ R_mult only UN
2. ~~**Run rank robustness on LIB fullfield**~~ ✅ rank=3 confirmed cross-chemistry
3. **Start Route A paper writing** — all experimental results available
4. ~~**Write `103_paper_figures.py`**~~ ✅ 8 figures in `outputs/paper_figures/`

### Short-term (1-2 weeks)
5. **Refine Route A figures** for journal submission (font size, colorblind palette, etc.)
6. **Paper methods + results sections** — combine all Phase A/B results into manuscript
7. **Synthetic 10K-cell benchmark** for universality pipeline validation at scale

### Pending external data
8. **Route C full run** when proprietary 10K data arrives
9. **Profile Likelihood on real experimental data** when Neware xlsx available

## §9 Key Scripts (on remote server)

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/94_parse_severson.py` | MIT/Severson parser (mat73) | Done |
| `scripts/95_severson_bayesian_validation.py` | BNN + PCA + info map validation | Done |
| `scripts/96c_profile_likelihood_gp.py` | Profile likelihood (GP surrogate) | Done (R_mult only UN) |
| `scripts/96b_profile_likelihood_pybamm.py` | Profile likelihood (PyBaMM forward) | Needs adaptation for simulation data |
| `scripts/97_signature_ablation.py` | Signature library ablation | Done |
| `scripts/98b_sensitivity_profile.py` | Sensitivity ranking + profile | Done |
| `scripts/99_rank_robustness.py` | η-rank robustness + bootstrap + multi-rate | Done |
| `scripts/100_health_check.py` | Universality dataset pre-check | Done |
| `scripts/101_severson_dryrun.py` | Universality Severson dry run | Done (ALL PASS) |
| `scripts/102_universality_pipeline.py` | Full universality pipeline (10K data) | Waiting for data |
| `scripts/103_paper_figures.py` | Paper figure generation | Done (8 figs) |
| `scripts/11_generate_fullfield.py` | Generate fullfield simulation data | Existing |
| `scripts/30_baseline_pybamm_fit.py` | PyBaMM MLE baseline | Existing |
| `scripts/18_fisher_analysis.py` | Fisher analysis | Existing |

## §10 Output Files

| Path | Content |
|------|---------|
| `outputs/rigorous_identifiability/` | Phase A Jacobian, subspace, practical results |
| `outputs/severson_validation/severson_validation_results.json` | MIT BNN/PCA/rank validation |
| `outputs/signature_ablation/signature_ablation_results.json` | Ablation study results |
| `outputs/profile_likelihood/gp_profile_likelihood_results.json` | GP PL results (R_mult UN) |
| `outputs/profile_likelihood/sensitivity_profile_results.json` | Sensitivity + profile analysis |
| `outputs/profile_likelihood/empirical_profile_likelihood_results.json` | Empirical PL (limited) |
| `outputs/profile_likelihood/gp_profile_likelihood.png` | GP PL figure |
| `outputs/rank_robustness/` | LFP rank table, spectra, η-sweep plots |
| `outputs/rank_robustness/lib/` | LIB rank table, spectra, η-sweep plots |
| `outputs/paper_figures/` | 8 publication figures (fig1-fig8) |
| `outputs/universality/severson/` | Severson dryrun results (archetype, scaling, knee) |
| `data/external/severson/severson_lfp.h5` | Parsed Severson dataset (138 cells) |

## §10 Conventions

- All scripts use `os.environ["MKL_THREADING_LAYER"] = "GNU"` for PyTorch compat
- Remote commands: `ssh 10.239.68.24 "source /root/miniconda3/etc/profile.d/conda.sh && conda activate autobattery && cd /AI4S/Users/howardwang/h204/autobattery && ..."`
- Lint/typecheck: no standard commands defined yet — check before running
- Commit style: conventional commits (feat:, fix:, docs:)
