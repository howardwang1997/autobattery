# Universality Paper — Plan & Code Map

**Working title:** *Universal scaling and degradation archetypes across N lithium-ion cells*

**Target venue:**
- Joule / Nat. Comm. (best case) — requires ≥10 formula variants + clean collapse
- Energy Storage Materials / npj Comp. Mater. (realistic) — archetype taxonomy + scaling
- Applied Energy / JPS (downgrade) — small-scale or single-formulation validation

**Updated 2026-05-08.** Paper 1 (rank=3 identifiability) framing deemed insufficient for
2026 novelty standards. Paper 2 (universality) is the primary path. See §Strategic
Decisions below for the full decision tree.

---

## Strategic Decisions (2026-05-08)

### Paper 1 (identifiability): NOT the primary path

**Reason:** "Effective rank ≈ 2-3" is a 2019-era claim (Bizeray, Sulzer), not novel
in 2026. Reviewers will ask "what is new compared to Severson 2019 / Sulzer 2021?"

Literature claims:

| Ref | Model | Rank reported | Novelty level |
|-----|-------|---------------|---------------|
| Forman 2012 (JPS 210) | DFN | "3-5 strongly identifiable" | Early work |
| Bizeray 2019 ( JES 166) | SPM | rank = 2 | Landmark SPM |
| Sulzer 2021 ( JES 168) | DFN | rank = 3 (single C-rate) | Landmark DFN |
| **This work** | DFN + Severson | rank 2-3 @ η=1e-3 | **Validated on 138 real cells, quantifies multi-rate lift 3→4** |

**Revised Paper 1 claim** (if pursued separately):  
"Multi-rate electrochemical protocols lift per-cell identifiability rank from 3 to 4,
providing quantitative justification for GITT/HPPC/EIS observables."  
Target: Applied Energy / J. Energy Storage.

### Paper 2 (universality): PRIMARY path

**Central claim:** Battery degradation trajectories cluster into archetypes, each
following a master curve Q(N)/Q₀ = Φ(N/N★), where N★ depends on formulation.

**What requires what:**

| Analysis | Needs formula data? | Claim novelty | Target |
|----------|---------------------|---------------|--------|
| Archetype taxonomy | ❌ | Medium (scale confirmation) | JPS / Applied Energy |
| Scaling collapse | ❌ | Medium (already in Bloom 1977) | Applied Energy |
| Phase diagram | ✅ Required | High (no prior work) | Joule / ESM |
| N★ = f(formulation) | ✅ Required | High (closed-form law) | Joule / Nat. Energy |
| Lifetime prediction | ❌ | ❌ Very low (saturated since 2019) | Not recommended |

### Lifetime prediction is NOT the claim

**Confirmed saturated:** Severson 2019 (Nat. Energy) → hundreds of follow-ups 2020-2026.
A pure Q(N) → EOL prediction model with no formula dimension offers no novelty.

**The novel claim is not "predict lifetime" but "discover the formula-space structure
of degradation behavior."**

---

## Formula diversity decision tree

| Formula variants | Analysis possible | Claim | Target |
|-----------------|-------------------|-------|--------|
| **1** | Archetype + scaling only | "Large-scale validation of archetype taxonomy" | JPS (weak) |
| **2** | 2-group comparison | "Two formulations have significantly different archetype distributions (χ², p<0.001)" | JPS / JES |
| **3-5** | Multinomial logistic | "Formulation class predicts degradation archetype (AUC ≥ 0.8)" | Applied Energy |
| **≥10 (continuous)** | Full phase diagram | "Archetype phase boundaries in formula space" + N★ closed-form | Joule / ESM |

**Key question before writing claim:**  
Is formula variance (between-config) >> within-config variance?  
→ Compute eta² = SS_between / SS_total for EOL cycles. If eta² < 0.06: effect too
small to publish. If eta² > 0.14: publishable effect.

---

## η (eta) parameter definition

The relative eigenvalue threshold η defines the **practical identifiability rank**:

```
η-rank = count(λᵢ / λ₁ > η)
```

where λ₁ ≥ λ₂ ≥ ... are Fisher eigenvalues sorted descending.

**Physical meaning:** η corresponds to voltage noise floor.
- η = 1e-2 → signals above ~10 mV → practical rank
- η = 1e-6 → signals above ~0.1 mV → below typical cycler resolution

**Our results:** rank = 3 @ η=1e-3 (typical lab noise ~1-5 mV), rank = 7 @ η=1e-6
(full mathematical rank but practically unidentifiable).

**Paper framing for η-rank:** "rank ≈ 2-3 at typical lab noise levels (η ∈ [1e-3, 1e-2]),
not a universal constant."

---

## Literature review (2026)

**Core references:**

| Ref | DOI | Key claim | Required citation |
|-----|-----|-----------|-------------------|
| Severson 2019 Nat. Energy | 10.1038/s41560-019-0356-8 | RF + 11 features, 124 cells, R²≈0.9 | Dataset source |
| Bizeray 2019 JES | 10.1149/2.0501913jes | SPM rank = 2 | SPM baseline |
| Sulzer 2021 JES | 10.1149/1945-7111/ac1a85 | DFN rank = 3 (single C-rate) | DFN baseline |
| Forman 2012 JPS | 10.1016/j.jpowsour.2012.03.009 | "3-5 parameters strongly identifiable" | Early work |
| Transtrum 2014 PRE | 10.1103/PhysRevE.90.042127 | Sloppy models, geometric eigenvalue decay | Theory |
| Fermín-Cueto 2020 Nat. Energy | — | Knee detection + ML prediction | Knee method |
| Raue 2009 Bioinformatics | — | Profile likelihood for identifiability | Method |
| O'Kane 2022 | — | Lithium plating DFN model | PyBaMM |

---

## Data contract

Two CSV (or parquet) tables.

```
cell_metadata.csv      cell_id, feature_1, feature_2, …
cycling_data.csv       cell_id, cycle_number, discharge_capacity
```

Minimum scale to make the story land:

| | Floor (will run) | Recommended | Ideal |
|---|---|---|---|
| n_cells | 500 | 2 000 | 5 000+ |
| cycles/cell | 100 | 300 | to-EOL |
| % reaching 80% SOH | 30% | 50% | 70% |
| continuous formulation vars | 1 | 2-3 | ≥3 |
| cells per config | 1 | 3 | 5+ |

Run `scripts/100_health_check.py` first; it prints which targets are met and
which downstream methods are blocked.

**Unknown formula diversity?** Start with eta² calculation in §Formula diversity
decision tree above before designing the paper claim.

---

## Pipeline

```
Q(N) raw  ──► align/interp ──► knee detection (defines N★)
                                        │
                                        ├──► FPCA + GMM  ──► archetype labels
                                        │
                                        ├──► rescale to (N/N★, Q/Q₀)
                                        │     └► collapse metric
                                        │
                                        └──► (formulation, archetype, N★)
                                              ├──► multinomial logit → phase diagram
                                              └──► PySR / linear  →  N★ = f(formulation)
```

Stages map to module files:

| Stage | Module | Tested in |
|---|---|---|
| Curve I/O + alignment | `src/universality/curves.py` | `tests/test_universality.py` |
| Knee detection (Kneedle + piecewise) | `src/universality/knee.py` | `tests/test_universality.py` |
| FPCA + archetype clustering | `src/universality/archetype.py` | `tests/test_universality.py` |
| Scaling collapse + master curve | `src/universality/scaling.py` | `tests/test_universality.py` |
| Phase diagram + N★ regression | `src/universality/phase_diagram.py` | `tests/test_universality.py` |

Top-level scripts:

| Script | Purpose |
|---|---|
| `scripts/100_health_check.py` | The 4 pre-checks — gates the rest |
| `scripts/101_severson_dryrun.py` | Whole pipeline on Severson, single-formulation. Validates archetype + scaling code; phase diagram is a no-op. |
| `scripts/102_universality_pipeline.py` | Full pipeline on the proprietary data |
| `scripts/103_paper_figures.py` | Generates Fig 1-5 from the JSON outputs |

---

## Success criteria

**Severson dry-run** (completed 2026-05-08):

- [x] ≥ 2 archetypes recovered with BIC-justified k → k=6 PASS
- [x] Within-archetype scaling-collapse residual ≤ 5% RMS → RMS=3.78% PASS
- [x] Knee detector reproduces 80%-SOH cycle for ≥ 90% → recall=100% PASS
- [x] All `pytest tests/test_universality_*.py` green → 14/14 PASS

**Proprietary data run** (pending):

- [ ] N★ regression: R² ≥ 0.7 from formulation alone
- [ ] Phase boundary: multinomial-logit AUC ≥ 0.8 for archetype prediction
- [ ] Symbolic-regression law for N★ extrapolates to held-out configs within 20% MAE

If the run hits all three: target Joule. Misses one: ESM/npj. Misses
two: write a smaller-scale methods paper (Applied Energy / JPS).

---

## What this paper deliberately does **not** do

- No mechanism (LLI / LAM / SEI) decomposition — needs dQ/dV or post-mortem
- No causal ATE — needs process metadata (N/P ratio, porosity, etc.)
- No early-cycle prediction benchmark — orthogonal problem; Fermín-Cueto
  2020 *Nat. Energy* covers it. We cite, do not redo.
- No BMS / online inference claim — purely offline cohort analysis
- No "rank = 3 is universal" claim — we frame as "effective rank ≈ 2-3 at
  typical lab noise levels (η ∈ [1e-3, 1e-2])"
- No pure lifetime prediction claim — too saturated since Severson 2019

These are deliberate scope cuts, not oversights. State them in §Limitations.

---

## Open questions before data arrives

1. **Formula diversity:** How many distinct formulation configs exist?
2. **EOL coverage:** What fraction of cells reach 80% SOH?
3. **NDA boundaries:** Can we release (a) model weights, (b) aggregate stats ≥10 cells,
   (c) anonymised continuous formula values?
4. **Within vs between variance:** Compute eta² before designing the paper claim.

---

## Sequencing

1. **Now (no data needed):** modules + tests + Severson dry-run ✓
2. **Contact data holder** for NDA clarification (questions above)
3. **Acquire data** → run `100_health_check.py`
4. **Adjust claim** based on health check result (Joule / ESM / JPS path)
5. **Run full pipeline** (`102_…py`) → `103_paper_figures.py` for paper figures

---

## Session Log

### 2026-05-08

**Decisions made:**
- Paper 1 (rank=3 identifiability) framing abandoned — insufficient 2026 novelty.
  Existing literature (Bizeray 2019, Sulzer 2021) already established "effective rank ≈ 2-3".
  The only novel contribution in Paper 1 is the multi-rate rank lift (3→4), which alone
  is insufficient for a separate paper.
- Paper 2 (universality / archetype / phase diagram) is the primary path forward.
- "Lifetime prediction" explicitly ruled out as a claim — saturated since Severson 2019.
- η (eta) is the relative eigenvalue threshold in Fisher analysis. Physical interpretation:
  η = 1e-3 corresponds to ~3 mV voltage noise floor, which is typical lab resolution.
  Rank ≈ 2-3 at η ∈ [1e-3, 1e-2] is the honest framing.
- Formula diversity decision tree established:
  * 1 variant → archetype taxonomy + scaling (JPS, weak novelty)
  * 2 variants → 2-group comparison (JPS/JES)
  * 3-5 variants → multinomial logistic (Applied Energy)
  * ≥10 variants (continuous) → full phase diagram + N★ formula (Joule/ESM)

**Code deliverables (completed):**
- `src/universality/` — 5 modules (curves, knee, archetype, scaling, phase_diagram)
- `scripts/100_health_check.py`, `101_severson_dryrun.py`, `102_universality_pipeline.py`
- `tests/test_universality.py` — 14 tests, all pass
- Severson dry-run: k=6 archetypes, RMS=3.78%, 91% within ±5%, knee recall=100%

**Critical open questions (must answer before writing paper):**
1. How many distinct formulation configs exist in the data?
2. What fraction of cells reach 80% SOH?
3. NDA boundaries: weights / aggregate stats / continuous formula values?
4. Within-config vs between-config variance (compute eta²)?

**Next steps:**
1. Contact data holder for NDA clarification
2. Acquire data → run `100_health_check.py`
3. Determine paper claim from health check result
