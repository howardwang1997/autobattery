# Universality Paper — Plan & Code Map

**Working title:** *Universal scaling and degradation archetypes across 10⁴ lithium-ion cells*

**Target venue:** Joule / Nat. Comm. (best case if scaling collapse is clean) ·
Energy Storage Materials / npj Comp. Mater. (realistic floor)

**Story (one paragraph).** Battery degradation is usually treated cell-by-cell:
each curve fitted in isolation, each "knee" predicted with a bespoke model.
We invert this: given Q(N) for ~10⁴ cells spanning many electrolyte
formulations, we ask whether degradation has *universal structure* across the
formulation space. Three results: (1) discharge-capacity trajectories cluster
into a small number of **archetypes** (gradual / knee / cliff) under
functional PCA + GMM; (2) within each archetype, curves admit a
**scaling collapse** Q(N)/Q₀ = Φ(N/N★) where N★ is a single per-cell
characteristic cycle; (3) N★ is a smooth, near-closed-form function of
formulation, and the archetype boundaries form a clean **phase diagram** in
formulation space.  Validation on the public Severson 138-cell dataset
shows the archetype + scaling pipeline reproduces; the formulation-space
results require the proprietary 10⁴ dataset.

**NDA fit.** Only Q(N) per cycle and formulation features are required. No
V(t), no temperature, no process. The publishable artefacts (archetype
definitions, master curves, the closed-form for N★, the phase boundary)
do not require releasing raw data — anonymised feature axes suffice.

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
| Curve I/O + alignment | `src/universality/curves.py` | `tests/test_universality_curves.py` |
| Knee detection (Kneedle + piecewise) | `src/universality/knee.py` | `tests/test_universality_knee.py` |
| FPCA + archetype clustering | `src/universality/archetype.py` | `tests/test_universality_archetype.py` |
| Scaling collapse + master curve | `src/universality/scaling.py` | `tests/test_universality_scaling.py` |
| Phase diagram + N★ regression | `src/universality/phase_diagram.py` | `tests/test_universality_phase.py` |

Top-level scripts:

| Script | Purpose |
|---|---|
| `scripts/100_health_check.py` | The 4 pre-checks — gates the rest |
| `scripts/101_severson_dryrun.py` | Whole pipeline on Severson, single-formulation. Validates archetype + scaling code; phase diagram is a no-op. |
| `scripts/102_universality_pipeline.py` | Full pipeline on the proprietary 10K data |
| `scripts/103_paper_figures.py` | Generates Fig 1-5 from the JSON outputs |

---

## Success criteria

**Severson dry-run** (we run this now, before 10K data arrives):

- [ ] ≥ 2 archetypes recovered with BIC-justified k
- [ ] Within-archetype scaling-collapse residual ≤ 5% RMS (Q/Q₀ space)
- [ ] Knee detector reproduces published 80%-SOH cycle for ≥ 90% of cells
- [ ] All `pytest tests/test_universality_*.py` green

**10K-cell run** (ceiling depends on these):

- [ ] N★ regression: R² ≥ 0.7 from formulation alone
- [ ] Phase boundary: multinomial-logit AUC ≥ 0.8 for archetype prediction
- [ ] Symbolic-regression law for N★ extrapolates to held-out configs within 20% MAE

If the 10K run hits all three: target Joule. Misses one: ESM/npj. Misses
two: write a Severson-only methods paper (JPS).

---

## What this paper deliberately does **not** do

- No mechanism (LLI / LAM / SEI) decomposition — needs dQ/dV or post-mortem
- No causal ATE — needs process metadata
- No early-cycle prediction benchmark — orthogonal problem; Fermín-Cueto
  2020 *Nat. Energy* covers it. We cite, do not redo.
- No BMS / online inference claim — purely offline cohort analysis

These are deliberate scope cuts, not oversights. State them in §Limitations.

---

## Sequencing

1. **Now (no data needed):** modules + tests + Severson dry-run
2. **When data arrives:** `100_health_check.py` → if green, `102_…py`
3. **Paper figures:** `103_paper_figures.py` builds Fig 1-5 deterministically
   from `outputs/universality/*.json`
