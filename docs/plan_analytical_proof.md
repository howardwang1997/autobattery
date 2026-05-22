# Plan: Analytical Proof of Low-Rank Structure

## Key Results (from `scripts/112_analytical_jacobian.py`)

### SPM (rigorous analytical result)
- **Rank = 2** at ALL degradation states (pristine, mild, heavy)
- Only D_n and D_p have non-zero voltage sensitivity
- SEI, t+, LAM_neg, LAM_pos, R_mult all have **exactly zero** sensitivity
- **Physical reason**: SPM doesn't model electrolyte (t+ absent), SEI/LAM/R_mult not connected to SPM equations in Prada2013
- **Conclusion**: 5/7 parameters are structurally unidentifiable in SPM — this is rigorous

### DFN (at single operating point)
| State | η-rank(1e-3) | SEI/R_mult corr | SEI/t+ corr | t+/R_mult corr |
|-------|-------------|-----------------|-------------|----------------|
| Pristine | 7 | 0.64 | 0.59 | 0.95 |
| Mild | 5 | **1.0000** | 0.86 | 0.86 |
| Heavy | **3** | **0.9995** | 0.08 | 0.08 |

**Key insight**: 
- SEI/R_mult pair is ALWAYS correlated (≥0.64, reaches 1.0000 at mild degradation)
- At heavy degradation: rank=3 matches fullfield, but only SEI/R_mult remain degenerate (0.9995)
- t+ joins the pair only at mild degradation (0.86), decouples at heavy degradation (0.08)

### Implication for Paper
- The "degenerate triplet" from fullfield analysis is actually a **degenerate pair** (SEI/R_mult) + context-dependent t+
- SEI and R_mult both add series resistance → proportional voltage change under CC → correlation = 1.0
- This is the analytical proof: **under CC discharge, any parameter that adds series resistance produces voltage sensitivity proportional to I(t), making them indistinguishable**

## Three-Layer Evidence for Paper
1. **SPM**: rank=2, 5/7 params zero sensitivity (rigorous, analytical)
2. **DFN single-point**: SEI/R_mult corr=1.0, rank varies 3-7 by degradation state
3. **DFN fullfield (integrated)**: rank=3, all three methods agree on low-rank
4. **Experimental**: Severson eff_rank=1.22 + 164 experimental cells (pending)

## Next Steps
- [ ] Verify SEI/R_mult analytical proof: derive ∂V/∂SEI and ∂V/∂R_mult from DFN equations, show proportionality
- [ ] Run PCA on experimental_cycling.h5 (164 cells) to validate low-rank
- [ ] Update manuscript with SPM rank=2 result and degenerate pair finding
