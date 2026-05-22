# Plan: Analytical Proof of Low-Rank Structure

## Goal
Rigorously prove (not just numerically observe) that SEI, t+, R_mult form a degenerate triplet
in the SPM/DFN voltage equation, explaining the low Fisher Information rank.

## Strategy
Three-layer evidence:
1. **SPM analytical derivation** — derive ∂V/∂θ symbolically, show linear dependence
2. **DFN numerical verification** — compute Jacobian via PyBaMM AD, confirm same structure
3. **Existing experimental confirmation** — Severson + experimental_cycling PCA (already done)

## SPM Voltage Equation

```
V(t) = U_p(c_p(t)) - U_n(c_n(t)) - η_p(t) - η_n(t) - I·R_total(t)
```

Degradation parameter effects:
- D_n → affects c_n(t) through solid diffusion PDE in negative particle
- D_p → affects c_p(t) through solid diffusion PDE in positive particle
- t+ → affects electrolyte concentration (DFN only); ≈ 0 sensitivity in SPM
- SEI → adds interphase resistance R_SEI(δ_SEI) + consumes cyclable Li
- LAM_neg → reduces negative active material area a_n·A·L_n
- LAM_pos → reduces positive active material area a_p·A·L_p
- R_mult → multiplies contact/ohmic resistance

## Key Hypothesis

For constant-current discharge in SPM:
- ∂V/∂R_mult = -I·R_contact  (proportional to I, constant in time for CC)
- ∂V/∂SEI ≈ -I·(∂R_SEI/∂δ_SEI) + (∂U_n/∂c_n)·(∂c_n/∂Q_SEI)
  - Resistance part: constant × I (same form as R_mult)
  - Li consumption part: small for thin SEI, proportional to ∂U_n/∂c_n
- ∂V/∂t+ ≈ 0 in SPM (no electrolyte), but in DFN:
  - Affects electrolyte concentration → affects overpotential
  - Net effect produces similar voltage signature as resistance change

If the three sensitivities are proportional to the same function α(t),
the Jacobian columns are linearly dependent → Fisher rank < 7.

## Implementation Plan

### Script: `scripts/112_analytical_jacobian.py`
1. Build SPM model in PyBaMM with degradation parameters
2. Use PyBaMM's `calculate_sensitivities` to compute ∂V/∂θ at 100 time points
3. Build Jacobian matrix J[i,j] = ∂V(t_i)/∂θ_j
4. SVD of J → verify rank
5. Compute column correlations → verify degenerate triplet
6. Repeat for DFN model
7. Generate publication figure: Jacobian structure + column correlation

### Expected Results
- SPM: rank ≤ 3 (SEI/R_mult proportional, t+ ≈ 0, diffusion/LAM at different timescales)
- DFN: rank ≤ 4 (t+ gains sensitivity through electrolyte, but still correlated with SEI/R_mult)

### Resources
- CPU only (no GPU needed)
- ~10 minutes runtime
- Remote server: conda env autobattery

## Status
- [ ] Write `scripts/112_analytical_jacobian.py`
- [ ] Run on remote server
- [ ] Analyze results
- [ ] Update manuscript with analytical proof section
