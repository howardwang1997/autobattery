# H20 runbook — Phase A LMB diagnosis pipeline

This folder contains the launchers used to execute Phase A of
`docs/plan_publication_roadmap.md` on the 4×H20 GPU box. Run them in
order from the repository root.

## Prerequisites (one-time per box)

```bash
bash scripts/h20/00_setup_env.sh
```

Creates / updates the `autobattery` conda env and verifies that
PyBaMM's lithium plating model integrates (smoke test).

## Pipeline (Phase A)

| Step | Script | Wall-time (4×H20) | Output |
|------|--------|-------------------|--------|
| 1. Synthetic LMB sweep | `01_generate_synthetic_lmb.sh` | 6–10 h CPU (multi-process) | `data/synthetic/synthetic_lmb.npz` |
| 2. Build signature library | `02_signature_library.sh` | <5 min | `outputs/diagnosis/signature_library_lmb.npz` |
| 3. PyBaMM direct-fit baseline | `03_pybamm_baseline.sh` | 1–3 h per cell, parallel over `--cell` | `outputs/baselines/pybamm_fit/<cell>/` |
| 4. Severson feature baseline | `04_severson_baseline.sh` | minutes | `outputs/baselines/severson/` |
| 5. Per-cell diagnosis | `05_diagnose_cells.sh` | ~minute per cell | `outputs/diagnosis/cells/` |
| 6. Forward-PINN (with PDE residual) training | `06_train_forward_pinn.sh` | 6–12 h on 1×H20 | `outputs/checkpoints/forward_pinn_pde_*.pt` |
| 7. FNO surrogate training (optional Phase A) | `07_train_fno.sh` | 8–16 h | `outputs/checkpoints/fno_lmb_*.pt` |

Step 6 still needs the PINN refactor (PDE loss path). For now use it as
a placeholder — the script enables the existing `--use-pde` switch but
the loss is unstable until `src/pinn/forward.py` is rewritten. Track
progress in `docs/plan_publication_roadmap.md` §4 Phase A1.

## Convention

All scripts:
* expect to be run from repo root,
* read paths from CLI flags so a different mount on the H20 box doesn't
  require code edits,
* tee logs into `logs/h20/<script>_<UTC-timestamp>.log`,
* exit non-zero on any inner step failure (`set -euo pipefail`).

## Required data layout (you provide)

```
data/raw/lmb_long_cycle/
  cell_001.xlsx            # Neware xlsx, one per cell
  cell_002.xlsx
  ...
data/landmark/manifest.json   # cryo-EM / XPS / SEM, optional in Phase A
```

Anything under `data/raw/` is gitignored; sync the cycling files onto
the H20 box separately.
