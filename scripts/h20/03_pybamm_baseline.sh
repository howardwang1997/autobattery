#!/usr/bin/env bash
# Run the PyBaMM direct-fit baseline across every cell in DATA_DIR for a
# user-supplied list of cycles. Dispatches one Python process per cell at
# a time; each fit uses scipy DE inside (CPU-bound).
#
# Usage:
#   DATA_DIR=data/raw/lmb_long_cycle CYCLES="10 100 500 1000" bash scripts/h20/03_pybamm_baseline.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw/lmb_long_cycle}"
CYCLES="${CYCLES:-10 100 500 1000}"
C_RATE="${C_RATE:-1.0}"
MAXITER="${MAXITER:-60}"
MODE="${MODE:-plating_dominant}"
PARAMETER_SET="${PARAMETER_SET:-OKane2022}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines/pybamm_fit}"

mkdir -p logs/h20
LOG="logs/h20/03_pybamm_baseline_$(date -u +%Y%m%dT%H%M%SZ).log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[03] data_dir=${DATA_DIR} cycles=${CYCLES} c_rate=${C_RATE}" | tee -a "${LOG}"

shopt -s nullglob
for f in "${DATA_DIR}"/*.xlsx "${DATA_DIR}"/*.csv; do
  cell="$(basename "${f%.*}")"
  for cyc in ${CYCLES}; do
    out="${OUT_ROOT}/${cell}"
    echo "[03] ${cell} cycle=${cyc}" | tee -a "${LOG}"
    python scripts/30_baseline_pybamm_fit.py \
      --data "${f}" \
      --cycle "${cyc}" \
      --c-rate "${C_RATE}" \
      --maxiter "${MAXITER}" \
      --mode "${MODE}" \
      --parameter-set "${PARAMETER_SET}" \
      --output "${out}" 2>&1 | tee -a "${LOG}" || \
      echo "[03] WARN: ${cell} cycle=${cyc} failed" | tee -a "${LOG}"
  done
done

echo "[03] done. results under ${OUT_ROOT}" | tee -a "${LOG}"
