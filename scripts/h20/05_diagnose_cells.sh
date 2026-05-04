#!/usr/bin/env bash
# End-to-end LMB degradation diagnosis on every cell in DATA_DIR using the
# precomputed signature library from step 02.

set -euo pipefail

LIBRARY="${LIBRARY:-outputs/diagnosis/signature_library_lmb.npz}"
DATA_DIR="${DATA_DIR:-data/raw/lmb_long_cycle}"
OUTPUT="${OUTPUT:-outputs/diagnosis/cells}"
REGRESSOR="${REGRESSOR:-ridge}"
ALPHA="${ALPHA:-0.1}"
N_REF="${N_REF:-5}"
SMOOTH_SIGMA="${SMOOTH_SIGMA:-3.0}"
BOOTSTRAP="${BOOTSTRAP:-100}"
ABLATION="${ABLATION:-1}"

mkdir -p logs/h20
LOG="logs/h20/05_diagnose_cells_$(date -u +%Y%m%dT%H%M%SZ).log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[05] library=${LIBRARY} data_dir=${DATA_DIR} regressor=${REGRESSOR}" | tee -a "${LOG}"

EXTRA=()
[[ "${ABLATION}" == "1" ]] && EXTRA+=(--ablation)

python scripts/33_diagnose_cells.py \
  --library "${LIBRARY}" \
  --data-dir "${DATA_DIR}" \
  --output "${OUTPUT}" \
  --regressor "${REGRESSOR}" \
  --alpha "${ALPHA}" \
  --n-ref-cycles "${N_REF}" \
  --smooth-sigma "${SMOOTH_SIGMA}" \
  --bootstrap "${BOOTSTRAP}" \
  "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"

echo "[05] done. results under ${OUTPUT}" | tee -a "${LOG}"
