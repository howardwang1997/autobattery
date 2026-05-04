#!/usr/bin/env bash
# Build the LMB differential-voltage signature library from the synthetic
# dataset produced in step 01.

set -euo pipefail

CONFIG="${CONFIG:-configs/lmb.yaml}"
DATA="${DATA:-data/synthetic/synthetic_lmb.npz}"
OUTPUT="${OUTPUT:-outputs/diagnosis/signature_library_lmb.npz}"
BOOTSTRAP="${BOOTSTRAP:-200}"

mkdir -p logs/h20
LOG="logs/h20/02_signature_library_$(date -u +%Y%m%dT%H%M%SZ).log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[02] config=${CONFIG} data=${DATA} output=${OUTPUT} bootstrap=${BOOTSTRAP}" | tee -a "${LOG}"

python scripts/32_signature_library_lmb.py \
  --config "${CONFIG}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --bootstrap "${BOOTSTRAP}" 2>&1 | tee -a "${LOG}"

echo "[02] library written to ${OUTPUT}" | tee -a "${LOG}"
