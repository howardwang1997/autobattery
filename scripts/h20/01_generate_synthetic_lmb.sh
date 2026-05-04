#!/usr/bin/env bash
# Generate the LMB synthetic dataset under the new plating-aware config.
# CPU-bound (PyBaMM). Default uses 32 workers; override via NUM_WORKERS.

set -euo pipefail

CONFIG="${CONFIG:-configs/lmb.yaml}"
OUTPUT="${OUTPUT:-data/synthetic}"
NUM_WORKERS="${NUM_WORKERS:-32}"
SEED="${SEED:-42}"

mkdir -p logs/h20
LOG="logs/h20/01_generate_synthetic_lmb_$(date -u +%Y%m%dT%H%M%SZ).log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[01] config=${CONFIG} output=${OUTPUT} workers=${NUM_WORKERS}" | tee -a "${LOG}"

python scripts/01_generate_synthetic.py \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" 2>&1 | tee -a "${LOG}"

echo "[01] dataset written; verify with ls -lh ${OUTPUT}/synthetic_lmb.npz" | tee -a "${LOG}"
