#!/usr/bin/env bash
# Severson early-prediction baseline: per-cell features + leave-one-cell-out
# Random Forest (default) regression on EOL cycle.

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw/lmb_long_cycle}"
FEATURE_CYCLES="${FEATURE_CYCLES:-10 100}"
RETENTION="${RETENTION:-0.8}"
MODEL="${MODEL:-rf}"      # ridge | rf | xgb
OUTPUT="${OUTPUT:-outputs/baselines/severson}"

mkdir -p logs/h20
LOG="logs/h20/04_severson_baseline_$(date -u +%Y%m%dT%H%M%SZ).log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[04] data_dir=${DATA_DIR} feature_cycles=${FEATURE_CYCLES} model=${MODEL}" | tee -a "${LOG}"

# shellcheck disable=SC2086
python scripts/31_baseline_severson_features.py \
  --data-dir "${DATA_DIR}" \
  --feature-cycles ${FEATURE_CYCLES} \
  --retention "${RETENTION}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" 2>&1 | tee -a "${LOG}"

echo "[04] done. metrics in ${OUTPUT}/cv_metrics.json" | tee -a "${LOG}"
