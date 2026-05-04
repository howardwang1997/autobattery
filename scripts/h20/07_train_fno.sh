#!/usr/bin/env bash
# FNO surrogate training. Requires step 11_generate_fullfield.py output.
# This is Phase A optional (only needed if pursuing the surrogate-speedup
# bullet in the abstract); skip if focusing on Phase B morphology
# anchoring.

set -euo pipefail

DATA="${DATA:-data/fullfield/fullfield_lmb.h5}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-16}"
TAG="${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p logs/h20 outputs/checkpoints
LOG="logs/h20/07_train_fno_${TAG}.log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[07] gpu=${GPU} data=${DATA} epochs=${EPOCHS} batch=${BATCH}" | tee -a "${LOG}"

if [[ ! -f "${DATA}" ]]; then
  echo "[07] full-field data missing; running scripts/11_generate_fullfield.py first" | tee -a "${LOG}"
  python scripts/11_generate_fullfield.py 2>&1 | tee -a "${LOG}"
fi

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/12_train_fno.py \
  --data "${DATA}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" 2>&1 | tee -a "${LOG}"

echo "[07] FNO checkpoint written under outputs/checkpoints/" | tee -a "${LOG}"
