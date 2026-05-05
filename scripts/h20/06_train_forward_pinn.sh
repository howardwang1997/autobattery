#!/usr/bin/env bash
# Forward PINN training launcher. As of Phase A this still uses the
# legacy VoltageMLP head; the --use-pde switch is wired but the PDE loss
# path is unstable until the network refactor (Phase A1) lands. Use this
# script to (1) baseline the legacy MLP without per-sim normalization
# (--no-per-sim-norm, once implemented) and (2) experiment with the
# --use-pde knob as work proceeds.

set -euo pipefail

CONFIG="${CONFIG:-configs/lmb.yaml}"
DATA="${DATA:-data/synthetic/synthetic_lmb.npz}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1000}"
USE_PDE="${USE_PDE:-0}"
PDE_WARMUP="${PDE_WARMUP:-200}"
MODEL="${MODEL:-mlp}"
NORM_MODE="${NORM_MODE:-global}"        # global | per_sim (legacy, leaky)
ADAPTIVE="${ADAPTIVE:-none}"            # none | softadapt
TAG="${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p logs/h20 outputs/checkpoints
LOG="logs/h20/06_train_forward_pinn_${TAG}.log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-autobattery}"

echo "[06] gpu=${GPU} model=${MODEL} use_pde=${USE_PDE} epochs=${EPOCHS} \
norm_mode=${NORM_MODE} adaptive=${ADAPTIVE}" | tee -a "${LOG}"

EXTRA=()
[[ "${USE_PDE}" == "1" ]] && EXTRA+=(--use-pde --pde-warmup-epochs "${PDE_WARMUP}")

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/02_train_forward.py \
  --config "${CONFIG}" \
  --data "${DATA}" \
  --gpu 0 \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --norm-mode "${NORM_MODE}" \
  --adaptive-weighting "${ADAPTIVE}" \
  "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"

echo "[06] checkpoints under outputs/checkpoints/" | tee -a "${LOG}"
echo "[06] Phase A1 recommended runs:" | tee -a "${LOG}"
echo "[06]   1. baseline:  MODEL=mlp NORM_MODE=global USE_PDE=0" | tee -a "${LOG}"
echo "[06]   2. PINN:      MODEL=pinn NORM_MODE=global USE_PDE=1 ADAPTIVE=softadapt" | tee -a "${LOG}"
echo "[06]   3. legacy:    MODEL=mlp NORM_MODE=per_sim  (reproducibility only)" | tee -a "${LOG}"
