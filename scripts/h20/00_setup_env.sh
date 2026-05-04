#!/usr/bin/env bash
# One-shot environment bootstrap for the H20 box.
# Idempotent: safe to re-run.

set -euo pipefail

ENV_NAME="${ENV_NAME:-autobattery}"
PY_VERSION="${PY_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "FATAL: conda not on PATH. Install miniconda first."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[setup] creating env ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"

# CPU + GPU dependencies. PyTorch 2.5.1 is what work-log:166-167 verified
# matches the H20 box's cuDNN 9.1 install.
pip install --upgrade pip
pip install \
  "torch==2.5.1" \
  "pybamm>=24.5" \
  numpy scipy pandas matplotlib pyyaml tqdm openpyxl h5py \
  scikit-learn xgboost \
  pytest

# Editable install of the repo so scripts can `from src import ...`.
pip install -e .

echo
echo "[setup] verifying PyBaMM lithium-plating smoke test..."
python - <<'PY'
from src.simulation.models import quick_lmb_smoke_test
res = quick_lmb_smoke_test(c_rate=0.5, t_end=1800, n_points=50)
print("smoke-test OK; voltage range %.3f .. %.3f V" % (res['voltage'].min(), res['voltage'].max()))
PY

echo
echo "[setup] checking GPU"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name}  {props.total_memory/1e9:.0f} GB")
PY

echo
echo "[setup] running unit tests"
pytest -q tests/

echo "[setup] done."
