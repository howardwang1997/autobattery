#!/bin/bash
# Master pipeline — runs remaining tasks after current ones finish
# Phase A (running): CNF, Ensemble, 2000-cell gen, Exp eval, Noisy Bayesian
# Phase B (after): Bayesian on experimental, Multi-chem gen, Final results

set -e
source /root/miniconda3/bin/activate autobattery
cd /root/autobattery

echo "=== Phase B Pipeline Started: $(date) ==="

# Wait for GPU to be free (ensemble/noisy bayesian to finish)
echo "Waiting for GPU tasks to finish..."
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | grep -q '[1-9]'; do
    sleep 60
done
echo "GPU free at $(date)"

# B1: Bayesian diagnosis on experimental data
echo "=== B1: Bayesian on Experimental ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/66_bayesian_experimental.py \
    --exp-data data/experimental/experimental_cycling.h5 \
    --ckpt outputs/bayesian/noisy/noisy_5mv.pt \
    --output outputs/bayesian/experimental/ 2>&1 | tee -a logs/pipeline_phaseB.log

# B2: Generate more multi-chem cycling data
echo "=== B2: Multi-chem 1000 cells ==="
python3 scripts/52_gen_multi_chem_cycling.py 2>&1 | tee -a logs/pipeline_phaseB.log || true

# B3: Generate noisy degradation data
echo "=== B3: Noisy degradation data ==="
python3 << 'PYEOF'
import numpy as np, h5py
with h5py.File('data/fullfield/fullfield_lfp_degradation.h5', 'r') as f:
    V = f['V'][:]; params = f['params'][:]; cr = f['c_rates'][:]
rng = np.random.RandomState(99)
for noise in [5, 10, 50]:
    Vn = V + rng.normal(0, noise/1000, V.shape).astype(np.float32)
    with h5py.File(f'data/fullfield/noisy_{noise}mv.h5', 'w') as f:
        f.create_dataset('V', data=Vn); f.create_dataset('params', data=params); f.create_dataset('c_rates', data=cr)
    print(f'  noisy_{noise}mv.h5 saved')
PYEOF

# B4: Early-cycle prediction on LFP 2000 cells (if generated)
echo "=== B4: Early-cycle on 2000 LFP cells ==="
if [ -f data/synthetic_cycling/lfp_2000.h5 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 scripts/51_early_cycle_predict.py \
        --data data/synthetic_cycling/lfp_2000.h5 \
        --n-early 5 --epochs 200 --model cnn --batch-size 64 2>&1 | tee -a logs/pipeline_phaseB.log || true
fi

# B5: Final results table
echo "=== B5: Final Results ==="
python3 scripts/68_results_table.py --output outputs/final_results.json 2>&1 | tee -a logs/pipeline_phaseB.log

echo "=== Phase B Pipeline Completed: $(date) ==="
