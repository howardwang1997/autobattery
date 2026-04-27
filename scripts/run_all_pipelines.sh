#!/bin/bash
# Master launcher: kick off all 4 pipelines on separate GPUs
# Usage: bash scripts/run_all_pipelines.sh

set -e
LOGDIR=/root/autobattery/outputs/pipeline_logs
mkdir -p $LOGDIR

echo "$(date): Starting all 4 pipelines..."

# Pipeline 1: LIB (GPU 0) — longest, start first
echo "Starting Pipeline 1 (LIB) on GPU 0..."
cd /root/autobattery
PYTHONPATH=/root/autobattery setsid python3 -u scripts/pipeline1_lib.py > $LOGDIR/pipeline1.log 2>&1 &
P1_PID=$!
echo "  PID: $P1_PID"

# Pipeline 2: Design Optimization (GPU 1)
echo "Starting Pipeline 2 (Design) on GPU 1..."
sleep 5
PYTHONPATH=/root/autobattery setsid python3 -u scripts/pipeline2_design.py > $LOGDIR/pipeline2.log 2>&1 &
P2_PID=$!
echo "  PID: $P2_PID"

# Pipeline 3: Fisher Analysis (GPU 2) — uses existing LMB model
echo "Starting Pipeline 3 (Fisher) on GPU 2..."
sleep 5
PYTHONPATH=/root/autobattery setsid python3 -u scripts/18_fisher_analysis.py \
    --data data/fullfield/fullfield_lmb_v2.h5 \
    --checkpoint outputs/checkpoints/fno_final.pt \
    --gpu 2 > $LOGDIR/pipeline3.log 2>&1 &
P3_PID=$!
echo "  PID: $P3_PID"

# Pipeline 4: Experimental Validation (GPU 3)
echo "Starting Pipeline 4 (Experimental) on GPU 3..."
sleep 5
PYTHONPATH=/root/autobattery setsid python3 -u scripts/pipeline4_experimental.py > $LOGDIR/pipeline4.log 2>&1 &
P4_PID=$!
echo "  PID: $P4_PID"

echo ""
echo "All pipelines started!"
echo "  P1 (LIB):    PID $P1_PID -> $LOGDIR/pipeline1.log"
echo "  P2 (Design): PID $P2_PID -> $LOGDIR/pipeline2.log"
echo "  P3 (Fisher): PID $P3_PID -> $LOGDIR/pipeline3.log"
echo "  P4 (Exp):    PID $P4_PID -> $LOGDIR/pipeline4.log"
echo ""
echo "Monitor with: tail -f $LOGDIR/pipelineN.log"
echo "Check status: bash scripts/check_pipelines.sh"
