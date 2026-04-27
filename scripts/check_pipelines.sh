#!/bin/bash
# Quick status check for all pipelines
echo "=== Pipeline Status ==="
for i in 1 2 3 4; do
    LOG=/root/autobattery/outputs/pipeline_logs/pipeline${i}.log
    if [ -f "$LOG" ]; then
        LAST=$(tail -1 "$LOG" 2>/dev/null)
        LINES=$(wc -l < "$LOG")
        echo "Pipeline $i: ${LINES} lines | Last: ${LAST}"
    else
        echo "Pipeline $i: not started"
    fi
done
echo ""
echo "=== Running Python processes ==="
ps aux | grep python3 | grep pipeline | grep -v grep | awk '{print $2, $11, $12, $13}'
echo ""
echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
