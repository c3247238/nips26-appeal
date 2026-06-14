#!/bin/bash
set -e

# Wait for GPU 2 to have enough free memory for ResNet-101 (need ~25 GB)
echo "[$(date)] Waiting for GPU 2 to have sufficient free memory..."
MAX_WAIT=3600  # max 1 hour wait
ELAPSED=0
INTERVAL=30

while [ $ELAPSED -lt $MAX_WAIT ]; do
    FREE_MEM=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "[$(date)] GPU 2 free memory: ${FREE_MEM} MiB"
    
    if [ -n "$FREE_MEM" ] && [ "$FREE_MEM" -gt 25000 ]; then
        echo "[$(date)] GPU 2 has ${FREE_MEM} MiB free, launching ResNet-101 pilot..."
        break
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "[$(date)] Timeout waiting for GPU memory. Launching anyway with smaller batch size..."
fi

cd /home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
conda run -n sibyl_dynamic-wd python exp/code/pilot_imagenet_resnet101_v2.py \
    --gpu_id 0 \
    --epochs 2 \
    --train_shards 8 \
    --val_shards 3 \
    --max_train_samples 30000 \
    --max_val_samples 5000 \
    --timeout 1800 \
    --batch_size 256 \
    2>&1

echo "[$(date)] ResNet-101 pilot v2 completed."
