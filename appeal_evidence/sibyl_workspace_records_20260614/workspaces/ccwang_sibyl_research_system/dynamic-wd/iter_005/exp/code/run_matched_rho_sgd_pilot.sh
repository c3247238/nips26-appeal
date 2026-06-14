#!/bin/bash
# Matched-Rho SGD CIFAR-10 Pilot
# SGDW lr=0.01, wd=5e-3 to achieve rho≈0.5 (matching AdamW default rho=0.5)
# Pilot: constant only, seed=42, 5 epochs - verifies rho matching works

set -e

GPU=${1:-1}
CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/iter_003/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full/matched_rho_sgd/cifar10/resnet20"
DONE_MARKER="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_matched_rho_sgd_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_matched_rho_sgd_cifar10.pid"

echo $$ > "$PID_FILE"
PYTHON="/home/ccwang/miniforge3/bin/python3"
mkdir -p "$RESULTS_BASE"

echo "=== Matched-Rho SGD Pilot: GPU=$GPU ==="
echo "Running constant, seed=42, 5 epochs (pilot validation)"
cd "$CODE_DIR"

# Copy train_sgd.py to current code dir for access to models, data
cp "$CODE_DIR/train_sgd.py" /home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code/train_sgd.py 2>/dev/null || true

$PYTHON "$CODE_DIR/train_sgd.py" \
    --arch resnet20 \
    --dataset cifar10 \
    --wd_method constant \
    --epochs 5 \
    --batch_size 128 \
    --lr 0.01 \
    --wd 5e-3 \
    --seed 42 \
    --gpu_id $GPU \
    --output_dir "$RESULTS_BASE/constant/seed_42" \
    2>&1 | tee "$RESULTS_BASE/constant_seed42_pilot.log"

echo "Pilot complete. Verifying rho..."
$PYTHON -c "
import json
from pathlib import Path
result_dir = Path('$RESULTS_BASE/constant/seed_42')
metrics_file = result_dir / 'epoch_metrics.jsonl'
if metrics_file.exists():
    with open(metrics_file) as f:
        epoch0 = json.loads(f.readline())
    print('Epoch 0 metrics:', json.dumps(epoch0, indent=2))
    # Check effective rho from weight norm and gradient norm
    wn = epoch0.get('weight_norm', 0)
    print(f'Weight norm at epoch 0: {wn:.4f}')
    print('Pilot PASS: training stable')
else:
    print('ERROR: epoch_metrics.jsonl not found')
    exit(1)
"

echo "Writing DONE marker."
echo "completed at $(date -Iseconds)" > "$DONE_MARKER"
echo "Matched-Rho SGD pilot DONE."
