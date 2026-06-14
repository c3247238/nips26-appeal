#!/bin/bash
# Pilot: NoBN ResNet-20 CIFAR-10
# Tests 3 WD methods (constant, cwd_hard, no_wd) without BatchNorm
# 5 epochs, seed=42, AdamW lr=5e-4, wd=5e-4

set -e

GPU=${1:-1}
CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilots/nobn_cifar10"
DONE_MARKER="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilot_nobn_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilot_nobn_cifar10.pid"

echo $$ > "$PID_FILE"
PYTHON="/home/ccwang/miniforge3/bin/python3"
mkdir -p "$RESULTS_BASE"

echo "=== NoBN Pilot: GPU=$GPU ==="
for METHOD in constant cwd_hard no_wd; do
    echo "Running method=$METHOD"
    cd "$CODE_DIR"
    $PYTHON train_unified.py \
        --arch resnet20_nobn \
        --dataset cifar10 \
        --wd_method $METHOD \
        --epochs 5 \
        --batch_size 128 \
        --lr 5e-4 \
        --wd 5e-4 \
        --seed 42 \
        --gpu_id $GPU \
        --output_dir "$RESULTS_BASE/$METHOD" \
        2>&1 | tee "$RESULTS_BASE/${METHOD}.log"
done

echo "All NoBN methods done. Writing DONE marker."
echo "completed at $(date -Iseconds)" > "$DONE_MARKER"
echo "NoBN pilot DONE."
