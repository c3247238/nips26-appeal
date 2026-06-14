#!/bin/bash
# Pilot: Rho Sweep CIFAR-10
# Two rho regimes: rho=0.05 (wd=5e-5) and rho=5.0 (wd=5e-3)
# ResNet-20, constant WD method, 5 epochs, seed=42

set -e

GPU=${1:-1}
CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/iter_004/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilots/rho_sweep"
DONE_MARKER="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilot_rho_sweep_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilot_rho_sweep_cifar10.pid"

# Write PID
echo $$ > "$PID_FILE"

PYTHON="/home/ccwang/miniforge3/bin/python3"

mkdir -p "$RESULTS_BASE"

echo "=== Rho Sweep Pilot: GPU=$GPU, python=$PYTHON ==="
echo "Run 1: rho=0.05 (wd=5e-5)"
cd "$CODE_DIR"
$PYTHON train_unified.py \
    --arch resnet20 \
    --dataset cifar10 \
    --wd_method constant \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-3 \
    --wd 5e-5 \
    --seed 42 \
    --gpu_id $GPU \
    --output_dir "$RESULTS_BASE/rho_low" \
    2>&1 | tee "$RESULTS_BASE/rho_low.log"

echo "Run 2: rho=5.0 (wd=5e-3)"
$PYTHON train_unified.py \
    --arch resnet20 \
    --dataset cifar10 \
    --wd_method constant \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-3 \
    --wd 5e-3 \
    --seed 42 \
    --gpu_id $GPU \
    --output_dir "$RESULTS_BASE/rho_high" \
    2>&1 | tee "$RESULTS_BASE/rho_high.log"

echo "Both runs complete. Writing DONE marker."
echo "completed at $(date -Iseconds)" > "$DONE_MARKER"
echo "Rho sweep pilot DONE."
