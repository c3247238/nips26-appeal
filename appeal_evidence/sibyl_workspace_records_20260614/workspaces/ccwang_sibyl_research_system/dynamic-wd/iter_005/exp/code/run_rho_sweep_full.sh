#!/bin/bash
# Rho Sweep Full Experiments (low and high rho)
# rho_low: wd=5e-5, 4 methods x 3 seeds
# rho_high: wd=5e-3, 4 methods x 3 seeds, 200 epochs

GPU=${1:-7}
CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full/rho_sweep/cifar10"
DONE_MARKER_LOW="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_rho_low_cifar10_DONE"
DONE_MARKER_HIGH="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_rho_high_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_rho_sweep_cifar10.pid"

echo $$ > "$PID_FILE"
PYTHON="/home/ccwang/miniforge3/bin/python3"
mkdir -p "$RESULTS_BASE/rho_low" "$RESULTS_BASE/rho_high"

echo "=== Rho Sweep Full: GPU=$GPU ==="

echo "--- rho_low (wd=5e-5) ---"
for METHOD in constant cwd_hard half_lambda no_wd; do
    for SEED in 42 123 456; do
        OUT_DIR="$RESULTS_BASE/rho_low/$METHOD/seed_$SEED"
        mkdir -p "$OUT_DIR"
        [ -f "$OUT_DIR/_DONE" ] && echo "  $METHOD/seed_$SEED: already done" && continue
        echo "rho_low Method=$METHOD Seed=$SEED"
        cd "$CODE_DIR"
        $PYTHON train_unified.py \
            --arch resnet20 --dataset cifar10 --wd_method $METHOD \
            --epochs 200 --batch_size 128 --lr 1e-3 --wd 5e-5 \
            --seed $SEED --gpu_id $GPU \
            --output_dir "$OUT_DIR" \
            2>&1 | tee "$OUT_DIR/train.log"
    done
done
echo "completed at $(date -Iseconds)" > "$DONE_MARKER_LOW"

echo "--- rho_high (wd=5e-3) ---"
for METHOD in constant cwd_hard half_lambda no_wd; do
    for SEED in 42 123 456; do
        OUT_DIR="$RESULTS_BASE/rho_high/$METHOD/seed_$SEED"
        mkdir -p "$OUT_DIR"
        [ -f "$OUT_DIR/_DONE" ] && echo "  $METHOD/seed_$SEED: already done" && continue
        echo "rho_high Method=$METHOD Seed=$SEED"
        cd "$CODE_DIR"
        $PYTHON train_unified.py \
            --arch resnet20 --dataset cifar10 --wd_method $METHOD \
            --epochs 200 --batch_size 128 --lr 1e-3 --wd 5e-3 \
            --seed $SEED --gpu_id $GPU \
            --output_dir "$OUT_DIR" \
            2>&1 | tee "$OUT_DIR/train.log"
    done
done
echo "completed at $(date -Iseconds)" > "$DONE_MARKER_HIGH"

echo "All Rho Sweep runs done."
