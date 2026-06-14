#!/bin/bash
# Matched-Rho SGD CIFAR-10 Full Experiment
# SGDW lr=0.01, wd=5e-3 → rho≈0.5 (matching AdamW default)
# 3 methods x 3 seeds = 9 runs, 200 epochs

GPU=${1:-7}
CODE_DIR_SGD="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/iter_003/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full/matched_rho_sgd/cifar10/resnet20"
DONE_MARKER="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_matched_rho_sgd_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_matched_rho_sgd_cifar10.pid"

# PID file was already written by pilot script; update it
echo $$ > "$PID_FILE"
PYTHON="/home/ccwang/miniforge3/bin/python3"
mkdir -p "$RESULTS_BASE"

echo "=== Matched-Rho SGD CIFAR-10 Full: GPU=$GPU ==="
for METHOD in constant cwd_hard no_wd; do
    for SEED in 42 123 456; do
        OUT_DIR="$RESULTS_BASE/$METHOD/seed_$SEED"
        mkdir -p "$OUT_DIR"
        if [ -f "$OUT_DIR/_DONE" ]; then
            echo "  $METHOD/seed_$SEED: already done"
            continue
        fi
        echo "Method=$METHOD Seed=$SEED"
        cd "$CODE_DIR_SGD"
        $PYTHON train_sgd.py \
            --arch resnet20 \
            --dataset cifar10 \
            --wd_method $METHOD \
            --epochs 200 \
            --batch_size 128 \
            --lr 0.01 \
            --wd 5e-3 \
            --seed $SEED \
            --gpu_id $GPU \
            --output_dir "$OUT_DIR" \
            2>&1 | tee "$OUT_DIR/train.log"
    done
done

echo "All Matched-Rho SGD runs done."
echo "completed at $(date -Iseconds)" > "$DONE_MARKER"
