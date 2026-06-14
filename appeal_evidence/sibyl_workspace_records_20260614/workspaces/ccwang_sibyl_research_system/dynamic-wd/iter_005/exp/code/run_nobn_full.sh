#!/bin/bash
# ResNet-20-NoBN CIFAR-10 Full Experiment
# 3 methods x 3 seeds = 9 runs, 200 epochs

GPU=${1:-7}
CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full/nobn/cifar10/resnet20_nobn"
DONE_MARKER="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_nobn_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_nobn_cifar10.pid"

echo $$ > "$PID_FILE"
PYTHON="/home/ccwang/miniforge3/bin/python3"
mkdir -p "$RESULTS_BASE"

echo "=== ResNet-20-NoBN CIFAR-10 Full: GPU=$GPU ==="
for METHOD in constant cwd_hard no_wd; do
    for SEED in 42 123 456; do
        echo "Method=$METHOD Seed=$SEED"
        OUT_DIR="$RESULTS_BASE/$METHOD/seed_$SEED"
        mkdir -p "$OUT_DIR"
        if [ -f "$OUT_DIR/_DONE" ]; then
            echo "  Already done, skipping"
            continue
        fi
        cd "$CODE_DIR"
        $PYTHON train_unified.py \
            --arch resnet20_nobn \
            --dataset cifar10 \
            --wd_method $METHOD \
            --epochs 200 \
            --batch_size 128 \
            --lr 5e-4 \
            --wd 5e-4 \
            --seed $SEED \
            --gpu_id $GPU \
            --output_dir "$OUT_DIR" \
            2>&1 | tee "$OUT_DIR/train.log"
    done
done

echo "All NoBN runs done."
echo "completed at $(date -Iseconds)" > "$DONE_MARKER"
