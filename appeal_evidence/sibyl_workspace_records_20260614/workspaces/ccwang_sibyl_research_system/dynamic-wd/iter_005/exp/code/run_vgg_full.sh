#!/bin/bash
# VGG-16-BN CIFAR-10 Full Experiment
# 7 methods x 3 seeds = 21 runs, 200 epochs each

GPU=${1:-1}
CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full/vgg16bn/cifar10"
DONE_MARKER="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_vgg16bn_cifar10_DONE"
PID_FILE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full_vgg16bn_cifar10.pid"

echo $$ > "$PID_FILE"
PYTHON="/home/ccwang/miniforge3/bin/python3"
mkdir -p "$RESULTS_BASE"

echo "=== VGG-16-BN CIFAR-10 Full: GPU=$GPU ==="
for METHOD in constant cwd_hard half_lambda cosine_schedule swd random_mask no_wd; do
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
            --arch vgg16_bn \
            --dataset cifar10 \
            --wd_method $METHOD \
            --epochs 200 \
            --batch_size 128 \
            --lr 1e-3 \
            --wd 5e-4 \
            --seed $SEED \
            --gpu_id $GPU \
            --output_dir "$OUT_DIR" \
            2>&1 | tee "$OUT_DIR/train.log"
    done
done

echo "All VGG-16-BN runs done."
echo "completed at $(date -Iseconds)" > "$DONE_MARKER"
