#!/bin/bash
# Run VGG-16-BN on CIFAR-100 with all 7 WD methods, 3 seeds each
# Reduced to 100 epochs to fit within 60 minute budget

set -e

CODE_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code"
RESULTS_DIR="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results"
PYTHON="/home/ccwang/sibyl-research-system/.venv/bin/python3"
EPOCHS=100
GPU="4"

METHODS=("NoWD" "FixedWD" "SWD" "CWD" "CPR" "CAWD" "EqWD")
SEEDS=(42 123 456)

echo "Starting cifar100_vgg16bn_full experiment at $(date)"
echo "Total runs: $((${#METHODS[@]} * ${#SEEDS[@]}))"
echo "Epochs: $EPOCHS, GPU: $GPU"

TOTAL=0
FAILED=0

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${RESULTS_DIR}/cifar100_vgg16bn_full/${METHOD}_seed${SEED}"
        mkdir -p "$OUT_DIR"

        # Build command
        CMD="$PYTHON train.py \
            --dataset cifar100 --arch vgg16bn \
            --epochs $EPOCHS --batch_size 128 \
            --lr 0.1 --lr_schedule cosine \
            --momentum 0.9 \
            --seed $SEED \
            --wd_method $METHOD \
            --wd_lambda 0.0005 \
            --output_dir $OUT_DIR"

        # EqWD has extra params
        if [ "$METHOD" = "EqWD" ]; then
            CMD="$CMD --eqwd_beta 1.0 --eqwd_ema_alpha 0.9"
        fi

        # NoWD: wd_lambda = 0
        if [ "$METHOD" = "NoWD" ]; then
            CMD="$CMD"
            # NoWD ignores wd_lambda, no change needed
        fi

        echo ""
        echo "=== Running $METHOD seed=$SEED at $(date) ==="
        echo "Output: $OUT_DIR"

        START_TIME=$(date +%s)

        if cd "$CODE_DIR" && CUDA_VISIBLE_DEVICES=$GPU $CMD 2>&1 | tee "$OUT_DIR/train.log"; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "DONE: $METHOD seed=$SEED in ${ELAPSED}s"
            TOTAL=$((TOTAL + 1))
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "FAILED: $METHOD seed=$SEED after ${ELAPSED}s"
            FAILED=$((FAILED + 1))
            TOTAL=$((TOTAL + 1))
        fi
    done
done

echo ""
echo "=== All runs complete at $(date) ==="
echo "Total: $TOTAL, Failed: $FAILED"
