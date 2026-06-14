#!/bin/bash
# VGG-16-BN experiments - architecture diversity (no skip connections)
# Tests whether Phi Invariance holds without residual connections

set -e

VENV="/home/ccwang/sibyl-research-system/.venv/bin/python3"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(dirname "$CODE_DIR")/results/vgg"
SCRIPT="$CODE_DIR/train_unified.py"
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# Focus on key methods (skip half_lambda and random_mask for efficiency)
METHODS=(constant cwd_hard swd cosine_schedule no_wd)
SEEDS=(42 123 456)
DATASETS=(cifar10 cifar100)
EPOCHS=200
LR="1e-3"
WD="5e-4"

echo "=============================================="
echo "VGG-16-BN Experiments"
echo "=============================================="
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "=============================================="

declare -a TASK_NAMES TASK_ARGS
task_idx=0

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            name="${dataset}_vgg16bn_${method}_seed${seed}"
            outdir="$RESULTS_DIR/${dataset}/vgg16_bn/${method}/seed_${seed}"

            extra_args=""
            if [ "$method" = "cosine_schedule" ]; then
                extra_args="--wd_min 0"
            elif [ "$method" = "swd" ]; then
                extra_args="--swd_sensitivity 1.0"
            fi

            TASK_NAMES[$task_idx]="$name"
            TASK_ARGS[$task_idx]="--arch vgg16_bn --dataset $dataset --wd_method $method --epochs $EPOCHS --lr $LR --wd $WD --seed $seed --lr_schedule cosine --output_dir $outdir $extra_args"
            task_idx=$((task_idx + 1))
        done
    done
done

total_tasks=$task_idx
echo "Total tasks: $total_tasks"
echo ""

completed=0
batch=0
while [ $completed -lt $total_tasks ]; do
    batch=$((batch + 1))
    batch_size=$((total_tasks - completed < NUM_GPUS ? total_tasks - completed : NUM_GPUS))
    echo "=== Batch $batch: tasks $((completed+1))-$((completed+batch_size))/$total_tasks ==="

    pids=()
    for i in $(seq 0 $((batch_size - 1))); do
        idx=$((completed + i))
        gpu_idx=$((i % NUM_GPUS))
        gpu=${GPUS[$gpu_idx]}
        name="${TASK_NAMES[$idx]}"
        args="${TASK_ARGS[$idx]}"

        echo "  [$name] -> GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu $VENV $SCRIPT --gpu_id 0 $args > "/tmp/vgg_${name}.log" 2>&1 &
        pids+=($!)
    done

    echo "  Waiting for batch $batch to complete..."
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    completed=$((completed + batch_size))
    echo "  Batch $batch done. Total completed: $completed/$total_tasks"
    echo ""
done

echo "=============================================="
echo "VGG-16-BN experiments complete!"
echo "Results in: $RESULTS_DIR"
echo "=============================================="
