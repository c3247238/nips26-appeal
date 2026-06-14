#!/bin/bash
# SGD + Momentum baseline experiments
# Tests whether WD method matters under SGD (non-adaptive optimizer)
# This is the P0 supplementary experiment from result_debate

set -e

VENV="/home/ccwang/sibyl-research-system/.venv/bin/python3"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(dirname "$CODE_DIR")/results/sgd_baseline"
SCRIPT="$CODE_DIR/train_sgd.py"
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

METHODS=(constant cwd_hard swd cosine_schedule random_mask half_lambda no_wd)
SEEDS=(42 123 456)
DATASETS=(cifar10 cifar100)
EPOCHS=200
LR="0.1"  # Standard SGD LR for CIFAR
WD="5e-4"

echo "=============================================="
echo "SGD Baseline Experiments"
echo "=============================================="
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Optimizer: SGD + Momentum(0.9) + Cosine LR"
echo "=============================================="

# Build task queue
declare -a TASK_NAMES TASK_ARGS
task_idx=0

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            name="${dataset}_resnet20_sgd_${method}_seed${seed}"
            outdir="$RESULTS_DIR/${dataset}/resnet20/${method}/seed_${seed}"

            extra_args=""
            if [ "$method" = "cosine_schedule" ]; then
                extra_args="--wd_min 0"
            elif [ "$method" = "swd" ]; then
                extra_args="--swd_sensitivity 1.0"
            elif [ "$method" = "random_mask" ]; then
                extra_args="--mask_prob 0.5"
            fi

            TASK_NAMES[$task_idx]="$name"
            TASK_ARGS[$task_idx]="--arch resnet20 --dataset $dataset --wd_method $method --epochs $EPOCHS --lr $LR --wd $WD --seed $seed --output_dir $outdir $extra_args"
            task_idx=$((task_idx + 1))
        done
    done
done

total_tasks=$task_idx
echo "Total tasks: $total_tasks"
echo ""

# Execute tasks in batches of NUM_GPUS
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
        CUDA_VISIBLE_DEVICES=$gpu $VENV $SCRIPT --gpu_id 0 $args > "/tmp/sgd_${name}.log" 2>&1 &
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
echo "SGD baseline experiments complete!"
echo "Results in: $RESULTS_DIR"
echo "=============================================="
