#!/bin/bash
# Full experiment runner for Unified Dynamic WD Framework
# CIFAR-10/100, ResNet-20, 200 epochs, 3 seeds, 7 WD methods
# Uses 8 GPUs in parallel

set -e

VENV="/home/ccwang/sibyl-research-system/.venv/bin/python3"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(dirname "$CODE_DIR")/results/full"
SCRIPT="$CODE_DIR/train_unified.py"
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

METHODS=(constant cwd_hard swd cosine_schedule random_mask half_lambda no_wd)
SEEDS=(42 123 456)
DATASETS=(cifar10 cifar100)
EPOCHS=200
LR="1e-3"
WD="5e-4"

echo "=============================================="
echo "Unified Dynamic WD Framework - Full Experiments"
echo "=============================================="
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Epochs: $EPOCHS"
echo "GPUs: ${GPUS[*]}"
echo "=============================================="

# Build task queue
declare -a TASK_NAMES TASK_ARGS TASK_GPUS
task_idx=0

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            name="${dataset}_resnet20_${method}_seed${seed}"
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
            TASK_ARGS[$task_idx]="--arch resnet20 --dataset $dataset --wd_method $method --epochs $EPOCHS --lr $LR --wd $WD --seed $seed --lr_schedule cosine --output_dir $outdir $extra_args"
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
        CUDA_VISIBLE_DEVICES=$gpu $VENV $SCRIPT --gpu_id 0 $args > "/tmp/full_${name}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all in this batch
    echo "  Waiting for batch $batch to complete..."
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    completed=$((completed + batch_size))
    echo "  Batch $batch done. Total completed: $completed/$total_tasks"
    echo ""
done

echo "=============================================="
echo "All experiments complete!"
echo "Results in: $RESULTS_DIR"
echo "=============================================="

# Generate summary
$VENV -c "
import json, os
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
summary = {}

for dataset_dir in sorted(results_dir.iterdir()):
    if not dataset_dir.is_dir():
        continue
    dataset = dataset_dir.name
    summary[dataset] = {}

    for arch_dir in sorted(dataset_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        summary[dataset][arch] = {}

        for method_dir in sorted(arch_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            accs = []

            for seed_dir in sorted(method_dir.iterdir()):
                s_file = seed_dir / 'summary.json'
                if s_file.exists():
                    d = json.loads(s_file.read_text())
                    accs.append(d.get('best_test_acc', 0))

            if accs:
                import statistics
                mean_acc = statistics.mean(accs)
                std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
                summary[dataset][arch][method] = {
                    'mean_acc': round(mean_acc, 2),
                    'std_acc': round(std_acc, 2),
                    'n_seeds': len(accs),
                    'accs': accs,
                }

# Print table
for dataset in sorted(summary):
    for arch in sorted(summary[dataset]):
        print(f'\n{dataset} / {arch}:')
        print(f'{\"Method\":20s} {\"Mean Acc\":>10s} {\"Std\":>8s} {\"Seeds\":>6s}')
        print('-' * 50)
        for method in sorted(summary[dataset][arch]):
            d = summary[dataset][arch][method]
            print(f'{method:20s} {d[\"mean_acc\"]:10.2f}% {d[\"std_acc\"]:7.2f}% {d[\"n_seeds\"]:5d}')

# Save
with open(results_dir / 'full_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to {results_dir / \"full_summary.json\"}')
"
