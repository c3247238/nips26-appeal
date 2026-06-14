#!/bin/bash
# Full PMP-WD experiments + instrumented reruns for iter_006
# Uses available GPUs: 1, 4, 5, 6, 7
set -e

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="$(dirname "$CODE_DIR")/results/full"
PROJECT_DIR="$(dirname "$(dirname "$CODE_DIR")")"

echo "=== Iteration 6 Full Experiments ==="
echo "Code: $CODE_DIR"
echo "Results: $RESULTS_BASE"
echo "Start: $(date)"

# ── Task 1: PMP-WD on CIFAR-10/ResNet-20 (3 seeds) ── GPU 1
run_pmpwd_cifar10() {
    local GPU=1
    echo "[GPU $GPU] Starting PMP-WD CIFAR-10 (3 seeds)..."
    for SEED in 42 123 456; do
        local OUT="$RESULTS_BASE/pmpwd/cifar10/resnet20/seed_${SEED}"
        mkdir -p "$OUT"
        if [ -f "$OUT/_DONE" ]; then
            echo "[GPU $GPU] Seed $SEED already done, skipping"
            continue
        fi
        echo "[GPU $GPU] PMP-WD CIFAR-10 seed=$SEED"
        CUDA_VISIBLE_DEVICES=$GPU python3 "$CODE_DIR/train_unified.py" \
            --arch resnet20 --dataset cifar10 --wd_method pmpwd \
            --epochs 200 --lr 1e-3 --wd 5e-4 --seed $SEED \
            --output_dir "$OUT" --gpu_id 0 2>&1 | tail -5
    done
    echo "[GPU $GPU] PMP-WD CIFAR-10 complete!"
}

# ── Task 2: PMP-WD on CIFAR-100/ResNet-20 (3 seeds) ── GPU 4
run_pmpwd_cifar100() {
    local GPU=4
    echo "[GPU $GPU] Starting PMP-WD CIFAR-100 (3 seeds)..."
    for SEED in 42 123 456; do
        local OUT="$RESULTS_BASE/pmpwd/cifar100/resnet20/seed_${SEED}"
        mkdir -p "$OUT"
        if [ -f "$OUT/_DONE" ]; then
            echo "[GPU $GPU] Seed $SEED already done, skipping"
            continue
        fi
        echo "[GPU $GPU] PMP-WD CIFAR-100 seed=$SEED"
        CUDA_VISIBLE_DEVICES=$GPU python3 "$CODE_DIR/train_unified.py" \
            --arch resnet20 --dataset cifar100 --wd_method pmpwd \
            --epochs 200 --lr 1e-3 --wd 5e-4 --seed $SEED \
            --output_dir "$OUT" --gpu_id 0 2>&1 | tail -5
    done
    echo "[GPU $GPU] PMP-WD CIFAR-100 complete!"
}

# ── Task 3: Instrumented reruns CIFAR-10 (5 methods x 3 seeds) ── GPUs 5,6,7
run_instrumented_reruns() {
    echo "[GPU 5,6,7] Starting instrumented reruns..."
    local METHODS=(constant cosine_schedule cwd_hard swd no_wd)
    local GPUS=(5 6 7)
    local GPU_IDX=0

    for METHOD in "${METHODS[@]}"; do
        for SEED in 42 123 456; do
            local OUT="$RESULTS_BASE/instrumented/cifar10/resnet20/${METHOD}/seed_${SEED}"
            mkdir -p "$OUT"
            if [ -f "$OUT/_DONE" ]; then
                echo "  $METHOD seed=$SEED already done"
                continue
            fi
            local GPU=${GPUS[$((GPU_IDX % 3))]}
            echo "  [GPU $GPU] $METHOD seed=$SEED"
            CUDA_VISIBLE_DEVICES=$GPU python3 "$CODE_DIR/train_unified.py" \
                --arch resnet20 --dataset cifar10 --wd_method "$METHOD" \
                --epochs 200 --lr 1e-3 --wd 5e-4 --seed $SEED \
                --output_dir "$OUT" --gpu_id 0 &
            GPU_IDX=$((GPU_IDX + 1))
            # Every 3 jobs, wait for them to finish
            if [ $((GPU_IDX % 3)) -eq 0 ]; then
                wait
            fi
        done
    done
    wait
    echo "[GPU 5,6,7] Instrumented reruns complete!"
}

# Write PID file for monitoring
echo $$ > "$PROJECT_DIR/exp/results/full_pmpwd_cifar10.pid"

# Run tasks in parallel
run_pmpwd_cifar10 &
PID1=$!
run_pmpwd_cifar100 &
PID2=$!
run_instrumented_reruns &
PID3=$!

echo "Launched: PMP-WD CIFAR-10 (PID $PID1), PMP-WD CIFAR-100 (PID $PID2), Instrumented (PID $PID3)"

# Wait for CIFAR tasks first
wait $PID1
echo "$(date '+%Y-%m-%dT%H:%M:%S') full_pmpwd_cifar10 done" > "$PROJECT_DIR/exp/results/full_pmpwd_cifar10_DONE"
wait $PID2
echo "$(date '+%Y-%m-%dT%H:%M:%S') full_pmpwd_cifar100 done" > "$PROJECT_DIR/exp/results/full_pmpwd_cifar100_DONE"
wait $PID3
echo "$(date '+%Y-%m-%dT%H:%M:%S') instrumented_reruns_cifar10 done" > "$PROJECT_DIR/exp/results/instrumented_reruns_cifar10_DONE"

echo "=== All experiments complete! ==="
echo "End: $(date)"
