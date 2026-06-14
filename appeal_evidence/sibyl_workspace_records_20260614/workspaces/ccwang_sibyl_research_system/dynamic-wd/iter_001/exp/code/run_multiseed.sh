#!/bin/bash
# Multi-seed experiment runner for iteration 1
# Runs key tier1 experiments with seeds 123 and 456 on free GPUs
# (seed=42 results already exist from iteration 0)

set -e

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results"
PYTHON="/home/ccwang/sibyl-research-system/.venv/bin/python3"

echo "=========================================="
echo "Multi-seed experiments - Iteration 1"
echo "Started: $(date)"
echo "=========================================="

# GPU allocation: 0, 2, 3 are free
# Seed 123: tier1_fixed_wd_grid (GPU 0) + tier1_aadwd_variants (GPU 2) + tier1_stagewise_cwd (GPU 3)
# Then seed 456: same 3 tasks when GPUs free up

for SEED in 123 456; do
    SEED_DIR="${RESULTS_BASE}/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    echo ""
    echo "=== Seed ${SEED} ==="

    # tier1_fixed_wd_grid on GPU 0
    echo "[$(date)] Launching tier1_fixed_wd_grid seed=${SEED} on GPU 0"
    CUDA_VISIBLE_DEVICES=0 nohup $PYTHON "$CODE_DIR/run_full.py" \
        --task tier1_fixed_wd_grid \
        --gpu 0 \
        --seed $SEED \
        --results_dir "$SEED_DIR" \
        > "${SEED_DIR}/tier1_fixed_wd_grid.log" 2>&1 &
    PID1=$!
    echo $PID1 > "${SEED_DIR}/tier1_fixed_wd_grid.pid"
    echo "  PID: $PID1"

    # tier1_aadwd_variants on GPU 2
    echo "[$(date)] Launching tier1_aadwd_variants seed=${SEED} on GPU 2"
    CUDA_VISIBLE_DEVICES=2 nohup $PYTHON "$CODE_DIR/run_full.py" \
        --task tier1_aadwd_variants \
        --gpu 0 \
        --seed $SEED \
        --results_dir "$SEED_DIR" \
        > "${SEED_DIR}/tier1_aadwd_variants.log" 2>&1 &
    PID2=$!
    echo $PID2 > "${SEED_DIR}/tier1_aadwd_variants.pid"
    echo "  PID: $PID2"

    # tier1_stagewise_cwd on GPU 3
    echo "[$(date)] Launching tier1_stagewise_cwd seed=${SEED} on GPU 3"
    CUDA_VISIBLE_DEVICES=3 nohup $PYTHON "$CODE_DIR/run_full.py" \
        --task tier1_stagewise_cwd \
        --gpu 0 \
        --seed $SEED \
        --results_dir "$SEED_DIR" \
        > "${SEED_DIR}/tier1_stagewise_cwd.log" 2>&1 &
    PID3=$!
    echo $PID3 > "${SEED_DIR}/tier1_stagewise_cwd.pid"
    echo "  PID: $PID3"

    # Wait for all 3 tasks of this seed to complete before starting next seed
    echo "[$(date)] Waiting for seed ${SEED} tasks to finish..."
    wait $PID1 $PID2 $PID3
    echo "[$(date)] Seed ${SEED} tasks completed."
done

# Write overall completion marker
echo "[$(date)] All multi-seed experiments completed!" > "${RESULTS_BASE}/multiseed_DONE"
echo ""
echo "=========================================="
echo "All experiments completed: $(date)"
echo "=========================================="
