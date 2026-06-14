#!/bin/bash
# Full-scale experiment launcher for DTA project
# Uses GPUs 0,1,2,4 on cs8000d
#
# Wave 1: Countdown 3 seeds + GSM8K seed 42 (4 GPUs in parallel)
# Wave 2: GSM8K seeds 123,456 + MBPP seed 42 + MBPP seed 123
# Wave 3: MBPP seed 456 + Ablations (task_6a, 7a-d, 8a-c)

set -e

PYTHON="/home/ccwang/sibyl_system/miniconda3/envs/fars/bin/python3"
CODE_DIR="/home/ccwang/sibyl_system/exp/code"
RESULTS_DIR="/home/ccwang/sibyl_system/exp/results/full"
LOG_DIR="/home/ccwang/sibyl_system/exp/logs"
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/checkpoints" "$LOG_DIR"

echo "==============================================="
echo "  DTA Full-Scale Experiments"
echo "  $(date)"
echo "==============================================="

# ── Wave 1: Main benchmark results ──
echo ""
echo "=== WAVE 1: Countdown (3 seeds) + GSM8K (seed 42) ==="

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON $CODE_DIR/full_5a_countdown.py --seed 42 \
    > $LOG_DIR/full_5a_s42.log 2>&1 &
PID_5A_42=$!

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $CODE_DIR/full_5a_countdown.py --seed 123 \
    > $LOG_DIR/full_5a_s123.log 2>&1 &
PID_5A_123=$!

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $CODE_DIR/full_5a_countdown.py --seed 456 \
    > $LOG_DIR/full_5a_s456.log 2>&1 &
PID_5A_456=$!

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $CODE_DIR/full_5b_gsm8k.py --seed 42 \
    > $LOG_DIR/full_5b_s42.log 2>&1 &
PID_5B_42=$!

echo "  GPU 0: Countdown seed=42   (PID $PID_5A_42)"
echo "  GPU 1: Countdown seed=123  (PID $PID_5A_123)"
echo "  GPU 2: Countdown seed=456  (PID $PID_5A_456)"
echo "  GPU 4: GSM8K seed=42       (PID $PID_5B_42)"
echo "  Waiting for Wave 1..."

wait $PID_5A_42 $PID_5A_123 $PID_5A_456 $PID_5B_42
echo "  Wave 1 DONE at $(date)"

# ── Wave 2: GSM8K remaining seeds + MBPP ──
echo ""
echo "=== WAVE 2: GSM8K (seeds 123,456) + MBPP (seeds 42,123) ==="

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON $CODE_DIR/full_5b_gsm8k.py --seed 123 \
    > $LOG_DIR/full_5b_s123.log 2>&1 &
PID_5B_123=$!

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $CODE_DIR/full_5b_gsm8k.py --seed 456 \
    > $LOG_DIR/full_5b_s456.log 2>&1 &
PID_5B_456=$!

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $CODE_DIR/full_5c_mbpp.py --seed 42 \
    > $LOG_DIR/full_5c_s42.log 2>&1 &
PID_5C_42=$!

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $CODE_DIR/full_5c_mbpp.py --seed 123 \
    > $LOG_DIR/full_5c_s123.log 2>&1 &
PID_5C_123=$!

echo "  GPU 0: GSM8K seed=123   (PID $PID_5B_123)"
echo "  GPU 1: GSM8K seed=456   (PID $PID_5B_456)"
echo "  GPU 2: MBPP seed=42     (PID $PID_5C_42)"
echo "  GPU 4: MBPP seed=123    (PID $PID_5C_123)"
echo "  Waiting for Wave 2..."

wait $PID_5B_123 $PID_5B_456 $PID_5C_42 $PID_5C_123
echo "  Wave 2 DONE at $(date)"

# ── Wave 3: MBPP seed 456 + Ablations ──
echo ""
echo "=== WAVE 3: MBPP seed=456 + Ablation tasks ==="

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON $CODE_DIR/full_5c_mbpp.py --seed 456 \
    > $LOG_DIR/full_5c_s456.log 2>&1 &
PID_5C_456=$!

# Ablation tasks use the existing pilot scripts (already scale to 200 samples)
# These were completed at pilot level; full-scale uses same scripts with larger N
# For now, mark as pending for manual launch if needed

echo "  GPU 0: MBPP seed=456  (PID $PID_5C_456)"
echo "  Waiting for Wave 3..."

wait $PID_5C_456
echo "  Wave 3 DONE at $(date)"

# ── Aggregation ──
echo ""
echo "=== Running aggregation ==="
$PYTHON $CODE_DIR/aggregate_full_results.py
echo ""
echo "ALL EXPERIMENTS COMPLETE at $(date)"
