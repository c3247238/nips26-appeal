#!/bin/bash
# C1: SOTA Comparison - PILOT mode
# GPUs: 1,4 (use GPU 1 as primary)
set -e

cd /home/ccwang/sibyl_system/projects/ttt-dlm

export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Unset LD_LIBRARY_PATH to avoid Blackwell CUBLAS conflicts in tmux
unset LD_LIBRARY_PATH

echo "=== SOTA Comparison PILOT ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

/home/ccwang/miniforge3/bin/conda run -n sibyl_ttt-dlm \
    python exp/code/dal/sota_comparison.py \
    2>&1 | tee exp/results/sota_comparison.log

echo "End: $(date)"
