#!/bin/bash
# Auto-restart wrapper for train_dal_mlp.py
# Handles CUBLAS errors by restarting from latest checkpoint
# Exit code 42 = restart requested (CUDA error)

set -u
cd /home/ccwang/sibyl_system/projects/ttt-dlm
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUBLAS deterministic workspace to avoid INVALID_VALUE errors
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=0

MAX_RESTARTS=50
RESTART_COUNT=0

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "=== Run attempt $((RESTART_COUNT + 1)) / $MAX_RESTARTS ==="

    # Remove DONE marker from previous CUDA error
    rm -f exp/results/train_dal_mlp_DONE

    /home/ccwang/miniforge3/envs/sibyl_ttt-dlm/bin/python -u exp/code/dal/train_dal_mlp.py
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 1 ]; then
        # Normal exit (success or failure)
        echo "Script exited with code $EXIT_CODE"
        exit $EXIT_CODE
    elif [ $EXIT_CODE -eq 42 ]; then
        # CUDA error, restart from checkpoint
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "CUDA error detected, restarting (attempt $RESTART_COUNT)..."
        sleep 5  # Brief pause to let GPU state clear
        rm -f exp/results/train_dal_mlp_DONE
    else
        echo "Unexpected exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi
done

echo "Max restarts ($MAX_RESTARTS) reached, giving up"
exit 1
