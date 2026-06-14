#!/bin/bash
# M4b: Cross-Backbone Validation on Dream-7B (PILOT)
# Uses GPUs 4,5 (DataParallel)
set -e

cd /home/ccwang/sibyl_system/projects/ttt-dlm

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=/home/ccwang/sibyl_system/projects/ttt-dlm/exp/code/dal:$PYTHONPATH

echo "Starting M4b: Dream-7B backbone eval at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

/home/ccwang/miniforge3/bin/conda run -n sibyl_ttt-dlm \
    python3 exp/code/dal/dream_backbone_eval.py \
    > exp/results/dream_backbone_eval.log 2>&1

EXIT_CODE=$?
echo "Finished at $(date) with exit code $EXIT_CODE"

# If CUBLAS error (exit 42), auto-restart
if [ $EXIT_CODE -eq 42 ]; then
    echo "CUBLAS error detected, restarting..."
    /home/ccwang/miniforge3/bin/conda run -n sibyl_ttt-dlm \
        python3 exp/code/dal/dream_backbone_eval.py \
        >> exp/results/dream_backbone_eval.log 2>&1
fi
