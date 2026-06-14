#!/bin/bash
# M3b: Progressive Signal Enrichment Analysis
# GPU: single GPU, assigned via CUDA_VISIBLE_DEVICES
# Task: progressive_signal

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${1:-6}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

REMOTE_BASE="/home/ccwang/sibyl_system"
PROJECT_DIR="${REMOTE_BASE}/projects/ttt-dlm"
CODE_DIR="${PROJECT_DIR}/exp/code/dal"
RESULTS_DIR="${PROJECT_DIR}/exp/results"
CONDA_RUN="/home/ccwang/miniforge3/bin/conda run -n sibyl_ttt-dlm --no-capture-output"

TASK_ID="progressive_signal"
LOG_FILE="${RESULTS_DIR}/${TASK_ID}.log"

echo "=========================================="
echo "M3b: Progressive Signal Enrichment (H3)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Time: $(date -Iseconds)"
echo "Log: ${LOG_FILE}"
echo "=========================================="

# Write PID
echo $$ > "${RESULTS_DIR}/${TASK_ID}.pid"

# Run the analysis
exec ${CONDA_RUN} python3 "${CODE_DIR}/progressive_signal.py" \
    2>&1 | tee "${LOG_FILE}"
