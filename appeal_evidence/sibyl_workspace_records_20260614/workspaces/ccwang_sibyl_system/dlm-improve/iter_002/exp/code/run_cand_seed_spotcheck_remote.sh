#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ccwang/sibyl_system/projects/dlm-improve"
RESULTS_DIR="${PROJECT_ROOT}/exp/results"
SCRIPT_PATH="${PROJECT_ROOT}/exp/code/cand_seed_spotcheck.py"
LOG_PATH="${RESULTS_DIR}/cand_seed_spotcheck.launch.log"

mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"
nohup bash -lc "cd ${PROJECT_ROOT} && export CUDA_VISIBLE_DEVICES=0,1 && /home/ccwang/sibyl_system/miniconda3/bin/conda run -n sibyl_dlm-improve python -u ${SCRIPT_PATH}" > "${LOG_PATH}" 2>&1 &
launcher_pid=$!
echo "${launcher_pid}" > "${RESULTS_DIR}/cand_seed_spotcheck_launcher.pid"
echo "launched cand_seed_spotcheck with launcher pid=${launcher_pid}"
