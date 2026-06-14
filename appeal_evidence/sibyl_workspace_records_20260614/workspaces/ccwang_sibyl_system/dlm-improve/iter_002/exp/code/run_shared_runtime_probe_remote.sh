#!/usr/bin/env bash
set -euo pipefail

REMOTE_BASE="/home/ccwang/sibyl_system"
PROJECT_ROOT="${REMOTE_BASE}/projects/dlm-improve"
RESULTS_DIR="${PROJECT_ROOT}/exp/results"
SCRIPT_PATH="${PROJECT_ROOT}/exp/code/shared_runtime_probe.py"
LOG_PATH="${RESULTS_DIR}/shared_runtime_probe.launch.log"

mkdir -p "${PROJECT_ROOT}/exp/code" "${RESULTS_DIR}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

nohup /home/ccwang/sibyl_system/miniconda3/bin/conda run -n sibyl_dlm-improve python -u "${SCRIPT_PATH}" > "${LOG_PATH}" 2>&1 &
launcher_pid=$!
echo "${launcher_pid}" > "${RESULTS_DIR}/shared_runtime_probe_launcher.pid"
echo "${launcher_pid}"
