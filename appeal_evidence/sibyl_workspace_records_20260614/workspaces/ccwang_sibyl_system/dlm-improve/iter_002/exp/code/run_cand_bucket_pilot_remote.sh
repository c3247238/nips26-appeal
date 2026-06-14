#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ccwang/sibyl_system/projects/dlm-improve"
RESULTS_DIR="${PROJECT_ROOT}/exp/results"
SCRIPT_PATH="${PROJECT_ROOT}/exp/code/cand_bucket_pilot.py"
LOG_PATH="${RESULTS_DIR}/cand_bucket_pilot.launch.log"
TASK_ID="cand_bucket_pilot"
GPU_IDS="${GPU_IDS:-0}"
REMOTE_ENV_CMD="${REMOTE_ENV_CMD:-/home/ccwang/sibyl_system/miniconda3/bin/conda run -n sibyl_dlm-improve}"

mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

rm -f \
  "${RESULTS_DIR}/${TASK_ID}.pid" \
  "${RESULTS_DIR}/${TASK_ID}_DONE" \
  "${RESULTS_DIR}/${TASK_ID}_PROGRESS.json" \
  "${RESULTS_DIR}/${TASK_ID}.json"

nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" bash -lc "${REMOTE_ENV_CMD} python \"${SCRIPT_PATH}\"" > "${LOG_PATH}" 2>&1 < /dev/null &
launcher_pid=$!
echo "${launcher_pid}" > "${RESULTS_DIR}/cand_bucket_pilot_launcher.pid"
echo "launched ${TASK_ID} with launcher pid=${launcher_pid} on gpu=${GPU_IDS}"
