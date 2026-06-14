#!/bin/bash
set -euo pipefail

WORKSPACE="/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current"
OWNER_ID="${1:?owner id required}"

exec /Users/cwan0785/sibyl-system/workspaces/dlm-improve/.venv/bin/python3 \
  /Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/exp/experiment_supervisor_sidecar.py \
  "$WORKSPACE" \
  PILOT \
  cs8000d \
  /home/ccwang/sibyl_system \
  "/home/ccwang/sibyl_system/miniconda3/bin/conda run -n sibyl_dlm-improve" \
  run_dnb64_baseline,run_dnb84_matched_compute \
  300 \
  600 \
  2000 \
  4 \
  true \
  25 \
  "$OWNER_ID"
