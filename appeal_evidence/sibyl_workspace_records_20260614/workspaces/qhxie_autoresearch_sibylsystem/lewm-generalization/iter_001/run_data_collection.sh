#!/bin/bash
# Full data collection runner
set -e
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3

WORKSPACE=/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current
LOG_FILE=/tmp/data_collection_full_output.log

echo "Starting full data collection at $(date)" | tee $LOG_FILE
cd $WORKSPACE

conda run -n sibyl_lewm-generalization python3 -u exp/code/collect_data_pilot.py --full >> $LOG_FILE 2>&1
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE" | tee -a $LOG_FILE
echo "Finished at $(date)" | tee -a $LOG_FILE
