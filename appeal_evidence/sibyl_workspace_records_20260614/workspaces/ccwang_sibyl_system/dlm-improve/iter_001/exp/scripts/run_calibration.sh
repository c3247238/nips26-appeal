#!/bin/bash
# Run calibration study in background with proper logging
cd /home/ccwang/sibyl_system/projects/dlm-improve
export CUDA_VISIBLE_DEVICES=2

PYTHON=/home/ccwang/sibyl_system/miniconda3/envs/sibyl_dlm-improve/bin/python
SCRIPT=exp/scripts/calibration_study.py
LOG=exp/results/calibration_study.log

nohup $PYTHON $SCRIPT > $LOG 2>&1 &
PID=$!
echo "Started calibration_study with PID=$PID"
echo $PID > exp/results/calibration_study_launcher.pid

# Wait briefly to check it didn't crash immediately
sleep 5
if kill -0 $PID 2>/dev/null; then
    echo "Process is running."
else
    echo "Process exited! Check log:"
    tail -20 $LOG
fi
