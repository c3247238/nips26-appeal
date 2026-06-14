#!/bin/bash
set -e
cd /home/ccwang/sibyl_system/projects/ttt-dlm
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
exec /home/ccwang/miniforge3/envs/sibyl_ttt-dlm/bin/python -u exp/code/dal/train_dal_linear.py
