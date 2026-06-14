#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
cd /home/qhxie/sibyl_system/projects/augmentation-order
source /home/qhxie/miniforge3/etc/profile.d/conda.sh
conda activate sibyl_augmentation-order
python -u exp/baselines_cifar10_pilot.py
