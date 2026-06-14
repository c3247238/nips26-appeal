#!/bin/bash
# Launch tier1_resnet18_cifar10_full experiment on GPU 3
export CUDA_VISIBLE_DEVICES=3
cd /home/qhxie/sibyl_system/projects/augmentation-order
source /home/qhxie/miniforge3/etc/profile.d/conda.sh
conda activate sibyl_augmentation-order
exec python exp/tier1_resnet18_cifar10_full.py
