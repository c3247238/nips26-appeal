#!/bin/bash
set -e
CODE=/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code
RESULTS=/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results
PYTHON=/home/ccwang/sibyl-research-system/.venv/bin/python3

for BETA in 1.0 2.0 5.0 10.0; do
    echo "=== Running beta=$BETA ==="
    OUTDIR="$RESULTS/ablation_beta/beta_$BETA"
    if [ -f "$OUTDIR/default_results.json" ]; then
        echo "Already done, skipping."
        continue
    fi
    cd "$CODE"
    CUDA_VISIBLE_DEVICES=5 "$PYTHON" train.py \
      --dataset cifar10 --arch resnet20 --epochs 100 --batch_size 128 \
      --lr 0.1 --lr_schedule cosine --momentum 0.9 \
      --seed 42 --wd_method EqWD --wd_lambda 0.0005 \
      --eqwd_beta "$BETA" --eqwd_ema_alpha 0.9 \
      --diag_interval 50 --output_dir "$OUTDIR"
    echo "=== Done beta=$BETA ==="
done

echo "ALL BETAS DONE"
