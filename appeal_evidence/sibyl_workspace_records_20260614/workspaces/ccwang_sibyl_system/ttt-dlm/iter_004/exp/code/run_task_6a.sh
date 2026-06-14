#!/bin/bash
# Task 6a: Inference-Time Scaling Curve (PILOT)
# Runs 4 step counts (64, 128, 256, 512) in parallel on 4 GPUs.
# Each GPU runs all 4 methods at its assigned step count.

set -e

SCRIPT_DIR="/home/ccwang/sibyl_system/exp/code"
LOG_DIR="/home/ccwang/sibyl_system/exp/logs"
mkdir -p "$LOG_DIR"

echo "=== Task 6a: Launching scaling curve experiment ==="
echo "Time: $(date -Iseconds)"

source /home/ccwang/miniforge3/etc/profile.d/conda.sh
conda activate base

cd "$SCRIPT_DIR"

# Launch 4 step counts in parallel on 4 GPUs
CUDA_VISIBLE_DEVICES=0 python3 task_6a_scaling_curve.py --steps=64  > "$LOG_DIR/task_6a_T64.log"  2>&1 &
PID_64=$!

CUDA_VISIBLE_DEVICES=1 python3 task_6a_scaling_curve.py --steps=128 > "$LOG_DIR/task_6a_T128.log" 2>&1 &
PID_128=$!

CUDA_VISIBLE_DEVICES=2 python3 task_6a_scaling_curve.py --steps=256 > "$LOG_DIR/task_6a_T256.log" 2>&1 &
PID_256=$!

CUDA_VISIBLE_DEVICES=4 python3 task_6a_scaling_curve.py --steps=512 > "$LOG_DIR/task_6a_T512.log" 2>&1 &
PID_512=$!

echo "PIDs: T=64($PID_64) T=128($PID_128) T=256($PID_256) T=512($PID_512)"
echo "Waiting for all to finish..."

FAILED=0
for PID_NAME in "64:$PID_64" "128:$PID_128" "256:$PID_256" "512:$PID_512"; do
    STEPS=${PID_NAME%%:*}
    PID=${PID_NAME##*:}
    if wait $PID; then
        echo "  T=$STEPS completed successfully"
    else
        echo "  T=$STEPS FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== All processes finished. Failed: $FAILED/4 ==="
echo "Time: $(date -Iseconds)"

# Merge partial results into combined file
python3 -c "
import json
from pathlib import Path

results_dir = Path('/home/ccwang/sibyl_system/exp/results/pilots')
combined_table = {}
all_steps = [64, 128, 256, 512]
methods = ['vanilla', 'remdm_conf', 'dta', 'dta_remdm']

for T in all_steps:
    f = results_dir / f'task_6a_T{T}.json'
    if f.exists():
        data = json.loads(f.read_text())
        combined_table[str(T)] = data.get('results', {})
    else:
        print(f'WARNING: missing {f}')

# H3 check
dta_accs = {}
remdm_accs = {}
for T_str, method_results in combined_table.items():
    T = int(T_str)
    if 'dta' in method_results:
        dta_accs[T] = method_results['dta']['accuracy']
    if 'remdm_conf' in method_results:
        remdm_accs[T] = method_results['remdm_conf']['accuracy']

dta_128 = dta_accs.get(128, 0)
dta_256 = dta_accs.get(256, 0)
dta_512 = dta_accs.get(512, 0)
dta_still_improving = (dta_256 >= dta_128) or (dta_512 >= dta_128)

combined = {
    'task': 'task_6a',
    'mode': 'pilot',
    'model': 'Dream-v0-Instruct-7B',
    'n_samples': 16,
    'seed': 42,
    'steps_tested': all_steps,
    'methods': methods,
    'summary_table': combined_table,
    'pass_criteria': {
        'all_step_counts_ran': len(combined_table) == 4,
        'h3_dta_still_improving': dta_still_improving,
        'dta_accs': {str(k): v for k, v in sorted(dta_accs.items())},
        'remdm_accs': {str(k): v for k, v in sorted(remdm_accs.items())},
        'overall': 'GO' if dta_still_improving else 'CONDITIONAL-GO',
    },
}

out = results_dir / 'task_6a_scaling_curve.json'
out.write_text(json.dumps(combined, indent=2, default=str))
print(f'Combined results saved to {out}')

# Print summary table
print()
print('Scaling Curve Summary (Accuracy):')
print(f\"  {'T':>5} | {'Vanilla':>10} | {'ReMDM-conf':>12} | {'DTA':>10} | {'DTA+ReMDM':>12}\")
for T in all_steps:
    T_str = str(T)
    if T_str in combined_table:
        row = combined_table[T_str]
        vals = [f\"{row.get(m, {}).get('accuracy', 0):.1%}\" if m in row else 'N/A' for m in methods]
        print(f'  {T:>5} | {vals[0]:>10} | {vals[1]:>12} | {vals[2]:>10} | {vals[3]:>12}')
"

echo ""
echo "=== Task 6a complete ==="
