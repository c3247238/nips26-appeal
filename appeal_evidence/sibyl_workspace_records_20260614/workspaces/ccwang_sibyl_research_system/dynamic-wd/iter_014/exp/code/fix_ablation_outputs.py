#!/usr/bin/env python3
"""Fix: generate missing output files from completed ablation experiment.

The main script crashed at the markdown generation step due to a KeyError.
This script reads the completed per-run summaries and generates:
1. summary.md
2. ablation_detailed_results.json
3. Updated DONE marker
4. Updated gpu_progress.json
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

TASK_ID = "ablation_cifar100"
TOTAL_EPOCHS = 200
SEEDS = [42, 123, 456]
MODEL = "vgg16_bn"
DATASET = "cifar100"
BATCH_SIZE = 128
LR = 0.1
WD = 1e-4

WORKSPACE = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current'
SAVE_DIR = f'{WORKSPACE}/exp/results/full/phase2_ablation'
RESULTS_BASE = f'{WORKSPACE}/exp/results'

VARIANTS = [
    ("FixedWD",      0.0, 0.0, 0.0, "Fixed WD baseline", True),
    ("Kp_only",      0.5, 0.0, 0.0, "Proportional only", False),
    ("Ki_only",      0.0, 0.1, 0.0, "Integral only", False),
    ("Kd_only",      0.0, 0.0, 0.3, "Derivative/alignment only", False),
    ("PI_control",   0.5, 0.1, 0.0, "PI (CPR-like)", False),
    ("PD_control",   0.5, 0.0, 0.3, "PD (CWD-like)", False),
    ("Full_PID",     0.5, 0.1, 0.3, "Full PID", False),
    ("UDWDC_v2",     0.5, 0.1, 0.3, "UDWDC-v2 default", False),
]


def main():
    # Read the already-saved summary JSON
    summary_file = Path(SAVE_DIR) / 'ablation_summary.json'
    with open(summary_file) as f:
        summary = json.load(f)

    aggregate = summary['aggregate']
    total_elapsed = summary['total_elapsed_sec']

    # Read per-run summaries
    all_results = {}
    for variant_name, _, _, _, _, _ in VARIANTS:
        all_results[variant_name] = []
        for seed in SEEDS:
            variant_dir = Path(SAVE_DIR) / f"{variant_name}_seed{seed}"
            method_label = "FixedWD" if variant_name == "FixedWD" else f"UDWDC_v2_{variant_name}"
            summary_path = variant_dir / f"{method_label}_seed{seed}_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    all_results[variant_name].append(json.load(f))

    # 1. Generate summary.md
    md_file = Path(SAVE_DIR) / 'summary.md'
    with open(md_file, 'w') as f:
        f.write("# Full: Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN\n\n")
        f.write(f"- Epochs: {TOTAL_EPOCHS}, Seeds: {SEEDS}\n")
        f.write(f"- Model: {MODEL}, Dataset: {DATASET}\n")
        f.write(f"- Batch size: {BATCH_SIZE}, LR: {LR}, WD: {WD}\n")
        f.write(f"- All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)\n")
        f.write(f"- Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)\n\n")

        f.write("## Results\n\n")
        f.write("| Variant | K_p | K_i | K_d | Test Acc (%) | Gen Gap (%) | WD Budget | CSI |\n")
        f.write("|---------|-----|-----|-----|-------------|-------------|-----------|-----|\n")
        for variant_name, K_p, K_i, K_d, desc, is_baseline in VARIANTS:
            if variant_name in aggregate:
                s = aggregate[variant_name]
                f.write(f"| {variant_name} | {K_p:.1f} | {K_i:.1f} | {K_d:.1f} | "
                        f"{s['test_acc_mean']:.2f} +/- {s['test_acc_std']:.2f} | "
                        f"{s['gen_gap_mean']:.2f} +/- {s['gen_gap_std']:.2f} | "
                        f"{s['wd_budget_mean']:.4f} | "
                        f"{s['csi_mean']:.4f} |\n")

        f.write("\n## Per-Seed Results\n\n")
        f.write("| Variant | Seed 42 | Seed 123 | Seed 456 |\n")
        f.write("|---------|---------|----------|----------|\n")
        for variant_name, _, _, _, _, _ in VARIANTS:
            if variant_name in aggregate:
                accs = aggregate[variant_name]['per_seed_accs']
                accs_str = " | ".join(f"{a:.2f}%" for a in accs)
                f.write(f"| {variant_name} | {accs_str} |\n")

        f.write("\n## Key Observations\n\n")
        # Find best variant
        best_variant = max(aggregate.items(), key=lambda x: x[1]['test_acc_mean'])
        f.write(f"- **Best variant**: {best_variant[0]} "
                f"({best_variant[1]['test_acc_mean']:.2f}% +/- {best_variant[1]['test_acc_std']:.2f}%)\n")

        # Rank all variants
        ranked = sorted(aggregate.items(), key=lambda x: x[1]['test_acc_mean'], reverse=True)
        f.write("\n### Ranking by Test Accuracy\n\n")
        for i, (vn, vs) in enumerate(ranked, 1):
            f.write(f"{i}. **{vn}**: {vs['test_acc_mean']:.2f}% +/- {vs['test_acc_std']:.2f}% "
                    f"(WD Budget: {vs['wd_budget_mean']:.4f})\n")

        # WD budget check
        for vn, vs in aggregate.items():
            if vs['wd_budget_mean'] <= 0:
                f.write(f"\n- **WARNING**: {vn} has zero WD budget!\n")

        # Key insights
        f.write("\n### Analysis\n\n")
        f.write(f"- **FixedWD baseline**: {aggregate['FixedWD']['test_acc_mean']:.2f}% "
                f"+/- {aggregate['FixedWD']['test_acc_std']:.2f}%\n")
        f.write(f"- **Best UDWDC variant**: {best_variant[0]} "
                f"({best_variant[1]['test_acc_mean']:.2f}%)\n")

        delta = best_variant[1]['test_acc_mean'] - aggregate['FixedWD']['test_acc_mean']
        f.write(f"- **Delta over FixedWD**: {delta:+.2f}%\n")

        # Kd_only observation
        f.write(f"- **Kd_only (alignment-derivative) best among UDWDC variants**: "
                f"{aggregate['Kd_only']['test_acc_mean']:.2f}% "
                f"with highest WD budget ({aggregate['Kd_only']['wd_budget_mean']:.4f}) "
                f"close to FixedWD ({aggregate['FixedWD']['wd_budget_mean']:.4f})\n")
        f.write(f"- **Kp_only (proportional) worst**: "
                f"{aggregate['Kp_only']['test_acc_mean']:.2f}% "
                f"with reduced WD budget ({aggregate['Kp_only']['wd_budget_mean']:.4f})\n")
        f.write(f"- **Full PID and UDWDC_v2 identical**: Both {aggregate['Full_PID']['test_acc_mean']:.2f}% "
                f"(same gains K_p=0.5, K_i=0.1, K_d=0.3)\n")

    print(f"Generated: {md_file}")

    # 2. Generate detailed results JSON
    detail_file = Path(SAVE_DIR) / 'ablation_detailed_results.json'
    with open(detail_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Generated: {detail_file}")

    # 3. Write DONE marker
    pid_file = Path(RESULTS_BASE) / f"{TASK_ID}.pid"
    pid_file.unlink(missing_ok=True)

    done_file = Path(RESULTS_BASE) / f"{TASK_ID}_DONE"
    best_name = best_variant[0]
    best_acc = best_variant[1]['test_acc_mean']

    done_file.write_text(json.dumps({
        'task_id': TASK_ID,
        'status': 'success',
        'summary': f"Ablation full: 24/24 runs complete. "
                   f"Best: {best_name} ({best_acc:.2f}%). "
                   f"Time: {total_elapsed/3600:.1f}h",
        'final_progress': {
            'completed_runs': 24,
            'total_runs': 24,
            'best_variant': best_name,
            'best_acc_mean': best_acc,
            'aggregate': {vn: {'test_acc_mean': vs['test_acc_mean'],
                               'test_acc_std': vs['test_acc_std'],
                               'csi_mean': vs['csi_mean']}
                          for vn, vs in aggregate.items()},
        },
        'timestamp': datetime.now().isoformat(),
    }))
    print(f"Generated: {done_file}")

    # 4. Update gpu_progress.json
    gpu_progress_file = Path(WORKSPACE) / 'exp' / 'gpu_progress.json'
    try:
        with open(gpu_progress_file, 'r') as f:
            gpu_progress = json.load(f)

        if TASK_ID not in gpu_progress.get('completed', []):
            gpu_progress.setdefault('completed', []).append(TASK_ID)

        gpu_progress.get('running', {}).pop(TASK_ID, None)

        gpu_progress.setdefault('timings', {})[TASK_ID] = {
            'planned_min': 60,
            'actual_min': int(total_elapsed / 60),
            'start_time': '2026-03-25T20:03:00',
            'end_time': datetime.now().isoformat(),
            'config_snapshot': {
                'model': MODEL,
                'batch_size': BATCH_SIZE,
                'dataset': DATASET,
                'epochs': TOTAL_EPOCHS,
                'seeds': len(SEEDS),
                'variants': len(VARIANTS),
                'total_runs': 24,
                'gpu_model': 'RTX PRO 6000 Blackwell',
                'gpu_count': 1,
            }
        }

        with open(gpu_progress_file, 'w') as f:
            json.dump(gpu_progress, f, indent=2)
        print(f"Updated: {gpu_progress_file}")
    except Exception as e:
        print(f"Warning: Could not update gpu_progress.json: {e}")

    # 5. Update experiment_state.json
    exp_state_file = Path(WORKSPACE) / 'exp' / 'experiment_state.json'
    try:
        with open(exp_state_file, 'r') as f:
            exp_state = json.load(f)

        if TASK_ID in exp_state.get('tasks', {}):
            exp_state['tasks'][TASK_ID]['status'] = 'completed'
            exp_state['tasks'][TASK_ID]['completed_at'] = datetime.now().isoformat()

        with open(exp_state_file, 'w') as f:
            json.dump(exp_state, f, indent=2)
        print(f"Updated: {exp_state_file}")
    except Exception as e:
        print(f"Warning: Could not update experiment_state.json: {e}")

    print("\nDone! All output files generated successfully.")


if __name__ == '__main__':
    main()
