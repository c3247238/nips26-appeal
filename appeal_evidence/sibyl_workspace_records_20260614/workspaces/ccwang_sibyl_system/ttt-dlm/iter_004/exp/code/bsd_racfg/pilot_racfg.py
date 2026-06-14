"""
RACFG Pilot — Stability-Guided Re-masking on Countdown-16.

Task: pilot_racfg
GPU: 2 (single)
Mode: PILOT (16 samples, seed 42)

Tests:
  1. RACFG generation with threshold_70_30 temporal scheduling
  2. Stability scores show meaningful variance across positions
  3. Accuracy >= vanilla baseline
  4. No degeneration (rep-3 < 2x vanilla)

Also runs vanilla baseline for comparison.

Pass criteria:
  - Accuracy >= vanilla
  - Stability scores show meaningful variance
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics,
    compute_per_sample_metrics, vanilla_generate, format_results,
    save_results, print_qualitative_samples, print_comparison_table,
    check_degeneration, PROJECT_DIR, MASK_TOKEN_ID,
)
from bsd_racfg.racfg import RACFGConfig, racfg_generate

TASK_ID = "pilot_racfg"
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/pilots")
N_SAMPLES = 16
SEED = 42


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def run_vanilla_baseline(model, tokenizer, problems, device):
    """Run vanilla baseline on all problems."""
    print(f"\n{'='*60}")
    print(f"  Vanilla Baseline ({len(problems)} samples)")
    print(f"{'='*60}")

    per_sample = []
    gen_times = []
    texts = []

    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)

        text, elapsed, diag = vanilla_generate(model, tokenizer, prompt, device)
        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        per_sample.append({
            "idx": idx,
            "target": problem["target"],
            "numbers": problem["numbers"],
            "generated_text": text,
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "gen_time_s": round(elapsed, 2),
            **metrics,
        })
        gen_times.append(elapsed)
        texts.append(text)

        status = "OK" if verification["is_correct"] else "WRONG"
        eq_str = verification.get('extracted_equation') or 'N/A'
        print(f"  [{idx:2d}] {status} | target={problem['target']} | "
              f"eq={eq_str[:40]} | {elapsed:.1f}s")

    n_correct = sum(1 for s in per_sample if s["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)

    result = format_results(
        method="vanilla",
        benchmark=f"Countdown-{len(problems)}",
        n_samples=len(problems),
        seed=SEED,
        accuracy=accuracy,
        n_correct=n_correct,
        diversity=diversity,
        gen_times=gen_times,
        per_sample=per_sample,
    )

    print(f"\n  Vanilla Accuracy: {accuracy:.1%} ({n_correct}/{len(problems)})")
    print(f"  rep-2: {diversity['rep_2']:.4f} | rep-3: {diversity['rep_3']:.4f}")
    print(f"  distinct-3: {diversity['distinct_3']:.4f}")
    print(f"  Avg gen time: {np.mean(gen_times):.1f}s")

    return result


def run_racfg_pilot(model, tokenizer, problems, device):
    """Run RACFG pilot on all problems."""
    print(f"\n{'='*60}")
    print(f"  RACFG Pilot ({len(problems)} samples)")
    print(f"  Config: remask_pct=0.10, w_base=1.0, schedule=threshold_70_30")
    print(f"{'='*60}")

    config = RACFGConfig(
        remask_pct=0.10,
        w_base=1.0,
        w_max=2.0,
        schedule_type="threshold_70_30",
        stability_ema_lambda=0.7,
        gen_len=256,
        steps=128,
        temperature=0.4,
    )

    per_sample = []
    gen_times = []
    texts = []
    all_stability_data = []

    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)

        text, elapsed, diag = racfg_generate(
            model, tokenizer, prompt, config, device, track_stability=True
        )
        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        # Collect stability data
        stability_data = diag.get("stability_data", [])
        all_stability_data.append({
            "idx": idx,
            "stability_records": stability_data,
        })

        # Count guidance steps
        n_guidance = sum(1 for s in diag["step_diagnostics"] if s.get("guidance_applied"))

        per_sample.append({
            "idx": idx,
            "target": problem["target"],
            "numbers": problem["numbers"],
            "generated_text": text,
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "gen_time_s": round(elapsed, 2),
            "n_guidance_steps": n_guidance,
            "stability_mean_final": (
                stability_data[-1]["stability_mean"] if stability_data else None
            ),
            **metrics,
        })
        gen_times.append(elapsed)
        texts.append(text)

        status = "OK" if verification["is_correct"] else "WRONG"
        eq_str = verification.get('extracted_equation') or 'N/A'
        print(f"  [{idx:2d}] {status} | target={problem['target']} | "
              f"eq={eq_str[:40]} | "
              f"guide={n_guidance} steps | {elapsed:.1f}s")

        # Progress update
        report_progress(
            TASK_ID, str(RESULTS_DIR),
            epoch=idx + 1, total_epochs=len(problems),
            metric={"accuracy_so_far": sum(1 for s in per_sample if s["is_correct"]) / len(per_sample)},
        )

    n_correct = sum(1 for s in per_sample if s["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)

    # Aggregate stability analysis
    all_stab_means = []
    all_stab_stds = []
    for sample_data in all_stability_data:
        for rec in sample_data["stability_records"]:
            all_stab_means.append(rec["stability_mean"])
            all_stab_stds.append(rec["stability_std"])

    stability_analysis = {
        "n_total_records": len(all_stab_means),
        "overall_stability_mean": float(np.mean(all_stab_means)) if all_stab_means else 0,
        "overall_stability_std": float(np.std(all_stab_means)) if all_stab_means else 0,
        "per_sample_stability_std_mean": float(np.mean(all_stab_stds)) if all_stab_stds else 0,
        "stability_variance_meaningful": (
            float(np.std(all_stab_means)) > 0.01 if all_stab_means else False
        ),
    }

    result = format_results(
        method="RACFG",
        benchmark=f"Countdown-{len(problems)}",
        n_samples=len(problems),
        seed=SEED,
        accuracy=accuracy,
        n_correct=n_correct,
        diversity=diversity,
        gen_times=gen_times,
        per_sample=per_sample,
        extra_config={
            "remask_pct": config.remask_pct,
            "w_base": config.w_base,
            "w_max": config.w_max,
            "schedule_type": config.schedule_type,
            "stability_ema_lambda": config.stability_ema_lambda,
        },
    )

    # Add stability analysis to result
    result["stability_analysis"] = stability_analysis
    result["stability_per_sample"] = all_stability_data

    print(f"\n  RACFG Accuracy: {accuracy:.1%} ({n_correct}/{len(problems)})")
    print(f"  rep-2: {diversity['rep_2']:.4f} | rep-3: {diversity['rep_3']:.4f}")
    print(f"  distinct-3: {diversity['distinct_3']:.4f}")
    print(f"  Avg gen time: {np.mean(gen_times):.1f}s")
    print(f"  Stability analysis:")
    print(f"    Overall mean: {stability_analysis['overall_stability_mean']:.4f}")
    print(f"    Overall std: {stability_analysis['overall_stability_std']:.4f}")
    print(f"    Per-sample std mean: {stability_analysis['per_sample_stability_std_mean']:.4f}")
    print(f"    Variance meaningful: {stability_analysis['stability_variance_meaningful']}")

    return result


def main():
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES maps GPU 2 to cuda:0
    start_time = datetime.now()

    print("=" * 70)
    print(f"  RACFG Pilot Experiment — Countdown-{N_SAMPLES}")
    print(f"  Task ID: {TASK_ID}")
    print(f"  Time: {start_time.isoformat()}")
    print(f"  Device: GPU 2 (mapped to {device})")
    print(f"  Seed: {SEED}")
    print("=" * 70)

    # Write PID file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        # Generate problems
        problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
        print(f"\nGenerated {len(problems)} Countdown problems")

        # Load model
        print("\nLoading Dream-7B...")
        model, tokenizer, embedding_layer = load_dream(device)

        # 1. Vanilla baseline
        vanilla_result = run_vanilla_baseline(model, tokenizer, problems, device)

        # 2. RACFG pilot
        racfg_result = run_racfg_pilot(model, tokenizer, problems, device)

        # 3. Comparison
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*60}")
        print_comparison_table([vanilla_result, racfg_result])

        # 4. Degeneration check
        vanilla_div = vanilla_result["metrics"]
        racfg_div = racfg_result["metrics"]
        warnings = check_degeneration(racfg_div, vanilla_div)
        if warnings:
            print(f"\n  DEGENERATION WARNINGS:")
            for k, v in warnings.items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  No degeneration warnings.")

        # 5. Qualitative samples
        print(f"\n--- RACFG Qualitative Samples ---")
        print_qualitative_samples(racfg_result["per_sample"], n=10)

        # 6. Pass/fail assessment
        v_acc = vanilla_result["metrics"]["accuracy"]
        r_acc = racfg_result["metrics"]["accuracy"]
        stab_meaningful = racfg_result["stability_analysis"]["stability_variance_meaningful"]

        pass_accuracy = r_acc >= v_acc
        pass_stability = stab_meaningful
        overall_pass = pass_accuracy and pass_stability

        verdict = {
            "overall": "GO" if overall_pass else "NO-GO",
            "accuracy_pass": pass_accuracy,
            "stability_pass": pass_stability,
            "vanilla_accuracy": v_acc,
            "racfg_accuracy": r_acc,
            "accuracy_delta_pp": round((r_acc - v_acc) * 100, 1),
            "stability_variance_meaningful": stab_meaningful,
            "degeneration_warnings": warnings,
        }

        print(f"\n{'='*60}")
        print(f"  PILOT VERDICT: {verdict['overall']}")
        print(f"{'='*60}")
        print(f"  Accuracy pass (RACFG >= vanilla): {pass_accuracy}")
        print(f"    vanilla={v_acc:.1%}, RACFG={r_acc:.1%}, delta={verdict['accuracy_delta_pp']:+.1f}pp")
        print(f"  Stability pass (meaningful variance): {pass_stability}")
        if warnings:
            print(f"  Degeneration warnings: {list(warnings.keys())}")
        print(f"{'='*60}")

        # 7. Save results
        combined_result = {
            "task_id": TASK_ID,
            "mode": "PILOT",
            "benchmark": f"Countdown-{N_SAMPLES}",
            "seed": SEED,
            "timestamp": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "verdict": verdict,
            "vanilla": vanilla_result,
            "racfg": racfg_result,
        }

        out_file = RESULTS_DIR / f"racfg_pilot_countdown{N_SAMPLES}.json"
        save_results(combined_result, str(out_file))

        # Also save to expected output location
        expected_out = RESULTS_DIR / "pilot_racfg.json"
        save_results(combined_result, str(expected_out))

        end_time = datetime.now()
        elapsed_min = (end_time - start_time).total_seconds() / 60

        # Mark done
        mark_task_done(
            TASK_ID, str(RESULTS_DIR),
            status="success",
            summary=f"RACFG pilot {verdict['overall']}: acc={r_acc:.1%} vs vanilla={v_acc:.1%}, "
                    f"delta={verdict['accuracy_delta_pp']:+.1f}pp, "
                    f"stability_meaningful={stab_meaningful}, "
                    f"elapsed={elapsed_min:.1f}min",
        )

        print(f"\nTotal elapsed: {elapsed_min:.1f} minutes")
        return overall_pass

    except Exception as e:
        traceback.print_exc()
        mark_task_done(
            TASK_ID, str(RESULTS_DIR),
            status="failed",
            summary=f"Error: {type(e).__name__}: {e}",
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
