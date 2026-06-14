"""
BSD Pilot — Belief Vector Feasibility on Countdown-16.

Task: pilot_bsd
Mode: PILOT (16 samples, seed 42)
GPU: 1

Tests:
  1. No OOD collapse (rep-3 < 2x vanilla)
  2. Belief entropy decreases across denoising steps
  3. Outputs are well-formed
  4. If OOD collapse detected, activate fallback_beta mixing

Pass criteria: Outputs are well-formed (rep-3 < 2x vanilla) AND
               belief entropy shows decreasing trend.
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

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics,
    compute_per_sample_metrics, vanilla_generate, format_results,
    save_results, print_qualitative_samples, print_comparison_table,
    check_degeneration, PROJECT_DIR, MASK_TOKEN_ID,
)
from bsd_racfg.bsd import BSDConfig, bsd_generate


# ── Constants ──
TASK_ID = "pilot_bsd"
N_SAMPLES = 16
SEED = 42
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/pilots")
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES is set externally


def write_pid():
    """Write PID file for system recovery detection."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[pilot_bsd] PID {os.getpid()} written to {pid_file}")


def report_progress(stage, total_stages, detail=""):
    """Write progress file for system monitor."""
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": stage, "total_epochs": total_stages,
        "step": 0, "total_steps": 0,
        "loss": None,
        "metric": {"detail": detail},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    # Clean up PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    # Merge final progress
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    # Write DONE marker
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[pilot_bsd] DONE marker written: {status}")


def run_vanilla_baseline(model, tokenizer, problems, prompts):
    """Run vanilla baseline for comparison."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Vanilla Baseline ({N_SAMPLES} samples)")
    print(f"{'='*60}")

    vanilla_results = []
    vanilla_texts = []
    vanilla_times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)
        text, elapsed, diag = vanilla_generate(model, tokenizer, prompt, DEVICE)
        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        vanilla_results.append({
            "idx": idx,
            "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text,
            "gen_time_s": elapsed,
            **metrics,
        })
        vanilla_texts.append(text)
        vanilla_times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        eq_str = (verification.get('extracted_equation') or 'N/A')[:40]
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"eq={eq_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in vanilla_results if r["is_correct"])
    accuracy = n_correct / N_SAMPLES
    diversity = compute_diversity_metrics(vanilla_texts)

    print(f"\n  Vanilla: {n_correct}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")

    result = format_results(
        method="vanilla", benchmark=f"Countdown-{N_SAMPLES}",
        n_samples=N_SAMPLES, seed=SEED,
        accuracy=accuracy, n_correct=n_correct,
        diversity=diversity, gen_times=vanilla_times,
        per_sample=vanilla_results,
    )
    return result, diversity


def run_bsd_config(model, tokenizer, embedding_layer, problems, prompts,
                   config: BSDConfig, label: str):
    """Run BSD with a specific configuration."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  k_frac={config.k_frac}, alpha={config.alpha_schedule} "
          f"({config.alpha_start}->{config.alpha_end}), "
          f"tau={config.tau_start}->{config.tau_end}")
    if config.fallback_beta_start > 0:
        print(f"  fallback_beta={config.fallback_beta_start}->{config.fallback_beta_end}")
    print(f"{'='*60}")

    bsd_results = []
    bsd_texts = []
    bsd_times = []
    all_entropy_trajectories = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)
        text, elapsed, diag = bsd_generate(
            model, tokenizer, prompt, embedding_layer, config, DEVICE,
            track_entropy=True,
        )
        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        # Extract entropy info
        entropy_traj = diag.get("entropy_trajectory", [])
        belief_entropies = [e["mean_entropy"] for e in entropy_traj if e["phase"] == "belief"]
        entropy_decreasing = (
            belief_entropies[-1] < belief_entropies[0]
            if len(belief_entropies) >= 2 else None
        )

        bsd_results.append({
            "idx": idx,
            "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text,
            "gen_time_s": elapsed,
            "entropy_start": belief_entropies[0] if belief_entropies else None,
            "entropy_end": belief_entropies[-1] if belief_entropies else None,
            "entropy_decreasing": entropy_decreasing,
            **metrics,
        })
        bsd_texts.append(text)
        bsd_times.append(elapsed)
        all_entropy_trajectories.append(entropy_traj)

        status = "OK" if verification["is_correct"] else "X"
        ent_str = ""
        if belief_entropies:
            ent_str = f"ent={belief_entropies[0]:.1f}->{belief_entropies[-1]:.1f}"
        eq_str = (verification.get('extracted_equation') or 'N/A')[:35]
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"eq={eq_str} | {ent_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in bsd_results if r["is_correct"])
    accuracy = n_correct / N_SAMPLES
    diversity = compute_diversity_metrics(bsd_texts)

    # Entropy analysis
    n_decreasing = sum(1 for r in bsd_results if r.get("entropy_decreasing") is True)
    avg_ent_start = np.mean([r["entropy_start"] for r in bsd_results
                             if r["entropy_start"] is not None])
    avg_ent_end = np.mean([r["entropy_end"] for r in bsd_results
                           if r["entropy_end"] is not None])

    print(f"\n  {label}: {n_correct}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")
    print(f"  Entropy: {avg_ent_start:.2f} -> {avg_ent_end:.2f} | "
          f"decreasing={n_decreasing}/{N_SAMPLES}")

    extra_config = {
        "k_frac": config.k_frac,
        "alpha_schedule": config.alpha_schedule,
        "alpha_start": config.alpha_start,
        "alpha_end": config.alpha_end,
        "tau_start": config.tau_start,
        "tau_end": config.tau_end,
        "fallback_beta_start": config.fallback_beta_start,
        "fallback_beta_end": config.fallback_beta_end,
    }

    result = format_results(
        method=label, benchmark=f"Countdown-{N_SAMPLES}",
        n_samples=N_SAMPLES, seed=SEED,
        accuracy=accuracy, n_correct=n_correct,
        diversity=diversity, gen_times=bsd_times,
        per_sample=bsd_results, extra_config=extra_config,
    )
    result["entropy_analysis"] = {
        "n_decreasing": n_decreasing,
        "n_total": N_SAMPLES,
        "avg_entropy_start": float(avg_ent_start),
        "avg_entropy_end": float(avg_ent_end),
        "entropy_trajectories": all_entropy_trajectories,
    }
    return result, diversity


def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 4, "Loading model")

    print("=" * 70)
    print("  BSD Pilot — Belief Vector Feasibility (Countdown-16)")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, 4, "Model loaded, generating problems")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    prompts = [format_countdown_prompt(p) for p in problems]
    print(f"\nGenerated {len(problems)} Countdown problems")

    # ── Phase 1: Vanilla baseline ──
    report_progress(1, 4, "Running vanilla baseline")
    vanilla_result, vanilla_diversity = run_vanilla_baseline(model, tokenizer, problems, prompts)

    # ── Phase 2: BSD default config (k=T/2, linear alpha 0.1->0.8) ──
    report_progress(2, 4, "Running BSD default config")
    bsd_default_config = BSDConfig(
        k_frac=0.5,
        alpha_schedule="linear",
        alpha_start=0.1, alpha_end=0.8,
        tau_start=1.0, tau_end=0.3,
        fallback_beta_start=0.0, fallback_beta_end=0.0,
    )
    bsd_result, bsd_diversity = run_bsd_config(
        model, tokenizer, embedding_layer, problems, prompts,
        bsd_default_config, "BSD (k=T/2, linear alpha)"
    )

    # ── Check for OOD collapse ──
    degeneration_warnings = check_degeneration(bsd_diversity, vanilla_diversity)
    ood_collapse = "rep_3_alert" in degeneration_warnings

    bsd_fallback_result = None
    if ood_collapse:
        print(f"\n!!! OOD COLLAPSE DETECTED: {degeneration_warnings}")
        print("Activating fallback_beta mixing...")

        # ── Phase 3: BSD with fallback (graceful degradation) ──
        report_progress(3, 4, "Running BSD with fallback_beta (OOD mitigation)")
        bsd_fallback_config = BSDConfig(
            k_frac=0.5,
            alpha_schedule="linear",
            alpha_start=0.1, alpha_end=0.8,
            tau_start=1.0, tau_end=0.3,
            fallback_beta_start=0.9, fallback_beta_end=0.1,
        )
        bsd_fallback_result, bsd_fallback_diversity = run_bsd_config(
            model, tokenizer, embedding_layer, problems, prompts,
            bsd_fallback_config, "BSD-fallback (beta=0.9->0.1)"
        )
        fallback_warnings = check_degeneration(bsd_fallback_diversity, vanilla_diversity)
        bsd_fallback_result["degeneration_warnings"] = fallback_warnings
    else:
        print("\nNo OOD collapse detected. Skipping fallback test.")
        report_progress(3, 4, "No OOD collapse, skipping fallback")

    # ── Comparison ──
    report_progress(4, 4, "Computing final comparison")
    all_results = [vanilla_result, bsd_result]
    if bsd_fallback_result:
        all_results.append(bsd_fallback_result)

    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print_comparison_table(all_results)

    # Print qualitative samples
    print(f"\n{'='*70}")
    print("  QUALITATIVE SAMPLES — Vanilla")
    print_qualitative_samples(vanilla_result["per_sample"], n=5)
    print(f"\n  QUALITATIVE SAMPLES — BSD")
    print_qualitative_samples(bsd_result["per_sample"], n=10)

    # ── Pass/Fail Decision ──
    bsd_accuracy = bsd_result["metrics"]["accuracy"]
    vanilla_accuracy = vanilla_result["metrics"]["accuracy"]
    bsd_rep3 = bsd_result["metrics"]["rep_3"]
    vanilla_rep3 = vanilla_result["metrics"]["rep_3"]
    entropy_analysis = bsd_result["entropy_analysis"]
    n_decreasing = entropy_analysis["n_decreasing"]
    entropy_trend = n_decreasing >= N_SAMPLES * 0.7  # 70%+ samples show decreasing trend

    well_formed = bsd_rep3 < vanilla_rep3 * 2 + 0.01  # +0.01 for epsilon when vanilla_rep3=0

    pass_criteria = well_formed and entropy_trend
    verdict = "GO" if pass_criteria else "NO-GO"

    print(f"\n{'='*70}")
    print(f"  PILOT VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  BSD accuracy:        {bsd_accuracy:.1%} (vanilla: {vanilla_accuracy:.1%})")
    print(f"  Well-formed:         {well_formed} (rep-3: {bsd_rep3:.4f} vs vanilla: {vanilla_rep3:.4f})")
    print(f"  Entropy decreasing:  {n_decreasing}/{N_SAMPLES} ({entropy_trend})")
    print(f"  OOD collapse:        {ood_collapse}")
    if degeneration_warnings:
        for k, v in degeneration_warnings.items():
            print(f"    WARNING: {v}")

    # ── Save results ──
    pilot_summary = {
        "task_id": TASK_ID,
        "verdict": verdict,
        "pass_criteria": {
            "well_formed": well_formed,
            "entropy_decreasing_trend": entropy_trend,
            "n_entropy_decreasing": n_decreasing,
        },
        "bsd_accuracy": bsd_accuracy,
        "vanilla_accuracy": vanilla_accuracy,
        "ood_collapse": ood_collapse,
        "degeneration_warnings": degeneration_warnings,
        "total_time_s": round(time.time() - start_time, 1),
        "timestamp": datetime.now().isoformat(),
        "vanilla": vanilla_result,
        "bsd": bsd_result,
        "bsd_fallback": bsd_fallback_result,
    }

    out_file = RESULTS_DIR / "bsd_pilot_countdown16.json"
    save_results(pilot_summary, str(out_file))

    # Also write pilot summary markdown
    summary_md = RESULTS_DIR / "bsd_pilot_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# BSD Pilot Summary — Countdown-{N_SAMPLES}\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"| Metric | Vanilla | BSD (k=T/2) |")
        if bsd_fallback_result:
            f.write(f" BSD-fallback |")
        f.write(f"\n|--------|---------|-------------|")
        if bsd_fallback_result:
            f.write(f"--------------|")
        f.write(f"\n| Accuracy | {vanilla_accuracy:.1%} | {bsd_accuracy:.1%} |")
        if bsd_fallback_result:
            fb_acc = bsd_fallback_result['metrics']['accuracy']
            f.write(f" {fb_acc:.1%} |")
        f.write(f"\n| rep-3 | {vanilla_rep3:.4f} | {bsd_rep3:.4f} |")
        if bsd_fallback_result:
            fb_rep3 = bsd_fallback_result['metrics']['rep_3']
            f.write(f" {fb_rep3:.4f} |")
        f.write(f"\n| distinct-3 | {vanilla_result['metrics']['distinct_3']:.4f} | "
                f"{bsd_result['metrics']['distinct_3']:.4f} |")
        if bsd_fallback_result:
            fb_d3 = bsd_fallback_result['metrics']['distinct_3']
            f.write(f" {fb_d3:.4f} |")
        f.write(f"\n| Avg time (s) | {vanilla_result['metrics']['avg_gen_time_s']:.1f} | "
                f"{bsd_result['metrics']['avg_gen_time_s']:.1f} |")
        if bsd_fallback_result:
            fb_time = bsd_fallback_result['metrics']['avg_gen_time_s']
            f.write(f" {fb_time:.1f} |")
        f.write(f"\n\n### Entropy Analysis\n")
        f.write(f"- Start: {entropy_analysis['avg_entropy_start']:.2f}\n")
        f.write(f"- End: {entropy_analysis['avg_entropy_end']:.2f}\n")
        f.write(f"- Decreasing: {n_decreasing}/{N_SAMPLES}\n")
        f.write(f"- OOD collapse: {ood_collapse}\n")
        if degeneration_warnings:
            f.write(f"\n### Warnings\n")
            for k, v in degeneration_warnings.items():
                f.write(f"- {v}\n")
    print(f"[pilot_bsd] Summary saved to {summary_md}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    elapsed_total = time.time() - start_time
    mark_done(
        status="success",
        summary=f"BSD pilot {verdict}: accuracy={bsd_accuracy:.1%}, "
                f"entropy_trend={entropy_trend}, ood={ood_collapse}, "
                f"time={elapsed_total:.0f}s",
    )

    # Update gpu_progress.json
    gpu_progress_path = Path(f"{PROJECT_DIR}/exp/gpu_progress.json")
    try:
        progress = json.loads(gpu_progress_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in progress["completed"]:
        progress["completed"].append(TASK_ID)
    # Remove from running if present
    progress["running"].pop(TASK_ID, None)
    # Record timing
    progress["timings"][TASK_ID] = {
        "planned_min": 30,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "Dream-v0-Instruct-7B",
            "method": "BSD",
            "k_frac": 0.5,
            "alpha_schedule": "linear",
            "n_samples": N_SAMPLES,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(progress, indent=2))
    print(f"[pilot_bsd] gpu_progress.json updated")

    print(f"\n[pilot_bsd] Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    return pass_criteria


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
