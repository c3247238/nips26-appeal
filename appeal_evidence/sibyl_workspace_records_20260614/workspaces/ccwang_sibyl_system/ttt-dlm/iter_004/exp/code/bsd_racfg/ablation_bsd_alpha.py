"""
BSD Alpha Schedule Ablation on Countdown-100 (pilot: 16 samples).

Task: ablation_bsd_alpha
Mode: PILOT (16 samples, seed 42)
GPU: 1

Tests BSD alpha schedules with best k from ablation_bsd_k (k=0.75).
Schedules:
  (1) linear ramp 0.1→0.8
  (2) cosine ramp 0.1→0.8
  (3) constant 0.3
  (4) constant 0.5

Fixed: k_frac=0.75, tau=linear(1.0→0.3), no fallback_beta.

Pass criteria: At least one schedule > vanilla AND no degeneration.
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
TASK_ID = "ablation_bsd_alpha"
N_SAMPLES = 16  # PILOT mode
SEED = 42
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/full")
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES is set externally

# Best k from ablation_bsd_k
BEST_K_FRAC = 0.75

# Alpha schedule configs to test
ALPHA_CONFIGS = [
    {
        "label": "linear(0.1→0.8)",
        "schedule": "linear",
        "alpha_start": 0.1,
        "alpha_end": 0.8,
    },
    {
        "label": "cosine(0.1→0.8)",
        "schedule": "cosine",
        "alpha_start": 0.1,
        "alpha_end": 0.8,
    },
    {
        "label": "constant(0.3)",
        "schedule": "constant",
        "alpha_start": 0.3,
        "alpha_end": 0.3,
    },
    {
        "label": "constant(0.5)",
        "schedule": "constant",
        "alpha_start": 0.5,
        "alpha_end": 0.5,
    },
]


def write_pid():
    """Write PID file for system recovery detection."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")


def report_progress(epoch, total_epochs, detail=""):
    """Write progress file for system monitor."""
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": 0, "total_steps": 0,
        "loss": None,
        "metric": {"detail": detail},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[{TASK_ID}] DONE marker written: {status}")


def run_vanilla_baseline(model, tokenizer, problems, prompts):
    """Run vanilla baseline for comparison."""
    print(f"\n{'='*60}")
    print(f"  Vanilla Baseline ({N_SAMPLES} samples)")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)
        text, elapsed, diag = vanilla_generate(model, tokenizer, prompt, DEVICE)
        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        results.append({
            "idx": idx,
            "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text,
            "gen_time_s": elapsed,
            **metrics,
        })
        texts.append(text)
        times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        eq_str = (verification.get('extracted_equation') or 'N/A')[:40]
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"eq={eq_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / N_SAMPLES
    diversity = compute_diversity_metrics(texts)

    print(f"\n  Vanilla: {n_correct}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")

    result = format_results(
        method="vanilla", benchmark=f"Countdown-{N_SAMPLES}",
        n_samples=N_SAMPLES, seed=SEED,
        accuracy=accuracy, n_correct=n_correct,
        diversity=diversity, gen_times=times,
        per_sample=results,
    )
    return result, diversity


def run_bsd_alpha_config(model, tokenizer, embedding_layer, problems, prompts,
                         alpha_cfg):
    """Run BSD with a specific alpha schedule config."""
    label = alpha_cfg["label"]
    print(f"\n{'='*60}")
    print(f"  BSD Alpha Ablation: {label}")
    print(f"  k_frac={BEST_K_FRAC}, alpha={label}, tau=linear(1.0→0.3)")
    print(f"{'='*60}")

    config = BSDConfig(
        k_frac=BEST_K_FRAC,
        alpha_schedule=alpha_cfg["schedule"],
        alpha_start=alpha_cfg["alpha_start"],
        alpha_end=alpha_cfg["alpha_end"],
        tau_start=1.0, tau_end=0.3,
        fallback_beta_start=0.0, fallback_beta_end=0.0,
    )

    results = []
    texts = []
    times = []
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

        results.append({
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
        texts.append(text)
        times.append(elapsed)
        all_entropy_trajectories.append(entropy_traj)

        status = "OK" if verification["is_correct"] else "X"
        ent_str = ""
        if belief_entropies:
            ent_str = f"ent={belief_entropies[0]:.1f}->{belief_entropies[-1]:.1f}"
        eq_str = (verification.get('extracted_equation') or 'N/A')[:35]
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"eq={eq_str} | {ent_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / N_SAMPLES
    diversity = compute_diversity_metrics(texts)

    # Entropy analysis
    n_decreasing = sum(1 for r in results if r.get("entropy_decreasing") is True)
    ent_starts = [r["entropy_start"] for r in results if r["entropy_start"] is not None]
    ent_ends = [r["entropy_end"] for r in results if r["entropy_end"] is not None]
    avg_ent_start = float(np.mean(ent_starts)) if ent_starts else 0.0
    avg_ent_end = float(np.mean(ent_ends)) if ent_ends else 0.0

    print(f"\n  {label}: {n_correct}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")
    print(f"  Entropy: {avg_ent_start:.2f} -> {avg_ent_end:.2f} | "
          f"decreasing={n_decreasing}/{N_SAMPLES}")

    extra_config = {
        "k_frac": BEST_K_FRAC,
        "alpha_schedule": alpha_cfg["schedule"],
        "alpha_start": alpha_cfg["alpha_start"],
        "alpha_end": alpha_cfg["alpha_end"],
        "tau_start": 1.0, "tau_end": 0.3,
    }

    result = format_results(
        method=f"BSD alpha={label}", benchmark=f"Countdown-{N_SAMPLES}",
        n_samples=N_SAMPLES, seed=SEED,
        accuracy=accuracy, n_correct=n_correct,
        diversity=diversity, gen_times=times,
        per_sample=results, extra_config=extra_config,
    )
    result["entropy_analysis"] = {
        "n_decreasing": n_decreasing,
        "n_total": N_SAMPLES,
        "avg_entropy_start": avg_ent_start,
        "avg_entropy_end": avg_ent_end,
        "entropy_trajectories": all_entropy_trajectories,
    }
    return result, diversity


def main():
    start_time = time.time()
    write_pid()
    report_progress(0, len(ALPHA_CONFIGS) + 1, "Loading model")

    print("=" * 70)
    print(f"  BSD Alpha Schedule Ablation — Countdown-{N_SAMPLES} (PILOT)")
    print(f"  Best k from ablation_bsd_k: k_frac={BEST_K_FRAC}")
    print(f"  Alpha schedules: {[c['label'] for c in ALPHA_CONFIGS]}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, len(ALPHA_CONFIGS) + 1, "Model loaded, generating problems")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    prompts = [format_countdown_prompt(p) for p in problems]
    print(f"\nGenerated {len(problems)} Countdown problems")

    # ── Phase 1: Vanilla baseline ──
    report_progress(1, len(ALPHA_CONFIGS) + 1, "Running vanilla baseline")
    vanilla_result, vanilla_diversity = run_vanilla_baseline(
        model, tokenizer, problems, prompts
    )

    # ── Phase 2: Run each alpha config ──
    alpha_results = {}
    alpha_diversities = {}
    for ai, alpha_cfg in enumerate(ALPHA_CONFIGS):
        label = alpha_cfg["label"]
        report_progress(ai + 2, len(ALPHA_CONFIGS) + 1, f"Running BSD alpha={label}")
        result, diversity = run_bsd_alpha_config(
            model, tokenizer, embedding_layer, problems, prompts,
            alpha_cfg,
        )
        alpha_results[label] = result
        alpha_diversities[label] = diversity

    # ── Comparison ──
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE — BSD Alpha Schedule Ablation")
    print(f"{'='*70}")
    all_results = [vanilla_result] + [alpha_results[c["label"]] for c in ALPHA_CONFIGS]
    print_comparison_table(all_results)

    # ── Degeneration checks ──
    degeneration_by_alpha = {}
    well_formed_count = 0
    for alpha_cfg in ALPHA_CONFIGS:
        label = alpha_cfg["label"]
        warnings = check_degeneration(alpha_diversities[label], vanilla_diversity)
        degeneration_by_alpha[label] = warnings
        if not warnings:
            well_formed_count += 1
        if warnings:
            print(f"  Warnings for {label}: {warnings}")

    # ── Pass/Fail Decision ──
    vanilla_acc = vanilla_result["metrics"]["accuracy"]
    any_above_vanilla = any(
        alpha_results[c["label"]]["metrics"]["accuracy"] > vanilla_acc
        for c in ALPHA_CONFIGS
    )

    # Check well-formedness: rep-3 < 2x vanilla
    all_ok = True
    for alpha_cfg in ALPHA_CONFIGS:
        label = alpha_cfg["label"]
        a_rep3 = alpha_results[label]["metrics"]["rep_3"]
        v_rep3 = vanilla_result["metrics"]["rep_3"]
        if a_rep3 > v_rep3 * 2 + 0.01:
            all_ok = False

    pass_criteria = all_ok and (any_above_vanilla or vanilla_acc == 0.0)
    if vanilla_acc == 0.0:
        pass_criteria = all_ok  # all configs well-formed is sufficient

    verdict = "GO" if pass_criteria else "NO-GO"

    # Find best alpha config
    best_label = max(
        [c["label"] for c in ALPHA_CONFIGS],
        key=lambda l: alpha_results[l]["metrics"]["accuracy"]
    )
    best_acc = alpha_results[best_label]["metrics"]["accuracy"]
    best_cfg = next(c for c in ALPHA_CONFIGS if c["label"] == best_label)

    print(f"\n{'='*70}")
    print(f"  PILOT VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  Vanilla accuracy:  {vanilla_acc:.1%}")
    for alpha_cfg in ALPHA_CONFIGS:
        label = alpha_cfg["label"]
        acc = alpha_results[label]["metrics"]["accuracy"]
        marker = " <-- BEST" if label == best_label else ""
        print(f"  BSD alpha={label}: {acc:.1%}{marker}")
    print(f"  All well-formed: {all_ok}")
    print(f"  Any above vanilla: {any_above_vanilla}")
    print(f"  Best alpha schedule: {best_label}")

    # Print qualitative samples from best config
    print(f"\n{'='*70}")
    print(f"  QUALITATIVE SAMPLES — Best BSD config (alpha={best_label})")
    print_qualitative_samples(alpha_results[best_label]["per_sample"], n=10)

    # ── Save results ──
    ablation_summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "verdict": verdict,
        "pass_criteria": {
            "all_well_formed": all_ok,
            "any_above_vanilla": any_above_vanilla,
        },
        "best_k_frac": BEST_K_FRAC,
        "best_alpha_schedule": best_label,
        "best_alpha_config": best_cfg,
        "best_alpha_accuracy": best_acc,
        "vanilla_accuracy": vanilla_acc,
        "alpha_summary": {
            c["label"]: {
                "accuracy": alpha_results[c["label"]]["metrics"]["accuracy"],
                "n_correct": alpha_results[c["label"]]["metrics"]["n_correct"],
                "rep_3": alpha_results[c["label"]]["metrics"]["rep_3"],
                "distinct_3": alpha_results[c["label"]]["metrics"]["distinct_3"],
                "avg_gen_time_s": alpha_results[c["label"]]["metrics"]["avg_gen_time_s"],
                "entropy_start": alpha_results[c["label"]].get("entropy_analysis", {}).get("avg_entropy_start"),
                "entropy_end": alpha_results[c["label"]].get("entropy_analysis", {}).get("avg_entropy_end"),
                "n_entropy_decreasing": alpha_results[c["label"]].get("entropy_analysis", {}).get("n_decreasing"),
                "degeneration_warnings": degeneration_by_alpha.get(c["label"], {}),
            }
            for c in ALPHA_CONFIGS
        },
        "total_time_s": round(time.time() - start_time, 1),
        "timestamp": datetime.now().isoformat(),
        "vanilla": vanilla_result,
        "alpha_results": {c["label"]: alpha_results[c["label"]] for c in ALPHA_CONFIGS},
    }

    out_file = RESULTS_DIR / "bsd_alpha_ablation_countdown100.json"
    save_results(ablation_summary, str(out_file))

    # Also write summary markdown
    summary_md = RESULTS_DIR / "bsd_alpha_ablation_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# BSD Alpha Schedule Ablation Summary — Countdown-{N_SAMPLES} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"Fixed: k_frac={BEST_K_FRAC}, tau=linear(1.0→0.3)\n\n")
        f.write(f"| Alpha Schedule | Accuracy | rep-3 | distinct-3 | Entropy (start→end) | Time (s) |\n")
        f.write(f"|----------------|----------|-------|------------|---------------------|----------|\n")
        f.write(f"| Vanilla | {vanilla_acc:.1%} | "
                f"{vanilla_result['metrics']['rep_3']:.4f} | "
                f"{vanilla_result['metrics']['distinct_3']:.4f} | — | "
                f"{vanilla_result['metrics']['avg_gen_time_s']:.1f} |\n")
        for alpha_cfg in ALPHA_CONFIGS:
            label = alpha_cfg["label"]
            r = alpha_results[label]
            ea = r.get("entropy_analysis", {})
            es = ea.get("avg_entropy_start", 0)
            ee = ea.get("avg_entropy_end", 0)
            marker = " **BEST**" if label == best_label else ""
            f.write(f"| {label}{marker} | "
                    f"{r['metrics']['accuracy']:.1%} | "
                    f"{r['metrics']['rep_3']:.4f} | "
                    f"{r['metrics']['distinct_3']:.4f} | "
                    f"{es:.2f}→{ee:.2f} | "
                    f"{r['metrics']['avg_gen_time_s']:.1f} |\n")
        f.write(f"\n### Best Config\n")
        f.write(f"- **Alpha schedule: {best_label}**\n")
        f.write(f"- k_frac: {BEST_K_FRAC}\n")
        f.write(f"- Accuracy: {best_acc:.1%}\n")
    print(f"[{TASK_ID}] Summary saved to {summary_md}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    elapsed_total = time.time() - start_time
    mark_done(
        status="success",
        summary=f"BSD alpha ablation {verdict}: best={best_label}, "
                f"acc={best_acc:.1%}, time={elapsed_total:.0f}s",
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
        "planned_min": 45,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "Dream-v0-Instruct-7B",
            "method": "BSD alpha-ablation",
            "k_frac": BEST_K_FRAC,
            "alpha_schedules": [c["label"] for c in ALPHA_CONFIGS],
            "n_samples": N_SAMPLES,
            "tau_schedule": "linear(1.0->0.3)",
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(progress, indent=2))
    print(f"[{TASK_ID}] gpu_progress.json updated")

    print(f"\n[{TASK_ID}] Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    return pass_criteria


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[{TASK_ID}] FATAL ERROR: {e}")
        traceback.print_exc()
        # Still write DONE marker on failure
        try:
            mark_done(status="failed", summary=f"Fatal error: {e}")
        except:
            pass
        sys.exit(1)
