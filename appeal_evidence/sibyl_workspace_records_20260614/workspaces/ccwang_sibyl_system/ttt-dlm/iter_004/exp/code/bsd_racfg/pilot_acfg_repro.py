"""
Pilot A-CFG Reproduction on Dream-7B (task: pilot_acfg_repro).

Reproduces Adaptive CFG (arXiv 2505.20199) on Dream-7B with Countdown-100
(16 samples in pilot mode). Validates that CFG works on Dream-7B before
investing in RACFG.

Decision Gate 1: if accuracy < vanilla, switch all CFG experiments to LLaDA-8B.

Pass criteria: Accuracy > vanilla AND no degeneration (rep-3 < vanilla + 20%)
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics, compute_per_sample_metrics,
    vanilla_generate, format_results, save_results, print_qualitative_samples,
    print_comparison_table, check_degeneration,
    PROJECT_DIR, MASK_TOKEN_ID,
)
from bsd_racfg.racfg import ACFGConfig, acfg_generate


TASK_ID = "pilot_acfg_repro"
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/pilots")
N_SAMPLES = 16  # pilot mode
SEED = 42
DEVICE = "cuda:0"

# A-CFG config matching arXiv 2505.20199
ACFG_W = 1.0
ACFG_REMASK_PCT = 0.10  # 10% re-masking


def write_pid():
    """Write PID file for system recovery detection."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")


def report_progress(stage, current, total, extra=None):
    """Write progress file for system monitor."""
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "stage": stage,
        "current": current,
        "total": total,
        "extra": extra or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress.write_text(json.dumps(data))


def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    # Clean up PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    # Read final progress
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
    print(f"[{TASK_ID}] DONE marker written (status={status})")


def run_vanilla(model, tokenizer, problems):
    """Run vanilla baseline on all problems."""
    print(f"\n{'='*60}")
    print(f"  VANILLA BASELINE ({len(problems)} samples)")
    print(f"{'='*60}")

    per_sample = []
    gen_times = []
    n_correct = 0

    for idx, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)

        text, elapsed, diag = vanilla_generate(model, tokenizer, prompt, DEVICE)
        verification = verify_countdown_answer(text, prob)
        metrics = compute_per_sample_metrics(text)

        is_correct = verification["is_correct"]
        if is_correct:
            n_correct += 1

        per_sample.append({
            "idx": idx,
            "target": prob["target"],
            "numbers": prob["numbers"],
            "generated_text": text,
            "is_correct": is_correct,
            "extracted_equation": verification.get("extracted_equation"),
            "gen_time_s": round(elapsed, 2),
            **metrics,
        })
        gen_times.append(elapsed)

        report_progress("vanilla", idx + 1, len(problems),
                        {"n_correct": n_correct})

        if (idx + 1) % 4 == 0:
            print(f"  [{idx+1}/{len(problems)}] acc={n_correct/(idx+1):.1%} "
                  f"avg_time={np.mean(gen_times):.1f}s")

    accuracy = n_correct / len(problems)
    texts = [s["generated_text"] for s in per_sample]
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

    print(f"\n  Vanilla accuracy: {accuracy:.1%} ({n_correct}/{len(problems)})")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")
    print(f"  Avg time: {np.mean(gen_times):.1f}s")

    return result


def run_acfg(model, tokenizer, problems):
    """Run A-CFG on all problems."""
    print(f"\n{'='*60}")
    print(f"  A-CFG (w={ACFG_W}, remask={ACFG_REMASK_PCT*100:.0f}%) "
          f"({len(problems)} samples)")
    print(f"{'='*60}")

    config = ACFGConfig(remask_pct=ACFG_REMASK_PCT, w=ACFG_W)

    per_sample = []
    gen_times = []
    n_correct = 0

    for idx, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)

        text, elapsed, diag = acfg_generate(model, tokenizer, prompt, config, DEVICE)
        verification = verify_countdown_answer(text, prob)
        metrics = compute_per_sample_metrics(text)

        is_correct = verification["is_correct"]
        if is_correct:
            n_correct += 1

        n_guidance_steps = sum(1 for s in diag["step_diagnostics"]
                               if s.get("guidance_applied"))

        per_sample.append({
            "idx": idx,
            "target": prob["target"],
            "numbers": prob["numbers"],
            "generated_text": text,
            "is_correct": is_correct,
            "extracted_equation": verification.get("extracted_equation"),
            "gen_time_s": round(elapsed, 2),
            "n_guidance_steps": n_guidance_steps,
            **metrics,
        })
        gen_times.append(elapsed)

        report_progress("acfg", idx + 1, len(problems),
                        {"n_correct": n_correct})

        if (idx + 1) % 4 == 0:
            print(f"  [{idx+1}/{len(problems)}] acc={n_correct/(idx+1):.1%} "
                  f"avg_time={np.mean(gen_times):.1f}s")

    accuracy = n_correct / len(problems)
    texts = [s["generated_text"] for s in per_sample]
    diversity = compute_diversity_metrics(texts)

    result = format_results(
        method="A-CFG",
        benchmark=f"Countdown-{len(problems)}",
        n_samples=len(problems),
        seed=SEED,
        accuracy=accuracy,
        n_correct=n_correct,
        diversity=diversity,
        gen_times=gen_times,
        per_sample=per_sample,
        extra_config={
            "acfg_w": ACFG_W,
            "acfg_remask_pct": ACFG_REMASK_PCT,
        },
    )

    print(f"\n  A-CFG accuracy: {accuracy:.1%} ({n_correct}/{len(problems)})")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")
    print(f"  Avg time: {np.mean(gen_times):.1f}s")
    print(f"  Avg guidance steps: {np.mean([s['n_guidance_steps'] for s in per_sample]):.0f}")

    return result


def main():
    start_time = datetime.now()
    print(f"{'='*60}")
    print(f"  Pilot A-CFG Reproduction on Dream-7B")
    print(f"  Task: {TASK_ID}")
    print(f"  Time: {start_time.isoformat()}")
    print(f"  Samples: {N_SAMPLES} (pilot mode)")
    print(f"  Seed: {SEED}")
    print(f"  A-CFG config: w={ACFG_W}, remask_pct={ACFG_REMASK_PCT}")
    print(f"{'='*60}")

    write_pid()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Generate problems
        report_progress("setup", 0, 2, {"stage": "generating_problems"})
        problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
        print(f"\nGenerated {len(problems)} Countdown problems")
        for i, p in enumerate(problems[:3]):
            print(f"  Problem {i}: numbers={p['numbers']}, target={p['target']}")

        # Load model
        report_progress("setup", 1, 2, {"stage": "loading_model"})
        model, tokenizer, embedding_layer = load_dream(DEVICE)

        # Run vanilla baseline
        report_progress("vanilla", 0, N_SAMPLES)
        vanilla_result = run_vanilla(model, tokenizer, problems)

        # Run A-CFG
        report_progress("acfg", 0, N_SAMPLES)
        acfg_result = run_acfg(model, tokenizer, problems)

        # ═══════════════════════════════════════════
        # Comparison & Decision Gate 1
        # ═══════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  COMPARISON & DECISION GATE 1")
        print(f"{'='*60}")

        v_acc = vanilla_result["metrics"]["accuracy"]
        a_acc = acfg_result["metrics"]["accuracy"]
        delta = a_acc - v_acc

        print(f"\n  Vanilla accuracy:  {v_acc:.1%}")
        print(f"  A-CFG accuracy:    {a_acc:.1%}")
        print(f"  Delta:             {delta:+.1%}")

        # Degeneration check
        v_div = {k: vanilla_result["metrics"].get(k, 0) for k in
                 ["rep_2", "rep_3", "distinct_3", "std_length_words"]}
        a_div = {k: acfg_result["metrics"].get(k, 0) for k in
                 ["rep_2", "rep_3", "distinct_3", "std_length_words"]}
        degen_warnings = check_degeneration(a_div, v_div)

        if degen_warnings:
            print(f"\n  DEGENERATION WARNINGS:")
            for k, v in degen_warnings.items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  No degeneration detected.")

        # Decision Gate
        acfg_beats_vanilla = a_acc > v_acc
        no_degeneration = "rep_3_alert" not in degen_warnings

        if acfg_beats_vanilla and no_degeneration:
            decision = "GO"
            decision_detail = (
                f"A-CFG ({a_acc:.1%}) > vanilla ({v_acc:.1%}), "
                f"no degeneration → CFG works on Dream-7B, proceed with RACFG"
            )
        elif acfg_beats_vanilla and not no_degeneration:
            decision = "CAUTION"
            decision_detail = (
                f"A-CFG ({a_acc:.1%}) > vanilla ({v_acc:.1%}), "
                f"but degeneration detected → proceed with caution, may need w reduction"
            )
        else:
            decision = "NO-GO"
            decision_detail = (
                f"A-CFG ({a_acc:.1%}) <= vanilla ({v_acc:.1%}) → "
                f"Decision Gate 1 FAILED: consider switching CFG experiments to LLaDA-8B"
            )

        print(f"\n  Decision Gate 1: {decision}")
        print(f"  {decision_detail}")

        # Print qualitative samples
        print(f"\n--- A-CFG Qualitative Samples ---")
        print_qualitative_samples(acfg_result["per_sample"], n=10)

        # Print comparison table
        print_comparison_table([vanilla_result, acfg_result])

        # Compile final results
        final_result = {
            "task_id": TASK_ID,
            "mode": "pilot",
            "timestamp": start_time.isoformat(),
            "duration_s": round((datetime.now() - start_time).total_seconds()),
            "seed": SEED,
            "n_samples": N_SAMPLES,
            "decision_gate_1": {
                "decision": decision,
                "detail": decision_detail,
                "acfg_beats_vanilla": acfg_beats_vanilla,
                "no_degeneration": no_degeneration,
            },
            "vanilla": vanilla_result,
            "acfg": acfg_result,
            "comparison": {
                "accuracy_delta": delta,
                "acfg_w": ACFG_W,
                "acfg_remask_pct": ACFG_REMASK_PCT,
                "degeneration_warnings": degen_warnings,
            },
            "pass_criteria": {
                "accuracy_gt_vanilla": acfg_beats_vanilla,
                "no_degeneration": no_degeneration,
                "overall": decision in ("GO", "CAUTION"),
            },
        }

        # Save results
        out_path = RESULTS_DIR / "acfg_repro_countdown100.json"
        save_results(final_result, str(out_path))

        # Write pilot summary
        summary_path = RESULTS_DIR / f"{TASK_ID}_summary.md"
        summary_path.write_text(
            f"# Pilot A-CFG Reproduction Summary\n\n"
            f"**Task**: {TASK_ID}\n"
            f"**Date**: {start_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"**Model**: Dream-v0-Instruct-7B\n"
            f"**Benchmark**: Countdown-{N_SAMPLES} (pilot)\n"
            f"**Seed**: {SEED}\n\n"
            f"## Results\n\n"
            f"| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time |\n"
            f"|--------|----------|-------|-------|------------|----------|\n"
            f"| Vanilla | {v_acc:.1%} | {v_div['rep_2']:.4f} | {v_div['rep_3']:.4f} | "
            f"{vanilla_result['metrics'].get('distinct_3', 0):.4f} | "
            f"{vanilla_result['metrics']['avg_gen_time_s']:.1f}s |\n"
            f"| A-CFG (w={ACFG_W}) | {a_acc:.1%} | {a_div['rep_2']:.4f} | {a_div['rep_3']:.4f} | "
            f"{acfg_result['metrics'].get('distinct_3', 0):.4f} | "
            f"{acfg_result['metrics']['avg_gen_time_s']:.1f}s |\n\n"
            f"## Decision Gate 1\n\n"
            f"**Decision**: {decision}\n\n"
            f"{decision_detail}\n\n"
            f"## Degeneration Check\n\n"
            f"{'No degeneration detected.' if not degen_warnings else chr(10).join(f'- {k}: {v}' for k, v in degen_warnings.items())}\n"
        )
        print(f"\nSummary saved to {summary_path}")

        # Mark done
        mark_done(
            status="success",
            summary=f"A-CFG pilot: {decision}. "
                    f"Vanilla={v_acc:.1%}, A-CFG={a_acc:.1%}, "
                    f"delta={delta:+.1%}"
        )

        return final_result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n!!! ERROR: {error_msg}")
        traceback.print_exc()
        mark_done(status="failed", summary=error_msg)
        raise


if __name__ == "__main__":
    result = main()
    sys.exit(0)
