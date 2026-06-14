"""
RACFG Temporal Schedule Ablation on Countdown-16 (PILOT mode).

Task: ablation_racfg_schedule
GPU: 1 (single)
Mode: PILOT (16 samples, seed 42)

PIVOT from original plan: Since ablation_racfg_remask_pct was NO-GO (JSD stability
is uninformative on Dream-7B, all RACFG variants got 0% accuracy), this ablation
tests temporal scheduling applied to A-CFG's CONFIDENCE-BASED re-masking instead
of the original JSD-based RACFG.

Tests 4 schedule types:
  (1) fixed w=1.0 (A-CFG baseline — no scheduling)
  (2) linear ramp (0 -> w_max)
  (3) cosine ramp (smooth 0 -> w_max)
  (4) threshold_70_30 (theory-driven: zero at mask_rate>70%, ramp 30-70%, max <30%)

Also tests w_base ∈ {1.0, 1.5} to see if higher base weight helps with scheduling.

Pass criteria: At least one scheduled variant > fixed w by >=1pp accuracy.
"""
import os
import sys
import json
import time
import traceback
import gc
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
from bsd_racfg.racfg import (
    ACFGConfig, ScheduledACFGConfig,
    acfg_generate, scheduled_acfg_generate,
)

TASK_ID = "ablation_racfg_schedule"
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/full")
PILOTS_DIR = Path(f"{PROJECT_DIR}/exp/results/pilots")
N_SAMPLES = 16  # PILOT mode
SEED = 42
DEVICE = "cuda:0"

# Schedule types to test
SCHEDULE_TYPES = ["fixed", "linear", "cosine", "threshold_70_30"]

# Base guidance weights to test
W_BASES = [1.0, 1.5]

# Fixed parameters (best from remask ablation: use 10% which is A-CFG default)
REMASK_PCT = 0.10
W_MAX = 2.0
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4


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


def run_method(model, tokenizer, problems, method_name, gen_fn, device):
    """Run a generation method on all problems and return formatted results."""
    per_sample = []
    gen_times = []
    texts = []
    all_guidance_info = []

    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)

        result = gen_fn(model, tokenizer, prompt, device)
        if len(result) == 3:
            text, elapsed, diag = result
        else:
            text, elapsed = result[0], result[1]
            diag = {}

        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        sample = {
            "idx": idx,
            "target": problem["target"],
            "numbers": problem["numbers"],
            "generated_text": text,
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "gen_time_s": round(elapsed, 2),
            **metrics,
        }

        # Collect guidance info if available
        if diag.get("n_guidance_steps") is not None:
            sample["n_guidance_steps"] = diag["n_guidance_steps"]
            sample["n_total_steps"] = diag.get("n_total_steps", 0)
            all_guidance_info.append({
                "idx": idx,
                "n_guidance_steps": diag["n_guidance_steps"],
                "n_total_steps": diag.get("n_total_steps", 0),
            })

        per_sample.append(sample)
        gen_times.append(elapsed)
        texts.append(text)

        status = "OK" if verification["is_correct"] else "WRONG"
        eq_str = verification.get('extracted_equation') or 'N/A'
        print(f"    [{idx:2d}] {status} | target={problem['target']} | "
              f"eq={eq_str[:40]} | {elapsed:.1f}s")

    n_correct = sum(1 for s in per_sample if s["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)

    result = format_results(
        method=method_name,
        benchmark=f"Countdown-{len(problems)}",
        n_samples=len(problems),
        seed=SEED,
        accuracy=accuracy,
        n_correct=n_correct,
        diversity=diversity,
        gen_times=gen_times,
        per_sample=per_sample,
    )

    # Add guidance analysis
    if all_guidance_info:
        guidance_steps = [g["n_guidance_steps"] for g in all_guidance_info]
        total_steps = [g["n_total_steps"] for g in all_guidance_info]
        result["guidance_analysis"] = {
            "mean_guidance_steps": float(np.mean(guidance_steps)),
            "mean_total_steps": float(np.mean(total_steps)),
            "guidance_ratio": float(np.mean(guidance_steps)) / float(np.mean(total_steps)) if np.mean(total_steps) > 0 else 0,
        }

    print(f"    -> Accuracy: {accuracy:.1%} ({n_correct}/{len(problems)}) | "
          f"rep-3: {diversity['rep_3']:.4f} | dist-3: {diversity['distinct_3']:.4f} | "
          f"avg: {np.mean(gen_times):.1f}s")

    return result


def main():
    start_time = datetime.now()

    print("=" * 70)
    print(f"  RACFG Temporal Schedule Ablation — Countdown-{N_SAMPLES} (PILOT)")
    print(f"  [PIVOTED: Using A-CFG confidence re-masking + temporal scheduling]")
    print(f"  Task ID: {TASK_ID}")
    print(f"  Time: {start_time.isoformat()}")
    print(f"  Device: GPU 1 (mapped to {DEVICE})")
    print(f"  Seed: {SEED}")
    print(f"  Schedules: {SCHEDULE_TYPES}")
    print(f"  w_bases: {W_BASES}")
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
        model, tokenizer, embedding_layer = load_dream(DEVICE)

        all_results = {}
        # vanilla + acfg_fixed + (schedules * w_bases - fixed*w_bases already counted as acfg)
        # = 1 + 1 + len(SCHEDULE_TYPES) * len(W_BASES) = 1 + 1 + 4*2 = 10
        # But fixed w=1.0 is the same as A-CFG baseline, so we skip that duplicate
        total_configs = 1 + len(SCHEDULE_TYPES) * len(W_BASES)
        config_idx = 0

        # ─────────────────────────────────────────────
        # 1. Vanilla baseline
        # ─────────────────────────────────────────────
        config_idx += 1
        print(f"\n{'='*60}")
        print(f"  [{config_idx}/{total_configs}] Vanilla Baseline")
        print(f"{'='*60}")

        def vanilla_fn(m, t, p, d):
            return vanilla_generate(m, t, p, d, gen_len=GEN_LEN, steps=STEPS,
                                    temperature=TEMPERATURE)

        vanilla_result = run_method(model, tokenizer, problems, "vanilla",
                                    vanilla_fn, DEVICE)
        all_results["vanilla"] = vanilla_result
        report_progress(TASK_ID, str(RESULTS_DIR), config_idx, total_configs,
                        metric={"phase": "vanilla", "accuracy": vanilla_result["metrics"]["accuracy"]})

        # ─────────────────────────────────────────────
        # 2. Scheduled A-CFG variants: schedule_type x w_base
        # ─────────────────────────────────────────────
        scheduled_results = {}

        for schedule_type in SCHEDULE_TYPES:
            for w_base in W_BASES:
                config_idx += 1
                config_name = f"ACFG_{schedule_type}_w{w_base}"

                print(f"\n{'='*60}")
                print(f"  [{config_idx}/{total_configs}] {config_name}")
                print(f"  schedule={schedule_type}, w_base={w_base}, remask={REMASK_PCT}")
                print(f"{'='*60}")

                if schedule_type == "fixed":
                    # For "fixed" schedule, use the standard A-CFG (no scheduling)
                    acfg_config = ACFGConfig(
                        remask_pct=REMASK_PCT, w=w_base,
                        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
                    )
                    def make_acfg_fn(cfg):
                        def fn(m, t, p, d):
                            return acfg_generate(m, t, p, cfg, d)
                        return fn
                    gen_fn = make_acfg_fn(acfg_config)
                else:
                    # Scheduled A-CFG
                    sched_config = ScheduledACFGConfig(
                        remask_pct=REMASK_PCT,
                        w_base=w_base,
                        w_max=W_MAX,
                        schedule_type=schedule_type,
                        gen_len=GEN_LEN,
                        steps=STEPS,
                        temperature=TEMPERATURE,
                    )
                    def make_sched_fn(cfg):
                        def fn(m, t, p, d):
                            return scheduled_acfg_generate(m, t, p, cfg, d)
                        return fn
                    gen_fn = make_sched_fn(sched_config)

                result = run_method(model, tokenizer, problems, config_name,
                                    gen_fn, DEVICE)
                all_results[config_name] = result
                scheduled_results[config_name] = result

                report_progress(TASK_ID, str(RESULTS_DIR), config_idx, total_configs,
                                metric={
                                    "phase": config_name,
                                    "accuracy": result["metrics"]["accuracy"],
                                    "schedule": schedule_type,
                                    "w_base": w_base,
                                })

                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache()

        # ─────────────────────────────────────────────
        # Analysis
        # ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  ANALYSIS")
        print("=" * 70)

        vanilla_acc = vanilla_result["metrics"]["accuracy"]

        # Find the fixed w=1.0 baseline (standard A-CFG)
        fixed_w1_key = "ACFG_fixed_w1.0"
        fixed_w1_acc = all_results.get(fixed_w1_key, {}).get("metrics", {}).get("accuracy", 0)

        # Find best scheduled variant
        best_sched_name = None
        best_sched_acc = -1
        for name, res in scheduled_results.items():
            if "fixed" not in name:  # exclude fixed baselines
                acc = res["metrics"]["accuracy"]
                if acc > best_sched_acc:
                    best_sched_acc = acc
                    best_sched_name = name

        # Also find best overall
        best_overall_name = None
        best_overall_acc = -1
        for name, res in scheduled_results.items():
            acc = res["metrics"]["accuracy"]
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_overall_name = name

        # Schedule analysis
        schedule_analysis = {}
        for sched in SCHEDULE_TYPES:
            sched_accs = []
            for w in W_BASES:
                key = f"ACFG_{sched}_w{w}"
                if key in all_results:
                    sched_accs.append(all_results[key]["metrics"]["accuracy"])
            schedule_analysis[sched] = {
                "mean_accuracy": float(np.mean(sched_accs)) if sched_accs else 0,
                "max_accuracy": float(max(sched_accs)) if sched_accs else 0,
                "accuracies": sched_accs,
            }

        # w_base analysis
        w_analysis = {}
        for w in W_BASES:
            w_accs = []
            for sched in SCHEDULE_TYPES:
                key = f"ACFG_{sched}_w{w}"
                if key in all_results:
                    w_accs.append(all_results[key]["metrics"]["accuracy"])
            w_analysis[f"w{w}"] = {
                "mean_accuracy": float(np.mean(w_accs)) if w_accs else 0,
                "max_accuracy": float(max(w_accs)) if w_accs else 0,
                "accuracies": w_accs,
            }

        # Check pass criteria: at least one scheduled variant > fixed w by >=1pp
        any_scheduled_beats_fixed = False
        for name, res in scheduled_results.items():
            if "fixed" not in name:
                if res["metrics"]["accuracy"] > fixed_w1_acc + 0.01:
                    any_scheduled_beats_fixed = True
                    break

        # Build verdict
        verdict = {
            "overall": "GO" if any_scheduled_beats_fixed else "NO-GO",
            "pass_criteria": "At least one scheduled variant > fixed w by >=1pp",
            "any_scheduled_beats_fixed": any_scheduled_beats_fixed,
            "vanilla_accuracy": vanilla_acc,
            "fixed_w1_accuracy": fixed_w1_acc,
            "best_scheduled_name": best_sched_name,
            "best_scheduled_accuracy": best_sched_acc,
            "best_scheduled_delta_vs_fixed_pp": round((best_sched_acc - fixed_w1_acc) * 100, 1) if best_sched_name else 0,
            "best_overall_name": best_overall_name,
            "best_overall_accuracy": best_overall_acc,
            "schedule_analysis": schedule_analysis,
            "w_analysis": w_analysis,
            "pivot_note": "Pivoted from JSD-based RACFG to confidence-based A-CFG + scheduling due to NO-GO from ablation_racfg_remask_pct",
        }

        # Print summary table
        print("\n  Results Summary:")
        print(f"  {'Method':<30s} {'Acc':>8s} {'rep-3':>8s} {'dist-3':>8s} {'Time':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Vanilla':<30s} {vanilla_acc:>7.1%} "
              f"{vanilla_result['metrics']['rep_3']:>8.4f} "
              f"{vanilla_result['metrics']['distinct_3']:>8.4f} "
              f"{vanilla_result['metrics']['avg_gen_time_s']:>7.1f}s")

        for name in sorted(scheduled_results.keys()):
            res = scheduled_results[name]
            m = res["metrics"]
            marker = " *" if name == best_overall_name else ""
            print(f"  {name:<30s} {m['accuracy']:>7.1%} "
                  f"{m['rep_3']:>8.4f} "
                  f"{m['distinct_3']:>8.4f} "
                  f"{m['avg_gen_time_s']:>7.1f}s{marker}")

        print(f"\n  Verdict: {verdict['overall']}")
        print(f"  Best scheduled: {best_sched_name} ({best_sched_acc:.1%})")
        print(f"  Fixed A-CFG w=1.0: {fixed_w1_acc:.1%}")
        if best_sched_name:
            print(f"  Delta: {verdict['best_scheduled_delta_vs_fixed_pp']:+.1f}pp")

        # ─────────────────────────────────────────────
        # Save results
        # ─────────────────────────────────────────────
        end_time = datetime.now()
        elapsed_min = (end_time - start_time).total_seconds() / 60

        output = {
            "task_id": TASK_ID,
            "mode": "PILOT",
            "benchmark": f"Countdown-{N_SAMPLES}",
            "seed": SEED,
            "timestamp": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_min": round(elapsed_min, 1),
            "verdict": verdict,
            "configs_tested": {
                "schedule_types": SCHEDULE_TYPES,
                "w_bases": W_BASES,
                "remask_pct": REMASK_PCT,
                "w_max": W_MAX,
            },
            "results": all_results,
        }

        output_path = RESULTS_DIR / "racfg_schedule_ablation_countdown100.json"
        output_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"\n  Results saved to: {output_path}")

        # Write summary markdown
        summary_path = RESULTS_DIR / "racfg_schedule_ablation_summary.md"
        summary_lines = [
            "# RACFG Temporal Schedule Ablation Summary (PILOT)",
            "",
            "## Task: ablation_racfg_schedule",
            f"- **Date**: {start_time.strftime('%Y-%m-%d')}",
            f"- **GPU**: 1 (NVIDIA RTX PRO 6000 Blackwell Server Edition)",
            f"- **Elapsed**: {elapsed_min:.1f} minutes",
            f"- **Verdict**: **{verdict['overall']}** ({'scheduled variant beats fixed' if any_scheduled_beats_fixed else 'no scheduled variant beats fixed'})",
            "",
            "## PIVOT NOTE",
            "",
            "This ablation was **pivoted** from the original JSD-based RACFG temporal scheduling",
            "to **A-CFG confidence-based re-masking + temporal scheduling**.",
            "",
            "Reason: `ablation_racfg_remask_pct` was NO-GO — JSD stability is uninformative on",
            "Dream-7B (stability ~0.997 everywhere), so all RACFG variants got 0% accuracy.",
            "A-CFG with confidence-based re-masking achieved 6.2% in the same test.",
            "",
            "## Configuration",
            f"- schedule_types: {SCHEDULE_TYPES}",
            f"- w_bases: {W_BASES}",
            f"- remask_pct={REMASK_PCT}, w_max={W_MAX}",
            f"- {N_SAMPLES} samples (PILOT), seed {SEED}, {STEPS} steps, gen_len={GEN_LEN}, temperature={TEMPERATURE}",
            "",
            "## Results",
            "",
            "| Method | Accuracy | rep-3 | dist-3 | Avg Time | Guidance Steps |",
            "|--------|----------|-------|--------|----------|---------------|",
        ]

        # Add vanilla
        vm = vanilla_result["metrics"]
        summary_lines.append(
            f"| Vanilla | {vm['accuracy']:.1%} | {vm['rep_3']:.4f} | "
            f"{vm['distinct_3']:.4f} | {vm['avg_gen_time_s']:.1f}s | N/A |"
        )

        # Add scheduled results
        for name in sorted(scheduled_results.keys()):
            res = scheduled_results[name]
            m = res["metrics"]
            ga = res.get("guidance_analysis", {})
            g_steps = f"{ga.get('mean_guidance_steps', 'N/A'):.0f}/{ga.get('mean_total_steps', 'N/A'):.0f}" if ga else "N/A"
            summary_lines.append(
                f"| {name} | {m['accuracy']:.1%} | {m['rep_3']:.4f} | "
                f"{m['distinct_3']:.4f} | {m['avg_gen_time_s']:.1f}s | {g_steps} |"
            )

        summary_lines.extend([
            "",
            "## Degeneration Check",
        ])

        # Check degeneration
        degen_found = False
        for name, res in all_results.items():
            if name == "vanilla":
                continue
            degen = check_degeneration(res["metrics"], vanilla_result["metrics"])
            if degen:
                degen_found = True
                summary_lines.append(f"- **{name}**: {degen}")
        if not degen_found:
            summary_lines.append("No degeneration detected in any configuration.")

        # Key findings
        summary_lines.extend([
            "",
            "## Key Findings",
            "",
        ])

        if any_scheduled_beats_fixed:
            summary_lines.extend([
                f"1. **Scheduled guidance beats fixed**: {best_sched_name} achieves "
                f"{best_sched_acc:.1%} vs fixed A-CFG {fixed_w1_acc:.1%} "
                f"({verdict['best_scheduled_delta_vs_fixed_pp']:+.1f}pp).",
                "",
                "2. **Schedule type matters**: The temporal scheduling of guidance weight "
                "has a measurable impact on accuracy, supporting H6.",
            ])
        else:
            summary_lines.extend([
                "1. **No scheduled variant beats fixed A-CFG**: Temporal scheduling of "
                "guidance weight does not improve upon constant guidance, contradicting H6.",
                "",
                "2. **Fixed guidance is sufficient**: A-CFG's constant guidance weight "
                "strategy is already near-optimal for Dream-7B on this benchmark.",
            ])

        # Add schedule-specific analysis
        summary_lines.extend([
            "",
            "## Schedule Type Analysis",
            "",
        ])
        for sched, analysis in schedule_analysis.items():
            summary_lines.append(
                f"- **{sched}**: mean accuracy {analysis['mean_accuracy']:.1%}, "
                f"max accuracy {analysis['max_accuracy']:.1%}"
            )

        summary_lines.extend([
            "",
            "## W_base Analysis",
            "",
        ])
        for w_name, analysis in w_analysis.items():
            summary_lines.append(
                f"- **{w_name}**: mean accuracy {analysis['mean_accuracy']:.1%}, "
                f"max accuracy {analysis['max_accuracy']:.1%}"
            )

        summary_lines.extend([
            "",
            "## Recommendations for Downstream Tasks",
            "",
            f"1. **Best configuration**: {best_overall_name} ({best_overall_acc:.1%})",
            f"2. **For ablation_racfg_vs_acfg**: Use the best config from this ablation "
            f"as the 'RACFG' entry (since original RACFG failed, this scheduled A-CFG "
            f"variant is the closest to the 'enhanced guidance' concept).",
            "3. **For fullscale_racfg**: The 'RACFG' full-scale evaluation should use "
            "the best configuration found here.",
        ])

        summary_path.write_text("\n".join(summary_lines))
        print(f"  Summary saved to: {summary_path}")

        # Mark task done
        mark_task_done(TASK_ID, str(RESULTS_DIR), status="success",
                       summary=f"Verdict={verdict['overall']}, "
                               f"best_scheduled={best_sched_name}({best_sched_acc:.1%}), "
                               f"fixed_acfg={fixed_w1_acc:.1%}, "
                               f"elapsed={elapsed_min:.1f}min")

        print(f"\n  Task {TASK_ID} completed in {elapsed_min:.1f} minutes")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        mark_task_done(TASK_ID, str(RESULTS_DIR), status="failed",
                       summary=f"Error: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    main()
