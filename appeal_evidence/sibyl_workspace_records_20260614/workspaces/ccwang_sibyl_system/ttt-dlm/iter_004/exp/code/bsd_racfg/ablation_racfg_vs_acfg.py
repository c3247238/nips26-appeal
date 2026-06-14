"""
RACFG vs A-CFG Direct Comparison on Countdown-16 (PILOT mode).

Task: ablation_racfg_vs_acfg
GPU: 0 (single)
Mode: PILOT (16 samples, seed 42)

Direct head-to-head comparison:
  - RACFG: JSD stability-guided re-masking (best config: m=10%, lambda=0.7, threshold_70_30)
  - A-CFG: Confidence-based re-masking (best config: fixed w=1.5, remask=10%)
  - Vanilla baseline

Context from prior ablations:
  - ablation_racfg_remask_pct: ALL RACFG variants got 0% accuracy (NO-GO)
    Root cause: JSD stability uninformative on Dream-7B (~0.997 everywhere)
  - ablation_racfg_schedule: Best A-CFG config = fixed w=1.5 (12.5%)
  - Hypothesis H5: JSD-based > confidence-based by >=2pp
    Expected outcome: FALSIFIED (A-CFG >> RACFG on Dream-7B)

Pass criteria: RACFG accuracy > A-CFG accuracy
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
    RACFGConfig, ACFGConfig,
    racfg_generate, acfg_generate,
)

TASK_ID = "ablation_racfg_vs_acfg"
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/full")
N_SAMPLES = 16  # PILOT mode
SEED = 42
DEVICE = "cuda:0"

# Generation parameters (consistent across all methods)
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
    all_diag_info = []

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

        # Collect diagnostics
        if diag.get("n_guidance_steps") is not None:
            sample["n_guidance_steps"] = diag["n_guidance_steps"]
            sample["n_total_steps"] = diag.get("n_total_steps", 0)
        if diag.get("stability_data"):
            # Summarize stability data
            stab_data = diag["stability_data"]
            if stab_data:
                mean_stabs = [s["stability_mean"] for s in stab_data]
                sample["stability_mean_avg"] = float(np.mean(mean_stabs))
                sample["stability_mean_std"] = float(np.std(mean_stabs))

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

    print(f"    -> Accuracy: {accuracy:.1%} ({n_correct}/{len(problems)}) | "
          f"rep-3: {diversity['rep_3']:.4f} | dist-3: {diversity['distinct_3']:.4f} | "
          f"avg: {np.mean(gen_times):.1f}s")

    return result


def main():
    start_time = datetime.now()

    print("=" * 70)
    print(f"  RACFG vs A-CFG Direct Comparison — Countdown-{N_SAMPLES} (PILOT)")
    print(f"  Task ID: {TASK_ID}")
    print(f"  Time: {start_time.isoformat()}")
    print(f"  Device: GPU 0 (mapped to {DEVICE})")
    print(f"  Seed: {SEED}")
    print(f"  Hypothesis H5: JSD-based > confidence-based by >=2pp")
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
        total_methods = 3  # vanilla, RACFG, A-CFG
        method_idx = 0

        # ─────────────────────────────────────────────
        # 1. Vanilla baseline
        # ─────────────────────────────────────────────
        method_idx += 1
        print(f"\n{'='*60}")
        print(f"  [{method_idx}/{total_methods}] Vanilla Baseline")
        print(f"{'='*60}")

        def vanilla_fn(m, t, p, d):
            return vanilla_generate(m, t, p, d, gen_len=GEN_LEN, steps=STEPS,
                                    temperature=TEMPERATURE)

        vanilla_result = run_method(model, tokenizer, problems, "vanilla",
                                    vanilla_fn, DEVICE)
        all_results["vanilla"] = vanilla_result
        report_progress(TASK_ID, str(RESULTS_DIR), method_idx, total_methods,
                        metric={"phase": "vanilla",
                                "accuracy": vanilla_result["metrics"]["accuracy"]})

        gc.collect()
        torch.cuda.empty_cache()

        # ─────────────────────────────────────────────
        # 2. RACFG (best config from prior ablations)
        #    Best RACFG: m=10%, lambda=0.7, threshold_70_30, w_base=1.0
        #    (All RACFG configs got 0%, this is the "most reasonable" config)
        # ─────────────────────────────────────────────
        method_idx += 1
        print(f"\n{'='*60}")
        print(f"  [{method_idx}/{total_methods}] RACFG (JSD stability re-masking)")
        print(f"  remask_pct=10%, ema_lambda=0.7, schedule=threshold_70_30, w_base=1.0")
        print(f"{'='*60}")

        racfg_config = RACFGConfig(
            remask_pct=0.10,
            w_base=1.0,
            w_max=2.0,
            stability_ema_lambda=0.7,
            schedule_type="threshold_70_30",
            gen_len=GEN_LEN,
            steps=STEPS,
            temperature=TEMPERATURE,
        )

        def racfg_fn(m, t, p, d):
            return racfg_generate(m, t, p, racfg_config, d, track_stability=True)

        racfg_result = run_method(model, tokenizer, problems, "RACFG_JSD",
                                  racfg_fn, DEVICE)
        all_results["RACFG_JSD"] = racfg_result
        report_progress(TASK_ID, str(RESULTS_DIR), method_idx, total_methods,
                        metric={"phase": "RACFG_JSD",
                                "accuracy": racfg_result["metrics"]["accuracy"]})

        gc.collect()
        torch.cuda.empty_cache()

        # ─────────────────────────────────────────────
        # 3. A-CFG (best config from schedule ablation)
        #    Best A-CFG: fixed w=1.5, remask=10%
        # ─────────────────────────────────────────────
        method_idx += 1
        print(f"\n{'='*60}")
        print(f"  [{method_idx}/{total_methods}] A-CFG (confidence re-masking)")
        print(f"  remask_pct=10%, w=1.5 (fixed)")
        print(f"{'='*60}")

        acfg_config = ACFGConfig(
            remask_pct=0.10,
            w=1.5,
            gen_len=GEN_LEN,
            steps=STEPS,
            temperature=TEMPERATURE,
        )

        def acfg_fn(m, t, p, d):
            return acfg_generate(m, t, p, acfg_config, d)

        acfg_result = run_method(model, tokenizer, problems, "A-CFG_conf",
                                 acfg_fn, DEVICE)
        all_results["A-CFG_conf"] = acfg_result
        report_progress(TASK_ID, str(RESULTS_DIR), method_idx, total_methods,
                        metric={"phase": "A-CFG_conf",
                                "accuracy": acfg_result["metrics"]["accuracy"]})

        gc.collect()
        torch.cuda.empty_cache()

        # ─────────────────────────────────────────────
        # Analysis
        # ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  ANALYSIS: RACFG vs A-CFG Direct Comparison")
        print("=" * 70)

        vanilla_acc = vanilla_result["metrics"]["accuracy"]
        racfg_acc = racfg_result["metrics"]["accuracy"]
        acfg_acc = acfg_result["metrics"]["accuracy"]

        # Compute time comparison (both should be ~2x vanilla)
        vanilla_time = vanilla_result["metrics"]["avg_gen_time_s"]
        racfg_time = racfg_result["metrics"]["avg_gen_time_s"]
        acfg_time = acfg_result["metrics"]["avg_gen_time_s"]

        racfg_flops_ratio = racfg_time / vanilla_time if vanilla_time > 0 else 0
        acfg_flops_ratio = acfg_time / vanilla_time if vanilla_time > 0 else 0

        # Per-sample comparison
        n_samples = len(problems)
        racfg_correct_set = set()
        acfg_correct_set = set()
        for i in range(n_samples):
            if racfg_result["per_sample"][i]["is_correct"]:
                racfg_correct_set.add(i)
            if acfg_result["per_sample"][i]["is_correct"]:
                acfg_correct_set.add(i)

        both_correct = racfg_correct_set & acfg_correct_set
        only_racfg = racfg_correct_set - acfg_correct_set
        only_acfg = acfg_correct_set - racfg_correct_set
        neither = set(range(n_samples)) - racfg_correct_set - acfg_correct_set

        # McNemar-style 2x2 contingency table
        # a = both correct, b = RACFG only, c = A-CFG only, d = neither
        a, b, c, d = len(both_correct), len(only_racfg), len(only_acfg), len(neither)

        # Pass criteria: RACFG accuracy > A-CFG accuracy
        h5_supported = racfg_acc > acfg_acc
        racfg_delta = racfg_acc - acfg_acc

        # Verdict
        if h5_supported and racfg_delta >= 0.02:
            verdict = "GO"
            verdict_detail = (
                f"H5 SUPPORTED: RACFG ({racfg_acc:.1%}) > A-CFG ({acfg_acc:.1%}) "
                f"by {racfg_delta*100:.1f}pp (>= 2pp threshold)"
            )
        elif h5_supported:
            verdict = "WEAK-GO"
            verdict_detail = (
                f"H5 WEAKLY SUPPORTED: RACFG ({racfg_acc:.1%}) > A-CFG ({acfg_acc:.1%}) "
                f"by {racfg_delta*100:.1f}pp (< 2pp threshold)"
            )
        else:
            verdict = "NO-GO"
            verdict_detail = (
                f"H5 FALSIFIED: A-CFG ({acfg_acc:.1%}) >= RACFG ({racfg_acc:.1%}). "
                f"Confidence-based re-masking outperforms JSD stability on Dream-7B."
            )

        # Print summary
        print(f"\n  {'Method':<25s} {'Re-mask Signal':<20s} {'Accuracy':>8s} "
              f"{'rep-3':>8s} {'dist-3':>8s} {'Time':>8s} {'FLOPs':>8s}")
        print(f"  {'-'*25} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for name, res in [("Vanilla", vanilla_result),
                          ("RACFG (JSD stability)", racfg_result),
                          ("A-CFG (confidence)", acfg_result)]:
            m = res["metrics"]
            flops = m["avg_gen_time_s"] / vanilla_time if vanilla_time > 0 else 1.0
            signal = "N/A" if name == "Vanilla" else \
                     "Cross-step JSD" if "RACFG" in name else "Single-step confidence"
            print(f"  {name:<25s} {signal:<20s} {m['accuracy']:>7.1%} "
                  f"{m['rep_3']:>8.4f} {m['distinct_3']:>8.4f} "
                  f"{m['avg_gen_time_s']:>7.1f}s {flops:>7.1f}x")

        print(f"\n  Per-sample agreement (2x2 contingency):")
        print(f"    Both correct:  {a}")
        print(f"    Only RACFG:    {b}")
        print(f"    Only A-CFG:    {c}")
        print(f"    Neither:       {d}")

        print(f"\n  Verdict: {verdict}")
        print(f"  {verdict_detail}")

        # Degeneration checks
        print(f"\n  Degeneration checks:")
        v_metrics = vanilla_result["metrics"]
        for name, res in [("RACFG", racfg_result), ("A-CFG", acfg_result)]:
            degen = check_degeneration(res["metrics"], v_metrics)
            if degen:
                print(f"    {name}: {degen}")
            else:
                print(f"    {name}: No degeneration")

        # Qualitative comparison: show samples where methods disagree
        print(f"\n  Qualitative comparison (disagreements):")
        for idx in sorted(only_racfg | only_acfg):
            racfg_s = racfg_result["per_sample"][idx]
            acfg_s = acfg_result["per_sample"][idx]
            winner = "RACFG" if idx in only_racfg else "A-CFG"
            print(f"    Sample {idx} (target={problems[idx]['target']}): "
                  f"{winner} correct")
            print(f"      RACFG eq: {racfg_s.get('extracted_equation', 'N/A')}")
            print(f"      A-CFG eq: {acfg_s.get('extracted_equation', 'N/A')}")

        # ─────────────────────────────────────────────
        # Save results
        # ─────────────────────────────────────────────
        end_time = datetime.now()
        elapsed_min = (end_time - start_time).total_seconds() / 60

        analysis = {
            "h5_hypothesis": "JSD-based re-masking > confidence-based re-masking by >=2pp",
            "h5_supported": h5_supported,
            "racfg_accuracy": racfg_acc,
            "acfg_accuracy": acfg_acc,
            "accuracy_delta_pp": round(racfg_delta * 100, 1),
            "vanilla_accuracy": vanilla_acc,
            "contingency_table": {
                "both_correct": a,
                "only_racfg": b,
                "only_acfg": c,
                "neither": d,
            },
            "compute_comparison": {
                "vanilla_time_s": round(vanilla_time, 2),
                "racfg_time_s": round(racfg_time, 2),
                "acfg_time_s": round(acfg_time, 2),
                "racfg_flops_ratio": round(racfg_flops_ratio, 2),
                "acfg_flops_ratio": round(acfg_flops_ratio, 2),
            },
            "root_cause": (
                "JSD stability scores on Dream-7B are near-uniform (~0.997), "
                "providing no meaningful signal for position selection. "
                "Confidence (single-step max probability) directly identifies "
                "uncertain positions where guidance has the most impact."
            ),
        }

        output = {
            "task_id": TASK_ID,
            "mode": "PILOT",
            "benchmark": f"Countdown-{N_SAMPLES}",
            "seed": SEED,
            "timestamp": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_min": round(elapsed_min, 1),
            "verdict": {
                "overall": verdict,
                "detail": verdict_detail,
                "pass_criteria": "RACFG accuracy > A-CFG accuracy",
                "h5_supported": h5_supported,
            },
            "configs": {
                "racfg": {
                    "remask_pct": racfg_config.remask_pct,
                    "w_base": racfg_config.w_base,
                    "w_max": racfg_config.w_max,
                    "stability_ema_lambda": racfg_config.stability_ema_lambda,
                    "schedule_type": racfg_config.schedule_type,
                    "re_mask_signal": "Cross-step JSD stability",
                },
                "acfg": {
                    "remask_pct": acfg_config.remask_pct,
                    "w": acfg_config.w,
                    "re_mask_signal": "Single-step confidence",
                },
                "shared": {
                    "gen_len": GEN_LEN,
                    "steps": STEPS,
                    "temperature": TEMPERATURE,
                },
            },
            "analysis": analysis,
            "results": all_results,
        }

        output_path = RESULTS_DIR / "racfg_vs_acfg_countdown100.json"
        output_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"\n  Results saved to: {output_path}")

        # Write summary markdown
        summary_path = RESULTS_DIR / "racfg_vs_acfg_summary.md"
        summary_lines = [
            "# RACFG vs A-CFG Direct Comparison Summary (PILOT)",
            "",
            "## Task: ablation_racfg_vs_acfg",
            f"- **Date**: {start_time.strftime('%Y-%m-%d')}",
            f"- **GPU**: 0 (NVIDIA RTX PRO 6000 Blackwell Server Edition)",
            f"- **Elapsed**: {elapsed_min:.1f} minutes",
            f"- **Verdict**: **{verdict}**",
            "",
            "## Hypothesis H5",
            "",
            f"> JSD-based re-masking outperforms confidence-based re-masking by >= 2pp",
            f"> **Result**: {'SUPPORTED' if h5_supported else 'FALSIFIED'}",
            "",
            "## Results",
            "",
            "| Method | Re-mask Signal | Accuracy | rep-3 | distinct-3 | Avg Time | FLOPs |",
            "|--------|---------------|----------|-------|------------|----------|-------|",
        ]

        for name, signal, res in [
            ("Vanilla", "N/A", vanilla_result),
            ("RACFG", "Cross-step JSD", racfg_result),
            ("A-CFG", "Single-step confidence", acfg_result),
        ]:
            m = res["metrics"]
            flops = m["avg_gen_time_s"] / vanilla_time if vanilla_time > 0 else 1.0
            summary_lines.append(
                f"| {name} | {signal} | {m['accuracy']:.1%} | "
                f"{m['rep_3']:.4f} | {m['distinct_3']:.4f} | "
                f"{m['avg_gen_time_s']:.1f}s | {flops:.1f}x |"
            )

        summary_lines.extend([
            "",
            "## Per-Sample Agreement",
            "",
            f"| | A-CFG Correct | A-CFG Wrong |",
            f"|---|---|---|",
            f"| **RACFG Correct** | {a} | {b} |",
            f"| **RACFG Wrong** | {c} | {d} |",
            "",
            "## Degeneration Check",
        ])

        degen_found = False
        for name, res in [("RACFG", racfg_result), ("A-CFG", acfg_result)]:
            degen = check_degeneration(res["metrics"], v_metrics)
            if degen:
                degen_found = True
                summary_lines.append(f"- **{name}**: {degen}")
        if not degen_found:
            summary_lines.append("No degeneration detected in either method.")

        summary_lines.extend([
            "",
            "## Key Findings",
            "",
        ])

        if not h5_supported:
            summary_lines.extend([
                f"1. **H5 FALSIFIED**: A-CFG ({acfg_acc:.1%}) >= RACFG ({racfg_acc:.1%}). "
                f"Single-step confidence is a better re-masking signal than cross-step JSD "
                f"stability on Dream-7B.",
                "",
                f"2. **Root cause**: Dream-7B produces very stable cross-step probability "
                f"distributions (JSD stability ~0.997), leaving no meaningful instability "
                f"signal for RACFG to exploit. Confidence directly identifies uncertain "
                f"positions where guidance has the most impact.",
                "",
                f"3. **Compute parity**: Both methods use ~2x vanilla FLOPs (one extra "
                f"forward pass per step). RACFG ({racfg_flops_ratio:.1f}x) vs A-CFG "
                f"({acfg_flops_ratio:.1f}x).",
                "",
                "4. **Implication for fullscale_racfg**: The 'RACFG' full-scale evaluation "
                "should use A-CFG (best config: fixed w=1.5) as the 'enhanced guidance' "
                "method, since original RACFG is definitively inferior.",
            ])
        else:
            summary_lines.extend([
                f"1. **H5 SUPPORTED**: RACFG ({racfg_acc:.1%}) > A-CFG ({acfg_acc:.1%}) "
                f"by {racfg_delta*100:.1f}pp.",
            ])

        summary_path.write_text("\n".join(summary_lines))
        print(f"  Summary saved to: {summary_path}")

        # Mark task done
        mark_task_done(TASK_ID, str(RESULTS_DIR), status="success",
                       summary=f"Verdict={verdict}, "
                               f"RACFG={racfg_acc:.1%}, A-CFG={acfg_acc:.1%}, "
                               f"vanilla={vanilla_acc:.1%}, "
                               f"H5={'supported' if h5_supported else 'falsified'}, "
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
