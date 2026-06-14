"""
RACFG Re-mask Percentage Ablation on Countdown-100 (PILOT mode).

Task: ablation_racfg_remask_pct
GPU: 2 (single)
Mode: PILOT (16 samples, seed 42)

Tests RACFG with remask_pct ∈ {5%, 10%, 20%} on Countdown-100 (pilot=16 samples).
Also runs vanilla and A-CFG baselines for comparison.

Based on pilot_racfg findings (stability scores uniformly ~0.997, NO-GO),
we test with BOTH:
  - Original EMA lambda=0.7 (for consistency)
  - Reduced EMA lambda=0.3 (pilot recommendation to amplify step-to-step differences)

Fixed: w_base=1.0, schedule=threshold_70_30 (from task_plan.json), JSD stability.

Pass criteria: At least one m% > A-CFG baseline accuracy.
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
from bsd_racfg.racfg import RACFGConfig, ACFGConfig, racfg_generate, acfg_generate

TASK_ID = "ablation_racfg_remask_pct"
RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/full")
PILOTS_DIR = Path(f"{PROJECT_DIR}/exp/results/pilots")
N_SAMPLES = 16  # PILOT mode
SEED = 42
DEVICE = "cuda:0"

# Ablation configurations
REMASK_PCTS = [0.05, 0.10, 0.20]
# Test with both original and reduced EMA lambda
EMA_LAMBDAS = [0.7, 0.3]

# Fixed parameters
W_BASE = 1.0
W_MAX = 2.0
SCHEDULE_TYPE = "threshold_70_30"
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4

# A-CFG baseline config
ACFG_W = 1.0
ACFG_REMASK_PCT = 0.10


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
    all_stability = []

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

        # Collect stability data if available
        stability_data = diag.get("stability_data", [])
        if stability_data:
            sample["n_guidance_steps"] = sum(
                1 for s in diag.get("step_diagnostics", [])
                if s.get("guidance_applied")
            )
            all_stability.append({
                "idx": idx,
                "stability_records": stability_data,
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

    # Add stability analysis if available
    if all_stability:
        all_stab_means = []
        all_stab_stds = []
        for sd in all_stability:
            for rec in sd["stability_records"]:
                all_stab_means.append(rec["stability_mean"])
                all_stab_stds.append(rec["stability_std"])
        result["stability_analysis"] = {
            "overall_stability_mean": float(np.mean(all_stab_means)) if all_stab_means else 0,
            "overall_stability_std": float(np.std(all_stab_means)) if all_stab_means else 0,
            "per_sample_stability_std_mean": float(np.mean(all_stab_stds)) if all_stab_stds else 0,
            "stability_variance_meaningful": (
                float(np.std(all_stab_means)) > 0.01 if all_stab_means else False
            ),
        }

    print(f"    → Accuracy: {accuracy:.1%} ({n_correct}/{len(problems)}) | "
          f"rep-3: {diversity['rep_3']:.4f} | dist-3: {diversity['distinct_3']:.4f} | "
          f"avg: {np.mean(gen_times):.1f}s")

    return result


def main():
    start_time = datetime.now()

    print("=" * 70)
    print(f"  RACFG Re-mask % Ablation — Countdown-{N_SAMPLES} (PILOT)")
    print(f"  Task ID: {TASK_ID}")
    print(f"  Time: {start_time.isoformat()}")
    print(f"  Device: GPU 2 (mapped to {DEVICE})")
    print(f"  Seed: {SEED}")
    print(f"  Configs: remask_pct={REMASK_PCTS}, ema_lambda={EMA_LAMBDAS}")
    print("=" * 70)

    # Write PID file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        # Generate problems (same as all other tasks, consistent seeding)
        problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
        print(f"\nGenerated {len(problems)} Countdown problems")

        # Load model
        print("\nLoading Dream-7B...")
        model, tokenizer, embedding_layer = load_dream(DEVICE)

        all_results = {}
        total_configs = 1 + 1 + len(REMASK_PCTS) * len(EMA_LAMBDAS)  # vanilla + acfg + racfg variants
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
        # 2. A-CFG baseline
        # ─────────────────────────────────────────────
        config_idx += 1
        print(f"\n{'='*60}")
        print(f"  [{config_idx}/{total_configs}] A-CFG Baseline (w={ACFG_W}, remask={ACFG_REMASK_PCT})")
        print(f"{'='*60}")

        acfg_config = ACFGConfig(
            remask_pct=ACFG_REMASK_PCT, w=ACFG_W,
            gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
        )

        def acfg_fn(m, t, p, d):
            return acfg_generate(m, t, p, acfg_config, d)

        acfg_result = run_method(model, tokenizer, problems, "A-CFG",
                                 acfg_fn, DEVICE)
        all_results["acfg"] = acfg_result
        report_progress(TASK_ID, str(RESULTS_DIR), config_idx, total_configs,
                        metric={"phase": "acfg", "accuracy": acfg_result["metrics"]["accuracy"]})

        # ─────────────────────────────────────────────
        # 3. RACFG variants: remask_pct × ema_lambda
        # ─────────────────────────────────────────────
        racfg_results = {}

        for remask_pct in REMASK_PCTS:
            for ema_lambda in EMA_LAMBDAS:
                config_idx += 1
                config_name = f"RACFG_m{int(remask_pct*100)}_lam{ema_lambda}"

                print(f"\n{'='*60}")
                print(f"  [{config_idx}/{total_configs}] {config_name}")
                print(f"  remask_pct={remask_pct}, ema_lambda={ema_lambda}")
                print(f"  w_base={W_BASE}, schedule={SCHEDULE_TYPE}")
                print(f"{'='*60}")

                racfg_config = RACFGConfig(
                    remask_pct=remask_pct,
                    w_base=W_BASE,
                    w_max=W_MAX,
                    schedule_type=SCHEDULE_TYPE,
                    stability_ema_lambda=ema_lambda,
                    gen_len=GEN_LEN,
                    steps=STEPS,
                    temperature=TEMPERATURE,
                )

                def make_racfg_fn(cfg):
                    def fn(m, t, p, d):
                        return racfg_generate(m, t, p, cfg, d, track_stability=True)
                    return fn

                r = run_method(model, tokenizer, problems, config_name,
                               make_racfg_fn(racfg_config), DEVICE)
                r["config"].update({
                    "remask_pct": remask_pct,
                    "w_base": W_BASE,
                    "w_max": W_MAX,
                    "schedule_type": SCHEDULE_TYPE,
                    "stability_ema_lambda": ema_lambda,
                })
                racfg_results[config_name] = r
                all_results[config_name] = r

                report_progress(
                    TASK_ID, str(RESULTS_DIR), config_idx, total_configs,
                    metric={
                        "phase": config_name,
                        "accuracy": r["metrics"]["accuracy"],
                        "remask_pct": remask_pct,
                        "ema_lambda": ema_lambda,
                    },
                )

                # Clear CUDA cache between configs
                torch.cuda.empty_cache()
                gc.collect()

        # ─────────────────────────────────────────────
        # 4. Analysis & Comparison
        # ─────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*70}")

        all_result_list = [vanilla_result, acfg_result] + list(racfg_results.values())
        print_comparison_table(all_result_list)

        # Degeneration checks
        vanilla_div = vanilla_result["metrics"]
        print(f"\n  Degeneration Checks (relative to vanilla):")
        for name, r in all_results.items():
            if name == "vanilla":
                continue
            warnings = check_degeneration(r["metrics"], vanilla_div)
            if warnings:
                print(f"    {name}: {list(warnings.keys())}")
            else:
                print(f"    {name}: OK")

        # Best RACFG config
        acfg_acc = acfg_result["metrics"]["accuracy"]
        best_racfg_name = None
        best_racfg_acc = -1
        for name, r in racfg_results.items():
            acc = r["metrics"]["accuracy"]
            if acc > best_racfg_acc:
                best_racfg_acc = acc
                best_racfg_name = name

        # Pass criteria: at least one m% > A-CFG baseline
        any_beats_acfg = any(
            r["metrics"]["accuracy"] > acfg_acc
            for r in racfg_results.values()
        )

        # Also check which remask_pct is best (aggregated across lambdas)
        pct_accs = {}
        for pct in REMASK_PCTS:
            pct_key = f"m{int(pct*100)}"
            accs = []
            for lam in EMA_LAMBDAS:
                config_name = f"RACFG_m{int(pct*100)}_lam{lam}"
                accs.append(racfg_results[config_name]["metrics"]["accuracy"])
            pct_accs[pct_key] = {
                "mean_accuracy": float(np.mean(accs)),
                "max_accuracy": float(max(accs)),
                "accuracies": accs,
            }

        # Lambda comparison
        lam_accs = {}
        for lam in EMA_LAMBDAS:
            accs = []
            for pct in REMASK_PCTS:
                config_name = f"RACFG_m{int(pct*100)}_lam{lam}"
                accs.append(racfg_results[config_name]["metrics"]["accuracy"])
            lam_accs[f"lambda_{lam}"] = {
                "mean_accuracy": float(np.mean(accs)),
                "max_accuracy": float(max(accs)),
            }

        # Qualitative samples from best config
        if best_racfg_name:
            print(f"\n--- Best RACFG ({best_racfg_name}) Qualitative Samples ---")
            print_qualitative_samples(racfg_results[best_racfg_name]["per_sample"], n=10)

        # ─────────────────────────────────────────────
        # 5. Verdict
        # ─────────────────────────────────────────────
        verdict = {
            "overall": "GO" if any_beats_acfg else "NO-GO",
            "pass_criteria": "At least one m% > A-CFG baseline",
            "any_beats_acfg": any_beats_acfg,
            "vanilla_accuracy": vanilla_result["metrics"]["accuracy"],
            "acfg_accuracy": acfg_acc,
            "best_racfg": best_racfg_name,
            "best_racfg_accuracy": best_racfg_acc,
            "best_racfg_delta_vs_acfg_pp": round((best_racfg_acc - acfg_acc) * 100, 1),
            "remask_pct_analysis": pct_accs,
            "ema_lambda_analysis": lam_accs,
        }

        print(f"\n{'='*70}")
        print(f"  ABLATION VERDICT: {verdict['overall']}")
        print(f"{'='*70}")
        print(f"  Vanilla:     {vanilla_result['metrics']['accuracy']:.1%}")
        print(f"  A-CFG:       {acfg_acc:.1%}")
        print(f"  Best RACFG:  {best_racfg_acc:.1%} ({best_racfg_name})")
        print(f"  Delta vs A-CFG: {verdict['best_racfg_delta_vs_acfg_pp']:+.1f}pp")
        print(f"  Any beats A-CFG: {any_beats_acfg}")
        print(f"\n  Remask % analysis:")
        for pct_key, info in pct_accs.items():
            print(f"    {pct_key}: mean={info['mean_accuracy']:.1%}, max={info['max_accuracy']:.1%}")
        print(f"\n  EMA Lambda analysis:")
        for lam_key, info in lam_accs.items():
            print(f"    {lam_key}: mean={info['mean_accuracy']:.1%}, max={info['max_accuracy']:.1%}")
        print(f"{'='*70}")

        # ─────────────────────────────────────────────
        # 6. Save results
        # ─────────────────────────────────────────────
        end_time = datetime.now()
        elapsed_min = (end_time - start_time).total_seconds() / 60

        combined_result = {
            "task_id": TASK_ID,
            "mode": "PILOT",
            "benchmark": f"Countdown-{N_SAMPLES}",
            "seed": SEED,
            "timestamp": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_min": round(elapsed_min, 1),
            "verdict": verdict,
            "configs_tested": {
                "remask_pcts": REMASK_PCTS,
                "ema_lambdas": EMA_LAMBDAS,
                "w_base": W_BASE,
                "w_max": W_MAX,
                "schedule_type": SCHEDULE_TYPE,
            },
            "results": {
                "vanilla": vanilla_result,
                "acfg": acfg_result,
                **racfg_results,
            },
        }

        # Save to full results dir (ablation output location)
        out_file = RESULTS_DIR / "racfg_remask_pct_ablation_countdown100.json"
        save_results(combined_result, str(out_file))

        # Mark done
        mark_task_done(
            TASK_ID, str(RESULTS_DIR),
            status="success",
            summary=(
                f"RACFG remask ablation {verdict['overall']}: "
                f"best={best_racfg_name} ({best_racfg_acc:.1%}) vs A-CFG ({acfg_acc:.1%}), "
                f"delta={verdict['best_racfg_delta_vs_acfg_pp']:+.1f}pp, "
                f"elapsed={elapsed_min:.1f}min"
            ),
        )

        print(f"\nTotal elapsed: {elapsed_min:.1f} minutes")
        return any_beats_acfg

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
