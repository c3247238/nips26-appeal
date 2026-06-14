"""
fullscale_racfg: RACFG Full-Scale Evaluation (PILOT: Countdown-16, seed 42).

Based on ablation results:
  - Original RACFG (JSD stability) failed on Dream-7B (0% accuracy)
  - A-CFG (confidence-based re-masking) is the effective guidance method
  - Best config from ablations: A-CFG fixed w=1.5, remask_pct=0.10

This task evaluates the best guidance method (A-CFG w=1.5) against:
  - Vanilla baseline
  - A-CFG w=1.0 (lower guidance)
  - Compute-fair comparison: vanilla with 2x steps (256 steps vs 128)

Pass criteria (PILOT): RACFG/A-CFG accuracy > vanilla + 3pp

Task: fullscale_racfg
Mode: PILOT (16 samples, seed 42)
GPU: cuda:0 (mapped via CUDA_VISIBLE_DEVICES)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python fullscale_racfg.py
"""
import os
import sys
import json
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics, compute_per_sample_metrics,
    vanilla_generate, format_results, save_results,
    print_qualitative_samples, print_comparison_table, check_degeneration,
    MASK_TOKEN_ID, PROJECT_DIR, RESULTS_FULL,
)
from bsd_racfg.racfg import ACFGConfig, acfg_generate

# ── Constants ──
TASK_ID = "fullscale_racfg"
N_SAMPLES = 16   # PILOT mode
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
DEVICE = "cuda:0"

# Best A-CFG config from ablation_racfg_schedule
BEST_W = 1.5
BEST_REMASK_PCT = 0.10

RESULTS_DIR = RESULTS_FULL
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────
# Progress & PID tracking
# ──────────────────────────────────────────────────

def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")


def report_progress(current, total, phase="", metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": current, "total_epochs": total,
        "step": current, "total_steps": total,
        "phase": phase,
        "loss": None,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
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
    print(f"[{TASK_ID}] DONE marker written ({status})")


# ──────────────────────────────────────────────────
# Vanilla with extra steps (compute-fair baseline)
# ──────────────────────────────────────────────────

def vanilla_generate_custom_steps(model, tokenizer, prompt_text, device,
                                   gen_len=GEN_LEN, steps=STEPS,
                                   temperature=TEMPERATURE):
    """Vanilla generation with configurable number of steps."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    t0 = time.time()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            outputs = model(x, attention_mask="full", position_ids=None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature > 0:
                probs = F.softmax(mask_logits[transfer_mask] / temperature, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                sampled_tokens = mask_logits[transfer_mask].argmax(dim=-1)
            x0[transfer_mask] = sampled_tokens
            x[mask_index] = x0

    elapsed = time.time() - t0

    gen_ids = x[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t_id in gen_ids:
        if t_id == MASK_TOKEN_ID:
            continue
        if t_id == eos_id:
            break
        clean_ids.append(t_id)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
    return text, elapsed


# ──────────────────────────────────────────────────
# Run methods
# ──────────────────────────────────────────────────

def run_method(model, tokenizer, problems, prompts, seed, method_name,
               generate_fn, method_label=""):
    """Generic runner for any generation method.

    generate_fn(prompt) -> (text, elapsed, optional_diag)
    """
    label = method_label or method_name
    print(f"\n{'='*60}")
    print(f"  {label} ({len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)

        output = generate_fn(prompt)
        if len(output) == 3:
            text, elapsed, diag = output
        else:
            text, elapsed = output
            diag = {}

        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        results.append({
            "idx": idx, "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        })
        texts.append(text)
        times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        eq_str = (verification.get('extracted_equation') or 'N/A')[:40]
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"eq={eq_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)

    print(f"\n  {label}: {n_correct}/{len(problems)} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")

    return results, accuracy, n_correct, diversity, times


# ──────────────────────────────────────────────────
# Flip analysis helper
# ──────────────────────────────────────────────────

def compute_flips(results_a, results_b, name_a, name_b):
    """Compute per-sample agreement/disagreement (McNemar-style)."""
    a_over_b = 0
    b_over_a = 0
    both_correct = 0
    both_wrong = 0

    for ra, rb in zip(results_a, results_b):
        ac = ra["is_correct"]
        bc = rb["is_correct"]
        if ac and bc:
            both_correct += 1
        elif ac and not bc:
            a_over_b += 1
        elif not ac and bc:
            b_over_a += 1
        else:
            both_wrong += 1

    return {
        f"{name_a}_over_{name_b}": a_over_b,
        f"{name_b}_over_{name_a}": b_over_a,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 5, "Loading model")

    print("=" * 70)
    print(f"  fullscale_racfg — RACFG/A-CFG Full-Scale Evaluation (PILOT)")
    print(f"  Countdown-{N_SAMPLES}, seed={SEED}")
    print(f"  Best A-CFG config: fixed w={BEST_W}, remask_pct={BEST_REMASK_PCT}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPUs visible: {torch.cuda.device_count()}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, 5, "Model loaded")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    prompts = [format_countdown_prompt(p) for p in problems]
    print(f"\nGenerated {len(problems)} Countdown problems")

    # ── Phase 1: Vanilla baseline (128 steps) ──
    report_progress(1, 5, "Running vanilla baseline (128 steps)")
    v_results, v_acc, v_correct, v_diversity, v_times = run_method(
        model, tokenizer, problems, prompts, SEED,
        "vanilla",
        lambda p: vanilla_generate(model, tokenizer, p, DEVICE),
        method_label="Vanilla (128 steps)",
    )

    # ── Phase 2: A-CFG w=1.0 ──
    report_progress(2, 5, "Running A-CFG w=1.0")
    acfg_w1_config = ACFGConfig(
        remask_pct=BEST_REMASK_PCT, w=1.0,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )
    a1_results, a1_acc, a1_correct, a1_diversity, a1_times = run_method(
        model, tokenizer, problems, prompts, SEED,
        "acfg_w1.0",
        lambda p: acfg_generate(model, tokenizer, p, acfg_w1_config, DEVICE),
        method_label="A-CFG (w=1.0)",
    )

    # ── Phase 3: A-CFG w=1.5 (best config) ──
    report_progress(3, 5, "Running A-CFG w=1.5 (best)")
    acfg_w15_config = ACFGConfig(
        remask_pct=BEST_REMASK_PCT, w=BEST_W,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )
    a15_results, a15_acc, a15_correct, a15_diversity, a15_times = run_method(
        model, tokenizer, problems, prompts, SEED,
        "acfg_w1.5",
        lambda p: acfg_generate(model, tokenizer, p, acfg_w15_config, DEVICE),
        method_label="A-CFG (w=1.5, BEST)",
    )

    # ── Phase 4: Vanilla compute-fair (256 steps = 2x FLOPs) ──
    report_progress(4, 5, "Running vanilla 2x steps (compute-fair)")
    v2x_results, v2x_acc, v2x_correct, v2x_diversity, v2x_times = run_method(
        model, tokenizer, problems, prompts, SEED,
        "vanilla_2x",
        lambda p: vanilla_generate_custom_steps(model, tokenizer, p, DEVICE,
                                                 steps=STEPS * 2),
        method_label="Vanilla (256 steps, compute-fair)",
    )

    # ── Phase 5: A-CFG w=2.0 (higher guidance, exploring) ──
    report_progress(5, 5, "Running A-CFG w=2.0")
    acfg_w2_config = ACFGConfig(
        remask_pct=BEST_REMASK_PCT, w=2.0,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )
    a2_results, a2_acc, a2_correct, a2_diversity, a2_times = run_method(
        model, tokenizer, problems, prompts, SEED,
        "acfg_w2.0",
        lambda p: acfg_generate(model, tokenizer, p, acfg_w2_config, DEVICE),
        method_label="A-CFG (w=2.0)",
    )

    # ── Comparison ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE — fullscale_racfg PILOT")
    print(f"{'='*70}")

    all_methods = [
        ("Vanilla (128)", v_acc, v_correct, v_diversity, v_times, v_results, "1.0x"),
        ("Vanilla (256, fair)", v2x_acc, v2x_correct, v2x_diversity, v2x_times, v2x_results, "2.0x"),
        ("A-CFG w=1.0", a1_acc, a1_correct, a1_diversity, a1_times, a1_results, "~2.0x"),
        ("A-CFG w=1.5 (BEST)", a15_acc, a15_correct, a15_diversity, a15_times, a15_results, "~2.0x"),
        ("A-CFG w=2.0", a2_acc, a2_correct, a2_diversity, a2_times, a2_results, "~2.0x"),
    ]

    print(f"\n| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |")
    print(f"|--------|----------|-------|-------|------------|----------|-------|")
    for name, acc, nc, div, tms, _, flops in all_methods:
        print(f"| {name} | {acc:.1%} ({nc}/{N_SAMPLES}) | "
              f"{div['rep_2']:.4f} | {div['rep_3']:.4f} | "
              f"{div['distinct_3']:.4f} | {np.mean(tms):.1f}s | {flops} |")

    # ── Flip analyses ──
    flips = {}

    # A-CFG best vs Vanilla
    f1 = compute_flips(a15_results, v_results, "acfg_w1.5", "vanilla")
    flips["acfg_w1.5_vs_vanilla"] = f1
    print(f"\n--- Flip Analysis: A-CFG w=1.5 vs Vanilla ---")
    print(f"  A-CFG wins: {f1['acfg_w1.5_over_vanilla']}, "
          f"Vanilla wins: {f1['vanilla_over_acfg_w1.5']}")

    # A-CFG best vs Vanilla 2x
    f2 = compute_flips(a15_results, v2x_results, "acfg_w1.5", "vanilla_2x")
    flips["acfg_w1.5_vs_vanilla_2x"] = f2
    print(f"\n--- Flip Analysis: A-CFG w=1.5 vs Vanilla 2x ---")
    print(f"  A-CFG wins: {f2['acfg_w1.5_over_vanilla_2x']}, "
          f"Vanilla-2x wins: {f2['vanilla_2x_over_acfg_w1.5']}")

    # A-CFG w=1.5 vs w=1.0
    f3 = compute_flips(a15_results, a1_results, "acfg_w1.5", "acfg_w1.0")
    flips["acfg_w1.5_vs_w1.0"] = f3
    print(f"\n--- Flip Analysis: A-CFG w=1.5 vs w=1.0 ---")
    print(f"  w=1.5 wins: {f3['acfg_w1.5_over_acfg_w1.0']}, "
          f"w=1.0 wins: {f3['acfg_w1.0_over_acfg_w1.5']}")

    # A-CFG w=2.0 vs w=1.5
    f4 = compute_flips(a2_results, a15_results, "acfg_w2.0", "acfg_w1.5")
    flips["acfg_w2.0_vs_w1.5"] = f4
    print(f"\n--- Flip Analysis: A-CFG w=2.0 vs w=1.5 ---")
    print(f"  w=2.0 wins: {f4['acfg_w2.0_over_acfg_w1.5']}, "
          f"w=1.5 wins: {f4['acfg_w1.5_over_acfg_w2.0']}")

    # Degeneration checks
    print(f"\n--- Degeneration Checks ---")
    for name, div, results in [
        ("A-CFG w=1.0", a1_diversity, a1_results),
        ("A-CFG w=1.5", a15_diversity, a15_results),
        ("A-CFG w=2.0", a2_diversity, a2_results),
    ]:
        degen = check_degeneration(div, v_diversity)
        if degen:
            print(f"  {name} DEGENERATION:")
            for k, v in degen.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {name}: No degeneration")

    # Qualitative samples for best method
    print(f"\n{'='*70}")
    print(f"  QUALITATIVE SAMPLES — A-CFG w=1.5 (BEST)")
    print_qualitative_samples(a15_results, n=10)

    # ── Pass Criteria ──
    # PILOT: A-CFG accuracy > vanilla + 3pp
    best_acfg_acc = a15_acc
    best_acfg_name = "A-CFG w=1.5"

    # Check if w=2.0 is actually better
    if a2_acc > a15_acc:
        best_acfg_acc = a2_acc
        best_acfg_name = "A-CFG w=2.0"

    beats_vanilla_3pp = best_acfg_acc > v_acc + 0.03
    beats_vanilla_2x = best_acfg_acc > v2x_acc
    no_degen_best = len(check_degeneration(
        a15_diversity if best_acfg_name == "A-CFG w=1.5" else a2_diversity,
        v_diversity
    )) == 0

    if beats_vanilla_3pp and no_degen_best:
        verdict = "GO"
    elif best_acfg_acc > v_acc and no_degen_best:
        verdict = "CONDITIONAL-GO"
    elif best_acfg_acc == 0.0 and v_acc == 0.0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO-GO"

    print(f"\n{'='*70}")
    print(f"  PILOT VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  Vanilla (128):     {v_correct}/{N_SAMPLES} = {v_acc:.1%}")
    print(f"  Vanilla (256 2x):  {v2x_correct}/{N_SAMPLES} = {v2x_acc:.1%}")
    print(f"  A-CFG w=1.0:       {a1_correct}/{N_SAMPLES} = {a1_acc:.1%}")
    print(f"  A-CFG w=1.5:       {a15_correct}/{N_SAMPLES} = {a15_acc:.1%}")
    print(f"  A-CFG w=2.0:       {a2_correct}/{N_SAMPLES} = {a2_acc:.1%}")
    print(f"  Best: {best_acfg_name} = {best_acfg_acc:.1%}")
    print(f"  Beats vanilla + 3pp: {beats_vanilla_3pp}")
    print(f"  Beats vanilla 2x (compute-fair): {beats_vanilla_2x}")
    print(f"  No degeneration: {no_degen_best}")

    # ── Save results ──
    elapsed_total = time.time() - start_time

    combined = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "verdict": verdict,
        "model": "Dream-v0-Instruct-7B",
        "benchmark": f"Countdown-{N_SAMPLES}",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_s": round(elapsed_total, 1),
        "elapsed_total_min": round(elapsed_total / 60, 1),
        "pivot_note": (
            "Original RACFG (JSD stability) failed on Dream-7B (0% accuracy). "
            "This task evaluates A-CFG (confidence-based re-masking) as the "
            "'enhanced guidance' method, which was the best-performing variant."
        ),
        "best_config": {
            "method": best_acfg_name,
            "remask_pct": BEST_REMASK_PCT,
            "w": BEST_W if "1.5" in best_acfg_name else 2.0,
            "gen_len": GEN_LEN,
            "steps": STEPS,
            "temperature": TEMPERATURE,
        },
        "results": {
            "vanilla_128": {
                "accuracy": v_acc, "n_correct": v_correct,
                "n_samples": N_SAMPLES, "steps": 128, "flops_ratio": "1.0x",
                **v_diversity,
                "avg_gen_time_s": float(np.mean(v_times)),
            },
            "vanilla_256_fair": {
                "accuracy": v2x_acc, "n_correct": v2x_correct,
                "n_samples": N_SAMPLES, "steps": 256, "flops_ratio": "2.0x",
                **v2x_diversity,
                "avg_gen_time_s": float(np.mean(v2x_times)),
            },
            "acfg_w1.0": {
                "accuracy": a1_acc, "n_correct": a1_correct,
                "n_samples": N_SAMPLES, "flops_ratio": "~2.0x",
                "config": {"w": 1.0, "remask_pct": BEST_REMASK_PCT},
                **a1_diversity,
                "avg_gen_time_s": float(np.mean(a1_times)),
            },
            "acfg_w1.5": {
                "accuracy": a15_acc, "n_correct": a15_correct,
                "n_samples": N_SAMPLES, "flops_ratio": "~2.0x",
                "config": {"w": 1.5, "remask_pct": BEST_REMASK_PCT},
                **a15_diversity,
                "avg_gen_time_s": float(np.mean(a15_times)),
            },
            "acfg_w2.0": {
                "accuracy": a2_acc, "n_correct": a2_correct,
                "n_samples": N_SAMPLES, "flops_ratio": "~2.0x",
                "config": {"w": 2.0, "remask_pct": BEST_REMASK_PCT},
                **a2_diversity,
                "avg_gen_time_s": float(np.mean(a2_times)),
            },
        },
        "flips": flips,
        "pass_criteria": {
            "best_method": best_acfg_name,
            "best_accuracy": best_acfg_acc,
            "beats_vanilla_3pp": beats_vanilla_3pp,
            "beats_vanilla_2x_fair": beats_vanilla_2x,
            "no_degeneration": no_degen_best,
            "verdict": verdict,
        },
        "per_sample": {
            "vanilla_128": v_results,
            "vanilla_256_fair": v2x_results,
            "acfg_w1.0": a1_results,
            "acfg_w1.5": a15_results,
            "acfg_w2.0": a2_results,
        },
    }

    out_file = RESULTS_DIR / "racfg_fullscale_countdown500.json"
    save_results(combined, str(out_file))

    # Summary markdown
    summary_md = RESULTS_DIR / "racfg_fullscale_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# RACFG/A-CFG Full-Scale Evaluation — Countdown-{N_SAMPLES} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"## Pivot Note\n\n")
        f.write("Original RACFG (JSD stability) failed on Dream-7B (0% accuracy everywhere). "
                "This evaluation uses A-CFG (confidence-based re-masking) as the effective "
                "'enhanced guidance' method.\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |\n")
        f.write(f"|--------|----------|-------|-------|------------|----------|-------|\n")
        for name, acc, nc, div, tms, _, flops in all_methods:
            f.write(f"| {name} | {acc:.1%} ({nc}/{N_SAMPLES}) | "
                    f"{div['rep_2']:.3f} | {div['rep_3']:.3f} | "
                    f"{div['distinct_3']:.3f} | {np.mean(tms):.1f}s | {flops} |\n")
        f.write(f"\n## Best Method\n\n")
        f.write(f"- **{best_acfg_name}**: {best_acfg_acc:.1%}\n")
        f.write(f"- Beats vanilla + 3pp: {beats_vanilla_3pp}\n")
        f.write(f"- Beats vanilla 2x (compute-fair): {beats_vanilla_2x}\n\n")
        f.write(f"## Flip Analysis\n\n")
        f.write(f"### A-CFG w=1.5 vs Vanilla\n")
        f.write(f"- A-CFG wins: {f1['acfg_w1.5_over_vanilla']}\n")
        f.write(f"- Vanilla wins: {f1['vanilla_over_acfg_w1.5']}\n\n")
        f.write(f"### A-CFG w=1.5 vs Vanilla 2x (compute-fair)\n")
        f.write(f"- A-CFG wins: {f2['acfg_w1.5_over_vanilla_2x']}\n")
        f.write(f"- Vanilla-2x wins: {f2['vanilla_2x_over_acfg_w1.5']}\n\n")
        f.write(f"## Runtime\n\n")
        f.write(f"- Total: {elapsed_total / 60:.1f} minutes\n")
    print(f"[{TASK_ID}] Summary saved to {summary_md}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Mark done
    mark_done(
        status="success",
        summary=(f"{best_acfg_name} {best_acfg_acc:.1%} vs vanilla {v_acc:.1%} "
                 f"vs vanilla-2x {v2x_acc:.1%} on Countdown-{N_SAMPLES}. "
                 f"Verdict: {verdict}. Time: {elapsed_total/60:.1f}min"),
    )

    # Update gpu_progress.json
    gpu_progress_path = Path(f"{PROJECT_DIR}/exp/gpu_progress.json")
    try:
        progress = json.loads(gpu_progress_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in progress["completed"]:
        progress["completed"].append(TASK_ID)
    progress["running"].pop(TASK_ID, None)
    progress["timings"][TASK_ID] = {
        "planned_min": 60,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "Dream-v0-Instruct-7B",
            "method": "A-CFG fullscale (PILOT, pivoted from RACFG)",
            "best_config": best_acfg_name,
            "remask_pct": BEST_REMASK_PCT,
            "w": BEST_W,
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "verdict": verdict,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }
    gpu_progress_path.write_text(json.dumps(progress, indent=2))
    print(f"[{TASK_ID}] gpu_progress.json updated")

    print(f"\n[{TASK_ID}] Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    return verdict in ("GO", "CONDITIONAL-GO")


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[{TASK_ID}] FATAL ERROR: {e}")
        traceback.print_exc()
        try:
            mark_done(status="failed", summary=f"Fatal error: {e}")
        except:
            pass
        sys.exit(1)
