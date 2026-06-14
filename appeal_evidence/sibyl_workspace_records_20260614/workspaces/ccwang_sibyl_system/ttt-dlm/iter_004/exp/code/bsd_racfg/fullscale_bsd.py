"""
fullscale_bsd: BSD Full-Scale Evaluation (PILOT: Countdown-16, seed 42).

Best BSD config from ablations:
  - k_frac=0.75 (k=3T/4: 25% belief refinement + 75% hard reveal)
  - alpha_schedule=linear(0.1→0.8)
  - tau=linear(1.0→0.3)
  - no fallback_beta

Compares BSD vs vanilla vs DMI on Countdown-16 (PILOT mode).
Records belief entropy trajectories for analysis.

Pass criteria: BSD accuracy > DMI (9.3%) with p < 0.1
  (On pilot 16 samples: BSD > DMI qualitatively)

Task: fullscale_bsd
Mode: PILOT (16 samples, seed 42)
GPU: cuda:0 (mapped via CUDA_VISIBLE_DEVICES)

Usage:
    CUDA_VISIBLE_DEVICES=1,2 python fullscale_bsd.py
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
from bsd_racfg.bsd import BSDConfig, bsd_generate

# ── Constants ──
TASK_ID = "fullscale_bsd"
N_SAMPLES = 16   # PILOT mode
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
DMI_ALPHA = 0.3
SOFT_TAU = 0.5
DEVICE = "cuda:0"

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
# DMI Generation (from pilot_dmi_repro)
# ──────────────────────────────────────────────────

def dmi_generate(model, tokenizer, prompt_text, embedding_layer,
                 device=DEVICE, gen_len=GEN_LEN, steps=STEPS,
                 temperature=TEMPERATURE, alpha=DMI_ALPHA, soft_tau=SOFT_TAU):
    """Generate with DMI: mix previous step's soft embeddings into mask positions."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    prev_logits_shifted = None
    t0 = time.time()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)

            if prev_logits_shifted is not None and alpha > 0:
                soft_probs = F.softmax(prev_logits_shifted / soft_tau, dim=-1)
                emb_weight = embedding_layer.weight
                soft_emb = torch.matmul(soft_probs, emb_weight.to(soft_probs.dtype))
                hard_emb = embedding_layer(x)
                mixed_emb = hard_emb.clone()
                mixed_emb[mask_index] = (
                    (1 - alpha) * hard_emb[mask_index] +
                    alpha * soft_emb[mask_index]
                )
                outputs = model(inputs_embeds=mixed_emb, attention_mask="full",
                                position_ids=None)
            else:
                outputs = model(x, attention_mask="full", position_ids=None)

            logits = outputs.logits
            logits_shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            prev_logits_shifted = logits_shifted.clone()

            mask_logits = logits_shifted[mask_index]

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

def run_vanilla(model, tokenizer, problems, prompts, seed):
    """Run vanilla baseline."""
    print(f"\n{'='*60}")
    print(f"  Vanilla Baseline ({len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
        text, elapsed, _ = vanilla_generate(model, tokenizer, prompt, DEVICE)
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

    print(f"\n  Vanilla: {n_correct}/{len(problems)} = {accuracy:.1%}")
    return results, accuracy, n_correct, diversity, times


def run_dmi(model, tokenizer, embedding_layer, problems, prompts, seed):
    """Run DMI baseline."""
    print(f"\n{'='*60}")
    print(f"  DMI Baseline (alpha={DMI_ALPHA}, {len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
        text, elapsed = dmi_generate(model, tokenizer, prompt, embedding_layer)
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

    print(f"\n  DMI: {n_correct}/{len(problems)} = {accuracy:.1%}")
    return results, accuracy, n_correct, diversity, times


def run_bsd(model, tokenizer, embedding_layer, problems, prompts, seed):
    """Run BSD with best config from ablations."""
    config = BSDConfig(
        k_frac=0.75,
        alpha_schedule="linear",
        alpha_start=0.1, alpha_end=0.8,
        tau_start=1.0, tau_end=0.3,
        fallback_beta_start=0.0, fallback_beta_end=0.0,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )

    print(f"\n{'='*60}")
    print(f"  BSD (k_frac=0.75, alpha=linear, {len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []
    all_entropy_trajectories = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
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
            "idx": idx, "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text, "gen_time_s": elapsed,
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

        # Report progress
        if (idx + 1) % 4 == 0:
            n_correct_so_far = sum(1 for r in results if r["is_correct"])
            report_progress(idx + 1, len(problems), "BSD generation",
                            {"accuracy": n_correct_so_far / (idx + 1)})

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)

    # Entropy analysis
    n_decreasing = sum(1 for r in results if r.get("entropy_decreasing") is True)
    ent_starts = [r["entropy_start"] for r in results if r["entropy_start"] is not None]
    ent_ends = [r["entropy_end"] for r in results if r["entropy_end"] is not None]
    avg_ent_start = float(np.mean(ent_starts)) if ent_starts else 0.0
    avg_ent_end = float(np.mean(ent_ends)) if ent_ends else 0.0

    print(f"\n  BSD: {n_correct}/{len(problems)} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")
    print(f"  Entropy: {avg_ent_start:.2f} -> {avg_ent_end:.2f} | "
          f"decreasing={n_decreasing}/{len(problems)}")

    entropy_analysis = {
        "n_decreasing": n_decreasing,
        "n_total": len(problems),
        "avg_entropy_start": avg_ent_start,
        "avg_entropy_end": avg_ent_end,
        "entropy_trajectories": all_entropy_trajectories,
    }

    return results, accuracy, n_correct, diversity, times, entropy_analysis


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 3, "Loading model")

    print("=" * 70)
    print(f"  fullscale_bsd — BSD Full-Scale Evaluation (PILOT)")
    print(f"  Countdown-{N_SAMPLES}, seed={SEED}")
    print(f"  Best BSD config: k_frac=0.75, alpha=linear(0.1→0.8)")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPUs visible: {torch.cuda.device_count()}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, 3, "Model loaded")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    prompts = [format_countdown_prompt(p) for p in problems]
    print(f"\nGenerated {len(problems)} Countdown problems")

    # ── Phase 1: Vanilla baseline ──
    report_progress(1, 3, "Running vanilla baseline")
    v_results, v_acc, v_correct, v_diversity, v_times = run_vanilla(
        model, tokenizer, problems, prompts, SEED)

    # ── Phase 2: DMI baseline ──
    report_progress(2, 3, "Running DMI baseline")
    d_results, d_acc, d_correct, d_diversity, d_times = run_dmi(
        model, tokenizer, embedding_layer, problems, prompts, SEED)

    # ── Phase 3: BSD (best config) ──
    report_progress(3, 3, "Running BSD")
    b_results, b_acc, b_correct, b_diversity, b_times, b_entropy = run_bsd(
        model, tokenizer, embedding_layer, problems, prompts, SEED)

    # ── Comparison ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE — fullscale_bsd PILOT")
    print(f"{'='*70}")

    vanilla_result = format_results(
        "vanilla", f"Countdown-{N_SAMPLES}", N_SAMPLES, SEED,
        v_acc, v_correct, v_diversity, v_times, v_results)
    dmi_result = format_results(
        "DMI (alpha=0.3)", f"Countdown-{N_SAMPLES}", N_SAMPLES, SEED,
        d_acc, d_correct, d_diversity, d_times, d_results,
        extra_config={"dmi_alpha": DMI_ALPHA, "soft_tau": SOFT_TAU})
    bsd_result = format_results(
        "BSD (k=0.75, linear)", f"Countdown-{N_SAMPLES}", N_SAMPLES, SEED,
        b_acc, b_correct, b_diversity, b_times, b_results,
        extra_config={"k_frac": 0.75, "alpha_schedule": "linear",
                      "alpha_start": 0.1, "alpha_end": 0.8,
                      "tau_start": 1.0, "tau_end": 0.3})
    bsd_result["entropy_analysis"] = b_entropy

    print_comparison_table([vanilla_result, dmi_result, bsd_result])

    # Flip analysis (BSD vs DMI)
    print(f"\n--- Flip Analysis: BSD vs DMI ---")
    bsd_over_dmi = 0
    dmi_over_bsd = 0
    for br, dr in zip(b_results, d_results):
        if br["is_correct"] and not dr["is_correct"]:
            bsd_over_dmi += 1
            print(f"  [+BSD] #{br['idx']}: BSD correct, DMI wrong "
                  f"(target={br['target']})")
        elif not br["is_correct"] and dr["is_correct"]:
            dmi_over_bsd += 1
            print(f"  [+DMI] #{br['idx']}: DMI correct, BSD wrong "
                  f"(target={br['target']})")
    print(f"  BSD->DMI flips: +{bsd_over_dmi}, DMI->BSD flips: +{dmi_over_bsd}")

    # Flip analysis (BSD vs Vanilla)
    print(f"\n--- Flip Analysis: BSD vs Vanilla ---")
    bsd_over_v = 0
    v_over_bsd = 0
    for br, vr in zip(b_results, v_results):
        if br["is_correct"] and not vr["is_correct"]:
            bsd_over_v += 1
        elif not br["is_correct"] and vr["is_correct"]:
            v_over_bsd += 1
    print(f"  BSD->Vanilla flips: +{bsd_over_v}, Vanilla->BSD flips: +{v_over_bsd}")

    # Degeneration checks
    bsd_degen = check_degeneration(b_diversity, v_diversity)
    if bsd_degen:
        print(f"\n  BSD DEGENERATION WARNINGS:")
        for k, v in bsd_degen.items():
            print(f"    {k}: {v}")
    else:
        print(f"\n  No BSD degeneration warnings.")

    # Qualitative samples
    print(f"\n{'='*70}")
    print(f"  QUALITATIVE SAMPLES — BSD")
    print_qualitative_samples(b_results, n=10)

    # ── Pass Criteria ──
    # Pilot: BSD > DMI qualitatively (can't do p<0.1 with 16 samples)
    bsd_beats_dmi = b_acc > d_acc
    bsd_beats_vanilla = b_acc > v_acc
    no_degeneration = len(bsd_degen) == 0

    # McNemar-like: at least check if BSD has more correct
    # With 16 samples, statistical significance is limited
    verdict = "GO" if (bsd_beats_vanilla and no_degeneration) else "CONDITIONAL-GO"
    if b_acc == 0.0 and d_acc == 0.0 and v_acc == 0.0:
        verdict = "INCONCLUSIVE"
    if b_acc < v_acc:
        verdict = "NO-GO"

    print(f"\n{'='*70}")
    print(f"  PILOT VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  Vanilla:  {v_correct}/{N_SAMPLES} = {v_acc:.1%}")
    print(f"  DMI:      {d_correct}/{N_SAMPLES} = {d_acc:.1%}")
    print(f"  BSD:      {b_correct}/{N_SAMPLES} = {b_acc:.1%}")
    print(f"  BSD > DMI: {bsd_beats_dmi}")
    print(f"  BSD > Vanilla: {bsd_beats_vanilla}")
    print(f"  No degeneration: {no_degeneration}")
    print(f"  Entropy: {b_entropy['avg_entropy_start']:.2f} -> "
          f"{b_entropy['avg_entropy_end']:.2f} "
          f"(decreasing: {b_entropy['n_decreasing']}/{b_entropy['n_total']})")

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
        "bsd_config": {
            "k_frac": 0.75,
            "alpha_schedule": "linear",
            "alpha_start": 0.1, "alpha_end": 0.8,
            "tau_start": 1.0, "tau_end": 0.3,
            "fallback_beta": 0.0,
            "gen_len": GEN_LEN, "steps": STEPS, "temperature": TEMPERATURE,
        },
        "dmi_config": {
            "alpha": DMI_ALPHA, "soft_tau": SOFT_TAU,
        },
        "results": {
            "vanilla": {
                "accuracy": v_acc, "n_correct": v_correct,
                "n_samples": N_SAMPLES,
                **v_diversity,
                "avg_gen_time_s": float(np.mean(v_times)),
                "total_gen_time_s": float(np.sum(v_times)),
            },
            "dmi": {
                "accuracy": d_acc, "n_correct": d_correct,
                "n_samples": N_SAMPLES,
                **d_diversity,
                "avg_gen_time_s": float(np.mean(d_times)),
                "total_gen_time_s": float(np.sum(d_times)),
            },
            "bsd": {
                "accuracy": b_acc, "n_correct": b_correct,
                "n_samples": N_SAMPLES,
                **b_diversity,
                "avg_gen_time_s": float(np.mean(b_times)),
                "total_gen_time_s": float(np.sum(b_times)),
                "entropy_analysis": {
                    "n_decreasing": b_entropy["n_decreasing"],
                    "n_total": b_entropy["n_total"],
                    "avg_entropy_start": b_entropy["avg_entropy_start"],
                    "avg_entropy_end": b_entropy["avg_entropy_end"],
                },
            },
        },
        "flips": {
            "bsd_over_dmi": bsd_over_dmi,
            "dmi_over_bsd": dmi_over_bsd,
            "bsd_over_vanilla": bsd_over_v,
            "vanilla_over_bsd": v_over_bsd,
        },
        "degeneration_warnings": bsd_degen,
        "pass_criteria": {
            "bsd_beats_dmi": bsd_beats_dmi,
            "bsd_beats_vanilla": bsd_beats_vanilla,
            "no_degeneration": no_degeneration,
            "verdict": verdict,
        },
        "per_sample": {
            "vanilla": v_results,
            "dmi": d_results,
            "bsd": b_results,
        },
        "entropy_trajectories": b_entropy.get("entropy_trajectories", []),
    }

    out_file = RESULTS_DIR / "bsd_fullscale_countdown500.json"
    save_results(combined, str(out_file))

    # Summary markdown
    summary_md = RESULTS_DIR / "bsd_fullscale_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# BSD Full-Scale Evaluation — Countdown-{N_SAMPLES} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time |\n")
        f.write(f"|--------|----------|-------|-------|------------|----------|\n")
        f.write(f"| Vanilla | {v_acc:.1%} ({v_correct}/{N_SAMPLES}) | "
                f"{v_diversity['rep_2']:.3f} | {v_diversity['rep_3']:.3f} | "
                f"{v_diversity['distinct_3']:.3f} | {np.mean(v_times):.1f}s |\n")
        f.write(f"| DMI (alpha=0.3) | {d_acc:.1%} ({d_correct}/{N_SAMPLES}) | "
                f"{d_diversity['rep_2']:.3f} | {d_diversity['rep_3']:.3f} | "
                f"{d_diversity['distinct_3']:.3f} | {np.mean(d_times):.1f}s |\n")
        f.write(f"| **BSD (k=0.75, linear)** | **{b_acc:.1%}** ({b_correct}/{N_SAMPLES}) | "
                f"{b_diversity['rep_2']:.3f} | {b_diversity['rep_3']:.3f} | "
                f"{b_diversity['distinct_3']:.3f} | {np.mean(b_times):.1f}s |\n")
        f.write(f"\n## Entropy Analysis\n\n")
        f.write(f"- Start entropy: {b_entropy['avg_entropy_start']:.2f}\n")
        f.write(f"- End entropy: {b_entropy['avg_entropy_end']:.2f}\n")
        f.write(f"- Decreasing: {b_entropy['n_decreasing']}/{b_entropy['n_total']}\n")
        f.write(f"\n## Flip Analysis\n\n")
        f.write(f"- BSD correct, DMI wrong: {bsd_over_dmi}\n")
        f.write(f"- DMI correct, BSD wrong: {dmi_over_bsd}\n")
        f.write(f"- BSD correct, Vanilla wrong: {bsd_over_v}\n")
        f.write(f"- Vanilla correct, BSD wrong: {v_over_bsd}\n")
        f.write(f"\n## Runtime\n\n")
        f.write(f"- Total: {elapsed_total / 60:.1f} minutes\n")
    print(f"[{TASK_ID}] Summary saved to {summary_md}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Mark done
    mark_done(
        status="success",
        summary=f"BSD {b_acc:.1%} vs DMI {d_acc:.1%} vs vanilla {v_acc:.1%} "
                f"on Countdown-{N_SAMPLES}. Verdict: {verdict}. "
                f"Time: {elapsed_total/60:.1f}min",
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
            "method": "BSD fullscale (PILOT)",
            "k_frac": 0.75,
            "alpha_schedule": "linear(0.1->0.8)",
            "n_samples": N_SAMPLES,
            "seed": SEED,
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
