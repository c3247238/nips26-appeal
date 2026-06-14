"""
gsm8k_extension: GSM8K Generalization Test (PILOT: 16 samples, seed 42).

Evaluates best methods (BSD and A-CFG) on GSM8K to test generalization
beyond Countdown. The combination (BSD+ACFG) was NO-GO on Countdown,
so we test the two individual methods that showed promise:
  - BSD (k_frac=0.75, alpha=linear(0.1->0.8)): 12.5% on Countdown-16
  - A-CFG (w=1.5, remask_pct=0.1): 12.5% on Countdown-16

Hypothesis H8: significant improvement over vanilla Dream-7B on GSM8K.

Task: gsm8k_extension
Mode: PILOT (16 samples, seed 42)
GPU: cuda:0 (mapped via CUDA_VISIBLE_DEVICES)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python gsm8k_extension.py
"""
import os
import sys
import json
import time
import re
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
    load_dream, compute_diversity_metrics, compute_per_sample_metrics,
    vanilla_generate, save_results, print_qualitative_samples,
    print_comparison_table, check_degeneration, format_results,
    MASK_TOKEN_ID, PROJECT_DIR, RESULTS_FULL,
)
from bsd_racfg.bsd import BSDConfig, bsd_generate
from bsd_racfg.racfg import ACFGConfig, acfg_generate

# ── Constants ──
TASK_ID = "gsm8k_extension"
N_SAMPLES = 16   # PILOT mode
SEED = 42
GEN_LEN = 512    # GSM8K needs longer generation
STEPS = 128
TEMPERATURE = 0.4
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
# GSM8K Data Loading & Evaluation
# ──────────────────────────────────────────────────

def load_gsm8k(n_samples=None, seed=42):
    """Load GSM8K test set. Returns list of problem dicts."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        answer_text = item["answer"]
        match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', answer_text)
        target = None
        if match:
            try:
                target = float(match.group(1).replace(',', ''))
            except:
                pass
        problems.append({
            "id": i,
            "question": item["question"],
            "target": target,
        })
    if n_samples is not None:
        import random
        rng = random.Random(seed)
        problems = rng.sample(problems, min(n_samples, len(problems)))
    return problems


def format_gsm8k_prompt(problem):
    """Format a GSM8K problem as a chat prompt."""
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


def extract_model_answer(text):
    """Extract numerical answer from model output."""
    # Try #### pattern first (standard GSM8K format)
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    # Fallback patterns
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:=]?\s*([\-]?\d[\d,]*\.?\d*)',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+answer\s+is\s+)?([\-]?\d[\d,]*\.?\d*)',
        r'\\boxed\{([\-]?\d[\d,]*\.?\d*)\}',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except:
                continue
    # Try "= number" at end of line
    match = re.search(r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    # Last resort: last number in text
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    return None


def verify_gsm8k(text, problem):
    """Verify GSM8K answer against target."""
    extracted = extract_model_answer(text)
    target = problem["target"]
    if extracted is None or target is None:
        return {
            "is_correct": False,
            "extracted_answer": extracted,
            "target": target,
            "extraction_method": "none",
        }
    is_correct = abs(extracted - target) < 1e-3
    return {
        "is_correct": is_correct,
        "extracted_answer": extracted,
        "target": target,
        "extraction_method": "auto",
    }


# ──────────────────────────────────────────────────
# DMI Generation (baseline from prior iterations)
# ──────────────────────────────────────────────────

DMI_ALPHA = 0.3
SOFT_TAU = 0.5


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
# Run methods on GSM8K
# ──────────────────────────────────────────────────

def run_gsm8k_vanilla(model, tokenizer, problems, prompts, seed):
    """Run vanilla baseline on GSM8K."""
    print(f"\n{'='*60}")
    print(f"  Vanilla Baseline — GSM8K ({len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
        text, elapsed, _ = vanilla_generate(
            model, tokenizer, prompt, DEVICE,
            gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
        )
        verification = verify_gsm8k(text, problem)
        metrics = compute_per_sample_metrics(text)

        results.append({
            "idx": idx, "id": problem["id"],
            "question": problem["question"][:100],
            "target": problem["target"],
            **verification,
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        })
        texts.append(text)
        times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        ans_str = f"ext={verification['extracted_answer']}" if verification['extracted_answer'] is not None else "N/A"
        print(f"  [{idx:2d}] {status} | target={problem['target']} | "
              f"{ans_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)
    print(f"\n  Vanilla: {n_correct}/{len(problems)} = {accuracy:.1%}")
    return results, accuracy, n_correct, diversity, times


def run_gsm8k_dmi(model, tokenizer, embedding_layer, problems, prompts, seed):
    """Run DMI on GSM8K."""
    print(f"\n{'='*60}")
    print(f"  DMI (alpha={DMI_ALPHA}) — GSM8K ({len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
        text, elapsed = dmi_generate(model, tokenizer, prompt, embedding_layer)
        verification = verify_gsm8k(text, problem)
        metrics = compute_per_sample_metrics(text)

        results.append({
            "idx": idx, "id": problem["id"],
            "question": problem["question"][:100],
            "target": problem["target"],
            **verification,
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        })
        texts.append(text)
        times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        ans_str = f"ext={verification['extracted_answer']}" if verification['extracted_answer'] is not None else "N/A"
        print(f"  [{idx:2d}] {status} | target={problem['target']} | "
              f"{ans_str} | {elapsed:.1f}s")

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)
    print(f"\n  DMI: {n_correct}/{len(problems)} = {accuracy:.1%}")
    return results, accuracy, n_correct, diversity, times


def run_gsm8k_bsd(model, tokenizer, embedding_layer, problems, prompts, seed):
    """Run BSD (best config) on GSM8K."""
    config = BSDConfig(
        k_frac=0.75,
        alpha_schedule="linear",
        alpha_start=0.1, alpha_end=0.8,
        tau_start=1.0, tau_end=0.3,
        fallback_beta_start=0.0, fallback_beta_end=0.0,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )

    print(f"\n{'='*60}")
    print(f"  BSD (k=0.75, linear) — GSM8K ({len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
        text, elapsed, diag = bsd_generate(
            model, tokenizer, prompt, embedding_layer, config, DEVICE,
            track_entropy=False,  # Skip entropy tracking for GSM8K speed
        )
        verification = verify_gsm8k(text, problem)
        metrics = compute_per_sample_metrics(text)

        results.append({
            "idx": idx, "id": problem["id"],
            "question": problem["question"][:100],
            "target": problem["target"],
            **verification,
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        })
        texts.append(text)
        times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        ans_str = f"ext={verification['extracted_answer']}" if verification['extracted_answer'] is not None else "N/A"
        print(f"  [{idx:2d}] {status} | target={problem['target']} | "
              f"{ans_str} | {elapsed:.1f}s")

        if (idx + 1) % 4 == 0:
            n_so_far = sum(1 for r in results if r["is_correct"])
            report_progress(idx + 1, len(problems), "BSD GSM8K",
                            {"accuracy": n_so_far / (idx + 1)})

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)
    print(f"\n  BSD: {n_correct}/{len(problems)} = {accuracy:.1%}")
    return results, accuracy, n_correct, diversity, times


def run_gsm8k_acfg(model, tokenizer, problems, prompts, seed):
    """Run A-CFG (best config w=1.5) on GSM8K."""
    config = ACFGConfig(
        remask_pct=0.1,
        w=1.5,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )

    print(f"\n{'='*60}")
    print(f"  A-CFG (w=1.5) — GSM8K ({len(problems)} samples, seed={seed})")
    print(f"{'='*60}")

    results = []
    texts = []
    times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)
        text, elapsed, diag = acfg_generate(
            model, tokenizer, prompt, config, DEVICE,
        )
        verification = verify_gsm8k(text, problem)
        metrics = compute_per_sample_metrics(text)

        results.append({
            "idx": idx, "id": problem["id"],
            "question": problem["question"][:100],
            "target": problem["target"],
            **verification,
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        })
        texts.append(text)
        times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        ans_str = f"ext={verification['extracted_answer']}" if verification['extracted_answer'] is not None else "N/A"
        print(f"  [{idx:2d}] {status} | target={problem['target']} | "
              f"{ans_str} | {elapsed:.1f}s")

        if (idx + 1) % 4 == 0:
            n_so_far = sum(1 for r in results if r["is_correct"])
            report_progress(idx + 1, len(problems), "A-CFG GSM8K",
                            {"accuracy": n_so_far / (idx + 1)})

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(texts)
    print(f"\n  A-CFG: {n_correct}/{len(problems)} = {accuracy:.1%}")
    return results, accuracy, n_correct, diversity, times


# ──────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────

def bootstrap_ci(correct_flags, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(correct_flags)
    arr = np.array(correct_flags, dtype=float)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(lo), float(hi)


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 5, "Loading model")

    print("=" * 70)
    print(f"  gsm8k_extension — GSM8K Generalization Test (PILOT)")
    print(f"  GSM8K-{N_SAMPLES}, seed={SEED}")
    print(f"  Methods: Vanilla, DMI, BSD (k=0.75), A-CFG (w=1.5)")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPUs visible: {torch.cuda.device_count()}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, 5, "Model loaded")

    # Load GSM8K problems
    print(f"\nLoading GSM8K test set...")
    problems = load_gsm8k(n_samples=N_SAMPLES, seed=SEED)
    prompts = [format_gsm8k_prompt(p) for p in problems]
    print(f"Loaded {len(problems)} GSM8K problems")

    # ── Phase 1: Vanilla baseline ──
    report_progress(1, 5, "Running vanilla baseline")
    v_results, v_acc, v_correct, v_diversity, v_times = run_gsm8k_vanilla(
        model, tokenizer, problems, prompts, SEED)

    # ── Phase 2: DMI baseline ──
    report_progress(2, 5, "Running DMI")
    d_results, d_acc, d_correct, d_diversity, d_times = run_gsm8k_dmi(
        model, tokenizer, embedding_layer, problems, prompts, SEED)

    # ── Phase 3: BSD ──
    report_progress(3, 5, "Running BSD")
    b_results, b_acc, b_correct, b_diversity, b_times = run_gsm8k_bsd(
        model, tokenizer, embedding_layer, problems, prompts, SEED)

    # ── Phase 4: A-CFG ──
    report_progress(4, 5, "Running A-CFG")
    a_results, a_acc, a_correct, a_diversity, a_times = run_gsm8k_acfg(
        model, tokenizer, problems, prompts, SEED)

    # ── Phase 5: Analysis ──
    report_progress(5, 5, "Analysis")

    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE — GSM8K PILOT ({N_SAMPLES} samples)")
    print(f"{'='*70}")

    # Format results for comparison table
    vanilla_result = format_results(
        "Vanilla", f"GSM8K-{N_SAMPLES}", N_SAMPLES, SEED,
        v_acc, v_correct, v_diversity, v_times, v_results)
    dmi_result = format_results(
        "DMI (alpha=0.3)", f"GSM8K-{N_SAMPLES}", N_SAMPLES, SEED,
        d_acc, d_correct, d_diversity, d_times, d_results,
        extra_config={"dmi_alpha": DMI_ALPHA})
    bsd_result = format_results(
        "BSD (k=0.75)", f"GSM8K-{N_SAMPLES}", N_SAMPLES, SEED,
        b_acc, b_correct, b_diversity, b_times, b_results,
        extra_config={"k_frac": 0.75, "alpha_schedule": "linear"})
    acfg_result = format_results(
        "A-CFG (w=1.5)", f"GSM8K-{N_SAMPLES}", N_SAMPLES, SEED,
        a_acc, a_correct, a_diversity, a_times, a_results,
        extra_config={"remask_pct": 0.1, "w": 1.5})

    print_comparison_table([vanilla_result, dmi_result, bsd_result, acfg_result])

    # Bootstrap CIs
    v_ci = bootstrap_ci([r["is_correct"] for r in v_results])
    d_ci = bootstrap_ci([r["is_correct"] for r in d_results])
    b_ci = bootstrap_ci([r["is_correct"] for r in b_results])
    a_ci = bootstrap_ci([r["is_correct"] for r in a_results])

    print(f"\n--- Bootstrap 95% CIs ---")
    print(f"  Vanilla:  {v_acc:.1%} [{v_ci[0]:.1%}, {v_ci[1]:.1%}]")
    print(f"  DMI:      {d_acc:.1%} [{d_ci[0]:.1%}, {d_ci[1]:.1%}]")
    print(f"  BSD:      {b_acc:.1%} [{b_ci[0]:.1%}, {b_ci[1]:.1%}]")
    print(f"  A-CFG:    {a_acc:.1%} [{a_ci[0]:.1%}, {a_ci[1]:.1%}]")

    # Flip analysis
    print(f"\n--- Flip Analysis ---")
    methods = [
        ("BSD", b_results), ("A-CFG", a_results), ("DMI", d_results),
    ]
    flips = {}
    for name, m_results in methods:
        m_over_v = sum(1 for mr, vr in zip(m_results, v_results)
                       if mr["is_correct"] and not vr["is_correct"])
        v_over_m = sum(1 for mr, vr in zip(m_results, v_results)
                       if not mr["is_correct"] and vr["is_correct"])
        print(f"  {name} vs Vanilla: +{m_over_v} / -{v_over_m}")
        flips[f"{name}_over_vanilla"] = m_over_v
        flips[f"vanilla_over_{name}"] = v_over_m

    # Degeneration checks
    print(f"\n--- Degeneration Checks ---")
    for name, div in [("BSD", b_diversity), ("A-CFG", a_diversity), ("DMI", d_diversity)]:
        degen = check_degeneration(div, v_diversity)
        if degen:
            print(f"  {name} WARNINGS: {degen}")
        else:
            print(f"  {name}: No degeneration detected")

    # Qualitative samples
    print(f"\n{'='*70}")
    print(f"  QUALITATIVE SAMPLES — BSD on GSM8K")
    print_qualitative_samples(b_results, n=5)
    print(f"\n{'='*70}")
    print(f"  QUALITATIVE SAMPLES — A-CFG on GSM8K")
    print_qualitative_samples(a_results, n=5)

    # ── Best method determination ──
    best_acc = max(v_acc, d_acc, b_acc, a_acc)
    best_method = "vanilla"
    if b_acc == best_acc:
        best_method = "BSD"
    elif a_acc == best_acc:
        best_method = "A-CFG"
    elif d_acc == best_acc:
        best_method = "DMI"

    # Pass criteria: accuracy > vanilla Dream-7B on GSM8K subset
    any_beats_vanilla = (b_acc > v_acc) or (a_acc > v_acc) or (d_acc > v_acc)
    verdict = "GO" if any_beats_vanilla else "NO-GO"
    if v_acc == 0 and b_acc == 0 and a_acc == 0:
        verdict = "INCONCLUSIVE"

    print(f"\n{'='*70}")
    print(f"  PILOT VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  Vanilla:   {v_correct}/{N_SAMPLES} = {v_acc:.1%}  CI={v_ci}")
    print(f"  DMI:       {d_correct}/{N_SAMPLES} = {d_acc:.1%}  CI={d_ci}")
    print(f"  BSD:       {b_correct}/{N_SAMPLES} = {b_acc:.1%}  CI={b_ci}")
    print(f"  A-CFG:     {a_correct}/{N_SAMPLES} = {a_acc:.1%}  CI={a_ci}")
    print(f"  Best: {best_method} ({best_acc:.1%})")
    print(f"  Any > Vanilla: {any_beats_vanilla}")

    # ── Save results ──
    elapsed_total = time.time() - start_time

    combined = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "verdict": verdict,
        "model": "Dream-v0-Instruct-7B",
        "benchmark": f"GSM8K-{N_SAMPLES}",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_s": round(elapsed_total, 1),
        "elapsed_total_min": round(elapsed_total / 60, 1),
        "best_method": best_method,
        "best_accuracy": best_acc,
        "methods_config": {
            "vanilla": {"steps": STEPS, "gen_len": GEN_LEN, "temperature": TEMPERATURE},
            "dmi": {"alpha": DMI_ALPHA, "soft_tau": SOFT_TAU},
            "bsd": {"k_frac": 0.75, "alpha_schedule": "linear(0.1->0.8)",
                    "tau_schedule": "linear(1.0->0.3)"},
            "acfg": {"w": 1.5, "remask_pct": 0.1},
        },
        "results": {
            "vanilla": {
                "accuracy": v_acc, "n_correct": v_correct, "n_samples": N_SAMPLES,
                "bootstrap_95ci": v_ci,
                **v_diversity,
                "avg_gen_time_s": float(np.mean(v_times)),
                "flops_ratio": 1.0,
            },
            "dmi": {
                "accuracy": d_acc, "n_correct": d_correct, "n_samples": N_SAMPLES,
                "bootstrap_95ci": d_ci,
                **d_diversity,
                "avg_gen_time_s": float(np.mean(d_times)),
                "flops_ratio": 1.05,
            },
            "bsd": {
                "accuracy": b_acc, "n_correct": b_correct, "n_samples": N_SAMPLES,
                "bootstrap_95ci": b_ci,
                **b_diversity,
                "avg_gen_time_s": float(np.mean(b_times)),
                "flops_ratio": 1.1,
            },
            "acfg": {
                "accuracy": a_acc, "n_correct": a_correct, "n_samples": N_SAMPLES,
                "bootstrap_95ci": a_ci,
                **a_diversity,
                "avg_gen_time_s": float(np.mean(a_times)),
                "flops_ratio": 2.0,
            },
        },
        "flips": flips,
        "pass_criteria": {
            "any_beats_vanilla": any_beats_vanilla,
            "verdict": verdict,
        },
        "per_sample": {
            "vanilla": v_results,
            "dmi": d_results,
            "bsd": b_results,
            "acfg": a_results,
        },
    }

    out_file = RESULTS_DIR / "gsm8k_best_method.json"
    save_results(combined, str(out_file))

    # Summary markdown
    summary_md = RESULTS_DIR / "gsm8k_extension_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# GSM8K Generalization Test — PILOT ({N_SAMPLES} samples)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Method | Accuracy | 95% CI | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |\n")
        f.write(f"|--------|----------|--------|-------|-------|------------|----------|-------|\n")
        for name, acc, correct, ci, div, times_list, flops in [
            ("Vanilla", v_acc, v_correct, v_ci, v_diversity, v_times, "1.0x"),
            ("DMI (alpha=0.3)", d_acc, d_correct, d_ci, d_diversity, d_times, "~1.05x"),
            ("BSD (k=0.75)", b_acc, b_correct, b_ci, b_diversity, b_times, "~1.1x"),
            ("A-CFG (w=1.5)", a_acc, a_correct, a_ci, a_diversity, a_times, "~2.0x"),
        ]:
            bold = "**" if acc == best_acc and acc > 0 else ""
            f.write(f"| {bold}{name}{bold} | {bold}{acc:.1%} ({correct}/{N_SAMPLES}){bold} | "
                    f"[{ci[0]:.1%}, {ci[1]:.1%}] | "
                    f"{div['rep_2']:.3f} | {div['rep_3']:.3f} | "
                    f"{div['distinct_3']:.3f} | {np.mean(times_list):.1f}s | {flops} |\n")
        f.write(f"\n## Best Method\n\n")
        f.write(f"- **{best_method}**: {best_acc:.1%}\n")
        f.write(f"- Beats vanilla: {any_beats_vanilla}\n")
        f.write(f"\n## Flip Analysis\n\n")
        for k, v in flips.items():
            f.write(f"- {k}: {v}\n")
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
        summary=f"GSM8K PILOT: Vanilla={v_acc:.1%}, DMI={d_acc:.1%}, "
                f"BSD={b_acc:.1%}, A-CFG={a_acc:.1%}. "
                f"Best={best_method}({best_acc:.1%}). "
                f"Verdict={verdict}. Time={elapsed_total/60:.1f}min",
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
        "planned_min": 90,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "Dream-v0-Instruct-7B",
            "method": "GSM8K extension (PILOT)",
            "methods_tested": ["vanilla", "DMI", "BSD", "A-CFG"],
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "gen_len": GEN_LEN,
            "best_method": best_method,
            "best_accuracy": best_acc,
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
