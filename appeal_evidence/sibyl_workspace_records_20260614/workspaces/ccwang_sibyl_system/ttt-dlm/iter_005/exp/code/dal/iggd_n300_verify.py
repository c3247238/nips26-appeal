#!/usr/bin/env python3
"""
iggd_n300_verify — Phase 0: IGGD n=300 Verification on GSM8K.

Replicate alt_a_pilot result at n=300 for statistical significance.
Run Info-Gain Soft (tau=1.0) on Dream-7B, GSM8K first 300 problems,
128 denoising steps, seed 42.

Compute:
  - Accuracy (both methods)
  - Paired bootstrap p-value (n=1000 resamples)
  - Cohen's d
  - Distinct-1/2/3
  - Per-sample predictions saved for downstream analysis

Also run vanilla baseline on same 300 samples for paired comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 iggd_n300_verify.py
"""
import os, sys, json, time, random, re, math, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── Config ──
MODEL_DIR = "/home/ccwang/sibyl_system/shared/checkpoints/Dream-v0-Instruct-7B"
PROJECT_DIR = Path("/home/ccwang/sibyl_system/projects/ttt-dlm")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "iggd_n300_verify"
N_SAMPLES = 300
SEED = 42
GEN_LEN = 512
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666
ENTROPY_TAU = 1.0  # temperature for entropy-based selection (from pilot)

# ── Progress / PID / DONE markers ──

def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

def report_progress(phase, completed, total, extra=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "phase": phase,
        "completed": completed,
        "total": total,
        "updated_at": datetime.now().isoformat(),
    }
    if extra:
        data.update(extra)
    progress.write_text(json.dumps(data))

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


# ── GSM8K Data ──

def load_gsm8k_first_n(n_samples=300):
    """Load the FIRST n_samples from GSM8K test set (deterministic, no shuffle)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
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
    return problems


def format_gsm8k_prompt(problem):
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


def extract_model_answer(text):
    """Extract numerical answer from model output."""
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "####"
        except:
            pass
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:=]?\s*([\-]?\d[\d,]*\.?\d*)',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+answer\s+is\s+)?([\-]?\d[\d,]*\.?\d*)',
        r'\\boxed\{([\-]?\d[\d,]*\.?\d*)\}',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', '')), "pattern"
            except:
                continue
    match = re.search(r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "equation"
        except:
            pass
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', '')), "last_number"
        except:
            pass
    return None, "none"


def verify_gsm8k(text, problem):
    extracted, method = extract_model_answer(text)
    target = problem["target"]
    if extracted is None or target is None:
        return {"is_correct": False, "extracted": extracted, "target": target, "method": method}
    is_correct = abs(extracted - target) < 1e-3
    return {"is_correct": is_correct, "extracted": extracted, "target": target, "method": method}


# ── Text Quality ──

def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


# ── Model Loading ──

def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading Dream-7B from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}, dtype=bfloat16")
    return model, tokenizer


# ── Inference Utilities ──

def get_shifted_logits(model, x):
    """Forward pass with Dream's shifted-logits convention."""
    outputs = model(x, attention_mask="full", position_ids=None)
    logits = outputs.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def sample_tokens_from_logits(logits, temperature=0.4):
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, sampled = probs.max(dim=-1)
    return confidence, sampled


def prepare_input(tokenizer, prompt_text, device):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + GEN_LEN
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    return x, prompt_len


def decode_output(tokenizer, x, prompt_len):
    gen_ids = x[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t_id in gen_ids:
        if t_id == MASK_TOKEN_ID:
            continue
        if t_id == eos_id:
            break
        clean_ids.append(t_id)
    return tokenizer.decode(clean_ids, skip_special_tokens=True).strip()


def compute_entropy(logits):
    """Compute per-position entropy from logits. Lower entropy = higher confidence."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


# ── Generation Methods ──

def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0"):
    """Standard Dream denoising (random unmasking order)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]
            t = timesteps[i]; s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
    text = decode_output(tokenizer, x, prompt_len)
    return text, time.time() - t0


def generate_iggd_soft(model, tokenizer, prompt_text, device="cuda:0", tau=1.0):
    """
    IGGD-Soft: Information-Gain Guided Decoding with soft temperature-controlled selection.

    At each denoising step:
      1. Forward pass -> logits for all masked positions
      2. Compute entropy for each masked position (= information gain proxy)
      3. Sample positions to unmask with P(i) ~ exp(-entropy_i / tau)
         Lower entropy (higher confidence) -> higher selection probability
      4. Sample tokens for selected positions

    tau=1.0 was the best setting from the n=100 pilot (+5pp over vanilla).
    """
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break

            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]

            t = timesteps[i]; s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

            n_masked = mask_logits.shape[0]
            n_to_unmask = max(1, int(n_masked * p_transfer))
            n_to_unmask = min(n_to_unmask, n_masked)

            if n_to_unmask >= n_masked:
                selected_indices = torch.arange(n_masked, device=device)
            else:
                # Compute entropy and use as selection weights
                entropy = compute_entropy(mask_logits)
                # Lower entropy -> higher selection probability
                selection_logits = -entropy / tau
                selection_probs = F.softmax(selection_logits, dim=0)
                selected_indices = torch.multinomial(
                    selection_probs, num_samples=n_to_unmask, replacement=False
                )

            selected_logits = mask_logits[selected_indices]
            _, sampled = sample_tokens_from_logits(selected_logits, TEMPERATURE)

            mask_positions = mask_index.nonzero(as_tuple=True)
            batch_idx = mask_positions[0][selected_indices]
            seq_idx = mask_positions[1][selected_indices]
            x[batch_idx, seq_idx] = sampled

    text = decode_output(tokenizer, x, prompt_len)
    elapsed = time.time() - t0
    return text, elapsed


# ── Statistical Utilities ──

def paired_bootstrap_test(correct_a, correct_b, n_bootstrap=1000, seed=42):
    """
    Paired bootstrap significance test.
    Tests whether method B is significantly better than method A.

    Returns: observed_diff, p_value (one-sided), p_value (two-sided),
             95% CI of the difference, Cohen's d.
    """
    rng = np.random.RandomState(seed)
    a = np.array(correct_a, dtype=float)
    b = np.array(correct_b, dtype=float)
    n = len(a)

    observed_diff = b.mean() - a.mean()

    # Bootstrap distribution of the difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs.append(b[idx].mean() - a[idx].mean())
    boot_diffs = np.array(boot_diffs)

    # One-sided p-value: P(boot_diff <= 0) under resampling
    # If observed_diff > 0, we test H0: diff <= 0
    p_one_sided = np.mean(boot_diffs <= 0)

    # Two-sided p-value
    p_two_sided = np.mean(np.abs(boot_diffs) >= abs(observed_diff))

    # 95% CI
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Cohen's d (paired)
    diff = b - a
    diff_std = diff.std(ddof=1)
    cohens_d = float(observed_diff / diff_std) if diff_std > 0 else 0.0

    return {
        "observed_diff": float(observed_diff),
        "p_one_sided": float(p_one_sided),
        "p_two_sided": float(p_two_sided),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "cohens_d": cohens_d,
        "n_bootstrap": n_bootstrap,
    }


def agg_diversity(results):
    """Aggregate diversity metrics across results."""
    d1 = [distinct_n(r["generated_text"], 1) for r in results]
    d2 = [distinct_n(r["generated_text"], 2) for r in results]
    d3 = [distinct_n(r["generated_text"], 3) for r in results]
    wc = [r["word_count"] for r in results]
    return {
        "distinct_1_mean": float(np.mean(d1)),
        "distinct_2_mean": float(np.mean(d2)),
        "distinct_3_mean": float(np.mean(d3)),
        "avg_word_count": float(np.mean(wc)),
    }


# ── Checkpoint/Resume Support ──

def load_checkpoint():
    """Load checkpoint if exists for resumability."""
    ckpt_file = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
    if ckpt_file.exists():
        try:
            data = json.loads(ckpt_file.read_text())
            print(f"Resuming from checkpoint: vanilla={data.get('vanilla_done', 0)}, "
                  f"iggd={data.get('iggd_done', 0)}")
            return data
        except:
            pass
    return None


def save_checkpoint(vanilla_results, iggd_results, vanilla_done, iggd_done):
    """Save checkpoint for resumability."""
    ckpt_file = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
    ckpt_file.write_text(json.dumps({
        "vanilla_done": vanilla_done,
        "iggd_done": iggd_done,
        "vanilla_results": vanilla_results,
        "iggd_results": iggd_results,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Main ──

def main():
    device = "cuda:0"
    start_time = datetime.now()

    print(f"{'='*70}")
    print(f"  iggd_n300_verify: IGGD n=300 Verification on GSM8K")
    print(f"  Model: Dream-7B-Instruct | Steps: {STEPS} | Temp: {TEMPERATURE}")
    print(f"  N_samples: {N_SAMPLES} | Seed: {SEED} | Device: {device}")
    print(f"  IGGD tau: {ENTROPY_TAU}")
    print(f"  Methods: vanilla (paired baseline), IGGD-Soft (tau={ENTROPY_TAU})")
    print(f"{'='*70}")

    write_pid()
    total_work = N_SAMPLES * 2  # vanilla + IGGD
    report_progress("init", 0, total_work)

    # Load data — first 300 problems (deterministic, no shuffle)
    problems = load_gsm8k_first_n(n_samples=N_SAMPLES)
    print(f"Loaded first {len(problems)} GSM8K problems (deterministic order)")

    # Load model
    model, tokenizer = load_dream(device)

    # GPU profile
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024*1024)
    print(f"GPU: {gpu_name}, VRAM: {vram_total_mb} MB")

    # Check for checkpoint
    ckpt = load_checkpoint()
    vanilla_results = ckpt["vanilla_results"] if ckpt and "vanilla_results" in ckpt else []
    iggd_results = ckpt["iggd_results"] if ckpt and "iggd_results" in ckpt else []
    vanilla_start = ckpt.get("vanilla_done", 0) if ckpt else 0
    iggd_start = ckpt.get("iggd_done", 0) if ckpt else 0

    # ── Phase 1: Vanilla baseline ──
    print(f"\n{'─'*50}")
    print(f"Phase 1: Vanilla baseline ({N_SAMPLES} samples, starting from {vanilla_start})")
    print(f"{'─'*50}")

    vanilla_correct = sum(1 for r in vanilla_results if r["is_correct"])
    vanilla_total_time = sum(r["gen_time_s"] for r in vanilla_results)

    for i in range(vanilla_start, N_SAMPLES):
        problem = problems[i]
        prompt = format_gsm8k_prompt(problem)
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        text, elapsed = generate_vanilla(model, tokenizer, prompt, device)
        vanilla_total_time += elapsed

        verification = verify_gsm8k(text, problem)
        if verification["is_correct"]:
            vanilla_correct += 1

        result = {
            "idx": i, "problem_id": problem["id"],
            "target": problem["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "generated_text": text[:500],
        }
        vanilla_results.append(result)

        if (i + 1) % 20 == 0 or i < 3:
            acc = vanilla_correct / (i + 1)
            st = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {st} | target={problem['target']} "
                  f"got={verification['extracted']} | acc={acc:.1%} | {elapsed:.1f}s",
                  flush=True)
            report_progress("vanilla", i + 1, total_work, {"vanilla_acc": acc})

        # Periodic GPU memory cleanup
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            save_checkpoint(vanilla_results, iggd_results, i + 1, iggd_start)

    vanilla_accuracy = vanilla_correct / N_SAMPLES
    print(f"\n  Vanilla: {vanilla_correct}/{N_SAMPLES} = {vanilla_accuracy:.1%} "
          f"({vanilla_total_time/N_SAMPLES:.1f}s/sample)")

    # Save checkpoint after vanilla phase
    save_checkpoint(vanilla_results, iggd_results, N_SAMPLES, iggd_start)

    # ── Phase 2: IGGD-Soft ──
    print(f"\n{'─'*50}")
    print(f"Phase 2: IGGD-Soft tau={ENTROPY_TAU} ({N_SAMPLES} samples, starting from {iggd_start})")
    print(f"{'─'*50}")

    iggd_correct = sum(1 for r in iggd_results if r["is_correct"])
    iggd_total_time = sum(r["gen_time_s"] for r in iggd_results)

    for i in range(iggd_start, N_SAMPLES):
        problem = problems[i]
        prompt = format_gsm8k_prompt(problem)
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        text, elapsed = generate_iggd_soft(
            model, tokenizer, prompt, device, tau=ENTROPY_TAU
        )
        iggd_total_time += elapsed

        verification = verify_gsm8k(text, problem)
        if verification["is_correct"]:
            iggd_correct += 1

        result = {
            "idx": i, "problem_id": problem["id"],
            "target": problem["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "generated_text": text[:500],
        }
        iggd_results.append(result)

        if (i + 1) % 20 == 0 or i < 3:
            acc = iggd_correct / (i + 1)
            st = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {st} | target={problem['target']} "
                  f"got={verification['extracted']} | acc={acc:.1%} | {elapsed:.1f}s",
                  flush=True)
            report_progress("iggd_soft", N_SAMPLES + i + 1, total_work,
                           {"vanilla_acc": vanilla_accuracy, "iggd_acc": acc})

        # Periodic GPU memory cleanup
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            save_checkpoint(vanilla_results, iggd_results, N_SAMPLES, i + 1)

    iggd_accuracy = iggd_correct / N_SAMPLES
    print(f"\n  IGGD-Soft: {iggd_correct}/{N_SAMPLES} = {iggd_accuracy:.1%} "
          f"({iggd_total_time/N_SAMPLES:.1f}s/sample)")

    # ── Statistical Analysis ──
    print(f"\n{'─'*50}")
    print(f"Phase 3: Statistical Analysis")
    print(f"{'─'*50}")

    vanilla_correct_arr = [r["is_correct"] for r in vanilla_results]
    iggd_correct_arr = [r["is_correct"] for r in iggd_results]

    # Paired bootstrap test
    bootstrap_result = paired_bootstrap_test(
        vanilla_correct_arr, iggd_correct_arr,
        n_bootstrap=1000, seed=SEED
    )

    print(f"  Observed diff: {bootstrap_result['observed_diff']:+.4f}")
    print(f"  p-value (one-sided): {bootstrap_result['p_one_sided']:.4f}")
    print(f"  p-value (two-sided): {bootstrap_result['p_two_sided']:.4f}")
    print(f"  95% CI: [{bootstrap_result['ci_95_lower']:.4f}, {bootstrap_result['ci_95_upper']:.4f}]")
    print(f"  Cohen's d: {bootstrap_result['cohens_d']:.4f}")

    # ── Agreement Analysis ──
    agreement = {
        "both_correct": 0, "both_wrong": 0,
        "iggd_only": 0, "vanilla_only": 0
    }
    for i in range(N_SAMPLES):
        v = vanilla_results[i]["is_correct"]
        ig = iggd_results[i]["is_correct"]
        if v and ig:
            agreement["both_correct"] += 1
        elif not v and not ig:
            agreement["both_wrong"] += 1
        elif ig and not v:
            agreement["iggd_only"] += 1
        elif v and not ig:
            agreement["vanilla_only"] += 1

    print(f"\n  Agreement analysis:")
    print(f"    Both correct: {agreement['both_correct']}")
    print(f"    Both wrong:   {agreement['both_wrong']}")
    print(f"    IGGD only:    {agreement['iggd_only']}")
    print(f"    Vanilla only: {agreement['vanilla_only']}")

    # ── Diversity metrics ──
    vanilla_div = agg_diversity(vanilla_results)
    iggd_div = agg_diversity(iggd_results)

    print(f"\n  Diversity (Distinct-2): vanilla={vanilla_div['distinct_2_mean']:.3f}, "
          f"IGGD={iggd_div['distinct_2_mean']:.3f}")

    # ── Qualitative samples (first 10) ──
    qualitative_samples = []
    for i in range(min(10, N_SAMPLES)):
        qualitative_samples.append({
            "idx": i,
            "question": problems[i]["question"][:200],
            "target": problems[i]["target"],
            "vanilla_answer": vanilla_results[i]["extracted"],
            "vanilla_correct": vanilla_results[i]["is_correct"],
            "vanilla_text": vanilla_results[i]["generated_text"][:300],
            "iggd_answer": iggd_results[i]["extracted"],
            "iggd_correct": iggd_results[i]["is_correct"],
            "iggd_text": iggd_results[i]["generated_text"][:300],
        })

    # ── Pass Criteria ──
    # "IGGD-Soft accuracy >= vanilla + 3% absolute on GSM8K-300 AND p<0.05 (paired bootstrap)"
    delta = iggd_accuracy - vanilla_accuracy
    pass_acc = delta >= 0.03
    pass_sig = bootstrap_result["p_one_sided"] < 0.05
    pass_criteria_met = pass_acc and pass_sig
    go_no_go = "GO" if pass_criteria_met else "NO_GO"

    # Relaxed criteria for directional signal
    directional_positive = delta > 0
    directional_strong = delta >= 0.02
    relaxed_sig = bootstrap_result["p_one_sided"] < 0.10

    # ── Build summary ──
    end_time = datetime.now()
    total_wall_clock = (end_time - start_time).total_seconds()

    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "candidate_id": "cand_iggd",
        "benchmark": "gsm8k",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,
        "iggd_tau": ENTROPY_TAU,

        "vanilla": {
            "accuracy": vanilla_accuracy,
            "correct": vanilla_correct,
            "total": N_SAMPLES,
            "avg_time_s": round(vanilla_total_time / N_SAMPLES, 2),
            "total_time_s": round(vanilla_total_time, 1),
            **vanilla_div,
        },
        "iggd_soft": {
            "accuracy": iggd_accuracy,
            "correct": iggd_correct,
            "total": N_SAMPLES,
            "avg_time_s": round(iggd_total_time / N_SAMPLES, 2),
            "total_time_s": round(iggd_total_time, 1),
            "overhead_vs_vanilla": round(iggd_total_time / vanilla_total_time, 3) if vanilla_total_time > 0 else None,
            **iggd_div,
        },

        "statistical_test": {
            "test_type": "paired_bootstrap",
            "n_bootstrap": 1000,
            "delta_accuracy": round(delta, 4),
            "delta_pct": f"{delta:+.1%}",
            **bootstrap_result,
        },

        "agreement_analysis": agreement,

        "qualitative_samples": qualitative_samples[:5],

        "pass_criteria": {
            "primary_criterion": "IGGD-Soft >= vanilla + 3% absolute AND p<0.05",
            "vanilla_accuracy": vanilla_accuracy,
            "iggd_accuracy": iggd_accuracy,
            "delta": round(delta, 4),
            "p_value": bootstrap_result["p_one_sided"],
            "cohens_d": bootstrap_result["cohens_d"],
            "pass_acc_3pct": pass_acc,
            "pass_sig_005": pass_sig,
            "primary_met": pass_criteria_met,
            "go_no_go": go_no_go,
            "relaxed_criteria": {
                "directional_positive": directional_positive,
                "delta_ge_2pct": directional_strong,
                "p_lt_010": relaxed_sig,
            },
        },

        "wall_clock_total_s": round(total_wall_clock, 1),
        "wall_clock_total_min": round(total_wall_clock / 60, 1),
        "timestamp": end_time.isoformat(),
        "torch_version": torch.__version__,
        "gpu_info": {
            "device": device,
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total_mb,
        },
    }

    # ── Print final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY: iggd_n300_verify")
    print(f"{'='*70}")
    print(f"  Vanilla:    {vanilla_correct}/{N_SAMPLES} = {vanilla_accuracy:.1%}  "
          f"({vanilla_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  IGGD-Soft:  {iggd_correct}/{N_SAMPLES} = {iggd_accuracy:.1%}  "
          f"({iggd_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  ─────────────────────────────────")
    print(f"  Delta: {delta:+.1%} (absolute: {delta:+.4f})")
    print(f"  p-value (one-sided): {bootstrap_result['p_one_sided']:.4f}")
    print(f"  Cohen's d: {bootstrap_result['cohens_d']:.4f}")
    print(f"  95% CI: [{bootstrap_result['ci_95_lower']:.4f}, {bootstrap_result['ci_95_upper']:.4f}]")
    print(f"  ─────────────────────────────────")
    print(f"  Agreement: both_correct={agreement['both_correct']}, "
          f"both_wrong={agreement['both_wrong']}, "
          f"iggd_only={agreement['iggd_only']}, vanilla_only={agreement['vanilla_only']}")
    print(f"  Diversity: vanilla_d2={vanilla_div['distinct_2_mean']:.3f}, "
          f"iggd_d2={iggd_div['distinct_2_mean']:.3f}")
    print(f"  Overhead: {summary['iggd_soft']['overhead_vs_vanilla']:.2f}x")
    print(f"  Total time: {total_wall_clock/60:.1f} min")
    print(f"  ─────────────────────────────────")
    print(f"  PRIMARY GO/NO-GO: {go_no_go}")
    print(f"    >= +3% absolute: {'PASS' if pass_acc else 'FAIL'} (delta={delta:+.1%})")
    print(f"    p < 0.05:        {'PASS' if pass_sig else 'FAIL'} (p={bootstrap_result['p_one_sided']:.4f})")
    if not pass_criteria_met:
        print(f"  RELAXED CRITERIA:")
        print(f"    Directional positive: {'YES' if directional_positive else 'NO'}")
        print(f"    >= +2% absolute:      {'YES' if directional_strong else 'NO'}")
        print(f"    p < 0.10:             {'YES' if relaxed_sig else 'NO'}")
    print(f"{'='*70}")

    # Save results
    out_file = PILOT_DIR / "iggd_n300_gsm8k.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Save per-sample details for downstream analysis
    details_file = PILOT_DIR / "iggd_n300_gsm8k_details.json"
    with open(details_file, "w") as f:
        json.dump({
            "vanilla_results": vanilla_results,
            "iggd_results": iggd_results,
            "qualitative_samples": qualitative_samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"Per-sample details saved to {details_file}")

    # Clean up checkpoint
    ckpt_file = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
    if ckpt_file.exists():
        ckpt_file.unlink()
        print("Checkpoint file cleaned up.")

    # Mark done
    mark_done(
        status="success",
        summary=f"Vanilla={vanilla_accuracy:.1%}, IGGD-Soft={iggd_accuracy:.1%}, "
                f"delta={delta:+.1%}, p={bootstrap_result['p_one_sided']:.4f}, "
                f"Cohen's d={bootstrap_result['cohens_d']:.4f}, GO/NO-GO={go_no_go}"
    )
    print("DONE marker written.")


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", file=sys.stderr, flush=True)
        print(f"\n\nFATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        mark_done(status="failed", summary=f"FATAL: {e}")
        sys.exit(1)
