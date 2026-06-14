#!/usr/bin/env python3
"""
alt_a_info_gain — Alternative A: Information-Gain Adaptive Unmasking Pilot.

Training-free approach: instead of random unmasking order (Dream default),
score each masked position by information gain (entropy of the predicted
distribution) and unmask highest-confidence (lowest-entropy) positions first.

Rationale: positions where the model is most confident should be unmasked first,
providing reliable context for subsequent steps. This is equivalent to
"easy-first" decoding — unmask what you know, then use that to figure out
what you don't.

Two strategies implemented:
  1. info_gain_greedy: At each step, compute entropy for all masked positions,
     unmask the lowest-entropy (highest-confidence) positions first.
  2. info_gain_block: Same idea but applied in blocks — sort masked positions
     by entropy, unmask the lowest-entropy block at each step.

Evaluated on GSM8K-100 against vanilla baseline (random unmasking).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 alt_a_info_gain.py
"""
import os, sys, json, time, random, re, math, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── Config ──
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
PROJECT_DIR = Path("/home/ccwang/sibyl_system/projects/ttt-dlm")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "alt_a_pilot"
N_SAMPLES = 100
SEED = 42
GEN_LEN = 512
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666


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

def load_gsm8k(n_samples=100, seed=42):
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
    rng = random.Random(seed)
    indices = list(range(len(problems)))
    rng.shuffle(indices)
    return [problems[i] for i in indices[:n_samples]]


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


def generate_info_gain(model, tokenizer, prompt_text, device="cuda:0"):
    """
    Information-Gain Adaptive Unmasking.

    At each denoising step:
      1. Run forward pass to get logits for all masked positions
      2. Compute entropy for each masked position
      3. Instead of random selection of which positions to unmask,
         preferentially unmask LOW-entropy (high-confidence) positions first
      4. Sample tokens for the selected positions

    This is "easy-first" decoding: resolve confident positions first,
    providing stable context for harder positions later.

    Implementation: We use the same number of steps and same transfer
    probability schedule as vanilla. The only difference is WHICH masked
    positions get unmasked at each step — we bias toward low-entropy ones.
    """
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)

    # Track entropy statistics for analysis
    entropy_stats = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break

            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]  # (n_masked, vocab)

            t = timesteps[i]; s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

            n_masked = mask_logits.shape[0]
            n_to_unmask = max(1, int(n_masked * p_transfer))
            # Ensure we don't try to unmask more than available
            n_to_unmask = min(n_to_unmask, n_masked)

            # Compute entropy for each masked position
            entropy = compute_entropy(mask_logits)  # (n_masked,)

            # Select positions to unmask: prefer LOW entropy (high confidence)
            # Use a soft selection: sample with probability inversely proportional to entropy
            # But for clean comparison, use deterministic: pick the n_to_unmask lowest-entropy positions
            if n_to_unmask >= n_masked:
                # Unmask all remaining
                selected_indices = torch.arange(n_masked, device=device)
            else:
                # Pick lowest-entropy positions (most confident)
                _, sorted_idx = entropy.sort()  # ascending: lowest entropy first
                selected_indices = sorted_idx[:n_to_unmask]

            # Record stats (subsample to avoid memory issues)
            if i % 16 == 0 or i < 4:
                entropy_stats.append({
                    "step": i,
                    "n_masked": n_masked,
                    "n_to_unmask": n_to_unmask,
                    "entropy_mean": float(entropy.mean().item()),
                    "entropy_std": float(entropy.std().item()),
                    "entropy_min": float(entropy.min().item()),
                    "entropy_max": float(entropy.max().item()),
                    "selected_entropy_mean": float(entropy[selected_indices].mean().item()),
                })

            # Sample tokens for selected positions
            selected_logits = mask_logits[selected_indices]
            _, sampled = sample_tokens_from_logits(selected_logits, TEMPERATURE)

            # Build the update: only unmask selected positions
            # We need to map selected_indices back to positions in x
            # mask_index is a boolean mask over x[0]; we need the actual positions
            mask_positions = mask_index.nonzero(as_tuple=True)  # (batch_idx, seq_idx)
            batch_idx = mask_positions[0][selected_indices]
            seq_idx = mask_positions[1][selected_indices]
            x[batch_idx, seq_idx] = sampled

    text = decode_output(tokenizer, x, prompt_len)
    elapsed = time.time() - t0
    return text, elapsed, entropy_stats


def generate_info_gain_soft(model, tokenizer, prompt_text, device="cuda:0"):
    """
    Soft Information-Gain Unmasking.

    Instead of deterministically picking the lowest-entropy positions,
    use entropy-weighted sampling: positions with lower entropy have
    higher probability of being unmasked, but there's still randomness.

    This avoids potential issues with deterministic ordering (e.g., always
    unmasking function words first) while still biasing toward confident positions.

    Weighting: P(unmask position i) proportional to exp(-entropy_i / tau)
    where tau controls the sharpness of the preference.
    """
    ENTROPY_TAU = 1.0  # temperature for entropy-based selection

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
                selection_logits = -entropy / ENTROPY_TAU
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
    return text, elapsed, []


# ── Main ──

def main():
    device = "cuda:0"
    start_time = datetime.now()

    print(f"{'='*70}")
    print(f"  alt_a_info_gain: Information-Gain Adaptive Unmasking Pilot")
    print(f"  Model: Dream-7B-Instruct | Steps: {STEPS} | Temp: {TEMPERATURE}")
    print(f"  N_samples: {N_SAMPLES} | Seed: {SEED} | Device: {device}")
    print(f"  Methods: vanilla, info_gain_greedy, info_gain_soft")
    print(f"{'='*70}")

    write_pid()
    # 3 methods x N_SAMPLES = total work items
    total_work = N_SAMPLES * 3
    report_progress("init", 0, total_work)

    # Load data
    problems = load_gsm8k(n_samples=N_SAMPLES, seed=SEED)
    print(f"Loaded {len(problems)} GSM8K problems")

    # Load model
    model, tokenizer = load_dream(device)

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ── Phase 1: Vanilla baseline ──
    print(f"\n{'─'*50}")
    print(f"Phase 1: Vanilla baseline ({N_SAMPLES} samples)")
    print(f"{'─'*50}")

    vanilla_results = []
    vanilla_correct = 0
    vanilla_total_time = 0

    for i, problem in enumerate(problems):
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
            "distinct_2": distinct_n(text, 2),
            "generated_text": text[:500],
        }
        vanilla_results.append(result)

        if (i + 1) % 10 == 0 or i < 3:
            acc = vanilla_correct / (i + 1)
            st = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {st} | target={problem['target']} "
                  f"got={verification['extracted']} | acc={acc:.1%} | {elapsed:.1f}s")
            report_progress("vanilla", i + 1, total_work, {"vanilla_acc": acc})

    vanilla_accuracy = vanilla_correct / N_SAMPLES
    print(f"\n  Vanilla: {vanilla_correct}/{N_SAMPLES} = {vanilla_accuracy:.1%} "
          f"({vanilla_total_time/N_SAMPLES:.1f}s/sample)")

    # ── Phase 2: Info-Gain Greedy ──
    print(f"\n{'─'*50}")
    print(f"Phase 2: Info-Gain Greedy ({N_SAMPLES} samples)")
    print(f"{'─'*50}")

    ig_greedy_results = []
    ig_greedy_correct = 0
    ig_greedy_total_time = 0
    all_entropy_stats = []

    for i, problem in enumerate(problems):
        prompt = format_gsm8k_prompt(problem)
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        text, elapsed, entropy_stats = generate_info_gain(
            model, tokenizer, prompt, device
        )
        ig_greedy_total_time += elapsed

        verification = verify_gsm8k(text, problem)
        if verification["is_correct"]:
            ig_greedy_correct += 1

        result = {
            "idx": i, "problem_id": problem["id"],
            "target": problem["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "distinct_2": distinct_n(text, 2),
            "generated_text": text[:500],
        }
        ig_greedy_results.append(result)

        # Save entropy stats for first 10 samples
        if i < 10:
            all_entropy_stats.append({
                "sample_idx": i,
                "stats": entropy_stats,
            })

        if (i + 1) % 10 == 0 or i < 3:
            acc = ig_greedy_correct / (i + 1)
            st = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {st} | target={problem['target']} "
                  f"got={verification['extracted']} | acc={acc:.1%} | {elapsed:.1f}s")
            report_progress("info_gain_greedy", N_SAMPLES + i + 1, total_work,
                           {"vanilla_acc": vanilla_accuracy,
                            "ig_greedy_acc": acc})

    ig_greedy_accuracy = ig_greedy_correct / N_SAMPLES
    print(f"\n  Info-Gain Greedy: {ig_greedy_correct}/{N_SAMPLES} = {ig_greedy_accuracy:.1%} "
          f"({ig_greedy_total_time/N_SAMPLES:.1f}s/sample)")

    # ── Phase 3: Info-Gain Soft ──
    print(f"\n{'─'*50}")
    print(f"Phase 3: Info-Gain Soft (tau=1.0, {N_SAMPLES} samples)")
    print(f"{'─'*50}")

    ig_soft_results = []
    ig_soft_correct = 0
    ig_soft_total_time = 0

    for i, problem in enumerate(problems):
        prompt = format_gsm8k_prompt(problem)
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        text, elapsed, _ = generate_info_gain_soft(
            model, tokenizer, prompt, device
        )
        ig_soft_total_time += elapsed

        verification = verify_gsm8k(text, problem)
        if verification["is_correct"]:
            ig_soft_correct += 1

        result = {
            "idx": i, "problem_id": problem["id"],
            "target": problem["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "distinct_2": distinct_n(text, 2),
            "generated_text": text[:500],
        }
        ig_soft_results.append(result)

        if (i + 1) % 10 == 0 or i < 3:
            acc = ig_soft_correct / (i + 1)
            st = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {st} | target={problem['target']} "
                  f"got={verification['extracted']} | acc={acc:.1%} | {elapsed:.1f}s")
            report_progress("info_gain_soft", 2 * N_SAMPLES + i + 1, total_work,
                           {"vanilla_acc": vanilla_accuracy,
                            "ig_greedy_acc": ig_greedy_accuracy,
                            "ig_soft_acc": acc})

    ig_soft_accuracy = ig_soft_correct / N_SAMPLES
    print(f"\n  Info-Gain Soft: {ig_soft_correct}/{N_SAMPLES} = {ig_soft_accuracy:.1%} "
          f"({ig_soft_total_time/N_SAMPLES:.1f}s/sample)")

    # ── Compute per-sample agreement analysis ──
    # Which samples did each method get right/wrong?
    agreement = {"all_correct": 0, "all_wrong": 0, "ig_only": 0, "vanilla_only": 0}
    for i in range(N_SAMPLES):
        v = vanilla_results[i]["is_correct"]
        g = ig_greedy_results[i]["is_correct"]
        if v and g:
            agreement["all_correct"] += 1
        elif not v and not g:
            agreement["all_wrong"] += 1
        elif g and not v:
            agreement["ig_only"] += 1
        elif v and not g:
            agreement["vanilla_only"] += 1

    # ── Diversity metrics ──
    def agg_diversity(results):
        d1 = [distinct_n(r["generated_text"], 1) for r in results]
        d2 = [r["distinct_2"] for r in results]
        d3 = [distinct_n(r["generated_text"], 3) for r in results]
        wc = [r["word_count"] for r in results]
        return {
            "distinct_1_mean": float(np.mean(d1)),
            "distinct_2_mean": float(np.mean(d2)),
            "distinct_3_mean": float(np.mean(d3)),
            "avg_word_count": float(np.mean(wc)),
        }

    vanilla_div = agg_diversity(vanilla_results)
    ig_greedy_div = agg_diversity(ig_greedy_results)
    ig_soft_div = agg_diversity(ig_soft_results)

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
            "ig_greedy_answer": ig_greedy_results[i]["extracted"],
            "ig_greedy_correct": ig_greedy_results[i]["is_correct"],
            "ig_greedy_text": ig_greedy_results[i]["generated_text"][:300],
            "ig_soft_answer": ig_soft_results[i]["extracted"],
            "ig_soft_correct": ig_soft_results[i]["is_correct"],
            "ig_soft_text": ig_soft_results[i]["generated_text"][:300],
        })

    # ── Paired bootstrap significance test ──
    def paired_bootstrap_p(results_a, results_b, n_bootstrap=1000, seed=42):
        """One-sided test: is B better than A?"""
        rng = np.random.RandomState(seed)
        a = np.array([r["is_correct"] for r in results_a])
        b = np.array([r["is_correct"] for r in results_b])
        observed_diff = b.mean() - a.mean()
        n = len(a)
        count_ge = 0
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            diff = b[idx].mean() - a[idx].mean()
            if diff >= observed_diff:
                count_ge += 1
        # Two-sided: proportion of bootstraps where diff is at least as extreme
        # For one-sided (B > A): count where bootstrap diff >= 0
        count_positive = 0
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            diff = b[idx].mean() - a[idx].mean()
            if diff >= 0:
                count_positive += 1
        return {
            "observed_diff": float(observed_diff),
            "bootstrap_p_two_sided": float(count_ge / n_bootstrap),
            "bootstrap_p_one_sided": float(1 - count_positive / n_bootstrap),
        }

    ig_greedy_vs_vanilla = paired_bootstrap_p(vanilla_results, ig_greedy_results)
    ig_soft_vs_vanilla = paired_bootstrap_p(vanilla_results, ig_soft_results)

    # ── Determine best method ──
    best_method = "vanilla"
    best_acc = vanilla_accuracy
    if ig_greedy_accuracy > best_acc:
        best_method = "info_gain_greedy"
        best_acc = ig_greedy_accuracy
    if ig_soft_accuracy > best_acc:
        best_method = "info_gain_soft"
        best_acc = ig_soft_accuracy

    # ── Pass criteria ──
    # "Info-gain unmasking accuracy > vanilla baseline + 1% on GSM8K-100"
    ig_best_acc = max(ig_greedy_accuracy, ig_soft_accuracy)
    ig_best_name = "greedy" if ig_greedy_accuracy >= ig_soft_accuracy else "soft"
    pass_criteria_met = ig_best_acc > vanilla_accuracy + 0.01
    go_no_go = "GO" if pass_criteria_met else "NO_GO"

    # ── Build summary ──
    end_time = datetime.now()
    total_wall_clock = (end_time - start_time).total_seconds()

    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "candidate_id": "cand_info_gain",
        "benchmark": "gsm8k",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,

        "vanilla": {
            "accuracy": vanilla_accuracy,
            "correct": vanilla_correct,
            "total": N_SAMPLES,
            "avg_time_s": round(vanilla_total_time / N_SAMPLES, 2),
            "total_time_s": round(vanilla_total_time, 1),
            **vanilla_div,
        },
        "info_gain_greedy": {
            "accuracy": ig_greedy_accuracy,
            "correct": ig_greedy_correct,
            "total": N_SAMPLES,
            "avg_time_s": round(ig_greedy_total_time / N_SAMPLES, 2),
            "total_time_s": round(ig_greedy_total_time, 1),
            "overhead_vs_vanilla": round(ig_greedy_total_time / vanilla_total_time, 3) if vanilla_total_time > 0 else None,
            **ig_greedy_div,
        },
        "info_gain_soft": {
            "accuracy": ig_soft_accuracy,
            "correct": ig_soft_correct,
            "total": N_SAMPLES,
            "avg_time_s": round(ig_soft_total_time / N_SAMPLES, 2),
            "total_time_s": round(ig_soft_total_time, 1),
            "overhead_vs_vanilla": round(ig_soft_total_time / vanilla_total_time, 3) if vanilla_total_time > 0 else None,
            **ig_soft_div,
        },

        "comparison": {
            "ig_greedy_vs_vanilla": {
                "delta_accuracy": ig_greedy_accuracy - vanilla_accuracy,
                "delta_pct": f"{ig_greedy_accuracy - vanilla_accuracy:+.1%}",
                **ig_greedy_vs_vanilla,
            },
            "ig_soft_vs_vanilla": {
                "delta_accuracy": ig_soft_accuracy - vanilla_accuracy,
                "delta_pct": f"{ig_soft_accuracy - vanilla_accuracy:+.1%}",
                **ig_soft_vs_vanilla,
            },
            "best_ig_method": ig_best_name,
            "best_ig_accuracy": ig_best_acc,
            "best_overall_method": best_method,
            "best_overall_accuracy": best_acc,
        },

        "agreement_analysis": agreement,
        "entropy_analysis": {
            "description": "Entropy statistics from info_gain_greedy for first 10 samples",
            "samples": all_entropy_stats[:5],  # keep compact
        },
        "qualitative_samples": qualitative_samples[:5],  # keep compact in main file

        "pass_criteria": {
            "criterion": "Info-gain unmasking accuracy > vanilla baseline + 1% on GSM8K-100",
            "vanilla_accuracy": vanilla_accuracy,
            "best_ig_accuracy": ig_best_acc,
            "threshold": vanilla_accuracy + 0.01,
            "met": pass_criteria_met,
            "go_no_go": go_no_go,
        },

        "wall_clock_total_s": round(total_wall_clock, 1),
        "wall_clock_total_min": round(total_wall_clock / 60, 1),
        "timestamp": end_time.isoformat(),
        "torch_version": torch.__version__,
        "gpu_info": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 0,
        },
    }

    # ── Print final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY: alt_a_info_gain")
    print(f"{'='*70}")
    print(f"  Vanilla:          {vanilla_correct}/{N_SAMPLES} = {vanilla_accuracy:.1%}  ({vanilla_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  Info-Gain Greedy: {ig_greedy_correct}/{N_SAMPLES} = {ig_greedy_accuracy:.1%}  ({ig_greedy_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  Info-Gain Soft:   {ig_soft_correct}/{N_SAMPLES} = {ig_soft_accuracy:.1%}  ({ig_soft_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  ─────────────────────────────────")
    print(f"  Best IG method:   {ig_best_name} ({ig_best_acc:.1%})")
    print(f"  Delta vs vanilla: {ig_best_acc - vanilla_accuracy:+.1%}")
    print(f"  Agreement: both_correct={agreement['all_correct']}, "
          f"both_wrong={agreement['all_wrong']}, "
          f"ig_only={agreement['ig_only']}, vanilla_only={agreement['vanilla_only']}")
    print(f"  Bootstrap p (greedy vs vanilla): {ig_greedy_vs_vanilla['observed_diff']:+.3f}, "
          f"p={ig_greedy_vs_vanilla['bootstrap_p_two_sided']:.3f}")
    print(f"  Diversity: vanilla_d2={vanilla_div['distinct_2_mean']:.3f}, "
          f"ig_greedy_d2={ig_greedy_div['distinct_2_mean']:.3f}, "
          f"ig_soft_d2={ig_soft_div['distinct_2_mean']:.3f}")
    print(f"  Overhead: greedy={summary['info_gain_greedy']['overhead_vs_vanilla']:.2f}x, "
          f"soft={summary['info_gain_soft']['overhead_vs_vanilla']:.2f}x")
    print(f"  Total time: {total_wall_clock/60:.1f} min")
    print(f"  GO/NO-GO: {go_no_go}")
    print(f"{'='*70}")

    # Save results
    out_file = PILOT_DIR / "alt_a_info_gain.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Save full details separately
    details_file = PILOT_DIR / "alt_a_info_gain_details.json"
    with open(details_file, "w") as f:
        json.dump({
            "vanilla_results": vanilla_results,
            "ig_greedy_results": ig_greedy_results,
            "ig_soft_results": ig_soft_results,
            "entropy_stats": all_entropy_stats,
            "qualitative_samples": qualitative_samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"Details saved to {details_file}")

    # Mark done
    mark_done(
        status="success",
        summary=f"Vanilla={vanilla_accuracy:.1%}, IG-Greedy={ig_greedy_accuracy:.1%}, "
                f"IG-Soft={ig_soft_accuracy:.1%}, GO/NO-GO={go_no_go}"
    )
    print("DONE marker written.")


if __name__ == "__main__":
    main()
