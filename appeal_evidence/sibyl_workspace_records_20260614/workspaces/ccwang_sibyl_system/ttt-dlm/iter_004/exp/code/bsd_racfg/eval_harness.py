"""
Shared Evaluation Harness for BSD/RACFG Experiments (Iteration 4).

Provides:
  - Countdown problem generation (deterministic, matches prior iterations)
  - Accuracy evaluation with answer parsing
  - Diversity metrics: rep-2/3, distinct-1/2/3
  - Length statistics
  - Qualitative sample printing
  - Unified result serialization

All experiment scripts import from this module for consistent evaluation.
"""
import os
import re
import json
import time
import random
import math
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any


# ──────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
MASK_TOKEN_ID = 151666
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
RESULTS_PILOTS = Path(f"{PROJECT_DIR}/exp/results/pilots")
RESULTS_FULL = Path(f"{PROJECT_DIR}/exp/results/full")


# ──────────────────────────────────────────────────
# 1. Countdown Problem Generator
# ──────────────────────────────────────────────────

def generate_countdown_problems(n: int, seed: int = 42,
                                 min_nums: int = 3, max_nums: int = 5) -> List[Dict]:
    """Generate Countdown problems with deterministic seeding.

    Matches the generator from prior iterations (task_3a_dmi_pilot.py) for
    reproducibility and fair comparison.
    """
    rng = random.Random(seed)
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 50:
        attempts += 1
        num_count = rng.randint(min_nums, max_nums)
        numbers = [rng.randint(1, 25) for _ in range(num_count)]
        if num_count >= 3:
            a, b, c = numbers[0], numbers[1], numbers[2]
            ops = ['+', '-', '*']
            op1 = rng.choice(ops)
            op2 = rng.choice(ops)
            try:
                intermediate = eval(f"{a} {op1} {b}")
                target = eval(f"{intermediate} {op2} {c}")
                if target > 0 and target == int(target) and 10 <= target <= 999:
                    target = int(target)
                    problems.append({
                        "numbers": numbers,
                        "target": target,
                        "solution_hint": f"({a} {op1} {b}) {op2} {c} = {target}"
                    })
                    continue
            except:
                pass
        a, b = numbers[0], numbers[1]
        target = a * b + (numbers[2] if num_count > 2 else 0)
        if 10 <= target <= 999:
            problems.append({
                "numbers": numbers,
                "target": target,
                "solution_hint": f"{a}*{b}+{numbers[2] if num_count > 2 else 0} = {target}"
            })
    return problems[:n]


def format_countdown_prompt(problem: Dict) -> str:
    """Format a Countdown problem as a chat prompt for Dream-7B."""
    nums = problem["numbers"]
    target = problem["target"]
    nums_str = ", ".join(str(n) for n in nums)
    return (
        f"Use the numbers [{nums_str}] with basic arithmetic operations "
        f"(+, -, *, /) to obtain {target}. "
        f"Each number can only be used once. "
        f"Show your work step by step, then provide the final equation.\n"
        f"Format: Step 1: ... Step 2: ... Answer: <equation> = {target}"
    )


# ──────────────────────────────────────────────────
# 2. Answer Parsing & Verification
# ──────────────────────────────────────────────────

def verify_countdown_answer(text: str, problem: Dict) -> Dict:
    """Verify a model's Countdown answer.

    Returns dict with is_correct, extracted_equation, evaluated_result, etc.
    """
    target = problem["target"]
    numbers = problem["numbers"]

    # Try "Answer: ..." pattern first
    answer_match = re.search(r'Answer:\s*(.+)', text, re.IGNORECASE)
    if answer_match:
        equation_str = answer_match.group(1).strip()
    else:
        # Fallback: find equations like "expr = number"
        eq_matches = re.findall(r'([\d\s+\-*/()]+)\s*=\s*(\d+)', text)
        if eq_matches:
            equation_str = eq_matches[-1][0].strip()
        else:
            return {"is_correct": False, "extracted_equation": None,
                    "explanation": "No equation found in output"}

    # Clean trailing "= number"
    equation_str = re.sub(r'\s*=\s*\d+\s*$', '', equation_str).strip()
    if not equation_str:
        return {"is_correct": False, "extracted_equation": None,
                "explanation": "Empty equation after cleanup"}

    try:
        safe_eq = re.sub(r'[^\d+\-*/(). ]', '', equation_str)
        if not safe_eq:
            return {"is_correct": False, "extracted_equation": equation_str,
                    "explanation": "Equation contains invalid characters"}
        result = eval(safe_eq, {"__builtins__": {}}, {})
        is_correct = abs(result - target) < 1e-6

        # Validate used numbers
        used_numbers = [int(x) for x in re.findall(r'\d+', safe_eq)]
        avail = list(numbers)
        numbers_valid = True
        for u in used_numbers:
            if u in avail:
                avail.remove(u)
            else:
                numbers_valid = False
                break

        return {
            "is_correct": is_correct and numbers_valid,
            "result_correct": is_correct,
            "numbers_valid": numbers_valid,
            "extracted_equation": equation_str,
            "evaluated_result": float(result),
            "target": target,
            "used_numbers": used_numbers,
            "explanation": f"eval({safe_eq}) = {result}, target = {target}, "
                           f"numbers_valid = {numbers_valid}"
        }
    except Exception as e:
        return {"is_correct": False, "extracted_equation": equation_str,
                "explanation": f"Eval error: {e}"}


# ──────────────────────────────────────────────────
# 3. Diversity & Quality Metrics
# ──────────────────────────────────────────────────

def distinct_n(text: str, n: int) -> float:
    """Compute distinct-n for a single text."""
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def distinct_n_corpus(texts: List[str], n: int) -> float:
    """Compute corpus-level distinct-n."""
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def rep_n(text: str, n: int) -> float:
    """Compute rep-n (repetition rate) for a single text."""
    return 1.0 - distinct_n(text, n)


def rep_n_corpus(texts: List[str], n: int) -> float:
    """Compute corpus-level average rep-n."""
    rates = []
    for text in texts:
        tokens = text.split()
        if len(tokens) < n + 1:
            rates.append(0.0)
            continue
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counter = Counter(ngrams)
        repeated = sum(c - 1 for c in counter.values() if c > 1)
        rates.append(repeated / len(ngrams) if ngrams else 0.0)
    return float(np.mean(rates)) if rates else 0.0


def compute_diversity_metrics(texts: List[str]) -> Dict:
    """Compute all diversity and length metrics for a batch of texts."""
    word_counts = [len(t.split()) for t in texts]
    char_counts = [len(t) for t in texts]
    return {
        "distinct_1": distinct_n_corpus(texts, 1),
        "distinct_2": distinct_n_corpus(texts, 2),
        "distinct_3": distinct_n_corpus(texts, 3),
        "rep_2": rep_n_corpus(texts, 2),
        "rep_3": rep_n_corpus(texts, 3),
        "avg_length_words": float(np.mean(word_counts)) if word_counts else 0,
        "std_length_words": float(np.std(word_counts)) if word_counts else 0,
        "avg_length_chars": float(np.mean(char_counts)) if char_counts else 0,
        "std_length_chars": float(np.std(char_counts)) if char_counts else 0,
        "min_length_words": int(np.min(word_counts)) if word_counts else 0,
        "max_length_words": int(np.max(word_counts)) if word_counts else 0,
    }


def compute_per_sample_metrics(text: str) -> Dict:
    """Compute diversity metrics for a single text."""
    return {
        "word_count": len(text.split()),
        "char_count": len(text),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "rep_2": rep_n(text, 2),
        "rep_3": rep_n(text, 3),
    }


# ──────────────────────────────────────────────────
# 4. Model Loading
# ──────────────────────────────────────────────────

def load_dream(device: str = "cuda:0"):
    """Load Dream-7B model and tokenizer.

    Returns (model, tokenizer, embedding_layer).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Get embedding layer for BSD/DMI
    embedding_layer = model.model.embed_tokens

    print(f"[eval_harness] Dream-7B loaded on {device}")
    print(f"[eval_harness] Embedding: {embedding_layer.weight.shape}")
    return model, tokenizer, embedding_layer


# ──────────────────────────────────────────────────
# 5. Vanilla Generation (baseline denoising loop)
# ──────────────────────────────────────────────────

def vanilla_generate(model, tokenizer, prompt_text: str, device: str = "cuda:0",
                     gen_len: int = 256, steps: int = 128,
                     temperature: float = 0.4) -> Tuple[str, float, Dict]:
    """Standard Dream-7B denoising generation (vanilla baseline).

    Returns (text, elapsed_seconds, diagnostics_dict).
    """
    import torch
    import torch.nn.functional as F

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

    # Decode
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

    diagnostics = {"steps": steps, "gen_len": gen_len, "temperature": temperature}
    return text, elapsed, diagnostics


# ──────────────────────────────────────────────────
# 6. Result Formatting & Saving
# ──────────────────────────────────────────────────

def format_results(method: str, benchmark: str, n_samples: int, seed: int,
                   accuracy: float, n_correct: int,
                   diversity: Dict, gen_times: List[float],
                   per_sample: List[Dict],
                   extra_config: Optional[Dict] = None) -> Dict:
    """Create a standardized result dict."""
    result = {
        "method": method,
        "benchmark": benchmark,
        "model": "Dream-v0-Instruct-7B",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_samples": n_samples,
            "seed": seed,
            "gen_len": 256,
            "steps": 128,
            "temperature": 0.4,
            **(extra_config or {}),
        },
        "metrics": {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_samples": n_samples,
            **diversity,
            "avg_gen_time_s": float(np.mean(gen_times)) if gen_times else 0,
            "total_gen_time_s": float(np.sum(gen_times)) if gen_times else 0,
        },
        "per_sample": per_sample,
    }
    return result


def save_results(result: Dict, filepath: str):
    """Save results to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[eval_harness] Results saved to {filepath}")


def print_qualitative_samples(per_sample: List[Dict], n: int = 10):
    """Print qualitative output samples for inspection."""
    print(f"\n--- Qualitative Samples (first {n}) ---")
    for i, s in enumerate(per_sample[:n]):
        status = "CORRECT" if s.get("is_correct") else "WRONG"
        target = s.get("target", "?")
        eq = s.get("extracted_equation", "N/A")
        text = s.get("generated_text", "")[:200]
        print(f"  [{i}] {status} | target={target} | eq={eq}")
        print(f"       {text}")
    print("---")


def print_comparison_table(results: List[Dict]):
    """Print a comparison table for multiple methods."""
    print(f"\n{'Method':<20} {'Accuracy':>10} {'rep-2':>8} {'rep-3':>8} "
          f"{'dist-3':>8} {'AvgTime':>8}")
    print("-" * 70)
    for r in results:
        m = r["metrics"]
        print(f"{r['method']:<20} {m['accuracy']:>10.1%} {m.get('rep_2', 0):>8.3f} "
              f"{m.get('rep_3', 0):>8.3f} {m.get('distinct_3', 0):>8.3f} "
              f"{m.get('avg_gen_time_s', 0):>8.1f}s")


# ──────────────────────────────────────────────────
# 7. Degeneration Checks
# ──────────────────────────────────────────────────

def check_degeneration(method_diversity: Dict, vanilla_diversity: Dict) -> Dict:
    """Check for degeneration relative to vanilla baseline.

    Returns warnings dict with alert flags.
    """
    warnings = {}

    # rep-3 alert: > vanilla + 20%
    v_rep3 = vanilla_diversity.get("rep_3", 0)
    m_rep3 = method_diversity.get("rep_3", 0)
    if m_rep3 > v_rep3 + 0.20:
        warnings["rep_3_alert"] = (
            f"rep-3 ({m_rep3:.3f}) exceeds vanilla ({v_rep3:.3f}) + 20% threshold"
        )

    # distinct-3 alert: < vanilla - 15%
    v_dist3 = vanilla_diversity.get("distinct_3", 1)
    m_dist3 = method_diversity.get("distinct_3", 1)
    if m_dist3 < v_dist3 - 0.15:
        warnings["distinct_3_alert"] = (
            f"distinct-3 ({m_dist3:.3f}) below vanilla ({v_dist3:.3f}) - 15% threshold"
        )

    # Length variance alert: variance < vanilla/2
    v_std = vanilla_diversity.get("std_length_words", 1)
    m_std = method_diversity.get("std_length_words", 1)
    if v_std > 0 and m_std < v_std / 2:
        warnings["length_collapse_alert"] = (
            f"Length std ({m_std:.1f}) < vanilla/2 ({v_std/2:.1f}): possible mode collapse"
        )

    return warnings
