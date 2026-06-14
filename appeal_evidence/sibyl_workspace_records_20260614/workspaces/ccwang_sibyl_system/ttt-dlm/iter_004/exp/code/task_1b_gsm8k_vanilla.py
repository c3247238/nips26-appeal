"""
Task 1b (PILOT): Vanilla Baseline on GSM8K (16 samples).
Dream-7B vanilla (origin, 128 steps, temp=0.4) on GSM8K test set.
Metrics: EM accuracy, answer extraction rate, gen length, distinct-1/2/3, rep-2/3, inference time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_1b_gsm8k_vanilla.py
"""
import os, sys, json, time, re
from pathlib import Path
from collections import Counter

import numpy as np
import torch

# === Config ===
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 16
SEED = 42
GEN_LEN = 512       # GSM8K needs longer reasoning chains than Countdown
STEPS = 128
TEMPERATURE = 0.4
ALG = "origin"
MASK_TOKEN_ID = 151666


# === GSM8K Data Loading ===
def load_gsm8k(n=None, seed=42):
    """Load GSM8K test set from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    problems = []
    for i, item in enumerate(ds):
        if n is not None and i >= n:
            break
        # Extract numeric answer from "#### N" format
        answer_text = item["answer"]
        target = extract_gsm8k_target(answer_text)
        problems.append({
            "id": i,
            "question": item["question"],
            "answer_text": answer_text,
            "target": target,
        })
    return problems


def extract_gsm8k_target(text):
    """Extract the ground truth numeric answer from GSM8K '#### N' format."""
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        num_str = match.group(1).replace(',', '')
        try:
            return float(num_str)
        except:
            return None
    return None


# === Prompt Formatting ===
def format_gsm8k_prompt(problem):
    """Format a GSM8K problem as a chat prompt for Dream-7B."""
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


# === Answer Extraction ===
def extract_model_answer(text):
    """Extract numeric answer from model generation.

    Returns (value, method) or (None, None).
    Tries multiple patterns in order of reliability.
    """
    # Pattern 1: #### N (GSM8K standard)
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "####"
        except:
            pass

    # Pattern 2: "the answer is N" / "answer: N" / "answer is N"
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

    # Pattern 3: "= N" at end of a line
    match = re.search(r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "trailing_eq"
        except:
            pass

    # Pattern 4: last number in text (fallback)
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', '')), "last_number"
        except:
            pass

    return None, None


# === Text Quality Metrics ===
def distinct_n(text, n):
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def rep_n(text, n):
    """Compute rep-n: repetition rate."""
    return 1.0 - distinct_n(text, n)


# === Model Loading & Generation ===
def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}, dtype=bfloat16")
    return model, tokenizer


def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0"):
    """Generate with Dream-7B vanilla decoding."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)

    t0 = time.time()
    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=GEN_LEN,
        output_history=False,
        return_dict_in_generate=True,
        steps=STEPS,
        temperature=TEMPERATURE,
        alg=ALG,
    )
    elapsed = time.time() - t0

    seq = output.sequences
    prompt_len = len(prompt_ids)
    gen_ids = seq[0, prompt_len:].tolist()
    # Remove mask tokens and stop at EOS
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t in gen_ids:
        if t == MASK_TOKEN_ID:
            continue
        if t == eos_id:
            break
        clean_ids.append(t)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
    return text, elapsed


# === Main ===
def main():
    device = "cuda:0"
    print(f"=== Task 1b PILOT: Vanilla Baseline on GSM8K ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Gen length: {GEN_LEN}, Device: {device}")

    # Load GSM8K
    print(f"\nLoading GSM8K test set (first {N_SAMPLES} samples)...")
    problems = load_gsm8k(n=N_SAMPLES, seed=SEED)
    print(f"Loaded {len(problems)} problems")
    for i, p in enumerate(problems[:3]):
        print(f"  [{i}] Q: {p['question'][:80]}...")
        print(f"      Target: {p['target']}")

    # Load model
    print(f"\nLoading Dream-7B...")
    model, tokenizer = load_dream(device)

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Run generation
    results = []
    correct_count = 0
    extracted_count = 0
    total_time = 0

    print(f"\n--- Running {N_SAMPLES} GSM8K samples ---")
    for i, problem in enumerate(problems):
        prompt_text = format_gsm8k_prompt(problem)
        text, elapsed = generate_vanilla(model, tokenizer, prompt_text, device)
        total_time += elapsed

        # Extract and evaluate answer
        extracted_val, extraction_method = extract_model_answer(text)
        target = problem["target"]

        is_correct = False
        if extracted_val is not None and target is not None:
            is_correct = abs(extracted_val - target) < 1e-6
            extracted_count += 1

        if is_correct:
            correct_count += 1

        # Text quality metrics
        d1 = distinct_n(text, 1)
        d2 = distinct_n(text, 2)
        d3 = distinct_n(text, 3)
        r2 = rep_n(text, 2)
        r3 = rep_n(text, 3)
        word_count = len(text.split())

        result = {
            "idx": i,
            "question": problem["question"][:200],
            "target": target,
            "extracted_answer": extracted_val,
            "extraction_method": extraction_method,
            "is_correct": is_correct,
            "generated_text": text,
            "gen_time_s": elapsed,
            "word_count": word_count,
            "distinct_1": d1,
            "distinct_2": d2,
            "distinct_3": d3,
            "rep_2": r2,
            "rep_3": r3,
        }
        results.append(result)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  [{i+1}/{N_SAMPLES}] {status} | target={target} | extracted={extracted_val} "
              f"({extraction_method}) | time={elapsed:.1f}s | words={word_count}")
        if i < 5 or is_correct:
            # Print first 200 chars of generation
            print(f"    Text: {text[:200]}")

    # Aggregate metrics
    accuracy = correct_count / N_SAMPLES
    extraction_rate = extracted_count / N_SAMPLES
    avg_time = total_time / N_SAMPLES
    texts = [r["generated_text"] for r in results]

    all_d1 = [r["distinct_1"] for r in results]
    all_d2 = [r["distinct_2"] for r in results]
    all_d3 = [r["distinct_3"] for r in results]
    all_r2 = [r["rep_2"] for r in results]
    all_r3 = [r["rep_3"] for r in results]
    all_wc = [r["word_count"] for r in results]

    summary = {
        "task": "task_1b",
        "mode": "pilot",
        "method": "vanilla",
        "benchmark": "gsm8k",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "alg": ALG,
        "gen_len": GEN_LEN,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "extracted_count": extracted_count,
        "extraction_rate": extraction_rate,
        "total_count": N_SAMPLES,
        "avg_gen_time_s": avg_time,
        "total_time_s": total_time,
        "distinct_1_mean": float(np.mean(all_d1)),
        "distinct_2_mean": float(np.mean(all_d2)),
        "distinct_3_mean": float(np.mean(all_d3)),
        "rep_2_mean": float(np.mean(all_r2)),
        "rep_3_mean": float(np.mean(all_r3)),
        "avg_word_count": float(np.mean(all_wc)),
        "results": results,
        "torch_version": torch.__version__,
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 1b Pilot - Vanilla GSM8K Baseline")
    print(f"{'='*70}")
    print(f"  Accuracy (EM):     {correct_count}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  Extraction rate:   {extracted_count}/{N_SAMPLES} = {extraction_rate:.1%}")
    print(f"  Avg time:          {avg_time:.1f}s per sample")
    print(f"  Total time:        {total_time:.0f}s")
    print(f"  Distinct-1:        {np.mean(all_d1):.3f}")
    print(f"  Distinct-2:        {np.mean(all_d2):.3f}")
    print(f"  Distinct-3:        {np.mean(all_d3):.3f}")
    print(f"  Rep-2:             {np.mean(all_r2):.3f}")
    print(f"  Rep-3:             {np.mean(all_r3):.3f}")
    print(f"  Avg words:         {np.mean(all_wc):.0f}")

    # Per-extraction-method breakdown
    method_counts = {}
    for r in results:
        m = r["extraction_method"] or "none"
        method_counts[m] = method_counts.get(m, 0) + 1
    print(f"\n  Extraction methods: {method_counts}")

    # Pass criteria check
    print(f"\n--- Pass Criteria ---")
    gen_ok = all(len(r["generated_text"].strip()) > 0 for r in results)
    extract_ok = extraction_rate >= 0.80
    acc_ok = accuracy > 0
    print(f"  16 samples generated successfully: {'PASS' if gen_ok else 'FAIL'}")
    print(f"  Answer extraction rate >= 80%: {extraction_rate:.1%} -> {'PASS' if extract_ok else 'FAIL'}")
    print(f"  Accuracy > 0%: {accuracy:.1%} -> {'PASS' if acc_ok else 'FAIL'}")

    overall = "GO" if (gen_ok and extract_ok and acc_ok) else \
              "CONDITIONAL-GO" if gen_ok else "NO-GO"
    print(f"  Overall: {overall}")

    # Save results
    out_file = RESULTS_DIR / "task_1b_gsm8k_vanilla.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
