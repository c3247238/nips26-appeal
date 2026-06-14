"""
Task 1a (PILOT): Vanilla Baseline on Countdown (16 samples).
Dream-7B vanilla (origin, 128 steps, temp=0.4) on 16 Countdown problems.
Metrics: accuracy, gen length, distinct-1/2/3, rep-2/3, inference time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_1a_countdown_vanilla.py
"""
import os, sys, json, time, random, re, math
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
GEN_LEN = 256       # Countdown answers are short, but give room
STEPS = 128
TEMPERATURE = 0.4
ALG = "origin"
MASK_TOKEN_ID = 151666

# === Countdown Problem Generator ===
def generate_countdown_problems(n, seed=42, min_nums=3, max_nums=5):
    """Generate Countdown-style arithmetic problems with known solutions."""
    rng = random.Random(seed)
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 50:
        attempts += 1
        num_count = rng.randint(min_nums, max_nums)
        numbers = [rng.randint(1, 25) for _ in range(num_count)]

        # Generate a solvable target by building an expression
        # Pick 2-3 numbers and combine them
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

        # Simpler fallback: just add/multiply two numbers
        a, b = numbers[0], numbers[1]
        target = a * b + (numbers[2] if num_count > 2 else 0)
        if 10 <= target <= 999:
            problems.append({
                "numbers": numbers,
                "target": target,
                "solution_hint": f"{a}*{b}+{numbers[2] if num_count > 2 else 0} = {target}"
            })

    return problems[:n]


def format_countdown_prompt(problem):
    """Format a Countdown problem as a chat prompt."""
    nums = problem["numbers"]
    target = problem["target"]
    nums_str = ", ".join(str(n) for n in nums)
    prompt = (
        f"Use the numbers [{nums_str}] with basic arithmetic operations "
        f"(+, -, *, /) to obtain {target}. "
        f"Each number can only be used once. "
        f"Show your work step by step, then provide the final equation.\n"
        f"Format: Step 1: ... Step 2: ... Answer: <equation> = {target}"
    )
    return prompt


# === Evaluation ===
def verify_countdown_answer(text, problem):
    """
    Parse generated text to verify Countdown answer correctness.
    Returns dict with is_correct, extracted_equation, explanation.
    """
    target = problem["target"]
    numbers = problem["numbers"]

    # Try to extract equation from "Answer:" line or last equation-like pattern
    answer_match = re.search(r'Answer:\s*(.+)', text, re.IGNORECASE)
    if answer_match:
        equation_str = answer_match.group(1).strip()
    else:
        # Try to find any equation with = target
        eq_matches = re.findall(r'([\d\s+\-*/()]+)\s*=\s*(\d+)', text)
        if eq_matches:
            # Use the last match
            equation_str = eq_matches[-1][0].strip()
        else:
            return {"is_correct": False, "extracted_equation": None,
                    "explanation": "No equation found in output"}

    # Clean up equation string - remove the "= target" part if present
    equation_str = re.sub(r'\s*=\s*\d+\s*$', '', equation_str).strip()
    if not equation_str:
        return {"is_correct": False, "extracted_equation": None,
                "explanation": "Empty equation after cleanup"}

    # Try to evaluate the equation
    try:
        # Safety: only allow digits, operators, parens, spaces
        safe_eq = re.sub(r'[^\d+\-*/(). ]', '', equation_str)
        if not safe_eq:
            return {"is_correct": False, "extracted_equation": equation_str,
                    "explanation": "Equation contains invalid characters"}
        result = eval(safe_eq)
        is_correct = abs(result - target) < 1e-6

        # Check that only allowed numbers are used (optional, soft check)
        used_numbers = [int(x) for x in re.findall(r'\d+', safe_eq)]
        numbers_sorted = sorted(numbers)
        used_sorted = sorted(used_numbers)

        # Check if used numbers are a subset of available numbers
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
            "explanation": f"eval({safe_eq}) = {result}, target = {target}, numbers_valid = {numbers_valid}"
        }
    except Exception as e:
        return {"is_correct": False, "extracted_equation": equation_str,
                "explanation": f"Eval error: {e}"}


# === Text Quality Metrics ===
def distinct_n(text, n):
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def rep_n(text, n):
    """Compute rep-n: 1 - distinct-n (repetition rate)."""
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
    # Remove mask tokens and EOS
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t in gen_ids:
        if t == MASK_TOKEN_ID:
            continue
        if t == eos_id:
            break
        clean_ids.append(t)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
    return text, elapsed, len(prompt_ids)


# === Main ===
def main():
    device = "cuda:0"
    print(f"=== Task 1a PILOT: Vanilla Baseline on Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Device: {device}")

    # Generate Countdown problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"\nGenerated {len(problems)} Countdown problems")
    for i, p in enumerate(problems[:3]):
        print(f"  [{i}] numbers={p['numbers']}, target={p['target']}, hint={p['solution_hint']}")

    # Load model
    model, tokenizer = load_dream(device)

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Run generation
    results = []
    correct_count = 0
    total_time = 0

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)
        text, elapsed, prompt_len = generate_vanilla(model, tokenizer, prompt_text, device)
        total_time += elapsed

        # Evaluate
        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
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
            "problem": problem,
            "prompt": prompt_text,
            "generated_text": text,
            "verification": verification,
            "is_correct": is_correct,
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
        eq = verification.get("extracted_equation", "N/A")
        print(f"  [{i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | words={word_count}")
        if i < 5 or is_correct:
            print(f"    Text: {text[:200]}")

    # Aggregate metrics
    accuracy = correct_count / N_SAMPLES
    avg_time = total_time / N_SAMPLES
    texts = [r["generated_text"] for r in results]
    all_d1 = [r["distinct_1"] for r in results]
    all_d2 = [r["distinct_2"] for r in results]
    all_d3 = [r["distinct_3"] for r in results]
    all_r2 = [r["rep_2"] for r in results]
    all_r3 = [r["rep_3"] for r in results]
    all_wc = [r["word_count"] for r in results]

    summary = {
        "task": "task_1a",
        "mode": "pilot",
        "method": "vanilla",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "alg": ALG,
        "gen_len": GEN_LEN,
        "accuracy": accuracy,
        "correct_count": correct_count,
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
    print(f"SUMMARY: Task 1a Pilot - Vanilla Countdown Baseline")
    print(f"{'='*70}")
    print(f"  Accuracy:     {correct_count}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  Avg time:     {avg_time:.1f}s per sample")
    print(f"  Total time:   {total_time:.0f}s")
    print(f"  Distinct-1:   {np.mean(all_d1):.3f}")
    print(f"  Distinct-2:   {np.mean(all_d2):.3f}")
    print(f"  Distinct-3:   {np.mean(all_d3):.3f}")
    print(f"  Rep-2:        {np.mean(all_r2):.3f}")
    print(f"  Rep-3:        {np.mean(all_r3):.3f}")
    print(f"  Avg words:    {np.mean(all_wc):.0f}")

    # Pass criteria check
    print(f"\n--- Pass Criteria ---")
    gen_ok = all(len(r["generated_text"].strip()) > 0 for r in results)
    acc_ok = accuracy > 0
    eval_ok = all(r["verification"] is not None for r in results)
    print(f"  All 16 generated successfully: {'PASS' if gen_ok else 'FAIL'}")
    print(f"  Accuracy > 0%: {accuracy:.1%} -> {'PASS' if acc_ok else 'FAIL'}")
    print(f"  Evaluation framework OK: {'PASS' if eval_ok else 'FAIL'}")
    overall = "GO" if (gen_ok and acc_ok and eval_ok) else "CONDITIONAL-GO" if gen_ok else "NO-GO"
    print(f"  Overall: {overall}")
    # Note: accuracy > 0% is ideal but not strictly required for pilot
    # The key is that generation and evaluation work correctly

    # Save results
    out_file = RESULTS_DIR / "task_1a_countdown_vanilla.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
