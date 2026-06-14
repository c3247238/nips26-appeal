"""
Task 2c (PILOT): RCR (Running Confidence Remasking) Baseline on Countdown (16 samples).

RCR Algorithm:
  After each denoising step, compute confidence for all revealed tokens.
  Maintain a running average of confidence scores across steps.
  Re-mask tokens whose confidence falls below a fraction of the running average.

Uses Dream's native `diffusion_generate` with `generation_tokens_hook_func`
to inject remasking logic while keeping base denoising identical to vanilla.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 task_2c_rcr_baseline.py
"""
import os, sys, json, time, random, re, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

# === Config ===
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 16
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
ALG = "origin"
MASK_TOKEN_ID = 151666

# RCR hyperparameters
RCR_EMA_DECAY = 0.9       # exponential moving average decay for running confidence
RCR_WARMUP_FRAC = 0.3     # don't remask during first 30% of steps
RCR_MIN_REVEALED = 0.15   # don't remask if fewer than 15% tokens revealed
RCR_THRESHOLD_SCALE = 0.7 # remask tokens below threshold_scale * running_avg
RCR_REMASK_CAP = 0.15     # cap remasking at 15% of revealed tokens per step

# === Countdown Problem Generator (same as task_1a) ===
def generate_countdown_problems(n, seed=42, min_nums=3, max_nums=5):
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


def format_countdown_prompt(problem):
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


# === Evaluation (same as task_1a) ===
def verify_countdown_answer(text, problem):
    target = problem["target"]
    numbers = problem["numbers"]
    answer_match = re.search(r'Answer:\s*(.+)', text, re.IGNORECASE)
    if answer_match:
        equation_str = answer_match.group(1).strip()
    else:
        eq_matches = re.findall(r'([\d\s+\-*/()]+)\s*=\s*(\d+)', text)
        if eq_matches:
            equation_str = eq_matches[-1][0].strip()
        else:
            return {"is_correct": False, "extracted_equation": None,
                    "explanation": "No equation found in output"}
    equation_str = re.sub(r'\s*=\s*\d+\s*$', '', equation_str).strip()
    if not equation_str:
        return {"is_correct": False, "extracted_equation": None,
                "explanation": "Empty equation after cleanup"}
    try:
        safe_eq = re.sub(r'[^\d+\-*/(). ]', '', equation_str)
        if not safe_eq:
            return {"is_correct": False, "extracted_equation": equation_str,
                    "explanation": "Equation contains invalid characters"}
        result = eval(safe_eq)
        is_correct = abs(result - target) < 1e-6
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
            "explanation": f"eval({safe_eq}) = {result}, target = {target}, numbers_valid = {numbers_valid}"
        }
    except Exception as e:
        return {"is_correct": False, "extracted_equation": equation_str,
                "explanation": f"Eval error: {e}"}


# === Text Quality Metrics ===
def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def rep_n(text, n):
    return 1.0 - distinct_n(text, n)


# === Model Loading ===
def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}, dtype=bfloat16")
    return model, tokenizer


# === RCR Hook Factory ===
class RCRHook:
    """
    Running Confidence Remasking hook for Dream's diffusion_generate.

    Injected via generation_tokens_hook_func. After each step, checks
    confidence of revealed tokens and remasks those below a running average.
    """

    def __init__(self, prompt_len, gen_len, steps, mask_token_id, temperature):
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.steps = steps
        self.mask_token_id = mask_token_id
        self.temperature = temperature

        self.warmup_steps = int(steps * RCR_WARMUP_FRAC)
        self.running_confidence = None
        self.step_count = 0

        # Stats tracking
        self.remask_counts = []
        self.running_conf_history = []
        self.revealed_frac_history = []

    def __call__(self, step, x, logits):
        """
        Called after each denoising step.
        step: int or None (None = initial call before first step)
        x: [B, seq_len] current token sequence
        logits: [B, seq_len, vocab_size] or None
        """
        if step is None:
            # Initial call, no remasking
            return x

        self.step_count += 1

        # Don't remask during warmup or last step
        if step < self.warmup_steps or step >= self.steps - 1:
            self.remask_counts.append(0)
            return x

        # Don't remask if logits not available
        if logits is None:
            self.remask_counts.append(0)
            return x

        gen_start = self.prompt_len
        gen_end = min(self.prompt_len + self.gen_len, x.shape[1])

        # Get generation region
        gen_tokens = x[0, gen_start:gen_end]
        revealed_mask = (gen_tokens != self.mask_token_id)
        n_revealed = revealed_mask.sum().item()
        n_total = gen_end - gen_start
        revealed_frac = n_revealed / n_total if n_total > 0 else 0

        self.revealed_frac_history.append(revealed_frac)

        if n_revealed < 2 or revealed_frac < RCR_MIN_REVEALED:
            self.remask_counts.append(0)
            return x

        # Compute confidence for revealed tokens
        gen_logits = logits[0, gen_start:gen_end]  # [gen_len, vocab]
        gen_probs = torch.softmax(gen_logits / self.temperature, dim=-1)

        # Confidence = P(current_token) for each position
        token_confidence = torch.gather(
            gen_probs, -1, gen_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [gen_len]

        revealed_conf = token_confidence[revealed_mask]
        mean_conf = revealed_conf.mean().item()

        # Update EMA
        if self.running_confidence is None:
            self.running_confidence = mean_conf
        else:
            self.running_confidence = (
                RCR_EMA_DECAY * self.running_confidence +
                (1 - RCR_EMA_DECAY) * mean_conf
            )
        self.running_conf_history.append(self.running_confidence)

        # Find tokens below threshold
        threshold = self.running_confidence * RCR_THRESHOLD_SCALE
        below_threshold = revealed_mask & (token_confidence < threshold)
        n_remask = below_threshold.sum().item()

        if n_remask > 0:
            # Cap remasking
            max_remask = max(1, int(n_revealed * RCR_REMASK_CAP))
            if n_remask > max_remask:
                # Only remask lowest confidence ones
                conf_for_sort = token_confidence.clone()
                conf_for_sort[~below_threshold] = float('inf')
                _, remask_indices = torch.topk(conf_for_sort, max_remask, largest=False)
                new_below = torch.zeros_like(below_threshold)
                new_below[remask_indices] = True
                below_threshold = new_below
                n_remask = max_remask

            # Apply remasking
            x[0, gen_start:gen_end][below_threshold] = self.mask_token_id

        self.remask_counts.append(n_remask)
        return x

    def get_stats(self):
        return {
            "remask_counts": self.remask_counts,
            "running_conf_history": self.running_conf_history,
            "revealed_frac_history": self.revealed_frac_history,
            "total_remasked": sum(self.remask_counts),
        }


# === RCR Generation ===
def generate_rcr(model, tokenizer, prompt_text, device="cuda:0"):
    """Generate with Dream-7B + RCR using native diffusion_generate hooks."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    # Create RCR hook
    rcr_hook = RCRHook(
        prompt_len=prompt_len,
        gen_len=GEN_LEN,
        steps=STEPS,
        mask_token_id=MASK_TOKEN_ID,
        temperature=TEMPERATURE,
    )

    t0 = time.time()
    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=GEN_LEN,
        output_history=False,
        return_dict_in_generate=True,
        steps=STEPS,
        temperature=TEMPERATURE,
        alg=ALG,
        generation_tokens_hook_func=rcr_hook,
    )
    elapsed = time.time() - t0

    # Decode
    seq = output.sequences
    gen_ids = seq[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t_id in gen_ids:
        if t_id == MASK_TOKEN_ID:
            continue
        if t_id == eos_id:
            break
        clean_ids.append(t_id)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

    return text, elapsed, rcr_hook.get_stats()


# === Main ===
def main():
    device = "cuda:0"
    print(f"=== Task 2c PILOT: RCR Baseline on Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"RCR Config: EMA_decay={RCR_EMA_DECAY}, warmup_frac={RCR_WARMUP_FRAC}, "
          f"min_revealed={RCR_MIN_REVEALED}, threshold_scale={RCR_THRESHOLD_SCALE}, "
          f"remask_cap={RCR_REMASK_CAP}")
    print(f"Device: {device}")

    # Generate problems (same seed as task_1a for comparability)
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
        text, elapsed, rcr_stats = generate_rcr(model, tokenizer, prompt_text, device)
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
            "total_remasked": rcr_stats["total_remasked"],
            "avg_running_conf": (
                float(np.mean(rcr_stats["running_conf_history"]))
                if rcr_stats["running_conf_history"] else 0.0
            ),
        }
        results.append(result)

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        total_remask = rcr_stats["total_remasked"]
        print(f"  [{i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | words={word_count} | remasked={total_remask}")
        if i < 5 or is_correct:
            print(f"    Text: {text[:200]}")

    # Aggregate metrics
    accuracy = correct_count / N_SAMPLES
    avg_time = total_time / N_SAMPLES
    all_d1 = [r["distinct_1"] for r in results]
    all_d2 = [r["distinct_2"] for r in results]
    all_d3 = [r["distinct_3"] for r in results]
    all_r2 = [r["rep_2"] for r in results]
    all_r3 = [r["rep_3"] for r in results]
    all_wc = [r["word_count"] for r in results]
    all_total_remask = [r["total_remasked"] for r in results]

    # Load vanilla baseline for comparison
    vanilla_path = RESULTS_DIR / "task_1a_countdown_vanilla.json"
    vanilla_acc = None
    if vanilla_path.exists():
        with open(vanilla_path) as f:
            vanilla_data = json.load(f)
        vanilla_acc = vanilla_data.get("accuracy", None)

    summary = {
        "task": "task_2c",
        "mode": "pilot",
        "method": "rcr",
        "method_description": "Running Confidence Remasking via generation_tokens_hook_func",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "alg": ALG,
        "gen_len": GEN_LEN,
        "rcr_config": {
            "ema_decay": RCR_EMA_DECAY,
            "warmup_frac": RCR_WARMUP_FRAC,
            "min_revealed_frac": RCR_MIN_REVEALED,
            "threshold_scale": RCR_THRESHOLD_SCALE,
            "remask_cap": RCR_REMASK_CAP,
        },
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
        "avg_total_remasked": float(np.mean(all_total_remask)),
        "vanilla_pilot_accuracy": vanilla_acc,
        "results": results,
        "torch_version": torch.__version__,
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 2c Pilot - RCR Countdown Baseline")
    print(f"{'='*70}")
    print(f"  Accuracy:     {correct_count}/{N_SAMPLES} = {accuracy:.1%}")
    if vanilla_acc is not None:
        print(f"  Vanilla acc:  {vanilla_acc:.1%} (from task_1a pilot)")
        diff = accuracy - vanilla_acc
        print(f"  Delta:        {diff:+.1%}")
    print(f"  Avg time:     {avg_time:.1f}s per sample")
    print(f"  Total time:   {total_time:.0f}s")
    print(f"  Distinct-1:   {np.mean(all_d1):.3f}")
    print(f"  Distinct-2:   {np.mean(all_d2):.3f}")
    print(f"  Distinct-3:   {np.mean(all_d3):.3f}")
    print(f"  Rep-2:        {np.mean(all_r2):.3f}")
    print(f"  Rep-3:        {np.mean(all_r3):.3f}")
    print(f"  Avg words:    {np.mean(all_wc):.0f}")
    print(f"  Avg remasked: {np.mean(all_total_remask):.0f} tokens total per sample")

    # Pass criteria check
    print(f"\n--- Pass Criteria ---")
    gen_ok = all(len(r["generated_text"].strip()) > 0 for r in results)
    d2_ok = np.mean(all_d2) >= 0.7
    print(f"  RCR runs successfully: {'PASS' if gen_ok else 'FAIL'}")
    print(f"  Diversity (distinct-2) >= 0.7: {np.mean(all_d2):.3f} -> {'PASS' if d2_ok else 'FAIL'}")
    overall = "GO" if (gen_ok and d2_ok) else "CONDITIONAL-GO" if gen_ok else "NO-GO"
    print(f"  Overall: {overall}")

    # Save results
    out_file = RESULTS_DIR / "task_2c_rcr_baseline.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
