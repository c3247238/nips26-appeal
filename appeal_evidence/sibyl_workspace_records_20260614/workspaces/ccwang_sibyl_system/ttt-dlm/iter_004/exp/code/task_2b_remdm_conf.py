"""
Task 2b (PILOT): ReMDM-conf Baseline on Countdown (16 samples).

Implements ReMDM-conf (confidence-based remasking) on Dream-7B:
- At each denoising step, after revealing tokens, compute confidence (softmax prob)
  for ALL revealed tokens (not just newly revealed ones)
- Remask the lowest-confidence k% of revealed tokens back to [MASK]
- This allows the model to "reconsider" low-confidence decisions

Reference: Nisonoff et al. (2025). "Remasking Discrete Diffusion Models" arXiv:2503.00307

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 task_2b_remdm_conf.py
"""
import os, sys, json, time, random, re, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

# === sample_tokens (from Dream generation_utils.py) ===
def _sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    """Sample tokens from logits and return (confidence, x0)."""
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    return confidence, x0


# === Config ===
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 16
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

# ReMDM-conf hyperparameters
# In ReMDM, at each step the model reveals tokens and then re-masks a fraction
# of them based on low confidence. We remask among ALL revealed gen tokens,
# but with a schedule: only remask if confidence is below a threshold,
# and stop remasking in the final 20% of steps.
REMASK_RATIO = 0.1  # fraction of revealed tokens to remask at each step
REMASK_STOP_FRAC = 0.8  # stop remasking after this fraction of total steps


# === Countdown Problem Generator (identical to task_1a) ===
def generate_countdown_problems(n, seed=42, min_nums=3, max_nums=5):
    """Generate Countdown-style arithmetic problems with known solutions."""
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


# === Evaluation (identical to task_1a) ===
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


# === ReMDM-conf Custom Denoising Loop ===
def generate_remdm_conf(model, tokenizer, prompt_text, device="cuda:0",
                        remask_ratio=REMASK_RATIO, steps=STEPS,
                        temperature=TEMPERATURE, gen_len=GEN_LEN):
    """
    Generate with Dream-7B using ReMDM-conf (confidence-based remasking).

    Key design (faithful to ReMDM paper):
    - At each denoising step, after the standard origin sampling reveals some tokens,
      we do a FRESH forward pass to re-evaluate ALL revealed tokens' confidence.
    - We remask the lowest-confidence fraction of revealed tokens.
    - Remasking stops in the final 20% of steps to let the sequence converge.
    - The remask count is based on the SCHEDULE: at step i, the number of tokens
      that SHOULD be masked is ceil(s/t * n_gen). If fewer are masked, we remask
      the lowest-confidence revealed tokens to match the schedule.

    This is the "ReMDM-conf" variant: confidence = P(x_i | context).
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)

    max_length = prompt_len + gen_len
    mask_token_id = MASK_TOKEN_ID
    eps = 1e-3

    # Pad input to max_length with mask tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

    tok_idx = None
    attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    stop_remask_step = int(REMASK_STOP_FRAC * steps)  # stop remasking after this step

    remask_stats = {
        "total_remasked": 0,
        "remask_per_step": [],
    }

    for i in range(steps):
        mask_index = (x == mask_token_id)

        # Forward pass
        with torch.no_grad():
            logits = model(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]

        # --- Standard Dream 'origin' sampling for masked tokens ---
        p_transfer = 1 - s / t if i < steps - 1 else 1
        x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + mask_token_id
        transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
        if transfer_index_t_s.any():
            _, sampled = _sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature)
            x0[transfer_index_t_s] = sampled
        x[mask_index] = x0.clone()

        # --- ReMDM-conf: schedule-based confidence remasking ---
        # Only remask in the generation region, not in the last steps, not on final step
        if i < stop_remask_step and i < steps - 1:
            gen_region = x[0, prompt_len:]
            revealed_mask = (gen_region != mask_token_id)
            n_revealed = revealed_mask.sum().item()
            n_gen = gen_len

            # According to the noise schedule, at time s, the expected number of
            # masked tokens should be s * n_gen. Currently we have (n_gen - n_revealed)
            # masked. If too many are revealed, remask some.
            expected_masked = int(math.ceil(s.item() * n_gen))
            current_masked = n_gen - n_revealed
            n_remask = max(0, expected_masked - current_masked)

            # Cap: don't remask more than remask_ratio * n_revealed
            n_remask = min(n_remask, max(1, int(remask_ratio * n_revealed)))

            if n_remask > 0 and n_revealed > 1:
                # Fresh forward pass to get updated confidence for revealed tokens
                with torch.no_grad():
                    logits_fresh = model(x, attention_mask, tok_idx).logits
                    logits_fresh = torch.cat([logits_fresh[:, :1], logits_fresh[:, :-1]], dim=1)

                gen_logits = logits_fresh[0, prompt_len:]
                gen_probs = torch.softmax(gen_logits, dim=-1)

                revealed_positions = torch.where(revealed_mask)[0]
                revealed_tokens = gen_region[revealed_positions]
                revealed_confidence = gen_probs[revealed_positions].gather(
                    1, revealed_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Don't remask more than we have
                n_remask = min(n_remask, n_revealed - 1)  # keep at least 1 revealed

                if n_remask > 0:
                    _, low_conf_idx = torch.topk(revealed_confidence, n_remask, largest=False)
                    positions_to_remask = revealed_positions[low_conf_idx]
                    x[0, prompt_len + positions_to_remask] = mask_token_id

                    remask_stats["total_remasked"] += n_remask
                    remask_stats["remask_per_step"].append(n_remask)
                else:
                    remask_stats["remask_per_step"].append(0)
            else:
                remask_stats["remask_per_step"].append(0)
        else:
            remask_stats["remask_per_step"].append(0)

    # Extract generated text
    gen_ids = x[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t_id in gen_ids:
        if t_id == mask_token_id:
            continue
        if t_id == eos_id:
            break
        clean_ids.append(t_id)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

    return text, remask_stats


# === Main ===
def main():
    device = "cuda:0"
    print(f"=== Task 2b PILOT: ReMDM-conf Baseline on Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Remask ratio: {REMASK_RATIO}")
    print(f"Device: {device}")

    # Generate same Countdown problems as task_1a (same seed)
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
    random.seed(SEED)

    # Run generation
    results = []
    correct_count = 0
    total_time = 0

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)

        t0 = time.time()
        text, remask_stats = generate_remdm_conf(
            model, tokenizer, prompt_text, device,
            remask_ratio=REMASK_RATIO, steps=STEPS,
            temperature=TEMPERATURE, gen_len=GEN_LEN,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        # Evaluate
        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            correct_count += 1

        # Text quality
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
            "remask_stats": {
                "total_remasked": remask_stats["total_remasked"],
                "avg_remask_per_step": (
                    np.mean(remask_stats["remask_per_step"])
                    if remask_stats["remask_per_step"] else 0
                ),
            },
        }
        results.append(result)

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        print(f"  [{i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | words={word_count} | "
              f"remasked={remask_stats['total_remasked']}")
        if i < 5 or is_correct:
            print(f"    Text: {text[:200]}")

    # Aggregate
    accuracy = correct_count / N_SAMPLES
    avg_time = total_time / N_SAMPLES
    all_d1 = [r["distinct_1"] for r in results]
    all_d2 = [r["distinct_2"] for r in results]
    all_d3 = [r["distinct_3"] for r in results]
    all_r2 = [r["rep_2"] for r in results]
    all_r3 = [r["rep_3"] for r in results]
    all_wc = [r["word_count"] for r in results]
    all_remasked = [r["remask_stats"]["total_remasked"] for r in results]

    # Load vanilla baseline for comparison
    vanilla_file = RESULTS_DIR / "task_1a_countdown_vanilla.json"
    vanilla_accuracy = None
    if vanilla_file.exists():
        with open(vanilla_file) as f:
            vanilla_data = json.load(f)
            vanilla_accuracy = vanilla_data.get("accuracy")

    summary = {
        "task": "task_2b",
        "mode": "pilot",
        "method": "remdm_conf",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,
        "remask_ratio": REMASK_RATIO,
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
        "avg_total_remasked": float(np.mean(all_remasked)),
        "vanilla_accuracy": vanilla_accuracy,
        "results": results,
        "torch_version": torch.__version__,
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 2b Pilot - ReMDM-conf Countdown Baseline")
    print(f"{'='*70}")
    print(f"  Accuracy:        {correct_count}/{N_SAMPLES} = {accuracy:.1%}")
    if vanilla_accuracy is not None:
        delta = accuracy - vanilla_accuracy
        print(f"  Vanilla acc:     {vanilla_accuracy:.1%}")
        print(f"  Delta:           {delta:+.1%}")
    print(f"  Avg time:        {avg_time:.1f}s per sample")
    print(f"  Total time:      {total_time:.0f}s")
    print(f"  Avg remasked:    {np.mean(all_remasked):.0f} tokens total per sample")
    print(f"  Distinct-1:      {np.mean(all_d1):.3f}")
    print(f"  Distinct-2:      {np.mean(all_d2):.3f}")
    print(f"  Distinct-3:      {np.mean(all_d3):.3f}")
    print(f"  Rep-2:           {np.mean(all_r2):.3f}")
    print(f"  Rep-3:           {np.mean(all_r3):.3f}")
    print(f"  Avg words:       {np.mean(all_wc):.0f}")

    # Pass criteria
    print(f"\n--- Pass Criteria ---")
    run_ok = all(len(r["generated_text"].strip()) > 0 for r in results)
    div_ok = np.mean(all_d2) >= 0.7
    compare_ok = vanilla_accuracy is not None
    print(f"  ReMDM-conf runs successfully: {'PASS' if run_ok else 'FAIL'}")
    print(f"  Diversity (distinct-2) >= 0.7: {np.mean(all_d2):.3f} -> {'PASS' if div_ok else 'FAIL'}")
    print(f"  Vanilla comparison available: {'PASS' if compare_ok else 'WARN (no vanilla results)'}")
    overall = "GO" if (run_ok and div_ok) else "NO-GO"
    print(f"  Overall: {overall}")

    # Save
    out_file = RESULTS_DIR / "task_2b_remdm_conf.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
