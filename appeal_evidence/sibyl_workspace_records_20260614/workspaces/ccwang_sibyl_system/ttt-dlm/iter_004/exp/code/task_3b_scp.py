"""
Task 3b (PILOT): Self-Contradiction Probing (SCP) on Countdown (16 samples).

Implements SCP (Level 2 ablation in the information augmentation spectrum):
- At configurable intervals during denoising, for each revealed token,
  perform leave-one-out masking: mask that single token and re-predict it.
- If the model's top prediction disagrees with the current token, mark it
  as "self-contradictory".
- Re-mask all self-contradictory tokens so the model can re-predict them
  in subsequent denoising steps.

This tests whether leveraging the model's bidirectional attention for
self-verification can improve generation quality, at a cost of ~1 extra
forward pass per probing step.

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 task_3b_scp.py
"""
import os, sys, json, time, random, re, math
from pathlib import Path

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
MASK_TOKEN_ID = 151666

# SCP Hyperparameters
SCP_INTERVAL = 8       # Run SCP every N denoising steps (to control overhead)
SCP_WARMUP_FRAC = 0.25 # Don't probe until 25% of steps done (need enough revealed tokens)
SCP_STOP_FRAC = 0.85   # Stop probing in last 15% of steps (let sequence converge)
SCP_MIN_REVEALED = 8   # Minimum revealed tokens before probing
SCP_BATCH_SIZE = 32    # Process leave-one-out in batches to manage VRAM
# When checking contradiction, use top-k: if current token NOT in top-k predictions,
# it is a contradiction. k=1 means strict (must be argmax), k=3 means lenient.
SCP_TOP_K = 1


# === Token sampling (from Dream generation_utils.py) ===
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


# === Countdown Problem Generator (identical to task_1a) ===
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
    return (
        f"Use the numbers [{nums_str}] with basic arithmetic operations "
        f"(+, -, *, /) to obtain {target}. "
        f"Each number can only be used once. "
        f"Show your work step by step, then provide the final equation.\n"
        f"Format: Step 1: ... Step 2: ... Answer: <equation> = {target}"
    )


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


# ──────────────────────────────────────────────────
# SCP Core: Leave-One-Out Contradiction Detection
# ──────────────────────────────────────────────────

def detect_contradictions(model, x, prompt_len, device, top_k=SCP_TOP_K,
                          batch_size=SCP_BATCH_SIZE):
    """
    For each revealed (non-mask) token in the generation region,
    mask it out, run a forward pass, and check if the model's top-k
    predictions include the current token.

    Returns:
        contradiction_positions: list of absolute positions (in x) that are contradictory
        n_probed: total number of tokens probed
        n_contradictions: number of contradictions found
    """
    gen_region = x[0, prompt_len:]  # [gen_len]
    revealed_mask = (gen_region != MASK_TOKEN_ID)
    revealed_positions = torch.where(revealed_mask)[0]  # relative to prompt_len
    n_probed = len(revealed_positions)

    if n_probed == 0:
        return [], 0, 0

    # For efficiency, batch the leave-one-out forward passes.
    # We create copies of x where one revealed token at a time is masked.
    contradiction_positions = []

    for batch_start in range(0, n_probed, batch_size):
        batch_end = min(batch_start + batch_size, n_probed)
        batch_positions = revealed_positions[batch_start:batch_end]
        abs_positions = batch_positions + prompt_len  # absolute positions in x
        batch_n = len(batch_positions)

        # Create batch of inputs: each has one token masked
        # x shape: [1, seq_len] -> replicate to [batch_n, seq_len]
        x_batch = x.expand(batch_n, -1).clone()  # [batch_n, seq_len]

        # Mask one position per batch element
        for j, abs_pos in enumerate(abs_positions):
            x_batch[j, abs_pos] = MASK_TOKEN_ID

        # Forward pass on the batch
        with torch.no_grad():
            outputs = model(x_batch, attention_mask="full", position_ids=None)
            logits = outputs.logits
            # Dream shifted logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Check each position
        for j, abs_pos in enumerate(abs_positions):
            current_token = x[0, abs_pos].item()
            pos_logits = logits[j, abs_pos]  # [vocab_size]

            # Get top-k predictions
            topk_tokens = torch.topk(pos_logits, top_k).indices.tolist()

            if current_token not in topk_tokens:
                contradiction_positions.append(abs_pos.item())

    n_contradictions = len(contradiction_positions)
    return contradiction_positions, n_probed, n_contradictions


# ──────────────────────────────────────────────────
# SCP Generation: Custom Denoising Loop with Self-Contradiction Probing
# ──────────────────────────────────────────────────

def generate_scp(model, tokenizer, prompt_text, device="cuda:0",
                 steps=STEPS, temperature=TEMPERATURE, gen_len=GEN_LEN,
                 scp_interval=SCP_INTERVAL, warmup_frac=SCP_WARMUP_FRAC,
                 stop_frac=SCP_STOP_FRAC, min_revealed=SCP_MIN_REVEALED,
                 top_k=SCP_TOP_K):
    """
    Generate with Dream-7B using SCP (Self-Contradiction Probing).

    At every scp_interval steps (after warmup, before stop):
    1. For each revealed token, mask it and check if model predicts it back.
    2. Tokens that fail (contradiction) are re-masked.
    3. Subsequent denoising steps then re-predict these positions.

    Returns: generated text, elapsed time, SCP diagnostics
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)

    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    warmup_step = int(steps * warmup_frac)
    stop_step = int(steps * stop_frac)

    # Diagnostics
    scp_diagnostics = {
        "probe_steps": [],
        "total_probed": 0,
        "total_contradictions": 0,
        "total_remasked": 0,
    }

    t0 = time.time()

    for i in range(steps):
        mask_index = (x == MASK_TOKEN_ID)

        # === E-step: Standard Dream 'origin' denoising ===
        with torch.no_grad():
            outputs = model(x, attention_mask="full", position_ids=None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        if transfer_mask.any():
            _, sampled = _sample_tokens(mask_logits[transfer_mask], temperature=temperature)
            x0[transfer_mask] = sampled
        x[mask_index] = x0

        # === SCP: Self-Contradiction Probing ===
        # Only at specific intervals, within the active window
        if (i >= warmup_step and i < stop_step and
                i % scp_interval == 0 and i < steps - 1):
            # Check if enough tokens are revealed
            gen_region = x[0, prompt_len:]
            n_revealed = (gen_region != MASK_TOKEN_ID).sum().item()

            if n_revealed >= min_revealed:
                contradiction_positions, n_probed, n_contradictions = detect_contradictions(
                    model, x, prompt_len, device, top_k=top_k
                )

                # Re-mask contradictory tokens
                n_remasked = 0
                if contradiction_positions:
                    for pos in contradiction_positions:
                        x[0, pos] = MASK_TOKEN_ID
                    n_remasked = len(contradiction_positions)

                scp_diagnostics["probe_steps"].append({
                    "step": i,
                    "n_revealed": n_revealed,
                    "n_probed": n_probed,
                    "n_contradictions": n_contradictions,
                    "n_remasked": n_remasked,
                    "contradiction_rate": n_contradictions / n_probed if n_probed > 0 else 0,
                })
                scp_diagnostics["total_probed"] += n_probed
                scp_diagnostics["total_contradictions"] += n_contradictions
                scp_diagnostics["total_remasked"] += n_remasked

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

    return text, elapsed, scp_diagnostics


# === Vanilla generation for comparison ===
def vanilla_generate(model, tokenizer, prompt_text, device="cuda:0"):
    """Vanilla generation (no SCP) using same custom loop for fair comparison."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)
    max_length = prompt_len + GEN_LEN
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            outputs = model(x, attention_mask="full", position_ids=None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_logits = logits[mask_index]

            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if transfer_mask.any():
                _, sampled = _sample_tokens(mask_logits[transfer_mask], temperature=TEMPERATURE)
                x0[transfer_mask] = sampled
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
# Main
# ──────────────────────────────────────────────────

def main():
    device = "cuda:0"
    print(f"=== Task 3b PILOT: SCP (Self-Contradiction Probing) on Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"SCP: interval={SCP_INTERVAL}, warmup={SCP_WARMUP_FRAC}, "
          f"stop={SCP_STOP_FRAC}, min_revealed={SCP_MIN_REVEALED}, top_k={SCP_TOP_K}")
    print(f"Device: {device}")

    # Generate problems (same seed as all other tasks)
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"\nGenerated {len(problems)} Countdown problems")

    # Load model
    model, tokenizer = load_dream(device)

    # === Phase 1: Vanilla baseline ===
    print(f"\n{'='*60}")
    print(f"Phase 1: Vanilla Baseline (for direct comparison)")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    vanilla_results = []
    vanilla_correct = 0

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)
        text, elapsed = vanilla_generate(model, tokenizer, prompt_text, device)

        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            vanilla_correct += 1

        vanilla_results.append({
            "idx": i,
            "problem": problem,
            "generated_text": text,
            "verification": verification,
            "is_correct": is_correct,
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "distinct_1": distinct_n(text, 1),
            "distinct_2": distinct_n(text, 2),
            "distinct_3": distinct_n(text, 3),
            "rep_2": rep_n(text, 2),
            "rep_3": rep_n(text, 3),
        })

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        print(f"  [V {i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | time={elapsed:.1f}s")

    vanilla_acc = vanilla_correct / N_SAMPLES
    print(f"\nVanilla: {vanilla_correct}/{N_SAMPLES} = {vanilla_acc:.1%}")

    # === Phase 2: SCP generation ===
    print(f"\n{'='*60}")
    print(f"Phase 2: SCP Generation")
    print(f"{'='*60}")

    # Re-seed for fair comparison
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    scp_results = []
    scp_correct = 0

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)

        text, elapsed, scp_diag = generate_scp(
            model, tokenizer, prompt_text, device,
            steps=STEPS, temperature=TEMPERATURE, gen_len=GEN_LEN,
        )

        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            scp_correct += 1

        # Aggregate SCP stats for this sample
        probe_steps = scp_diag["probe_steps"]
        n_probes = len(probe_steps)
        avg_contradiction_rate = (
            float(np.mean([s["contradiction_rate"] for s in probe_steps]))
            if probe_steps else 0.0
        )

        scp_results.append({
            "idx": i,
            "problem": problem,
            "generated_text": text,
            "verification": verification,
            "is_correct": is_correct,
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "distinct_1": distinct_n(text, 1),
            "distinct_2": distinct_n(text, 2),
            "distinct_3": distinct_n(text, 3),
            "rep_2": rep_n(text, 2),
            "rep_3": rep_n(text, 3),
            "scp_stats": {
                "n_probe_steps": n_probes,
                "total_probed": scp_diag["total_probed"],
                "total_contradictions": scp_diag["total_contradictions"],
                "total_remasked": scp_diag["total_remasked"],
                "avg_contradiction_rate": avg_contradiction_rate,
                "probe_details": probe_steps,
            },
        })

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        print(f"  [S {i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | probes={n_probes} | "
              f"contradictions={scp_diag['total_contradictions']}/{scp_diag['total_probed']} "
              f"({avg_contradiction_rate:.1%}) | remasked={scp_diag['total_remasked']}")

    scp_acc = scp_correct / N_SAMPLES

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 3b Pilot - SCP vs Vanilla on Countdown")
    print(f"{'='*70}")
    print(f"  Vanilla:  {vanilla_correct}/{N_SAMPLES} = {vanilla_acc:.1%}")
    print(f"  SCP:      {scp_correct}/{N_SAMPLES} = {scp_acc:.1%}")
    print(f"  Delta:    {scp_acc - vanilla_acc:+.1%}")

    # Vanilla metrics
    v_d2 = float(np.mean([r["distinct_2"] for r in vanilla_results]))
    v_r3 = float(np.mean([r["rep_3"] for r in vanilla_results]))
    v_time = float(np.mean([r["gen_time_s"] for r in vanilla_results]))

    # SCP metrics
    s_d2 = float(np.mean([r["distinct_2"] for r in scp_results]))
    s_r3 = float(np.mean([r["rep_3"] for r in scp_results]))
    s_time = float(np.mean([r["gen_time_s"] for r in scp_results]))

    # SCP-specific stats
    all_contradiction_rates = [
        r["scp_stats"]["avg_contradiction_rate"] for r in scp_results
    ]
    overall_contradiction_rate = float(np.mean(all_contradiction_rates)) if all_contradiction_rates else 0
    total_remasked_all = sum(r["scp_stats"]["total_remasked"] for r in scp_results)
    total_probed_all = sum(r["scp_stats"]["total_probed"] for r in scp_results)

    print(f"\n  Vanilla: distinct-2={v_d2:.3f}, rep-3={v_r3:.3f}, time={v_time:.1f}s")
    print(f"  SCP:     distinct-2={s_d2:.3f}, rep-3={s_r3:.3f}, time={s_time:.1f}s")
    print(f"  Time overhead: {s_time / v_time:.2f}x")
    print(f"\n  SCP Contradiction Stats:")
    print(f"    Overall contradiction rate: {overall_contradiction_rate:.1%}")
    print(f"    Total tokens probed:        {total_probed_all}")
    print(f"    Total contradictions found:  {sum(r['scp_stats']['total_contradictions'] for r in scp_results)}")
    print(f"    Total tokens remasked:       {total_remasked_all}")

    # Pass criteria
    print(f"\n--- Pass Criteria ---")
    run_ok = all(len(r["generated_text"].strip()) > 0 for r in scp_results)
    # Contradiction rate should be in 5%-50% (not all or none)
    rate_ok = 0.05 <= overall_contradiction_rate <= 0.50
    acc_ok = scp_acc >= vanilla_acc
    d2_ok = s_d2 >= 0.7

    print(f"  SCP runs without errors:              {'PASS' if run_ok else 'FAIL'}")
    print(f"  Contradiction rate in [5%, 50%]:       {'PASS' if rate_ok else 'FAIL'} ({overall_contradiction_rate:.1%})")
    print(f"  SCP acc >= vanilla ({vanilla_acc:.1%}):       {'PASS' if acc_ok else 'FAIL'} ({scp_acc:.1%})")
    print(f"  distinct-2 >= 0.7:                    {'PASS' if d2_ok else 'FAIL'} ({s_d2:.3f})")

    if run_ok and rate_ok and acc_ok and d2_ok:
        overall = "GO"
    elif run_ok:
        issues = []
        if not rate_ok:
            issues.append(f"contradiction rate {overall_contradiction_rate:.1%} outside [5%,50%]")
        if not acc_ok:
            issues.append(f"acc {scp_acc:.1%} < vanilla {vanilla_acc:.1%}")
        if not d2_ok:
            issues.append(f"distinct-2 {s_d2:.3f} < 0.7")
        overall = f"CONDITIONAL-GO ({'; '.join(issues)})"
    else:
        overall = "NO-GO"
    print(f"  Overall: {overall}")

    # === Save results ===
    summary = {
        "task": "task_3b",
        "mode": "pilot",
        "method": "scp",
        "method_description": "Self-Contradiction Probing: leave-one-out verification + re-masking",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,
        "scp_config": {
            "interval": SCP_INTERVAL,
            "warmup_frac": SCP_WARMUP_FRAC,
            "stop_frac": SCP_STOP_FRAC,
            "min_revealed": SCP_MIN_REVEALED,
            "top_k": SCP_TOP_K,
            "batch_size": SCP_BATCH_SIZE,
        },
        "vanilla_accuracy": vanilla_acc,
        "vanilla_correct": vanilla_correct,
        "scp_accuracy": scp_acc,
        "scp_correct": scp_correct,
        "total_count": N_SAMPLES,
        "delta_accuracy": scp_acc - vanilla_acc,
        "vanilla_distinct_2_mean": v_d2,
        "scp_distinct_2_mean": s_d2,
        "vanilla_rep_3_mean": v_r3,
        "scp_rep_3_mean": s_r3,
        "vanilla_avg_time_s": v_time,
        "scp_avg_time_s": s_time,
        "time_overhead": s_time / v_time if v_time > 0 else 0,
        "overall_contradiction_rate": overall_contradiction_rate,
        "total_probed": total_probed_all,
        "total_contradictions": sum(r["scp_stats"]["total_contradictions"] for r in scp_results),
        "total_remasked": total_remasked_all,
        "pass_criteria": {
            "no_error": run_ok,
            "contradiction_rate_ok": rate_ok,
            "acc_ok": acc_ok,
            "d2_ok": d2_ok,
            "overall": overall,
        },
        "vanilla_results": vanilla_results,
        "scp_results": scp_results,
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / "task_3b_scp.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
