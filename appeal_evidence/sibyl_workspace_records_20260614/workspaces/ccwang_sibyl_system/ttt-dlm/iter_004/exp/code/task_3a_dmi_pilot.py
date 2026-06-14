"""
Task 3a (PILOT): DMI (Diffusion Memory Injection) on Countdown (16 samples).

DMI = Level 1 ablation in the information augmentation spectrum.
At each denoising step, take the previous step's logits, apply softmax to get
soft token distributions, compute weighted embedding, and mix into the current
step's input embeddings with coefficient alpha=0.3.

This is essentially "free" (no extra forward/backward pass), just an embedding
interpolation before each forward pass.

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 task_3a_dmi_pilot.py
"""
import os, sys, json, time, random, re, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
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

# DMI Hyperparams
DMI_ALPHA = 0.3   # Mixing coefficient: x_emb = (1-alpha)*orig_emb + alpha*soft_emb


# ──────────────────────────────────────────────────
# Countdown Problem Generator (same as task_1a)
# ──────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────
# Text Quality Metrics
# ──────────────────────────────────────────────────

def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0

def rep_n(text, n):
    return 1.0 - distinct_n(text, n)


# ──────────────────────────────────────────────────
# Sampling helper
# ──────────────────────────────────────────────────

def sample_tokens_from_logits(logits, temperature=1.0):
    """Sample tokens from logits with temperature."""
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return confidence, sampled


# ──────────────────────────────────────────────────
# DMI Generation: Denoising with Diffusion Memory Injection
# ──────────────────────────────────────────────────

def dmi_generate(model, tokenizer, prompt_text, device="cuda:0",
                 embedding_layer=None, gen_len=GEN_LEN, steps=STEPS,
                 temperature=TEMPERATURE, alpha=DMI_ALPHA):
    """
    Generate with DMI: at each denoising step, mix previous step's soft
    token embeddings into the current input embeddings.

    DMI mechanism:
    1. At step t, we have logits from step t-1 (prev_logits)
    2. Compute soft embeddings: soft_emb = softmax(prev_logits / tau) @ E
       where E is the token embedding matrix
    3. For MASKED positions only, replace input embedding:
       x_emb[masked] = (1-alpha)*embed(x[masked]) + alpha*soft_emb[masked]
    4. Run forward pass with modified embeddings

    This injects "memory" from the previous step's predictions into the
    current step, breaking the information island between steps.

    Returns: generated text, elapsed time, per-step diagnostics
    """
    # Prepare input
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    max_length = prompt_len + gen_len
    eps = 1e-3

    # Pad with mask tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    # Previous step's logits (None for first step)
    prev_logits = None
    prev_logits_shifted = None

    # Diagnostics
    step_diagnostics = []
    t0 = time.time()

    # Temperature for softmax over prev logits (sharper = more like hard tokens)
    soft_tau = 0.5

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()

            # === DMI: Inject previous step's soft embeddings ===
            if prev_logits_shifted is not None and alpha > 0:
                # Compute soft token distribution from previous logits
                # prev_logits_shifted: [1, seq_len, vocab_size]
                soft_probs = F.softmax(prev_logits_shifted / soft_tau, dim=-1)  # [1, seq_len, V]

                # Compute soft embeddings: weighted sum of token embeddings
                # E: [V, D], soft_probs: [1, seq_len, V]
                # soft_emb = soft_probs @ E -> [1, seq_len, D]
                emb_weight = embedding_layer.weight  # [V, D]
                soft_emb = torch.matmul(soft_probs, emb_weight.to(soft_probs.dtype))  # [1, seq_len, D]

                # Get hard embeddings for current tokens
                hard_emb = embedding_layer(x)  # [1, seq_len, D]

                # Mix: only for MASKED positions (non-masked tokens keep their hard embeddings)
                mixed_emb = hard_emb.clone()
                # For masked positions, interpolate between hard (mask token) embedding and soft embedding
                mixed_emb[mask_index] = (
                    (1 - alpha) * hard_emb[mask_index] +
                    alpha * soft_emb[mask_index]
                )

                # Forward with modified embeddings
                # Dream model expects inputs_embeds when not using input_ids
                outputs = model(
                    inputs_embeds=mixed_emb,
                    attention_mask="full",
                    position_ids=None,
                )
                dmi_applied = True
            else:
                # Standard forward (first step or alpha=0)
                outputs = model(x, attention_mask="full", position_ids=None)
                dmi_applied = False

            logits = outputs.logits
            # Dream uses shifted logits
            logits_shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Store for next step's DMI
            prev_logits_shifted = logits_shifted.clone()

            # Sample from masked positions
            mask_logits = logits_shifted[mask_index]

            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled_tokens = sample_tokens_from_logits(
                mask_logits[transfer_mask], temperature=temperature
            )
            x0[transfer_mask] = sampled_tokens

            # Update x: reveal new tokens
            x_old = x.clone()
            x[mask_index] = x0

            # Count changes
            newly_revealed = ((x_old == MASK_TOKEN_ID) & (x != MASK_TOKEN_ID))
            n_revealed = newly_revealed.sum().item()
            revealed_mask = (x[0, prompt_len:] != MASK_TOKEN_ID)
            total_revealed = revealed_mask.sum().item()

            step_diagnostics.append({
                "step": i,
                "n_masked": n_masked,
                "n_revealed_this_step": n_revealed,
                "total_revealed": total_revealed,
                "dmi_applied": dmi_applied,
            })

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

    return text, elapsed, step_diagnostics


# ──────────────────────────────────────────────────
# Vanilla Generation (same custom loop for fair comparison)
# ──────────────────────────────────────────────────

def vanilla_generate(model, tokenizer, prompt_text, device="cuda:0"):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
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
            _, sampled_tokens = sample_tokens_from_logits(
                mask_logits[transfer_mask], temperature=TEMPERATURE
            )
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
# Main
# ──────────────────────────────────────────────────

def main():
    device = "cuda:0"
    print(f"=== Task 3a PILOT: DMI (Diffusion Memory Injection) on Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"DMI alpha: {DMI_ALPHA}")
    print(f"Device: {device}")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"\nGenerated {len(problems)} Countdown problems")

    # Load model
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}")

    # Get embedding layer for DMI
    # Dream model uses model.model.embed_tokens
    embedding_layer = model.model.embed_tokens
    print(f"Embedding layer: {embedding_layer.weight.shape}")

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Phase 1: Vanilla baseline (same seed, same loop) ──
    print(f"\n{'='*60}")
    print(f"Phase 1: Vanilla Baseline (for direct comparison)")
    print(f"{'='*60}")

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

    # ── Phase 2: DMI generation ──
    print(f"\n{'='*60}")
    print(f"Phase 2: DMI Generation (alpha={DMI_ALPHA})")
    print(f"{'='*60}")

    # Re-seed for fair comparison
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    dmi_results = []
    dmi_correct = 0

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)

        text, elapsed, step_diag = dmi_generate(
            model, tokenizer, prompt_text, device,
            embedding_layer=embedding_layer,
        )

        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            dmi_correct += 1

        n_dmi_steps = sum(1 for s in step_diag if s["dmi_applied"])

        dmi_results.append({
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
            "n_dmi_steps": n_dmi_steps,
        })

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        print(f"  [DMI {i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | dmi_steps={n_dmi_steps}")
        if i < 3 or is_correct:
            print(f"    Text: {text[:200]}")

    dmi_acc = dmi_correct / N_SAMPLES

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 3a Pilot - DMI vs Vanilla on Countdown")
    print(f"{'='*70}")
    print(f"  Vanilla:  {vanilla_correct}/{N_SAMPLES} = {vanilla_acc:.1%}")
    print(f"  DMI:      {dmi_correct}/{N_SAMPLES} = {dmi_acc:.1%}")
    print(f"  Delta:    {dmi_acc - vanilla_acc:+.1%}")

    # Vanilla metrics
    v_d1 = float(np.mean([r["distinct_1"] for r in vanilla_results]))
    v_d2 = float(np.mean([r["distinct_2"] for r in vanilla_results]))
    v_d3 = float(np.mean([r["distinct_3"] for r in vanilla_results]))
    v_r2 = float(np.mean([r["rep_2"] for r in vanilla_results]))
    v_r3 = float(np.mean([r["rep_3"] for r in vanilla_results]))
    v_time = float(np.mean([r["gen_time_s"] for r in vanilla_results]))
    v_wc = float(np.mean([r["word_count"] for r in vanilla_results]))

    # DMI metrics
    d_d1 = float(np.mean([r["distinct_1"] for r in dmi_results]))
    d_d2 = float(np.mean([r["distinct_2"] for r in dmi_results]))
    d_d3 = float(np.mean([r["distinct_3"] for r in dmi_results]))
    d_r2 = float(np.mean([r["rep_2"] for r in dmi_results]))
    d_r3 = float(np.mean([r["rep_3"] for r in dmi_results]))
    d_time = float(np.mean([r["gen_time_s"] for r in dmi_results]))
    d_wc = float(np.mean([r["word_count"] for r in dmi_results]))

    print(f"\n  Vanilla: distinct-1={v_d1:.3f}, distinct-2={v_d2:.3f}, distinct-3={v_d3:.3f}")
    print(f"           rep-2={v_r2:.3f}, rep-3={v_r3:.3f}, avg_words={v_wc:.0f}, time={v_time:.1f}s")
    print(f"  DMI:     distinct-1={d_d1:.3f}, distinct-2={d_d2:.3f}, distinct-3={d_d3:.3f}")
    print(f"           rep-2={d_r2:.3f}, rep-3={d_r3:.3f}, avg_words={d_wc:.0f}, time={d_time:.1f}s")
    print(f"  Time overhead: {d_time / v_time:.2f}x" if v_time > 0 else "  Time overhead: N/A")

    # Per-sample comparison
    print(f"\n--- Per-sample Comparison ---")
    v_to_d = 0  # vanilla wrong, dmi correct
    d_to_v = 0  # vanilla correct, dmi wrong
    for vr, dr in zip(vanilla_results, dmi_results):
        if not vr["is_correct"] and dr["is_correct"]:
            v_to_d += 1
            print(f"  [+] Sample {vr['idx']}: vanilla WRONG -> DMI CORRECT "
                  f"(target={vr['problem']['target']})")
        elif vr["is_correct"] and not dr["is_correct"]:
            d_to_v += 1
            print(f"  [-] Sample {vr['idx']}: vanilla CORRECT -> DMI WRONG "
                  f"(target={vr['problem']['target']})")
    print(f"  Flips: +{v_to_d} (vanilla->DMI correct), -{d_to_v} (vanilla->DMI wrong)")

    # Pass criteria
    print(f"\n--- Pass Criteria ---")
    no_error = all(len(r["generated_text"].strip()) > 0 for r in dmi_results)
    acc_ok = dmi_acc >= vanilla_acc
    print(f"  DMI runs without errors:              {'PASS' if no_error else 'FAIL'}")
    print(f"  DMI acc >= vanilla ({vanilla_acc:.1%}):  {'PASS' if acc_ok else 'FAIL'} ({dmi_acc:.1%})")

    overall = "GO" if (no_error and acc_ok) else "CONDITIONAL-GO" if no_error else "NO-GO"
    if not acc_ok and no_error:
        overall = "CONDITIONAL-GO (accuracy below vanilla, but runs correctly)"
    print(f"  Overall: {overall}")

    # ── Save results ──
    summary = {
        "task": "task_3a",
        "mode": "pilot",
        "method": "dmi",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "alg": ALG,
        "gen_len": GEN_LEN,
        "dmi_config": {
            "alpha": DMI_ALPHA,
            "soft_tau": 0.5,
        },
        "vanilla_accuracy": vanilla_acc,
        "vanilla_correct": vanilla_correct,
        "dmi_accuracy": dmi_acc,
        "dmi_correct": dmi_correct,
        "total_count": N_SAMPLES,
        "delta_accuracy": dmi_acc - vanilla_acc,
        "vanilla_metrics": {
            "distinct_1_mean": v_d1,
            "distinct_2_mean": v_d2,
            "distinct_3_mean": v_d3,
            "rep_2_mean": v_r2,
            "rep_3_mean": v_r3,
            "avg_time_s": v_time,
            "avg_word_count": v_wc,
        },
        "dmi_metrics": {
            "distinct_1_mean": d_d1,
            "distinct_2_mean": d_d2,
            "distinct_3_mean": d_d3,
            "rep_2_mean": d_r2,
            "rep_3_mean": d_r3,
            "avg_time_s": d_time,
            "avg_word_count": d_wc,
        },
        "time_overhead": d_time / v_time if v_time > 0 else 0,
        "pass_criteria": {
            "no_error": no_error,
            "acc_ok": acc_ok,
            "overall": overall,
        },
        "vanilla_results": vanilla_results,
        "dmi_results": dmi_results,
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / "task_3a_dmi_pilot.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
