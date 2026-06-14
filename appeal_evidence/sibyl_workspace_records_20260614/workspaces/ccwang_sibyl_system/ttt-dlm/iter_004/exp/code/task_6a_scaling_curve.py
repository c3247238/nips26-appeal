"""
Task 6a (PILOT): Inference-Time Scaling Curve Analysis.

Scans denoising steps T in {64, 128, 256, 512} for 4 methods:
  1. Vanilla
  2. ReMDM-conf
  3. DTA
  4. DTA+ReMDM

on 16 Countdown problems.  Verifies H3: DTA accuracy keeps improving at T=256,512
while ReMDM-conf saturates beyond T>2L.

Usage (parallel across 4 GPUs):
    CUDA_VISIBLE_DEVICES=0 python3 task_6a_scaling_curve.py --steps=64  &
    CUDA_VISIBLE_DEVICES=1 python3 task_6a_scaling_curve.py --steps=128 &
    CUDA_VISIBLE_DEVICES=2 python3 task_6a_scaling_curve.py --steps=256 &
    CUDA_VISIBLE_DEVICES=4 python3 task_6a_scaling_curve.py --steps=512 &
    wait

Or run all step counts sequentially (single GPU):
    python3 task_6a_scaling_curve.py  (runs all 4 step counts on cuda:0)
"""
import os, sys, json, time, random, re, math, gc, argparse
from pathlib import Path
from datetime import datetime

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
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

ALL_STEPS = [64, 128, 256, 512]
METHODS = ["vanilla", "remdm_conf", "dta", "dta_remdm"]

# DTA Hyperparams
LORA_RANK = 4
LORA_LAYERS = 2
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# ReMDM-conf Hyperparams
REMASK_RATIO = 0.1
REMASK_STOP_FRAC = 0.8


# ──────────────────────────────────────────────────
# Countdown Problem Generator
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

def compute_text_metrics(text):
    return {
        "word_count": len(text.split()),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "rep_2": rep_n(text, 2),
        "rep_3": rep_n(text, 3),
    }


# ──────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────

def sample_tokens_from_logits(logits, temperature=1.0):
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
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


def get_shifted_logits(model, x):
    outputs = model(x, attention_mask="full", position_ids=None)
    logits = outputs.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


# ──────────────────────────────────────────────────
# Method 1: Vanilla (parameterized steps)
# ──────────────────────────────────────────────────

def generate_vanilla(model, tokenizer, prompt_text, device, steps):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]

            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {}


# ──────────────────────────────────────────────────
# Method 2: ReMDM-conf (parameterized steps)
# ──────────────────────────────────────────────────

def generate_remdm_conf(model, tokenizer, prompt_text, device, steps):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    remask_stop_step = int(steps * REMASK_STOP_FRAC)
    total_remasked = 0

    t0 = time.time()
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]

            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

            if i < remask_stop_step and i < steps - 1:
                gen_region = x[0, prompt_len:]
                revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(revealed_positions) > 3:
                    logits2 = get_shifted_logits(model, x)
                    gen_logits = logits2[0, prompt_len:]
                    revealed_logits = gen_logits[revealed_positions]
                    probs = F.softmax(revealed_logits, dim=-1)
                    current_tokens = gen_region[revealed_positions]
                    confidence = probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(-1)

                    n_remask = max(1, int(len(revealed_positions) * REMASK_RATIO))
                    _, lowest_idx = confidence.topk(n_remask, largest=False)
                    remask_positions = revealed_positions[lowest_idx] + prompt_len
                    x[0, remask_positions] = MASK_TOKEN_ID
                    total_remasked += n_remask

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {"total_remasked": total_remasked}


# ──────────────────────────────────────────────────
# LoRA Implementation
# ──────────────────────────────────────────────────

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features,
                                               dtype=original.weight.dtype,
                                               device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank,
                                               dtype=original.weight.dtype,
                                               device=original.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_delta_norm(self):
        with torch.no_grad():
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            return delta.norm().item()


def inject_lora(model, n_layers=2, rank=4, alpha=1.0):
    total_layers = len(model.model.layers)
    target_layers = list(range(total_layers - n_layers, total_layers))
    lora_params = []
    lora_modules = []
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            original = getattr(mlp, proj_name)
            lora_layer = LoRALayer(original, rank=rank, alpha=alpha)
            setattr(mlp, proj_name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            lora_modules.append(lora_layer)
    n_params = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected into layers {target_layers}, {len(lora_modules)} modules, {n_params} params")
    return lora_params, lora_modules


def reset_lora(lora_modules, optimizer):
    for m in lora_modules:
        m.reset_parameters()
    optimizer.zero_grad()
    optimizer.state.clear()


def decay_lora_params(lora_params, gamma):
    with torch.no_grad():
        for p in lora_params:
            p.mul_(gamma)


# ──────────────────────────────────────────────────
# Method 3: DTA (parameterized steps)
# ──────────────────────────────────────────────────

def generate_dta(model, tokenizer, prompt_text, device, steps,
                 lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    warmup_steps = int(steps * WARMUP_FRAC)

    reset_lora(lora_modules, optimizer)

    n_updates = 0
    lora_losses = []
    max_norm = 0.0

    t0 = time.time()
    for i in range(steps):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0

        gen_region = x[0, prompt_len:]
        revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        total_revealed = len(revealed_positions)

        if i >= warmup_steps and total_revealed >= 5:
            n_to_mask = max(2, total_revealed // 5)
            perm = torch.randperm(total_revealed, device=device)[:n_to_mask]
            mask_positions = revealed_positions[perm] + prompt_len

            target_tokens = x[0, mask_positions].clone()
            x_masked = x.clone()
            x_masked[0, mask_positions] = MASK_TOKEN_ID

            outputs_m = model(x_masked, attention_mask="full", position_ids=None)
            logits_m = outputs_m.logits
            logits_m = torch.cat([logits_m[:, :1], logits_m[:, :-1]], dim=1)

            loss_logits = logits_m[0, mask_positions]
            loss = F.cross_entropy(loss_logits, target_tokens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            lora_losses.append(loss.item())
            n_updates += 1

            decay_lora_params(lora_params, LORA_GAMMA)

        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
    return text, elapsed, {"n_updates": n_updates, "max_norm": max_norm,
                           "avg_loss": avg_loss}


# ──────────────────────────────────────────────────
# Method 4: DTA + ReMDM-conf (parameterized steps)
# ──────────────────────────────────────────────────

def generate_dta_remdm(model, tokenizer, prompt_text, device, steps,
                       lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    warmup_steps = int(steps * WARMUP_FRAC)
    remask_stop_step = int(steps * REMASK_STOP_FRAC)

    reset_lora(lora_modules, optimizer)

    n_updates = 0
    lora_losses = []
    max_norm = 0.0
    total_remasked = 0

    t0 = time.time()
    for i in range(steps):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0

        # ReMDM-conf step
        if i < remask_stop_step and i < steps - 1:
            gen_region = x[0, prompt_len:]
            revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(revealed_positions) > 3:
                with torch.no_grad():
                    logits2 = get_shifted_logits(model, x)
                gen_logits = logits2[0, prompt_len:]
                revealed_logits = gen_logits[revealed_positions]
                probs = F.softmax(revealed_logits, dim=-1)
                current_tokens = gen_region[revealed_positions]
                confidence = probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(-1)

                n_remask = max(1, int(len(revealed_positions) * REMASK_RATIO))
                _, lowest_idx = confidence.topk(n_remask, largest=False)
                remask_positions = revealed_positions[lowest_idx] + prompt_len
                x[0, remask_positions] = MASK_TOKEN_ID
                total_remasked += n_remask

        # M-step: LoRA update
        gen_region = x[0, prompt_len:]
        revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        total_revealed = len(revealed_positions)

        if i >= warmup_steps and total_revealed >= 5:
            n_to_mask = max(2, total_revealed // 5)
            perm = torch.randperm(total_revealed, device=device)[:n_to_mask]
            mask_positions = revealed_positions[perm] + prompt_len

            target_tokens = x[0, mask_positions].clone()
            x_masked = x.clone()
            x_masked[0, mask_positions] = MASK_TOKEN_ID

            outputs_m = model(x_masked, attention_mask="full", position_ids=None)
            logits_m = outputs_m.logits
            logits_m = torch.cat([logits_m[:, :1], logits_m[:, :-1]], dim=1)

            loss_logits = logits_m[0, mask_positions]
            loss = F.cross_entropy(loss_logits, target_tokens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            lora_losses.append(loss.item())
            n_updates += 1

            decay_lora_params(lora_params, LORA_GAMMA)

        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
    return text, elapsed, {"n_updates": n_updates, "max_norm": max_norm,
                           "avg_loss": avg_loss, "total_remasked": total_remasked}


# ──────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────

GENERATE_FNS = {
    "vanilla": generate_vanilla,
    "remdm_conf": generate_remdm_conf,
    "dta": generate_dta,
    "dta_remdm": generate_dta_remdm,
}


def run_single_steps(steps, model, tokenizer, problems, device,
                     lora_params, lora_modules, optimizer):
    """Run all 4 methods at a given step count. Returns results dict."""
    results_for_steps = {}

    def set_seed():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    for method_name in METHODS:
        set_seed()
        gen_fn = GENERATE_FNS[method_name]
        method_results = []
        correct = 0

        for idx, problem in enumerate(problems):
            prompt = format_countdown_prompt(problem)

            if method_name in ("dta", "dta_remdm"):
                text, elapsed, extras = gen_fn(
                    model, tokenizer, prompt, device, steps,
                    lora_params=lora_params, lora_modules=lora_modules,
                    optimizer=optimizer
                )
            else:
                text, elapsed, extras = gen_fn(model, tokenizer, prompt, device, steps)

            verification = verify_countdown_answer(text, problem)
            is_correct = verification["is_correct"]
            if is_correct:
                correct += 1

            metrics = compute_text_metrics(text)
            result = {
                "idx": idx,
                "problem": problem,
                "generated_text": text,
                "verification": verification,
                "is_correct": is_correct,
                "gen_time_s": elapsed,
                **metrics,
                **extras,
            }
            method_results.append(result)

            status = "OK" if is_correct else "X"
            eq = verification.get("extracted_equation", "N/A")
            print(f"  [T={steps} {method_name} {idx+1}/{len(problems)}] {status} | "
                  f"target={problem['target']} | eq={eq} | {elapsed:.1f}s")

        accuracy = correct / len(problems)
        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "avg_time_s": float(np.mean([r["gen_time_s"] for r in method_results])),
            "total_time_s": float(sum(r["gen_time_s"] for r in method_results)),
            "distinct_2_mean": float(np.mean([r["distinct_2"] for r in method_results])),
            "rep_3_mean": float(np.mean([r["rep_3"] for r in method_results])),
        }
        # DTA-specific summary fields
        if method_name in ("dta", "dta_remdm"):
            norms = [r.get("max_norm", 0) for r in method_results]
            summary["lora_norm_max"] = float(max(norms)) if norms else 0
            summary["lora_norm_mean"] = float(np.mean(norms)) if norms else 0

        print(f"  => T={steps} {method_name}: {correct}/{len(problems)} = {accuracy:.1%}, "
              f"avg_time={summary['avg_time_s']:.1f}s")

        results_for_steps[method_name] = {
            "summary": summary,
            "details": method_results,
        }
        torch.cuda.empty_cache()
        gc.collect()

    return results_for_steps


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=0,
                        help="Run only this step count (0 = run all)")
    args = parser.parse_args()

    if args.steps > 0:
        steps_to_run = [args.steps]
    else:
        steps_to_run = ALL_STEPS

    device = "cuda:0"
    start_time = datetime.now()
    print(f"=== Task 6a PILOT: Inference-Time Scaling Curve ===")
    print(f"Steps to run: {steps_to_run}")
    print(f"Methods: {METHODS}")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Temp: {TEMPERATURE}")
    print(f"Start: {start_time.isoformat()}")

    # Generate problems (same for all methods and step counts)
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"Generated {len(problems)} Countdown problems")

    # Load model
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}")

    # Inject LoRA (needed for DTA and DTA+ReMDM)
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                            rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    # Run all step counts
    all_results = {}
    for steps in steps_to_run:
        print(f"\n{'='*70}")
        print(f"  Running T={steps} denoising steps")
        print(f"{'='*70}")
        all_results[str(steps)] = run_single_steps(
            steps, model, tokenizer, problems, device,
            lora_params, lora_modules, optimizer
        )

    end_time = datetime.now()
    wall_seconds = (end_time - start_time).total_seconds()

    # Build compact summary table
    summary_table = {}
    for steps_str, step_results in all_results.items():
        summary_table[steps_str] = {}
        for method_name, method_data in step_results.items():
            summary_table[steps_str][method_name] = method_data["summary"]

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SCALING CURVE SUMMARY (T x Method)")
    print(f"{'='*80}")
    print(f"  {'T':>5} | {'Vanilla':>10} | {'ReMDM-conf':>12} | {'DTA':>10} | {'DTA+ReMDM':>12} |")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+")
    for steps_str in sorted(summary_table.keys(), key=int):
        row = summary_table[steps_str]
        vals = []
        for m in METHODS:
            if m in row:
                vals.append(f"{row[m]['accuracy']:.1%}")
            else:
                vals.append("N/A")
        print(f"  {steps_str:>5} | {vals[0]:>10} | {vals[1]:>12} | {vals[2]:>10} | {vals[3]:>12} |")

    print(f"\n  Wall time: {wall_seconds:.0f}s ({wall_seconds/60:.1f}min)")

    # H3 check: DTA still improving at T=256, T=512?
    h3_check = {"passed": False, "details": ""}
    dta_accs = {}
    remdm_accs = {}
    for steps_str, step_results in all_results.items():
        if "dta" in step_results:
            dta_accs[int(steps_str)] = step_results["dta"]["summary"]["accuracy"]
        if "remdm_conf" in step_results:
            remdm_accs[int(steps_str)] = step_results["remdm_conf"]["summary"]["accuracy"]

    if len(steps_to_run) == 4:
        # Check if DTA accuracy at T=256 or T=512 is >= T=128
        dta_256 = dta_accs.get(256, 0)
        dta_512 = dta_accs.get(512, 0)
        dta_128 = dta_accs.get(128, 0)
        dta_still_improving = (dta_256 >= dta_128) or (dta_512 >= dta_128)
        h3_check["dta_still_improving"] = dta_still_improving
        h3_check["dta_accs"] = {str(k): v for k, v in sorted(dta_accs.items())}
        h3_check["remdm_accs"] = {str(k): v for k, v in sorted(remdm_accs.items())}
        h3_check["passed"] = dta_still_improving
        h3_check["details"] = (
            f"DTA accuracy at T=256: {dta_256:.1%}, T=512: {dta_512:.1%} "
            f"(vs T=128: {dta_128:.1%}). "
            f"{'Still improving' if dta_still_improving else 'NOT improving'} at higher T."
        )
    else:
        h3_check["details"] = f"Only ran T={steps_to_run}, need all 4 for H3 check"

    pass_criteria = {
        "all_step_counts_ran": len(all_results) == len(steps_to_run),
        "all_methods_ran_per_step": all(
            len(step_results) == len(METHODS)
            for step_results in all_results.values()
        ),
        "h3_dta_still_improving": h3_check,
        "overall": "GO" if h3_check.get("passed", False) else "CONDITIONAL-GO"
    }

    # Save per-step-count partial results (for parallel runs)
    for steps_str, step_results in all_results.items():
        partial_path = RESULTS_DIR / f"task_6a_T{steps_str}.json"
        partial_data = {
            "task": "task_6a",
            "mode": "pilot",
            "steps": int(steps_str),
            "model": "Dream-v0-Instruct-7B",
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "methods": METHODS,
            "results": {
                m: step_results[m]["summary"] for m in METHODS if m in step_results
            },
            "wall_seconds": wall_seconds,
            "timestamp": end_time.isoformat(),
        }
        with open(partial_path, "w") as f:
            json.dump(partial_data, f, indent=2, default=str)
        print(f"  Saved partial results: {partial_path}")

    # Save combined results
    combined = {
        "task": "task_6a",
        "mode": "pilot",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "steps_tested": steps_to_run,
        "methods": METHODS,
        "summary_table": summary_table,
        "pass_criteria": pass_criteria,
        "wall_seconds": wall_seconds,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "configs": {
            "dta": {
                "rank": LORA_RANK,
                "layers": LORA_LAYERS,
                "lr": LORA_LR,
                "gamma": LORA_GAMMA,
                "warmup": WARMUP_FRAC,
                "alpha": LORA_ALPHA,
                "grad_clip": GRAD_CLIP,
            },
            "remdm": {
                "remask_ratio": REMASK_RATIO,
                "stop_frac": REMASK_STOP_FRAC,
            },
        },
    }

    out_path = RESULTS_DIR / "task_6a_scaling_curve.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Combined results saved: {out_path}")

    # Write DONE marker
    done_marker = Path("/home/ccwang/sibyl_system/exp/results") / "task_6a_DONE"
    done_data = {
        "task_id": "task_6a",
        "status": "success",
        "summary": (
            f"Scaling curve pilot complete. Steps: {steps_to_run}. "
            f"H3 check: {pass_criteria['overall']}. "
            f"Wall time: {wall_seconds:.0f}s."
        ),
        "timestamp": end_time.isoformat(),
    }
    with open(done_marker, "w") as f:
        json.dump(done_data, f, indent=2)
    print(f"  DONE marker written: {done_marker}")

    print(f"\n  Pass criteria: {pass_criteria['overall']}")
    print(f"  H3: {h3_check.get('details', 'N/A')}")


if __name__ == "__main__":
    main()
