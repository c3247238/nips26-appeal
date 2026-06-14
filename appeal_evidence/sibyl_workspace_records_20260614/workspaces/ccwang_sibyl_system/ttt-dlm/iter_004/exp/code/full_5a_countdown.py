"""
Task 5a (FULL): Countdown All-Methods Comparison.
500 problems, parameterized seed, all 7 methods.

McNemar test + Bootstrap 95% CI.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 full_5a_countdown.py --seed 42
    CUDA_VISIBLE_DEVICES=1 python3 full_5a_countdown.py --seed 123
    CUDA_VISIBLE_DEVICES=2 python3 full_5a_countdown.py --seed 456
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
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 500
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

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

# RCR Hyperparams
RCR_EMA_DECAY = 0.9
RCR_WARMUP_FRAC = 0.3
RCR_MIN_REVEALED_FRAC = 0.15
RCR_THRESHOLD_SCALE = 0.7
RCR_REMASK_CAP = 0.15

# DMI Hyperparams
DMI_ALPHA = 0.3
DMI_TAU = 0.5

# SCP Hyperparams
SCP_INTERVAL = 8
SCP_WARMUP_FRAC = 0.25
SCP_STOP_FRAC = 0.85
SCP_MIN_REVEALED = 8
SCP_TOP_K = 1
SCP_BATCH_SIZE = 32

# Checkpoint interval
CHECKPOINT_EVERY = 50


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
# Sampling helper
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
# Method 1: Vanilla
# ──────────────────────────────────────────────────

def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0"):
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
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed


# ──────────────────────────────────────────────────
# Method 2: ReMDM-conf
# ──────────────────────────────────────────────────

def generate_remdm_conf(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    remask_stop_step = int(STEPS * REMASK_STOP_FRAC)
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
            if i < remask_stop_step and i < STEPS - 1:
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
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed


# ──────────────────────────────────────────────────
# Method 3: RCR
# ──────────────────────────────────────────────────

def generate_rcr(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * RCR_WARMUP_FRAC)
    running_conf = None
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            conf_vals, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
            if i >= warmup_steps and i < STEPS - 1:
                gen_region = x[0, prompt_len:]
                revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                frac_revealed = len(revealed_positions) / GEN_LEN
                if frac_revealed >= RCR_MIN_REVEALED_FRAC and len(revealed_positions) > 3:
                    logits2 = get_shifted_logits(model, x)
                    gen_logits = logits2[0, prompt_len:]
                    revealed_logits = gen_logits[revealed_positions]
                    probs = F.softmax(revealed_logits, dim=-1)
                    current_tokens = gen_region[revealed_positions]
                    confidence = probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(-1)
                    mean_conf = confidence.mean().item()
                    if running_conf is None:
                        running_conf = mean_conf
                    else:
                        running_conf = RCR_EMA_DECAY * running_conf + (1 - RCR_EMA_DECAY) * mean_conf
                    threshold = running_conf * RCR_THRESHOLD_SCALE
                    below_threshold = confidence < threshold
                    max_remask = max(1, int(len(revealed_positions) * RCR_REMASK_CAP))
                    remask_candidates = revealed_positions[below_threshold]
                    if len(remask_candidates) > max_remask:
                        candidate_conf = confidence[below_threshold]
                        _, lowest_idx = candidate_conf.topk(max_remask, largest=False)
                        remask_candidates = remask_candidates[lowest_idx]
                    if len(remask_candidates) > 0:
                        x[0, remask_candidates + prompt_len] = MASK_TOKEN_ID
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed


# ──────────────────────────────────────────────────
# Method 4: DMI
# ──────────────────────────────────────────────────

def generate_dmi(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    embed_tokens = model.model.embed_tokens
    prev_logits = None
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            if prev_logits is not None and mask_index.any():
                inputs_embeds = embed_tokens(x)
                soft_probs = F.softmax(prev_logits / DMI_TAU, dim=-1)
                soft_embeds = soft_probs @ embed_tokens.weight
                mask_2d = mask_index.unsqueeze(-1).expand_as(inputs_embeds)
                mixed = (1 - DMI_ALPHA) * inputs_embeds + DMI_ALPHA * soft_embeds
                inputs_embeds = torch.where(mask_2d, mixed, inputs_embeds)
                outputs = model(inputs_embeds=inputs_embeds, attention_mask="full", position_ids=None)
                logits = outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                logits = get_shifted_logits(model, x)
            prev_logits = logits.clone()
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed


# ──────────────────────────────────────────────────
# Method 5: SCP
# ──────────────────────────────────────────────────

def generate_scp(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * SCP_WARMUP_FRAC)
    stop_step = int(STEPS * SCP_STOP_FRAC)
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
            if (i >= warmup_steps and i < stop_step and
                i % SCP_INTERVAL == 0 and i < STEPS - 1):
                gen_region = x[0, prompt_len:]
                revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(revealed_positions) >= SCP_MIN_REVEALED:
                    contradictions = []
                    for batch_start in range(0, len(revealed_positions), SCP_BATCH_SIZE):
                        batch_pos = revealed_positions[batch_start:batch_start + SCP_BATCH_SIZE]
                        batch_size = len(batch_pos)
                        x_batch = x.expand(batch_size, -1).clone()
                        abs_pos = batch_pos + prompt_len
                        for j in range(batch_size):
                            x_batch[j, abs_pos[j]] = MASK_TOKEN_ID
                        outputs_b = model(x_batch, attention_mask="full", position_ids=None)
                        logits_b = outputs_b.logits
                        logits_b = torch.cat([logits_b[:, :1], logits_b[:, :-1]], dim=1)
                        for j in range(batch_size):
                            pos = abs_pos[j]
                            current_token = x[0, pos].item()
                            pred_logits = logits_b[j, pos]
                            _, top_k_tokens = pred_logits.topk(SCP_TOP_K)
                            if current_token not in top_k_tokens.tolist():
                                contradictions.append(batch_pos[j].item())
                    if contradictions:
                        n_remask = min(len(contradictions), max(1, len(revealed_positions) // 8))
                        remask_pos = contradictions[:n_remask]
                        for rp in remask_pos:
                            x[0, rp + prompt_len] = MASK_TOKEN_ID
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed


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
# Method 6: DTA
# ──────────────────────────────────────────────────

def generate_dta(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)
    reset_lora(lora_modules, optimizer)
    n_updates = 0
    max_norm = 0.0
    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
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
            n_updates += 1
            decay_lora_params(lora_params, LORA_GAMMA)
        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {"n_updates": n_updates, "max_norm": max_norm}


# ──────────────────────────────────────────────────
# Method 7: DTA + ReMDM-conf
# ──────────────────────────────────────────────────

def generate_dta_remdm(model, tokenizer, prompt_text, device="cuda:0",
                       lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)
    remask_stop_step = int(STEPS * REMASK_STOP_FRAC)
    reset_lora(lora_modules, optimizer)
    n_updates = 0
    max_norm = 0.0
    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0
        if i < remask_stop_step and i < STEPS - 1:
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
                remask_positions_r = revealed_positions[lowest_idx] + prompt_len
                x[0, remask_positions_r] = MASK_TOKEN_ID
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
            n_updates += 1
            decay_lora_params(lora_params, LORA_GAMMA)
        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max
    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {"n_updates": n_updates, "max_norm": max_norm}


# ──────────────────────────────────────────────────
# Runner with checkpointing
# ──────────────────────────────────────────────────

def run_method(method_name, generate_fn, model, tokenizer, problems, device,
               seed, lora_params=None, lora_modules=None, optimizer=None,
               checkpoint_file=None):
    """Run a method on all problems with checkpointing."""
    # Load checkpoint if exists
    completed_results = []
    start_idx = 0
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        completed_results = ckpt.get("results", [])
        start_idx = len(completed_results)
        print(f"  Resuming {method_name} from checkpoint: {start_idx}/{len(problems)}")

    results = completed_results
    correct = sum(1 for r in results if r.get("is_correct", False))

    for idx in range(start_idx, len(problems)):
        problem = problems[idx]
        prompt = format_countdown_prompt(problem)

        # Set per-sample seed for reproducibility
        torch.manual_seed(seed + idx)
        torch.cuda.manual_seed(seed + idx)

        try:
            if method_name in ("dta", "dta_remdm"):
                text, elapsed, extras = generate_fn(
                    model, tokenizer, prompt, device,
                    lora_params=lora_params, lora_modules=lora_modules,
                    optimizer=optimizer
                )
            elif method_name in ("remdm_conf", "rcr", "scp"):
                ret = generate_fn(model, tokenizer, prompt, device)
                if len(ret) == 3:
                    text, elapsed, extras = ret
                else:
                    text, elapsed = ret
                    extras = {}
            else:
                ret = generate_fn(model, tokenizer, prompt, device)
                if isinstance(ret, tuple) and len(ret) == 3:
                    text, elapsed, extras = ret
                else:
                    text, elapsed = ret
                    extras = {}
        except Exception as e:
            print(f"  [ERROR] {method_name} sample {idx}: {e}")
            text = ""
            elapsed = 0
            extras = {"error": str(e)}

        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            correct += 1

        metrics = compute_text_metrics(text) if text else {
            "word_count": 0, "distinct_1": 0, "distinct_2": 0,
            "distinct_3": 0, "rep_2": 0, "rep_3": 0
        }
        result = {
            "idx": idx,
            "is_correct": is_correct,
            "gen_time_s": elapsed,
            "target": problem["target"],
            "extracted_equation": verification.get("extracted_equation"),
            **metrics,
        }
        # Only store max_norm for DTA methods (keep results compact)
        if "max_norm" in extras:
            result["max_norm"] = extras["max_norm"]
        results.append(result)

        if (idx + 1) % 10 == 0:
            acc_so_far = correct / (idx + 1)
            print(f"  [{method_name} {idx+1}/{len(problems)}] "
                  f"acc={acc_so_far:.1%} ({correct}/{idx+1}) | {elapsed:.1f}s")

        # Checkpoint periodically
        if checkpoint_file and (idx + 1) % CHECKPOINT_EVERY == 0:
            ckpt_data = {"method": method_name, "seed": seed, "results": results}
            with open(checkpoint_file, "w") as f:
                json.dump(ckpt_data, f)
            print(f"  [checkpoint] Saved at {idx+1}/{len(problems)}")

        torch.cuda.empty_cache()

    accuracy = correct / len(problems)
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in results])),
    }
    print(f"  => {method_name}: {correct}/{len(problems)} = {accuracy:.1%}")
    return results, summary


# ──────────────────────────────────────────────────
# Statistical Tests
# ──────────────────────────────────────────────────

def mcnemar_test(correct_a, correct_b):
    """McNemar's test for paired binary outcomes."""
    n = len(correct_a)
    assert len(correct_b) == n
    # b=1,c=0: a wrong, b right  |  b=0,c=1: a right, b wrong
    b_count = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)  # a wrong, b right
    c_count = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)  # a right, b wrong
    if b_count + c_count == 0:
        return {"chi2": 0, "p_value": 1.0, "b": b_count, "c": c_count}
    chi2 = (abs(b_count - c_count) - 1)**2 / (b_count + c_count)
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"chi2": float(chi2), "p_value": float(p_value), "b": b_count, "c": c_count}


def bootstrap_ci(correct_list, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(correct_list)
    arr = np.array(correct_list, dtype=float)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    means = sorted(means)
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi)}


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated methods or 'all'")
    args = parser.parse_args()

    seed = args.seed
    device = "cuda:0"
    start_time = datetime.now()

    if args.methods == "all":
        methods_to_run = ["vanilla", "remdm_conf", "rcr", "dmi", "scp", "dta", "dta_remdm"]
    else:
        methods_to_run = args.methods.split(",")

    print(f"=== Task 5a FULL: Countdown All-Methods (seed={seed}) ===")
    print(f"Samples: {N_SAMPLES}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Methods: {methods_to_run}")
    print(f"Start: {start_time.isoformat()}")

    # Use FIXED seed for problem generation (same problems across all seeds)
    problems = generate_countdown_problems(N_SAMPLES, seed=42)
    print(f"Generated {len(problems)} Countdown problems")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}")

    all_results = {}
    all_summaries = {}

    # Methods without LoRA
    non_lora_methods = {
        "vanilla": generate_vanilla,
        "remdm_conf": generate_remdm_conf,
        "rcr": generate_rcr,
        "dmi": generate_dmi,
        "scp": generate_scp,
    }

    for method_name in methods_to_run:
        if method_name in non_lora_methods:
            print(f"\n{'='*60}\n  {method_name} (seed={seed})\n{'='*60}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            ckpt_file = CHECKPOINT_DIR / f"task5a_{method_name}_s{seed}.json"
            results, summary = run_method(
                method_name, non_lora_methods[method_name], model, tokenizer,
                problems, device, seed, checkpoint_file=ckpt_file
            )
            all_results[method_name] = results
            all_summaries[method_name] = summary
            torch.cuda.empty_cache(); gc.collect()

    # Methods with LoRA
    lora_methods = [m for m in methods_to_run if m in ("dta", "dta_remdm")]
    if lora_methods:
        lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                                rank=LORA_RANK, alpha=LORA_ALPHA)
        optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

        lora_generate_fns = {
            "dta": generate_dta,
            "dta_remdm": generate_dta_remdm,
        }

        for method_name in lora_methods:
            print(f"\n{'='*60}\n  {method_name} (seed={seed})\n{'='*60}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            ckpt_file = CHECKPOINT_DIR / f"task5a_{method_name}_s{seed}.json"
            results, summary = run_method(
                method_name, lora_generate_fns[method_name], model, tokenizer,
                problems, device, seed,
                lora_params=lora_params, lora_modules=lora_modules,
                optimizer=optimizer, checkpoint_file=ckpt_file
            )
            all_results[method_name] = results
            all_summaries[method_name] = summary
            dta_norms = [r.get("max_norm", 0) for r in results]
            all_summaries[method_name]["lora_norm_max"] = float(max(dta_norms)) if dta_norms else 0
            all_summaries[method_name]["lora_norm_mean"] = float(np.mean(dta_norms)) if dta_norms else 0
            torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()

    # Bootstrap CIs for each method
    for method_name, results in all_results.items():
        correct_list = [r["is_correct"] for r in results]
        ci = bootstrap_ci(correct_list)
        all_summaries[method_name]["bootstrap_ci"] = ci

    # McNemar tests vs vanilla (if vanilla is present)
    mcnemar_results = {}
    if "vanilla" in all_results:
        vanilla_correct = [r["is_correct"] for r in all_results["vanilla"]]
        for method_name, results in all_results.items():
            if method_name != "vanilla":
                other_correct = [r["is_correct"] for r in results]
                try:
                    mcn = mcnemar_test(vanilla_correct, other_correct)
                    mcnemar_results[f"vanilla_vs_{method_name}"] = mcn
                except Exception as e:
                    mcnemar_results[f"vanilla_vs_{method_name}"] = {"error": str(e)}

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY: seed={seed}, N={N_SAMPLES}")
    print(f"{'='*80}")
    print(f"  {'Method':<18} {'Acc':>8} {'95% CI':>18} {'Correct':>8} {'AvgTime':>10}")
    print(f"  {'-'*62}")
    for m in methods_to_run:
        if m in all_summaries:
            s = all_summaries[m]
            ci = s.get("bootstrap_ci", {})
            ci_str = f"[{ci.get('ci_low',0):.3f}, {ci.get('ci_high',0):.3f}]"
            print(f"  {m:<18} {s['accuracy']:>7.1%} {ci_str:>18} {s['correct']:>8} {s['avg_time_s']:>9.1f}s")

    # Save
    output = {
        "task": "task_5a",
        "mode": "full",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": seed,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "methods": methods_to_run,
        "summaries": all_summaries,
        "mcnemar_tests": mcnemar_results,
        "configs": {
            "dta": {"rank": LORA_RANK, "layers": LORA_LAYERS, "lr": LORA_LR,
                    "gamma": LORA_GAMMA, "warmup": WARMUP_FRAC, "alpha": LORA_ALPHA},
            "remdm": {"remask_ratio": REMASK_RATIO, "stop_frac": REMASK_STOP_FRAC},
        },
        "results": all_results,
        "wall_clock_s": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / f"task5a_countdown_s{seed}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Write DONE marker
    done_file = RESULTS_DIR / f"task5a_s{seed}_DONE"
    done_file.write_text(json.dumps({
        "task_id": f"task_5a_s{seed}",
        "status": "success",
        "summary": f"Countdown 500 x seed {seed}: " + ", ".join(
            f"{m}={all_summaries[m]['accuracy']:.1%}" for m in methods_to_run if m in all_summaries
        ),
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"DONE marker written to {done_file}")


if __name__ == "__main__":
    main()
