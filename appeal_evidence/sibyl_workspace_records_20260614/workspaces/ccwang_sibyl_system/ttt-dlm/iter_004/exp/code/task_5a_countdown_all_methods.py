"""
Task 5a (PILOT): Countdown All-Methods Comparison (16 samples).

Runs all 7 methods on the same 16 Countdown problems with a single model load:
  1. Vanilla (standard Dream origin denoising)
  2. ReMDM-conf (confidence-based remasking)
  3. RCR (running confidence remasking)
  4. DMI (diffusion memory injection - Level 1 ablation)
  5. SCP (self-contradiction probing - Level 2 ablation)
  6. DTA (denoising-time adaptation - Level 3, LoRA online update)
  7. DTA+ReMDM (DTA combined with ReMDM-conf)

Pass criteria (pilot): All 7 methods run successfully, DTA accuracy > vanilla.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_5a_countdown_all_methods.py
"""
import os, sys, json, time, random, re, math, gc
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
    """Forward pass + Dream's shifted logits."""
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
    total_remasked = 0

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

            # ReMDM-conf: remask low-confidence revealed tokens
            if i < remask_stop_step and i < STEPS - 1:
                gen_region = x[0, prompt_len:]
                revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(revealed_positions) > 3:
                    # Fresh forward pass to get current confidence
                    logits2 = get_shifted_logits(model, x)
                    gen_logits = logits2[0, prompt_len:]
                    revealed_logits = gen_logits[revealed_positions]
                    probs = F.softmax(revealed_logits, dim=-1)
                    current_tokens = gen_region[revealed_positions]
                    confidence = probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(-1)

                    # Remask lowest-confidence tokens
                    n_remask = max(1, int(len(revealed_positions) * REMASK_RATIO))
                    _, lowest_idx = confidence.topk(n_remask, largest=False)
                    remask_positions = revealed_positions[lowest_idx] + prompt_len
                    x[0, remask_positions] = MASK_TOKEN_ID
                    total_remasked += n_remask

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {"total_remasked": total_remasked}


# ──────────────────────────────────────────────────
# Method 3: RCR (Running Confidence Remasking)
# ──────────────────────────────────────────────────

def generate_rcr(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * RCR_WARMUP_FRAC)
    running_conf = None
    total_remasked = 0

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

            # RCR: remask tokens below running confidence average
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
                        total_remasked += len(remask_candidates)

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {"total_remasked": total_remasked}


# ──────────────────────────────────────────────────
# Method 4: DMI (Diffusion Memory Injection)
# ──────────────────────────────────────────────────

def generate_dmi(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)

    # Get embedding layer
    embed_tokens = model.model.embed_tokens
    prev_logits = None

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break

            # DMI: inject previous logits as soft embeddings for masked positions
            if prev_logits is not None and mask_index.any():
                # Get base embeddings
                inputs_embeds = embed_tokens(x)
                # Compute soft embeddings from previous logits
                soft_probs = F.softmax(prev_logits / DMI_TAU, dim=-1)
                soft_embeds = soft_probs @ embed_tokens.weight

                # Mix only at masked positions
                mask_2d = mask_index.unsqueeze(-1).expand_as(inputs_embeds)
                mixed = (1 - DMI_ALPHA) * inputs_embeds + DMI_ALPHA * soft_embeds
                inputs_embeds = torch.where(mask_2d, mixed, inputs_embeds)

                # Forward with modified embeddings
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
# Method 5: SCP (Self-Contradiction Probing)
# ──────────────────────────────────────────────────

def generate_scp(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * SCP_WARMUP_FRAC)
    stop_step = int(STEPS * SCP_STOP_FRAC)
    total_contradictions = 0
    total_remasked = 0

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

            # SCP: leave-one-out probing at intervals
            if (i >= warmup_steps and i < stop_step and
                i % SCP_INTERVAL == 0 and i < STEPS - 1):
                gen_region = x[0, prompt_len:]
                revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]

                if len(revealed_positions) >= SCP_MIN_REVEALED:
                    contradictions = []
                    # Process in batches
                    for batch_start in range(0, len(revealed_positions), SCP_BATCH_SIZE):
                        batch_pos = revealed_positions[batch_start:batch_start + SCP_BATCH_SIZE]
                        batch_size = len(batch_pos)

                        # Create batch of sequences with one token masked each
                        x_batch = x.expand(batch_size, -1).clone()
                        abs_pos = batch_pos + prompt_len
                        for j in range(batch_size):
                            x_batch[j, abs_pos[j]] = MASK_TOKEN_ID

                        # Forward pass on batch
                        outputs_b = model(x_batch, attention_mask="full", position_ids=None)
                        logits_b = outputs_b.logits
                        logits_b = torch.cat([logits_b[:, :1], logits_b[:, :-1]], dim=1)

                        # Check if top-k prediction matches current token
                        for j in range(batch_size):
                            pos = abs_pos[j]
                            current_token = x[0, pos].item()
                            pred_logits = logits_b[j, pos]
                            _, top_k_tokens = pred_logits.topk(SCP_TOP_K)
                            if current_token not in top_k_tokens.tolist():
                                contradictions.append(batch_pos[j].item())

                    total_contradictions += len(contradictions)

                    # Remask contradictory tokens (at most top-1 per interval)
                    if contradictions:
                        n_remask = min(len(contradictions), max(1, len(revealed_positions) // 8))
                        remask_pos = contradictions[:n_remask]
                        for rp in remask_pos:
                            x[0, rp + prompt_len] = MASK_TOKEN_ID
                        total_remasked += n_remask

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, {"total_contradictions": total_contradictions,
                           "total_remasked": total_remasked}


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
    lora_losses = []
    max_norm = 0.0

    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        # E-step: standard denoising
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

        # Track norms
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
    lora_losses = []
    max_norm = 0.0
    total_remasked = 0

    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        # E-step: standard denoising
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

        # ReMDM-conf step
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

def run_method(method_name, generate_fn, model, tokenizer, problems, device,
               lora_params=None, lora_modules=None, optimizer=None):
    """Run a method on all problems, return results list and summary."""
    results = []
    correct = 0

    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)

        # Call the appropriate generate function
        if method_name in ("dta", "dta_remdm"):
            text, elapsed, extras = generate_fn(
                model, tokenizer, prompt, device,
                lora_params=lora_params, lora_modules=lora_modules,
                optimizer=optimizer
            )
        elif method_name in ("remdm_conf", "rcr", "scp"):
            text, elapsed, extras = generate_fn(model, tokenizer, prompt, device)
        else:  # vanilla, dmi
            ret = generate_fn(model, tokenizer, prompt, device)
            if len(ret) == 2:
                text, elapsed = ret
                extras = {}
            else:
                text, elapsed, extras = ret

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
        results.append(result)

        status = "OK" if is_correct else "X"
        eq = verification.get("extracted_equation", "N/A")
        print(f"  [{method_name} {idx+1}/{len(problems)}] {status} | "
              f"target={problem['target']} | eq={eq} | {elapsed:.1f}s")

    accuracy = correct / len(problems)
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in results])),
    }
    print(f"  => {method_name}: {correct}/{len(problems)} = {accuracy:.1%}, "
          f"avg_time={summary['avg_time_s']:.1f}s")
    return results, summary


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = "cuda:0"
    start_time = datetime.now()
    print(f"=== Task 5a PILOT: Countdown All-Methods Comparison ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Start: {start_time.isoformat()}")

    # Generate problems (same for all methods)
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

    all_method_results = {}
    all_method_summaries = {}

    # ── Methods without LoRA (run first to avoid model modification) ──

    def set_seed():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    # 1. Vanilla
    print(f"\n{'='*60}\n  Method 1/7: Vanilla\n{'='*60}")
    set_seed()
    results, summary = run_method("vanilla", generate_vanilla, model, tokenizer,
                                  problems, device)
    all_method_results["vanilla"] = results
    all_method_summaries["vanilla"] = summary
    torch.cuda.empty_cache(); gc.collect()

    # 2. ReMDM-conf
    print(f"\n{'='*60}\n  Method 2/7: ReMDM-conf\n{'='*60}")
    set_seed()
    results, summary = run_method("remdm_conf", generate_remdm_conf, model,
                                  tokenizer, problems, device)
    all_method_results["remdm_conf"] = results
    all_method_summaries["remdm_conf"] = summary
    torch.cuda.empty_cache(); gc.collect()

    # 3. RCR
    print(f"\n{'='*60}\n  Method 3/7: RCR\n{'='*60}")
    set_seed()
    results, summary = run_method("rcr", generate_rcr, model, tokenizer,
                                  problems, device)
    all_method_results["rcr"] = results
    all_method_summaries["rcr"] = summary
    torch.cuda.empty_cache(); gc.collect()

    # 4. DMI
    print(f"\n{'='*60}\n  Method 4/7: DMI\n{'='*60}")
    set_seed()
    results, summary = run_method("dmi", generate_dmi, model, tokenizer,
                                  problems, device)
    all_method_results["dmi"] = results
    all_method_summaries["dmi"] = summary
    torch.cuda.empty_cache(); gc.collect()

    # 5. SCP
    print(f"\n{'='*60}\n  Method 5/7: SCP\n{'='*60}")
    set_seed()
    results, summary = run_method("scp", generate_scp, model, tokenizer,
                                  problems, device)
    all_method_results["scp"] = results
    all_method_summaries["scp"] = summary
    torch.cuda.empty_cache(); gc.collect()

    # ── Methods with LoRA (inject once, reset per sample) ──

    # 6. DTA
    print(f"\n{'='*60}\n  Method 6/7: DTA\n{'='*60}")
    set_seed()
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                            rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    results, summary = run_method("dta", generate_dta, model, tokenizer,
                                  problems, device,
                                  lora_params=lora_params,
                                  lora_modules=lora_modules,
                                  optimizer=optimizer)
    all_method_results["dta"] = results
    all_method_summaries["dta"] = summary
    # Add DTA-specific stats
    dta_norms = [r.get("max_norm", 0) for r in results]
    all_method_summaries["dta"]["lora_norm_max"] = float(max(dta_norms)) if dta_norms else 0
    all_method_summaries["dta"]["lora_norm_mean"] = float(np.mean(dta_norms)) if dta_norms else 0
    torch.cuda.empty_cache(); gc.collect()

    # 7. DTA+ReMDM
    print(f"\n{'='*60}\n  Method 7/7: DTA+ReMDM\n{'='*60}")
    set_seed()
    results, summary = run_method("dta_remdm", generate_dta_remdm, model,
                                  tokenizer, problems, device,
                                  lora_params=lora_params,
                                  lora_modules=lora_modules,
                                  optimizer=optimizer)
    all_method_results["dta_remdm"] = results
    all_method_summaries["dta_remdm"] = summary
    combo_norms = [r.get("max_norm", 0) for r in results]
    all_method_summaries["dta_remdm"]["lora_norm_max"] = float(max(combo_norms)) if combo_norms else 0
    torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()

    # ── Summary Table ──
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY: Task 5a Pilot - All Methods on Countdown ({N_SAMPLES} samples)")
    print(f"{'='*80}")
    print(f"  {'Method':<18} {'Acc':>8} {'Correct':>8} {'AvgTime':>10} {'Dist-2':>8} {'Rep-3':>8}")
    print(f"  {'-'*70}")

    methods_order = ["vanilla", "remdm_conf", "rcr", "dmi", "scp", "dta", "dta_remdm"]
    for m in methods_order:
        s = all_method_summaries[m]
        print(f"  {m:<18} {s['accuracy']:>7.1%} {s['correct']:>8} "
              f"{s['avg_time_s']:>9.1f}s {s['distinct_2_mean']:>7.3f} {s['rep_3_mean']:>7.3f}")

    # Pass criteria
    vanilla_acc = all_method_summaries["vanilla"]["accuracy"]
    dta_acc = all_method_summaries["dta"]["accuracy"]
    all_ran = all(m in all_method_results for m in methods_order)
    all_have_results = all(len(all_method_results[m]) == N_SAMPLES for m in methods_order)
    dta_better = dta_acc > vanilla_acc

    print(f"\n--- Pass Criteria ---")
    print(f"  All 7 methods ran:              {'PASS' if all_ran and all_have_results else 'FAIL'}")
    print(f"  DTA acc > vanilla:              {'PASS' if dta_better else 'FAIL'} "
          f"(DTA={dta_acc:.1%} vs Vanilla={vanilla_acc:.1%})")
    overall = "GO" if (all_ran and all_have_results and dta_better) else "CONDITIONAL-GO"
    if not all_ran or not all_have_results:
        overall = "NO-GO"
    print(f"  Overall: {overall}")
    print(f"\n  Wall-clock: {(end_time - start_time).total_seconds():.0f}s")

    # ── Save results ──
    output = {
        "task": "task_5a",
        "mode": "pilot",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "methods": methods_order,
        "summaries": all_method_summaries,
        "pass_criteria": {
            "all_methods_ran": all_ran and all_have_results,
            "dta_better_than_vanilla": dta_better,
            "overall": overall,
        },
        "configs": {
            "dta": {"rank": LORA_RANK, "layers": LORA_LAYERS, "lr": LORA_LR,
                    "gamma": LORA_GAMMA, "warmup": WARMUP_FRAC, "alpha": LORA_ALPHA},
            "remdm": {"remask_ratio": REMASK_RATIO, "stop_frac": REMASK_STOP_FRAC},
            "rcr": {"ema_decay": RCR_EMA_DECAY, "warmup_frac": RCR_WARMUP_FRAC,
                    "min_revealed_frac": RCR_MIN_REVEALED_FRAC,
                    "threshold_scale": RCR_THRESHOLD_SCALE, "remask_cap": RCR_REMASK_CAP},
            "dmi": {"alpha": DMI_ALPHA, "tau": DMI_TAU},
            "scp": {"interval": SCP_INTERVAL, "warmup_frac": SCP_WARMUP_FRAC,
                    "stop_frac": SCP_STOP_FRAC, "top_k": SCP_TOP_K},
        },
        "results": {m: all_method_results[m] for m in methods_order},
        "wall_clock_s": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / "task_5a_countdown_all_methods.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
