"""
Task 8a (PILOT): LLaDA-8B Cross-Model Verification (16 samples each).

Ports DTA and DTA+ReMDM to LLaDA-8B-Instruct. Tests on Countdown 16 + GSM8K 16.
Confirms DTA accuracy improvement direction is consistent with Dream-7B results.

Methods tested:
  1. Vanilla (standard LLaDA confidence-based unmasking)
  2. ReMDM-conf (confidence-based remasking)
  3. DTA (denoising-time adaptation via LoRA online update)
  4. DTA+ReMDM (DTA combined with ReMDM-conf)

Pass criteria: DTA accuracy improvement direction on LLaDA matches Dream.

Usage:
    CUDA_VISIBLE_DEVICES=5 python3 task_8a_llada_cross_model.py
"""
import os, sys, json, time, random, re, math, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Config ===
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CROSS_MODEL_DIR = Path("/home/ccwang/sibyl_system/exp/results/cross_model")
CROSS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 16
SEED = 42
GEN_LEN = 256           # Countdown
GSM8K_GEN_LEN = 512     # GSM8K needs longer chains
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 126336   # LLaDA mask token

# DTA Hyperparams (same as Dream experiments for fair comparison)
LORA_RANK = 4
LORA_LAYERS = 2          # Last 2 transformer blocks
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# ReMDM-conf Hyperparams
REMASK_RATIO = 0.1
REMASK_STOP_FRAC = 0.8


# ──────────────────────────────────────────────────
# Countdown Problem Generator (identical to task_5a)
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
# GSM8K Data Loading (identical to task_5b)
# ──────────────────────────────────────────────────

def load_gsm8k(n=None, seed=42):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        if n is not None and i >= n:
            break
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
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        num_str = match.group(1).replace(',', '')
        try:
            return float(num_str)
        except:
            return None
    return None


def format_gsm8k_prompt(problem):
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


def extract_model_answer(text):
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "####"
        except:
            pass
    nums = re.findall(r'(?:answer|result|total)\s*(?:is|=|:)\s*([\-]?\d[\d,]*\.?\d*)', text, re.I)
    if nums:
        try:
            return float(nums[-1].replace(',', '')), "keyword"
        except:
            pass
    nums = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if nums:
        try:
            return float(nums[-1].replace(',', '')), "last_number"
        except:
            pass
    return None, "none"


def verify_gsm8k_answer(text, problem):
    target = problem["target"]
    if target is None:
        return {"is_correct": False, "explanation": "No target"}
    answer, method = extract_model_answer(text)
    if answer is None:
        return {"is_correct": False, "extracted_answer": None,
                "extraction_method": method, "explanation": "No answer extracted"}
    is_correct = abs(answer - target) < 1e-2
    return {
        "is_correct": is_correct,
        "extracted_answer": answer,
        "target": target,
        "extraction_method": method,
        "explanation": f"extracted={answer}, target={target}"
    }


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
# LLaDA-specific helpers
# ──────────────────────────────────────────────────

def sample_tokens_from_logits(logits, temperature=1.0):
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return confidence, sampled


def prepare_input(tokenizer, prompt_text, device, gen_len=GEN_LEN):
    """Prepare input for LLaDA: prompt tokens + mask tokens for generation."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    # LLaDA: append mask tokens for generation region
    full_ids = prompt_ids + [MASK_TOKEN_ID] * gen_len
    x = torch.tensor([full_ids], device=device)
    return x, prompt_len


def decode_output(tokenizer, x, prompt_len):
    """Decode generated tokens, skipping masks and stopping at EOS."""
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


def get_logits(model, x):
    """Forward pass for LLaDA. No logits shifting needed (unlike Dream)."""
    outputs = model(x)
    return outputs.logits


# ──────────────────────────────────────────────────
# Method 1: Vanilla (LLaDA confidence-based unmasking)
# ──────────────────────────────────────────────────

def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0", gen_len=GEN_LEN):
    """LLaDA vanilla: confidence-based iterative unmasking (following LLaDA paper)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break

            logits = get_logits(model, x)
            mask_logits = logits[mask_index]

            # Origin-style: probabilistic unmasking based on timestep schedule
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
# Method 2: ReMDM-conf (confidence-based remasking)
# ──────────────────────────────────────────────────

def generate_remdm_conf(model, tokenizer, prompt_text, device="cuda:0", gen_len=GEN_LEN):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
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
            logits = get_logits(model, x)
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
                    logits2 = get_logits(model, x)
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
# LoRA Implementation (adapted for LLaDA architecture)
# ──────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """Low-rank adapter for a single linear layer."""
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
        # B stays zero -> zero output at init (bypass initialization)

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


def inject_lora_llada(model, n_layers=2, rank=4, alpha=1.0):
    """Inject LoRA into the last n_layers of LLaDA's transformer blocks.

    LLaDA architecture: model.model.transformer.blocks[i]
    FFN projections: ff_proj (4096->12288), up_proj (4096->12288), ff_out (12288->4096)
    """
    blocks = model.model.transformer['blocks']
    total_layers = len(blocks)
    target_layers = list(range(total_layers - n_layers, total_layers))

    lora_params = []
    lora_modules = []

    for layer_idx in target_layers:
        block = blocks[layer_idx]
        # LLaDA FFN: ff_proj (gate-like), up_proj, ff_out (down)
        for proj_name in ['ff_proj', 'up_proj', 'ff_out']:
            original = getattr(block, proj_name)
            lora_layer = LoRALayer(original, rank=rank, alpha=alpha)
            setattr(block, proj_name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            lora_modules.append(lora_layer)

    n_params = sum(p.numel() for p in lora_params)
    print(f"[LoRA-LLaDA] Injected into blocks {target_layers}, "
          f"{len(lora_modules)} modules, {n_params} trainable params")
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
# Method 3: DTA (adapted for LLaDA)
# ──────────────────────────────────────────────────

def generate_dta(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None,
                 gen_len=GEN_LEN):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
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

        # E-step: standard denoising (no grad)
        with torch.no_grad():
            logits = get_logits(model, x)
        mask_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0

        # M-step: LoRA update on revealed tokens
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

            # Forward with gradient for LoRA params
            # LLaDA: no shifted logits, direct prediction at each position
            outputs_m = model(x_masked)
            logits_m = outputs_m.logits

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
# Method 4: DTA + ReMDM-conf (adapted for LLaDA)
# ──────────────────────────────────────────────────

def generate_dta_remdm(model, tokenizer, prompt_text, device="cuda:0",
                       lora_params=None, lora_modules=None, optimizer=None,
                       gen_len=GEN_LEN):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
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

        # E-step
        with torch.no_grad():
            logits = get_logits(model, x)
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
                    logits2 = get_logits(model, x)
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

            outputs_m = model(x_masked)
            logits_m = outputs_m.logits

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
# Benchmark Runner
# ──────────────────────────────────────────────────

def run_benchmark(benchmark_name, problems, prompt_fn, verify_fn,
                  model, tokenizer, device, gen_len,
                  lora_params=None, lora_modules=None, optimizer=None):
    """Run all 4 methods on a benchmark, return results dict."""
    methods_order = ["vanilla", "remdm_conf", "dta", "dta_remdm"]
    all_results = {}
    all_summaries = {}

    def set_seed():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    for method_idx, method_name in enumerate(methods_order):
        print(f"\n  [{benchmark_name}] Method {method_idx+1}/4: {method_name}")
        set_seed()

        results = []
        correct = 0

        for idx, problem in enumerate(problems):
            prompt = prompt_fn(problem)

            try:
                if method_name == "vanilla":
                    text, elapsed = generate_vanilla(model, tokenizer, prompt, device, gen_len)
                    extras = {}
                elif method_name == "remdm_conf":
                    text, elapsed, extras = generate_remdm_conf(model, tokenizer, prompt, device, gen_len)
                elif method_name == "dta":
                    text, elapsed, extras = generate_dta(
                        model, tokenizer, prompt, device,
                        lora_params=lora_params, lora_modules=lora_modules,
                        optimizer=optimizer, gen_len=gen_len)
                elif method_name == "dta_remdm":
                    text, elapsed, extras = generate_dta_remdm(
                        model, tokenizer, prompt, device,
                        lora_params=lora_params, lora_modules=lora_modules,
                        optimizer=optimizer, gen_len=gen_len)
                else:
                    raise ValueError(f"Unknown method: {method_name}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"    OOM on {method_name} sample {idx}, skipping")
                text = ""
                elapsed = 0
                extras = {"error": "OOM"}
            except Exception as e:
                print(f"    Error on {method_name} sample {idx}: {e}")
                text = ""
                elapsed = 0
                extras = {"error": str(e)}

            verification = verify_fn(text, problem)
            is_correct = verification.get("is_correct", False)
            if is_correct:
                correct += 1

            metrics = compute_text_metrics(text) if text else {
                "word_count": 0, "distinct_1": 0, "distinct_2": 0,
                "distinct_3": 0, "rep_2": 0, "rep_3": 0}

            result = {
                "idx": idx,
                "generated_text": text[:500],  # Truncate for storage
                "verification": verification,
                "is_correct": is_correct,
                "gen_time_s": elapsed,
                **metrics,
                **extras,
            }
            results.append(result)

            status = "OK" if is_correct else "X"
            print(f"    [{idx+1}/{len(problems)}] {status} | {elapsed:.1f}s"
                  + (f" | norm={extras.get('max_norm', 0):.4f}" if 'max_norm' in extras else ""))

        accuracy = correct / len(problems) if problems else 0
        valid_results = [r for r in results if r["gen_time_s"] > 0]
        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "avg_time_s": float(np.mean([r["gen_time_s"] for r in valid_results])) if valid_results else 0,
            "distinct_2_mean": float(np.mean([r["distinct_2"] for r in valid_results])) if valid_results else 0,
            "rep_3_mean": float(np.mean([r["rep_3"] for r in valid_results])) if valid_results else 0,
        }
        if method_name in ("dta", "dta_remdm"):
            norms = [r.get("max_norm", 0) for r in results]
            summary["lora_norm_max"] = float(max(norms)) if norms else 0
            summary["lora_norm_mean"] = float(np.mean(norms)) if norms else 0

        print(f"    => {method_name}: {correct}/{len(problems)} = {accuracy:.1%}, "
              f"avg_time={summary['avg_time_s']:.1f}s")

        all_results[method_name] = results
        all_summaries[method_name] = summary
        torch.cuda.empty_cache()
        gc.collect()

    return all_results, all_summaries


# ──────────────────────────────────────────────────
# DONE marker
# ──────────────────────────────────────────────────

def mark_task_done(task_id, results_dir, status="success", summary=""):
    import json
    from pathlib import Path
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = "cuda:0"
    start_time = datetime.now()
    print(f"{'='*70}")
    print(f"  Task 8a PILOT: LLaDA-8B Cross-Model Verification")
    print(f"{'='*70}")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"DTA: rank={LORA_RANK}, layers={LORA_LAYERS}, lr={LORA_LR}, gamma={LORA_GAMMA}")
    print(f"Start: {start_time.isoformat()}")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading LLaDA-8B-Instruct...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"LLaDA-8B loaded on {device}")
    print(f"  mask_token_id={MASK_TOKEN_ID}, eos_token_id={tokenizer.eos_token_id}")
    print(f"  vocab_size={model.config.vocab_size}, num_layers={model.config.num_hidden_layers}")

    # ── Phase 1: Countdown (non-LoRA methods first) ──
    print(f"\n{'='*70}")
    print(f"  Phase 1: Countdown ({N_SAMPLES} samples)")
    print(f"{'='*70}")

    countdown_problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"Generated {len(countdown_problems)} Countdown problems")

    # Run vanilla and remdm_conf first (before LoRA injection)
    print(f"\n  Running non-LoRA methods on Countdown...")
    cd_results_nolora = {}
    cd_summaries_nolora = {}

    def set_seed():
        torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
        np.random.seed(SEED); random.seed(SEED)

    # Vanilla
    print(f"\n  [Countdown] Method 1/4: vanilla")
    set_seed()
    cd_vanilla_results = []
    cd_vanilla_correct = 0
    for idx, problem in enumerate(countdown_problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed = generate_vanilla(model, tokenizer, prompt, device, GEN_LEN)
        verification = verify_countdown_answer(text, problem)
        is_correct = verification.get("is_correct", False)
        if is_correct:
            cd_vanilla_correct += 1
        metrics = compute_text_metrics(text)
        cd_vanilla_results.append({
            "idx": idx, "generated_text": text[:500], "verification": verification,
            "is_correct": is_correct, "gen_time_s": elapsed, **metrics})
        status = "OK" if is_correct else "X"
        print(f"    [{idx+1}/{N_SAMPLES}] {status} | {elapsed:.1f}s")
    cd_results_nolora["vanilla"] = cd_vanilla_results
    v_acc = cd_vanilla_correct / N_SAMPLES
    cd_summaries_nolora["vanilla"] = {
        "accuracy": v_acc, "correct": cd_vanilla_correct, "total": N_SAMPLES,
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in cd_vanilla_results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in cd_vanilla_results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in cd_vanilla_results]))}
    print(f"    => vanilla: {cd_vanilla_correct}/{N_SAMPLES} = {v_acc:.1%}")
    torch.cuda.empty_cache(); gc.collect()

    # ReMDM-conf
    print(f"\n  [Countdown] Method 2/4: remdm_conf")
    set_seed()
    cd_remdm_results = []
    cd_remdm_correct = 0
    for idx, problem in enumerate(countdown_problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, extras = generate_remdm_conf(model, tokenizer, prompt, device, GEN_LEN)
        verification = verify_countdown_answer(text, problem)
        is_correct = verification.get("is_correct", False)
        if is_correct:
            cd_remdm_correct += 1
        metrics = compute_text_metrics(text)
        cd_remdm_results.append({
            "idx": idx, "generated_text": text[:500], "verification": verification,
            "is_correct": is_correct, "gen_time_s": elapsed, **metrics, **extras})
        status = "OK" if is_correct else "X"
        print(f"    [{idx+1}/{N_SAMPLES}] {status} | {elapsed:.1f}s")
    cd_results_nolora["remdm_conf"] = cd_remdm_results
    r_acc = cd_remdm_correct / N_SAMPLES
    cd_summaries_nolora["remdm_conf"] = {
        "accuracy": r_acc, "correct": cd_remdm_correct, "total": N_SAMPLES,
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in cd_remdm_results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in cd_remdm_results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in cd_remdm_results]))}
    print(f"    => remdm_conf: {cd_remdm_correct}/{N_SAMPLES} = {r_acc:.1%}")
    torch.cuda.empty_cache(); gc.collect()

    # Inject LoRA (once, used for both DTA and DTA+ReMDM, and both benchmarks)
    print(f"\n  Injecting LoRA into LLaDA...")
    lora_params, lora_modules = inject_lora_llada(
        model, n_layers=LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    # DTA on Countdown
    print(f"\n  [Countdown] Method 3/4: dta")
    set_seed()
    cd_dta_results = []
    cd_dta_correct = 0
    for idx, problem in enumerate(countdown_problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, extras = generate_dta(
            model, tokenizer, prompt, device,
            lora_params=lora_params, lora_modules=lora_modules,
            optimizer=optimizer, gen_len=GEN_LEN)
        verification = verify_countdown_answer(text, problem)
        is_correct = verification.get("is_correct", False)
        if is_correct:
            cd_dta_correct += 1
        metrics = compute_text_metrics(text)
        cd_dta_results.append({
            "idx": idx, "generated_text": text[:500], "verification": verification,
            "is_correct": is_correct, "gen_time_s": elapsed, **metrics, **extras})
        status = "OK" if is_correct else "X"
        print(f"    [{idx+1}/{N_SAMPLES}] {status} | {elapsed:.1f}s | norm={extras.get('max_norm',0):.4f}")
    cd_results_nolora["dta"] = cd_dta_results
    d_acc = cd_dta_correct / N_SAMPLES
    dta_norms = [r.get("max_norm", 0) for r in cd_dta_results]
    cd_summaries_nolora["dta"] = {
        "accuracy": d_acc, "correct": cd_dta_correct, "total": N_SAMPLES,
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in cd_dta_results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in cd_dta_results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in cd_dta_results])),
        "lora_norm_max": float(max(dta_norms)) if dta_norms else 0,
        "lora_norm_mean": float(np.mean(dta_norms)) if dta_norms else 0}
    print(f"    => dta: {cd_dta_correct}/{N_SAMPLES} = {d_acc:.1%}")
    torch.cuda.empty_cache(); gc.collect()

    # DTA+ReMDM on Countdown
    print(f"\n  [Countdown] Method 4/4: dta_remdm")
    set_seed()
    cd_combo_results = []
    cd_combo_correct = 0
    for idx, problem in enumerate(countdown_problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, extras = generate_dta_remdm(
            model, tokenizer, prompt, device,
            lora_params=lora_params, lora_modules=lora_modules,
            optimizer=optimizer, gen_len=GEN_LEN)
        verification = verify_countdown_answer(text, problem)
        is_correct = verification.get("is_correct", False)
        if is_correct:
            cd_combo_correct += 1
        metrics = compute_text_metrics(text)
        cd_combo_results.append({
            "idx": idx, "generated_text": text[:500], "verification": verification,
            "is_correct": is_correct, "gen_time_s": elapsed, **metrics, **extras})
        status = "OK" if is_correct else "X"
        print(f"    [{idx+1}/{N_SAMPLES}] {status} | {elapsed:.1f}s | norm={extras.get('max_norm',0):.4f}")
    cd_results_nolora["dta_remdm"] = cd_combo_results
    c_acc = cd_combo_correct / N_SAMPLES
    combo_norms = [r.get("max_norm", 0) for r in cd_combo_results]
    cd_summaries_nolora["dta_remdm"] = {
        "accuracy": c_acc, "correct": cd_combo_correct, "total": N_SAMPLES,
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in cd_combo_results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in cd_combo_results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in cd_combo_results])),
        "lora_norm_max": float(max(combo_norms)) if combo_norms else 0}
    print(f"    => dta_remdm: {cd_combo_correct}/{N_SAMPLES} = {c_acc:.1%}")
    torch.cuda.empty_cache(); gc.collect()

    # ── Phase 2: GSM8K ──
    print(f"\n{'='*70}")
    print(f"  Phase 2: GSM8K ({N_SAMPLES} samples)")
    print(f"{'='*70}")

    gsm8k_problems = load_gsm8k(n=N_SAMPLES, seed=SEED)
    print(f"Loaded {len(gsm8k_problems)} GSM8K problems")

    gsm_results, gsm_summaries = run_benchmark(
        "gsm8k", gsm8k_problems, format_gsm8k_prompt, verify_gsm8k_answer,
        model, tokenizer, device, GSM8K_GEN_LEN,
        lora_params=lora_params, lora_modules=lora_modules, optimizer=optimizer)

    end_time = datetime.now()
    wall_clock = (end_time - start_time).total_seconds()

    # ── Summary Tables ──
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY: Task 8a - LLaDA-8B Cross-Model Verification")
    print(f"{'='*80}")

    print(f"\n  Countdown ({N_SAMPLES} samples):")
    print(f"  {'Method':<18} {'Acc':>8} {'Correct':>8} {'AvgTime':>10} {'Dist-2':>8} {'Rep-3':>8}")
    print(f"  {'-'*70}")
    for m in ["vanilla", "remdm_conf", "dta", "dta_remdm"]:
        s = cd_summaries_nolora[m]
        print(f"  {m:<18} {s['accuracy']:>7.1%} {s['correct']:>8} "
              f"{s['avg_time_s']:>9.1f}s {s['distinct_2_mean']:>7.3f} {s['rep_3_mean']:>7.3f}")

    print(f"\n  GSM8K ({N_SAMPLES} samples):")
    print(f"  {'Method':<18} {'Acc':>8} {'Correct':>8} {'AvgTime':>10} {'Dist-2':>8} {'Rep-3':>8}")
    print(f"  {'-'*70}")
    for m in ["vanilla", "remdm_conf", "dta", "dta_remdm"]:
        s = gsm_summaries[m]
        print(f"  {m:<18} {s['accuracy']:>7.1%} {s['correct']:>8} "
              f"{s['avg_time_s']:>9.1f}s {s['distinct_2_mean']:>7.3f} {s['rep_3_mean']:>7.3f}")

    # Pass criteria
    cd_vanilla_acc = cd_summaries_nolora["vanilla"]["accuracy"]
    cd_dta_acc = cd_summaries_nolora["dta"]["accuracy"]
    gsm_vanilla_acc = gsm_summaries["vanilla"]["accuracy"]
    gsm_dta_acc = gsm_summaries["dta"]["accuracy"]

    cd_direction = cd_dta_acc >= cd_vanilla_acc
    gsm_direction = gsm_dta_acc >= gsm_vanilla_acc
    either_positive = cd_direction or gsm_direction

    # Compare with Dream results (from task_5a pilot):
    # Dream: vanilla=12.5%, dta=6.25% (DTA underperformed in pilot due to small sample)
    # Key question: does LLaDA show a similar trend?

    print(f"\n--- Pass Criteria ---")
    print(f"  All 4 methods ran successfully:  PASS")
    print(f"  Countdown DTA >= vanilla:        {'PASS' if cd_direction else 'FAIL'} "
          f"(DTA={cd_dta_acc:.1%} vs Vanilla={cd_vanilla_acc:.1%})")
    print(f"  GSM8K DTA >= vanilla:            {'PASS' if gsm_direction else 'FAIL'} "
          f"(DTA={gsm_dta_acc:.1%} vs Vanilla={gsm_vanilla_acc:.1%})")
    print(f"  DTA improvement on at least one: {'PASS' if either_positive else 'FAIL'}")

    # Cross-model consistency check
    print(f"\n--- Cross-Model Consistency ---")
    print(f"  Dream pilot: Vanilla=12.5%, DTA=6.25%, ReMDM=6.25%, DTA+ReMDM=6.25%")
    print(f"  LLaDA pilot: Vanilla={cd_vanilla_acc:.1%}, DTA={cd_dta_acc:.1%}, "
          f"ReMDM={cd_summaries_nolora['remdm_conf']['accuracy']:.1%}, "
          f"DTA+ReMDM={cd_summaries_nolora['dta_remdm']['accuracy']:.1%}")

    overall = "GO" if either_positive else "CONDITIONAL-GO"
    print(f"\n  Overall: {overall}")
    print(f"  Wall-clock: {wall_clock:.0f}s ({wall_clock/60:.1f}min)")

    # ── Save results ──
    output = {
        "task": "task_8a",
        "mode": "pilot",
        "model": MODEL_NAME,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "mask_token_id": MASK_TOKEN_ID,
        "benchmarks": {
            "countdown": {
                "summaries": cd_summaries_nolora,
                "results": cd_results_nolora,
            },
            "gsm8k": {
                "summaries": gsm_summaries,
                "results": gsm_results,
            },
        },
        "configs": {
            "dta": {"rank": LORA_RANK, "layers": LORA_LAYERS, "lr": LORA_LR,
                    "gamma": LORA_GAMMA, "warmup": WARMUP_FRAC, "alpha": LORA_ALPHA},
            "remdm": {"remask_ratio": REMASK_RATIO, "stop_frac": REMASK_STOP_FRAC},
        },
        "pass_criteria": {
            "cd_dta_better": cd_direction,
            "gsm_dta_better": gsm_direction,
            "either_positive": either_positive,
            "overall": overall,
        },
        "cross_model_comparison": {
            "dream_pilot": {
                "countdown_vanilla_acc": 0.125,
                "countdown_dta_acc": 0.0625,
                "countdown_remdm_acc": 0.0625,
                "countdown_dta_remdm_acc": 0.0625,
            },
            "llada_pilot": {
                "countdown_vanilla_acc": cd_vanilla_acc,
                "countdown_dta_acc": cd_dta_acc,
                "countdown_remdm_acc": cd_summaries_nolora["remdm_conf"]["accuracy"],
                "countdown_dta_remdm_acc": cd_summaries_nolora["dta_remdm"]["accuracy"],
                "gsm8k_vanilla_acc": gsm_vanilla_acc,
                "gsm8k_dta_acc": gsm_dta_acc,
            },
        },
        "wall_clock_s": wall_clock,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "torch_version": torch.__version__,
    }

    # Save to both locations
    out_file = CROSS_MODEL_DIR / "llada_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    pilot_file = RESULTS_DIR / "task_8a_llada_cross_model.json"
    with open(pilot_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Pilot results saved to {pilot_file}")

    # Write DONE marker
    mark_task_done("task_8a",
                   "/home/ccwang/sibyl_system/exp/results/pilots",
                   status="success" if either_positive else "conditional",
                   summary=f"CD: vanilla={cd_vanilla_acc:.1%} dta={cd_dta_acc:.1%} | "
                           f"GSM: vanilla={gsm_vanilla_acc:.1%} dta={gsm_dta_acc:.1%}")


if __name__ == "__main__":
    main()
