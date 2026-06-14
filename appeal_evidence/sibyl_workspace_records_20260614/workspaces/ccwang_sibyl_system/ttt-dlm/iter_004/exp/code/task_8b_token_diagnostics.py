"""
Task 8b (PILOT): Token-Level Diagnostic Analysis (16 samples).

Records per-step token change trajectories for Vanilla, ReMDM-conf, SCP, and DTA.
Computes Correction Precision/Recall for remasking methods.

Key metrics:
  - Correction Precision: fraction of remasked tokens that were actually wrong
  - Correction Recall: fraction of wrong tokens that were remasked
  - Token change rate: fraction of tokens that change between consecutive steps
  - Trajectory stability: how many times each position changes value

Hypotheses tested:
  H9: ReMDM-conf Correction Precision < 50%
  H6: SCP Correction Precision > ReMDM-conf Correction Precision

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 task_8b_token_diagnostics.py
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
# Trajectory-Recording Generation Functions
# ──────────────────────────────────────────────────

def generate_vanilla_with_trajectory(model, tokenizer, prompt_text, device="cuda:0"):
    """Vanilla generation recording token state at each step."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)

    # trajectory: list of gen_region snapshots (as lists of ints)
    trajectory = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                trajectory.append(x[0, prompt_len:].tolist())
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

            trajectory.append(x[0, prompt_len:].tolist())

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, trajectory


def generate_remdm_with_trajectory(model, tokenizer, prompt_text, device="cuda:0"):
    """ReMDM-conf with per-step trajectory and remask event logging."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    remask_stop_step = int(STEPS * REMASK_STOP_FRAC)

    trajectory = []
    remask_events = []  # list of (step, [positions remasked])

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                trajectory.append(x[0, prompt_len:].tolist())
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

            # Record state BEFORE remasking (to identify what gets remasked)
            pre_remask_state = x[0, prompt_len:].clone()

            # ReMDM-conf: remask low-confidence revealed tokens
            remasked_positions = []
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
                    remask_pos = revealed_positions[lowest_idx]
                    # Record what tokens were at those positions before remasking
                    remasked_tokens = pre_remask_state[remask_pos].tolist()
                    remasked_positions = remask_pos.tolist()
                    x[0, remask_pos + prompt_len] = MASK_TOKEN_ID

            if remasked_positions:
                remask_events.append({
                    "step": i,
                    "positions": remasked_positions,
                    "tokens_before_remask": remasked_tokens,
                })

            trajectory.append(x[0, prompt_len:].tolist())

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, trajectory, remask_events


def generate_scp_with_trajectory(model, tokenizer, prompt_text, device="cuda:0"):
    """SCP with per-step trajectory and contradiction/remask event logging."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * SCP_WARMUP_FRAC)
    stop_step = int(STEPS * SCP_STOP_FRAC)

    trajectory = []
    remask_events = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                trajectory.append(x[0, prompt_len:].tolist())
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

            pre_remask_state = x[0, prompt_len:].clone()

            # SCP: leave-one-out probing
            remasked_positions = []
            contradictions_found = []
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

                    contradictions_found = contradictions

                    if contradictions:
                        n_remask = min(len(contradictions), max(1, len(revealed_positions) // 8))
                        remask_pos_list = contradictions[:n_remask]
                        remasked_tokens = [pre_remask_state[p].item() for p in remask_pos_list]
                        remasked_positions = remask_pos_list
                        for rp in remask_pos_list:
                            x[0, rp + prompt_len] = MASK_TOKEN_ID

            if remasked_positions or contradictions_found:
                remask_events.append({
                    "step": i,
                    "contradictions_found": contradictions_found,
                    "positions_remasked": remasked_positions,
                    "tokens_before_remask": [pre_remask_state[p].item() for p in remasked_positions] if remasked_positions else [],
                })

            trajectory.append(x[0, prompt_len:].tolist())

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, trajectory, remask_events


def generate_dta_with_trajectory(model, tokenizer, prompt_text, device="cuda:0",
                                  lora_params=None, lora_modules=None, optimizer=None):
    """DTA with per-step trajectory and LoRA norm tracking."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)

    reset_lora(lora_modules, optimizer)

    trajectory = []
    lora_norms = []
    lora_losses = []
    n_updates = 0

    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            trajectory.append(x[0, prompt_len:].tolist())
            lora_norms.append(max(m.get_delta_norm() for m in lora_modules))
            break

        # E-step
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

        # M-step
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

        cur_norm = max(m.get_delta_norm() for m in lora_modules) if lora_modules else 0
        lora_norms.append(cur_norm)
        trajectory.append(x[0, prompt_len:].tolist())

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed, trajectory, lora_norms, lora_losses, n_updates


# ──────────────────────────────────────────────────
# Diagnostic Analysis Functions
# ──────────────────────────────────────────────────

def compute_trajectory_stats(trajectory, mask_token_id=MASK_TOKEN_ID):
    """Compute per-step statistics from a token trajectory."""
    n_steps = len(trajectory)
    if n_steps == 0:
        return {}
    gen_len = len(trajectory[0])

    # Per-position: how many times did the token change?
    position_changes = [0] * gen_len
    # Per-step: token change rate (fraction of non-mask positions that changed)
    step_change_rates = []
    # Per-step: fraction revealed
    step_revealed_frac = []

    for s in range(n_steps):
        revealed = sum(1 for t in trajectory[s] if t != mask_token_id)
        step_revealed_frac.append(revealed / gen_len)

        if s > 0:
            changes = 0
            non_mask_count = 0
            for p in range(gen_len):
                cur = trajectory[s][p]
                prev = trajectory[s-1][p]
                if cur != mask_token_id or prev != mask_token_id:
                    non_mask_count += 1
                    if cur != prev:
                        changes += 1
                        position_changes[p] += 1
            step_change_rates.append(changes / max(non_mask_count, 1))

    # Stability: positions that never changed after first reveal
    stable_positions = sum(1 for c in position_changes if c <= 1)
    unstable_positions = sum(1 for c in position_changes if c > 1)
    highly_unstable = sum(1 for c in position_changes if c > 5)

    return {
        "n_steps_recorded": n_steps,
        "avg_change_rate": float(np.mean(step_change_rates)) if step_change_rates else 0,
        "max_change_rate": float(np.max(step_change_rates)) if step_change_rates else 0,
        "stable_positions": stable_positions,
        "unstable_positions": unstable_positions,
        "highly_unstable_positions": highly_unstable,
        "final_revealed_frac": step_revealed_frac[-1] if step_revealed_frac else 0,
        "position_change_histogram": {
            "0_changes": sum(1 for c in position_changes if c == 0),
            "1_change": sum(1 for c in position_changes if c == 1),
            "2-3_changes": sum(1 for c in position_changes if 2 <= c <= 3),
            "4-10_changes": sum(1 for c in position_changes if 4 <= c <= 10),
            "10+_changes": sum(1 for c in position_changes if c > 10),
        },
        "step_change_rate_curve": step_change_rates[:20] + step_change_rates[-5:] if len(step_change_rates) > 25 else step_change_rates,
        "step_revealed_frac_curve": step_revealed_frac[:20] + step_revealed_frac[-5:] if len(step_revealed_frac) > 25 else step_revealed_frac,
    }


def compute_correction_metrics(trajectory, remask_events, mask_token_id=MASK_TOKEN_ID):
    """
    Compute Correction Precision and Recall for remasking methods.

    "Oracle" = final output tokens. A token at position p is "wrong" at step s
    if it differs from the final token at position p.

    Correction Precision = |remasked AND wrong| / |remasked|
    Correction Recall = |remasked AND wrong| / |wrong at that step|
    """
    if not trajectory or not remask_events:
        return {
            "correction_precision": None,
            "correction_recall": None,
            "total_remasked": 0,
            "total_wrong_remasked": 0,
            "total_correct_remasked": 0,
            "total_wrong_at_remask_steps": 0,
        }

    final_state = trajectory[-1]
    gen_len = len(final_state)

    total_remasked = 0
    total_wrong_remasked = 0
    total_correct_remasked = 0
    total_wrong_at_remask_steps = 0

    per_step_precision = []
    per_step_recall = []

    for event in remask_events:
        step = event["step"]
        # Get positions that were remasked
        if "positions" in event:
            positions = event["positions"]
            tokens_before = event.get("tokens_before_remask", [])
        elif "positions_remasked" in event:
            positions = event["positions_remasked"]
            tokens_before = event.get("tokens_before_remask", [])
        else:
            continue

        if not positions:
            continue

        # Get the state just before remasking (the trajectory state at this step
        # was recorded AFTER remasking, so we look at state before)
        # We use tokens_before_remask which was recorded
        if step < len(trajectory):
            # Count wrong tokens among revealed at this step
            # We need the pre-remask state. For ReMDM, the trajectory was recorded
            # after remasking. We reconstruct pre-remask state from trajectory[step]
            # by restoring remasked positions with their original tokens.
            state_after_remask = trajectory[step] if step < len(trajectory) else trajectory[-1]
            pre_remask = list(state_after_remask)
            for idx_p, pos in enumerate(positions):
                if idx_p < len(tokens_before) and pos < gen_len:
                    pre_remask[pos] = tokens_before[idx_p]

            # Count how many revealed tokens are wrong (differ from final)
            n_wrong_at_step = 0
            for p in range(gen_len):
                if pre_remask[p] != mask_token_id and pre_remask[p] != final_state[p]:
                    n_wrong_at_step += 1

            # Count remasked tokens that were wrong
            step_wrong_remasked = 0
            step_correct_remasked = 0
            for idx_p, pos in enumerate(positions):
                if pos >= gen_len:
                    continue
                if idx_p < len(tokens_before):
                    token = tokens_before[idx_p]
                else:
                    continue
                total_remasked += 1
                if token != final_state[pos]:
                    step_wrong_remasked += 1
                    total_wrong_remasked += 1
                else:
                    step_correct_remasked += 1
                    total_correct_remasked += 1

            total_wrong_at_remask_steps += n_wrong_at_step

            # Per-step precision
            n_remasked_this_step = len(positions)
            if n_remasked_this_step > 0:
                per_step_precision.append(step_wrong_remasked / n_remasked_this_step)
            if n_wrong_at_step > 0:
                per_step_recall.append(step_wrong_remasked / n_wrong_at_step)

    overall_precision = total_wrong_remasked / total_remasked if total_remasked > 0 else None
    overall_recall = total_wrong_remasked / total_wrong_at_remask_steps if total_wrong_at_remask_steps > 0 else None

    return {
        "correction_precision": overall_precision,
        "correction_recall": overall_recall,
        "total_remasked": total_remasked,
        "total_wrong_remasked": total_wrong_remasked,
        "total_correct_remasked": total_correct_remasked,
        "total_wrong_at_remask_steps": total_wrong_at_remask_steps,
        "per_step_precision_mean": float(np.mean(per_step_precision)) if per_step_precision else None,
        "per_step_precision_std": float(np.std(per_step_precision)) if per_step_precision else None,
        "per_step_recall_mean": float(np.mean(per_step_recall)) if per_step_recall else None,
        "per_step_recall_std": float(np.std(per_step_recall)) if per_step_recall else None,
        "n_remask_events": len(remask_events),
    }


def compute_dta_future_prediction_accuracy(trajectory, lora_norms, mask_token_id=MASK_TOKEN_ID):
    """
    Analyze whether DTA's LoRA norms correlate with prediction accuracy improvement.
    Check H7 proxy: do tokens stabilize more as LoRA learns?
    """
    if not trajectory or len(trajectory) < 10:
        return {}

    final_state = trajectory[-1]
    gen_len = len(final_state)

    # At each step, compute fraction of revealed tokens that match final output
    accuracy_curve = []
    for s, state in enumerate(trajectory):
        correct = 0
        revealed = 0
        for p in range(gen_len):
            if state[p] != mask_token_id:
                revealed += 1
                if state[p] == final_state[p]:
                    correct += 1
        accuracy_curve.append(correct / max(revealed, 1))

    # Split into early (first 25%) and late (last 25%) phases
    n = len(accuracy_curve)
    early = accuracy_curve[:n//4]
    late = accuracy_curve[3*n//4:]

    return {
        "early_phase_accuracy_mean": float(np.mean(early)) if early else None,
        "late_phase_accuracy_mean": float(np.mean(late)) if late else None,
        "accuracy_improvement": (float(np.mean(late)) - float(np.mean(early))) if early and late else None,
        "accuracy_curve_sampled": accuracy_curve[::max(1, n//20)],
        "lora_norms_sampled": lora_norms[::max(1, len(lora_norms)//20)] if lora_norms else [],
        "is_monotonically_improving": all(accuracy_curve[i] <= accuracy_curve[i+1] for i in range(len(accuracy_curve)-1)) if len(accuracy_curve) > 1 else None,
    }


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    import sys as _sys
    LOG_FILE = RESULTS_DIR / "task_8b_log.txt"

    class TeeOutput:
        def __init__(self, file_path):
            self.terminal = _sys.stdout
            self.log = open(file_path, "w", buffering=1)
        def write(self, msg):
            self.terminal.write(msg)
            self.log.write(msg)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    _sys.stdout = TeeOutput(LOG_FILE)
    _sys.stderr = TeeOutput(RESULTS_DIR / "task_8b_err.txt")

    device = "cuda:0"
    print(f"[task_8b] Token-Level Diagnostic Analysis")
    print(f"[task_8b] Device: {device}")
    print(f"[task_8b] N_SAMPLES={N_SAMPLES}, STEPS={STEPS}, SEED={SEED}")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"[task_8b] Generated {len(problems)} Countdown problems")

    # Load model
    print("[task_8b] Loading model...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print("[task_8b] Model loaded")

    all_results = {
        "meta": {
            "task": "task_8b",
            "timestamp": datetime.now().isoformat(),
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "steps": STEPS,
            "gen_len": GEN_LEN,
            "mode": "pilot",
        },
        "methods": {},
    }

    # ──── Method 1: Vanilla ────
    print("\n[task_8b] === Vanilla ===")
    vanilla_results = []
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, traj = generate_vanilla_with_trajectory(model, tokenizer, prompt, device)
        verification = verify_countdown_answer(text, problem)
        traj_stats = compute_trajectory_stats(traj)

        vanilla_results.append({
            "idx": idx,
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "trajectory_stats": traj_stats,
            "extracted_equation": verification.get("extracted_equation"),
        })
        status = "OK" if verification["is_correct"] else "X"
        print(f"  [{idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | {elapsed:.1f}s")

    vanilla_correct = sum(1 for r in vanilla_results if r["is_correct"])
    print(f"  Vanilla accuracy: {vanilla_correct}/{N_SAMPLES} ({vanilla_correct/N_SAMPLES*100:.1f}%)")
    all_results["methods"]["vanilla"] = {
        "accuracy": vanilla_correct / N_SAMPLES,
        "n_correct": vanilla_correct,
        "per_sample": vanilla_results,
    }

    gc.collect(); torch.cuda.empty_cache()

    # ──── Method 2: ReMDM-conf ────
    print("\n[task_8b] === ReMDM-conf ===")
    remdm_results = []
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, traj, remask_events = generate_remdm_with_trajectory(model, tokenizer, prompt, device)
        verification = verify_countdown_answer(text, problem)
        traj_stats = compute_trajectory_stats(traj)
        correction = compute_correction_metrics(traj, remask_events)

        remdm_results.append({
            "idx": idx,
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "trajectory_stats": traj_stats,
            "correction_metrics": correction,
            "n_remask_events": len(remask_events),
            "extracted_equation": verification.get("extracted_equation"),
        })
        prec = correction["correction_precision"]
        prec_str = f"{prec:.3f}" if prec is not None else "N/A"
        status = "OK" if verification["is_correct"] else "X"
        print(f"  [{idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | "
              f"CorrPrec={prec_str} | remasks={correction['total_remasked']} | {elapsed:.1f}s")

    remdm_correct = sum(1 for r in remdm_results if r["is_correct"])
    # Aggregate correction metrics
    remdm_precisions = [r["correction_metrics"]["correction_precision"]
                        for r in remdm_results if r["correction_metrics"]["correction_precision"] is not None]
    remdm_recalls = [r["correction_metrics"]["correction_recall"]
                     for r in remdm_results if r["correction_metrics"]["correction_recall"] is not None]
    print(f"  ReMDM-conf accuracy: {remdm_correct}/{N_SAMPLES} ({remdm_correct/N_SAMPLES*100:.1f}%)")
    if remdm_precisions:
        print(f"  Mean Correction Precision: {np.mean(remdm_precisions):.3f} +/- {np.std(remdm_precisions):.3f}")
    if remdm_recalls:
        print(f"  Mean Correction Recall: {np.mean(remdm_recalls):.3f} +/- {np.std(remdm_recalls):.3f}")

    all_results["methods"]["remdm_conf"] = {
        "accuracy": remdm_correct / N_SAMPLES,
        "n_correct": remdm_correct,
        "aggregate_correction_precision_mean": float(np.mean(remdm_precisions)) if remdm_precisions else None,
        "aggregate_correction_precision_std": float(np.std(remdm_precisions)) if remdm_precisions else None,
        "aggregate_correction_recall_mean": float(np.mean(remdm_recalls)) if remdm_recalls else None,
        "aggregate_correction_recall_std": float(np.std(remdm_recalls)) if remdm_recalls else None,
        "per_sample": remdm_results,
    }

    gc.collect(); torch.cuda.empty_cache()

    # ──── Method 3: SCP ────
    print("\n[task_8b] === SCP ===")
    scp_results = []
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, traj, remask_events = generate_scp_with_trajectory(model, tokenizer, prompt, device)
        verification = verify_countdown_answer(text, problem)
        traj_stats = compute_trajectory_stats(traj)
        correction = compute_correction_metrics(traj, remask_events)

        scp_results.append({
            "idx": idx,
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "trajectory_stats": traj_stats,
            "correction_metrics": correction,
            "n_remask_events": len(remask_events),
            "extracted_equation": verification.get("extracted_equation"),
        })
        prec = correction["correction_precision"]
        prec_str = f"{prec:.3f}" if prec is not None else "N/A"
        status = "OK" if verification["is_correct"] else "X"
        print(f"  [{idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | "
              f"CorrPrec={prec_str} | contradictions={correction['total_remasked']} | {elapsed:.1f}s")

    scp_correct = sum(1 for r in scp_results if r["is_correct"])
    scp_precisions = [r["correction_metrics"]["correction_precision"]
                      for r in scp_results if r["correction_metrics"]["correction_precision"] is not None]
    scp_recalls = [r["correction_metrics"]["correction_recall"]
                   for r in scp_results if r["correction_metrics"]["correction_recall"] is not None]
    print(f"  SCP accuracy: {scp_correct}/{N_SAMPLES} ({scp_correct/N_SAMPLES*100:.1f}%)")
    if scp_precisions:
        print(f"  Mean Correction Precision: {np.mean(scp_precisions):.3f} +/- {np.std(scp_precisions):.3f}")
    if scp_recalls:
        print(f"  Mean Correction Recall: {np.mean(scp_recalls):.3f} +/- {np.std(scp_recalls):.3f}")

    all_results["methods"]["scp"] = {
        "accuracy": scp_correct / N_SAMPLES,
        "n_correct": scp_correct,
        "aggregate_correction_precision_mean": float(np.mean(scp_precisions)) if scp_precisions else None,
        "aggregate_correction_precision_std": float(np.std(scp_precisions)) if scp_precisions else None,
        "aggregate_correction_recall_mean": float(np.mean(scp_recalls)) if scp_recalls else None,
        "aggregate_correction_recall_std": float(np.std(scp_recalls)) if scp_recalls else None,
        "per_sample": scp_results,
    }

    gc.collect(); torch.cuda.empty_cache()

    # ──── Inject LoRA for DTA ────
    print("\n[task_8b] Injecting LoRA for DTA...")
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                             rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    # ──── Method 4: DTA ────
    print("\n[task_8b] === DTA ===")
    dta_results = []
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed, traj, lora_norms, losses, n_updates = generate_dta_with_trajectory(
            model, tokenizer, prompt, device,
            lora_params=lora_params, lora_modules=lora_modules, optimizer=optimizer
        )
        verification = verify_countdown_answer(text, problem)
        traj_stats = compute_trajectory_stats(traj)
        future_acc = compute_dta_future_prediction_accuracy(traj, lora_norms)

        dta_results.append({
            "idx": idx,
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "trajectory_stats": traj_stats,
            "future_prediction_accuracy": future_acc,
            "n_lora_updates": n_updates,
            "max_lora_norm": max(lora_norms) if lora_norms else 0,
            "avg_lora_loss": float(np.mean(losses)) if losses else 0,
            "extracted_equation": verification.get("extracted_equation"),
        })
        status = "OK" if verification["is_correct"] else "X"
        early_acc = future_acc.get("early_phase_accuracy_mean", 0)
        late_acc = future_acc.get("late_phase_accuracy_mean", 0)
        print(f"  [{idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | "
              f"early_acc={early_acc:.3f} late_acc={late_acc:.3f} | "
              f"updates={n_updates} | {elapsed:.1f}s")

    dta_correct = sum(1 for r in dta_results if r["is_correct"])
    print(f"  DTA accuracy: {dta_correct}/{N_SAMPLES} ({dta_correct/N_SAMPLES*100:.1f}%)")

    all_results["methods"]["dta"] = {
        "accuracy": dta_correct / N_SAMPLES,
        "n_correct": dta_correct,
        "per_sample": dta_results,
    }

    # ──── Hypothesis Testing ────
    print("\n" + "="*60)
    print("[task_8b] HYPOTHESIS TESTING")
    print("="*60)

    # H9: ReMDM-conf Correction Precision < 50%
    h9_result = None
    if remdm_precisions:
        h9_mean = np.mean(remdm_precisions)
        h9_result = h9_mean < 0.50
        print(f"\n  H9: ReMDM-conf Correction Precision < 50%")
        print(f"      Mean Precision = {h9_mean:.3f}")
        print(f"      Result: {'SUPPORTED' if h9_result else 'NOT SUPPORTED'}")

    # H6: SCP Correction Precision > ReMDM-conf Correction Precision
    h6_result = None
    if scp_precisions and remdm_precisions:
        scp_mean = np.mean(scp_precisions)
        remdm_mean = np.mean(remdm_precisions)
        h6_result = scp_mean > remdm_mean
        print(f"\n  H6: SCP Correction Precision > ReMDM-conf Correction Precision")
        print(f"      SCP Precision = {scp_mean:.3f}, ReMDM Precision = {remdm_mean:.3f}")
        print(f"      Difference = {scp_mean - remdm_mean:+.3f}")
        print(f"      Result: {'SUPPORTED' if h6_result else 'NOT SUPPORTED'}")

    # Trajectory stability comparison
    print(f"\n  Trajectory Stability Comparison:")
    for method_name, results_list in [("vanilla", vanilla_results), ("remdm_conf", remdm_results),
                                       ("scp", scp_results), ("dta", dta_results)]:
        avg_changes = np.mean([r["trajectory_stats"]["avg_change_rate"] for r in results_list])
        avg_unstable = np.mean([r["trajectory_stats"]["unstable_positions"] for r in results_list])
        print(f"    {method_name:12s}: avg_change_rate={avg_changes:.4f}, avg_unstable_pos={avg_unstable:.1f}")

    # DTA information accumulation (H7 proxy)
    print(f"\n  DTA Information Accumulation (H7 proxy):")
    improvements = [r["future_prediction_accuracy"].get("accuracy_improvement", 0) for r in dta_results
                   if r["future_prediction_accuracy"].get("accuracy_improvement") is not None]
    if improvements:
        print(f"    Mean accuracy improvement (late - early): {np.mean(improvements):+.4f}")
        print(f"    All positive: {all(i > 0 for i in improvements)}")

    all_results["hypothesis_tests"] = {
        "H9_remdm_precision_below_50pct": {
            "supported": h9_result,
            "remdm_precision_mean": float(np.mean(remdm_precisions)) if remdm_precisions else None,
        },
        "H6_scp_precision_gt_remdm": {
            "supported": h6_result,
            "scp_precision_mean": float(np.mean(scp_precisions)) if scp_precisions else None,
            "remdm_precision_mean": float(np.mean(remdm_precisions)) if remdm_precisions else None,
        },
    }

    # ──── Pass Criteria ────
    print("\n" + "="*60)
    print("[task_8b] PASS CRITERIA")
    print("="*60)

    # 1. Successfully recorded token trajectories
    traj_recorded = all(len(r["trajectory_stats"]) > 0 for r in vanilla_results)
    print(f"  Token trajectories recorded: {'PASS' if traj_recorded else 'FAIL'}")

    # 2. Correction Precision/Recall computed without error
    corr_computed = len(remdm_precisions) > 0 and len(scp_precisions) > 0
    print(f"  Correction metrics computed: {'PASS' if corr_computed else 'FAIL'}")

    overall_pass = traj_recorded and corr_computed
    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    all_results["pass_criteria"] = {
        "trajectories_recorded": traj_recorded,
        "correction_metrics_computed": corr_computed,
        "overall": overall_pass,
    }

    # ──── Save Results ────
    # Trim trajectories from per-sample results to reduce file size
    # (keep only stats, not raw trajectories)
    output_path = RESULTS_DIR / "task_8b_token_diagnostics.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[task_8b] Results saved to {output_path}")

    # Write DONE marker
    done_marker = RESULTS_DIR / "task_8b_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "task_8b",
        "status": "success" if overall_pass else "failed",
        "summary": f"Token diagnostics: traj_ok={traj_recorded}, corr_ok={corr_computed}, "
                   f"H9={'supported' if h9_result else 'not_supported'}, "
                   f"H6={'supported' if h6_result else 'not_supported'}",
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[task_8b] DONE marker written")

    return all_results


if __name__ == "__main__":
    results = main()
