"""
Task 4a (PILOT): DTA + ReMDM-conf Combination on Countdown (16 samples).

Combines DTA (LoRA online update) with ReMDM-conf (confidence-based remasking):
  - E-step: Standard Dream 'origin' denoising reveals tokens
  - ReMDM-conf step: Re-evaluate confidence, remask lowest-confidence tokens
  - M-step: DTA LoRA update on revealed tokens (masked re-prediction loss)

This tests H2: DTA and ReMDM-conf are orthogonally complementary.
DTA improves the model's understanding; ReMDM-conf corrects discrete token choices.

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 task_4a_dta_remdm.py
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

# DTA Hyperparams (same as task_2a)
LORA_RANK = 4
LORA_LAYERS = 2
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# ReMDM-conf Hyperparams (same as task_2b)
REMASK_RATIO = 0.1
REMASK_STOP_FRAC = 0.8


# ──────────────────────────────────────────────────
# LoRA Implementation (same as task_2a)
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

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=original.weight.dtype, device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=original.weight.dtype, device=original.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero -> initial output is zero (bypass init)

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

    def get_param_norms(self):
        with torch.no_grad():
            return self.lora_A.float().norm().item(), self.lora_B.float().norm().item()


def inject_lora(model, n_layers=2, rank=4, alpha=1.0):
    """Inject LoRA into the last n_layers' MLP (gate_proj, up_proj, down_proj)."""
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

    print(f"[DTA] Injected LoRA (rank={rank}) into layers {target_layers}, "
          f"3 projections each = {len(lora_modules)} LoRA modules, "
          f"{sum(p.numel() for p in lora_params)} trainable params")

    return lora_params, lora_modules


def get_lora_delta_norms(lora_modules):
    return [m.get_delta_norm() for m in lora_modules]


def decay_lora_params(lora_params, gamma):
    with torch.no_grad():
        for p in lora_params:
            p.mul_(gamma)


def reset_lora_params(lora_modules):
    for m in lora_modules:
        m.reset_parameters()


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


# ──────────────────────────────────────────────────
# Sampling utility
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
# DTA + ReMDM-conf Combined Generation
# ──────────────────────────────────────────────────

def dta_remdm_generate(model, tokenizer, prompt_text, device="cuda:0",
                       lora_params=None, lora_modules=None, optimizer=None,
                       gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
                       warmup_frac=WARMUP_FRAC, gamma=LORA_GAMMA,
                       remask_ratio=REMASK_RATIO):
    """
    Generate with DTA + ReMDM-conf combined:

    For each denoising step t:
      1. E-step: Standard Dream 'origin' denoising - predict & reveal tokens
      2. ReMDM-conf: Fresh forward pass to re-evaluate confidence of all revealed
         tokens. Remask lowest-confidence tokens back to [MASK].
      3. M-step: DTA LoRA update - mask ~20% of (remaining) revealed tokens,
         compute masked LM loss, update LoRA params via 1-step SGD.

    The key insight is that DTA and ReMDM-conf are ORTHOGONAL:
    - ReMDM-conf improves discrete token selection (corrects bad token choices)
    - DTA improves the model's parametric understanding (learns current context)
    - Running ReMDM-conf BEFORE DTA means DTA learns on higher-quality tokens
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
    warmup_steps = int(steps * warmup_frac)
    stop_remask_step = int(REMASK_STOP_FRAC * steps)

    # Reset LoRA before each generation
    for m in lora_modules:
        m.reset_parameters()
    optimizer.zero_grad()
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            state.clear()
    optimizer.state.clear()

    # Diagnostics
    step_diagnostics = []
    t0 = time.time()

    for i in range(steps):
        mask_index = (x == MASK_TOKEN_ID)
        n_masked = mask_index.sum().item()

        # ═══ Step 1: E-step (standard denoising) ═══
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
        _, sampled_tokens = sample_tokens_from_logits(
            mask_logits[transfer_mask], temperature=temperature
        )
        x0[transfer_mask] = sampled_tokens
        x[mask_index] = x0

        # ═══ Step 2: ReMDM-conf (confidence-based remasking) ═══
        n_remasked = 0
        if i < stop_remask_step and i < steps - 1:
            gen_region = x[0, prompt_len:]
            revealed_mask = (gen_region != MASK_TOKEN_ID)
            n_revealed = revealed_mask.sum().item()

            # Schedule-based: how many should still be masked at time s
            expected_masked = int(math.ceil(s.item() * gen_len))
            current_masked = gen_len - n_revealed
            n_remask = max(0, expected_masked - current_masked)
            n_remask = min(n_remask, max(1, int(remask_ratio * n_revealed)))

            if n_remask > 0 and n_revealed > 1:
                # Fresh forward pass to evaluate confidence of revealed tokens
                with torch.no_grad():
                    logits_fresh = model(x, attention_mask="full", position_ids=None).logits
                    logits_fresh = torch.cat([logits_fresh[:, :1], logits_fresh[:, :-1]], dim=1)

                gen_logits = logits_fresh[0, prompt_len:]
                gen_probs = torch.softmax(gen_logits, dim=-1)

                revealed_positions = torch.where(revealed_mask)[0]
                revealed_tokens = gen_region[revealed_positions]
                revealed_confidence = gen_probs[revealed_positions].gather(
                    1, revealed_tokens.unsqueeze(-1)
                ).squeeze(-1)

                n_remask = min(n_remask, n_revealed - 1)  # keep at least 1

                if n_remask > 0:
                    _, low_conf_idx = torch.topk(revealed_confidence, n_remask, largest=False)
                    positions_to_remask = revealed_positions[low_conf_idx]
                    x[0, prompt_len + positions_to_remask] = MASK_TOKEN_ID
                    n_remasked = n_remask

        # Count current revealed tokens (after remasking)
        revealed_mask_current = (x[0, prompt_len:] != MASK_TOKEN_ID)
        total_revealed = revealed_mask_current.sum().item()

        # ═══ Step 3: M-step (DTA LoRA update) ═══
        did_update = False
        lora_loss_val = 0.0
        lora_norms_before = get_lora_delta_norms(lora_modules)

        if i >= warmup_steps and total_revealed >= 5:
            gen_region_current = x[0, prompt_len:]
            revealed_positions = (gen_region_current != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]

            if len(revealed_positions) >= 5:
                # Mask ~20% of revealed tokens for self-supervised loss
                n_to_mask = max(2, len(revealed_positions) // 5)
                perm = torch.randperm(len(revealed_positions), device=device)[:n_to_mask]
                mask_positions = revealed_positions[perm] + prompt_len

                target_tokens = x[0, mask_positions].clone()
                x_masked = x.clone()
                x_masked[0, mask_positions] = MASK_TOKEN_ID

                # Forward with gradient for LoRA params
                outputs_m = model(x_masked, attention_mask="full", position_ids=None)
                logits_m = outputs_m.logits
                logits_m = torch.cat([logits_m[:, :1], logits_m[:, :-1]], dim=1)

                loss_logits = logits_m[0, mask_positions]
                loss = F.cross_entropy(loss_logits, target_tokens)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                lora_loss_val = loss.item()
                did_update = True

                # Decay LoRA params
                decay_lora_params(lora_params, gamma)

        lora_norms_after = get_lora_delta_norms(lora_modules)

        step_diagnostics.append({
            "step": i,
            "n_masked": n_masked,
            "total_revealed": total_revealed,
            "n_remasked": n_remasked,
            "did_dta_update": did_update,
            "lora_loss": lora_loss_val,
            "lora_norm_max_after": max(lora_norms_after) if lora_norms_after else 0,
            "lora_norm_mean_after": float(np.mean(lora_norms_after)) if lora_norms_after else 0,
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
# Vanilla generation for fair comparison
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
    print(f"=== Task 4a PILOT: DTA + ReMDM-conf on Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"DTA: rank={LORA_RANK}, layers={LORA_LAYERS}, lr={LORA_LR}, "
          f"gamma={LORA_GAMMA}, warmup={WARMUP_FRAC}")
    print(f"ReMDM-conf: remask_ratio={REMASK_RATIO}, stop_frac={REMASK_STOP_FRAC}")
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
    print(f"Dream-7B loaded on {device}")

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Phase 1: Vanilla baseline ──
    print(f"\n{'='*60}")
    print(f"Phase 1: Vanilla Baseline")
    print(f"{'='*60}")

    model.eval()
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

    # ── Phase 2: DTA + ReMDM-conf ──
    print(f"\n{'='*60}")
    print(f"Phase 2: DTA + ReMDM-conf Combined")
    print(f"{'='*60}")

    # Re-seed for fair comparison
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Inject LoRA
    lora_params, lora_modules = inject_lora(
        model, n_layers=LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA
    )

    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    combo_results = []
    combo_correct = 0

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)

        text, elapsed, step_diag = dta_remdm_generate(
            model, tokenizer, prompt_text, device,
            lora_params=lora_params, lora_modules=lora_modules,
            optimizer=optimizer,
        )

        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            combo_correct += 1

        lora_norms = [s["lora_norm_max_after"] for s in step_diag]
        lora_losses = [s["lora_loss"] for s in step_diag if s["did_dta_update"]]
        n_updates = sum(1 for s in step_diag if s["did_dta_update"])
        total_remasked = sum(s["n_remasked"] for s in step_diag)

        combo_results.append({
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
            "n_dta_updates": n_updates,
            "total_remasked": total_remasked,
            "max_lora_norm": max(lora_norms) if lora_norms else 0,
            "final_lora_norm": lora_norms[-1] if lora_norms else 0,
            "avg_dta_loss": float(np.mean(lora_losses)) if lora_losses else 0,
        })

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        max_norm = max(lora_norms) if lora_norms else 0
        avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
        print(f"  [C {i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | updates={n_updates} | remasked={total_remasked} | "
              f"norm={max_norm:.4f} | loss={avg_loss:.3f}")

    combo_acc = combo_correct / N_SAMPLES

    # ── Load reference results for comparison ──
    dta_alone_file = RESULTS_DIR / "task_2a_dta_pilot.json"
    remdm_alone_file = RESULTS_DIR / "task_2b_remdm_conf.json"

    dta_alone_acc = None
    remdm_alone_acc = None

    if dta_alone_file.exists():
        with open(dta_alone_file) as f:
            dta_data = json.load(f)
            dta_alone_acc = dta_data.get("dta_accuracy")

    if remdm_alone_file.exists():
        with open(remdm_alone_file) as f:
            remdm_data = json.load(f)
            remdm_alone_acc = remdm_data.get("accuracy")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 4a Pilot - DTA + ReMDM-conf vs Individual Methods")
    print(f"{'='*70}")
    print(f"  Vanilla:              {vanilla_correct}/{N_SAMPLES} = {vanilla_acc:.1%}")
    if dta_alone_acc is not None:
        print(f"  DTA alone (task_2a):  {dta_alone_acc:.1%}")
    if remdm_alone_acc is not None:
        print(f"  ReMDM alone (task_2b):{remdm_alone_acc:.1%}")
    print(f"  DTA + ReMDM (combo):  {combo_correct}/{N_SAMPLES} = {combo_acc:.1%}")

    max_individual = max(
        dta_alone_acc or 0,
        remdm_alone_acc or 0,
        vanilla_acc
    )
    delta_vs_best = combo_acc - max_individual
    print(f"\n  Best individual:      {max_individual:.1%}")
    print(f"  Combo delta vs best:  {delta_vs_best:+.1%}")

    # Metrics
    v_d2 = float(np.mean([r["distinct_2"] for r in vanilla_results]))
    v_r3 = float(np.mean([r["rep_3"] for r in vanilla_results]))
    v_time = float(np.mean([r["gen_time_s"] for r in vanilla_results]))

    c_d2 = float(np.mean([r["distinct_2"] for r in combo_results]))
    c_r3 = float(np.mean([r["rep_3"] for r in combo_results]))
    c_time = float(np.mean([r["gen_time_s"] for r in combo_results]))
    c_max_norms = [r["max_lora_norm"] for r in combo_results]
    c_losses = [r["avg_dta_loss"] for r in combo_results if r["avg_dta_loss"] > 0]
    c_remasked = [r["total_remasked"] for r in combo_results]

    print(f"\n  Vanilla: distinct-2={v_d2:.3f}, rep-3={v_r3:.3f}, time={v_time:.1f}s")
    print(f"  Combo:   distinct-2={c_d2:.3f}, rep-3={c_r3:.3f}, time={c_time:.1f}s")
    print(f"  Combo LoRA max norms: min={min(c_max_norms):.4f}, max={max(c_max_norms):.4f}, "
          f"mean={np.mean(c_max_norms):.4f}")
    print(f"  Avg remasked tokens: {np.mean(c_remasked):.1f}")

    # Pass criteria
    print(f"\n--- Pass Criteria ---")
    no_error = all(len(r["generated_text"].strip()) > 0 for r in combo_results)
    norm_ok = all(n <= 1.0 for n in c_max_norms)
    # Main criterion: combo >= max(DTA alone, ReMDM alone)
    acc_ok = combo_acc >= max_individual
    d2_ok = c_d2 >= 0.7

    print(f"  Combo runs without errors:           {'PASS' if no_error else 'FAIL'}")
    print(f"  ||delta_theta||_F <= 1.0:            {'PASS' if norm_ok else 'FAIL'} (max={max(c_max_norms):.4f})")
    print(f"  Combo acc >= max individual ({max_individual:.1%}): {'PASS' if acc_ok else 'FAIL'} ({combo_acc:.1%})")
    print(f"  distinct-2 >= 0.7:                   {'PASS' if d2_ok else 'FAIL'} ({c_d2:.3f})")

    if acc_ok and no_error and norm_ok:
        overall = "GO"
    elif no_error and norm_ok:
        overall = f"CONDITIONAL-GO (combo {combo_acc:.1%} < best individual {max_individual:.1%})"
    else:
        overall = "NO-GO"
    print(f"  Overall: {overall}")

    # ── Save results ──
    summary = {
        "task": "task_4a",
        "mode": "pilot",
        "method": "dta_remdm_conf",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "alg": ALG,
        "gen_len": GEN_LEN,
        "dta_config": {
            "lora_rank": LORA_RANK,
            "lora_layers": LORA_LAYERS,
            "lr": LORA_LR,
            "gamma": LORA_GAMMA,
            "warmup_frac": WARMUP_FRAC,
            "lora_alpha": LORA_ALPHA,
        },
        "remdm_config": {
            "remask_ratio": REMASK_RATIO,
            "remask_stop_frac": REMASK_STOP_FRAC,
        },
        "vanilla_accuracy": vanilla_acc,
        "vanilla_correct": vanilla_correct,
        "dta_alone_accuracy": dta_alone_acc,
        "remdm_alone_accuracy": remdm_alone_acc,
        "combo_accuracy": combo_acc,
        "combo_correct": combo_correct,
        "total_count": N_SAMPLES,
        "best_individual_accuracy": max_individual,
        "delta_vs_best_individual": delta_vs_best,
        "vanilla_distinct_2_mean": v_d2,
        "combo_distinct_2_mean": c_d2,
        "vanilla_rep_3_mean": v_r3,
        "combo_rep_3_mean": c_r3,
        "vanilla_avg_time_s": v_time,
        "combo_avg_time_s": c_time,
        "time_overhead_vs_vanilla": c_time / v_time if v_time > 0 else 0,
        "lora_norm_max": float(max(c_max_norms)),
        "lora_norm_mean": float(np.mean(c_max_norms)),
        "avg_total_remasked": float(np.mean(c_remasked)),
        "pass_criteria": {
            "no_error": no_error,
            "norm_ok": norm_ok,
            "acc_ok": acc_ok,
            "d2_ok": d2_ok,
            "overall": overall,
        },
        "vanilla_results": vanilla_results,
        "combo_results": combo_results,
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / "task_4a_dta_remdm.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
