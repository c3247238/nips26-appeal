"""
Task 2a (PILOT): DTA Core Implementation + Countdown Pilot (16 samples).

Implements Denoising-Time Adaptation (DTA):
  - During Dream-7B denoising, after each step reveals tokens (E-step),
    compute masked LM loss on revealed tokens and update LoRA params (M-step).
  - LoRA: rank=4, last 2 layers FFN (gate_proj, up_proj, down_proj)
  - Hyperparams: lr=1e-4, gamma=0.95, warmup=20%

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 task_2a_dta_pilot.py
"""
import os, sys, json, time, random, re, math, copy
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

# DTA Hyperparams
LORA_RANK = 4
LORA_LAYERS = 2        # Last N layers
LORA_LR = 5e-4         # Balanced: enough signal, no degradation
LORA_GAMMA = 0.95      # Decay factor to prevent drift
WARMUP_FRAC = 0.20     # First 20% steps: no DTA
LORA_ALPHA = 1.0       # LoRA scaling (alpha / rank)
GRAD_CLIP = 1.0        # Gradient norm clipping


# ──────────────────────────────────────────────────
# LoRA Implementation (lightweight, no PEFT dependency)
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

        # Initialize A with kaiming, B with zeros -> zero output at init
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=original.weight.dtype, device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=original.weight.dtype, device=original.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero -> initial output is zero (bypass init)

        # Freeze original
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def reset_parameters(self):
        """Reset LoRA params to zero (restore original behavior)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_delta_norm(self):
        """Return Frobenius norm of the effective delta weight."""
        with torch.no_grad():
            # Compute in float32 for numerical precision
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            return delta.norm().item()

    def get_param_norms(self):
        """Return norms of A and B separately for debugging."""
        with torch.no_grad():
            return self.lora_A.float().norm().item(), self.lora_B.float().norm().item()


def inject_lora(model, n_layers=2, rank=4, alpha=1.0):
    """Inject LoRA into the last n_layers' MLP (gate_proj, up_proj, down_proj).

    Returns list of LoRA parameters for optimizer, and list of LoRALayer modules.
    """
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
    """Get Frobenius norms of all LoRA delta weights."""
    return [m.get_delta_norm() for m in lora_modules]


def decay_lora_params(lora_params, gamma):
    """Apply decay: Δθ <- γ * Δθ (in-place)."""
    with torch.no_grad():
        for p in lora_params:
            p.mul_(gamma)


def reset_lora_params(lora_modules):
    """Reset all LoRA modules to zero."""
    for m in lora_modules:
        m.reset_parameters()


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
# DTA Generation: Custom Denoising Loop with LoRA Updates
# ──────────────────────────────────────────────────

def sample_tokens_from_logits(logits, temperature=1.0, top_p=None, top_k=None):
    """Sample tokens from logits with temperature, top-p, top-k."""
    if temperature > 0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = top_k_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, -float('inf'))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float('inf'))

    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return confidence, sampled


def dta_generate(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None,
                 gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
                 warmup_frac=WARMUP_FRAC, gamma=LORA_GAMMA):
    """
    Generate with DTA: custom denoising loop with LoRA updates.

    E-step: Standard denoising (predict & reveal tokens)
    M-step: Compute masked LM loss on revealed tokens, update LoRA

    Returns: generated text, elapsed time, per-step diagnostics
    """
    # Prepare input
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    max_length = prompt_len + gen_len
    eps = 1e-3  # from Dream default

    # Pad with mask tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    warmup_steps = int(steps * warmup_frac)

    # Reset LoRA before each generation (each sample gets fresh LoRA)
    for m in lora_modules:
        m.reset_parameters()
    # Reset optimizer state (momentum, variance) for fresh sample
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

        # === E-step: Forward pass & token sampling ===
        # We need gradients only for M-step, but forward must be done
        # in no_grad for E-step sampling to avoid memory explosion
        with torch.no_grad():
            outputs = model(x, attention_mask="full", position_ids=None)
            logits = outputs.logits
            # Dream uses shifted logits: cat([logits[:,:1], logits[:,:-1]], dim=1)
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]

        # Origin algorithm: randomly reveal tokens
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

        # Count newly revealed tokens
        newly_revealed = ((x_old == MASK_TOKEN_ID) & (x != MASK_TOKEN_ID))
        n_revealed = newly_revealed.sum().item()
        # Total revealed so far (non-mask, non-prompt)
        revealed_mask = (x[0, prompt_len:] != MASK_TOKEN_ID)
        total_revealed = revealed_mask.sum().item()

        # === M-step: LoRA update on revealed tokens ===
        did_update = False
        lora_loss_val = 0.0
        lora_norms_before = get_lora_delta_norms(lora_modules)

        if i >= warmup_steps and total_revealed >= 5:
            # Get indices of revealed (non-mask) positions in generation region
            gen_region = x[0, prompt_len:]  # shape [gen_len]
            revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]

            if len(revealed_positions) >= 5:
                # M-step: Masked re-prediction (self-supervised)
                # Mask ~20% of revealed tokens, predict from remaining context
                # This forces cross-position information flow through LoRA
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

                # Loss on masked positions
                loss_logits = logits_m[0, mask_positions]
                loss = F.cross_entropy(loss_logits, target_tokens)

                # Backward & update with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                lora_loss_val = loss.item()
                did_update = True

                # Restore x (don't keep the masking)
                # x was not modified since x_masked is a clone

                # Decay LoRA params
                decay_lora_params(lora_params, gamma)

        lora_norms_after = get_lora_delta_norms(lora_modules)

        # Debug: print param norms at key steps of first 2 samples
        if did_update and (i == warmup_steps or i == warmup_steps + 5 or i == STEPS - 1):
            a_norms = [m.get_param_norms()[0] for m in lora_modules]
            b_norms = [m.get_param_norms()[1] for m in lora_modules]
            if not hasattr(dta_generate, '_debug_count'):
                dta_generate._debug_count = 0
            dta_generate._debug_count += 1
            if dta_generate._debug_count <= 6:
                print(f"    [DEBUG step {i}] loss={lora_loss_val:.4f}, "
                      f"A_norm={np.mean(a_norms):.6f}, B_norm={np.mean(b_norms):.6f}, "
                      f"delta_norm={np.mean(lora_norms_after):.6f}")

        step_diagnostics.append({
            "step": i,
            "n_masked": n_masked,
            "n_revealed_this_step": n_revealed,
            "total_revealed": total_revealed,
            "did_dta_update": did_update,
            "lora_loss": lora_loss_val,
            "lora_norm_max_before": max(lora_norms_before) if lora_norms_before else 0,
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
# Also generate vanilla for direct comparison
# ──────────────────────────────────────────────────

def vanilla_generate(model, tokenizer, prompt_text, device="cuda:0"):
    """Vanilla generation (no DTA) using same custom loop for fair comparison."""
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
    print(f"=== Task 2a PILOT: DTA Core + Countdown ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"DTA: rank={LORA_RANK}, layers={LORA_LAYERS}, lr={LORA_LR}, "
          f"gamma={LORA_GAMMA}, warmup={WARMUP_FRAC}")
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

    # ── Phase 1: Vanilla baseline (same seed, same loop) ──
    print(f"\n{'='*60}")
    print(f"Phase 1: Vanilla Baseline (for direct comparison)")
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

    # ── Phase 2: DTA generation ──
    print(f"\n{'='*60}")
    print(f"Phase 2: DTA Generation")
    print(f"{'='*60}")

    # Re-seed for fair comparison
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Inject LoRA
    lora_params, lora_modules = inject_lora(
        model, n_layers=LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA
    )

    # Create optimizer (AdamW works better for online few-step learning)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    dta_results = []
    dta_correct = 0
    all_step_diagnostics = []

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)

        text, elapsed, step_diag = dta_generate(
            model, tokenizer, prompt_text, device,
            lora_params=lora_params, lora_modules=lora_modules,
            optimizer=optimizer,
        )

        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            dta_correct += 1

        # Collect LoRA norm trajectory for this sample
        lora_norms = [s["lora_norm_max_after"] for s in step_diag]
        lora_losses = [s["lora_loss"] for s in step_diag if s["did_dta_update"]]
        n_updates = sum(1 for s in step_diag if s["did_dta_update"])

        dta_results.append({
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
            "max_lora_norm": max(lora_norms) if lora_norms else 0,
            "final_lora_norm": lora_norms[-1] if lora_norms else 0,
            "avg_dta_loss": float(np.mean(lora_losses)) if lora_losses else 0,
        })

        all_step_diagnostics.append({
            "sample_idx": i,
            "steps": step_diag,
        })

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        max_norm = max(lora_norms) if lora_norms else 0
        avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
        print(f"  [D {i+1}/{N_SAMPLES}] {status} | target={problem['target']} | eq={eq} | "
              f"time={elapsed:.1f}s | updates={n_updates} | norm={max_norm:.4f} | loss={avg_loss:.3f}")

    dta_acc = dta_correct / N_SAMPLES

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 2a Pilot - DTA vs Vanilla on Countdown")
    print(f"{'='*70}")
    print(f"  Vanilla:  {vanilla_correct}/{N_SAMPLES} = {vanilla_acc:.1%}")
    print(f"  DTA:      {dta_correct}/{N_SAMPLES} = {dta_acc:.1%}")
    print(f"  Delta:    {dta_acc - vanilla_acc:+.1%}")

    # Vanilla metrics
    v_d2 = float(np.mean([r["distinct_2"] for r in vanilla_results]))
    v_r3 = float(np.mean([r["rep_3"] for r in vanilla_results]))
    v_time = float(np.mean([r["gen_time_s"] for r in vanilla_results]))

    # DTA metrics
    d_d2 = float(np.mean([r["distinct_2"] for r in dta_results]))
    d_r3 = float(np.mean([r["rep_3"] for r in dta_results]))
    d_time = float(np.mean([r["gen_time_s"] for r in dta_results]))
    d_max_norms = [r["max_lora_norm"] for r in dta_results]
    d_losses = [r["avg_dta_loss"] for r in dta_results if r["avg_dta_loss"] > 0]

    print(f"\n  Vanilla: distinct-2={v_d2:.3f}, rep-3={v_r3:.3f}, time={v_time:.1f}s")
    print(f"  DTA:     distinct-2={d_d2:.3f}, rep-3={d_r3:.3f}, time={d_time:.1f}s")
    print(f"  DTA LoRA max norms: min={min(d_max_norms):.4f}, max={max(d_max_norms):.4f}, "
          f"mean={np.mean(d_max_norms):.4f}")
    if d_losses:
        print(f"  DTA avg losses: min={min(d_losses):.3f}, max={max(d_losses):.3f}, "
              f"mean={np.mean(d_losses):.3f}")

    # Pass criteria
    print(f"\n--- Pass Criteria ---")
    no_error = all(len(r["generated_text"].strip()) > 0 for r in dta_results)
    norm_ok = all(n <= 1.0 for n in d_max_norms)
    acc_ok = dta_acc >= vanilla_acc
    d2_ok = d_d2 >= 0.7

    print(f"  DTA runs without errors:         {'PASS' if no_error else 'FAIL'}")
    print(f"  ||Δθ||_F <= 1.0:                 {'PASS' if norm_ok else 'FAIL'} (max={max(d_max_norms):.4f})")
    print(f"  DTA acc >= vanilla ({vanilla_acc:.1%}): {'PASS' if acc_ok else 'FAIL'} ({dta_acc:.1%})")
    print(f"  distinct-2 >= 0.7:               {'PASS' if d2_ok else 'FAIL'} ({d_d2:.3f})")

    overall = "GO" if (no_error and norm_ok and acc_ok and d2_ok) else "CONDITIONAL-GO" if no_error else "NO-GO"
    if not acc_ok and no_error:
        overall = "CONDITIONAL-GO (accuracy below vanilla, but runs correctly)"
    print(f"  Overall: {overall}")

    # ── Save results ──
    summary = {
        "task": "task_2a",
        "mode": "pilot",
        "method": "dta",
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
        "vanilla_accuracy": vanilla_acc,
        "vanilla_correct": vanilla_correct,
        "dta_accuracy": dta_acc,
        "dta_correct": dta_correct,
        "total_count": N_SAMPLES,
        "delta_accuracy": dta_acc - vanilla_acc,
        "vanilla_distinct_2_mean": v_d2,
        "dta_distinct_2_mean": d_d2,
        "vanilla_rep_3_mean": v_r3,
        "dta_rep_3_mean": d_r3,
        "vanilla_avg_time_s": v_time,
        "dta_avg_time_s": d_time,
        "time_overhead": d_time / v_time if v_time > 0 else 0,
        "lora_norm_max": float(max(d_max_norms)),
        "lora_norm_mean": float(np.mean(d_max_norms)),
        "pass_criteria": {
            "no_error": no_error,
            "norm_ok": norm_ok,
            "acc_ok": acc_ok,
            "d2_ok": d2_ok,
            "overall": overall,
        },
        "vanilla_results": vanilla_results,
        "dta_results": dta_results,
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / "task_2a_dta_pilot.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Save step diagnostics separately (large)
    diag_file = RESULTS_DIR / "task_2a_dta_step_diagnostics.json"
    with open(diag_file, "w") as f:
        json.dump(all_step_diagnostics, f, indent=2)
    print(f"Step diagnostics saved to {diag_file}")


if __name__ == "__main__":
    main()
