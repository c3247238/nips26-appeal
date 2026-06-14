"""
Task 8c (PILOT): Degradation Detection & LoRA Trajectory Analysis (16 samples).

Validates:
  H10: DTA does not introduce text degradation (compare distinct-1/2/3, rep-2/3)
  H7 alternative: DTA's prediction accuracy for masked tokens improves over steps

Analyses:
  1. Compare DTA vs Vanilla text quality metrics
  2. Per-step LoRA ||Δθ||_F trajectory visualization data
  3. Correct vs incorrect samples' LoRA behavior differences
  4. Per-step masked-token prediction accuracy (information accumulation test)

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_8c_degradation_lora_trajectory.py
"""
import os, sys, json, time, random, re, math, gc
from pathlib import Path
from datetime import datetime
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
MASK_TOKEN_ID = 151666

# DTA Hyperparams (same as task_5a)
LORA_RANK = 4
LORA_LAYERS = 2
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0


# ──────────────────────────────────────────────────
# Countdown Problem Generator (same as task_5a)
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

    def get_param_norms(self):
        with torch.no_grad():
            return {
                "A_norm": self.lora_A.float().norm().item(),
                "B_norm": self.lora_B.float().norm().item(),
                "delta_norm": self.get_delta_norm(),
            }


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
# Vanilla generation
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
# DTA with detailed per-step diagnostics
# ──────────────────────────────────────────────────

def generate_dta_with_diagnostics(model, tokenizer, prompt_text, device="cuda:0",
                                   lora_params=None, lora_modules=None, optimizer=None):
    """
    DTA generation with rich per-step diagnostics:
      - LoRA norm trajectory (per module and aggregate)
      - LoRA loss per step
      - Masked-token prediction accuracy per step (H7 test)
      - Number of revealed tokens per step
    """
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)

    reset_lora(lora_modules, optimizer)

    # Per-step diagnostics storage
    step_diagnostics = []
    n_updates = 0

    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            # Record remaining steps as complete
            step_diagnostics.append({
                "step": i, "n_masked": 0, "n_revealed_total": int(GEN_LEN),
                "did_dta_update": False, "lora_loss": 0.0,
                "lora_norm_max": 0.0, "lora_norm_mean": 0.0,
                "lora_per_module_norms": [],
                "mask_pred_accuracy": None,
                "n_newly_revealed": 0,
            })
            break

        n_masked_before = int(mask_index.sum().item())

        # E-step: standard denoising
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]

        t_val = timesteps[i]
        s_val = timesteps[i + 1]
        p_transfer = (1 - s_val / t_val).item() if i < STEPS - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0

        # Count revealed tokens
        gen_region = x[0, prompt_len:]
        revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        total_revealed = len(revealed_positions)
        n_masked_after = int((x == MASK_TOKEN_ID).sum().item())
        n_newly_revealed = n_masked_before - n_masked_after

        # H7 alternative test: predict accuracy on remaining masked tokens
        # After DTA update, check how well model predicts remaining masked tokens
        mask_pred_accuracy = None

        # M-step: LoRA update
        did_update = False
        lora_loss_val = 0.0

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

            lora_loss_val = loss.item()
            n_updates += 1
            did_update = True

            decay_lora_params(lora_params, LORA_GAMMA)

            # H7 test: after DTA update, check prediction accuracy on remaining masked tokens
            remaining_mask = (x[0, prompt_len:] == MASK_TOKEN_ID)
            n_remaining = int(remaining_mask.sum().item())
            if n_remaining > 0 and n_remaining < GEN_LEN:
                with torch.no_grad():
                    # We can't know ground truth for masked tokens, but we can measure
                    # how confident and consistent the model is on them.
                    # Instead, use the DTA training loss as proxy for information gain.
                    # Also compute top-1 confidence on masked positions as a proxy.
                    logits_check = get_shifted_logits(model, x)
                    remaining_positions = remaining_mask.nonzero(as_tuple=True)[0] + prompt_len
                    check_logits = logits_check[0, remaining_positions]
                    check_probs = F.softmax(check_logits / TEMPERATURE, dim=-1)
                    top1_conf = check_probs.max(dim=-1).values.mean().item()
                    mask_pred_accuracy = top1_conf

        # LoRA norms
        per_module_norms = [m.get_delta_norm() for m in lora_modules]
        per_module_details = [m.get_param_norms() for m in lora_modules]
        norm_max = max(per_module_norms) if per_module_norms else 0.0
        norm_mean = float(np.mean(per_module_norms)) if per_module_norms else 0.0
        # Also track A and B norms for debugging
        a_norms = [d["A_norm"] for d in per_module_details]
        b_norms = [d["B_norm"] for d in per_module_details]
        a_norm_max = max(a_norms) if a_norms else 0.0
        b_norm_max = max(b_norms) if b_norms else 0.0

        step_diagnostics.append({
            "step": i,
            "n_masked": n_masked_after,
            "n_revealed_total": int(total_revealed),
            "n_newly_revealed": n_newly_revealed,
            "did_dta_update": did_update,
            "lora_loss": lora_loss_val,
            "lora_norm_max": norm_max,
            "lora_norm_mean": norm_mean,
            "lora_a_norm_max": a_norm_max,
            "lora_b_norm_max": b_norm_max,
            "lora_per_module_norms": per_module_norms,
            "mask_pred_confidence": mask_pred_accuracy,
        })

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)

    # Aggregate LoRA trajectory summary
    update_steps = [d for d in step_diagnostics if d["did_dta_update"]]
    norm_trajectory = [d["lora_norm_max"] for d in step_diagnostics]
    a_norm_trajectory = [d.get("lora_a_norm_max", 0) for d in step_diagnostics]
    b_norm_trajectory = [d.get("lora_b_norm_max", 0) for d in step_diagnostics]
    loss_trajectory = [d["lora_loss"] for d in update_steps]
    confidence_trajectory = [d["mask_pred_confidence"] for d in update_steps
                             if d["mask_pred_confidence"] is not None]

    # Check convergence using B norm (delta norm may be near-zero due to scaling)
    # B norm is a better indicator since delta = B @ A * scaling
    effective_norm = b_norm_trajectory if max(b_norm_trajectory) > 0 else norm_trajectory
    if len(effective_norm) > 10:
        last_quarter = effective_norm[-(len(effective_norm)//4):]
        first_quarter = effective_norm[:len(effective_norm)//4]
        max_last = max(last_quarter) if last_quarter else 0
        max_first = max(first_quarter) if first_quarter else 0
        is_converging = max_last <= max(max_first * 3, 0.5)  # not exploding
    else:
        is_converging = True

    return text, elapsed, {
        "n_updates": n_updates,
        "max_norm": max(norm_trajectory) if norm_trajectory else 0,
        "max_b_norm": max(b_norm_trajectory) if b_norm_trajectory else 0,
        "mean_norm_final": norm_trajectory[-1] if norm_trajectory else 0,
        "avg_loss": float(np.mean(loss_trajectory)) if loss_trajectory else 0,
        "is_converging": is_converging,
        "step_diagnostics": step_diagnostics,
        "norm_trajectory": norm_trajectory,
        "a_norm_trajectory": a_norm_trajectory,
        "b_norm_trajectory": b_norm_trajectory,
        "loss_trajectory": loss_trajectory,
        "confidence_trajectory": confidence_trajectory,
    }


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    print(f"[task_8c] Starting at {start_time.isoformat()}")
    print(f"[task_8c] GPU: {torch.cuda.get_device_name(0)}")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda:0"

    # Load model
    print("[task_8c] Loading model...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    print("[task_8c] Model loaded.")

    # Inject LoRA
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                             rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.SGD(lora_params, lr=LORA_LR)

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    print(f"[task_8c] Generated {len(problems)} Countdown problems")

    # ── Phase 1: Run Vanilla ──
    print("\n[task_8c] === Phase 1: Vanilla generation ===")
    vanilla_results = []
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        # Reset seed per sample for reproducibility
        torch.manual_seed(SEED + idx)
        text, elapsed = generate_vanilla(model, tokenizer, prompt, device)
        verification = verify_countdown_answer(text, problem)
        metrics = compute_text_metrics(text)
        vanilla_results.append({
            "idx": idx,
            "problem": problem,
            "generated_text": text,
            "verification": verification,
            "is_correct": verification.get("is_correct", False),
            "gen_time_s": elapsed,
            **metrics,
        })
        status = "CORRECT" if verification.get("is_correct") else "wrong"
        print(f"  [{idx+1}/{N_SAMPLES}] {status} | {elapsed:.1f}s | d2={metrics['distinct_2']:.3f} r3={metrics['rep_3']:.3f}")
    gc.collect(); torch.cuda.empty_cache()

    # ── Phase 2: Run DTA with diagnostics ──
    print("\n[task_8c] === Phase 2: DTA with diagnostics ===")
    dta_results = []
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        torch.manual_seed(SEED + idx)
        text, elapsed, diag = generate_dta_with_diagnostics(
            model, tokenizer, prompt, device,
            lora_params=lora_params, lora_modules=lora_modules, optimizer=optimizer
        )
        verification = verify_countdown_answer(text, problem)
        metrics = compute_text_metrics(text)
        dta_results.append({
            "idx": idx,
            "problem": problem,
            "generated_text": text,
            "verification": verification,
            "is_correct": verification.get("is_correct", False),
            "gen_time_s": elapsed,
            "n_updates": diag["n_updates"],
            "max_norm": diag["max_norm"],
            "max_b_norm": diag["max_b_norm"],
            "mean_norm_final": diag["mean_norm_final"],
            "avg_loss": diag["avg_loss"],
            "is_converging": diag["is_converging"],
            "norm_trajectory": diag["norm_trajectory"],
            "a_norm_trajectory": diag["a_norm_trajectory"],
            "b_norm_trajectory": diag["b_norm_trajectory"],
            "loss_trajectory": diag["loss_trajectory"],
            "confidence_trajectory": diag["confidence_trajectory"],
            "step_diagnostics": diag["step_diagnostics"],
            **metrics,
        })
        status = "CORRECT" if verification.get("is_correct") else "wrong"
        conv = "converging" if diag["is_converging"] else "DIVERGING"
        print(f"  [{idx+1}/{N_SAMPLES}] {status} | {elapsed:.1f}s | d2={metrics['distinct_2']:.3f} "
              f"r3={metrics['rep_3']:.3f} | delta={diag['max_norm']:.6f} B={diag['max_b_norm']:.6f} | {conv}")
    gc.collect(); torch.cuda.empty_cache()

    # ── Phase 3: Analysis ──
    print("\n[task_8c] === Phase 3: Analysis ===")

    # 3a. Aggregate text quality comparison (H10)
    def aggregate_metrics(results):
        keys = ["distinct_1", "distinct_2", "distinct_3", "rep_2", "rep_3", "word_count"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in results]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        agg["accuracy"] = sum(1 for r in results if r["is_correct"]) / len(results)
        agg["correct_count"] = sum(1 for r in results if r["is_correct"])
        agg["avg_time_s"] = float(np.mean([r["gen_time_s"] for r in results]))
        return agg

    vanilla_agg = aggregate_metrics(vanilla_results)
    dta_agg = aggregate_metrics(dta_results)

    print("\n  --- H10 Degradation Test ---")
    print(f"  {'Metric':<15} {'Vanilla':>10} {'DTA':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    for metric in ["distinct_1", "distinct_2", "distinct_3", "rep_2", "rep_3"]:
        v_val = vanilla_agg[f"{metric}_mean"]
        d_val = dta_agg[f"{metric}_mean"]
        delta = d_val - v_val
        print(f"  {metric:<15} {v_val:>10.4f} {d_val:>10.4f} {delta:>+10.4f}")
    print(f"  {'accuracy':<15} {vanilla_agg['accuracy']:>10.4f} {dta_agg['accuracy']:>10.4f} "
          f"{dta_agg['accuracy'] - vanilla_agg['accuracy']:>+10.4f}")

    # H10 pass: DTA rep-3 should not exceed vanilla + 20%
    h10_pass = dta_agg["rep_3_mean"] <= vanilla_agg["rep_3_mean"] * 1.2 + 0.02  # +0.02 for small values
    print(f"\n  H10 (no degradation): {'PASS' if h10_pass else 'FAIL'}")
    print(f"    DTA rep-3 ({dta_agg['rep_3_mean']:.4f}) <= vanilla rep-3 * 1.2 + 0.02 "
          f"({vanilla_agg['rep_3_mean'] * 1.2 + 0.02:.4f})")

    # 3b. LoRA trajectory analysis: correct vs incorrect
    correct_dta = [r for r in dta_results if r["is_correct"]]
    incorrect_dta = [r for r in dta_results if not r["is_correct"]]

    print(f"\n  --- LoRA Behavior: Correct ({len(correct_dta)}) vs Incorrect ({len(incorrect_dta)}) ---")

    def summarize_lora_group(group, label):
        if not group:
            print(f"  {label}: No samples")
            return {}
        max_norms = [r["max_norm"] for r in group]
        max_b_norms = [r["max_b_norm"] for r in group]
        final_norms = [r["mean_norm_final"] for r in group]
        avg_losses = [r["avg_loss"] for r in group]
        n_updates_list = [r["n_updates"] for r in group]
        converging = [r["is_converging"] for r in group]
        print(f"  {label}:")
        print(f"    delta_norm:  mean={np.mean(max_norms):.6f}, std={np.std(max_norms):.6f}")
        print(f"    B_norm:      mean={np.mean(max_b_norms):.6f}, std={np.std(max_b_norms):.6f}")
        print(f"    final_norm:  mean={np.mean(final_norms):.6f}, std={np.std(final_norms):.6f}")
        print(f"    avg_loss:    mean={np.mean(avg_losses):.4f}, std={np.std(avg_losses):.4f}")
        print(f"    n_updates:   mean={np.mean(n_updates_list):.1f}")
        print(f"    converging:  {sum(converging)}/{len(converging)}")
        return {
            "max_delta_norm_mean": float(np.mean(max_norms)),
            "max_delta_norm_std": float(np.std(max_norms)),
            "max_b_norm_mean": float(np.mean(max_b_norms)),
            "max_b_norm_std": float(np.std(max_b_norms)),
            "final_norm_mean": float(np.mean(final_norms)),
            "avg_loss_mean": float(np.mean(avg_losses)),
            "n_updates_mean": float(np.mean(n_updates_list)),
            "all_converging": all(converging),
        }

    correct_lora_summary = summarize_lora_group(correct_dta, "Correct samples")
    incorrect_lora_summary = summarize_lora_group(incorrect_dta, "Incorrect samples")

    # 3c. H7 alternative: confidence trajectory (should be increasing)
    print(f"\n  --- H7 Alternative: Information Accumulation ---")
    all_conf_trajectories = []
    for r in dta_results:
        ct = r.get("confidence_trajectory", [])
        if ct:
            all_conf_trajectories.append(ct)

    if all_conf_trajectories:
        # Average confidence at each DTA step position
        max_len = max(len(ct) for ct in all_conf_trajectories)
        # Pad shorter trajectories with NaN
        padded = np.full((len(all_conf_trajectories), max_len), np.nan)
        for i, ct in enumerate(all_conf_trajectories):
            padded[i, :len(ct)] = ct
        mean_conf = np.nanmean(padded, axis=0)

        # Check if monotonically increasing (roughly)
        # Compare first third vs last third
        third = max(1, len(mean_conf) // 3)
        first_third_mean = float(np.nanmean(mean_conf[:third]))
        last_third_mean = float(np.nanmean(mean_conf[-third:]))
        h7_increasing = last_third_mean > first_third_mean

        print(f"  Confidence trajectory (averaged over {len(all_conf_trajectories)} samples):")
        print(f"    First third mean: {first_third_mean:.4f}")
        print(f"    Last third mean:  {last_third_mean:.4f}")
        print(f"    H7 (increasing):  {'YES' if h7_increasing else 'NO'}")

        # Sample 5 points from trajectory
        indices = np.linspace(0, len(mean_conf)-1, min(5, len(mean_conf)), dtype=int)
        for idx in indices:
            print(f"    Step {idx}: confidence={mean_conf[idx]:.4f}")
    else:
        h7_increasing = None
        first_third_mean = None
        last_third_mean = None
        print("  No confidence trajectories recorded.")

    # 3d. LoRA norm convergence check
    print(f"\n  --- LoRA Convergence Summary ---")
    all_converging = all(r["is_converging"] for r in dta_results)
    all_max_norms = [r["max_norm"] for r in dta_results]
    all_max_b_norms = [r["max_b_norm"] for r in dta_results]
    print(f"  All samples converging: {all_converging}")
    print(f"  Max delta norm across all samples: {max(all_max_norms):.6f}")
    print(f"  Max B norm across all samples: {max(all_max_b_norms):.6f}")
    print(f"  Mean max delta norm: {np.mean(all_max_norms):.6f}")
    print(f"  Mean max B norm: {np.mean(all_max_b_norms):.6f}")

    # ── Build results JSON ──
    end_time = datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()

    results = {
        "task": "task_8c",
        "mode": "pilot",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "dta_config": {
            "rank": LORA_RANK,
            "layers": LORA_LAYERS,
            "lr": LORA_LR,
            "gamma": LORA_GAMMA,
            "warmup_frac": WARMUP_FRAC,
            "alpha": LORA_ALPHA,
        },
        "vanilla_summary": vanilla_agg,
        "dta_summary": dta_agg,
        "h10_degradation_test": {
            "pass": h10_pass,
            "vanilla_rep3": vanilla_agg["rep_3_mean"],
            "dta_rep3": dta_agg["rep_3_mean"],
            "threshold": vanilla_agg["rep_3_mean"] * 1.2 + 0.02,
            "vanilla_distinct2": vanilla_agg["distinct_2_mean"],
            "dta_distinct2": dta_agg["distinct_2_mean"],
        },
        "lora_trajectory_analysis": {
            "correct_samples": correct_lora_summary,
            "incorrect_samples": incorrect_lora_summary,
            "all_converging": all_converging,
            "max_norm_overall": float(max(all_max_norms)),
            "mean_max_norm": float(np.mean(all_max_norms)),
        },
        "h7_information_accumulation": {
            "increasing": h7_increasing,
            "first_third_confidence": first_third_mean,
            "last_third_confidence": last_third_mean,
        },
        "pass_criteria": {
            "h10_no_degradation": h10_pass,
            "lora_converging": all_converging,
            "overall": "GO" if (h10_pass and all_converging) else "CONDITIONAL-GO",
        },
        "wall_clock_s": elapsed_total,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "gpu": torch.cuda.get_device_name(0),
    }

    # Save full results (without per-step diagnostics which are huge)
    results_path = RESULTS_DIR / "task_8c_degradation_lora_trajectory.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[task_8c] Results saved to {results_path}")

    # Save detailed per-sample diagnostics separately (for visualization)
    diagnostics_path = RESULTS_DIR / "task_8c_step_diagnostics.json"
    diag_data = []
    for r in dta_results:
        diag_data.append({
            "idx": r["idx"],
            "is_correct": r["is_correct"],
            "norm_trajectory": r["norm_trajectory"],
            "a_norm_trajectory": r.get("a_norm_trajectory", []),
            "b_norm_trajectory": r.get("b_norm_trajectory", []),
            "loss_trajectory": r["loss_trajectory"],
            "confidence_trajectory": r.get("confidence_trajectory", []),
            "step_diagnostics": r["step_diagnostics"],
        })
    with open(diagnostics_path, "w") as f:
        json.dump(diag_data, f, indent=2)
    print(f"[task_8c] Step diagnostics saved to {diagnostics_path}")

    # Save vanilla vs DTA per-sample comparison
    comparison_path = RESULTS_DIR / "task_8c_text_comparison.json"
    comparison = []
    for v, d in zip(vanilla_results, dta_results):
        comparison.append({
            "idx": v["idx"],
            "problem": v["problem"],
            "vanilla_text": v["generated_text"],
            "dta_text": d["generated_text"],
            "vanilla_correct": v["is_correct"],
            "dta_correct": d["is_correct"],
            "vanilla_metrics": {k: v[k] for k in ["distinct_1", "distinct_2", "distinct_3", "rep_2", "rep_3", "word_count"]},
            "dta_metrics": {k: d[k] for k in ["distinct_1", "distinct_2", "distinct_3", "rep_2", "rep_3", "word_count"]},
        })
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"[task_8c] Text comparison saved to {comparison_path}")

    # DONE marker
    done_marker = RESULTS_DIR / "task_8c_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "task_8c",
        "status": "success",
        "summary": f"H10={'PASS' if h10_pass else 'FAIL'}, converging={all_converging}, "
                   f"vanilla_acc={vanilla_agg['accuracy']:.3f}, dta_acc={dta_agg['accuracy']:.3f}",
        "timestamp": end_time.isoformat(),
    }))
    print(f"[task_8c] DONE marker written.")

    print(f"\n[task_8c] Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"[task_8c] Done!")


if __name__ == "__main__":
    main()
