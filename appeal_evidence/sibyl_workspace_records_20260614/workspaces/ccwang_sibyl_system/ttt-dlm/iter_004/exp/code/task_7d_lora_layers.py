"""
Task 7d (PILOT): LoRA Insertion Layers Ablation (16 samples).

Sweeps the number of transformer layers where LoRA is inserted on Countdown:
  - last_1: LoRA on last 1 layer FFN
  - last_2: LoRA on last 2 layers FFN (default)
  - last_4: LoRA on last 4 layers FFN

Also runs vanilla baseline for reference.

Pass criteria: All 3 configurations run successfully.

Usage:
    CUDA_VISIBLE_DEVICES=7 /home/ccwang/miniforge3/bin/python3 task_7d_lora_layers.py
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

# DTA Hyperparams (fixed, only n_layers varies)
LORA_RANK = 4
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# Layer configurations to test
LAYER_CONFIGS = [1, 2, 4]


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
# Shared Utilities
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
    """Inject LoRA into the last n_layers of the model's FFN."""
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
    return lora_params, lora_modules, target_layers


def remove_lora(model, target_layers):
    """Remove LoRA layers and restore original linear layers."""
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            current = getattr(mlp, proj_name)
            if isinstance(current, LoRALayer):
                # Restore original linear layer
                original = current.original
                original.weight.requires_grad_(False)
                if original.bias is not None:
                    original.bias.requires_grad_(False)
                setattr(mlp, proj_name, original)
    gc.collect()
    torch.cuda.empty_cache()


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
# Vanilla Generation (baseline reference)
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
# DTA Generation (standard, used with each layer config)
# ──────────────────────────────────────────────────

def generate_dta(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None):
    """Standard DTA generation (update every step, default hyperparams)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)

    reset_lora(lora_modules, optimizer)

    n_updates = 0
    lora_losses = []
    max_norm = 0.0
    norm_trajectory = []

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

        # Record norm trajectory every 8 steps
        if i % 8 == 0 or i == STEPS - 1:
            norm_trajectory.append({
                "step": i,
                "max_norm": cur_max,
                "mean_norm": float(np.mean(norms)) if norms else 0,
                "total_revealed": total_revealed,
                "n_updates_so_far": n_updates,
            })

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    avg_loss = float(np.mean(lora_losses)) if lora_losses else 0

    return text, elapsed, {
        "n_updates": n_updates,
        "max_norm": max_norm,
        "avg_loss": avg_loss,
        "norm_trajectory": norm_trajectory,
    }


# ──────────────────────────────────────────────────
# Run one method on all problems
# ──────────────────────────────────────────────────

def run_method(method_name, generate_fn, model, tokenizer, problems, device,
               lora_params=None, lora_modules=None, optimizer=None,
               is_dta=False):
    """Run a single method on all problems, return summary + per-sample results."""
    results = []
    correct = 0

    for idx, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)

        if is_dta:
            text, elapsed, extras = generate_fn(
                model, tokenizer, prompt_text, device,
                lora_params=lora_params, lora_modules=lora_modules,
                optimizer=optimizer,
            )
        else:
            text, elapsed = generate_fn(model, tokenizer, prompt_text, device)
            extras = {}

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

        status = "CORRECT" if is_correct else "WRONG"
        eq = verification.get("extracted_equation", "N/A")
        extra_info = ""
        if "n_updates" in extras:
            extra_info = f" | updates={extras['n_updates']} | max_norm={extras['max_norm']:.4f}"
        print(f"  [{method_name} {idx+1}/{len(problems)}] {status} | "
              f"target={problem['target']} | eq={eq} | time={elapsed:.1f}s{extra_info}")

    accuracy = correct / len(problems)
    avg_time = float(np.mean([r["gen_time_s"] for r in results]))
    d2_mean = float(np.mean([r["distinct_2"] for r in results]))
    r3_mean = float(np.mean([r["rep_3"] for r in results]))

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_time_s": avg_time,
        "distinct_2_mean": d2_mean,
        "rep_3_mean": r3_mean,
    }

    # Add DTA-specific summary stats
    if any("n_updates" in r for r in results):
        summary["avg_n_updates"] = float(np.mean([r.get("n_updates", 0) for r in results]))
        max_norms = [r.get("max_norm", 0) for r in results]
        summary["lora_norm_max"] = float(max(max_norms))
        summary["lora_norm_mean"] = float(np.mean(max_norms))
        avg_losses = [r.get("avg_loss", 0) for r in results if r.get("avg_loss", 0) > 0]
        if avg_losses:
            summary["avg_dta_loss"] = float(np.mean(avg_losses))

    print(f"  => {method_name}: {correct}/{len(problems)} = {accuracy:.1%}, "
          f"avg_time={avg_time:.1f}s, d2={d2_mean:.3f}, r3={r3_mean:.3f}")

    return summary, results


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = "cuda:0"
    start_time = datetime.now()
    print(f"=== Task 7d PILOT: LoRA Insertion Layers Ablation ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"DTA config: rank={LORA_RANK}, lr={LORA_LR}, "
          f"gamma={LORA_GAMMA}, warmup={WARMUP_FRAC}")
    print(f"Layer configurations to test: last {LAYER_CONFIGS} layers")
    print(f"Device: {device}")
    print(f"Start time: {start_time.isoformat()}")

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
    total_transformer_layers = len(model.model.layers)
    print(f"Dream-7B loaded on {device}, total transformer layers: {total_transformer_layers}")

    all_summaries = {}
    all_results = {}

    # ── 1. Vanilla baseline ──
    print(f"\n{'='*60}")
    print(f"  Vanilla Baseline")
    print(f"{'='*60}")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    v_summary, v_results = run_method(
        "vanilla", generate_vanilla, model, tokenizer, problems, device
    )
    all_summaries["vanilla"] = v_summary
    all_results["vanilla"] = v_results

    # ── 2. Run DTA with each layer configuration ──
    # We need to inject/remove LoRA for each config since they target different layers
    for n_layers in LAYER_CONFIGS:
        method_name = f"dta_layers_{n_layers}"
        print(f"\n{'='*60}")
        print(f"  DTA n_layers={n_layers} (last {n_layers} layers FFN)")
        print(f"{'='*60}")

        # Inject LoRA for this configuration
        lora_params, lora_modules, target_layers = inject_lora(
            model, n_layers=n_layers, rank=LORA_RANK, alpha=LORA_ALPHA
        )
        n_lora_params = sum(p.numel() for p in lora_params)
        print(f"  LoRA params: {n_lora_params} ({n_lora_params/1e6:.3f}M)")

        optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

        # Re-seed for fair comparison
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        summary, results = run_method(
            method_name, generate_dta, model, tokenizer, problems, device,
            lora_params=lora_params, lora_modules=lora_modules,
            optimizer=optimizer, is_dta=True,
        )

        # Record additional info about this config
        summary["n_lora_layers"] = n_layers
        summary["n_lora_params"] = n_lora_params
        summary["target_layers"] = target_layers

        all_summaries[method_name] = summary
        all_results[method_name] = results

        # Remove LoRA before next config to avoid double-wrapping
        remove_lora(model, target_layers)
        del lora_params, lora_modules, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  LoRA removed from layers {target_layers}")

    # ── Summary ──
    end_time = datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"SUMMARY: Task 7d Pilot - LoRA Insertion Layers Ablation")
    print(f"{'='*80}")
    print(f"\n{'Method':<20} {'Acc':>8} {'Correct':>8} {'Time(s)':>8} {'D2':>8} {'R3':>8} "
          f"{'Updates':>8} {'MaxNorm':>8} {'Params':>10}")
    print("-" * 100)

    for name, s in all_summaries.items():
        updates = s.get("avg_n_updates", "-")
        max_norm = s.get("lora_norm_max", "-")
        n_params = s.get("n_lora_params", "-")
        if isinstance(updates, float):
            updates = f"{updates:.0f}"
        if isinstance(max_norm, float):
            max_norm = f"{max_norm:.4f}"
        if isinstance(n_params, int):
            n_params = f"{n_params}"
        print(f"{name:<20} {s['accuracy']:>7.1%} {s['correct']:>8} {s['avg_time_s']:>8.1f} "
              f"{s['distinct_2_mean']:>8.3f} {s['rep_3_mean']:>8.3f} {updates:>8} "
              f"{max_norm:>8} {n_params:>10}")

    # Compute time overhead relative to vanilla
    vanilla_time = all_summaries["vanilla"]["avg_time_s"]
    print(f"\nTime overhead relative to vanilla ({vanilla_time:.1f}s):")
    for n_layers in LAYER_CONFIGS:
        name = f"dta_layers_{n_layers}"
        t = all_summaries[name]["avg_time_s"]
        overhead = t / vanilla_time if vanilla_time > 0 else 0
        print(f"  layers={n_layers}: {t:.1f}s ({overhead:.2f}x vanilla), "
              f"params={all_summaries[name].get('n_lora_params', 'N/A')}")

    # Accuracy comparison
    print(f"\nAccuracy comparison:")
    vanilla_acc = all_summaries["vanilla"]["accuracy"]
    print(f"  vanilla: {vanilla_acc:.1%}")
    for n_layers in LAYER_CONFIGS:
        name = f"dta_layers_{n_layers}"
        acc = all_summaries[name]["accuracy"]
        delta = acc - vanilla_acc
        print(f"  layers={n_layers}: {acc:.1%} (delta={delta:+.1%})")

    # Pass criteria
    print(f"\n--- Pass Criteria ---")
    all_ran = all(f"dta_layers_{n}" in all_summaries for n in LAYER_CONFIGS)
    print(f"  All 3 configurations ran: {'PASS' if all_ran else 'FAIL'}")

    # Check all generated non-empty text
    all_nonempty = True
    for n_layers in LAYER_CONFIGS:
        name = f"dta_layers_{n_layers}"
        for r in all_results[name]:
            if len(r["generated_text"].strip()) == 0:
                all_nonempty = False
                break
    print(f"  All outputs non-empty: {'PASS' if all_nonempty else 'FAIL'}")

    # Norm check
    all_norm_ok = True
    for n_layers in LAYER_CONFIGS:
        name = f"dta_layers_{n_layers}"
        if all_summaries[name].get("lora_norm_max", 0) > 1.0:
            all_norm_ok = False
    print(f"  LoRA norms <= 1.0:     {'PASS' if all_norm_ok else 'FAIL (warning only)'}")

    overall = "GO" if (all_ran and all_nonempty) else "CONDITIONAL-GO"
    print(f"  Overall: {overall}")

    print(f"\nTotal wall-clock time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # ── Save results ──
    output = {
        "task": "task_7d",
        "mode": "pilot",
        "ablation": "lora_layers",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "layer_configs": LAYER_CONFIGS,
        "total_transformer_layers": total_transformer_layers,
        "dta_config": {
            "lora_rank": LORA_RANK,
            "lr": LORA_LR,
            "gamma": LORA_GAMMA,
            "warmup_frac": WARMUP_FRAC,
            "lora_alpha": LORA_ALPHA,
            "grad_clip": GRAD_CLIP,
        },
        "summaries": all_summaries,
        "pass_criteria": {
            "all_ran": all_ran,
            "all_nonempty": all_nonempty,
            "all_norm_ok": all_norm_ok,
            "overall": overall,
        },
        "wall_clock_s": elapsed_total,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "torch_version": torch.__version__,
        "per_sample": {},
    }

    for name, results in all_results.items():
        # Strip norm_trajectory for main output (save separately)
        stripped = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if k != "norm_trajectory"}
            stripped.append(r_copy)
        output["per_sample"][name] = stripped

    out_file = RESULTS_DIR / "task_7d_lora_layers.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Save norm trajectories separately for detailed analysis
    trajectories = {}
    for n_layers in LAYER_CONFIGS:
        name = f"dta_layers_{n_layers}"
        trajectories[name] = []
        for r in all_results[name]:
            if "norm_trajectory" in r:
                trajectories[name].append({
                    "sample_idx": r["idx"],
                    "is_correct": r["is_correct"],
                    "trajectory": r["norm_trajectory"],
                })

    traj_file = RESULTS_DIR / "task_7d_norm_trajectories.json"
    with open(traj_file, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"Norm trajectories saved to {traj_file}")

    # Write DONE marker
    done_marker = RESULTS_DIR / "task_7d_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "task_7d",
        "status": "success",
        "summary": (
            f"LoRA layers ablation: "
            + ", ".join(
                f"layers={n} acc={all_summaries[f'dta_layers_{n}']['accuracy']:.1%}"
                for n in LAYER_CONFIGS
            )
        ),
        "timestamp": end_time.isoformat(),
    }))
    print(f"DONE marker written to {done_marker}")


if __name__ == "__main__":
    main()
