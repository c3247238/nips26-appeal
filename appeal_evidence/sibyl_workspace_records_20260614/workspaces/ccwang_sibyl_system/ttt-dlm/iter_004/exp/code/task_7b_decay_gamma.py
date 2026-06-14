"""
Task 7b (PILOT): Decay Factor gamma Ablation on Countdown (16 samples).

Sweeps gamma in {0.0, 0.8, 0.9, 0.95, 0.99, 1.0} with fixed hyperparams:
  - lr=5e-4, rank=4, warmup=20%, last 2 layers FFN
  - Also runs Vanilla baseline for reference

gamma=0.0 means full reset each step (no cumulative memory).
gamma=1.0 means no decay (risk of drift).
H8 predicts optimal gamma in [0.9, 0.99].

Pass criteria: All 6 gamma values run successfully, gamma=0.95 vicinity is best.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_7b_decay_gamma.py [--gammas 0.0,0.8,0.9,0.95,0.99,1.0]
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
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

# DTA Hyperparams (fixed across ablation)
LORA_RANK = 4
LORA_LAYERS = 2        # Last 2 layers
LORA_LR = 5e-4
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# Ablation: gamma values to sweep
DEFAULT_GAMMAS = [0.0, 0.8, 0.9, 0.95, 0.99, 1.0]


# --------------------------------------------------
# Countdown Problem Generator
# --------------------------------------------------

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


# --------------------------------------------------
# Text Quality Metrics
# --------------------------------------------------

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


# --------------------------------------------------
# Sampling & Generation Helpers
# --------------------------------------------------

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


# --------------------------------------------------
# LoRA Implementation
# --------------------------------------------------

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
            if isinstance(original, LoRALayer):
                original = original.original
            lora_layer = LoRALayer(original, rank=rank, alpha=alpha)
            setattr(mlp, proj_name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            lora_modules.append(lora_layer)
    n_params = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected rank={rank} into layers {target_layers}, "
          f"{len(lora_modules)} modules, {n_params:,} params")
    return lora_params, lora_modules


def remove_lora(model, lora_modules):
    """Remove LoRA layers and restore original linear layers."""
    for m in lora_modules:
        original = m.original
        for layer in model.model.layers:
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if getattr(mlp, proj_name) is m:
                    setattr(mlp, proj_name, original)
    for layer in model.model.layers:
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            if isinstance(proj, nn.Linear):
                proj.weight.requires_grad_(False)
                if proj.bias is not None:
                    proj.bias.requires_grad_(False)


def reset_lora(lora_modules, optimizer):
    for m in lora_modules:
        m.reset_parameters()
    optimizer.zero_grad()
    optimizer.state.clear()


def decay_lora_params(lora_params, gamma):
    """Apply decay to LoRA params. gamma=0 means full reset to zero."""
    with torch.no_grad():
        for p in lora_params:
            p.mul_(gamma)


# --------------------------------------------------
# Vanilla Generation (baseline reference)
# --------------------------------------------------

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


# --------------------------------------------------
# DTA Generation with configurable gamma
# --------------------------------------------------

def generate_dta(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None,
                 gamma=0.95):
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

            # Apply decay (gamma=0 means full reset after each update)
            decay_lora_params(lora_params, gamma)

        # Track norms
        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max

        # Record trajectory at every 8th step
        if i % 8 == 0:
            norm_trajectory.append({
                "step": i,
                "max_norm": cur_max,
                "mean_norm": float(np.mean(norms)) if norms else 0,
            })

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
    final_norms = [m.get_delta_norm() for m in lora_modules]
    final_max_norm = max(final_norms) if final_norms else 0
    return text, elapsed, {
        "n_updates": n_updates,
        "max_norm": max_norm,
        "final_norm": final_max_norm,
        "avg_loss": avg_loss,
        "norm_trajectory": norm_trajectory,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=str,
                        default=",".join(str(g) for g in DEFAULT_GAMMAS),
                        help="Comma-separated gamma values to sweep")
    args = parser.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]

    device = "cuda:0"
    start_time = datetime.now()
    print(f"=== Task 7b PILOT: Decay Factor gamma Ablation ===")
    print(f"Gammas: {gammas}")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Fixed: rank={LORA_RANK}, lr={LORA_LR}, warmup={WARMUP_FRAC}, layers={LORA_LAYERS}")
    print(f"Start: {start_time.isoformat()}")

    # Generate problems (same for all)
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

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    def set_seed():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    all_results = {}

    # -- Vanilla baseline --
    print(f"\n{'='*60}\n  Vanilla Baseline\n{'='*60}")
    set_seed()
    vanilla_results = []
    vanilla_correct = 0
    for idx, problem in enumerate(problems):
        prompt = format_countdown_prompt(problem)
        text, elapsed = generate_vanilla(model, tokenizer, prompt, device)
        verification = verify_countdown_answer(text, problem)
        is_correct = verification["is_correct"]
        if is_correct:
            vanilla_correct += 1
        metrics = compute_text_metrics(text)
        vanilla_results.append({
            "idx": idx, "is_correct": is_correct, "gen_time_s": elapsed,
            "generated_text": text, "verification": verification, **metrics,
        })
        status = "OK" if is_correct else "X"
        print(f"  [vanilla {idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | {elapsed:.1f}s")

    vanilla_acc = vanilla_correct / N_SAMPLES
    vanilla_avg_time = float(np.mean([r["gen_time_s"] for r in vanilla_results]))
    print(f"  => Vanilla: {vanilla_correct}/{N_SAMPLES} = {vanilla_acc:.1%}, avg_time={vanilla_avg_time:.1f}s")

    all_results["vanilla"] = {
        "gamma": None,
        "accuracy": vanilla_acc,
        "correct": vanilla_correct,
        "total": N_SAMPLES,
        "avg_time_s": vanilla_avg_time,
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in vanilla_results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in vanilla_results])),
        "per_sample": vanilla_results,
    }
    torch.cuda.empty_cache(); gc.collect()

    # -- Inject LoRA once (rank is fixed, only gamma changes) --
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                             rank=LORA_RANK, alpha=LORA_ALPHA)
    n_lora_params = sum(p.numel() for p in lora_params)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    # -- DTA with different gamma values --
    for gamma in gammas:
        gamma_label = f"gamma_{gamma}"
        print(f"\n{'='*60}\n  DTA gamma={gamma}\n{'='*60}")
        set_seed()

        gamma_results = []
        gamma_correct = 0
        all_norm_trajectories = []

        for idx, problem in enumerate(problems):
            prompt = format_countdown_prompt(problem)
            text, elapsed, extras = generate_dta(
                model, tokenizer, prompt, device,
                lora_params=lora_params, lora_modules=lora_modules,
                optimizer=optimizer, gamma=gamma
            )
            verification = verify_countdown_answer(text, problem)
            is_correct = verification["is_correct"]
            if is_correct:
                gamma_correct += 1
            metrics = compute_text_metrics(text)
            gamma_results.append({
                "idx": idx, "is_correct": is_correct, "gen_time_s": elapsed,
                "generated_text": text, "verification": verification,
                "n_updates": extras["n_updates"],
                "max_norm": extras["max_norm"],
                "final_norm": extras["final_norm"],
                "avg_loss": extras["avg_loss"],
                **metrics,
            })
            all_norm_trajectories.append(extras["norm_trajectory"])
            status = "OK" if is_correct else "X"
            eq = verification.get("extracted_equation", "N/A")
            print(f"  [gamma={gamma} {idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | "
                  f"eq={eq} | maxnorm={extras['max_norm']:.4f} | fnorm={extras['final_norm']:.4f} | {elapsed:.1f}s")

        gamma_acc = gamma_correct / N_SAMPLES
        gamma_avg_time = float(np.mean([r["gen_time_s"] for r in gamma_results]))
        print(f"  => DTA gamma={gamma}: {gamma_correct}/{N_SAMPLES} = {gamma_acc:.1%}, "
              f"avg_time={gamma_avg_time:.1f}s")

        all_results[gamma_label] = {
            "gamma": gamma,
            "accuracy": gamma_acc,
            "correct": gamma_correct,
            "total": N_SAMPLES,
            "avg_time_s": gamma_avg_time,
            "distinct_2_mean": float(np.mean([r["distinct_2"] for r in gamma_results])),
            "rep_3_mean": float(np.mean([r["rep_3"] for r in gamma_results])),
            "n_lora_params": n_lora_params,
            "max_norm_mean": float(np.mean([r["max_norm"] for r in gamma_results])),
            "max_norm_max": float(max(r["max_norm"] for r in gamma_results)),
            "final_norm_mean": float(np.mean([r["final_norm"] for r in gamma_results])),
            "avg_loss_mean": float(np.mean([r["avg_loss"] for r in gamma_results])),
            "per_sample": gamma_results,
            "norm_trajectories": all_norm_trajectories,
        }

        torch.cuda.empty_cache(); gc.collect()

    # Remove LoRA
    remove_lora(model, lora_modules)
    del optimizer, lora_params, lora_modules
    torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()
    wall_clock_min = (end_time - start_time).total_seconds() / 60

    # -- Summary Table --
    print(f"\n{'='*90}")
    print(f"  FINAL SUMMARY: Task 7b Pilot - Decay Factor gamma Ablation")
    print(f"{'='*90}")
    print(f"  {'Config':<15} {'Acc':>8} {'Correct':>8} {'AvgTime':>10} {'Dist-2':>8} "
          f"{'Rep-3':>8} {'MaxNorm':>10} {'FinalNorm':>10}")
    print(f"  {'-'*95}")

    configs = ["vanilla"] + [f"gamma_{g}" for g in gammas]
    for cfg_name in configs:
        s = all_results[cfg_name]
        if s["gamma"] is not None:
            norm_str = f"{s['max_norm_mean']:.4f}"
            fnorm_str = f"{s['final_norm_mean']:.4f}"
        else:
            norm_str = "N/A"
            fnorm_str = "N/A"
        print(f"  {cfg_name:<15} {s['accuracy']:>7.1%} {s['correct']:>8} "
              f"{s['avg_time_s']:>9.1f}s {s['distinct_2_mean']:>7.3f} {s['rep_3_mean']:>7.3f} "
              f"{norm_str:>10} {fnorm_str:>10}")

    # -- Pass Criteria --
    all_ran = all(f"gamma_{g}" in all_results for g in gammas)
    gamma_accs = {g: all_results[f"gamma_{g}"]["accuracy"] for g in gammas}
    best_gamma = max(gammas, key=lambda g: gamma_accs[g])
    best_acc = gamma_accs[best_gamma]

    # Check if gamma=0.95 vicinity is best (within 1 sample of best)
    g095_acc = gamma_accs.get(0.95, 0)
    g095_near_best = (best_acc - g095_acc) <= (1.0 / N_SAMPLES + 1e-6)

    # Check norm behavior: gamma=1.0 should have highest final norms (drift risk)
    g10_final = all_results.get("gamma_1.0", {}).get("final_norm_mean", 0)
    g095_final = all_results.get("gamma_0.95", {}).get("final_norm_mean", 0)
    g00_final = all_results.get("gamma_0.0", {}).get("final_norm_mean", 0)

    print(f"\n  Pass Criteria:")
    print(f"    All 6 gammas ran: {'PASS' if all_ran else 'FAIL'}")
    print(f"    gamma=0.95 near-best: {'PASS' if g095_near_best else 'FAIL'} "
          f"(g=0.95: {g095_acc:.1%}, best: g={best_gamma} at {best_acc:.1%})")
    print(f"    Overall: {'PASS' if all_ran else 'FAIL'}")
    print(f"\n  Norm Analysis:")
    print(f"    gamma=0.0 final_norm:  {g00_final:.4f} (should be ~0, full reset)")
    print(f"    gamma=0.95 final_norm: {g095_final:.4f}")
    print(f"    gamma=1.0 final_norm:  {g10_final:.4f} (highest = drift risk)")
    print(f"\n  Wall clock: {wall_clock_min:.1f} min")

    # -- Save Results --
    output = {
        "task": "task_7b",
        "mode": "pilot",
        "ablation": "decay_gamma",
        "config": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "steps": STEPS,
            "temperature": TEMPERATURE,
            "fixed_rank": LORA_RANK,
            "fixed_lr": LORA_LR,
            "fixed_warmup": WARMUP_FRAC,
            "fixed_layers": LORA_LAYERS,
            "gammas_swept": gammas,
        },
        "gpu_info": {
            "gpu_name": gpu_name,
            "vram_total_mb": int(gpu_mem),
        },
        "results": {},
        "summary_table": [],
        "pass_criteria": {
            "all_gammas_ran": all_ran,
            "g095_near_best": g095_near_best,
            "best_gamma": best_gamma,
            "best_accuracy": best_acc,
        },
        "norm_analysis": {
            "gamma_0.0_final_norm": g00_final,
            "gamma_0.95_final_norm": g095_final,
            "gamma_1.0_final_norm": g10_final,
        },
        "wall_clock_min": wall_clock_min,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }

    # Store per-gamma summaries
    for cfg_name in configs:
        s = all_results[cfg_name]
        summary_row = {
            "config": cfg_name,
            "gamma": s["gamma"],
            "accuracy": s["accuracy"],
            "correct": s["correct"],
            "total": s["total"],
            "avg_time_s": s["avg_time_s"],
            "distinct_2_mean": s["distinct_2_mean"],
            "rep_3_mean": s["rep_3_mean"],
        }
        if s["gamma"] is not None:
            summary_row["n_lora_params"] = s["n_lora_params"]
            summary_row["max_norm_mean"] = s["max_norm_mean"]
            summary_row["max_norm_max"] = s["max_norm_max"]
            summary_row["final_norm_mean"] = s["final_norm_mean"]
            summary_row["avg_loss_mean"] = s["avg_loss_mean"]
        output["summary_table"].append(summary_row)

    # Full results with per-sample data
    for cfg_name in configs:
        s = all_results[cfg_name]
        output["results"][cfg_name] = {
            "gamma": s["gamma"],
            "accuracy": s["accuracy"],
            "correct": s["correct"],
            "avg_time_s": s["avg_time_s"],
            "per_sample": s["per_sample"],
        }
        if "norm_trajectories" in s:
            output["results"][cfg_name]["norm_trajectories"] = s["norm_trajectories"]

    # Save main results
    results_path = RESULTS_DIR / "task_7b_decay_gamma.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Write DONE marker
    done_path = RESULTS_DIR / "task_7b_DONE"
    done_path.write_text(json.dumps({
        "task_id": "task_7b",
        "status": "success" if all_ran else "failed",
        "summary": f"Gamma ablation: best g={best_gamma} ({best_acc:.1%}), "
                   f"g=0.95: {g095_acc:.1%}, vanilla={vanilla_acc:.1%}, "
                   f"wall={wall_clock_min:.1f}min",
        "timestamp": end_time.isoformat(),
    }))
    print(f"DONE marker written to {done_path}")


if __name__ == "__main__":
    main()
