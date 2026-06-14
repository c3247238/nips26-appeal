"""
Task 7a (PILOT): LoRA Rank Ablation on Countdown (16 samples).

Sweeps LoRA rank r in {2, 4, 8, 16} with fixed hyperparams:
  - lr=5e-4, gamma=0.95, warmup=20%, last 2 layers FFN
  - Also runs Vanilla baseline for reference

Analyzes rank vs accuracy/compute trade-off.

Pass criteria: All 4 rank values run successfully, r=4 accuracy near-optimal.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_7a_lora_rank_ablation.py [--ranks 2,4,8,16]
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
LORA_LAYERS = 2        # Last 2 layers
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# Ablation: LoRA ranks to sweep
DEFAULT_RANKS = [2, 4, 8, 16]


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
# Sampling & Generation Helpers
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


def remove_lora(model, lora_modules):
    """Remove LoRA layers and restore original linear layers."""
    for m in lora_modules:
        # Find and restore the original layer
        original = m.original
        # Search through model layers to find where this LoRA is attached
        for layer in model.model.layers:
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if getattr(mlp, proj_name) is m:
                    setattr(mlp, proj_name, original)
    # Ensure original weights are not requiring grad
    for layer in model.model.layers:
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            if isinstance(proj, nn.Linear):
                proj.weight.requires_grad_(False)
                if proj.bias is not None:
                    proj.bias.requires_grad_(False)


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
            # If already a LoRALayer, get the underlying original
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
# DTA Generation with configurable rank
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

        # Record trajectory at every 8th step for analysis
        if i % 8 == 0:
            norm_trajectory.append({
                "step": i,
                "max_norm": cur_max,
                "mean_norm": float(np.mean(norms)) if norms else 0,
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
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=str, default=",".join(str(r) for r in DEFAULT_RANKS),
                        help="Comma-separated LoRA ranks to sweep")
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]

    device = "cuda:0"
    start_time = datetime.now()
    print(f"=== Task 7a PILOT: LoRA Rank Ablation ===")
    print(f"Ranks: {ranks}")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Fixed: lr={LORA_LR}, gamma={LORA_GAMMA}, warmup={WARMUP_FRAC}, layers={LORA_LAYERS}")
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

    # ── Vanilla baseline ──
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
        "rank": 0,
        "accuracy": vanilla_acc,
        "correct": vanilla_correct,
        "total": N_SAMPLES,
        "avg_time_s": vanilla_avg_time,
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in vanilla_results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in vanilla_results])),
        "n_lora_params": 0,
        "per_sample": vanilla_results,
    }
    torch.cuda.empty_cache(); gc.collect()

    # ── DTA with different ranks ──
    for rank in ranks:
        print(f"\n{'='*60}\n  DTA rank={rank}\n{'='*60}")
        set_seed()

        # Remove previous LoRA if any, then inject new rank
        # Clean model state by re-checking for LoRA layers
        for layer in model.model.layers:
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                if isinstance(proj, LoRALayer):
                    setattr(mlp, proj_name, proj.original)

        torch.cuda.empty_cache(); gc.collect()

        lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                                 rank=rank, alpha=LORA_ALPHA)
        optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)
        n_lora_params = sum(p.numel() for p in lora_params)

        rank_results = []
        rank_correct = 0
        all_norm_trajectories = []

        for idx, problem in enumerate(problems):
            prompt = format_countdown_prompt(problem)
            text, elapsed, extras = generate_dta(
                model, tokenizer, prompt, device,
                lora_params=lora_params, lora_modules=lora_modules,
                optimizer=optimizer
            )
            verification = verify_countdown_answer(text, problem)
            is_correct = verification["is_correct"]
            if is_correct:
                rank_correct += 1
            metrics = compute_text_metrics(text)
            rank_results.append({
                "idx": idx, "is_correct": is_correct, "gen_time_s": elapsed,
                "generated_text": text, "verification": verification,
                "n_updates": extras["n_updates"],
                "max_norm": extras["max_norm"],
                "avg_loss": extras["avg_loss"],
                **metrics,
            })
            all_norm_trajectories.append(extras["norm_trajectory"])
            status = "OK" if is_correct else "X"
            eq = verification.get("extracted_equation", "N/A")
            print(f"  [rank={rank} {idx+1}/{N_SAMPLES}] {status} | target={problem['target']} | "
                  f"eq={eq} | norm={extras['max_norm']:.4f} | {elapsed:.1f}s")

        rank_acc = rank_correct / N_SAMPLES
        rank_avg_time = float(np.mean([r["gen_time_s"] for r in rank_results]))
        print(f"  => DTA rank={rank}: {rank_correct}/{N_SAMPLES} = {rank_acc:.1%}, "
              f"avg_time={rank_avg_time:.1f}s, params={n_lora_params:,}")

        all_results[f"rank_{rank}"] = {
            "rank": rank,
            "accuracy": rank_acc,
            "correct": rank_correct,
            "total": N_SAMPLES,
            "avg_time_s": rank_avg_time,
            "distinct_2_mean": float(np.mean([r["distinct_2"] for r in rank_results])),
            "rep_3_mean": float(np.mean([r["rep_3"] for r in rank_results])),
            "n_lora_params": n_lora_params,
            "max_norm_mean": float(np.mean([r["max_norm"] for r in rank_results])),
            "max_norm_max": float(max(r["max_norm"] for r in rank_results)),
            "avg_loss_mean": float(np.mean([r["avg_loss"] for r in rank_results])),
            "per_sample": rank_results,
            "norm_trajectories": all_norm_trajectories,
        }

        # Remove LoRA before next rank
        remove_lora(model, lora_modules)
        del optimizer, lora_params, lora_modules
        torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()
    wall_clock_min = (end_time - start_time).total_seconds() / 60

    # ── Summary Table ──
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY: Task 7a Pilot - LoRA Rank Ablation")
    print(f"{'='*80}")
    print(f"  {'Config':<15} {'Acc':>8} {'Correct':>8} {'AvgTime':>10} {'Dist-2':>8} "
          f"{'Rep-3':>8} {'Params':>10} {'MaxNorm':>10}")
    print(f"  {'-'*85}")

    configs = ["vanilla"] + [f"rank_{r}" for r in ranks]
    for cfg_name in configs:
        s = all_results[cfg_name]
        norm_str = f"{s.get('max_norm_mean', 0):.4f}" if s['rank'] > 0 else "N/A"
        print(f"  {cfg_name:<15} {s['accuracy']:>7.1%} {s['correct']:>8} "
              f"{s['avg_time_s']:>9.1f}s {s['distinct_2_mean']:>7.3f} {s['rep_3_mean']:>7.3f} "
              f"{s['n_lora_params']:>10,} {norm_str:>10}")

    # ── Pass Criteria ──
    all_ran = all(f"rank_{r}" in all_results for r in ranks)
    rank_4_acc = all_results.get("rank_4", {}).get("accuracy", 0)
    best_rank = max(ranks, key=lambda r: all_results[f"rank_{r}"]["accuracy"])
    best_acc = all_results[f"rank_{best_rank}"]["accuracy"]

    # r=4 is near-optimal if within 1 sample of the best
    r4_near_optimal = (best_acc - rank_4_acc) <= (1.0 / N_SAMPLES + 1e-6)

    print(f"\n  Pass Criteria:")
    print(f"    All 4 ranks ran: {'PASS' if all_ran else 'FAIL'}")
    print(f"    r=4 near-optimal: {'PASS' if r4_near_optimal else 'FAIL'} "
          f"(r=4: {rank_4_acc:.1%}, best: r={best_rank} at {best_acc:.1%})")
    print(f"    Overall: {'PASS' if all_ran else 'FAIL'}")
    print(f"\n  Wall clock: {wall_clock_min:.1f} min")

    # ── Save Results ──
    output = {
        "task": "task_7a",
        "mode": "pilot",
        "ablation": "lora_rank",
        "config": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "steps": STEPS,
            "temperature": TEMPERATURE,
            "fixed_lr": LORA_LR,
            "fixed_gamma": LORA_GAMMA,
            "fixed_warmup": WARMUP_FRAC,
            "fixed_layers": LORA_LAYERS,
            "ranks_swept": ranks,
        },
        "gpu_info": {
            "gpu_name": gpu_name,
            "vram_total_mb": int(gpu_mem),
        },
        "results": {},
        "summary_table": [],
        "pass_criteria": {
            "all_ranks_ran": all_ran,
            "r4_near_optimal": r4_near_optimal,
            "best_rank": best_rank,
            "best_accuracy": best_acc,
        },
        "wall_clock_min": wall_clock_min,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }

    # Store per-rank summaries (without full per_sample for the main JSON)
    for cfg_name in configs:
        s = all_results[cfg_name]
        summary_row = {
            "config": cfg_name,
            "rank": s["rank"],
            "accuracy": s["accuracy"],
            "correct": s["correct"],
            "total": s["total"],
            "avg_time_s": s["avg_time_s"],
            "distinct_2_mean": s["distinct_2_mean"],
            "rep_3_mean": s["rep_3_mean"],
            "n_lora_params": s["n_lora_params"],
        }
        if s["rank"] > 0:
            summary_row["max_norm_mean"] = s["max_norm_mean"]
            summary_row["max_norm_max"] = s["max_norm_max"]
            summary_row["avg_loss_mean"] = s["avg_loss_mean"]
        output["summary_table"].append(summary_row)

    # Full results with per-sample data
    for cfg_name in configs:
        s = all_results[cfg_name]
        output["results"][cfg_name] = {
            "rank": s["rank"],
            "accuracy": s["accuracy"],
            "correct": s["correct"],
            "avg_time_s": s["avg_time_s"],
            "per_sample": s["per_sample"],
        }
        if "norm_trajectories" in s:
            output["results"][cfg_name]["norm_trajectories"] = s["norm_trajectories"]

    # Save main results
    results_path = RESULTS_DIR / "task_7a_lora_rank.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Write DONE marker
    done_path = RESULTS_DIR / "task_7a_DONE"
    done_path.write_text(json.dumps({
        "task_id": "task_7a",
        "status": "success" if all_ran else "failed",
        "summary": f"Rank ablation: best r={best_rank} ({best_acc:.1%}), "
                   f"vanilla={vanilla_acc:.1%}, wall={wall_clock_min:.1f}min",
        "timestamp": end_time.isoformat(),
    }))
    print(f"DONE marker written to {done_path}")


if __name__ == "__main__":
    main()
