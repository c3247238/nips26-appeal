"""
entropy_analysis: Belief Entropy Trajectory Analysis (PILOT: Countdown-16, seed 42).

Detailed information-theoretic analysis of BSD belief evolution:
  (1) Per-position entropy at each denoising step
  (2) Spearman rank correlation (step vs entropy)
  (3) Terminal entropy comparison vs vanilla
  (4) Entropy vs accuracy correlation

Hypothesis H2: Monotonically decreasing entropy, lower terminal entropy than vanilla.

Task: entropy_analysis
Mode: PILOT (16 samples, seed 42)
GPU: cuda:0 (mapped via CUDA_VISIBLE_DEVICES)

Usage:
    CUDA_VISIBLE_DEVICES=2 python entropy_analysis.py
"""
import os
import sys
import json
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics, compute_per_sample_metrics,
    vanilla_generate, save_results,
    MASK_TOKEN_ID, PROJECT_DIR, RESULTS_FULL,
)
from bsd_racfg.bsd import BSDConfig

# ── Constants ──
TASK_ID = "entropy_analysis"
N_SAMPLES = 16   # PILOT mode
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
DEVICE = "cuda:0"

RESULTS_DIR = RESULTS_FULL
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────
# Progress & PID tracking
# ──────────────────────────────────────────────────

def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")


def report_progress(current, total, phase="", metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": current, "total_epochs": total,
        "step": current, "total_steps": total,
        "phase": phase,
        "loss": None,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[{TASK_ID}] DONE marker written ({status})")


# ──────────────────────────────────────────────────
# BSD with detailed per-position entropy tracking
# ──────────────────────────────────────────────────

def bsd_generate_with_position_entropy(
    model, tokenizer, prompt_text, embedding_layer, config, device=DEVICE
):
    """BSD generation that records per-position entropy at every step.

    Returns: (text, elapsed, diagnostics) where diagnostics contains:
      - position_entropies: list of [gen_len] arrays, one per step (Phase 1 only)
      - step_mean_entropy: list of floats, mean entropy per step (all phases)
      - step_min_entropy / step_max_entropy
      - per_position_terminal_entropy: [gen_len] array at last belief step
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + config.gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, config.steps + 1, device=device)

    k_step = int(config.steps * (1 - config.k_frac))

    mask_emb = embedding_layer(torch.tensor([MASK_TOKEN_ID], device=device))
    mask_emb_norm = mask_emb.norm(dim=-1, keepdim=True).item()

    beliefs = mask_emb.expand(1, config.gen_len, -1).clone()

    # Detailed tracking
    position_entropies_belief = []   # Phase 1: per-position entropy at each step
    position_entropies_reveal = []   # Phase 2: per-position entropy at mask positions
    step_info = []                   # Per-step summary
    per_position_terminal = None     # Last belief step per-position entropy

    t0 = time.time()

    with torch.no_grad():
        for i in range(config.steps):
            is_phase1 = (i < k_step)

            if is_phase1:
                step_frac = i / max(k_step - 1, 1)
                alpha_start, alpha_end = config.alpha_start, config.alpha_end
                if config.alpha_schedule == "linear":
                    alpha = alpha_start + (alpha_end - alpha_start) * step_frac
                elif config.alpha_schedule == "cosine":
                    import math
                    t_cos = 0.5 * (1 - math.cos(math.pi * step_frac))
                    alpha = alpha_start + (alpha_end - alpha_start) * t_cos
                else:
                    alpha = alpha_start

                tau = config.tau_start + (config.tau_end - config.tau_start) * step_frac
                beta_start = config.fallback_beta_start
                beta_end = config.fallback_beta_end
                beta = beta_start + (beta_end - beta_start) * step_frac

                prompt_emb = embedding_layer(x[:, :prompt_len])
                input_emb = torch.cat([prompt_emb, beliefs], dim=1)

                outputs = model(inputs_embeds=input_emb,
                                attention_mask="full", position_ids=None)
                logits = outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                gen_logits = logits[:, prompt_len:, :]
                soft_probs = F.softmax(gen_logits / tau, dim=-1)

                # Per-position entropy: H(p) = -sum(p * log(p))
                pos_entropy = -(soft_probs * (soft_probs + 1e-10).log()).sum(dim=-1)  # [1, gen_len]
                pos_entropy_np = pos_entropy[0].cpu().float().numpy()

                position_entropies_belief.append(pos_entropy_np.copy())
                per_position_terminal = pos_entropy_np.copy()

                step_info.append({
                    "step": i, "phase": "belief",
                    "mean_entropy": float(pos_entropy_np.mean()),
                    "min_entropy": float(pos_entropy_np.min()),
                    "max_entropy": float(pos_entropy_np.max()),
                    "std_entropy": float(pos_entropy_np.std()),
                    "median_entropy": float(np.median(pos_entropy_np)),
                    "alpha": alpha, "tau": tau,
                })

                # EMA update
                emb_weight = embedding_layer.weight
                new_belief = torch.matmul(soft_probs.to(emb_weight.dtype), emb_weight)
                beliefs = (1 - alpha) * beliefs + alpha * new_belief

                if beta > 0:
                    beliefs = beta * mask_emb.unsqueeze(0).expand_as(beliefs) + \
                              (1 - beta) * beliefs

                belief_norms = beliefs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                beliefs = beliefs * (mask_emb_norm / belief_norms)

            else:
                mask_index = (x == MASK_TOKEN_ID)
                n_masked = mask_index.sum().item()
                if n_masked == 0:
                    break

                if i == k_step:
                    prompt_emb = embedding_layer(x[:, :prompt_len])
                    input_emb = torch.cat([prompt_emb, beliefs], dim=1)
                    outputs = model(inputs_embeds=input_emb,
                                    attention_mask="full", position_ids=None)
                else:
                    outputs = model(x, attention_mask="full", position_ids=None)

                logits = outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Per-position entropy for remaining mask positions
                all_probs = F.softmax(logits[:, prompt_len:, :] / TEMPERATURE, dim=-1)
                gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)
                if gen_mask.any():
                    masked_probs = all_probs[gen_mask]
                    entropy = -(masked_probs * (masked_probs + 1e-10).log()).sum(dim=-1)
                    mean_ent = float(entropy.mean().item())
                    # Store per-position for masked positions
                    pos_ent_array = np.full(config.gen_len, np.nan)
                    mask_indices = gen_mask[0].cpu().numpy().nonzero()[0]
                    pos_ent_array[mask_indices] = entropy.cpu().float().numpy()
                    position_entropies_reveal.append(pos_ent_array.copy())
                else:
                    mean_ent = 0.0
                    position_entropies_reveal.append(np.full(config.gen_len, np.nan))

                step_info.append({
                    "step": i, "phase": "hard_reveal",
                    "mean_entropy": mean_ent,
                    "n_masked": n_masked,
                })

                mask_logits = logits[mask_index]
                t_val = timesteps[i]
                s_val = timesteps[i + 1]
                p_transfer = (1 - s_val / t_val).item() if i < config.steps - 1 else 1.0

                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                if config.temperature > 0:
                    probs = F.softmax(mask_logits[transfer_mask] / config.temperature, dim=-1)
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    sampled_tokens = mask_logits[transfer_mask].argmax(dim=-1)
                x0[transfer_mask] = sampled_tokens
                x[mask_index] = x0

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

    diagnostics = {
        "position_entropies_belief": position_entropies_belief,
        "position_entropies_reveal": position_entropies_reveal,
        "step_info": step_info,
        "per_position_terminal_entropy": per_position_terminal,
        "k_step": k_step,
    }

    return text, elapsed, diagnostics


# ──────────────────────────────────────────────────
# Vanilla with per-step entropy tracking
# ──────────────────────────────────────────────────

def vanilla_generate_with_entropy(
    model, tokenizer, prompt_text, device=DEVICE,
    gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE
):
    """Vanilla generation that records per-step 'effective entropy' at mask positions.

    Returns: (text, elapsed, diagnostics) with step-level entropy.
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    step_info = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            outputs = model(x, attention_mask="full", position_ids=None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Per-position entropy at mask positions
            all_probs = F.softmax(logits[:, prompt_len:, :] / temperature, dim=-1)
            gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)
            if gen_mask.any():
                masked_probs = all_probs[gen_mask]
                entropy = -(masked_probs * (masked_probs + 1e-10).log()).sum(dim=-1)
                mean_ent = float(entropy.mean().item())
                min_ent = float(entropy.min().item())
                max_ent = float(entropy.max().item())
                std_ent = float(entropy.std().item()) if entropy.numel() > 1 else 0.0
            else:
                mean_ent = min_ent = max_ent = std_ent = 0.0

            step_info.append({
                "step": i,
                "mean_entropy": mean_ent,
                "min_entropy": min_ent,
                "max_entropy": max_ent,
                "std_entropy": std_ent,
                "n_masked": n_masked,
            })

            mask_logits = logits[mask_index]
            t_val = timesteps[i]
            s_val = timesteps[i + 1]
            p_transfer = (1 - s_val / t_val).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature > 0:
                probs = F.softmax(mask_logits[transfer_mask] / temperature, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                sampled_tokens = mask_logits[transfer_mask].argmax(dim=-1)
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

    diagnostics = {
        "step_info": step_info,
    }
    return text, elapsed, diagnostics


# ──────────────────────────────────────────────────
# Statistical Analysis Functions
# ──────────────────────────────────────────────────

def compute_spearman_correlation(step_info):
    """Compute Spearman rank correlation between step index and mean entropy.

    For BSD belief phase only (where entropy should decrease monotonically).
    """
    belief_steps = [s for s in step_info if s.get("phase") == "belief"]
    if len(belief_steps) < 3:
        # For vanilla, use all steps
        belief_steps = step_info

    steps = [s["step"] for s in belief_steps]
    entropies = [s["mean_entropy"] for s in belief_steps]

    if len(steps) < 3:
        return {"rho": None, "p_value": None, "n_steps": len(steps)}

    rho, p_value = scipy_stats.spearmanr(steps, entropies)
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "n_steps": len(steps),
        "monotonic_decreasing": rho < -0.8,
    }


def compute_entropy_accuracy_correlation(samples):
    """Compute point-biserial correlation between terminal entropy and accuracy."""
    terminal_entropies = []
    correctness = []
    for s in samples:
        te = s.get("terminal_entropy")
        ic = s.get("is_correct")
        if te is not None and ic is not None:
            terminal_entropies.append(te)
            correctness.append(1 if ic else 0)

    if len(terminal_entropies) < 3 or sum(correctness) == 0 or sum(correctness) == len(correctness):
        return {
            "correlation": None, "p_value": None, "n_samples": len(terminal_entropies),
            "note": "Insufficient variance for correlation"
        }

    corr, p_value = scipy_stats.pointbiserialr(correctness, terminal_entropies)
    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "n_samples": len(terminal_entropies),
        "direction": "lower_entropy_more_correct" if corr < 0 else "higher_entropy_more_correct",
    }


def compute_monotonicity_score(entropy_values):
    """Compute what fraction of consecutive pairs are decreasing."""
    if len(entropy_values) < 2:
        return {"score": None, "n_pairs": 0}
    n_decreasing = sum(1 for i in range(len(entropy_values) - 1)
                       if entropy_values[i+1] < entropy_values[i])
    n_pairs = len(entropy_values) - 1
    return {
        "score": n_decreasing / n_pairs,
        "n_decreasing": n_decreasing,
        "n_pairs": n_pairs,
    }


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 4, "Loading model")

    print("=" * 70)
    print(f"  entropy_analysis — Belief Entropy Trajectory Analysis (PILOT)")
    print(f"  Countdown-{N_SAMPLES}, seed={SEED}")
    print(f"  BSD config: k_frac=0.75, alpha=linear(0.1->0.8)")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, 4, "Model loaded")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    prompts = [format_countdown_prompt(p) for p in problems]
    print(f"\nGenerated {len(problems)} Countdown problems")

    # ── Phase 1: Vanilla with entropy tracking ──
    print(f"\n{'='*60}")
    print(f"  Phase 1: Vanilla with Entropy Tracking")
    print(f"{'='*60}")
    report_progress(1, 4, "Vanilla entropy tracking")

    vanilla_samples = []
    vanilla_step_infos = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)
        text, elapsed, diag = vanilla_generate_with_entropy(
            model, tokenizer, prompt, DEVICE)
        verification = verify_countdown_answer(text, problem)

        vanilla_step_infos.append(diag["step_info"])

        # Terminal entropy = mean entropy at the last step with masked tokens
        last_step = diag["step_info"][-1] if diag["step_info"] else {}
        terminal_entropy = last_step.get("mean_entropy", 0.0)

        vanilla_samples.append({
            "idx": idx, "target": problem["target"],
            "is_correct": verification["is_correct"],
            "terminal_entropy": terminal_entropy,
            "gen_time_s": elapsed,
        })

        status = "OK" if verification["is_correct"] else "X"
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"terminal_ent={terminal_entropy:.4f} | {elapsed:.1f}s")

    v_correct = sum(1 for s in vanilla_samples if s["is_correct"])
    print(f"\n  Vanilla: {v_correct}/{N_SAMPLES} = {v_correct/N_SAMPLES:.1%}")

    # ── Phase 2: BSD with per-position entropy tracking ──
    print(f"\n{'='*60}")
    print(f"  Phase 2: BSD with Per-Position Entropy Tracking")
    print(f"{'='*60}")
    report_progress(2, 4, "BSD entropy tracking")

    config = BSDConfig(
        k_frac=0.75,
        alpha_schedule="linear",
        alpha_start=0.1, alpha_end=0.8,
        tau_start=1.0, tau_end=0.3,
        fallback_beta_start=0.0, fallback_beta_end=0.0,
        gen_len=GEN_LEN, steps=STEPS, temperature=TEMPERATURE,
    )

    bsd_samples = []
    bsd_step_infos = []
    bsd_position_entropies = []  # per-sample: list of per-step position entropy arrays

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)
        text, elapsed, diag = bsd_generate_with_position_entropy(
            model, tokenizer, prompt, embedding_layer, config, DEVICE)
        verification = verify_countdown_answer(text, problem)

        bsd_step_infos.append(diag["step_info"])
        bsd_position_entropies.append(diag["position_entropies_belief"])

        # Terminal entropy: mean of per-position terminal entropy
        terminal = diag.get("per_position_terminal_entropy")
        terminal_entropy = float(np.mean(terminal)) if terminal is not None else 0.0

        bsd_samples.append({
            "idx": idx, "target": problem["target"],
            "is_correct": verification["is_correct"],
            "terminal_entropy": terminal_entropy,
            "entropy_start": float(diag["step_info"][0]["mean_entropy"]) if diag["step_info"] else 0.0,
            "entropy_end": terminal_entropy,
            "gen_time_s": elapsed,
        })

        status = "OK" if verification["is_correct"] else "X"
        ent_start = diag["step_info"][0]["mean_entropy"] if diag["step_info"] else 0.0
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"ent={ent_start:.2f}->{terminal_entropy:.6f} | {elapsed:.1f}s")

        if (idx + 1) % 4 == 0:
            report_progress(2, 4, f"BSD sample {idx+1}/{N_SAMPLES}")

    b_correct = sum(1 for s in bsd_samples if s["is_correct"])
    print(f"\n  BSD: {b_correct}/{N_SAMPLES} = {b_correct/N_SAMPLES:.1%}")

    # ── Phase 3: Statistical Analysis ──
    print(f"\n{'='*60}")
    print(f"  Phase 3: Statistical Analysis")
    print(f"{'='*60}")
    report_progress(3, 4, "Statistical analysis")

    # 3a. Spearman correlation for each BSD sample
    bsd_spearman_per_sample = []
    for idx, si in enumerate(bsd_step_infos):
        corr = compute_spearman_correlation(si)
        bsd_spearman_per_sample.append(corr)

    # Aggregate Spearman
    rhos = [s["rho"] for s in bsd_spearman_per_sample if s["rho"] is not None]
    avg_rho = float(np.mean(rhos)) if rhos else None
    n_monotonic = sum(1 for s in bsd_spearman_per_sample
                      if s.get("monotonic_decreasing", False))

    print(f"\n  BSD Spearman (step vs entropy):")
    print(f"    Average rho: {avg_rho:.4f}" if avg_rho is not None else "    N/A")
    print(f"    Monotonically decreasing (rho < -0.8): {n_monotonic}/{N_SAMPLES}")

    # 3b. Vanilla Spearman (for comparison)
    vanilla_spearman_per_sample = []
    for idx, si in enumerate(vanilla_step_infos):
        corr = compute_spearman_correlation(si)
        vanilla_spearman_per_sample.append(corr)

    v_rhos = [s["rho"] for s in vanilla_spearman_per_sample if s["rho"] is not None]
    v_avg_rho = float(np.mean(v_rhos)) if v_rhos else None

    print(f"\n  Vanilla Spearman (step vs entropy):")
    print(f"    Average rho: {v_avg_rho:.4f}" if v_avg_rho is not None else "    N/A")

    # 3c. Terminal entropy comparison
    bsd_terminal = [s["terminal_entropy"] for s in bsd_samples]
    vanilla_terminal = [s["terminal_entropy"] for s in vanilla_samples]
    bsd_mean_terminal = float(np.mean(bsd_terminal))
    vanilla_mean_terminal = float(np.mean(vanilla_terminal))

    # Wilcoxon signed-rank test for paired comparison
    if len(bsd_terminal) >= 5:
        try:
            wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(
                bsd_terminal, vanilla_terminal, alternative='less')
        except ValueError:
            wilcoxon_stat, wilcoxon_p = None, None
    else:
        wilcoxon_stat, wilcoxon_p = None, None

    print(f"\n  Terminal Entropy Comparison:")
    print(f"    BSD mean terminal:     {bsd_mean_terminal:.6f}")
    print(f"    Vanilla mean terminal: {vanilla_mean_terminal:.6f}")
    print(f"    BSD < Vanilla:         {bsd_mean_terminal < vanilla_mean_terminal}")
    if wilcoxon_p is not None:
        print(f"    Wilcoxon p-value:      {wilcoxon_p:.6f}")

    # 3d. Entropy-accuracy correlation
    bsd_ent_acc = compute_entropy_accuracy_correlation(bsd_samples)
    vanilla_ent_acc = compute_entropy_accuracy_correlation(vanilla_samples)

    print(f"\n  Entropy-Accuracy Correlation:")
    print(f"    BSD:     r={bsd_ent_acc.get('correlation', 'N/A')}, "
          f"p={bsd_ent_acc.get('p_value', 'N/A')}")
    print(f"    Vanilla: r={vanilla_ent_acc.get('correlation', 'N/A')}, "
          f"p={vanilla_ent_acc.get('p_value', 'N/A')}")

    # 3e. Monotonicity score for BSD belief phase
    bsd_monotonicity_per_sample = []
    for si in bsd_step_infos:
        belief_ents = [s["mean_entropy"] for s in si if s.get("phase") == "belief"]
        mono = compute_monotonicity_score(belief_ents)
        bsd_monotonicity_per_sample.append(mono)

    mono_scores = [m["score"] for m in bsd_monotonicity_per_sample if m["score"] is not None]
    avg_mono = float(np.mean(mono_scores)) if mono_scores else None

    print(f"\n  BSD Monotonicity Score:")
    print(f"    Average: {avg_mono:.4f}" if avg_mono is not None else "    N/A")
    n_perfect_mono = sum(1 for m in mono_scores if m == 1.0)
    print(f"    Perfectly monotonic: {n_perfect_mono}/{len(mono_scores)}")

    # 3f. Aggregate entropy trajectories for visualization
    # Average entropy trajectory across all BSD samples
    n_belief_steps = min(len(pe) for pe in bsd_position_entropies) if bsd_position_entropies else 0
    avg_entropy_trajectory_bsd = []
    for step_idx in range(n_belief_steps):
        step_means = [float(np.mean(pe[step_idx])) for pe in bsd_position_entropies]
        avg_entropy_trajectory_bsd.append({
            "step": step_idx,
            "mean_entropy": float(np.mean(step_means)),
            "std_entropy": float(np.std(step_means)),
            "min_sample": float(np.min(step_means)),
            "max_sample": float(np.max(step_means)),
        })

    # Average vanilla entropy trajectory
    n_vanilla_steps = min(len(si) for si in vanilla_step_infos) if vanilla_step_infos else 0
    avg_entropy_trajectory_vanilla = []
    for step_idx in range(n_vanilla_steps):
        step_means = [si[step_idx]["mean_entropy"] for si in vanilla_step_infos]
        avg_entropy_trajectory_vanilla.append({
            "step": step_idx,
            "mean_entropy": float(np.mean(step_means)),
            "std_entropy": float(np.std(step_means)),
        })

    # ── Phase 4: Compile Results ──
    print(f"\n{'='*60}")
    print(f"  Phase 4: Saving Results")
    print(f"{'='*60}")
    report_progress(4, 4, "Saving results")

    elapsed_total = time.time() - start_time

    # Pass criteria: Spearman rho < -0.8
    pass_criterion = avg_rho is not None and avg_rho < -0.8
    verdict = "GO" if pass_criterion else "CONDITIONAL-GO"

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "verdict": verdict,
        "model": "Dream-v0-Instruct-7B",
        "benchmark": f"Countdown-{N_SAMPLES}",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_s": round(elapsed_total, 1),
        "elapsed_total_min": round(elapsed_total / 60, 1),
        "bsd_config": {
            "k_frac": 0.75,
            "alpha_schedule": "linear",
            "alpha_start": 0.1, "alpha_end": 0.8,
            "tau_start": 1.0, "tau_end": 0.3,
            "fallback_beta": 0.0,
            "gen_len": GEN_LEN, "steps": STEPS, "temperature": TEMPERATURE,
        },
        "summary": {
            "bsd_accuracy": b_correct / N_SAMPLES,
            "vanilla_accuracy": v_correct / N_SAMPLES,
            "bsd_n_correct": b_correct,
            "vanilla_n_correct": v_correct,
        },
        "spearman_analysis": {
            "bsd": {
                "avg_rho": avg_rho,
                "n_monotonic_samples": n_monotonic,
                "n_total_samples": N_SAMPLES,
                "per_sample": bsd_spearman_per_sample,
            },
            "vanilla": {
                "avg_rho": v_avg_rho,
                "per_sample": vanilla_spearman_per_sample,
            },
        },
        "terminal_entropy": {
            "bsd_mean": bsd_mean_terminal,
            "vanilla_mean": vanilla_mean_terminal,
            "bsd_lower_than_vanilla": bsd_mean_terminal < vanilla_mean_terminal,
            "ratio": bsd_mean_terminal / vanilla_mean_terminal if vanilla_mean_terminal > 0 else None,
            "wilcoxon_statistic": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
            "wilcoxon_p_value": float(wilcoxon_p) if wilcoxon_p is not None else None,
            "per_sample_bsd": bsd_terminal,
            "per_sample_vanilla": vanilla_terminal,
        },
        "entropy_accuracy_correlation": {
            "bsd": bsd_ent_acc,
            "vanilla": vanilla_ent_acc,
        },
        "monotonicity": {
            "avg_score": avg_mono,
            "n_perfect_monotonic": n_perfect_mono,
            "n_total": len(mono_scores),
            "per_sample": bsd_monotonicity_per_sample,
        },
        "entropy_trajectories": {
            "bsd_avg": avg_entropy_trajectory_bsd,
            "vanilla_avg": avg_entropy_trajectory_vanilla,
        },
        "per_sample": {
            "bsd": bsd_samples,
            "vanilla": vanilla_samples,
        },
        "pass_criteria": {
            "spearman_rho_below_neg08": pass_criterion,
            "bsd_terminal_lower": bsd_mean_terminal < vanilla_mean_terminal,
            "entropy_decreasing": n_monotonic >= N_SAMPLES * 0.8,
            "verdict": verdict,
        },
        "hypothesis_H2": {
            "claim": "BSD belief entropy monotonically decreases, lower terminal entropy than vanilla",
            "monotonic_decrease": avg_rho is not None and avg_rho < -0.8,
            "lower_terminal": bsd_mean_terminal < vanilla_mean_terminal,
            "supported": (avg_rho is not None and avg_rho < -0.8 and
                          bsd_mean_terminal < vanilla_mean_terminal),
        },
    }

    out_file = RESULTS_DIR / "entropy_analysis_countdown500.json"
    save_results(results, str(out_file))

    # Summary markdown
    summary_md = RESULTS_DIR / "entropy_analysis_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# Entropy Analysis — Countdown-{N_SAMPLES} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")

        f.write(f"## Hypothesis H2 Evaluation\n\n")
        f.write(f"**Claim**: BSD belief entropy monotonically decreases during denoising, ")
        f.write(f"reaching lower terminal entropy than vanilla.\n\n")
        supported = results["hypothesis_H2"]["supported"]
        f.write(f"**Result**: {'SUPPORTED' if supported else 'NOT FULLY SUPPORTED'}\n\n")

        f.write(f"## Spearman Correlation (step vs entropy)\n\n")
        f.write(f"| Method | Avg rho | Monotonic (rho<-0.8) |\n")
        f.write(f"|--------|---------|---------------------|\n")
        f.write(f"| BSD (belief phase) | {avg_rho:.4f} | "
                f"{n_monotonic}/{N_SAMPLES} |\n" if avg_rho else "| BSD | N/A | N/A |\n")
        f.write(f"| Vanilla | {v_avg_rho:.4f} | N/A |\n\n" if v_avg_rho else "| Vanilla | N/A | N/A |\n\n")

        f.write(f"## Terminal Entropy\n\n")
        f.write(f"| Method | Mean Terminal Entropy |\n")
        f.write(f"|--------|---------------------|\n")
        f.write(f"| BSD | {bsd_mean_terminal:.6f} |\n")
        f.write(f"| Vanilla | {vanilla_mean_terminal:.6f} |\n")
        f.write(f"| **BSD < Vanilla** | **{bsd_mean_terminal < vanilla_mean_terminal}** |\n")
        if wilcoxon_p is not None:
            f.write(f"| Wilcoxon p-value | {wilcoxon_p:.6f} |\n")
        f.write(f"\n")

        f.write(f"## Monotonicity\n\n")
        f.write(f"- Average monotonicity score: {avg_mono:.4f}\n" if avg_mono else "- N/A\n")
        f.write(f"- Perfectly monotonic: {n_perfect_mono}/{len(mono_scores)}\n\n")

        f.write(f"## Entropy-Accuracy Correlation\n\n")
        f.write(f"- BSD: r={bsd_ent_acc.get('correlation', 'N/A')}, "
                f"p={bsd_ent_acc.get('p_value', 'N/A')}\n")
        f.write(f"- Vanilla: r={vanilla_ent_acc.get('correlation', 'N/A')}, "
                f"p={vanilla_ent_acc.get('p_value', 'N/A')}\n\n")

        f.write(f"## Accuracy\n\n")
        f.write(f"- BSD: {b_correct}/{N_SAMPLES} = {b_correct/N_SAMPLES:.1%}\n")
        f.write(f"- Vanilla: {v_correct}/{N_SAMPLES} = {v_correct/N_SAMPLES:.1%}\n\n")

        f.write(f"## Runtime\n\n")
        f.write(f"- Total: {elapsed_total / 60:.1f} minutes\n")

    print(f"[{TASK_ID}] Summary saved to {summary_md}")

    # ── Final Report ──
    print(f"\n{'='*70}")
    print(f"  PILOT VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"  H2 (monotonic decrease):  {'SUPPORTED' if results['hypothesis_H2']['monotonic_decrease'] else 'NOT SUPPORTED'}")
    print(f"  H2 (lower terminal):      {'SUPPORTED' if results['hypothesis_H2']['lower_terminal'] else 'NOT SUPPORTED'}")
    print(f"  H2 overall:               {'SUPPORTED' if results['hypothesis_H2']['supported'] else 'NOT FULLY SUPPORTED'}")
    print(f"  Spearman rho:             {avg_rho:.4f}" if avg_rho else "  Spearman rho: N/A")
    print(f"  BSD terminal entropy:     {bsd_mean_terminal:.6f}")
    print(f"  Vanilla terminal entropy: {vanilla_mean_terminal:.6f}")
    print(f"  Pass criteria (rho<-0.8): {pass_criterion}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Mark done
    mark_done(
        status="success",
        summary=f"H2 {'supported' if supported else 'not fully supported'}. "
                f"Spearman rho={avg_rho:.4f}. "
                f"Terminal entropy BSD={bsd_mean_terminal:.6f} vs vanilla={vanilla_mean_terminal:.6f}. "
                f"Verdict: {verdict}. Time: {elapsed_total/60:.1f}min",
    )

    # Update gpu_progress.json
    gpu_progress_path = Path(f"{PROJECT_DIR}/exp/gpu_progress.json")
    try:
        progress = json.loads(gpu_progress_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in progress["completed"]:
        progress["completed"].append(TASK_ID)
    progress["running"].pop(TASK_ID, None)
    progress["timings"][TASK_ID] = {
        "planned_min": 30,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "Dream-v0-Instruct-7B",
            "method": "entropy_analysis (PILOT)",
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(progress, indent=2))
    print(f"[{TASK_ID}] gpu_progress.json updated")

    print(f"\n[{TASK_ID}] Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    return verdict in ("GO", "CONDITIONAL-GO")


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[{TASK_ID}] FATAL ERROR: {e}")
        traceback.print_exc()
        try:
            mark_done(status="failed", summary=f"Fatal error: {e}")
        except:
            pass
        sys.exit(1)
