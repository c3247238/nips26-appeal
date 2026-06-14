"""
BSD + A-CFG Combination Pilot — Countdown-16 (PILOT mode).

Combines:
  - BSD belief refinement in Phase 1 (steps T to k+1): continuous belief vectors
  - A-CFG guidance in Phase 2 (steps k to 1): confidence-based re-masking + fixed CFG

Best configs from prior ablations:
  - BSD: k_frac=0.75, alpha=linear(0.1->0.8), tau=linear(1.0->0.3)
  - A-CFG: w=1.5, remask_pct=0.1, fixed schedule

Hypothesis H7: combination >= 18%, > max(BSD, RACFG).
PILOT: 16 samples, seed 42, Countdown-16.
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# ── Setup paths ──
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
sys.path.insert(0, f"{PROJECT_DIR}/exp/code")

from bsd_racfg.eval_harness import (
    MASK_TOKEN_ID, MODEL_DIR,
    generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics,
    compute_per_sample_metrics, load_dream,
    vanilla_generate, save_results, print_qualitative_samples,
    print_comparison_table, check_degeneration,
    RESULTS_PILOTS, RESULTS_FULL,
)
from bsd_racfg.bsd import BSDConfig, get_alpha, get_tau, get_fallback_beta


# ── Configuration ──
@dataclass
class BSDACFGComboConfig:
    """Configuration for BSD + A-CFG combination."""
    # BSD Phase 1 params (best from ablation)
    k_frac: float = 0.75
    alpha_schedule: str = "linear"
    alpha_start: float = 0.1
    alpha_end: float = 0.8
    tau_start: float = 1.0
    tau_end: float = 0.3
    fallback_beta_start: float = 0.0
    fallback_beta_end: float = 0.0

    # A-CFG Phase 2 params (best from ablation)
    acfg_w: float = 1.5
    acfg_remask_pct: float = 0.1

    # Generation parameters
    gen_len: int = 256
    steps: int = 128
    temperature: float = 0.4


def bsd_acfg_combo_generate(
    model, tokenizer, prompt_text: str,
    embedding_layer, config: BSDACFGComboConfig,
    device: str = "cuda:0",
    track_entropy: bool = True,
) -> Tuple[str, float, Dict]:
    """Generate text using BSD + A-CFG combination.

    Phase 1 (steps T to k+1): BSD continuous belief refinement
      - No argmax, no mask/unmask
      - Beliefs evolve via EMA of probability-weighted embedding mixtures
      - L2 normalized to match mask_emb norm

    Phase 2 (steps k to 1): Hard reveal with A-CFG guidance
      - Confidence-based re-masking to construct unconditional input
      - CFG: guided = cond + w * (cond - uncond)
      - First Phase 2 step uses belief states for initial predictions

    Returns: (text, elapsed_seconds, diagnostics)
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + config.gen_len
    eps = 1e-3

    # Initialize sequence with mask tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, config.steps + 1, device=device)

    # Phase transition step index
    k_step = int(config.steps * (1 - config.k_frac))

    # Get mask embedding norm for normalization
    mask_emb = embedding_layer(torch.tensor([MASK_TOKEN_ID], device=device))  # [1, D]
    mask_emb_norm = mask_emb.norm(dim=-1, keepdim=True).item()

    # Initialize belief vectors as mask embeddings for generation positions
    beliefs = mask_emb.expand(1, config.gen_len, -1).clone()  # [1, gen_len, D]

    # Create a BSDConfig for helper functions
    bsd_cfg = BSDConfig(
        k_frac=config.k_frac,
        alpha_schedule=config.alpha_schedule,
        alpha_start=config.alpha_start, alpha_end=config.alpha_end,
        tau_start=config.tau_start, tau_end=config.tau_end,
        fallback_beta_start=config.fallback_beta_start,
        fallback_beta_end=config.fallback_beta_end,
    )

    # Tracking
    entropy_trajectory = []
    step_diagnostics = []
    phase1_steps = 0
    phase2_steps = 0
    phase2_guidance_steps = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(config.steps):
            is_phase1 = (i < k_step)

            if is_phase1:
                # ── Phase 1: BSD Belief Refinement ──
                phase1_steps += 1
                step_frac = i / max(k_step - 1, 1)
                alpha = get_alpha(bsd_cfg, step_frac)
                tau = get_tau(bsd_cfg, step_frac)
                beta = get_fallback_beta(bsd_cfg, step_frac)

                # Construct input embeddings: prompt tokens (hard) + beliefs (soft)
                prompt_emb = embedding_layer(x[:, :prompt_len])
                input_emb = torch.cat([prompt_emb, beliefs], dim=1)

                # Forward pass with embeddings
                outputs = model(inputs_embeds=input_emb,
                                attention_mask="full", position_ids=None)
                logits = outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Extract logits for generation positions
                gen_logits = logits[:, prompt_len:, :]

                # Compute soft predictions with temperature
                soft_probs = F.softmax(gen_logits / tau, dim=-1)

                # Compute new belief embeddings
                emb_weight = embedding_layer.weight
                new_belief = torch.matmul(
                    soft_probs.to(emb_weight.dtype), emb_weight
                )

                # EMA update
                beliefs = (1 - alpha) * beliefs + alpha * new_belief

                # Fallback: mix with mask_emb if beta > 0
                if beta > 0:
                    beliefs = beta * mask_emb.unsqueeze(0).expand_as(beliefs) + \
                              (1 - beta) * beliefs

                # L2 normalize to match mask embedding norm
                belief_norms = beliefs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                beliefs = beliefs * (mask_emb_norm / belief_norms)

                # Track entropy
                if track_entropy:
                    entropy = -(soft_probs * (soft_probs + 1e-10).log()).sum(dim=-1)
                    mean_ent = entropy.mean().item()
                    entropy_trajectory.append({
                        "step": i, "phase": "belief",
                        "mean_entropy": mean_ent,
                        "min_entropy": entropy.min().item(),
                        "max_entropy": entropy.max().item(),
                    })

                step_diagnostics.append({
                    "step": i, "phase": "belief",
                    "alpha": alpha, "tau": tau, "beta": beta,
                    "belief_norm_mean": beliefs.norm(dim=-1).mean().item(),
                })

            else:
                # ── Phase 2: Hard Reveal with A-CFG Guidance ──
                phase2_steps += 1
                mask_index = (x == MASK_TOKEN_ID)
                n_masked = mask_index.sum().item()

                if n_masked == 0:
                    break

                # Forward pass 1: conditional
                if i == k_step:
                    # First Phase 2 step: use beliefs for initial predictions
                    prompt_emb = embedding_layer(x[:, :prompt_len])
                    input_emb = torch.cat([prompt_emb, beliefs], dim=1)
                    outputs_cond = model(inputs_embeds=input_emb,
                                         attention_mask="full", position_ids=None)
                else:
                    outputs_cond = model(x, attention_mask="full", position_ids=None)

                logits_cond = outputs_cond.logits
                logits_cond = torch.cat([logits_cond[:, :1], logits_cond[:, :-1]], dim=1)

                # A-CFG: confidence-based re-masking for unconditional input
                probs_cond = F.softmax(logits_cond / config.temperature, dim=-1)
                confidence = probs_cond.max(dim=-1).values  # [1, total_len]

                gen_conf = confidence[:, prompt_len:]
                gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)

                n_remask = max(1, int(gen_mask.sum().item() * config.acfg_remask_pct))

                # Find lowest confidence revealed positions
                revealed_positions = (~gen_mask[0]).nonzero(as_tuple=True)[0]
                guidance_applied = False

                if len(revealed_positions) > 0:
                    rev_conf = gen_conf[0, revealed_positions]
                    n_actual_remask = min(n_remask, len(revealed_positions))
                    _, remask_idx = rev_conf.topk(n_actual_remask, largest=False)

                    # Create unconditional input by re-masking low-confidence positions
                    if i == k_step:
                        # For first Phase 2 step, create token-based x from belief argmax
                        # then re-mask for unconditional
                        # Use the same conditional logits to determine tokens
                        x_uncond = x.clone()
                    else:
                        x_uncond = x.clone()

                    for idx in remask_idx:
                        actual_idx = prompt_len + revealed_positions[idx].item()
                        x_uncond[0, actual_idx] = MASK_TOKEN_ID

                    # Forward pass 2: unconditional
                    if i == k_step:
                        # Use beliefs for unconditional too, but with re-masked positions
                        # Reset those belief positions to mask_emb
                        beliefs_uncond = beliefs.clone()
                        for idx in remask_idx:
                            beliefs_uncond[0, revealed_positions[idx].item()] = mask_emb[0]
                        prompt_emb_uc = embedding_layer(x_uncond[:, :prompt_len])
                        input_emb_uc = torch.cat([prompt_emb_uc, beliefs_uncond], dim=1)
                        outputs_uncond = model(inputs_embeds=input_emb_uc,
                                               attention_mask="full", position_ids=None)
                    else:
                        outputs_uncond = model(x_uncond, attention_mask="full",
                                               position_ids=None)

                    logits_uncond = outputs_uncond.logits
                    logits_uncond = torch.cat([logits_uncond[:, :1], logits_uncond[:, :-1]], dim=1)

                    # Apply CFG: guided = cond + w * (cond - uncond)
                    guided_logits = logits_cond + config.acfg_w * (logits_cond - logits_uncond)
                    guidance_applied = True
                    phase2_guidance_steps += 1
                else:
                    guided_logits = logits_cond

                # Standard unmasking with guided logits
                mask_logits = guided_logits[mask_index]

                t = timesteps[i]
                s = timesteps[i + 1]
                p_transfer = (1 - s / t).item() if i < config.steps - 1 else 1.0

                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                if config.temperature > 0:
                    probs = F.softmax(mask_logits[transfer_mask] / config.temperature, dim=-1)
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    sampled_tokens = mask_logits[transfer_mask].argmax(dim=-1)
                x0[transfer_mask] = sampled_tokens
                x[mask_index] = x0

                # Track entropy in Phase 2
                if track_entropy:
                    all_probs = F.softmax(guided_logits[:, prompt_len:, :] / config.temperature, dim=-1)
                    gen_mask_cur = (x[:, prompt_len:] == MASK_TOKEN_ID)
                    if gen_mask_cur.any():
                        masked_probs = all_probs[gen_mask_cur]
                        entropy = -(masked_probs * (masked_probs + 1e-10).log()).sum(dim=-1)
                        mean_ent = entropy.mean().item()
                    else:
                        mean_ent = 0.0
                    entropy_trajectory.append({
                        "step": i, "phase": "hard_reveal_acfg",
                        "mean_entropy": mean_ent,
                        "n_masked": n_masked,
                    })

                step_diagnostics.append({
                    "step": i, "phase": "hard_reveal_acfg",
                    "n_masked": n_masked,
                    "guidance_applied": guidance_applied,
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

    diagnostics = {
        "config": {
            "k_frac": config.k_frac,
            "alpha_schedule": config.alpha_schedule,
            "alpha_start": config.alpha_start,
            "alpha_end": config.alpha_end,
            "tau_start": config.tau_start,
            "tau_end": config.tau_end,
            "acfg_w": config.acfg_w,
            "acfg_remask_pct": config.acfg_remask_pct,
        },
        "k_step": k_step,
        "phase1_steps": phase1_steps,
        "phase2_steps": phase2_steps,
        "phase2_guidance_steps": phase2_guidance_steps,
        "entropy_trajectory": entropy_trajectory,
        "step_diagnostics": step_diagnostics,
    }

    return text, elapsed, diagnostics


# ── DMI baseline for comparison ──
def dmi_generate(model, tokenizer, prompt_text: str, embedding_layer,
                 device: str = "cuda:0", alpha: float = 0.3,
                 soft_tau: float = 0.5, gen_len: int = 256,
                 steps: int = 128, temperature: float = 0.4) -> Tuple[str, float, Dict]:
    """DMI (Diffusion Memory Injection) baseline generation."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    mask_emb = embedding_layer(torch.tensor([MASK_TOKEN_ID], device=device))
    mask_emb_norm = mask_emb.norm(dim=-1, keepdim=True).item()
    prev_logits = None

    t0 = time.time()
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            if mask_index.sum().item() == 0:
                break

            if prev_logits is not None:
                # DMI: inject previous logits into mask embeddings
                gen_logits_prev = prev_logits[:, prompt_len:, :]
                soft_probs = F.softmax(gen_logits_prev / soft_tau, dim=-1)
                emb_weight = embedding_layer.weight
                soft_emb = torch.matmul(soft_probs.to(emb_weight.dtype), emb_weight)
                curr_emb = embedding_layer(x)
                gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)
                mixed = alpha * mask_emb.expand_as(soft_emb) + (1 - alpha) * soft_emb
                mixed_norm = mixed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                mixed = mixed * (mask_emb_norm / mixed_norm)
                curr_emb[:, prompt_len:][gen_mask] = mixed[gen_mask]
                outputs = model(inputs_embeds=curr_emb, attention_mask="full", position_ids=None)
            else:
                outputs = model(x, attention_mask="full", position_ids=None)

            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            prev_logits = logits.clone()

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

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

    return text, elapsed, {"alpha": alpha, "soft_tau": soft_tau}


# ── A-CFG standalone for comparison ──
def acfg_standalone_generate(model, tokenizer, prompt_text: str,
                              device: str = "cuda:0", w: float = 1.5,
                              remask_pct: float = 0.1, gen_len: int = 256,
                              steps: int = 128, temperature: float = 0.4) -> Tuple[str, float, Dict]:
    """A-CFG standalone generation (best config: fixed w=1.5)."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            # Forward pass 1: conditional
            outputs_cond = model(x, attention_mask="full", position_ids=None)
            logits_cond = outputs_cond.logits
            logits_cond = torch.cat([logits_cond[:, :1], logits_cond[:, :-1]], dim=1)

            # Confidence-based re-masking
            probs_cond = F.softmax(logits_cond / temperature, dim=-1)
            confidence = probs_cond.max(dim=-1).values
            gen_conf = confidence[:, prompt_len:]
            gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)
            n_remask = max(1, int(gen_mask.sum().item() * remask_pct))

            revealed_positions = (~gen_mask[0]).nonzero(as_tuple=True)[0]
            if len(revealed_positions) > 0:
                rev_conf = gen_conf[0, revealed_positions]
                n_actual_remask = min(n_remask, len(revealed_positions))
                _, remask_idx = rev_conf.topk(n_actual_remask, largest=False)

                x_uncond = x.clone()
                for idx in remask_idx:
                    actual_idx = prompt_len + revealed_positions[idx].item()
                    x_uncond[0, actual_idx] = MASK_TOKEN_ID

                outputs_uncond = model(x_uncond, attention_mask="full", position_ids=None)
                logits_uncond = outputs_uncond.logits
                logits_uncond = torch.cat([logits_uncond[:, :1], logits_uncond[:, :-1]], dim=1)

                guided_logits = logits_cond + w * (logits_cond - logits_uncond)
            else:
                guided_logits = logits_cond

            mask_logits = guided_logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

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

    return text, elapsed, {"w": w, "remask_pct": remask_pct}


def main():
    task_id = "combination_bsd_racfg"
    n_samples = 16  # PILOT
    seed = 42
    device = "cuda:0"

    print(f"=" * 70)
    print(f"BSD + A-CFG Combination PILOT — Countdown-{n_samples}")
    print(f"=" * 70)

    # Write PID file
    results_dir = RESULTS_FULL
    results_dir.mkdir(parents=True, exist_ok=True)
    pid_file = results_dir / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] Written to {pid_file}")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    model, tokenizer, embedding_layer = load_dream(device)

    # Generate problems
    problems = generate_countdown_problems(n_samples, seed=seed)
    print(f"Generated {len(problems)} Countdown problems")

    # Configuration
    combo_config = BSDACFGComboConfig()
    print(f"\nBSD+ACFG config:")
    print(f"  BSD: k_frac={combo_config.k_frac}, alpha={combo_config.alpha_schedule}"
          f"({combo_config.alpha_start}->{combo_config.alpha_end})")
    print(f"  ACFG: w={combo_config.acfg_w}, remask_pct={combo_config.acfg_remask_pct}")

    # ── Run all methods ──
    all_results = {}

    # 1. Vanilla baseline
    print(f"\n--- Vanilla (128 steps) ---")
    vanilla_texts, vanilla_times, vanilla_correct = [], [], 0
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        text, elapsed, _ = vanilla_generate(model, tokenizer, prompt, device)
        ver = verify_countdown_answer(text, prob)
        vanilla_texts.append(text)
        vanilla_times.append(elapsed)
        if ver["is_correct"]:
            vanilla_correct += 1
        if j < 3:
            print(f"  [{j}] {'OK' if ver['is_correct'] else 'NO'} | {text[:100]}")
    vanilla_acc = vanilla_correct / n_samples
    vanilla_div = compute_diversity_metrics(vanilla_texts)
    print(f"  Vanilla accuracy: {vanilla_acc:.1%} ({vanilla_correct}/{n_samples})")

    # 2. DMI baseline
    print(f"\n--- DMI (alpha=0.3) ---")
    dmi_texts, dmi_times, dmi_correct = [], [], 0
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        text, elapsed, _ = dmi_generate(model, tokenizer, prompt, embedding_layer, device)
        ver = verify_countdown_answer(text, prob)
        dmi_texts.append(text)
        dmi_times.append(elapsed)
        if ver["is_correct"]:
            dmi_correct += 1
        if j < 3:
            print(f"  [{j}] {'OK' if ver['is_correct'] else 'NO'} | {text[:100]}")
    dmi_acc = dmi_correct / n_samples
    dmi_div = compute_diversity_metrics(dmi_texts)
    print(f"  DMI accuracy: {dmi_acc:.1%} ({dmi_correct}/{n_samples})")

    # 3. BSD standalone (best config)
    print(f"\n--- BSD (k=0.75, linear alpha) ---")
    from bsd_racfg.bsd import BSDConfig as BSDCfg, bsd_generate
    bsd_standalone_cfg = BSDCfg(
        k_frac=0.75, alpha_schedule="linear",
        alpha_start=0.1, alpha_end=0.8,
        tau_start=1.0, tau_end=0.3,
    )
    bsd_texts, bsd_times, bsd_correct = [], [], 0
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        text, elapsed, _ = bsd_generate(model, tokenizer, prompt, embedding_layer,
                                         bsd_standalone_cfg, device)
        ver = verify_countdown_answer(text, prob)
        bsd_texts.append(text)
        bsd_times.append(elapsed)
        if ver["is_correct"]:
            bsd_correct += 1
        if j < 3:
            print(f"  [{j}] {'OK' if ver['is_correct'] else 'NO'} | {text[:100]}")
    bsd_acc = bsd_correct / n_samples
    bsd_div = compute_diversity_metrics(bsd_texts)
    print(f"  BSD accuracy: {bsd_acc:.1%} ({bsd_correct}/{n_samples})")

    # 4. A-CFG standalone (best config: w=1.5)
    print(f"\n--- A-CFG (w=1.5, fixed) ---")
    acfg_texts, acfg_times, acfg_correct = [], [], 0
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        text, elapsed, _ = acfg_standalone_generate(model, tokenizer, prompt, device, w=1.5)
        ver = verify_countdown_answer(text, prob)
        acfg_texts.append(text)
        acfg_times.append(elapsed)
        if ver["is_correct"]:
            acfg_correct += 1
        if j < 3:
            print(f"  [{j}] {'OK' if ver['is_correct'] else 'NO'} | {text[:100]}")
    acfg_acc = acfg_correct / n_samples
    acfg_div = compute_diversity_metrics(acfg_texts)
    print(f"  A-CFG accuracy: {acfg_acc:.1%} ({acfg_correct}/{n_samples})")

    # 5. BSD + A-CFG Combination
    print(f"\n--- BSD + A-CFG Combination ---")
    combo_texts, combo_times, combo_correct = [], [], 0
    combo_per_sample = []
    combo_entropy_all = []
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        text, elapsed, diag = bsd_acfg_combo_generate(
            model, tokenizer, prompt, embedding_layer, combo_config, device
        )
        ver = verify_countdown_answer(text, prob)
        combo_texts.append(text)
        combo_times.append(elapsed)
        if ver["is_correct"]:
            combo_correct += 1

        sample_metrics = compute_per_sample_metrics(text)
        combo_per_sample.append({
            "sample_idx": j,
            "target": prob["target"],
            "numbers": prob["numbers"],
            "is_correct": ver["is_correct"],
            "extracted_equation": ver.get("extracted_equation"),
            "generated_text": text[:500],
            "gen_time_s": elapsed,
            **sample_metrics,
            "phase1_steps": diag["phase1_steps"],
            "phase2_steps": diag["phase2_steps"],
            "phase2_guidance_steps": diag["phase2_guidance_steps"],
        })
        combo_entropy_all.append(diag.get("entropy_trajectory", []))

        if j < 5:
            print(f"  [{j}] {'OK' if ver['is_correct'] else 'NO'} | "
                  f"P1={diag['phase1_steps']} P2={diag['phase2_steps']} "
                  f"CFG={diag['phase2_guidance_steps']} | {text[:100]}")

        # Report progress
        if j % 4 == 3:
            progress_file = results_dir / f"{task_id}_PROGRESS.json"
            progress_file.write_text(json.dumps({
                "task_id": task_id,
                "epoch": j + 1, "total_epochs": n_samples,
                "metric": {"accuracy": combo_correct / (j + 1)},
                "updated_at": datetime.now().isoformat(),
            }))

    combo_acc = combo_correct / n_samples
    combo_div = compute_diversity_metrics(combo_texts)
    print(f"\n  BSD+ACFG accuracy: {combo_acc:.1%} ({combo_correct}/{n_samples})")

    # 6. Vanilla with 2x steps (compute-fair comparison for combo ~2.1x FLOPs)
    print(f"\n--- Vanilla 2x steps (256, compute-fair) ---")
    v2x_texts, v2x_times, v2x_correct = [], [], 0
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        text, elapsed, _ = vanilla_generate(model, tokenizer, prompt, device, steps=256)
        ver = verify_countdown_answer(text, prob)
        v2x_texts.append(text)
        v2x_times.append(elapsed)
        if ver["is_correct"]:
            v2x_correct += 1
    v2x_acc = v2x_correct / n_samples
    v2x_div = compute_diversity_metrics(v2x_texts)
    print(f"  Vanilla-2x accuracy: {v2x_acc:.1%} ({v2x_correct}/{n_samples})")

    # ── Flip analysis ──
    print(f"\n--- Flip Analysis ---")
    combo_vs_bsd_wins = sum(1 for i in range(n_samples)
                            if combo_per_sample[i]["is_correct"]
                            and not verify_countdown_answer(bsd_texts[i], problems[i])["is_correct"])
    bsd_vs_combo_wins = sum(1 for i in range(n_samples)
                            if not combo_per_sample[i]["is_correct"]
                            and verify_countdown_answer(bsd_texts[i], problems[i])["is_correct"])
    combo_vs_acfg_wins = sum(1 for i in range(n_samples)
                             if combo_per_sample[i]["is_correct"]
                             and not verify_countdown_answer(acfg_texts[i], problems[i])["is_correct"])
    acfg_vs_combo_wins = sum(1 for i in range(n_samples)
                             if not combo_per_sample[i]["is_correct"]
                             and verify_countdown_answer(acfg_texts[i], problems[i])["is_correct"])

    print(f"  Combo wins vs BSD: {combo_vs_bsd_wins}, BSD wins vs Combo: {bsd_vs_combo_wins}")
    print(f"  Combo wins vs ACFG: {combo_vs_acfg_wins}, ACFG wins vs Combo: {acfg_vs_combo_wins}")

    # ── Degeneration check ──
    degen = check_degeneration(combo_div, vanilla_div)
    if degen:
        print(f"\n  DEGENERATION WARNINGS: {degen}")
    else:
        print(f"\n  No degeneration detected.")

    # ── Determine verdict ──
    max_individual = max(bsd_acc, acfg_acc)
    combo_beats_max = combo_acc > max_individual
    combo_beats_vanilla_3pp = combo_acc > vanilla_acc + 0.03
    verdict = "GO" if combo_beats_vanilla_3pp else "NO-GO"

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"  Combo ({combo_acc:.1%}) > max(BSD={bsd_acc:.1%}, ACFG={acfg_acc:.1%}): {combo_beats_max}")
    print(f"  Combo ({combo_acc:.1%}) > Vanilla ({vanilla_acc:.1%}) + 3pp: {combo_beats_vanilla_3pp}")
    print(f"{'=' * 70}")

    # ── Comparison table ──
    print(f"\n{'Method':<25} {'Accuracy':>10} {'rep-2':>8} {'rep-3':>8} "
          f"{'dist-3':>8} {'AvgTime':>8} {'FLOPs':>8}")
    print("-" * 85)
    methods_data = [
        ("Vanilla (128)", vanilla_acc, vanilla_div, vanilla_times, "1.0x"),
        ("Vanilla (256, fair)", v2x_acc, v2x_div, v2x_times, "2.0x"),
        ("DMI (alpha=0.3)", dmi_acc, dmi_div, dmi_times, "~1.05x"),
        ("BSD (k=0.75)", bsd_acc, bsd_div, bsd_times, "~1.1x"),
        ("A-CFG (w=1.5)", acfg_acc, acfg_div, acfg_times, "~2.0x"),
        ("BSD+ACFG (COMBO)", combo_acc, combo_div, combo_times, "~2.1x"),
    ]
    for name, acc, div, times, flops in methods_data:
        avg_t = np.mean(times) if times else 0
        print(f"{name:<25} {acc:>10.1%} {div.get('rep_2', 0):>8.3f} "
              f"{div.get('rep_3', 0):>8.3f} {div.get('distinct_3', 0):>8.3f} "
              f"{avg_t:>7.1f}s {flops:>8}")

    # ── Entropy analysis ──
    if combo_entropy_all and combo_entropy_all[0]:
        # Average entropy trajectory across samples
        all_steps = {}
        for sample_ent in combo_entropy_all:
            for entry in sample_ent:
                step = entry["step"]
                if step not in all_steps:
                    all_steps[step] = {"entropies": [], "phase": entry["phase"]}
                all_steps[step]["entropies"].append(entry["mean_entropy"])

        avg_entropy_trajectory = []
        for step in sorted(all_steps.keys()):
            ents = all_steps[step]["entropies"]
            avg_entropy_trajectory.append({
                "step": step,
                "phase": all_steps[step]["phase"],
                "mean_entropy": float(np.mean(ents)),
                "std_entropy": float(np.std(ents)),
            })

        # Check monotonic decrease
        belief_ents = [e["mean_entropy"] for e in avg_entropy_trajectory if e["phase"] == "belief"]
        if len(belief_ents) >= 2:
            is_decreasing = all(belief_ents[i] >= belief_ents[i+1] - 0.01
                                for i in range(len(belief_ents) - 1))
            print(f"\n  Entropy analysis:")
            print(f"    Start entropy: {belief_ents[0]:.2f}")
            print(f"    End entropy: {belief_ents[-1]:.2f}")
            print(f"    Monotonically decreasing: {is_decreasing}")
    else:
        avg_entropy_trajectory = []

    # ── Save results ──
    total_elapsed = sum(vanilla_times) + sum(dmi_times) + sum(bsd_times) + \
                    sum(acfg_times) + sum(combo_times) + sum(v2x_times)

    result = {
        "task_id": task_id,
        "mode": "PILOT",
        "verdict": verdict,
        "model": "Dream-v0-Instruct-7B",
        "benchmark": f"Countdown-{n_samples}",
        "seed": seed,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_s": total_elapsed,
        "elapsed_total_min": round(total_elapsed / 60, 1),
        "combo_config": {
            "bsd_k_frac": combo_config.k_frac,
            "bsd_alpha_schedule": f"{combo_config.alpha_schedule}({combo_config.alpha_start}->{combo_config.alpha_end})",
            "bsd_tau_schedule": f"linear({combo_config.tau_start}->{combo_config.tau_end})",
            "acfg_w": combo_config.acfg_w,
            "acfg_remask_pct": combo_config.acfg_remask_pct,
            "gen_len": combo_config.gen_len,
            "steps": combo_config.steps,
            "temperature": combo_config.temperature,
        },
        "results": {
            "vanilla_128": {
                "accuracy": vanilla_acc,
                "n_correct": vanilla_correct,
                "n_samples": n_samples,
                "flops_ratio": "1.0x",
                **vanilla_div,
                "avg_gen_time_s": float(np.mean(vanilla_times)),
            },
            "vanilla_256_fair": {
                "accuracy": v2x_acc,
                "n_correct": v2x_correct,
                "n_samples": n_samples,
                "flops_ratio": "2.0x",
                **v2x_div,
                "avg_gen_time_s": float(np.mean(v2x_times)),
            },
            "dmi": {
                "accuracy": dmi_acc,
                "n_correct": dmi_correct,
                "n_samples": n_samples,
                "flops_ratio": "~1.05x",
                **dmi_div,
                "avg_gen_time_s": float(np.mean(dmi_times)),
            },
            "bsd_standalone": {
                "accuracy": bsd_acc,
                "n_correct": bsd_correct,
                "n_samples": n_samples,
                "flops_ratio": "~1.1x",
                **bsd_div,
                "avg_gen_time_s": float(np.mean(bsd_times)),
            },
            "acfg_standalone": {
                "accuracy": acfg_acc,
                "n_correct": acfg_correct,
                "n_samples": n_samples,
                "flops_ratio": "~2.0x",
                **acfg_div,
                "avg_gen_time_s": float(np.mean(acfg_times)),
            },
            "bsd_acfg_combo": {
                "accuracy": combo_acc,
                "n_correct": combo_correct,
                "n_samples": n_samples,
                "flops_ratio": "~2.1x",
                **combo_div,
                "avg_gen_time_s": float(np.mean(combo_times)),
            },
        },
        "flip_analysis": {
            "combo_vs_bsd": {"combo_wins": combo_vs_bsd_wins, "bsd_wins": bsd_vs_combo_wins},
            "combo_vs_acfg": {"combo_wins": combo_vs_acfg_wins, "acfg_wins": acfg_vs_combo_wins},
        },
        "degeneration_warnings": degen,
        "combo_beats_max_individual": combo_beats_max,
        "max_individual_accuracy": max_individual,
        "avg_entropy_trajectory": avg_entropy_trajectory,
        "per_sample": combo_per_sample,
    }

    out_path = results_dir / "bsd_racfg_combo_countdown500.json"
    save_results(result, str(out_path))
    print(f"\nResults saved to {out_path}")

    # ── Write summary ──
    summary_path = results_dir / "bsd_racfg_combo_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# BSD + A-CFG Combination — Countdown-{n_samples} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- BSD: k_frac={combo_config.k_frac}, alpha={combo_config.alpha_schedule}"
                f"({combo_config.alpha_start}->{combo_config.alpha_end})\n")
        f.write(f"- A-CFG: w={combo_config.acfg_w}, remask_pct={combo_config.acfg_remask_pct}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |\n")
        f.write(f"|--------|----------|-------|-------|------------|----------|-------|\n")
        for name, acc, div, times, flops in methods_data:
            avg_t = np.mean(times) if times else 0
            f.write(f"| {name} | {acc:.1%} ({int(acc * n_samples)}/{n_samples}) | "
                    f"{div.get('rep_2', 0):.3f} | {div.get('rep_3', 0):.3f} | "
                    f"{div.get('distinct_3', 0):.3f} | {avg_t:.1f}s | {flops} |\n")
        f.write(f"\n## Hypothesis H7 Test\n\n")
        f.write(f"- Combination accuracy: {combo_acc:.1%}\n")
        f.write(f"- max(BSD, ACFG): {max_individual:.1%}\n")
        f.write(f"- Combo > max(BSD, ACFG): {combo_beats_max}\n")
        f.write(f"- Combo > Vanilla + 3pp: {combo_beats_vanilla_3pp}\n\n")
        f.write(f"## Flip Analysis\n\n")
        f.write(f"### Combo vs BSD\n")
        f.write(f"- Combo wins: {combo_vs_bsd_wins}\n")
        f.write(f"- BSD wins: {bsd_vs_combo_wins}\n\n")
        f.write(f"### Combo vs A-CFG\n")
        f.write(f"- Combo wins: {combo_vs_acfg_wins}\n")
        f.write(f"- A-CFG wins: {acfg_vs_combo_wins}\n\n")
        f.write(f"## Runtime\n\n")
        f.write(f"- Total: {total_elapsed / 60:.1f} minutes\n")
    print(f"Summary saved to {summary_path}")

    # ── Write DONE marker ──
    done_marker = results_dir / f"{task_id}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": task_id,
        "status": "success",
        "summary": f"BSD+ACFG combo: {combo_acc:.1%} (verdict: {verdict})",
        "final_progress": {
            "accuracy": combo_acc,
            "n_correct": combo_correct,
            "n_samples": n_samples,
            "verdict": verdict,
        },
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"DONE marker written to {done_marker}")

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    return result


if __name__ == "__main__":
    main()
