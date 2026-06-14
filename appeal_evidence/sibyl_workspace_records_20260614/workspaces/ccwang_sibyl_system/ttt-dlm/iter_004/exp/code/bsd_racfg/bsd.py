"""
Belief-State Diffusion (BSD) — Core Module.

Replaces hard mask embeddings with continuously evolving belief vectors
during denoising Phase 1 (steps T to k+1), then performs standard
confidence-based unmasking in Phase 2 (steps k to 1).

Key design:
  - Belief vector = probability-weighted embedding mixture via EMA
  - L2 normalization to match mask_emb norm (prevent OOD)
  - Graceful degradation via fallback_beta mixing with mask_emb
  - Tracks belief entropy trajectory for analysis
"""
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .eval_harness import MASK_TOKEN_ID


@dataclass
class BSDConfig:
    """Configuration for Belief-State Diffusion."""
    # Phase transition: k = fraction of total steps for hard-reveal phase
    # e.g., k_frac=0.5 means T/2 steps belief refinement, T/2 steps hard reveal
    k_frac: float = 0.5

    # Alpha schedule for EMA update rate
    # 'linear': ramp from alpha_start to alpha_end
    # 'cosine': cosine ramp from alpha_start to alpha_end
    # 'constant': fixed alpha_start
    alpha_schedule: str = "linear"
    alpha_start: float = 0.1
    alpha_end: float = 0.8

    # Temperature schedule for softmax over logits
    tau_start: float = 1.0
    tau_end: float = 0.3

    # Fallback: mix belief with mask_emb for graceful degradation
    # fallback_beta=0 means full belief replacement (no fallback)
    # fallback_beta=0.9 means 90% mask_emb + 10% belief (very conservative)
    fallback_beta_start: float = 0.0  # 0 = no fallback (default)
    fallback_beta_end: float = 0.0

    # Generation parameters
    gen_len: int = 256
    steps: int = 128
    temperature: float = 0.4  # sampling temperature in Phase 2


def get_alpha(config: BSDConfig, step_frac: float) -> float:
    """Get alpha EMA rate for current step fraction (0=start, 1=end of Phase 1)."""
    if config.alpha_schedule == "constant":
        return config.alpha_start
    elif config.alpha_schedule == "cosine":
        # Cosine ramp from alpha_start to alpha_end
        t = 0.5 * (1 - math.cos(math.pi * step_frac))
        return config.alpha_start + (config.alpha_end - config.alpha_start) * t
    else:  # linear
        return config.alpha_start + (config.alpha_end - config.alpha_start) * step_frac


def get_tau(config: BSDConfig, step_frac: float) -> float:
    """Get temperature for current step fraction."""
    return config.tau_start + (config.tau_end - config.tau_start) * step_frac


def get_fallback_beta(config: BSDConfig, step_frac: float) -> float:
    """Get fallback mixing ratio for current step fraction."""
    return config.fallback_beta_start + \
           (config.fallback_beta_end - config.fallback_beta_start) * step_frac


def bsd_generate(model, tokenizer, prompt_text: str,
                 embedding_layer, config: BSDConfig,
                 device: str = "cuda:0",
                 track_entropy: bool = True) -> Tuple[str, float, Dict]:
    """Generate text using Belief-State Diffusion.

    Phase 1 (steps T to k+1): Continuous belief refinement
      - No argmax sampling, no mask/unmask
      - Beliefs evolve via EMA of probability-weighted embedding mixtures
      - L2 normalized to match mask_emb norm

    Phase 2 (steps k to 1): Standard confidence-based unmasking
      - From belief states (not hard masks)
      - Optional RACFG guidance (applied externally)

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
    k_step = int(config.steps * (1 - config.k_frac))  # step index where Phase 2 begins

    # Get mask embedding norm for normalization
    mask_emb = embedding_layer(torch.tensor([MASK_TOKEN_ID], device=device))  # [1, D]
    mask_emb_norm = mask_emb.norm(dim=-1, keepdim=True).item()

    # Initialize belief vectors as mask embeddings for generation positions
    # beliefs: [1, gen_len, D] — only for generation positions
    gen_positions = torch.arange(prompt_len, max_length, device=device)
    beliefs = mask_emb.expand(1, config.gen_len, -1).clone()  # [1, gen_len, D]

    # Tracking
    entropy_trajectory = []  # per-step mean entropy
    step_diagnostics = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(config.steps):
            is_phase1 = (i < k_step)

            if is_phase1:
                # ── Phase 1: Belief Refinement ──
                step_frac = i / max(k_step - 1, 1)
                alpha = get_alpha(config, step_frac)
                tau = get_tau(config, step_frac)
                beta = get_fallback_beta(config, step_frac)

                # Construct input embeddings: prompt tokens (hard) + beliefs (soft)
                prompt_emb = embedding_layer(x[:, :prompt_len])  # [1, prompt_len, D]
                input_emb = torch.cat([prompt_emb, beliefs], dim=1)  # [1, total_len, D]

                # Forward pass with embeddings
                outputs = model(inputs_embeds=input_emb,
                                attention_mask="full", position_ids=None)
                logits = outputs.logits
                # Dream uses shifted logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Extract logits for generation positions only
                gen_logits = logits[:, prompt_len:, :]  # [1, gen_len, V]

                # Compute soft predictions with temperature
                soft_probs = F.softmax(gen_logits / tau, dim=-1)  # [1, gen_len, V]

                # Compute new belief embeddings: weighted sum of token embeddings
                emb_weight = embedding_layer.weight  # [V, D]
                new_belief = torch.matmul(
                    soft_probs.to(emb_weight.dtype), emb_weight
                )  # [1, gen_len, D]

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
                    entropy = -(soft_probs * (soft_probs + 1e-10).log()).sum(dim=-1)  # [1, gen_len]
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
                # ── Phase 2: Hard Reveal with Confidence-Based Unmasking ──
                mask_index = (x == MASK_TOKEN_ID)
                n_masked = mask_index.sum().item()

                if n_masked == 0:
                    break

                # First Phase 2 step: convert beliefs to initial token predictions
                if i == k_step:
                    # Use beliefs for the final predictions
                    prompt_emb = embedding_layer(x[:, :prompt_len])
                    input_emb = torch.cat([prompt_emb, beliefs], dim=1)
                    outputs = model(inputs_embeds=input_emb,
                                    attention_mask="full", position_ids=None)
                else:
                    # Standard forward with current tokens
                    outputs = model(x, attention_mask="full", position_ids=None)

                logits = outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                mask_logits = logits[mask_index]

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
                    all_probs = F.softmax(logits[:, prompt_len:, :] / config.temperature, dim=-1)
                    gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)
                    if gen_mask.any():
                        masked_probs = all_probs[gen_mask]
                        entropy = -(masked_probs * (masked_probs + 1e-10).log()).sum(dim=-1)
                        mean_ent = entropy.mean().item()
                    else:
                        mean_ent = 0.0
                    entropy_trajectory.append({
                        "step": i, "phase": "hard_reveal",
                        "mean_entropy": mean_ent,
                        "n_masked": n_masked,
                    })

                step_diagnostics.append({
                    "step": i, "phase": "hard_reveal",
                    "n_masked": n_masked,
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
            "fallback_beta_start": config.fallback_beta_start,
            "fallback_beta_end": config.fallback_beta_end,
        },
        "k_step": k_step,
        "entropy_trajectory": entropy_trajectory,
        "step_diagnostics": step_diagnostics,
    }

    return text, elapsed, diagnostics
