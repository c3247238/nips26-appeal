"""
Reasoning-Aware Classifier-Free Guidance (RACFG) — Core Module.

Enhances A-CFG with:
  1. Cross-step JSD stability scoring (vs single-step confidence)
  2. Stability-guided re-masking for unconditional input construction
  3. Temporal scheduling (theory-driven: zero early, max late)
  4. Position-adaptive guidance weights

The module also includes a standard A-CFG implementation for baseline comparison.
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
class RACFGConfig:
    """Configuration for Reasoning-Aware CFG."""
    # Re-mask percentage: fraction of positions to re-mask for unconditional input
    remask_pct: float = 0.10  # 10%

    # Guidance weight
    w_base: float = 1.0
    w_max: float = 2.0  # cap to prevent over-extrapolation

    # EMA smoothing for cross-step logit history
    stability_ema_lambda: float = 0.7

    # Temporal schedule type
    # 'fixed': constant w_base
    # 'linear': linear ramp 0 -> w_max
    # 'cosine': cosine ramp
    # 'threshold_70_30': zero at mask_rate>70%, ramp at 30-70%, max at <30%
    schedule_type: str = "threshold_70_30"

    # Generation parameters
    gen_len: int = 256
    steps: int = 128
    temperature: float = 0.4


@dataclass
class ACFGConfig:
    """Configuration for standard A-CFG (baseline)."""
    # Re-mask percentage based on confidence
    remask_pct: float = 0.10
    # Fixed guidance weight
    w: float = 1.0
    # Generation parameters
    gen_len: int = 256
    steps: int = 128
    temperature: float = 0.4


def _compute_jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute Jensen-Shannon Divergence between two probability distributions.

    Args:
        p, q: [N, V] probability distributions
    Returns:
        [N] JSD values
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (p / (m + 1e-10) + 1e-10).log()).sum(dim=-1)
    kl_qm = (q * (q / (m + 1e-10) + 1e-10).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _get_temporal_weight(config: RACFGConfig, mask_rate: float,
                          step_frac: float) -> float:
    """Compute temporal guidance weight scaling factor.

    Args:
        config: RACFG configuration
        mask_rate: fraction of positions still masked (1.0 = all masked, 0.0 = none)
        step_frac: fraction of total steps completed (0.0 = start, 1.0 = end)

    Returns:
        Guidance weight scale factor [0, 1]
    """
    if config.schedule_type == "fixed":
        return 1.0
    elif config.schedule_type == "linear":
        return step_frac  # 0 at start, 1 at end
    elif config.schedule_type == "cosine":
        return 0.5 * (1 - math.cos(math.pi * step_frac))
    elif config.schedule_type == "threshold_70_30":
        # Theory-driven: no guidance when insufficient info, max at low mask rate
        if mask_rate > 0.70:
            return 0.0
        elif mask_rate > 0.30:
            # Linear ramp from 0 to 1 in the 30-70% range
            return (0.70 - mask_rate) / 0.40
        else:
            return 1.0
    else:
        return 1.0


def racfg_generate(model, tokenizer, prompt_text: str,
                   config: RACFGConfig, device: str = "cuda:0",
                   track_stability: bool = True) -> Tuple[str, float, Dict]:
    """Generate text using RACFG (Reasoning-Aware Classifier-Free Guidance).

    At each denoising step:
    1. Maintain EMA-smoothed logit history for stability scoring
    2. Compute JSD-based stability scores per position
    3. Select least-stable positions for re-masking
    4. Two forward passes: conditional (original) + unconditional (re-masked)
    5. Apply position-adaptive, temporally-scheduled guidance

    Returns: (text, elapsed_seconds, diagnostics)
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + config.gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, config.steps + 1, device=device)

    # Logit EMA history for stability scoring
    ema_probs = None  # [1, total_len, V] — EMA-smoothed probabilities
    lam = config.stability_ema_lambda

    # Tracking
    stability_data = []
    step_diagnostics = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(config.steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()

            if n_masked == 0:
                break

            step_frac = i / config.steps
            mask_rate = n_masked / config.gen_len

            # === Forward pass 1: Conditional (original sequence) ===
            outputs_cond = model(x, attention_mask="full", position_ids=None)
            logits_cond = outputs_cond.logits
            logits_cond = torch.cat([logits_cond[:, :1], logits_cond[:, :-1]], dim=1)

            # Current probabilities
            current_probs = F.softmax(logits_cond / config.temperature, dim=-1)

            # === Stability scoring via JSD ===
            # Compute temporal weight first to decide if guidance is needed
            w_scale = _get_temporal_weight(config, mask_rate, step_frac)

            if ema_probs is not None and w_scale > 0:
                # Compute JSD between current and EMA-smoothed probs
                stability_scores = 1.0 - _compute_jsd(
                    current_probs.view(-1, current_probs.shape[-1]),
                    ema_probs.view(-1, ema_probs.shape[-1])
                ).view(current_probs.shape[:-1])  # [1, total_len]

                # Extract stability for generation (masked) positions
                gen_stability = stability_scores[:, prompt_len:]  # [1, gen_len]
                gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)  # [1, gen_len]

                # Select least-stable masked positions for re-masking
                n_remask = max(1, int(gen_mask.sum().item() * config.remask_pct))

                # Get stability scores only for masked positions
                masked_stability = gen_stability.clone()
                masked_stability[~gen_mask] = float('inf')  # don't re-mask revealed tokens

                # Find least stable positions (lowest stability = highest uncertainty)
                _, remask_indices = masked_stability[0].topk(
                    min(n_remask, gen_mask[0].sum().item()),
                    largest=False
                )

                # === Forward pass 2: Unconditional (with re-masked positions) ===
                x_uncond = x.clone()
                # Re-mask the least stable positions
                for idx in remask_indices:
                    actual_idx = prompt_len + idx.item()
                    if x_uncond[0, actual_idx] != MASK_TOKEN_ID:
                        # This position was already revealed — re-mask it
                        x_uncond[0, actual_idx] = MASK_TOKEN_ID

                # Also re-mask random non-masked positions if remask targets were already masked
                # (since we want m% of positions to differ from conditional input)
                n_actually_remasked = (x_uncond != x).sum().item()
                if n_actually_remasked < n_remask:
                    # Some target positions were already masked, try additional positions
                    revealed_indices = (x[0, prompt_len:] != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                    if len(revealed_indices) > 0:
                        extra_need = n_remask - n_actually_remasked
                        # Pick lowest-stability revealed positions
                        rev_stab = gen_stability[0, revealed_indices]
                        n_extra = min(extra_need, len(revealed_indices))
                        _, extra_idx = rev_stab.topk(n_extra, largest=False)
                        for ei in extra_idx:
                            actual_idx = prompt_len + revealed_indices[ei].item()
                            x_uncond[0, actual_idx] = MASK_TOKEN_ID

                outputs_uncond = model(x_uncond, attention_mask="full", position_ids=None)
                logits_uncond = outputs_uncond.logits
                logits_uncond = torch.cat([logits_uncond[:, :1], logits_uncond[:, :-1]], dim=1)

                # === Apply CFG: guided = cond + w * (cond - uncond) ===
                w_effective = config.w_base * w_scale
                w_effective = min(w_effective, config.w_max)

                # Position-adaptive: scale by (1 - stability) per position
                # Lower stability = more guidance
                pos_weights = (1.0 - stability_scores).unsqueeze(-1)  # [1, total_len, 1]
                pos_weights = pos_weights.clamp(0, 1)

                guided_logits = logits_cond + w_effective * pos_weights * (logits_cond - logits_uncond)

                # Track stability data
                if track_stability and i % 4 == 0:  # every 4 steps to save memory
                    gen_stab_masked = gen_stability[gen_mask].float().cpu().numpy()
                    stability_data.append({
                        "step": i,
                        "mask_rate": mask_rate,
                        "w_scale": w_scale,
                        "w_effective": w_effective,
                        "n_remasked": n_remask,
                        "stability_mean": float(gen_stab_masked.mean()) if len(gen_stab_masked) > 0 else 0,
                        "stability_std": float(gen_stab_masked.std()) if len(gen_stab_masked) > 0 else 0,
                        "stability_min": float(gen_stab_masked.min()) if len(gen_stab_masked) > 0 else 0,
                        "stability_max": float(gen_stab_masked.max()) if len(gen_stab_masked) > 0 else 0,
                    })
            else:
                # No guidance: first step or w_scale=0
                guided_logits = logits_cond

            # Update EMA
            if ema_probs is None:
                ema_probs = current_probs.clone()
            else:
                ema_probs = lam * ema_probs + (1 - lam) * current_probs

            # === Standard unmasking with guided logits ===
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

            step_diagnostics.append({
                "step": i, "n_masked": n_masked,
                "mask_rate": mask_rate, "w_scale": w_scale,
                "guidance_applied": w_scale > 0 and ema_probs is not None,
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
            "remask_pct": config.remask_pct,
            "w_base": config.w_base,
            "w_max": config.w_max,
            "schedule_type": config.schedule_type,
            "stability_ema_lambda": config.stability_ema_lambda,
        },
        "stability_data": stability_data,
        "step_diagnostics": step_diagnostics,
    }

    return text, elapsed, diagnostics


def acfg_generate(model, tokenizer, prompt_text: str,
                  config: ACFGConfig, device: str = "cuda:0") -> Tuple[str, float, Dict]:
    """Generate text using standard A-CFG (Adaptive CFG baseline).

    A-CFG uses single-step confidence to select positions for re-masking.
    No cross-step memory, no temporal scheduling, fixed guidance weight.

    Returns: (text, elapsed_seconds, diagnostics)
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + config.gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, config.steps + 1, device=device)

    step_diagnostics = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(config.steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            # Forward pass 1: conditional
            outputs_cond = model(x, attention_mask="full", position_ids=None)
            logits_cond = outputs_cond.logits
            logits_cond = torch.cat([logits_cond[:, :1], logits_cond[:, :-1]], dim=1)

            # Confidence-based re-masking (A-CFG style)
            probs_cond = F.softmax(logits_cond / config.temperature, dim=-1)
            confidence = probs_cond.max(dim=-1).values  # [1, total_len]

            # Select low-confidence positions for re-masking
            gen_conf = confidence[:, prompt_len:]  # [1, gen_len]
            gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)

            n_remask = max(1, int(gen_mask.sum().item() * config.remask_pct))

            # Mask out already-masked positions (they can't be re-masked)
            conf_for_selection = gen_conf.clone()
            conf_for_selection[gen_mask] = float('inf')  # don't select already-masked

            # Find lowest confidence revealed positions
            revealed_positions = (~gen_mask[0]).nonzero(as_tuple=True)[0]
            if len(revealed_positions) > 0:
                rev_conf = gen_conf[0, revealed_positions]
                n_actual_remask = min(n_remask, len(revealed_positions))
                _, remask_idx = rev_conf.topk(n_actual_remask, largest=False)

                # Create unconditional input
                x_uncond = x.clone()
                for idx in remask_idx:
                    actual_idx = prompt_len + revealed_positions[idx].item()
                    x_uncond[0, actual_idx] = MASK_TOKEN_ID

                # Forward pass 2: unconditional
                outputs_uncond = model(x_uncond, attention_mask="full", position_ids=None)
                logits_uncond = outputs_uncond.logits
                logits_uncond = torch.cat([logits_uncond[:, :1], logits_uncond[:, :-1]], dim=1)

                # Apply CFG
                guided_logits = logits_cond + config.w * (logits_cond - logits_uncond)
            else:
                guided_logits = logits_cond

            # Standard unmasking
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

            step_diagnostics.append({
                "step": i, "n_masked": n_masked,
                "guidance_applied": len(revealed_positions) > 0 if 'revealed_positions' in dir() else False,
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
        "config": {"remask_pct": config.remask_pct, "w": config.w},
        "step_diagnostics": step_diagnostics,
    }

    return text, elapsed, diagnostics


@dataclass
class ScheduledACFGConfig:
    """Configuration for A-CFG with temporal scheduling (confidence-based re-masking + scheduled w)."""
    # Re-mask percentage based on confidence
    remask_pct: float = 0.10
    # Base guidance weight
    w_base: float = 1.0
    w_max: float = 2.0  # cap
    # Temporal schedule type: 'fixed', 'linear', 'cosine', 'threshold_70_30'
    schedule_type: str = "threshold_70_30"
    # Generation parameters
    gen_len: int = 256
    steps: int = 128
    temperature: float = 0.4


def scheduled_acfg_generate(model, tokenizer, prompt_text: str,
                            config: ScheduledACFGConfig,
                            device: str = "cuda:0") -> Tuple[str, float, Dict]:
    """Generate text using A-CFG with temporal scheduling.

    Combines A-CFG's confidence-based re-masking (which works on Dream-7B)
    with temporal scheduling of guidance weight (theory-driven: zero early,
    max late). This tests hypothesis H6: scheduled > fixed by >=2pp.

    Returns: (text, elapsed_seconds, diagnostics)
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + config.gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, config.steps + 1, device=device)

    step_diagnostics = []
    t0 = time.time()

    # Create a RACFGConfig just for the temporal weight function
    _sched_cfg = RACFGConfig(schedule_type=config.schedule_type)

    with torch.no_grad():
        for i in range(config.steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            step_frac = i / config.steps
            mask_rate = n_masked / config.gen_len

            # Compute temporal weight scaling
            w_scale = _get_temporal_weight(_sched_cfg, mask_rate, step_frac)

            # Forward pass 1: conditional
            outputs_cond = model(x, attention_mask="full", position_ids=None)
            logits_cond = outputs_cond.logits
            logits_cond = torch.cat([logits_cond[:, :1], logits_cond[:, :-1]], dim=1)

            guidance_applied = False
            w_effective = 0.0

            if w_scale > 0:
                # Confidence-based re-masking (A-CFG style)
                probs_cond = F.softmax(logits_cond / config.temperature, dim=-1)
                confidence = probs_cond.max(dim=-1).values  # [1, total_len]

                gen_conf = confidence[:, prompt_len:]  # [1, gen_len]
                gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)

                n_remask = max(1, int(gen_mask.sum().item() * config.remask_pct))

                # Find lowest confidence revealed positions
                revealed_positions = (~gen_mask[0]).nonzero(as_tuple=True)[0]
                if len(revealed_positions) > 0:
                    rev_conf = gen_conf[0, revealed_positions]
                    n_actual_remask = min(n_remask, len(revealed_positions))
                    _, remask_idx = rev_conf.topk(n_actual_remask, largest=False)

                    # Create unconditional input
                    x_uncond = x.clone()
                    for idx in remask_idx:
                        actual_idx = prompt_len + revealed_positions[idx].item()
                        x_uncond[0, actual_idx] = MASK_TOKEN_ID

                    # Forward pass 2: unconditional
                    outputs_uncond = model(x_uncond, attention_mask="full", position_ids=None)
                    logits_uncond = outputs_uncond.logits
                    logits_uncond = torch.cat([logits_uncond[:, :1], logits_uncond[:, :-1]], dim=1)

                    # Apply scheduled CFG
                    w_effective = min(config.w_base * w_scale, config.w_max)
                    guided_logits = logits_cond + w_effective * (logits_cond - logits_uncond)
                    guidance_applied = True
                else:
                    guided_logits = logits_cond
            else:
                guided_logits = logits_cond

            # Standard unmasking
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

            step_diagnostics.append({
                "step": i, "n_masked": n_masked,
                "mask_rate": round(mask_rate, 4),
                "w_scale": round(w_scale, 4),
                "w_effective": round(w_effective, 4),
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

    # Count guidance steps
    n_guidance_steps = sum(1 for s in step_diagnostics if s["guidance_applied"])

    diagnostics = {
        "config": {
            "remask_pct": config.remask_pct,
            "w_base": config.w_base,
            "w_max": config.w_max,
            "schedule_type": config.schedule_type,
        },
        "n_guidance_steps": n_guidance_steps,
        "n_total_steps": len(step_diagnostics),
        "step_diagnostics": step_diagnostics,
    }

    return text, elapsed, diagnostics
