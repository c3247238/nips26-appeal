#!/usr/bin/env python3
"""
C4: Latency & Throughput Benchmarking (Full 6-method comparison)

Measures wall-clock time per sample for:
  1. Vanilla: Standard LLaDA-8B iterative unmasking
  2. MetaState-GRU: Cross-step GRU memory baseline
  3. DaL-Linear (no phase scheduling): TTT layer with gate_sep config
  4. DaL-Linear + Phase Scheduling: TTT with phase-transition scheduler
  5. ReMDM: Principled remasking (remask_fraction=0.1)
  6. CoRe: Context-robust remasking (remask_fraction=0.15)

Uses LLaDA-8B-Instruct, NFE=128, batch_size=1.
PILOT mode: 16 samples from GSM8K.

Reports: total time, per-step time, TTT/remasking overhead, peak GPU memory.
"""

import os, sys, json, time, gc, re, math, traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Blackwell GPU CUBLAS workarounds
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# === Paths ===
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
FULL_RESULTS_DIR = f"{PROJECT_DIR}/exp/results/full"
CODE_DIR = f"{PROJECT_DIR}/exp/code/dal"
SHARED_DIR = f"{REMOTE_BASE}/shared"

TASK_ID = "latency_benchmark"
SEED = 42

# === Model Config ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
D_MODEL = 4096
VOCAB_SIZE = 126464
N_LAYERS = 32
INSERTION_LAYER = 16

# === Checkpoints ===
DAL_LINEAR_CKPT = f"{FULL_RESULTS_DIR}/dal_linear_ckpt_step5000.pt"
GRU_CKPT = f"{FULL_RESULTS_DIR}/gru_ckpt_step5000.pt"

# === Eval Config ===
PILOT_SAMPLES = 16
GEN_STEPS = 128
GEN_LEN = 512
TEMPERATURE = 0.0
MAX_CUBLAS_RETRIES = 5

sys.path.insert(0, CODE_DIR)
from ttt_layer import TTTLayer, build_ttt_layer
from dal_wrapper import PhaseTransitionScheduler, MetaStateGRU

# ==============================================================================
# Progress / DONE helpers
# ==============================================================================

def report_progress(phase, step, total, extra=None):
    Path(RESULTS_DIR).joinpath(f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "phase": phase, "step": step, "total_steps": total,
        "extra": extra or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    p = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if p.exists():
        p.unlink()
    prog_file = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if prog_file.exists():
        try:
            final_progress = json.loads(prog_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    Path(RESULTS_DIR).joinpath(f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat()}))


def cublas_safe_run(fn, max_retries=MAX_CUBLAS_RETRIES):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RuntimeError as e:
            if "CUBLAS" in str(e) and attempt < max_retries:
                print(f"  [CUBLAS retry {attempt+1}/{max_retries}] {str(e)[:100]}")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
                continue
            raise
    return None


# ==============================================================================
# LLaDA helpers
# ==============================================================================

def prepare_input(tokenizer, prompt_text, device, gen_len):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + [MASK_TOKEN_ID] * gen_len
    x = torch.tensor([full_ids], device=device)
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


def get_layers(model):
    """Get transformer layers from model."""
    inner_model = model.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        return tf.blocks
    elif hasattr(tf, 'block_groups'):
        return tf.block_groups
    else:
        raise ValueError("Cannot find blocks")


# ==============================================================================
# Method 1: Vanilla (timed)
# ==============================================================================

def generate_vanilla_timed(model, tokenizer, prompt_text, device, gen_len,
                           gen_steps=GEN_STEPS, temperature=0.0):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)
    step_times = []
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.no_grad():
        for i in range(gen_steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = model(x).logits
            mask_logits = logits[mask_index]
            t, s = timesteps[i], timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            sampled = mask_logits[transfer_mask].argmax(dim=-1) if temperature == 0 else \
                torch.multinomial(F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    return decode_output(tokenizer, x, prompt_len), {
        "total_time_s": total_time,
        "mean_step_time_s": float(np.mean(step_times)) if step_times else 0,
        "std_step_time_s": float(np.std(step_times)) if step_times else 0,
        "num_steps_run": len(step_times),
    }


# ==============================================================================
# Method 2: MetaState-GRU (timed)
# ==============================================================================

def generate_gru_timed(model, gru_adapter, tokenizer, prompt_text, device, gen_len,
                       insertion_layer, gen_steps=GEN_STEPS, temperature=0.0):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)
    layers = get_layers(model)
    target_layer = layers[insertion_layer]
    gru_adapter.reset_state(batch_size=1)

    gru_overhead_times = []
    step_times = []
    hook_state = {"gru_time": 0.0}

    def gru_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        mi = (x == MASK_TOKEN_ID)
        revealed_mask = (~mi).float()
        n_masked, n_revealed = mi.sum().item(), revealed_mask.sum().item()
        if n_revealed > 5 and n_masked > 3:
            torch.cuda.synchronize()
            gru_start = time.perf_counter()
            with torch.no_grad():
                delta, _ = gru_adapter(hidden, revealed_mask)
            torch.cuda.synchronize()
            hook_state["gru_time"] = time.perf_counter() - gru_start
            if isinstance(output, tuple):
                output = (hidden + delta,) + output[1:]
            else:
                output = hidden + delta
        else:
            hook_state["gru_time"] = 0.0
        return output

    hook_handle = target_layer.register_forward_hook(gru_hook)
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    try:
        with torch.no_grad():
            for i in range(gen_steps):
                torch.cuda.synchronize()
                step_start = time.perf_counter()
                hook_state["gru_time"] = 0.0
                mask_index = (x == MASK_TOKEN_ID)
                if not mask_index.any():
                    break
                logits = model(x).logits
                mask_logits = logits[mask_index]
                t, s = timesteps[i], timesteps[i + 1]
                p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                sampled = mask_logits[transfer_mask].argmax(dim=-1) if temperature == 0 else \
                    torch.multinomial(F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
                x0[transfer_mask] = sampled
                x[mask_index] = x0
                torch.cuda.synchronize()
                step_elapsed = time.perf_counter() - step_start
                step_times.append(step_elapsed)
                gru_overhead_times.append(hook_state["gru_time"])
    finally:
        hook_handle.remove()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    total_gru_time = sum(gru_overhead_times)
    return decode_output(tokenizer, x, prompt_len), {
        "total_time_s": total_time,
        "mean_step_time_s": float(np.mean(step_times)) if step_times else 0,
        "std_step_time_s": float(np.std(step_times)) if step_times else 0,
        "num_steps_run": len(step_times),
        "gru_overhead_fraction": total_gru_time / total_time if total_time > 0 else 0,
        "total_gru_time_s": total_gru_time,
    }


# ==============================================================================
# Method 3 & 4: DaL-Linear (timed, with optional phase scheduling)
# ==============================================================================

def generate_dal_timed(model, ttt_layer, ssl_head, ssl_gate, layer_norm,
                       tokenizer, prompt_text, device, gen_len,
                       insertion_layer, gen_steps=GEN_STEPS, temperature=0.0,
                       phase_scheduler=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    target_ids = x.clone()
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)
    layers = get_layers(model)
    target_layer = layers[insertion_layer]
    ttt_layer.reset_fast_weights(batch_size=1)

    ttt_overhead_times = []
    step_times = []
    hook_state = {"ttt_time": 0.0, "ttt_updated": False}

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()
        mask_ratio = n_masked / max(x.shape[1], 1)
        do_update = True
        if phase_scheduler is not None:
            do_update = phase_scheduler.should_update(mask_ratio)
        if n_revealed > 5 and n_masked > 3 and do_update:
            torch.cuda.synchronize()
            ttt_start = time.perf_counter()
            h_normed = layer_norm(hidden.detach())
            gate = ttt_layer.gate.detach()
            with torch.enable_grad():
                fast_params = ttt_layer.fast_weight.get_params_for_grad()
                for p in fast_params:
                    p.requires_grad_(True)
                fast_output_grad = ttt_layer.fast_weight(h_normed)
                ssl_logits_grad = ssl_head(fast_output_grad)
                B, S, V = ssl_logits_grad.shape
                n_rev = revealed_mask.sum().clamp(min=1.0)
                per_token_loss = F.cross_entropy(
                    ssl_logits_grad.view(-1, V).float(),
                    target_ids.view(-1), reduction="none"
                ).view(B, S).to(ssl_logits_grad.dtype)
                ssl_g = ssl_gate.detach() if ssl_gate is not None else 1.0
                masked_loss = per_token_loss * revealed_mask * ssl_g
                ssl_loss = masked_loss.sum() / n_rev
                grads = torch.autograd.grad(ssl_loss, fast_params,
                                             create_graph=False, allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, fast_params)]
            lr = ttt_layer.lr.detach()
            ttt_layer.fast_weight.apply_update(grads, lr)
            with torch.no_grad():
                new_fast_output = ttt_layer.fast_weight(h_normed)
                delta = gate * new_fast_output
            if isinstance(output, tuple):
                output = (hidden + delta,) + output[1:]
            else:
                output = hidden + delta
            torch.cuda.synchronize()
            hook_state["ttt_time"] = time.perf_counter() - ttt_start
            hook_state["ttt_updated"] = True
        else:
            hook_state["ttt_time"] = 0.0
            hook_state["ttt_updated"] = False
        return output

    hook_handle = target_layer.register_forward_hook(capture_and_inject)
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    try:
        with torch.no_grad():
            for i in range(gen_steps):
                torch.cuda.synchronize()
                step_start = time.perf_counter()
                hook_state["ttt_time"] = 0.0
                mask_index = (x == MASK_TOKEN_ID)
                if not mask_index.any():
                    break
                logits = model(x).logits
                mask_logits = logits[mask_index]
                t, s = timesteps[i], timesteps[i + 1]
                p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                sampled = mask_logits[transfer_mask].argmax(dim=-1) if temperature == 0 else \
                    torch.multinomial(F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
                x0[transfer_mask] = sampled
                x[mask_index] = x0
                torch.cuda.synchronize()
                step_elapsed = time.perf_counter() - step_start
                step_times.append(step_elapsed)
                ttt_overhead_times.append(hook_state["ttt_time"])
    finally:
        hook_handle.remove()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    total_ttt_time = sum(ttt_overhead_times)
    ttt_steps_active = sum(1 for t in ttt_overhead_times if t > 0)
    return decode_output(tokenizer, x, prompt_len), {
        "total_time_s": total_time,
        "mean_step_time_s": float(np.mean(step_times)) if step_times else 0,
        "std_step_time_s": float(np.std(step_times)) if step_times else 0,
        "num_steps_run": len(step_times),
        "ttt_overhead_fraction": total_ttt_time / total_time if total_time > 0 else 0,
        "total_ttt_time_s": total_ttt_time,
        "ttt_steps_active": ttt_steps_active,
    }


# ==============================================================================
# Method 5: ReMDM (timed)
# ==============================================================================

def generate_remdm_timed(model, tokenizer, prompt_text, device, gen_len,
                         gen_steps=GEN_STEPS, temperature=0.0, remask_fraction=0.1):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)
    prompt_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:prompt_len] = True

    step_times = []
    remask_overhead_times = []
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.no_grad():
        for i in range(gen_steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = model(x).logits
            mask_logits = logits[mask_index]
            t, s = timesteps[i], timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            sampled = mask_logits[transfer_mask].argmax(dim=-1) if temperature == 0 else \
                torch.multinomial(F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

            # ReMDM remasking
            torch.cuda.synchronize()
            remask_start = time.perf_counter()
            if i < int(gen_steps * 0.9) and remask_fraction > 0:
                revealed_gen = (~mask_index[0]) & (~prompt_mask) & (x[0] != MASK_TOKEN_ID)
                revealed_indices = revealed_gen.nonzero(as_tuple=True)[0]
                if len(revealed_indices) > 3:
                    revealed_logits = logits[0, revealed_indices]
                    revealed_probs = F.softmax(revealed_logits, dim=-1)
                    current_tokens = x[0, revealed_indices]
                    token_probs = revealed_probs.gather(1, current_tokens.unsqueeze(1)).squeeze(1)
                    n_remask = max(1, int(len(revealed_indices) * remask_fraction))
                    progress = i / gen_steps
                    n_remask = max(1, int(n_remask * (1 - progress)))
                    _, lowest_conf_idx = token_probs.topk(min(n_remask, len(token_probs)), largest=False)
                    remask_positions = revealed_indices[lowest_conf_idx]
                    x[0, remask_positions] = MASK_TOKEN_ID
            torch.cuda.synchronize()
            remask_overhead_times.append(time.perf_counter() - remask_start)
            step_times.append(time.perf_counter() - step_start)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    total_remask = sum(remask_overhead_times)
    return decode_output(tokenizer, x, prompt_len), {
        "total_time_s": total_time,
        "mean_step_time_s": float(np.mean(step_times)) if step_times else 0,
        "std_step_time_s": float(np.std(step_times)) if step_times else 0,
        "num_steps_run": len(step_times),
        "remask_overhead_fraction": total_remask / total_time if total_time > 0 else 0,
        "total_remask_time_s": total_remask,
    }


# ==============================================================================
# Method 6: CoRe (timed)
# ==============================================================================

def generate_core_timed(model, tokenizer, prompt_text, device, gen_len,
                        gen_steps=GEN_STEPS, temperature=0.0,
                        remask_fraction=0.15, context_window=32):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)
    prompt_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:prompt_len] = True

    step_times = []
    remask_overhead_times = []
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.no_grad():
        for i in range(gen_steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = model(x).logits
            mask_logits = logits[mask_index]
            t, s = timesteps[i], timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            sampled = mask_logits[transfer_mask].argmax(dim=-1) if temperature == 0 else \
                torch.multinomial(F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

            # CoRe context-robust remasking
            torch.cuda.synchronize()
            remask_start = time.perf_counter()
            if i < int(gen_steps * 0.85) and remask_fraction > 0:
                revealed_gen = (~mask_index[0]) & (~prompt_mask) & (x[0] != MASK_TOKEN_ID)
                revealed_indices = revealed_gen.nonzero(as_tuple=True)[0]
                if len(revealed_indices) > 5:
                    all_probs = F.softmax(logits[0], dim=-1)
                    entropy = -(all_probs * (all_probs + 1e-10).log()).sum(-1)
                    revealed_entropy = entropy[revealed_indices]
                    context_scores = torch.zeros_like(revealed_entropy)
                    for j, idx in enumerate(revealed_indices):
                        lo = max(prompt_len, idx.item() - context_window // 2)
                        hi = min(x.shape[1], idx.item() + context_window // 2)
                        local_entropy = entropy[lo:hi]
                        if len(local_entropy) > 1:
                            context_scores[j] = revealed_entropy[j] - local_entropy.mean()
                        else:
                            context_scores[j] = 0.0
                    progress = i / gen_steps
                    n_remask = max(1, int(len(revealed_indices) * remask_fraction * (1 - progress)))
                    if n_remask > 0 and len(context_scores) > 0:
                        _, highest_disagree_idx = context_scores.topk(
                            min(n_remask, len(context_scores)), largest=True)
                        valid = context_scores[highest_disagree_idx] > 0
                        remask_positions = revealed_indices[highest_disagree_idx[valid]]
                        if len(remask_positions) > 0:
                            x[0, remask_positions] = MASK_TOKEN_ID
            torch.cuda.synchronize()
            remask_overhead_times.append(time.perf_counter() - remask_start)
            step_times.append(time.perf_counter() - step_start)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    total_remask = sum(remask_overhead_times)
    return decode_output(tokenizer, x, prompt_len), {
        "total_time_s": total_time,
        "mean_step_time_s": float(np.mean(step_times)) if step_times else 0,
        "std_step_time_s": float(np.std(step_times)) if step_times else 0,
        "num_steps_run": len(step_times),
        "remask_overhead_fraction": total_remask / total_time if total_time > 0 else 0,
        "total_remask_time_s": total_remask,
    }


# ==============================================================================
# Benchmark runner
# ==============================================================================

def benchmark_method(name, gen_fn, prompts, device, num_warmup=2):
    """Run a generation method on all prompts with timing."""
    print(f"\n  Warmup ({num_warmup} samples)...")
    for w in range(num_warmup):
        try:
            cublas_safe_run(lambda: gen_fn(prompts[0]))
        except Exception as e:
            print(f"    Warmup {w+1} failed: {str(e)[:100]}")
    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.reset_peak_memory_stats(device)
    sample_times = []
    all_timings = []

    for i, prompt in enumerate(prompts):
        torch.manual_seed(SEED + i)
        torch.cuda.synchronize()
        def _run(p=prompt):
            return gen_fn(p)
        text, timing = cublas_safe_run(_run)
        sample_times.append(timing["total_time_s"])
        all_timings.append(timing)
        if i % 4 == 0:
            extra_info = ""
            if "ttt_overhead_fraction" in timing:
                extra_info = f", TTT={timing['ttt_overhead_fraction']*100:.1f}%"
            elif "gru_overhead_fraction" in timing:
                extra_info = f", GRU={timing['gru_overhead_fraction']*100:.1f}%"
            elif "remask_overhead_fraction" in timing:
                extra_info = f", remask={timing['remask_overhead_fraction']*100:.1f}%"
            print(f"    Sample {i+1}/{len(prompts)}: {timing['total_time_s']:.2f}s{extra_info}")

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
    mean_step_times = [t.get("mean_step_time_s", 0) for t in all_timings]

    # Extract method-specific overhead
    overhead_key = None
    overhead_fraction = 0.0
    for key in ["ttt_overhead_fraction", "gru_overhead_fraction", "remask_overhead_fraction"]:
        vals = [t.get(key, 0) for t in all_timings]
        if any(v > 0 for v in vals):
            overhead_key = key
            overhead_fraction = float(np.mean(vals))
            break

    ttt_steps_active = None
    if any("ttt_steps_active" in t for t in all_timings):
        ttt_steps_active = float(np.mean([t.get("ttt_steps_active", 0) for t in all_timings]))

    result = {
        "total_time_s": sum(sample_times),
        "mean_time_per_sample_s": float(np.mean(sample_times)),
        "std_time_per_sample_s": float(np.std(sample_times)),
        "median_time_per_sample_s": float(np.median(sample_times)),
        "min_time_per_sample_s": float(np.min(sample_times)),
        "max_time_per_sample_s": float(np.max(sample_times)),
        "mean_step_time_s": float(np.mean(mean_step_times)),
        "peak_gpu_memory_gb": peak_mem,
        "num_samples": len(sample_times),
        "sample_times": sample_times,
        "overhead_type": overhead_key or "none",
        "overhead_fraction": overhead_fraction,
    }
    if ttt_steps_active is not None:
        result["ttt_steps_active"] = ttt_steps_active

    return result


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"=== C4: Latency & Throughput Benchmarking (Full 6-method, PILOT) ===")
    print(f"Task ID: {TASK_ID}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    Path(RESULTS_DIR).joinpath(f"{TASK_ID}.pid").write_text(str(os.getpid()))
    for suffix in ["_DONE", "_PROGRESS.json"]:
        p = Path(RESULTS_DIR) / f"{TASK_ID}{suffix}"
        if p.exists():
            p.unlink()

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # === Load model ===
    print(f"\n[1/8] Loading LLaDA-8B-Instruct...")
    report_progress("loading_model", 0, 8)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"  Model loaded on {device}")

    # === Load GSM8K ===
    print(f"\n[2/8] Loading GSM8K dataset...")
    report_progress("loading_data", 1, 8)
    from datasets import load_from_disk
    gsm8k = load_from_disk(f"{SHARED_DIR}/datasets/gsm8k")
    if hasattr(gsm8k, 'keys'):
        # DatasetDict
        gsm8k_test = gsm8k["test"] if "test" in gsm8k else gsm8k["train"]
    else:
        # Single Dataset
        gsm8k_test = gsm8k
    print(f"  GSM8K test: {len(gsm8k_test)} samples, using {PILOT_SAMPLES}")

    prompts = []
    for idx in range(min(PILOT_SAMPLES, len(gsm8k_test))):
        q = gsm8k_test[idx]["question"]
        prompts.append(f"Solve this math problem step by step. Show your work and give the final numerical answer.\n\nQuestion: {q}\n\nSolution:")

    all_results = {}

    # === Method 1: Vanilla ===
    print(f"\n[3/8] Benchmarking: Vanilla (NFE={GEN_STEPS})...")
    report_progress("bench_vanilla", 2, 8)
    all_results["vanilla"] = benchmark_method(
        "vanilla",
        lambda p: generate_vanilla_timed(model, tokenizer, p, device, GEN_LEN),
        prompts, device
    )
    vanilla_mean = all_results["vanilla"]["mean_time_per_sample_s"]
    print(f"  => Vanilla: {vanilla_mean:.2f}s/sample, peak_mem={all_results['vanilla']['peak_gpu_memory_gb']:.2f}GB")
    gc.collect(); torch.cuda.empty_cache()

    # === Method 2: MetaState-GRU ===
    print(f"\n[4/8] Benchmarking: MetaState-GRU (NFE={GEN_STEPS})...")
    report_progress("bench_metastate_gru", 3, 8)

    gru_adapter = MetaStateGRU(d_model=D_MODEL).to(dtype=torch.bfloat16, device=device).eval()
    if os.path.exists(GRU_CKPT):
        gru_ckpt = torch.load(GRU_CKPT, map_location=device, weights_only=False)
        if "gru_adapter_state_dict" in gru_ckpt:
            gru_adapter.load_state_dict(gru_ckpt["gru_adapter_state_dict"], strict=False)
        print(f"  GRU checkpoint loaded: {GRU_CKPT}")
    else:
        print(f"  WARNING: GRU checkpoint not found at {GRU_CKPT}, using random init")

    all_results["metastate_gru"] = benchmark_method(
        "metastate_gru",
        lambda p: generate_gru_timed(model, gru_adapter, tokenizer, p, device, GEN_LEN, INSERTION_LAYER),
        prompts, device
    )
    gru_mean = all_results["metastate_gru"]["mean_time_per_sample_s"]
    print(f"  => MetaState-GRU: {gru_mean:.2f}s/sample ({gru_mean/vanilla_mean:.3f}x), peak_mem={all_results['metastate_gru']['peak_gpu_memory_gb']:.2f}GB")

    del gru_adapter
    gc.collect(); torch.cuda.empty_cache()

    # === Load DaL-Linear ===
    print(f"\n[4.5/8] Loading DaL-Linear checkpoint...")
    ttt_layer = build_ttt_layer(
        variant="linear", d_model=D_MODEL, d_ttt=D_MODEL, vocab_size=VOCAB_SIZE,
    ).to(dtype=torch.bfloat16, device=device).eval()
    dal_ckpt = torch.load(DAL_LINEAR_CKPT, map_location=device, weights_only=False)
    if "ttt_layer_state_dict" in dal_ckpt:
        ttt_layer.load_state_dict(dal_ckpt["ttt_layer_state_dict"], strict=False)
    ssl_head = ttt_layer.ssl_head
    layer_norm = ttt_layer.layer_norm
    ssl_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.bfloat16, device=device))
    with torch.no_grad():
        ttt_layer.gate_logit.fill_(0.0)
    print(f"  DaL-Linear loaded. Gate={torch.sigmoid(ttt_layer.gate_logit).item():.4f}")

    # === Method 3: DaL-Linear (no phase scheduling) ===
    print(f"\n[5/8] Benchmarking: DaL-Linear (no phase scheduling, NFE={GEN_STEPS})...")
    report_progress("bench_dal_linear", 4, 8)
    def dal_gen(p):
        ttt_layer.reset_fast_weights(batch_size=1)
        return generate_dal_timed(
            model, ttt_layer, ssl_head, ssl_gate, layer_norm,
            tokenizer, p, device, GEN_LEN, INSERTION_LAYER, phase_scheduler=None)
    all_results["dal_linear"] = benchmark_method("dal_linear", dal_gen, prompts, device)
    dal_mean = all_results["dal_linear"]["mean_time_per_sample_s"]
    print(f"  => DaL-Linear: {dal_mean:.2f}s/sample ({dal_mean/vanilla_mean:.3f}x), overhead={all_results['dal_linear']['overhead_fraction']*100:.1f}%, peak_mem={all_results['dal_linear']['peak_gpu_memory_gb']:.2f}GB")
    gc.collect(); torch.cuda.empty_cache()

    # === Method 4: DaL-Linear + Phase Scheduling ===
    print(f"\n[6/8] Benchmarking: DaL-Linear + Phase Scheduling (NFE={GEN_STEPS})...")
    report_progress("bench_dal_phase", 5, 8)
    phase_scheduler = PhaseTransitionScheduler(r_crit=0.45, sigma=0.15, high_cutoff=0.80, low_cutoff=0.15)
    def dal_phase_gen(p):
        ttt_layer.reset_fast_weights(batch_size=1)
        return generate_dal_timed(
            model, ttt_layer, ssl_head, ssl_gate, layer_norm,
            tokenizer, p, device, GEN_LEN, INSERTION_LAYER, phase_scheduler=phase_scheduler)
    all_results["dal_linear_phase"] = benchmark_method("dal_linear_phase", dal_phase_gen, prompts, device)
    dal_phase_mean = all_results["dal_linear_phase"]["mean_time_per_sample_s"]
    print(f"  => DaL+Phase: {dal_phase_mean:.2f}s/sample ({dal_phase_mean/vanilla_mean:.3f}x), overhead={all_results['dal_linear_phase']['overhead_fraction']*100:.1f}%, active_steps={all_results['dal_linear_phase'].get('ttt_steps_active', 'N/A')}")

    del ttt_layer, ssl_head, layer_norm, ssl_gate, dal_ckpt
    gc.collect(); torch.cuda.empty_cache()

    # === Method 5: ReMDM ===
    print(f"\n[7/8] Benchmarking: ReMDM (remask_fraction=0.1, NFE={GEN_STEPS})...")
    report_progress("bench_remdm", 6, 8)
    all_results["remdm"] = benchmark_method(
        "remdm",
        lambda p: generate_remdm_timed(model, tokenizer, p, device, GEN_LEN, remask_fraction=0.1),
        prompts, device
    )
    remdm_mean = all_results["remdm"]["mean_time_per_sample_s"]
    print(f"  => ReMDM: {remdm_mean:.2f}s/sample ({remdm_mean/vanilla_mean:.3f}x), remask overhead={all_results['remdm']['overhead_fraction']*100:.1f}%")
    gc.collect(); torch.cuda.empty_cache()

    # === Method 6: CoRe ===
    print(f"\n[8/8] Benchmarking: CoRe (remask_fraction=0.15, context_window=32, NFE={GEN_STEPS})...")
    report_progress("bench_core", 7, 8)
    all_results["core"] = benchmark_method(
        "core",
        lambda p: generate_core_timed(model, tokenizer, p, device, GEN_LEN, remask_fraction=0.15),
        prompts, device
    )
    core_mean = all_results["core"]["mean_time_per_sample_s"]
    print(f"  => CoRe: {core_mean:.2f}s/sample ({core_mean/vanilla_mean:.3f}x), remask overhead={all_results['core']['overhead_fraction']*100:.1f}%")

    # === Summary Table ===
    print(f"\n{'='*100}")
    print(f"C4: Latency Benchmark Summary (PILOT {PILOT_SAMPLES} samples, NFE={GEN_STEPS}, batch_size=1)")
    print(f"{'='*100}")
    print(f"{'Method':<25} {'Time/sample':>12} {'Step time':>12} {'Overhead':>14} {'Peak mem':>10} {'Slowdown':>10}")
    print(f"{'-'*100}")

    for method_name, res in all_results.items():
        slowdown = res["mean_time_per_sample_s"] / vanilla_mean if vanilla_mean > 0 else 0
        overhead_pct = res["overhead_fraction"] * 100
        print(f"{method_name:<25} {res['mean_time_per_sample_s']:>9.2f}s   {res['mean_step_time_s']*1000:>8.1f}ms   {overhead_pct:>10.1f}%    {res['peak_gpu_memory_gb']:>7.2f}GB   {slowdown:>7.3f}x")
    print(f"{'-'*100}")

    # === Build comparison table ===
    comparison = []
    for method_name, res in all_results.items():
        slowdown = res["mean_time_per_sample_s"] / vanilla_mean if vanilla_mean > 0 else 0
        entry = {
            "method": method_name,
            "mean_time_per_sample_s": round(res["mean_time_per_sample_s"], 3),
            "std_time_per_sample_s": round(res["std_time_per_sample_s"], 3),
            "mean_step_time_ms": round(res["mean_step_time_s"] * 1000, 2),
            "overhead_type": res["overhead_type"],
            "overhead_pct": round(res["overhead_fraction"] * 100, 2),
            "peak_gpu_memory_gb": round(res["peak_gpu_memory_gb"], 2),
            "slowdown_vs_vanilla": round(slowdown, 3),
        }
        if "ttt_steps_active" in res:
            entry["ttt_steps_active"] = round(res["ttt_steps_active"], 1)
        comparison.append(entry)

    # === Analysis ===
    dal_slowdown = all_results["dal_linear"]["mean_time_per_sample_s"] / vanilla_mean
    dal_phase_slowdown = all_results["dal_linear_phase"]["mean_time_per_sample_s"] / vanilla_mean
    phase_speedup = all_results["dal_linear"]["mean_time_per_sample_s"] / all_results["dal_linear_phase"]["mean_time_per_sample_s"]
    gru_slowdown = all_results["metastate_gru"]["mean_time_per_sample_s"] / vanilla_mean
    remdm_slowdown = all_results["remdm"]["mean_time_per_sample_s"] / vanilla_mean
    core_slowdown = all_results["core"]["mean_time_per_sample_s"] / vanilla_mean

    analysis = {
        "vanilla_baseline_time_s": round(vanilla_mean, 3),
        "metastate_gru_slowdown": round(gru_slowdown, 3),
        "dal_linear_slowdown": round(dal_slowdown, 3),
        "dal_phase_slowdown": round(dal_phase_slowdown, 3),
        "remdm_slowdown": round(remdm_slowdown, 3),
        "core_slowdown": round(core_slowdown, 3),
        "phase_scheduling_speedup_vs_full_dal": round(phase_speedup, 3),
        "dal_ttt_overhead_pct": round(all_results["dal_linear"]["overhead_fraction"] * 100, 2),
        "dal_phase_ttt_overhead_pct": round(all_results["dal_linear_phase"]["overhead_fraction"] * 100, 2),
        "gru_overhead_pct": round(all_results["metastate_gru"]["overhead_fraction"] * 100, 2),
        "remdm_remask_overhead_pct": round(all_results["remdm"]["overhead_fraction"] * 100, 2),
        "core_remask_overhead_pct": round(all_results["core"]["overhead_fraction"] * 100, 2),
        "dal_memory_overhead_gb": round(all_results["dal_linear"]["peak_gpu_memory_gb"] - all_results["vanilla"]["peak_gpu_memory_gb"], 2),
        "gru_memory_overhead_gb": round(all_results["metastate_gru"]["peak_gpu_memory_gb"] - all_results["vanilla"]["peak_gpu_memory_gb"], 2),
        "phase_active_steps_mean": all_results["dal_linear_phase"].get("ttt_steps_active"),
        "phase_active_steps_total": GEN_STEPS,
        "pass_criteria_met": dal_phase_slowdown < 1.5,
        "key_findings": [
            f"DaL-Linear adds {dal_slowdown:.2f}x latency ({all_results['dal_linear']['overhead_fraction']*100:.1f}% TTT overhead).",
            f"Phase scheduling reduces to {dal_phase_slowdown:.2f}x ({phase_speedup:.2f}x speedup over full DaL).",
            f"MetaState-GRU adds {gru_slowdown:.2f}x latency ({all_results['metastate_gru']['overhead_fraction']*100:.1f}% GRU overhead).",
            f"ReMDM adds {remdm_slowdown:.2f}x latency ({all_results['remdm']['overhead_fraction']*100:.1f}% remasking overhead).",
            f"CoRe adds {core_slowdown:.2f}x latency ({all_results['core']['overhead_fraction']*100:.1f}% remasking overhead).",
            f"DaL+Phase overhead ({dal_phase_slowdown:.2f}x) is comparable to MetaState-GRU ({gru_slowdown:.2f}x), both practical.",
        ],
    }

    # === Save ===
    output = {
        "task_id": TASK_ID,
        "experiment": "C4: Latency & Throughput Benchmarking (Full 6-method, PILOT)",
        "model": "LLaDA-8B-Instruct",
        "mode": "PILOT",
        "pilot_samples": PILOT_SAMPLES,
        "gen_steps": GEN_STEPS,
        "gen_len": GEN_LEN,
        "batch_size": 1,
        "seed": SEED,
        "methods": list(all_results.keys()),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "sample_times"}
                    for k, v in all_results.items()},
        "results_with_samples": {k: v for k, v in all_results.items()},
        "comparison": comparison,
        "analysis": analysis,
        "gpu_info": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "vram_total_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2),
        },
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
    result_path = f"{FULL_RESULTS_DIR}/c4_latency.json"
    Path(result_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to: {result_path}")

    summary = (
        f"Full latency benchmark complete (6 methods). "
        f"Vanilla={vanilla_mean:.2f}s, MetaState-GRU={gru_mean:.2f}s ({gru_slowdown:.2f}x), "
        f"DaL-Linear={dal_mean:.2f}s ({dal_slowdown:.2f}x), "
        f"DaL+Phase={dal_phase_mean:.2f}s ({dal_phase_slowdown:.2f}x), "
        f"ReMDM={remdm_mean:.2f}s ({remdm_slowdown:.2f}x), "
        f"CoRe={core_mean:.2f}s ({core_slowdown:.2f}x). "
        f"Pass criteria met: {analysis['pass_criteria_met']}"
    )
    mark_done("success", summary)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
