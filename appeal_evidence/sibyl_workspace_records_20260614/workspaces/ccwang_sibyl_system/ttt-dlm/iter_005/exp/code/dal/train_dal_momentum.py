#!/usr/bin/env python3
"""
M2c: Train DaL-Momentum on LLaDA-8B-Instruct (PILOT mode)

Based on the working train_dal_mlp.py with momentum-specific changes:
  1. variant="momentum" — TTT-MLP + Titans-style momentum (beta=0.9) + weight decay
  2. Weight decay lambda coupled to remasking ratio (DLM-specific forgetting)
  3. Momentum buffer persistence across K-step unrolling
  4. Gate init at -3.0 (same as MLP)

Output:
  - DONE marker: exp/results/train_dal_momentum_DONE
  - Progress: exp/results/train_dal_momentum_PROGRESS.json
  - Full results: exp/results/full/m2c_dal_momentum.json
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

def safe_backbone_forward(backbone, input_ids, max_retries=5):
    """Wrapper for backbone forward that handles transient CUBLAS errors on Blackwell."""
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                return backbone(input_ids=input_ids, output_hidden_states=False)
        except RuntimeError as e:
            if ("CUBLAS" in str(e) or "CUDA" in str(e)) and attempt < max_retries - 1:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1.0 + attempt * 0.5)
                continue
            raise

# === Paths ===
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
FULL_RESULTS_DIR = f"{PROJECT_DIR}/exp/results/full"
CODE_DIR = f"{PROJECT_DIR}/exp/code/dal"
SHARED_DIR = f"{REMOTE_BASE}/shared"

TASK_ID = "train_dal_momentum"
SEED = 42

# === Model Config (LLaDA-8B-Instruct) ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
D_MODEL = 4096
D_TTT = 512  # d_model / 8
VOCAB_SIZE = 126464
N_LAYERS = 32
INSERTION_LAYER = 16  # L/2

# === Training Config ===
META_STEPS = 5000
K_UNROLL = 4
META_LR = 1e-4
META_BATCH = 2
SEQ_LEN = 256
CHECKPOINT_INTERVAL = 100
NUM_TRAIN_SAMPLES = 200  # Reduced from 800 to avoid Blackwell CUBLAS issues with large tensors on GPU

# === Momentum Config (Titans-style) ===
MOMENTUM_BETA = 0.9
WEIGHT_DECAY_BASE = 0.01  # Base weight decay, coupled to mask ratio

# === Eval Config (PILOT: 16 samples) ===
PILOT_SAMPLES = 16
GEN_STEPS = 128
GEN_LEN_GSM8K = 512
TEMPERATURE = 0.0  # greedy

sys.path.insert(0, CODE_DIR)
from ttt_layer import TTTLayer, build_ttt_layer
from dal_wrapper import create_masked_input


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


# ==============================================================================
# LLaDA Generation Helpers
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


def generate_vanilla(model, tokenizer, prompt_text, device, gen_len, temperature=0.0):
    """LLaDA vanilla: confidence-based iterative unmasking."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)

    with torch.no_grad():
        for i in range(GEN_STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = model(x).logits
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < GEN_STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature == 0:
                sampled = mask_logits[transfer_mask].argmax(dim=-1)
            else:
                probs = F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

    text = decode_output(tokenizer, x, prompt_len)
    return text


def generate_with_ttt(model, ttt_layer, tokenizer,
                      prompt_text, device, gen_len, insertion_layer,
                      temperature=0.0):
    """LLaDA generation with TTT-Momentum injection at layer L/2."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    target_ids = x.clone()
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)

    # Get backbone layers
    inner_model = model.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        layers = tf.blocks
    elif hasattr(tf, 'block_groups'):
        layers = tf.block_groups
    else:
        raise ValueError("Cannot find blocks in LLaDA transformer")

    target_layer = layers[insertion_layer]
    ttt_layer.reset_fast_weights(batch_size=1)

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()

        if n_revealed > 5 and n_masked > 3:
            h_normed = ttt_layer.layer_norm(hidden.detach())
            gate = ttt_layer.gate.detach()

            with torch.enable_grad():
                fast_params = ttt_layer.fast_weight.get_params_for_grad()
                for p in fast_params:
                    p.requires_grad_(True)

                fast_output_grad = ttt_layer.fast_weight(h_normed)
                ssl_logits_grad = ttt_layer.ssl_head(fast_output_grad)
                B, S, V = ssl_logits_grad.shape
                n_rev = revealed_mask.sum().clamp(min=1.0)
                per_token_loss_grad = F.cross_entropy(
                    ssl_logits_grad.view(-1, V).float(),
                    target_ids.view(-1),
                    reduction="none"
                ).view(B, S)
                masked_loss_grad = per_token_loss_grad * revealed_mask
                ssl_loss = masked_loss_grad.sum() / n_rev

                grads = torch.autograd.grad(ssl_loss, fast_params,
                                             create_graph=False, allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, fast_params)]

            # Gradient clipping
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
            if grad_norm > ttt_layer.max_grad_norm:
                scale = ttt_layer.max_grad_norm / (grad_norm + 1e-8)
                grads = [g * scale for g in grads]

            # Momentum update: m = beta * m + grad; W = (1-lambda)*W - lr*m
            lr = ttt_layer.lr.detach()
            if ttt_layer._momentum_bufs is not None:
                for i, (g, m) in enumerate(zip(grads, ttt_layer._momentum_bufs)):
                    ttt_layer._momentum_bufs[i] = MOMENTUM_BETA * m.detach() + g.detach()
                # Compute current mask ratio for coupled weight decay
                total_positions = float(S)
                current_mask_ratio = n_masked / total_positions if total_positions > 0 else 0.5
                wd = WEIGHT_DECAY_BASE * current_mask_ratio  # Coupled to mask ratio

                new_W1 = ttt_layer.fast_weight.W1_fast.detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[0]
                new_b1 = ttt_layer.fast_weight.b1_fast.detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[1]
                new_W2 = ttt_layer.fast_weight.W2_fast.detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[2]
                new_b2 = ttt_layer.fast_weight.b2_fast.detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[3]
                ttt_layer.fast_weight.W1_fast = new_W1
                ttt_layer.fast_weight.b1_fast = new_b1
                ttt_layer.fast_weight.W2_fast = new_W2
                ttt_layer.fast_weight.b2_fast = new_b2
            else:
                # Fallback: standard SGD
                ttt_layer.fast_weight.apply_update(grads, lr)

            with torch.no_grad():
                new_fast_output = ttt_layer.fast_weight(h_normed)
                delta = gate * new_fast_output

            if isinstance(output, tuple):
                output = (hidden + delta,) + output[1:]
            else:
                output = hidden + delta

        return output

    hook_handle = target_layer.register_forward_hook(capture_and_inject)

    try:
        with torch.no_grad():
            for i in range(GEN_STEPS):
                mask_index = (x == MASK_TOKEN_ID)
                if not mask_index.any():
                    break
                logits = model(x).logits
                mask_logits = logits[mask_index]
                t = timesteps[i]
                s = timesteps[i + 1]
                p_transfer = (1 - s / t).item() if i < GEN_STEPS - 1 else 1.0
                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                if temperature == 0:
                    sampled = mask_logits[transfer_mask].argmax(dim=-1)
                else:
                    probs = F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                x0[transfer_mask] = sampled
                x[mask_index] = x0
                target_ids = x.clone()
    finally:
        hook_handle.remove()

    text = decode_output(tokenizer, x, prompt_len)
    return text


# ==============================================================================
# Answer Extraction
# ==============================================================================

def extract_gsm8k_answer(text):
    for pat in [r'####\s*(-?[\d,]+\.?\d*)', r'[Tt]he answer is[:\s]*(-?[\d,]+\.?\d*)']:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except:
                pass
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if nums:
        try:
            return float(nums[-1].replace(',', ''))
        except:
            pass
    return None


def extract_gsm8k_target(ans):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', ans)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except:
            pass
    return None


# ==============================================================================
# Data Loading
# ==============================================================================

def load_openwebtext_samples(num_samples, seq_len, tokenizer):
    from datasets import load_from_disk
    dataset_path = f"{SHARED_DIR}/datasets/openwebtext_10k"
    ds = load_from_disk(dataset_path)
    all_ids = []
    for item in ds:
        if len(all_ids) >= num_samples:
            break
        text = item.get("text", item.get("content", ""))
        if not text or len(text) < 100:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=seq_len, padding="max_length")
        input_ids = tokens["input_ids"].squeeze(0)
        non_pad = (input_ids != tokenizer.pad_token_id).sum().item()
        if non_pad >= seq_len // 2:
            all_ids.append(input_ids)
    return torch.stack(all_ids[:num_samples])


# ==============================================================================
# Meta-Training with Momentum
# ==============================================================================

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file for resume."""
    if not checkpoint_dir:
        return None, 0
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None, 0
    ckpts = sorted(ckpt_dir.glob("dal_momentum_ckpt_step*.pt"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if ckpts:
        last = ckpts[-1]
        step = int(last.stem.split("step")[1])
        return str(last), step
    return None, 0


def meta_train_ttt_momentum(backbone, tokenizer, device, num_steps=5000,
                             checkpoint_dir=None):
    """
    Meta-train TTT-Momentum on OpenWebText with K-step unrolling.

    Key differences from TTT-MLP:
      1. variant="momentum" — MLP fast weight with momentum buffer
      2. Momentum update: m = beta*m + grad; W = (1-lambda)*W - lr*m
      3. Weight decay lambda coupled to current mask ratio (DLM-specific)
      4. Momentum buffers persist across K-step unrolling within a meta-step
    """
    print(f"\n{'='*60}")
    print(f"Meta-training TTT-Momentum for {num_steps} steps")
    print(f"  K={K_UNROLL}, meta_lr={META_LR}, batch={META_BATCH}")
    print(f"  d_model={D_MODEL}, d_ttt={D_TTT}")
    print(f"  momentum_beta={MOMENTUM_BETA}, weight_decay_base={WEIGHT_DECAY_BASE}")
    print(f"{'='*60}")

    # Build TTT-Momentum layer
    ttt_layer = build_ttt_layer(
        d_model=D_MODEL,
        variant="momentum",
        vocab_size=VOCAB_SIZE,
        ttt_lr=0.1,  # Best from pilot (P1)
        precision_weighted=False,
        momentum_beta=MOMENTUM_BETA,
        weight_decay=WEIGHT_DECAY_BASE,
    ).to(device).to(torch.bfloat16)

    # Same gate init as MLP: -3.0 (sigmoid ~ 0.047)
    with torch.no_grad():
        ttt_layer.gate_logit.fill_(-3.0)

    param_count = ttt_layer.get_trainable_param_count()
    fast_param_count = ttt_layer.get_fast_weight_param_count()
    print(f"  TTT-Momentum total trainable params: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"  Fast weight params: {fast_param_count:,} ({fast_param_count/1e6:.2f}M)")
    print(f"  Initial gate value: {ttt_layer.gate.item():.4f}")

    # Load training data
    print("Loading OpenWebText training data...")
    train_data = load_openwebtext_samples(NUM_TRAIN_SAMPLES, SEQ_LEN, tokenizer)
    train_data = train_data.to(device)
    print(f"  Loaded {len(train_data)} samples")

    # Get backbone layers for hook
    inner_model = backbone.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        layers = tf.blocks
    elif hasattr(tf, 'block_groups'):
        layers = tf.block_groups
    target_layer = layers[INSERTION_LAYER]

    # Meta-optimizer with separate gate lr (same as MLP)
    gate_params = [ttt_layer.gate_logit]
    other_params = [p for n, p in ttt_layer.named_parameters()
                    if p.requires_grad and 'gate_logit' not in n]

    meta_optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": META_LR, "weight_decay": 0.01},
        {"params": gate_params, "lr": META_LR * 10, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer, T_max=num_steps, eta_min=META_LR * 0.1)

    train_losses = []
    gate_values = []
    lr_values = []
    momentum_norms = []  # Track momentum buffer norms
    start_step = 0
    captured_hidden = [None]

    # Try to resume from checkpoint
    ckpt_path, ckpt_step = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path and ckpt_step > 0:
        print(f"  Resuming from checkpoint: {ckpt_path} (step {ckpt_step})")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ttt_layer.load_state_dict(ckpt["ttt_layer_state_dict"], strict=False)
        try:
            meta_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except:
            print("  WARNING: Could not load optimizer state, using fresh optimizer")
        train_losses = ckpt.get("train_losses", [])
        gate_values = ckpt.get("gate_values", [])
        lr_values = ckpt.get("lr_values", [])
        momentum_norms = ckpt.get("momentum_norms", [])
        start_step = ckpt_step
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resumed: gate={ttt_layer.gate.item():.4f}, "
              f"ttt_lr={ttt_layer.lr.item():.4f}, "
              f"ssl_loss={train_losses[-1] if train_losses else 'N/A'}")
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    start_time = time.time()

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured_hidden[0] = output[0].clone()
        else:
            captured_hidden[0] = output.clone()
        return output

    cublas_error_count = 0
    max_cublas_errors = 5000  # Blackwell GPU 2 has frequent transient CUBLAS errors

    for step in range(start_step, num_steps):
        torch.manual_seed(SEED + step * 100)
        meta_optimizer.zero_grad()

        # Select batch
        batch_start = (step * META_BATCH) % len(train_data)
        batch_end = batch_start + META_BATCH
        if batch_end > len(train_data):
            batch = torch.cat([
                train_data[batch_start:],
                train_data[:batch_end - len(train_data)]
            ], dim=0)
        else:
            batch = train_data[batch_start:batch_end]

        # Reset fast weights AND momentum buffers
        ttt_layer.reset_fast_weights(META_BATCH)

        # Create initial masked input
        input_ids, revealed_mask = create_masked_input(batch, MASK_TOKEN_ID, 0.8)
        input_ids = input_ids.to(device)
        revealed_mask = revealed_mask.to(device)
        mask_ratios = np.linspace(0.8, 0.2, K_UNROLL + 1)

        step_ssl_losses = []
        meta_losses = []
        step_had_cublas_error = False

        try:
            for k in range(K_UNROLL):
                current_mr = mask_ratios[k]
                target_mr = mask_ratios[k + 1]
                revealed_mask_k = revealed_mask.clone()

                # Forward through backbone to capture hidden states
                hook_handle = target_layer.register_forward_hook(capture_hook)
                backbone_ok = True
                try:
                    safe_backbone_forward(backbone, input_ids, max_retries=5)
                except RuntimeError as cublas_err:
                    cublas_error_count += 1
                    step_had_cublas_error = True
                    if cublas_error_count % 50 == 1:
                        print(f"  [Step {step}, k={k}] CUBLAS error #{cublas_error_count}: {str(cublas_err)[:80]}")
                    backbone_ok = False
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                finally:
                    try:
                        hook_handle.remove()
                    except Exception:
                        pass

                if not backbone_ok:
                    continue

                if captured_hidden[0] is not None:
                    hidden = captured_hidden[0].detach()

                    # Layer norm
                    h_normed = ttt_layer.layer_norm(hidden)

                    # Compute fast weight output
                    fast_params = ttt_layer.fast_weight.get_params_for_grad()
                    for p in fast_params:
                        p.requires_grad_(True)

                    fast_output = ttt_layer.fast_weight(h_normed)

                    # SSL loss
                    ssl_logits = ttt_layer.ssl_head(fast_output)
                    B, S, V = ssl_logits.shape
                    per_token_loss = F.cross_entropy(
                        ssl_logits.view(-1, V).float(),
                        batch.view(-1),
                        reduction="none"
                    ).view(B, S)
                    masked_loss = per_token_loss * revealed_mask_k
                    n_revealed = revealed_mask_k.sum().clamp(min=1.0)
                    ssl_loss = masked_loss.sum() / n_revealed

                    step_ssl_losses.append(ssl_loss.item())

                    # Gradients w.r.t. fast weights
                    grads = torch.autograd.grad(
                        ssl_loss, fast_params, create_graph=False, allow_unused=True)
                    grads = [g if g is not None else torch.zeros_like(p)
                             for g, p in zip(grads, fast_params)]

                    # Gradient clipping
                    grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
                    if grad_norm > ttt_layer.max_grad_norm:
                        scale = ttt_layer.max_grad_norm / (grad_norm + 1e-8)
                        grads = [g * scale for g in grads]

                    # === MOMENTUM UPDATE (key difference from MLP) ===
                    lr = ttt_layer.lr.detach()

                    # Compute weight decay coupled to current mask ratio
                    wd = WEIGHT_DECAY_BASE * current_mr  # Higher decay at higher mask ratio

                    # Update momentum buffers: m = beta * m + grad
                    if ttt_layer._momentum_bufs is not None:
                        for i, (g, m) in enumerate(zip(grads, ttt_layer._momentum_bufs)):
                            ttt_layer._momentum_bufs[i] = MOMENTUM_BETA * m.detach() + g.detach()

                        # Apply momentum update with weight decay: W = (1-wd)*W - lr*m
                        new_W1 = fast_params[0].detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[0]
                        new_b1 = fast_params[1].detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[1]
                        new_W2 = fast_params[2].detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[2]
                        new_b2 = fast_params[3].detach() * (1.0 - wd) - lr * ttt_layer._momentum_bufs[3]
                        ttt_layer.fast_weight.W1_fast = new_W1
                        ttt_layer.fast_weight.b1_fast = new_b1
                        ttt_layer.fast_weight.W2_fast = new_W2
                        ttt_layer.fast_weight.b2_fast = new_b2
                    else:
                        # Fallback if momentum buffers not initialized
                        ttt_layer.fast_weight.apply_update(grads, lr)

                    # Meta-loss: evaluate updated fast weights
                    fast_output_after = ttt_layer.fast_weight(h_normed)
                    ssl_logits_after = ttt_layer.ssl_head(fast_output_after.detach())
                    per_token_loss_after = F.cross_entropy(
                        ssl_logits_after.view(-1, V).float(),
                        batch.view(-1),
                        reduction="none"
                    ).view(B, S)
                    masked_loss_after = per_token_loss_after * revealed_mask_k
                    meta_loss = masked_loss_after.sum() / n_revealed

                    # Gate regularization (same as MLP)
                    gate_val = ttt_layer.gate
                    gate_reg = 0.1 * (gate_val - 0.1).pow(2)
                    gate_floor = 0.05 * F.relu(0.01 - gate_val)
                    meta_loss = meta_loss + gate_reg + gate_floor

                    meta_losses.append(meta_loss)

                    del ssl_logits, ssl_logits_after, per_token_loss, per_token_loss_after
                    del fast_output, fast_output_after, ssl_loss

                # Reveal more tokens for next unrolling step
                n_to_reveal = int((current_mr - target_mr) * SEQ_LEN)
                n_to_reveal = max(1, n_to_reveal)
                masked_positions = (input_ids == MASK_TOKEN_ID)
                for b in range(META_BATCH):
                    masked_pos = masked_positions[b].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        n = min(n_to_reveal, len(masked_pos))
                        perm = torch.randperm(len(masked_pos), device=device)[:n]
                        positions = masked_pos[perm]
                        input_ids[b, positions] = batch[b, positions]
                        revealed_mask[b, positions] = 1.0

        except RuntimeError as step_err:
            if "CUBLAS" in str(step_err) or "CUDA" in str(step_err):
                cublas_error_count += 1
                step_had_cublas_error = True
                if cublas_error_count % 10 == 1:
                    print(f"  [Step {step}] CUBLAS error #{cublas_error_count} in step body, skipping step")
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(0.5)
                train_losses.append(train_losses[-1] if train_losses else 10.0)
                gate_values.append(float(ttt_layer.gate.item()))
                lr_values.append(float(ttt_layer.lr.item()))
                momentum_norms.append(0.0)
                scheduler.step()
                if cublas_error_count >= max_cublas_errors:
                    print(f"  Too many CUBLAS errors ({cublas_error_count}), saving checkpoint and exiting")
                    if checkpoint_dir:
                        ckpt_path = Path(checkpoint_dir) / f"dal_momentum_ckpt_step{step+1}.pt"
                        torch.save({
                            "step": step + 1,
                            "ttt_layer_state_dict": ttt_layer.state_dict(),
                            "optimizer_state_dict": meta_optimizer.state_dict(),
                            "train_losses": train_losses,
                            "gate_values": gate_values,
                            "lr_values": lr_values,
                            "momentum_norms": momentum_norms,
                        }, ckpt_path)
                    sys.exit(42)
                continue
            else:
                raise

        # Meta-gradient step
        if meta_losses:
            total_meta_loss = sum(meta_losses) / len(meta_losses)
            total_meta_loss.backward()

            has_nan = False
            for name, p in ttt_layer.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    p.grad.nan_to_num_(nan=0.0)

            torch.nn.utils.clip_grad_norm_(ttt_layer.parameters(), max_norm=1.0)
            if not has_nan:
                meta_optimizer.step()
            scheduler.step()

            del total_meta_loss

        avg_loss = np.mean(step_ssl_losses) if step_ssl_losses else (train_losses[-1] if train_losses else 10.0)
        train_losses.append(float(avg_loss))
        gate_values.append(float(ttt_layer.gate.item()))
        lr_values.append(float(ttt_layer.lr.item()))

        # Track momentum buffer norm
        mom_norm = 0.0
        if ttt_layer._momentum_bufs is not None:
            mom_norm = sum(m.norm().item() for m in ttt_layer._momentum_bufs)
        momentum_norms.append(float(mom_norm))

        if step % 100 == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (step - start_step + 1) * (num_steps - step - 1) if step > start_step else 0
            current_meta_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:5d}/{num_steps}: ssl_loss={avg_loss:.4f} "
                  f"gate={ttt_layer.gate.item():.4f} "
                  f"ttt_lr={ttt_layer.lr.item():.4f} "
                  f"mom_norm={mom_norm:.2f} "
                  f"meta_lr={current_meta_lr:.6f} "
                  f"cublas_errs={cublas_error_count} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            report_progress("meta_training", step, num_steps, {
                "ssl_loss": avg_loss,
                "gate": ttt_layer.gate.item(),
                "ttt_lr": ttt_layer.lr.item(),
                "momentum_norm": mom_norm,
                "cublas_errors": cublas_error_count,
            })

        # Save checkpoints
        if checkpoint_dir and (step + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = Path(checkpoint_dir) / f"dal_momentum_ckpt_step{step+1}.pt"
            torch.save({
                "step": step + 1,
                "ttt_layer_state_dict": ttt_layer.state_dict(),
                "optimizer_state_dict": meta_optimizer.state_dict(),
                "train_losses": train_losses,
                "gate_values": gate_values,
                "lr_values": lr_values,
                "momentum_norms": momentum_norms,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Memory cleanup every 20 steps
        if step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Periodic CUDA sync
        if step % 50 == 0:
            torch.cuda.synchronize()

    total_time = time.time() - start_time
    first_100 = np.mean(train_losses[:100]) if len(train_losses) >= 100 else train_losses[0]
    last_100 = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else train_losses[-1]
    decrease_pct = (first_100 - last_100) / first_100 * 100 if first_100 > 0 else 0

    gate_first_100 = np.mean(gate_values[:100]) if len(gate_values) >= 100 else gate_values[0]
    gate_last_100 = np.mean(gate_values[-100:]) if len(gate_values) >= 100 else gate_values[-1]

    mom_first_100 = np.mean(momentum_norms[:100]) if len(momentum_norms) >= 100 else momentum_norms[0]
    mom_last_100 = np.mean(momentum_norms[-100:]) if len(momentum_norms) >= 100 else momentum_norms[-1]

    print(f"\nMeta-training complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  First 100 avg loss: {first_100:.4f}")
    print(f"  Last 100 avg loss:  {last_100:.4f}")
    print(f"  Decrease: {decrease_pct:.1f}%")
    print(f"  Final gate: {ttt_layer.gate.item():.4f}")
    print(f"  Gate trajectory: {gate_first_100:.4f} -> {gate_last_100:.4f}")
    print(f"  Final TTT lr: {ttt_layer.lr.item():.4f}")
    print(f"  Momentum norm trajectory: {mom_first_100:.4f} -> {mom_last_100:.4f}")

    train_info = {
        "variant": "momentum",
        "d_ttt": D_TTT,
        "num_steps": num_steps,
        "k_unroll": K_UNROLL,
        "meta_lr": META_LR,
        "momentum_beta": MOMENTUM_BETA,
        "weight_decay_base": WEIGHT_DECAY_BASE,
        "first_100_avg": float(first_100),
        "last_100_avg": float(last_100),
        "decrease_pct": float(decrease_pct),
        "gate_final": float(ttt_layer.gate.item()),
        "gate_first_100": float(gate_first_100),
        "gate_last_100": float(gate_last_100),
        "ttt_lr_final": float(ttt_layer.lr.item()),
        "momentum_norm_first_100": float(mom_first_100),
        "momentum_norm_last_100": float(mom_last_100),
        "total_time_s": float(total_time),
        "total_time_min": float(total_time / 60),
        "train_losses_sampled": train_losses[::10],
        "gate_values_sampled": gate_values[::10],
        "lr_values_sampled": lr_values[::10],
        "momentum_norms_sampled": momentum_norms[::10],
        "trainable_params": param_count,
        "fast_weight_params": fast_param_count,
    }

    converged = last_100 < 2 * first_100
    train_info["converged"] = converged
    print(f"  Convergence check (loss < 2x initial): {converged}")

    return ttt_layer, train_info


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_gsm8k(model, tokenizer, device, ttt_layer=None,
                   insertion_layer=None, num_samples=None, gen_len=512):
    """Evaluate on GSM8K (full or subset)."""
    from datasets import load_from_disk
    gsm8k = load_from_disk(f"{SHARED_DIR}/datasets/gsm8k")
    if num_samples:
        gsm8k = gsm8k.select(range(min(num_samples, len(gsm8k))))

    method = "dal_momentum" if ttt_layer else "vanilla"
    print(f"\n  Evaluating {method} on GSM8K ({len(gsm8k)} problems)...")

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for idx in range(len(gsm8k)):
        item = gsm8k[idx]
        question = item['question']
        target = extract_gsm8k_target(item['answer'])
        if target is None:
            continue

        prompt = (
            "Solve the math problem step by step. "
            "End your answer with #### followed by the final numerical answer.\n\n"
            f"Problem: {question}\n\nSolution:"
        )

        try:
            if ttt_layer:
                gen_text = generate_with_ttt(
                    model, ttt_layer, tokenizer, prompt, device, gen_len,
                    insertion_layer, temperature=0.0)
            else:
                gen_text = generate_vanilla(
                    model, tokenizer, prompt, device, gen_len, temperature=0.0)
        except Exception as e:
            gen_text = ""
            if idx < 5:
                print(f"    [{idx}] Error: {e}")
                traceback.print_exc()

        extracted = extract_gsm8k_answer(gen_text)
        is_correct = (extracted is not None and abs(extracted - target) < 1e-3)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": idx, "target": target, "extracted": extracted,
            "is_correct": is_correct,
            "gen_text": gen_text[:300] if idx < 10 else None,
        })

        if (idx + 1) % 8 == 0 or idx == len(gsm8k) - 1:
            acc = correct / total if total > 0 else 0
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (len(gsm8k) - idx - 1) if idx > 0 else 0
            print(f"    [{idx+1}/{len(gsm8k)}] acc={acc:.3f} ({correct}/{total}) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            report_progress(f"eval_gsm8k_{method}", idx + 1, len(gsm8k), {
                "correct": correct, "total": total, "accuracy": acc})

        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"  GSM8K {method}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")
    return {
        "accuracy": accuracy, "correct": correct, "total": total,
        "time_s": elapsed, "results": results,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"{'='*70}")
    print(f"M2c: Train DaL-Momentum on LLaDA-8B-Instruct (PILOT)")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Config: meta_steps={META_STEPS}, K={K_UNROLL}, "
          f"insertion_layer={INSERTION_LAYER}, d_ttt={D_TTT}")
    print(f"Momentum: beta={MOMENTUM_BETA}, weight_decay_base={WEIGHT_DECAY_BASE}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Mode: PILOT (eval on {PILOT_SAMPLES} samples)")
    print()

    overall_start = time.time()

    # Write PID
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")

    # === Load LLaDA-8B-Instruct ===
    print("Loading LLaDA-8B-Instruct...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    backbone.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loaded: d_model={D_MODEL}, layers={N_LAYERS}, vocab={VOCAB_SIZE}")

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # GPU memory info
    gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU memory: {gpu_mem:.1f}GB / {gpu_total:.1f}GB")

    # === Phase 1: Meta-train TTT-Momentum ===
    checkpoint_dir = FULL_RESULTS_DIR
    ttt_layer, train_info = meta_train_ttt_momentum(
        backbone, tokenizer, device,
        num_steps=META_STEPS, checkpoint_dir=checkpoint_dir)

    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 2: Evaluate Vanilla Baseline ===
    print(f"\n{'='*70}")
    print(f"Phase 2: Vanilla LLaDA-8B Evaluation (GSM8K-{PILOT_SAMPLES})")
    print(f"{'='*70}")

    report_progress("eval_vanilla", 0, 1)
    vanilla_gsm8k = evaluate_gsm8k(
        model=backbone, tokenizer=tokenizer, device=device,
        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN_GSM8K)
    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 3: Evaluate DaL-Momentum ===
    print(f"\n{'='*70}")
    print(f"Phase 3: DaL-Momentum Augmented Evaluation (GSM8K-{PILOT_SAMPLES})")
    print(f"{'='*70}")

    report_progress("eval_dal_momentum", 0, 1)
    dal_gsm8k = evaluate_gsm8k(
        model=backbone, tokenizer=tokenizer, device=device,
        ttt_layer=ttt_layer, insertion_layer=INSERTION_LAYER,
        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN_GSM8K)
    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 4: Results ===
    total_elapsed = time.time() - overall_start

    gsm8k_delta = dal_gsm8k['accuracy'] - vanilla_gsm8k['accuracy']

    print(f"\n{'='*70}")
    print("RESULTS COMPARISON (PILOT)")
    print(f"{'='*70}")
    print(f"{'Benchmark':<15} {'Vanilla':<12} {'DaL-Mom':<12} {'Delta':<12}")
    print(f"{'-'*51}")
    print(f"{'GSM8K':<15} {vanilla_gsm8k['accuracy']:.3f}        "
          f"{dal_gsm8k['accuracy']:.3f}        {gsm8k_delta:+.3f}")

    # Compare with DaL-Linear and DaL-MLP if available
    for ref_name, ref_file, ref_key in [
        ("DaL-Linear", "m2a_dal_linear.json", "dal_linear"),
        ("DaL-MLP", "m2b_dal_mlp.json", "dal_mlp"),
    ]:
        ref_path = Path(FULL_RESULTS_DIR) / ref_file
        if ref_path.exists():
            try:
                ref_results = json.loads(ref_path.read_text())
                ref_acc = ref_results.get(ref_key, {}).get("gsm8k", {}).get("accuracy")
                if ref_acc is not None:
                    print(f"{ref_name:<15} {ref_acc:.3f}        (ref from earlier)")
            except:
                pass

    converged = train_info.get('converged', False)
    pilot_pass = converged and (dal_gsm8k['accuracy'] > vanilla_gsm8k['accuracy'])

    print(f"\n  Pass criteria:")
    print(f"    Training converged: {converged}")
    print(f"    GSM8K accuracy vs vanilla: {dal_gsm8k['accuracy']:.3f} vs "
          f"{vanilla_gsm8k['accuracy']:.3f} (delta={gsm8k_delta:+.3f})")
    print(f"    Gate: {train_info.get('gate_first_100', 'N/A')} -> "
          f"{train_info.get('gate_last_100', 'N/A')}")
    print(f"    Momentum: beta={MOMENTUM_BETA}, wd_base={WEIGHT_DECAY_BASE}")
    print(f"    Momentum norm: {train_info.get('momentum_norm_first_100', 'N/A'):.4f} -> "
          f"{train_info.get('momentum_norm_last_100', 'N/A'):.4f}")
    print(f"    Pilot assessment: {'PASS' if pilot_pass else 'FAIL'}")
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # === Save Results ===
    full_results = {
        "task_id": TASK_ID,
        "experiment": "M2c: DaL-Momentum on LLaDA-8B (PILOT)",
        "model": "LLaDA-8B-Instruct",
        "mode": "PILOT",
        "variant": "momentum",
        "config": {
            "meta_steps": META_STEPS, "k_unroll": K_UNROLL,
            "meta_lr": META_LR, "meta_batch": META_BATCH,
            "seq_len": SEQ_LEN, "insertion_layer": INSERTION_LAYER,
            "d_model": D_MODEL, "d_ttt": D_TTT, "vocab_size": VOCAB_SIZE,
            "gen_steps": GEN_STEPS, "pilot_samples": PILOT_SAMPLES,
            "momentum_beta": MOMENTUM_BETA,
            "weight_decay_base": WEIGHT_DECAY_BASE,
        },
        "meta_training": train_info,
        "vanilla": {
            "gsm8k": {k: v for k, v in vanilla_gsm8k.items() if k != "results"},
        },
        "dal_momentum": {
            "gsm8k": {k: v for k, v in dal_gsm8k.items() if k != "results"},
        },
        "comparison": {
            "gsm8k_delta": float(gsm8k_delta),
        },
        "pass_criteria": {
            "converged": converged,
            "pilot_assessment": "PASS" if pilot_pass else "FAIL",
        },
        "timing": {
            "total_elapsed_s": float(total_elapsed),
            "total_elapsed_min": float(total_elapsed / 60),
        },
        "gpu_info": {
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2),
        },
        "timestamp": datetime.now().isoformat(),
        "sample_results": {
            "vanilla": vanilla_gsm8k.get("results", [])[:10],
            "dal_momentum": dal_gsm8k.get("results", [])[:10],
        },
    }

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return str(obj)

    full_path = Path(FULL_RESULTS_DIR) / "m2c_dal_momentum.json"
    full_path.write_text(json.dumps(full_results, indent=2, default=json_default))
    print(f"\nFull results saved to {full_path}")

    # DONE marker
    status = "PASS" if pilot_pass else "FAIL"
    summary = (f"{status}: DaL-Momentum, train_decrease={train_info['decrease_pct']:.1f}%, "
               f"gate={train_info['gate_final']:.4f}, "
               f"ttt_lr={train_info['ttt_lr_final']:.4f}, "
               f"mom_beta={MOMENTUM_BETA}, wd={WEIGHT_DECAY_BASE}, "
               f"GSM8K-{PILOT_SAMPLES}: vanilla={vanilla_gsm8k['accuracy']:.3f} "
               f"dal_momentum={dal_gsm8k['accuracy']:.3f} delta={gsm8k_delta:+.3f}")
    mark_done(status, summary)
    print(f"DONE marker written: {status}")

    return pilot_pass


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except RuntimeError as e:
        if "CUDA" in str(e) or "CUBLAS" in str(e):
            print(f"\nCUDA ERROR (recoverable): {e}")
            traceback.print_exc()
            mark_done("cuda_error", f"CUDA error: {str(e)[:200]}. Restart with checkpoint.")
            sys.exit(42)
        else:
            print(f"\nFATAL ERROR: {e}")
            traceback.print_exc()
            mark_done("error", f"Fatal: {str(e)[:200]}")
            sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        mark_done("error", f"Fatal: {str(e)[:200]}")
        sys.exit(1)
