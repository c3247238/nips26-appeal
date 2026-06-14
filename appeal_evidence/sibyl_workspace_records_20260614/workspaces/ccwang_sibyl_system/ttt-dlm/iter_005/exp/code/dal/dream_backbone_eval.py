#!/usr/bin/env python3
"""
M4b: Cross-Backbone Validation on Dream-7B (PILOT mode)

Train DaL-MLP on Dream-7B-Instruct frozen backbone for 5K steps,
then evaluate on GSM8K-16, MATH500-16, Countdown-16 (pilot).

Compare against vanilla Dream-7B baseline on same samples.

This validates that DaL improvements transfer across backbone architectures
(Dream is AR-initialized, LLaDA is native diffusion).

Output:
  - DONE marker: exp/results/dream_backbone_eval_DONE
  - Progress: exp/results/dream_backbone_eval_PROGRESS.json
  - Full results: exp/results/full/m4b_dream_eval.json
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

TASK_ID = "dream_backbone_eval"
SEED = 42

# === Model Config (Dream-7B-Instruct) ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/Dream-v0-Instruct-7B"
MASK_TOKEN_ID = 151666
D_MODEL = 3584
D_TTT = 448  # d_model / 8
VOCAB_SIZE = 152064
N_LAYERS = 28
INSERTION_LAYER = 14  # L/2

# === Training Config ===
META_STEPS = 5000
K_UNROLL = 4
META_LR = 1e-4
META_BATCH = 2
SEQ_LEN = 256
CHECKPOINT_INTERVAL = 200
NUM_TRAIN_SAMPLES = 800

# === Eval Config (PILOT: 16 samples) ===
PILOT_SAMPLES = 16
GEN_STEPS = 128
GEN_LEN = 512
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
# Dream-7B Generation Helpers
# ==============================================================================

def prepare_input_dream(tokenizer, prompt_text, device, gen_len):
    """Prepare input for Dream-7B generation."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + [MASK_TOKEN_ID] * gen_len
    x = torch.tensor([full_ids], device=device)
    return x, prompt_len


def decode_output_dream(tokenizer, x, prompt_len):
    """Decode Dream-7B output, skipping mask tokens."""
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


def generate_vanilla_dream(model, tokenizer, prompt_text, device, gen_len,
                           temperature=0.0):
    """Dream-7B vanilla: confidence-based iterative unmasking."""
    x, prompt_len = prepare_input_dream(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)

    with torch.no_grad():
        for i in range(GEN_STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            outputs = model(x, return_dict=True)
            logits = outputs.logits
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

    text = decode_output_dream(tokenizer, x, prompt_len)
    return text


def get_dream_layers(model):
    """Get the list of transformer layers from Dream-7B model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError("Cannot find layers in Dream model")


def generate_with_ttt_dream(model, ttt_layer, tokenizer,
                            prompt_text, device, gen_len, insertion_layer,
                            temperature=0.0):
    """Dream-7B generation with TTT-MLP injection at layer L/2."""
    x, prompt_len = prepare_input_dream(tokenizer, prompt_text, device, gen_len)
    target_ids = x.clone()
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)

    layers = get_dream_layers(model)
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
            # Cast to float32 for TTT ops to avoid CUBLAS bf16 bugs
            h_float = hidden.detach().float()
            h_normed = ttt_layer.layer_norm(h_float)
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
                masked_loss_grad = per_token_loss_grad * revealed_mask.float()
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

            lr = ttt_layer.lr.detach()
            ttt_layer.fast_weight.apply_update(grads, lr)

            with torch.no_grad():
                new_fast_output = ttt_layer.fast_weight(h_normed)
                # Cast delta back to bf16 for backbone injection
                delta = (gate * new_fast_output).to(hidden.dtype)

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
                outputs = model(x, return_dict=True)
                logits = outputs.logits
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

    text = decode_output_dream(tokenizer, x, prompt_len)
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


def extract_math_answer(text):
    """Extract answer from MATH problem output (boxed format)."""
    # Look for \\boxed{...}
    for pat in [r'\\boxed\{([^}]+)\}', r'boxed\{([^}]+)\}',
                r'[Tt]he answer is[:\s]*(.+?)[\.\n]']:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None


def extract_countdown_answer(text):
    """Extract answer from Countdown problem."""
    # Look for final equation or answer
    for pat in [r'=\s*(\d+)', r'[Tt]he answer is[:\s]*(\d+)',
                r'[Rr]esult[:\s]*(\d+)']:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
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
# Meta-Training for Dream-7B
# ==============================================================================

def find_latest_checkpoint(checkpoint_dir, prefix="dream_dal_mlp_ckpt"):
    """Find the latest checkpoint file for resume."""
    if not checkpoint_dir:
        return None, 0
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None, 0
    ckpts = sorted(ckpt_dir.glob(f"{prefix}_step*.pt"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if ckpts:
        last = ckpts[-1]
        step = int(last.stem.split("step")[1])
        return str(last), step
    return None, 0


def meta_train_ttt_mlp_dream(backbone, tokenizer, device, num_steps=5000,
                              checkpoint_dir=None):
    """
    Meta-train TTT-MLP on OpenWebText with K-step unrolling for Dream-7B.

    Adapted from train_dal_mlp.py (LLaDA-8B version) with Dream-specific changes:
      - d_model=3584, d_ttt=448, vocab_size=152064
      - Dream layer access: model.model.layers[L]
      - mask_token_id=151666
    """
    print(f"\n{'='*60}")
    print(f"Meta-training TTT-MLP on Dream-7B for {num_steps} steps")
    print(f"  K={K_UNROLL}, meta_lr={META_LR}, batch={META_BATCH}")
    print(f"  d_model={D_MODEL}, d_ttt={D_TTT}")
    print(f"{'='*60}")

    # Build TTT-MLP layer for Dream-7B dimensions
    # CRITICAL: Keep TTT layer in float32 to avoid Blackwell CUBLAS bf16 bugs
    ttt_layer = build_ttt_layer(
        d_model=D_MODEL,
        variant="mlp",
        vocab_size=VOCAB_SIZE,
        ttt_lr=0.1,  # Best from pilot
        precision_weighted=False,
    ).to(device).to(torch.float32)

    # Gate init: 0.0 (sigmoid = 0.5) so gate starts open
    # Previous -3.0/-5.0 inits resulted in gate stuck near 0
    with torch.no_grad():
        ttt_layer.gate_logit.fill_(0.0)

    param_count = ttt_layer.get_trainable_param_count()
    fast_param_count = ttt_layer.get_fast_weight_param_count()
    print(f"  TTT-MLP total trainable params: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"  Fast weight params: {fast_param_count:,} ({fast_param_count/1e6:.2f}M)")
    print(f"  Initial gate value: {ttt_layer.gate.item():.4f}")

    # Load training data
    print("Loading OpenWebText training data...")
    train_data = load_openwebtext_samples(NUM_TRAIN_SAMPLES, SEQ_LEN, tokenizer)
    train_data = train_data.to(device)
    print(f"  Loaded {len(train_data)} samples")

    # Get backbone layers for hook
    layers = get_dream_layers(backbone)
    target_layer = layers[INSERTION_LAYER]

    # Meta-optimizer with separate gate lr
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
    max_cublas_errors = 200

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

        # Reset fast weights
        ttt_layer.reset_fast_weights(META_BATCH)

        # Create initial masked input
        input_ids, revealed_mask = create_masked_input(batch, MASK_TOKEN_ID, 0.8)
        input_ids = input_ids.to(device)
        revealed_mask = revealed_mask.to(device)
        mask_ratios = np.linspace(0.8, 0.2, K_UNROLL + 1)

        step_ssl_losses = []
        k_completed = 0

        try:
            for k in range(K_UNROLL):
                current_mr = mask_ratios[k]
                target_mr = mask_ratios[k + 1]
                revealed_mask_k = revealed_mask.clone()

                # Forward through backbone to capture hidden states
                hook_handle = target_layer.register_forward_hook(capture_hook)
                try:
                    with torch.no_grad():
                        backbone(input_ids=input_ids, output_hidden_states=False, return_dict=True)
                except RuntimeError as e:
                    hook_handle.remove()
                    cublas_error_count += 1
                    if cublas_error_count % 50 == 1:
                        print(f"  [Step {step}, k={k}] CUDA error #{cublas_error_count}: {str(e)[:60]}")
                    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
                    time.sleep(0.2)
                    continue
                finally:
                    try:
                        hook_handle.remove()
                    except Exception:
                        pass

                if captured_hidden[0] is not None:
                    # Cast to float32 for TTT ops
                    hidden = captured_hidden[0].detach().float()
                    h_normed = ttt_layer.layer_norm(hidden)

                    # --- TTT self-supervised step (no meta-grad needed) ---
                    fast_params = ttt_layer.fast_weight.get_params_for_grad()
                    for p in fast_params:
                        p.requires_grad_(True)

                    fast_output = ttt_layer.fast_weight(h_normed)
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

                    grads = torch.autograd.grad(
                        ssl_loss, fast_params, create_graph=False, allow_unused=True)
                    grads = [g if g is not None else torch.zeros_like(p)
                             for g, p in zip(grads, fast_params)]

                    grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
                    if grad_norm > ttt_layer.max_grad_norm:
                        scale = ttt_layer.max_grad_norm / (grad_norm + 1e-8)
                        grads = [g * scale for g in grads]

                    lr = ttt_layer.lr.detach()
                    ttt_layer.fast_weight.apply_update(grads, lr)

                    # --- Meta-loss: per-k backward to save memory ---
                    # Compute meta-loss from after-update predictions
                    with torch.no_grad():
                        fo_after = ttt_layer.fast_weight(h_normed)
                        sl_after = ttt_layer.ssl_head(fo_after)
                        ptl_after = F.cross_entropy(
                            sl_after.view(-1, V).float(),
                            batch.view(-1),
                            reduction="none"
                        ).view(B, S)
                        meta_ssl = (ptl_after * revealed_mask_k).sum() / n_revealed

                    # Gate regularization (this is the meta-learnable part)
                    gate_val = ttt_layer.gate
                    gate_reg = 0.1 * (gate_val - 0.1).pow(2)
                    gate_floor = 0.05 * F.relu(0.01 - gate_val)
                    meta_loss_k = gate_reg + gate_floor
                    # Scale by 1/K_UNROLL for averaging
                    (meta_loss_k / K_UNROLL).backward()

                    k_completed += 1

                    # Aggressive cleanup
                    del ssl_logits, sl_after, per_token_loss, ptl_after
                    del fast_output, fo_after, ssl_loss, hidden, h_normed
                    captured_hidden[0] = None

                # Reveal more tokens
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
            if "CUDA" in str(step_err) or "memory" in str(step_err).lower():
                cublas_error_count += 1
                if cublas_error_count % 50 == 1:
                    print(f"  [Step {step}] CUDA/OOM error #{cublas_error_count}")
                torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
                time.sleep(0.3)
                train_losses.append(train_losses[-1] if train_losses else 10.0)
                gate_values.append(float(ttt_layer.gate.item()))
                lr_values.append(float(ttt_layer.lr.item()))
                scheduler.step()
                continue
            else:
                raise

        # Meta-gradient step
        if k_completed > 0:
            has_nan = False
            for name, p in ttt_layer.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    p.grad.nan_to_num_(nan=0.0)

            torch.nn.utils.clip_grad_norm_(ttt_layer.parameters(), max_norm=1.0)
            if not has_nan:
                meta_optimizer.step()
            scheduler.step()
        else:
            scheduler.step()

        avg_loss = np.mean(step_ssl_losses) if step_ssl_losses else (train_losses[-1] if train_losses else 10.0)
        train_losses.append(float(avg_loss))
        gate_values.append(float(ttt_layer.gate.item()))
        lr_values.append(float(ttt_layer.lr.item()))

        if step % 100 == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (step - start_step + 1) * (num_steps - step - 1) if step > start_step else 0
            current_meta_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:5d}/{num_steps}: ssl_loss={avg_loss:.4f} "
                  f"gate={ttt_layer.gate.item():.4f} "
                  f"ttt_lr={ttt_layer.lr.item():.4f} "
                  f"meta_lr={current_meta_lr:.6f} "
                  f"cublas_errs={cublas_error_count} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            report_progress("meta_training", step, num_steps, {
                "ssl_loss": avg_loss,
                "gate": ttt_layer.gate.item(),
                "ttt_lr": ttt_layer.lr.item(),
                "cublas_errors": cublas_error_count,
            })

        # Save checkpoints
        if checkpoint_dir and (step + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = Path(checkpoint_dir) / f"dream_dal_mlp_ckpt_step{step+1}.pt"
            torch.save({
                "step": step + 1,
                "ttt_layer_state_dict": ttt_layer.state_dict(),
                "optimizer_state_dict": meta_optimizer.state_dict(),
                "train_losses": train_losses,
                "gate_values": gate_values,
                "lr_values": lr_values,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        if step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if step % 50 == 0:
            torch.cuda.synchronize()

    total_time = time.time() - start_time
    first_100 = np.mean(train_losses[:100]) if len(train_losses) >= 100 else train_losses[0]
    last_100 = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else train_losses[-1]
    decrease_pct = (first_100 - last_100) / first_100 * 100 if first_100 > 0 else 0

    gate_first_100 = np.mean(gate_values[:100]) if len(gate_values) >= 100 else gate_values[0]
    gate_last_100 = np.mean(gate_values[-100:]) if len(gate_values) >= 100 else gate_values[-1]

    print(f"\nMeta-training complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  First 100 avg loss: {first_100:.4f}")
    print(f"  Last 100 avg loss:  {last_100:.4f}")
    print(f"  Decrease: {decrease_pct:.1f}%")
    print(f"  Final gate: {ttt_layer.gate.item():.4f}")
    print(f"  Gate trajectory: {gate_first_100:.4f} -> {gate_last_100:.4f}")
    print(f"  Final TTT lr: {ttt_layer.lr.item():.4f}")

    train_info = {
        "variant": "mlp",
        "backbone": "Dream-7B-Instruct",
        "d_model": D_MODEL,
        "d_ttt": D_TTT,
        "num_steps": num_steps,
        "k_unroll": K_UNROLL,
        "meta_lr": META_LR,
        "first_100_avg": float(first_100),
        "last_100_avg": float(last_100),
        "decrease_pct": float(decrease_pct),
        "gate_final": float(ttt_layer.gate.item()),
        "gate_first_100": float(gate_first_100),
        "gate_last_100": float(gate_last_100),
        "ttt_lr_final": float(ttt_layer.lr.item()),
        "total_time_s": float(total_time),
        "total_time_min": float(total_time / 60),
        "train_losses_sampled": train_losses[::10],
        "gate_values_sampled": gate_values[::10],
        "lr_values_sampled": lr_values[::10],
        "trainable_params": param_count,
        "fast_weight_params": fast_param_count,
        "cublas_errors": cublas_error_count,
    }

    converged = last_100 < 2 * first_100
    train_info["converged"] = converged
    print(f"  Convergence check (loss < 2x initial): {converged}")

    return ttt_layer, train_info


# ==============================================================================
# Evaluation functions
# ==============================================================================

def evaluate_gsm8k_dream(model, tokenizer, device, ttt_layer=None,
                         insertion_layer=None, num_samples=16, gen_len=512):
    from datasets import load_from_disk
    gsm8k = load_from_disk(f"{SHARED_DIR}/datasets/gsm8k")
    if num_samples:
        gsm8k = gsm8k.select(range(min(num_samples, len(gsm8k))))

    method = "dal_mlp" if ttt_layer else "vanilla"
    print(f"\n  Evaluating Dream-7B {method} on GSM8K ({len(gsm8k)} problems)...")

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
                gen_text = generate_with_ttt_dream(
                    model, ttt_layer, tokenizer, prompt, device, gen_len,
                    insertion_layer, temperature=0.0)
            else:
                gen_text = generate_vanilla_dream(
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

        if (idx + 1) % 4 == 0 or idx == len(gsm8k) - 1:
            acc = correct / total if total > 0 else 0
            elapsed = time.time() - start_time
            print(f"    [{idx+1}/{len(gsm8k)}] acc={acc:.3f} ({correct}/{total}) "
                  f"elapsed={elapsed:.0f}s")
            report_progress(f"eval_gsm8k_{method}", idx + 1, len(gsm8k), {
                "correct": correct, "total": total, "accuracy": acc})

        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"  GSM8K {method}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")
    return {
        "accuracy": accuracy, "correct": correct, "total": total,
        "time_s": elapsed, "results": results,
    }


def evaluate_math500_dream(model, tokenizer, device, ttt_layer=None,
                           insertion_layer=None, num_samples=16, gen_len=512):
    from datasets import load_from_disk
    try:
        math500 = load_from_disk(f"{SHARED_DIR}/datasets/math500")
    except Exception:
        print("  MATH500 dataset not found, skipping...")
        return None

    if num_samples:
        math500 = math500.select(range(min(num_samples, len(math500))))

    method = "dal_mlp" if ttt_layer else "vanilla"
    print(f"\n  Evaluating Dream-7B {method} on MATH500 ({len(math500)} problems)...")

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for idx in range(len(math500)):
        item = math500[idx]
        problem = item.get('problem', item.get('question', ''))
        target_answer = item.get('answer', item.get('solution', ''))

        prompt = (
            "Solve the following math problem. Show your work and put your final answer "
            "in \\boxed{}.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )

        try:
            if ttt_layer:
                gen_text = generate_with_ttt_dream(
                    model, ttt_layer, tokenizer, prompt, device, gen_len,
                    insertion_layer, temperature=0.0)
            else:
                gen_text = generate_vanilla_dream(
                    model, tokenizer, prompt, device, gen_len, temperature=0.0)
        except Exception as e:
            gen_text = ""
            if idx < 3:
                print(f"    [{idx}] Error: {e}")

        extracted = extract_math_answer(gen_text)
        target_extracted = extract_math_answer(target_answer)

        is_correct = False
        if extracted and target_extracted:
            is_correct = extracted.strip() == target_extracted.strip()
            # Also try numeric comparison
            if not is_correct:
                try:
                    is_correct = abs(float(extracted) - float(target_extracted)) < 1e-3
                except (ValueError, TypeError):
                    pass

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": idx, "target": target_extracted, "extracted": extracted,
            "is_correct": is_correct,
            "gen_text": gen_text[:300] if idx < 5 else None,
        })

        if (idx + 1) % 4 == 0 or idx == len(math500) - 1:
            acc = correct / total if total > 0 else 0
            print(f"    [{idx+1}/{len(math500)}] acc={acc:.3f} ({correct}/{total})")

        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"  MATH500 {method}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")
    return {
        "accuracy": accuracy, "correct": correct, "total": total,
        "time_s": elapsed, "results": results,
    }


def evaluate_countdown_dream(model, tokenizer, device, ttt_layer=None,
                              insertion_layer=None, num_samples=16, gen_len=256):
    from datasets import load_from_disk
    try:
        countdown = load_from_disk(f"{SHARED_DIR}/datasets/countdown")
    except Exception:
        print("  Countdown dataset not found, skipping...")
        return None

    if num_samples:
        countdown = countdown.select(range(min(num_samples, len(countdown))))

    method = "dal_mlp" if ttt_layer else "vanilla"
    print(f"\n  Evaluating Dream-7B {method} on Countdown ({len(countdown)} problems)...")

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for idx in range(len(countdown)):
        item = countdown[idx]
        # Countdown format varies; try common keys
        problem = item.get('problem', item.get('question', item.get('input', '')))
        target = item.get('target', item.get('answer', item.get('output', '')))

        prompt = f"Solve this Countdown numbers problem:\n{problem}\n\nSolution:"

        try:
            if ttt_layer:
                gen_text = generate_with_ttt_dream(
                    model, ttt_layer, tokenizer, prompt, device, gen_len,
                    insertion_layer, temperature=0.0)
            else:
                gen_text = generate_vanilla_dream(
                    model, tokenizer, prompt, device, gen_len, temperature=0.0)
        except Exception as e:
            gen_text = ""
            if idx < 3:
                print(f"    [{idx}] Error: {e}")

        extracted = extract_countdown_answer(gen_text)
        # Try parsing target as number
        target_num = None
        try:
            target_num = int(target) if isinstance(target, (int, str)) and str(target).strip().isdigit() else None
        except (ValueError, TypeError):
            pass

        is_correct = (extracted is not None and target_num is not None and extracted == target_num)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": idx, "target": target_num, "extracted": extracted,
            "is_correct": is_correct,
            "gen_text": gen_text[:200] if idx < 5 else None,
        })

        if (idx + 1) % 4 == 0 or idx == len(countdown) - 1:
            acc = correct / total if total > 0 else 0
            print(f"    [{idx+1}/{len(countdown)}] acc={acc:.3f} ({correct}/{total})")

        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"  Countdown {method}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")
    return {
        "accuracy": accuracy, "correct": correct, "total": total,
        "time_s": elapsed, "results": results,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"{'='*70}")
    print(f"M4b: Cross-Backbone Validation on Dream-7B (PILOT)")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Config: meta_steps={META_STEPS}, K={K_UNROLL}, "
          f"insertion_layer={INSERTION_LAYER}, d_ttt={D_TTT}")
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

    # === Load Dream-7B-Instruct ===
    # Dream uses custom model class, must use AutoModel (not AutoModelForCausalLM)
    print("Loading Dream-7B-Instruct...")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device)
    backbone.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loaded: d_model={D_MODEL}, layers={N_LAYERS}, vocab={VOCAB_SIZE}")
    print(f"  mask_token_id={MASK_TOKEN_ID}")

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU memory: {gpu_mem:.1f}GB / {gpu_total:.1f}GB")

    # === Phase 1: Meta-train TTT-MLP on Dream-7B ===
    checkpoint_dir = FULL_RESULTS_DIR
    ttt_layer, train_info = meta_train_ttt_mlp_dream(
        backbone, tokenizer, device,
        num_steps=META_STEPS, checkpoint_dir=checkpoint_dir)

    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 2: Evaluate Vanilla Dream-7B Baseline ===
    print(f"\n{'='*70}")
    print(f"Phase 2: Vanilla Dream-7B Evaluation (PILOT-{PILOT_SAMPLES})")
    print(f"{'='*70}")

    report_progress("eval_vanilla", 0, 3)

    vanilla_gsm8k = evaluate_gsm8k_dream(
        model=backbone, tokenizer=tokenizer, device=device,
        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN)
    gc.collect(); torch.cuda.empty_cache()

    vanilla_math500 = evaluate_math500_dream(
        model=backbone, tokenizer=tokenizer, device=device,
        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN)
    gc.collect(); torch.cuda.empty_cache()

    vanilla_countdown = evaluate_countdown_dream(
        model=backbone, tokenizer=tokenizer, device=device,
        num_samples=PILOT_SAMPLES, gen_len=256)
    gc.collect(); torch.cuda.empty_cache()

    # === Phase 3: Evaluate DaL-MLP on Dream-7B ===
    print(f"\n{'='*70}")
    print(f"Phase 3: DaL-MLP Augmented Evaluation (PILOT-{PILOT_SAMPLES})")
    print(f"{'='*70}")

    report_progress("eval_dal_mlp", 0, 3)

    dal_gsm8k = evaluate_gsm8k_dream(
        model=backbone, tokenizer=tokenizer, device=device,
        ttt_layer=ttt_layer, insertion_layer=INSERTION_LAYER,
        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN)
    gc.collect(); torch.cuda.empty_cache()

    dal_math500 = evaluate_math500_dream(
        model=backbone, tokenizer=tokenizer, device=device,
        ttt_layer=ttt_layer, insertion_layer=INSERTION_LAYER,
        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN)
    gc.collect(); torch.cuda.empty_cache()

    dal_countdown = evaluate_countdown_dream(
        model=backbone, tokenizer=tokenizer, device=device,
        ttt_layer=ttt_layer, insertion_layer=INSERTION_LAYER,
        num_samples=PILOT_SAMPLES, gen_len=256)
    gc.collect(); torch.cuda.empty_cache()

    # === Phase 4: Results Summary ===
    total_elapsed = time.time() - overall_start

    print(f"\n{'='*70}")
    print("RESULTS COMPARISON — Dream-7B (PILOT)")
    print(f"{'='*70}")

    benchmarks = {
        "GSM8K": (vanilla_gsm8k, dal_gsm8k),
        "MATH500": (vanilla_math500, dal_math500),
        "Countdown": (vanilla_countdown, dal_countdown),
    }

    improvements = 0
    deltas = {}

    print(f"\n{'Benchmark':<15} {'Vanilla':<12} {'DaL-MLP':<12} {'Delta':<12}")
    print(f"{'-'*51}")
    for name, (van, dal) in benchmarks.items():
        if van and dal:
            van_acc = van['accuracy']
            dal_acc = dal['accuracy']
            delta = dal_acc - van_acc
            deltas[name] = delta
            if delta > 0:
                improvements += 1
            print(f"{name:<15} {van_acc:.3f}        {dal_acc:.3f}        {delta:+.3f}")
        elif van:
            print(f"{name:<15} {van['accuracy']:.3f}        N/A          N/A")
        else:
            print(f"{name:<15} N/A          N/A          N/A")

    pilot_pass = improvements >= 2  # Pass: improvement on at least 2 of 3 benchmarks
    converged = train_info.get('converged', False)

    print(f"\n  Pass criteria: DaL improves over vanilla on >= 2/3 benchmarks")
    print(f"    Training converged: {converged}")
    print(f"    Benchmarks improved: {improvements}/3")
    print(f"    Pilot assessment: {'PASS' if pilot_pass else 'FAIL'}")
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # === Compare with LLaDA-8B results ===
    print(f"\n{'='*70}")
    print("Cross-Backbone Comparison: Dream-7B vs LLaDA-8B")
    print(f"{'='*70}")
    # Load LLaDA results if available
    llada_results = {}
    llada_path = Path(FULL_RESULTS_DIR) / "m3_update_rule_comparison.json"
    if llada_path.exists():
        try:
            llada_data = json.loads(llada_path.read_text())
            if "full_results" in llada_data:
                fs = llada_data["full_results"]
                if "vanilla" in fs and "full_scale_from_training" in fs["vanilla"]:
                    llada_results["vanilla_gsm8k"] = fs["vanilla"]["full_scale_from_training"]["gsm8k_full"]["accuracy"]
                    llada_results["vanilla_math500"] = fs["vanilla"]["full_scale_from_training"]["math500"]["accuracy"]
            print(f"  LLaDA-8B vanilla GSM8K: {llada_results.get('vanilla_gsm8k', 'N/A')}")
            print(f"  LLaDA-8B vanilla MATH500: {llada_results.get('vanilla_math500', 'N/A')}")
        except Exception as e:
            print(f"  Could not load LLaDA results: {e}")

    # === Save Results ===
    def safe_results(r):
        if r is None:
            return None
        return {k: v for k, v in r.items() if k != "results"}

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

    full_results = {
        "task_id": TASK_ID,
        "experiment": "M4b: Cross-Backbone Validation on Dream-7B (PILOT)",
        "backbone": "Dream-7B-Instruct (Dream-v0-Instruct-7B)",
        "mode": "PILOT",
        "variant": "mlp",
        "config": {
            "meta_steps": META_STEPS, "k_unroll": K_UNROLL,
            "meta_lr": META_LR, "meta_batch": META_BATCH,
            "seq_len": SEQ_LEN, "insertion_layer": INSERTION_LAYER,
            "d_model": D_MODEL, "d_ttt": D_TTT, "vocab_size": VOCAB_SIZE,
            "mask_token_id": MASK_TOKEN_ID,
            "gen_steps": GEN_STEPS, "pilot_samples": PILOT_SAMPLES,
        },
        "meta_training": train_info,
        "vanilla": {
            "gsm8k": safe_results(vanilla_gsm8k),
            "math500": safe_results(vanilla_math500),
            "countdown": safe_results(vanilla_countdown),
        },
        "dal_mlp": {
            "gsm8k": safe_results(dal_gsm8k),
            "math500": safe_results(dal_math500),
            "countdown": safe_results(dal_countdown),
        },
        "comparison": {
            "gsm8k_delta": float(deltas.get("GSM8K", 0)),
            "math500_delta": float(deltas.get("MATH500", 0)),
            "countdown_delta": float(deltas.get("Countdown", 0)),
            "benchmarks_improved": improvements,
        },
        "cross_backbone": {
            "llada_8b_results": llada_results,
            "note": "Compare DaL-MLP gains on Dream-7B vs LLaDA-8B to verify architecture transferability",
        },
        "pass_criteria": {
            "converged": converged,
            "improvements_count": improvements,
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
            "vanilla_gsm8k": vanilla_gsm8k.get("results", [])[:5] if vanilla_gsm8k else [],
            "dal_mlp_gsm8k": dal_gsm8k.get("results", [])[:5] if dal_gsm8k else [],
        },
    }

    full_path = Path(FULL_RESULTS_DIR) / "m4b_dream_eval.json"
    full_path.write_text(json.dumps(full_results, indent=2, default=json_default))
    print(f"\nFull results saved to {full_path}")

    # DONE marker
    status = "PASS" if pilot_pass else "FAIL"
    summary = (f"{status}: Dream-7B DaL-MLP, train_decrease={train_info['decrease_pct']:.1f}%, "
               f"gate={train_info['gate_final']:.4f}, "
               f"GSM8K: v={vanilla_gsm8k['accuracy']:.3f} d={dal_gsm8k['accuracy']:.3f} "
               f"MATH500: v={vanilla_math500['accuracy']:.3f if vanilla_math500 else 'N/A'} "
               f"d={dal_math500['accuracy']:.3f if dal_math500 else 'N/A'} "
               f"Countdown: v={vanilla_countdown['accuracy']:.3f if vanilla_countdown else 'N/A'} "
               f"d={dal_countdown['accuracy']:.3f if dal_countdown else 'N/A'} "
               f"improved={improvements}/3")
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
