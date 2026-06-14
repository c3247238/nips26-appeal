#!/usr/bin/env python3
"""
M6: Enhancement Ablation on DaL-Linear (v2 - CUBLAS retry logic)

Ablates each enhancement component starting from DaL-Linear base.
Key fix: gate_logit initialized to 0 (sigmoid=0.5) instead of -5 (sigmoid=0.007).

Configurations:
  (a) DaL-Linear base (no enhancements, gate_init=0)
  (b) +precision weighting (entropy-based)
  (c) +phase-transition scheduling (only update at critical mask ratios)
  (d) +residual gate separation (separate gates for SSL vs output path)
  (e) DaL-Full (all three enhancements)

Each config: 1000-step meta-train + GSM8K-16 pilot eval.

v2 changes:
  - CUBLAS error retry logic for Blackwell GPUs
  - Clean old DONE/PROGRESS markers on start
  - Add MATH500-16 evaluation alongside GSM8K-16
  - Track wall-clock time per config
  - Track logit cosine similarity (backbone vs TTT output) for residual gate analysis
"""

import os, sys, json, time, gc, re, math, traceback, copy
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Blackwell cuBLAS / allocator workarounds need to be set before CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

TASK_ID = "enhancement_ablation"
SEED = 42

# === Model Config ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
D_MODEL = 4096
VOCAB_SIZE = 126464
N_LAYERS = 32
INSERTION_LAYER = 16

# === Training Config (reduced for ablation) ===
META_STEPS = 1000
K_UNROLL = 4
META_LR = 1e-4
META_BATCH = 2
SEQ_LEN = 256
NUM_TRAIN_SAMPLES = 400

# === Eval Config ===
PILOT_SAMPLES = 16
GEN_STEPS = 128
GEN_LEN_GSM8K = 512
GEN_LEN_MATH = 512
MAX_CUBLAS_RETRIES = 3  # Per-sample retry limit for CUBLAS errors

sys.path.insert(0, CODE_DIR)
from dal_wrapper import create_masked_input


# ==============================================================================
# CUBLAS Error Retry Helper
# ==============================================================================

def is_cublas_error(e):
    """Check if an exception is a CUBLAS/CUDA error."""
    msg = str(e)
    return "CUBLAS" in msg or "CUDA error" in msg


def cublas_retry(func, max_retries=MAX_CUBLAS_RETRIES, sleep_sec=2.0):
    """Retry a function on CUBLAS errors with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except RuntimeError as e:
            if is_cublas_error(e) and attempt < max_retries:
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(sleep_sec * (2 ** attempt))
                continue
            raise


# ==============================================================================
# Phase-Transition Scheduler
# ==============================================================================

class PhaseTransitionScheduler:
    def __init__(self, r_crit=0.45, sigma=0.15, high_cutoff=0.80, low_cutoff=0.15):
        self.r_crit = r_crit
        self.sigma = sigma
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff

    def should_update(self, mask_ratio):
        return self.low_cutoff <= mask_ratio <= self.high_cutoff

    def get_weight(self, mask_ratio):
        if not self.should_update(mask_ratio):
            return 0.0
        z = (mask_ratio - self.r_crit) / self.sigma
        return math.exp(-0.5 * z * z)


# ==============================================================================
# Enhanced TTT-Linear Layer with configurable enhancements
# ==============================================================================

class EnhancedTTTLinear(nn.Module):
    """TTT-Linear with individually toggleable enhancements."""

    def __init__(self, d_model, vocab_size=126464, ttt_lr=0.1,
                 gate_init=0.0, precision_weighted=False,
                 phase_transition=False, gate_separation=False,
                 max_grad_norm=10.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.use_precision = precision_weighted
        self.use_phase_transition = phase_transition
        self.use_gate_separation = gate_separation

        self.W_init = nn.Parameter(torch.zeros(d_model, d_model))
        nn.init.xavier_uniform_(self.W_init, gain=1.0)
        self.register_buffer("W_fast", None)

        self.log_lr = nn.Parameter(torch.tensor(math.log(ttt_lr)))
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

        if gate_separation:
            self.ssl_gate_logit = nn.Parameter(torch.tensor(gate_init))

        self.layer_norm = nn.LayerNorm(d_model)

        if phase_transition:
            self.scheduler = PhaseTransitionScheduler()
        else:
            self.scheduler = None

        self._batch_size = None

    @property
    def lr(self):
        return self.log_lr.exp()

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    @property
    def ssl_gate(self):
        if self.use_gate_separation:
            return torch.sigmoid(self.ssl_gate_logit)
        return self.gate

    def reset_fast_weights(self, batch_size=1):
        self._batch_size = batch_size
        self.W_fast = self.W_init.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward_fast(self, h):
        return torch.bmm(h, self.W_fast.transpose(1, 2))

    def compute_precision_weights(self, logits, revealed_mask):
        if not self.use_precision or logits is None:
            return revealed_mask.float()
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        variance = 1.0 - max_probs
        precision = 1.0 / (variance + 1e-6)
        precision = precision * revealed_mask.float()
        num_revealed = revealed_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        weight_sum = precision.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        precision = precision * (num_revealed / weight_sum)
        return precision

    def ttt_step(self, h_normed, target_ids, revealed_mask, precision_weights=None,
                 mask_ratio=None):
        if self.use_phase_transition and mask_ratio is not None:
            if not self.scheduler.should_update(mask_ratio):
                return {"ssl_loss": 0.0, "grad_norm": 0.0, "skipped": True,
                        "lr": self.lr.item(), "gate": self.gate.item()}

        self.W_fast.requires_grad_(True)
        fast_output = self.forward_fast(h_normed)

        B, S, D = fast_output.shape
        fast_norm = F.normalize(fast_output.float(), dim=-1)
        h_norm = F.normalize(h_normed.float().detach(), dim=-1)
        per_token_sim = (fast_norm * h_norm).sum(dim=-1)
        per_token_loss = 1.0 - per_token_sim

        if precision_weights is not None:
            weighted_loss = per_token_loss * precision_weights
        else:
            weighted_loss = per_token_loss * revealed_mask.float()

        num_revealed = revealed_mask.float().sum().clamp(min=1.0)
        lr_scale = 1.0
        if self.use_phase_transition and mask_ratio is not None:
            lr_scale = self.scheduler.get_weight(mask_ratio)

        ssl_loss = weighted_loss.sum() / num_revealed
        grads = torch.autograd.grad(ssl_loss, [self.W_fast], create_graph=False)
        grad = grads[0]
        if grad is None:
            grad = torch.zeros_like(self.W_fast)

        grad_norm = grad.norm().item()
        if grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / (grad_norm + 1e-8))

        lr = self.lr.detach() * lr_scale
        if self.use_gate_separation:
            ssl_gate = self.ssl_gate.detach()
            self.W_fast = (self.W_fast.detach() - lr * ssl_gate * grad.detach())
        else:
            self.W_fast = (self.W_fast.detach() - lr * grad.detach())

        return {"ssl_loss": ssl_loss.item(), "grad_norm": grad_norm,
                "lr": self.lr.item(), "gate": self.gate.item(), "skipped": False}


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
        except:
            pass
    Path(RESULTS_DIR).joinpath(f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat()}))


def clean_old_markers():
    """Remove old DONE/PROGRESS markers from previous failed runs."""
    for suffix in ["_DONE", "_PROGRESS.json", ".pid"]:
        p = Path(RESULTS_DIR) / f"{TASK_ID}{suffix}"
        if p.exists():
            p.unlink()
            print(f"  Cleaned old marker: {p.name}")


# ==============================================================================
# LLaDA Generation
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
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)
    with torch.no_grad():
        for i in range(GEN_STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break

            def _fwd():
                return model(x).logits
            logits = cublas_retry(_fwd)

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
    return decode_output(tokenizer, x, prompt_len)


def generate_with_enhanced_ttt(model, ttt_layer, tokenizer,
                                prompt_text, device, gen_len,
                                insertion_layer, temperature=0.0,
                                collect_similarity=False):
    """Generate with TTT injection. Optionally collect logit cosine similarity."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    target_ids = x.clone()
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)

    inner_model = model.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        layers = tf.blocks
    elif hasattr(tf, 'block_groups'):
        layers = tf.block_groups
    else:
        raise ValueError("Cannot find blocks")

    target_layer = layers[insertion_layer]
    ttt_layer.reset_fast_weights(batch_size=1)

    similarity_log = [] if collect_similarity else None

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()
        total_tokens = x.shape[1]
        current_mask_ratio = n_masked / total_tokens if total_tokens > 0 else 0

        if n_revealed > 5 and n_masked > 3:
            h_normed = ttt_layer.layer_norm(hidden.detach())
            gate = ttt_layer.gate.detach()

            with torch.enable_grad():
                ttt_layer.ttt_step(
                    h_normed, target_ids,
                    revealed_mask.unsqueeze(0) if revealed_mask.dim() == 1 else revealed_mask,
                    mask_ratio=current_mask_ratio)

            with torch.no_grad():
                new_fast_output = ttt_layer.forward_fast(h_normed)
                delta = gate * new_fast_output

                # Collect cosine similarity between backbone hidden and TTT output
                if collect_similarity and similarity_log is not None:
                    backbone_norm = F.normalize(hidden.float(), dim=-1)
                    ttt_norm = F.normalize(new_fast_output.float(), dim=-1)
                    cos_sim = (backbone_norm * ttt_norm).sum(dim=-1).mean().item()
                    similarity_log.append({
                        "step": len(similarity_log),
                        "mask_ratio": current_mask_ratio,
                        "cosine_similarity": cos_sim,
                        "gate": gate.item(),
                    })

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

                def _fwd():
                    return model(x).logits
                logits = cublas_retry(_fwd)

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
    return text, similarity_log


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
    """Extract answer from MATH problem response."""
    # Look for boxed answer
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    # Look for "the answer is"
    m = re.search(r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)', text)
    if m:
        return m.group(1).strip()
    # Last line heuristic
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return None


def normalize_math_answer(ans):
    """Normalize math answer for comparison."""
    if ans is None:
        return None
    ans = str(ans).strip()
    # Remove common wrappers
    ans = ans.replace('$', '').replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\%', '%').replace('\\frac', 'frac')
    ans = ans.strip()
    # Try numeric
    try:
        return float(ans.replace(',', ''))
    except:
        return ans.lower()


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
# Meta-Training
# ==============================================================================

def meta_train_config(backbone, tokenizer, device, ttt_layer, config_name,
                      num_steps=1000):
    print(f"\n{'='*60}")
    print(f"Meta-training config: {config_name} for {num_steps} steps")
    print(f"  precision={ttt_layer.use_precision}, phase_trans={ttt_layer.use_phase_transition}, gate_sep={ttt_layer.use_gate_separation}")
    print(f"  gate_init={ttt_layer.gate.item():.4f}")
    print(f"{'='*60}")

    print("Loading training data...")
    train_data = load_openwebtext_samples(NUM_TRAIN_SAMPLES, SEQ_LEN, tokenizer)
    train_data = train_data.to(device)
    print(f"  Loaded {len(train_data)} samples")

    inner_model = backbone.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        layers = tf.blocks
    elif hasattr(tf, 'block_groups'):
        layers = tf.block_groups
    target_layer = layers[INSERTION_LAYER]

    meta_optimizer = torch.optim.AdamW(ttt_layer.parameters(), lr=META_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=num_steps, eta_min=META_LR * 0.1)

    train_losses = []
    gate_values = []
    lr_values = []
    start_time = time.time()
    captured_hidden = [None]
    cublas_error_count = 0

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured_hidden[0] = output[0].clone()
        else:
            captured_hidden[0] = output.clone()
        return output

    for step in range(num_steps):
        torch.manual_seed(SEED + step * 100)
        meta_optimizer.zero_grad()

        batch_start = (step * META_BATCH) % len(train_data)
        batch_end = batch_start + META_BATCH
        if batch_end > len(train_data):
            batch = torch.cat([train_data[batch_start:], train_data[:batch_end - len(train_data)]], dim=0)
        else:
            batch = train_data[batch_start:batch_end]

        ttt_layer.reset_fast_weights(META_BATCH)
        input_ids, revealed_mask = create_masked_input(batch, MASK_TOKEN_ID, 0.8)
        input_ids = input_ids.to(device)
        revealed_mask = revealed_mask.to(device)
        mask_ratios = np.linspace(0.8, 0.2, K_UNROLL + 1)

        step_ssl_losses = []
        meta_losses = []
        step_had_cublas_error = False

        for k in range(K_UNROLL):
            current_mr = mask_ratios[k]
            target_mr = mask_ratios[k + 1]
            revealed_mask_k = revealed_mask.clone()

            # Forward pass with CUBLAS retry
            captured_hidden[0] = None
            hook_handle = target_layer.register_forward_hook(capture_hook)
            try:
                def _backbone_fwd():
                    with torch.no_grad():
                        backbone(input_ids=input_ids.contiguous(), output_hidden_states=False)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                try:
                    cublas_retry(_backbone_fwd)
                except RuntimeError as e:
                    if is_cublas_error(e):
                        cublas_error_count += 1
                        step_had_cublas_error = True
                        if cublas_error_count % 10 == 0:
                            print(f"  [Step {step}, k={k}] CUBLAS error #{cublas_error_count}, skipping")
                        break
                    raise
            finally:
                hook_handle.remove()

            if step_had_cublas_error:
                break

            if captured_hidden[0] is not None:
                hidden = captured_hidden[0].detach()
                h_normed = ttt_layer.layer_norm(hidden)

                if ttt_layer.W_fast is not None:
                    ttt_layer.W_fast.requires_grad_(True)

                fast_output = ttt_layer.forward_fast(h_normed)

                B, S, D = fast_output.shape
                fast_norm_vec = F.normalize(fast_output.float(), dim=-1)
                h_norm_target = F.normalize(h_normed.float().detach(), dim=-1)
                per_token_sim = (fast_norm_vec * h_norm_target).sum(dim=-1)
                per_token_loss = 1.0 - per_token_sim

                # Precision weighting
                if ttt_layer.use_precision:
                    try:
                        def _prec_fwd():
                            logits = backbone(input_ids=input_ids.contiguous()).logits
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            return logits
                        with torch.no_grad():
                            backbone_logits = cublas_retry(_prec_fwd)
                        pw = ttt_layer.compute_precision_weights(backbone_logits, revealed_mask_k)
                        weighted_loss = per_token_loss * pw
                    except RuntimeError as e:
                        if is_cublas_error(e):
                            cublas_error_count += 1
                            step_had_cublas_error = True
                            break
                        raise
                else:
                    weighted_loss = per_token_loss * revealed_mask_k

                n_revealed = revealed_mask_k.sum().clamp(min=1.0)

                # Phase transition
                lr_scale = 1.0
                if ttt_layer.use_phase_transition:
                    if not ttt_layer.scheduler.should_update(current_mr):
                        weighted_loss = weighted_loss * 0.0
                    else:
                        lr_scale = ttt_layer.scheduler.get_weight(current_mr)

                ssl_loss = weighted_loss.sum() / n_revealed
                step_ssl_losses.append(ssl_loss.item())

                if ttt_layer.W_fast is not None and ttt_layer.W_fast.requires_grad:
                    grads = torch.autograd.grad(ssl_loss, [ttt_layer.W_fast], create_graph=False, allow_unused=True)
                    grad = grads[0]
                    if grad is not None:
                        grad_norm_val = grad.norm().item()
                        if grad_norm_val > ttt_layer.max_grad_norm:
                            grad = grad * (ttt_layer.max_grad_norm / (grad_norm_val + 1e-8))
                        lr = ttt_layer.lr.detach() * lr_scale
                        ttt_layer.W_fast = (ttt_layer.W_fast.detach() - lr * grad.detach())

                # Meta-loss
                fast_output_after = ttt_layer.forward_fast(h_normed)
                fast_norm_after = F.normalize(fast_output_after.float(), dim=-1)
                per_token_loss_after = 1.0 - (fast_norm_after * h_norm_target).sum(dim=-1)

                if ttt_layer.use_precision:
                    meta_loss_val = (per_token_loss_after * pw).sum() / n_revealed
                else:
                    meta_loss_val = (per_token_loss_after * revealed_mask_k).sum() / n_revealed

                gate_val = ttt_layer.gate
                gate_reg = -0.01 * gate_val.log().clamp(min=-10)
                meta_loss_val = meta_loss_val + gate_reg
                meta_losses.append(meta_loss_val)

            # Reveal more tokens
            n_to_reveal = max(1, int((current_mr - target_mr) * SEQ_LEN))
            masked_positions = (input_ids == MASK_TOKEN_ID)
            for b in range(META_BATCH):
                masked_pos = masked_positions[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) > 0:
                    n = min(n_to_reveal, len(masked_pos))
                    perm = torch.randperm(len(masked_pos), device=device)[:n]
                    positions = masked_pos[perm]
                    input_ids[b, positions] = batch[b, positions]
                    revealed_mask[b, positions] = 1.0

        if step_had_cublas_error:
            gc.collect()
            torch.cuda.empty_cache()
            train_losses.append(train_losses[-1] if train_losses else 0.0)
            gate_values.append(float(ttt_layer.gate.item()))
            lr_values.append(float(ttt_layer.lr.item()))
            continue

        if meta_losses:
            total_meta_loss = sum(meta_losses) / len(meta_losses)
            total_meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(ttt_layer.parameters(), max_norm=1.0)
            meta_optimizer.step()
            scheduler.step()

        avg_loss = np.mean(step_ssl_losses) if step_ssl_losses else 0.0
        train_losses.append(float(avg_loss))
        gate_values.append(float(ttt_layer.gate.item()))
        lr_values.append(float(ttt_layer.lr.item()))

        if step % 100 == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (num_steps - step - 1) if step > 0 else 0
            gate_sep_str = ""
            if ttt_layer.use_gate_separation:
                gate_sep_str = f" ssl_gate={ttt_layer.ssl_gate.item():.4f}"
            print(f"  [{config_name}] Step {step:4d}/{num_steps}: "
                  f"loss={avg_loss:.4f} gate={ttt_layer.gate.item():.4f}"
                  f"{gate_sep_str} ttt_lr={ttt_layer.lr.item():.4f} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                  f"{f' cublas_errs={cublas_error_count}' if cublas_error_count > 0 else ''}")
            report_progress(f"train_{config_name}", step, num_steps, {
                "ssl_loss": avg_loss, "gate": ttt_layer.gate.item(), "ttt_lr": ttt_layer.lr.item()})

        if step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    half = max(1, len(train_losses) // 2)
    first_half = np.mean(train_losses[:half])
    last_half = np.mean(train_losses[half:])
    decrease_pct = (first_half - last_half) / (first_half + 1e-8) * 100

    print(f"\n  [{config_name}] Training complete in {total_time:.0f}s")
    print(f"    Loss: {first_half:.4f} -> {last_half:.4f} ({decrease_pct:.1f}% decrease)")
    print(f"    Final gate: {ttt_layer.gate.item():.4f}")
    if cublas_error_count > 0:
        print(f"    CUBLAS errors encountered: {cublas_error_count}")

    return {
        "config": config_name, "num_steps": num_steps,
        "first_half_avg": float(first_half), "last_half_avg": float(last_half),
        "decrease_pct": float(decrease_pct),
        "gate_final": float(ttt_layer.gate.item()),
        "ssl_gate_final": float(ttt_layer.ssl_gate.item()) if ttt_layer.use_gate_separation else None,
        "ttt_lr_final": float(ttt_layer.lr.item()),
        "total_time_s": float(total_time),
        "cublas_errors": cublas_error_count,
        "train_losses": train_losses[::10],
        "gate_values": gate_values[::10],
    }


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_gsm8k(model, tokenizer, device, ttt_layer=None,
                   insertion_layer=None, num_samples=16, gen_len=512,
                   config_name="vanilla", collect_similarity=False):
    from datasets import load_from_disk
    gsm8k = load_from_disk(f"{SHARED_DIR}/datasets/gsm8k")
    gsm8k = gsm8k.select(range(min(num_samples, len(gsm8k))))
    print(f"\n  Evaluating {config_name} on GSM8K ({len(gsm8k)} problems)...")

    correct = 0
    total = 0
    results = []
    all_similarity = []
    start_time = time.time()

    for idx in range(len(gsm8k)):
        item = gsm8k[idx]
        question = item['question']
        target = extract_gsm8k_target(item['answer'])
        if target is None:
            continue

        prompt = ("Solve the math problem step by step. "
                  "End your answer with #### followed by the final numerical answer.\n\n"
                  f"Problem: {question}\n\nSolution:")

        gen_text = ""
        sim_log = None
        try:
            if ttt_layer:
                gen_text, sim_log = generate_with_enhanced_ttt(
                    model, ttt_layer, tokenizer, prompt, device, gen_len,
                    insertion_layer, temperature=0.0,
                    collect_similarity=(collect_similarity and idx < 5))
            else:
                gen_text = generate_vanilla(
                    model, tokenizer, prompt, device, gen_len, temperature=0.0)
        except Exception as e:
            if idx < 3:
                print(f"    [{idx}] Error: {e}")
            gen_text = ""

        if sim_log:
            all_similarity.extend(sim_log)

        extracted = extract_gsm8k_answer(gen_text)
        is_correct = (extracted is not None and abs(extracted - target) < 1e-3)
        if is_correct:
            correct += 1
        total += 1

        results.append({"idx": idx, "target": target, "extracted": extracted,
                        "is_correct": is_correct,
                        "gen_text_preview": gen_text[:200] if idx < 5 else None})

        if (idx + 1) % 8 == 0 or idx == len(gsm8k) - 1:
            acc = correct / total if total > 0 else 0
            elapsed = time.time() - start_time
            print(f"    [{idx+1}/{len(gsm8k)}] acc={acc:.3f} ({correct}/{total}) elapsed={elapsed:.0f}s")

        if idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"  GSM8K {config_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")

    result = {"accuracy": accuracy, "correct": correct, "total": total,
              "time_s": elapsed, "results": results}
    if all_similarity:
        avg_sim = np.mean([s["cosine_similarity"] for s in all_similarity])
        result["avg_logit_similarity"] = float(avg_sim)
        result["similarity_log"] = all_similarity[:20]  # Keep first 20 entries
    return result


def evaluate_math500(model, tokenizer, device, ttt_layer=None,
                     insertion_layer=None, num_samples=16, gen_len=512,
                     config_name="vanilla"):
    """Evaluate on MATH500 subset."""
    from datasets import load_from_disk
    math_path = f"{SHARED_DIR}/datasets/math500"
    try:
        math_ds = load_from_disk(math_path)
    except Exception as e:
        print(f"  Warning: Cannot load MATH500 from {math_path}: {e}")
        return {"accuracy": None, "error": str(e)}

    math_ds = math_ds.select(range(min(num_samples, len(math_ds))))
    print(f"\n  Evaluating {config_name} on MATH500 ({len(math_ds)} problems)...")

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for idx in range(len(math_ds)):
        item = math_ds[idx]
        question = item.get('problem', item.get('question', ''))
        target_ans = item.get('answer', item.get('solution', ''))

        prompt = ("Solve the following math problem step by step. "
                  "Put your final answer in \\boxed{}.\n\n"
                  f"Problem: {question}\n\nSolution:")

        gen_text = ""
        try:
            if ttt_layer:
                gen_text, _ = generate_with_enhanced_ttt(
                    model, ttt_layer, tokenizer, prompt, device, gen_len,
                    insertion_layer, temperature=0.0)
            else:
                gen_text = generate_vanilla(
                    model, tokenizer, prompt, device, gen_len, temperature=0.0)
        except Exception as e:
            if idx < 3:
                print(f"    [{idx}] Error: {e}")
            gen_text = ""

        pred = normalize_math_answer(extract_math_answer(gen_text))
        gold = normalize_math_answer(extract_math_answer(target_ans))
        if gold is None:
            gold = normalize_math_answer(target_ans)

        is_correct = (pred is not None and gold is not None and pred == gold)
        if is_correct:
            correct += 1
        total += 1

        results.append({"idx": idx, "target": str(gold), "extracted": str(pred),
                        "is_correct": is_correct,
                        "gen_text_preview": gen_text[:200] if idx < 3 else None})

        if (idx + 1) % 8 == 0 or idx == len(math_ds) - 1:
            acc = correct / total if total > 0 else 0
            elapsed = time.time() - start_time
            print(f"    [{idx+1}/{len(math_ds)}] acc={acc:.3f} ({correct}/{total}) elapsed={elapsed:.0f}s")

        if idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"  MATH500 {config_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total,
            "time_s": elapsed, "results": results}


# ==============================================================================
# Ablation Configurations
# ==============================================================================

ABLATION_CONFIGS = {
    "base": {
        "desc": "DaL-Linear base (no enhancements, gate_init=0)",
        "precision_weighted": False, "phase_transition": False,
        "gate_separation": False, "gate_init": 0.0,
    },
    "precision": {
        "desc": "DaL-Linear + precision weighting (entropy-based)",
        "precision_weighted": True, "phase_transition": False,
        "gate_separation": False, "gate_init": 0.0,
    },
    "phase_transition": {
        "desc": "DaL-Linear + phase-transition scheduling",
        "precision_weighted": False, "phase_transition": True,
        "gate_separation": False, "gate_init": 0.0,
    },
    "gate_sep": {
        "desc": "DaL-Linear + residual gate separation",
        "precision_weighted": False, "phase_transition": False,
        "gate_separation": True, "gate_init": 0.0,
    },
    "full": {
        "desc": "DaL-Full (all three enhancements)",
        "precision_weighted": True, "phase_transition": True,
        "gate_separation": True, "gate_init": 0.0,
    },
}


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"{'='*70}")
    print(f"M6: Enhancement Ablation on DaL-Linear (v2)")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Configs: {list(ABLATION_CONFIGS.keys())}")
    print(f"Steps per config: {META_STEPS}")
    print(f"Eval: GSM8K-{PILOT_SAMPLES} + MATH500-{PILOT_SAMPLES}")
    print()

    overall_start = time.time()
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)

    # Clean old markers
    clean_old_markers()

    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")

    print("Loading LLaDA-8B-Instruct...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    backbone.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for param in backbone.parameters():
        param.requires_grad = False

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"  Model loaded. GPU: {gpu_name}, VRAM: {vram_total}MB, Used: {gpu_mem:.1f}GB")

    # Warmup forward pass to catch CUBLAS issues early
    print("  Warmup forward pass...")
    try:
        with torch.no_grad():
            warmup_x = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            _ = backbone(warmup_x)
        print("  Warmup OK")
    except RuntimeError as e:
        if is_cublas_error(e):
            print(f"  WARNING: CUBLAS error in warmup, retrying...")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
            with torch.no_grad():
                _ = backbone(torch.tensor([[1, 2, 3, 4, 5]], device=device))
            print("  Warmup OK (after retry)")
        else:
            raise

    # === Run ablation configs ===
    all_gsm8k = {}
    all_math500 = {}
    all_train_info = {}
    config_wall_clock = {}

    for i, (config_name, config) in enumerate(ABLATION_CONFIGS.items()):
        config_start = time.time()
        print(f"\n{'='*70}")
        print(f"Config {i+1}/{len(ABLATION_CONFIGS)}: {config_name}")
        print(f"  {config['desc']}")
        print(f"{'='*70}")

        report_progress(f"config_{config_name}", i, len(ABLATION_CONFIGS))

        ttt_layer = EnhancedTTTLinear(
            d_model=D_MODEL, vocab_size=VOCAB_SIZE, ttt_lr=0.1,
            gate_init=config["gate_init"],
            precision_weighted=config["precision_weighted"],
            phase_transition=config["phase_transition"],
            gate_separation=config["gate_separation"],
        ).to(device).to(torch.bfloat16)

        # Meta-train
        train_info = meta_train_config(backbone, tokenizer, device, ttt_layer,
                                        config_name, num_steps=META_STEPS)
        all_train_info[config_name] = train_info
        gc.collect()
        torch.cuda.empty_cache()

        # Eval GSM8K (collect similarity for gate_sep and full configs)
        collect_sim = config_name in ("gate_sep", "full")
        gsm8k_result = evaluate_gsm8k(backbone, tokenizer, device,
                                       ttt_layer=ttt_layer, insertion_layer=INSERTION_LAYER,
                                       num_samples=PILOT_SAMPLES, gen_len=GEN_LEN_GSM8K,
                                       config_name=config_name,
                                       collect_similarity=collect_sim)
        all_gsm8k[config_name] = gsm8k_result
        gc.collect()
        torch.cuda.empty_cache()

        # Eval MATH500
        math_result = evaluate_math500(backbone, tokenizer, device,
                                        ttt_layer=ttt_layer, insertion_layer=INSERTION_LAYER,
                                        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN_MATH,
                                        config_name=config_name)
        all_math500[config_name] = math_result
        del ttt_layer
        gc.collect()
        torch.cuda.empty_cache()

        config_elapsed = time.time() - config_start
        config_wall_clock[config_name] = config_elapsed
        print(f"\n  Config {config_name} total wall-clock: {config_elapsed:.0f}s ({config_elapsed/60:.1f}min)")

    # === Vanilla baseline ===
    # Run the long autoregressive eval after training. On Blackwell this avoids
    # poisoning the subsequent meta-training forward with a stale async cuBLAS error.
    print(f"\n{'='*70}")
    print("Phase 0: Vanilla baseline")
    print(f"{'='*70}")
    report_progress("eval_vanilla", 0, 1)
    vanilla_gsm8k = evaluate_gsm8k(backbone, tokenizer, device,
                                     num_samples=PILOT_SAMPLES, gen_len=GEN_LEN_GSM8K,
                                     config_name="vanilla")
    vanilla_math500 = evaluate_math500(backbone, tokenizer, device,
                                        num_samples=PILOT_SAMPLES, gen_len=GEN_LEN_MATH,
                                        config_name="vanilla")
    all_gsm8k["vanilla"] = vanilla_gsm8k
    all_math500["vanilla"] = vanilla_math500
    gc.collect()
    torch.cuda.empty_cache()

    # === Summary ===
    total_elapsed = time.time() - overall_start
    vanilla_gsm8k_acc = vanilla_gsm8k['accuracy']
    vanilla_math_acc = vanilla_math500.get('accuracy', 0) or 0

    print(f"\n{'='*70}")
    print("ENHANCEMENT ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'GSM8K':<8} {'MATH':<8} {'Gate':<8} {'Loss↓%':<8} {'Time(s)':<8} {'SimBT':<8}")
    print(f"{'-'*66}")
    print(f"{'vanilla':<18} {vanilla_gsm8k_acc:.3f}   {vanilla_math_acc:.3f}   {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

    for config_name in ABLATION_CONFIGS:
        gsm_acc = all_gsm8k[config_name]['accuracy']
        math_acc = all_math500[config_name].get('accuracy', 0) or 0
        gate = all_train_info[config_name]['gate_final']
        decrease = all_train_info[config_name]['decrease_pct']
        wc = config_wall_clock.get(config_name, 0)
        sim = all_gsm8k[config_name].get('avg_logit_similarity', None)
        sim_str = f"{sim:.3f}" if sim is not None else "N/A"
        delta_g = gsm_acc - vanilla_gsm8k_acc
        delta_m = math_acc - vanilla_math_acc
        print(f"{config_name:<18} {gsm_acc:.3f}   {math_acc:.3f}   {gate:.4f}  {decrease:5.1f}%  {wc:6.0f}   {sim_str:<8} "
              f"(Δg={delta_g:+.3f}, Δm={delta_m:+.3f})")

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    best_config = max(ABLATION_CONFIGS.keys(),
                      key=lambda c: (all_gsm8k[c]['accuracy'] + (all_math500[c].get('accuracy', 0) or 0)) / 2)
    best_gsm = all_gsm8k[best_config]['accuracy']
    best_math = all_math500[best_config].get('accuracy', 0) or 0
    print(f"  Best config: {best_config} (GSM8K={best_gsm:.3f}, MATH={best_math:.3f})")

    # === Save results ===
    def json_default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return str(obj)

    full_results = {
        "task_id": TASK_ID,
        "experiment": "M6: Enhancement Ablation on DaL-Linear",
        "model": "LLaDA-8B-Instruct", "mode": "PILOT",
        "pilot_samples": PILOT_SAMPLES,
        "meta_steps_per_config": META_STEPS,
        "gate_init_fix": "Changed from sigmoid(-5)=0.007 to sigmoid(0)=0.5",
        "configs": {name: cfg["desc"] for name, cfg in ABLATION_CONFIGS.items()},
        "vanilla_baseline": {
            "gsm8k": {k: v for k, v in vanilla_gsm8k.items() if k != "results"},
            "math500": {k: v for k, v in vanilla_math500.items() if k != "results"},
        },
        "training": all_train_info,
        "evaluation": {
            name: {
                "gsm8k": {k: v for k, v in all_gsm8k[name].items() if k not in ("results", "similarity_log")},
                "math500": {k: v for k, v in all_math500[name].items() if k != "results"},
                "wall_clock_s": config_wall_clock.get(name, 0),
            }
            for name in ABLATION_CONFIGS
        },
        "comparison": {
            name: {
                "gsm8k_accuracy": all_gsm8k[name]["accuracy"],
                "gsm8k_delta_vs_vanilla": all_gsm8k[name]["accuracy"] - vanilla_gsm8k_acc,
                "math500_accuracy": all_math500[name].get("accuracy"),
                "math500_delta_vs_vanilla": (all_math500[name].get("accuracy", 0) or 0) - vanilla_math_acc,
                "gate_final": all_train_info[name]["gate_final"],
                "loss_decrease_pct": all_train_info[name]["decrease_pct"],
                "wall_clock_s": config_wall_clock.get(name, 0),
                "logit_similarity": all_gsm8k[name].get("avg_logit_similarity"),
            }
            for name in ABLATION_CONFIGS
        },
        "best_config": best_config,
        "best_gsm8k": best_gsm,
        "best_math500": best_math,
        "analysis": {
            "gate_fix_effective": all(all_train_info[c]["gate_final"] > 0.1 for c in ABLATION_CONFIGS),
            "any_gsm8k_improvement": any(all_gsm8k[c]["accuracy"] > vanilla_gsm8k_acc for c in ABLATION_CONFIGS),
            "any_math_improvement": any((all_math500[c].get("accuracy", 0) or 0) > vanilla_math_acc for c in ABLATION_CONFIGS),
            "phase_transition_time_savings": (
                config_wall_clock.get("phase_transition", 0) / max(config_wall_clock.get("base", 1), 1)
                if "phase_transition" in config_wall_clock and "base" in config_wall_clock else None
            ),
        },
        "timing": {"total_elapsed_s": float(total_elapsed), "total_elapsed_min": float(total_elapsed / 60)},
        "gpu_info": {"device": str(device), "gpu_name": gpu_name,
                     "vram_total_mb": vram_total},
        "timestamp": datetime.now().isoformat(),
        "sample_results": {
            name: {
                "gsm8k": all_gsm8k[name].get("results", [])[:3],
                "math500": all_math500[name].get("results", [])[:3],
            }
            for name in list(ABLATION_CONFIGS.keys())[:2]
        },
    }

    full_path = Path(FULL_RESULTS_DIR) / "m6_enhancement_ablation.json"
    full_path.write_text(json.dumps(full_results, indent=2, default=json_default))
    print(f"\nResults saved to {full_path}")

    # Also save log
    log_path = Path(FULL_RESULTS_DIR) / "m6_enhancement_ablation_log.txt"
    # (Log is captured by nohup redirect)

    status = "success"
    if not full_results["analysis"]["gate_fix_effective"]:
        status = "partial"
    summary = (f"Enhancement ablation complete: vanilla_gsm8k={vanilla_gsm8k_acc:.3f}, "
               f"best={best_config} gsm8k={best_gsm:.3f} math={best_math:.3f}, "
               f"gate_fix={'effective' if full_results['analysis']['gate_fix_effective'] else 'partial'}, "
               f"total_time={total_elapsed/60:.1f}min")
    mark_done(status, summary)
    print(f"DONE marker written: {status}")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        mark_done("error", f"Fatal: {str(e)[:200]}")
        sys.exit(1)
