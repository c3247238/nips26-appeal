#!/usr/bin/env python3
"""
M3b: Progressive Signal Enrichment Analysis (PILOT mode)

For the trained DaL-Linear model, record self-supervised loss L_ssl(W_t) at every
denoising step t for 100 GSM8K sequences. Analyze progressive signal enrichment (H3).

Measurements:
  1. L_ssl trajectory across all denoising steps (per sequence)
  2. Revealed token count |R_t| at each step
  3. Spearman correlation between L_ssl improvement rate and |R_t| growth rate
  4. Mid-denoising fast weight reset experiment (H9): reset W_f to W_0 at t=T/2,
     measure accuracy drop

Known Blackwell CUBLAS issues:
  - TTT layer uses float32 for gradient computation
  - CUBLAS_WORKSPACE_CONFIG=:4096:8

Output:
  - DONE marker: exp/results/progressive_signal_DONE
  - Progress: exp/results/progressive_signal_PROGRESS.json
  - Full results: exp/results/full/m3b_progressive_signal.json
"""

import os, sys, json, time, gc, re, math, traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats

# Blackwell GPU CUBLAS workarounds
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# === Paths ===
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
FULL_RESULTS_DIR = f"{PROJECT_DIR}/exp/results/full"
CODE_DIR = f"{PROJECT_DIR}/exp/code/dal"
SHARED_DIR = f"{REMOTE_BASE}/shared"

TASK_ID = "progressive_signal"
SEED = 42

# === Model Config (LLaDA-8B-Instruct) ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
D_MODEL = 4096
VOCAB_SIZE = 126464
N_LAYERS = 32
INSERTION_LAYER = 16  # L/2

# === Analysis Config ===
NUM_SEQUENCES = 100  # 100 GSM8K sequences for analysis
GEN_STEPS = 128
GEN_LEN_GSM8K = 512
TEMPERATURE = 0.0  # greedy

# === Checkpoint ===
DAL_LINEAR_CKPT = f"{FULL_RESULTS_DIR}/dal_linear_ckpt_step5000.pt"

sys.path.insert(0, CODE_DIR)
from ttt_layer import TTTLayer, build_ttt_layer


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
# Load checkpoint
# ==============================================================================

def load_ttt_linear_checkpoint(ckpt_path, device):
    """Load TTT-Linear layer from checkpoint."""
    print(f"  Loading DaL-Linear checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    ttt_layer = build_ttt_layer(
        variant="linear",
        d_model=D_MODEL,
        d_ttt=512,
        vocab_size=VOCAB_SIZE,
    )
    # W_fast is a runtime buffer, not a parameter — filter it out
    state_dict = {k: v for k, v in ckpt["ttt_layer_state_dict"].items()
                  if "W_fast" not in k and "b1_fast" not in k and "b2_fast" not in k
                  and "W1_fast" not in k and "W2_fast" not in k}
    ttt_layer.load_state_dict(state_dict, strict=False)
    # Keep TTT layer in float32 for CUBLAS stability on Blackwell
    ttt_layer = ttt_layer.to(device).to(torch.float32)
    ttt_layer.eval()

    step = ckpt.get("step", "?")
    gate = ttt_layer.gate.item()
    ttt_lr = ttt_layer.lr.item()
    print(f"  DaL-Linear loaded: step={step}, gate={gate:.4f}, ttt_lr={ttt_lr:.4f}")
    print(f"  Using float32 for TTT layer (Blackwell CUBLAS workaround)")

    return ttt_layer, ckpt


# ==============================================================================
# Core analysis: generate with TTT and record SSL loss per step
# ==============================================================================

def generate_with_ssl_tracking(
    model, ttt_layer, tokenizer, prompt_text, device, gen_len,
    insertion_layer, temperature=0.0, reset_at_half=False,
):
    """
    LLaDA generation with TTT-Linear injection, recording SSL loss at every step.

    Args:
        reset_at_half: If True, reset fast weights to W_0 at step GEN_STEPS//2 (H9 test)

    Returns:
        text: Generated text
        step_data: List of dicts with per-step metrics
    """
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

    # Reset TTT fast weights
    ttt_layer.reset_fast_weights(batch_size=1)

    # Save initial fast weight state for potential mid-reset
    initial_W_fast = ttt_layer.fast_weight.W_fast.clone()

    ssl_head = ttt_layer.ssl_head
    layer_norm = ttt_layer.layer_norm

    step_data = []
    captured_hidden = [None]

    # Track current step for the hook
    current_step = [0]

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        captured_hidden[0] = hidden.clone()

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()
        total_gen_tokens = x.shape[1] - prompt_len
        mask_ratio = n_masked / max(total_gen_tokens, 1)

        ssl_loss_val = float('nan')
        grad_norm_val = float('nan')

        if n_revealed > 5 and n_masked > 3:
            # Cast hidden to float32 for TTT computation
            h_normed = layer_norm(hidden.detach().float())
            gate = ttt_layer.gate.detach()

            with torch.enable_grad():
                fast_params = ttt_layer.fast_weight.get_params_for_grad()
                for p in fast_params:
                    p.requires_grad_(True)

                fast_output_grad = ttt_layer.fast_weight(h_normed)
                ssl_logits_grad = ssl_head(fast_output_grad)
                B, S, V = ssl_logits_grad.shape
                n_rev = revealed_mask.sum().clamp(min=1.0)
                per_token_loss_grad = F.cross_entropy(
                    ssl_logits_grad.view(-1, V).float(),
                    target_ids.view(-1),
                    reduction="none"
                ).view(B, S)
                masked_loss_grad = per_token_loss_grad * revealed_mask
                ssl_loss = masked_loss_grad.sum() / n_rev

                ssl_loss_val = ssl_loss.item()

                grads = torch.autograd.grad(ssl_loss, fast_params,
                                             create_graph=False, allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, fast_params)]

                grad_norm_val = torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()

            # Clip gradients
            if grad_norm_val > ttt_layer.max_grad_norm:
                scale = ttt_layer.max_grad_norm / (grad_norm_val + 1e-8)
                grads = [g * scale for g in grads]

            lr = ttt_layer.lr.detach()
            ttt_layer.fast_weight.apply_update(grads, lr)

            # Compute fast weight norm after update
            fw_norm = ttt_layer.fast_weight.W_fast.norm().item()

            with torch.no_grad():
                new_fast_output = ttt_layer.fast_weight(h_normed)
                delta = gate * new_fast_output

            if isinstance(output, tuple):
                output = (hidden + delta.to(hidden.dtype),) + output[1:]
            else:
                output = hidden + delta.to(hidden.dtype)
        else:
            fw_norm = ttt_layer.fast_weight.W_fast.norm().item()

        step_data.append({
            "step": current_step[0],
            "ssl_loss": ssl_loss_val,
            "grad_norm": grad_norm_val,
            "n_revealed": int(n_revealed),
            "n_masked": int(n_masked),
            "mask_ratio": float(mask_ratio),
            "fast_weight_norm": float(fw_norm),
        })

        return output

    hook_handle = target_layer.register_forward_hook(capture_and_inject)

    try:
        with torch.no_grad():
            for i in range(GEN_STEPS):
                current_step[0] = i

                # H9: Mid-denoising reset
                if reset_at_half and i == GEN_STEPS // 2:
                    ttt_layer.fast_weight.W_fast = initial_W_fast.clone()

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
    return text, step_data


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


# ==============================================================================
# Analysis Functions
# ==============================================================================

def compute_signal_analysis(all_step_data: List[List[Dict]]) -> Dict:
    """
    Compute progressive signal enrichment metrics from per-step data.

    Returns analysis dict with:
      - mean_ssl_loss_trajectory: averaged across sequences
      - mean_revealed_trajectory: averaged across sequences
      - spearman_rho: correlation between ssl_loss change rate and |R_t| growth
      - monotonicity_score: fraction of steps where ssl_loss decreases
    """
    n_seq = len(all_step_data)

    # Find max steps across sequences
    max_steps = max(len(sd) for sd in all_step_data)

    # Aggregate trajectories (pad with NaN for sequences that ended early)
    ssl_loss_matrix = np.full((n_seq, max_steps), np.nan)
    revealed_matrix = np.full((n_seq, max_steps), np.nan)
    mask_ratio_matrix = np.full((n_seq, max_steps), np.nan)
    fw_norm_matrix = np.full((n_seq, max_steps), np.nan)
    grad_norm_matrix = np.full((n_seq, max_steps), np.nan)

    for i, sd in enumerate(all_step_data):
        for j, d in enumerate(sd):
            ssl_loss_matrix[i, j] = d["ssl_loss"]
            revealed_matrix[i, j] = d["n_revealed"]
            mask_ratio_matrix[i, j] = d["mask_ratio"]
            fw_norm_matrix[i, j] = d["fast_weight_norm"]
            grad_norm_matrix[i, j] = d["grad_norm"]

    # Mean trajectories (ignore NaN)
    mean_ssl = np.nanmean(ssl_loss_matrix, axis=0)
    std_ssl = np.nanstd(ssl_loss_matrix, axis=0)
    mean_revealed = np.nanmean(revealed_matrix, axis=0)
    mean_mask_ratio = np.nanmean(mask_ratio_matrix, axis=0)
    mean_fw_norm = np.nanmean(fw_norm_matrix, axis=0)
    mean_grad_norm = np.nanmean(grad_norm_matrix, axis=0)

    # Spearman correlation: ssl_loss improvement rate vs |R_t| growth rate
    # For each sequence, compute:
    #   - ssl_loss improvement: delta_L[t] = L[t] - L[t-1]  (negative = improving)
    #   - revealed growth: delta_R[t] = R[t] - R[t-1]
    all_delta_L = []
    all_delta_R = []
    per_seq_spearman = []

    for i in range(n_seq):
        valid = ~np.isnan(ssl_loss_matrix[i])
        losses = ssl_loss_matrix[i][valid]
        reveals = revealed_matrix[i][valid]

        if len(losses) < 10:
            continue

        delta_L = np.diff(losses)
        delta_R = np.diff(reveals)

        # Filter out steps where both are zero
        mask = (delta_R != 0) | (delta_L != 0)
        if mask.sum() < 5:
            continue

        all_delta_L.extend(delta_L[mask].tolist())
        all_delta_R.extend(delta_R[mask].tolist())

        # Per-sequence Spearman
        rho, pval = stats.spearmanr(delta_L[mask], delta_R[mask])
        per_seq_spearman.append({"seq_idx": i, "rho": float(rho), "pval": float(pval)})

    # Global Spearman
    global_rho, global_pval = float('nan'), float('nan')
    if len(all_delta_L) > 10:
        global_rho, global_pval = stats.spearmanr(all_delta_L, all_delta_R)
        global_rho = float(global_rho)
        global_pval = float(global_pval)

    # Monotonicity: fraction of steps where mean ssl_loss decreases
    valid_mean = mean_ssl[~np.isnan(mean_ssl)]
    if len(valid_mean) > 1:
        diffs = np.diff(valid_mean)
        monotonicity = float(np.mean(diffs < 0))
    else:
        monotonicity = float('nan')

    # Identify phase transition zone: where ssl_loss drop rate is highest
    if len(valid_mean) > 10:
        # Smooth with moving average
        window = 5
        smoothed = np.convolve(valid_mean, np.ones(window)/window, mode='valid')
        if len(smoothed) > 2:
            drop_rates = -np.diff(smoothed)
            peak_idx = np.argmax(drop_rates) + window // 2
            peak_mask_ratio = float(mean_mask_ratio[peak_idx]) if peak_idx < len(mean_mask_ratio) else float('nan')
        else:
            peak_idx = 0
            peak_mask_ratio = float('nan')
    else:
        peak_idx = 0
        peak_mask_ratio = float('nan')

    return {
        "n_sequences": n_seq,
        "max_steps": int(max_steps),
        "mean_ssl_trajectory": [float(v) if not np.isnan(v) else None for v in mean_ssl],
        "std_ssl_trajectory": [float(v) if not np.isnan(v) else None for v in std_ssl],
        "mean_revealed_trajectory": [float(v) if not np.isnan(v) else None for v in mean_revealed],
        "mean_mask_ratio_trajectory": [float(v) if not np.isnan(v) else None for v in mean_mask_ratio],
        "mean_fw_norm_trajectory": [float(v) if not np.isnan(v) else None for v in mean_fw_norm],
        "mean_grad_norm_trajectory": [float(v) if not np.isnan(v) else None for v in mean_grad_norm],
        "spearman_global": {"rho": global_rho, "pval": global_pval},
        "spearman_per_sequence": per_seq_spearman,
        "spearman_mean_rho": float(np.nanmean([s["rho"] for s in per_seq_spearman])) if per_seq_spearman else float('nan'),
        "monotonicity_score": monotonicity,
        "peak_drop_step": int(peak_idx),
        "peak_drop_mask_ratio": peak_mask_ratio,
        # Sampled trajectories (every 4 steps) for visualization
        "ssl_trajectory_sampled": [float(v) if not np.isnan(v) else None for v in mean_ssl[::4]],
        "revealed_trajectory_sampled": [float(v) if not np.isnan(v) else None for v in mean_revealed[::4]],
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    print("=" * 70)
    print("M3b: Progressive Signal Enrichment Analysis (PILOT)")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {physical_gpu} (CUDA_VISIBLE_DEVICES)")
    print(f"Analyzing: {NUM_SEQUENCES} GSM8K sequences, {GEN_STEPS} denoising steps each")
    print(f"Checkpoint: {DAL_LINEAR_CKPT}")
    print()

    # Write PID
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)

    overall_start = time.time()

    # === Load backbone model ===
    report_progress("loading_model", 0, 3)
    print("Loading LLaDA-8B-Instruct...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Loaded. GPU: {gpu_name}")

    # === Load TTT-Linear checkpoint ===
    report_progress("loading_checkpoint", 1, 3)
    ttt_layer, ckpt = load_ttt_linear_checkpoint(DAL_LINEAR_CKPT, device)

    gc.collect()
    torch.cuda.empty_cache()

    # === Load GSM8K dataset ===
    report_progress("loading_data", 2, 3)
    from datasets import load_from_disk
    gsm8k = load_from_disk(f"{SHARED_DIR}/datasets/gsm8k")
    gsm8k = gsm8k.select(range(min(NUM_SEQUENCES, len(gsm8k))))
    print(f"  Loaded GSM8K: {len(gsm8k)} sequences")

    # =================================================================
    # Phase 1: SSL Loss Trajectory Recording (Normal DaL)
    # =================================================================
    print(f"\n{'='*70}")
    print(f"Phase 1: Recording SSL loss trajectories ({NUM_SEQUENCES} sequences)")
    print(f"{'='*70}")

    all_step_data = []
    dal_results = []
    dal_correct = 0
    dal_total = 0
    phase1_start = time.time()

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
            gen_text, step_data = generate_with_ssl_tracking(
                model, ttt_layer, tokenizer, prompt, device,
                GEN_LEN_GSM8K, INSERTION_LAYER, temperature=TEMPERATURE,
                reset_at_half=False,
            )
        except Exception as e:
            gen_text = ""
            step_data = []
            if idx < 5:
                print(f"    [{idx}] Error: {e}")
                traceback.print_exc()

        all_step_data.append(step_data)

        extracted = extract_gsm8k_answer(gen_text)
        is_correct = (extracted is not None and abs(extracted - target) < 1e-3)
        if is_correct:
            dal_correct += 1
        dal_total += 1

        dal_results.append({
            "idx": idx, "target": float(target),
            "extracted": float(extracted) if extracted else None,
            "is_correct": bool(is_correct),
            "n_steps_recorded": len(step_data),
        })

        if (idx + 1) % 10 == 0 or idx == len(gsm8k) - 1:
            acc = dal_correct / dal_total if dal_total > 0 else 0
            elapsed = time.time() - phase1_start
            eta = elapsed / (idx + 1) * (len(gsm8k) - idx - 1) if idx > 0 else 0
            # Show a few ssl_loss values from the latest sequence
            loss_preview = ""
            if step_data and len(step_data) > 10:
                l0 = step_data[0]["ssl_loss"]
                lmid = step_data[len(step_data)//2]["ssl_loss"]
                lend = step_data[-1]["ssl_loss"]
                loss_preview = f" ssl[0,mid,end]=[{l0:.2f},{lmid:.2f},{lend:.2f}]"
            print(f"    [{idx+1}/{len(gsm8k)}] acc={acc:.3f} ({dal_correct}/{dal_total}) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s{loss_preview}")
            report_progress("phase1_ssl_tracking", idx + 1, len(gsm8k), {
                "correct": dal_correct, "total": dal_total, "accuracy": acc,
            })

        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    phase1_elapsed = time.time() - phase1_start
    dal_accuracy = dal_correct / dal_total if dal_total > 0 else 0
    print(f"\n  Phase 1 complete: {dal_correct}/{dal_total} = {dal_accuracy:.3f} "
          f"({phase1_elapsed:.0f}s)")

    # =================================================================
    # Phase 2: Mid-Reset Experiment (H9)
    # =================================================================
    print(f"\n{'='*70}")
    print(f"Phase 2: Mid-Denoising Fast Weight Reset Experiment (H9)")
    print(f"{'='*70}")
    print(f"  Resetting W_f to W_0 at step {GEN_STEPS//2} for {NUM_SEQUENCES} sequences")

    reset_results = []
    reset_correct = 0
    reset_total = 0
    phase2_start = time.time()

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
            gen_text, step_data = generate_with_ssl_tracking(
                model, ttt_layer, tokenizer, prompt, device,
                GEN_LEN_GSM8K, INSERTION_LAYER, temperature=TEMPERATURE,
                reset_at_half=True,
            )
        except Exception as e:
            gen_text = ""
            step_data = []
            if idx < 5:
                print(f"    [{idx}] Error: {e}")
                traceback.print_exc()

        extracted = extract_gsm8k_answer(gen_text)
        is_correct = (extracted is not None and abs(extracted - target) < 1e-3)
        if is_correct:
            reset_correct += 1
        reset_total += 1

        reset_results.append({
            "idx": idx, "target": float(target),
            "extracted": float(extracted) if extracted else None,
            "is_correct": bool(is_correct),
        })

        if (idx + 1) % 10 == 0 or idx == len(gsm8k) - 1:
            acc = reset_correct / reset_total if reset_total > 0 else 0
            elapsed = time.time() - phase2_start
            eta = elapsed / (idx + 1) * (len(gsm8k) - idx - 1) if idx > 0 else 0
            print(f"    [{idx+1}/{len(gsm8k)}] acc={acc:.3f} ({reset_correct}/{reset_total}) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            report_progress("phase2_mid_reset", idx + 1, len(gsm8k), {
                "correct": reset_correct, "total": reset_total, "accuracy": acc,
            })

        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    phase2_elapsed = time.time() - phase2_start
    reset_accuracy = reset_correct / reset_total if reset_total > 0 else 0
    print(f"\n  Phase 2 complete: {reset_correct}/{reset_total} = {reset_accuracy:.3f} "
          f"({phase2_elapsed:.0f}s)")

    # =================================================================
    # Phase 3: Vanilla baseline for comparison
    # =================================================================
    print(f"\n{'='*70}")
    print(f"Phase 3: Vanilla Baseline ({NUM_SEQUENCES} sequences)")
    print(f"{'='*70}")

    vanilla_correct = 0
    vanilla_total = 0
    phase3_start = time.time()

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
            gen_text = generate_vanilla(
                model, tokenizer, prompt, device,
                GEN_LEN_GSM8K, temperature=TEMPERATURE)
        except Exception as e:
            gen_text = ""

        extracted = extract_gsm8k_answer(gen_text)
        is_correct = (extracted is not None and abs(extracted - target) < 1e-3)
        if is_correct:
            vanilla_correct += 1
        vanilla_total += 1

        if (idx + 1) % 10 == 0 or idx == len(gsm8k) - 1:
            acc = vanilla_correct / vanilla_total if vanilla_total > 0 else 0
            elapsed = time.time() - phase3_start
            eta = elapsed / (idx + 1) * (len(gsm8k) - idx - 1) if idx > 0 else 0
            print(f"    [{idx+1}/{len(gsm8k)}] acc={acc:.3f} ({vanilla_correct}/{vanilla_total}) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            report_progress("phase3_vanilla", idx + 1, len(gsm8k), {
                "correct": vanilla_correct, "total": vanilla_total, "accuracy": acc,
            })

        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    phase3_elapsed = time.time() - phase3_start
    vanilla_accuracy = vanilla_correct / vanilla_total if vanilla_total > 0 else 0
    print(f"\n  Phase 3 complete: {vanilla_correct}/{vanilla_total} = {vanilla_accuracy:.3f} "
          f"({phase3_elapsed:.0f}s)")

    # =================================================================
    # Phase 4: Compute Analysis
    # =================================================================
    print(f"\n{'='*70}")
    print(f"Phase 4: Signal Analysis")
    print(f"{'='*70}")

    report_progress("analysis", 0, 1)
    signal_analysis = compute_signal_analysis(all_step_data)

    print(f"  Monotonicity score: {signal_analysis['monotonicity_score']:.3f}")
    print(f"  Spearman rho (global): {signal_analysis['spearman_global']['rho']:.4f} "
          f"(p={signal_analysis['spearman_global']['pval']:.4f})")
    print(f"  Spearman rho (mean per-seq): {signal_analysis['spearman_mean_rho']:.4f}")
    print(f"  Peak drop at step {signal_analysis['peak_drop_step']} "
          f"(mask_ratio={signal_analysis['peak_drop_mask_ratio']:.3f})")

    # Show mean SSL trajectory (sampled)
    traj = signal_analysis['ssl_trajectory_sampled']
    print(f"  SSL loss trajectory (sampled every 4 steps, first 8):")
    for i, v in enumerate(traj[:8]):
        step_num = i * 4
        print(f"    step {step_num:3d}: {v:.4f}" if v is not None else f"    step {step_num:3d}: NaN")

    # =================================================================
    # Phase 5: Summary & Results
    # =================================================================
    total_elapsed = time.time() - overall_start

    accuracy_drop_from_reset = dal_accuracy - reset_accuracy

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<25} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Vanilla':<25} {vanilla_accuracy:>10.3f} {vanilla_correct:>10}/{vanilla_total:<5}")
    print(f"  {'DaL-Linear (normal)':<25} {dal_accuracy:>10.3f} {dal_correct:>10}/{dal_total:<5}")
    print(f"  {'DaL-Linear (mid-reset)':<25} {reset_accuracy:>10.3f} {reset_correct:>10}/{reset_total:<5}")
    print(f"\n  Accuracy drop from mid-reset (H9): {accuracy_drop_from_reset:+.3f}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Pass criteria: Spearman rho < -0.5 (negative correlation: as more revealed, loss drops)
    spearman_pass = (not np.isnan(signal_analysis['spearman_global']['rho']) and
                     signal_analysis['spearman_global']['rho'] < -0.5)

    # Build results JSON
    full_results = {
        "task_id": TASK_ID,
        "experiment": "M3b: Progressive Signal Enrichment Analysis",
        "model": "LLaDA-8B-Instruct",
        "mode": "PILOT",
        "variant": "linear",
        "checkpoint": "dal_linear_ckpt_step5000.pt",
        "config": {
            "num_sequences": NUM_SEQUENCES,
            "gen_steps": GEN_STEPS,
            "gen_len": GEN_LEN_GSM8K,
            "insertion_layer": INSERTION_LAYER,
            "d_model": D_MODEL,
            "ttt_lr": float(ttt_layer.lr.item()),
            "gate": float(ttt_layer.gate.item()),
            "temperature": TEMPERATURE,
        },
        "signal_analysis": signal_analysis,
        "accuracy_results": {
            "vanilla": {
                "accuracy": float(vanilla_accuracy),
                "correct": int(vanilla_correct),
                "total": int(vanilla_total),
                "time_s": float(phase3_elapsed),
            },
            "dal_linear_normal": {
                "accuracy": float(dal_accuracy),
                "correct": int(dal_correct),
                "total": int(dal_total),
                "time_s": float(phase1_elapsed),
            },
            "dal_linear_mid_reset": {
                "accuracy": float(reset_accuracy),
                "correct": int(reset_correct),
                "total": int(reset_total),
                "time_s": float(phase2_elapsed),
                "reset_step": GEN_STEPS // 2,
            },
        },
        "h9_mid_reset": {
            "accuracy_drop": float(accuracy_drop_from_reset),
            "normal_accuracy": float(dal_accuracy),
            "reset_accuracy": float(reset_accuracy),
            "hypothesis": "Fast weight memory is important: reset should cause accuracy drop",
            "finding": (f"Accuracy drop from mid-reset: {accuracy_drop_from_reset:+.3f}. "
                       f"{'Supports H9' if accuracy_drop_from_reset > 0.01 else 'Does not clearly support H9'}"),
        },
        "h3_progressive_signal": {
            "spearman_global_rho": signal_analysis['spearman_global']['rho'],
            "spearman_global_pval": signal_analysis['spearman_global']['pval'],
            "spearman_mean_per_seq": signal_analysis['spearman_mean_rho'],
            "monotonicity_score": signal_analysis['monotonicity_score'],
            "peak_drop_mask_ratio": signal_analysis['peak_drop_mask_ratio'],
            "hypothesis": "SSL loss monotonically decreases as more tokens are revealed (signal enrichment)",
            "pass_criteria": "Spearman rho < -0.5",
            "pass": spearman_pass,
        },
        "pass_criteria": {
            "spearman_rho_pass": spearman_pass,
            "trajectories_recorded": len(all_step_data) >= NUM_SEQUENCES * 0.8,
            "overall": "PASS" if spearman_pass else "PARTIAL",
        },
        "timing": {
            "phase1_ssl_tracking_s": float(phase1_elapsed),
            "phase2_mid_reset_s": float(phase2_elapsed),
            "phase3_vanilla_s": float(phase3_elapsed),
            "total_elapsed_s": float(total_elapsed),
            "total_elapsed_min": float(total_elapsed / 60),
        },
        "gpu_info": {
            "device": str(device),
            "gpu_name": gpu_name,
        },
        "timestamp": datetime.now().isoformat(),
        # Save first 5 sequences' full step data for detailed plotting
        "detailed_trajectories": [
            {
                "seq_idx": i,
                "steps": all_step_data[i],
            }
            for i in range(min(5, len(all_step_data)))
        ],
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
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return str(obj)

    results_path = Path(FULL_RESULTS_DIR) / "m3b_progressive_signal.json"
    results_path.write_text(json.dumps(full_results, indent=2, default=json_default))
    print(f"\nResults saved: {results_path}")

    # DONE marker
    status = "PASS" if spearman_pass else "PARTIAL"
    summary = (
        f"{status}: Progressive signal analysis on {NUM_SEQUENCES} GSM8K sequences. "
        f"Spearman rho={signal_analysis['spearman_global']['rho']:.4f} "
        f"(pass: rho<-0.5 = {spearman_pass}). "
        f"Monotonicity={signal_analysis['monotonicity_score']:.3f}. "
        f"Accuracy: vanilla={vanilla_accuracy:.3f}, "
        f"dal_normal={dal_accuracy:.3f}, "
        f"dal_mid_reset={reset_accuracy:.3f} "
        f"(H9 drop={accuracy_drop_from_reset:+.3f})."
    )
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
