#!/usr/bin/env python3
"""
C3: Qualitative Analysis & Visualization (PILOT mode)

For 10 random GSM8K samples:
  (a) Inspect generated text from vanilla vs DaL-MLP (best available ckpt)
  (b) Visualize TTT fast weight evolution across denoising steps (PCA projection)
  (c) Visualize per-position precision weights
  (d) Identify failure modes

For 16 sequences (pilot): compute pairwise fast weight cosine similarity (H9).

Available models:
  - Vanilla LLaDA-8B-Instruct (no memory)
  - DaL-MLP (dal_mlp_ckpt_step2500.pt — best available)
  - DaL-Linear (dal_linear_ckpt_step5000.pt)

Output:
  - DONE marker: exp/results/qualitative_analysis_DONE
  - Progress: exp/results/qualitative_analysis_PROGRESS.json
  - Full results: exp/results/full/c3_qualitative.json
  - Figures: exp/results/full/figures/
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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# === Paths ===
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
FULL_RESULTS_DIR = f"{PROJECT_DIR}/exp/results/full"
FIGURES_DIR = f"{FULL_RESULTS_DIR}/figures"
CODE_DIR = f"{PROJECT_DIR}/exp/code/dal"
SHARED_DIR = f"{REMOTE_BASE}/shared"

TASK_ID = "qualitative_analysis"
SEED = 42

# === Model Config (LLaDA-8B-Instruct) ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
D_MODEL = 4096
D_TTT = 512
VOCAB_SIZE = 126464
N_LAYERS = 32
INSERTION_LAYER = 16

# === Eval Config (PILOT) ===
PILOT_SAMPLES = 16  # For cosine similarity matrix
QUAL_SAMPLES = 10   # For qualitative inspection
GEN_STEPS = 128
GEN_LEN_GSM8K = 512
TEMPERATURE = 0.0

# === Checkpoint paths ===
DAL_MLP_CKPT = f"{FULL_RESULTS_DIR}/dal_mlp_ckpt_step2500.pt"
DAL_LINEAR_CKPT = f"{FULL_RESULTS_DIR}/dal_linear_ckpt_step5000.pt"

sys.path.insert(0, CODE_DIR)
from ttt_layer import TTTLayer, build_ttt_layer
from dal_wrapper import MetaStateGRU, create_masked_input


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
# Generation with TTT + Fast Weight Tracking
# ==============================================================================

def generate_vanilla(model, tokenizer, prompt_text, device, gen_len, temperature=0.0):
    """Standard LLaDA generation."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, GEN_STEPS + 1, device=device)

    per_step_entropy = []

    with torch.no_grad():
        for i in range(GEN_STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = model(x).logits

            # Compute per-position entropy for masked positions
            mask_logits = logits[mask_index]
            probs_for_entropy = F.softmax(mask_logits.float(), dim=-1)
            entropy = -(probs_for_entropy * (probs_for_entropy + 1e-10).log()).sum(dim=-1)
            per_step_entropy.append({
                "step": i,
                "mask_ratio": mask_index.float().mean().item(),
                "mean_entropy": entropy.mean().item(),
                "max_entropy": entropy.max().item(),
                "min_entropy": entropy.min().item(),
            })

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
    return text, per_step_entropy


def generate_with_ttt_tracking(model, ttt_layer, ssl_head, layer_norm, tokenizer,
                                prompt_text, device, gen_len, insertion_layer,
                                temperature=0.0, track_weights=True):
    """
    LLaDA generation with TTT injection + tracking of:
      - Fast weight snapshots at each step (for PCA / cosine similarity)
      - SSL loss at each step
      - Per-position precision weights
      - Per-step entropy
    """
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
        raise ValueError("Cannot find blocks in LLaDA transformer")

    target_layer = layers[insertion_layer]
    ttt_layer.reset_fast_weights(batch_size=1)

    # Tracking data
    weight_snapshots = []  # Flattened fast weight vectors at each step
    ssl_losses = []
    precision_weights_log = []
    per_step_entropy = []
    step_counter = [0]

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()
        current_step = step_counter[0]

        if n_revealed > 5 and n_masked > 3:
            h_normed = layer_norm(hidden.detach())
            gate = ttt_layer.gate.detach()

            # Track fast weight state before update
            if track_weights and current_step % 4 == 0:  # Every 4th step to save memory
                with torch.no_grad():
                    fw_params = ttt_layer.fast_weight.get_params_for_grad()
                    flat = torch.cat([p.detach().flatten() for p in fw_params])
                    # Subsample to keep manageable (take every 100th element)
                    subsample = flat[::100].cpu().float().numpy().tolist()
                    weight_snapshots.append({
                        "step": current_step,
                        "mask_ratio": mask_index.float().mean().item(),
                        "weight_norm": float(flat.norm().item()),
                        "weight_sample": subsample[:500],  # Cap at 500 elements
                    })

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
                    target_ids.view(-1),
                    reduction="none"
                ).view(B, S)

                # Compute precision weights (prediction variance proxy)
                with torch.no_grad():
                    probs_ssl = F.softmax(ssl_logits_grad.float(), dim=-1)
                    variance = (probs_ssl * (1 - probs_ssl)).sum(dim=-1)  # (B, S)
                    precision = 1.0 / (variance + 1e-6)
                    precision = precision * revealed_mask
                    precision_sum = precision.sum().clamp(min=1.0)
                    precision_normalized = precision / precision_sum * n_rev

                # Log precision weights for revealed positions
                if current_step % 8 == 0:
                    rev_indices = revealed_mask[0].nonzero(as_tuple=True)[0]
                    if len(rev_indices) > 0:
                        prec_vals = precision_normalized[0, rev_indices].cpu().float().tolist()
                        precision_weights_log.append({
                            "step": current_step,
                            "mask_ratio": mask_index.float().mean().item(),
                            "positions": rev_indices.cpu().tolist()[:50],
                            "precision_weights": prec_vals[:50],
                            "mean_precision": float(np.mean(prec_vals)),
                            "max_precision": float(np.max(prec_vals)) if prec_vals else 0,
                            "std_precision": float(np.std(prec_vals)) if len(prec_vals) > 1 else 0,
                        })

                masked_loss = per_token_loss * revealed_mask
                ssl_loss = masked_loss.sum() / n_rev

                ssl_losses.append({
                    "step": current_step,
                    "ssl_loss": float(ssl_loss.item()),
                    "mask_ratio": mask_index.float().mean().item(),
                    "n_revealed": int(n_revealed),
                })

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

        return output

    hook_handle = target_layer.register_forward_hook(capture_and_inject)

    try:
        with torch.no_grad():
            for i in range(GEN_STEPS):
                step_counter[0] = i
                mask_index = (x == MASK_TOKEN_ID)
                if not mask_index.any():
                    break
                logits = model(x).logits
                mask_logits = logits[mask_index]

                # Compute per-position entropy
                probs_for_entropy = F.softmax(mask_logits.float(), dim=-1)
                entropy = -(probs_for_entropy * (probs_for_entropy + 1e-10).log()).sum(dim=-1)
                per_step_entropy.append({
                    "step": i,
                    "mask_ratio": mask_index.float().mean().item(),
                    "mean_entropy": entropy.mean().item(),
                    "max_entropy": entropy.max().item(),
                    "min_entropy": entropy.min().item(),
                })

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

    tracking_data = {
        "weight_snapshots": weight_snapshots,
        "ssl_losses": ssl_losses,
        "precision_weights": precision_weights_log,
        "per_step_entropy": per_step_entropy,
    }

    return text, tracking_data


def collect_final_fast_weights(model, ttt_layer, ssl_head, layer_norm, tokenizer,
                                prompt_text, device, gen_len, insertion_layer,
                                temperature=0.0):
    """
    Run generation and return the final fast weight vector (for cosine similarity).
    Lightweight version — no tracking, just returns final weight state.
    """
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

    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()

        if n_revealed > 5 and n_masked > 3:
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
                    target_ids.view(-1),
                    reduction="none"
                ).view(B, S)
                ssl_loss = (per_token_loss * revealed_mask).sum() / n_rev
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
        return output

    hook_handle = target_layer.register_forward_hook(inject_hook)

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

    # Extract final fast weight vector
    with torch.no_grad():
        fw_params = ttt_layer.fast_weight.get_params_for_grad()
        flat = torch.cat([p.detach().flatten() for p in fw_params])

    text = decode_output(tokenizer, x, prompt_len)
    return flat, text


# ==============================================================================
# Load checkpoints
# ==============================================================================

def load_ttt_checkpoint(ckpt_path, variant, device):
    """
    Load TTT layer + SSL head + layer norm from checkpoint.
    Handles two checkpoint formats:
      1. Unified: ssl_head and layer_norm inside ttt_layer state_dict
      2. Split: separate ssl_head_state_dict and layer_norm_state_dict keys
    """
    print(f"  Loading DaL-{variant} checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    ttt_layer = build_ttt_layer(
        variant=variant,
        d_model=D_MODEL,
        d_ttt=D_TTT,
        vocab_size=VOCAB_SIZE,
    )

    ttt_sd = ckpt["ttt_layer_state_dict"]

    # Check if ssl_head/layer_norm are inside ttt_layer state_dict (unified format)
    unified = any(k.startswith("ssl_head.") for k in ttt_sd)

    if unified:
        # Load with strict=False to skip runtime buffers (W1_fast etc.)
        missing, unexpected = ttt_layer.load_state_dict(ttt_sd, strict=False)
        if unexpected:
            print(f"    Unexpected keys (ignored): {unexpected[:5]}")
        ttt_layer = ttt_layer.to(device).to(torch.bfloat16)
        ttt_layer.eval()
        # Extract ssl_head and layer_norm from the ttt_layer
        ssl_head = ttt_layer.ssl_head
        layer_norm = ttt_layer.layer_norm
    else:
        # Split format (older checkpoints like dal_linear)
        missing, unexpected = ttt_layer.load_state_dict(ttt_sd, strict=False)
        if unexpected:
            print(f"    Unexpected keys (ignored): {unexpected[:5]}")
        ttt_layer = ttt_layer.to(device).to(torch.bfloat16)
        ttt_layer.eval()
        ssl_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False).to(device).to(torch.bfloat16)
        ssl_head.load_state_dict(ckpt["ssl_head_state_dict"])
        layer_norm = nn.LayerNorm(D_MODEL).to(device).to(torch.bfloat16)
        layer_norm.load_state_dict(ckpt["layer_norm_state_dict"])

    step = ckpt.get("step", "?")
    # Gate and lr are stored as log/logit values
    gate_val = torch.sigmoid(ttt_layer.gate_logit).item()
    lr_val = torch.exp(ttt_layer.log_lr).item()
    print(f"  DaL-{variant} loaded: step={step}, gate={gate_val:.4f}, ttt_lr={lr_val:.4f}")

    return ttt_layer, ssl_head, layer_norm, ckpt


def load_ttt_mlp_checkpoint(ckpt_path, device):
    return load_ttt_checkpoint(ckpt_path, "mlp", device)


def load_ttt_linear_checkpoint(ckpt_path, device):
    return load_ttt_checkpoint(ckpt_path, "linear", device)


# ==============================================================================
# Visualization Helpers
# ==============================================================================

def compute_pca_trajectory(weight_snapshots):
    """
    Compute PCA projection of fast weight evolution across denoising steps.
    Returns PC1, PC2 coordinates and explained variance.
    """
    if len(weight_snapshots) < 3:
        return None

    # Build matrix: each row is a weight sample at a step
    max_len = min(len(s["weight_sample"]) for s in weight_snapshots)
    if max_len < 10:
        return None

    matrix = np.array([s["weight_sample"][:max_len] for s in weight_snapshots])

    # Center
    mean = matrix.mean(axis=0)
    centered = matrix - mean

    # SVD for PCA
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained_var = (S ** 2) / (S ** 2).sum()
        pc1 = centered @ Vt[0]
        pc2 = centered @ Vt[1] if len(Vt) > 1 else np.zeros_like(pc1)

        return {
            "steps": [s["step"] for s in weight_snapshots],
            "mask_ratios": [s["mask_ratio"] for s in weight_snapshots],
            "pc1": pc1.tolist(),
            "pc2": pc2.tolist(),
            "explained_variance_ratio": explained_var[:5].tolist(),
            "weight_norms": [s["weight_norm"] for s in weight_snapshots],
        }
    except Exception as e:
        print(f"  PCA failed: {e}")
        return None


def compute_cosine_similarity_matrix(weight_vectors):
    """
    Compute pairwise cosine similarity between final fast weight vectors.
    weight_vectors: list of (flat_tensor, text) tuples
    """
    n = len(weight_vectors)
    if n < 2:
        return None

    # Stack into matrix
    vecs = torch.stack([wv[0].float() for wv in weight_vectors])
    # Normalize
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vecs_normed = vecs / norms
    # Cosine similarity matrix
    sim_matrix = (vecs_normed @ vecs_normed.T).cpu().numpy()

    return {
        "similarity_matrix": sim_matrix.tolist(),
        "n_sequences": n,
        "mean_similarity": float(sim_matrix[np.triu_indices(n, k=1)].mean()),
        "std_similarity": float(sim_matrix[np.triu_indices(n, k=1)].std()),
        "min_similarity": float(sim_matrix[np.triu_indices(n, k=1)].min()),
        "max_similarity": float(sim_matrix[np.triu_indices(n, k=1)].max()),
    }


def generate_figures(all_data, figures_dir):
    """Generate matplotlib figures for paper."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("  matplotlib not available, skipping figure generation")
        return []

    os.makedirs(figures_dir, exist_ok=True)
    generated = []

    # --- Figure 1: Fast Weight PCA Trajectory ---
    for method_name, method_data in all_data.get("qualitative_samples", {}).items():
        if method_name == "vanilla":
            continue
        for sample_idx, sample in enumerate(method_data[:3]):  # First 3 samples
            pca = sample.get("pca_trajectory")
            if pca is None:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            pc1 = np.array(pca["pc1"])
            pc2 = np.array(pca["pc2"])
            steps = np.array(pca["steps"])
            mask_ratios = np.array(pca["mask_ratios"])

            scatter = ax.scatter(pc1, pc2, c=mask_ratios, cmap='viridis',
                               s=30, alpha=0.8, edgecolors='k', linewidths=0.3)
            # Draw trajectory line
            ax.plot(pc1, pc2, 'k-', alpha=0.3, linewidth=0.5)
            # Mark start and end
            ax.scatter([pc1[0]], [pc2[0]], c='red', s=100, marker='s',
                      zorder=5, label='Start (high mask)')
            ax.scatter([pc1[-1]], [pc2[-1]], c='blue', s=100, marker='*',
                      zorder=5, label='End (low mask)')

            plt.colorbar(scatter, label='Mask Ratio')
            ax.set_xlabel(f"PC1 ({pca['explained_variance_ratio'][0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pca['explained_variance_ratio'][1]*100:.1f}% var)" if len(pca['explained_variance_ratio']) > 1 else "PC2")
            ax.set_title(f"Fast Weight Evolution — {method_name} (Sample {sample_idx})")
            ax.legend(loc='best', fontsize=8)

            fname = f"pca_trajectory_{method_name}_sample{sample_idx}.png"
            fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(fname)

    # --- Figure 2: SSL Loss Trajectory ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for method_name, method_data in all_data.get("qualitative_samples", {}).items():
        if method_name == "vanilla":
            continue
        for sample_idx, sample in enumerate(method_data[:1]):  # First sample only
            ssl_data = sample.get("ssl_losses", [])
            if ssl_data:
                steps_ssl = [d["step"] for d in ssl_data]
                losses = [d["ssl_loss"] for d in ssl_data]
                ax.plot(steps_ssl, losses, label=f"{method_name} (sample {sample_idx})",
                       alpha=0.7)

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Self-Supervised Loss")
    ax.set_title("SSL Loss Trajectory During Denoising")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fname = "ssl_loss_trajectory.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(fname)

    # --- Figure 3: Precision Weight Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (method_name, method_data) in enumerate(
        [(k, v) for k, v in all_data.get("qualitative_samples", {}).items() if k != "vanilla"]
    ):
        if ax_idx >= 2:
            break
        ax = axes[ax_idx]
        sample = method_data[0] if method_data else None
        if sample and sample.get("precision_weights"):
            pw_data = sample["precision_weights"]
            # Plot precision weights at different mask ratios
            for pw in pw_data:
                mr = pw["mask_ratio"]
                weights = pw["precision_weights"]
                positions = pw["positions"]
                if len(positions) > 0:
                    ax.scatter(positions, weights, s=5, alpha=0.5,
                             label=f"mask={mr:.2f}" if mr > 0.3 else None)
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Precision Weight")
            ax.set_title(f"Precision Weights — {method_name}")
            ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fname = "precision_weights.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(fname)

    # --- Figure 4: Cosine Similarity Matrix ---
    sim_data = all_data.get("cosine_similarity")
    if sim_data and sim_data.get("similarity_matrix"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        sim_matrix = np.array(sim_data["similarity_matrix"])
        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Cosine Similarity')
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel("Sequence Index")
        ax.set_title(f"Cross-Sequence Fast Weight Cosine Similarity (H9)\n"
                     f"Mean={sim_data['mean_similarity']:.3f}, "
                     f"Std={sim_data['std_similarity']:.3f}")
        fname = "cosine_similarity_matrix.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated.append(fname)

    # --- Figure 5: Entropy Comparison (Vanilla vs DaL) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for method_name, method_data in all_data.get("qualitative_samples", {}).items():
        sample = method_data[0] if method_data else None
        if sample:
            entropy_key = "per_step_entropy" if "per_step_entropy" in sample else "entropy_data"
            entropy_data = sample.get(entropy_key, [])
            if entropy_data:
                steps_e = [d["step"] for d in entropy_data]
                mean_e = [d["mean_entropy"] for d in entropy_data]
                ax.plot(steps_e, mean_e, label=method_name, alpha=0.7)

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Mean Entropy (masked positions)")
    ax.set_title("Per-Step Entropy: Vanilla vs DaL Methods")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fname = "entropy_comparison.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(fname)

    # --- Figure 6: Weight Norm Evolution ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for method_name, method_data in all_data.get("qualitative_samples", {}).items():
        if method_name == "vanilla":
            continue
        for sample_idx, sample in enumerate(method_data[:3]):
            ws = sample.get("weight_snapshots", [])
            if ws:
                steps_w = [s["step"] for s in ws]
                norms = [s["weight_norm"] for s in ws]
                ax.plot(steps_w, norms, alpha=0.5,
                       label=f"{method_name} s{sample_idx}" if sample_idx < 2 else None)

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Fast Weight Frobenius Norm")
    ax.set_title("Fast Weight Norm Evolution During Denoising")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fname = "weight_norm_evolution.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(fname)

    print(f"  Generated {len(generated)} figures in {figures_dir}")
    return generated


# ==============================================================================
# Failure Mode Analysis
# ==============================================================================

def analyze_failure_modes(vanilla_results, dal_results):
    """Categorize failure modes by comparing vanilla and DaL outputs."""
    categories = {
        "dal_fixes_vanilla_error": [],
        "dal_introduces_new_error": [],
        "both_correct": [],
        "both_wrong_same_error": [],
        "both_wrong_different_error": [],
    }

    for v, d in zip(vanilla_results, dal_results):
        v_correct = v.get("is_correct", False)
        d_correct = d.get("is_correct", False)
        idx = v.get("idx", "?")

        if v_correct and d_correct:
            categories["both_correct"].append(idx)
        elif not v_correct and d_correct:
            categories["dal_fixes_vanilla_error"].append(idx)
        elif v_correct and not d_correct:
            categories["dal_introduces_new_error"].append(idx)
        else:
            # Both wrong - check if same wrong answer
            v_ans = v.get("extracted")
            d_ans = d.get("extracted")
            if v_ans is not None and d_ans is not None and abs(v_ans - d_ans) < 1e-3:
                categories["both_wrong_same_error"].append(idx)
            else:
                categories["both_wrong_different_error"].append(idx)

    return {k: {"count": len(v), "indices": v} for k, v in categories.items()}


# ==============================================================================
# Main
# ==============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    print("=" * 70)
    print("C3: Qualitative Analysis & Visualization (PILOT)")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {physical_gpu} (CUDA_VISIBLE_DEVICES)")
    print(f"Qualitative samples: {QUAL_SAMPLES}")
    print(f"Cosine similarity samples: {PILOT_SAMPLES}")
    print()

    # Write PID
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    os.makedirs(FIGURES_DIR, exist_ok=True)

    report_progress("loading_model", 0, 5)

    # Load backbone model
    print("Loading LLaDA-8B-Instruct...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e6 if torch.cuda.is_available() else 0
    print(f"  Loaded. GPU: {gpu_name}, VRAM: {vram:.0f}MB")

    # Load GSM8K
    from datasets import load_from_disk
    gsm8k = load_from_disk(f"{SHARED_DIR}/datasets/gsm8k")

    # Select samples deterministically
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(gsm8k))[:max(QUAL_SAMPLES, PILOT_SAMPLES)]
    qual_indices = indices[:QUAL_SAMPLES]
    sim_indices = indices[:PILOT_SAMPLES]

    all_data = {"qualitative_samples": {}, "cosine_similarity": None, "failure_analysis": None}

    # ====================================================================
    # Phase 1: Vanilla qualitative samples
    # ====================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Vanilla Baseline — Qualitative Samples")
    print("=" * 70)
    report_progress("vanilla_qual", 0, QUAL_SAMPLES)

    vanilla_samples = []
    for si, idx in enumerate(qual_indices):
        item = gsm8k[int(idx)]
        question = item['question']
        target = extract_gsm8k_target(item['answer'])

        prompt = (
            "Solve the math problem step by step. "
            "End your answer with #### followed by the final numerical answer.\n\n"
            f"Problem: {question}\n\nSolution:"
        )

        try:
            gen_text, entropy_data = generate_vanilla(
                model, tokenizer, prompt, device, GEN_LEN_GSM8K, TEMPERATURE
            )
        except Exception as e:
            gen_text, entropy_data = "", []
            print(f"  [{si}] Vanilla error: {e}")

        extracted = extract_gsm8k_answer(gen_text)
        is_correct = (extracted is not None and target is not None
                      and abs(extracted - target) < 1e-3)

        vanilla_samples.append({
            "idx": int(idx),
            "question": question[:200],
            "target": float(target) if target else None,
            "extracted": float(extracted) if extracted else None,
            "is_correct": bool(is_correct),
            "generated_text": gen_text[:1000],
            "per_step_entropy": entropy_data[-20:],  # Last 20 steps to save space
        })

        print(f"  [{si}] idx={idx} correct={is_correct} target={target} extracted={extracted}")
        report_progress("vanilla_qual", si + 1, QUAL_SAMPLES,
                        {"correct_so_far": sum(1 for s in vanilla_samples if s["is_correct"])})

        gc.collect()
        torch.cuda.empty_cache()

    all_data["qualitative_samples"]["vanilla"] = vanilla_samples

    # ====================================================================
    # Phase 2: DaL-MLP qualitative samples + tracking
    # ====================================================================
    print("\n" + "=" * 70)
    print("Phase 2: DaL-MLP — Qualitative Samples with Tracking")
    print("=" * 70)
    report_progress("loading_dal_mlp", 0, 1)

    dal_mlp_ok = False
    ttt_layer_mlp = ssl_head_mlp = ln_mlp = None
    try:
        ttt_layer_mlp, ssl_head_mlp, ln_mlp, _ = load_ttt_mlp_checkpoint(DAL_MLP_CKPT, device)
        dal_mlp_ok = True
    except Exception as e:
        print(f"  Failed to load DaL-MLP: {e}")
        traceback.print_exc()
        # Try alternative checkpoint
        for alt_step in [2800, 2500, 2000, 1500, 1000]:
            alt_path = f"{FULL_RESULTS_DIR}/dal_mlp_ckpt_step{alt_step}.pt"
            try:
                print(f"  Trying alternative: {alt_path}")
                ttt_layer_mlp, ssl_head_mlp, ln_mlp, _ = load_ttt_mlp_checkpoint(alt_path, device)
                dal_mlp_ok = True
                break
            except:
                continue

    dal_mlp_samples = []
    if dal_mlp_ok:
        for si, idx in enumerate(qual_indices):
            item = gsm8k[int(idx)]
            question = item['question']
            target = extract_gsm8k_target(item['answer'])

            prompt = (
                "Solve the math problem step by step. "
                "End your answer with #### followed by the final numerical answer.\n\n"
                f"Problem: {question}\n\nSolution:"
            )

            try:
                gen_text, tracking = generate_with_ttt_tracking(
                    model, ttt_layer_mlp, ssl_head_mlp, ln_mlp, tokenizer,
                    prompt, device, GEN_LEN_GSM8K, INSERTION_LAYER, TEMPERATURE,
                    track_weights=True
                )
            except Exception as e:
                gen_text, tracking = "", {"weight_snapshots": [], "ssl_losses": [],
                                          "precision_weights": [], "per_step_entropy": []}
                print(f"  [{si}] DaL-MLP error: {e}")
                traceback.print_exc()

            extracted = extract_gsm8k_answer(gen_text)
            is_correct = (extracted is not None and target is not None
                          and abs(extracted - target) < 1e-3)

            pca_traj = compute_pca_trajectory(tracking["weight_snapshots"])

            dal_mlp_samples.append({
                "idx": int(idx),
                "question": question[:200],
                "target": float(target) if target else None,
                "extracted": float(extracted) if extracted else None,
                "is_correct": bool(is_correct),
                "generated_text": gen_text[:1000],
                "pca_trajectory": pca_traj,
                "ssl_losses": tracking["ssl_losses"][-30:],  # Last 30 to save space
                "weight_snapshots": [
                    {"step": s["step"], "mask_ratio": s["mask_ratio"],
                     "weight_norm": s["weight_norm"]}
                    for s in tracking["weight_snapshots"]
                ],
                "precision_weights": tracking["precision_weights"][-10:],
                "per_step_entropy": tracking["per_step_entropy"][-20:],
            })

            print(f"  [{si}] idx={idx} correct={is_correct} target={target} extracted={extracted}")
            report_progress("dal_mlp_qual", si + 1, QUAL_SAMPLES,
                            {"correct_so_far": sum(1 for s in dal_mlp_samples if s["is_correct"])})

            gc.collect()
            torch.cuda.empty_cache()

        all_data["qualitative_samples"]["dal_mlp"] = dal_mlp_samples

        # ====================================================================
        # Phase 3: DaL-MLP Cosine Similarity Matrix (H9)
        # ====================================================================
        print("\n" + "=" * 70)
        print(f"Phase 3: Cross-Sequence Fast Weight Cosine Similarity ({PILOT_SAMPLES} seqs)")
        print("=" * 70)
        report_progress("cosine_similarity", 0, PILOT_SAMPLES)

        weight_vectors = []
        for si, idx in enumerate(sim_indices):
            item = gsm8k[int(idx)]
            question = item['question']

            prompt = (
                "Solve the math problem step by step. "
                "End your answer with #### followed by the final numerical answer.\n\n"
                f"Problem: {question}\n\nSolution:"
            )

            try:
                fw, gen_text = collect_final_fast_weights(
                    model, ttt_layer_mlp, ssl_head_mlp, ln_mlp, tokenizer,
                    prompt, device, GEN_LEN_GSM8K, INSERTION_LAYER, TEMPERATURE
                )
                weight_vectors.append((fw, gen_text[:200]))
            except Exception as e:
                print(f"  [{si}] Cosine sim error: {e}")

            if si % 4 == 0:
                report_progress("cosine_similarity", si + 1, PILOT_SAMPLES,
                                {"collected": len(weight_vectors)})
                gc.collect()
                torch.cuda.empty_cache()

            print(f"  [{si}/{PILOT_SAMPLES}] Collected fast weight vector")

        if len(weight_vectors) >= 2:
            sim_result = compute_cosine_similarity_matrix(weight_vectors)
            all_data["cosine_similarity"] = sim_result
            if sim_result:
                print(f"\n  Cosine similarity stats:")
                print(f"    Mean: {sim_result['mean_similarity']:.4f}")
                print(f"    Std:  {sim_result['std_similarity']:.4f}")
                print(f"    Min:  {sim_result['min_similarity']:.4f}")
                print(f"    Max:  {sim_result['max_similarity']:.4f}")
        else:
            print("  Not enough vectors for cosine similarity")

        # Clean up MLP
        del ttt_layer_mlp, ssl_head_mlp, ln_mlp
        gc.collect()
        torch.cuda.empty_cache()

    # ====================================================================
    # Phase 4: Failure Mode Analysis
    # ====================================================================
    print("\n" + "=" * 70)
    print("Phase 4: Failure Mode Analysis")
    print("=" * 70)

    if dal_mlp_samples:
        failure_analysis = analyze_failure_modes(vanilla_samples, dal_mlp_samples)
        all_data["failure_analysis"] = failure_analysis
        print("\n  Failure mode breakdown:")
        for cat, data in failure_analysis.items():
            print(f"    {cat}: {data['count']} samples {data['indices']}")
    else:
        print("  Skipping (no DaL-MLP results)")

    # ====================================================================
    # Phase 5: Generate Figures
    # ====================================================================
    print("\n" + "=" * 70)
    print("Phase 5: Generating Figures")
    print("=" * 70)
    report_progress("generating_figures", 0, 1)

    generated_figures = generate_figures(all_data, FIGURES_DIR)

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("QUALITATIVE ANALYSIS SUMMARY")
    print("=" * 70)

    vanilla_correct = sum(1 for s in vanilla_samples if s["is_correct"])
    dal_correct = sum(1 for s in dal_mlp_samples if s["is_correct"]) if dal_mlp_samples else 0

    print(f"\n  Vanilla: {vanilla_correct}/{QUAL_SAMPLES} correct")
    if dal_mlp_samples:
        print(f"  DaL-MLP: {dal_correct}/{QUAL_SAMPLES} correct")

    # Print comparison table
    print(f"\n  {'Idx':<6} {'Target':>10} {'Vanilla':>10} {'V OK':>5} {'DaL-MLP':>10} {'D OK':>5}")
    print("  " + "-" * 50)
    for i, (v, d) in enumerate(zip(vanilla_samples, dal_mlp_samples if dal_mlp_samples else vanilla_samples)):
        v_ans = f"{v['extracted']:.1f}" if v['extracted'] is not None else "None"
        d_ans = f"{d['extracted']:.1f}" if d['extracted'] is not None else "None"
        print(f"  {v['idx']:<6} {v['target']:>10.1f} {v_ans:>10} {'Y' if v['is_correct'] else 'N':>5} "
              f"{d_ans:>10} {'Y' if d['is_correct'] else 'N':>5}")

    # Build full results
    full_results = {
        "task_id": TASK_ID,
        "experiment": "C3: Qualitative Analysis & Visualization (PILOT)",
        "model": "LLaDA-8B-Instruct",
        "mode": "PILOT",
        "qual_samples": QUAL_SAMPLES,
        "sim_samples": PILOT_SAMPLES,
        "gen_steps": GEN_STEPS,
        "methods": {
            "vanilla": {
                "accuracy": vanilla_correct / QUAL_SAMPLES if QUAL_SAMPLES > 0 else 0,
                "correct": vanilla_correct,
                "total": QUAL_SAMPLES,
            },
        },
        "qualitative_samples": all_data["qualitative_samples"],
        "cosine_similarity": all_data["cosine_similarity"],
        "failure_analysis": all_data["failure_analysis"],
        "generated_figures": generated_figures,
        "gpu_info": {
            "device": str(device),
            "gpu_name": gpu_name,
            "vram_total_mb": float(vram),
        },
        "timestamp": datetime.now().isoformat(),
    }

    if dal_mlp_samples:
        full_results["methods"]["dal_mlp"] = {
            "accuracy": dal_correct / QUAL_SAMPLES if QUAL_SAMPLES > 0 else 0,
            "correct": dal_correct,
            "total": QUAL_SAMPLES,
            "checkpoint": os.path.basename(DAL_MLP_CKPT),
        }

    # Save results
    results_path = Path(FULL_RESULTS_DIR) / "c3_qualitative.json"
    results_path.write_text(json.dumps(full_results, indent=2, default=str))
    print(f"\nResults saved: {results_path}")

    status = "success"
    summary = (f"Qualitative analysis on {QUAL_SAMPLES} GSM8K samples. "
               f"Vanilla={vanilla_correct}/{QUAL_SAMPLES}, "
               f"DaL-MLP={dal_correct}/{QUAL_SAMPLES}. "
               f"Cosine sim matrix: {PILOT_SAMPLES} sequences. "
               f"Generated {len(generated_figures)} figures.")

    mark_done(status, summary)
    print(f"\nDone. Status: {status}")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        traceback.print_exc()
        mark_done("error", f"Fatal: {str(e)[:200]}")
        sys.exit(1)
