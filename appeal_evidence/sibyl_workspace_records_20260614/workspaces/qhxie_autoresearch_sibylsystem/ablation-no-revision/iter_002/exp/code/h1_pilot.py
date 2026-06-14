#!/usr/bin/env python3
"""
H1 Pilot: Absorption Prevalence in Sparse Autoencoders
Pilot on 100 sequences, layer 8, 8K dict.
Compute absorption score for all latents and report % with score >0.5.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
LAYER = 8
DICT_SIZE = 8192
N_SEQS = 100
SEQ_LEN = 128
THRESHOLD_PCT = 0.01  # activation threshold = 1% of max
COFIRER_TOPK = 5
VARIANCE_THRESHOLD = 0.80

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results")
SHARED_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/shared")
TASK_ID = "h1_pilot"

# ---------------------------------------------------------------------------
# PID + Progress helpers
# ---------------------------------------------------------------------------
def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss,
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
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ---------------------------------------------------------------------------
# Absorption metric
# ---------------------------------------------------------------------------
def compute_absorption_scores(sae, model, tokens, layer, device):
    """
    Compute absorption score for every latent in the SAE.

    For each latent f:
      1. Find activating tokens (feature_f > threshold)
      2. For each activating token, find top-5 co-firing latents
      3. Compute partial reconstruction using f + top-5 co-firers
      4. Check if co-firers explain >80% of reconstruction variance
      5. Absorption score = fraction of activating tokens where this holds

    Returns:
      absorption_scores: np.ndarray [d_sae]
      activation_counts: np.ndarray [d_sae]  (how many tokens activated)
    """
    sae.eval()
    model.eval()

    d_sae = sae.cfg.d_sae
    batch_size = tokens.shape[0]

    # Run model to get residual activations
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_pre")
        activations = cache[f"blocks.{layer}.hook_resid_pre"]  # [batch, pos, d_model]

    # Flatten to [N, d_model]
    N = batch_size * SEQ_LEN
    acts_flat = activations.reshape(N, -1).to(device)

    # Encode to SAE features [N, d_sae]
    with torch.no_grad():
        features = sae.encode(acts_flat)  # [N, d_sae]

    # Compute per-latent max activation for thresholding
    max_acts = features.max(dim=0).values  # [d_sae]
    thresholds = max_acts * THRESHOLD_PCT

    absorption_scores = np.zeros(d_sae, dtype=np.float32)
    activation_counts = np.zeros(d_sae, dtype=np.int32)

    # Process in chunks to avoid OOM with large d_sae
    chunk_size = 256
    W_dec = sae.W_dec  # [d_sae, d_model]

    for chunk_start in range(0, d_sae, chunk_size):
        chunk_end = min(chunk_start + chunk_size, d_sae)
        chunk_indices = torch.arange(chunk_start, chunk_end, device=device)

        for i, feat_idx in enumerate(chunk_indices):
            feat_idx_int = feat_idx.item()
            threshold = thresholds[feat_idx_int].item()

            # Find activating tokens
            activations_f = features[:, feat_idx_int]
            active_mask = activations_f > threshold
            active_indices = torch.where(active_mask)[0]
            n_active = active_indices.shape[0]
            activation_counts[feat_idx_int] = n_active

            if n_active == 0:
                absorption_scores[feat_idx_int] = 0.0
                continue

            # For each activating token, find top-5 co-firers
            active_features = features[active_indices]  # [n_active, d_sae]
            # Zero out the target feature so it doesn't appear in its own top-k
            active_features_copy = active_features.clone()
            active_features_copy[:, feat_idx_int] = -float('inf')

            topk_vals, topk_idx = torch.topk(active_features_copy, k=COFIRER_TOPK, dim=1)
            # topk_idx: [n_active, 5]

            # Compute partial reconstruction: W_dec[f] * act_f + sum_c W_dec[c] * act_c
            # Full reconstruction for reference
            act_active = acts_flat[active_indices]  # [n_active, d_model]
            var_x = act_active.var(dim=1, unbiased=False)  # [n_active]

            # Partial reconstruction
            partial = W_dec[feat_idx_int].unsqueeze(0) * activations_f[active_indices].unsqueeze(1)  # [n_active, d_model]
            for c in range(COFIRER_TOPK):
                cofirer_idx = topk_idx[:, c]  # [n_active]
                cofirer_act = active_features[torch.arange(n_active, device=device), cofirer_idx]  # [n_active]
                partial += W_dec[cofirer_idx] * cofirer_act.unsqueeze(1)

            var_residual = (act_active - partial).var(dim=1, unbiased=False)
            # Avoid division by zero
            var_x = torch.clamp(var_x, min=1e-8)
            var_explained = 1.0 - var_residual / var_x

            n_absorbed = (var_explained > VARIANCE_THRESHOLD).sum().item()
            absorption_scores[feat_idx_int] = n_absorbed / n_active

    return absorption_scores, activation_counts

# ---------------------------------------------------------------------------
# Random dictionary control
# ---------------------------------------------------------------------------
def compute_random_control(tokens, layer, device, d_sae=8192, d_model=768):
    """Compute absorption scores on a random Gaussian decoder (normalized per column)."""
    # Load or generate random decoder
    rand_path = SHARED_DIR / f"random_decoders_{d_sae}.pt"
    if rand_path.exists():
        W_dec = torch.load(rand_path, map_location=device, weights_only=True)
    else:
        W_dec = torch.randn(d_sae, d_model, device=device)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)

    # For random control, we don't have an encoder. Use random encoder weights.
    W_enc = torch.randn(d_model, d_sae, device=device)
    b_enc = torch.zeros(d_sae, device=device)
    b_dec = torch.zeros(d_model, device=device)

    # Simple encode: relu(x @ W_enc + b_enc)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_pre")
        activations = cache[f"blocks.{layer}.hook_resid_pre"]

    N = tokens.shape[0] * SEQ_LEN
    acts_flat = activations.reshape(N, -1).to(device)

    with torch.no_grad():
        features = torch.relu(acts_flat @ W_enc + b_enc)

    max_acts = features.max(dim=0).values
    thresholds = max_acts * THRESHOLD_PCT

    absorption_scores = np.zeros(d_sae, dtype=np.float32)
    activation_counts = np.zeros(d_sae, dtype=np.int32)

    chunk_size = 256
    for chunk_start in range(0, d_sae, chunk_size):
        chunk_end = min(chunk_start + chunk_size, d_sae)
        for feat_idx_int in range(chunk_start, chunk_end):
            threshold = thresholds[feat_idx_int].item()
            activations_f = features[:, feat_idx_int]
            active_mask = activations_f > threshold
            active_indices = torch.where(active_mask)[0]
            n_active = active_indices.shape[0]
            activation_counts[feat_idx_int] = n_active

            if n_active == 0:
                absorption_scores[feat_idx_int] = 0.0
                continue

            active_features = features[active_indices]
            active_features_copy = active_features.clone()
            active_features_copy[:, feat_idx_int] = -float('inf')

            topk_vals, topk_idx = torch.topk(active_features_copy, k=COFIRER_TOPK, dim=1)

            act_active = acts_flat[active_indices]
            var_x = act_active.var(dim=1, unbiased=False)

            partial = W_dec[feat_idx_int].unsqueeze(0) * activations_f[active_indices].unsqueeze(1)
            for c in range(COFIRER_TOPK):
                cofirer_idx = topk_idx[:, c]
                cofirer_act = active_features[torch.arange(n_active, device=device), cofirer_idx]
                partial += W_dec[cofirer_idx] * cofirer_act.unsqueeze(1)

            var_residual = (act_active - partial).var(dim=1, unbiased=False)
            var_x = torch.clamp(var_x, min=1e-8)
            var_explained = 1.0 - var_residual / var_x

            n_absorbed = (var_explained > VARIANCE_THRESHOLD).sum().item()
            absorption_scores[feat_idx_int] = n_absorbed / n_active

    return absorption_scores, activation_counts

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    write_pid()
    start_time = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[{TASK_ID}] Starting on device={DEVICE}, GPU={GPU_ID}")
    print(f"[{TASK_ID}] Layer={LAYER}, DictSize={DICT_SIZE}, NSeqs={N_SEQS}")

    report_progress(0, 3, step=0, total_steps=3, metric={"stage": "loading_model"})

    # Load model
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"[{TASK_ID}] Loading gpt2-small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"[{TASK_ID}] Model loaded.")

    # Load SAE
    print(f"[{TASK_ID}] Loading SAE for layer {LAYER}...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE,
    )
    print(f"[{TASK_ID}] SAE loaded: d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")

    # Load tokens
    print(f"[{TASK_ID}] Loading tokens...")
    tokens_path = SHARED_DIR / "pile_1k_128.pt"
    if tokens_path.exists():
        all_tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
        # Take first N_SEQS
        if all_tokens.shape[0] >= N_SEQS:
            tokens = all_tokens[:N_SEQS].to(DEVICE)
        else:
            tokens = all_tokens.to(DEVICE)
            print(f"[{TASK_ID}] Warning: only {all_tokens.shape[0]} sequences available")
    else:
        print(f"[{TASK_ID}] Token file not found at {tokens_path}, generating random tokens...")
        tokens = torch.randint(0, model.cfg.d_vocab, (N_SEQS, SEQ_LEN), device=DEVICE)

    print(f"[{TASK_ID}] Token shape: {tokens.shape}")

    report_progress(1, 3, step=1, total_steps=3, metric={"stage": "computing_sae"})

    # Compute absorption scores for real SAE
    print(f"[{TASK_ID}] Computing absorption scores for real SAE...")
    sae_scores, sae_counts = compute_absorption_scores(sae, model, tokens, LAYER, DEVICE)

    report_progress(2, 3, step=2, total_steps=3, metric={"stage": "computing_random"})

    # Compute absorption scores for random control
    print(f"[{TASK_ID}] Computing absorption scores for random control...")
    rand_scores, rand_counts = compute_random_control(tokens, LAYER, DEVICE, d_sae=DICT_SIZE, d_model=model.cfg.d_model)

    # Statistics
    sae_pct_gt_05 = (sae_scores > 0.5).mean() * 100
    rand_pct_gt_05 = (rand_scores > 0.5).mean() * 100
    sae_mean = sae_scores.mean()
    rand_mean = rand_scores.mean()

    print(f"\n[{TASK_ID}] === RESULTS ===")
    print(f"  Real SAE:   % latents with absorption >0.5 = {sae_pct_gt_05:.2f}%")
    print(f"  Random ctrl: % latents with absorption >0.5 = {rand_pct_gt_05:.2f}%")
    print(f"  Real SAE:   mean absorption score = {sae_mean:.4f}")
    print(f"  Random ctrl: mean absorption score = {rand_mean:.4f}")

    # Save results
    results = {
        "task_id": TASK_ID,
        "layer": LAYER,
        "dict_size": DICT_SIZE,
        "n_sequences": tokens.shape[0],
        "seed": SEED,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "sae": {
            "absorption_scores": sae_scores.tolist(),
            "activation_counts": sae_counts.tolist(),
            "pct_gt_0_5": float(sae_pct_gt_05),
            "mean_score": float(sae_mean),
            "median_score": float(np.median(sae_scores)),
            "std_score": float(np.std(sae_scores)),
        },
        "random_control": {
            "absorption_scores": rand_scores.tolist(),
            "activation_counts": rand_counts.tolist(),
            "pct_gt_0_5": float(rand_pct_gt_05),
            "mean_score": float(rand_mean),
            "median_score": float(np.median(rand_scores)),
            "std_score": float(np.std(rand_scores)),
        },
        "pass_criteria": {
            "sae_pct_gt_0_5": float(sae_pct_gt_05),
            "random_pct_gt_0_5": float(rand_pct_gt_05),
            "signal_detected": bool(sae_pct_gt_05 >= 5.0 and rand_pct_gt_05 < 2.0),
        },
        "runtime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "pilots" / f"{TASK_ID}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{TASK_ID}] Results saved to {out_path}")

    # Qualitative inspection: top 10 most absorbed latents
    top_absorbed = np.argsort(sae_scores)[-10:][::-1]
    print(f"\n[{TASK_ID}] Top 10 most absorbed latents:")
    for rank, idx in enumerate(top_absorbed, 1):
        print(f"  {rank}. Latent {idx}: score={sae_scores[idx]:.4f}, activations={sae_counts[idx]}")

    # Qualitative inspection: bottom 10
    bottom_absorbed = np.argsort(sae_scores)[:10]
    print(f"\n[{TASK_ID}] Bottom 10 least absorbed latents:")
    for rank, idx in enumerate(bottom_absorbed, 1):
        print(f"  {rank}. Latent {idx}: score={sae_scores[idx]:.4f}, activations={sae_counts[idx]}")

    summary = f"SAE: {sae_pct_gt_05:.1f}% latents absorbed >0.5 (random: {rand_pct_gt_05:.1f}%). Signal: {'YES' if sae_pct_gt_05 >= 5.0 and rand_pct_gt_05 < 2.0 else 'NO'}."
    mark_done(status="success", summary=summary)
    print(f"[{TASK_ID}] Done in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[{TASK_ID}] ERROR: {e}")
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
