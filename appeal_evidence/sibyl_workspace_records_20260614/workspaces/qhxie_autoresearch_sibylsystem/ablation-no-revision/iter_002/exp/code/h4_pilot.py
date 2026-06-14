#!/usr/bin/env python3
"""
H4 Pilot: Circuit Faithfulness via Activation Patching
Pilot on 1 factual recall task at layer 8.
Compare raw residual patching vs. SAE latent patching for low-absorption vs. high-absorption latents.
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
THRESHOLD_PCT = 0.01
COFIRER_TOPK = 5
VARIANCE_THRESHOLD = 0.80

# Patching config
CLEAN_PROMPT = "The capital of France is"
CORRUPTED_PROMPT = "The capital of Germany is"
TARGET_TOKENS = [" Paris", " Berlin"]  # space-prefixed for GPT-2 tokenizer

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results")
SHARED_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/shared")
TASK_ID = "h4_pilot"

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
# Absorption score computation (needed to select low/high absorption latents)
# ---------------------------------------------------------------------------
def compute_absorption_scores(sae, model, tokens, layer, device):
    """Compute absorption score for every latent."""
    sae.eval()
    model.eval()
    d_sae = sae.cfg.d_sae
    batch_size = tokens.shape[0]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_pre")
        activations = cache[f"blocks.{layer}.hook_resid_pre"]

    N = batch_size * SEQ_LEN
    acts_flat = activations.reshape(N, -1).to(device)

    with torch.no_grad():
        features = sae.encode(acts_flat)

    max_acts = features.max(dim=0).values
    thresholds = max_acts * THRESHOLD_PCT

    absorption_scores = np.zeros(d_sae, dtype=np.float32)
    activation_counts = np.zeros(d_sae, dtype=np.int32)

    chunk_size = 256
    W_dec = sae.W_dec

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
# Activation Patching
# ---------------------------------------------------------------------------
def compute_patching_faithfulness(model, sae, clean_tokens, corrupted_tokens, layer, device,
                                   patching_mode="raw", selected_latents=None):
    """
    Compute faithfulness of activation patching.

    patching_mode:
      - "raw": patch full residual activations
      - "sae_low": patch through SAE, using low-absorption latents
      - "sae_high": patch through SAE, using high-absorption latents
      - "sae_all": patch through SAE using all latents

    Faithfulness metric: fraction of token positions where patching restores
    >=50% of the clean->corrupted logit difference for the target token.
    """
    model.eval()
    sae.eval()

    # Run clean and corrupted to get baseline logits
    with torch.no_grad():
        clean_logits = model(clean_tokens)  # [1, seq_len, vocab]
        corrupted_logits = model(corrupted_tokens)

    # Target token IDs
    tokenizer = model.tokenizer
    clean_target_id = tokenizer.encode(TARGET_TOKENS[0], add_special_tokens=False)[0]
    corrupted_target_id = tokenizer.encode(TARGET_TOKENS[1], add_special_tokens=False)[0]

    # For faithfulness, we compare the logit of the correct answer (Paris)
    # between clean and corrupted runs
    clean_paris_logits = clean_logits[0, :, clean_target_id]
    corrupted_paris_logits = corrupted_logits[0, :, clean_target_id]
    baseline_diff = clean_paris_logits - corrupted_paris_logits  # [seq_len]

    # Patching: run corrupted with patches from clean at the specified layer
    hook_name = f"blocks.{layer}.hook_resid_pre"

    # Pre-compute clean activations ONCE outside the hook to avoid recursion
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda n: n == hook_name
        )
    clean_acts = clean_cache[hook_name]  # [1, seq_len, d_model]

    # Pre-compute SAE-encoded clean activations for SAE modes
    if patching_mode != "raw":
        batch_size, seq_len, d_model = clean_acts.shape
        clean_flat = clean_acts.reshape(-1, d_model)
        with torch.no_grad():
            clean_features = sae.encode(clean_flat)  # [batch*seq, d_sae]
        if selected_latents is not None:
            mask = torch.zeros(clean_features.shape[1], device=device, dtype=torch.bool)
            mask[selected_latents] = True
            clean_features = clean_features * mask.unsqueeze(0)
        with torch.no_grad():
            sae_patched_acts = sae.decode(clean_features)
        sae_patched_acts = sae_patched_acts.reshape(batch_size, seq_len, d_model)
    else:
        sae_patched_acts = None

    def patch_hook(activations, hook):
        """Replace activations with pre-computed clean activations (or SAE-reconstructed)."""
        if patching_mode == "raw":
            return clean_acts
        else:
            return sae_patched_acts

    # Run corrupted with patch
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )

    patched_paris_logits = patched_logits[0, :, clean_target_id]
    patched_diff = patched_paris_logits - corrupted_paris_logits

    # Faithfulness: what fraction of baseline diff is restored?
    # Avoid division by zero
    faithfulness_per_pos = torch.zeros_like(baseline_diff)
    valid_mask = baseline_diff.abs() > 1e-6
    faithfulness_per_pos[valid_mask] = patched_diff[valid_mask] / baseline_diff[valid_mask]

    # Clip to [0, 1] (can't restore more than 100%)
    faithfulness_per_pos = torch.clamp(faithfulness_per_pos, min=0.0, max=1.0)

    # Overall faithfulness: mean across positions
    mean_faithfulness = faithfulness_per_pos.mean().item()

    # Also compute: fraction of positions where >=50% restored
    half_restored = (faithfulness_per_pos >= 0.5).float().mean().item()

    return {
        "mean_faithfulness": mean_faithfulness,
        "half_restored_fraction": half_restored,
        "faithfulness_per_position": faithfulness_per_pos.cpu().tolist(),
        "baseline_diff": baseline_diff.cpu().tolist(),
        "patched_diff": patched_diff.cpu().tolist(),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    write_pid()
    start_time = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[{TASK_ID}] Starting on device={DEVICE}, GPU={GPU_ID}")
    print(f"[{TASK_ID}] Layer={LAYER}, Task=France/Paris vs Germany/Berlin")

    report_progress(0, 4, step=0, total_steps=4, metric={"stage": "loading_model"})

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"[{TASK_ID}] Loading gpt2-small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"[{TASK_ID}] Model loaded.")

    print(f"[{TASK_ID}] Loading SAE for layer {LAYER}...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE,
    )
    print(f"[{TASK_ID}] SAE loaded: d_sae={sae.cfg.d_sae}")

    # Load tokens for absorption score computation
    print(f"[{TASK_ID}] Loading tokens for absorption scoring...")
    tokens_path = SHARED_DIR / "pile_1k_128.pt"
    if tokens_path.exists():
        all_tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
        tokens = all_tokens[:N_SEQS].to(DEVICE)
    else:
        tokens = torch.randint(0, model.cfg.d_vocab, (N_SEQS, SEQ_LEN), device=DEVICE)
    print(f"[{TASK_ID}] Token shape: {tokens.shape}")

    report_progress(1, 4, step=1, total_steps=4, metric={"stage": "computing_absorption"})

    # Compute absorption scores to select low/high absorption latents
    print(f"[{TASK_ID}] Computing absorption scores...")
    absorption_scores, activation_counts = compute_absorption_scores(sae, model, tokens, LAYER, DEVICE)

    # Select latents with sufficient activations
    min_activations = 10
    valid_mask = activation_counts >= min_activations
    valid_indices = np.where(valid_mask)[0]
    valid_scores = absorption_scores[valid_mask]

    print(f"[{TASK_ID}] {len(valid_indices)}/{len(absorption_scores)} latents have >= {min_activations} activations")

    # Select top/bottom 10% by absorption score
    if len(valid_indices) >= 20:
        n_select = max(10, int(len(valid_indices) * 0.1))
        sorted_by_score = np.argsort(valid_scores)
        low_abs_indices = valid_indices[sorted_by_score[:n_select]]
        high_abs_indices = valid_indices[sorted_by_score[-n_select:]]
    else:
        # Fallback: just use all valid, split by median
        median_idx = len(valid_indices) // 2
        sorted_by_score = np.argsort(valid_scores)
        low_abs_indices = valid_indices[sorted_by_score[:median_idx]]
        high_abs_indices = valid_indices[sorted_by_score[median_idx:]]

    print(f"[{TASK_ID}] Low-absorption latents: {len(low_abs_indices)} (mean score: {absorption_scores[low_abs_indices].mean():.4f})")
    print(f"[{TASK_ID}] High-absorption latents: {len(high_abs_indices)} (mean score: {absorption_scores[high_abs_indices].mean():.4f})")

    report_progress(2, 4, step=2, total_steps=4, metric={"stage": "patching"})

    # Prepare clean and corrupted prompts
    tokenizer = model.tokenizer
    clean_tokens = tokenizer(CLEAN_PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    corrupted_tokens = tokenizer(CORRUPTED_PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    print(f"[{TASK_ID}] Clean tokens: {clean_tokens.shape} -> '{CLEAN_PROMPT}'")
    print(f"[{TASK_ID}] Corrupted tokens: {corrupted_tokens.shape} -> '{CORRUPTED_PROMPT}'")

    # Run patching experiments
    print(f"[{TASK_ID}] Running activation patching experiments...")

    # 1. Raw residual patching
    print(f"[{TASK_ID}] 1. Raw residual patching...")
    raw_results = compute_patching_faithfulness(
        model, sae, clean_tokens, corrupted_tokens, LAYER, DEVICE,
        patching_mode="raw"
    )
    print(f"[{TASK_ID}]    Raw faithfulness: {raw_results['mean_faithfulness']:.4f}")

    # 2. SAE all latents
    print(f"[{TASK_ID}] 2. SAE all latents patching...")
    sae_all_results = compute_patching_faithfulness(
        model, sae, clean_tokens, corrupted_tokens, LAYER, DEVICE,
        patching_mode="sae_all", selected_latents=None
    )
    print(f"[{TASK_ID}]    SAE all faithfulness: {sae_all_results['mean_faithfulness']:.4f}")

    # 3. SAE low-absorption latents
    print(f"[{TASK_ID}] 3. SAE low-absorption latents patching...")
    sae_low_results = compute_patching_faithfulness(
        model, sae, clean_tokens, corrupted_tokens, LAYER, DEVICE,
        patching_mode="sae_low", selected_latents=low_abs_indices.tolist()
    )
    print(f"[{TASK_ID}]    SAE low-abs faithfulness: {sae_low_results['mean_faithfulness']:.4f}")

    # 4. SAE high-absorption latents
    print(f"[{TASK_ID}] 4. SAE high-absorption latents patching...")
    sae_high_results = compute_patching_faithfulness(
        model, sae, clean_tokens, corrupted_tokens, LAYER, DEVICE,
        patching_mode="sae_high", selected_latents=high_abs_indices.tolist()
    )
    print(f"[{TASK_ID}]    SAE high-abs faithfulness: {sae_high_results['mean_faithfulness']:.4f}")

    report_progress(3, 4, step=3, total_steps=4, metric={"stage": "saving_results"})

    # Compile results
    results = {
        "task_id": TASK_ID,
        "layer": LAYER,
        "dict_size": DICT_SIZE,
        "clean_prompt": CLEAN_PROMPT,
        "corrupted_prompt": CORRUPTED_PROMPT,
        "target_tokens": TARGET_TOKENS,
        "seed": SEED,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "absorption_scores": {
            "mean": float(absorption_scores.mean()),
            "median": float(np.median(absorption_scores)),
            "std": float(np.std(absorption_scores)),
            "low_abs_latents": low_abs_indices.tolist(),
            "low_abs_mean_score": float(absorption_scores[low_abs_indices].mean()),
            "high_abs_latents": high_abs_indices.tolist(),
            "high_abs_mean_score": float(absorption_scores[high_abs_indices].mean()),
        },
        "patching_results": {
            "raw_residual": raw_results,
            "sae_all_latents": sae_all_results,
            "sae_low_absorption": sae_low_results,
            "sae_high_absorption": sae_high_results,
        },
        "faithfulness_comparison": {
            "raw": raw_results["mean_faithfulness"],
            "sae_all": sae_all_results["mean_faithfulness"],
            "sae_low": sae_low_results["mean_faithfulness"],
            "sae_high": sae_high_results["mean_faithfulness"],
            "raw_vs_sae_all_diff": raw_results["mean_faithfulness"] - sae_all_results["mean_faithfulness"],
            "low_vs_high_diff": sae_low_results["mean_faithfulness"] - sae_high_results["mean_faithfulness"],
        },
        "pass_criteria": {
            "raw_faithfulness_computable": raw_results["mean_faithfulness"] > 0,
            "sae_within_20pp_of_raw": abs(raw_results["mean_faithfulness"] - sae_all_results["mean_faithfulness"]) < 0.20,
        },
        "runtime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "pilots" / f"{TASK_ID}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{TASK_ID}] Results saved to {out_path}")

    # Print summary
    print(f"\n[{TASK_ID}] === RESULTS ===")
    print(f"  Raw residual faithfulness:     {raw_results['mean_faithfulness']:.4f}")
    print(f"  SAE all latents faithfulness:  {sae_all_results['mean_faithfulness']:.4f}")
    print(f"  SAE low-abs faithfulness:      {sae_low_results['mean_faithfulness']:.4f}")
    print(f"  SAE high-abs faithfulness:     {sae_high_results['mean_faithfulness']:.4f}")
    print(f"  Raw vs SAE all diff:           {results['faithfulness_comparison']['raw_vs_sae_all_diff']:.4f}")
    print(f"  Low vs High abs diff:          {results['faithfulness_comparison']['low_vs_high_diff']:.4f}")

    # Qualitative: show token-by-token faithfulness for raw patching
    print(f"\n[{TASK_ID}] Token-by-token faithfulness (raw residual patching):")
    clean_tokens_list = clean_tokens[0].cpu().tolist()
    for pos, (tok_id, faith) in enumerate(zip(clean_tokens_list, raw_results["faithfulness_per_position"])):
        tok_str = tokenizer.decode([tok_id])
        print(f"  Pos {pos}: '{tok_str}' -> faithfulness={faith:.4f}")

    summary = (
        f"Raw={raw_results['mean_faithfulness']:.3f}, "
        f"SAE_all={sae_all_results['mean_faithfulness']:.3f}, "
        f"SAE_low={sae_low_results['mean_faithfulness']:.3f}, "
        f"SAE_high={sae_high_results['mean_faithfulness']:.3f}. "
        f"Low-vs-High diff={results['faithfulness_comparison']['low_vs_high_diff']:+.3f}."
    )
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
