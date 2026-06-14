#!/usr/bin/env python3
"""
H5 Pilot: Dictionary Size Effect on Feature Absorption
Pilot on 100 sequences, layer 8.

Since gpt2-small-res-jb only provides d_sae=24576, we test dictionary size effect
by using CUMULATIVE latent sets: starting with the top 2048 most-active latents,
then adding more to reach 8192, then the full 24576. This tests whether adding more
features to the dictionary reduces per-feature absorption rates for the same core set.

Alternative interpretation: We first identify which latents are "absorbable" in the
full dictionary, then test whether smaller subsets of those same latents show higher
absorption (because fewer alternative features are available to explain variance).

Hypothesis: Larger dictionary sizes reduce per-feature absorption rates.
Falsification: Absorption rate does not decrease with dictionary size.
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
N_SEQS = 100
SEQ_LEN = 128
THRESHOLD_PCT = 0.01  # activation threshold = 1% of max
COFIRER_TOPK = 5
VARIANCE_THRESHOLD = 0.80
DICT_SIZES = [2048, 8192, 24576]

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results")
SHARED_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/shared")
TASK_ID = "h5_pilot"

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
# Absorption metric with subsampled dictionary
# ---------------------------------------------------------------------------
def compute_absorption_scores_subsampled(sae, model, tokens, layer, device, latent_indices):
    """
    Compute absorption score for a SUBSAMPLED set of latents from the SAE.
    Co-firers are also drawn from the subsampled set.
    """
    sae.eval()
    model.eval()

    d_sae_sub = len(latent_indices)
    batch_size = tokens.shape[0]

    latent_indices_t = torch.tensor(latent_indices, device=device, dtype=torch.long)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_pre")
        activations = cache[f"blocks.{layer}.hook_resid_pre"]

    N = batch_size * SEQ_LEN
    acts_flat = activations.reshape(N, -1).to(device)

    with torch.no_grad():
        features_full = sae.encode(acts_flat)
    features = features_full[:, latent_indices_t]

    max_acts = features.max(dim=0).values
    thresholds = max_acts * THRESHOLD_PCT

    absorption_scores = np.zeros(d_sae_sub, dtype=np.float32)
    activation_counts = np.zeros(d_sae_sub, dtype=np.int32)

    chunk_size = 256
    W_dec_full = sae.W_dec
    W_dec = W_dec_full[latent_indices_t]

    for chunk_start in range(0, d_sae_sub, chunk_size):
        chunk_end = min(chunk_start + chunk_size, d_sae_sub)

        for idx_in_chunk in range(chunk_start, chunk_end):
            threshold = thresholds[idx_in_chunk].item()
            activations_f = features[:, idx_in_chunk]
            active_mask = activations_f > threshold
            active_indices = torch.where(active_mask)[0]
            n_active = active_indices.shape[0]
            activation_counts[idx_in_chunk] = n_active

            if n_active == 0:
                absorption_scores[idx_in_chunk] = 0.0
                continue

            active_features = features[active_indices]
            active_features_copy = active_features.clone()
            active_features_copy[:, idx_in_chunk] = -float('inf')

            topk_vals, topk_idx = torch.topk(active_features_copy, k=COFIRER_TOPK, dim=1)

            act_active = acts_flat[active_indices]
            var_x = act_active.var(dim=1, unbiased=False)

            partial = W_dec[idx_in_chunk].unsqueeze(0) * activations_f[active_indices].unsqueeze(1)
            for c in range(COFIRER_TOPK):
                cofirer_idx_sub = topk_idx[:, c]
                cofirer_act = active_features[torch.arange(n_active, device=device), cofirer_idx_sub]
                partial += W_dec[cofirer_idx_sub] * cofirer_act.unsqueeze(1)

            var_residual = (act_active - partial).var(dim=1, unbiased=False)
            var_x = torch.clamp(var_x, min=1e-8)
            var_explained = 1.0 - var_residual / var_x

            n_absorbed = (var_explained > VARIANCE_THRESHOLD).sum().item()
            absorption_scores[idx_in_chunk] = n_absorbed / n_active

    return absorption_scores, activation_counts

# ---------------------------------------------------------------------------
# Random dictionary control
# ---------------------------------------------------------------------------
def compute_random_control(tokens, layer, device, d_sae, d_model=768):
    rand_path = SHARED_DIR / f"random_decoders_{d_sae}.pt"
    if rand_path.exists():
        W_dec = torch.load(rand_path, map_location=device, weights_only=True)
    else:
        W_dec = torch.randn(d_sae, d_model, device=device)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)

    W_enc = torch.randn(d_model, d_sae, device=device)
    b_enc = torch.zeros(d_sae, device=device)

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
    print(f"[{TASK_ID}] Layer={LAYER}, DictSizes={DICT_SIZES}, NSeqs={N_SEQS}")
    print(f"[{TASK_ID}] NOTE: Using cumulative subsampling from d_sae=24576")

    report_progress(0, 5, step=0, total_steps=5, metric={"stage": "loading_model"})

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"[{TASK_ID}] Loading gpt2-small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"[{TASK_ID}] Model loaded.")

    print(f"[{TASK_ID}] Loading SAE for layer {LAYER}...")
    sae_full = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE,
    )
    d_sae_full = sae_full.cfg.d_sae
    print(f"[{TASK_ID}] Full SAE loaded: d_sae={d_sae_full}, d_in={sae_full.cfg.d_in}")

    # Load tokens
    print(f"[{TASK_ID}] Loading tokens...")
    tokens_path = SHARED_DIR / "pile_1k_128.pt"
    if tokens_path.exists():
        all_tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
        if all_tokens.shape[0] >= N_SEQS:
            tokens = all_tokens[:N_SEQS].to(DEVICE)
        else:
            tokens = all_tokens.to(DEVICE)
            print(f"[{TASK_ID}] Warning: only {all_tokens.shape[0]} sequences available")
    else:
        print(f"[{TASK_ID}] Token file not found at {tokens_path}, generating random tokens...")
        tokens = torch.randint(0, model.cfg.d_vocab, (N_SEQS, SEQ_LEN), device=DEVICE)

    print(f"[{TASK_ID}] Token shape: {tokens.shape}")

    # Step 1: Compute activation counts for ALL latents to rank by activity
    report_progress(1, 5, step=1, total_steps=5, metric={"stage": "ranking_latents"})
    print(f"[{TASK_ID}] Ranking all {d_sae_full} latents by activation frequency...")

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == f"blocks.{LAYER}.hook_resid_pre")
        activations = cache[f"blocks.{LAYER}.hook_resid_pre"]
        N = tokens.shape[0] * SEQ_LEN
        acts_flat = activations.reshape(N, -1).to(DEVICE)
        features_full = sae_full.encode(acts_flat)

        # Count activations per latent (feature > 0)
        activation_counts_full = (features_full > 0).sum(dim=0).cpu().numpy()  # [d_sae_full]

    # Rank latents by activation count (most active first)
    ranked_indices = np.argsort(-activation_counts_full).tolist()  # Descending
    print(f"[{TASK_ID}] Top 10 most active latents: {ranked_indices[:10]}")
    print(f"[{TASK_ID}] Their activation counts: {activation_counts_full[ranked_indices[:10]].tolist()}")

    # Step 2: First compute FULL dict absorption to identify "absorbable" latents
    report_progress(2, 5, step=2, total_steps=5, metric={"stage": "computing_full_24576"})
    print(f"\n[{TASK_ID}] === Computing FULL dict (24576) absorption ===")
    full_scores, full_counts = compute_absorption_scores_subsampled(
        sae_full, model, tokens, LAYER, DEVICE, list(range(d_sae_full))
    )

    # Identify latents with highest absorption scores
    absorption_ranked = np.argsort(-full_scores).tolist()
    print(f"[{TASK_ID}] Top 10 most absorbed latents: {absorption_ranked[:10]}")
    print(f"[{TASK_ID}] Their absorption scores: {full_scores[absorption_ranked[:10]].tolist()}")

    # Step 3: Build cumulative latent sets
    # Strategy: include top-absorbed latents first, then fill with most-active latents
    # This ensures the "absorbable" latents are present in all dict sizes
    top_absorbed = [i for i in absorption_ranked if full_scores[i] > 0]
    print(f"[{TASK_ID}] Latents with absorption > 0: {len(top_absorbed)}")

    results_by_size = {}
    step = 3

    for d_sae in DICT_SIZES:
        print(f"\n[{TASK_ID}] === Processing dict_size={d_sae} ===")
        report_progress(2, 5, step=step, total_steps=5, metric={"stage": f"computing_sae_{d_sae}"})

        if d_sae > d_sae_full:
            print(f"[{TASK_ID}] ERROR: requested d_sae={d_sae} > full d_sae={d_sae_full}")
            results_by_size[d_sae] = {"error": f"Requested d_sae={d_sae} > full d_sae={d_sae_full}"}
            step += 1
            continue

        # Build latent set: prioritize absorbable latents, then most-active
        if len(top_absorbed) >= d_sae:
            # More absorbable latents than dict size - take top d_sae
            latent_indices = top_absorbed[:d_sae]
        else:
            # Include all absorbable latents, fill rest with most-active (excluding already chosen)
            chosen = set(top_absorbed)
            remaining = [i for i in ranked_indices if i not in chosen]
            needed = d_sae - len(top_absorbed)
            latent_indices = top_absorbed + remaining[:needed]

        latent_indices.sort()
        print(f"[{TASK_ID}] Latent set: {len(latent_indices)} latents (includes {len([i for i in latent_indices if i in top_absorbed])} absorbable)")

        # Compute absorption for this dict size
        print(f"[{TASK_ID}] Computing absorption scores for dict={d_sae}...")
        sae_scores, sae_counts = compute_absorption_scores_subsampled(
            sae_full, model, tokens, LAYER, DEVICE, latent_indices
        )

        # Also compute random control
        print(f"[{TASK_ID}] Computing random control for dict={d_sae}...")
        rand_scores, rand_counts = compute_random_control(tokens, LAYER, DEVICE, d_sae=d_sae, d_model=model.cfg.d_model)

        # Statistics
        sae_pct_gt_05 = (sae_scores > 0.5).mean() * 100
        rand_pct_gt_05 = (rand_scores > 0.5).mean() * 100
        sae_mean = sae_scores.mean()
        rand_mean = rand_scores.mean()

        print(f"\n[{TASK_ID}] --- Dict {d_sae} Results ---")
        print(f"  Real SAE:   % latents with absorption >0.5 = {sae_pct_gt_05:.2f}%")
        print(f"  Random ctrl: % latents with absorption >0.5 = {rand_pct_gt_05:.2f}%")
        print(f"  Real SAE:   mean absorption score = {sae_mean:.4f}")
        print(f"  Random ctrl: mean absorption score = {rand_mean:.4f}")

        results_by_size[d_sae] = {
            "dict_size": d_sae,
            "latent_indices": latent_indices,
            "n_absorbable_included": len([i for i in latent_indices if i in top_absorbed]),
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
        }

        step += 1

    # Aggregate comparison
    print(f"\n[{TASK_ID}] === AGGREGATE COMPARISON ===")
    dict_sizes_processed = [k for k in sorted(results_by_size.keys()) if "error" not in results_by_size[k]]

    if len(dict_sizes_processed) >= 2:
        print(f"  Dict sizes tested: {dict_sizes_processed}")
        for ds in dict_sizes_processed:
            r = results_by_size[ds]
            print(f"  {ds}: SAE mean={r['sae']['mean_score']:.4f}, random mean={r['random_control']['mean_score']:.4f}, %>0.5={r['sae']['pct_gt_0_5']:.2f}%")

        means = [(ds, results_by_size[ds]["sae"]["mean_score"]) for ds in dict_sizes_processed]
        means_sorted = sorted(means, key=lambda x: x[0])
        print(f"\n  Mean absorption by dict size:")
        for ds, m in means_sorted:
            print(f"    {ds}: {m:.4f}")

        pass_dict_size_effect = all(means_sorted[i][1] >= means_sorted[i+1][1] for i in range(len(means_sorted)-1))
        print(f"  Monotonic decrease (larger dict -> lower absorption): {pass_dict_size_effect}")

        rand_scaling_appropriate = all(
            results_by_size[ds]["random_control"]["mean_score"] <= results_by_size[ds]["sae"]["mean_score"] * 2
            for ds in dict_sizes_processed
        )
        print(f"  Random controls scale appropriately: {rand_scaling_appropriate}")
    else:
        pass_dict_size_effect = False
        rand_scaling_appropriate = False

    # Save results
    results = {
        "task_id": TASK_ID,
        "layer": LAYER,
        "dict_sizes": DICT_SIZES,
        "full_sae_d_sae": d_sae_full,
        "method": "cumulative_subsample_prioritizing_absorbable",
        "note": "Since gpt2-small-res-jb only provides d_sae=24576, smaller dicts are simulated by subsampling latents. Absorbable latents are prioritized.",
        "n_sequences": tokens.shape[0],
        "seed": SEED,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "results_by_size": results_by_size,
        "pass_criteria": {
            "pass_dict_size_effect": bool(pass_dict_size_effect),
            "rand_scaling_appropriate": bool(rand_scaling_appropriate),
            "larger_dict_lower_absorption": bool(pass_dict_size_effect),
        },
        "runtime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "pilots" / f"{TASK_ID}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{TASK_ID}] Results saved to {out_path}")

    # Qualitative
    for d_sae in dict_sizes_processed:
        scores = np.array(results_by_size[d_sae]["sae"]["absorption_scores"])
        counts = np.array(results_by_size[d_sae]["sae"]["activation_counts"])
        top5 = np.argsort(scores)[-5:][::-1]
        print(f"\n[{TASK_ID}] Top 5 most absorbed latents (dict={d_sae}):")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. Subsampled idx {idx}: score={scores[idx]:.4f}, activations={counts[idx]}")

    summary = f"Dict size effect (cumulative): {dict_sizes_processed}. "
    if pass_dict_size_effect:
        summary += "Larger dict shows lower absorption. PASS."
    else:
        summary += "No clear monotonic dict size effect. FAIL."

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
