#!/usr/bin/env python3
"""
H3 Pilot: Sparsity Trade-off Pilot
Examines relationship between SAE sparsity (L0) and absorption rate.

Approach: Since no pretrained SAEs with varying L1 coefficients are available,
we use L0 proxy across different layers (layers 0,2,4,6,8,10) and compute
correlation between mean L0 and mean absorption rate.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
TASK_ID = "h3_pilot"
SEED = 42
N_SEQUENCES = 100
SEQ_LEN = 128
BATCH_SIZE = 8
LAYERS = [0, 2, 4, 6, 8, 10]
THRESHOLD_PCT = 0.01  # 1% of max activation
COFIRER_TOPK = 5
VARIANCE_THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
SHARED_DIR = WORKSPACE / "shared"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Process tracking
# ------------------------------------------------------------------
def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
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
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
def load_data():
    data_path = SHARED_DIR / "pile_1k_128.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    token_ids = torch.load(data_path, map_location="cpu", weights_only=True)
    token_ids = token_ids[:N_SEQUENCES, :SEQ_LEN]
    return token_ids

# ------------------------------------------------------------------
# Absorption score computation (reused from h1_pilot)
# ------------------------------------------------------------------
def compute_absorption_scores(sae, model, token_ids, tokenizer, layer_idx):
    """Compute absorption scores for all latents in an SAE."""
    n_latents = sae.cfg.d_sae
    d_model = sae.cfg.d_in
    
    # Track activations per latent
    latent_activations = [[] for _ in range(n_latents)]
    
    # Hook to capture residual activations
    activation_cache = {}
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    
    def hook_fn(value, hook):
        activation_cache["resid"] = value.detach()
    
    model.add_hook(hook_name, hook_fn)
    
    n_batches = (N_SEQUENCES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, N_SEQUENCES)
        batch_tokens = token_ids[start:end].to(DEVICE)
        
        with torch.no_grad():
            model(batch_tokens)
            resid = activation_cache["resid"]  # [batch, seq, d_model]
            
            # Encode through SAE
            sae_acts = sae.encode(resid)  # [batch, seq, n_latents]
            
            # For each latent, record (token_idx, activation, top5_cofirers)
            for latent_idx in range(n_latents):
                acts = sae_acts[:, :, latent_idx]  # [batch, seq]
                mask = acts > 0
                if mask.any():
                    positions = torch.nonzero(mask, as_tuple=False)
                    for pos in positions:
                        bi, si = pos[0].item(), pos[1].item()
                        act_val = acts[bi, si].item()
                        # Get top-5 other latents at this position
                        other_acts = sae_acts[bi, si, :].clone()
                        other_acts[latent_idx] = -1e9
                        top5_vals, top5_idx = torch.topk(other_acts, min(COFIRER_TOPK, n_latents - 1))
                        top5_vals = top5_vals[top5_vals > 0]
                        top5_idx = top5_idx[:len(top5_vals)]
                        
                        latent_activations[latent_idx].append({
                            "batch": start + bi,
                            "seq": si,
                            "activation": act_val,
                            "top5_idx": top5_idx.cpu().tolist(),
                            "top5_vals": top5_vals.cpu().tolist(),
                            "resid": resid[bi, si].cpu(),
                        })
        
        report_progress(
            epoch=b + 1, total_epochs=n_batches,
            step=b + 1, total_steps=n_batches,
            metric={"layer": layer_idx, "batch": b + 1}
        )
    
    model.reset_hooks(clear_contexts=True)

    # Compute absorption score per latent
    absorption_scores = []
    activation_counts = []
    
    for latent_idx in range(n_latents):
        acts = latent_activations[latent_idx]
        activation_counts.append(len(acts))
        
        if len(acts) == 0:
            absorption_scores.append(0.0)
            continue
        
        # Compute max activation for threshold
        max_act = max(a["activation"] for a in acts) if acts else 1.0
        threshold = max_act * THRESHOLD_PCT
        
        absorbed_count = 0
        total_count = 0
        
        for a in acts:
            if a["activation"] < threshold:
                continue
            total_count += 1
            
            # Compute partial reconstruction using feature + top5 cofirers
            x = a["resid"].to(DEVICE)
            x_var = x.var().item()
            if x_var < 1e-10:
                continue
            
            # Full SAE reconstruction for this token
            with torch.no_grad():
                x_hat = sae.decode(sae.encode(x.unsqueeze(0).unsqueeze(0))).squeeze()
            
            # Partial reconstruction: only this feature + cofirers
            partial_acts = torch.zeros(n_latents, device=DEVICE)
            partial_acts[latent_idx] = a["activation"]
            for cof_idx, cof_val in zip(a["top5_idx"], a["top5_vals"]):
                partial_acts[cof_idx] = cof_val
            
            with torch.no_grad():
                x_partial = sae.decode(partial_acts.unsqueeze(0).unsqueeze(0)).squeeze()
            
            var_explained = 1.0 - ((x - x_partial).var() / x_var).item()
            if var_explained > VARIANCE_THRESHOLD:
                absorbed_count += 1
        
        if total_count > 0:
            absorption_scores.append(absorbed_count / total_count)
        else:
            absorption_scores.append(0.0)
    
    return absorption_scores, activation_counts

# ------------------------------------------------------------------
# L0 computation
# ------------------------------------------------------------------
def compute_l0(sae, model, token_ids, layer_idx):
    """Compute mean L0 (number of active features per token) for this SAE."""
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    activation_cache = {}
    
    def hook_fn(value, hook):
        activation_cache["resid"] = value.detach()
    
    model.add_hook(hook_name, hook_fn)
    
    l0_values = []
    n_batches = (N_SEQUENCES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, N_SEQUENCES)
        batch_tokens = token_ids[start:end].to(DEVICE)
        
        with torch.no_grad():
            model(batch_tokens)
            resid = activation_cache["resid"]
            sae_acts = sae.encode(resid)  # [batch, seq, n_latents]
            
            # Count non-zero activations per token
            for bi in range(sae_acts.shape[0]):
                for si in range(sae_acts.shape[1]):
                    n_active = (sae_acts[bi, si] > 0).sum().item()
                    l0_values.append(n_active)
    
    model.reset_hooks(clear_contexts=True)

    return np.mean(l0_values), np.std(l0_values), l0_values

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    write_pid()
    start_time = time.time()
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print(f"[{TASK_ID}] Starting H3 pilot on device={DEVICE}, gpu={GPU_ID}")
    
    # Load data
    print("Loading data...")
    token_ids = load_data()
    print(f"  Data shape: {token_ids.shape}")
    
    # Load model
    print("Loading GPT-2 small...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load SAEs for each layer and compute absorption + L0
    from sae_lens import SAE
    
    layer_results = []
    
    for layer_idx in LAYERS:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer_idx}")
        print(f"{'='*60}")
        
        # Load SAE
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer_idx}.hook_resid_pre",
            device=DEVICE,
        )[0]
        sae.eval()
        
        print(f"  SAE: d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
        
        # Compute L0
        print("  Computing L0 (sparsity)...")
        mean_l0, std_l0, l0_values = compute_l0(sae, model, token_ids, layer_idx)
        print(f"  L0: mean={mean_l0:.2f}, std={std_l0:.2f}")
        
        # Compute absorption scores
        print("  Computing absorption scores...")
        absorption_scores, activation_counts = compute_absorption_scores(
            sae, model, token_ids, tokenizer, layer_idx
        )
        
        mean_abs = np.mean(absorption_scores)
        median_abs = np.median(absorption_scores)
        std_abs = np.std(absorption_scores)
        pct_gt_0_5 = sum(1 for s in absorption_scores if s > 0.5) / len(absorption_scores) * 100
        
        print(f"  Absorption: mean={mean_abs:.4f}, median={median_abs:.4f}, std={std_abs:.4f}")
        print(f"  % > 0.5: {pct_gt_0_5:.2f}%")
        
        layer_results.append({
            "layer": layer_idx,
            "d_sae": sae.cfg.d_sae,
            "mean_l0": float(mean_l0),
            "std_l0": float(std_l0),
            "mean_absorption": float(mean_abs),
            "median_absorption": float(median_abs),
            "std_absorption": float(std_abs),
            "pct_gt_0_5": float(pct_gt_0_5),
            "absorption_scores": [float(s) for s in absorption_scores],
            "activation_counts": activation_counts,
        })
        
        # Clean up
        del sae
        torch.cuda.empty_cache()
    
    # Compute correlation between L0 and absorption
    mean_l0s = [r["mean_l0"] for r in layer_results]
    mean_abss = [r["mean_absorption"] for r in layer_results]
    
    from scipy.stats import spearmanr, pearsonr
    spr_r, spr_p = spearmanr(mean_l0s, mean_abss)
    pr_r, pr_p = pearsonr(mean_l0s, mean_abss)
    
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Spearman r={spr_r:.4f}, p={spr_p:.4f}")
    print(f"Pearson  r={pr_r:.4f}, p={pr_p:.4f}")
    
    # Determine if monotonic trend exists
    monotonic = all(mean_abss[i] <= mean_abss[i+1] for i in range(len(mean_abss)-1)) or \
                all(mean_abss[i] >= mean_abss[i+1] for i in range(len(mean_abss)-1))
    
    print(f"Monotonic trend: {monotonic}")
    
    # Save results
    runtime = time.time() - start_time
    results = {
        "task_id": TASK_ID,
        "n_sequences": N_SEQUENCES,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "layers": LAYERS,
        "layer_results": layer_results,
        "correlation": {
            "spearman_r": float(spr_r),
            "spearman_p": float(spr_p),
            "pearson_r": float(pr_r),
            "pearson_p": float(pr_p),
            "monotonic_trend": monotonic,
        },
        "runtime_seconds": runtime,
        "timestamp": datetime.now().isoformat(),
    }
    
    out_path = RESULTS_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    
    # Summary
    summary = (
        f"H3 pilot complete. {len(LAYERS)} layers analyzed. "
        f"Spearman r={spr_r:.3f} (p={spr_p:.3f}). "
        f"Monotonic={monotonic}. "
        f"L0 range: {min(mean_l0s):.1f}-{max(mean_l0s):.1f}. "
        f"Absorption range: {min(mean_abss):.4f}-{max(mean_abss):.4f}."
    )
    print(summary)
    
    mark_done(status="success", summary=summary)
    print(f"\nTotal runtime: {runtime:.1f}s")

if __name__ == "__main__":
    main()
