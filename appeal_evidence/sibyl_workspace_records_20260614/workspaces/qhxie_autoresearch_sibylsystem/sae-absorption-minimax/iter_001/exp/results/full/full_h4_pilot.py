"""
Full H4: UAS validation across models and layers (PILOT mode).
Validates calibrated UAS (alpha=1.0, beta=0.5) from pilot_h1_h4
across GPT-2 (layers 4, 8) and Gemma-2B (layer 8).

Pass criteria: Mean Spearman r > 0.5 across 3 (model, layer) pairs.
"""

import json
import os
import sys
import gc
import warnings
import numpy as np
import torch
from scipy.stats import spearmanr, skew
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
REMOTE_BASE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = REMOTE_BASE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "full_h4"
DEVICE = "cuda"  # Use "cuda" since CUDA_VISIBLE_DEVICES is set externally
N_TOKENS = 100  # pilot budget per pair
SEED = 42
N_FEATURES = 100

# UAS hyperparameters (calibrated from pilot_h1_h4: alpha=1.0, beta=0.5)
UAS_ALPHA = 1.0
UAS_BETA = 0.5

# Experiment pairs for pilot (3 pairs from GPT-2 Small at different layers)
# Note: Gemma-2B is gated (requires HF auth) and its SAEs are not locally cached.
# Using GPT-2 Small at layers 4, 6, 8 to validate UAS across different layers.
EXP_PAIRS = [
    {"model": "gpt2-small", "layer": 4, "sae_release": "gpt2-small-res-jb"},
    {"model": "gpt2-small", "layer": 6, "sae_release": "gpt2-small-res-jb"},
    {"model": "gpt2-small", "layer": 8, "sae_release": "gpt2-small-res-jb"},
]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_uas(sae, feature_acts, topk_indices):
    """Compute Unsupervised Absorption Score (UAS) for each feature."""
    W_dec = sae.W_dec.data.cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    uas_scores = {}
    cos_sim_variances = {}
    freq_skewness_dict = {}

    for feat_idx in topk_indices:
        # Cosine similarity variance
        other_indices = [i for i in topk_indices if i != feat_idx]
        if len(other_indices) < 2:
            cos_sim_variances[feat_idx] = 0.0
            freq_skewness_dict[feat_idx] = 0.0
            uas_scores[feat_idx] = 0.0
            continue

        cos_sims = np.dot(W_dec_norm[feat_idx], W_dec_norm[other_indices].T)
        cos_sim_variances[feat_idx] = float(np.var(cos_sims))

        # Frequency skewness
        feat_act_values = feature_acts[0, :, feat_idx].cpu().numpy()
        non_zero_acts = feat_act_values[feat_act_values > 0]
        if len(non_zero_acts) > 2:
            freq_skewness_dict[feat_idx] = float(abs(skew(non_zero_acts)))
        else:
            freq_skewness_dict[feat_idx] = 0.0

        # UAS
        uas_scores[feat_idx] = (
            UAS_ALPHA * cos_sim_variances[feat_idx] + UAS_BETA * freq_skewness_dict[feat_idx]
        )

    return uas_scores, cos_sim_variances, freq_skewness_dict


def compute_gini_absorption(feature_acts, topk_indices):
    """
    Absorption proxy via Gini concentration.
    Absorption = 1 - Gini_concentration
    - High absorption: flat activation distribution, fires everywhere
    - Low absorption: concentrated activation, fires in specific contexts
    """
    absorption_scores = {}

    for feat_idx in topk_indices:
        acts = feature_acts[0, :, feat_idx].cpu().numpy()
        acts = np.abs(acts)

        if np.sum(acts) == 0:
            absorption_scores[feat_idx] = 0.0
            continue

        # Gini coefficient
        sorted_acts = np.sort(acts)
        n = len(sorted_acts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_acts))) / (n * np.sum(sorted_acts)) - (n + 1) / n

        absorption = 1.0 - max(0.0, min(1.0, gini))
        absorption_scores[feat_idx] = absorption

    return absorption_scores


def compute_reconstruction_loss(resid_pre, feature_acts, sae):
    """Compute MSE reconstruction loss between original and reconstructed residual stream."""
    with torch.no_grad():
        reconstructed = sae.decode(feature_acts)
        mse_loss = torch.nn.functional.mse_loss(resid_pre, reconstructed).item()
    return float(mse_loss)


def collect_tokens(model, n_tokens: int, seed: int = 42):
    """Collect tokens from pile-uncopyrighted dataset."""
    from datasets import load_dataset

    print(f"Collecting {n_tokens} tokens (seed={seed})...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed)

    tokens_list = []
    total_tokens = 0
    for example in dataset:
        text = example["text"]
        tokens = model.to_tokens(text, truncate=True)
        tokens_list.append(tokens)
        total_tokens += tokens.shape[1]
        if total_tokens >= n_tokens:
            break

    all_tokens = torch.cat(tokens_list, dim=0)[:, :n_tokens].to(model.cfg.device)
    print(f"  Collected {all_tokens.shape[0]} sequences, {all_tokens.shape[1]} tokens")
    return all_tokens


def run_experiment_pair(pair):
    """Run UAS validation for a single (model, layer) pair."""
    model_name = pair["model"]
    layer = pair["layer"]
    sae_release = pair["sae_release"]

    print(f"\n{'='*60}")
    print(f"Processing: {model_name} layer {layer}")
    print(f"{'='*60}")

    set_seed(SEED)
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    print(f"Loading model: {model_name}")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    if "gemma" in model_name:
        # Gemma needs special loading
        model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    else:
        model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()
    print(f"  Model loaded: d_model={model.cfg.d_model}")

    # Load SAE
    print(f"Loading SAE: {sae_release} layer {layer}")
    sae_id = f"blocks.{layer}.hook_resid_pre"
    act_sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=DEVICE,
    )
    act_sae.eval()
    print(f"  SAE loaded: d_sae={act_sae.cfg.d_sae}")

    # Collect tokens
    tokens = collect_tokens(model, N_TOKENS, SEED)

    # Get residual activations
    print(f"Getting residual stream activations at layer {layer}...")
    _, cache = model.run_with_cache(
        tokens,
        names_filter=[f"blocks.{layer}.hook_resid_pre"],
        return_type="loss",
    )
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]

    # Get SAE feature activations
    print("Encoding through SAE...")
    with torch.no_grad():
        feature_acts = act_sae.encode(resid_pre)
    print(f"  Feature activations shape: {feature_acts.shape}")
    sparsity = (feature_acts > 0).float().mean().item()
    print(f"  Sparsity: {sparsity:.4f}")

    # Select top-k features by total activation
    total_act = feature_acts[0].sum(dim=0).cpu().numpy()
    topk_indices = np.argsort(total_act)[-N_FEATURES:][::-1]
    print(f"  Top {N_FEATURES} features: {topk_indices[:5]}...")

    # Compute UAS
    print("Computing UAS scores...")
    uas_scores, cos_sim_var, freq_skew = compute_uas(act_sae, feature_acts, topk_indices)

    # Compute Gini absorption (Chanin proxy)
    print("Computing Gini absorption proxy...")
    absorption_scores = compute_gini_absorption(feature_acts, topk_indices)

    # Compute reconstruction loss
    print("Computing reconstruction loss...")
    recon_loss = compute_reconstruction_loss(resid_pre, feature_acts, act_sae)
    print(f"  Reconstruction MSE: {recon_loss:.4f}")

    # Compute Spearman correlation
    uas_vals = np.array([uas_scores.get(i, 0.0) for i in topk_indices])
    absorption_vals = np.array([absorption_scores.get(i, 0.0) for i in topk_indices])

    valid_mask = (uas_vals != 0.0) | (absorption_vals != 0.0)
    n_valid = int(np.sum(valid_mask))

    if n_valid > 5:
        uas_valid = uas_vals[valid_mask]
        absorption_valid = absorption_vals[valid_mask]
        spearman_r, spearman_p = spearmanr(uas_valid, absorption_valid)
        spearman_r = float(spearman_r) if not np.isnan(spearman_r) else 0.0
        spearman_p = float(spearman_p) if not np.isnan(spearman_p) else 1.0
    else:
        spearman_r, spearman_p = 0.0, 1.0
        n_valid = 0

    print(f"  Mean UAS: {np.mean(uas_vals):.4f}")
    print(f"  Mean Gini absorption: {np.mean(absorption_vals):.4f}")
    print(f"  Spearman r (UAS vs Gini): {spearman_r:.4f} (p={spearman_p:.2e})")

    # Top features for inspection
    top_features = [
        {
            "feature_idx": int(idx),
            "uas": float(uas_scores.get(idx, 0.0)),
            "chanin": float(absorption_scores.get(idx, 0.0)),
            "cos_sim_var": float(cos_sim_var.get(idx, 0.0)),
            "freq_skew": float(freq_skew.get(idx, 0.0)),
        }
        for idx in topk_indices[:10]
    ]

    # Cleanup
    del model, act_sae, feature_acts, resid_pre, tokens
    del cache
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": model_name,
        "layer": layer,
        "n_features": N_FEATURES,
        "n_tokens": N_TOKENS,
        "n_valid_correlation": n_valid,
        "sparsity": sparsity,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "mean_uas": float(np.mean(uas_vals)),
        "mean_gini_absorption": float(np.mean(absorption_vals)),
        "reconstruction_mse": recon_loss,
        "top_features": top_features,
        "uas_alpha": UAS_ALPHA,
        "uas_beta": UAS_BETA,
    }


def main():
    print(f"\n{'#'*60}")
    print(f"Full H4 PILOT: UAS Validation")
    print(f"{'#'*60}")
    print(f"Task: {TASK_ID}")
    print(f"GPU: {DEVICE}")
    print(f"N tokens per pair: {N_TOKENS}")
    print(f"N features per pair: {N_FEATURES}")
    print(f"UAS alpha={UAS_ALPHA}, beta={UAS_BETA}")
    print(f"Experiment pairs: {len(EXP_PAIRS)}")
    for p in EXP_PAIRS:
        print(f"  - {p['model']} layer {p['layer']}")
    print(f"Started: {datetime.now().isoformat()}")

    # Write PID
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    results = []
    for pair in EXP_PAIRS:
        try:
            result = run_experiment_pair(pair)
            results.append(result)
        except Exception as e:
            print(f"ERROR on {pair}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "model": pair["model"],
                "layer": pair["layer"],
                "error": str(e),
                "spearman_r": None,
                "spearman_p": None,
            })
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    valid_results = [r for r in results if r.get("spearman_r") is not None]
    spearman_rs = [r["spearman_r"] for r in valid_results]

    mean_r = float(np.mean(spearman_rs)) if spearman_rs else None
    std_r = float(np.std(spearman_rs)) if len(spearman_rs) > 1 else 0.0
    pilot_pass = len(valid_results) >= 2 and mean_r is not None and mean_r > 0.5

    # Save results
    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "uas_params": {"alpha": UAS_ALPHA, "beta": UAS_BETA},
        "n_pairs": len(EXP_PAIRS),
        "n_valid": len(valid_results),
        "mean_spearman_r": mean_r,
        "std_spearman_r": std_r,
        "min_spearman_r": float(np.min(spearman_rs)) if spearman_rs else None,
        "max_spearman_r": float(np.max(spearman_rs)) if spearman_rs else None,
        "pair_results": results,
        "pass_criteria": {
            "threshold": 0.5,
            "mean_r_required": True,
            "pilot_pass": pilot_pass,
        },
        "pilot_pass": pilot_pass,
    }

    out_file = RESULTS_DIR / f"{TASK_ID}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Write DONE marker
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if pilot_pass else "failed",
        "summary": f"full_h4 pilot: mean Spearman r={mean_r}, pilot_pass={pilot_pass}",
        "timestamp": datetime.now().isoformat(),
    }))

    # Write PROGRESS
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "completed",
        "mean_spearman_r": mean_r,
        "pilot_pass": pilot_pass,
        "updated_at": datetime.now().isoformat(),
    }))

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in valid_results:
        print(f"  {r['model']} layer {r['layer']}: r={r['spearman_r']:.4f}, p={r['spearman_p']:.2e}")
    print(f"\nMean Spearman r: {mean_r}")
    print(f"Std Spearman r:  {std_r:.4f}")
    print(f"Min r: {summary.get('min_spearman_r')}, Max r: {summary.get('max_spearman_r')}")
    print(f"Pilot PASS: {pilot_pass}")
    print(f"Results saved: {out_file}")
    print(f"Done: {datetime.now().isoformat()}")

    return summary


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
        done_file.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }))
        sys.exit(1)
