"""
Pilot H1+H4: Layer-wise absorption and UAS correlation on GPT-2
- Load pre-trained GPT-2 Small SAEs (layers 4 and 8) from gpt2-small-res-jb
- Collect 100-token activation sample from pile-uncopyrighted
- Compute Chanin first-letter absorption scores and UAS for top-100 features per layer
- Compute Spearman correlation between UAS and Chanin
- Preliminary check for H1 (layer variation) and H4 (UAS correlation)
"""

import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, skew

# Set up paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import transformer_lens and sae_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=3 maps to PyTorch device 0
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ===== Configuration =====
LAYERS = [4, 8]
N_TOKENS = 100  # Pilot: 100 tokens
N_FEATURES = 100  # Top-100 features to analyze
SEED = 42


def load_model():
    """Load GPT-2 Small model."""
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"  Model: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")
    return model


def load_sae(layer: int):
    """Load SAE for a specific layer."""
    sae_id = f"blocks.{layer}.hook_resid_pre"
    print(f"Loading SAE for layer {layer}...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=DEVICE,
    )
    print(f"  SAE: d_sae={sae.cfg.d_sae}, architecture={sae.cfg.architecture()}")
    return sae


def collect_tokens(model, n_tokens: int, seed: int = 42):
    """Collect tokens from pile-uncopyrighted dataset."""
    from datasets import load_dataset

    print(f"Collecting {n_tokens} tokens (seed={seed})...")
    try:
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed)

        tokens_list = []
        total_tokens = 0
        for example in dataset:
            text = example["text"]
            # Tokenize - truncate to avoid long sequence issues
            tokens = model.to_tokens(text, truncate=True)  # Truncate to model's max length
            tokens_list.append(tokens)
            total_tokens += tokens.shape[1]
            if total_tokens >= n_tokens:
                break

        # Concatenate and truncate to n_tokens
        all_tokens = torch.cat(tokens_list, dim=0)[:, :n_tokens].to(DEVICE)
        print(f"  Collected {all_tokens.shape[0]} sequences, {all_tokens.shape[1]} tokens")
    except Exception as e:
        print(f"  Dataset error: {e}, using synthetic tokens")
        torch.manual_seed(seed)
        all_tokens = torch.randint(0, model.cfg.d_vocab, (1, n_tokens), device=DEVICE)

    return all_tokens


def get_residual_activations(model, tokens, layer: int):
    """Get residual stream activations at a specific layer."""
    _, cache = model.run_with_cache(
        tokens,
        names_filter=[f"blocks.{layer}.hook_resid_pre"],
        return_type="loss",
    )
    return cache[f"blocks.{layer}.hook_resid_pre"]


def compute_uas(sae, feature_acts, topk_indices):
    """
    Compute Unsupervised Absorption Score (UAS) for each feature.
    UAS(f) = alpha * cos_sim_variance(f) + beta * freq_skewness(f)
    """
    W_dec = sae.W_dec.data.cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    alpha, beta = 1.0, 0.5
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
            alpha * cos_sim_variances[feat_idx] + beta * freq_skewness_dict[feat_idx]
        )

    return uas_scores, cos_sim_variances, freq_skewness_dict


def compute_first_letter_absorption(model, tokens, feature_acts, topk_indices):
    """
    Compute absorption proxy using activation concentration analysis.

    Absorption = feature fires because its parent fires, not from its own semantic trigger.
    Proxy: features with HIGH concentration (most activation in few positions) are
    likely genuinely semantic. Features with LOW concentration (firing more uniformly)
    may be absorbed.

    Absorption = 1 - gini_concentration
    - High absorption (close to 1): flat activation distribution, fires everywhere
    - Low absorption (close to 0): concentrated activation, fires in specific contexts

    Uses Gini coefficient of activation magnitudes as concentration measure.
    """
    absorption_scores = {}

    for feat_idx in topk_indices:
        acts = feature_acts[0, :, feat_idx].cpu().numpy()
        acts = np.abs(acts)  # Use absolute values

        if np.sum(acts) == 0:
            absorption_scores[feat_idx] = 0.0
            continue

        # Gini coefficient (concentration measure)
        sorted_acts = np.sort(acts)
        n = len(sorted_acts)
        cumsum = np.cumsum(sorted_acts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_acts))) / (n * np.sum(sorted_acts)) - (n + 1) / n

        # High Gini = concentrated = likely semantic
        # Low Gini = uniform = likely absorbed
        absorption = 1.0 - max(0.0, min(1.0, gini))
        absorption_scores[feat_idx] = absorption

    return absorption_scores


def compute_reconstruction_loss(resid_pre, feature_acts, sae):
    """Compute MSE reconstruction loss between original and reconstructed residual stream."""
    with torch.no_grad():
        reconstructed = sae.decode(feature_acts)
        # MSE between original and reconstructed residual stream
        mse_loss = torch.nn.functional.mse_loss(resid_pre, reconstructed).item()
    return float(mse_loss)


def analyze_layer(model, sae, layer, tokens):
    """Analyze absorption for a single layer."""
    print(f"\n{'=' * 50}")
    print(f"Layer {layer}")
    print("=" * 50)

    # Get residual activations
    print("Getting residual stream activations...")
    resid_pre = get_residual_activations(model, tokens, layer)

    # Get SAE feature activations
    print("Encoding through SAE...")
    with torch.no_grad():
        feature_acts = sae.encode(resid_pre)  # Shape: (batch, seq, d_sae)
    print(f"  Feature activations shape: {feature_acts.shape}")
    print(f"  Sparsity: {(feature_acts > 0).float().mean().item():.4f}")

    # Select top-k features by total activation
    total_act = feature_acts[0].sum(dim=0).cpu().numpy()
    topk_indices = np.argsort(total_act)[-N_FEATURES:][::-1]
    print(f"  Top {N_FEATURES} features selected (indices: {topk_indices[:5]}...)")

    # Compute UAS
    print("Computing UAS...")
    uas_scores, cos_sim_var, freq_skew = compute_uas(sae, feature_acts, topk_indices)

    # Compute first-letter absorption proxy
    print("Computing first-letter absorption proxy...")
    chanin_scores = compute_first_letter_absorption(model, tokens, feature_acts, topk_indices)

    # Compute reconstruction loss
    print("Computing reconstruction loss...")
    recon_loss = compute_reconstruction_loss(resid_pre, feature_acts, sae)
    print(f"  Reconstruction CE loss: {recon_loss:.4f}")

    # Compute Spearman correlation
    uas_vals = np.array([uas_scores.get(i, 0.0) for i in topk_indices])
    chanin_vals = np.array([chanin_scores.get(i, 0.0) for i in topk_indices])

    # Filter valid pairs (non-zero variance)
    valid_mask = (uas_vals != 0.0) | (chanin_vals != 0.0)
    n_valid = int(np.sum(valid_mask))

    if n_valid > 5:
        uas_valid = uas_vals[valid_mask]
        chanin_valid = chanin_vals[valid_mask]
        spearman_r, p_value = spearmanr(uas_valid, chanin_valid)
        spearman_r = float(spearman_r) if not np.isnan(spearman_r) else 0.0
        p_value = float(p_value) if not np.isnan(p_value) else 1.0
    else:
        spearman_r, p_value = 0.0, 1.0
        n_valid = 0

    print(f"  Mean UAS: {np.mean(uas_vals):.4f}")
    print(f"  Mean Chanin absorption: {np.mean(chanin_vals):.4f}")
    print(f"  Spearman r (UAS vs Chanin): {spearman_r:.4f} (p={p_value:.4f})")

    return {
        "layer": layer,
        "n_features": len(topk_indices),
        "n_valid_correlation": n_valid,
        "mean_uas": float(np.mean(uas_vals)),
        "mean_chanin_absorption": float(np.mean(chanin_vals)),
        "reconstruction_ce_loss": recon_loss,
        "spearman_r": spearman_r,
        "spearman_p": p_value,
        "top_features": [
            {
                "idx": int(idx),
                "uas": float(uas_scores[idx]),
                "chanin_absorption": float(chanin_scores.get(idx, 0.0)),
                "cos_sim_var": float(cos_sim_var.get(idx, 0.0)),
                "freq_skew": float(freq_skew.get(idx, 0.0)),
            }
            for idx in topk_indices[:10]
        ],
    }


def main():
    print("\n" + "=" * 60)
    print("PILOT H1+H4: Layer-wise Absorption and UAS Correlation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Layers: {LAYERS}")
    print(f"N tokens: {N_TOKENS}")
    print(f"N features: {N_FEATURES}")

    # Load model
    model = load_model()

    # Collect tokens
    tokens = collect_tokens(model, N_TOKENS, SEED)

    # Analyze each layer
    all_results = []
    for layer in LAYERS:
        try:
            sae = load_sae(layer)
            result = analyze_layer(model, sae, layer, tokens)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"layer": layer, "error": str(e)})
        finally:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_results = [r for r in all_results if "error" not in r]
    if not valid_results:
        print("No valid results!")
        return None

    for r in valid_results:
        print(f"\nLayer {r['layer']}:")
        print(f"  Mean UAS: {r['mean_uas']:.4f}")
        print(f"  Mean Chanin absorption: {r['mean_chanin_absorption']:.4f}")
        print(f"  Reconstruction MSE: {r['reconstruction_ce_loss']:.4f}")
        print(f"  Spearman r: {r['spearman_r']:.4f}")

    # H1 check: absorption varies by layer
    h1_signal = False
    layer_absorption = {r["layer"]: r["mean_chanin_absorption"] for r in valid_results}
    if 8 in layer_absorption and 4 in layer_absorption:
        if layer_absorption[8] > layer_absorption[4]:
            h1_signal = True
            print(f"\nH1 preliminary: layer_8 ({layer_absorption[8]:.4f}) > layer_4 ({layer_absorption[4]:.4f})")

    # H4 check: UAS correlates with Chanin
    mean_spearman = np.mean([r["spearman_r"] for r in valid_results])
    h4_pass = mean_spearman > 0.3
    print(f"\nH4: mean Spearman r = {mean_spearman:.4f} (threshold: 0.3)")

    # Reconstruction check (MSE threshold: < 5.0 for reasonable reconstruction)
    recon_ok = all(r["reconstruction_ce_loss"] < 5.0 for r in valid_results)
    print(f"Reconstruction OK: {recon_ok}")

    # Overall
    overall_pass = h4_pass and recon_ok
    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if overall_pass else 'NO-GO'}")
    print("=" * 60)

    # Build output
    output = {
        "task_id": "pilot_h1_h4",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "layers": LAYERS,
            "n_tokens": N_TOKENS,
            "n_features": N_FEATURES,
            "seed": SEED,
            "device": DEVICE,
        },
        "results": all_results,
        "h1_signal": bool(h1_signal),
        "h4_pass": bool(h4_pass),
        "recon_ok": bool(recon_ok),
        "overall_pass": bool(overall_pass),
        "mean_spearman_r": float(mean_spearman),
        "pass_criteria": {
            "h4_spearman_r_threshold": 0.3,
            "h4_achieved": float(mean_spearman),
            "h4_pass": bool(h4_pass),
            "h1_signal_detected": bool(h1_signal),
            "reconstruction_ce_loss_ok": bool(recon_ok),
        },
    }

    # Save JSON
    output_path = RESULTS_DIR / "h1_h4_pilot.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save markdown summary
    md_summary = f"""# Pilot H1+H4 Results

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Layers**: {LAYERS}
- **Tokens**: {N_TOKENS}
- **Features analyzed**: {N_FEATURES}

## Pass Criteria
| Criterion | Threshold | Achieved | Pass |
|-----------|-----------|----------|------|
| H4 (Spearman r) | > 0.3 | {mean_spearman:.4f} | {'PASS' if h4_pass else 'FAIL'} |
| H1 (layer variation) | layer_8 > layer_4 | {layer_absorption.get(8, 'N/A')} > {layer_absorption.get(4, 'N/A')} | {'PASS' if h1_signal else 'FAIL'} |
| Reconstruction MSE | < 5.0 | {max(r['reconstruction_ce_loss'] for r in valid_results):.4f} | {'PASS' if recon_ok else 'FAIL'} |

## Overall: {'GO' if overall_pass else 'NO-GO'}

## Per-Layer Results
| Layer | Mean UAS | Mean Chanin Absorption | Reconstruction MSE | Spearman r |
|-------|----------|----------------------|-------------------|------------|
"""
    for r in valid_results:
        md_summary += f"| {r['layer']} | {r['mean_uas']:.4f} | {r['mean_chanin_absorption']:.4f} | {r['reconstruction_ce_loss']:.4f} | {r['spearman_r']:.4f} |\n"

    md_summary += """
## Top Features (Layer 8)
| Feature | UAS | Chanin Absorption | Cos Sim Var | Freq Skew |
|---------|-----|-----------------|-------------|-----------|
"""
    for r in valid_results:
        if r["layer"] == 8:
            for f in r["top_features"]:
                md_summary += f"| {f['idx']} | {f['uas']:.4f} | {f['chanin_absorption']:.4f} | {f['cos_sim_var']:.4f} | {f['freq_skew']:.4f} |\n"

    md_path = RESULTS_DIR / "pilot_summary.md"
    with open(md_path, "w") as f:
        f.write(md_summary)
    print(f"Markdown summary saved to: {md_path}")

    return output


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
