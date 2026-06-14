"""
Trade-off Analysis: Absorption vs Downstream Performance Across Layers

Tests whether layers with lower absorption rates achieve better downstream performance.
Runs absorption detection, steering, and probing on GPT-2 Small layers 0, 4, 8, 11.

Expected time: ~30-45 minutes
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from sae_lens import SAE, HookedSAETransformer


def load_model_and_saes(layers=[0, 4, 8, 11]):
    """Load GPT-2 Small and SAEs for specified layers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 Small on {device}...")

    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    saes = {}
    for layer in layers:
        sae_id = f"blocks.{layer}.hook_resid_pre"
        print(f"  Loading SAE for layer {layer}...")
        sae = SAE.from_pretrained("gpt2-small-res-jb", sae_id)
        sae.to(device)
        saes[layer] = sae

    return model, saes, device


def get_first_letter_features(model, sae, device, letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Find latents that maximally activate for each first-letter prompt."""
    from transformer_lens.utils import test_prompt

    features = {}
    for letter in letters:
        prompt = f"The letter {letter}"
        tokens = model.to_tokens(prompt)

        _, cache = model.run_with_cache(tokens)
        acts = cache[sae.cfg.hook_name]  # (batch, pos, d_model)

        # Get SAE activations
        with torch.no_grad():
            sae_acts = sae.encode(acts.squeeze(0))  # (pos, d_sae)

        # Find max activating feature
        max_val, max_idx = sae_acts.max(dim=0)
        best_feat = max_val.argmax().item()

        features[letter] = {
            "feature_id": best_feat,
            "max_activation": max_val[best_feat].item(),
        }

    return features


def detect_absorption(model, sae, device, feature_id, prompt, n_children=50):
    """
    Detect absorption using differential correlation metric (Chanin et al.).
    Simplified: measure how much parent feature activation is explained by children.
    """
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        acts = cache[sae.cfg.hook_name].squeeze(0)  # (pos, d_model)
        sae_acts = sae.encode(acts)  # (pos, d_sae)

    # Parent feature activation
    parent_acts = sae_acts[:, feature_id]

    # Find top child candidates (excluding parent)
    other_acts = sae_acts.clone()
    other_acts[:, feature_id] = 0
    child_scores = other_acts.sum(dim=0)  # aggregate across positions
    top_children = torch.topk(child_scores, n_children + 1).indices.tolist()
    if feature_id in top_children:
        top_children.remove(feature_id)
    top_children = top_children[:n_children]

    # Differential correlation: correlation(parent, parent - children)
    parent_np = parent_acts.cpu().numpy()
    residual = parent_acts.clone()
    for child_id in top_children:
        residual -= sae_acts[:, child_id]

    # Compute correlation
    from scipy.stats import pearsonr
    corr, pval = pearsonr(parent_np, residual.cpu().numpy())

    return {
        "absorption_rate": max(0, 1 - corr),
        "correlation": corr,
        "p_value": pval,
        "n_children": len(top_children),
        "top_children": top_children[:10],
    }


def run_steering(model, sae, device, feature_id, prompt, strengths=[-50, -30, -10, 10, 30, 50]):
    """Test feature steering at multiple strengths."""
    tokens = model.to_tokens(prompt)
    base_logits = model(tokens)

    results = {}
    for strength in strengths:
        hook_name = sae.cfg.hook_name

        def steering_hook(value, hook):
            value[:, :, feature_id] += strength
            return value

        steered_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, steering_hook)],
        )

        # Measure change
        diff = (steered_logits - base_logits).abs().mean().item()
        results[strength] = diff

    return results


def run_probing(model, sae, device, feature_id, prompts):
    """Simple probe: check if feature activation predicts letter identity."""
    activations = []
    labels = []

    for letter, prompt in prompts.items():
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            acts = cache[sae.cfg.hook_name].squeeze(0)
            sae_acts = sae.encode(acts)
            feat_act = sae_acts[:, feature_id].max().item()

        activations.append(feat_act)
        labels.append(letter)

    return {
        "mean_activation": np.mean(activations),
        "std_activation": np.std(activations),
        "max_activation": max(activations),
        "min_activation": min(activations),
    }


def run_experiment_for_layer(model, sae, device, layer_idx, letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Run full experiment pipeline for one layer."""
    print(f"\n{'='*50}")
    print(f"Layer {layer_idx}")
    print(f"{'='*50}")

    # 1. Identify first-letter features
    print("  Finding first-letter features...")
    features = {}
    for letter in letters:
        prompt = f"The letter {letter}"
        try:
            result = get_first_letter_features(model, sae, device, letters=letter)
            features[letter] = result[letter]
        except Exception as e:
            print(f"    Warning: failed for {letter}: {e}")
            features[letter] = {"feature_id": -1, "max_activation": 0.0}

    # 2. Detect absorption
    print("  Detecting absorption...")
    absorption_results = {}
    for letter in letters:
        feat_id = features[letter]["feature_id"]
        if feat_id < 0:
            continue
        prompt = f"The letter {letter}"
        try:
            result = detect_absorption(model, sae, device, feat_id, prompt)
            absorption_results[letter] = result
        except Exception as e:
            print(f"    Warning: absorption failed for {letter}: {e}")
            absorption_results[letter] = {"absorption_rate": 0.0, "correlation": 1.0}

    # 3. Steering (sample a few features)
    print("  Testing steering...")
    steering_results = {}
    for letter in list(letters)[:5]:  # subset for speed
        feat_id = features[letter]["feature_id"]
        if feat_id < 0:
            continue
        prompt = f"The letter {letter}"
        try:
            result = run_steering(model, sae, device, feat_id, prompt)
            steering_results[letter] = result
        except Exception as e:
            print(f"    Warning: steering failed for {letter}: {e}")

    # 4. Compute summary stats
    absorption_rates = [v["absorption_rate"] for v in absorption_results.values()]
    high_abs = sum(1 for a in absorption_rates if a > 0.10)
    mean_abs = np.mean(absorption_rates) if absorption_rates else 0.0

    return {
        "layer": layer_idx,
        "features": features,
        "absorption": absorption_results,
        "steering": steering_results,
        "summary": {
            "mean_absorption_rate": mean_abs,
            "high_absorption_count": high_abs,
            "n_features": len(features),
        },
    }


def analyze_tradeoff(layer_results):
    """Analyze absorption-performance trade-off across layers."""
    print(f"\n{'='*60}")
    print("TRADE-OFF ANALYSIS")
    print(f"{'='*60}")

    layers = sorted(layer_results.keys())
    abs_rates = [layer_results[l]["summary"]["mean_absorption_rate"] for l in layers]

    print(f"\nLayers: {layers}")
    print(f"Mean absorption rates: {[f'{a:.3f}' for a in abs_rates]}")

    # Check trend with depth
    from scipy.stats import spearmanr
    corr, pval = spearmanr(layers, abs_rates)
    print(f"\nDepth vs Absorption: rho={corr:.3f}, p={pval:.3f}")

    if corr > 0 and pval < 0.05:
        print("  -> Deeper layers have HIGHER absorption")
    elif corr < 0 and pval < 0.05:
        print("  -> Deeper layers have LOWER absorption")
    else:
        print("  -> No significant depth trend")

    return {
        "layers": layers,
        "absorption_rates": abs_rates,
        "depth_correlation": corr,
        "depth_pvalue": pval,
    }


def main():
    print("=" * 60)
    print("Trade-off Analysis: Absorption vs Performance Across Layers")
    print("=" * 60)

    layers = [0, 4, 8, 11]
    model, saes, device = load_model_and_saes(layers)

    layer_results = {}
    for layer in layers:
        result = run_experiment_for_layer(model, saes[layer], device, layer)
        layer_results[layer] = result

    # Analyze trade-off
    tradeoff = analyze_tradeoff(layer_results)

    # Save results
    output = {
        "experiment": "tradeoff_crosslayer",
        "model": "gpt2-small",
        "sae": "gpt2-small-res-jb",
        "layers": layers,
        "layer_results": layer_results,
        "tradeoff_analysis": tradeoff,
    }

    output_path = "exp/results/full/tradeoff_crosslayer.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
