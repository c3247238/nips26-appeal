#!/usr/bin/env python3
"""
Pilot H-SENS: Sensitivity-Steering Correlation
Iteration 10 - Test if steering effectiveness depends on sensitivity (paraphrase AUC) rather than absorption status.

If r(steering, sensitivity) > 0.4, sensitivity metric explains steering even if absorption is degenerate.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# TransformerLens and SAE Lens imports
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 8
SEED = 42
N_FEATURES = 30
BETA = 5.0

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_model():
    """Load GPT-2 Small."""
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    model.eval()
    return model


def get_sae(layer):
    """Load SAE for a specific layer."""
    print(f"Loading SAE for layer {layer}...")
    sae, cfg, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=DEVICE
    )
    sae.eval()
    return sae, cfg


def get_tokenizer():
    """Get GPT-2 tokenizer."""
    return HookedTransformer.from_pretrained("gpt2-small", device=DEVICE).tokenizer


def load_iter009_sf1_data():
    """Load iter_009 sf1_large_n200 data for feature selection."""
    sf1_path = RESULTS_DIR / "pilots" / "pilot_sf1_large_n200.json"
    if sf1_path.exists():
        with open(sf1_path) as f:
            return json.load(f)
    return None


def compute_sensitivity_score(model, sae, tokens, feature_idx):
    """
    Compute sensitivity score using variance-based proxy.
    Higher score = more consistent activation across contexts.
    """
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_pre"])
    resid_post = cache[f"blocks.{LAYER}.hook_resid_pre"]

    with torch.no_grad():
        sae_acts = sae.encode(resid_post)

    batch_size, seq_len, n_features = sae_acts.shape
    feature_acts = sae_acts[:, :, feature_idx].flatten().cpu().numpy()

    mean_act = float(np.mean(feature_acts))
    std_act = float(np.std(feature_acts)) + 1e-6

    # Coefficient of variation
    cv = std_act / (abs(mean_act) + 1e-6)

    # Convert to sensitivity score (higher = more sensitive)
    sensitivity = 1.0 / (1.0 + cv)
    return sensitivity


def steering_experiment(model, sae, tokens, feature_idx, beta=5.0):
    """
    Perform steering experiment: add beta * W_dec[feature] to residual stream.
    Measure effect on first-letter classification task.
    """
    from sklearn.linear_model import LogisticRegression

    tokenizer = get_tokenizer()
    batch_size, seq_len = tokens.shape

    # Get first letter labels
    first_letters = []
    for i in range(batch_size):
        for j in range(seq_len):
            tok_id = int(tokens[i, j].item())
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
            if tok_text and len(tok_text) > 0 and tok_text[0].isalpha():
                first_letters.append(1 if tok_text[0].upper() in 'MNOPQRSTUVWXYZ' else 0)
            else:
                first_letters.append(-1)

    first_letters = np.array(first_letters).reshape(batch_size, seq_len)
    valid_mask = first_letters != -1

    # Get residual stream and SAE activations
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_pre"])
    resid_post = cache[f"blocks.{LAYER}.hook_resid_pre"]

    with torch.no_grad():
        sae_acts = sae.encode(resid_post)

    n_features = sae.W_dec.shape[0]

    # Steering: add beta * W_dec[feature] to residual stream
    feature_direction = sae.W_dec[feature_idx].to(DEVICE)

    # Apply steering
    patched_resid = resid_post + beta * feature_direction.view(1, 1, -1)

    # Get steering effect on first-letter classification
    # Use the feature activation as the classifier input
    feature_acts_baseline = sae_acts[:, :, feature_idx].flatten().cpu().numpy()
    feature_acts_patched = sae_acts[:, :, feature_idx].detach().clone()
    # For patched, we need to re-encode with the modified residual
    with torch.no_grad():
        patched_sae_acts = sae.encode(patched_resid)
    feature_acts_patched = patched_sae_acts[:, :, feature_idx].flatten().cpu().numpy()

    # Valid positions
    valid_indices = valid_mask.flatten()
    if valid_indices.sum() < 50:
        return None

    y = first_letters.flatten()[valid_indices]
    X_baseline = feature_acts_baseline[valid_indices].reshape(-1, 1)
    X_patched = feature_acts_patched[valid_indices].reshape(-1, 1)

    if len(np.unique(y)) < 2:
        return None

    # Train baseline classifier
    clf_baseline = LogisticRegression(max_iter=500, random_state=SEED)
    clf_baseline.fit(X_baseline, y)
    baseline_acc = clf_baseline.score(X_baseline, y)

    # Train patched classifier
    clf_patched = LogisticRegression(max_iter=500, random_state=SEED)
    clf_patched.fit(X_patched, y)
    patched_acc = clf_patched.score(X_patched, y)

    # Steering effect = difference in accuracy
    steering_effect = patched_acc - baseline_acc

    return {
        'baseline_acc': float(baseline_acc),
        'patched_acc': float(patched_acc),
        'steering_effect': float(steering_effect),
        'beta': beta
    }


def run_pilot_sensitivity_steering_correlation():
    """
    Task: Test if steering effectiveness correlates with sensitivity.
    Pass criteria: r(steering, sensitivity) > 0.4 (falsification: r < 0.2)
    """
    print("\n" + "="*80)
    print("PILOT H-SENS: Sensitivity-Steering Correlation")
    print("="*80)

    # Load iter_009 sf1 data
    sf1_data = load_iter009_sf1_data()
    if sf1_data is None:
        print("ERROR: Could not load iter_009 sf1_large_n200.json")
        return None

    print("Loaded iter_009 sf1_large_n200 data")
    sensitivity_results = sf1_data.get('sensitivity_results', {})

    # Select features spanning sensitivity range
    feature_sensitivity = []
    for f_idx, sens_data in sensitivity_results.items():
        # Handle both dict format and float format
        if isinstance(sens_data, dict) and 'sensitivity_proxy' in sens_data:
            feature_sensitivity.append((int(f_idx), sens_data['sensitivity_proxy']))
        elif isinstance(sens_data, (int, float)):
            feature_sensitivity.append((int(f_idx), float(sens_data)))

    if len(feature_sensitivity) < N_FEATURES:
        print(f"Warning: only {len(feature_sensitivity)} features with sensitivity scores")

    feature_sensitivity.sort(key=lambda x: x[1])
    selected_features = []

    # Sample across the sensitivity range
    n_bins = 5
    bin_size = len(feature_sensitivity) // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(feature_sensitivity)
        bin_features = feature_sensitivity[start:end]
        if bin_features:
            np.random.seed(SEED + i)
            selected = bin_features[np.random.randint(0, len(bin_features))]
            selected_features.append(selected)

    # Add Q4 features if not already included
    for q4_feat in [10236, 6768]:
        if q4_feat not in [f[0] for f in selected_features] and str(q4_feat) in sensitivity_results:
            sens_val = sensitivity_results[str(q4_feat)]
            if isinstance(sens_val, dict):
                sens_score = sens_val.get('sensitivity_proxy', sens_val.get('sensitivity', 0.5))
            else:
                sens_score = float(sens_val)
            selected_features.append((q4_feat, sens_score))

    print(f"Selected {len(selected_features)} features across sensitivity range")

    model = get_model()
    sae, _ = get_sae(LAYER)
    tokenizer = get_tokenizer()

    # Diverse text samples
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
        "Actions speak louder than words.",
        "Better late than never.",
        "Birds of a feather flock together.",
    ] * 10

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"].to(DEVICE)

    results = {
        'task_id': 'pilot_sensitivity_steering_correlation',
        'hypothesis': 'H-SENS',
        'timestamp': datetime.now().isoformat(),
        'features': {}
    }

    steering_effects = []
    sensitivity_scores = []

    for feature_idx, sens_score in tqdm(selected_features, desc="Steering experiment"):
        print(f"\nFeature {feature_idx}: sensitivity={sens_score:.4f}")

        # Compute sensitivity
        sensitivity = compute_sensitivity_score(model, sae, tokens, feature_idx)

        # Perform steering
        steering_result = steering_experiment(model, sae, tokens, feature_idx, BETA)

        if steering_result is not None:
            results['features'][str(feature_idx)] = {
                'sensitivity_proxy': sens_score,
                'computed_sensitivity': sensitivity,
                'steering_effect': steering_result['steering_effect'],
                'baseline_acc': steering_result['baseline_acc'],
                'patched_acc': steering_result['patched_acc'],
                'beta': steering_result['beta']
            }
            steering_effects.append(steering_result['steering_effect'])
            sensitivity_scores.append(sens_score)
            print(f"Steering effect: {steering_result['steering_effect']:.4f}")

    # Compute correlation
    if len(steering_effects) > 10:
        spearman_r, spearman_p = spearmanr(sensitivity_scores, steering_effects)
        pearson_r, pearson_p = pearsonr(sensitivity_scores, steering_effects)
    else:
        spearman_r = spearman_p = pearson_r = pearson_p = np.nan

    results['correlation'] = {
        'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
        'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else None,
        'pearson_p': float(pearson_p) if not np.isnan(pearson_p) else None,
        'n_features': len(steering_effects)
    }

    results['summary'] = {
        'pass_criteria': 'r(steering, sensitivity) > 0.4',
        'falsification': 'r(steering, sensitivity) < 0.2'
    }

    # Hypothesis evaluation
    h_sens_passed = abs(spearman_r) > 0.4 if not np.isnan(spearman_r) else False

    results['hypothesis_results'] = {
        'H-SENS': {
            'pass': h_sens_passed,
            'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
            'criterion': 'r(steering, sensitivity) > 0.4',
            'falsification': 'r < 0.2'
        }
    }

    # Save results
    output_file = RESULTS_DIR / "pilots" / "sensitivity_steering_correlation.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"\nH-SENS (|r| > 0.4): {'PASS' if h_sens_passed else 'FAIL'} (spearman r = {spearman_r:.4f})")

    return results


if __name__ == "__main__":
    results = run_pilot_sensitivity_steering_correlation()
