#!/usr/bin/env python3
"""
Pilot H-DEC: Decoder Overlap by Sensitivity (Alternative Metric)
Iteration 10 - Test decoder specificity using cosine similarity to top-5 nearest neighbors in W_dec space.

High-sensitivity should have LOWER decoder overlap if sensitivity requires decoder specificity.
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
from scipy.stats import mannwhitneyu, ttest_ind
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
N_FEATURES = 50  # 25 high-sensitivity, 25 low-sensitivity
K_NEIGHBORS = 5
SENSITIVITY_THRESHOLD = 0.65

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


def load_iter009_sf1_data():
    """Load iter_009 sf1_large_n200 data for feature selection."""
    sf1_path = RESULTS_DIR / "pilots" / "pilot_sf1_large_n200.json"
    if sf1_path.exists():
        with open(sf1_path) as f:
            return json.load(f)
    return None


def compute_decoder_overlap(sae, feature_idx, k=5):
    """
    Compute decoder overlap: cosine similarity to top-k nearest neighbors in W_dec space.
    Lower overlap = more decoder specificity.
    """
    W_dec = sae.W_dec.detach().cpu().numpy()  # Shape: (n_features, d_model)
    n_features, d_model = W_dec.shape

    # Get the decoder direction for the target feature
    target_dec = W_dec[feature_idx]

    # Compute cosine similarity to all other features
    target_norm = np.linalg.norm(target_dec)
    all_norms = np.linalg.norm(W_dec, axis=1)

    # Cosine similarity
    cos_sims = np.dot(W_dec, target_dec) / (all_norms * target_norm + 1e-8)

    # Exclude self (set to -inf to ignore)
    cos_sims[feature_idx] = -np.inf

    # Find top-k neighbors
    top_k_indices = np.argsort(cos_sims)[-k:]
    top_k_sims = cos_sims[top_k_indices]

    # Mean overlap with top-k
    mean_overlap = float(np.mean(top_k_sims))

    return mean_overlap, top_k_indices.tolist()


def run_pilot_decoder_overlap_alternative():
    """
    Task: Test decoder specificity by sensitivity group.
    Pass criteria: High-sensitivity features have LOWER decoder overlap (p < 0.05)
    """
    print("\n" + "="*80)
    print("PILOT H-DEC: Decoder Overlap by Sensitivity (Alternative Metric)")
    print("="*80)

    # Load iter_009 sf1 data
    sf1_data = load_iter009_sf1_data()
    if sf1_data is None:
        print("ERROR: Could not load iter_009 sf1_large_n200.json")
        return None

    print("Loaded iter_009 sf1_large_n200 data")
    sensitivity_results = sf1_data.get('sensitivity_results', {})

    # Classify features by sensitivity
    high_sens_features = []
    low_sens_features = []

    for f_idx, sens_data in sensitivity_results.items():
        # Handle both dict format and float format
        if isinstance(sens_data, dict) and 'sensitivity_proxy' in sens_data:
            sens_score = sens_data['sensitivity_proxy']
        elif isinstance(sens_data, (int, float)):
            sens_score = float(sens_data)
        else:
            continue

        if sens_score >= SENSITIVITY_THRESHOLD:
            high_sens_features.append((int(f_idx), sens_score))
        else:
            low_sens_features.append((int(f_idx), sens_score))

    print(f"High-sensitivity features (AUC >= {SENSITIVITY_THRESHOLD}): {len(high_sens_features)}")
    print(f"Low-sensitivity features (AUC < {SENSITIVITY_THRESHOLD}): {len(low_sens_features)}")

    # Sample equal numbers
    n_per_group = min(N_FEATURES // 2, len(high_sens_features), len(low_sens_features))
    np.random.seed(SEED)
    selected_high = [high_sens_features[i] for i in np.random.choice(len(high_sens_features), n_per_group, replace=False)]
    selected_low = [low_sens_features[i] for i in np.random.choice(len(low_sens_features), n_per_group, replace=False)]

    # Add Q4 features to high-sensitivity group
    for q4_feat in [10236, 6768]:
        if q4_feat not in [f[0] for f in selected_high] and str(q4_feat) in sensitivity_results:
            selected_high.append((q4_feat, sensitivity_results[str(q4_feat)]['sensitivity_proxy']))

    print(f"Selected {len(selected_high)} high-sensitivity and {len(selected_low)} low-sensitivity features")

    model = get_model()
    sae, _ = get_sae(LAYER)

    results = {
        'task_id': 'pilot_decoder_overlap_alternative',
        'hypothesis': 'H-DEC',
        'timestamp': datetime.now().isoformat(),
        'high_sensitivity': {},
        'low_sensitivity': {}
    }

    # Compute decoder overlap for high-sensitivity group
    print("\n--- High-Sensitivity Features ---")
    high_overlaps = []
    for feature_idx, sens_score in tqdm(selected_high, desc="High-sens overlap"):
        mean_overlap, neighbors = compute_decoder_overlap(sae, feature_idx, K_NEIGHBORS)
        results['high_sensitivity'][str(feature_idx)] = {
            'sensitivity_proxy': sens_score,
            'mean_overlap': mean_overlap,
            'neighbors': neighbors
        }
        high_overlaps.append(mean_overlap)
        print(f"Feature {feature_idx}: sens={sens_score:.4f}, overlap={mean_overlap:.4f}")

    # Compute decoder overlap for low-sensitivity group
    print("\n--- Low-Sensitivity Features ---")
    low_overlaps = []
    for feature_idx, sens_score in tqdm(selected_low, desc="Low-sens overlap"):
        mean_overlap, neighbors = compute_decoder_overlap(sae, feature_idx, K_NEIGHBORS)
        results['low_sensitivity'][str(feature_idx)] = {
            'sensitivity_proxy': sens_score,
            'mean_overlap': mean_overlap,
            'neighbors': neighbors
        }
        low_overlaps.append(mean_overlap)
        print(f"Feature {feature_idx}: sens={sens_score:.4f}, overlap={mean_overlap:.4f}")

    # Compute statistics
    high_mean = np.mean(high_overlaps)
    high_std = np.std(high_overlaps)
    low_mean = np.mean(low_overlaps)
    low_std = np.std(low_overlaps)

    # Statistical test
    if len(high_overlaps) >= 5 and len(low_overlaps) >= 5:
        # Mann-Whitney U test
        u_stat, mw_p = mannwhitneyu(high_overlaps, low_overlaps, alternative='two-sided')
        # T-test
        t_stat, t_p = ttest_ind(high_overlaps, low_overlaps)
    else:
        u_stat = mw_p = t_stat = t_p = np.nan

    results['summary'] = {
        'high_sens_mean_overlap': float(high_mean),
        'high_sens_std_overlap': float(high_std),
        'low_sens_mean_overlap': float(low_mean),
        'low_sens_std_overlap': float(low_std),
        'difference': float(high_mean - low_mean),
        'mannwhitney_p': float(mw_p) if not np.isnan(mw_p) else None,
        'ttest_p': float(t_p) if not np.isnan(t_p) else None,
        'n_high_sens': len(high_overlaps),
        'n_low_sens': len(low_overlaps),
        'pass_criteria': 'High-sens < Low-sens overlap (p < 0.05)'
    }

    # Hypothesis evaluation
    # High-sensitivity should have LOWER decoder overlap
    # So we want high_mean < low_mean
    h_dec_passed = (high_mean < low_mean) and (mw_p < 0.05) if not np.isnan(mw_p) else False

    results['hypothesis_results'] = {
        'H-DEC': {
            'pass': h_dec_passed,
            'high_sens_mean': float(high_mean),
            'low_sens_mean': float(low_mean),
            'difference': float(high_mean - low_mean),
            'p_value': float(mw_p) if not np.isnan(mw_p) else None,
            'criterion': 'High-sens < Low-sens overlap (p < 0.05)',
            'prediction': 'High-sensitivity requires decoder specificity (lower overlap)'
        }
    }

    # Save results
    output_file = RESULTS_DIR / "pilots" / "decoder_overlap_alternative.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"\nH-DEC (High-sens < Low-sens overlap, p < 0.05): {'PASS' if h_dec_passed else 'FAIL'}")
    print(f"High-sensitivity mean overlap: {high_mean:.4f} (std: {high_std:.4f})")
    print(f"Low-sensitivity mean overlap: {low_mean:.4f} (std: {low_std:.4f})")
    print(f"Mann-Whitney p-value: {mw_p:.6f}")

    return results


if __name__ == "__main__":
    results = run_pilot_decoder_overlap_alternative()
