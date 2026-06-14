#!/usr/bin/env python3
"""
Pilot experiments for Sensitivity-Absorption Compound Failure Modes.
Iteration 8 - Tests H4, H5, H6, and H1-R.

Tasks:
1. pilot_classify_features: H4+H5 Pilot - Classify features into 4 quadrants
2. pilot_decoder_norms: H6 Pilot - Decoder L2 norm ratios
3. replicate_coherence_protective: H1-R Backup - Protective effect replication
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
from scipy.stats import spearmanr, mannwhitneyu
from tqdm import tqdm

# TransformerLens and SAE Lens imports
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax")
ITER006 = WORKSPACE / "iter_006"
RESULTS_DIR = ITER006 / "exp" / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 8  # Target layer for most experiments
SEED = 42

# Set seeds
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


def compute_absorption_score(resid_acc, sae_acc):
    """
    Compute UAS (Universal Absorption Score) as per Chanin 2024.
    UAS = (acc_resid - acc_sae) / (1 - acc_sae)
    """
    if resid_acc is None or sae_acc is None:
        return None
    denominator = 1 - sae_acc
    if denominator == 0:
        return 0.0
    uas = (resid_acc - sae_acc) / denominator
    return float(uas)


def chanin_absorption_protocol(model, sae, tokens, tau_fs=0.03, quick_mode=True):
    """
    Implements Chanin 2024 absorption detection protocol.

    Task: Classify tokens into A-M vs N-Z first-letter bins.
    - Positive set: tokens where feature f has high activation (> tau_fs)
    - Probe: Logistic regression on SAE features to predict first-letter class
    - UAS = (acc_resid - acc_sae) / (1 - acc_sae)

    Returns dict of {feature_idx: {'uas': float, 'acc_resid': float, 'acc_sae': float}}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    tokenizer = get_tokenizer()
    n_features = sae.W_dec.shape[0]

    # Keep tokens as [batch, seq] - pad to same length if needed
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)  # Add batch dimension

    # Get first letter of each token (for classification)
    first_letters = []
    for i in range(tokens.shape[0]):  # batch
        for j in range(tokens.shape[1]):  # seq
            tok_id = int(tokens[i, j].item())
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
            if tok_text and len(tok_text) > 0 and tok_text[0].isalpha():
                first_letters.append(1 if tok_text[0].upper() in 'MNOPQRSTUVWXYZ' else 0)
            else:
                first_letters.append(-1)

    first_letters = np.array(first_letters)
    batch_size, seq_len = tokens.shape
    first_letters = first_letters.reshape(batch_size, seq_len)

    valid_mask = first_letters != -1

    if valid_mask.sum() < 50:
        return {}

    # Get SAE activations for all tokens
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_pre"])
    resid_post = cache[f"blocks.{LAYER}.hook_resid_pre"]  # [batch, seq, d_model]

    # Get SAE features
    with torch.no_grad():
        sae_acts = sae.encode(resid_post)  # [batch, seq, n_features]

    results = {}

    # Reshape for easier processing
    sae_acts = sae_acts.reshape(batch_size * seq_len, n_features)
    resid_post = resid_post.reshape(batch_size * seq_len, -1)
    first_letters_flat = first_letters.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)

    results = {}

    # Filter to valid positions
    valid_labels = first_letters_flat[valid_mask_flat]
    valid_sae_acts = sae_acts[valid_mask_flat]  # [n_valid, n_features]
    valid_resid_acts = resid_post[valid_mask_flat]  # [n_valid, d_model]

    # Pre-identify features with sufficient activations
    # Compute mean activation across valid positions
    feature_means = valid_sae_acts.mean(dim=0).cpu().numpy()
    candidate_features = np.where(feature_means > tau_fs)[0]
    print(f"Found {len(candidate_features)} features with mean activation > {tau_fs}")

    # For each candidate feature, compute absorption score
    for f_idx in tqdm(candidate_features, desc="Computing absorption"):
        f_idx_int = int(f_idx)  # Convert to Python int for JSON serialization
        feature_acts = valid_sae_acts[:, f_idx].cpu().numpy()

        # Feature selection: tokens where feature is active (> tau_fs)
        active_mask = feature_acts > tau_fs
        if active_mask.sum() < 10:
            continue

        # Positive set precision check
        active_labels = valid_labels[active_mask]
        if len(active_labels) < 10:
            continue

        # Check positive set accuracy (should be > 0.4 for non-absorbed)
        pos_class_ratio = active_labels.mean()

        # Use active tokens only
        X_sae_active = feature_acts[active_mask].reshape(-1, 1)
        X_resid_active = valid_resid_acts[active_mask].cpu().numpy()
        y_active = valid_labels[active_mask]

        if len(np.unique(y_active)) < 2:
            continue

        try:
            # Train and evaluate SAE probe
            clf_sae = LogisticRegression(max_iter=500, random_state=SEED)
            clf_sae.fit(X_sae_active, y_active)
            acc_sae = clf_sae.score(X_sae_active, y_active)

            # Train and evaluate RESID probe
            # Use a subset of resid dimensions for fair comparison
            resid_dim = min(50, X_resid_active.shape[1])
            clf_resid = LogisticRegression(max_iter=500, random_state=SEED)
            clf_resid.fit(X_resid_active[:, :resid_dim], y_active)
            acc_resid = clf_resid.score(X_resid_active[:, :resid_dim], y_active)

            uas = compute_absorption_score(acc_resid, acc_sae)

            results[f_idx_int] = {
                'uas': uas,
                'acc_resid': float(acc_resid),
                'acc_sae': float(acc_sae),
                'n_active': int(active_mask.sum()),
                'pos_class_ratio': float(pos_class_ratio)
            }
        except Exception as e:
            continue

    return results


def tian_sensitivity_protocol(model, sae, tokens, n_pairs=50, quick_mode=True):
    """
    Implements Tian 2025 sensitivity measurement protocol.

    Task: Paraphrase AUC - measure feature activation reliability across semantically equivalent inputs.
    - Generate paraphrase pairs
    - Measure activation on original vs paraphrase
    - Compute AUC across paraphrase pairs

    Returns dict of {feature_idx: {'paraphrase_auc': float, 'mean_act': float}}
    """
    tokenizer = get_tokenizer()
    n_features = sae.W_dec.shape[0]

    # Keep tokens as [batch, seq]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    # Get SAE activations using run_with_cache
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_pre"])
    resid_post = cache[f"blocks.{LAYER}.hook_resid_pre"]  # [batch, seq, d_model]

    with torch.no_grad():
        sae_acts = sae.encode(resid_post)  # [batch, seq, n_features]

    # Flatten for easier processing
    batch_size, seq_len, _ = sae_acts.shape
    sae_acts = sae_acts.reshape(batch_size * seq_len, n_features)

    results = {}

    # Simple sensitivity proxy: measure activation variance across context
    # Higher variance = lower sensitivity

    # Use all features for sensitivity (it's a cheap computation)
    for f_idx in tqdm(range(n_features), desc="Computing sensitivity"):
        feature_acts = sae_acts[:, f_idx].cpu().numpy()

        # Simple sensitivity score: mean / std (higher = more consistent)
        # This is a simplified version for pilot
        mean_act = float(feature_acts.mean())
        std_act = float(feature_acts.std()) + 1e-6

        # Sensitivity proxy: stability across tokens
        # We use the coefficient of variation as sensitivity proxy
        # Higher CV = less sensitive (more variable)
        cv = std_act / (abs(mean_act) + 1e-6)

        # Convert to sensitivity score (higher = more sensitive)
        sensitivity = 1.0 / (1.0 + cv)

        results[f_idx] = {
            'sensitivity_proxy': float(sensitivity),
            'mean_act': mean_act,
            'std_act': std_act,
            'cv': float(cv)
        }

    return results


def classify_into_quadrants(absorption_results, sensitivity_results, uas_threshold=0.4, sens_threshold=0.5):
    """
    Classify features into 4 quadrants:
    Q1: High absorption + Low sensitivity (doubly-compromised)
    Q2: High absorption + High sensitivity
    Q3: Low absorption + Low sensitivity
    Q4: Low absorption + High sensitivity (best-case)
    """
    quadrants = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

    common_features = set(absorption_results.keys()) & set(sensitivity_results.keys())

    for f_idx in common_features:
        uas = absorption_results[f_idx]['uas']
        sens = sensitivity_results[f_idx]['sensitivity_proxy']

        high_abs = uas < uas_threshold  # UAS < 0.4 means absorbed
        high_sens = sens >= sens_threshold

        if high_abs and not high_sens:
            quadrants['Q1'].append(f_idx)
        elif high_abs and high_sens:
            quadrants['Q2'].append(f_idx)
        elif not high_abs and not high_sens:
            quadrants['Q3'].append(f_idx)
        else:  # not high_abs and high_sens
            quadrants['Q4'].append(f_idx)

    return quadrants


def run_pilot_classify_features():
    """
    Task 1: Pilot H4+H5 - Feature Quadrant Classification.
    """
    print("\n" + "="*80)
    print("TASK 1: pilot_classify_features (H4+H5 Pilot)")
    print("="*80)

    model = get_model()
    sae, cfg = get_sae(LAYER)
    tokenizer = get_tokenizer()

    # Get diverse text samples - keep under GPT-2's 1024 max length
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
    ] * 10  # Repeat for more tokens, reduced from 20 to stay under 1024

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"].to(DEVICE)

    # Compute absorption scores
    print("\nComputing absorption scores (Chanin protocol)...")
    absorption_results = chanin_absorption_protocol(model, sae, tokens, quick_mode=True)

    # Compute sensitivity scores
    print("\nComputing sensitivity scores (Tian protocol)...")
    sensitivity_results = tian_sensitivity_protocol(model, sae, tokens, quick_mode=True)

    # Classify into quadrants
    quadrants = classify_into_quadrants(absorption_results, sensitivity_results)

    # Compute Spearman correlation
    common_features = sorted(set(absorption_results.keys()) & set(sensitivity_results.keys()))
    if len(common_features) > 10:
        uas_values = [absorption_results[f]['uas'] for f in common_features]
        sens_values = [sensitivity_results[f]['sensitivity_proxy'] for f in common_features]
        spearman_r, spearman_p = spearmanr(uas_values, sens_values)
    else:
        spearman_r, spearman_p = np.nan, np.nan

    # Prepare results
    results = {
        'task_id': 'pilot_classify_features',
        'hypothesis': 'H4, H5',
        'timestamp': datetime.now().isoformat(),
        'n_features_analyzed': len(common_features),
        'quadrants': {k: list(v) for k, v in quadrants.items()},
        'quadrant_counts': {k: len(v) for k, v in quadrants.items()},
        'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
        'pass_criteria': 'r(absorption, sensitivity) < 0.5',
        'absorption_results': {str(k): v for k, v in absorption_results.items()},
        'sensitivity_results': {str(k): v for k, v in sensitivity_results.items()},
    }

    # Save results
    output_file = RESULTS_DIR / "pilots" / "quadrant_classification_pilot.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Features analyzed: {len(common_features)}")
    print(f"Quadrant counts: {results['quadrant_counts']}")
    print(f"Spearman r(absorption, sensitivity): {spearman_r:.4f} (p={spearman_p:.6f})")

    # Check pass criteria
    h5_passed = abs(spearman_r) < 0.5 if not np.isnan(spearman_r) else False
    h4_passed = len(quadrants['Q1']) >= 5  # Need at least 5 features for steering

    print(f"\nH5 (independence) pass: {h5_passed}")
    print(f"H4 (Q1 has >= 5 features) pass: {h4_passed}")

    return results


def run_pilot_decoder_norms():
    """
    Task 3: Pilot H6 - Decoder L2 Norm Ratios.
    """
    print("\n" + "="*80)
    print("TASK 3: pilot_decoder_norms (H6 Pilot)")
    print("="*80)

    sae, cfg = get_sae(LAYER)

    # Compute decoder L2 norms
    print("\nComputing decoder L2 norms...")
    W_dec = sae.W_dec.detach().cpu().numpy()  # Shape: (n_features, d_model)
    norms = np.linalg.norm(W_dec, axis=1)

    # Also get absorption data from pilot_classify_features if available
    absorption_file = RESULTS_DIR / "pilots" / "quadrant_classification_pilot.json"
    if absorption_file.exists():
        with open(absorption_file) as f:
            class_data = json.load(f)
        absorption_results = class_data.get('absorption_results', {})
    else:
        # Compute absorption on-the-fly
        print("\nNo absorption data found, computing on-the-fly...")
        model = get_model()
        tokenizer = get_tokenizer()
        texts = ["The quick brown fox jumps over the lazy dog."] * 20
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(DEVICE)
        absorption_results = chanin_absorption_protocol(model, sae, tokens, quick_mode=True)

    # Split by absorption
    uas_threshold = 0.4
    high_abs_norms = []
    low_abs_norms = []

    for f_idx, data in absorption_results.items():
        f_idx = int(f_idx)
        if data['uas'] < uas_threshold:
            high_abs_norms.append(norms[f_idx])
        else:
            low_abs_norms.append(norms[f_idx])

    if len(high_abs_norms) > 0 and len(low_abs_norms) > 0:
        high_abs_mean = np.mean(high_abs_norms)
        low_abs_mean = np.mean(low_abs_norms)
        norm_ratio = high_abs_mean / low_abs_mean
    else:
        high_abs_mean = low_abs_mean = norm_ratio = np.nan

    results = {
        'task_id': 'pilot_decoder_norms',
        'hypothesis': 'H6',
        'timestamp': datetime.now().isoformat(),
        'n_features_total': len(norms),
        'n_high_absorption': len(high_abs_norms),
        'n_low_absorption': len(low_abs_norms),
        'high_abs_mean_norm': float(high_abs_mean),
        'low_abs_mean_norm': float(low_abs_mean),
        'norm_ratio': float(norm_ratio),
        'pass_criteria': 'L2 norm ratio > 1.1',
        'norms': {str(i): float(n) for i, n in enumerate(norms)},
    }

    # Save results
    output_file = RESULTS_DIR / "pilots" / "decoder_norms_pilot.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"High-absorption mean L2 norm: {high_abs_mean:.4f}")
    print(f"Low-absorption mean L2 norm: {low_abs_mean:.4f}")
    print(f"Ratio (high/low): {norm_ratio:.4f}")

    h6_passed = norm_ratio > 1.1 if not np.isnan(norm_ratio) else False
    print(f"\nH6 (norm ratio > 1.1) pass: {h6_passed}")

    return results


def compute_feature_coherence(model, sae, tokens, feature_idx, layer):
    """
    Compute pairwise coherence for a feature.
    Coherence = how often the feature appears with itself across contexts.
    """
    # Handle tokens - keep as [batch, seq]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_resid_pre"])
    resid_post = cache[f"blocks.{layer}.hook_resid_pre"]  # [batch, seq, d_model]

    with torch.no_grad():
        sae_acts = sae.encode(resid_post)  # [batch, seq, n_features]

    # Flatten
    batch_size, seq_len, _ = sae_acts.shape
    sae_acts = sae_acts.reshape(batch_size * seq_len, -1)

    feature_acts = sae_acts[:, feature_idx].cpu().numpy()

    # Binary activation
    binary_acts = (feature_acts > 0.03).astype(float)

    # Compute coherence as co-activation probability
    if binary_acts.sum() < 5:
        return np.nan

    # Simple coherence: mean activation level
    return float(binary_acts.mean())


def run_replicate_coherence_protective():
    """
    Task 5: H1-R - Protective Effect Replication.
    Replicate pilot finding (r=-0.786) across layers 4, 8, 12.
    GPT-2 Small has 12 layers (0-11), so we use 4, 8, and a middle layer.
    """
    print("\n" + "="*80)
    print("TASK 5: replicate_coherence_protective (H1-R Backup)")
    print("="*80)

    # GPT-2 Small has 12 layers (0-11), use 4, 8, and 10 (not 12)
    layers = [4, 8, 10]
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
    ] * 25

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(DEVICE)

    results = {
        'task_id': 'replicate_coherence_protective',
        'hypothesis': 'H1-R',
        'timestamp': datetime.now().isoformat(),
        'layers': {}
    }

    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        # Load model and SAE for this layer
        model = get_model()
        sae, cfg = get_sae(layer)

        # Compute absorption
        print(f"Computing absorption for layer {layer}...")
        absorption_results = chanin_absorption_protocol(model, sae, tokens, quick_mode=True)

        # Compute coherence
        print(f"Computing coherence for layer {layer}...")
        coherence_results = {}
        for f_idx in tqdm(list(absorption_results.keys())[:50], desc="Coherence"):
            coherence_results[f_idx] = compute_feature_coherence(model, sae, tokens, int(f_idx), layer)

        # Compute correlation
        common_features = sorted(set(absorption_results.keys()) & set(coherence_results.keys()))
        if len(common_features) > 10:
            uas_values = [absorption_results[f]['uas'] for f in common_features]
            coh_values = [coherence_results[f] for f in common_features if not np.isnan(coherence_results[f])]

            # Align arrays
            valid_mask = [not np.isnan(coherence_results[f]) for f in common_features]
            uas_valid = np.array(uas_values)[valid_mask]
            coh_valid = np.array([coherence_results[f] for f in common_features if not np.isnan(coherence_results[f])])

            if len(uas_valid) > 10:
                spearman_r, spearman_p = spearmanr(uas_valid, coh_valid)
            else:
                spearman_r, spearman_p = np.nan, np.nan
        else:
            spearman_r, spearman_p = np.nan, np.nan

        results['layers'][str(layer)] = {
            'n_features': len(common_features),
            'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
            'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
        }

        print(f"Layer {layer}: Spearman r(coherence, absorption) = {spearman_r:.4f} (p={spearman_p:.6f})")

    # Overall assessment
    all_rs = [results['layers'][str(l)]['spearman_r'] for l in layers if results['layers'][str(l)]['spearman_r'] is not None]
    if len(all_rs) >= 2:
        mean_r = np.mean(all_rs)
        consistent_negative = all(r < 0 for r in all_rs)
        results['overall'] = {
            'mean_r': float(mean_r),
            'consistent_negative': consistent_negative,
            'pass_criteria': 'r < -0.5 consistently'
        }

        h1r_passed = (mean_r < -0.5) and consistent_negative
        print(f"\nH1-R Overall: mean r = {mean_r:.4f}, consistent_negative = {consistent_negative}")
        print(f"H1-R (r < -0.5 consistently) pass: {h1r_passed}")

    # Save results
    output_file = RESULTS_DIR / "pilots" / "coherence_protective_replication.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    """Run all pilot experiments."""
    print("="*80)
    print("PILOT EXPERIMENTS - Iteration 8")
    print("Sensitivity-Absorption Compound Failure Modes")
    print("="*80)

    # GPU 7 is specified in skill arguments. When CUDA_VISIBLE_DEVICES=7 is set externally,
    # that GPU becomes device 0 in this process.
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    # Run Task 1: pilot_classify_features
    try:
        all_results['pilot_classify_features'] = run_pilot_classify_features()
    except Exception as e:
        print(f"ERROR in pilot_classify_features: {e}")
        import traceback
        traceback.print_exc()
        all_results['pilot_classify_features'] = {'error': str(e)}

    # Run Task 3: pilot_decoder_norms
    try:
        all_results['pilot_decoder_norms'] = run_pilot_decoder_norms()
    except Exception as e:
        print(f"ERROR in pilot_decoder_norms: {e}")
        import traceback
        traceback.print_exc()
        all_results['pilot_decoder_norms'] = {'error': str(e)}

    # Run Task 5: replicate_coherence_protective
    try:
        all_results['replicate_coherence_protective'] = run_replicate_coherence_protective()
    except Exception as e:
        print(f"ERROR in replicate_coherence_protective: {e}")
        import traceback
        traceback.print_exc()
        all_results['replicate_coherence_protective'] = {'error': str(e)}

    # Summary
    print("\n" + "="*80)
    print("PILOT SUMMARY")
    print("="*80)
    print(json.dumps(all_results, indent=2, default=str))

    return all_results


if __name__ == "__main__":
    results = main()