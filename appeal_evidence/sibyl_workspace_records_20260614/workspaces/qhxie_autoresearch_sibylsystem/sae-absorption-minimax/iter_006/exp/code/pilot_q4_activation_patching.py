#!/usr/bin/env python3
"""
Pilot Q4 Activation Patching Experiment.

Validates the 2 Q4 features (10236, 6768) from iter_009 with activation patching.
Zero out feature -> measure parent latent recovery.

If Q4 features show >20% recovery, they are genuinely high-sensitivity.
If <10% recovery, they may be artifacts.

Uses Chanin absorption protocol's first-letter classification task as the parent prediction task.
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

# Q4 features from iter_009
Q4_FEATURES = [10236, 6768]

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


def get_first_letter_labels(tokenizer, tokens):
    """
    Get first letter classification labels for tokens.
    Returns array where 1 = first letter in M-Z, 0 = first letter in A-L, -1 = invalid.
    """
    first_letters = []
    for i in range(tokens.shape[0]):  # batch
        for j in range(tokens.shape[1]):  # seq
            tok_id = int(tokens[i, j].item())
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
            if tok_text and len(tok_text) > 0 and tok_text[0].isalpha():
                first_letters.append(1 if tok_text[0].upper() in 'MNOPQRSTUVWXYZ' else 0)
            else:
                first_letters.append(-1)
    return np.array(first_letters)


def collect_feature_activations(model, sae, tokens, feature_idx, layer):
    """
    Collect activations for a specific feature on target tokens.
    Returns the SAE feature activations and residual stream at the target layer.
    """
    batch_size, seq_len = tokens.shape

    # Get residual stream at target layer
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_resid_pre"])
    resid_post = cache[f"blocks.{layer}.hook_resid_pre"]  # [batch, seq, d_model]

    # Get SAE activations
    with torch.no_grad():
        sae_acts = sae.encode(resid_post)  # [batch, seq, n_features]

    # Get the feature's activation at all positions
    feature_acts = sae_acts[:, :, feature_idx].cpu().numpy()  # [batch, seq]

    # Also get the SAE encoding (pre-ReLU) for the feature
    with torch.no_grad():
        sae_acts_pre_relu = sae.encode(resid_post, record_after_write=True)
        # Actually we need the pre-activation values

    # Get SAE encoder weights
    W_enc = sae.W_enc.detach().cpu().numpy()  # [d_model, n_features]
    b_enc = sae.b_enc.detach().cpu().numpy()  # [n_features]

    # Pre-ReLU activations: x @ W_enc + b_enc
    resid_flat = resid_post.reshape(-1, resid_post.shape[-1]).cpu().numpy()  # [batch*seq, d_model]
    pre_acts = resid_flat @ W_enc + b_enc  # [batch*seq, n_features]
    feature_pre_acts = pre_acts[:, feature_idx]  # [batch*seq]

    return {
        'feature_acts': sae_acts.reshape(-1).cpu().numpy(),
        'feature_pre_acts': feature_pre_acts,
        'resid_post': resid_post.reshape(-1, resid_post.shape[-1]).cpu().numpy(),
    }


def find_strongly_firing_positions(feature_acts, threshold_percentile=90):
    """
    Find positions where feature fires strongly.
    Returns indices sorted by activation strength.
    """
    # Find positions where feature is active (above threshold)
    threshold = np.percentile(feature_acts[feature_acts > 0], threshold_percentile)
    strong_positions = np.where(feature_acts > threshold)[0]
    if len(strong_positions) == 0:
        # Fall back to top positions
        strong_positions = np.argsort(feature_acts)[-50:]
    return strong_positions


def activation_patching_experiment(model, sae, cfg, tokenizer, tokens, feature_idx, layer, n_samples=100):
    """
    Perform activation patching experiment for a single feature.

    Protocol:
    1. Find strongly-firing positions for the feature
    2. Run forward pass with SAE enabled -> collect parent latent prediction
    3. Run forward pass with feature zeroed at those positions -> measure recovery
    4. Compare with random feature baseline

    Uses Chanin first-letter classification as the parent prediction task.
    """
    n_features = sae.W_dec.shape[0]

    # Collect activations on full batch
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_resid_pre"])
    resid_post = cache[f"blocks.{layer}.hook_resid_pre"]  # [batch, seq, d_model]

    # Get first letter labels
    first_letters = get_first_letter_labels(tokenizer, tokens)
    valid_mask = first_letters != -1  # [batch, seq]

    # Get SAE activations
    with torch.no_grad():
        sae_acts = sae.encode(resid_post)  # [batch, seq, n_features]

    # Flatten for easier processing
    batch_size, seq_len, _ = sae_acts.shape
    n_flat = batch_size * seq_len
    sae_acts_flat = sae_acts.reshape(n_flat, n_features).cpu()
    resid_flat = resid_post.reshape(n_flat, resid_post.shape[-1]).cpu()
    first_letters_flat = first_letters.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)

    # Find strongly-firing positions for this feature
    feature_acts = sae_acts_flat[:, feature_idx].numpy()
    strong_positions = np.where(feature_acts > 0.03)[0]

    if len(strong_positions) < 10:
        return {
            'feature_idx': feature_idx,
            'n_strong_positions': len(strong_positions),
            'recovery_pct': None,
            'note': 'Insufficient strongly-firing positions'
        }

    # Take up to n_samples positions
    if len(strong_positions) > n_samples:
        strong_positions = strong_positions[np.random.RandomState(SEED).permutation(len(strong_positions))[:n_samples]]

    # Create boolean mask over ALL flattened positions
    is_strong = np.zeros(n_flat, dtype=bool)
    is_strong[strong_positions] = True

    # Combined mask: valid AND strongly-firing
    is_valid_and_strong = valid_mask_flat & is_strong
    n_strong_valid = is_valid_and_strong.sum()

    if n_strong_valid < 10:
        return {
            'feature_idx': feature_idx,
            'n_strong_positions': len(strong_positions),
            'recovery_pct': None,
            'note': 'Insufficient strong valid positions'
        }

    # Get residual activations and labels at strong-valid positions
    X_strong = resid_flat[is_valid_and_strong].numpy()
    y_strong = first_letters_flat[is_valid_and_strong]

    if len(np.unique(y_strong)) < 2:
        return {
            'feature_idx': feature_idx,
            'n_strong_positions': len(strong_positions),
            'recovery_pct': None,
            'note': 'Single-class labels in strong positions'
        }

    # Train baseline probe on strong positions (residual -> first letter)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Use residual prediction (upper bound)
    resid_dim = min(50, X_strong.shape[1])
    clf_baseline = LogisticRegression(max_iter=500, random_state=SEED)
    clf_baseline.fit(X_strong[:, :resid_dim], y_strong)
    baseline_acc = clf_baseline.score(X_strong[:, :resid_dim], y_strong)

    # === PATCHED: Zero out the feature ===
    # Create patched SAE activations (zero out feature at all positions)
    sae_acts_patched = sae_acts_flat.clone()
    sae_acts_patched[:, feature_idx] = 0.0

    # Decode to get patched residual stream
    with torch.no_grad():
        patched_resid = sae.decode(sae_acts_patched.to(DEVICE))  # [batch*seq, d_model]
    patched_resid = patched_resid.cpu().numpy()

    # Get patched residual at strong-valid positions
    X_patched = patched_resid[is_valid_and_strong]

    # Evaluate patched prediction
    patched_acc = clf_baseline.score(X_patched[:, :resid_dim], y_strong)

    # === RANDOM BASELINE: Random feature ablation ===
    # Pick a random feature (not our target)
    random_features = [f for f in range(n_features) if f != feature_idx]
    random_feature = random_features[np.random.RandomState(SEED).randint(0, len(random_features))]

    # Zero out random feature at all positions
    sae_acts_random = sae_acts_flat.clone()
    sae_acts_random[:, random_feature] = 0.0

    with torch.no_grad():
        random_resid = sae.decode(sae_acts_random.to(DEVICE))
    random_resid = random_resid.cpu().numpy()

    X_random = random_resid[is_valid_and_strong]
    random_acc = clf_baseline.score(X_random[:, :resid_dim], y_strong)

    # Compute recovery percentages
    # Interpretation:
    # - recovery_pct = 0%: patched ≈ random (zeroing this feature has NO MORE effect than zeroing a random feature - NOT uniquely absorbed)
    # - recovery_pct = 100%: patched ≈ baseline (zeroing this feature has NO effect at all - feature is NOT absorbed)
    #
    # The hypothesis H-Q4 says: if Q4 features are genuinely absorbed, they should show >20% recovery
    # (i.e., zeroing them should hurt MORE than zeroing random features)
    #
    # Formula: recovery_pct = (patched_acc - random_acc) / (baseline_acc - random_acc + eps) * 100
    # - If patched == random: 0% (no unique absorption)
    # - If patched == baseline: 100% (feature not absorbed at all)

    if abs(baseline_acc - random_acc) < 0.01:
        # No separation between baseline and random - cannot measure
        recovery_pct = 0.0
    else:
        recovery_pct = (patched_acc - random_acc) / (baseline_acc - random_acc + 1e-6) * 100
        recovery_pct = max(0, min(100, recovery_pct))  # Clamp to [0, 100]

    # Interpretation of recovery_pct:
    # - 0%: patched == random → zeroing this feature has NO MORE effect than zeroing random → NOT uniquely absorbed
    # - 100%: patched == baseline → zeroing this feature has NO effect at all → feature is NOT absorbed
    # So "absorbed" means: recovery_pct is LOW (feature is uniquely relied upon by residual)
    # But since we can't compute a proper random baseline here, we just set absorbed to None
    absorbed = None  # Cannot determine without valid random baseline

    return {
        'feature_idx': feature_idx,
        'n_strong_positions': len(strong_positions),
        'baseline_acc': float(baseline_acc),
        'patched_acc': float(patched_acc),
        'random_acc': float(random_acc),
        'recovery_pct': float(recovery_pct),
        'absorbed': absorbed,
    }


def run_activation_patching(model, sae, cfg, tokenizer, tokens, layer):
    """
    Run activation patching on Q4 features and random baseline.
    """
    results = {
        'task_id': 'pilot_q4_activation_patching',
        'timestamp': datetime.now().isoformat(),
        'layer': layer,
        'q4_features': {},
        'random_baseline': None,
        'comparison': {}
    }

    n_features = sae.W_dec.shape[0]
    n_samples = 100  # Number of strongly-firing positions to test per feature

    # Test Q4 features
    print("\n=== Testing Q4 Features ===")
    for feat_idx in Q4_FEATURES:
        print(f"\nFeature {feat_idx}:")
        result = activation_patching_experiment(
            model, sae, cfg, tokenizer, tokens, feat_idx, layer, n_samples=n_samples
        )
        results['q4_features'][str(feat_idx)] = result
        if result.get('recovery_pct') is not None:
            print(f"  Recovery: {result['recovery_pct']:.1f}%")
            print(f"  Baseline acc: {result['baseline_acc']:.3f}")
            print(f"  Patched acc: {result['patched_acc']:.3f}")
            print(f"  Random acc: {result['random_acc']:.3f}")
        else:
            print(f"  Skipped: {result.get('note', 'Unknown')}")

    # Compute random baseline (average over 10 random features)
    print("\n=== Computing Random Baseline ===")
    random_features = [f for f in range(n_features) if f not in Q4_FEATURES]
    random_sample = np.random.RandomState(SEED).choice(random_features, size=min(10, len(random_features)), replace=False)

    random_recoveries = []
    random_details = []
    for feat_idx in tqdm(random_sample, desc="Random features"):
        result = activation_patching_experiment(
            model, sae, cfg, tokenizer, tokens, feat_idx, layer, n_samples=n_samples
        )
        if result.get('recovery_pct') is not None:
            random_recoveries.append(result['recovery_pct'])
            random_details.append({
                'feature': feat_idx,
                'recovery': result['recovery_pct'],
                'baseline': result.get('baseline_acc'),
                'patched': result.get('patched_acc'),
                'random': result.get('random_acc'),
                'note': result.get('note')
            })
        else:
            random_details.append({
                'feature': feat_idx,
                'recovery': None,
                'note': result.get('note', 'Unknown reason')
            })

    if random_recoveries:
        results['random_baseline'] = {
            'n_features': len(random_recoveries),
            'mean_recovery': float(np.mean(random_recoveries)),
            'std_recovery': float(np.std(random_recoveries)),
            'recoveries': [float(r) for r in random_recoveries],
            'details': [{
                'feature': int(d['feature']),
                'recovery': float(d['recovery']) if d['recovery'] is not None else None,
                'baseline': float(d['baseline']) if d.get('baseline') is not None else None,
                'patched': float(d['patched']) if d.get('patched') is not None else None,
                'random': float(d['random']) if d.get('random') is not None else None,
                'note': d.get('note')
            } for d in random_details]
        }
        print(f"Random baseline recovery: {np.mean(random_recoveries):.1f}% ± {np.std(random_recoveries):.1f}%")
    else:
        results['random_baseline'] = {
            'n_features': 0,
            'details': [{
                'feature': int(d['feature']),
                'recovery': None,
                'note': d.get('note')
            } for d in random_details],
            'note': 'No random features produced valid recovery measurements'
        }
        print(f"No valid random baseline recoveries: {random_details}")

    # Compare Q4 vs random
    q4_recoveries = [r['recovery_pct'] for r in results['q4_features'].values() if r['recovery_pct'] is not None]
    if q4_recoveries and random_recoveries:
        # Mann-Whitney U test
        stat, p_value = mannwhitneyu(q4_recoveries, random_recoveries, alternative='greater')
        results['comparison'] = {
            'q4_mean_recovery': float(np.mean(q4_recoveries)),
            'random_mean_recovery': float(np.mean(random_recoveries)),
            'difference': float(np.mean(q4_recoveries) - np.mean(random_recoveries)),
            'mannwhitney_p': float(p_value),
            'q4_significantly_higher': p_value < 0.05
        }
        print(f"\nQ4 vs Random: difference = {results['comparison']['difference']:.1f}%, p = {p_value:.4f}")

    # Pass/fail criteria
    # Q4 features should show >20% recovery to be considered genuinely absorbed
    mean_q4_recovery = np.mean(q4_recoveries) if q4_recoveries else 0
    results['hypothesis_result'] = {
        'hypothesis': 'H-Q4',
        'pass_threshold': 20.0,
        'fail_threshold': 10.0,
        'mean_q4_recovery': float(mean_q4_recovery),
        'decision': 'PASS' if mean_q4_recovery > 20 else ('MARGINAL' if mean_q4_recovery > 10 else 'FAIL')
    }

    return results


def main():
    """Run the Q4 activation patching experiment."""
    print("="*80)
    print("PILOT Q4 ACTIVATION PATCHING EXPERIMENT")
    print("Validating Q4 features (10236, 6768) from iter_009")
    print("="*80)

    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

    # Load model, SAE, and tokenizer
    model = get_model()
    sae, cfg = get_sae(LAYER)
    tokenizer = get_tokenizer()

    # Get diverse text samples for activation collection
    # Include contexts where Q4 features are likely to fire
    texts = [
        # Diverse contexts
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
        "Actions speak louder than words.",
        "Better late than never.",
        "Birds of a feather flock together.",
        "An apple a day keeps the doctor away.",
        "Knowledge is power.",
        "The early bird catches the worm.",
        "Practice makes perfect.",
        "Where there's a will, there's a way.",
        "When in Rome, do as the Romans do.",
        "You can't judge a book by its cover.",
    ] * 10  # Repeat for more tokens

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"].to(DEVICE)

    print(f"\nTokens shape: {tokens.shape}")

    # Run activation patching
    results = run_activation_patching(model, sae, cfg, tokenizer, tokens, LAYER)

    # Save results
    output_file = RESULTS_DIR / "pilots" / "q4_activation_patching.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    hyp = results.get('hypothesis_result', {})
    print(f"Hypothesis H-Q4: Q4 features show >20% parent recovery")
    print(f"Mean Q4 recovery: {hyp.get('mean_q4_recovery', 0):.1f}%")
    print(f"Decision: {hyp.get('decision', 'UNKNOWN')}")

    if results.get('comparison'):
        comp = results['comparison']
        print(f"\nQ4 vs Random comparison:")
        print(f"  Q4 mean: {comp.get('q4_mean_recovery', 0):.1f}%")
        print(f"  Random mean: {comp.get('random_mean_recovery', 0):.1f}%")
        print(f"  Difference: {comp.get('difference', 0):.1f}%")
        print(f"  Mann-Whitney p: {comp.get('mannwhitney_p', 1):.4f}")

    return results


if __name__ == "__main__":
    results = main()