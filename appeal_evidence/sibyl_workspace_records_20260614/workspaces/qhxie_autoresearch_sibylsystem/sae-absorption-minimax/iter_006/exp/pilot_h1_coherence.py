#!/usr/bin/env python3
"""
Pilot H1: Coherence-Absorption Correlation

Tests whether decoder weight mutual coherence at SAE initialization predicts
final absorption severity on pre-trained GPT-2 Small SAEs.

Methodology:
1. Load GPT-2 Small SAEs (gpt2-small-res-jb) from SAELens for layer 8
2. Compute pairwise mutual coherence for 1000 randomly sampled feature pairs
3. Run Chanin first-letter protocol (A-M vs N-Z) on 200 tokens per feature
4. Measure Spearman correlation between coherence and absorption

Pass criteria: Spearman r(coherence, absorption) > 0.3 with p < 0.05
               AND high-coherence features show absorption probability > 0.5
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Try to import transformer_lens and sae_lens
try:
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("Please install: pip install transformer_lens sae_lens")
    sys.exit(1)

# Configuration
SEED = 42
N_PAIRS = 1000  # Number of feature pairs to sample for coherence computation
N_TOKENS = 200  # Number of tokens for Chanin protocol (quick mode)
LAYER = 8  # Primary target layer
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chanin protocol thresholds
TAU_FS = 0.03  # Feature selection threshold
TAU_PS = 0.025  # Positive set precision threshold
TAU_PA = 0.4   # Positive set accuracy threshold (absorption threshold)

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def compute_mutual_coherence(W_dec: torch.Tensor, n_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise mutual coherence for sampled feature pairs.

    μ_ij = |<w_i, w_j>| / (||w_i|| × ||w_j||) for i ≠ j

    Args:
        W_dec: Decoder weight matrix [d_sae, d_model]
        n_pairs: Number of random pairs to sample

    Returns:
        (pair_indices, coherences, max_coherences)
        - pair_indices: (n_pairs, 2) array of feature indices
        - coherences: (n_pairs,) array of mutual coherence values
        - max_coherences: (d_sae,) array of max coherence for each feature
    """
    d_sae, d_model = W_dec.shape

    # Normalize decoder weights
    W_norm = W_dec / W_dec.norm(dim=1, keepdim=True)  # [d_sae, d_model]

    # Compute all pairwise inner products (symmetric matrix)
    # This is O(d_sae^2 * d_model) but we can compute efficiently
    inner_products = W_norm @ W_norm.T  # [d_sae, d_sae]

    # Sample random pairs (excluding diagonal)
    valid_pairs = []
    while len(valid_pairs) < n_pairs:
        i = np.random.randint(0, d_sae)
        j = np.random.randint(0, d_sae)
        if i != j:
            valid_pairs.append((i, j))

    pair_indices = np.array(valid_pairs)
    coherences = np.abs(inner_products[pair_indices[:, 0], pair_indices[:, 1]].cpu().numpy())

    # Compute max coherence for each feature (excluding self)
    max_coherences = []
    for i in range(d_sae):
        row = np.abs(inner_products[i].cpu().numpy())
        row[i] = 0  # Exclude self
        max_coherences.append(np.max(row))
    max_coherences = np.array(max_coherences)

    return pair_indices, coherences, max_coherences


def get_first_letter_bucket(token: str) -> int:
    """Classify token into A-M (0) vs N-Z (1) first-letter buckets."""
    if not token:
        return -1
    first_char = token[0].upper()
    if first_char.isalpha():
        return 0 if first_char <= 'M' else 1
    return -1  # Non-alphabetic


def run_chanin_protocol(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
    feature_indices: List[int],
    n_tokens: int = 200,
    layer: int = 8
) -> Dict[int, Dict]:
    """
    Run Chanin et al. (2024) first-letter classification protocol.

    For each feature:
    1. Select tokens where feature activation > tau_fs
    2. Compute probe accuracy on residual stream (upper bound)
    3. Compute probe accuracy on SAE features (lower bound)
    4. UAS = (acc_resid - acc_sae) / (1 - acc_sae)

    Args:
        model: HookedTransformer model
        sae: SAE instance
        tokens: Token tensor [n_tokens, seq_len]
        feature_indices: List of feature indices to test
        n_tokens: Number of tokens to use
        layer: Layer to extract activations from

    Returns:
        Dictionary mapping feature_idx -> {uas, acc_resid, acc_sae, n_positive}
    """
    results = {}

    # Get residual stream activations and SAE activations
    with torch.no_grad():
        # Run model to get all activations
        logits, cache = model.run_with_cache(tokens[:n_tokens])

        # Get residual stream activations at the target layer
        resid_post = cache[f"blocks.{layer}.hook_resid_post"]  # [n_tokens, seq_len, d_model]

        # Flatten sequence dimension
        resid_flat = resid_post.reshape(-1, resid_post.shape[-1])  # [n_tokens * seq_len, d_model]

        # Get SAE activations
        sae_acts = sae.encode(resid_flat.to(DEVICE))  # [n_tokens * seq_len, d_sae]

        # Get token strings for first-letter classification
        # Flatten tokens to [batch * seq_len] and decode
        tokens_2d = tokens[:n_tokens]
        batch_size, seq_len = tokens_2d.shape
        tokens_flat = tokens_2d.reshape(-1)  # [batch * seq_len]

        # Decode using tokenizer
        token_strings = model.tokenizer.batch_decode(tokens_flat)

    # Get first-letter labels for each position
    labels = []
    for tok_str in token_strings:
        bucket = get_first_letter_bucket(tok_str)
        labels.append(bucket)
    labels = np.array(labels)

    # Filter to alphabetic tokens
    alphabetic_mask = labels >= 0
    n_positions = alphabetic_mask.sum()

    if n_positions < 100:
        print(f"WARNING: Only {n_positions} alphabetic positions, may be unreliable")

    for feat_idx in feature_indices:
        # Get feature activations
        feat_acts = sae_acts[:, feat_idx].cpu().numpy()

        # Feature selection: tokens where feature has high activation
        high_act_mask = feat_acts > TAU_FS

        if high_act_mask.sum() < 10:
            results[feat_idx] = {
                "uas": np.nan,
                "acc_resid": np.nan,
                "acc_sae": np.nan,
                "n_positive": int(high_act_mask.sum()),
                "absorption": np.nan
            }
            continue

        # Create positive/negative set based on feature activation
        # (this is the "feature" class for the probe)
        y = high_act_mask.astype(int)

        # Get indices for both sets
        pos_indices = np.where(high_act_mask)[0]
        neg_indices = np.where(~high_act_mask & alphabetic_mask)[0]

        if len(pos_indices) < 5 or len(neg_indices) < 5:
            results[feat_idx] = {
                "uas": np.nan,
                "acc_resid": np.nan,
                "acc_sae": np.nan,
                "n_positive": int(high_act_mask.sum()),
                "absorption": np.nan
            }
            continue

        # Balance sets for probe training
        n_train = min(len(pos_indices), len(neg_indices), 100)
        np.random.seed(SEED)
        train_pos = np.random.choice(pos_indices, n_train, replace=False)
        train_neg = np.random.choice(neg_indices, n_train, replace=False)
        train_indices = np.concatenate([train_pos, train_neg])
        np.random.shuffle(train_indices)

        # First-letter labels for training
        y_train = labels[train_indices]
        train_mask = y_train >= 0

        if train_mask.sum() < 10:
            results[feat_idx] = {
                "uas": np.nan,
                "acc_resid": np.nan,
                "acc_sae": np.nan,
                "n_positive": int(high_act_mask.sum()),
                "absorption": np.nan
            }
            continue

        # Move residual activations to CPU for sklearn
        resid_flat_cpu = resid_flat.cpu()

        # Probe on residual stream (upper bound)
        X_resid = resid_flat_cpu[train_indices][train_mask].numpy()
        y_resid = y_train[train_mask]

        probe_resid = LogisticRegression(max_iter=500, random_state=SEED)
        probe_resid.fit(X_resid, y_resid)

        # Evaluate on held-out alphabetic tokens
        test_mask = alphabetic_mask & ~np.isin(np.arange(len(labels)), train_indices)
        if test_mask.sum() < 50:
            test_mask = alphabetic_mask  # Fallback to all if needed

        X_test_resid = resid_flat_cpu[test_mask].numpy()
        y_test = labels[test_mask]

        acc_resid = accuracy_score(y_test, probe_resid.predict(X_test_resid))

        # Probe on SAE features
        X_sae = sae_acts[train_indices][train_mask].cpu().numpy()

        probe_sae = LogisticRegression(max_iter=500, random_state=SEED)
        probe_sae.fit(X_sae, y_resid)

        X_test_sae = sae_acts[test_mask].cpu().numpy()
        acc_sae = accuracy_score(y_test, probe_sae.predict(X_test_sae))

        # Compute UAS (Upper bound Absorption Score)
        # UAS = (acc_resid - acc_sae) / (1 - acc_sae)
        if acc_sae < 1.0:
            uas = (acc_resid - acc_sae) / (1 - acc_sae)
        else:
            uas = np.nan

        # Absorption: feature is absorbed if UAS < TAU_PA (0.4)
        absorption = 1.0 if uas < TAU_PA else 0.0

        results[feat_idx] = {
            "uas": float(uas) if not np.isnan(uas) else None,
            "acc_resid": float(acc_resid),
            "acc_sae": float(acc_sae),
            "n_positive": int(high_act_mask.sum()),
            "absorption": absorption
        }

    return results


def main():
    print("=" * 60)
    print("Pilot H1: Coherence-Absorption Correlation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Layer: {LAYER}")
    print(f"Feature pairs for coherence: {N_PAIRS}")
    print(f"Tokens for Chanin protocol: {N_TOKENS}")
    print()

    # Results directory
    results_dir = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/iter_006/exp/results/pilots")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "h1_coherence_pilot.json"

    # Write PID file for system monitoring
    pid_file = results_dir.parent / "h1_coherence_pilot.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        # Step 1: Load GPT-2 Small SAEs
        print("Step 1: Loading GPT-2 Small SAEs from SAELens...")

        # Load model
        model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
        print(f"  Model loaded: gpt2-small")

        # Load SAE (gpt2-small-res-jb is the standard release)
        # SAE ID for gpt2-small layer 8 is "blocks.8.hook_resid_pre"
        # Note: SAE.from_pretrained() returns only SAE object in newer versions
        result = SAE.from_pretrained_with_cfg_and_sparsity(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{LAYER}.hook_resid_pre",
            device=DEVICE
        )
        if isinstance(result, tuple):
            sae, cfg_dict, sparsity = result
        else:
            # New API: just SAE object
            sae = result
            cfg_dict = {}
            sparsity = None

        print(f"  SAE loaded: gpt2-small-res-jb / blocks.{LAYER}.hook_resid_pre")
        print(f"  d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")
        if sparsity is not None:
            print(f"  Sparsity: {sparsity.mean():.4f}")

        # Get decoder weights
        W_dec = sae.W_dec.detach()  # [d_sae, d_model]
        d_sae = W_dec.shape[0]
        print(f"  W_dec shape: {W_dec.shape}")

        # Step 2: Compute mutual coherence
        print(f"\nStep 2: Computing mutual coherence for {N_PAIRS} feature pairs...")

        pair_indices, coherences, max_coherences = compute_mutual_coherence(W_dec, N_PAIRS)

        print(f"  Coherence range: [{coherences.min():.4f}, {coherences.max():.4f}]")
        print(f"  Coherence mean: {coherences.mean():.4f}")
        print(f"  Max coherence range: [{max_coherences.min():.4f}, {max_coherences.max():.4f}]")

        # Step 3: Prepare tokens for Chanin protocol
        print(f"\nStep 3: Preparing {N_TOKENS} tokens for Chanin protocol...")

        # Generate random text tokens using a simple approach
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Use the tokenizer to encode actual text
        # We'll generate random text sequences with alphabetic characters
        sample_texts = []
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        for i in range(N_TOKENS):
            # Generate random length text
            length = np.random.randint(5, 20)
            text = ''.join(np.random.choice(list(alphabet + alphabet.upper()), size=length))
            sample_texts.append(text)

        tokens = model.tokenizer.batch_encode_plus(
            sample_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )["input_ids"].to(DEVICE)

        print(f"  Token batch shape: {tokens.shape}")

        # Step 4: Select features based on max coherence
        print(f"\nStep 4: Selecting features for absorption testing...")

        # Select top and bottom coherence features for comparison
        n_features_per_group = 50
        sorted_indices = np.argsort(max_coherences)

        low_coh_indices = sorted_indices[:n_features_per_group].tolist()
        high_coh_indices = sorted_indices[-n_features_per_group:].tolist()
        all_test_indices = low_coh_indices + high_coh_indices

        print(f"  Low coherence features: {len(low_coh_indices)} (mean max_coh: {max_coherences[low_coh_indices].mean():.4f})")
        print(f"  High coherence features: {len(high_coh_indices)} (mean max_coh: {max_coherences[high_coh_indices].mean():.4f})")

        # Step 5: Run Chanin absorption protocol
        print(f"\nStep 5: Running Chanin first-letter protocol...")

        absorption_results = run_chanin_protocol(
            model=model,
            sae=sae,
            tokens=tokens,
            feature_indices=all_test_indices,
            n_tokens=N_TOKENS,
            layer=LAYER
        )

        # Step 6: Compute correlation
        print(f"\nStep 6: Computing coherence-absorption correlation...")

        # Build feature coherence map
        feature_coherences = max_coherences[all_test_indices]
        feature_absorptions = []
        feature_uas = []

        valid_indices = []
        for i, feat_idx in enumerate(all_test_indices):
            uas_val = absorption_results[feat_idx]["uas"]
            # Filter out None and NaN values
            if uas_val is not None and not (isinstance(uas_val, float) and uas_val != uas_val):
                valid_indices.append(i)
                feature_absorptions.append(absorption_results[feat_idx]["absorption"])
                feature_uas.append(uas_val)

        feature_coherences = feature_coherences[valid_indices]
        feature_absorptions = np.array(feature_absorptions)
        feature_uas = np.array(feature_uas)

        # Spearman correlation between coherence and absorption
        if len(feature_absorptions) > 10:
            spearman_r, spearman_p = spearmanr(feature_coherences, feature_absorptions)
        else:
            spearman_r, spearman_p = np.nan, np.nan

        # High-coherence absorption rate
        high_coh_mask = feature_coherences > np.median(feature_coherences)
        if high_coh_mask.sum() > 0:
            high_coh_absorption_rate = feature_absorptions[high_coh_mask].mean()
        else:
            high_coh_absorption_rate = np.nan

        low_coh_mask = ~high_coh_mask
        if low_coh_mask.sum() > 0:
            low_coh_absorption_rate = feature_absorptions[low_coh_mask].mean()
        else:
            low_coh_absorption_rate = np.nan

        print(f"  Spearman r(coherence, absorption): {spearman_r:.4f} (p={spearman_p:.4f})")
        print(f"  High-coherence absorption rate: {high_coh_absorption_rate:.4f}")
        print(f"  Low-coherence absorption rate: {low_coh_absorption_rate:.4f}")

        # Step 7: Determine GO/NO-GO
        print(f"\nStep 7: Evaluating pass criteria...")

        pass_criteria_met = (
            spearman_r > 0.3 and
            spearman_p < 0.05 and
            high_coh_absorption_rate > 0.5
        )

        if pass_criteria_met:
            recommendation = "GO"
            print("  RESULT: GO - Pass criteria met!")
        else:
            recommendation = "NO_GO"
            print("  RESULT: NO-GO - Pass criteria not met")

        print(f"  - Spearman r > 0.3: {spearman_r > 0.3} (r={spearman_r:.4f})")
        print(f"  - p < 0.05: {spearman_p < 0.05} (p={spearman_p:.4f})")
        print(f"  - High-coh P_abs > 0.5: {high_coh_absorption_rate > 0.5} (P_abs={high_coh_absorption_rate:.4f})")

        # Step 8: Save results
        print(f"\nStep 8: Saving results to {output_file}...")

        result_data = {
            "task_id": "pilot_h1_coherence",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "seed": SEED,
                "n_pairs": N_PAIRS,
                "n_tokens": N_TOKENS,
                "layer": LAYER,
                "tau_fs": TAU_FS,
                "tau_ps": TAU_PS,
                "tau_pa": TAU_PA
            },
            "coherence_stats": {
                "pair_coherence_mean": float(coherences.mean()),
                "pair_coherence_std": float(coherences.std()),
                "max_coherence_mean": float(max_coherences.mean()),
                "max_coherence_std": float(max_coherences.std())
            },
            "absorption_results": {
                str(k): v for k, v in absorption_results.items()
            },
            "correlation": {
                "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
                "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
                "high_coh_absorption_rate": float(high_coh_absorption_rate) if not np.isnan(high_coh_absorption_rate) else None,
                "low_coh_absorption_rate": float(low_coh_absorption_rate) if not np.isnan(low_coh_absorption_rate) else None,
                "n_valid_features": len(valid_indices)
            },
            "pass_criteria": {
                "r_threshold": 0.3,
                "p_threshold": 0.05,
                "high_coh_absorption_threshold": 0.5,
                "met": bool(pass_criteria_met)
            },
            "recommendation": recommendation
        }

        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"  Results saved successfully")

        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink()

        # Write DONE marker
        done_file = results_dir.parent / "h1_coherence_pilot_DONE"
        done_file.write_text(json.dumps({
            "task_id": "pilot_h1_coherence",
            "status": "success" if pass_criteria_met else "completed_with_no_go",
            "recommendation": recommendation,
            "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
            "timestamp": datetime.now().isoformat()
        }))

        print("\n" + "=" * 60)
        print("Pilot H1 Complete")
        print("=" * 60)

        return 0 if pass_criteria_met else 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Write error DONE marker
        done_file = results_dir.parent / "h1_coherence_pilot_DONE"
        done_file.write_text(json.dumps({
            "task_id": "pilot_h1_coherence",
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))

        return 1


if __name__ == "__main__":
    sys.exit(main())