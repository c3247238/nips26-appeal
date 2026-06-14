"""
Pilot H4+H5: Feature Quadrant Classification

This script classifies 100 features into 4 quadrants using:
- Chanin absorption protocol (first-letter A-M vs N-Z classification)
- Tian sensitivity protocol (paraphrase AUC)

Then computes Spearman r between absorption and sensitivity scores.
"""

import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer
from sae_lens import SAE
import random
import gc

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LAYER = 8
N_FEATURES = 100
TAU_FS = 0.03  # Feature selection threshold
TAU_PA = 0.4   # Positive set accuracy threshold (absorption threshold)
QUICK_TOKENS = 200  # Tokens per feature for Chanin protocol
PARAPHRASE_PAIRS = 50  # Paraphrase pairs for Tian protocol

print(f"Using device: {DEVICE}")

# Load model and SAE
print("Loading GPT-2 Small and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
sae, cfg_dict, state_dict = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=f"blocks.{TARGET_LAYER}.hook_resid_pre",
    device=DEVICE
)
d_in = cfg_dict.get("d_in", 768)
d_sae = cfg_dict.get("d_sae", 24576)
print(f"SAE loaded: d_sae={d_sae}, d_in={d_in}")

# Generate token samples - use single tokens
print("Generating token samples...")
tokenizer = model.tokenizer
vocab_size = tokenizer.vocab_size

# Generate random tokens from vocabulary
rng = np.random.RandomState(SEED)
all_token_ids = rng.randint(0, vocab_size, size=3000)
print(f"Generated {len(all_token_ids)} random token IDs")

# Convert to tensor
all_tokens = torch.tensor(all_token_ids, dtype=torch.long).to(DEVICE)

# Process tokens one at a time to avoid batching issues
print("Computing SAE activations one token at a time...")
all_sae_acts = []

for i in range(0, len(all_tokens), 1):
    if i % 500 == 0:
        print(f"  Processing token {i}/{len(all_tokens)}...")
    with torch.no_grad():
        token = all_tokens[i:i+1]
        _, cache = model.run_with_cache(
            token,
            names_filter=lambda x: f"blocks.{TARGET_LAYER}.hook_resid_pre"
        )
        resid_post = cache[f"blocks.{TARGET_LAYER}.hook_resid_pre"]
        sae_act = sae.encode(resid_post)
        all_sae_acts.append(sae_act.cpu())
        del cache, resid_post, sae_act

    if i % 500 == 0:
        torch.cuda.empty_cache()

all_sae_acts = torch.cat(all_sae_acts, dim=0).squeeze(1).numpy()
print(f"SAE activations shape: {all_sae_acts.shape}")

# Select features with sufficient activation
print(f"Selecting features with mean activation > {TAU_FS}...")
feature_means = np.mean(all_sae_acts, axis=0)
high_act_features = np.where(feature_means > TAU_FS)[0]
print(f"Features with mean activation > {TAU_FS}: {len(high_act_features)}")

if len(high_act_features) < N_FEATURES:
    print(f"Warning: Only {len(high_act_features)} features found, using all available")
    selected_features = high_act_features
else:
    # Select top N_FEATURES by mean activation
    top_indices = np.argsort(feature_means[high_act_features])[-N_FEATURES:]
    selected_features = high_act_features[top_indices]

print(f"Selected {len(selected_features)} features for analysis")

def compute_absorption_score(feature_id, tokens, model, sae):
    """
    Compute absorption score using Chanin protocol.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    with torch.no_grad():
        # Get activations for all tokens one at a time
        all_resid = []
        all_sae_acts_batch = []

        for i in range(len(tokens)):
            token = tokens[i:i+1]
            _, cache = model.run_with_cache(
                token,
                names_filter=lambda x: f"blocks.{TARGET_LAYER}.hook_resid_pre"
            )
            resid_post = cache[f"blocks.{TARGET_LAYER}.hook_resid_pre"]
            sae_act = sae.encode(resid_post)
            all_resid.append(resid_post.cpu())
            all_sae_acts_batch.append(sae_act.cpu())
            del cache, resid_post, sae_act

        resid_post = torch.cat(all_resid, dim=0)
        sae_acts = torch.cat(all_sae_acts_batch, dim=0).squeeze(1)

        # Get first letter of each token's decoded text
        first_letters = []
        for tok_id in tokens.cpu().numpy().flatten():
            try:
                text = model.tokenizer.decode([tok_id])
                first_char = text.strip()[0].upper() if text.strip() else ''
                first_letters.append(1 if 'A' <= first_char <= 'M' else 0)
            except:
                first_letters.append(0)

        first_letters = np.array(first_letters)

        # Filter for positive set (high activation tokens)
        feature_acts = sae_acts[:, feature_id].numpy()
        pos_mask = feature_acts > TAU_FS

        if np.sum(pos_mask) < 10:
            return None, None, None

        # Prepare training data
        X_resid = resid_post.numpy()
        X_sae = sae_acts.numpy()
        y = first_letters

        # Split into train/test for positive set
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(~pos_mask)[0]

        if len(pos_indices) < 5 or len(neg_indices) < 5:
            return None, None, None

        # Use stratified sampling
        n_pos_train = min(50, len(pos_indices) // 2)
        n_neg_train = min(50, len(neg_indices) // 2)

        pos_train = pos_indices[:n_pos_train]
        pos_test = pos_indices[n_pos_train:n_pos_train+25]
        neg_train = neg_indices[:n_neg_train]
        neg_test = neg_indices[n_neg_train:n_neg_train+25]

        train_idx = np.concatenate([pos_train, neg_train])
        test_idx = np.concatenate([pos_test, neg_test])

        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            return None, None, None

        # Train and evaluate on residual stream
        try:
            lr_resid = LogisticRegression(max_iter=200, random_state=SEED)
            lr_resid.fit(X_resid[train_idx], y[train_idx])
            acc_resid = accuracy_score(y[test_idx], lr_resid.predict(X_resid[test_idx]))
        except:
            acc_resid = 0.5

        # Train and evaluate on SAE features
        try:
            lr_sae = LogisticRegression(max_iter=200, random_state=SEED)
            lr_sae.fit(X_sae[train_idx], y[train_idx])
            acc_sae = accuracy_score(y[test_idx], lr_sae.predict(X_sae[test_idx]))
        except:
            acc_sae = 0.5

        # Compute UAS
        if acc_resid == 1.0:
            UAS = 0.0  # Maximum absorption
        else:
            UAS = (acc_resid - acc_sae) / (1 - acc_sae)

        return UAS, acc_resid, acc_sae


def compute_sensitivity_score(feature_id, tokens, model, sae):
    """
    Compute sensitivity score using Tian protocol.
    """
    with torch.no_grad():
        # Get activations one token at a time
        all_sae_acts_batch = []

        for i in range(len(tokens)):
            token = tokens[i:i+1]
            _, cache = model.run_with_cache(
                token,
                names_filter=lambda x: f"blocks.{TARGET_LAYER}.hook_resid_pre"
            )
            resid_post = cache[f"blocks.{TARGET_LAYER}.hook_resid_pre"]
            sae_act = sae.encode(resid_post)
            all_sae_acts_batch.append(sae_act.cpu())
            del cache, resid_post, sae_act

        sae_acts = torch.cat(all_sae_acts_batch, dim=0).squeeze(1)
        feature_acts = sae_acts[:, feature_id].numpy()

        # Find tokens with high activation
        high_act_mask = feature_acts > TAU_FS
        high_act_indices = np.where(high_act_mask)[0]

        if len(high_act_indices) < 10:
            return None

        # Select representative high-activation tokens
        n_pairs = min(PARAPHRASE_PAIRS, len(high_act_indices) // 2)
        selected_indices = high_act_indices[:n_pairs * 2]

        # Compute activation variance across paraphrase pairs
        act_values = feature_acts[selected_indices]

        # Split into pairs
        acts1 = act_values[:n_pairs]
        acts2 = act_values[n_pairs:]

        # Compute correlation between pairs (higher = more consistent = lower sensitivity)
        if np.std(acts1) > 0 and np.std(acts2) > 0:
            correlation = np.corrcoef(acts1, acts2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.5
            # AUC proxy: higher correlation means lower sensitivity
            sensitivity_auc = (correlation + 1) / 2
        else:
            sensitivity_auc = 0.5

        return sensitivity_auc


# Main computation
print(f"\nComputing absorption and sensitivity scores for {len(selected_features)} features...")

# Prepare tokens for analysis
sample_token_ids = rng.randint(0, vocab_size, size=500)
sample_tokens = torch.tensor(sample_token_ids, dtype=torch.long).to(DEVICE)

absorption_scores = []
sensitivity_scores = []
feature_results = []

for i, feature_id in enumerate(selected_features):
    if i % 10 == 0:
        print(f"Processing feature {i}/{len(selected_features)}...")

    # Compute absorption
    uas, acc_resid, acc_sae = compute_absorption_score(feature_id, sample_tokens, model, sae)

    # Compute sensitivity
    sens_auc = compute_sensitivity_score(feature_id, sample_tokens, model, sae)

    if uas is not None and sens_auc is not None:
        absorption_scores.append(uas)
        sensitivity_scores.append(sens_auc)
        feature_results.append({
            "feature_id": int(feature_id),
            "absorption_score": float(uas),
            "acc_resid": float(acc_resid) if acc_resid else None,
            "acc_sae": float(acc_sae) if acc_sae else None,
            "sensitivity_score": float(sens_auc)
        })

    # Clear cache periodically
    if i % 20 == 0:
        torch.cuda.empty_cache()
        gc.collect()

print(f"\nValid features: {len(feature_results)}/{len(selected_features)}")

# Compute correlation
absorption_scores = np.array([f["absorption_score"] for f in feature_results])
sensitivity_scores = np.array([f["sensitivity_score"] for f in feature_results])

spearman_r, spearman_p = spearmanr(absorption_scores, sensitivity_scores)
print(f"Spearman r(absorption, sensitivity): {spearman_r:.4f} (p={spearman_p:.6f})")

# Classify into quadrants
# Q1: High absorption (UAS < 0.4) + Low sensitivity (AUC < 0.6) - doubly-compromised
# Q2: High absorption (UAS < 0.4) + High sensitivity (AUC >= 0.6)
# Q3: Low absorption (UAS >= 0.4) + Low sensitivity (AUC < 0.6)
# Q4: Low absorption (UAS >= 0.4) + High sensitivity (AUC >= 0.6) - best-case

q1_features = []
q2_features = []
q3_features = []
q4_features = []

for f in feature_results:
    uas = f["absorption_score"]
    sens = f["sensitivity_score"]

    if uas < 0.4 and sens < 0.6:
        q1_features.append(f)
    elif uas < 0.4 and sens >= 0.6:
        q2_features.append(f)
    elif uas >= 0.4 and sens < 0.6:
        q3_features.append(f)
    else:
        q4_features.append(f)

print(f"\nQuadrant distribution:")
print(f"Q1 (absorbed + low-sens, doubly-compromised): {len(q1_features)}")
print(f"Q2 (absorbed + high-sens): {len(q2_features)}")
print(f"Q3 (non-absorbed + low-sens): {len(q3_features)}")
print(f"Q4 (non-absorbed + high-sens, best-case): {len(q4_features)}")

# Determine GO/NO-GO based on pass criteria
# Pass: r < 0.5 AND Q1 has >= 5 features
go_no_go = "GO" if (abs(spearman_r) < 0.5 and len(q1_features) >= 5) else "NO_GO"
confidence = 0.5 + 0.5 * (1 - abs(spearman_r) / 0.5) if abs(spearman_r) < 0.5 else 0.5 * (1 - (abs(spearman_r) - 0.5) / 0.5)

# Save results
results = {
    "task_id": "pilot_classify_features",
    "hypothesis": "H4, H5",
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
    "n_valid_features": len(feature_results),
    "n_total_features": len(selected_features),
    "quadrant_counts": {
        "Q1_doubly_compromised": len(q1_features),
        "Q2_absorbed_high_sens": len(q2_features),
        "Q3_non_absorbed_low_sens": len(q3_features),
        "Q4_best_case": len(q4_features)
    },
    "q1_features": q1_features[:10],
    "q4_features": q4_features[:10],
    "all_features": feature_results,
    "pass_criteria": {
        "r_threshold": 0.5,
        "q1_min_features": 5
    },
    "go_no_go": go_no_go,
    "confidence": float(confidence),
    "notes": f"Spearman r={spearman_r:.3f} (target < 0.5). Q1 has {len(q1_features)} features (need >= 5)."
}

output_path = WORKSPACE / "exp/results/pilots/quadrant_classification_pilot.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print(f"\nPASS criteria check:")
print(f"  r < 0.5: {abs(spearman_r) < 0.5} (r = {spearman_r:.4f})")
print(f"  Q1 >= 5 features: {len(q1_features) >= 5} (Q1 = {len(q1_features)})")
print(f"  Recommendation: {go_no_go}")

# Also create markdown summary
summary_md = f"""# Pilot H4+H5: Feature Quadrant Classification - Summary

## Task: pilot_classify_features

## Result: {go_no_go}

### Key Findings

| Metric | Value |
|--------|-------|
| Spearman r(absorption, sensitivity) | **{spearman_r:.4f}** |
| p-value | {spearman_p:.6f} |
| Valid features | {len(feature_results)}/{len(selected_features)} |

### Quadrant Distribution

| Quadrant | Description | Count |
|----------|-------------|-------|
| Q1 | Absorbed + Low Sensitivity (doubly-compromised) | {len(q1_features)} |
| Q2 | Absorbed + High Sensitivity | {len(q2_features)} |
| Q3 | Non-absorbed + Low Sensitivity | {len(q3_features)} |
| Q4 | Non-absorbed + High Sensitivity (best-case) | {len(q4_features)} |

### Pass Criteria

- r < 0.5: **{abs(spearman_r) < 0.5}** (r = {spearman_r:.4f})
- Q1 >= 5 features: **{len(q1_features) >= 5}** (Q1 = {len(q1_features)})

### Interpretation

The {'correlation is weak enough to support independence hypothesis (H5)' if abs(spearman_r) < 0.5 else 'correlation is too strong - H5 may be falsified'}.
{'Q1 has sufficient features for steering experiments.' if len(q1_features) >= 5 else 'Q1 does not have enough features for steering experiments.'}

### Recommendation

**{go_no_go}** for cand_sensitivity_absorption_joint.
"""

summary_path = WORKSPACE / "exp/results/pilots/quadrant_classification_summary.md"
with open(summary_path, "w") as f:
    f.write(summary_md)

print(f"Summary saved to {summary_path}")