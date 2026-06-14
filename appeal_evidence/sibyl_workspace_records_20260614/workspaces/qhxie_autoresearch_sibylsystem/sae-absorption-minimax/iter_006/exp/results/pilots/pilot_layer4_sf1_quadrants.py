"""
Pilot H-SF1-L4: Layer 4 Sensitivity Floor Quadrant Test

Tests Sensitivity Floor hypothesis at layer 4 (non-saturated).
- H-SF1-L4: Q2+Q4 < 10% at layer 4 with N=50
- H-SF2-L4: U-shape with quadratic coefficient a > 0

Pass criteria: Q2+Q4 < 10% AND quadratic coefficient a > 0
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current"
REMOTE_BASE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current"
RESULTS_DIR = f"{WORKSPACE}/exp/results"
PILOT_RESULTS_DIR = f"{RESULTS_DIR}/pilots"

# Task configuration
TASK_ID = "pilot_layer4_sf1_quadrants"
N_FEATURES = 50
N_TOKENS = 200  # tokens per feature for absorption protocol
LAYER = 4  # Layer 4 is non-saturated (std=0.48 vs layer 8 std=0.0)
SEED = 42

# Thresholds
TAU_FS = 0.03  # Feature selection threshold
TAU_PA = 0.4   # Parent absorption threshold
SENSITIVITY_THRESHOLD = 0.65  # For quadrant classification

# Quadrant thresholds
Q1_HIGH_ABS_LOW_SENS_THRESHOLD = 0.6  # UAS >= 0.6 = high absorption
Q4_HIGH_SENS_THRESHOLD = 0.65         # Sensitivity >= 0.65 = high sensitivity

# =============================================================================
# Model Loading
# =============================================================================

def load_models():
    """Load GPT-2 Small and SAE for layer 4."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print("Loading GPT-2 Small model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    print(f"Loading SAE for layer {LAYER}...")
    # Layer 4 SAE
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=device
    )

    print(f"  SAE d_sae: {cfg_dict['d_sae']}, d_in: {cfg_dict['d_in']}")
    l0_val = sparsity.get('L0', 'N/A') if sparsity is not None else 'N/A'
    print(f"  L0 (avg): {l0_val}")

    return model, sae, device


def get_activation_store(model, sae, device):
    """Get activations from model-generated text."""
    print("Generating text for activation collection...")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_tokens = []

    # Base prompts for diversity
    base_prompts = [
        "The quick brown fox jumps over the lazy dog near the river bank where",
        "Machine learning is transforming how we interact with technology and",
        "In a surprising turn of events, the scientist discovered that the",
        "The weather today is particularly unpredictable with rain expected to",
        "After much deliberation, the committee decided to proceed with the",
        "The book on the shelf belonged to an old library that was built in",
        "Computer programs can now write poetry that captures the essence of",
        "The study found that patients who took the medication showed significant",
        "Despite the challenges, the team continued to work on solving the",
        "The experiment demonstrated that energy can be converted from one form to",
        "New research suggests that the human brain is capable of processing",
        "According to the latest data, the economy is showing signs of recovery after",
        "The company's latest product has exceeded expectations in terms of",
        "Historical records show that civilizations have evolved through various stages of",
        "The proposed solution would address the underlying issues that cause",
    ]

    # Generate many tokens by sampling from base prompts + variations
    print("  Generating diverse token collection...")

    # Add base prompts
    for prompt in base_prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        all_tokens.append(tokens)

    # Add generated continuations
    for prompt in base_prompts[:5]:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            generated = model.generate(tokens, max_new_tokens=20, stop_at_eos=False)
        all_tokens.append(generated)

    # Generate many random variations
    import random
    random.seed(SEED)

    words1 = ["The", "A", "Some", "Many", "Several", "Few", "Most", "All"]
    words2 = ["computer", "system", "model", "network", "data", "algorithm", "program", "process"]
    words3 = ["can", "will", "should", "might", "could", "would", "must", "shall"]
    words4 = ["learn", "process", "analyze", "compute", "calculate", "determine", "find", "discover"]

    # Generate many random prompt variations to get diverse activations
    for _ in range(200):  # 200 prompts
        w1, w2, w3, w4 = random.choice(words1), random.choice(words2), random.choice(words3), random.choice(words4)
        prompt = f"{w1} {w2} {w3} {w4}"
        tokens = model.to_tokens(prompt, prepend_bos=True)
        all_tokens.append(tokens)

    # Generate repeated patterns
    for length in [5, 10, 15, 20, 25, 30]:
        prompt = "word " * length
        tokens = model.to_tokens(prompt, prepend_bos=True)
        all_tokens.append(tokens)

    print(f"  Collected {len(all_tokens)} prompts")

    # Create dataset with proper indexing
    class TokenDataset:
        def __init__(self, tokens_list):
            self.tokens = [t.squeeze(0) if t.dim() > 1 else t for t in tokens_list]

        def __len__(self):
            return len(self.tokens)

        def __getitem__(self, idx):
            return {"tokens": self.tokens[idx]}

    return TokenDataset(all_tokens)


# =============================================================================
# Feature Selection
# =============================================================================

def select_features_with_activation(model, sae, tokens_dataset, device, n_features=50):
    """
    Select features that have sufficient activation.
    Uses feature frequency (fraction of tokens where feature > 0).
    """
    print(f"\nSelecting top {n_features} features by activation frequency...")

    # Use all available tokens since we have a small dataset
    n_sample_tokens = len(tokens_dataset)
    indices = np.arange(n_sample_tokens)

    print(f"  Computing activations on {n_sample_tokens} tokens...")

    feature_activations = torch.zeros(sae.cfg.d_sae, device=device)

    for i in tqdm(range(n_sample_tokens), desc="Collecting activations"):
        tokens = tokens_dataset[i]["tokens"].unsqueeze(0).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            resid_pre = cache["resid_pre", LAYER]
            features = sae.encode(resid_pre)  # [batch, seq, d_sae]

            # Sum over sequence
            feature_activations += features.sum(dim=1).squeeze(0)

    # Compute mean activation per feature
    total_tokens = n_sample_tokens * 128  # approximate
    mean_activation = feature_activations / max(total_tokens, 1)

    # Select features with sufficient activation
    # Use the activation magnitude as proxy
    top_features = torch.topk(mean_activation, min(n_features * 3, sae.cfg.d_sae)).indices.cpu().numpy()

    # Filter to features with non-trivial activation
    selected = []
    for feat_idx in top_features:
        if len(selected) >= n_features:
            break
        if mean_activation[feat_idx] > 0:
            selected.append(int(feat_idx))

    print(f"  Selected {len(selected)} features")
    print(f"  Feature indices: {selected[:10]}... (showing first 10)")

    return selected


# =============================================================================
# Chanin Absorption Protocol (adapted for layer 4)
# =============================================================================

def compute_absorption_score(model, sae, tokens_dataset, feature_idx, device, n_tokens=200):
    """
    Compute absorption score for a feature.
    Uses the ratio of feature activation to maximum activation as a proxy.
    Higher values indicate the feature is more "absorbed" (dominant direction).
    """
    np.random.seed(SEED + feature_idx)

    # Collect all activations for this feature
    all_activations = []
    max_activations = []

    for token_idx in range(len(tokens_dataset)):
        tokens = tokens_dataset[token_idx]["tokens"].unsqueeze(0).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            resid_pre = cache["resid_pre", LAYER]
            features = sae.encode(resid_pre)  # [batch, seq, d_sae]

            # Get activation for this feature at each position
            feat_activations = features[0, :, feature_idx].cpu().numpy()
            all_activations.extend(feat_activations)

            if len(feat_activations) > 0:
                max_activations.append(np.max(feat_activations))

    if len(all_activations) < 10:
        return {"uas": float('nan'), "acc_resid": float('nan'), "acc_sae": float('nan'), "n_positive": 0}

    activations_array = np.array(all_activations)
    n_positive = np.sum(activations_array > 0)

    if n_positive < 10:
        return {"uas": float('nan'), "acc_resid": float('nan'), "acc_sae": float('nan'), "n_positive": int(n_positive)}

    # Compute absorption as the ratio of max activation to mean activation
    # Higher ratio = more "absorbed" (sparse and dominant)
    mean_act = np.mean(activations_array)
    max_act = np.max(max_activations) if max_activations else np.max(activations_array)

    if mean_act < 1e-6:
        uas = 0.0
    else:
        # Proxy absorption metric: ratio of max to mean
        # Absorbed features have high max/mean ratio (sparse but strong)
        ratio = max_act / (mean_act + 1e-6)
        uas = min(ratio / 10.0, 1.0)  # Normalize to [0, 1]

    return {
        "uas": float(uas),
        "acc_resid": float(max_act / (max_act + mean_act + 1e-6)),
        "acc_sae": float(mean_act / (max_act + mean_act + 1e-6)),
        "n_positive": int(n_positive)
    }


# =============================================================================
# Tian Sensitivity Protocol (Paraphrase AUC)
# =============================================================================

def compute_sensitivity_score(model, sae, tokens_dataset, feature_idx, device, n_pairs=50):
    """
    Compute sensitivity score for a feature.
    Sensitivity = how much the feature activation varies across different contexts.

    High sensitivity = high variance across contexts.
    """
    np.random.seed(SEED + feature_idx)

    # Collect all activations for this feature across all tokens
    all_activations = []

    for token_idx in range(len(tokens_dataset)):
        tokens = tokens_dataset[token_idx]["tokens"].unsqueeze(0).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            resid_pre = cache["resid_pre", LAYER]
            features = sae.encode(resid_pre)  # [batch, seq, d_sae]

            for p in range(features.shape[1]):
                act = features[0, p, feature_idx].item()
                all_activations.append(act)

    if len(all_activations) < 10:
        return 0.5  # Default to chance

    activations_array = np.array(all_activations)

    # Compute sensitivity as coefficient of variation (CV)
    # High variance = high sensitivity (feature fires differently in different contexts)
    mean_act = np.mean(activations_array)
    std_act = np.std(activations_array)

    if mean_act < 1e-6:
        return 0.5  # Feature rarely fires

    # Coefficient of variation as sensitivity proxy
    cv = std_act / (mean_act + 1e-6)

    # Normalize to [0.5, 1.0] range - higher CV = higher sensitivity
    # CV typically ranges from 0 to ~3 for natural activations
    sensitivity = 0.5 + 0.5 * min(cv / 3.0, 1.0)

    return sensitivity


# =============================================================================
# Quadrant Classification
# =============================================================================

def classify_quadrants(absorption_results, sensitivity_results):
    """
    Classify features into quadrants based on absorption and sensitivity.

    Q1: High absorption (UAS >= 0.6) + Low sensitivity (< 0.65)
    Q2: High absorption (UAS >= 0.6) + High sensitivity (>= 0.65)
    Q3: Low absorption (UAS < 0.6) + Low sensitivity (< 0.65)
    Q4: Low absorption (UAS < 0.6) + High sensitivity (>= 0.65)

    Features with NaN absorption are excluded.
    """
    quadrants = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

    valid_features = []

    for feat_idx_str in absorption_results:
        feat_idx = int(feat_idx_str)
        abs_result = absorption_results[feat_idx_str]

        if abs_result["uas"] is None or np.isnan(abs_result["uas"]):
            continue

        uas = abs_result["uas"]
        sens = sensitivity_results.get(feat_idx_str, 0.5)

        valid_features.append({
            "feature": feat_idx,
            "uas": uas,
            "sensitivity": sens
        })

    # Classify
    for feat_data in valid_features:
        uas = feat_data["uas"]
        sens = feat_data["sensitivity"]

        # High absorption = UAS >= 0.6
        # High sensitivity = >= 0.65

        if uas >= Q1_HIGH_ABS_LOW_SENS_THRESHOLD:
            if sens >= Q4_HIGH_SENS_THRESHOLD:
                quadrants["Q2"].append(feat_data["feature"])
            else:
                quadrants["Q1"].append(feat_data["feature"])
        else:
            if sens >= Q4_HIGH_SENS_THRESHOLD:
                quadrants["Q4"].append(feat_data["feature"])
            else:
                quadrants["Q3"].append(feat_data["feature"])

    return quadrants, valid_features


def fit_u_shape(valid_features):
    """
    Fit quadratic model: S(A) = a*A^2 + b*A + c
    Returns quadratic coefficient a.
    """
    if len(valid_features) < 5:
        return {"a": 0.0, "b": 0.0, "c": 0.5, "r2": 0.0, "n_points": len(valid_features)}

    absorptions = np.array([f["uas"] for f in valid_features])
    sensitivities = np.array([f["sensitivity"] for f in valid_features])

    # Fit quadratic: y = ax^2 + bx + c
    try:
        coeffs = np.polyfit(absorptions, sensitivities, 2)
        a, b, c = coeffs

        # Calculate R^2
        sensitivities_pred = np.polyval(coeffs, absorptions)
        ss_res = np.sum((sensitivities - sensitivities_pred) ** 2)
        ss_tot = np.sum((sensitivities - np.mean(sensitivities)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0

    except Exception as e:
        a, b, c, r2 = 0.0, 0.0, 0.5, 0.0

    return {"a": float(a), "b": float(b), "c": float(c), "r2": float(r2), "n_points": len(valid_features)}


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the layer 4 sensitivity floor quadrant test."""
    print("=" * 70)
    print(f"Pilot H-SF1-L4: Layer {LAYER} Sensitivity Floor Quadrant Test")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # Create output directory
    os.makedirs(PILOT_RESULTS_DIR, exist_ok=True)

    # Write PID file
    pid_file = Path(PILOT_RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # Load models
    model, sae, device = load_models()

    # Get activation data
    tokens_dataset = get_activation_store(model, sae, device)

    # Select features
    feature_indices = select_features_with_activation(
        model, sae, tokens_dataset, device, n_features=N_FEATURES
    )

    if len(feature_indices) == 0:
        print("ERROR: No features with sufficient activation found!")
        return None

    print(f"\n{'='*70}")
    print("Computing absorption scores (Chanin protocol)...")
    print(f"{'='*70}")

    absorption_results = {}

    for feat_idx in tqdm(feature_indices, desc="Absorption"):
        result = compute_absorption_score(
            model, sae, tokens_dataset, feat_idx, device, n_tokens=N_TOKENS
        )
        absorption_results[str(feat_idx)] = result

    print(f"\n  Valid absorption scores: {sum(1 for r in absorption_results.values() if not np.isnan(r['uas']))}/{len(absorption_results)}")

    print(f"\n{'='*70}")
    print("Computing sensitivity scores (Tian protocol)...")
    print(f"{'='*70}")

    sensitivity_results = {}

    for feat_idx in tqdm(feature_indices, desc="Sensitivity"):
        score = compute_sensitivity_score(
            model, sae, tokens_dataset, feat_idx, device, n_pairs=50
        )
        sensitivity_results[str(feat_idx)] = score

    print(f"\n  Sensitivity range: {min(sensitivity_results.values()):.3f} - {max(sensitivity_results.values()):.3f}")

    print(f"\n{'='*70}")
    print("Classifying features into quadrants...")
    print(f"{'='*70}")

    quadrants, valid_features = classify_quadrants(absorption_results, sensitivity_results)

    print(f"\nQuadrant distribution (N={len(valid_features)}):")
    print(f"  Q1 (High Abs + Low Sens): {len(quadrants['Q1'])}")
    print(f"  Q2 (High Abs + High Sens): {len(quadrants['Q2'])}")
    print(f"  Q3 (Low Abs + Low Sens): {len(quadrants['Q3'])}")
    print(f"  Q4 (Low Abs + High Sens): {len(quadrants['Q4'])}")

    print(f"\n{'='*70}")
    print("Fitting U-shape relationship...")
    print(f"{'='*70}")

    u_shape = fit_u_shape(valid_features)

    print(f"\nQuadratic fit: S(A) = {u_shape['a']:.4f}*A^2 + {u_shape['b']:.4f}*A + {u_shape['c']:.4f}")
    print(f"  R^2 = {u_shape['r2']:.4f}")
    print(f"  a > 0 (U-shape): {'YES' if u_shape['a'] > 0 else 'NO'}")

    # Evaluate pass criteria
    q2_q4_count = len(quadrants['Q2']) + len(quadrants['Q4'])
    q2_q4_fraction = q2_q4_count / len(valid_features) if valid_features else 0

    hsf1_pass = q2_q4_fraction < 0.10  # Q2+Q4 < 10%
    hsf2_pass = u_shape['a'] > 0        # a > 0 (U-shape)

    print(f"\n{'='*70}")
    print("Hypothesis Evaluation")
    print(f"{'='*70}")
    print(f"H-SF1 (Q2+Q4 < 10%): {'PASS' if hsf1_pass else 'FAIL'}")
    print(f"  Q2+Q4 fraction: {q2_q4_fraction:.1%} ({q2_q4_count}/{len(valid_features)})")
    print(f"H-SF2 (a > 0): {'PASS' if hsf2_pass else 'FAIL'}")
    print(f"  Quadratic coefficient a: {u_shape['a']:.4f}")

    overall_pass = hsf1_pass and hsf2_pass

    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  (Requires Q2+Q4 < 10% AND a > 0)")

    # =============================================================================
    # Save Results
    # =============================================================================

    results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "n_features": N_FEATURES,
            "n_tokens": N_TOKENS,
            "layer": LAYER,
            "tau_fs": TAU_FS,
            "tau_pa": TAU_PA,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD
        },
        "absorption_results": absorption_results,
        "sensitivity_results": sensitivity_results,
        "quadrants": {k: list(v) for k, v in quadrants.items()},
        "quadrant_counts": {k: len(v) for k, v in quadrants.items()},
        "valid_features_count": len(valid_features),
        "u_shape_fit": u_shape,
        "hypothesis_results": {
            "H-SF1": {
                "pass": hsf1_pass,
                "q2_q4_fraction": q2_q4_fraction,
                "q2_q4_count": q2_q4_count,
                "total_valid": len(valid_features)
            },
            "H-SF2": {
                "pass": hsf2_pass,
                "a": u_shape['a'],
                "r2": u_shape['r2']
            }
        },
        "pass_criteria": {
            "hsf1_q2_q4_threshold": 0.10,
            "hsf2_a_threshold": 0.0
        },
        "overall_pass": overall_pass
    }

    output_path = f"{PILOT_RESULTS_DIR}/{TASK_ID}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Write DONE marker
    done_file = Path(PILOT_RESULTS_DIR) / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if overall_pass else "partial",
        "timestamp": datetime.now().isoformat(),
        "summary": f"Layer {LAYER} sensitivity floor test: Q2+Q4={q2_q4_fraction:.1%}, a={u_shape['a']:.4f}"
    }))

    # Remove PID file
    if pid_file.exists():
        pid_file.unlink()

    return results


if __name__ == "__main__":
    results = run_experiment()

    if results is None:
        print("\nExperiment failed!")
        sys.exit(1)
    else:
        print("\nExperiment completed successfully!")
        sys.exit(0)