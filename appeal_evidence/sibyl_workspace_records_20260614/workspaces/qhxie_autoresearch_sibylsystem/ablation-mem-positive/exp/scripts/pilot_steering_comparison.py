#!/usr/bin/env python3
"""
Pilot Steering Comparison: High-CV vs Low-CV Absorbed Features
Tests steering effectiveness on 30 high-CV vs 30 low-CV absorbed features at +5 strength.
"""
import json
import sys
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Setup
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
RESULTS_DIR = WORKSPACE / "exp/results/pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Config
SEED = 42
N_FEATURES_PER_GROUP = 30
STEERING_STRENGTH = 5
PROMPT_TEMPLATES = [
    "The movie was very",
    "The food was extremely",
    "The weather today is",
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("\n" + "="*60)
    print("PILOT: Steering Comparison by CV")
    print("="*60)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    # Load GPT-2 model
    print("\n[1/4] Loading GPT-2 Small model...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    print(f"  Model: GPT-2 Small")

    # Load SAE
    print("\n[2/4] Loading GPT-2 Layer 6 SAE...")
    from sae_lens import SAE
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        device=str(device)
    )
    print(f"  SAE: gpt2-small-res-jb, layer 6, d_sae={sae.cfg.d_sae}")

    # Load classification results
    print("\n[3/4] Loading CV classification results...")
    class_path = WORKSPACE / "exp/results/pilot_cv_classification.json"
    with open(class_path) as f:
        classification = json.load(f)

    high_cv_features = classification['groups']['high_cv']['indices'][:N_FEATURES_PER_GROUP]
    low_cv_features = classification['groups']['low_cv']['indices'][:N_FEATURES_PER_GROUP]

    print(f"  High-CV features: {len(high_cv_features)}")
    print(f"  Low-CV features: {len(low_cv_features)}")

    # Run steering experiments
    print("\n[4/4] Running steering experiments...")

    # Steering function
    def run_steering_for_feature(feature_idx, prompt, strength, model, sae, device):
        """Run steering for a single feature and return logit change."""
        # Get the feature direction from decoder weights
        feature_dir = sae.W_dec[feature_idx].float().to(device)
        feature_dir = feature_dir / (feature_dir.norm() + 1e-6)

        # Tokenize
        input_ids = model.tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

        # Get clean logits
        with torch.no_grad():
            clean_logits = model(input_ids)

        # Define steering hook
        def steering_hook(resid, hook):
            # Add steering signal to residual
            steering_signal = strength * feature_dir.unsqueeze(0).unsqueeze(0)
            return resid + steering_signal

        # Run with steering
        with model.hooks(fwd_hooks=[(f"blocks.6.hook_resid_pre", steering_hook)]):
            steered_logits = model(input_ids)

        # Get logits at the last token position
        target_pos = input_ids.shape[1] - 1
        clean_logit = clean_logits[0, target_pos, :].max().item()
        steered_logit = steered_logits[0, target_pos, :].max().item()
        logit_change = steered_logit - clean_logit

        return {
            'feature': int(feature_idx),
            'prompt': prompt,
            'strength': strength,
            'logit_change': float(logit_change),
            'abs_effect': float(abs(logit_change))
        }

    # Run for high-CV features
    high_cv_results = []
    for feat_idx in tqdm(high_cv_features, desc="High-CV steering"):
        for prompt in PROMPT_TEMPLATES:
            result = run_steering_for_feature(feat_idx, prompt, STEERING_STRENGTH, model, sae, device)
            high_cv_results.append(result)

    # Run for low-CV features
    low_cv_results = []
    for feat_idx in tqdm(low_cv_features, desc="Low-CV steering"):
        for prompt in PROMPT_TEMPLATES:
            result = run_steering_for_feature(feat_idx, prompt, STEERING_STRENGTH, model, sae, device)
            low_cv_results.append(result)

    # Aggregate
    high_cv_effects = [r['logit_change'] for r in high_cv_results]
    low_cv_effects = [r['logit_change'] for r in low_cv_results]

    high_cv_mean = np.mean(high_cv_effects)
    low_cv_mean = np.mean(low_cv_effects)
    high_cv_std = np.std(high_cv_effects)
    low_cv_std = np.std(low_cv_effects)

    # Statistical test (Welch's t-test, one-sided)
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(high_cv_effects, low_cv_effects, equal_var=False)

    high_cv_larger = bool(high_cv_mean > low_cv_mean)
    significant = bool(p_value < 0.05)
    pass_criteria_met = high_cv_larger and significant

    # Compile results
    results = {
        'task_id': 'pilot_steering_comparison',
        'status': 'success' if pass_criteria_met else 'needs_review',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': SEED,
            'n_features_per_group': N_FEATURES_PER_GROUP,
            'steering_strength': STEERING_STRENGTH,
            'model': 'gpt2-small',
            'sae': 'gpt2-small-res-jb',
            'layer': 6,
            'prompts': PROMPT_TEMPLATES
        },
        'steering_results': {
            'high_cv': high_cv_results,
            'low_cv': low_cv_results
        },
        'aggregate': {
            'high_cv_mean_effect': float(high_cv_mean),
            'low_cv_mean_effect': float(low_cv_mean),
            'high_cv_std_effect': float(high_cv_std),
            'low_cv_std_effect': float(low_cv_std),
            'n_high_cv_samples': len(high_cv_results),
            'n_low_cv_samples': len(low_cv_results),
            'difference': float(high_cv_mean - low_cv_mean),
            'ratio': float(high_cv_mean / low_cv_mean) if low_cv_mean != 0 else float('inf')
        },
        'statistical_test': {
            'test': 'Welch t-test (one-sided)',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': significant,
            'high_cv_larger': high_cv_larger
        },
        'pass_criteria': {
            'required': 'High-CV steering effect > Low-CV with p < 0.05',
            'high_cv_mean': float(high_cv_mean),
            'low_cv_mean': float(low_cv_mean),
            'high_cv_larger': high_cv_larger,
            'p_value': float(p_value),
            'significant': significant,
            'overall_pass': pass_criteria_met
        },
        'gpu': {
            'id': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }

    # Save results
    output_path = RESULTS_DIR / "pilot_steering_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Write DONE marker
    done_path = RESULTS_DIR / "pilot_steering_comparison_DONE"
    with open(done_path, 'w') as f:
        f.write(json.dumps({'task_id': 'pilot_steering_comparison', 'status': results['status']}))

    print(f"\nSaved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"High-CV mean effect: {high_cv_mean:.4f}")
    print(f"Low-CV mean effect: {low_cv_mean:.4f}")
    print(f"Difference: {high_cv_mean - low_cv_mean:.4f}")
    print(f"Ratio: {high_cv_mean / low_cv_mean:.2f}x" if low_cv_mean != 0 else "Ratio: inf")
    print(f"p-value: {p_value:.6f}")
    print(f"Pass: {pass_criteria_met}")

    return results

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results['pass_criteria']['overall_pass'] else 1)