#!/usr/bin/env python3
"""
full_non_absorbed_baseline: Compare steering effects for absorbed vs non-absorbed features.
Establishes whether 'robust absorbed' is comparable to non-absorbed or still degraded.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive"
RESULTS_DIR = f"{WORKSPACE}/exp/results/full"
PILOT_RESULTS_DIR = f"{WORKSPACE}/exp/results/pilots"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("FULL NON-ABSORBED BASELINE EXPERIMENT")
print("=" * 60)

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

device = torch.device("cuda")
print(f"Device: {device}")

task_id = "full_non_absorbed_baseline"
pid_file = Path(RESULTS_DIR) / f"{task_id}.pid"
pid_file.write_text(str(os.getpid()))

# Load models
print("\n[1/6] Loading GPT-2 Small model...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dim")

print("\n[2/6] Loading GPT-2 Layer 6 SAE...")
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device="cpu"  # Load to CPU first to avoid device issues
)
sae = sae.to(device)  # Move to GPU after loading
cfg_dict = sae.cfg
d_sae = cfg_dict.d_sae
d_in = cfg_dict.d_in
print(f"  SAE loaded: d_sae={d_sae}, d_in={d_in}")

# Parameters
n_samples = 500  # Reduced for faster computation
seed = 42
torch.manual_seed(seed)

steering_strength = 5
n_features_per_group = 30

prompts = [
    "The movie was very",
    "The food was extremely",
    "The weather today is"
]

print(f"\n[3/6] Computing feature statistics on {n_samples} samples...")

base_prompt = "The quick brown fox jumps over the lazy dog. " * 3
tokens = model.to_tokens(base_prompt)
seq_len = tokens.shape[1]
print(f"  Sequence length: {seq_len}")

# Collect activations on CPU to avoid CUDA issues
print("  Collecting activations (CPU)...")
all_activations = []

for i in range(n_samples):
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        acts = cache["blocks.6.hook_resid_pre"].cpu()
        all_activations.append(acts)

    if (i + 1) % 100 == 0:
        print(f"  Collected {i + 1}/{n_samples} samples")

all_activations = torch.cat(all_activations, dim=0)
print(f"  Activations shape: {all_activations.shape}")

# Compute per-feature statistics on CPU
print("  Computing per-feature statistics (CPU)...")
flat_acts = all_activations.view(-1, d_in).numpy()  # [n_samples * seq, d_in]

# Compute mean and std per input dimension
feature_means = flat_acts.mean(axis=0)  # [d_in]
feature_stds = flat_acts.std(axis=0)    # [d_in]

print(f"  Feature means range: {feature_means.min():.6f} to {feature_means.max():.6f}")
print(f"  Feature stds range: {feature_stds.min():.6f} to {feature_stds.max():.6f}")

# Clear GPU memory before SAE operations
del all_activations, flat_acts
torch.cuda.empty_cache()

# For non-absorbed identification, we use sparsity-based approach
# Re-compute on GPU carefully
print("\n  Computing feature sparsity on GPU...")
tokens_gpu = tokens.to(device)

all_sae_acts = []
with torch.no_grad():
    for i in range(min(n_samples, 100)):  # Use fewer for sparsity estimation
        _, cache = model.run_with_cache(tokens_gpu)
        resid_pre = cache["blocks.6.hook_resid_pre"]
        sae_acts = sae.encode(resid_pre)  # [1, seq, d_sae]
        all_sae_acts.append(sae_acts.cpu())

        if (i + 1) % 20 == 0:
            print(f"  SAE encoding {i + 1}/100")

all_sae_acts = torch.cat(all_sae_acts, dim=0)
print(f"  SAE activations shape: {all_sae_acts.shape}")

# Compute sparsity (fraction of time feature is active)
is_active = (all_sae_acts > 0).float()
feature_sparsity = is_active.mean(dim=(0, 1)).numpy()  # [d_sae]
print(f"  Sparsity range: {feature_sparsity.min():.3f} to {feature_sparsity.max():.3f}")

del all_sae_acts, is_active
torch.cuda.empty_cache()

# Load absorbed features from pilot
pilot_file = Path(PILOT_RESULTS_DIR) / "pilot_steering_comparison.json"
if pilot_file.exists():
    with open(pilot_file) as f:
        pilot_data = json.load(f)

    absorbed_features = set()
    for r in pilot_data["steering_results"]["high_cv"]:
        absorbed_features.add(r["feature"])
    for r in pilot_data["steering_results"]["low_cv"]:
        absorbed_features.add(r["feature"])

    high_cv_absorbed = set()
    low_cv_absorbed = set()
    for r in pilot_data["steering_results"]["high_cv"]:
        high_cv_absorbed.add(r["feature"])
    for r in pilot_data["steering_results"]["low_cv"]:
        low_cv_absorbed.add(r["feature"])

    print(f"  Loaded {len(absorbed_features)} absorbed features from pilot")
    print(f"    High-CV absorbed: {len(high_cv_absorbed)}")
    print(f"    Low-CV absorbed: {len(low_cv_absorbed)}")
else:
    absorbed_mask = feature_sparsity < 0.5
    absorbed_features = set(np.where(absorbed_mask)[0].tolist())
    high_cv_absorbed = absorbed_features.copy()
    low_cv_absorbed = set()
    print(f"  Using sparsity threshold: {len(absorbed_features)} absorbed features")

# Identify non-absorbed features
all_feature_indices = set(range(d_sae))
non_absorbed_features = list(all_feature_indices - absorbed_features)
print(f"  Non-absorbed features: {len(non_absorbed_features)}")

# Select non-absorbed features by highest sparsity
non_absorbed_sparsity = [(feature_sparsity[i], i) for i in non_absorbed_features]
non_absorbed_sparsity.sort(reverse=True)
non_absorbed_selected = [idx for _, idx in non_absorbed_sparsity[:n_features_per_group]]

# Select absorbed features for comparison
absorbed_selected = list(high_cv_absorbed)[:n_features_per_group]

print(f"\n  Selected features:")
print(f"    Non-absorbed: {len(non_absorbed_selected)} features (sparsest non-absorbed)")
print(f"    Absorbed (high-CV): {len(absorbed_selected)} features")

# Move model and SAE to GPU
print("\n[4/6] Running steering on non-absorbed features...")
hook_name = "blocks.6.hook_resid_pre"
W_dec = sae.W_dec  # [d_sae, d_in]

def measure_steering_effect(feature_idx, prompt, strength):
    tokens = model.to_tokens(prompt)
    feat_direction = W_dec[feature_idx].to(device)

    def steering_hook(activations, hook):
        return activations + strength * feat_direction

    with torch.no_grad():
        clean_logits = model(tokens)

    steered_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, steering_hook)]
    )

    target_pos = tokens.shape[1] - 1
    clean_logit = clean_logits[0, target_pos].max().item()
    steered_logit = steered_logits[0, target_pos].max().item()
    logit_change = steered_logit - clean_logit

    return {
        "feature": feature_idx,
        "prompt": prompt,
        "strength": strength,
        "logit_change": logit_change,
        "abs_effect": abs(logit_change),
    }

non_absorbed_results = []
for i, feature_idx in enumerate(non_absorbed_selected):
    for prompt in prompts:
        result = measure_steering_effect(feature_idx, prompt, steering_strength)
        non_absorbed_results.append(result)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(non_absorbed_selected)} non-absorbed features")

print(f"  Total non-absorbed steering measurements: {len(non_absorbed_results)}")

# Progress file
progress_data = {
    "task_id": task_id,
    "epoch": 1,
    "total_epochs": 1,
    "step": len(non_absorbed_results),
    "total_steps": len(non_absorbed_selected) * len(prompts),
    "status": "running",
    "updated_at": datetime.now().isoformat()
}
progress_file = Path(RESULTS_DIR) / f"{task_id}_PROGRESS.json"
progress_file.write_text(json.dumps(progress_data))

print(f"\n[5/6] Running steering on absorbed (high-CV) features...")

absorbed_results = []
for i, feature_idx in enumerate(absorbed_selected):
    for prompt in prompts:
        result = measure_steering_effect(feature_idx, prompt, steering_strength)
        absorbed_results.append(result)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(absorbed_selected)} absorbed features")

print(f"  Total absorbed steering measurements: {len(absorbed_results)}")

print(f"\n[6/6] Computing statistics...")

# Compute statistics
non_absorbed_effects = [r["logit_change"] for r in non_absorbed_results]
non_absorbed_abs_effects = [r["abs_effect"] for r in non_absorbed_results]

stats_non_absorbed = {
    "mean_logit_change": float(np.mean(non_absorbed_effects)),
    "std_logit_change": float(np.std(non_absorbed_effects)),
    "mean_abs_effect": float(np.mean(non_absorbed_abs_effects)),
    "std_abs_effect": float(np.std(non_absorbed_abs_effects)),
    "n_samples": len(non_absorbed_results),
    "n_features": len(non_absorbed_selected)
}

absorbed_effects = [r["logit_change"] for r in absorbed_results]
absorbed_abs_effects = [r["abs_effect"] for r in absorbed_results]

stats_absorbed = {
    "mean_logit_change": float(np.mean(absorbed_effects)),
    "std_logit_change": float(np.std(absorbed_effects)),
    "mean_abs_effect": float(np.mean(absorbed_abs_effects)),
    "std_abs_effect": float(np.std(absorbed_abs_effects)),
    "n_samples": len(absorbed_results),
    "n_features": len(absorbed_selected)
}

# Get pilot stats
pilot_aggregate = pilot_data.get("aggregate", {}) if pilot_file.exists() else {}
stats_high_cv_pilot = {
    "mean_logit_change": pilot_aggregate.get("high_cv_mean_raw_effect", 0),
    "mean_abs_effect": pilot_aggregate.get("high_cv_mean_abs_effect", 0),
}
stats_low_cv_pilot = {
    "mean_logit_change": pilot_aggregate.get("low_cv_mean_raw_effect", 0),
    "mean_abs_effect": pilot_aggregate.get("low_cv_mean_abs_effect", 0),
}

print("\n  Summary Statistics:")
print(f"    Non-absorbed: mean_abs={stats_non_absorbed['mean_abs_effect']:.4f}, n={stats_non_absorbed['n_features']} features")
print(f"    Absorbed high-CV: mean_abs={stats_absorbed['mean_abs_effect']:.4f}, n={stats_absorbed['n_features']} features")
print(f"    High-CV absorbed (pilot): mean_abs={stats_high_cv_pilot['mean_abs_effect']:.4f}")
print(f"    Low-CV absorbed (pilot): mean_abs={stats_low_cv_pilot['mean_abs_effect']:.4f}")

# Compile output
output = {
    "task_id": task_id,
    "timestamp": datetime.now().isoformat(),
    "status": "success",
    "config": {
        "seed": seed,
        "n_features_per_group": n_features_per_group,
        "steering_strength": steering_strength,
        "model": "gpt2-small",
        "sae": "gpt2-small-res-jb",
        "layer": 6,
        "prompts": prompts,
        "n_samples_cv_computation": n_samples
    },
    "steering_results": {
        "non_absorbed": non_absorbed_results,
        "absorbed_high_cv": absorbed_results
    },
    "statistics": {
        "non_absorbed": stats_non_absorbed,
        "absorbed_high_cv": stats_absorbed,
        "high_cv_absorbed_pilot": stats_high_cv_pilot,
        "low_cv_absorbed_pilot": stats_low_cv_pilot
    },
    "comparison": {
        "non_absorbed_vs_absorbed_high_cv": {
            "abs_effect_difference": stats_non_absorbed['mean_abs_effect'] - stats_absorbed['mean_abs_effect'],
        },
        "non_absorbed_vs_low_cv_pilot": {
            "abs_effect_difference": stats_non_absorbed['mean_abs_effect'] - stats_low_cv_pilot['mean_abs_effect'],
        }
    },
    "interpretation": {
        "finding": "baseline_comparison",
        "summary": f"Non-absorbed features show mean steering effect of {stats_non_absorbed['mean_abs_effect']:.4f} (abs). "
                   f"Absorbed high-CV features show {stats_absorbed['mean_abs_effect']:.4f}. "
                   f"Low-CV absorbed (pilot) showed {stats_low_cv_pilot['mean_abs_effect']:.4f}."
    },
    "gpu": {
        "id": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(0)
    }
}

output_file = Path(RESULTS_DIR) / f"{task_id}.json"
output_file.write_text(json.dumps(output, indent=2))
print(f"\n  Written to: {output_file}")

# DONE marker
done_file = Path(RESULTS_DIR) / f"{task_id}_DONE"
done_marker = {
    "task_id": task_id,
    "status": "success",
    "summary": f"Non-absorbed baseline: mean_abs_effect={stats_non_absorbed['mean_abs_effect']:.4f}",
    "timestamp": datetime.now().isoformat()
}
done_file.write_text(json.dumps(done_marker))
print(f"  Written DONE marker to: {done_file}")

if pid_file.exists():
    pid_file.unlink()

print("\n" + "=" * 60)
print("COMPLETED: full_non_absorbed_baseline")
print("=" * 60)