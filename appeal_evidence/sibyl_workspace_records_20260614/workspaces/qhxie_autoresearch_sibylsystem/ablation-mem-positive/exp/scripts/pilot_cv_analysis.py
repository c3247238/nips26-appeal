#!/usr/bin/env python3
"""
Pilot: Coefficient of Variation Analysis (H4)
Measure CV for absorbed vs non-absorbed features at layer 6.
Hypothesis H4: CV_low < CV_high at critical point (layer 6).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ========== Paths ==========
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "pilot_cv_analysis"
SCRIPT_START = datetime.now().isoformat()

# ========== Device ==========
assigned_gpu = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[{TASK_ID}] Using device: {device} (assigned GPU {assigned_gpu})")

# ========== Load Model & SAE ==========
print(f"[{TASK_ID}] Loading GPT-2 + SAE...")
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=str(device)
)
print(f"[{TASK_ID}] SAE loaded: d_sae={sae.cfg.d_sae}, hook={sae.cfg.metadata.get('hook_name', 'unknown')}")

# ========== Config ==========
N_SAMPLES = 100
SEED = 42
LAYER = 6
ABSORPTION_THRESHOLD = 0.001  # Features with absorption score > this are "absorbed"

np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Prompts ==========
prompts = [
    "The capital of France is",
    "I walked down the street to",
    "The theory of relativity was",
    "In the beginning,",
    "The solution to the problem",
    "When the rain fell, the",
    "A new discovery in science",
    "The historical event took place",
    "The author wrote that",
    "After the sun set, the",
]
prompts = prompts * (N_SAMPLES // len(prompts) + 1)
prompts = prompts[:N_SAMPLES]

print(f"[{TASK_ID}] Running CV analysis on {len(prompts)} prompts...")

# ========== Compute absorption scores for feature classification ==========
print(f"[{TASK_ID}] Computing absorption scores (weight_norm * activation)...")
tokens = model.to_tokens(prompts, truncate=None)
max_len = tokens.shape[1]
tokens = tokens[:, :max_len]

_, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_pre"])
activations = cache[f"blocks.{LAYER}.hook_resid_pre"]

sae_features = sae.encode(activations).detach()  # [batch, seq, d_sae]

# Decoder weight norms
W_dec = sae.W_dec.detach()  # [d_sae, d_in]
weight_norms = W_dec.norm(dim=1).cpu().numpy()  # [d_sae]

# Mean activation magnitude per feature
mean_activations = sae_features.abs().mean(dim=(0, 1)).cpu().numpy()  # [d_sae]

# Absorption score
absorption_scores = weight_norms * mean_activations

# Classify features: absorbed (score > threshold) vs non-absorbed
absorbed_mask = absorption_scores > ABSORPTION_THRESHOLD
non_absorbed_mask = ~absorbed_mask

print(f"[{TASK_ID}] Features: {absorbed_mask.sum()} absorbed, {non_absorbed_mask.sum()} non-absorbed")

# ========== Compute CV for each group ==========
# CV = std / mean for activation magnitudes
# For each feature, compute CV across all prompts/positions

# Per-feature: CV across samples
feature_cv = {}  # feature_idx -> CV

all_activations_by_feature = sae_features.abs().cpu().numpy()  # [batch, seq, d_sae]

# Compute CV per feature (std/mean across all positions for each feature)
for feat_idx in range(sae.cfg.d_sae):
    activations_for_feat = all_activations_by_feature[:, :, feat_idx].flatten()
    mean_val = activations_for_feat.mean()
    std_val = activations_for_feat.std()
    if mean_val > 1e-8:
        feature_cv[feat_idx] = std_val / mean_val
    else:
        feature_cv[feat_idx] = 0.0

# Group into absorbed and non-absorbed
absorbed_cv = [feature_cv[i] for i in range(sae.cfg.d_sae) if absorbed_mask[i]]
non_absorbed_cv = [feature_cv[i] for i in range(sae.cfg.d_sae) if non_absorbed_mask[i]]

# Compute overall CV statistics for each group
cv_low_mean = np.mean(absorbed_cv) if absorbed_cv else 0.0
cv_low_std = np.std(absorbed_cv) if absorbed_cv else 0.0
cv_high_mean = np.mean(non_absorbed_cv) if non_absorbed_cv else 0.0
cv_high_std = np.std(non_absorbed_cv) if non_absorbed_cv else 0.0

print(f"[{TASK_ID}] CV (absorbed):  mean={cv_low_mean:.4f}, std={cv_low_std:.4f}")
print(f"[{TASK_ID}] CV (non-abs):   mean={cv_high_mean:.4f}, std={cv_high_std:.4f}")

# Compute difference direction
cv_diff = cv_high_mean - cv_low_mean  # positive means CV_high > CV_low
cv_ratio = cv_low_mean / (cv_high_mean + 1e-8)

print(f"[{TASK_ID}] CV difference (high - low): {cv_diff:.4f}")
print(f"[{TASK_ID}] CV ratio (low/high): {cv_ratio:.4f}")

# ========== Per-sample CV for statistical test ==========
# For each prompt, compute the "aggregate CV" separately for absorbed vs non-absorbed groups
sample_cv_low = []
sample_cv_high = []

for sample_idx in range(len(prompts)):
    for pos_idx in range(max_len):
        # Get activations for this position
        acts_at_pos = sae_features[sample_idx, pos_idx].cpu().numpy()  # [d_sae]

        # Split by group
        absorbed_acts = acts_at_pos[absorbed_mask]
        non_absorbed_acts = acts_at_pos[non_absorbed_mask]

        # CV for this position's absorbed features
        if len(absorbed_acts) > 0 and absorbed_acts.mean() > 1e-8:
            sample_cv_low.append(absorbed_acts.std() / absorbed_acts.mean())
        if len(non_absorbed_acts) > 0 and non_absorbed_acts.mean() > 1e-8:
            sample_cv_high.append(non_absorbed_acts.std() / non_absorbed_acts.mean())

# Aggregate per-sample CV
n_samples_cv = min(len(sample_cv_low), len(sample_cv_high), 1000)
if n_samples_cv > 0:
    # Take first n_samples for comparison
    sample_cv_low_arr = np.array(sample_cv_low[:n_samples_cv])
    sample_cv_high_arr = np.array(sample_cv_high[:n_samples_cv])
else:
    sample_cv_low_arr = np.array([])
    sample_cv_high_arr = np.array([])

# Statistical test (simple t-test)
from scipy import stats
if len(sample_cv_low_arr) >= 5 and len(sample_cv_high_arr) >= 5:
    t_stat, p_value = stats.ttest_ind(sample_cv_low_arr, sample_cv_high_arr)
else:
    t_stat, p_value = 0.0, 1.0

print(f"[{TASK_ID}] t-test: t={t_stat:.4f}, p={p_value:.4f}")

# ========== Check pass criteria ==========
# Criteria:
# 1. CV computable for both groups
# 2. CV_low and CV_high have non-zero variance
# 3. Difference direction detectable (H4 predicts CV_low < CV_high)

pass_criteria = {
    "cv_computable_both_groups": bool(len(absorbed_cv) > 0 and len(non_absorbed_cv) > 0),
    "both_have_nonzero_variance": bool(cv_low_std > 0 and cv_high_std > 0),
    "difference_detectable": bool(abs(cv_diff) > 0.01),
    "h4_direction_correct": bool(cv_diff > 0),  # CV_high > CV_low means CV_low < CV_high ✓
}

overall_pass = all(pass_criteria.values())

# ========== Summary ==========
summary = {
    "task_id": TASK_ID,
    "status": "success" if overall_pass else "needs_review",
    "timestamp": datetime.now().isoformat(),
    "script_start": SCRIPT_START,
    "script_end": datetime.now().isoformat(),
    "pass_criteria": pass_criteria,
    "overall_pass": overall_pass,
    "cv_absorbed": {
        "mean": float(cv_low_mean),
        "std": float(cv_low_std),
        "n_features": len(absorbed_cv),
        "per_feature_values": [float(v) for v in absorbed_cv[:100]],  # sample
    },
    "cv_non_absorbed": {
        "mean": float(cv_high_mean),
        "std": float(cv_high_std),
        "n_features": len(non_absorbed_cv),
        "per_feature_values": [float(v) for v in non_absorbed_cv[:100]],  # sample
    },
    "cv_difference": {
        "high_minus_low": float(cv_diff),
        "ratio_low_over_high": float(cv_ratio),
        "h4_prediction": "CV_low < CV_high at critical point",
        "h4_observed": "CV_low < CV_high" if cv_diff > 0 else "CV_low >= CV_high (direction reversed)",
        "h4_direction_correct": bool(cv_diff > 0),
    },
    "statistical_test": {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0_01": bool(p_value < 0.01),
        "significant_at_0_05": bool(p_value < 0.05),
    },
    "gpu": {
        "id": 6,
        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    },
    "config": {
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "layer": LAYER,
        "absorption_threshold": ABSORPTION_THRESHOLD,
    },
}

output_path = RESULTS_DIR / f"{TASK_ID}.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[{TASK_ID}] DONE. Saved to {output_path}")
print(f"  cv_computable_both_groups: {pass_criteria['cv_computable_both_groups']}")
print(f"  both_have_nonzero_variance: {pass_criteria['both_have_nonzero_variance']}")
print(f"  difference_detectable: {pass_criteria['difference_detectable']}")
print(f"  h4_direction_correct: {pass_criteria['h4_direction_correct']}")
print(f"  overall_pass: {overall_pass}")

# ========== Write DONE marker ==========
done_path = RESULTS_DIR / f"{TASK_ID}_DONE"
done_path.write_text(json.dumps({
    "task_id": TASK_ID,
    "status": "success" if overall_pass else "needs_review",
    "timestamp": datetime.now().isoformat(),
}))

print(f"[{TASK_ID}] DONE marker written.")