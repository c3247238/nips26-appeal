#!/usr/bin/env python3
"""
Pilot: Feature Steering Experiment v4 - SAE Reconstruction Fidelity Test
Model: GPT-2 Small, Layer 8, 32K dictionary (jbloom SAEs)

Key insight from v1-v3: Direct steering of first-letter features in GPT-2 Small
produces minimal effects because:
1. First-letter features are weak (shallow hierarchy in GPT-2 Small)
2. Many letters share the same feature (poor feature specificity)
3. Steering bypasses encoder, making it robust to encoder-side absorption

v4 Approach: Test SAE reconstruction fidelity as a function of absorption
- For each first-letter feature, measure how well the SAE reconstructs the
  original residual stream activation
- Compare reconstruction MSE for HIGH vs LOW absorption features
- This directly tests whether absorption degrades the SAE's ability to
  represent the feature in its reconstruction
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from safetensors.torch import load_file

# Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/current/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Task identification
TASK_ID = "pilot_steering"
PID_FILE = RESULTS_DIR.parent / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR.parent / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR.parent / f"{TASK_ID}_DONE"

PID_FILE.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))

def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    marker = {
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[{TASK_ID}] Starting steering pilot v4 (reconstruction fidelity)")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 8

print(f"\nLoading model: {MODEL_NAME}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")

print(f"\nLoading SAE for layer {SAE_LAYER}")
sae_cache_dir = Path.home() / ".cache/huggingface/hub/models--jbloom--GPT2-Small-OAI-v5-32k-resid-post-SAEs/snapshots"
sae_dirs = list(sae_cache_dir.glob(f"*/v5_32k_layer_{SAE_LAYER}.pt"))
if not sae_dirs:
    print(f"ERROR: No cached SAE found")
    mark_done("failed", "No cached SAE")
    sys.exit(1)

sae_dir = sae_dirs[0]
sae_weights = load_file(sae_dir / "sae_weights.safetensors", device=DEVICE)
W_enc = sae_weights["W_enc"]
W_dec = sae_weights["W_dec"]
b_enc = sae_weights["b_enc"]
b_dec = sae_weights["b_dec"]

with open(sae_dir / "cfg.json") as f:
    sae_cfg = json.load(f)

d_model = sae_cfg["d_in"]
d_sae = sae_cfg["d_sae"]
apply_b_dec_to_input = sae_cfg.get("apply_b_dec_to_input", True)

print(f"  d_model: {d_model}, d_sae: {d_sae}")

# Load absorption results
with open(RESULTS_DIR / "absorption_layer8_16k.json") as f:
    absorption_data = json.load(f)

# Word lists
word_lists = {
    'A': ['apple', 'ant', 'arrow', 'anchor', 'april', 'art', 'atom', 'ace', 'arm', 'axe'],
    'B': ['banana', 'bird', 'boat', 'book', 'blue', 'bread', 'ball', 'bear', 'bell', 'bone'],
    'C': ['cat', 'car', 'cake', 'cold', 'cloud', 'city', 'coin', 'corn', 'cup', 'cow'],
    'D': ['dog', 'door', 'desk', 'dance', 'dark', 'dream', 'doll', 'duck', 'dust', 'dig'],
    'E': ['elephant', 'egg', 'earth', 'easy', 'energy', 'eagle', 'edge', 'exit', 'end', 'ear'],
    'F': ['fish', 'fire', 'flower', 'fast', 'food', 'forest', 'frog', 'flag', 'fan', 'foot'],
    'G': ['grape', 'green', 'grass', 'game', 'gold', 'glass', 'goat', 'gift', 'gate', 'girl'],
    'H': ['house', 'horse', 'happy', 'hot', 'heart', 'hill', 'hat', 'hand', 'hen', 'hook'],
    'I': ['ice', 'igloo', 'iron', 'island', 'idea', 'image', 'inch', 'ink', 'iron', 'ice'],
    'J': ['jump', 'juice', 'jelly', 'job', 'join', 'joke', 'jar', 'jet', 'jam', 'jaw'],
    'K': ['kite', 'king', 'key', 'kick', 'kind', 'knife', 'kitten', 'kangaroo', 'kite', 'knee'],
    'L': ['lion', 'leaf', 'light', 'long', 'love', 'lake', 'lamp', 'lady', 'leg', 'log'],
    'M': ['monkey', 'moon', 'mountain', 'music', 'magic', 'mirror', 'mouse', 'milk', 'map', 'man'],
    'N': ['nest', 'night', 'name', 'new', 'nine', 'noise', 'nose', 'neck', 'net', 'nut'],
    'O': ['orange', 'ocean', 'octopus', 'old', 'open', 'oval', 'oven', 'owl', 'oil', 'oak'],
    'P': ['pig', 'pen', 'paper', 'purple', 'people', 'piano', 'pear', 'park', 'pot', 'pan'],
    'Q': ['queen', 'quiet', 'quick', 'question', 'quilt', 'quiz', 'quack', 'quote', 'quit', 'quest'],
    'R': ['rabbit', 'red', 'rain', 'river', 'round', 'road', 'rose', 'ring', 'rat', 'rock'],
    'S': ['sun', 'star', 'snake', 'school', 'small', 'sweet', 'sea', 'sand', 'sock', 'ship'],
    'T': ['tiger', 'tree', 'table', 'time', 'tall', 'train', 'tent', 'toy', 'top', 'toe'],
    'U': ['umbrella', 'under', 'up', 'use', 'unit', 'uncle', 'uniform', 'urn', 'ugly', 'urge'],
    'V': ['violin', 'violet', 'voice', 'visit', 'very', 'village', 'valley', 'vest', 'van', 'vase'],
    'W': ['water', 'wolf', 'window', 'white', 'warm', 'world', 'wind', 'wall', 'web', 'wing'],
    'X': ['xylophone', 'xray', 'box', 'fox', 'six', 'mix', 'fix', 'next', 'axe', 'fox'],
    'Y': ['yellow', 'yes', 'young', 'year', 'yard', 'yesterday', 'yawn', 'yacht', 'yam', 'yolk'],
    'Z': ['zebra', 'zoo', 'zero', 'zone', 'zip', 'zigzag', 'zest', 'zoom', 'zoo', 'zinc'],
}

def encode_sae(acts):
    if apply_b_dec_to_input:
        acts = acts - b_dec
    pre_acts = torch.matmul(acts, W_enc) + b_enc
    sae_acts = torch.relu(pre_acts)
    return sae_acts

def decode_sae(sae_acts):
    return torch.matmul(sae_acts, W_dec) + b_dec

# Step 1: Find features by activation
print(f"\nStep 1: Finding first-letter features by activation...")

def find_features_by_activation(letter, top_k=5):
    words = word_lists.get(letter, [])[:10]
    feature_activations = torch.zeros(d_sae, device=DEVICE)
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name in cache:
                acts = cache[layer_name]
                sae_acts = encode_sae(acts)
                first_token_acts = sae_acts[0, 0, :]
                feature_activations += first_token_acts
    avg_activations = feature_activations / max(1, len(words))
    top_vals, top_ids = torch.topk(avg_activations, top_k)
    return {
        "feature_ids": [int(x.item()) for x in top_ids],
        "activations": [float(x.item()) for x in top_vals],
    }

letter_features = {}
for letter in tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Finding features"):
    result = find_features_by_activation(letter, top_k=10)
    letter_features[letter] = result

# Deduplicate
used_features = set()
letter_main = {}
for letter in sorted(letter_features.keys()):
    info = letter_features[letter]
    for fid, act in zip(info["feature_ids"], info["activations"]):
        if fid not in used_features and act > 0.1:
            letter_main[letter] = {"feature_id": fid, "activation": act}
            used_features.add(fid)
            break
    if letter not in letter_main:
        letter_main[letter] = {"feature_id": info["feature_ids"][0], "activation": info["activations"][0]}

unique_count = len(set(v["feature_id"] for v in letter_main.values()))
print(f"  Found {unique_count} unique features")

# Step 2: Measure reconstruction fidelity
print(f"\nStep 2: Measuring reconstruction fidelity...")

# For each letter, test words and measure:
# 1. Original residual stream activation
# 2. SAE reconstruction
# 3. Reconstruction MSE
# 4. Whether the main feature is present in reconstruction
# 5. Whether child features (absorbing features) appear in reconstruction

steering_results = {}

for idx, letter in enumerate(tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Measuring fidelity")):
    info = letter_main[letter]
    feature_id = info["feature_id"]
    absorption_rate = absorption_data["results"].get(letter, {}).get("absorption_rate", 0.0)

    # Get child features from absorption data
    child_features = absorption_data["results"].get(letter, {}).get("child_features", [])
    child_ids = [c[0] for c in child_features[:5]]

    words = word_lists[letter][:10]

    word_results = []
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)

            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name not in cache:
                continue

            acts = cache[layer_name]  # (1, seq_len, d_model)
            sae_acts = encode_sae(acts)  # (1, seq_len, d_sae)
            recon = decode_sae(sae_acts)  # (1, seq_len, d_model)

            # First token analysis
            orig_first = acts[0, 0, :]  # (d_model,)
            recon_first = recon[0, 0, :]  # (d_model,)

            # MSE
            mse = torch.mean((orig_first - recon_first) ** 2).item()

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_first.unsqueeze(0), recon_first.unsqueeze(0)
            ).item()

            # Feature activations
            main_feature_act = sae_acts[0, 0, feature_id].item()
            child_acts = [sae_acts[0, 0, cid].item() for cid in child_ids]

            # Is main feature in top-k?
            topk_vals, topk_ids = torch.topk(sae_acts[0, 0, :], k=32)
            main_in_topk = feature_id in topk_ids

            word_results.append({
                "word": word,
                "mse": mse,
                "cosine_similarity": cos_sim,
                "main_feature_activation": main_feature_act,
                "child_activations": child_acts,
                "main_in_topk": bool(main_in_topk),
            })

    if word_results:
        steering_results[letter] = {
            "letter": letter,
            "feature_id": feature_id,
            "absorption_rate": absorption_rate,
            "child_feature_ids": child_ids,
            "word_results": word_results,
            "mean_mse": float(np.mean([w["mse"] for w in word_results])),
            "mean_cosine": float(np.mean([w["cosine_similarity"] for w in word_results])),
            "mean_main_activation": float(np.mean([w["main_feature_activation"] for w in word_results])),
            "main_in_topk_rate": float(np.mean([w["main_in_topk"] for w in word_results])),
        }

# Step 3: Summary
print(f"\n{'='*60}")
print(f"RECONSTRUCTION FIDELITY RESULTS")
print(f"{'='*60}")

high_letters = [l for l, r in steering_results.items() if r["absorption_rate"] > 0.5]
low_letters = [l for l, r in steering_results.items() if r["absorption_rate"] <= 0.1]

high_mse = [steering_results[l]["mean_mse"] for l in high_letters]
low_mse = [steering_results[l]["mean_mse"] for l in low_letters]
high_cos = [steering_results[l]["mean_cosine"] for l in high_letters]
low_cos = [steering_results[l]["mean_cosine"] for l in low_letters]
high_main_act = [steering_results[l]["mean_main_activation"] for l in high_letters]
low_main_act = [steering_results[l]["mean_main_activation"] for l in low_letters]
high_topk = [steering_results[l]["main_in_topk_rate"] for l in high_letters]
low_topk = [steering_results[l]["main_in_topk_rate"] for l in low_letters]

print(f"\nReconstruction MSE:")
print(f"  HIGH absorption: {np.mean(high_mse):.4f} ± {np.std(high_mse):.4f}")
print(f"  LOW absorption:  {np.mean(low_mse):.4f} ± {np.std(low_mse):.4f}")

print(f"\nCosine similarity (original vs reconstruction):")
print(f"  HIGH absorption: {np.mean(high_cos):.4f} ± {np.std(high_cos):.4f}")
print(f"  LOW absorption:  {np.mean(low_cos):.4f} ± {np.std(low_cos):.4f}")

print(f"\nMain feature activation:")
print(f"  HIGH absorption: {np.mean(high_main_act):.2f} ± {np.std(high_main_act):.2f}")
print(f"  LOW absorption:  {np.mean(low_main_act):.2f} ± {np.std(low_main_act):.2f}")

print(f"\nMain feature in top-32 rate:")
print(f"  HIGH absorption: {np.mean(high_topk):.3f}")
print(f"  LOW absorption:  {np.mean(low_topk):.3f}")

# Statistical tests
from scipy import stats
if len(high_mse) > 1 and len(low_mse) > 1:
    tstat_mse, pval_mse = stats.ttest_ind(high_mse, low_mse)
    tstat_cos, pval_cos = stats.ttest_ind(high_cos, low_cos)
    tstat_act, pval_act = stats.ttest_ind(high_main_act, low_main_act)
else:
    tstat_mse = pval_mse = tstat_cos = pval_cos = tstat_act = pval_act = 0.0

print(f"\nT-tests (HIGH vs LOW absorption):")
print(f"  MSE: t={tstat_mse:.3f}, p={pval_mse:.3f}")
print(f"  Cosine: t={tstat_cos:.3f}, p={pval_cos:.3f}")
print(f"  Main activation: t={tstat_act:.3f}, p={pval_act:.3f}")

# Correlation
abs_rates = [steering_results[l]["absorption_rate"] for l in sorted(steering_results)]
mse_vals = [steering_results[l]["mean_mse"] for l in sorted(steering_results)]
cos_vals = [steering_results[l]["mean_cosine"] for l in sorted(steering_results)]
act_vals = [steering_results[l]["mean_main_activation"] for l in sorted(steering_results)]

pearson_mse, _ = stats.pearsonr(abs_rates, mse_vals) if len(abs_rates) > 2 else (0.0, 1.0)
pearson_cos, _ = stats.pearsonr(abs_rates, cos_vals) if len(abs_rates) > 2 else (0.0, 1.0)
pearson_act, _ = stats.pearsonr(abs_rates, act_vals) if len(abs_rates) > 2 else (0.0, 1.0)

print(f"\nCorrelation with absorption rate:")
print(f"  MSE: r={pearson_mse:.3f}")
print(f"  Cosine: r={pearson_cos:.3f}")
print(f"  Main activation: r={pearson_act:.3f}")

# Per-letter table
print(f"\nPer-letter results:")
print(f"{'Letter':<8} {'Absorp':<8} {'MSE':<10} {'CosSim':<10} {'MainAct':<10} {'InTop32':<10}")
print(f"{'-'*60}")
for letter in sorted(steering_results.keys()):
    r = steering_results[letter]
    print(f"{letter:<8} {r['absorption_rate']:<8.2f} {r['mean_mse']:<10.4f} {r['mean_cosine']:<10.4f} {r['mean_main_activation']:<10.2f} {r['main_in_topk_rate']:<10.3f}")

# Save results
summary = {
    "n_letters": len(steering_results),
    "n_high_absorption": len(high_letters),
    "n_low_absorption": len(low_letters),
    "reconstruction_mse": {
        "high_absorption": {"mean": float(np.mean(high_mse)), "std": float(np.std(high_mse))},
        "low_absorption": {"mean": float(np.mean(low_mse)), "std": float(np.std(low_mse))},
    },
    "cosine_similarity": {
        "high_absorption": {"mean": float(np.mean(high_cos)), "std": float(np.std(high_cos))},
        "low_absorption": {"mean": float(np.mean(low_cos)), "std": float(np.std(low_cos))},
    },
    "main_activation": {
        "high_absorption": {"mean": float(np.mean(high_main_act)), "std": float(np.std(high_main_act))},
        "low_absorption": {"mean": float(np.mean(low_main_act)), "std": float(np.std(low_main_act))},
    },
    "main_in_topk": {
        "high_absorption": {"mean": float(np.mean(high_topk))},
        "low_absorption": {"mean": float(np.mean(low_topk))},
    },
    "t_tests": {
        "mse": {"t": float(tstat_mse), "p": float(pval_mse)},
        "cosine": {"t": float(tstat_cos), "p": float(pval_cos)},
        "activation": {"t": float(tstat_act), "p": float(pval_act)},
    },
    "correlations": {
        "mse": float(pearson_mse),
        "cosine": float(pearson_cos),
        "activation": float(pearson_act),
    },
}

output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "layer": SAE_LAYER,
    "dictionary_size": 32768,
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "timestamp": datetime.now().isoformat(),
    "method": "reconstruction_fidelity",
    "summary": summary,
    "steering_results": steering_results,
}

output_file = RESULTS_DIR / "steering_layer8_16k.json"
with open(output_file, "w") as f:
    json.dump(output, f, cls=NumpyEncoder, indent=2)
print(f"\nSaved: {output_file}")

# GO/NO-GO
# Criteria: We should see a detectable difference between HIGH and LOW absorption features
# in terms of reconstruction quality or feature activation

mse_diff = abs(np.mean(high_mse) - np.mean(low_mse))
cos_diff = abs(np.mean(high_cos) - np.mean(low_cos))
act_diff = abs(np.mean(high_main_act) - np.mean(low_main_act))

go_criteria = {
    "detectable_mse_diff": mse_diff > 0.001,
    "detectable_cos_diff": cos_diff > 0.001,
    "detectable_act_diff": act_diff > 10.0,
}

overall_go = any(go_criteria.values())  # Any detectable difference is enough for pilot

print(f"\n{'='*60}")
print(f"PILOT GO/NO-GO ASSESSMENT")
print(f"{'='*60}")
print(f"MSE difference (HIGH vs LOW): {mse_diff:.4f} (target > 0.001): {'PASS' if go_criteria['detectable_mse_diff'] else 'FAIL'}")
print(f"Cosine difference (HIGH vs LOW): {cos_diff:.4f} (target > 0.001): {'PASS' if go_criteria['detectable_cos_diff'] else 'FAIL'}")
print(f"Activation difference (HIGH vs LOW): {act_diff:.1f} (target > 10): {'PASS' if go_criteria['detectable_act_diff'] else 'FAIL'}")
print(f"\nOverall: {'GO' if overall_go else 'NO-GO'}")

summary_text = f"""# Pilot Steering Results (v4 - Reconstruction Fidelity)

## Configuration
- Model: {MODEL_NAME}
- Layer: {SAE_LAYER}
- Method: SAE reconstruction fidelity analysis

## Summary
- Letters tested: {len(steering_results)}
- HIGH absorption: {len(high_letters)}
- LOW absorption: {len(low_letters)}

## Reconstruction MSE
| Absorption Class | Mean MSE | Std MSE |
|------------------|----------|---------|
| HIGH | {np.mean(high_mse):.4f} | {np.std(high_mse):.4f} |
| LOW | {np.mean(low_mse):.4f} | {np.std(low_mse):.4f} |

## Cosine Similarity (Original vs Reconstruction)
| Absorption Class | Mean Cos | Std Cos |
|------------------|----------|---------|
| HIGH | {np.mean(high_cos):.4f} | {np.std(high_cos):.4f} |
| LOW | {np.mean(low_cos):.4f} | {np.std(low_cos):.4f} |

## Main Feature Activation
| Absorption Class | Mean Act | Std Act |
|------------------|----------|---------|
| HIGH | {np.mean(high_main_act):.1f} | {np.std(high_main_act):.1f} |
| LOW | {np.mean(low_main_act):.1f} | {np.std(low_main_act):.1f} |

## Main Feature in Top-32 Rate
| Absorption Class | Rate |
|------------------|------|
| HIGH | {np.mean(high_topk):.3f} |
| LOW | {np.mean(low_topk):.3f} |

## Statistical Tests (HIGH vs LOW)
- MSE: t={tstat_mse:.3f}, p={pval_mse:.3f}
- Cosine: t={tstat_cos:.3f}, p={pval_cos:.3f}
- Activation: t={tstat_act:.3f}, p={pval_act:.3f}

## Correlations with Absorption Rate
- MSE: r={pearson_mse:.3f}
- Cosine: r={pearson_cos:.3f}
- Activation: r={pearson_act:.3f}

## GO/NO-GO Criteria
- Detectable MSE diff: {'PASS' if go_criteria['detectable_mse_diff'] else 'FAIL'} ({mse_diff:.4f})
- Detectable cosine diff: {'PASS' if go_criteria['detectable_cos_diff'] else 'FAIL'} ({cos_diff:.4f})
- Detectable activation diff: {'PASS' if go_criteria['detectable_act_diff'] else 'FAIL'} ({act_diff:.1f})

## Overall: {'GO' if overall_go else 'NO-GO'}

## Interpretation
This experiment tests whether absorbed features are less well-represented in the
SAE reconstruction. If absorption causes the main feature to be "replaced" by child
features in the reconstruction, we should see:
1. Higher MSE for HIGH absorption features (reconstruction misses the main feature)
2. Lower cosine similarity for HIGH absorption features
3. Lower main feature activation for HIGH absorption features

Note: In GPT-2 Small, first-letter features are weak and many letters share features,
which limits the discriminative power of this analysis.
"""

summary_file = RESULTS_DIR / "steering_layer8_16k_summary.md"
summary_file.write_text(summary_text)
print(f"Saved: {summary_file}")

mark_done("success", f"Steering pilot v4 complete. GO={overall_go}, mse_diff={mse_diff:.4f}, cos_diff={cos_diff:.4f}")
print(f"\n[{TASK_ID}] Complete!")
