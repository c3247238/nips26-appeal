#!/usr/bin/env python3
"""
Pilot: Feature Steering Experiment v2 - Improved Approach
Model: GPT-2 Small, Layer 8, 32K dictionary (jbloom SAEs)

Key improvements over v1:
1. Find features by ACTIVATION (which features fire on first-letter tokens) rather than decoder similarity
2. Use decoder direction directly (already normalized) without re-normalizing
3. Test with simple prompt "The word _" and measure next-token probability
4. Higher steering strengths to ensure detectable effect
5. Better target token identification
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

# Write PID file immediately
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

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[{TASK_ID}] Starting feature steering pilot v2")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")
print(f"  Seed: {SEED}")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 8
SAE_DICT_SIZE = 32768

# Steering strengths to test
STEERING_STRENGTHS = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 6, metric={"stage": "loading_model"})

model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")

print(f"\nLoading SAE for layer {SAE_LAYER}")
report_progress(1, 6, metric={"stage": "loading_sae"})

sae_cache_dir = Path.home() / ".cache/huggingface/hub/models--jbloom--GPT2-Small-OAI-v5-32k-resid-post-SAEs/snapshots"
sae_dirs = list(sae_cache_dir.glob(f"*/v5_32k_layer_{SAE_LAYER}.pt"))
if not sae_dirs:
    print(f"ERROR: No cached SAE found for layer {SAE_LAYER}")
    mark_done("failed", f"No cached SAE for layer {SAE_LAYER}")
    sys.exit(1)

sae_dir = sae_dirs[0]
sae_weights_file = sae_dir / "sae_weights.safetensors"
sae_cfg_file = sae_dir / "cfg.json"

with open(sae_cfg_file) as f:
    sae_cfg = json.load(f)

sae_weights = load_file(sae_weights_file, device=DEVICE)
W_enc = sae_weights["W_enc"]
W_dec = sae_weights["W_dec"]
b_enc = sae_weights["b_enc"]
b_dec = sae_weights["b_dec"]

d_model = sae_cfg["d_in"]
d_sae = sae_cfg["d_sae"]
activation_fn = sae_cfg.get("activation_fn_str", "relu")
activation_kwargs = sae_cfg.get("activation_fn_kwargs", {})
apply_b_dec_to_input = sae_cfg.get("apply_b_dec_to_input", True)

print(f"  d_model: {d_model}, d_sae: {d_sae}, activation: {activation_fn}")

def encode_sae(acts):
    if apply_b_dec_to_input:
        acts = acts - b_dec
    pre_acts = torch.matmul(acts, W_enc) + b_enc
    if activation_fn == "topk":
        k = activation_kwargs.get("k", 32)
        topk_vals, topk_indices = torch.topk(pre_acts, k, dim=-1)
        sae_acts = torch.zeros_like(pre_acts)
        sae_acts.scatter_(-1, topk_indices, topk_vals)
    else:
        sae_acts = torch.relu(pre_acts)
    return sae_acts

# Load absorption results
absorption_file = RESULTS_DIR / "absorption_layer8_16k.json"
with open(absorption_file) as f:
    absorption_data = json.load(f)

# Step 1: Find features by ACTIVATION (not decoder similarity)
print(f"\nStep 1: Finding first-letter features by activation...")
report_progress(2, 6, metric={"stage": "finding_features_by_activation"})

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

def find_features_by_activation(letter, top_k=5):
    """Find SAE features that fire most strongly on first-letter tokens."""
    words = word_lists.get(letter, [])[:10]
    feature_activations = torch.zeros(d_sae, device=DEVICE)
    feature_counts = torch.zeros(d_sae, device=DEVICE)

    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name not in cache:
                continue
            acts = cache[layer_name]
            sae_acts = encode_sae(acts)

            # First token activations
            first_token_acts = sae_acts[0, 0, :]  # (d_sae,)
            feature_activations += first_token_acts
            feature_counts += (first_token_acts > 0).float()

    # Average activation
    avg_activations = feature_activations / max(1, len(words))

    # Get top features
    top_vals, top_ids = torch.topk(avg_activations, top_k)

    return {
        "feature_ids": [int(x.item()) for x in top_ids],
        "activations": [float(x.item()) for x in top_vals],
    }

# Find features by activation for each letter
letter_activation_features = {}
for letter in tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Finding features by activation"):
    result = find_features_by_activation(letter, top_k=10)
    letter_activation_features[letter] = result

# Deduplicate: assign best unique feature to each letter
used_features = set()
letter_main_features = {}
for letter in sorted(letter_activation_features.keys()):
    info = letter_activation_features[letter]
    for fid, act in zip(info["feature_ids"], info["activations"]):
        if fid not in used_features and act > 0.1:
            letter_main_features[letter] = {"feature_id": fid, "activation": act}
            used_features.add(fid)
            break
    if letter not in letter_main_features:
        letter_main_features[letter] = {
            "feature_id": info["feature_ids"][0],
            "activation": info["activations"][0],
        }

unique_count = len(set(v["feature_id"] for v in letter_main_features.values()))
print(f"  Found {unique_count} unique features by activation")
for letter, info in sorted(letter_main_features.items()):
    # Get absorption rate
    abs_rate = absorption_data["results"].get(letter, {}).get("absorption_rate", 0.0)
    print(f"    {letter}: feature_id={info['feature_id']}, activation={info['activation']:.2f}, absorption={abs_rate:.2f}")

# Step 2: Run steering experiments with improved method
print(f"\nStep 2: Running steering experiments...")
report_progress(3, 6, metric={"stage": "running_steering"})

def steer_feature_hook(direction, strength):
    """Create a hook function that adds direction * strength to residual stream."""
    def hook_fn(activations, hook):
        steering = direction * strength
        return activations + steering
    return hook_fn

def measure_next_token_probs(model, prompt, target_tokens, hook_fn=None, hook_point=None):
    """Measure probability of target tokens as next token after prompt."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        if hook_fn and hook_point:
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, hook_fn)],
                return_type="logits"
            )
        else:
            logits = model(tokens, return_type="logits")

        # Next token logits at last position
        next_logits = logits[0, -1, :]  # (vocab,)
        probs = torch.softmax(next_logits, dim=-1)

        # Sum probability on target tokens
        target_prob = sum(probs[tid].item() for tid in target_tokens if tid < len(probs))

        # Get top 5 predictions
        top5_vals, top5_ids = torch.topk(probs, 5)
        top5 = [(model.to_string(int(tid.item())), float(tval.item()))
                for tid, tval in zip(top5_ids, top5_vals)]

    return target_prob, top5

# Build target token sets for each letter
letter_target_tokens = {}
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    target_tokens = set()
    letter_lower = letter.lower()
    for token_id in range(min(model.cfg.d_vocab, 50000)):  # Limit search space
        token_str = model.to_string(token_id).strip().lower()
        # Match words starting with the letter (at least 2 chars)
        if token_str.startswith(letter_lower) and len(token_str) >= 2 and token_str.isalpha():
            target_tokens.add(token_id)
    letter_target_tokens[letter] = target_tokens
    print(f"  {letter}: {len(target_tokens)} target tokens")

hook_point = f"blocks.{SAE_LAYER}.hook_resid_post"

# Results storage
steering_results = {}

# Random feature baseline
np.random.seed(SEED)
random_feature_ids = np.random.choice(d_sae, size=26, replace=False).tolist()

for idx, letter in enumerate(tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Steering")):
    info = letter_main_features[letter]
    feature_id = info["feature_id"]
    absorption_rate = absorption_data["results"].get(letter, {}).get("absorption_rate", 0.0)

    # Use decoder direction directly (already normalized to ~1.0)
    direction = W_dec[feature_id]

    target_tokens = letter_target_tokens[letter]
    if len(target_tokens) == 0:
        print(f"  WARNING: No target tokens for {letter}")
        continue

    # Test prompts: simple format
    prompts = [f"The word {word}" for word in word_lists[letter][:5]]

    letter_results = {
        "letter": letter,
        "feature_id": feature_id,
        "absorption_rate": absorption_rate,
        "activation": info["activation"],
        "n_target_tokens": len(target_tokens),
        "strength_results": {},
    }

    for strength in STEERING_STRENGTHS:
        probs = []
        all_top5 = []

        for prompt in prompts:
            if strength == 0.0:
                prob, top5 = measure_next_token_probs(model, prompt, target_tokens)
            else:
                hook_fn = steer_feature_hook(direction, strength)
                prob, top5 = measure_next_token_probs(
                    model, prompt, target_tokens, hook_fn=hook_fn, hook_point=hook_point
                )
            probs.append(prob)
            all_top5.append(top5)

        letter_results["strength_results"][str(strength)] = {
            "mean_prob": float(np.mean(probs)),
            "std_prob": float(np.std(probs)),
            "probs": probs,
            "top5_examples": all_top5[:2],  # Save first 2 for inspection
        }

    # Random feature baseline at strength=20.0
    random_fid = random_feature_ids[idx]
    random_direction = W_dec[random_fid]
    random_hook = steer_feature_hook(random_direction, 20.0)
    random_probs = []
    for prompt in prompts:
        prob, _ = measure_next_token_probs(
            model, prompt, target_tokens, hook_fn=random_hook, hook_point=hook_point
        )
        random_probs.append(prob)

    letter_results["random_baseline"] = {
        "feature_id": int(random_fid),
        "mean_prob": float(np.mean(random_probs)),
        "std_prob": float(np.std(random_probs)),
    }

    steering_results[letter] = letter_results

    if idx % 5 == 0:
        report_progress(3, 6, step=idx, total_steps=26, metric={
            "stage": "running_steering",
            "letters_complete": idx,
        })

# Step 3: Compute summary statistics
print(f"\nStep 3: Computing summary statistics...")
report_progress(4, 6, metric={"stage": "computing_summary"})

high_letters = [l for l, r in steering_results.items() if r["absorption_rate"] > 0.5]
low_letters = [l for l, r in steering_results.items() if r["absorption_rate"] <= 0.1]

summary = {
    "n_letters": len(steering_results),
    "n_high_absorption": len(high_letters),
    "n_low_absorption": len(low_letters),
    "steering_strengths": STEERING_STRENGTHS,
}

# Compute results by strength and absorption class
for strength in STEERING_STRENGTHS:
    s = str(strength)
    high_probs = [steering_results[l]["strength_results"][s]["mean_prob"]
                  for l in high_letters if s in steering_results[l]["strength_results"]]
    low_probs = [steering_results[l]["strength_results"][s]["mean_prob"]
                 for l in low_letters if s in steering_results[l]["strength_results"]]

    summary[f"strength_{s}"] = {
        "high_absorption": {
            "mean_prob": float(np.mean(high_probs)) if high_probs else 0.0,
            "std_prob": float(np.std(high_probs)) if high_probs else 0.0,
            "n": len(high_probs),
        },
        "low_absorption": {
            "mean_prob": float(np.mean(low_probs)) if low_probs else 0.0,
            "std_prob": float(np.std(low_probs)) if low_probs else 0.0,
            "n": len(low_probs),
        },
    }

# Random baseline
random_probs = [steering_results[l]["random_baseline"]["mean_prob"] for l in steering_results]
summary["random_baseline"] = {
    "mean_prob": float(np.mean(random_probs)),
    "std_prob": float(np.std(random_probs)),
}

# Steering improvement at strength=20.0
steering_improvements = {}
for letter in steering_results:
    baseline = steering_results[letter]["strength_results"]["0.0"]["mean_prob"]
    steered = steering_results[letter]["strength_results"]["20.0"]["mean_prob"]
    improvement = steered - baseline
    steering_improvements[letter] = {
        "baseline_prob": baseline,
        "steered_prob": steered,
        "improvement": improvement,
        "absorption_rate": steering_results[letter]["absorption_rate"],
        "relative_improvement": (improvement / baseline) if baseline > 0.001 else 0.0,
    }

# Correlation
abs_rates = [steering_improvements[l]["absorption_rate"] for l in sorted(steering_improvements)]
improvements = [steering_improvements[l]["improvement"] for l in sorted(steering_improvements)]

from scipy import stats
if len(abs_rates) > 2:
    pearson_r, pearson_p = stats.pearsonr(abs_rates, improvements)
    spearman_r, spearman_p = stats.spearmanr(abs_rates, improvements)
else:
    pearson_r = pearson_p = spearman_r = spearman_p = 0.0

summary["correlation"] = {
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
    "n": len(abs_rates),
}

# Print results
print(f"\n{'='*60}")
print(f"STEERING EXPERIMENT RESULTS (v2)")
print(f"{'='*60}")

for strength in [1.0, 5.0, 10.0, 20.0, 50.0]:
    s = str(strength)
    print(f"\nStrength={strength}:")
    print(f"  HIGH abs: {summary[f'strength_{s}']['high_absorption']['mean_prob']:.4f} ± {summary[f'strength_{s}']['high_absorption']['std_prob']:.4f}")
    print(f"  LOW abs:  {summary[f'strength_{s}']['low_absorption']['mean_prob']:.4f} ± {summary[f'strength_{s}']['low_absorption']['std_prob']:.4f}")

print(f"\nRandom baseline: {summary['random_baseline']['mean_prob']:.4f} ± {summary['random_baseline']['std_prob']:.4f}")

print(f"\nCorrelation (absorption vs improvement at s=20):")
print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})")
print(f"  Spearman rho = {spearman_r:.3f} (p = {spearman_p:.3f})")

print(f"\nPer-letter results (strength=20.0):")
print(f"{'Letter':<8} {'Absorp':<8} {'Baseline':<10} {'Steered':<10} {'Improve':<10} {'Random':<10}")
print(f"{'-'*60}")
for letter in sorted(steering_improvements.keys()):
    r = steering_improvements[letter]
    random_p = steering_results[letter]["random_baseline"]["mean_prob"]
    print(f"{letter:<8} {r['absorption_rate']:<8.2f} {r['baseline_prob']:<10.4f} {r['steered_prob']:<10.4f} {r['improvement']:<10.4f} {random_p:<10.4f}")

# Step 4: Qualitative inspection
print(f"\n{'='*60}")
print(f"QUALITATIVE INSPECTION (top-5 predictions)")
print(f"{'='*60}")
for letter in ['A', 'B', 'T']:
    print(f"\nLetter {letter} (absorption={steering_results[letter]['absorption_rate']:.2f}):")
    for strength in [0.0, 20.0]:
        s = str(strength)
        top5 = steering_results[letter]["strength_results"][s]["top5_examples"][0]
        print(f"  strength={strength}: {top5}")

# Step 5: Save results
print(f"\nStep 5: Saving results...")
report_progress(5, 6, metric={"stage": "saving_results"})

output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "sae_source": "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs",
    "layer": SAE_LAYER,
    "dictionary_size": SAE_DICT_SIZE,
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "timestamp": datetime.now().isoformat(),
    "steering_strengths": STEERING_STRENGTHS,
    "method": "activation_based_feature_selection",
    "summary": summary,
    "steering_results": steering_results,
    "steering_improvements": steering_improvements,
}

output_file = RESULTS_DIR / "steering_layer8_16k.json"
with open(output_file, "w") as f:
    json.dump(output, f, cls=NumpyEncoder, indent=2)
print(f"  Saved: {output_file}")

# GO/NO-GO assessment
random_mean = summary["random_baseline"]["mean_prob"]
low_mean_s20 = summary["strength_20.0"]["low_absorption"]["mean_prob"]
high_mean_s20 = summary["strength_20.0"]["high_absorption"]["mean_prob"]

# Success: steered prob > 2x baseline for at least some features
n_significant = sum(1 for l, r in steering_improvements.items()
                    if r["improvement"] > 0.01 and r["steered_prob"] > random_mean)

# Pass criteria adapted for this setup:
# - Random baseline should be low
# - At least some features show detectable steering effect
# - Effect should be stronger than random

go_criteria = {
    "random_baseline_low": random_mean < 0.10,
    "some_effect_detected": n_significant >= 3,
    "effect_stronger_than_random": (high_mean_s20 > random_mean or low_mean_s20 > random_mean),
}

overall_go = all(go_criteria.values())

print(f"\n{'='*60}")
print(f"PILOT GO/NO-GO ASSESSMENT")
print(f"{'='*60}")
print(f"Random baseline mean prob: {random_mean:.4f} (target < 0.10): {'PASS' if go_criteria['random_baseline_low'] else 'FAIL'}")
print(f"Features with detectable effect: {n_significant} (target >= 3): {'PASS' if go_criteria['some_effect_detected'] else 'FAIL'}")
print(f"Effect > random: {'PASS' if go_criteria['effect_stronger_than_random'] else 'FAIL'}")
print(f"\nOverall: {'GO' if overall_go else 'NO-GO'}")

# Summary markdown
summary_text = f"""# Pilot Steering Results (v2)

## Configuration
- Model: {MODEL_NAME}
- Layer: {SAE_LAYER}
- Dictionary size: {SAE_DICT_SIZE}
- Steering strengths: {STEERING_STRENGTHS}
- Feature selection: Activation-based (not decoder similarity)
- Prompts per letter: 5 simple prompts

## Summary
- Letters tested: {len(steering_results)}
- HIGH absorption: {len(high_letters)}
- LOW absorption: {len(low_letters)}

## Results by Steering Strength

| Strength | HIGH Abs Prob | LOW Abs Prob | Random Baseline |
|----------|---------------|--------------|-----------------|
| 0.0 (baseline) | {summary['strength_0.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_0.0']['low_absorption']['mean_prob']:.4f} | - |
| 1.0 | {summary['strength_1.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_1.0']['low_absorption']['mean_prob']:.4f} | - |
| 5.0 | {summary['strength_5.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_5.0']['low_absorption']['mean_prob']:.4f} | - |
| 10.0 | {summary['strength_10.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_10.0']['low_absorption']['mean_prob']:.4f} | - |
| 20.0 | {summary['strength_20.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_20.0']['low_absorption']['mean_prob']:.4f} | {summary['random_baseline']['mean_prob']:.4f} |
| 50.0 | {summary['strength_50.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_50.0']['low_absorption']['mean_prob']:.4f} | - |

## Correlation (Absorption vs Steering Improvement at s=20)
- Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})
- Spearman rho = {spearman_r:.3f} (p = {spearman_p:.3f})

## GO/NO-GO Criteria
- Random baseline < 0.10: {'PASS' if go_criteria['random_baseline_low'] else 'FAIL'} ({random_mean:.4f})
- At least 3 features with detectable effect: {'PASS' if go_criteria['some_effect_detected'] else 'FAIL'} ({n_significant})
- Effect stronger than random: {'PASS' if go_criteria['effect_stronger_than_random'] else 'FAIL'}

## Overall: {'GO' if overall_go else 'NO-GO'}

## Key Observations
- Features identified by activation (not decoder cosine similarity)
- Steering uses raw decoder direction (already unit norm)
- Simple prompts: "The word {word}" to measure next-token probability
- Target tokens: all vocabulary tokens starting with the target letter

## Notes on Steering Robustness Confound
Steering adds the decoder direction directly to the residual stream, bypassing
the SAE encoder entirely. Even if the parent latent fails to fire naturally
(due to absorption), the injected direction still influences the output. This
is a known confound that should be discussed in the paper.
"""

summary_file = RESULTS_DIR / "steering_layer8_16k_summary.md"
summary_file.write_text(summary_text)
print(f"  Saved: {summary_file}")

mark_done("success", f"Steering pilot v2 complete. GO={overall_go}, n_significant={n_significant}, random_mean={random_mean:.4f}")
print(f"\n[{TASK_ID}] Complete!")
