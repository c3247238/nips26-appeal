#!/usr/bin/env python3
"""
Pilot: Absorption Detection for First-Letter Features (A-Z)
Model: GPT-2 Small, Layer 8, 32K dictionary (jbloom SAEs)
Metric: Custom absorption metric based on decoder cosine similarity

Approach:
1. Use decoder directions (W_dec) to find first-letter features
2. For each feature, find "child" features via decoder cosine similarity
3. Measure absorption: when parent fires, do children also fire?
   (Positive correlation = no absorption, negative = absorption)
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
TASK_ID = "pilot_absorption"
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

print(f"[{TASK_ID}] Starting absorption detection pilot")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")
print(f"  Seed: {SEED}")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 8
SAE_DICT_SIZE = 32768

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 5, metric={"stage": "loading_model"})

model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")

print(f"\nLoading SAE for layer {SAE_LAYER}")
report_progress(1, 5, metric={"stage": "loading_sae"})

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

# Step 1: Find first-letter features using decoder directions
print(f"\nStep 1: Finding first-letter features via decoder directions...")
report_progress(2, 5, metric={"stage": "finding_features"})

# For each letter, compute the average decoder direction of tokens starting with that letter
# Then find SAE features with highest cosine similarity to that direction

def get_token_directions(letter):
    """Get average residual stream direction for first tokens of words starting with letter."""
    words = word_lists.get(letter, [])[:10]
    directions = []
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name in cache:
                # Get first token's residual stream direction
                acts = cache[layer_name][0, 0, :]  # First token
                directions.append(acts)
    if not directions:
        return None
    return torch.stack(directions).mean(dim=0)

def find_features_for_letter(letter, top_k=10):
    """Find SAE features most aligned with a letter's token direction."""
    direction = get_token_directions(letter)
    if direction is None:
        return None

    # Compute cosine similarity between direction and each decoder vector
    # W_dec: (d_sae, d_model)
    direction_norm = direction / (direction.norm() + 1e-8)
    dec_norms = W_dec.norm(dim=1, keepdim=True) + 1e-8
    W_dec_normalized = W_dec / dec_norms

    cos_sims = torch.matmul(W_dec_normalized, direction_norm)  # (d_sae,)

    # Get top features by cosine similarity
    top_values, top_indices = torch.topk(cos_sims, top_k)

    return {
        "feature_ids": [int(x.item()) for x in top_indices],
        "cosine_similarities": [float(x.item()) for x in top_values],
        "direction": direction.cpu().numpy().tolist(),
    }

letter_features = {}
for letter in tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Finding features"):
    result = find_features_for_letter(letter, top_k=20)
    if result:
        letter_features[letter] = result

print(f"  Found features for {len(letter_features)} letters")

# Deduplicate features across letters
used_features = set()
letter_main_features = {}
for letter in sorted(letter_features.keys()):
    info = letter_features[letter]
    for fid, cos_sim in zip(info["feature_ids"], info["cosine_similarities"]):
        if fid not in used_features and cos_sim > 0.01:  # Minimum similarity threshold
            letter_main_features[letter] = {
                "feature_id": fid,
                "cosine_similarity": cos_sim,
            }
            used_features.add(fid)
            break
    if letter not in letter_main_features:
        # Fallback: use best feature even if used
        letter_main_features[letter] = {
            "feature_id": info["feature_ids"][0],
            "cosine_similarity": info["cosine_similarities"][0],
        }

unique_features = len(set(v["feature_id"] for v in letter_main_features.values()))
print(f"  Unique main features: {unique_features} / {len(letter_main_features)}")
for letter, info in sorted(letter_main_features.items()):
    print(f"    {letter}: feature_id={info['feature_id']}, cos_sim={info['cosine_similarity']:.4f}")

# Step 2: Compute absorption metric
print(f"\nStep 2: Computing absorption metric...")
report_progress(3, 5, metric={"stage": "computing_absorption", "unique_features": unique_features})

def compute_absorption(feature_id, letter, n_words=15):
    """
    Compute absorption for a feature using the Chanin-inspired metric.

    For each test word:
    1. Check if the main feature fires
    2. Find child features (other features with high decoder cosine similarity to main)
    3. Check if children fire when main doesn't

    Absorption rate = fraction of cases where main doesn't fire but children do.
    """
    words = word_lists.get(letter, [])[:n_words]

    # Find child features: features with high decoder cosine similarity to parent
    parent_dec = W_dec[feature_id]  # (d_model,)
    parent_dec_norm = parent_dec / (parent_dec.norm() + 1e-8)

    dec_norms = W_dec.norm(dim=1) + 1e-8
    W_dec_norm = W_dec / dec_norms.unsqueeze(1)

    child_cos_sims = torch.matmul(W_dec_norm, parent_dec_norm)  # (d_sae,)

    # Exclude self and get top children
    child_cos_sims[feature_id] = -1.0
    top_child_vals, top_child_ids = torch.topk(child_cos_sims, k=10)

    child_features = [(int(cid.item()), float(cval.item()))
                      for cid, cval in zip(top_child_ids, top_child_vals)
                      if cval > 0.1]

    # Evaluate on test words
    results = []
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)

            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name not in cache:
                continue

            acts = cache[layer_name]
            sae_acts = encode_sae(acts)

            # First token
            first_token_acts = sae_acts[0, 0, :].cpu().numpy()

            main_fired = first_token_acts[feature_id] > 0.1
            main_activation = float(first_token_acts[feature_id])

            # Check child activations
            child_activations = []
            child_fired_count = 0
            for cid, ccos in child_features:
                cact = float(first_token_acts[cid])
                child_activations.append((cid, ccos, cact))
                if cact > 0.1:
                    child_fired_count += 1

            # Absorption criteria:
            # - Main feature did NOT fire
            # - At least one child feature fired
            is_absorbed = (not main_fired) and (child_fired_count > 0)

            results.append({
                "word": word,
                "main_fired": main_fired,
                "main_activation": main_activation,
                "child_fired_count": child_fired_count,
                "is_absorbed": is_absorbed,
                "child_activations": child_activations,
            })

    if not results:
        return {"absorption_rate": 0.0, "error": "no results"}

    n_total = len(results)
    n_absorbed = sum(1 for r in results if r["is_absorbed"])
    n_main_fired = sum(1 for r in results if r["main_fired"])
    n_main_not_fired = n_total - n_main_fired

    absorption_rate = n_absorbed / max(1, n_total)
    conditional_rate = n_absorbed / max(1, n_main_not_fired) if n_main_not_fired > 0 else 0.0

    return {
        "absorption_rate": float(absorption_rate),
        "conditional_rate": float(conditional_rate),
        "n_total": n_total,
        "n_absorbed": n_absorbed,
        "n_main_fired": n_main_fired,
        "n_main_not_fired": n_main_not_fired,
        "child_features": child_features,
        "per_word": results,
    }

absorption_results = {}
classification = {"HIGH": [], "MEDIUM": [], "LOW": []}

for letter in tqdm(sorted(letter_main_features.keys()), desc="Computing absorption"):
    info = letter_main_features[letter]
    result = compute_absorption(info["feature_id"], letter, n_words=15)
    result["letter"] = letter
    result["feature_id"] = info["feature_id"]
    result["cosine_similarity"] = info["cosine_similarity"]

    absorption_results[letter] = result

    rate = result["absorption_rate"]
    if rate > 0.5:
        classification["HIGH"].append(letter)
    elif rate > 0.1:
        classification["MEDIUM"].append(letter)
    else:
        classification["LOW"].append(letter)

# Summary
absorption_rates = [r["absorption_rate"] for r in absorption_results.values()]
print(f"\n{'='*60}")
print(f"ABSORPTION DETECTION RESULTS")
print(f"{'='*60}")
print(f"Features analyzed: {len(absorption_results)}")
print(f"Unique features: {len(set(r['feature_id'] for r in absorption_results.values()))}")
print(f"Mean absorption rate: {np.mean(absorption_rates):.4f}")
print(f"Std absorption rate: {np.std(absorption_rates):.4f}")
print(f"Min absorption rate: {np.min(absorption_rates):.4f}")
print(f"Max absorption rate: {np.max(absorption_rates):.4f}")
print(f"Median absorption rate: {np.median(absorption_rates):.4f}")
print(f"")
print(f"Classification:")
print(f"  HIGH (>50%):     {len(classification['HIGH'])} features {classification['HIGH']}")
print(f"  MEDIUM (10-50%): {len(classification['MEDIUM'])} features {classification['MEDIUM']}")
print(f"  LOW (<10%):      {len(classification['LOW'])} features {classification['LOW']}")
print(f"")

print(f"Per-letter absorption rates:")
for letter in sorted(absorption_results.keys()):
    r = absorption_results[letter]
    print(f"  {letter}: A={r['absorption_rate']:.4f}, conditional={r['conditional_rate']:.4f}, "
          f"main_fired={r['n_main_fired']}/{r['n_total']}, feat={r['feature_id']}")

# Save results
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
    "method": "decoder_cosine_similarity_with_child_firing",
    "summary": {
        "n_features": len(absorption_results),
        "n_unique_features": len(set(r["feature_id"] for r in absorption_results.values())),
        "mean_absorption": float(np.mean(absorption_rates)),
        "std_absorption": float(np.std(absorption_rates)),
        "min_absorption": float(np.min(absorption_rates)),
        "max_absorption": float(np.max(absorption_rates)),
        "median_absorption": float(np.median(absorption_rates)),
        "high_count": len(classification["HIGH"]),
        "medium_count": len(classification["MEDIUM"]),
        "low_count": len(classification["LOW"]),
    },
    "classification": classification,
    "results": absorption_results,
    "letter_features": {k: {"feature_id": v["feature_id"], "cosine_similarity": v["cosine_similarity"]} for k, v in letter_main_features.items()},
}

output_file = RESULTS_DIR / "absorption_layer8_16k.json"
output_file.write_text(json.dumps(output, cls=NumpyEncoder, indent=2))
print(f"\nResults saved to: {output_file}")

import csv
csv_file = RESULTS_DIR / "absorption_layer8_16k.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["letter", "feature_id", "absorption_rate", "conditional_rate",
                     "n_total", "n_absorbed", "n_main_fired", "cosine_similarity"])
    for letter in sorted(absorption_results.keys()):
        r = absorption_results[letter]
        writer.writerow([letter, r["feature_id"], f"{r['absorption_rate']:.4f}",
                        f"{r['conditional_rate']:.4f}", r["n_total"],
                        r["n_absorbed"], r["n_main_fired"],
                        f"{r['cosine_similarity']:.4f}"])
print(f"CSV saved to: {csv_file}")

report_progress(5, 5, metric={
    "stage": "completed",
    "n_features": len(absorption_results),
    "mean_absorption": float(np.mean(absorption_rates)),
    "high_count": len(classification["HIGH"]),
})

summary_text = f"Detected absorption for {len(absorption_results)} first-letter features ({len(set(r['feature_id'] for r in absorption_results.values()))} unique). "
summary_text += f"Mean={np.mean(absorption_rates):.3f}, HIGH={len(classification['HIGH'])}, "
summary_text += f"MEDIUM={len(classification['MEDIUM'])}, LOW={len(classification['LOW'])}"

mark_done(status="success", summary=summary_text)
print(f"\n{'='*60}")
print(f"Pilot absorption detection COMPLETE")
print(f"{'='*60}")

if PID_FILE.exists():
    PID_FILE.unlink()
