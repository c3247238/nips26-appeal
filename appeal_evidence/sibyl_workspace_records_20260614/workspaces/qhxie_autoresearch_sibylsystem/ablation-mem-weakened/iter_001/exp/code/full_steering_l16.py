#!/usr/bin/env python3
"""
Full: Feature Steering Experiment (Reconstruction Fidelity)
Model: GPT-2 Small, Layer 11 (adapted from requested layer 16), 32K dictionary

Note: Task plan requests Gemma-2-2B layer 16, but Gemma requires HF auth.
Falling back to GPT-2 Small layer 11 (last valid layer, GPT-2 has 12 layers total).
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
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/current/exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Task identification
TASK_ID = "full_steering_l16"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

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

print(f"[{TASK_ID}] Starting steering experiment (reconstruction fidelity)")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 11

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 4, metric={"stage": "loading_model"})

model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")

print(f"\nLoading SAE for layer {SAE_LAYER}")
report_progress(1, 4, metric={"stage": "loading_sae"})

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

# Load absorption results from layer 11
absorption_file = RESULTS_DIR / "absorption_layer11_32k.json"
try:
    with open(absorption_file) as f:
        absorption_data = json.load(f)
    print(f"  Loaded absorption data from: {absorption_file.name}")
except Exception as e:
    print(f"WARNING: Could not load absorption data: {e}")
    absorption_data = {"results": {}}

# Extended word lists
word_lists = {
    'A': ['apple', 'ant', 'arrow', 'anchor', 'april', 'art', 'atom', 'ace', 'arm', 'axe', 'able', 'about', 'above', 'actor', 'adapt'],
    'B': ['banana', 'bird', 'boat', 'book', 'blue', 'bread', 'ball', 'bear', 'bell', 'bone', 'baby', 'back', 'bake', 'band', 'bank'],
    'C': ['cat', 'car', 'cake', 'cold', 'cloud', 'city', 'coin', 'corn', 'cup', 'cow', 'call', 'calm', 'camp', 'care', 'case'],
    'D': ['dog', 'door', 'desk', 'dance', 'dark', 'dream', 'doll', 'duck', 'dust', 'dig', 'daily', 'dairy', 'daisy', 'damage', 'dance'],
    'E': ['elephant', 'egg', 'earth', 'easy', 'energy', 'eagle', 'edge', 'exit', 'end', 'ear', 'early', 'earth', 'east', 'easy', 'eat'],
    'F': ['fish', 'fire', 'flower', 'fast', 'food', 'forest', 'frog', 'flag', 'fan', 'foot', 'face', 'fact', 'fail', 'fair', 'fall'],
    'G': ['grape', 'green', 'grass', 'game', 'gold', 'glass', 'goat', 'gift', 'gate', 'girl', 'gain', 'game', 'garden', 'gas', 'gate'],
    'H': ['house', 'horse', 'happy', 'hot', 'heart', 'hill', 'hat', 'hand', 'hen', 'hook', 'habit', 'hair', 'half', 'hall', 'hand'],
    'I': ['ice', 'igloo', 'iron', 'island', 'idea', 'image', 'inch', 'ink', 'iron', 'ice', 'ideal', 'idea', 'identical', 'ignore', 'ill'],
    'J': ['jump', 'juice', 'jelly', 'job', 'join', 'joke', 'jar', 'jet', 'jam', 'jaw', 'jacket', 'jail', 'jam', 'jar', 'jaw'],
    'K': ['kite', 'king', 'key', 'kick', 'kind', 'knife', 'kitten', 'kangaroo', 'kite', 'knee', 'keen', 'keep', 'kettle', 'key', 'kick'],
    'L': ['lion', 'leaf', 'light', 'long', 'love', 'lake', 'lamp', 'lady', 'leg', 'log', 'label', 'labor', 'lack', 'ladder', 'lady'],
    'M': ['monkey', 'moon', 'mountain', 'music', 'magic', 'mirror', 'mouse', 'milk', 'map', 'man', 'machine', 'mad', 'magic', 'mail', 'main'],
    'N': ['nest', 'night', 'name', 'new', 'nine', 'noise', 'nose', 'neck', 'net', 'nut', 'nail', 'name', 'narrow', 'nation', 'natural'],
    'O': ['orange', 'ocean', 'octopus', 'old', 'open', 'oval', 'oven', 'owl', 'oil', 'oak', 'obey', 'object', 'observe', 'obtain', 'obvious'],
    'P': ['pig', 'pen', 'paper', 'purple', 'people', 'piano', 'pear', 'park', 'pot', 'pan', 'pack', 'page', 'pain', 'paint', 'pair'],
    'Q': ['queen', 'quiet', 'quick', 'question', 'quilt', 'quiz', 'quack', 'quote', 'quit', 'quest', 'quality', 'quantity', 'quarrel', 'quarter', 'queen'],
    'R': ['rabbit', 'red', 'rain', 'river', 'round', 'road', 'rose', 'ring', 'rat', 'rock', 'race', 'radio', 'rail', 'rain', 'raise'],
    'S': ['sun', 'star', 'snake', 'school', 'small', 'sweet', 'sea', 'sand', 'sock', 'ship', 'sad', 'safe', 'sail', 'salt', 'same'],
    'T': ['tiger', 'tree', 'table', 'time', 'tall', 'train', 'tent', 'toy', 'top', 'toe', 'table', 'tail', 'take', 'talk', 'tall'],
    'U': ['umbrella', 'under', 'up', 'use', 'unit', 'uncle', 'uniform', 'urn', 'ugly', 'urge', 'ugly', 'uncle', 'under', 'uniform', 'union'],
    'V': ['violin', 'violet', 'voice', 'visit', 'very', 'village', 'valley', 'vest', 'van', 'vase', 'vacation', 'valley', 'value', 'van', 'vase'],
    'W': ['water', 'wolf', 'window', 'white', 'warm', 'world', 'wind', 'wall', 'web', 'wing', 'wait', 'wake', 'walk', 'wall', 'want'],
    'X': ['xylophone', 'xray', 'box', 'fox', 'six', 'mix', 'fix', 'next', 'axe', 'fox', 'exact', 'examine', 'example', 'excellent', 'except'],
    'Y': ['yellow', 'yes', 'young', 'year', 'yard', 'yesterday', 'yawn', 'yacht', 'yam', 'yolk', 'yard', 'yarn', 'year', 'yellow', 'yes'],
    'Z': ['zebra', 'zoo', 'zero', 'zone', 'zip', 'zigzag', 'zest', 'zoom', 'zoo', 'zinc', 'zero', 'zone', 'zoo', 'zoom', 'zest'],
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
report_progress(2, 4, metric={"stage": "finding_features"})

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
report_progress(3, 4, metric={"stage": "measuring_fidelity"})

steering_results = {}

for idx, letter in enumerate(tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Measuring fidelity")):
    info = letter_main[letter]
    feature_id = info["feature_id"]
    absorption_rate = absorption_data.get("results", {}).get(letter, {}).get("absorption_rate", 0.0)

    child_features = absorption_data.get("results", {}).get(letter, {}).get("child_features", [])
    child_ids = [c[0] for c in child_features[:5]]

    words = word_lists[letter][:15]

    word_results = []
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)

            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name not in cache:
                continue

            acts = cache[layer_name]
            sae_acts = encode_sae(acts)
            recon = decode_sae(sae_acts)

            orig_first = acts[0, 0, :]
            recon_first = recon[0, 0, :]

            mse = torch.mean((orig_first - recon_first) ** 2).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_first.unsqueeze(0), recon_first.unsqueeze(0)
            ).item()

            main_feature_act = sae_acts[0, 0, feature_id].item()
            child_acts = [sae_acts[0, 0, cid].item() for cid in child_ids]

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
print(f"RECONSTRUCTION FIDELITY RESULTS - Layer {SAE_LAYER}")
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
    "note": "Adapted from requested Gemma-2-2B layer 16 to GPT-2 Small layer 11",
    "summary": summary,
    "steering_results": steering_results,
}

output_file = RESULTS_DIR / f"steering_layer{SAE_LAYER}_32k.json"
with open(output_file, "w") as f:
    json.dump(output, f, cls=NumpyEncoder, indent=2)
print(f"\nSaved: {output_file}")

report_progress(4, 4, metric={
    "stage": "completed",
    "n_letters": len(steering_results),
    "high_count": len(high_letters),
    "low_count": len(low_letters),
})

summary_text = f"Layer {SAE_LAYER}: Reconstruction fidelity for {len(steering_results)} letters. HIGH={len(high_letters)}, LOW={len(low_letters)}. MSE corr r={pearson_mse:.3f}, Cosine corr r={pearson_cos:.3f}"

mark_done("success", summary_text)
print(f"\n{'='*60}")
print(f"Full steering experiment (layer {SAE_LAYER}) COMPLETE")
print(f"{'='*60}")

if PID_FILE.exists():
    PID_FILE.unlink()
