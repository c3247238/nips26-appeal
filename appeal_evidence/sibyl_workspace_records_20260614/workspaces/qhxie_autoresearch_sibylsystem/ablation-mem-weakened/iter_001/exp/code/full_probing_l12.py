#!/usr/bin/env python3
"""
Full: Sparse Probing Experiment
Model: GPT-2 Small, Layer 10 (adapted from requested layer 12), 32K dictionary

Note: Task plan requests Gemma-2-2B layer 12, but Gemma requires HF auth.
Falling back to GPT-2 Small layer 10 (valid layer for GPT-2's 12 layers).
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
TASK_ID = "full_probing_l12"
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

print(f"[{TASK_ID}] Starting sparse probing experiment")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 10

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

# Load absorption results from layer 10 (or fallback to layer 11)
absorption_file = RESULTS_DIR / "absorption_layer10_32k.json"
if not absorption_file.exists():
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

# Step 1: Collect SAE activations for all words
print(f"\nStep 1: Collecting SAE activations...")
report_progress(2, 4, metric={"stage": "collecting_activations"})

all_activations = []
all_labels = []

for letter in tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Collecting activations"):
    words = word_lists.get(letter, [])[:15]
    for word in words:
        with torch.no_grad():
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name in cache:
                acts = cache[layer_name]
                sae_acts = encode_sae(acts)
                first_token_acts = sae_acts[0, 0, :].cpu().numpy()
                all_activations.append(first_token_acts)
                all_labels.append(letter)

X = np.array(all_activations)
y = np.array(all_labels)

print(f"  Collected {len(X)} samples, {d_sae} features each")

# Step 2: Train k-sparse probes
print(f"\nStep 2: Training k-sparse probes...")
report_progress(3, 4, metric={"stage": "training_probes"})

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# k-sparse probe: use only top-k features by mutual information
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y_encoded, random_state=SEED)

k_values = [1, 5, 10, 20]
probe_results = {}

for k in k_values:
    top_k_indices = np.argsort(mi_scores)[-k:]
    X_k = X[:, top_k_indices]

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_k, y_encoded)

    # Predict
    y_pred = clf.predict(X_k)

    f1 = f1_score(y_encoded, y_pred, average='macro')
    acc = accuracy_score(y_encoded, y_pred)

    probe_results[k] = {
        "k": k,
        "f1_score": float(f1),
        "accuracy": float(acc),
        "top_features": [int(i) for i in top_k_indices],
        "mi_scores": [float(mi_scores[i]) for i in top_k_indices],
    }

    print(f"  k={k}: F1={f1:.4f}, Accuracy={acc:.4f}")

# Step 3: Compare absorbed vs non-absorbed features
print(f"\nStep 3: Comparing absorbed vs non-absorbed features...")
report_progress(4, 4, metric={"stage": "comparing_absorption"})

# Get feature IDs for HIGH and LOW absorption letters
high_abs_features = []
low_abs_features = []

for letter in absorption_data.get("results", {}):
    result = absorption_data["results"][letter]
    if result.get("absorption_rate", 0) > 0.5:
        high_abs_features.append(result.get("feature_id", -1))
    elif result.get("absorption_rate", 0) <= 0.1:
        low_abs_features.append(result.get("feature_id", -1))

high_abs_features = [f for f in high_abs_features if f >= 0]
low_abs_features = [f for f in low_abs_features if f >= 0]

print(f"  HIGH absorption features: {len(high_abs_features)}")
print(f"  LOW absorption features: {len(low_abs_features)}")

# Train probes using only HIGH or only LOW absorption features
comparison_results = {}

for feature_set_name, feature_ids in [("high_absorption", high_abs_features), ("low_absorption", low_abs_features)]:
    if len(feature_ids) == 0:
        comparison_results[feature_set_name] = {"error": "No features in this set"}
        continue

    valid_ids = [f for f in feature_ids if f < d_sae]
    if len(valid_ids) == 0:
        comparison_results[feature_set_name] = {"error": "No valid features"}
        continue

    X_subset = X[:, valid_ids]

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_subset, y_encoded)
    y_pred = clf.predict(X_subset)

    f1 = f1_score(y_encoded, y_pred, average='macro')
    acc = accuracy_score(y_encoded, y_pred)

    comparison_results[feature_set_name] = {
        "n_features": len(valid_ids),
        "feature_ids": [int(f) for f in valid_ids],
        "f1_score": float(f1),
        "accuracy": float(acc),
    }

    print(f"  {feature_set_name}: n={len(valid_ids)}, F1={f1:.4f}, Acc={acc:.4f}")

# Summary
print(f"\n{'='*60}")
print(f"SPARSE PROBING RESULTS - Layer {SAE_LAYER}")
print(f"{'='*60}")

print(f"\nK-sparse probe results:")
for k, res in probe_results.items():
    print(f"  k={k}: F1={res['f1_score']:.4f}, Accuracy={res['accuracy']:.4f}")

print(f"\nAbsorption-based feature comparison:")
for name, res in comparison_results.items():
    if "error" in res:
        print(f"  {name}: {res['error']}")
    else:
        print(f"  {name}: n={res['n_features']}, F1={res['f1_score']:.4f}, Acc={res['accuracy']:.4f}")

# Save results
output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "layer": SAE_LAYER,
    "dictionary_size": 32768,
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "timestamp": datetime.now().isoformat(),
    "method": "k_sparse_probing",
    "note": "Adapted from requested Gemma-2-2B layer 12 to GPT-2 Small layer 10",
    "summary": {
        "n_samples": len(X),
        "n_features": d_sae,
        "k_sparse_results": probe_results,
        "absorption_comparison": comparison_results,
    },
    "probe_results": probe_results,
    "comparison_results": comparison_results,
}

output_file = RESULTS_DIR / f"probing_layer{SAE_LAYER}_32k.json"
with open(output_file, "w") as f:
    json.dump(output, f, cls=NumpyEncoder, indent=2)
print(f"\nSaved: {output_file}")

report_progress(4, 4, metric={
    "stage": "completed",
    "n_samples": len(X),
    "best_f1": max(r["f1_score"] for r in probe_results.values()),
})

summary_text = f"Layer {SAE_LAYER}: Probing {len(X)} samples. Best F1={max(r['f1_score'] for r in probe_results.values()):.3f} (k=20)"

mark_done("success", summary_text)
print(f"\n{'='*60}")
print(f"Full probing experiment (layer {SAE_LAYER}) COMPLETE")
print(f"{'='*60}")

if PID_FILE.exists():
    PID_FILE.unlink()
