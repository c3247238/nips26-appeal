#!/usr/bin/env python3
"""
Ablation: Random Feature Baseline Analysis
Test random SAE features to establish baseline for reconstruction fidelity.
Model: GPT-2 Small, Layer 8, 32K dictionary
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

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/current/exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "ablation_random_baseline"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

PID_FILE.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))

def mark_done(status="success", summary=""):
    if PID_FILE.exists(): PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try: final_progress = json.loads(PROGRESS_FILE.read_text())
        except: pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.bool_)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

print(f"[{TASK_ID}] Starting random feature baseline analysis")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 8

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 3, metric={"stage": "loading_model"})

model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
print(f"  Model loaded: {model.cfg.n_layers} layers")

print(f"\nLoading SAE for layer {SAE_LAYER}")
report_progress(1, 3, metric={"stage": "loading_sae"})

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

apply_b_dec_to_input = sae_cfg.get("apply_b_dec_to_input", True)
d_sae = sae_cfg["d_sae"]

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

# Test 5 sets of 26 random features each
n_sets = 5
n_features_per_set = 26

print(f"\nTesting {n_sets} sets of {n_features_per_set} random features each")
report_progress(2, 3, metric={"stage": "random_baseline", "n_sets": n_sets})

random_results = []

for set_idx in tqdm(range(n_sets), desc="Random sets"):
    # Set different seed for each set
    np.random.seed(SEED + set_idx)
    random_feature_ids = np.random.choice(d_sae, size=n_features_per_set, replace=False)

    # Assign one random feature to each letter
    letter_to_feature = {letter: int(random_feature_ids[i]) for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

    set_mses = []
    set_cos_sims = []

    with torch.no_grad():
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            words = word_lists.get(letter, [])[:5]
            feature_id = letter_to_feature[letter]

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

                set_mses.append(mse)
                set_cos_sims.append(cos_sim)

    random_results.append({
        "set_idx": set_idx,
        "mean_mse": float(np.mean(set_mses)),
        "std_mse": float(np.std(set_mses)),
        "mean_cosine": float(np.mean(set_cos_sims)),
        "std_cosine": float(np.std(set_cos_sims)),
    })

# Summary
print(f"\n{'='*60}")
print(f"RANDOM FEATURE BASELINE RESULTS")
print(f"{'='*60}")

all_mses = [r["mean_mse"] for r in random_results]
all_cos = [r["mean_cosine"] for r in random_results]

print(f"\nRandom feature sets:")
for r in random_results:
    print(f"  Set {r['set_idx']}: MSE={r['mean_mse']:.2f} ± {r['std_mse']:.2f}, Cos={r['mean_cosine']:.4f} ± {r['std_cosine']:.4f}")

print(f"\nAcross {n_sets} random sets:")
print(f"  MSE: {np.mean(all_mses):.2f} ± {np.std(all_mses):.2f}")
print(f"  Cosine: {np.mean(all_cos):.4f} ± {np.std(all_cos):.4f}")

# Save results
output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "layer": SAE_LAYER,
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "timestamp": datetime.now().isoformat(),
    "method": "random_feature_baseline",
    "n_sets": n_sets,
    "n_features_per_set": n_features_per_set,
    "summary": {
        "mean_mse": float(np.mean(all_mses)),
        "std_mse": float(np.std(all_mses)),
        "mean_cosine": float(np.mean(all_cos)),
        "std_cosine": float(np.std(all_cos)),
    },
    "random_results": random_results,
}

output_file = RESULTS_DIR / "ablation_random_baseline.json"
with open(output_file, "w") as f:
    json.dump(output, f, cls=NumpyEncoder, indent=2)
print(f"\nSaved: {output_file}")

summary_text = f"Random baseline: {n_sets} sets, MSE={np.mean(all_mses):.2f} ± {np.std(all_mses):.2f}, Cos={np.mean(all_cos):.4f}"

mark_done("success", summary_text)
print(f"\n{'='*60}")
print(f"Random feature baseline COMPLETE")
print(f"{'='*60}")

if PID_FILE.exists(): PID_FILE.unlink()
