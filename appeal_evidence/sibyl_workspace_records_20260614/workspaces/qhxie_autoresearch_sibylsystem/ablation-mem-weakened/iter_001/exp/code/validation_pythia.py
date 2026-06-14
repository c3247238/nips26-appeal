#!/usr/bin/env python3
"""
Validation: Pythia-160M Cross-Model Check
Run absorption detection on Pythia-160M to validate cross-model generalization.
Uses SAEBench SAEs (adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108).
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

TASK_ID = "validation_pythia"
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

print(f"[{TASK_ID}] Starting Pythia-160M validation")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")

from transformer_lens import HookedTransformer

MODEL_NAME = "EleutherAI/pythia-160m-deduped"
SAE_LAYER = 8

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 3, metric={"stage": "loading_model"})

try:
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
    print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")
except Exception as e:
    print(f"ERROR loading model: {e}")
    mark_done("failed", f"Model load error: {e}")
    sys.exit(1)

print(f"\nLoading SAE for layer {SAE_LAYER}")
report_progress(1, 3, metric={"stage": "loading_sae"})

sae_cache_dir = Path.home() / ".cache/huggingface/hub/models--adamkarvonen--saebench_pythia-160m-deduped_width-2pow14_date-0108/snapshots"
sae_dirs = list(sae_cache_dir.glob("*"))
if not sae_dirs:
    print(f"ERROR: No cached SAE found")
    mark_done("failed", "No cached SAE")
    sys.exit(1)

# Find the SAE weights for layer 8
sae_dir = sae_dirs[0]
sae_weights_file = sae_dir / "sae_weights.safetensors"
sae_cfg_file = sae_dir / "cfg.json"

if not sae_weights_file.exists():
    # Try alternate structure
    sae_weights_file = sae_dir / "sae_weights.safetensors"
    sae_cfg_file = sae_dir / "cfg.json"

if not sae_weights_file.exists():
    print(f"ERROR: SAE weights not found at {sae_weights_file}")
    mark_done("failed", "SAE weights not found")
    sys.exit(1)

with open(sae_cfg_file) as f:
    sae_cfg = json.load(f)

sae_weights = load_file(sae_weights_file, device=DEVICE)
W_enc = sae_weights["W_enc"]
W_dec = sae_weights["W_dec"]
b_enc = sae_weights["b_enc"]
b_dec = sae_weights["b_dec"]

d_model = sae_cfg["d_in"]
d_sae = sae_cfg["d_sae"]
apply_b_dec_to_input = sae_cfg.get("apply_b_dec_to_input", True)

print(f"  d_model: {d_model}, d_sae: {d_sae}")

def encode_sae(acts):
    if apply_b_dec_to_input:
        acts = acts - b_dec
    pre_acts = torch.matmul(acts, W_enc) + b_enc
    sae_acts = torch.relu(pre_acts)
    return sae_acts

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

# Find features by decoder cosine similarity
print(f"\nFinding first-letter features...")
report_progress(2, 3, metric={"stage": "finding_features"})

def get_token_directions(letter):
    words = word_lists.get(letter, [])[:10]
    directions = []
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name in cache:
                acts = cache[layer_name][0, 0, :]
                directions.append(acts)
    if not directions: return None
    return torch.stack(directions).mean(dim=0)

def find_features_for_letter(letter, top_k=10):
    direction = get_token_directions(letter)
    if direction is None: return None
    direction_norm = direction / (direction.norm() + 1e-8)
    dec_norms = W_dec.norm(dim=1, keepdim=True) + 1e-8
    W_dec_normalized = W_dec / dec_norms
    cos_sims = torch.matmul(W_dec_normalized, direction_norm)
    top_values, top_indices = torch.topk(cos_sims, top_k)
    return {
        "feature_ids": [int(x.item()) for x in top_indices],
        "cosine_similarities": [float(x.item()) for x in top_values],
    }

letter_features = {}
for letter in tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Finding features"):
    result = find_features_for_letter(letter, top_k=20)
    if result: letter_features[letter] = result

# Deduplicate
used_features = set()
letter_main_features = {}
for letter in sorted(letter_features.keys()):
    info = letter_features[letter]
    for fid, cos_sim in zip(info["feature_ids"], info["cosine_similarities"]):
        if fid not in used_features and cos_sim > 0.01:
            letter_main_features[letter] = {"feature_id": fid, "cosine_similarity": cos_sim}
            used_features.add(fid)
            break
    if letter not in letter_main_features:
        letter_main_features[letter] = {"feature_id": info["feature_ids"][0], "cosine_similarity": info["cosine_similarities"][0]}

print(f"  Found {len(letter_main_features)} features ({len(set(v['feature_id'] for v in letter_main_features.values()))} unique)")

# Compute absorption
print(f"\nComputing absorption metric...")
report_progress(3, 3, metric={"stage": "computing_absorption"})

def compute_absorption(feature_id, letter, n_words=10):
    words = word_lists.get(letter, [])[:n_words]
    parent_dec = W_dec[feature_id]
    parent_dec_norm = parent_dec / (parent_dec.norm() + 1e-8)
    dec_norms = W_dec.norm(dim=1) + 1e-8
    W_dec_norm = W_dec / dec_norms.unsqueeze(1)
    child_cos_sims = torch.matmul(W_dec_norm, parent_dec_norm)
    child_cos_sims[feature_id] = -1.0
    top_child_vals, top_child_ids = torch.topk(child_cos_sims, k=10)
    child_features = [(int(cid.item()), float(cval.item())) for cid, cval in zip(top_child_ids, top_child_vals) if cval > 0.1]

    results = []
    with torch.no_grad():
        for word in words:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            layer_name = f"blocks.{SAE_LAYER}.hook_resid_post"
            if layer_name not in cache: continue
            acts = cache[layer_name]
            sae_acts = encode_sae(acts)
            first_token_acts = sae_acts[0, 0, :].cpu().numpy()
            main_fired = first_token_acts[feature_id] > 0.1
            main_activation = float(first_token_acts[feature_id])
            child_fired_count = 0
            for cid, ccos in child_features:
                if float(first_token_acts[cid]) > 0.1: child_fired_count += 1
            is_absorbed = (not main_fired) and (child_fired_count > 0)
            results.append({"word": word, "main_fired": main_fired, "main_activation": main_activation, "child_fired_count": child_fired_count, "is_absorbed": is_absorbed})

    if not results: return {"absorption_rate": 0.0, "error": "no results"}
    n_total = len(results)
    n_absorbed = sum(1 for r in results if r["is_absorbed"])
    n_main_fired = sum(1 for r in results if r["main_fired"])
    n_main_not_fired = n_total - n_main_fired
    absorption_rate = n_absorbed / max(1, n_total)
    conditional_rate = n_absorbed / max(1, n_main_not_fired) if n_main_not_fired > 0 else 0.0
    return {"absorption_rate": float(absorption_rate), "conditional_rate": float(conditional_rate), "n_total": n_total, "n_absorbed": n_absorbed, "n_main_fired": n_main_fired, "n_main_not_fired": n_main_not_fired, "child_features": child_features, "per_word": results}

absorption_results = {}
classification = {"HIGH": [], "MEDIUM": [], "LOW": []}

for letter in tqdm(sorted(letter_main_features.keys()), desc="Computing absorption"):
    info = letter_main_features[letter]
    result = compute_absorption(info["feature_id"], letter, n_words=10)
    result["letter"] = letter
    result["feature_id"] = info["feature_id"]
    result["cosine_similarity"] = info["cosine_similarity"]
    absorption_results[letter] = result
    rate = result["absorption_rate"]
    if rate > 0.5: classification["HIGH"].append(letter)
    elif rate > 0.1: classification["MEDIUM"].append(letter)
    else: classification["LOW"].append(letter)

absorption_rates = [r["absorption_rate"] for r in absorption_results.values()]
print(f"\n{'='*60}")
print(f"PYTHIA-160M ABSORPTION RESULTS - Layer {SAE_LAYER}")
print(f"{'='*60}")
print(f"Features analyzed: {len(absorption_results)}")
print(f"Unique features: {len(set(r['feature_id'] for r in absorption_results.values()))}")
print(f"Mean absorption rate: {np.mean(absorption_rates):.4f}")
print(f"Std: {np.std(absorption_rates):.4f}")
print(f"HIGH (>50%): {len(classification['HIGH'])} {classification['HIGH']}")
print(f"MEDIUM (10-50%): {len(classification['MEDIUM'])} {classification['MEDIUM']}")
print(f"LOW (<10%): {len(classification['LOW'])} {classification['LOW']}")

# Save results
output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "sae_source": "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
    "layer": SAE_LAYER,
    "dictionary_size": d_sae,
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
        "high_count": len(classification["HIGH"]),
        "medium_count": len(classification["MEDIUM"]),
        "low_count": len(classification["LOW"]),
    },
    "classification": classification,
    "results": absorption_results,
    "letter_features": {k: {"feature_id": v["feature_id"], "cosine_similarity": v["cosine_similarity"]} for k, v in letter_main_features.items()},
}

output_file = RESULTS_DIR / "validation_pythia_160m.json"
output_file.write_text(json.dumps(output, cls=NumpyEncoder, indent=2))
print(f"\nSaved: {output_file}")

summary_text = f"Pythia-160M layer {SAE_LAYER}: {len(absorption_results)} features, mean={np.mean(absorption_rates):.3f}, HIGH={len(classification['HIGH'])}, LOW={len(classification['LOW'])}"

mark_done("success", summary_text)
print(f"\n{'='*60}")
print(f"Pythia-160M validation COMPLETE")
print(f"{'='*60}")

if PID_FILE.exists(): PID_FILE.unlink()
