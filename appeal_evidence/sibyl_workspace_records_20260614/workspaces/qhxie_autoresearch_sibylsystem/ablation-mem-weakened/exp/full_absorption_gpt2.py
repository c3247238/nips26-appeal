#!/usr/bin/env python3
"""
Full Experiment: Feature Absorption Detection on GPT-2 Small (Multi-Layer)
Tasks: full_absorption_gpt2_l0, full_absorption_gpt2_l4, full_absorption_gpt2_l8, full_absorption_gpt2_l10

Detect absorption on all 26 first-letter features (A-Z) across GPT-2 Small layers 0, 4, 8, 10.
SAE: gpt2-small-res-jb (24K latents)

This script runs all 4 layer configs sequentially and saves combined results.
"""

import json
import os
import sys
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
SAMPLES_PER_FEATURE = 100

VOCAB = {
    "A": ["apple", "arrow", "anchor", "amber", "angel", "arena", "artist", "atom", "avenue", "azure",
          "able", "about", "above", "abroad", "absent", "absolute", "academy", "accept", "accident", "account"],
    "B": ["book", "bird", "blue", "bridge", "bread", "butter", "basket", "beauty", "battle", "beach",
          "baby", "back", "balance", "ball", "band", "bank", "bar", "base", "bath", "bear"],
    "C": ["cat", "car", "city", "cloud", "castle", "candle", "camera", "coffee", "circle", "crystal",
          "cake", "call", "calm", "camp", "can", "cap", "car", "card", "care", "case"],
    "D": ["dragon", "diamond", "dream", "dawn", "dance", "desert", "doctor", "dolphin", "dagger", "dust",
          "daily", "damage", "danger", "dark", "daughter", "death", "decide", "deep", "defense", "degree"],
    "E": ["eagle", "earth", "echo", "edge", "effect", "effort", "egg", "eight", "either", "elbow",
          "elder", "elect", "elegant", "element", "elephant", "elite", "emotion", "empire", "empty", "energy"],
    "F": ["fire", "forest", "flower", "fish", "feather", "frost", "flame", "fountain", "fortune", "freedom",
          "face", "fact", "factor", "fail", "fair", "faith", "fall", "false", "fame", "family"],
    "G": ["gold", "garden", "ghost", "glass", "grain", "grass", "green", "ground", "group", "guide",
          "gain", "game", "gap", "garage", "garden", "gas", "gate", "gather", "gear", "general"],
    "H": ["heart", "hill", "house", "horse", "harbor", "hammer", "honey", "horizon", "harvest", "haven",
          "habit", "hair", "half", "hall", "hand", "handle", "hang", "happen", "happy", "hard"],
    "I": ["ice", "iron", "island", "image", "impact", "idea", "ideal", "identity", "ignore", "ill",
          "image", "imagine", "immediate", "impact", "import", "impress", "improve", "inch", "incident", "income"],
    "J": ["jewel", "jungle", "justice", "journey", "journal", "judge", "juice", "jump", "jury", "jar",
          "job", "join", "joint", "joke", "joy", "judge", "jump", "jungle", "junior", "jury"],
    "K": ["king", "key", "knight", "knowledge", "kitchen", "kind", "kiss", "kit", "knee", "knife",
          "keen", "keep", "kettle", "key", "kick", "kid", "kill", "kind", "king", "kiss"],
    "L": ["light", "lake", "leaf", "lion", "ladder", "lamp", "land", "language", "laugh", "law",
          "lab", "label", "labor", "lack", "ladder", "lady", "lake", "lamp", "land", "lane"],
    "M": ["moon", "mountain", "mirror", "magic", "music", "market", "master", "meadow", "mystery", "metal",
          "machine", "magazine", "maintain", "major", "manage", "manner", "march", "market", "marriage", "material"],
    "N": ["night", "north", "nest", "nature", "needle", "nerve", "net", "network", "news", "night",
          "nail", "name", "narrow", "nation", "native", "nature", "near", "nearly", "neat", "necessary"],
    "O": ["ocean", "orange", "orbit", "order", "organ", "origin", "object", "offer", "office", "often",
          "oak", "obey", "object", "observe", "obtain", "obvious", "occasion", "occupy", "occur", "ocean"],
    "P": ["pear", "palace", "paper", "peace", "pearl", "people", "picture", "pilot", "pioneer", "planet",
          "pace", "pack", "package", "page", "pain", "paint", "pair", "palace", "pale", "panel"],
    "Q": ["queen", "quest", "quiet", "quick", "quality", "quarter", "question", "queue", "quote", "quiz",
          "qualify", "quality", "quarter", "queen", "quest", "question", "quick", "quiet", "quit", "quite"],
    "R": ["river", "rose", "rain", "rock", "road", "ring", "room", "root", "rope", "rule",
          "race", "radio", "rail", "rain", "raise", "range", "rank", "rapid", "rare", "rate"],
    "S": ["silver", "sun", "star", "stone", "sound", "shadow", "stream", "spirit", "sphere", "sword",
          "school", "science", "season", "second", "secret", "section", "secure", "select", "senior", "service"],
    "T": ["tiger", "tree", "tower", "truth", "temple", "thunder", "treasure", "tribe", "tunnel", "tide",
          "table", "taste", "teach", "telephone", "television", "temperature", "tennis", "theater", "theory", "ticket"],
    "U": ["unity", "universe", "urban", "use", "usual", "under", "unit", "upper", "urgent", "user",
          "ugly", "ultimate", "umbrella", "unable", "uncle", "under", "understand", "uniform", "union", "unique"],
    "V": ["valley", "violet", "voice", "vision", "vessel", "village", "virtue", "victory", "video", "view",
          "vacant", "vacation", "vague", "valid", "valley", "value", "van", "variable", "variety", "various"],
    "W": ["water", "wind", "wolf", "wood", "world", "wonder", "winter", "wisdom", "wealth", "wheel",
          "walk", "wall", "want", "war", "warm", "warn", "wash", "waste", "watch", "water"],
    "X": ["xenon", "xerox", "xray", "xylophone", "xenial", "xeric", "xenolith", "xenograft", "xenophobia", "xenogenesis",
          "xray", "xenon", "xerox", "xylophone", "xenial", "xeric", "xenolith", "xenograft", "xenophobia", "xenogenesis"],
    "Y": ["yellow", "young", "year", "yard", "youth", "yarn", "yacht", "yawn", "yeast", "yield",
          "yard", "yarn", "yawn", "year", "yeast", "yellow", "yes", "yesterday", "yet", "yield"],
    "Z": ["zebra", "zone", "zero", "zinc", "zoo", "zoom", "zenith", "zest", "zombie", "zodiac",
          "zero", "zone", "zoo", "zoom", "zebra", "zenith", "zest", "zombie", "zodiac", "zinc"],
}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_pid(task_id, results_dir):
    pid_file = Path(results_dir) / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def find_feature_latents(model, sae, tokenizer, feature_letter, vocab_words, n_samples=100, device="cuda"):
    """Find SAE latents that are MOST SELECTIVE for words starting with the given letter."""
    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")

    target_prompts = [f"The word '{w}'" for w in vocab_words[:n_samples//2]]

    other_words = []
    for letter, words in VOCAB.items():
        if letter != feature_letter:
            other_words.extend(words[:10])
    np.random.seed(SEED)
    np.random.shuffle(other_words)
    other_prompts = [f"The word '{w}'" for w in other_words[:n_samples//2]]

    target_activations = []
    for prompt in target_prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_act = sae_acts.max(dim=0).values.cpu().numpy()
            target_activations.append(max_act)

    other_activations = []
    for prompt in other_prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_act = sae_acts.max(dim=0).values.cpu().numpy()
            other_activations.append(max_act)

    target_mean = np.mean(target_activations, axis=0)
    other_mean = np.mean(other_activations, axis=0)
    selectivity = target_mean - other_mean

    top_k = min(100, len(selectivity))
    top_indices = np.argsort(selectivity)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if selectivity[idx] > 0.001:
            results.append((int(idx), float(selectivity[idx]), float(target_mean[idx]), float(other_mean[idx])))

    return results


def detect_absorption(model, sae, tokenizer, feature_letter, vocab_words, top_latents, n_samples=100, device="cuda"):
    """Detect absorption: measure how much the parent's signal is 'spread' across children."""
    if len(top_latents) < 2:
        return {"absorption_rate": 0.0, "n_latents": len(top_latents), "method": "differential_correlation"}

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")
    parent_latent = top_latents[0][0]
    child_latents = [l[0] for l in top_latents[1:51]]

    prompts = [f"The word '{w}' means something" for w in vocab_words * (n_samples // len(vocab_words) + 1)]
    prompts = prompts[:n_samples]

    parent_acts_list = []
    child_acts_list = {c: [] for c in child_latents}

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_acts = sae_acts.max(dim=0).values.cpu().numpy()

            parent_acts_list.append(max_acts[parent_latent])
            for c in child_latents:
                child_acts_list[c].append(max_acts[c])

    parent_acts = np.array(parent_acts_list)

    absorbing_children = 0
    child_correlations = []
    correlation_values = []

    for c in child_latents:
        child_acts = np.array(child_acts_list[c])
        active_mask = parent_acts > np.percentile(parent_acts, 50)
        if active_mask.sum() >= 3:
            r, p = stats.pearsonr(parent_acts[active_mask], child_acts[active_mask])
            child_correlations.append({"latent": int(c), "r": float(r), "p": float(p)})
            correlation_values.append(max(0, r))
            if r > 0.3 and p < 0.05:
                absorbing_children += 1
        else:
            child_correlations.append({"latent": int(c), "r": 0.0, "p": 1.0})
            correlation_values.append(0.0)

    absorption_rate = np.mean(correlation_values) if correlation_values else 0.0

    return {
        "absorption_rate": float(absorption_rate),
        "parent_latent": int(parent_latent),
        "n_child_latents": len(child_latents),
        "absorbing_children": absorbing_children,
        "method": "differential_correlation",
        "top_latents": [(int(l[0]), float(l[1]), float(l[2]), float(l[3])) for l in top_latents[:10]],
        "child_correlations": child_correlations,
    }


def run_absorption_for_layer(layer, model, tokenizer, device="cuda"):
    """Run absorption detection for a single layer."""
    from sae_lens import SAE

    task_id = f"full_absorption_gpt2_l{layer}"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=1, step=0, total_steps=26)

    print(f"\n{'='*60}")
    print(f"Layer {layer}: Loading SAE...")

    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device,
    )

    hook_name = sae.cfg.metadata.get("hook_name", f"blocks.{layer}.hook_resid_pre")
    print(f"  SAE: {sae.cfg.d_sae} latents, hook={hook_name}")

    # Detect absorption for all 26 features
    absorption_rates = {}
    for i, feat in enumerate(FEATURES):
        top_latents = find_feature_latents(
            model, sae, tokenizer, feat, VOCAB[feat],
            n_samples=SAMPLES_PER_FEATURE, device=device
        )

        absorption = detect_absorption(
            model, sae, tokenizer, feat, VOCAB[feat], top_latents,
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        absorption_rates[feat] = absorption

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/26] '{feat}': rate={absorption['absorption_rate']:.3f}, parent={absorption.get('parent_latent', 'N/A')}")

        report_progress(task_id, results_dir, epoch=1, total_epochs=1, step=i+1, total_steps=26,
                        metric={"feature": feat, "absorption_rate": absorption['absorption_rate']})

    # Save per-layer results
    output = {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": sae_id,
        "layer": layer,
        "dict_size": 24576,
        "hook_name": hook_name,
        "features": FEATURES,
        "absorption_rates": absorption_rates,
        "summary": {
            "n_features": len(FEATURES),
            "n_high_absorption": sum(1 for a in absorption_rates.values() if a["absorption_rate"] > 0.5),
            "n_medium_absorption": sum(1 for a in absorption_rates.values() if 0.1 <= a["absorption_rate"] <= 0.5),
            "n_low_absorption": sum(1 for a in absorption_rates.values() if a["absorption_rate"] < 0.1),
            "mean_absorption": float(np.mean([a["absorption_rate"] for a in absorption_rates.values()])),
            "std_absorption": float(np.std([a["absorption_rate"] for a in absorption_rates.values()])),
            "max_absorption": float(max(a["absorption_rate"] for a in absorption_rates.values())),
        }
    }

    with open(results_dir / f"{task_id}_absorption_rates.json", "w") as f:
        json.dump(output, f, indent=2)

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Layer {layer}: {output['summary']['n_high_absorption']} high, {output['summary']['n_medium_absorption']} medium, {output['summary']['n_low_absorption']} low, mean={output['summary']['mean_absorption']:.3f}")

    print(f"  Layer {layer} complete: mean={output['summary']['mean_absorption']:.3f}, max={output['summary']['max_absorption']:.3f}")

    return output


def run_full_experiment(layers=None):
    task_id = "full_absorption_gpt2_all_layers"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)

    if layers is None:
        layers = [0, 4, 8, 10]

    print("=" * 70)
    print("Full Experiment: Absorption Detection on GPT-2 Small (Multi-Layer)")
    print(f"Layers: {layers}")
    print(f"Device: cuda:0")
    print("=" * 70)

    set_seed(SEED)

    print("\n[Loading] Loading GPT-2 Small...")
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2",
        device="cuda:0",
        dtype=torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model: GPT-2 Small ({model.cfg.n_params/1e6:.0f}M params)")

    # Run for each layer
    all_results = {}
    for layer in layers:
        layer_results = run_absorption_for_layer(layer, model, tokenizer, device="cuda:0")
        all_results[f"layer_{layer}"] = layer_results

    # Combined summary
    combined = {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "layers": layers,
        "dict_size": 24576,
        "layer_results": all_results,
        "cross_layer_summary": {
            "mean_by_layer": {f"layer_{l}": all_results[f"layer_{l}"]["summary"]["mean_absorption"] for l in layers},
            "max_by_layer": {f"layer_{l}": all_results[f"layer_{l}"]["summary"]["max_absorption"] for l in layers},
            "cv_mean": float(np.std([all_results[f"layer_{l}"]["summary"]["mean_absorption"] for l in layers]) /
                           np.mean([all_results[f"layer_{l}"]["summary"]["mean_absorption"] for l in layers])),
        }
    }

    with open(results_dir / "full_absorption_gpt2_all_layers_combined.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'=' * 70}")
    print("All Layers Complete")
    for l in layers:
        s = all_results[f"layer_{l}"]["summary"]
        print(f"  Layer {l}: mean={s['mean_absorption']:.3f}, max={s['max_absorption']:.3f}, high={s['n_high_absorption']}, med={s['n_medium_absorption']}")
    print(f"{'=' * 70}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"All {len(layers)} layers complete")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="0,4,8,10", help="Comma-separated layer indices")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]

    try:
        results = run_full_experiment(layers=layers)
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent / "results" / "full"
        mark_task_done("full_absorption_gpt2_all_layers", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
