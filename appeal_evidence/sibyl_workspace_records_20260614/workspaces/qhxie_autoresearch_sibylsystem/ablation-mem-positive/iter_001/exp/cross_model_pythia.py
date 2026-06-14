#!/usr/bin/env python3
"""
Cross-Model Validation: Pythia-70m-deduped
Task: I3_cross_model_pythia

Run the 3-condition absorption detection framework on Pythia-70m-deduped
to check if the middle-layer hotspot pattern (observed as Layer 6 in GPT-2 Small)
replicates in a different model architecture.

Framework (3 conditions):
1. Frequency condition: Parent feature fires on many samples
2. Co-occurrence condition: Child features fire together with parent
3. Decoder cosine condition: Parent-child decoder vectors have high cosine similarity

Model: EleutherAI/pythia-70m-deduped (19M params, 6 layers)
SAE: pythia-70m-deduped-res-sm (32K latents, layers 0-5)
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
N_PROMPTS_COOC = 500  # For co-occurrence analysis

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
    hook_name = sae.cfg.metadata.get("hook_name", "blocks.4.hook_resid_post")

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


def compute_frequency_condition(model, sae, tokenizer, feature_letter, vocab_words, parent_latent,
                                n_prompts=500, device="cuda"):
    """Condition 1: Parent latent fires frequently across prompts."""
    hook_name = sae.cfg.metadata.get("hook_name", "blocks.4.hook_resid_post")

    prompts = [f"The word '{w}' means something" for w in vocab_words * (n_prompts // len(vocab_words) + 1)]
    prompts = prompts[:n_prompts]

    parent_acts = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_act = sae_acts.max(dim=0).values[parent_latent].item()
            parent_acts.append(max_act)

    parent_acts = np.array(parent_acts)
    firing_rate = np.mean(parent_acts > 0.1)
    mean_activation = np.mean(parent_acts)

    return {
        "firing_rate": float(firing_rate),
        "mean_activation": float(mean_activation),
        "n_prompts": n_prompts,
    }


def compute_cooccurrence_condition(model, sae, tokenizer, feature_letter, vocab_words, parent_latent,
                                   child_latents, n_prompts=500, device="cuda"):
    """Condition 2: Child latents co-fire with parent."""
    hook_name = sae.cfg.metadata.get("hook_name", "blocks.4.hook_resid_post")

    prompts = [f"The word '{w}' means something" for w in vocab_words * (n_prompts // len(vocab_words) + 1)]
    prompts = prompts[:n_prompts]

    parent_acts = []
    child_acts = {c: [] for c in child_latents}

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_acts = sae_acts.max(dim=0).values.cpu().numpy()

            parent_acts.append(max_acts[parent_latent])
            for c in child_latents:
                child_acts[c].append(max_acts[c])

    parent_acts = np.array(parent_acts)

    cooccurrence_scores = []
    for c in child_latents:
        child_arr = np.array(child_acts[c])
        # P(child active | parent active)
        parent_active = parent_acts > np.percentile(parent_acts, 50)
        if parent_active.sum() >= 3:
            p_child_given_parent = np.mean(child_arr[parent_active] > 0.1)
            p_child_given_not_parent = np.mean(child_arr[~parent_active] > 0.1)
            cooccurrence_scores.append({
                "latent": int(c),
                "p_child_given_parent": float(p_child_given_parent),
                "p_child_given_not_parent": float(p_child_given_not_parent),
                "ratio": float(p_child_given_parent / (p_child_given_not_parent + 1e-8)),
            })

    mean_cooccurrence = np.mean([s["p_child_given_parent"] for s in cooccurrence_scores]) if cooccurrence_scores else 0.0
    mean_ratio = np.mean([s["ratio"] for s in cooccurrence_scores]) if cooccurrence_scores else 0.0

    return {
        "mean_cooccurrence": float(mean_cooccurrence),
        "mean_ratio": float(mean_ratio),
        "n_children": len(child_latents),
        "child_scores": cooccurrence_scores,
    }


def compute_decoder_cosine_condition(sae, parent_latent, child_latents):
    """Condition 3: Parent-child decoder vectors have high cosine similarity."""
    W_dec = sae.W_dec.detach().cpu().numpy()
    parent_vec = W_dec[parent_latent]
    parent_norm = parent_vec / (np.linalg.norm(parent_vec) + 1e-8)

    cosine_sims = []
    for c in child_latents:
        child_vec = W_dec[c]
        child_norm = child_vec / (np.linalg.norm(child_vec) + 1e-8)
        sim = np.dot(parent_norm, child_norm)
        cosine_sims.append({"latent": int(c), "cosine_sim": float(sim)})

    mean_cosine = np.mean([s["cosine_sim"] for s in cosine_sims]) if cosine_sims else 0.0
    n_high_sim = sum(1 for s in cosine_sims if s["cosine_sim"] > 0.3)

    return {
        "mean_cosine_sim": float(mean_cosine),
        "n_high_similarity": int(n_high_sim),
        "n_children": len(child_latents),
        "child_sims": cosine_sims,
    }


def detect_absorption_three_condition(model, sae, tokenizer, feature_letter, vocab_words, top_latents,
                                      n_samples=100, device="cuda"):
    """Detect absorption using the 3-condition framework."""
    if len(top_latents) < 2:
        return {
            "absorption_rate": 0.0,
            "n_latents": len(top_latents),
            "method": "three_condition",
            "conditions_met": 0,
        }

    parent_latent = top_latents[0][0]
    child_latents = [l[0] for l in top_latents[1:51]]

    # Condition 1: Frequency
    freq = compute_frequency_condition(model, sae, tokenizer, feature_letter, vocab_words,
                                       parent_latent, n_prompts=N_PROMPTS_COOC, device=device)

    # Condition 2: Co-occurrence
    cooc = compute_cooccurrence_condition(model, sae, tokenizer, feature_letter, vocab_words,
                                          parent_latent, child_latents, n_prompts=N_PROMPTS_COOC, device=device)

    # Condition 3: Decoder cosine
    cosine = compute_decoder_cosine_condition(sae, parent_latent, child_latents)

    # Count conditions met
    conditions_met = 0
    if freq["firing_rate"] > 0.3:
        conditions_met += 1
    if cooc["mean_cooccurrence"] > 0.3:
        conditions_met += 1
    if cosine["mean_cosine_sim"] > 0.1:
        conditions_met += 1

    # Absorption score: weighted by conditions met
    base_score = (cooc["mean_cooccurrence"] + max(0, cosine["mean_cosine_sim"])) / 2
    absorption_rate = base_score * (conditions_met / 3)

    return {
        "absorption_rate": float(absorption_rate),
        "parent_latent": int(parent_latent),
        "n_child_latents": len(child_latents),
        "conditions_met": conditions_met,
        "method": "three_condition",
        "frequency": freq,
        "cooccurrence": cooc,
        "decoder_cosine": cosine,
        "top_latents": [(int(l[0]), float(l[1]), float(l[2]), float(l[3])) for l in top_latents[:10]],
    }


def run_absorption_for_layer(layer, model, tokenizer, device="cuda"):
    """Run 3-condition absorption detection for a single layer."""
    from sae_lens import SAE

    task_id = f"cross_model_pythia_l{layer}"
    results_dir = Path(__file__).parent / "results" / "full" / "cross_model_pythia"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=1, step=0, total_steps=26)

    print(f"\n{'='*60}")
    print(f"Pythia-70m Layer {layer}: Loading SAE...")

    sae_id = f"blocks.{layer}.hook_resid_post"
    sae = SAE.from_pretrained(
        release="pythia-70m-deduped-res-sm",
        sae_id=sae_id,
        device=device,
    )

    hook_name = sae.cfg.metadata.get("hook_name", f"blocks.{layer}.hook_resid_post")
    print(f"  SAE: {sae.cfg.d_sae} latents, hook={hook_name}")

    # Detect absorption for all 26 features
    absorption_rates = {}
    for i, feat in enumerate(FEATURES):
        top_latents = find_feature_latents(
            model, sae, tokenizer, feat, VOCAB[feat],
            n_samples=SAMPLES_PER_FEATURE, device=device
        )

        absorption = detect_absorption_three_condition(
            model, sae, tokenizer, feat, VOCAB[feat], top_latents,
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        absorption_rates[feat] = absorption

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/26] '{feat}': rate={absorption['absorption_rate']:.3f}, "
                  f"conditions={absorption['conditions_met']}/3, "
                  f"parent={absorption.get('parent_latent', 'N/A')}")

        report_progress(task_id, results_dir, epoch=1, total_epochs=1, step=i+1, total_steps=26,
                        metric={"feature": feat, "absorption_rate": absorption['absorption_rate'],
                                "conditions_met": absorption['conditions_met']})

    # Save per-layer results
    output = {
        "model": "pythia-70m-deduped",
        "sae_release": "pythia-70m-deduped-res-sm",
        "sae_id": sae_id,
        "layer": layer,
        "dict_size": 32768,
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
            "mean_conditions_met": float(np.mean([a["conditions_met"] for a in absorption_rates.values()])),
        }
    }

    with open(results_dir / f"{task_id}_absorption_rates.json", "w") as f:
        json.dump(output, f, indent=2)

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Layer {layer}: {output['summary']['n_high_absorption']} high, "
                          f"{output['summary']['n_medium_absorption']} medium, "
                          f"{output['summary']['n_low_absorption']} low, "
                          f"mean={output['summary']['mean_absorption']:.3f}, "
                          f"avg_conditions={output['summary']['mean_conditions_met']:.2f}")

    print(f"  Layer {layer} complete: mean={output['summary']['mean_absorption']:.3f}, "
          f"max={output['summary']['max_absorption']:.3f}, "
          f"avg_conditions={output['summary']['mean_conditions_met']:.2f}/3")

    return output


def run_cross_model_validation(layers=None):
    task_id = "cross_model_pythia"
    results_dir = Path(__file__).parent / "results" / "full" / "cross_model_pythia"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)

    if layers is None:
        layers = [0, 1, 2, 3, 4, 5]  # All Pythia-70m layers

    print("=" * 70)
    print("Cross-Model Validation: Pythia-70m-deduped")
    print(f"Layers: {layers}")
    print(f"Device: cuda")
    print("=" * 70)

    set_seed(SEED)

    print("\n[Loading] Loading Pythia-70m-deduped...")
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "pythia-70m-deduped",
        device="cuda",
        dtype=torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model: Pythia-70m-deduped ({model.cfg.n_params/1e6:.0f}M params)")
    print(f"  Layers: {model.cfg.n_layers}")

    # Run for each layer
    all_results = {}
    for layer in layers:
        layer_results = run_absorption_for_layer(layer, model, tokenizer, device="cuda")
        all_results[f"layer_{layer}"] = layer_results

    # Combined summary with cross-layer analysis
    mean_by_layer = {f"layer_{l}": all_results[f"layer_{l}"]["summary"]["mean_absorption"] for l in layers}
    max_by_layer = {f"layer_{l}": all_results[f"layer_{l}"]["summary"]["max_absorption"] for l in layers}
    conditions_by_layer = {f"layer_{l}": all_results[f"layer_{l}"]["summary"]["mean_conditions_met"] for l in layers}

    # Find hotspot layer (highest mean absorption)
    hotspot_layer = max(mean_by_layer, key=mean_by_layer.get)
    hotspot_idx = int(hotspot_layer.split("_")[1])

    # Check if middle layers show higher absorption (hotspot pattern)
    middle_layers = [l for l in layers if len(layers)//3 <= l < 2*len(layers)//3]
    middle_mean = np.mean([mean_by_layer[f"layer_{l}"] for l in middle_layers]) if middle_layers else 0
    early_mean = np.mean([mean_by_layer[f"layer_{l}"] for l in layers if l < len(layers)//3])
    late_mean = np.mean([mean_by_layer[f"layer_{l}"] for l in layers if l >= 2*len(layers)//3])

    combined = {
        "model": "pythia-70m-deduped",
        "sae_release": "pythia-70m-deduped-res-sm",
        "layers": layers,
        "dict_size": 32768,
        "layer_results": all_results,
        "cross_layer_summary": {
            "mean_by_layer": mean_by_layer,
            "max_by_layer": max_by_layer,
            "conditions_met_by_layer": conditions_by_layer,
            "hotspot_layer": hotspot_layer,
            "hotspot_layer_idx": hotspot_idx,
            "early_layers_mean": float(early_mean),
            "middle_layers_mean": float(middle_mean),
            "late_layers_mean": float(late_mean),
            "middle_vs_early_ratio": float(middle_mean / (early_mean + 1e-8)),
            "middle_vs_late_ratio": float(middle_mean / (late_mean + 1e-8)),
            "hotspot_replicated": bool(middle_mean > early_mean and middle_mean > late_mean),
        },
        "comparison_with_gpt2": {
            "note": "GPT-2 Small (124M, 12 layers) showed Layer 6 hotspot (~middle). "
                    "Pythia-70m (19M, 6 layers) hotspot pattern indicates generalizability.",
            "gpt2_hotspot_layer": 6,
            "gpt2_total_layers": 12,
            "pythia_hotspot_layer": hotspot_idx,
            "pythia_total_layers": 6,
            "gpt2_hotspot_relative": 0.5,  # 6/12
            "pythia_hotspot_relative": hotspot_idx / 6 if layers else 0,
        }
    }

    with open(results_dir / "cross_model_pythia_combined.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Cross-Model Validation Complete")
    print(f"{'=' * 70}")
    for l in layers:
        s = all_results[f"layer_{l}"]["summary"]
        print(f"  Layer {l}: mean={s['mean_absorption']:.3f}, max={s['max_absorption']:.3f}, "
              f"high={s['n_high_absorption']}, med={s['n_medium_absorption']}, "
              f"cond={s['mean_conditions_met']:.2f}")
    print(f"\n  Hotspot layer: {hotspot_layer}")
    print(f"  Middle layer mean: {middle_mean:.3f}")
    print(f"  Early layer mean: {early_mean:.3f}")
    print(f"  Late layer mean: {late_mean:.3f}")
    print(f"  Hotspot replicated: {combined['cross_layer_summary']['hotspot_replicated']}")
    print(f"{'=' * 70}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Pythia cross-validation: hotspot={hotspot_layer}, "
                          f"replicated={combined['cross_layer_summary']['hotspot_replicated']}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5", help="Comma-separated layer indices")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]

    try:
        results = run_cross_model_validation(layers=layers)
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent / "results" / "full" / "cross_model_pythia"
        mark_task_done("cross_model_pythia", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
