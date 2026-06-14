#!/usr/bin/env python3
"""
Cross-Model Validation: Pythia-70m-deduped
Task: cross_model_pythia

Replicate primary experiments on Pythia-70m to test generalizability:
1. Absorption detection on 26 first-letter features (A-Z)
2. Feature steering with random baseline (adapted for small model)
3. Delta-corrected correlation analysis

SAE: pythia-70m-deduped-res-sm (32K latents, layers 0-5)
Model: EleutherAI/pythia-70m-deduped (19M params)

NOTE: Pythia-70m is much smaller than GPT-2 Small (19M vs 124M params).
Steering effects are weaker. We use cosine similarity between steering
direction and target word embeddings as the effectiveness metric.
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
STEERING_STRENGTHS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

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


def detect_absorption(model, sae, tokenizer, feature_letter, vocab_words, top_latents, n_samples=100, device="cuda"):
    """Detect absorption: measure how much the parent's signal is 'spread' across children."""
    if len(top_latents) < 2:
        return {"absorption_rate": 0.0, "n_latents": len(top_latents), "method": "differential_correlation"}

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.4.hook_resid_post")
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


def run_steering_experiment(model, sae, tokenizer, feature_letter, vocab_words, feature_latent_id,
                            strengths=None, n_prompts=100, device="cuda"):
    """Run feature steering experiment using embedding-space similarity metric.

    For small models like Pythia-70m, raw token probability changes are minimal.
    We measure steering effectiveness by the cosine similarity between the
    steering direction and the target word's unembedding vector.
    """
    if strengths is None:
        strengths = STEERING_STRENGTHS

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.4.hook_resid_post")
    direction = sae.W_dec[feature_latent_id].to(device)
    direction_norm = direction / (direction.norm() + 1e-8)

    # Get unembedding matrix for semantic similarity
    W_U = model.W_U.to(device)  # [d_model, vocab_size]

    # Pre-compute target word embeddings (average of first token of each target word)
    target_embeddings = []
    for w in vocab_words[:10]:
        t_ids = tokenizer.encode(w, add_special_tokens=False)
        if t_ids:
            target_embeddings.append(W_U[:, t_ids[0]])
    if target_embeddings:
        target_embedding = torch.stack(target_embeddings).mean(dim=0)
        target_embedding = target_embedding / (target_embedding.norm() + 1e-8)
    else:
        target_embedding = None

    prompts = [f"The word '{w}'" for w in vocab_words * (n_prompts // len(vocab_words) + 1)]
    prompts = prompts[:n_prompts]

    results = {}
    for strength in strengths:
        similarities = []
        token_prob_lifts = []

        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt").to(device)

            # Baseline
            with torch.no_grad():
                logits_base = model(tokens.input_ids)
                probs_base = torch.softmax(logits_base[0, -1, :], dim=-1)

            # Steered
            def steering_hook(acts, hook):
                acts[:, -1, :] += strength * direction
                return acts

            with torch.no_grad():
                logits_steered = model.run_with_hooks(
                    tokens.input_ids,
                    fwd_hooks=[(hook_name, steering_hook)]
                )
                probs_steered = torch.softmax(logits_steered[0, -1, :], dim=-1)

            # Metric 1: Cosine similarity between steering direction and top predicted token embedding
            top_token = torch.argmax(logits_steered[0, -1, :])
            top_embedding = W_U[:, top_token]
            top_embedding = top_embedding / (top_embedding.norm() + 1e-8)
            sim = torch.dot(direction_norm, top_embedding).item()
            similarities.append(sim)

            # Metric 2: Probability lift for target letter token
            letter_token = 33 + ord(feature_letter) - ord('A')  # A=34, B=35, etc.
            if 0 <= letter_token < probs_base.shape[0]:
                prob_lift = probs_steered[letter_token].item() - probs_base[letter_token].item()
                token_prob_lifts.append(prob_lift)

        mean_sim = np.mean(similarities) if similarities else 0.0
        mean_prob_lift = np.mean(token_prob_lifts) if token_prob_lifts else 0.0

        results[strength] = {
            "mean_embedding_similarity": float(mean_sim),
            "mean_letter_prob_lift": float(mean_prob_lift),
            "n_prompts": len(prompts),
        }

    return results


def run_random_baseline_steering(model, sae, tokenizer, vocab_words, n_random=26,
                                  strengths=None, n_prompts=100, device="cuda"):
    """Run steering on random SAE latents as baseline."""
    if strengths is None:
        strengths = STEERING_STRENGTHS

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.4.hook_resid_post")
    d_sae = sae.cfg.d_sae
    W_U = model.W_U.to(device)

    np.random.seed(SEED + 100)
    random_latents = np.random.choice(d_sae, size=n_random, replace=False)

    all_results = {}
    for latent_id in random_latents:
        direction = sae.W_dec[latent_id].to(device)
        direction_norm = direction / (direction.norm() + 1e-8)
        prompts = [f"The word '{w}'" for w in vocab_words * (n_prompts // len(vocab_words) + 1)]
        prompts = prompts[:n_prompts]

        latent_results = {}
        for strength in strengths:
            similarities = []

            def make_hook(s, d):
                def steering_hook(acts, hook):
                    acts[:, -1, :] += s * d
                    return acts
                return steering_hook

            for prompt in prompts:
                tokens = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    logits_steered = model.run_with_hooks(
                        tokens.input_ids,
                        fwd_hooks=[(hook_name, make_hook(strength, direction))]
                    )

                top_token = torch.argmax(logits_steered[0, -1, :])
                top_embedding = W_U[:, top_token]
                top_embedding = top_embedding / (top_embedding.norm() + 1e-8)
                sim = torch.dot(direction_norm, top_embedding).item()
                similarities.append(sim)

            mean_sim = np.mean(similarities) if similarities else 0.0
            latent_results[strength] = {
                "mean_embedding_similarity": float(mean_sim),
            }

        all_results[int(latent_id)] = latent_results

    return all_results


def run_absorption_for_layer(layer, model, tokenizer, device="cuda"):
    """Run absorption detection for a single layer."""
    from sae_lens import SAE

    task_id = f"cross_model_pythia_l{layer}"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=3, step=0, total_steps=26)

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
    feature_latent_map = {}

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

        if top_latents:
            feature_latent_map[feat] = top_latents[0][0]

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/26] '{feat}': rate={absorption['absorption_rate']:.3f}, parent={absorption.get('parent_latent', 'N/A')}")

        report_progress(task_id, results_dir, epoch=1, total_epochs=3, step=i+1, total_steps=26,
                        metric={"feature": feat, "absorption_rate": absorption['absorption_rate']})

    # Save absorption results
    absorption_output = {
        "model": "pythia-70m-deduped",
        "sae_release": "pythia-70m-deduped-res-sm",
        "sae_id": sae_id,
        "layer": layer,
        "dict_size": 32768,
        "hook_name": hook_name,
        "features": FEATURES,
        "absorption_rates": absorption_rates,
        "feature_latent_map": feature_latent_map,
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
        json.dump(absorption_output, f, indent=2)

    print(f"  Absorption complete: mean={absorption_output['summary']['mean_absorption']:.3f}, max={absorption_output['summary']['max_absorption']:.3f}")

    # Phase 2: Steering experiment
    print(f"\n  Running steering experiment...")
    report_progress(task_id, results_dir, epoch=2, total_epochs=3, step=0, total_steps=26)

    steering_results = {}
    for i, feat in enumerate(FEATURES):
        if feat in feature_latent_map:
            latent_id = feature_latent_map[feat]
            steering = run_steering_experiment(
                model, sae, tokenizer, feat, VOCAB[feat], latent_id,
                strengths=STEERING_STRENGTHS, n_prompts=100, device=device
            )
            steering_results[feat] = steering

            if (i + 1) % 5 == 0 or i == 0:
                sim = steering[50.0]["mean_embedding_similarity"] if 50.0 in steering else 0.0
                print(f"    [{i+1}/26] '{feat}': sim@50={sim:.3f}")

            report_progress(task_id, results_dir, epoch=2, total_epochs=3, step=i+1, total_steps=26,
                            metric={"feature": feat, "steering_sim_50": steering.get(50.0, {}).get("mean_embedding_similarity", 0)})

    # Phase 3: Random baseline
    print(f"\n  Running random baseline...")
    report_progress(task_id, results_dir, epoch=3, total_epochs=3, step=0, total_steps=1)

    all_vocab_words = []
    for words in VOCAB.values():
        all_vocab_words.extend(words[:10])

    random_baseline = run_random_baseline_steering(
        model, sae, tokenizer, all_vocab_words, n_random=26,
        strengths=STEERING_STRENGTHS, n_prompts=100, device=device
    )

    report_progress(task_id, results_dir, epoch=3, total_epochs=3, step=1, total_steps=1)

    # Compute delta-corrected metrics
    print(f"\n  Computing delta-corrected metrics...")
    delta_corrected = {}
    for feat in FEATURES:
        if feat in steering_results:
            feat_sim_50 = steering_results[feat].get(50.0, {}).get("mean_embedding_similarity", 0.0)
            random_sims = [random_baseline[k].get(50.0, {}).get("mean_embedding_similarity", 0.0)
                          for k in random_baseline.keys()]
            mean_random_sim = np.mean(random_sims) if random_sims else 0.0
            delta = feat_sim_50 - mean_random_sim
            delta_corrected[feat] = {
                "feature_similarity": float(feat_sim_50),
                "random_mean_similarity": float(mean_random_sim),
                "delta": float(delta),
            }

    # Correlation analysis
    absorption_vals = [absorption_rates[f]["absorption_rate"] for f in FEATURES]
    raw_sim_vals = [steering_results.get(f, {}).get(50.0, {}).get("mean_embedding_similarity", 0.0) for f in FEATURES]
    delta_vals = [delta_corrected.get(f, {}).get("delta", 0.0) for f in FEATURES]

    r_raw, p_raw = stats.pearsonr(absorption_vals, raw_sim_vals)
    r_delta, p_delta = stats.pearsonr(absorption_vals, delta_vals)

    correlation = {
        "raw": {"r": float(r_raw), "p": float(p_raw), "R2": float(r_raw**2)},
        "delta_corrected": {"r": float(r_delta), "p": float(p_delta), "R2": float(r_delta**2)},
    }

    print(f"  Raw correlation: r={r_raw:.3f}, p={p_raw:.3f}")
    print(f"  Delta-corrected: r={r_delta:.3f}, p={p_delta:.3f}")

    # Save combined results
    combined_output = {
        "model": "pythia-70m-deduped",
        "sae_release": "pythia-70m-deduped-res-sm",
        "layer": layer,
        "dict_size": 32768,
        "absorption": absorption_output,
        "steering": steering_results,
        "random_baseline": random_baseline,
        "delta_corrected": delta_corrected,
        "correlation": correlation,
        "summary": {
            "mean_absorption": float(np.mean(absorption_vals)),
            "max_absorption": float(max(absorption_vals)),
            "raw_r": float(r_raw),
            "raw_p": float(p_raw),
            "delta_r": float(r_delta),
            "delta_p": float(p_delta),
            "delta_significant": bool(p_delta < 0.05),
        }
    }

    with open(results_dir / f"{task_id}_combined.json", "w") as f:
        json.dump(combined_output, f, indent=2)

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Pythia L{layer}: mean_abs={np.mean(absorption_vals):.3f}, delta_r={r_delta:.3f}, p={p_delta:.3f}")

    print(f"\n  Layer {layer} complete!")
    print(f"  Mean absorption: {np.mean(absorption_vals):.3f}")
    print(f"  Delta-corrected r: {r_delta:.3f} (p={p_delta:.3f})")

    return combined_output


def run_cross_model_validation(layers=None):
    task_id = "cross_model_pythia"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)

    if layers is None:
        layers = [4]

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

    all_results = {}
    for layer in layers:
        layer_results = run_absorption_for_layer(layer, model, tokenizer, device="cuda")
        all_results[f"layer_{layer}"] = layer_results

    # Combined summary
    combined = {
        "model": "pythia-70m-deduped",
        "sae_release": "pythia-70m-deduped-res-sm",
        "layers": layers,
        "dict_size": 32768,
        "layer_results": all_results,
        "cross_model_summary": {
            "gpt2_comparison": {
                "note": "Compare with GPT-2 Small results for generalizability assessment",
            }
        }
    }

    with open(results_dir / "cross_model_pythia_combined.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Cross-Model Validation Complete")
    for l in layers:
        s = all_results[f"layer_{l}"]["summary"]
        print(f"  Layer {l}: mean_abs={s['mean_absorption']:.3f}, delta_r={s['delta_r']:.3f}, p={s['delta_p']:.3f}")
    print(f"{'=' * 70}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Pythia cross-validation complete for layers {layers}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="4", help="Comma-separated layer indices")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]

    try:
        results = run_cross_model_validation(layers=layers)
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent / "results" / "full"
        mark_task_done("cross_model_pythia", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
