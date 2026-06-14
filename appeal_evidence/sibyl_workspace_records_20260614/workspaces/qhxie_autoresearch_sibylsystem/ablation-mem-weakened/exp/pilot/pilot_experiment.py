#!/usr/bin/env python3
"""
Pilot Experiment: Feature Absorption Degrades Downstream SAE Reliability
Model: GPT-2 Small, Layer 8, SAE: res-jb (24K latents)

This script implements:
1. Absorption detection using differential activation analysis
2. Feature steering effectiveness measurement (SAE-level steering)
3. Sparse probing accuracy (k-sparse linear probes)
4. Correlation analysis between absorption and task degradation

All experiments are training-free -- we analyze pre-trained SAEs.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# Set CUDA device from env
# When CUDA_VISIBLE_DEVICES is set, the visible device is always cuda:0 in the process
if "CUDA_VISIBLE_DEVICES" in os.environ:
    device = "cuda:0"
else:
    device = "cuda:0"

SEED = 42
FEATURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
STRENGTHS = [10.0, 20.0, 50.0, 100.0]
K_VALUES = [1, 5, 10]
SAMPLES_PER_FEATURE = 50  # Reduced per feature to keep total time reasonable

# Vocabulary words for each first letter
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
    """
    Find SAE latents that are MOST SELECTIVE for words starting with the given letter.
    Uses differential activation: target words vs. non-target words.
    """
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
    """
    Detect absorption: measure how much the parent's signal is 'spread' across children.
    A child 'absorbs' if it is significantly positively correlated with parent activation.
    """
    if len(top_latents) < 2:
        return {"absorption_rate": 0.0, "n_latents": len(top_latents), "method": "differential_correlation"}

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")
    parent_latent = top_latents[0][0]
    child_latents = [l[0] for l in top_latents[1:51]]  # Top 50 children for more variance

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
            correlation_values.append(max(0, r))  # Only positive correlations count
            if r > 0.3 and p < 0.05:
                absorbing_children += 1
        else:
            child_correlations.append({"latent": int(c), "r": 0.0, "p": 1.0})
            correlation_values.append(0.0)

    # Use mean positive correlation as continuous absorption rate (more variance)
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


def run_steering_experiment(model, sae, tokenizer, feature_letter, vocab_words, absorption_info, strengths, n_samples=100, device="cuda"):
    """
    Test steering effectiveness using residual-stream steering.
    Add the SAE decoder direction for the parent latent to the residual stream.

    Uses RELATIVE probability lift (multiplicative) as the metric since absolute
    probabilities for specific tokens are extremely small in open-vocabulary settings.
    """
    parent_latent = absorption_info.get("parent_latent", None)
    if parent_latent is None:
        return {"error": "No parent latent found"}

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")

    # Get steering direction from SAE decoder
    direction = sae.W_dec[parent_latent].detach().clone()
    direction = direction / (direction.norm() + 1e-8)

    contexts = [
        "The word", "I like the word", "The best word is", "Consider the word",
        "A good example is the word", "The famous word", "Remember the word",
    ]

    np.random.seed(SEED)
    test_words = vocab_words[:n_samples]

    results = []
    for strength in strengths:
        successes = 0
        prob_lifts = []
        rel_lifts = []
        n_tested = 0

        for word in test_words:
            context = np.random.choice(contexts)
            prompt = f"{context}"
            tokens = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = model(tokens.input_ids)
                next_token_logits = logits[0, -1, :]
                probs_base = torch.softmax(next_token_logits, dim=-1)

            word_tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
            if not word_tokens:
                continue
            target_token = word_tokens[0]

            baseline_prob = probs_base[target_token].item()

            # Residual stream steering: add direction to last token position
            def steering_hook(value, hook):
                value[:, -1, :] += strength * direction
                return value

            with torch.no_grad():
                steered_logits = model.run_with_hooks(
                    tokens.input_ids,
                    fwd_hooks=[(hook_name, steering_hook)]
                )
                next_token_logits_steered = steered_logits[0, -1, :]
                probs_steered = torch.softmax(next_token_logits_steered, dim=-1)

            steered_prob = probs_steered[target_token].item()
            prob_lift = steered_prob - baseline_prob
            rel_lift = (steered_prob / (baseline_prob + 1e-10)) if baseline_prob > 0 else (100.0 if steered_prob > 0 else 1.0)
            prob_lifts.append(prob_lift)
            rel_lifts.append(rel_lift)
            n_tested += 1

            # Success: relative lift > 1.5x (50% increase of probability)
            if rel_lift > 1.5:
                successes += 1

        success_rate = successes / n_tested if n_tested > 0 else 0.0
        mean_lift = float(np.mean(prob_lifts)) if prob_lifts else 0.0
        median_lift = float(np.median(prob_lifts)) if prob_lifts else 0.0
        mean_rel_lift = float(np.mean(rel_lifts)) if rel_lifts else 1.0
        median_rel_lift = float(np.median(rel_lifts)) if rel_lifts else 1.0

        results.append({
            "strength": float(strength),
            "success_rate": float(success_rate),
            "mean_prob_lift": mean_lift,
            "median_prob_lift": median_lift,
            "mean_rel_lift": mean_rel_lift,
            "median_rel_lift": median_rel_lift,
            "n_tested": n_tested,
            "n_successes": successes,
        })

    return {"strength_results": results, "parent_latent": int(parent_latent)}


def run_probing_experiment(model, sae, tokenizer, feature_letter, vocab_words, absorption_info, k_values, device="cuda"):
    """Train k-sparse linear probes on first-letter classification."""
    parent_latent = absorption_info.get("parent_latent", None)
    if parent_latent is None:
        return {"error": "No parent latent found"}

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")

    all_words = []
    labels = []
    for letter, words in VOCAB.items():
        for w in words:
            all_words.append(w)
            labels.append(1 if letter == feature_letter else 0)

    activations = []
    for word in all_words:
        prompt = f"The word '{word}'"
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_act = sae_acts.max(dim=0).values.cpu().numpy()
            activations.append(max_act)

    X = np.array(activations)
    y = np.array(labels)

    results = []
    for k in k_values:
        if k == 1:
            X_k = X[:, parent_latent:parent_latent+1].reshape(-1, 1)
        else:
            correlations = []
            for i in range(X.shape[1]):
                if np.std(X[:, i]) > 0:
                    r, _ = stats.pearsonr(X[:, i], y)
                    correlations.append((i, abs(r)))
                else:
                    correlations.append((i, 0.0))

            top_k_indices = [c[0] for c in sorted(correlations, key=lambda x: x[1], reverse=True)[:k]]
            X_k = X[:, top_k_indices]

        probe = LogisticRegression(max_iter=1000, random_state=SEED)
        probe.fit(X_k, y)
        y_pred = probe.predict(X_k)

        f1 = float(f1_score(y, y_pred, zero_division=0))
        precision = float(precision_score(y, y_pred, zero_division=0))
        recall = float(recall_score(y, y_pred, zero_division=0))

        results.append({
            "k": k,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "n_features": int(X_k.shape[1]),
        })

    return {"k_results": results, "parent_latent": int(parent_latent), "n_samples": len(all_words)}


def analyze_correlation(absorption_rates, steering_results, probing_results):
    """Compute correlations between absorption rate and task performance."""
    features = list(absorption_rates.keys())
    abs_vals = [absorption_rates[f]["absorption_rate"] for f in features]

    # Steering: use success rate at strength=50.0
    steer_success = []
    for f in features:
        sr_50 = [r["success_rate"] for r in steering_results[f]["strength_results"] if r["strength"] == 50.0]
        steer_success.append(sr_50[0] if sr_50 else 0.0)

    probe_f1 = []
    for f in features:
        f1_k5 = [r["f1"] for r in probing_results[f]["k_results"] if r["k"] == 5]
        probe_f1.append(f1_k5[0] if f1_k5 else 0.0)

    if len(features) >= 2 and len(set(abs_vals)) > 1:
        r_steering, p_steering = stats.pearsonr(abs_vals, steer_success)
        r_probing, p_probing = stats.pearsonr(abs_vals, probe_f1)
        slope_steering, intercept_steering, r_val_s, _, _ = stats.linregress(abs_vals, steer_success)
        slope_probing, intercept_probing, r_val_p, _, _ = stats.linregress(abs_vals, probe_f1)
    else:
        r_steering = p_steering = r_probing = p_probing = float("nan")
        slope_steering = intercept_steering = r_val_s = slope_probing = intercept_probing = r_val_p = float("nan")

    h1_pass = bool(r_steering < 0 and p_steering < 0.1) if not (np.isnan(r_steering) or np.isnan(p_steering)) else False
    h2_pass = bool(r_probing < 0 and p_probing < 0.1) if not (np.isnan(r_probing) or np.isnan(p_probing)) else False

    return {
        "H1_steering": {
            "r": float(r_steering),
            "p": float(p_steering),
            "n": len(features),
            "R2": float(r_val_s ** 2) if not np.isnan(r_val_s) else float("nan"),
            "slope": float(slope_steering),
            "intercept": float(intercept_steering),
            "interpretation": "negative" if r_steering < 0 else "positive/null",
            "passes_pilot": h1_pass,
        },
        "H2_probing": {
            "r": float(r_probing),
            "p": float(p_probing),
            "n": len(features),
            "R2": float(r_val_p ** 2) if not np.isnan(r_val_p) else float("nan"),
            "slope": float(slope_probing),
            "intercept": float(intercept_probing),
            "interpretation": "negative" if r_probing < 0 else "positive/null",
            "passes_pilot": h2_pass,
        },
        "absorption_rates": {f: float(absorption_rates[f]["absorption_rate"]) for f in features},
        "steering_success_at_50": {f: float(s) for f, s in zip(features, steer_success)},
        "probing_f1_k5": {f: float(p) for f, p in zip(features, probe_f1)},
    }


def run_pilot():
    task_id = "pilot_cand_a"
    results_dir = Path(__file__).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=4, step=0, total_steps=4)

    print("=" * 70)
    print("Pilot Experiment: Feature Absorption -> Downstream Reliability")
    print("Model: GPT-2 Small, Layer 8, SAE: res-jb (24K latents)")
    print(f"Device: {device}")
    print(f"Features: {FEATURES}")
    print("=" * 70)

    set_seed(SEED)

    # Step 0: Load model and SAE
    print("\n[Step 0/4] Loading model and SAE...")
    from sae_lens import SAE
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2",
        device=device,
        dtype=torch.float32,
    )

    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")

    print(f"  Model: GPT-2 Small ({model.cfg.n_params/1e6:.0f}M params)")
    print(f"  SAE: {sae.cfg.d_sae} latents, hook={hook_name}")
    print(f"  d_model: {sae.cfg.d_in}")

    report_progress(task_id, results_dir, epoch=1, total_epochs=4, step=1, total_steps=4,
                    metric={"stage": "model_loaded"})

    # Step 1: Absorption Detection
    print("\n[Step 1/4] Detecting absorption rates...")
    absorption_rates = {}
    for feat in FEATURES:
        print(f"  Feature '{feat}': Finding selective latents...", end=" ", flush=True)
        top_latents = find_feature_latents(
            model, sae, tokenizer, feat, VOCAB[feat],
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        print(f"found {len(top_latents)} selective latents")

        if len(top_latents) > 0:
            print(f"    Top latent: id={top_latents[0][0]}, selectivity={top_latents[0][1]:.4f}")

        print(f"  Feature '{feat}': Detecting absorption...", end=" ", flush=True)
        absorption = detect_absorption(
            model, sae, tokenizer, feat, VOCAB[feat], top_latents,
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        absorption_rates[feat] = absorption
        print(f"rate={absorption['absorption_rate']:.3f} (parent={absorption.get('parent_latent', 'N/A')}, absorbing_children={absorption.get('absorbing_children', 0)}/{absorption.get('n_child_latents', 0)})")

    report_progress(task_id, results_dir, epoch=2, total_epochs=4, step=2, total_steps=4,
                    metric={"stage": "absorption_done"})

    # Step 2: Steering Experiment
    print("\n[Step 2/4] Running steering experiments...")
    steering_results = {}
    for feat in FEATURES:
        print(f"  Feature '{feat}': Steering...", end=" ", flush=True)
        result = run_steering_experiment(
            model, sae, tokenizer, feat, VOCAB[feat], absorption_rates[feat],
            STRENGTHS, n_samples=SAMPLES_PER_FEATURE, device=device
        )
        steering_results[feat] = result
        sr_50 = [r["success_rate"] for r in result["strength_results"] if r["strength"] == 50.0]
        rel_50 = [r["mean_rel_lift"] for r in result["strength_results"] if r["strength"] == 50.0]
        print(f"success@50.0={sr_50[0]:.3f}, rel_lift={rel_50[0]:.2f}x" if sr_50 else "N/A")

    report_progress(task_id, results_dir, epoch=3, total_epochs=4, step=3, total_steps=4,
                    metric={"stage": "steering_done"})

    # Step 3: Probing Experiment
    print("\n[Step 3/4] Running sparse probing...")
    probing_results = {}
    for feat in FEATURES:
        print(f"  Feature '{feat}': Probing...", end=" ", flush=True)
        result = run_probing_experiment(
            model, sae, tokenizer, feat, VOCAB[feat], absorption_rates[feat],
            K_VALUES, device=device
        )
        probing_results[feat] = result
        f1_at_5 = [r["f1"] for r in result["k_results"] if r["k"] == 5]
        print(f"F1@k=5={f1_at_5[0]:.3f}" if f1_at_5 else "N/A")

    report_progress(task_id, results_dir, epoch=4, total_epochs=4, step=4, total_steps=4,
                    metric={"stage": "probing_done"})

    # Step 4: Correlation Analysis
    print("\n[Step 4/4] Correlation analysis...")
    correlation = analyze_correlation(absorption_rates, steering_results, probing_results)

    h1_pass = correlation["H1_steering"]["passes_pilot"]
    h2_pass = correlation["H2_probing"]["passes_pilot"]

    print(f"\n  H1 (Steering): r={correlation['H1_steering']['r']:.3f}, p={correlation['H1_steering']['p']:.3f}, passes={h1_pass}")
    print(f"  H2 (Probing):  r={correlation['H2_probing']['r']:.3f}, p={correlation['H2_probing']['p']:.3f}, passes={h2_pass}")

    # Save results
    with open(results_dir / "absorption_rates.json", "w") as f:
        json.dump(absorption_rates, f, indent=2)
    with open(results_dir / "steering_results.json", "w") as f:
        json.dump(steering_results, f, indent=2)
    with open(results_dir / "probing_results.json", "w") as f:
        json.dump(probing_results, f, indent=2)
    with open(results_dir / "correlation_report.json", "w") as f:
        json.dump(correlation, f, indent=2)

    # Pilot summary
    n_high_absorption = sum(1 for a in absorption_rates.values() if a["absorption_rate"] > 0.1)

    pilot_summary = {
        "overall_recommendation": "GO" if (h1_pass or h2_pass) and n_high_absorption >= 2 else "NO_GO",
        "selected_candidate_id": "cand_a",
        "n_high_absorption": n_high_absorption,
        "H1_passes": h1_pass,
        "H2_passes": h2_pass,
        "candidates": [{
            "candidate_id": "cand_a",
            "go_no_go": "GO" if (h1_pass or h2_pass) and n_high_absorption >= 2 else "NO_GO",
            "confidence": 0.7 if (h1_pass or h2_pass) else 0.3,
            "supported_hypotheses": (["H1"] if h1_pass else []) + (["H2"] if h2_pass else []),
            "failed_assumptions": [],
            "key_metrics": {
                "absorption_rates": {f: float(absorption_rates[f]["absorption_rate"]) for f in FEATURES},
                "steering_success_at_50": correlation["steering_success_at_50"],
                "probing_f1_k5": correlation["probing_f1_k5"],
            },
            "notes": f"{n_high_absorption}/5 features show >10% absorption. H1={'PASS' if h1_pass else 'FAIL'}, H2={'PASS' if h2_pass else 'FAIL'}."
        }]
    }

    with open(results_dir / "pilot_summary.json", "w") as f:
        json.dump(pilot_summary, f, indent=2)

    # Markdown summary
    md = f"""# Pilot Results: Feature Absorption -> Downstream Reliability

## Configuration
- Model: GPT-2 Small
- SAE: res-jb, layer 8 (hook_resid_pre), 24K latents
- Features: {', '.join(FEATURES)}
- Samples per feature: {SAMPLES_PER_FEATURE}
- Seed: {SEED}

## Absorption Rates
| Feature | Absorption Rate | Parent Latent | Absorbing Children | Total Children |
|---------|----------------|---------------|-------------------|----------------|
"""
    for feat in FEATURES:
        a = absorption_rates[feat]
        md += f"| {feat} | {a['absorption_rate']:.3f} | {a.get('parent_latent', 'N/A')} | {a.get('absorbing_children', 'N/A')} | {a.get('n_child_latents', 'N/A')} |\n"

    md += f"""
## Steering Results (strength=5.0)
| Feature | Success Rate | Mean Prob Lift | Median Prob Lift | N Tested |
|---------|-------------|----------------|------------------|----------|
"""
    for feat in FEATURES:
        sr_5 = [r for r in steering_results[feat]["strength_results"] if r["strength"] == 5.0]
        if sr_5:
            md += f"| {feat} | {sr_5[0]['success_rate']:.3f} | {sr_5[0]['mean_prob_lift']:.4f} | {sr_5[0]['median_prob_lift']:.4f} | {sr_5[0]['n_tested']} |\n"

    md += f"""
## Probing Results (k=5)
| Feature | F1 | Precision | Recall |
|---------|-----|-----------|--------|
"""
    for feat in FEATURES:
        f1_5 = [r for r in probing_results[feat]["k_results"] if r["k"] == 5]
        if f1_5:
            md += f"| {feat} | {f1_5[0]['f1']:.3f} | {f1_5[0]['precision']:.3f} | {f1_5[0]['recall']:.3f} |\n"

    md += f"""
## Correlation Analysis
- **H1 (Steering)**: r={correlation['H1_steering']['r']:.3f}, p={correlation['H1_steering']['p']:.3f}, R2={correlation['H1_steering']['R2']:.3f}
- **H2 (Probing)**: r={correlation['H2_probing']['r']:.3f}, p={correlation['H2_probing']['p']:.3f}, R2={correlation['H2_probing']['R2']:.3f}

## Pilot Verdict: {pilot_summary['overall_recommendation']}
- H1 passes pilot threshold: {h1_pass}
- H2 passes pilot threshold: {h2_pass}
- Features with >10% absorption: {n_high_absorption}/5
"""

    with open(results_dir / "pilot_summary.md", "w") as f:
        f.write(md)

    print(f"\n{'=' * 70}")
    print(f"Pilot Complete -- Verdict: {pilot_summary['overall_recommendation']}")
    print(f"  H1 (Steering): {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H2 (Probing):  {'PASS' if h2_pass else 'FAIL'}")
    print(f"  High absorption features: {n_high_absorption}/5")
    print(f"  Results saved to: {results_dir}")
    print(f"{'=' * 70}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Pilot {pilot_summary['overall_recommendation']}: H1={'PASS' if h1_pass else 'FAIL'}, H2={'PASS' if h2_pass else 'FAIL'}, {n_high_absorption}/5 high absorption")

    return pilot_summary


if __name__ == "__main__":
    try:
        results = run_pilot()
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent
        mark_task_done("pilot_cand_a", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
