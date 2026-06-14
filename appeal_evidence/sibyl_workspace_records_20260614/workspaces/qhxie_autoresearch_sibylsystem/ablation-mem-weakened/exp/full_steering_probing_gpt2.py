#!/usr/bin/env python3
"""
Full Experiment: Feature Steering and Sparse Probing on GPT-2 Small
Tasks: full_steering_gpt2, full_probing_gpt2

Run steering and probing on all 26 first-letter features for layers 4 and 8.
Uses absorption data from full_absorption_gpt2.py to correlate with task performance.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
SAMPLES_PER_FEATURE = 100
STRENGTHS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
K_VALUES = [1, 5, 10, 20]

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


def run_steering_experiment(model, sae, tokenizer, feature_letter, vocab_words, parent_latent, strengths, n_samples=100, device="cuda"):
    """Test steering effectiveness using residual-stream steering."""
    hook_name = sae.cfg.metadata.get("hook_name", "blocks.8.hook_resid_pre")

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

            if rel_lift > 1.5:
                successes += 1

        success_rate = successes / n_tested if n_tested > 0 else 0.0
        mean_lift = float(np.mean(prob_lifts)) if prob_lifts else 0.0
        median_lift = float(np.median(prob_lifts)) if prob_lifts else 0.0
        mean_rel_lift = float(np.mean(rel_lifts)) if rel_lifts else 1.0

        results.append({
            "strength": float(strength),
            "success_rate": float(success_rate),
            "mean_prob_lift": mean_lift,
            "median_prob_lift": median_lift,
            "mean_rel_lift": mean_rel_lift,
            "n_tested": n_tested,
            "n_successes": successes,
        })

    return {"strength_results": results, "parent_latent": int(parent_latent)}


def run_probing_experiment(model, sae, tokenizer, feature_letter, vocab_words, parent_latent, k_values, device="cuda"):
    """Train k-sparse linear probes on first-letter classification."""
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


def run_layer_experiments(layer, model, tokenizer, device="cuda"):
    """Run steering and probing for a single layer."""
    from sae_lens import SAE

    task_id = f"full_steering_probing_gpt2_l{layer}"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)

    # Load absorption data
    absorption_file = results_dir / f"full_absorption_gpt2_l{layer}_absorption_rates.json"
    if not absorption_file.exists():
        print(f"ERROR: Absorption data not found for layer {layer}")
        mark_task_done(task_id, results_dir, status="failed", summary="Absorption data not found")
        return None

    with open(absorption_file) as f:
        absorption_data = json.load(f)

    # Load SAE
    print(f"\n[Layer {layer}] Loading SAE...")
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device,
    )

    # Steering experiment
    print(f"[Layer {layer}] Running steering experiments...")
    steering_results = {}
    report_progress(task_id, results_dir, epoch=1, total_epochs=2, step=0, total_steps=26)

    for i, feat in enumerate(FEATURES):
        parent_latent = absorption_data["absorption_rates"][feat].get("parent_latent", None)
        if parent_latent is None:
            continue

        result = run_steering_experiment(
            model, sae, tokenizer, feat, VOCAB[feat], parent_latent,
            STRENGTHS, n_samples=SAMPLES_PER_FEATURE, device=device
        )
        steering_results[feat] = result

        if (i + 1) % 5 == 0 or i == 0:
            sr_50 = [r["success_rate"] for r in result["strength_results"] if r["strength"] == 50.0]
            print(f"  [{i+1}/26] '{feat}': success@50={sr_50[0]:.2f}" if sr_50 else f"  [{i+1}/26] '{feat}': N/A")

        report_progress(task_id, results_dir, epoch=1, total_epochs=2, step=i+1, total_steps=26,
                        metric={"feature": feat, "stage": "steering"})

    # Probing experiment
    print(f"[Layer {layer}] Running probing experiments...")
    probing_results = {}
    report_progress(task_id, results_dir, epoch=2, total_epochs=2, step=0, total_steps=26)

    for i, feat in enumerate(FEATURES):
        parent_latent = absorption_data["absorption_rates"][feat].get("parent_latent", None)
        if parent_latent is None:
            continue

        result = run_probing_experiment(
            model, sae, tokenizer, feat, VOCAB[feat], parent_latent,
            K_VALUES, device=device
        )
        probing_results[feat] = result

        if (i + 1) % 5 == 0 or i == 0:
            f1_5 = [r["f1"] for r in result["k_results"] if r["k"] == 5]
            print(f"  [{i+1}/26] '{feat}': F1@5={f1_5[0]:.3f}" if f1_5 else f"  [{i+1}/26] '{feat}': N/A")

        report_progress(task_id, results_dir, epoch=2, total_epochs=2, step=i+1, total_steps=26,
                        metric={"feature": feat, "stage": "probing"})

    # Save results
    output = {
        "model": "gpt2-small",
        "layer": layer,
        "steering_results": steering_results,
        "probing_results": probing_results,
    }

    with open(results_dir / f"{task_id}_results.json", "w") as f:
        json.dump(output, f, indent=2)

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Layer {layer}: steering + probing complete for 26 features")

    print(f"  Layer {layer} complete. Results saved.")
    return output


def run_full_experiment(layers=None):
    task_id = "full_steering_probing_gpt2"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)

    if layers is None:
        layers = [4, 8]

    print("=" * 70)
    print("Full Experiment: Steering + Probing on GPT-2 Small")
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

    all_results = {}
    for layer in layers:
        layer_results = run_layer_experiments(layer, model, tokenizer, device="cuda:0")
        all_results[f"layer_{layer}"] = layer_results

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"All layers {layers} complete")

    print(f"\n{'=' * 70}")
    print("All steering + probing experiments complete")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="4,8", help="Comma-separated layer indices")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]

    try:
        results = run_full_experiment(layers=layers)
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent / "results" / "full"
        mark_task_done("full_steering_probing_gpt2", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
