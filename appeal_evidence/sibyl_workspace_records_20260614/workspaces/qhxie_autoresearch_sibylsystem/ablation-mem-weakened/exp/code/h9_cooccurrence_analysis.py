#!/usr/bin/env python3
"""
H9: Co-occurrence Strength vs. Absorption Rate

For each first-letter feature, compute p_11 = fraction of child prompts where
parent latent also fires. Test correlation between p_11 and absorption_rate.

Hypothesis: Features with stronger parent-child co-occurrence (higher p_11)
exhibit higher absorption rates.
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

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
SAMPLES_PER_FEATURE = 100
LAYER = 8  # Primary layer for H9

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


def find_parent_latent(model, sae, tokenizer, feature_letter, vocab_words, n_samples=100, device="cuda"):
    """Find the single most selective latent for words starting with the given letter."""
    hook_name = sae.cfg.metadata.get("hook_name", f"blocks.{LAYER}.hook_resid_pre")

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

    # Return the single most selective latent
    parent_idx = int(np.argmax(selectivity))
    return parent_idx, float(selectivity[parent_idx]), float(target_mean[parent_idx])


def compute_cooccurrence_and_absorption(model, sae, tokenizer, feature_letter, vocab_words,
                                        parent_latent, n_samples=100, device="cuda"):
    """
    For each child prompt (word starting with feature_letter), check:
    1. Does the parent latent fire? (for p_11)
    2. Does any child latent fire while parent is suppressed? (for absorption)
    """
    hook_name = sae.cfg.metadata.get("hook_name", f"blocks.{LAYER}.hook_resid_pre")

    # Generate child prompts
    prompts = [f"The word '{w}' means something" for w in vocab_words * (n_samples // len(vocab_words) + 1)]
    prompts = prompts[:n_samples]

    # First pass: find top child latents (most active on child prompts, excluding parent)
    child_activations = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_acts = sae_acts.max(dim=0).values.cpu().numpy()
            # Zero out parent latent
            max_acts_copy = max_acts.copy()
            max_acts_copy[parent_latent] = 0
            child_activations.append(max_acts_copy)

    child_mean = np.mean(child_activations, axis=0)
    # Top 50 child latents (excluding parent)
    child_mean[parent_latent] = -1e9
    top_child_indices = np.argsort(child_mean)[-50:][::-1]

    # Second pass: compute co-occurrence and absorption per prompt
    parent_fires_count = 0
    child_fires_parent_suppressed_count = 0
    total_prompts = len(prompts)

    parent_acts_list = []
    child_max_acts_list = []

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)[0]
            max_acts = sae_acts.max(dim=0).values.cpu().numpy()

            parent_act = max_acts[parent_latent]
            parent_acts_list.append(parent_act)

            # Check if parent fires (above median activation)
            parent_fires = parent_act > 0.1  # Threshold for "fires"
            if parent_fires:
                parent_fires_count += 1

            # Check if any child fires while parent is suppressed
            child_max = max(max_acts[c] for c in top_child_indices[:20])
            child_max_acts_list.append(child_max)

            if not parent_fires and child_max > 0.1:
                child_fires_parent_suppressed_count += 1

    # Compute p_11: fraction where parent fires (among child prompts)
    p_11 = parent_fires_count / total_prompts if total_prompts > 0 else 0.0

    # Compute absorption rate: fraction where child fires but parent is suppressed
    absorption_rate = child_fires_parent_suppressed_count / total_prompts if total_prompts > 0 else 0.0

    return {
        "feature": feature_letter,
        "parent_latent": int(parent_latent),
        "n_prompts": total_prompts,
        "parent_fires_count": parent_fires_count,
        "p_11": float(p_11),
        "child_fires_parent_suppressed_count": child_fires_parent_suppressed_count,
        "absorption_rate": float(absorption_rate),
        "parent_act_mean": float(np.mean(parent_acts_list)),
        "parent_act_std": float(np.std(parent_acts_list)),
        "child_act_mean": float(np.mean(child_max_acts_list)),
        "child_act_std": float(np.std(child_max_acts_list)),
    }


def run_h9_experiment():
    task_id = "h9_cooccurrence_analysis"
    results_dir = Path(__file__).parent.parent / "results" / "pilots"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=1, step=0, total_steps=len(FEATURES))

    print("=" * 70)
    print("H9: Co-occurrence Strength vs. Absorption Rate")
    print(f"Layer: {LAYER}, Samples per feature: {SAMPLES_PER_FEATURE}")
    print("=" * 70)

    set_seed(SEED)

    print("\n[Loading] Loading GPT-2 Small...")
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        "gpt2",
        device=device,
        dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model: GPT-2 Small ({model.cfg.n_params/1e6:.0f}M params)")

    # Load SAE
    sae_id = f"blocks.{LAYER}.hook_resid_pre"
    print(f"\n[Loading] Loading SAE: {sae_id}...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device,
    )
    print(f"  SAE: {sae.cfg.d_sae} latents")

    # Run analysis for all 26 features
    feature_results = []
    for i, feat in enumerate(FEATURES):
        print(f"\n[{i+1}/26] Feature '{feat}':")

        # Find parent latent
        parent_latent, selectivity, target_mean = find_parent_latent(
            model, sae, tokenizer, feat, VOCAB[feat],
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        print(f"  Parent latent: {parent_latent} (selectivity={selectivity:.3f})")

        # Compute co-occurrence and absorption
        result = compute_cooccurrence_and_absorption(
            model, sae, tokenizer, feat, VOCAB[feat],
            parent_latent, n_samples=SAMPLES_PER_FEATURE, device=device
        )
        feature_results.append(result)

        print(f"  p_11={result['p_11']:.3f}, absorption_rate={result['absorption_rate']:.3f}")

        report_progress(task_id, results_dir, epoch=1, total_epochs=1, step=i+1, total_steps=len(FEATURES),
                        metric={"feature": feat, "p_11": result["p_11"], "absorption_rate": result["absorption_rate"]})

    # Compute correlation
    p11_values = [r["p_11"] for r in feature_results]
    abs_values = [r["absorption_rate"] for r in feature_results]

    pearson_r, pearson_p = stats.pearsonr(p11_values, abs_values)
    spearman_r, spearman_p = stats.spearmanr(p11_values, abs_values)

    print(f"\n{'=' * 70}")
    print("CORRELATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.4f}")

    # Load existing absorption rates for comparison
    existing_absorption = {}
    existing_abs_path = results_dir.parent / "full" / f"full_absorption_gpt2_l{LAYER}_absorption_rates.json"
    if existing_abs_path.exists():
        with open(existing_abs_path) as f:
            existing_data = json.load(f)
            for feat in FEATURES:
                existing_absorption[feat] = existing_data["absorption_rates"][feat]["absorption_rate"]

        # Correlation with existing absorption rates
        existing_values = [existing_absorption.get(f, 0) for f in FEATURES]
        r_existing, p_existing = stats.pearsonr(p11_values, existing_values)
        print(f"Correlation with existing Chanin absorption: r = {r_existing:.4f}, p = {p_existing:.4f}")
    else:
        r_existing, p_existing = None, None

    # Determine GO/NO-GO
    if pearson_r > 0.2 and pearson_p < 0.10:
        go_no_go = "GO"
        confidence = min(0.5 + abs(pearson_r), 0.95)
    elif pearson_r < 0.1:
        go_no_go = "NO_GO"
        confidence = 0.7
    else:
        go_no_go = "UNCERTAIN"
        confidence = 0.5

    output = {
        "task_id": task_id,
        "layer": LAYER,
        "n_features": len(FEATURES),
        "n_samples_per_feature": SAMPLES_PER_FEATURE,
        "seed": SEED,
        "feature_results": feature_results,
        "correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_r),
            "spearman_p": float(spearman_p),
            "r_with_existing_chanin": float(r_existing) if r_existing is not None else None,
            "p_with_existing_chanin": float(p_existing) if p_existing is not None else None,
        },
        "go_no_go": go_no_go,
        "confidence": float(confidence),
        "summary": {
            "mean_p11": float(np.mean(p11_values)),
            "std_p11": float(np.std(p11_values)),
            "mean_absorption": float(np.mean(abs_values)),
            "std_absorption": float(np.std(abs_values)),
            "max_p11": float(max(p11_values)),
            "min_p11": float(min(p11_values)),
        }
    }

    output_path = results_dir / f"{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"GO/NO-GO: {go_no_go} (confidence={confidence:.2f})")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"H9: Pearson r={pearson_r:.3f}, p={pearson_p:.3f}, {go_no_go}")

    return output


if __name__ == "__main__":
    try:
        results = run_h9_experiment()
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent.parent / "results" / "pilots"
        mark_task_done("h9_cooccurrence_analysis", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
