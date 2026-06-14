#!/usr/bin/env python3
"""
H10: Random SAE Baseline Absorption

Construct a random SAE baseline by freezing decoder weights (or using orthogonal
random matrix) and random encoder. Run the Chanin absorption metric on the same
26 first-letter features to test if absorption is structural or learned.

Hypothesis: Random SAE baselines exhibit absorption-like patterns, confirming
absorption is partially a structural artifact.
"""

import json
import os
import sys
import copy
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
SAMPLES_PER_FEATURE = 100
LAYER = 8  # Primary layer for H10

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


def create_random_sae(sae, seed=SEED):
    """
    Create a random SAE baseline by:
    1. Freezing decoder weights at random initialization (orthonormal)
    2. Using random encoder weights
    3. Keeping the same architecture (d_sae, d_model, etc.)
    """
    random_sae = copy.deepcopy(sae)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_sae = sae.cfg.d_sae
    d_model = sae.cfg.d_in  # StandardSAEConfig uses d_in not d_model
    device = sae.W_dec.device

    # Random decoder: orthonormal rows (normalized random vectors)
    W_dec_random = torch.randn(d_sae, d_model, device=device)
    W_dec_random = F.normalize(W_dec_random, dim=1)
    random_sae.W_dec.data = W_dec_random

    # Random encoder
    W_enc_random = torch.randn(d_model, d_sae, device=device)
    random_sae.W_enc.data = W_enc_random

    # Random bias - ensure on same device
    if hasattr(random_sae, 'b_enc') and random_sae.b_enc is not None:
        random_sae.b_enc.data = torch.randn(d_sae, device=device) * 0.1
    if hasattr(random_sae, 'b_dec') and random_sae.b_dec is not None:
        random_sae.b_dec.data = torch.randn(d_model, device=device) * 0.1

    # Ensure all parameters are on the same device
    random_sae = random_sae.to(device)

    return random_sae


def find_feature_latents(model, sae, tokenizer, feature_letter, vocab_words, n_samples=100, device="cuda"):
    """Find SAE latents that are MOST SELECTIVE for words starting with the given letter."""
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

    top_k = min(100, len(selectivity))
    top_indices = np.argsort(selectivity)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if selectivity[idx] > 0.001:
            results.append((int(idx), float(selectivity[idx]), float(target_mean[idx]), float(other_mean[idx])))

    return results


def detect_absorption_chanin(model, sae, tokenizer, feature_letter, vocab_words, top_latents, n_samples=100, device="cuda"):
    """Detect absorption using Chanin et al. differential correlation metric."""
    if len(top_latents) < 2:
        return {"absorption_rate": 0.0, "n_latents": len(top_latents), "method": "differential_correlation"}

    hook_name = sae.cfg.metadata.get("hook_name", f"blocks.{LAYER}.hook_resid_pre")
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

    correlation_values = []
    for c in child_latents:
        child_acts = np.array(child_acts_list[c])
        active_mask = parent_acts > np.percentile(parent_acts, 50)
        if active_mask.sum() >= 3:
            r, p = stats.pearsonr(parent_acts[active_mask], child_acts[active_mask])
            correlation_values.append(max(0, r))
        else:
            correlation_values.append(0.0)

    absorption_rate = np.mean(correlation_values) if correlation_values else 0.0

    return {
        "absorption_rate": float(absorption_rate),
        "parent_latent": int(parent_latent),
        "n_child_latents": len(child_latents),
        "method": "differential_correlation",
    }


def run_h10_experiment():
    task_id = "h10_random_sae_baseline"
    results_dir = Path(__file__).parent.parent / "results" / "pilots"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=1, step=0, total_steps=len(FEATURES))

    print("=" * 70)
    print("H10: Random SAE Baseline Absorption")
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

    # Load trained SAE
    sae_id = f"blocks.{LAYER}.hook_resid_pre"
    print(f"\n[Loading] Loading trained SAE: {sae_id}...")
    trained_sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device,
    )
    print(f"  Trained SAE: {trained_sae.cfg.d_sae} latents")

    # Create random SAE
    print("\n[Creating] Random SAE baseline...")
    random_sae = create_random_sae(trained_sae, seed=SEED)
    print(f"  Random SAE: decoder frozen (orthonormal), encoder randomized")

    # Run absorption detection on TRAINED SAE
    print("\n" + "=" * 70)
    print("TRAINED SAE - Absorption Detection")
    print("=" * 70)
    trained_absorption = {}
    for i, feat in enumerate(FEATURES):
        top_latents = find_feature_latents(
            model, trained_sae, tokenizer, feat, VOCAB[feat],
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        absorption = detect_absorption_chanin(
            model, trained_sae, tokenizer, feat, VOCAB[feat], top_latents,
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        trained_absorption[feat] = absorption
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/26] '{feat}': rate={absorption['absorption_rate']:.3f}")

    # Run absorption detection on RANDOM SAE
    print("\n" + "=" * 70)
    print("RANDOM SAE - Absorption Detection")
    print("=" * 70)
    random_absorption = {}
    for i, feat in enumerate(FEATURES):
        top_latents = find_feature_latents(
            model, random_sae, tokenizer, feat, VOCAB[feat],
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        absorption = detect_absorption_chanin(
            model, random_sae, tokenizer, feat, VOCAB[feat], top_latents,
            n_samples=SAMPLES_PER_FEATURE, device=device
        )
        random_absorption[feat] = absorption
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/26] '{feat}': rate={absorption['absorption_rate']:.3f}")

        report_progress(task_id, results_dir, epoch=1, total_epochs=1, step=i+1, total_steps=len(FEATURES),
                        metric={"feature": feat, "trained_rate": trained_absorption[feat]["absorption_rate"],
                                "random_rate": absorption["absorption_rate"]})

    # Compare
    trained_rates = [trained_absorption[f]["absorption_rate"] for f in FEATURES]
    random_rates = [random_absorption[f]["absorption_rate"] for f in FEATURES]

    diff = np.array(trained_rates) - np.array(random_rates)

    # Statistical test
    t_stat, t_p = stats.ttest_rel(trained_rates, random_rates)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(trained_rates, random_rates)

    # Correlation
    r, r_p = stats.pearsonr(trained_rates, random_rates)

    print(f"\n{'=' * 70}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 70}")
    print(f"Trained SAE: mean={np.mean(trained_rates):.4f}, std={np.std(trained_rates):.4f}, max={max(trained_rates):.4f}")
    print(f"Random SAE:  mean={np.mean(random_rates):.4f}, std={np.std(random_rates):.4f}, max={max(random_rates):.4f}")
    print(f"Difference:  mean={np.mean(diff):.4f}, std={np.std(diff):.4f}")
    print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"Wilcoxon: W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}")
    print(f"Correlation: r={r:.3f}, p={r_p:.4f}")

    # Determine GO/NO-GO
    random_mean = np.mean(random_rates)
    if random_mean > 0.01:  # Non-zero absorption in random SAE
        go_no_go = "GO"
        confidence = min(0.5 + random_mean * 2, 0.95)
        notes = "Random SAE shows non-zero absorption, suggesting structural component"
    elif random_mean < 0.005:
        go_no_go = "NO_GO"
        confidence = 0.7
        notes = "Random SAE shows near-zero absorption; absorption appears learned"
    else:
        go_no_go = "UNCERTAIN"
        confidence = 0.5
        notes = "Marginal absorption in random SAE"

    output = {
        "task_id": task_id,
        "layer": LAYER,
        "n_features": len(FEATURES),
        "n_samples_per_feature": SAMPLES_PER_FEATURE,
        "seed": SEED,
        "trained_sae": {
            "absorption_by_feature": trained_absorption,
            "mean": float(np.mean(trained_rates)),
            "std": float(np.std(trained_rates)),
            "max": float(max(trained_rates)),
            "min": float(min(trained_rates)),
        },
        "random_sae": {
            "absorption_by_feature": random_absorption,
            "mean": float(np.mean(random_rates)),
            "std": float(np.std(random_rates)),
            "max": float(max(random_rates)),
            "min": float(min(random_rates)),
        },
        "comparison": {
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "wilcoxon_stat": float(wilcoxon_stat),
            "wilcoxon_p": float(wilcoxon_p),
            "pearson_r": float(r),
            "pearson_p": float(r_p),
        },
        "go_no_go": go_no_go,
        "confidence": float(confidence),
        "notes": notes,
    }

    output_path = results_dir / f"{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"GO/NO-GO: {go_no_go} (confidence={confidence:.2f})")
    print(f"Notes: {notes}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"H10: Trained={np.mean(trained_rates):.3f}, Random={np.mean(random_rates):.3f}, {go_no_go}")

    return output


if __name__ == "__main__":
    try:
        results = run_h10_experiment()
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent.parent / "results" / "pilots"
        mark_task_done("h10_random_sae_baseline", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
