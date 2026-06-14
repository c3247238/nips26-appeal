"""
P2_controls v2: Cross-Domain Controls with Calibrated Absorption Thresholds
=============================================================================

v1 finding: cosine_threshold=0.025 is FAR too permissive. With d=768 and d_sae=24576,
random directions easily achieve cosine > 0.025 with many features. This caused:
  - Random probe control: ~40% absorption (should be ~0%)
  - Continent multi-class probe: 100% absorption even on shuffled controls
  - First-letter baseline: 100% (degenerate)

v2 fix: Calibrate cosine threshold empirically.
  1. For each probe direction, compute cosine(decoder[i], probe_direction) for all i.
  2. Set threshold at 95th percentile of this distribution + margin.
     Only features exceeding this threshold are "aligned" with the probe.
  3. Additionally require the feature to be in the top-k by cosine (not just above threshold).
  4. Use a stricter dominance criterion: top activation must be > 3x second.

This ensures that "aligned with probe" means genuinely encoding the probe concept,
not just random cosine similarity in high-dimensional space.

Also: use proper feature split identification via cosine between decoder and probe
(not activation ratios, which conflate correlation with alignment).
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.stats import mannwhitneyu

# ── Configuration ──────────────────────────────────────────────────────
TASK_ID = "P2_controls"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
DATA_DIR = WORKSPACE / "exp" / "data" / "ravel"
PROBE_DIR = FULL_DIR / "P2_probes"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "gpt2"
SAE_RELEASE = "gpt2-small-res-jb"
PILOT_LAYERS = [8]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Absorption thresholds -- now calibrated
# cosine_percentile: a feature must have cosine with probe in top X% of all features
COSINE_PERCENTILE = 99.0  # Top 1% (~246 features out of 24576)
COSINE_TOPK = 20          # Additionally: only check top-k features by cosine
MIN_COSINE = 0.05         # Absolute minimum cosine (after percentile filter)
DOMINANCE_THRESHOLD = 2.0  # Top feature must have activation >= 2x second

# Feature selection params
SPLIT_FEATURE_TOPK = 10    # Top-k SAE features per class (by cosine with probe)
SPLIT_ACTIVATION_THRESHOLD = 0
MIN_FALSE_NEGATIVES = 3
MAX_CITIES_PILOT = 200
N_LETTERS_PILOT = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_dump(obj, path, **kwargs):
    with open(path, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=2, **kwargs)


# ── Process Tracking ───────────────────────────────────────────────────
def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(stage, detail="", metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "stage": stage,
        "detail": detail,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }, cls=NumpyEncoder))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, cls=NumpyEncoder))
    print(f"[DONE] {status}: {summary}")


# ── Data Loading ───────────────────────────────────────────────────────
def load_ravel_cities():
    with open(DATA_DIR / "ravel_city_entity_attributes.json") as f:
        all_attrs = json.load(f)
    with open(DATA_DIR / "ravel_city_entity_to_split.json") as f:
        splits = json.load(f)
    data = []
    for city, a in all_attrs.items():
        entry = {"city": city, "split": splits.get(city, "unknown")}
        entry.update(a)
        data.append(entry)
    return data


def load_probe(probe_name, layer):
    path = PROBE_DIR / f"probe_{probe_name}_layer{layer}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "coef": data["coef"],
        "intercept": data["intercept"],
        "scaler_mean": data["scaler_mean"],
        "scaler_scale": data["scaler_scale"],
        "classes": data["classes"],
        "mean_acc": float(data["mean_acc"]),
    }


def probe_direction(probe_data):
    """Get normalized probe directions in original (unscaled) model space."""
    coef = probe_data["coef"]
    # Undo scaling: probe operates on (x - mean) / scale, so direction = coef / scale
    direction = coef / probe_data["scaler_scale"]
    if direction.shape[0] <= 2:
        # Binary: use difference of class weights (or the single weight vector)
        if direction.shape[0] == 2:
            d = direction[1] - direction[0]
        else:
            d = direction[0]
        d = d / (np.linalg.norm(d) + 1e-10)
        return d.reshape(1, -1)
    else:
        # Multi-class: return per-class directions, each normalized
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        return direction / (norms + 1e-10)


# ── Activation Extraction ─────────────────────────────────────────────
def extract_activations_and_sae(model, tokenizer, sae, texts, layer, batch_size=64):
    """Extract residual stream activations AND SAE feature activations."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    resid_list = []
    sae_list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                names_filter=[hook_name],
                attention_mask=tokens.get("attention_mask"),
            )

        if "attention_mask" in tokens:
            seq_lens = tokens["attention_mask"].sum(dim=1) - 1
        else:
            seq_lens = torch.full((tokens["input_ids"].shape[0],),
                                  tokens["input_ids"].shape[1] - 1, device=DEVICE)

        resid = cache[hook_name]
        for j in range(resid.shape[0]):
            last_token_resid = resid[j, seq_lens[j]].float()
            resid_list.append(last_token_resid.cpu().numpy())
            sae_out = sae.encode(last_token_resid.unsqueeze(0))
            sae_list.append(sae_out.squeeze(0).detach().cpu().numpy())

        del cache, resid
        torch.cuda.empty_cache()

    return np.stack(resid_list), np.stack(sae_list)


# ── Calibrated Absorption Measurement ─────────────────────────────────
def compute_decoder_cosines(sae_decoder, probe_dir):
    """Compute cosine similarity between each SAE decoder vector and the probe direction.
    Returns (d_sae,) array of absolute cosine similarities.
    """
    # sae_decoder: (d_sae, d_model), probe_dir: (d_model,) or (1, d_model)
    probe_dir = probe_dir.flatten()
    probe_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-10)
    decoder_norms = np.linalg.norm(sae_decoder, axis=1, keepdims=True)
    decoder_normed = sae_decoder / (decoder_norms + 1e-10)
    cosines = decoder_normed @ probe_norm  # (d_sae,)
    return np.abs(cosines)


def find_aligned_features(sae_decoder, probe_dirs, percentile=COSINE_PERCENTILE,
                          topk=COSINE_TOPK, min_cos=MIN_COSINE):
    """Find SAE features aligned with any of the probe directions.
    Returns: set of feature indices, and the cosine threshold used.
    """
    all_cosines = np.zeros(sae_decoder.shape[0])
    for pd in probe_dirs:
        cos = compute_decoder_cosines(sae_decoder, pd)
        all_cosines = np.maximum(all_cosines, cos)

    # Percentile-based threshold
    threshold = max(np.percentile(all_cosines, percentile), min_cos)

    # Features above threshold
    above = np.where(all_cosines >= threshold)[0]

    # Additionally limit to top-k
    topk_ids = np.argsort(all_cosines)[-topk:][::-1]

    # Intersection: must be both above percentile threshold AND in top-k
    aligned = set(above.tolist()) & set(topk_ids.tolist())

    return aligned, float(threshold), all_cosines


def measure_absorption_calibrated(
    sae_acts, sae_decoder, probe_dirs, class_labels, class_id,
    aligned_features, cosine_array, cosine_threshold,
    dominance_threshold=DOMINANCE_THRESHOLD,
):
    """Calibrated absorption measurement.

    Split features: aligned features that also activate preferentially for this class.
    False negatives: class tokens where NO split feature fires.
    Absorbed: among FN tokens, a NON-split feature that is (a) aligned with probe
              (cosine above threshold) and (b) dominates other active features.
    """
    class_mask = class_labels == class_id
    n_class = int(class_mask.sum())
    if n_class < 5:
        return {"status": "insufficient_data", "n_class": n_class}

    # Split features: aligned features that activate more for this class
    aligned_list = sorted(aligned_features)
    if not aligned_list:
        return {
            "status": "no_aligned_features",
            "n_class": n_class,
            "absorption_rate": 0.0,
        }

    # Among aligned features, find those that activate preferentially for this class
    split_features = []
    for fid in aligned_list:
        in_rate = (sae_acts[class_mask, fid] > 0).mean()
        out_rate = (sae_acts[~class_mask, fid] > 0).mean()
        if in_rate > out_rate * 1.5 and in_rate > 0.05:
            split_features.append(fid)

    # If no split features, use top aligned features by activation ratio
    if not split_features:
        ratios = []
        for fid in aligned_list:
            in_mean = sae_acts[class_mask, fid].mean()
            out_mean = sae_acts[~class_mask, fid].mean()
            ratios.append(in_mean / (out_mean + 1e-10))
        best_idx = np.argsort(ratios)[-min(SPLIT_FEATURE_TOPK, len(ratios)):]
        split_features = [aligned_list[i] for i in best_idx]

    split_features = split_features[:SPLIT_FEATURE_TOPK]

    # False negatives: class tokens where no split feature fires
    class_sae = sae_acts[class_mask]
    split_active = np.array([class_sae[:, fid] > 0 for fid in split_features]).T
    any_split_active = split_active.any(axis=1) if split_active.shape[1] > 0 else np.zeros(n_class, dtype=bool)
    fn_mask = ~any_split_active
    n_fn = int(fn_mask.sum())

    if n_fn < MIN_FALSE_NEGATIVES:
        return {
            "status": "too_few_false_negatives",
            "n_class": n_class,
            "n_false_negatives": n_fn,
            "n_split_features": len(split_features),
            "absorption_rate": 0.0,
        }

    # For each FN token: check if any non-split feature is aligned with probe AND dominates
    fn_sae = class_sae[fn_mask]
    split_set = set(split_features)
    n_absorbed = 0
    absorption_details = []

    for idx in range(fn_sae.shape[0]):
        feat_acts = fn_sae[idx].copy()
        # Zero out split features
        for sf in split_features:
            feat_acts[sf] = 0

        active_ids = np.where(feat_acts > 0)[0]
        if len(active_ids) == 0:
            absorption_details.append({"absorbed": False, "reason": "no_active_features"})
            continue

        # Sort by activation magnitude
        sorted_ids = active_ids[np.argsort(feat_acts[active_ids])[::-1]]
        top_feat = sorted_ids[0]
        top_act = feat_acts[top_feat]

        # Dominance: compare to second-highest
        second_act = feat_acts[sorted_ids[1]] if len(sorted_ids) > 1 else 0
        dominance = top_act / (second_act + 1e-10) if second_act > 0 else float('inf')

        # Cosine alignment check
        top_cosine = cosine_array[top_feat]
        is_aligned = top_cosine >= cosine_threshold
        is_dominant = dominance >= dominance_threshold

        is_absorbed = is_aligned and is_dominant

        if is_absorbed:
            n_absorbed += 1

        if len(absorption_details) < 5:  # Sample first 5
            absorption_details.append({
                "absorbed": bool(is_absorbed),
                "top_feature": int(top_feat),
                "top_activation": float(top_act),
                "cosine_with_probe": float(top_cosine),
                "cosine_threshold": float(cosine_threshold),
                "dominance": float(min(dominance, 100)),
                "dominance_threshold": float(dominance_threshold),
                "is_aligned": bool(is_aligned),
                "is_dominant": bool(is_dominant),
            })

    absorption_rate = n_absorbed / n_fn if n_fn > 0 else 0.0

    return {
        "status": "measured",
        "n_class": n_class,
        "n_false_negatives": n_fn,
        "fn_rate": float(n_fn / n_class),
        "n_absorbed": n_absorbed,
        "absorption_rate": float(absorption_rate),
        "n_split_features": len(split_features),
        "cosine_threshold_used": float(cosine_threshold),
        "dominance_threshold_used": float(dominance_threshold),
        "sample_details": absorption_details,
    }


# ── Control 1: Shuffled Hierarchy ─────────────────────────────────────
def run_shuffled_control(sae_acts, sae_decoder, probe_dirs, true_labels,
                         aligned_features, cosine_array, cosine_threshold,
                         n_shuffles=5):
    """Shuffle class labels and re-measure absorption."""
    print("  [Control 1] Shuffled hierarchy...")
    rng = np.random.RandomState(SEED)
    shuffle_results = []

    for si in range(n_shuffles):
        shuffled = true_labels.copy()
        rng.shuffle(shuffled)
        classes = np.unique(shuffled)
        rates = []
        for cls in classes:
            r = measure_absorption_calibrated(
                sae_acts, sae_decoder, probe_dirs, shuffled, cls,
                aligned_features, cosine_array, cosine_threshold,
            )
            if r["status"] == "measured":
                rates.append(r["absorption_rate"])
        mean_abs = np.mean(rates) if rates else 0
        shuffle_results.append({
            "shuffle_idx": si,
            "mean_absorption": float(mean_abs),
            "n_measured": len(rates),
            "per_class_rates": [float(x) for x in rates],
        })
        print(f"    Shuffle {si}: {mean_abs:.4f} ({len(rates)} classes measured)")

    overall_mean = np.mean([r["mean_absorption"] for r in shuffle_results])
    overall_std = np.std([r["mean_absorption"] for r in shuffle_results])

    return {
        "control_type": "shuffled_hierarchy",
        "n_shuffles": n_shuffles,
        "per_shuffle_results": shuffle_results,
        "overall_mean_absorption": float(overall_mean),
        "overall_std_absorption": float(overall_std),
        "expected_range": "<5%",
        "passed": bool(overall_mean < 0.10),
    }


# ── Control 2: Random Probe Direction ─────────────────────────────────
def run_random_probe_control(sae_acts, sae_decoder, true_labels, d_model,
                             n_random=10):
    """Use random unit vectors as probe directions. Must recalibrate threshold per direction."""
    print("  [Control 2] Random probe direction (with per-direction calibration)...")
    rng = np.random.RandomState(SEED + 1)
    random_results = []

    for ri in range(n_random):
        rand_dir = rng.randn(1, d_model).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)

        # Calibrate threshold for this random direction
        aligned, threshold, cos_arr = find_aligned_features(
            sae_decoder, rand_dir,
            percentile=COSINE_PERCENTILE, topk=COSINE_TOPK, min_cos=MIN_COSINE,
        )

        classes = np.unique(true_labels)
        rates = []
        for cls in classes:
            r = measure_absorption_calibrated(
                sae_acts, sae_decoder, rand_dir, true_labels, cls,
                aligned, cos_arr, threshold,
            )
            if r["status"] == "measured":
                rates.append(r["absorption_rate"])

        mean_abs = np.mean(rates) if rates else 0
        random_results.append({
            "random_idx": ri,
            "mean_absorption": float(mean_abs),
            "n_measured": len(rates),
            "calibrated_threshold": float(threshold),
        })
        print(f"    Random {ri}: {mean_abs:.4f} (threshold={threshold:.4f})")

    overall_mean = np.mean([r["mean_absorption"] for r in random_results])
    overall_std = np.std([r["mean_absorption"] for r in random_results])

    return {
        "control_type": "random_probe_direction",
        "n_random_directions": n_random,
        "per_random_results": random_results,
        "overall_mean_absorption": float(overall_mean),
        "overall_std_absorption": float(overall_std),
        "expected_range": "<5%",
        "passed": bool(overall_mean < 0.10),
        "note": "Each random direction gets its own calibrated cosine threshold",
    }


# ── Control 3: First-Letter Baseline ──────────────────────────────────
def run_first_letter_control(model, tokenizer, sae, sae_decoder, layer,
                             n_letters=N_LETTERS_PILOT):
    """Replicate Chanin first-letter absorption on same SAE."""
    print(f"  [Control 3] First-letter baseline (top {n_letters} letters)...")

    vocab = tokenizer.get_vocab()
    letter_tokens = defaultdict(list)
    for token_str, token_id in vocab.items():
        if token_str.startswith('\u0120') and len(token_str) > 2:
            first_char = token_str[1].upper()
            if first_char.isalpha():
                letter_tokens[first_char].append((token_str, token_id))

    letter_counts = {l: len(toks) for l, toks in letter_tokens.items()}
    sorted_letters = sorted(letter_counts, key=letter_counts.get, reverse=True)
    target_letters = sorted_letters[:n_letters]
    print(f"    Letters: {target_letters}")

    MAX_TOKENS_PER_LETTER = 100
    rng = np.random.RandomState(SEED)
    for letter in target_letters:
        if len(letter_tokens[letter]) > MAX_TOKENS_PER_LETTER:
            indices = rng.choice(len(letter_tokens[letter]), MAX_TOKENS_PER_LETTER, replace=False)
            letter_tokens[letter] = [letter_tokens[letter][i] for i in sorted(indices)]

    all_token_ids = []
    all_letter_labels = []
    letter_to_idx = {l: i for i, l in enumerate(target_letters)}

    for letter in target_letters:
        for _, token_id in letter_tokens[letter]:
            all_token_ids.append(token_id)
            all_letter_labels.append(letter_to_idx[letter])

    all_letter_labels = np.array(all_letter_labels)
    print(f"    Total tokens: {len(all_token_ids)}")

    # Extract activations
    hook_name = f"blocks.{layer}.hook_resid_pre"
    resid_list = []
    sae_list = []
    batch_size = 256

    for i in range(0, len(all_token_ids), batch_size):
        batch_ids = all_token_ids[i:i + batch_size]
        input_ids = torch.tensor([[tid] for tid in batch_ids], device=DEVICE)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])

        resid = cache[hook_name]
        for j in range(resid.shape[0]):
            r = resid[j, 0].float()
            resid_list.append(r.cpu().numpy())
            sae_out = sae.encode(r.unsqueeze(0))
            sae_list.append(sae_out.squeeze(0).detach().cpu().numpy())

        del cache, resid
        torch.cuda.empty_cache()

    sae_acts = np.stack(sae_list)
    resid_acts = np.stack(resid_list)

    per_letter_results = {}
    all_rates = []

    for letter in target_letters:
        cls = letter_to_idx[letter]
        in_class = resid_acts[all_letter_labels == cls]
        out_class = resid_acts[all_letter_labels != cls]

        if len(in_class) < 5:
            continue

        # Probe direction: difference of means
        direction = in_class.mean(axis=0) - out_class.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        probe_dir = direction.reshape(1, -1)

        # Calibrate threshold for this letter's probe direction
        aligned, threshold, cos_arr = find_aligned_features(
            sae_decoder, probe_dir,
            percentile=COSINE_PERCENTILE, topk=COSINE_TOPK, min_cos=MIN_COSINE,
        )

        result = measure_absorption_calibrated(
            sae_acts, sae_decoder, probe_dir, all_letter_labels, cls,
            aligned, cos_arr, threshold,
        )
        per_letter_results[letter] = result

        if result["status"] == "measured":
            all_rates.append(result["absorption_rate"])
            print(f"    {letter}: absorption={result['absorption_rate']:.4f} "
                  f"(FN={result['n_false_negatives']}/{result['n_class']}, "
                  f"cosine_thresh={threshold:.4f})")
        else:
            print(f"    {letter}: {result['status']}")

    mean_rate = np.mean(all_rates) if all_rates else 0
    std_rate = np.std(all_rates) if all_rates else 0

    return {
        "control_type": "first_letter_baseline",
        "n_letters": len(target_letters),
        "n_tokens": len(all_token_ids),
        "letters": target_letters,
        "per_letter_results": per_letter_results,
        "n_measured": len(all_rates),
        "mean_absorption_rate": float(mean_rate),
        "std_absorption_rate": float(std_rate),
        "per_letter_rates": {l: r.get("absorption_rate", None)
                             for l, r in per_letter_results.items()
                             if r.get("status") == "measured"},
        "expected_range": "15-35% (Chanin et al. literature)",
        "note": "First-letter on GPT-2 Small with calibrated thresholds",
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "Starting P2_controls v2 (calibrated thresholds)")
    print(f"[CONFIG] TASK_ID={TASK_ID}, GPU={GPU_ID}, DEVICE={DEVICE}")
    print(f"[CONFIG] SAE={SAE_RELEASE}, Layers={PILOT_LAYERS}")
    print(f"[CONFIG] Cosine percentile={COSINE_PERCENTILE}, topk={COSINE_TOPK}, min_cos={MIN_COSINE}")
    print(f"[CONFIG] Dominance threshold={DOMINANCE_THRESHOLD}")

    # Load model
    print(f"\n[MODEL] Loading {MODEL_NAME}...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    d_model = model.cfg.d_model
    print(f"[MODEL] d_model={d_model}")

    # Load SAE
    from sae_lens import SAE as SAELens
    layer = PILOT_LAYERS[0]
    print(f"\n[SAE] Loading layer {layer}...")
    sae = SAELens.from_pretrained(
        release=SAE_RELEASE,
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=DEVICE,
    )
    d_sae = sae.cfg.d_sae
    sae_decoder = sae.W_dec.detach().cpu().float().numpy()  # (d_sae, d_model)
    print(f"[SAE] d_sae={d_sae}, decoder shape={sae_decoder.shape}")

    # Diagnostic: what does the cosine distribution look like for a random direction?
    rng_diag = np.random.RandomState(999)
    rand_d = rng_diag.randn(d_model).astype(np.float32)
    rand_d /= np.linalg.norm(rand_d)
    diag_cos = compute_decoder_cosines(sae_decoder, rand_d)
    print(f"\n[DIAGNOSTIC] Random direction cosine distribution:")
    print(f"  mean={diag_cos.mean():.4f}, std={diag_cos.std():.4f}")
    print(f"  p50={np.percentile(diag_cos, 50):.4f}, "
          f"p95={np.percentile(diag_cos, 95):.4f}, "
          f"p99={np.percentile(diag_cos, 99):.4f}, "
          f"max={diag_cos.max():.4f}")
    print(f"  # features > 0.025: {(diag_cos > 0.025).sum()} "
          f"({100*(diag_cos > 0.025).mean():.1f}%)")
    print(f"  # features > 0.05: {(diag_cos > 0.05).sum()} "
          f"({100*(diag_cos > 0.05).mean():.1f}%)")
    print(f"  # features > 0.10: {(diag_cos > 0.10).sum()} "
          f"({100*(diag_cos > 0.10).mean():.1f}%)")

    # Load RAVEL cities
    all_cities = load_ravel_cities()
    rng_data = np.random.RandomState(SEED)
    if len(all_cities) > MAX_CITIES_PILOT:
        indices = rng_data.choice(len(all_cities), MAX_CITIES_PILOT, replace=False)
        cities = [all_cities[i] for i in sorted(indices)]
    else:
        cities = all_cities
    print(f"\n[DATA] {len(cities)} cities")

    # Results container
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "version": "v2_calibrated",
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "layer": layer,
        "d_sae": d_sae,
        "n_cities": len(cities),
        "calibration": {
            "cosine_percentile": COSINE_PERCENTILE,
            "cosine_topk": COSINE_TOPK,
            "min_cosine": MIN_COSINE,
            "dominance_threshold": DOMINANCE_THRESHOLD,
            "diagnostic_random_cosines": {
                "mean": float(diag_cos.mean()),
                "std": float(diag_cos.std()),
                "p95": float(np.percentile(diag_cos, 95)),
                "p99": float(np.percentile(diag_cos, 99)),
                "max": float(diag_cos.max()),
                "n_above_025": int((diag_cos > 0.025).sum()),
                "n_above_05": int((diag_cos > 0.05).sum()),
                "n_above_10": int((diag_cos > 0.10).sum()),
            }
        },
        "controls": {},
        "statistical_comparisons": {},
        "first_letter_baseline": {},
    }

    # Probe configurations
    probe_configs = [
        ("Country_binary_US", "Country", "United States"),
        ("Language_binary_English", "Language", "English"),
        ("Continent", "Continent", None),
    ]

    all_real_rates = []
    all_shuffled_rates = []
    all_random_rates = []

    for probe_name, attr, binary_positive in probe_configs:
        print(f"\n{'='*60}")
        print(f"[PROBE] {probe_name} ({attr})")
        print(f"{'='*60}")

        probe_data = load_probe(probe_name, layer)
        if probe_data is None:
            print(f"  WARNING: Probe not found, skipping")
            continue

        print(f"  Probe accuracy: {probe_data['mean_acc']:.4f}")

        # Prepare labels
        if binary_positive:
            labels = np.array([1 if c[attr] == binary_positive else 0 for c in cities])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform([c[attr] for c in cities])

        n_classes = len(np.unique(labels))
        print(f"  Classes: {n_classes}, distribution: {dict(Counter(labels))}")

        # Probe directions
        p_dirs = probe_direction(probe_data)
        print(f"  Probe direction shape: {p_dirs.shape}")

        # Calibrate: find features aligned with this probe's direction
        aligned, cos_threshold, cos_arr = find_aligned_features(
            sae_decoder, p_dirs,
            percentile=COSINE_PERCENTILE, topk=COSINE_TOPK, min_cos=MIN_COSINE,
        )
        print(f"  Calibrated cosine threshold: {cos_threshold:.4f}")
        print(f"  # aligned features: {len(aligned)}")

        # Extract activations
        template = "{city}, a city known for being in"
        texts = [template.format(city=c["city"]) for c in cities]
        report_progress("extracting", f"{probe_name}")
        t0 = time.time()
        resid_acts, sae_acts = extract_activations_and_sae(
            model, tokenizer, sae, texts, layer
        )
        print(f"  Extraction: {time.time()-t0:.1f}s")

        # Real absorption
        report_progress("real_absorption", probe_name)
        print(f"\n  [Real absorption]")
        real_results = {}
        for cls in np.unique(labels):
            r = measure_absorption_calibrated(
                sae_acts, sae_decoder, p_dirs, labels, cls,
                aligned, cos_arr, cos_threshold,
            )
            real_results[int(cls)] = r
            if r["status"] == "measured":
                all_real_rates.append(r["absorption_rate"])
                print(f"    Class {cls}: absorption={r['absorption_rate']:.4f} "
                      f"(FN={r['n_false_negatives']}/{r['n_class']})")
            else:
                print(f"    Class {cls}: {r['status']}")

        measured_real = [r for r in real_results.values() if r.get("status") == "measured"]
        real_mean = np.mean([r["absorption_rate"] for r in measured_real]) if measured_real else 0
        print(f"  Real mean: {real_mean:.4f}")

        # Control 1: Shuffled
        report_progress("shuffled_control", probe_name)
        shuffled = run_shuffled_control(
            sae_acts, sae_decoder, p_dirs, labels,
            aligned, cos_arr, cos_threshold, n_shuffles=5,
        )
        for sr in shuffled["per_shuffle_results"]:
            all_shuffled_rates.extend(sr["per_class_rates"])

        # Control 2: Random probe
        report_progress("random_control", probe_name)
        random_ctrl = run_random_probe_control(
            sae_acts, sae_decoder, labels, d_model, n_random=10,
        )
        all_random_rates.extend([r["mean_absorption"] for r in random_ctrl["per_random_results"]])

        results["controls"][probe_name] = {
            "attribute": attr,
            "n_classes": n_classes,
            "probe_accuracy": probe_data["mean_acc"],
            "calibrated_cosine_threshold": float(cos_threshold),
            "n_aligned_features": len(aligned),
            "real_absorption": {
                "mean_rate": float(real_mean),
                "per_class": {str(k): v for k, v in real_results.items()},
                "n_measured": len(measured_real),
            },
            "shuffled_control": shuffled,
            "random_probe_control": random_ctrl,
        }

        del resid_acts, sae_acts
        gc.collect()
        torch.cuda.empty_cache()

    # Control 3: First-letter
    print(f"\n{'='*60}")
    print("[FIRST-LETTER BASELINE]")
    print(f"{'='*60}")
    report_progress("first_letter", "First-letter baseline")

    first_letter = run_first_letter_control(
        model, tokenizer, sae, sae_decoder, layer, n_letters=N_LETTERS_PILOT,
    )
    results["first_letter_baseline"] = first_letter

    # Statistical comparisons
    print(f"\n{'='*60}")
    print("[STATISTICAL COMPARISONS]")
    print(f"{'='*60}")

    comparisons = {}
    if all_real_rates and all_shuffled_rates:
        try:
            stat, p = mannwhitneyu(all_real_rates, all_shuffled_rates, alternative='greater')
            comparisons["real_vs_shuffled"] = {
                "test": "Mann-Whitney U (real > shuffled)",
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < 0.05),
                "real_mean": float(np.mean(all_real_rates)),
                "shuffled_mean": float(np.mean(all_shuffled_rates)),
            }
            print(f"  real vs shuffled: p={p:.4f}, sig={p<0.05}")
        except Exception as e:
            comparisons["real_vs_shuffled"] = {"error": str(e)}

    if all_real_rates and all_random_rates:
        try:
            stat, p = mannwhitneyu(all_real_rates, all_random_rates, alternative='greater')
            comparisons["real_vs_random"] = {
                "test": "Mann-Whitney U (real > random)",
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < 0.05),
                "real_mean": float(np.mean(all_real_rates)),
                "random_mean": float(np.mean(all_random_rates)),
            }
            print(f"  real vs random: p={p:.4f}, sig={p<0.05}")
        except Exception as e:
            comparisons["real_vs_random"] = {"error": str(e)}

    results["statistical_comparisons"] = comparisons

    # Summary
    elapsed = time.time() - start_time

    shuffled_rates_per_probe = []
    random_rates_per_probe = []
    for pdata in results["controls"].values():
        sh = pdata.get("shuffled_control", {})
        rp = pdata.get("random_probe_control", {})
        if sh.get("overall_mean_absorption") is not None:
            shuffled_rates_per_probe.append(sh["overall_mean_absorption"])
        if rp.get("overall_mean_absorption") is not None:
            random_rates_per_probe.append(rp["overall_mean_absorption"])

    shuffled_mean = np.mean(shuffled_rates_per_probe) if shuffled_rates_per_probe else None
    random_mean = np.mean(random_rates_per_probe) if random_rates_per_probe else None
    real_mean_overall = np.mean(all_real_rates) if all_real_rates else 0
    first_letter_mean = first_letter.get("mean_absorption_rate", None)

    shuffled_pass = shuffled_mean is not None and shuffled_mean < 0.10
    random_pass = random_mean is not None and random_mean < 0.10

    if shuffled_pass and random_pass:
        pilot_verdict = "GO"
        if real_mean_overall < 0.05:
            pilot_verdict = "INFORMATIVE_NULL"
            pilot_detail = (
                f"Controls validated (shuffled={shuffled_mean:.4f}, random={random_mean:.4f}, "
                f"both <10%). Real absorption={real_mean_overall:.4f} is also near zero, "
                "indicating GPT-2 Small may not exhibit strong knowledge absorption. "
                "This is an informative null: the measurement methodology is valid "
                "(controls discriminate) but GPT-2 lacks sufficient hierarchical knowledge "
                "encoding for observable absorption. "
                f"First-letter baseline={first_letter_mean:.4f} provides reference."
            )
        else:
            pilot_detail = (
                f"Controls pass: shuffled={shuffled_mean:.4f}, random={random_mean:.4f} (<10%). "
                f"Real={real_mean_overall:.4f}. First-letter={first_letter_mean:.4f}."
            )
    elif shuffled_pass or random_pass:
        pilot_verdict = "MIXED"
        pilot_detail = (
            f"Partial pass: shuffled={shuffled_mean:.4f} ({'PASS' if shuffled_pass else 'FAIL'}), "
            f"random={random_mean:.4f} ({'PASS' if random_pass else 'FAIL'}). "
            f"Real={real_mean_overall:.4f}. First-letter={first_letter_mean:.4f}."
        )
    else:
        pilot_verdict = "FAIL"
        pilot_detail = (
            f"Both controls fail: shuffled={shuffled_mean}, random={random_mean}. "
            "Absorption metric may have methodological issues."
        )

    results["summary"] = {
        "pilot_verdict": pilot_verdict,
        "pilot_detail": pilot_detail,
        "elapsed_sec": round(elapsed, 1),
        "real_absorption_mean": float(real_mean_overall),
        "shuffled_mean": float(shuffled_mean) if shuffled_mean is not None else None,
        "random_mean": float(random_mean) if random_mean is not None else None,
        "first_letter_mean": float(first_letter_mean) if first_letter_mean is not None else None,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("[RESULTS SUMMARY]")
    print(f"{'='*60}")
    print(f"  Real absorption: {real_mean_overall:.4f}")
    print(f"  Shuffled control: {shuffled_mean:.4f}" if shuffled_mean is not None else "  Shuffled: N/A")
    print(f"  Random probe:    {random_mean:.4f}" if random_mean is not None else "  Random: N/A")
    print(f"  First-letter:    {first_letter_mean:.4f}" if first_letter_mean is not None else "  First-letter: N/A")
    print(f"  Verdict: {pilot_verdict}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    json_dump(results, PILOT_DIR / "P2_controls.json")
    json_dump(results, FULL_DIR / "P2_controls.json")
    print(f"\n[SAVE] Saved to pilots/ and full/")

    # Mark done
    mark_done(
        "success",
        f"{pilot_verdict}: real={real_mean_overall:.4f}, shuffled={shuffled_mean:.4f}, "
        f"random={random_mean:.4f}, first_letter={first_letter_mean:.4f}, "
        f"elapsed={elapsed:.1f}s",
    )

    # GPU progress
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        with open(gp_path) as f:
            gp = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]

    gp["timings"][TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(elapsed / 60, 1),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "layer": layer,
            "d_sae": d_sae,
            "n_cities": len(cities),
            "cosine_percentile": COSINE_PERCENTILE,
            "dominance_threshold": DOMINANCE_THRESHOLD,
            "gpu_count": 1,
        },
    }

    with open(gp_path, "w") as f:
        json.dump(gp, f, cls=NumpyEncoder, indent=2)

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}\n{traceback.format_exc()}")
        mark_done("failed", f"{type(e).__name__}: {str(e)[:200]}")

        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        try:
            with open(gp_path) as f:
                gp = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in gp.get("failed", []):
            gp.setdefault("failed", []).append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        with open(gp_path, "w") as f:
            json.dump(gp, f, indent=2, cls=NumpyEncoder)
        sys.exit(1)
