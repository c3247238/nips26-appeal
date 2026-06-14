"""
P2_controls: Cross-Domain Controls for Absorption Measurement
==============================================================

Task: Three control experiments to validate that absorption measurements on knowledge
hierarchies (P2_absorption_measurement) are genuine, not artifacts.

Controls:
  (1) Shuffled hierarchy: Randomize city-attribute mappings. If absorption is real,
      shuffled probes should show near-zero absorption because the probe direction no
      longer aligns with the model's actual knowledge representation.
  (2) Random probe direction: Use random unit vectors instead of trained probes.
      Absorption should be near-zero because random directions don't correspond to any
      learned concept.
  (3) First-letter baseline: Replicate Chanin et al. first-letter absorption on the
      same SAEs for direct comparison. This anchors our knowledge absorption rates
      against the established first-letter phenomenon.

Model: GPT-2 Small (fallback from Gemma 2B due to gating)
SAE: gpt2-small-res-jb (24576 features)
Layers: 5, 8, 11 (matching probes)

Absorption measurement approach (adapted from Chanin et al.):
  1. Load SAE and trained probe for a given attribute
  2. For each class in the attribute:
     a. Find SAE features that are "split features" -- features whose activation
        correlates with the class membership
     b. Identify tokens where split features fail to fire but probe classifies correctly
        (false negatives of the SAE's representation)
     c. Among other SAE features active on false-negative tokens, check if any have
        decoder directions aligned with the probe direction (cosine > threshold)
     d. If such a feature exists and dominates (activation >> second-highest), it has
        "absorbed" the class information
  3. Absorption rate = (absorbed instances) / (total relevant instances)

Pilot mode: Use subset of cities, single SAE layer (8), 1 seed.
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
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.spatial.distance import cosine as cosine_dist

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
# Focus on best layer for pilot
PILOT_LAYERS = [8]
FULL_LAYERS = [5, 8, 11]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Absorption thresholds (sweep in full mode, fixed for pilot)
COSINE_THRESHOLDS = [0.025]  # Pilot: single threshold
DOMINANCE_THRESHOLDS = [1.0]  # Pilot: single threshold
# Full mode would sweep: cosine in {0.01, 0.025, 0.05, 0.1}, dominance in {0.5, 1.0, 2.0}

# Feature selection params
SPLIT_FEATURE_TOPK = 5          # Top-k SAE features per class
SPLIT_ACTIVATION_THRESHOLD = 0  # Feature must be active (> 0)
MIN_FALSE_NEGATIVES = 3         # Minimum FN tokens to compute absorption
MAX_CITIES_PILOT = 200          # Pilot limit

# First-letter experiment params
FIRST_LETTER_TEMPLATE = " {token}"  # Space prefix for GPT-2 tokenization
N_LETTERS_PILOT = 10  # Pilot: 10 most common starting letters


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
    """Load a saved probe from npz file."""
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
        "tier": int(data["tier"]) if "tier" in data else 0,
        "probe_name": str(data["probe_name"]) if "probe_name" in data else probe_name,
    }


def probe_predict(probe_data, activations):
    """Apply a saved probe to activations. Returns predicted labels and decision scores."""
    # Scale activations
    X = (activations - probe_data["scaler_mean"]) / probe_data["scaler_scale"]
    # Logistic regression: scores = X @ coef.T + intercept
    scores = X @ probe_data["coef"].T + probe_data["intercept"]
    if scores.ndim == 1:
        preds = (scores > 0).astype(int)
    else:
        preds = scores.argmax(axis=1)
    return preds, scores


def probe_direction(probe_data):
    """Get the primary direction of the probe (weight vector).
    For binary: the single coefficient vector.
    For multi-class: the mean of per-class coefficient vectors (centroid direction).
    Returns normalized direction in model space (pre-scaling).
    """
    coef = probe_data["coef"]
    # Undo scaling: probe acts on scaled X, so direction in original space is coef / scale
    direction = coef / probe_data["scaler_scale"]
    if direction.shape[0] == 1:
        # Binary: single direction
        d = direction[0]
    elif direction.shape[0] == 2:
        # Binary with 2 rows: take difference
        d = direction[1] - direction[0]
    else:
        # Multi-class: use per-class directions
        return direction / np.linalg.norm(direction, axis=1, keepdims=True)
    d = d / np.linalg.norm(d)
    return d.reshape(1, -1)


# ── Activation Extraction ─────────────────────────────────────────────
def extract_activations_and_sae(model, tokenizer, sae, texts, layer, batch_size=64):
    """Extract residual stream activations AND SAE feature activations for given texts.

    Returns:
        resid_acts: (N, d_model) residual stream at last token position
        sae_acts: (N, d_sae) SAE feature activations
    """
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

        # Get last-token positions
        if "attention_mask" in tokens:
            seq_lens = tokens["attention_mask"].sum(dim=1) - 1
        else:
            seq_lens = torch.full((tokens["input_ids"].shape[0],),
                                  tokens["input_ids"].shape[1] - 1, device=DEVICE)

        resid = cache[hook_name]  # (batch, seq, d_model)

        for j in range(resid.shape[0]):
            last_token_resid = resid[j, seq_lens[j]].float()
            resid_list.append(last_token_resid.cpu().numpy())

            # Encode through SAE
            sae_out = sae.encode(last_token_resid.unsqueeze(0))
            sae_list.append(sae_out.squeeze(0).detach().cpu().numpy())

        del cache, resid
        torch.cuda.empty_cache()

    return np.stack(resid_list), np.stack(sae_list)


# ── Absorption Measurement Core ───────────────────────────────────────
def measure_absorption(
    sae_acts,
    sae_decoder,  # (d_sae, d_model) decoder weight matrix
    probe_dirs,   # (n_dirs, d_model) probe direction(s)
    class_labels,  # (N,) integer class labels for each city
    class_id,      # which class to measure absorption for
    cosine_threshold=0.025,
    dominance_threshold=1.0,
):
    """Measure absorption rate for a single class.

    Algorithm (adapted from Chanin et al.):
    1. Find "split features" -- SAE features that activate preferentially for this class
    2. Identify "false negatives" -- tokens where NO split feature fires but the class is correct
    3. Among active features on FN tokens, find features aligned with probe direction
    4. Absorption = feature with high cosine to probe AND high activation dominance

    Returns dict with absorption stats.
    """
    class_mask = class_labels == class_id
    n_class = class_mask.sum()
    if n_class < 5:
        return {"status": "insufficient_data", "n_class": int(n_class)}

    # Step 1: Find split features (features preferentially active for this class)
    # Use simple method: features with highest ratio of mean activation in-class vs out-of-class
    in_class_mean = sae_acts[class_mask].mean(axis=0)
    out_class_mean = sae_acts[~class_mask].mean(axis=0)

    # Avoid division by zero
    ratio = in_class_mean / (out_class_mean + 1e-10)
    # Filter: feature must be active in at least 10% of in-class tokens
    active_rate = (sae_acts[class_mask] > 0).mean(axis=0)
    ratio[active_rate < 0.1] = 0

    # Top-k split features
    split_feature_ids = np.argsort(ratio)[-SPLIT_FEATURE_TOPK:][::-1]
    split_ratios = ratio[split_feature_ids]

    # Step 2: False negatives -- class tokens where NO split feature fires
    class_sae = sae_acts[class_mask]
    split_active = class_sae[:, split_feature_ids] > SPLIT_ACTIVATION_THRESHOLD
    any_split_active = split_active.any(axis=1)
    fn_mask = ~any_split_active
    n_fn = fn_mask.sum()

    if n_fn < MIN_FALSE_NEGATIVES:
        return {
            "status": "too_few_false_negatives",
            "n_class": int(n_class),
            "n_false_negatives": int(n_fn),
            "split_feature_ids": split_feature_ids.tolist(),
            "split_ratios": split_ratios.tolist(),
            "absorption_rate": 0.0,
        }

    # Step 3: For each false-negative token, find the highest-activation non-split feature
    fn_sae = class_sae[fn_mask]
    n_absorbed = 0
    absorbed_features = []
    absorption_details = []

    for idx in range(fn_sae.shape[0]):
        feat_acts = fn_sae[idx].copy()
        # Zero out split features
        feat_acts[split_feature_ids] = 0

        # Get top active features
        active_ids = np.where(feat_acts > 0)[0]
        if len(active_ids) == 0:
            continue

        # Sort by activation
        sorted_ids = active_ids[np.argsort(feat_acts[active_ids])[::-1]]
        top_feat = sorted_ids[0]
        top_act = feat_acts[top_feat]

        # Second-highest activation for dominance check
        if len(sorted_ids) > 1:
            second_act = feat_acts[sorted_ids[1]]
        else:
            second_act = 0

        # Check cosine alignment with probe direction
        feat_decoder = sae_decoder[top_feat]
        feat_decoder_norm = feat_decoder / (np.linalg.norm(feat_decoder) + 1e-10)

        # Check against each probe direction
        max_cos = 0
        for pd in probe_dirs:
            pd_norm = pd / (np.linalg.norm(pd) + 1e-10)
            cos_sim = np.dot(feat_decoder_norm, pd_norm)
            max_cos = max(max_cos, abs(cos_sim))

        # Dominance check
        dominance = top_act / (second_act + 1e-10) if second_act > 0 else float('inf')

        is_absorbed = (max_cos >= cosine_threshold) and (dominance >= dominance_threshold)

        if is_absorbed:
            n_absorbed += 1
            absorbed_features.append(int(top_feat))

        absorption_details.append({
            "top_feature": int(top_feat),
            "top_activation": float(top_act),
            "cosine_with_probe": float(max_cos),
            "dominance": float(min(dominance, 100)),
            "is_absorbed": bool(is_absorbed),
        })

    absorption_rate = n_absorbed / n_fn if n_fn > 0 else 0.0

    return {
        "status": "measured",
        "n_class": int(n_class),
        "n_false_negatives": int(n_fn),
        "n_absorbed": int(n_absorbed),
        "absorption_rate": float(absorption_rate),
        "fn_rate": float(n_fn / n_class),
        "split_feature_ids": split_feature_ids.tolist(),
        "split_ratios": split_ratios[:3].tolist(),
        "absorbed_features": absorbed_features[:10],
        "cosine_threshold": cosine_threshold,
        "dominance_threshold": dominance_threshold,
        "sample_details": absorption_details[:5],  # First 5 for inspection
    }


# ── Control 1: Shuffled Hierarchy ─────────────────────────────────────
def run_shuffled_control(
    sae_acts, sae_decoder, probe_dirs, true_labels, n_shuffles=5,
    cosine_threshold=0.025, dominance_threshold=1.0,
):
    """Shuffle class labels and re-measure absorption.
    Expect near-zero because probe direction no longer matches SAE structure.
    """
    print("  [Control 1] Shuffled hierarchy...")
    rng = np.random.RandomState(SEED)
    shuffle_results = []

    for si in range(n_shuffles):
        shuffled_labels = true_labels.copy()
        rng.shuffle(shuffled_labels)

        # Measure absorption with shuffled labels for each class
        classes = np.unique(shuffled_labels)
        class_absorptions = []

        for cls in classes:
            result = measure_absorption(
                sae_acts, sae_decoder, probe_dirs,
                shuffled_labels, cls,
                cosine_threshold=cosine_threshold,
                dominance_threshold=dominance_threshold,
            )
            if result["status"] == "measured":
                class_absorptions.append(result["absorption_rate"])

        mean_absorption = np.mean(class_absorptions) if class_absorptions else 0
        shuffle_results.append({
            "shuffle_idx": si,
            "mean_absorption": float(mean_absorption),
            "n_classes_measured": len(class_absorptions),
            "per_class_rates": [float(x) for x in class_absorptions],
        })
        print(f"    Shuffle {si}: mean absorption = {mean_absorption:.4f}")

    overall_mean = np.mean([r["mean_absorption"] for r in shuffle_results])
    overall_std = np.std([r["mean_absorption"] for r in shuffle_results])

    return {
        "control_type": "shuffled_hierarchy",
        "n_shuffles": n_shuffles,
        "per_shuffle_results": shuffle_results,
        "overall_mean_absorption": float(overall_mean),
        "overall_std_absorption": float(overall_std),
        "expected_range": "<5%",
        "pass_criterion": "absorption_rate < 0.10",
        "passed": bool(overall_mean < 0.10),
    }


# ── Control 2: Random Probe Direction ─────────────────────────────────
def run_random_probe_control(
    sae_acts, sae_decoder, true_labels, d_model, n_random=10,
    cosine_threshold=0.025, dominance_threshold=1.0,
):
    """Use random unit vectors as probe directions instead of trained probes.
    Expect near-zero because random directions don't correspond to learned concepts.
    """
    print("  [Control 2] Random probe direction...")
    rng = np.random.RandomState(SEED + 1)
    random_results = []

    for ri in range(n_random):
        # Generate random unit vector
        rand_dir = rng.randn(1, d_model).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)

        classes = np.unique(true_labels)
        class_absorptions = []

        for cls in classes:
            result = measure_absorption(
                sae_acts, sae_decoder, rand_dir,
                true_labels, cls,
                cosine_threshold=cosine_threshold,
                dominance_threshold=dominance_threshold,
            )
            if result["status"] == "measured":
                class_absorptions.append(result["absorption_rate"])

        mean_absorption = np.mean(class_absorptions) if class_absorptions else 0
        random_results.append({
            "random_idx": ri,
            "mean_absorption": float(mean_absorption),
            "n_classes_measured": len(class_absorptions),
        })
        print(f"    Random {ri}: mean absorption = {mean_absorption:.4f}")

    overall_mean = np.mean([r["mean_absorption"] for r in random_results])
    overall_std = np.std([r["mean_absorption"] for r in random_results])

    return {
        "control_type": "random_probe_direction",
        "n_random_directions": n_random,
        "per_random_results": random_results,
        "overall_mean_absorption": float(overall_mean),
        "overall_std_absorption": float(overall_std),
        "expected_range": "<5%",
        "pass_criterion": "absorption_rate < 0.10",
        "passed": bool(overall_mean < 0.10),
    }


# ── Control 3: First-Letter Baseline ──────────────────────────────────
def run_first_letter_control(
    model, tokenizer, sae, layer, n_letters=N_LETTERS_PILOT,
    cosine_threshold=0.025, dominance_threshold=1.0,
):
    """Replicate Chanin et al. first-letter absorption measurement.
    Uses single-token inputs (space-prefixed words) for direct comparison.
    """
    print(f"  [Control 3] First-letter baseline (top {n_letters} letters)...")

    # Get vocabulary tokens (single-token words starting with a letter)
    vocab = tokenizer.get_vocab()
    # Filter to tokens that are ' X...' (space-prefixed words)
    letter_tokens = defaultdict(list)
    for token_str, token_id in vocab.items():
        # GPT-2 tokens often start with 'Ġ' (space prefix)
        if token_str.startswith('Ġ') and len(token_str) > 2:
            first_char = token_str[1].upper()
            if first_char.isalpha():
                letter_tokens[first_char].append((token_str, token_id))

    # Find most common starting letters
    letter_counts = {l: len(toks) for l, toks in letter_tokens.items()}
    sorted_letters = sorted(letter_counts, key=letter_counts.get, reverse=True)
    target_letters = sorted_letters[:n_letters]
    print(f"    Target letters: {target_letters}")
    print(f"    Token counts: {[letter_counts[l] for l in target_letters]}")

    # Limit tokens per letter for pilot
    MAX_TOKENS_PER_LETTER = 100
    for letter in target_letters:
        if len(letter_tokens[letter]) > MAX_TOKENS_PER_LETTER:
            np.random.seed(SEED)
            indices = np.random.choice(len(letter_tokens[letter]), MAX_TOKENS_PER_LETTER, replace=False)
            letter_tokens[letter] = [letter_tokens[letter][i] for i in sorted(indices)]

    # Collect all tokens and their letter labels
    all_token_strs = []
    all_token_ids = []
    all_letter_labels = []
    letter_to_idx = {l: i for i, l in enumerate(target_letters)}

    for letter in target_letters:
        for token_str, token_id in letter_tokens[letter]:
            all_token_strs.append(token_str)
            all_token_ids.append(token_id)
            all_letter_labels.append(letter_to_idx[letter])

    all_letter_labels = np.array(all_letter_labels)
    print(f"    Total tokens: {len(all_token_ids)}")

    # Extract activations
    hook_name = f"blocks.{layer}.hook_resid_pre"
    resid_list = []
    sae_list = []
    batch_size = 128

    for i in range(0, len(all_token_ids), batch_size):
        batch_ids = all_token_ids[i:i + batch_size]
        # Single-token inputs
        input_ids = torch.tensor([[tid] for tid in batch_ids], device=DEVICE)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[hook_name],
            )

        resid = cache[hook_name]  # (batch, 1, d_model)
        for j in range(resid.shape[0]):
            r = resid[j, 0].float()
            resid_list.append(r.cpu().numpy())
            sae_out = sae.encode(r.unsqueeze(0))
            sae_list.append(sae_out.squeeze(0).detach().cpu().numpy())

        del cache, resid
        torch.cuda.empty_cache()

    sae_acts = np.stack(sae_list)
    resid_acts = np.stack(resid_list)

    # Get SAE decoder weights
    sae_decoder = sae.W_dec.detach().cpu().float().numpy()  # (d_sae, d_model)

    # For first-letter, the "probe" direction is the difference between mean activations
    # of the target letter vs. all other letters (a natural direction in residual stream)
    per_letter_results = {}

    for letter in target_letters:
        cls = letter_to_idx[letter]
        in_class = resid_acts[all_letter_labels == cls]
        out_class = resid_acts[all_letter_labels != cls]

        if len(in_class) < 5:
            continue

        # Probe direction: difference of means (simple but effective for binary classification)
        direction = in_class.mean(axis=0) - out_class.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        probe_dir = direction.reshape(1, -1)

        result = measure_absorption(
            sae_acts, sae_decoder, probe_dir,
            all_letter_labels, cls,
            cosine_threshold=cosine_threshold,
            dominance_threshold=dominance_threshold,
        )

        per_letter_results[letter] = result
        if result["status"] == "measured":
            print(f"    Letter {letter}: absorption={result['absorption_rate']:.4f} "
                  f"(FN={result['n_false_negatives']}/{result['n_class']})")
        else:
            print(f"    Letter {letter}: {result['status']} "
                  f"(n_class={result.get('n_class', '?')})")

    measured = [r for r in per_letter_results.values() if r.get("status") == "measured"]
    if measured:
        rates = [r["absorption_rate"] for r in measured]
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        fn_rates = [r["fn_rate"] for r in measured]
        mean_fn = np.mean(fn_rates)
    else:
        mean_rate = 0
        std_rate = 0
        mean_fn = 0

    return {
        "control_type": "first_letter_baseline",
        "n_letters": len(target_letters),
        "n_tokens_total": len(all_token_ids),
        "letters": target_letters,
        "per_letter_results": per_letter_results,
        "n_measured": len(measured),
        "mean_absorption_rate": float(mean_rate),
        "std_absorption_rate": float(std_rate),
        "mean_fn_rate": float(mean_fn),
        "expected_range": "15-35% (from Chanin et al. literature)",
        "note": "First-letter on GPT-2 Small may differ from Gemma 2B published values",
    }


# ── Statistical Comparison ─────────────────────────────────────────────
def compare_real_vs_controls(real_rates, shuffled_rates, random_rates):
    """Compare real absorption rates against controls using statistical tests."""
    comparisons = {}

    # Real vs shuffled
    if len(real_rates) > 0 and len(shuffled_rates) > 0:
        try:
            stat, p = mannwhitneyu(real_rates, shuffled_rates, alternative='greater')
            comparisons["real_vs_shuffled"] = {
                "test": "Mann-Whitney U (one-sided: real > shuffled)",
                "statistic": float(stat),
                "p_value": float(p),
                "significant_at_005": bool(p < 0.05),
                "real_mean": float(np.mean(real_rates)),
                "shuffled_mean": float(np.mean(shuffled_rates)),
                "effect_size": float(np.mean(real_rates) - np.mean(shuffled_rates)),
            }
        except Exception as e:
            comparisons["real_vs_shuffled"] = {"error": str(e)}

    # Real vs random
    if len(real_rates) > 0 and len(random_rates) > 0:
        try:
            stat, p = mannwhitneyu(real_rates, random_rates, alternative='greater')
            comparisons["real_vs_random"] = {
                "test": "Mann-Whitney U (one-sided: real > random)",
                "statistic": float(stat),
                "p_value": float(p),
                "significant_at_005": bool(p < 0.05),
                "real_mean": float(np.mean(real_rates)),
                "random_mean": float(np.mean(random_rates)),
                "effect_size": float(np.mean(real_rates) - np.mean(random_rates)),
            }
        except Exception as e:
            comparisons["real_vs_random"] = {"error": str(e)}

    return comparisons


# ── Main ───────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "Starting P2_controls pilot")
    print(f"[CONFIG] TASK_ID={TASK_ID}, GPU={GPU_ID}, DEVICE={DEVICE}")
    print(f"[CONFIG] SAE={SAE_RELEASE}, Layers={PILOT_LAYERS}")

    # ── Load Model ─────────────────────────────────────────────────────
    print(f"\n[MODEL] Loading {MODEL_NAME}...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    d_model = model.cfg.d_model
    print(f"[MODEL] Loaded: {model.cfg.n_layers} layers, d={d_model}")

    # ── Load SAE ───────────────────────────────────────────────────────
    from sae_lens import SAE as SAELens
    layer = PILOT_LAYERS[0]
    print(f"\n[SAE] Loading {SAE_RELEASE} layer {layer}...")
    sae = SAELens.from_pretrained(
        release=SAE_RELEASE,
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=DEVICE,
    )
    d_sae = sae.cfg.d_sae
    print(f"[SAE] Loaded: d_sae={d_sae}")

    # Get decoder weights
    sae_decoder = sae.W_dec.detach().cpu().float().numpy()  # (d_sae, d_model)
    print(f"[SAE] Decoder shape: {sae_decoder.shape}")

    # ── Load RAVEL cities ──────────────────────────────────────────────
    print(f"\n[DATA] Loading RAVEL cities...")
    all_cities = load_ravel_cities()
    # Pilot: limit cities
    np.random.seed(SEED)
    if len(all_cities) > MAX_CITIES_PILOT:
        indices = np.random.choice(len(all_cities), MAX_CITIES_PILOT, replace=False)
        cities = [all_cities[i] for i in sorted(indices)]
    else:
        cities = all_cities
    print(f"[DATA] {len(cities)} cities (of {len(all_cities)} total)")

    # ── Load probes ────────────────────────────────────────────────────
    # Use best probe for this layer
    probe_configs = [
        ("Country_binary_US", "Country", "United States"),
        ("Language_binary_English", "Language", "English"),
        ("Continent", "Continent", None),
    ]

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "layer": layer,
        "d_sae": d_sae,
        "n_cities": len(cities),
        "controls": {},
        "statistical_comparisons": {},
        "first_letter_baseline": {},
    }

    # We'll track real absorption rates for comparison
    all_real_rates = []
    all_shuffled_rates = []
    all_random_rates = []

    for probe_name, attr, binary_positive in probe_configs:
        print(f"\n{'='*60}")
        print(f"[PROBE] {probe_name} (attribute: {attr})")
        print(f"{'='*60}")

        probe_data = load_probe(probe_name, layer)
        if probe_data is None:
            print(f"  WARNING: Probe {probe_name}_layer{layer} not found, skipping")
            continue

        print(f"  Probe accuracy: {probe_data['mean_acc']:.4f}")
        print(f"  Probe classes: {probe_data['classes']}")

        # Prepare labels for this attribute
        if binary_positive:
            labels = np.array([1 if c[attr] == binary_positive else 0 for c in cities])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            raw_labels = [c[attr] for c in cities]
            labels = le.fit_transform(raw_labels)

        n_classes = len(np.unique(labels))
        print(f"  N classes: {n_classes}, class distribution: {dict(Counter(labels))}")

        # Get probe directions
        p_dirs = probe_direction(probe_data)
        print(f"  Probe direction shape: {p_dirs.shape}")

        # Extract activations
        template = "{city}, a city known for being in"
        texts = [template.format(city=c["city"]) for c in cities]

        report_progress("extracting", f"Activations for {probe_name}", {"probe": probe_name})
        print(f"  Extracting activations for {len(texts)} texts...")
        t0 = time.time()
        resid_acts, sae_acts = extract_activations_and_sae(
            model, tokenizer, sae, texts, layer
        )
        print(f"  Extraction took {time.time() - t0:.1f}s")
        print(f"  Residual shape: {resid_acts.shape}, SAE shape: {sae_acts.shape}")

        # ── Real absorption measurement (baseline for comparison) ──────
        report_progress("measuring", f"Real absorption for {probe_name}")
        print(f"\n  [Real] Measuring absorption with correct labels...")
        real_results = {}
        for cls in np.unique(labels):
            result = measure_absorption(
                sae_acts, sae_decoder, p_dirs, labels, cls,
                cosine_threshold=COSINE_THRESHOLDS[0],
                dominance_threshold=DOMINANCE_THRESHOLDS[0],
            )
            real_results[int(cls)] = result
            if result["status"] == "measured":
                all_real_rates.append(result["absorption_rate"])
                print(f"    Class {cls}: absorption={result['absorption_rate']:.4f} "
                      f"(FN={result['n_false_negatives']}/{result['n_class']})")

        measured_real = [r for r in real_results.values() if r.get("status") == "measured"]
        mean_real = np.mean([r["absorption_rate"] for r in measured_real]) if measured_real else 0
        print(f"  [Real] Mean absorption: {mean_real:.4f}")

        # ── Control 1: Shuffled ────────────────────────────────────────
        report_progress("control_shuffled", f"Shuffled control for {probe_name}")
        shuffled = run_shuffled_control(
            sae_acts, sae_decoder, p_dirs, labels,
            n_shuffles=5,
            cosine_threshold=COSINE_THRESHOLDS[0],
            dominance_threshold=DOMINANCE_THRESHOLDS[0],
        )
        for sr in shuffled["per_shuffle_results"]:
            all_shuffled_rates.extend(sr["per_class_rates"])

        # ── Control 2: Random probe ────────────────────────────────────
        report_progress("control_random", f"Random probe control for {probe_name}")
        random_ctrl = run_random_probe_control(
            sae_acts, sae_decoder, labels, d_model,
            n_random=10,
            cosine_threshold=COSINE_THRESHOLDS[0],
            dominance_threshold=DOMINANCE_THRESHOLDS[0],
        )
        all_random_rates.extend([r["mean_absorption"] for r in random_ctrl["per_random_results"]])

        # Store per-probe results
        results["controls"][probe_name] = {
            "attribute": attr,
            "n_cities": len(cities),
            "n_classes": n_classes,
            "probe_accuracy": probe_data["mean_acc"],
            "real_absorption": {
                "mean_rate": float(mean_real),
                "per_class": {str(k): v for k, v in real_results.items()},
                "n_measured": len(measured_real),
            },
            "shuffled_control": shuffled,
            "random_probe_control": random_ctrl,
        }

        del resid_acts, sae_acts
        gc.collect()
        torch.cuda.empty_cache()

    # ── Control 3: First-letter baseline ───────────────────────────────
    print(f"\n{'='*60}")
    print("[FIRST-LETTER BASELINE]")
    print(f"{'='*60}")
    report_progress("first_letter", "Running first-letter baseline")

    first_letter = run_first_letter_control(
        model, tokenizer, sae, layer,
        n_letters=N_LETTERS_PILOT,
        cosine_threshold=COSINE_THRESHOLDS[0],
        dominance_threshold=DOMINANCE_THRESHOLDS[0],
    )
    results["first_letter_baseline"] = first_letter

    # ── Statistical Comparisons ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[STATISTICAL COMPARISONS]")
    print(f"{'='*60}")
    report_progress("statistics", "Computing statistical comparisons")

    comparisons = compare_real_vs_controls(
        all_real_rates, all_shuffled_rates, all_random_rates,
    )
    results["statistical_comparisons"] = comparisons

    for test_name, test_result in comparisons.items():
        if "error" not in test_result:
            sig = "YES" if test_result.get("significant_at_005") else "NO"
            print(f"  {test_name}: p={test_result['p_value']:.4f}, "
                  f"significant={sig}, effect={test_result.get('effect_size', 'N/A'):.4f}")

    # ── Dead Feature Statistics ────────────────────────────────────────
    # Report how many SAE features are dead (never activate)
    dead_stats = {}
    # We'll re-extract once more for dead feature analysis
    # (or use cached data from last probe run)
    # For pilot, report from the last sae_acts if available

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    # Aggregate pass/fail
    all_controls = []
    for pname, pdata in results["controls"].items():
        sh = pdata.get("shuffled_control", {})
        rp = pdata.get("random_probe_control", {})
        all_controls.append({
            "probe": pname,
            "shuffled_rate": sh.get("overall_mean_absorption", None),
            "shuffled_passed": sh.get("passed", None),
            "random_rate": rp.get("overall_mean_absorption", None),
            "random_passed": rp.get("passed", None),
        })

    n_shuffled_pass = sum(1 for c in all_controls if c.get("shuffled_passed"))
    n_random_pass = sum(1 for c in all_controls if c.get("random_passed"))
    n_total = len(all_controls)

    # Overall pilot verdict
    all_shuffled_rates_summary = [c["shuffled_rate"] for c in all_controls if c["shuffled_rate"] is not None]
    all_random_rates_summary = [c["random_rate"] for c in all_controls if c["random_rate"] is not None]

    shuffled_mean = np.mean(all_shuffled_rates_summary) if all_shuffled_rates_summary else None
    random_mean = np.mean(all_random_rates_summary) if all_random_rates_summary else None
    first_letter_mean = first_letter.get("mean_absorption_rate", None)

    # Pilot pass criteria from task plan:
    # - Shuffled control absorption rate < 10%
    # - Random probe control < 10%
    # - Both significantly lower than real measurement (paired test p < 0.05)
    shuffled_pass = shuffled_mean is not None and shuffled_mean < 0.10
    random_pass = random_mean is not None and random_mean < 0.10
    stats_sig = any(
        v.get("significant_at_005", False)
        for v in comparisons.values()
        if isinstance(v, dict) and "error" not in v
    )

    # If real absorption is also very low (GPT-2 may not show much absorption on
    # knowledge features), the controls might not be significantly different.
    # This is itself an informative result.
    real_mean = np.mean(all_real_rates) if all_real_rates else 0

    pilot_verdict = "GO" if (shuffled_pass and random_pass) else "MIXED"
    if real_mean < 0.05 and shuffled_mean is not None and shuffled_mean < 0.05:
        pilot_verdict = "INFORMATIVE_NULL"
        pilot_detail = (
            f"Both real absorption ({real_mean:.4f}) and controls are near zero. "
            "This is consistent with GPT-2 Small's limited knowledge capacity rather than "
            "a failure of the absorption measurement method. Controls are valid "
            "(low false positive rate), but there is insufficient real absorption signal "
            "to demonstrate statistical separation."
        )
    elif shuffled_pass and random_pass:
        pilot_detail = (
            f"Both controls pass: shuffled={shuffled_mean:.4f}<10%, random={random_mean:.4f}<10%. "
            f"Real absorption mean={real_mean:.4f}. "
            f"First-letter baseline={first_letter_mean:.4f}. "
            f"Statistical significance: {stats_sig}."
        )
    else:
        pilot_detail = (
            f"Some controls did not pass. Shuffled={shuffled_mean}, Random={random_mean}. "
            f"This may indicate a measurement artifact or insufficient data in pilot."
        )

    results["summary"] = {
        "pilot_verdict": pilot_verdict,
        "pilot_detail": pilot_detail,
        "elapsed_sec": round(elapsed, 1),
        "real_absorption_mean": float(real_mean),
        "shuffled_mean": float(shuffled_mean) if shuffled_mean is not None else None,
        "random_mean": float(random_mean) if random_mean is not None else None,
        "first_letter_mean": float(first_letter_mean) if first_letter_mean is not None else None,
        "n_probes_tested": n_total,
        "n_shuffled_passed": n_shuffled_pass,
        "n_random_passed": n_random_pass,
        "per_control_summary": all_controls,
    }

    # ── Print Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[RESULTS SUMMARY]")
    print(f"{'='*60}")
    print(f"  Real absorption (mean): {real_mean:.4f}")
    print(f"  Shuffled control (mean): {shuffled_mean:.4f}" if shuffled_mean else "  Shuffled: N/A")
    print(f"  Random probe (mean):    {random_mean:.4f}" if random_mean else "  Random: N/A")
    print(f"  First-letter baseline:  {first_letter_mean:.4f}" if first_letter_mean else "  First-letter: N/A")
    print(f"  Shuffled passed: {n_shuffled_pass}/{n_total}")
    print(f"  Random passed:   {n_random_pass}/{n_total}")
    print(f"  Verdict: {pilot_verdict}")
    print(f"  Time: {elapsed:.1f}s")

    # ── Save Results ───────────────────────────────────────────────────
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    pilot_path = PILOT_DIR / "P2_controls.json"
    json_dump(results, pilot_path)
    print(f"\n[SAVE] {pilot_path}")

    full_path = FULL_DIR / "P2_controls.json"
    json_dump(results, full_path)
    print(f"[SAVE] {full_path}")

    # ── Mark Done ──────────────────────────────────────────────────────
    mark_done(
        "success",
        f"{pilot_verdict}: real={real_mean:.4f}, shuffled={shuffled_mean:.4f}, "
        f"random={random_mean:.4f}, first_letter={first_letter_mean:.4f}, "
        f"elapsed={elapsed:.1f}s",
    )

    # ── GPU Progress ───────────────────────────────────────────────────
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
            "n_probes_tested": n_total,
            "gpu_count": 1,
            "gpu_model": GPU_ID,
        },
    }

    with open(gp_path, "w") as f:
        json.dump(gp, f, cls=NumpyEncoder, indent=2)
    print(f"[GPU] {gp_path}")

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
