"""
P2_absorption_measurement: Cross-Domain Absorption Measurement on Knowledge Hierarchies
========================================================================================
Task: Measure feature absorption on RAVEL knowledge hierarchies using GPT-2 Small SAEs.

Adapted Chanin absorption metric pipeline:
1. For each (probe, SAE, layer), identify "split features": SAE features that
   selectively activate for one class vs others (class-specific features)
2. Find "false negatives": cities where the probe predicts correctly but NO
   split feature for the true class fires
3. Among false negatives, identify whether a single non-split feature dominates
   the SAE representation (the "absorber")
4. Absorption rate = |false negatives with dominant absorber| / |probe-correct|

The split feature identification uses activation statistics rather than decoder
cosine (which is too permissive in 768-dim space). A feature is "split" for class C
if it fires significantly more often on class-C cities than on non-class-C cities.

Pilot scope: 200 cities, layer 8, 1 SAE (24k), 5 probe types.
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
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────────────
TASK_ID = "P2_absorption_measurement"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
PROBE_DIR = FULL_DIR / "P2_probes"
DATA_DIR = WORKSPACE / "exp" / "data" / "ravel"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = 0

# Pilot parameters
PILOT_N_CITIES = 200
PILOT_LAYERS = [8]     # Best layer from probe training
SAE_RELEASE = "gpt2-small-res-jb"

# Split feature identification thresholds
SELECTIVITY_THRESHOLD = 3.0  # Feature must fire >=3x more on target class
MIN_ACTIVATION_RATE = 0.05   # Feature must fire on >= 5% of target class tokens
# Absorption detection thresholds to sweep
DOMINANCE_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]

# Which probes to use
PROBE_CONFIGS = [
    ("Country_binary_US", "binary", "US vs non-US cities"),
    ("Language_binary_English", "binary", "English vs non-English cities"),
    ("Continent", "multiclass", "6-continent classification"),
    ("Country_top10", "multiclass", "Top-10 countries"),
    ("Language_top10", "multiclass", "Top-10 languages"),
]

TEMPLATE = "{city}, a city known for being in"


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
    print(f"\n[DONE] {status}: {summary}")


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


def load_probe(probe_name, layer, prefer_pre=True):
    """Load a saved probe from .npz file.

    prefer_pre=True: use the probe trained on hook_resid_pre (matching SAE hook point).
    Falls back to the original (hook_resid_post) probe if pre version not available.
    """
    if prefer_pre:
        path = PROBE_DIR / f"probe_{probe_name}_layer{layer}_pre.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return {
                "coef": data["coef"],
                "intercept": data["intercept"],
                "scaler_mean": data["scaler_mean"],
                "scaler_scale": data["scaler_scale"],
                "classes": data["classes"],
                "mean_acc": float(data["mean_acc"]),
                "tier": int(data["tier"]),
                "hook_point": "hook_resid_pre",
            }

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
        "tier": int(data["tier"]),
        "hook_point": "hook_resid_post",
    }


# ── Label Assignment ───────────────────────────────────────────────────
def assign_label(city, probe_name, probe_type, classes):
    """Assign a label to a city for a given probe."""
    classes_str = [str(c) for c in classes]
    if probe_type == "binary":
        if "US" in probe_name:
            label = "United States" if city.get("Country") == "United States" else "non-US"
        elif "English" in probe_name:
            label = "English" if city.get("Language") == "English" else "non-English"
        else:
            label = "unknown"
    else:
        if "Country" in probe_name:
            label = city.get("Country", "unknown")
        elif "Continent" in probe_name:
            label = city.get("Continent", "unknown")
        elif "Language" in probe_name:
            label = city.get("Language", "unknown")
        else:
            label = "unknown"
    return label if label in classes_str else None


# ── Core: Extract Activations + SAE Encodings ──────────────────────────
def extract_all_representations(model, sae, cities, layer, template, batch_size=32,
                                probe_hook_point="hook_resid_pre"):
    """
    Extract residual stream activations and SAE encodings for all cities.

    Both the probe and SAE should operate on the same hook point for valid
    absorption measurement. The SAE's hook point is read from its config.
    The probe hook point should match (default: hook_resid_pre).

    Returns:
        probe_activations: (N, d_model) numpy array (for probe prediction)
        sae_acts: (N, d_sae) numpy array (SAE encoding)
    """
    tokenizer = model.tokenizer
    # SAE hook point
    sae_hook = sae.cfg.metadata.hook_name  # e.g., "blocks.8.hook_resid_pre"
    # Probe hook point (should match SAE for valid absorption measurement)
    probe_hook = f"blocks.{layer}.{probe_hook_point}"

    # If both are the same, only cache one
    hooks_to_cache = list(set([sae_hook, probe_hook]))

    all_probe_acts = []
    all_sae_acts = []

    for batch_start in range(0, len(cities), batch_size):
        batch_cities = cities[batch_start:batch_start + batch_size]
        prompts = [template.format(city=c["city"]) for c in batch_cities]

        tokens = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=64)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                names_filter=hooks_to_cache,
                attention_mask=tokens.get("attention_mask"),
            )

        if "attention_mask" in tokens:
            seq_lens = tokens["attention_mask"].sum(dim=1) - 1
        else:
            seq_lens = torch.full((tokens["input_ids"].shape[0],),
                                  tokens["input_ids"].shape[1] - 1, device=DEVICE)

        probe_resid = cache[probe_hook]
        sae_resid = cache[sae_hook]

        for j in range(len(batch_cities)):
            probe_act = probe_resid[j, seq_lens[j]].float()
            all_probe_acts.append(probe_act.detach().cpu().numpy())

            sae_input = sae_resid[j, seq_lens[j]].float()
            sae_enc = sae.encode(sae_input.unsqueeze(0)).squeeze(0)
            all_sae_acts.append(sae_enc.detach().cpu().numpy())

        del cache
        torch.cuda.empty_cache()

    return np.array(all_probe_acts), np.array(all_sae_acts)


# ── Core: Identify Split Features ──────────────────────────────────────
def identify_split_features(sae_acts, labels, classes,
                           selectivity_threshold=3.0, min_activation_rate=0.05):
    """
    Identify SAE features that selectively activate for specific classes.

    A feature is "split" for class C if:
    1. It fires on >= min_activation_rate of class-C tokens
    2. Its firing rate on class-C tokens is >= selectivity_threshold x its rate on non-C tokens

    Returns:
        split_features: dict {class_name: list of (feature_idx, selectivity_ratio)}
    """
    n_samples, d_sae = sae_acts.shape
    is_active = sae_acts > 0  # (N, d_sae) boolean

    split_features = {}
    for cls in classes:
        cls_mask = np.array([l == cls for l in labels])
        n_cls = cls_mask.sum()
        n_other = n_samples - n_cls

        if n_cls < 3 or n_other < 3:
            split_features[cls] = []
            continue

        # Firing rates
        cls_rate = is_active[cls_mask].mean(axis=0)     # (d_sae,)
        other_rate = is_active[~cls_mask].mean(axis=0)  # (d_sae,)

        # Selectivity: cls_rate / max(other_rate, epsilon)
        selectivity = cls_rate / np.maximum(other_rate, 1e-6)

        # Filter by both criteria
        selected = np.where(
            (cls_rate >= min_activation_rate) &
            (selectivity >= selectivity_threshold)
        )[0]

        # Sort by selectivity
        feats = [(int(f), float(selectivity[f]), float(cls_rate[f]))
                 for f in selected]
        feats.sort(key=lambda x: x[1], reverse=True)
        split_features[cls] = feats

    return split_features


# ── Core: Measure Absorption ───────────────────────────────────────────
def measure_absorption(
    activations, sae_acts, labels, classes,
    probe_data, split_features,
    dominance_threshold=1.0,
):
    """
    Measure absorption given pre-computed activations and split features.

    For each city token:
    1. Predict with probe (correct/incorrect)
    2. If correct: check if any split feature for true class fires
    3. If no split feature fires (false negative): check if a single non-split
       feature dominates the SAE encoding (absorption candidate)

    Returns detailed results.
    """
    n_samples = len(labels)
    coef = probe_data["coef"]
    intercept = probe_data["intercept"]
    scaler_mean = probe_data["scaler_mean"]
    scaler_scale = probe_data["scaler_scale"]
    classes_list = [str(c) for c in probe_data["classes"]]

    # Get all split feature indices per class
    split_idx_per_class = {}
    all_split_idx = set()
    for cls, feats in split_features.items():
        idx_set = set(f[0] for f in feats)
        split_idx_per_class[cls] = idx_set
        all_split_idx.update(idx_set)

    # Process each token
    probe_correct = []
    split_fired = []         # True if any split feature for true class fired
    absorption_candidates = []
    absorbed_tokens = []

    for i in range(n_samples):
        act_np = activations[i]
        sae_act = sae_acts[i]
        true_label = labels[i]

        # Probe prediction
        act_scaled = (act_np - scaler_mean) / scaler_scale
        logits = coef @ act_scaled + intercept
        if len(classes_list) == 2 and coef.shape[0] == 1:
            pred_idx = 1 if logits[0] > 0 else 0
        else:
            pred_idx = int(np.argmax(logits))
        pred_label = classes_list[pred_idx]
        is_correct = (pred_label == true_label)
        probe_correct.append(is_correct)

        if not is_correct:
            split_fired.append(False)
            continue

        # Check if any split feature for true class fires
        active_features = set(np.where(sae_act > 0)[0])
        class_split = split_idx_per_class.get(true_label, set())
        class_split_active = active_features.intersection(class_split)
        has_split = len(class_split_active) > 0
        split_fired.append(has_split)

        if has_split:
            continue

        # False negative: probe correct but no split feature fires
        # Check for absorption: is there a dominant non-split feature?
        non_split_active = active_features - all_split_idx
        if len(non_split_active) == 0 and len(active_features) == 0:
            absorption_candidates.append({
                "index": i,
                "status": "dead_representation",
                "n_active": 0,
            })
            continue

        # Find top activating feature (among all active)
        active_vals = [(f, float(sae_act[f])) for f in active_features]
        active_vals.sort(key=lambda x: x[1], reverse=True)

        top_feat, top_act = active_vals[0] if active_vals else (None, 0)
        second_act = active_vals[1][1] if len(active_vals) > 1 else 0
        dominance = top_act / max(second_act, 1e-10) if top_act > 0 else 0

        is_split = top_feat in all_split_idx if top_feat is not None else False

        detail = {
            "index": i,
            "n_active": len(active_features),
            "n_split_active": len(class_split_active),
            "top_feature": int(top_feat) if top_feat is not None else -1,
            "top_activation": float(top_act),
            "second_activation": float(second_act),
            "dominance_ratio": float(dominance),
            "top_is_split": is_split,
        }

        # Absorption: dominant non-split feature
        if not is_split and dominance >= dominance_threshold:
            detail["is_absorbed"] = True
            absorbed_tokens.append(detail)
        else:
            detail["is_absorbed"] = False

        absorption_candidates.append(detail)

    n_correct = sum(probe_correct)
    n_split_fired = sum(1 for c, s in zip(probe_correct, split_fired) if c and s)
    n_false_neg = sum(1 for c, s in zip(probe_correct, split_fired) if c and not s)
    n_absorbed = len(absorbed_tokens)

    return {
        "n_total": n_samples,
        "n_probe_correct": n_correct,
        "probe_accuracy": round(n_correct / max(n_samples, 1), 4),
        "n_split_fired": n_split_fired,
        "n_false_negatives": n_false_neg,
        "false_negative_rate": round(n_false_neg / max(n_correct, 1), 4),
        "n_absorbed": n_absorbed,
        "absorption_rate": round(n_absorbed / max(n_correct, 1), 4),
        "absorption_rate_of_fn": round(n_absorbed / max(n_false_neg, 1), 4),
        "dominance_threshold": dominance_threshold,
        "n_dead_representation": sum(1 for c in absorption_candidates
                                     if c.get("status") == "dead_representation"),
        "absorption_details_sample": absorbed_tokens[:10],
        "fn_details_sample": absorption_candidates[:10],
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "Starting P2 absorption measurement pilot")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    # Load model
    print(f"[MODEL] Loading {MODEL_NAME}...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loaded: {model.cfg.n_layers} layers, d={model.cfg.d_model}")

    # Load data
    print(f"\n[DATA] Loading RAVEL cities...")
    all_cities = load_ravel_cities()
    print(f"[DATA] {len(all_cities)} cities total")

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(all_cities))[:PILOT_N_CITIES]
    pilot_cities = [all_cities[i] for i in indices]
    print(f"[DATA] Pilot sample: {len(pilot_cities)} cities")

    # Results structure
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "n_cities_pilot": len(pilot_cities),
        "seed": SEED,
        "probe_source": "P2_probe_training_v3",
        "split_feature_config": {
            "selectivity_threshold": SELECTIVITY_THRESHOLD,
            "min_activation_rate": MIN_ACTIVATION_RATE,
        },
        "measurements": [],
        "threshold_sweep": [],
    }

    for layer in PILOT_LAYERS:
        print(f"\n{'='*70}")
        print(f"[SAE] Loading GPT-2 Small SAE for layer {layer}")
        print(f"{'='*70}")

        from sae_lens import SAE as SAELens
        sae = SAELens.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        sae = sae.to(DEVICE)
        sae.eval()
        print(f"[SAE] d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")

        # Step 1: Extract all representations once (shared across probes)
        # Use hook_resid_pre for both probe and SAE (correct alignment)
        print(f"\n[EXTRACT] Extracting residual stream + SAE activations for {len(pilot_cities)} cities...")
        print(f"[EXTRACT] SAE hook: {sae.cfg.metadata.hook_name}, Probe hook: blocks.{layer}.hook_resid_pre")
        t0 = time.time()
        activations, sae_acts_all = extract_all_representations(
            model, sae, pilot_cities, layer, TEMPLATE, batch_size=32,
            probe_hook_point="hook_resid_pre",
        )
        extract_time = time.time() - t0
        print(f"[EXTRACT] Done in {extract_time:.1f}s. Shape: acts={activations.shape}, sae={sae_acts_all.shape}")

        # Dead feature statistics (computed once)
        n_dead_features = int(np.sum(np.all(sae_acts_all == 0, axis=0)))
        n_alive = sae.cfg.d_sae - n_dead_features
        mean_active_per_token = float(np.mean(np.sum(sae_acts_all > 0, axis=1)))
        print(f"[SAE] Dead features: {n_dead_features}/{sae.cfg.d_sae} "
              f"({100*n_dead_features/sae.cfg.d_sae:.1f}%)")
        print(f"[SAE] Mean active features per token: {mean_active_per_token:.1f}")

        # Step 2: Measure absorption for each probe
        for probe_name, probe_type, desc in PROBE_CONFIGS:
            print(f"\n  {'─'*60}")
            print(f"  [PROBE] {probe_name} ({desc}) at layer {layer}")
            print(f"  {'─'*60}")

            probe_data = load_probe(probe_name, layer)
            if probe_data is None:
                print(f"    SKIP: No probe found for {probe_name}_layer{layer}")
                continue

            print(f"    Probe accuracy (train): {probe_data['mean_acc']:.4f}, Tier: {probe_data['tier']}")
            report_progress("measuring", f"{probe_name} layer {layer}")

            # Assign labels
            classes_list = [str(c) for c in probe_data["classes"]]
            labels = []
            valid_mask = []
            for c in pilot_cities:
                label = assign_label(c, probe_name, probe_type, probe_data["classes"])
                if label is not None:
                    labels.append(label)
                    valid_mask.append(True)
                else:
                    labels.append(None)
                    valid_mask.append(False)

            valid_idx = [i for i, v in enumerate(valid_mask) if v]
            valid_acts = activations[valid_idx]
            valid_sae = sae_acts_all[valid_idx]
            valid_labels = [labels[i] for i in valid_idx]
            valid_cities_sub = [pilot_cities[i] for i in valid_idx]

            print(f"    Valid cities: {len(valid_idx)}/{len(pilot_cities)}")
            label_counts = Counter(valid_labels)
            for cls, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {cls}: {cnt}")

            # Identify split features
            print(f"    Identifying split features (selectivity>{SELECTIVITY_THRESHOLD}, "
                  f"min_rate>{MIN_ACTIVATION_RATE})...")
            split_features = identify_split_features(
                valid_sae, valid_labels, classes_list,
                selectivity_threshold=SELECTIVITY_THRESHOLD,
                min_activation_rate=MIN_ACTIVATION_RATE,
            )
            n_total_split = sum(len(v) for v in split_features.values())
            print(f"    Total split features: {n_total_split}")
            for cls, feats in split_features.items():
                if feats:
                    top_sel = feats[0][1] if feats else 0
                    print(f"      {cls}: {len(feats)} features (top selectivity: {top_sel:.1f})")

            # Measure absorption at default dominance threshold
            result = measure_absorption(
                valid_acts, valid_sae, valid_labels, classes_list,
                probe_data, split_features,
                dominance_threshold=1.0,
            )

            # Add metadata
            result["probe_name"] = probe_name
            result["probe_type"] = probe_type
            result["probe_accuracy_reported"] = probe_data["mean_acc"]
            result["layer"] = layer
            result["d_sae"] = int(sae.cfg.d_sae)
            result["n_dead_features"] = n_dead_features
            result["n_alive_features"] = n_alive
            result["mean_active_per_token"] = mean_active_per_token
            result["n_split_features_total"] = n_total_split
            result["split_features_per_class"] = {
                cls: len(feats) for cls, feats in split_features.items()
            }
            result["selectivity_threshold"] = SELECTIVITY_THRESHOLD
            result["min_activation_rate"] = MIN_ACTIVATION_RATE
            # Add city names for sample absorption details
            for d in result.get("absorption_details_sample", []):
                if "index" in d:
                    d["city"] = valid_cities_sub[d["index"]]["city"]
                    d["true_label"] = valid_labels[d["index"]]
            for d in result.get("fn_details_sample", []):
                if "index" in d:
                    d["city"] = valid_cities_sub[d["index"]]["city"]
                    d["true_label"] = valid_labels[d["index"]]

            results["measurements"].append(result)

            print(f"\n    >>> RESULT: probe_acc={result['probe_accuracy']:.3f}, "
                  f"FN_rate={result['false_negative_rate']:.3f}, "
                  f"abs_rate={result['absorption_rate']:.3f}, "
                  f"n_absorbed={result['n_absorbed']}")

        # Step 3: Threshold sweep on Country_binary_US
        print(f"\n{'='*70}")
        print(f"[SWEEP] Dominance threshold sensitivity on Country_binary_US, layer {layer}")
        print(f"{'='*70}")

        us_probe = load_probe("Country_binary_US", layer)
        if us_probe is not None:
            us_labels = []
            us_valid = []
            for c in pilot_cities:
                label = assign_label(c, "Country_binary_US", "binary", us_probe["classes"])
                if label is not None:
                    us_labels.append(label)
                    us_valid.append(True)
                else:
                    us_labels.append(None)
                    us_valid.append(False)

            us_idx = [i for i, v in enumerate(us_valid) if v]
            us_acts = activations[us_idx]
            us_sae = sae_acts_all[us_idx]
            us_lab = [us_labels[i] for i in us_idx]
            us_classes = [str(c) for c in us_probe["classes"]]

            # Also sweep selectivity thresholds
            for sel_t in [2.0, 3.0, 5.0, 10.0]:
                split_f = identify_split_features(
                    us_sae, us_lab, us_classes,
                    selectivity_threshold=sel_t, min_activation_rate=MIN_ACTIVATION_RATE
                )
                n_split = sum(len(v) for v in split_f.values())

                for dom_t in DOMINANCE_THRESHOLDS:
                    res = measure_absorption(
                        us_acts, us_sae, us_lab, us_classes,
                        us_probe, split_f, dominance_threshold=dom_t
                    )
                    entry = {
                        "selectivity_threshold": sel_t,
                        "dominance_threshold": dom_t,
                        "n_split_features": n_split,
                        "absorption_rate": res["absorption_rate"],
                        "false_negative_rate": res["false_negative_rate"],
                        "n_absorbed": res["n_absorbed"],
                        "n_false_neg": res["n_false_negatives"],
                        "probe_accuracy": res["probe_accuracy"],
                    }
                    results["threshold_sweep"].append(entry)
                    print(f"  sel={sel_t}, dom={dom_t}: split={n_split}, "
                          f"FN_rate={res['false_negative_rate']:.3f}, "
                          f"abs_rate={res['absorption_rate']:.3f}")

        # Clean up
        del sae, activations, sae_acts_all
        torch.cuda.empty_cache()
        gc.collect()

    # ── Aggregate Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[SUMMARY]")
    print(f"{'='*70}")

    measurement_summary = []
    for m in results["measurements"]:
        s = {
            "probe": m["probe_name"],
            "type": m["probe_type"],
            "probe_acc": m["probe_accuracy"],
            "n_correct": m["n_probe_correct"],
            "fn_rate": m["false_negative_rate"],
            "absorption_rate": m["absorption_rate"],
            "n_absorbed": m["n_absorbed"],
            "n_split_features": m["n_split_features_total"],
        }
        measurement_summary.append(s)
        print(f"  {s['probe']:30s} | acc={s['probe_acc']:.3f} | "
              f"split={s['n_split_features']:4d} | "
              f"FN={s['fn_rate']:.3f} | abs={s['absorption_rate']:.3f}")

    binary_measurements = [m for m in results["measurements"] if m["probe_type"] == "binary"]
    multi_measurements = [m for m in results["measurements"] if m["probe_type"] == "multiclass"]

    avg_binary_abs = np.mean([m["absorption_rate"] for m in binary_measurements]) if binary_measurements else 0
    avg_multi_abs = np.mean([m["absorption_rate"] for m in multi_measurements]) if multi_measurements else 0
    avg_overall_abs = np.mean([m["absorption_rate"] for m in results["measurements"]]) if results["measurements"] else 0
    avg_fn_rate = np.mean([m["false_negative_rate"] for m in results["measurements"]]) if results["measurements"] else 0

    elapsed = round(time.time() - start_time, 1)

    # Pilot verdict
    any_completed = len(results["measurements"]) > 0
    any_above_3pct = any(m["absorption_rate"] > 0.03 for m in results["measurements"])
    any_fn_above_10pct = any(m["false_negative_rate"] > 0.10 for m in results["measurements"])

    if any_completed:
        if any_above_3pct:
            verdict = "GO"
            detail = (
                f"Absorption detected above 3% threshold. "
                f"Avg binary absorption: {avg_binary_abs:.3f}, "
                f"Avg multiclass: {avg_multi_abs:.3f}. "
                f"Avg FN rate: {avg_fn_rate:.3f}."
            )
        elif any_fn_above_10pct:
            verdict = "INFORMATIVE_PARTIAL"
            detail = (
                f"False negative rate > 10% detected (indicating split features miss some "
                f"correctly-classified tokens), but absorption rate < 3% (the absorbing "
                f"features are distributed rather than dominant). "
                f"Avg binary absorption: {avg_binary_abs:.3f}, FN: {avg_fn_rate:.3f}."
            )
        else:
            verdict = "INFORMATIVE_NULL"
            detail = (
                f"Absorption rates below 3% across all probes. "
                f"Avg binary: {avg_binary_abs:.3f}, multiclass: {avg_multi_abs:.3f}. "
                f"FN rate: {avg_fn_rate:.3f}. "
                f"Knowledge hierarchies may not exhibit absorption at the same rate as "
                f"syntactic features (first-letter spelling), potentially because GPT-2 Small "
                f"encodes geographic knowledge less crisply."
            )
    else:
        verdict = "NO_GO"
        detail = "No measurements completed."

    results["summary"] = {
        "pilot_verdict": verdict,
        "verdict_detail": detail,
        "avg_absorption_rate_binary": round(float(avg_binary_abs), 4),
        "avg_absorption_rate_multiclass": round(float(avg_multi_abs), 4),
        "avg_absorption_rate_overall": round(float(avg_overall_abs), 4),
        "avg_false_negative_rate": round(float(avg_fn_rate), 4),
        "n_probes_measured": len(results["measurements"]),
        "measurement_summary": measurement_summary,
        "model": MODEL_NAME,
        "layers": PILOT_LAYERS,
        "n_cities": PILOT_N_CITIES,
        "total_time_sec": elapsed,
        "note": (
            "GPT-2 Small (124M params) was used as fallback because Gemma 2 2B is gated. "
            "GPT-2 has limited factual knowledge compared to larger models, so probe "
            "accuracy is moderate (35-90% depending on task). The absorption measurement "
            "protocol is valid but results should be interpreted as a lower bound on what "
            "a larger model would show. Binary probes (US/English) provide the most reliable "
            "directions given GPT-2's knowledge capacity."
        ),
    }

    print(f"\n  Verdict: {verdict}")
    print(f"  Detail: {detail}")
    print(f"  Time: {elapsed}s")

    # ── Save Results ───────────────────────────────────────────────────
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    pilot_path = PILOT_DIR / "P2_absorption_measurement.json"
    with open(pilot_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[SAVE] {pilot_path}")

    full_path = FULL_DIR / "P2_absorption_knowledge.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVE] {full_path}")

    # ── Mark Done ──────────────────────────────────────────────────────
    mark_done(
        "success",
        f"{verdict}: avg_abs_binary={avg_binary_abs:.4f}, "
        f"avg_abs_multi={avg_multi_abs:.4f}, avg_fn={avg_fn_rate:.4f}, elapsed={elapsed}s"
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
        "planned_min": 60,
        "actual_min": round(elapsed / 60, 1),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "n_cities": PILOT_N_CITIES,
            "layers": PILOT_LAYERS,
            "n_probes": len(PROBE_CONFIGS),
            "gpu_count": 1,
            "gpu_model": "NVIDIA RTX PRO 6000",
        },
    }

    with open(gp_path, "w") as f:
        json.dump(gp, f, indent=2, cls=NumpyEncoder)
    print(f"[GPU] Updated {gp_path}")

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
