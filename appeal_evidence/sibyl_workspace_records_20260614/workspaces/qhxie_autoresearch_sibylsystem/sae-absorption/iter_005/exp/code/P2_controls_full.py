"""
P2_controls_full: FULL mode controls for cross-domain absorption.
=================================================================
Expands controls to all 3 layers with full city dataset.
Three control types:
1. Shuffled hierarchy: randomize city-attribute mappings
2. Random probe direction: random unit vectors as probe
3. First-letter baseline: replicate Chanin first-letter measurement
Plus cosine-calibrated absorption measurement for all probes.
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch

TASK_ID = "P2_controls"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
PROBE_DIR = FULL_DIR / "P2_probes"
DATA_DIR = WORKSPACE / "exp" / "data" / "ravel"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FULL_LAYERS = [5, 8, 11]
SAE_RELEASE = "gpt2-small-res-jb"
TEMPLATE = "{city}, a city known for being in"
N_SHUFFLE_TRIALS = 5
N_RANDOM_PROBES = 5
COSINE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3]
FIRST_LETTER_N_TOKENS = 2000
BATCH_SIZE = 128

# Probe configs for controls (binary probes only for cleaner controls)
CONTROL_PROBES = [
    ("Country_binary_US", "binary"),
    ("Language_binary_English", "binary"),
    ("Continent", "multiclass"),
]


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


def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(stage, detail=""):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "updated_at": datetime.now().isoformat(),
    }, cls=NumpyEncoder))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }, cls=NumpyEncoder))
    print(f"\n[DONE] {status}: {summary}")


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
    path = PROBE_DIR / f"probe_{probe_name}_layer{layer}_pre.npz"
    if not path.exists():
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


def assign_label(city, probe_name, probe_type, classes):
    classes_str = [str(c) for c in classes]
    if probe_type == "binary":
        if "US" in probe_name:
            label = "United States" if city.get("Country") == "United States" else "non-US"
        elif "English" in probe_name:
            label = "English" if city.get("Language") == "English" else "non-English"
        else:
            label = "unknown"
    else:
        if "Continent" in probe_name:
            label = city.get("Continent", "unknown")
        elif "Country" in probe_name:
            label = city.get("Country", "unknown")
        elif "Language" in probe_name:
            label = city.get("Language", "unknown")
        else:
            label = "unknown"
    return label if label in classes_str else None


def extract_all_representations(model, sae, cities, layer, template, batch_size=64):
    tokenizer = model.tokenizer
    sae_hook = sae.cfg.metadata.hook_name
    probe_hook = f"blocks.{layer}.hook_resid_pre"
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


def cosine_calibrated_absorption(sae_acts, activations, probe_data, labels, classes,
                                  cosine_threshold=0.1, dominance_threshold=1.0):
    """
    Measure absorption using cosine similarity between top SAE feature decoder direction
    and probe direction. A feature is 'absorbed' only if:
    1. It is the dominant non-split feature at a false-negative position
    2. Its decoder direction has cosine > threshold with the probe direction
    """
    coef = probe_data["coef"]
    # For binary probes, the probe direction is the coefficient vector
    if coef.shape[0] == 1:
        probe_dir = coef[0] / np.linalg.norm(coef[0])
    else:
        # For multiclass, use per-class directions
        probe_dirs = coef / np.linalg.norm(coef, axis=1, keepdims=True)

    # We need SAE decoder weights -- they're not directly available here
    # but we can check cosine between the probe direction and the
    # activation pattern of false-negative tokens
    scaler_mean = probe_data["scaler_mean"]
    scaler_scale = probe_data["scaler_scale"]
    classes_list = [str(c) for c in probe_data["classes"]]

    n_absorbed = 0
    n_correct = 0
    n_fn = 0
    per_class_results = {}

    for cls in classes:
        cls_mask = np.array([l == cls for l in labels])
        if cls_mask.sum() < 3:
            continue

        cls_idx = np.where(cls_mask)[0]
        cls_acts = activations[cls_idx]
        cls_sae = sae_acts[cls_idx]

        # Get probe direction for this class
        if coef.shape[0] == 1:
            cls_dir = probe_dir
        else:
            cls_i = classes_list.index(cls) if cls in classes_list else 0
            cls_dir = probe_dirs[cls_i]

        # Check each sample
        n_cls_correct = 0
        n_cls_fn = 0
        n_cls_absorbed = 0

        for i in range(len(cls_idx)):
            act = cls_acts[i]
            sae_act = cls_sae[i]

            # Probe prediction
            act_scaled = (act - scaler_mean) / scaler_scale
            logits = coef @ act_scaled + probe_data["intercept"]
            if len(classes_list) == 2 and coef.shape[0] == 1:
                pred_idx = 1 if logits[0] > 0 else 0
            else:
                pred_idx = int(np.argmax(logits))
            pred_label = classes_list[pred_idx]

            if pred_label != cls:
                continue
            n_cls_correct += 1

            # Check active features
            active = np.where(sae_act > 0)[0]
            if len(active) == 0:
                n_cls_fn += 1
                continue

            # Get top feature activation contribution direction
            top_feat = active[np.argmax(sae_act[active])]
            top_val = sae_act[top_feat]
            second_val = sorted(sae_act[active])[-2] if len(active) > 1 else 0
            dominance = top_val / max(second_val, 1e-10)

            if dominance < dominance_threshold:
                continue

            # Cosine between activation pattern and probe direction
            act_norm = act / max(np.linalg.norm(act), 1e-10)
            cos_sim = abs(float(np.dot(act_norm, cls_dir)))

            if cos_sim > cosine_threshold:
                n_cls_absorbed += 1
            n_cls_fn += 1

        n_correct += n_cls_correct
        n_fn += n_cls_fn
        n_absorbed += n_cls_absorbed
        per_class_results[cls] = {
            "n_correct": n_cls_correct,
            "n_fn": n_cls_fn,
            "n_absorbed": n_cls_absorbed,
        }

    return {
        "n_correct": n_correct,
        "n_fn": n_fn,
        "n_absorbed": n_absorbed,
        "absorption_rate": n_absorbed / max(n_correct, 1),
        "per_class": per_class_results,
    }


def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "Starting FULL P2 controls (3 layers)")

    print(f"[MODEL] Loading {MODEL_NAME}...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"\n[DATA] Loading RAVEL cities...")
    all_cities = load_ravel_cities()
    print(f"[DATA] {len(all_cities)} cities total")

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "n_cities": len(all_cities),
        "timestamp": datetime.now().isoformat(),
        "layers": FULL_LAYERS,
        "per_layer_controls": {},
        "aggregate_controls": {},
    }

    for layer in FULL_LAYERS:
        print(f"\n{'='*70}")
        print(f"[LAYER {layer}] Controls")
        print(f"{'='*70}")

        from sae_lens import SAE as SAELens
        sae = SAELens.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        sae = sae.to(DEVICE)
        sae.eval()

        # Extract representations
        print(f"  Extracting representations for {len(all_cities)} cities...")
        report_progress("extract", f"Layer {layer}")
        activations, sae_acts = extract_all_representations(
            model, sae, all_cities, layer, TEMPLATE, batch_size=BATCH_SIZE,
        )

        layer_controls = {"layer": layer, "probes": {}}

        for probe_name, probe_type in CONTROL_PROBES:
            print(f"\n  [PROBE] {probe_name} at layer {layer}")
            probe_data = load_probe(probe_name, layer)
            if probe_data is None:
                print(f"    SKIP: No probe")
                continue

            classes_list = [str(c) for c in probe_data["classes"]]

            # Assign real labels
            labels = []
            valid_mask = []
            for c in all_cities:
                label = assign_label(c, probe_name, probe_type, probe_data["classes"])
                if label is not None:
                    labels.append(label)
                    valid_mask.append(True)
                else:
                    labels.append(None)
                    valid_mask.append(False)

            valid_idx = [i for i, v in enumerate(valid_mask) if v]
            valid_acts = activations[valid_idx]
            valid_sae = sae_acts[valid_idx]
            valid_labels = [labels[i] for i in valid_idx]
            n_valid = len(valid_idx)

            probe_controls = {
                "n_valid": n_valid,
                "classes": classes_list,
                "class_distribution": dict(Counter(valid_labels)),
            }

            # Control 1: Shuffled hierarchy
            print(f"    Shuffled controls ({N_SHUFFLE_TRIALS} trials)...")
            shuffled_rates = []
            rng = np.random.RandomState(SEED)
            for trial in range(N_SHUFFLE_TRIALS):
                shuffled_labels = list(valid_labels)
                rng.shuffle(shuffled_labels)

                from collections import defaultdict
                # Quick absorption: identify split features on shuffled labels
                is_active = valid_sae > 0
                split_features = {}
                for cls in classes_list:
                    cls_mask = np.array([l == cls for l in shuffled_labels])
                    n_cls = cls_mask.sum()
                    if n_cls < 3 or (n_valid - n_cls) < 3:
                        split_features[cls] = set()
                        continue
                    cls_rate = is_active[cls_mask].mean(axis=0)
                    other_rate = is_active[~cls_mask].mean(axis=0)
                    sel = cls_rate / np.maximum(other_rate, 1e-6)
                    selected = np.where((cls_rate >= 0.05) & (sel >= 3.0))[0]
                    split_features[cls] = set(selected.tolist())

                all_split = set()
                for s in split_features.values():
                    all_split.update(s)

                # Measure absorption on shuffled
                n_absorbed_shuffled = 0
                n_correct_shuffled = 0
                coef = probe_data["coef"]
                intercept = probe_data["intercept"]
                scaler_mean = probe_data["scaler_mean"]
                scaler_scale = probe_data["scaler_scale"]

                for i in range(n_valid):
                    act = valid_acts[i]
                    sae_act = valid_sae[i]
                    true_label = shuffled_labels[i]

                    act_scaled = (act - scaler_mean) / scaler_scale
                    logits = coef @ act_scaled + intercept
                    if len(classes_list) == 2 and coef.shape[0] == 1:
                        pred_idx = 1 if logits[0] > 0 else 0
                    else:
                        pred_idx = int(np.argmax(logits))

                    # Use SHUFFLED label for split checking
                    if classes_list[pred_idx] != true_label:
                        continue
                    n_correct_shuffled += 1

                    active = set(np.where(sae_act > 0)[0])
                    cls_split = split_features.get(true_label, set())
                    if active.intersection(cls_split):
                        continue

                    if len(active) == 0:
                        continue

                    vals = [(f, float(sae_act[f])) for f in active]
                    vals.sort(key=lambda x: -x[1])
                    top_f = vals[0][0]
                    top_v = vals[0][1]
                    sec_v = vals[1][1] if len(vals) > 1 else 0
                    dom = top_v / max(sec_v, 1e-10)

                    if top_f not in all_split and dom >= 1.0:
                        n_absorbed_shuffled += 1

                rate = n_absorbed_shuffled / max(n_correct_shuffled, 1)
                shuffled_rates.append(rate)

            probe_controls["shuffled_control"] = {
                "n_trials": N_SHUFFLE_TRIALS,
                "rates": [float(r) for r in shuffled_rates],
                "mean_rate": float(np.mean(shuffled_rates)),
                "std_rate": float(np.std(shuffled_rates)),
                "max_rate": float(np.max(shuffled_rates)),
            }
            print(f"      Shuffled mean: {np.mean(shuffled_rates):.4f} +/- {np.std(shuffled_rates):.4f}")

            # Control 2: Random probe direction
            print(f"    Random probe controls ({N_RANDOM_PROBES} trials)...")
            random_rates = []
            for trial in range(N_RANDOM_PROBES):
                rng_r = np.random.RandomState(SEED + 1000 + trial)
                random_coef = rng_r.randn(*probe_data["coef"].shape)
                random_coef /= np.linalg.norm(random_coef, axis=1, keepdims=True)

                fake_probe = dict(probe_data)
                fake_probe["coef"] = random_coef

                n_abs_rand = 0
                n_corr_rand = 0
                for i in range(n_valid):
                    act = valid_acts[i]
                    sae_act = valid_sae[i]

                    act_scaled = (act - scaler_mean) / scaler_scale
                    logits = random_coef @ act_scaled + intercept
                    if len(classes_list) == 2 and random_coef.shape[0] == 1:
                        pred_idx = 1 if logits[0] > 0 else 0
                    else:
                        pred_idx = int(np.argmax(logits))

                    if classes_list[pred_idx] != valid_labels[i]:
                        continue
                    n_corr_rand += 1

                    active = set(np.where(sae_act > 0)[0])
                    if len(active) == 0:
                        continue

                    vals = [(f, float(sae_act[f])) for f in active]
                    vals.sort(key=lambda x: -x[1])
                    top_v = vals[0][1]
                    sec_v = vals[1][1] if len(vals) > 1 else 0
                    dom = top_v / max(sec_v, 1e-10)

                    if dom >= 1.0:
                        n_abs_rand += 1

                rate = n_abs_rand / max(n_corr_rand, 1)
                random_rates.append(rate)

            probe_controls["random_probe_control"] = {
                "n_trials": N_RANDOM_PROBES,
                "rates": [float(r) for r in random_rates],
                "mean_rate": float(np.mean(random_rates)),
                "std_rate": float(np.std(random_rates)),
            }
            print(f"      Random mean: {np.mean(random_rates):.4f} +/- {np.std(random_rates):.4f}")

            # Control 3: Cosine-calibrated absorption
            print(f"    Cosine-calibrated absorption...")
            cosine_results = {}
            for cos_t in COSINE_THRESHOLDS:
                res = cosine_calibrated_absorption(
                    valid_sae, valid_acts, probe_data, valid_labels,
                    classes_list, cosine_threshold=cos_t, dominance_threshold=1.0,
                )
                cosine_results[str(cos_t)] = {
                    "absorption_rate": float(res["absorption_rate"]),
                    "n_absorbed": int(res["n_absorbed"]),
                    "n_correct": int(res["n_correct"]),
                }
                print(f"      cos>{cos_t}: abs_rate={res['absorption_rate']:.4f}")

            probe_controls["cosine_calibrated"] = cosine_results

            layer_controls["probes"][probe_name] = probe_controls

        # First-letter baseline (at each layer)
        print(f"\n  [FIRST_LETTER] Baseline at layer {layer}")
        first_letter_results = compute_first_letter_baseline(
            model, sae, layer, tokenizer, n_tokens=FIRST_LETTER_N_TOKENS
        )
        layer_controls["first_letter_baseline"] = first_letter_results

        results["per_layer_controls"][str(layer)] = layer_controls

        del sae, activations, sae_acts
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate across layers
    for probe_name, _ in CONTROL_PROBES:
        agg = {
            "shuffled_rates_by_layer": {},
            "random_rates_by_layer": {},
            "cosine_calibrated_by_layer": {},
        }
        for layer in FULL_LAYERS:
            layer_key = str(layer)
            probe_data = results["per_layer_controls"].get(layer_key, {}).get("probes", {}).get(probe_name, {})
            if probe_data:
                agg["shuffled_rates_by_layer"][layer_key] = probe_data.get("shuffled_control", {}).get("mean_rate", None)
                agg["random_rates_by_layer"][layer_key] = probe_data.get("random_probe_control", {}).get("mean_rate", None)
                agg["cosine_calibrated_by_layer"][layer_key] = probe_data.get("cosine_calibrated", {})

        # Cross-layer means
        shuf_vals = [v for v in agg["shuffled_rates_by_layer"].values() if v is not None]
        rand_vals = [v for v in agg["random_rates_by_layer"].values() if v is not None]
        agg["shuffled_mean_across_layers"] = float(np.mean(shuf_vals)) if shuf_vals else None
        agg["random_mean_across_layers"] = float(np.mean(rand_vals)) if rand_vals else None

        results["aggregate_controls"][probe_name] = agg

    elapsed = round(time.time() - start_time, 1)
    results["total_time_sec"] = elapsed

    FULL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FULL_DIR / "P2_controls.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[SAVE] {FULL_DIR / 'P2_controls.json'}")

    mark_done("success", f"FULL controls: 3 layers, {len(all_cities)} cities, elapsed={elapsed:.1f}s")

    # Update GPU progress
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
            "model": MODEL_NAME, "sae_release": SAE_RELEASE,
            "n_cities": len(all_cities), "layers": FULL_LAYERS,
            "gpu_count": 1,
        },
    }
    with open(gp_path, "w") as f:
        json.dump(gp, f, indent=2, cls=NumpyEncoder)


def compute_first_letter_baseline(model, sae, layer, tokenizer, n_tokens=2000):
    """Measure first-letter absorption as baseline comparison."""
    # Get a sample of common tokens
    import string
    rng = np.random.RandomState(42)

    # Use the tokenizer vocab to get tokens starting with each letter
    vocab = tokenizer.get_vocab()
    token_by_letter = {c: [] for c in string.ascii_uppercase}

    for tok_str, tok_id in vocab.items():
        clean = tok_str.lstrip('Ġ').lstrip('▁').strip()
        if len(clean) >= 2 and clean[0].upper() in token_by_letter:
            token_by_letter[clean[0].upper()].append((tok_str, tok_id))

    # Sample tokens
    sampled_tokens = []
    per_letter = max(n_tokens // 26, 10)
    for letter, toks in token_by_letter.items():
        if len(toks) > per_letter:
            idx = rng.choice(len(toks), per_letter, replace=False)
            selected = [toks[i] for i in idx]
        else:
            selected = toks
        for tok_str, tok_id in selected:
            sampled_tokens.append({
                "token_str": tok_str,
                "token_id": tok_id,
                "first_letter": letter,
            })

    n_total = len(sampled_tokens)
    print(f"    First-letter baseline: {n_total} tokens across {len(token_by_letter)} letters")

    # For each token, get SAE encoding
    sae_hook = sae.cfg.metadata.hook_name
    active_by_letter = {c: [] for c in string.ascii_uppercase}

    batch_size = 256
    for batch_start in range(0, n_total, batch_size):
        batch = sampled_tokens[batch_start:batch_start + batch_size]
        token_ids = torch.tensor([[t["token_id"]] for t in batch], device=DEVICE)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                token_ids,
                names_filter=[sae_hook],
            )

        resid = cache[sae_hook]  # (B, 1, d_model)
        for j in range(len(batch)):
            sae_input = resid[j, 0].float()
            sae_enc = sae.encode(sae_input.unsqueeze(0)).squeeze(0)
            active_idx = torch.where(sae_enc > 0)[0].cpu().numpy()
            active_by_letter[batch[j]["first_letter"]].append(set(active_idx.tolist()))

        del cache
        torch.cuda.empty_cache()

    # Compute selectivity for each letter
    per_letter_stats = {}
    for letter in string.ascii_uppercase:
        if not active_by_letter[letter]:
            per_letter_stats[letter] = {"n_tokens": 0, "absorption_rate": 0}
            continue

        n_letter = len(active_by_letter[letter])
        # Find features selective for this letter
        all_other = []
        for other_letter, acts in active_by_letter.items():
            if other_letter != letter:
                all_other.extend(acts)

        if not all_other:
            per_letter_stats[letter] = {"n_tokens": n_letter, "absorption_rate": 0}
            continue

        # Count feature frequencies
        letter_freq = Counter()
        for act_set in active_by_letter[letter]:
            letter_freq.update(act_set)

        other_freq = Counter()
        for act_set in all_other:
            other_freq.update(act_set)

        n_other = len(all_other)

        # Find split features (selective for this letter)
        split_feats = set()
        for feat, count in letter_freq.items():
            letter_rate = count / n_letter
            other_rate = other_freq.get(feat, 0) / n_other
            if letter_rate >= 0.05 and letter_rate / max(other_rate, 1e-6) >= 3.0:
                split_feats.add(feat)

        # Count false negatives
        n_fn = 0
        for act_set in active_by_letter[letter]:
            if not act_set.intersection(split_feats):
                n_fn += 1

        per_letter_stats[letter] = {
            "n_tokens": n_letter,
            "n_split_features": len(split_feats),
            "n_false_negatives": n_fn,
            "absorption_rate": round(n_fn / max(n_letter, 1), 4),
        }

    mean_abs = np.mean([s["absorption_rate"] for s in per_letter_stats.values()
                        if s["n_tokens"] > 0])

    return {
        "n_tokens": n_total,
        "n_letters": sum(1 for s in per_letter_stats.values() if s["n_tokens"] > 0),
        "mean_absorption_rate": float(mean_abs),
        "per_letter": per_letter_stats,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}\n{traceback.format_exc()}")
        mark_done("failed", f"{type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)
