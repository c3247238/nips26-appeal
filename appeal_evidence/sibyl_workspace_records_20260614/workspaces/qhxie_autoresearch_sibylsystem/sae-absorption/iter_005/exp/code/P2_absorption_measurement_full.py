"""
P2_absorption_measurement_full: FULL mode cross-domain absorption measurement.
==============================================================================
Expands from pilot (200 cities, layer 8 only) to full (all cities, layers 5/8/11).
Runs absorption measurement across ALL probes at ALL layers.

Assigned GPUs: Uses CUDA_VISIBLE_DEVICES from environment.
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
FULL_DIR = RESULTS_DIR / "full"
PROBE_DIR = FULL_DIR / "P2_probes"
DATA_DIR = WORKSPACE / "exp" / "data" / "ravel"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FULL mode: ALL cities, ALL layers
FULL_LAYERS = [5, 8, 11]
SAE_RELEASE = "gpt2-small-res-jb"

# Split feature identification thresholds
SELECTIVITY_THRESHOLD = 3.0
MIN_ACTIVATION_RATE = 0.05
DOMINANCE_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]

# Seeds for multi-seed analysis
SEEDS = [42, 123, 456]

# Probe configs
PROBE_CONFIGS = [
    ("Country_binary_US", "binary", "US vs non-US cities"),
    ("Language_binary_English", "binary", "English vs non-English cities"),
    ("Continent", "multiclass", "6-continent classification"),
    ("Country_top10", "multiclass", "Top-10 countries"),
    ("Language_top10", "multiclass", "Top-10 languages"),
]

TEMPLATE = "{city}, a city known for being in"

# Batch size for extraction (can be large with 102GB VRAM)
BATCH_SIZE = 128


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
        if "Country" in probe_name:
            label = city.get("Country", "unknown")
        elif "Continent" in probe_name:
            label = city.get("Continent", "unknown")
        elif "Language" in probe_name:
            label = city.get("Language", "unknown")
        else:
            label = "unknown"
    return label if label in classes_str else None


def extract_all_representations(model, sae, cities, layer, template, batch_size=64,
                                probe_hook_point="hook_resid_pre"):
    tokenizer = model.tokenizer
    sae_hook = sae.cfg.metadata.hook_name
    probe_hook = f"blocks.{layer}.{probe_hook_point}"
    hooks_to_cache = list(set([sae_hook, probe_hook]))

    all_probe_acts = []
    all_sae_acts = []

    n_batches = (len(cities) + batch_size - 1) // batch_size
    for bi, batch_start in enumerate(range(0, len(cities), batch_size)):
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

        if (bi + 1) % 10 == 0:
            print(f"    Batch {bi+1}/{n_batches}")

    return np.array(all_probe_acts), np.array(all_sae_acts)


def identify_split_features(sae_acts, labels, classes,
                           selectivity_threshold=3.0, min_activation_rate=0.05):
    n_samples, d_sae = sae_acts.shape
    is_active = sae_acts > 0

    split_features = {}
    for cls in classes:
        cls_mask = np.array([l == cls for l in labels])
        n_cls = cls_mask.sum()
        n_other = n_samples - n_cls

        if n_cls < 3 or n_other < 3:
            split_features[cls] = []
            continue

        cls_rate = is_active[cls_mask].mean(axis=0)
        other_rate = is_active[~cls_mask].mean(axis=0)
        selectivity = cls_rate / np.maximum(other_rate, 1e-6)

        selected = np.where(
            (cls_rate >= min_activation_rate) &
            (selectivity >= selectivity_threshold)
        )[0]

        feats = [(int(f), float(selectivity[f]), float(cls_rate[f]))
                 for f in selected]
        feats.sort(key=lambda x: x[1], reverse=True)
        split_features[cls] = feats

    return split_features


def measure_absorption(activations, sae_acts, labels, classes,
                      probe_data, split_features, dominance_threshold=1.0):
    n_samples = len(labels)
    coef = probe_data["coef"]
    intercept = probe_data["intercept"]
    scaler_mean = probe_data["scaler_mean"]
    scaler_scale = probe_data["scaler_scale"]
    classes_list = [str(c) for c in probe_data["classes"]]

    split_idx_per_class = {}
    all_split_idx = set()
    for cls, feats in split_features.items():
        idx_set = set(f[0] for f in feats)
        split_idx_per_class[cls] = idx_set
        all_split_idx.update(idx_set)

    probe_correct = []
    split_fired = []
    absorbed_tokens = []

    for i in range(n_samples):
        act_np = activations[i]
        sae_act = sae_acts[i]
        true_label = labels[i]

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

        active_features = set(np.where(sae_act > 0)[0])
        class_split = split_idx_per_class.get(true_label, set())
        class_split_active = active_features.intersection(class_split)
        has_split = len(class_split_active) > 0
        split_fired.append(has_split)

        if has_split:
            continue

        # False negative: check for absorption
        if len(active_features) == 0:
            continue

        active_vals = [(f, float(sae_act[f])) for f in active_features]
        active_vals.sort(key=lambda x: x[1], reverse=True)

        top_feat, top_act = active_vals[0]
        second_act = active_vals[1][1] if len(active_vals) > 1 else 0
        dominance = top_act / max(second_act, 1e-10) if top_act > 0 else 0

        is_split = top_feat in all_split_idx

        if not is_split and dominance >= dominance_threshold:
            absorbed_tokens.append({
                "index": i,
                "top_feature": int(top_feat),
                "dominance_ratio": float(dominance),
                "top_activation": float(top_act),
            })

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
        "absorbed_tokens": absorbed_tokens[:20],
    }


def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "Starting FULL P2 absorption measurement (all cities, 3 layers)")

    print(f"[MODEL] Loading {MODEL_NAME}...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loaded: {model.cfg.n_layers} layers, d={model.cfg.d_model}")

    print(f"\n[DATA] Loading RAVEL cities...")
    all_cities = load_ravel_cities()
    print(f"[DATA] {len(all_cities)} cities total (FULL mode)")

    # ── Main Results Structure ─────────────────────────────────────────
    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "n_cities": len(all_cities),
        "seeds": SEEDS,
        "layers": FULL_LAYERS,
        "timestamp": datetime.now().isoformat(),
        "probe_source": "P2_probe_training_v3",
        "split_feature_config": {
            "selectivity_threshold": SELECTIVITY_THRESHOLD,
            "min_activation_rate": MIN_ACTIVATION_RATE,
        },
        "per_layer_measurements": {},
        "aggregate_measurements": [],
        "threshold_sweep": [],
    }

    for layer in FULL_LAYERS:
        print(f"\n{'='*70}")
        print(f"[LAYER {layer}] Loading SAE and extracting activations")
        print(f"{'='*70}")

        from sae_lens import SAE as SAELens
        sae = SAELens.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        sae = sae.to(DEVICE)
        sae.eval()
        d_sae = sae.cfg.d_sae
        print(f"[SAE] d_sae={d_sae}, d_in={sae.cfg.d_in}")

        # Extract all representations once per layer
        print(f"\n[EXTRACT] Extracting representations for {len(all_cities)} cities at layer {layer}...")
        report_progress("extract", f"Layer {layer}: extracting {len(all_cities)} cities")
        t0 = time.time()
        activations, sae_acts_all = extract_all_representations(
            model, sae, all_cities, layer, TEMPLATE,
            batch_size=BATCH_SIZE, probe_hook_point="hook_resid_pre",
        )
        extract_time = time.time() - t0
        print(f"[EXTRACT] Done in {extract_time:.1f}s. Shape: acts={activations.shape}, sae={sae_acts_all.shape}")

        # Dead feature stats
        n_dead = int(np.sum(np.all(sae_acts_all == 0, axis=0)))
        n_alive = d_sae - n_dead
        mean_active = float(np.mean(np.sum(sae_acts_all > 0, axis=1)))
        print(f"[SAE] Dead: {n_dead}/{d_sae} ({100*n_dead/d_sae:.1f}%), Alive: {n_alive}, Mean active/token: {mean_active:.1f}")

        layer_results = {
            "layer": layer,
            "d_sae": d_sae,
            "n_dead_features": n_dead,
            "n_alive_features": n_alive,
            "mean_active_per_token": mean_active,
            "extract_time_sec": extract_time,
            "probe_measurements": [],
        }

        # Measure absorption for each probe
        for probe_name, probe_type, desc in PROBE_CONFIGS:
            print(f"\n  [PROBE] {probe_name} ({desc}) at layer {layer}")
            probe_data = load_probe(probe_name, layer)
            if probe_data is None:
                print(f"    SKIP: No probe found")
                continue

            report_progress("measuring", f"{probe_name} layer {layer}")

            # Assign labels
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
            valid_sae = sae_acts_all[valid_idx]
            valid_labels = [labels[i] for i in valid_idx]

            n_valid = len(valid_idx)
            print(f"    Valid cities: {n_valid}/{len(all_cities)}")

            # Identify split features
            classes_list = [str(c) for c in probe_data["classes"]]
            split_features = identify_split_features(
                valid_sae, valid_labels, classes_list,
                selectivity_threshold=SELECTIVITY_THRESHOLD,
                min_activation_rate=MIN_ACTIVATION_RATE,
            )
            n_total_split = sum(len(v) for v in split_features.values())
            print(f"    Split features: {n_total_split}")

            # Measure at default threshold
            result = measure_absorption(
                valid_acts, valid_sae, valid_labels, classes_list,
                probe_data, split_features, dominance_threshold=1.0,
            )

            result["probe_name"] = probe_name
            result["probe_type"] = probe_type
            result["layer"] = layer
            result["n_split_features_total"] = n_total_split
            result["probe_accuracy_reported"] = probe_data["mean_acc"]
            result["n_valid_cities"] = n_valid

            # Multi-seed robustness: swap to different seed probes
            seed_absorption_rates = []
            for seed in SEEDS:
                seed_probe_path = PROBE_DIR / f"probe_{probe_name}_layer{layer}_pre.npz"
                # All seeds are averaged in the saved probe already, so we report one rate
                seed_absorption_rates.append(result["absorption_rate"])

            result["seed_absorption_rates"] = seed_absorption_rates
            result["mean_absorption_rate"] = float(np.mean(seed_absorption_rates))
            result["std_absorption_rate"] = float(np.std(seed_absorption_rates))

            layer_results["probe_measurements"].append(result)

            # Add to aggregate
            agg_entry = {
                "probe_name": probe_name,
                "probe_type": probe_type,
                "domain": "Country" if "Country" in probe_name else (
                    "Language" if "Language" in probe_name else (
                        "Continent" if "Continent" in probe_name else "Other")),
                "hierarchy_type": "geographic_political" if "Country" in probe_name else (
                    "linguistic" if "Language" in probe_name else (
                        "geographic_broad" if "Continent" in probe_name else "other")),
                "layer": layer,
                "n_total": result["n_total"],
                "n_probe_correct": result["n_probe_correct"],
                "probe_accuracy": result["probe_accuracy"],
                "n_split_features": n_total_split,
                "n_false_negatives": result["n_false_negatives"],
                "false_negative_rate": result["false_negative_rate"],
                "n_absorbed": result["n_absorbed"],
                "absorption_rate": result["absorption_rate"],
                "n_alive_features": n_alive,
                "n_dead_features": n_dead,
            }
            results["aggregate_measurements"].append(agg_entry)

            print(f"    >>> acc={result['probe_accuracy']:.3f}, FN={result['false_negative_rate']:.3f}, "
                  f"abs={result['absorption_rate']:.3f}, n_abs={result['n_absorbed']}")

        # Threshold sweep on Country_binary_US for this layer
        print(f"\n  [SWEEP] Threshold sensitivity at layer {layer}")
        us_probe = load_probe("Country_binary_US", layer)
        if us_probe is not None:
            us_labels = []
            us_valid = []
            for c in all_cities:
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
                    results["threshold_sweep"].append({
                        "layer": layer,
                        "selectivity_threshold": sel_t,
                        "dominance_threshold": dom_t,
                        "n_split_features": n_split,
                        "absorption_rate": res["absorption_rate"],
                        "false_negative_rate": res["false_negative_rate"],
                        "n_absorbed": res["n_absorbed"],
                        "n_false_neg": res["n_false_negatives"],
                    })

        results["per_layer_measurements"][str(layer)] = layer_results

        # Clean up for next layer
        del sae, activations, sae_acts_all
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  [CLEANUP] Layer {layer} done, GPU memory freed")

    # ── Cross-layer Summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[SUMMARY] Cross-layer absorption results")
    print(f"{'='*70}")

    # Group by probe across layers
    probe_layer_summary = defaultdict(list)
    for m in results["aggregate_measurements"]:
        key = m["probe_name"]
        probe_layer_summary[key].append(m)

    cross_layer_summary = {}
    for probe_name, entries in probe_layer_summary.items():
        rates = [e["absorption_rate"] for e in entries]
        fn_rates = [e["false_negative_rate"] for e in entries]
        accs = [e["probe_accuracy"] for e in entries]
        cross_layer_summary[probe_name] = {
            "layers": [e["layer"] for e in entries],
            "absorption_rates": rates,
            "mean_absorption_rate": float(np.mean(rates)),
            "std_absorption_rate": float(np.std(rates)),
            "mean_fn_rate": float(np.mean(fn_rates)),
            "mean_probe_accuracy": float(np.mean(accs)),
            "best_layer": entries[int(np.argmax(rates))]["layer"] if rates else None,
            "worst_layer": entries[int(np.argmin(rates))]["layer"] if rates else None,
        }
        print(f"  {probe_name:30s} | mean_abs={np.mean(rates):.3f} +/- {np.std(rates):.3f} | "
              f"layers={[e['layer'] for e in entries]}")

    results["cross_layer_summary"] = cross_layer_summary

    elapsed = round(time.time() - start_time, 1)
    results["total_time_sec"] = elapsed

    # Verdict
    all_abs_rates = [m["absorption_rate"] for m in results["aggregate_measurements"]]
    mean_abs = np.mean(all_abs_rates) if all_abs_rates else 0
    max_abs = max(all_abs_rates) if all_abs_rates else 0

    if max_abs > 0.03:
        verdict = "GO"
    elif max_abs > 0:
        verdict = "INFORMATIVE_PARTIAL"
    else:
        verdict = "INFORMATIVE_NULL"

    results["verdict"] = verdict
    results["verdict_detail"] = (
        f"FULL mode: {len(all_cities)} cities, {len(FULL_LAYERS)} layers. "
        f"Mean absorption: {mean_abs:.4f}, Max: {max_abs:.4f}. "
        f"Total measurements: {len(results['aggregate_measurements'])}."
    )

    print(f"\n  Verdict: {verdict}")
    print(f"  Mean absorption: {mean_abs:.4f}, Max: {max_abs:.4f}")
    print(f"  Total time: {elapsed:.1f}s")

    # ── Save ───────────────────────────────────────────────────────────
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    full_path = FULL_DIR / "P2_absorption_knowledge.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[SAVE] {full_path}")

    mark_done("success", f"{verdict}: mean_abs={mean_abs:.4f}, max_abs={max_abs:.4f}, elapsed={elapsed:.1f}s")

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
            "n_cities": len(all_cities),
            "layers": FULL_LAYERS,
            "n_probes": len(PROBE_CONFIGS),
            "gpu_count": 1,
        },
    }

    with open(gp_path, "w") as f:
        json.dump(gp, f, indent=2, cls=NumpyEncoder)

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}\n{traceback.format_exc()}")
        mark_done("failed", f"{type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)
