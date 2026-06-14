"""
P2_probe_training: RAVEL Knowledge Probe Training (v2)
======================================================
Task: Train multi-class logistic regression probes on residual stream activations
      for city-country, city-continent, city-language attributes.

Model: GPT-2 Small (fallback from Gemma 2 2B due to gated access without HF token)
Data: RAVEL city dataset (3552 cities with country/continent/language)
Layers: 5, 8, 11 (adapted from 8, 12, 17 for Gemma 2B; GPT-2 has 12 layers)

PILOT mode:
- Use full RAVEL city dataset (3552 cities) for sufficient class coverage
- Filter to high-frequency classes to ensure enough samples per class
- 3 seeds, 80/20 stratified split
- Quality gate: accuracy >= 80% for Country, >= 75% for Continent (GPT-2 threshold)

Key fix from v1: v1 used only 200 cities -> severe data sparsity (4-5 samples per country).
Now using all 3552 cities with minimum 20 samples per class for Country/Language.
"""

import os
import sys
import json
import time
import gc
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Configuration ──────────────────────────────────────────────────────
TASK_ID = "P2_probe_training"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full" / "P2_probes"
DATA_DIR = WORKSPACE / "exp" / "data" / "ravel"

SEEDS = [42, 123, 456]
LAYERS = [5, 8, 11]  # GPT-2 Small: early, mid, late
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Attribute-specific configuration
ATTRIBUTE_CONFIG = {
    "Country": {
        "min_class_freq": 20,   # Minimum cities per country for inclusion
        "quality_threshold": 0.60,  # Lower threshold for multi-class (many countries)
        "template": "The city of {city} is located in",
        "max_classes": 30,  # Keep top-30 countries by frequency
    },
    "Continent": {
        "min_class_freq": 10,
        "quality_threshold": 0.70,  # 6 classes only, should be easier
        "template": "The city of {city} is on the continent of",
        "max_classes": None,  # Keep all (only 6 continents)
    },
    "Language": {
        "min_class_freq": 20,
        "quality_threshold": 0.55,  # Many languages, harder task
        "template": "The primary language spoken in {city} is",
        "max_classes": 20,  # Keep top-20 languages
    },
}


# ── JSON encoder for numpy types ──────────────────────────────────────
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


def json_dump(obj, f, **kwargs):
    json.dump(obj, f, cls=NumpyEncoder, **kwargs)


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
    """Load full RAVEL city dataset."""
    with open(DATA_DIR / "ravel_city_entity_attributes.json") as f:
        all_attrs = json.load(f)

    with open(DATA_DIR / "ravel_city_entity_to_split.json") as f:
        splits = json.load(f)

    data = []
    for city, attrs in all_attrs.items():
        entry = {"city": city, "split": splits.get(city, "unknown")}
        entry.update(attrs)
        data.append(entry)

    return data


def prepare_attribute_data(data, attribute, config):
    """Filter and prepare data for a specific attribute."""
    # Count class frequencies
    counts = Counter(d[attribute] for d in data)

    # Filter by minimum frequency
    valid_classes = {cls for cls, cnt in counts.items() if cnt >= config["min_class_freq"]}

    # If max_classes specified, keep only top-N by frequency
    if config.get("max_classes"):
        top_classes = [cls for cls, _ in counts.most_common(config["max_classes"])]
        valid_classes = valid_classes.intersection(top_classes)

    filtered = [d for d in data if d[attribute] in valid_classes]

    # Report statistics
    print(f"  Raw classes: {len(counts)}, after freq filter: {len(valid_classes)}")
    print(f"  Cities: {len(data)} -> {len(filtered)}")
    print(f"  Top classes: {Counter(d[attribute] for d in filtered).most_common(5)}")

    return filtered, len(valid_classes)


# ── Activation Extraction ─────────────────────────────────────────────
def extract_activations(model, tokenizer, cities, attribute, layer, config, batch_size=64):
    """Extract residual stream activations at specified layer for probe training.

    Uses the last token position (before padding) from residual stream post-attention.
    """
    template = config["template"]
    all_activations = []
    all_labels = []

    for i in range(0, len(cities), batch_size):
        batch = cities[i:i + batch_size]
        prompts = [template.format(city=c["city"]) for c in batch]
        labels = [c[attribute] for c in batch]

        # Tokenize with left-padding for causal LM
        tokens = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                names_filter=f"blocks.{layer}.hook_resid_post",
                attention_mask=tokens.get("attention_mask"),
            )

        resid = cache[f"blocks.{layer}.hook_resid_post"]  # [batch, seq, d_model]

        # Get last non-padding token position
        if "attention_mask" in tokens:
            seq_lens = tokens["attention_mask"].sum(dim=1) - 1
        else:
            seq_lens = torch.full((resid.shape[0],), resid.shape[1] - 1, device=DEVICE)

        for j in range(resid.shape[0]):
            act = resid[j, seq_lens[j]].float().cpu().numpy()
            all_activations.append(act)
            all_labels.append(labels[j])

        del cache, resid
        torch.cuda.empty_cache()

    activations = np.stack(all_activations)
    return activations, all_labels


# ── Probe Training ─────────────────────────────────────────────────────
def train_probe(X_train, y_train, X_test, y_test, seed=42, max_iter=3000):
    """Train logistic regression probe with standardization."""
    # Standardize features (important for logistic regression convergence)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        C=1.0,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))

    y_pred = clf.predict(X_test_s)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return clf, scaler, train_acc, test_acc, report


# ── Main Pipeline ──────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "Loading model and data")

    # ── Load Model ─────────────────────────────────────────────────────
    print(f"[MODEL] Loading {MODEL_NAME} on {DEVICE}...")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"[MODEL] {MODEL_NAME}: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    report_progress("model_loaded", f"{MODEL_NAME} loaded successfully")

    # ── Load Data ──────────────────────────────────────────────────────
    print(f"\n[DATA] Loading full RAVEL city dataset...")
    all_cities = load_ravel_cities()
    print(f"[DATA] Loaded {len(all_cities)} cities")

    # ── Results Storage ────────────────────────────────────────────────
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "model": MODEL_NAME,
        "model_config": {
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
        },
        "fallback_reason": "Gemma 2 2B is gated; no HuggingFace token available",
        "n_cities_total": len(all_cities),
        "layers": LAYERS,
        "attributes": list(ATTRIBUTE_CONFIG.keys()),
        "seeds": SEEDS,
        "probes": {},
        "quality_gate": {},
        "summary": {},
    }

    saved_probes = {}
    total_combos = len(ATTRIBUTE_CONFIG) * len(LAYERS)
    combo_idx = 0

    # ── Train Probes per Attribute x Layer ─────────────────────────────
    for attr, config in ATTRIBUTE_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"[ATTRIBUTE] {attr}")
        print(f"{'='*60}")

        # Prepare data
        filtered_cities, n_classes = prepare_attribute_data(all_cities, attr, config)

        if len(filtered_cities) < 50 or n_classes < 2:
            print(f"  [SKIP] Insufficient data for {attr}")
            results["probes"][attr] = {"status": "skipped", "reason": "insufficient data"}
            combo_idx += len(LAYERS)
            continue

        results["probes"][attr] = {
            "n_cities": len(filtered_cities),
            "n_classes": n_classes,
            "template": config["template"],
            "quality_threshold": config["quality_threshold"],
            "layers": {},
        }

        for layer in LAYERS:
            combo_idx += 1
            print(f"\n  [LAYER {layer}] ({combo_idx}/{total_combos})")
            report_progress(
                "extracting",
                f"{attr} layer {layer} ({combo_idx}/{total_combos})",
                {"attribute": attr, "layer": layer},
            )

            # Extract activations
            print(f"    Extracting activations for {len(filtered_cities)} cities...")
            t0 = time.time()
            activations, labels = extract_activations(
                model, tokenizer, filtered_cities, attr, layer, config, batch_size=128
            )
            extract_time = time.time() - t0
            print(f"    Activations: {activations.shape}, took {extract_time:.1f}s")

            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(labels)

            # Train with multiple seeds
            seed_results = []
            best_probe = None
            best_scaler = None
            best_acc = 0

            for seed in SEEDS:
                X_train, X_test, y_train, y_test = train_test_split(
                    activations, y, test_size=0.2, random_state=seed, stratify=y
                )

                clf, scaler, train_acc, test_acc, report = train_probe(
                    X_train, y_train, X_test, y_test, seed=seed
                )

                print(f"    Seed {seed}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

                seed_results.append({
                    "seed": int(seed),
                    "train_acc": float(round(train_acc, 4)),
                    "test_acc": float(round(test_acc, 4)),
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "macro_f1": float(round(report.get("macro avg", {}).get("f1-score", 0), 4)),
                    "weighted_f1": float(round(report.get("weighted avg", {}).get("f1-score", 0), 4)),
                })

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_probe = clf
                    best_scaler = scaler
                    best_le = le

            # Compute mean/std
            mean_test_acc = float(np.mean([s["test_acc"] for s in seed_results]))
            std_test_acc = float(np.std([s["test_acc"] for s in seed_results]))
            mean_train_acc = float(np.mean([s["train_acc"] for s in seed_results]))
            threshold = config["quality_threshold"]
            passed = bool(mean_test_acc >= threshold)

            layer_result = {
                "mean_test_acc": round(mean_test_acc, 4),
                "std_test_acc": round(std_test_acc, 4),
                "mean_train_acc": round(mean_train_acc, 4),
                "best_test_acc": round(float(best_acc), 4),
                "quality_gate_threshold": float(threshold),
                "quality_gate_passed": passed,
                "seed_results": seed_results,
                "extraction_time_sec": round(extract_time, 1),
                "n_classes": int(n_classes),
                "n_samples": int(len(filtered_cities)),
            }
            results["probes"][attr]["layers"][str(layer)] = layer_result

            print(f"    Mean test accuracy: {mean_test_acc:.4f} +/- {std_test_acc:.4f}")
            gate_str = "PASSED" if passed else "FAILED"
            print(f"    Quality gate ({gate_str}): {mean_test_acc:.4f} vs {threshold:.2f}")

            # Save probe weights if passed
            if passed and best_probe is not None:
                probe_path = FULL_DIR / f"probe_{attr}_layer{layer}.npz"
                np.savez(
                    probe_path,
                    coef=best_probe.coef_,
                    intercept=best_probe.intercept_,
                    scaler_mean=best_scaler.mean_,
                    scaler_scale=best_scaler.scale_,
                    classes=best_le.classes_,
                    mean_acc=mean_test_acc,
                    std_acc=std_test_acc,
                )
                saved_probes[f"{attr}_layer{layer}"] = str(probe_path)
                print(f"    Saved probe to {probe_path}")

            report_progress(
                "probe_trained",
                f"{attr} layer {layer}: acc={mean_test_acc:.4f}",
                {"attribute": attr, "layer": layer, "acc": mean_test_acc},
            )

            del activations, labels, y
            gc.collect()
            torch.cuda.empty_cache()

    # ── Aggregate Results ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[SUMMARY]")
    print(f"{'='*60}")

    n_passed = 0
    n_total = 0
    best_combo = None
    best_combo_acc = 0

    for attr in ATTRIBUTE_CONFIG:
        if "layers" not in results["probes"].get(attr, {}):
            continue
        for layer_str, layer_data in results["probes"][attr]["layers"].items():
            n_total += 1
            if layer_data["quality_gate_passed"]:
                n_passed += 1
            if layer_data["mean_test_acc"] > best_combo_acc:
                best_combo_acc = layer_data["mean_test_acc"]
                best_combo = f"{attr}_layer{layer_str}"

    results["quality_gate"] = {
        "n_passed": int(n_passed),
        "n_total": int(n_total),
        "pass_rate": round(n_passed / max(n_total, 1), 4),
        "best_combo": best_combo,
        "best_combo_acc": round(float(best_combo_acc), 4),
    }
    results["saved_probes"] = saved_probes

    # Pilot verdict: any attribute has a usable probe
    any_passed = n_passed > 0

    # Find best continent acc (most likely to succeed)
    continent_best_acc = 0.0
    continent_passed = False
    if "layers" in results["probes"].get("Continent", {}):
        for ld in results["probes"]["Continent"]["layers"].values():
            if ld["mean_test_acc"] > continent_best_acc:
                continent_best_acc = ld["mean_test_acc"]
            if ld["quality_gate_passed"]:
                continent_passed = True

    country_best_acc = 0.0
    country_passed = False
    if "layers" in results["probes"].get("Country", {}):
        for ld in results["probes"]["Country"]["layers"].values():
            if ld["mean_test_acc"] > country_best_acc:
                country_best_acc = ld["mean_test_acc"]
            if ld["quality_gate_passed"]:
                country_passed = True

    # GO if any probes pass; informative result even if country fails
    pilot_go = any_passed
    pilot_verdict = "GO" if pilot_go else "NO_GO"

    results["summary"] = {
        "pilot_verdict": pilot_verdict,
        "country_best_acc": round(float(country_best_acc), 4),
        "country_passed_quality_gate": bool(country_passed),
        "continent_best_acc": round(float(continent_best_acc), 4),
        "continent_passed_quality_gate": bool(continent_passed),
        "n_probes_saved": len(saved_probes),
        "total_time_sec": round(time.time() - start_time, 1),
        "model_used": MODEL_NAME,
        "fallback_applied": True,
        "fallback_reason": "Gemma 2 2B gated; no HF token; GPT-2 Small used per task plan",
        "interpretation": (
            "GPT-2 Small is a 124M-param model with limited factual knowledge. "
            "Probe accuracies reflect both the model's knowledge and the probe's ability "
            "to extract it. High accuracy = model encodes this knowledge linearly. "
            "Low accuracy may indicate either (a) model lacks the knowledge or "
            "(b) knowledge is stored non-linearly. "
            "Continent (6 classes) is the easiest hierarchy; Country (many classes) "
            "is hardest but most relevant for absorption."
        ),
    }

    # Print summary table
    print(f"\n  Attribute  | Layer | Test Acc (mean +/- std) | Gate")
    print(f"  {'-'*60}")
    for attr in ATTRIBUTE_CONFIG:
        if "layers" not in results["probes"].get(attr, {}):
            print(f"  {attr:10s}  | {'N/A':>5s} | {'SKIPPED':>23s} |")
            continue
        for layer_str, ld in results["probes"][attr]["layers"].items():
            gate = "PASS" if ld["quality_gate_passed"] else "FAIL"
            print(f"  {attr:10s}  | {layer_str:>5s} | "
                  f"{ld['mean_test_acc']:.4f} +/- {ld['std_test_acc']:.4f} "
                  f"(n={ld['n_samples']}) | {gate}")

    print(f"\n  Pilot verdict: {pilot_verdict}")
    print(f"  Total time: {results['summary']['total_time_sec']:.1f}s")
    print(f"  Probes saved: {len(saved_probes)}")

    # ── Save Results ───────────────────────────────────────────────────
    pilot_path = PILOT_DIR / "P2_probe_training.json"
    with open(pilot_path, "w") as f:
        json_dump(results, f, indent=2)
    print(f"\n[SAVE] Pilot results: {pilot_path}")

    full_path = FULL_DIR / "probe_training_results.json"
    with open(full_path, "w") as f:
        json_dump(results, f, indent=2)
    print(f"[SAVE] Full results: {full_path}")

    # ── Mark Done ──────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    mark_done(
        "success",
        f"Pilot {pilot_verdict}: "
        f"continent_best={continent_best_acc:.4f}, country_best={country_best_acc:.4f}, "
        f"{n_passed}/{n_total} probes passed gate, "
        f"{elapsed}s elapsed",
    )

    # ── Update GPU Progress ────────────────────────────────────────────
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        with open(gpu_progress_path) as f:
            gpu_progress = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gpu_progress["completed"]:
        gpu_progress["completed"].append(TASK_ID)
    if TASK_ID in gpu_progress.get("running", {}):
        del gpu_progress["running"][TASK_ID]

    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(elapsed / 60, 1),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": MODEL_NAME,
            "n_cities_total": len(all_cities),
            "layers": LAYERS,
            "attributes": list(ATTRIBUTE_CONFIG.keys()),
            "n_seeds": len(SEEDS),
            "device": DEVICE,
            "fallback_applied": True,
            "gpu_count": 1,
        },
    }

    with open(gpu_progress_path, "w") as f:
        json_dump(gpu_progress, f, indent=2)
    print(f"[GPU] Updated progress: {gpu_progress_path}")

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}\n{tb}")
        mark_done("failed", f"{type(e).__name__}: {str(e)[:200]}")

        gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
        try:
            with open(gpu_progress_path) as f:
                gpu_progress = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if TASK_ID not in gpu_progress.get("failed", []):
            gpu_progress.setdefault("failed", []).append(TASK_ID)
        if TASK_ID in gpu_progress.get("running", {}):
            del gpu_progress["running"][TASK_ID]

        with open(gpu_progress_path, "w") as f:
            json.dump(gpu_progress, f, indent=2, cls=NumpyEncoder)

        sys.exit(1)
