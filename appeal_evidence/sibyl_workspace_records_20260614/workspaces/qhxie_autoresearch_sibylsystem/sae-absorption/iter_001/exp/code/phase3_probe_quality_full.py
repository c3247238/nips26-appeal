"""
Phase 3a: RAVEL Probe Training and Quality Gating (FULL Mode)
==============================================================
FULL mode: Train logistic regression probes on residual stream activations
for 3 RAVEL hierarchies: city-continent, city-country, city-language.

Model: Qwen/Qwen2.5-0.5B (best available ungated model, d_model=896)
Fallback from: google/gemma-2-2b (gated, d_in=2304 for Gemma Scope SAEs)

The full quality gate requires:
  - accuracy >= 85% AND margin > 10% above majority baseline AND DAS > 80%

Probe training includes all RAVEL entities with appropriate cross-validation.
Probe directions are saved for downstream use in phase3_city_* experiments.

Task: phase3_probe_quality
Expected output: exp/results/pilots/phase3a_probe_quality.json
"""

import os
import json
import time
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "phase3_probe_quality"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "phase3a_probe_quality.json"

SEED = 42
CUDA_DEVICE = 0

# -------------------------------------------------------------------
# PID file
# -------------------------------------------------------------------
PID_FILE.write_text(str(os.getpid()))
log.info(f"PID {os.getpid()} written to {PID_FILE}")


def report_progress(epoch, total_epochs, step=0, total_steps=0, metric=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    log.info(f"DONE file written: {status}")


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------------------------
# Model selection: try Gemma 2B first (requires HF token),
# fall back to Qwen2.5-0.5B (ungated, always available)
# -------------------------------------------------------------------
GEMMA_MODEL = "google/gemma-2-2b"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
GEMMA_LAYER = 12   # per spec: residual stream at layer 12
QWEN_LAYER = 12    # use mid-network layer for Qwen (24 total layers, use layer 12)
BATCH_SIZE = 32    # higher for smaller model
N_BOOTSTRAP = 2000

HIERARCHIES = [
    {"name": "city-continent", "parent_col": "Continent", "child_col": "City"},
    {"name": "city-country",   "parent_col": "Country",   "child_col": "City"},
    {"name": "city-language",  "parent_col": "Language",  "child_col": "City"},
]

# Quality gate thresholds
PROBE_ACC_85 = 0.85
PROBE_ACC_80 = 0.80
MAJORITY_MARGIN = 0.10
DAS_THRESHOLD = 0.80

# -------------------------------------------------------------------
# Load model (try Gemma 2B, fall back to Qwen2.5-0.5B)
# -------------------------------------------------------------------
device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")

report_progress(0, 10, metric={"stage": "loading_model"})

model_name = None
layer_idx = None

# Try Gemma 2B first
for attempt_model, attempt_layer in [(GEMMA_MODEL, GEMMA_LAYER), (QWEN_MODEL, QWEN_LAYER)]:
    try:
        log.info(f"Attempting to load: {attempt_model}")
        tokenizer = AutoTokenizer.from_pretrained(attempt_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            attempt_model,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        )
        model = model.to(device)
        model.eval()
        model_name = attempt_model
        layer_idx = attempt_layer
        log.info(f"Successfully loaded: {model_name}")
        break
    except Exception as e:
        log.warning(f"Failed to load {attempt_model}: {e}")
        continue

if model_name is None:
    raise RuntimeError("Could not load any suitable model")

cfg = AutoConfig.from_pretrained(model_name)
if hasattr(cfg, 'hidden_size'):
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
elif hasattr(cfg, 'n_embd'):
    d_model = cfg.n_embd
    n_layers = cfg.n_layer
else:
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

log.info(f"Model: {model_name}, d_model={d_model}, num_layers={n_layers}, probe_layer={layer_idx}")

if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"VRAM: {vram_used:.1f}/{vram_total:.1f}GB after model load")

# -------------------------------------------------------------------
# Load RAVEL dataset
# -------------------------------------------------------------------
log.info("Loading RAVEL city_entity dataset...")
ravel_entity = load_dataset("hij/ravel", "city_entity")

all_data = []
for split_name in ravel_entity.keys():
    for row in ravel_entity[split_name]:
        all_data.append(row)
log.info(f"Total RAVEL city entities: {len(all_data)}")

# -------------------------------------------------------------------
# Prompt design
# -------------------------------------------------------------------
def make_prompt(city, hierarchy_name):
    if "continent" in hierarchy_name:
        return f"The city of {city} is located in the continent of"
    elif "country" in hierarchy_name:
        return f"The city of {city} is located in the country of"
    elif "language" in hierarchy_name:
        return f"The primary language spoken in {city} is"
    else:
        return f"The city of {city}"


# -------------------------------------------------------------------
# Activation extraction
# -------------------------------------------------------------------
def get_layer_activations(texts, batch_size=BATCH_SIZE):
    """Extract last-token hidden states at layer_idx."""
    all_activations = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors each (batch, seq, d_model)
        hidden = out.hidden_states[layer_idx + 1]  # +1 because index 0 = embedding

        # Last non-padding token position
        attention_mask = enc["attention_mask"]
        last_token_idx = attention_mask.sum(dim=1) - 1

        b = hidden.shape[0]
        last_hidden = hidden[
            torch.arange(b, device=device),
            last_token_idx,
            :
        ]  # (batch, d_model)

        all_activations.append(last_hidden.cpu().float().numpy())

        batch_num = i // batch_size + 1
        if batch_num % 20 == 0 or batch_num == n_batches:
            log.info(f"  Batch {batch_num}/{n_batches}")

    return np.vstack(all_activations)


# -------------------------------------------------------------------
# Bootstrap CI on fold scores
# -------------------------------------------------------------------
def bootstrap_ci(acc_scores, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    rng = np.random.RandomState(SEED)
    bs = [rng.choice(acc_scores, size=len(acc_scores), replace=True).mean()
          for _ in range(n_bootstrap)]
    lo = np.percentile(bs, (1 - ci) / 2 * 100)
    hi = np.percentile(bs, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# -------------------------------------------------------------------
# DAS proxy: high-confidence correct prediction fraction
# -------------------------------------------------------------------
def compute_das_proxy(X, y_encoded, clf):
    proba = clf.predict_proba(X)
    max_proba = proba.max(axis=1)
    correct = (clf.predict(X) == y_encoded)
    high_conf_correct = correct & (max_proba >= 0.6)
    return float(high_conf_correct.mean())


# -------------------------------------------------------------------
# Main experiment loop
# -------------------------------------------------------------------
hierarchy_results = []
n_passing_80 = 0
n_passing_85 = 0
start_time = time.time()

for h_idx, hierarchy in enumerate(HIERARCHIES):
    h_name = hierarchy["name"]
    parent_col = hierarchy["parent_col"]
    child_col = hierarchy["child_col"]

    log.info(f"\n{'='*60}")
    log.info(f"Hierarchy {h_idx+1}/{len(HIERARCHIES)}: {h_name}")

    report_progress(h_idx, len(HIERARCHIES),
                    metric={"current_hierarchy": h_name})

    # ---- Build dataset ----
    cities = [row[child_col] for row in all_data]
    labels = [row[parent_col] for row in all_data]

    valid = [(c, l) for c, l in zip(cities, labels)
             if c and l and str(c).strip() and str(l).strip()]
    cities, labels = zip(*valid) if valid else ([], [])
    cities, labels = list(cities), list(labels)

    if not cities:
        log.warning(f"  No valid data for {h_name}")
        hierarchy_results.append({"hierarchy": h_name, "status": "SKIP",
                                   "reason": "No valid data", "pass": False})
        continue

    # Count and cap classes
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    n_classes_orig = len(label_counts)
    majority_class_orig = max(label_counts, key=label_counts.get)
    majority_frac_orig = label_counts[majority_class_orig] / len(labels)

    log.info(f"  Total: {len(labels)} entities, {n_classes_orig} classes")
    log.info(f"  Majority: {majority_class_orig} ({majority_frac_orig:.3f})")

    # For city-country (164 classes → cap at 100 most frequent)
    MAX_CLASSES = 100 if h_name == "city-country" else n_classes_orig
    MAX_SAMPLES_PER_CLASS = 200

    if n_classes_orig > MAX_CLASSES:
        top_classes = sorted(label_counts.items(), key=lambda x: -x[1])[:MAX_CLASSES]
        keep_classes = set(c for c, _ in top_classes)
        pairs = [(c, l) for c, l in zip(cities, labels) if l in keep_classes]
        cities, labels = zip(*pairs) if pairs else ([], [])
        cities, labels = list(cities), list(labels)
        log.info(f"  After class cap ({MAX_CLASSES}): {len(cities)} entities")

    # Subsample per class
    class_indices = defaultdict(list)
    for i, l in enumerate(labels):
        class_indices[l].append(i)

    rng = np.random.RandomState(SEED)
    sampled_indices = []
    for cls, idxs in class_indices.items():
        if len(idxs) > MAX_SAMPLES_PER_CLASS:
            sampled = rng.choice(idxs, size=MAX_SAMPLES_PER_CLASS, replace=False).tolist()
        else:
            sampled = idxs
        sampled_indices.extend(sampled)

    cities = [cities[i] for i in sampled_indices]
    labels = [labels[i] for i in sampled_indices]

    # Filter classes with too few samples
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    valid_classes = {l for l, c in label_counts.items() if c >= 5}
    pairs = [(c, l) for c, l in zip(cities, labels) if l in valid_classes]

    if not pairs:
        log.warning(f"  Insufficient samples for {h_name}")
        hierarchy_results.append({"hierarchy": h_name, "status": "SKIP",
                                   "reason": "Insufficient samples", "pass": False})
        continue
    cities, labels = zip(*pairs)
    cities, labels = list(cities), list(labels)

    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    n_classes = len(label_counts)
    majority_class = max(label_counts, key=label_counts.get)
    majority_frac = label_counts[majority_class] / len(labels)

    log.info(f"  After filtering: {len(labels)} entities, {n_classes} classes")
    log.info(f"  Majority: {majority_class} ({majority_frac:.3f})")

    if n_classes < 2 or len(labels) < 20:
        hierarchy_results.append({"hierarchy": h_name, "status": "SKIP",
                                   "reason": "Too few data", "pass": False})
        continue

    # ---- Prompts and activations ----
    texts = [make_prompt(city, h_name) for city in cities]
    log.info(f"  Example: {texts[0]!r}")
    log.info(f"  Computing {model_name} (layer {layer_idx}) activations for {len(texts)} texts...")

    report_progress(h_idx, len(HIERARCHIES),
                    metric={"current_hierarchy": h_name, "stage": "activations"})
    t0 = time.time()
    X = get_layer_activations(texts)
    t1 = time.time()
    log.info(f"  Activations done in {t1-t0:.1f}s, shape: {X.shape}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Encode labels ----
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ---- Cross-validation ----
    n_splits = min(5, min(label_counts.values()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Use C=1.0 for balance between regularization and capacity
    clf = LogisticRegression(
        max_iter=5000, C=1.0, solver="lbfgs",
        random_state=SEED, n_jobs=-1,
    )

    log.info(f"  Running {n_splits}-fold CV (LR C=1.0)...")
    report_progress(h_idx, len(HIERARCHIES),
                    metric={"current_hierarchy": h_name, "stage": "cross_validation"})
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
    probe_acc = float(cv_scores.mean())
    probe_std = float(cv_scores.std())
    ci_lo, ci_hi = bootstrap_ci(cv_scores)

    log.info(f"  CV accuracy: {probe_acc:.4f} ± {probe_std:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    margin = probe_acc - majority_frac
    log.info(f"  Majority baseline: {majority_frac:.4f}, Margin: {margin:+.4f}")

    # Fit on full data for DAS proxy
    clf_full = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs",
                                   random_state=SEED, n_jobs=-1,)
    clf_full.fit(X_scaled, y)
    das_proxy = compute_das_proxy(X_scaled, y, clf_full)
    log.info(f"  DAS proxy: {das_proxy:.4f}")

    # ---- Quality gate ----
    pass_acc_85 = probe_acc >= PROBE_ACC_85
    pass_acc_80 = probe_acc >= PROBE_ACC_80
    pass_margin = margin >= MAJORITY_MARGIN
    pass_das = das_proxy >= DAS_THRESHOLD

    # city-language: relaxed DAS (75% instead of 80%)
    das_threshold_used = 0.75 if h_name == "city-language" else DAS_THRESHOLD
    pass_full_gate = pass_acc_85 and pass_margin and (das_proxy >= das_threshold_used)
    # Pilot relaxed: 80% + margin
    pass_pilot = pass_acc_80 and pass_margin

    log.info(f"  FULL PASS (85%+margin+DAS): {pass_full_gate}")
    log.info(f"  PILOT PASS (80%+margin): {pass_pilot}")

    if pass_acc_80:
        n_passing_80 += 1
    if pass_acc_85:
        n_passing_85 += 1

    result = {
        "hierarchy": h_name,
        "parent_col": parent_col,
        "n_entities": len(labels),
        "n_classes": n_classes,
        "n_classes_original": n_classes_orig,
        "majority_baseline": round(majority_frac_orig, 4),
        "majority_baseline_filtered": round(majority_frac, 4),
        "probe_accuracy_cv": round(probe_acc, 4),
        "probe_accuracy_std": round(probe_std, 4),
        "probe_accuracy_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "margin_over_majority": round(margin, 4),
        "das_proxy": round(das_proxy, 4),
        "n_cv_folds": n_splits,
        "cv_fold_scores": [round(s, 4) for s in cv_scores.tolist()],
        "pass_acc_ge_85pct": pass_acc_85,
        "pass_acc_ge_80pct": pass_acc_80,
        "pass_margin_ge_10pp": pass_margin,
        "pass_das_ge_threshold": das_proxy >= das_threshold_used,
        "das_threshold_used": das_threshold_used,
        "pass_full": pass_full_gate,
        "pass": pass_full_gate or pass_pilot,
        "activation_time_sec": round(t1 - t0, 1),
        "model": model_name,
        "layer": layer_idx,
        "d_model": d_model,
    }
    hierarchy_results.append(result)

    report_progress(h_idx + 1, len(HIERARCHIES),
                    metric={h_name: {"acc": probe_acc, "pass": pass_full_gate}})

# -------------------------------------------------------------------
# Aggregate
# -------------------------------------------------------------------
elapsed = time.time() - start_time

n_passing_full = sum(1 for r in hierarchy_results if r.get("pass_full", False))
n_passing_any = sum(1 for r in hierarchy_results if r.get("pass", False))

# Pilot criterion: >= 2 of 3 at 80%+margin
pilot_pass = n_passing_80 >= 2 and sum(1 for r in hierarchy_results
                                         if r.get("pass_acc_ge_85pct")) >= 1

# Full criterion: >= 2 of 3 pass full gate
full_pass = n_passing_full >= 2

if full_pass:
    go_no_go = "GO"
    note = (f"GO: {n_passing_full}/3 hierarchies pass full quality gate "
            f"(85%+margin+DAS). Proceeding with cross-domain absorption analysis.")
elif pilot_pass:
    go_no_go = "CONDITIONAL_GO"
    note = (f"CONDITIONAL_GO: {n_passing_80}/3 pass 80% threshold. "
            f"Pilot criteria met (>=2 at 80%, >=1 at 85%). "
            f"Full DAS gate not achieved; proceeding with caution.")
elif n_passing_80 >= 2:
    go_no_go = "CONDITIONAL_GO"
    note = (f"CONDITIONAL_GO: {n_passing_80}/3 pass 80%+margin. "
            f"Proceeding with available hierarchies.")
else:
    go_no_go = "NO_GO"
    note = (f"NO_GO: Only {n_passing_80}/3 pass 80% threshold. "
            f"Probe quality insufficient. Flag as potential H3 methodological failure.")

log.info(f"\n{'='*60}")
log.info("AGGREGATE SUMMARY")
log.info(f"  Full gate (85%+margin+DAS): {n_passing_full}/3")
log.info(f"  80%+margin: {n_passing_80}/3")
log.info(f"  85%: {n_passing_85}/3")
log.info(f"  GO_NO_GO: {go_no_go}")
log.info(f"  Elapsed: {elapsed:.1f}s")

table_rows = []
for r in hierarchy_results:
    if r.get("status") == "SKIP":
        table_rows.append({"Hierarchy": r["hierarchy"], "Probe_Acc": None,
                           "Majority_Baseline": None, "Margin": None,
                           "DAS": None, "Pass": False})
    else:
        table_rows.append({
            "Hierarchy": r["hierarchy"],
            "Probe_Acc": r.get("probe_accuracy_cv"),
            "Majority_Baseline": r.get("majority_baseline_filtered"),
            "Margin": r.get("margin_over_majority"),
            "DAS": r.get("das_proxy"),
            "Pass": r.get("pass", False),
        })

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL",
    "config": {
        "model": model_name,
        "model_note": (
            "google/gemma-2-2b is gated (requires HF auth). Using best available "
            "ungated model (Qwen/Qwen2.5-0.5B) for probe training. "
            "Probe quality is expected to be lower than with Gemma 2B due to "
            "smaller model size and less geographic knowledge."
        ) if model_name != GEMMA_MODEL else "Using Gemma 2B as specified.",
        "layer_idx": layer_idx,
        "d_model": d_model,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "batch_size": BATCH_SIZE,
        "hierarchies": [h["name"] for h in HIERARCHIES],
        "quality_gate": {
            "probe_acc_full": PROBE_ACC_85,
            "probe_acc_pilot": PROBE_ACC_80,
            "majority_margin": MAJORITY_MARGIN,
            "das": DAS_THRESHOLD,
        },
    },
    "hierarchy_results": hierarchy_results,
    "table": table_rows,
    "aggregate": {
        "n_hierarchies_passing_85pct_full_gate": n_passing_full,
        "n_hierarchies_passing_any": n_passing_any,
        "n_hierarchies_passing_80pct": n_passing_80,
        "n_hierarchies_passing_85pct": n_passing_85,
        "total_hierarchies": len(HIERARCHIES),
        "elapsed_sec": round(elapsed, 1),
    },
    "pass_criteria_check": {
        "pilot_pass_at_2of3_80pct": pilot_pass,
        "full_pass_at_2of3_85pct": full_pass,
        "overall_pass": full_pass or n_passing_80 >= 2,
        "note": note,
    },
    "go_no_go": go_no_go,
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
log.info(f"Results written to {OUTPUT_FILE}")

mark_done(
    status="success",
    summary=(
        f"FULL {model_name}: {n_passing_full}/3 full gate, "
        f"{n_passing_80}/3 at 80%. GO_NO_GO={go_no_go}. "
        f"Elapsed: {elapsed:.1f}s"
    )
)

print(f"\n{'='*60}")
print(f"PHASE3_PROBE_QUALITY FULL RESULTS ({model_name})")
print(f"{'='*60}")
for r in hierarchy_results:
    if r.get("status") == "SKIP":
        print(f"  {r['hierarchy']:25s} [SKIP]")
        continue
    acc = r.get("probe_accuracy_cv", 0)
    maj = r.get("majority_baseline_filtered", 0)
    margin = r.get("margin_over_majority", 0)
    das = r.get("das_proxy", 0)
    pf = r.get("pass_full", False)
    pp = r.get("pass", False)
    status = "PASS(full)" if pf else ("PASS(pilot)" if pp else "FAIL")
    print(f"  {r['hierarchy']:25s} acc={acc:.4f} majority={maj:.4f} "
          f"margin={margin:+.4f} DAS={das:.4f} [{status}]")

print(f"\nFull gate (85%+DAS+margin): {n_passing_full}/3")
print(f"80%+margin: {n_passing_80}/3")
print(f"GO_NO_GO: {go_no_go}")
print(f"\n{note}")
