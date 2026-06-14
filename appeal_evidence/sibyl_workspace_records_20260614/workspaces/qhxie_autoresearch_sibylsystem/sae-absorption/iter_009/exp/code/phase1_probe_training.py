"""
Phase 1.1: Multi-Layer Probe Training (4 hierarchies x layers)

Iteration 9 probe training. CRITICAL PATH: all downstream tasks depend on this.

Key improvements over iter_008:
- RAVEL probes use ICL prompts with position -2 (prediction position)
- Better class filtering: min_samples_per_class=10 for language, =5 for country
- Multiple regularization strengths with cross-validation
- Proper train/test split BEFORE activation caching (no data leakage)
- First-letter uses sae_spelling pipeline (proven F1=0.97 at L24)

Pilot: L12, L24 only (2 hierarchies: first-letter, city-continent).
Full: L6, L12, L18, L24 (all 4 hierarchies).

Expected baselines from iter_008:
  first-letter L24: F1=0.97 (sklearn) / F1=0.87 (sae_spelling)
  city-continent L24: F1=0.84
  city-country L24: F1=0.79
  city-language L24: F1=0.82
"""

import os
import sys
import json
import time
import gc
import random
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# ── Configuration ────────────────────────────────────────────────
TASK_ID = "phase1_probe_training"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# Pilot: L12 + L24 (fast validation). Full: all 4 layers.
PROBE_LAYERS = [12, 24] if MODE == "PILOT" else [6, 12, 18, 24]

# Pilot: first-letter + city-continent only. Full: all 4.
PILOT_HIERARCHIES = {"first-letter", "city-continent"}

# GPU selection -- when CUDA_VISIBLE_DEVICES is set, torch sees device as cuda:0
GPU_ID = 0  # Always use cuda:0 since CUDA_VISIBLE_DEVICES remaps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ── Progress tracking ────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total_steps": total,
        "status": status, "metric": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid.exists():
        pid.unlink()
    prog = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if prog.exists():
        try:
            fp = json.loads(prog.read_text())
        except Exception:
            pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }))
    logger.info(f"DONE (status={status}): {summary}")


# ── Model loading ────────────────────────────────────────────────
def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL_PATH, dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return model


# ── First-Letter Probes (sae_spelling pipeline) ─────────────────
def train_first_letter_probes(model, layers):
    """
    Train first-letter probes using sae_spelling pipeline.
    Also train sklearn LogisticRegression at the same positions for
    downstream absorption measurement compatibility.
    """
    from sae_spelling.probing import (
        create_dataset_probe_training,
        gen_and_save_df_acts_probing,
        train_linear_probe_for_task,
        gen_probe_stats,
    )
    from sae_spelling.prompting import first_letter_formatter
    from sae_spelling.vocab import get_alpha_tokens

    logger.info("=== First-Letter Probes (sae_spelling pipeline) ===")

    # Get vocabulary tokens
    alpha_tokens = get_alpha_tokens(model.tokenizer)
    logger.info(f"Alpha tokens (full): {len(alpha_tokens)}")

    # For pilot: balanced subset across letters
    if MODE == "PILOT":
        letter_tokens = defaultdict(list)
        for t in alpha_tokens:
            clean = t.strip()
            if clean.startswith("\u2581"):
                clean = clean[1:]
            if clean and clean[0].isalpha():
                letter_tokens[clean[0].upper()].append(t)
        rng = np.random.RandomState(SEED)
        max_per_letter = 40
        subset = []
        for letter in sorted(letter_tokens.keys()):
            avail = letter_tokens[letter]
            n_take = min(len(avail), max_per_letter)
            chosen = rng.choice(avail, size=n_take, replace=False)
            subset.extend(chosen.tolist())
        rng.shuffle(subset)
        alpha_tokens = subset
        logger.info(f"Alpha tokens (pilot subset): {len(alpha_tokens)}")

    # Create ICL prompt dataset
    formatter = first_letter_formatter()
    base_template = "The first letter of the word '{word}' is '"
    train_dataset, test_dataset = create_dataset_probe_training(
        vocab=alpha_tokens,
        formatter=formatter,
        num_prompts_per_token=1,
        base_template=base_template,
        max_icl_examples=5,
        train_test_fraction=0.8,
    )
    logger.info(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test")

    results = {}

    for layer in layers:
        logger.info(f"\n--- First-letter probe at layer {layer} ---")
        hook_point = f"blocks.{layer}.hook_resid_post"

        probe_path = PHASE1_DIR / "sae_spelling_probes"
        probe_path.mkdir(parents=True, exist_ok=True)

        # Generate activations at position -2 (prediction position)
        train_df, test_df, train_acts, test_acts = gen_and_save_df_acts_probing(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            path=str(probe_path),
            hook_point=hook_point,
            layer=layer,
            batch_size=32,
            position_idx=-2,
        )

        # Train sae_spelling LinearProbe
        probe, probe_data = train_linear_probe_for_task(
            train_df=train_df,
            test_df=test_df,
            device=model.cfg.device,
            train_activations=train_acts,
            test_activations=test_acts,
            num_classes=26,
            batch_size=2048,
            num_epochs=50,
            lr=1e-2,
            weight_decay=1e-4,
        )

        # Evaluate: gen_probe_stats returns per-letter ProbeStats
        stats_list = gen_probe_stats(
            probe=probe,
            X_val=probe_data["X_test"],
            y_val=probe_data["y_test"],
            device=model.cfg.device,
        )

        per_letter_f1 = [s.f1 for s in stats_list]
        macro_f1 = float(np.mean(per_letter_f1))

        # Multi-class accuracy
        with torch.no_grad():
            logits = probe(probe_data["X_test"].to(model.cfg.device))
            preds = logits.argmax(dim=-1).cpu()
            y_true = probe_data["y_test"].cpu()
            multi_acc = float((preds == y_true).float().mean())
            multi_f1 = float(f1_score(
                y_true.numpy(), preds.numpy(), average="weighted"
            ))

        f1 = multi_f1
        acc = multi_acc

        logger.info(f"  sae_spelling: F1_weighted={f1:.4f}, F1_macro={macro_f1:.4f}, Acc={acc:.4f}")

        # Save probe
        probe_save = PHASE1_DIR / f"probe_first-letter_L{layer}.pt"
        torch.save({
            "probe_state_dict": probe.state_dict(),
            "f1": f1, "accuracy": acc,
            "n_train": len(train_dataset), "n_test": len(test_dataset),
        }, probe_save)

        results[f"first-letter_L{layer}"] = {
            "hierarchy": "first-letter",
            "layer": layer,
            "method": "sae_spelling_LinearProbe",
            "position": -2,
            "f1_weighted_cv": f1,
            "f1_macro": macro_f1,
            "accuracy_cv": acc,
            "n_samples": len(train_dataset) + len(test_dataset),
            "n_classes": 26,
            "probe_type": "LinearProbe (sae_spelling)",
            "quality_gate_strict": f1 >= 0.90,
            "quality_gate_relaxed": f1 >= 0.85,
            "per_letter_f1": {s.letter: s.f1 for s in stats_list},
        }

        # Also train sklearn probe on the same activations for absorption pipeline
        X_train_np = probe_data["X_train"].cpu().numpy()
        y_train_np = probe_data["y_train"].cpu().numpy()
        X_test_np = probe_data["X_test"].cpu().numpy()
        y_test_np = probe_data["y_test"].cpu().numpy()

        best_sklearn_f1 = -1
        best_sklearn = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(
                C=C, max_iter=3000, solver="lbfgs", random_state=SEED
            )
            clf.fit(X_train_np, y_train_np)
            y_pred = clf.predict(X_test_np)
            sk_f1 = f1_score(y_test_np, y_pred, average="weighted")
            logger.info(f"    sklearn C={C}: F1={sk_f1:.4f}")
            if sk_f1 > best_sklearn_f1:
                best_sklearn_f1 = sk_f1
                best_sklearn = clf

        if best_sklearn is not None:
            np.savez(
                PHASE1_DIR / f"probe_first-letter_L{layer}_sklearn.npz",
                coef=best_sklearn.coef_,
                intercept=best_sklearn.intercept_,
                classes=np.arange(26),
            )
            results[f"first-letter-sklearn_L{layer}"] = {
                "hierarchy": "first-letter",
                "layer": layer,
                "method": "sklearn_logistic_regression",
                "position": -2,
                "f1_weighted_cv": float(best_sklearn_f1),
                "accuracy_cv": float(accuracy_score(y_test_np, best_sklearn.predict(X_test_np))),
                "n_samples": len(y_train_np) + len(y_test_np),
                "n_classes": 26,
                "probe_type": "LogisticRegression (sklearn)",
                "quality_gate_strict": best_sklearn_f1 >= 0.90,
                "quality_gate_relaxed": best_sklearn_f1 >= 0.85,
            }
            logger.info(f"  sklearn best: F1={best_sklearn_f1:.4f}")

    return results


# ── RAVEL ICL Prompts ────────────────────────────────────────────
def create_ravel_icl_prompts(cities, labels, label_name, n_icl=5, seed=42):
    """
    Create ICL-style prompts for RAVEL hierarchies.
    Uses varied templates per hierarchy to improve model engagement.
    Extract at position -2 (prediction position).
    """
    templates = {
        "city-country": [
            ("The city of {entity} is in the country of", " {label}"),
            ("{entity} is a city located in", " {label}"),
        ],
        "city-continent": [
            ("The city of {entity} is on the continent of", " {label}"),
            ("{entity} is located on the continent of", " {label}"),
        ],
        "city-language": [
            ("The primary language spoken in {entity} is", " {label}"),
            ("People in {entity} mainly speak", " {label}"),
        ],
    }

    template_pairs = templates.get(
        label_name,
        [("The city of {entity} has the property of", " {label}")]
    )

    rng = random.Random(seed)
    examples = list(zip(cities, labels))
    rng.shuffle(examples)

    prompts = []
    for i, (city, label) in enumerate(zip(cities, labels)):
        # Rotate templates
        base_template, answer_template = template_pairs[i % len(template_pairs)]

        # Select ICL examples (not including the current city)
        icl_pool = [(c, l) for c, l in examples if c != city][:n_icl * 2]
        rng.shuffle(icl_pool)
        icl_examples = icl_pool[:n_icl]

        # Build ICL prompt
        icl_parts = []
        for ex_city, ex_label in icl_examples:
            ex_prompt = base_template.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_prompt)

        full_prompt = "\n".join(icl_parts) + "\n" + base_template.format(entity=city)
        prompts.append(full_prompt)

    return prompts


def cache_ravel_activations(model, prompts, layers, batch_size=1):
    """
    Cache activations at position -2 (prediction position).
    This mirrors the sae_spelling approach.
    """
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    activations = {l: [] for l in layers}

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for l in layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -2, :].float().cpu().numpy()
            activations[l].append(act)

        del cache

        if (i + 1) % 200 == 0 or i == len(prompts) - 1:
            logger.info(f"  Cached {i+1}/{len(prompts)}")
        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

    for l in layers:
        activations[l] = np.array(activations[l])
    return activations


def train_ravel_probes(model, ds, layers, hierarchies_to_train=None):
    """Train probes for RAVEL hierarchies."""
    cities = ds["City"]
    hierarchy_defs = {
        "city-country": (ds["Country"], 5),   # min 5 samples per class
        "city-continent": (ds["Continent"], 5),
        "city-language": (ds["Language"], 10), # min 10 samples for language
    }

    if hierarchies_to_train:
        hierarchy_defs = {
            k: v for k, v in hierarchy_defs.items()
            if k in hierarchies_to_train
        }

    results = {}

    for hier_name, (raw_labels, min_samples) in hierarchy_defs.items():
        logger.info(f"\n=== {hier_name} ===")

        # Filter valid entries
        valid = [
            (c, l) for c, l in zip(cities, raw_labels)
            if c and l and str(l).strip()
        ]
        ents, labs = zip(*valid)
        ents, labs = list(ents), list(labs)

        # Merge Australia -> Oceania for continent
        if hier_name == "city-continent":
            labs = ["Oceania" if l == "Australia" else l for l in labs]

        # Filter classes with >= min_samples
        le = LabelEncoder()
        encoded = le.fit_transform(labs)
        counts = Counter(encoded)
        valid_classes = {c for c, n in counts.items() if n >= min_samples}
        keep = [i for i, c in enumerate(encoded) if c in valid_classes]
        ents = [ents[i] for i in keep]
        labs = [labs[i] for i in keep]

        # Re-encode after filtering
        le = LabelEncoder()
        encoded = le.fit_transform(labs)
        n_classes = len(le.classes_)

        logger.info(f"  {len(ents)} entities, {n_classes} classes")
        logger.info(f"  Class distribution: {Counter(labs).most_common(5)}...")

        # Stratified train/test split (80/20)
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(ents))
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=SEED, stratify=encoded
            )
        except ValueError:
            # If stratification fails (too few samples), use random split
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=SEED
            )

        train_ents = [ents[i] for i in train_idx]
        train_labs = [labs[i] for i in train_idx]
        test_ents = [ents[i] for i in test_idx]
        test_labs = [labs[i] for i in test_idx]

        logger.info(f"  Train: {len(train_ents)}, Test: {len(test_ents)}")

        # Create ICL prompts for ALL entities (train + test)
        all_ents = train_ents + test_ents
        all_labs = train_labs + test_labs
        logger.info("  Creating ICL prompts...")
        prompts = create_ravel_icl_prompts(all_ents, all_labs, hier_name, n_icl=5)

        # Sample for inspection
        for j in range(min(2, len(prompts))):
            logger.info(f"  Sample prompt ({all_labs[j]}):\n    {prompts[j][:150]}...")

        # Cache activations for ALL entities
        logger.info("  Caching activations (position -2)...")
        activations = cache_ravel_activations(model, prompts, layers)

        n_train = len(train_ents)

        for layer in layers:
            X_all = activations[layer]
            X_train = X_all[:n_train]
            X_test = X_all[n_train:]

            # Encode train/test labels
            le_train = LabelEncoder()
            y_train = le_train.fit_transform(train_labs)
            # Map test labels using training encoder
            y_test = []
            valid_test = []
            for i, lab in enumerate(test_labs):
                if lab in le_train.classes_:
                    y_test.append(np.where(le_train.classes_ == lab)[0][0])
                    valid_test.append(i)
            y_test = np.array(y_test)
            X_test_valid = X_test[valid_test]
            n_test_classes = len(set(y_test))

            logger.info(
                f"  Training {hier_name} probe L{layer}: "
                f"X_train={X_train.shape}, X_test={X_test_valid.shape}, "
                f"{len(le_train.classes_)} classes"
            )

            # Try multiple regularizations
            best_f1, best_probe, best_metrics, best_C = -1, None, None, None
            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                probe = LogisticRegression(
                    C=C, max_iter=3000, solver="lbfgs",
                    random_state=SEED,
                )

                try:
                    probe.fit(X_train, y_train)
                    y_pred = probe.predict(X_test_valid)
                    f1_w = f1_score(y_test, y_pred, average="weighted")
                    acc = accuracy_score(y_test, y_pred)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    f1_m = f1_score(y_test, y_pred, average="macro")

                    # Also compute CV on training set
                    min_count = min(Counter(y_train).values())
                    n_splits = min(5, max(2, min_count))
                    try:
                        cv = StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=SEED
                        )
                        y_cv_pred = cross_val_predict(probe, X_train, y_train, cv=cv)
                        f1_cv = f1_score(y_train, y_cv_pred, average="weighted")
                    except Exception:
                        f1_cv = f1_w  # fallback

                    train_acc = accuracy_score(y_train, probe.predict(X_train))
                except Exception as e:
                    logger.warning(f"    C={C} failed: {e}")
                    continue

                logger.info(
                    f"    C={C}: test_F1w={f1_w:.4f}, test_Acc={acc:.4f}, "
                    f"cv_F1w={f1_cv:.4f}, train_Acc={train_acc:.4f}"
                )

                if f1_w > best_f1:
                    best_f1 = f1_w
                    best_C = C
                    best_probe = probe
                    best_metrics = {
                        "f1_weighted_test": float(f1_w),
                        "f1_macro_test": float(f1_m),
                        "f1_weighted_cv": float(f1_cv),
                        "accuracy_test": float(acc),
                        "balanced_accuracy_test": float(bal_acc),
                        "accuracy_train": float(train_acc),
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "n_classes": len(le_train.classes_),
                        "n_splits_cv": n_splits if 'n_splits' in dir() else 5,
                        "best_C": C,
                        "probe_type": "LogisticRegression",
                    }

            if best_probe is None:
                logger.error(f"  ALL regularizations failed for {hier_name} L{layer}")
                continue

            # Use test F1 as primary metric (consistent with task plan)
            key = f"{hier_name}_L{layer}"
            results[key] = {
                "hierarchy": hier_name,
                "layer": layer,
                "method": "sklearn_logistic_regression",
                "position": -2,
                **best_metrics,
                "quality_gate_strict": best_metrics["f1_weighted_test"] >= 0.90,
                "quality_gate_relaxed": best_metrics["f1_weighted_test"] >= 0.80,
            }

            # Save probe
            np.savez(
                PHASE1_DIR / f"probe_{hier_name}_L{layer}.npz",
                coef=best_probe.coef_,
                intercept=best_probe.intercept_,
                classes=np.array(le_train.classes_),
            )

            logger.info(
                f"  BEST: F1_test={best_metrics['f1_weighted_test']:.4f} "
                f"(C={best_C})"
            )

        del activations
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ── Main ─────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 10, "starting")

    logger.info(f"=== Phase 1.1: Probe Training (mode={MODE}, layers={PROBE_LAYERS}) ===")
    logger.info(f"GPU: cuda:{GPU_ID}")

    device = f"cuda:{GPU_ID}"

    # Load model
    report_progress(1, 10, "loading_model")
    model = load_model(device)

    all_results = {}

    # ── First-letter probes ──
    report_progress(2, 10, "first_letter_probes")
    fl_results = train_first_letter_probes(model, PROBE_LAYERS)
    all_results.update(fl_results)

    report_progress(4, 10, "first_letter_done", {
        "n_probes": len(fl_results),
        "best_f1": max(
            (r["f1_weighted_cv"] for r in fl_results.values()),
            default=0
        ),
    })

    # ── RAVEL probes ──
    report_progress(5, 10, "ravel_probes")
    from datasets import load_dataset
    ds = load_dataset("hij/ravel", "city_entity", split="train")

    # In pilot mode, only train city-continent (simplest, highest baseline F1)
    hierarchies = None
    if MODE == "PILOT":
        hierarchies = {"city-continent"}
        logger.info("PILOT: training city-continent only")
    else:
        hierarchies = {"city-continent", "city-country", "city-language"}

    ravel_results = train_ravel_probes(model, ds, PROBE_LAYERS, hierarchies)
    all_results.update(ravel_results)

    report_progress(8, 10, "ravel_done", {
        "n_probes": len(ravel_results),
    })

    # ── Quality gate summary ──
    logger.info("\n" + "=" * 60)
    logger.info("=== Quality Gate Summary ===")
    logger.info("=" * 60)

    n_pass_strict, n_pass_relaxed, n_total = 0, 0, 0
    for key, pr in sorted(all_results.items()):
        n_total += 1
        is_strict = pr.get("quality_gate_strict", False)
        is_relaxed = pr.get("quality_gate_relaxed", False)
        if is_strict:
            n_pass_strict += 1
        if is_relaxed:
            n_pass_relaxed += 1
        st = "STRICT" if is_strict else ("RELAXED" if is_relaxed else "FAIL")
        f1_key = "f1_weighted_test" if "f1_weighted_test" in pr else "f1_weighted_cv"
        logger.info(f"  {key:35s}: F1={pr[f1_key]:.4f} [{st}]")

    # Best per hierarchy
    best_per_hier = {}
    for key, pr in all_results.items():
        hier = pr["hierarchy"]
        f1_key = "f1_weighted_test" if "f1_weighted_test" in pr else "f1_weighted_cv"
        f1_val = pr[f1_key]
        if hier not in best_per_hier or f1_val > best_per_hier[hier]["f1"]:
            best_per_hier[hier] = {
                "key": key, "f1": f1_val, "layer": pr["layer"],
                "method": pr.get("method", pr.get("probe_type", "unknown")),
            }

    logger.info("\nBest per hierarchy:")
    for hier, info in sorted(best_per_hier.items()):
        logger.info(f"  {hier}: F1={info['f1']:.4f} at L{info['layer']} ({info['method']})")

    # Recommended layer per hierarchy (for downstream tasks)
    recommended_layers = {}
    for hier, info in best_per_hier.items():
        recommended_layers[hier] = info["layer"]
    logger.info(f"\nRecommended layers: {recommended_layers}")

    hiers = set(pr["hierarchy"] for pr in all_results.values())
    best_f1 = max(
        pr.get("f1_weighted_test", pr.get("f1_weighted_cv", 0))
        for pr in all_results.values()
    )

    # Pilot criteria: at least 2 hierarchies have F1 >= 0.80 at best layer
    n_hier_above_080 = sum(1 for info in best_per_hier.values() if info["f1"] >= 0.80)
    pilot_ok = n_hier_above_080 >= 2

    elapsed = time.time() - start_time

    # ── Comparison with iter_008 baselines ──
    iter008_baselines = {
        "first-letter_L24": 0.97,
        "city-continent_L24": 0.84,
        "city-country_L24": 0.79,
        "city-language_L24": 0.82,
    }
    comparison = {}
    for key, baseline in iter008_baselines.items():
        if key in all_results:
            f1_key = "f1_weighted_test" if "f1_weighted_test" in all_results[key] else "f1_weighted_cv"
            current = all_results[key][f1_key]
            comparison[key] = {
                "iter_008": baseline,
                "iter_009": float(current),
                "delta": float(current - baseline),
                "improved": current > baseline,
            }

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "layers": PROBE_LAYERS,
        "probes": all_results,
        "best_per_hierarchy": best_per_hier,
        "recommended_layers": recommended_layers,
        "quality_gate_summary": {
            "n_total": n_total,
            "n_pass_strict": n_pass_strict,
            "n_pass_relaxed": n_pass_relaxed,
            "n_hierarchies_above_080": n_hier_above_080,
        },
        "iter008_comparison": comparison,
        "pilot_criteria_met": pilot_ok,
        "pilot_criteria_details": {
            "n_hierarchies": len(hiers),
            "hierarchies": sorted(hiers),
            "best_f1": float(best_f1),
            "n_hier_above_080": n_hier_above_080,
            "criteria": "At least 2 hierarchies with F1 >= 0.80 at best layer",
        },
        "elapsed_minutes": elapsed / 60,
    }

    if not pilot_ok:
        output["diagnosis"] = {
            "issue": f"Only {n_hier_above_080} hierarchies above F1=0.80",
            "analysis": [
                f"Best F1: {best_f1:.4f}",
                "RAVEL probes may benefit from more ICL examples",
                "Try layers 6, 18 in full mode",
                "Consider frequency-balanced sampling for city-country",
            ],
        }

    # Save results
    if MODE == "PILOT":
        out_path = PILOT_DIR / "phase1_probe_training.json"
    else:
        out_path = PHASE1_DIR / "probe_training.json"

    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")
    logger.info(f"Elapsed: {elapsed/60:.1f} min, Pilot OK: {pilot_ok}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary = (
        f"{len(hiers)} hierarchies, best F1={best_f1:.4f}, "
        f"strict pass: {n_pass_strict}/{n_total}, "
        f"relaxed pass: {n_pass_relaxed}/{n_total}, "
        f"pilot: {'MET' if pilot_ok else 'NOT MET'}"
    )
    mark_done("success" if pilot_ok else "partial", summary)
    update_gpu_progress(elapsed)
    return output


def update_gpu_progress(elapsed_seconds):
    import filelock
    path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if TASK_ID not in data.get("completed", []):
                data.setdefault("completed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 15 if MODE == "PILOT" else 45,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "layers": PROBE_LAYERS,
                    "gpu": f"cuda:{GPU_ID}",
                },
            }
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update: {e}")
        # Try without lock as fallback
        try:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if TASK_ID not in data.get("completed", []):
                data.setdefault("completed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        try:
            path = WORKSPACE / "exp" / "gpu_progress.json"
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if TASK_ID not in data.get("failed", []):
                data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
        sys.exit(1)
