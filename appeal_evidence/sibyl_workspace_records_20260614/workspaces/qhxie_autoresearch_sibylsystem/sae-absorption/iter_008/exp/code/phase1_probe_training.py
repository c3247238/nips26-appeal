"""
Phase 1.1: Train Cross-Domain Linear Probes on Gemma 2 2B

Trains linear probes for 4 hierarchy types on Gemma 2 2B residual stream.

Key methodological choices:
- First-letter: Use sae_spelling's ICL prompt pipeline for direct comparison
  with Chanin et al. Extract at position -2 (the prediction position).
- RAVEL hierarchies: Use ICL-style prompts that elicit factual knowledge.
  Extract at position -2 (right before the model predicts the answer).
- Train neural LinearProbe (via sae_spelling) for first-letter.
- Train logistic regression for RAVEL (simpler, works well for factual probes).

Pilot: Layer 12 only. Full: Layers 6, 12, 18, 24.
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
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configuration
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

PROBE_LAYERS = [12] if MODE == "PILOT" else [6, 12, 18, 24]

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# PID/Progress/Done
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
    if pid.exists(): pid.unlink()
    prog = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if prog.exists():
        try: fp = json.loads(prog.read_text())
        except: pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }))
    logger.info(f"DONE (status={status})")


def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL_PATH, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model; gc.collect(); torch.cuda.empty_cache()
    return model


# ============================================================
# First-Letter Probes (using sae_spelling methodology)
# ============================================================
def train_first_letter_probes(model, layers):
    """
    Train first-letter probes using the sae_spelling pipeline.
    This ensures direct comparability with Chanin et al.
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

    # For pilot: use a subset balanced across letters to keep it fast
    # For full: use all tokens
    if MODE == "PILOT":
        from collections import defaultdict
        letter_tokens = defaultdict(list)
        for t in alpha_tokens:
            clean = t.strip()
            if clean.startswith("▁"):
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
    # first_letter_formatter() returns a callable formatter (functools.partial)
    formatter = first_letter_formatter()
    base_template = "The first letter of the word '{word}' is '"
    train_dataset, test_dataset = create_dataset_probe_training(
        vocab=alpha_tokens,
        formatter=formatter,
        num_prompts_per_token=1,  # 1 prompt per word for pilot speed
        base_template=base_template,
        max_icl_examples=5,
        train_test_fraction=0.8,
    )
    logger.info(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test")

    # Sample prompts for inspection
    for i in range(min(3, len(train_dataset))):
        prompt, cls = train_dataset[i]
        logger.info(f"  Sample: '{prompt.base[:80]}...' -> class={cls} ({prompt.answer})")

    results = {}
    for layer in layers:
        logger.info(f"\n--- First-letter probe at layer {layer} ---")
        hook_point = f"blocks.{layer}.hook_resid_post"

        # Generate activations (position_idx=-2 = prediction position)
        probe_path = PHASE1_DIR / "sae_spelling_probes"
        probe_path.mkdir(parents=True, exist_ok=True)

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

        # Train probe
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

        # Evaluate: gen_probe_stats returns list of per-letter ProbeStats
        stats_list = gen_probe_stats(
            probe=probe,
            X_val=probe_data["X_test"],
            y_val=probe_data["y_test"],
            device=model.cfg.device,
        )

        # Aggregate per-letter F1 (macro average)
        per_letter_f1 = [s.f1 for s in stats_list]
        macro_f1 = float(np.mean(per_letter_f1))

        # Also compute multi-class accuracy directly
        with torch.no_grad():
            logits = probe(probe_data["X_test"].to(model.cfg.device))
            preds = logits.argmax(dim=-1).cpu()
            y_true = probe_data["y_test"].cpu()
            multi_acc = float((preds == y_true).float().mean())
            # Compute weighted F1 over the multi-class predictions
            from sklearn.metrics import f1_score as sk_f1
            multi_f1 = float(sk_f1(y_true.numpy(), preds.numpy(), average="weighted"))

        f1 = multi_f1  # Use multi-class weighted F1 as primary metric
        acc = multi_acc

        logger.info(f"  F1={f1:.4f}, Accuracy={acc:.4f}")

        # Save probe weights
        probe_save = PHASE1_DIR / f"probe_first-letter_L{layer}.pt"
        torch.save({
            "probe_state_dict": probe.state_dict(),
            "f1": f1, "accuracy": acc,
            "n_train": len(train_dataset), "n_test": len(test_dataset),
        }, probe_save)

        results[f"first-letter_L{layer}"] = {
            "hierarchy": "first-letter",
            "layer": layer,
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

    return results


# ============================================================
# RAVEL Hierarchy Probes
# ============================================================
def create_ravel_icl_prompts(cities, labels, label_name, n_icl=5):
    """
    Create ICL-style prompts for RAVEL hierarchies.
    Template varies by hierarchy type for best model performance.
    """
    templates = {
        "city-country": ("The city of {entity} is in the country of", " {label}"),
        "city-continent": ("The city of {entity} is on the continent of", " {label}"),
        "city-language": ("The primary language spoken in {entity} is", " {label}"),
    }

    base_template, answer_template = templates.get(
        label_name, ("The city of {entity} has the property", " {label}")
    )

    # Build example pool
    examples = list(zip(cities, labels))
    random.shuffle(examples)

    prompts = []
    for i, (city, label) in enumerate(zip(cities, labels)):
        # Select ICL examples (not including the current city)
        icl_pool = [(c, l) for c, l in examples if c != city][:n_icl]

        # Build ICL prompt
        icl_parts = []
        for ex_city, ex_label in icl_pool:
            ex_prompt = base_template.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_prompt)

        full_prompt = "\n".join(icl_parts) + "\n" + base_template.format(entity=city)
        prompts.append(full_prompt)

    return prompts


def cache_ravel_activations(model, prompts, layers, batch_size=1):
    """
    Cache activations at position -2 (right before the model would predict the answer).
    This mirrors the sae_spelling approach.
    """
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    activations = {l: [] for l in layers}

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for l in layers:
            # Position -2: the last content token before the final token
            # This is where the model "knows" the answer
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


def train_ravel_probes(model, ds, layers):
    """Train probes for all 3 RAVEL hierarchies."""
    cities = ds["City"]
    hierarchy_defs = {
        "city-country": ds["Country"],
        "city-continent": ds["Continent"],
        "city-language": ds["Language"],
    }

    results = {}

    for hier_name, raw_labels in hierarchy_defs.items():
        logger.info(f"\n=== {hier_name} ===")

        # Filter valid entries and classes with >= 5 samples
        valid = [(c, l) for c, l in zip(cities, raw_labels) if c and l and str(l).strip()]
        ents, labs = zip(*valid)
        ents, labs = list(ents), list(labs)

        le = LabelEncoder()
        encoded = le.fit_transform(labs)
        counts = Counter(encoded)
        valid_classes = {c for c, n in counts.items() if n >= 5}
        keep = [i for i, c in enumerate(encoded) if c in valid_classes]
        ents = [ents[i] for i in keep]
        labs = [labs[i] for i in keep]
        le = LabelEncoder()
        encoded = le.fit_transform(labs)

        logger.info(f"  {len(ents)} entities, {len(le.classes_)} classes")

        # Create ICL prompts
        logger.info("  Creating ICL prompts...")
        prompts = create_ravel_icl_prompts(ents, labs, hier_name, n_icl=5)

        # Sample for inspection
        for j in range(min(2, len(prompts))):
            logger.info(f"  Sample prompt ({labs[j]}):\n    {prompts[j][:120]}...")

        # Cache activations
        logger.info("  Caching activations...")
        activations = cache_ravel_activations(model, prompts, layers)

        for layer in layers:
            X = activations[layer]
            y = encoded
            n_classes = len(le.classes_)

            logger.info(f"  Training {hier_name} probe L{layer}: X={X.shape}, {n_classes} classes")

            # Try multiple regularizations
            best_f1, best_probe, best_metrics = -1, None, None
            for C in [0.01, 0.1, 1.0, 10.0]:
                probe = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=SEED)
                min_count = min(Counter(y).values())
                n_splits = min(5, max(2, min_count))

                try:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                    y_pred = cross_val_predict(probe, X, y, cv=cv)
                    f1 = f1_score(y, y_pred, average="weighted")
                    acc = accuracy_score(y, y_pred)
                    bal_acc = balanced_accuracy_score(y, y_pred)
                except Exception as e:
                    logger.warning(f"    CV failed for C={C}: {e}")
                    f1, acc, bal_acc = 0.0, 0.0, 0.0

                probe.fit(X, y)
                train_acc = accuracy_score(y, probe.predict(X))

                logger.info(f"    C={C}: F1={f1:.4f}, Acc={acc:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_probe = probe
                    best_metrics = {
                        "f1_weighted_cv": float(f1),
                        "accuracy_cv": float(acc),
                        "balanced_accuracy_cv": float(bal_acc),
                        "accuracy_train": float(train_acc),
                        "n_samples": len(y),
                        "n_classes": n_classes,
                        "n_splits_cv": n_splits,
                        "best_C": C,
                        "probe_type": "LogisticRegression",
                    }

            key = f"{hier_name}_L{layer}"
            results[key] = {
                "hierarchy": hier_name, "layer": layer, **best_metrics,
                "quality_gate_strict": best_metrics["f1_weighted_cv"] >= 0.90,
                "quality_gate_relaxed": best_metrics["f1_weighted_cv"] >= 0.85,
            }

            # Save probe
            np.savez(PHASE1_DIR / f"probe_{hier_name}_L{layer}.npz",
                    coef=best_probe.coef_, intercept=best_probe.intercept_,
                    classes=np.array(le.classes_))

            logger.info(f"  BEST F1={best_metrics['f1_weighted_cv']:.4f}")

        del activations
        gc.collect(); torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 10, "starting")

    logger.info(f"=== Phase 1.1: Probe Training (mode={MODE}, layers={PROBE_LAYERS}) ===")

    # Load model
    report_progress(1, 10, "loading_model")
    model = load_model("cuda:0")

    all_results = {}

    # First-letter probes (sae_spelling pipeline)
    report_progress(2, 10, "first_letter_probes")
    fl_results = train_first_letter_probes(model, PROBE_LAYERS)
    all_results.update(fl_results)

    # RAVEL hierarchy probes
    report_progress(5, 10, "ravel_probes")
    from datasets import load_dataset
    ds = load_dataset("hij/ravel", "city_entity", split="train")
    ravel_results = train_ravel_probes(model, ds, PROBE_LAYERS)
    all_results.update(ravel_results)

    # Quality gate summary
    logger.info("\n=== Quality Gate Summary ===")
    n_pass_strict, n_pass_relaxed, n_total = 0, 0, 0
    for key, pr in all_results.items():
        n_total += 1
        if pr["quality_gate_strict"]: n_pass_strict += 1
        if pr["quality_gate_relaxed"]: n_pass_relaxed += 1
        st = "PASS" if pr["quality_gate_strict"] else ("RELAXED" if pr["quality_gate_relaxed"] else "FAIL")
        logger.info(f"  {key}: F1={pr['f1_weighted_cv']:.4f} [{st}]")

    hiers = set(pr["hierarchy"] for pr in all_results.values())
    best_f1 = max(pr["f1_weighted_cv"] for pr in all_results.values())
    pilot_ok = len(hiers) >= 2 and best_f1 >= 0.85

    elapsed = time.time() - start_time

    output = {
        "task_id": TASK_ID, "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED, "model": "gemma-2-2b",
        "layers": PROBE_LAYERS,
        "probes": all_results,
        "quality_gate_summary": {
            "n_total": n_total, "n_pass_strict": n_pass_strict,
            "n_pass_relaxed": n_pass_relaxed,
        },
        "pilot_criteria_met": pilot_ok,
        "pilot_criteria_details": {
            "n_hierarchies": len(hiers), "hierarchies": sorted(hiers),
            "best_f1": best_f1,
        },
        "elapsed_minutes": elapsed / 60,
    }

    if best_f1 < 0.85:
        output["diagnosis"] = {
            "issue": "No probe reaches F1 >= 0.85",
            "analysis": [
                f"Best F1: {best_f1:.4f}",
                "Layer 12 may not be optimal for all hierarchies",
                "ICL prompts should improve factual probe quality",
                "Consider trying layers 6, 18, 24 in full mode",
            ],
        }

    out_path = PILOT_DIR / "phase1_probe_training.json" if MODE == "PILOT" else PHASE1_DIR / "probe_training.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")
    logger.info(f"Elapsed: {elapsed/60:.1f} min, Pilot OK: {pilot_ok}")

    del model; gc.collect(); torch.cuda.empty_cache()

    summary = (f"{len(hiers)} hierarchies, best F1={best_f1:.4f}, "
               f"strict pass: {n_pass_strict}/{n_total}, "
               f"pilot: {'MET' if pilot_ok else 'NOT MET'}")
    mark_done("success" if pilot_ok else "partial", summary)
    update_gpu_progress(elapsed)
    return output


def update_gpu_progress(elapsed_seconds):
    path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        data = json.loads(path.read_text()) if path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in data.get("completed", []):
            data.setdefault("completed", []).append(TASK_ID)
        data.get("running", {}).pop(TASK_ID, None)
        data.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 45, "actual_min": int(elapsed_seconds / 60),
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {"model": "gemma-2-2b", "mode": MODE,
                               "layers": PROBE_LAYERS, "gpu": "RTX PRO 6000"}
        }
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        try:
            path = WORKSPACE / "exp" / "gpu_progress.json"
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}}
            if TASK_ID not in data.get("failed", []):
                data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            path.write_text(json.dumps(data, indent=2))
        except: pass
        sys.exit(1)
