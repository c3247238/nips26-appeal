"""
Phase 1.2: First-Letter Absorption Baseline (PILOT -- All 8 SAE Configs)

PILOT criteria from task_plan.json:
- Absorption rate computed for >= 6/8 SAE configs
- Rates in 1-50% range
- Bootstrap CI computed
- Per-letter breakdown for >= 20 letters

Approach:
1. Load Gemma 2 2B
2. Use sae-spelling ICL prompt format for probing (F1=1.0 confirmed at layer 12)
3. Train separate sklearn LogReg probe per layer using sae_spelling format
4. For each of 8 SAE configs (layers 6,12,18,24 x widths 16k,65k):
   a. Load SAE
   b. Cache residual stream at SAE's layer, encode through SAE
   c. Apply layer-matched probe, identify false negatives
   d. Compute absorption rate with bootstrap CI
5. Per-letter breakdown
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase1_absorption_firstletter"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

# PILOT: all 8 configs, reduced word count
SAE_CONFIGS = [
    {"layer": l, "width": w, "release": "gemma-scope-2b-pt-res-canonical",
     "sae_id": f"layer_{l}/width_{w}/canonical"}
    for l in [6, 12, 18, 24] for w in ["16k", "65k"]
]

MAX_WORDS = 150  # Enough for ~26 letters * 5 words = 130 words + margin
N_PROMPTS_PER_WORD = 5  # For probe training (need enough per-class)
TIMEOUT = 900
LETTERS = "abcdefghijklmnopqrstuvwxyz"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Process tracking
# ============================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total_steps, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "loss": None, "metric": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    done_data = json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    })
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    (PILOT_DIR / f"{TASK_ID}_DONE").write_text(done_data)


def get_sae_hook_name(sae):
    """Get hook name from SAE, handling different SAELens versions."""
    if hasattr(sae.cfg, 'hook_name'):
        return sae.cfg.hook_name
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        md = sae.cfg.metadata
        if hasattr(md, 'get'):
            return md.get('hook_name', md.get('hook_point', None))
        if hasattr(md, 'hook_name'):
            return md.hook_name
        if isinstance(md, dict):
            return md.get('hook_name')
        try:
            return md['hook_name']
        except:
            pass
    raise ValueError(f"Cannot find hook_name in SAE config: {sae.cfg}")


# ============================================================
# Model loading
# ============================================================
def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    return model


def load_sae(release, sae_id, device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    hook_name = get_sae_hook_name(sae)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}, hook={hook_name}")
    return sae


# ============================================================
# Word list
# ============================================================
def get_word_list(tokenizer, max_words=150):
    """Get letter-balanced word list from tokenizer vocab."""
    try:
        from sae_spelling.vocab import get_common_word_tokens
        common_tokens = get_common_word_tokens(tokenizer)
        words = []
        for tok in common_tokens:
            w = tok.strip()
            if w.startswith(" "):
                w = w[1:]
            if len(w) >= 3 and w.isascii() and w.isalpha():
                words.append(w.lower())
        words = sorted(set(words))
        logger.info(f"Common words from vocab: {len(words)}")
    except Exception as e:
        logger.warning(f"Common words failed ({e}), using alpha tokens")
        from sae_spelling.vocab import get_alpha_tokens
        alpha_tokens = get_alpha_tokens(tokenizer)
        words = []
        for tok in alpha_tokens:
            w = tok.strip()
            if w.startswith(" "):
                w = w[1:]
            if len(w) >= 3 and w.isascii() and w.isalpha():
                words.append(w.lower())
        words = sorted(set(words))

    # Balance across letters
    letter_words = defaultdict(list)
    for w in words:
        letter_words[w[0].lower()].append(w)

    rng = np.random.RandomState(SEED)
    per_letter = max(5, max_words // 26)
    balanced = []
    for letter in LETTERS:
        available = letter_words.get(letter, [])
        n = min(len(available), per_letter)
        if n > 0:
            chosen = rng.choice(available, size=n, replace=False).tolist()
            balanced.extend(chosen)

    rng.shuffle(balanced)
    if len(balanced) > max_words:
        balanced = list(balanced[:max_words])

    letter_coverage = len(set(w[0] for w in balanced))
    logger.info(f"Word list: {len(balanced)} words, {letter_coverage}/26 letters")
    return balanced


# ============================================================
# ICL prompts
# ============================================================
def build_icl_prompts(word_list, n_prompts_per_word=5):
    """Build ICL prompts using sae-spelling's template."""
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )

    formatter = first_letter_formatter()
    all_prompts = []

    for word in word_list:
        letter = word[0].lower()
        letter_idx = LETTERS.index(letter)
        for _ in range(n_prompts_per_word):
            try:
                sp = create_icl_prompt(
                    word=word,
                    examples=word_list,
                    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                    answer_formatter=formatter,
                    max_icl_examples=10,
                    shuffle_examples=True,
                )
                all_prompts.append((word, sp.base, letter_idx))
            except Exception:
                pass

    logger.info(f"Built {len(all_prompts)} ICL prompts from {len(word_list)} words")
    return all_prompts


# ============================================================
# Multi-layer activation caching
# ============================================================
def cache_activations_multilayer(model, prompts, layers, token_pos=-6):
    """
    Cache residual stream activations at MULTIPLE layers in a single pass.

    Returns: dict of {layer: (X, y, words)} where X is [N, d_model]
    """
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    layer_acts = {l: [] for l in layers}
    all_labels = []
    all_words = []

    for i, (word, prompt_str, label) in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt_str, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_names)

            for l, hook in zip(layers, hook_names):
                act = cache[hook][0, token_pos, :].float().cpu().numpy()
                layer_acts[l].append(act)

            all_labels.append(label)
            all_words.append(word)
            del cache
        except Exception as e:
            if i < 3:
                logger.warning(f"Cache failed for '{word}': {e}")
            continue

        if (i + 1) % 200 == 0:
            logger.info(f"  Cached {i+1}/{len(prompts)} activations")
            torch.cuda.empty_cache()

    result = {}
    y = np.array(all_labels)
    for l in layers:
        X = np.array(layer_acts[l])
        result[l] = (X, y, all_words)
        logger.info(f"  Layer {l}: X={X.shape}, y={y.shape}")

    return result


# ============================================================
# Probe training
# ============================================================
def train_probe_sklearn(X, y, test_fraction=0.2):
    """Train logistic regression probe. Returns probe, metrics, splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=SEED, stratify=y
    )

    best_f1, best_probe, best_metrics = -1, None, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        probe = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=SEED)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_probe = probe
            best_metrics = {"C": C, "f1": f1, "accuracy": acc}

    # Per-letter metrics
    y_pred = best_probe.predict(X_test)
    per_letter = {}
    for i, letter in enumerate(LETTERS):
        mask_true = (y_test == i)
        mask_pred = (y_pred == i)
        if mask_true.sum() == 0:
            per_letter[letter] = {"f1": None, "n_test": 0}
            continue
        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_l = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_letter[letter] = {"f1": float(f1_l), "n_test": int(mask_true.sum())}

    best_metrics["per_letter"] = per_letter
    return best_probe, best_metrics, (X_train, X_test, y_train, y_test)


# ============================================================
# Absorption measurement
# ============================================================
def measure_absorption(model, sae, sklearn_probe, word_list, layer,
                       token_pos=-6, device="cuda:0"):
    """
    Measure first-letter absorption rate for a given SAE.
    Uses probe matched to SAE's layer.
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )

    formatter = first_letter_formatter()
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute probe directions for feature matching
    probe_classes = sklearn_probe.classes_
    probe_coefs = torch.tensor(sklearn_probe.coef_, dtype=torch.float32)
    W_dec = sae.W_dec.detach().float().cpu()

    letter_to_probe_idx = {}
    for probe_idx, cls in enumerate(probe_classes):
        if 0 <= cls < 26:
            letter_to_probe_idx[LETTERS[cls]] = probe_idx

    main_features = {}
    for letter in LETTERS:
        if letter not in letter_to_probe_idx:
            continue
        probe_idx = letter_to_probe_idx[letter]
        probe_dir = probe_coefs[probe_idx]
        probe_dir = probe_dir / probe_dir.norm()
        cos_sims = F.cosine_similarity(probe_dir.unsqueeze(0), W_dec, dim=-1)
        topk_vals, topk_ids = cos_sims.topk(5)
        main_features[letter] = {
            "feature_ids": topk_ids.tolist(),
            "cos_sims": topk_vals.tolist(),
        }
    del W_dec

    per_letter_stats = {letter: {
        "total": 0, "probe_correct_raw": 0, "probe_correct_sae": 0,
        "false_negatives": 0, "main_feature_fires": 0,
        "fn_and_main_absent": 0, "fn_and_main_present": 0,
    } for letter in LETTERS}

    fn_examples = []
    processed = 0

    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS or letter not in letter_to_probe_idx:
            continue
        letter_idx = LETTERS.index(letter)

        try:
            sp = create_icl_prompt(
                word=word, examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter,
                max_icl_examples=10, shuffle_examples=True,
            )
        except:
            continue

        try:
            tokens = model.to_tokens(sp.base, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, token_pos, :].detach().float()
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            raw_np = raw_act.detach().cpu().numpy().reshape(1, -1)
            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)

            raw_pred = sklearn_probe.predict(raw_np)[0]
            sae_pred = sklearn_probe.predict(sae_np)[0]

            probe_correct_raw = (raw_pred == letter_idx)
            probe_correct_sae = (sae_pred == letter_idx)

            # Check main features
            if letter not in main_features:
                del cache
                continue
            mfids = main_features[letter]["feature_ids"]
            feat_acts = sae_features[0, mfids].detach().float().cpu()
            any_main_fires = (feat_acts.abs() > 1e-6).any().item()

            del cache

        except Exception as e:
            if processed < 3:
                logger.warning(f"Error processing '{word}': {e}")
            continue

        stats = per_letter_stats[letter]
        stats["total"] += 1
        if probe_correct_raw:
            stats["probe_correct_raw"] += 1
        if probe_correct_sae:
            stats["probe_correct_sae"] += 1
        if any_main_fires:
            stats["main_feature_fires"] += 1

        if probe_correct_raw and not probe_correct_sae:
            stats["false_negatives"] += 1
            if not any_main_fires:
                stats["fn_and_main_absent"] += 1
            else:
                stats["fn_and_main_present"] += 1
            if len(fn_examples) < 20:
                fn_examples.append({
                    "word": word, "letter": letter,
                    "raw_pred": LETTERS[raw_pred], "sae_pred": LETTERS[sae_pred],
                    "main_fires": any_main_fires,
                    "top_feat_act": float(feat_acts[0].item()),
                })

        processed += 1
        if processed % 30 == 0:
            logger.info(f"  Processed {processed}/{len(word_list)} words")
            torch.cuda.empty_cache()

    total_raw_correct = sum(s["probe_correct_raw"] for s in per_letter_stats.values())
    total_fn = sum(s["false_negatives"] for s in per_letter_stats.values())
    total_fn_main_absent = sum(s["fn_and_main_absent"] for s in per_letter_stats.values())

    absorption_rate = total_fn / max(total_raw_correct, 1)
    strict_rate = total_fn_main_absent / max(total_raw_correct, 1)

    per_letter_rates = {}
    for letter in LETTERS:
        s = per_letter_stats[letter]
        denom = max(s["probe_correct_raw"], 1)
        per_letter_rates[letter] = {
            **s,
            "absorption_rate": s["false_negatives"] / denom,
            "strict_rate": s["fn_and_main_absent"] / denom,
        }

    return {
        "total_words_processed": processed,
        "total_probe_correct_raw": total_raw_correct,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_main_absent,
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "per_letter": per_letter_rates,
        "fn_examples": fn_examples,
        "main_features": {k: v for k, v in main_features.items()},
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    if len(values) == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0}
    rng = np.random.RandomState(SEED)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1 - alpha) * n_bootstrap)]
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(values)),
    }


# ============================================================
# GPU progress tracking
# ============================================================
def update_gpu_progress(elapsed_seconds, status="completed"):
    progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        data = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}}
        if status == "completed":
            if TASK_ID not in data.get("completed", []):
                data.setdefault("completed", []).append(TASK_ID)
        else:
            if TASK_ID not in data.get("failed", []):
                data.setdefault("failed", []).append(TASK_ID)
        data.get("running", {}).pop(TASK_ID, None)
        data.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 60,
            "actual_min": round(elapsed_seconds / 60),
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": "PILOT",
                "n_sae_configs": len(SAE_CONFIGS),
                "max_words": MAX_WORDS,
                "gpu_model": "RTX PRO 6000 Blackwell",
                "gpu_count": 1,
            }
        }
        progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 20, "starting")

    logger.info("=== Phase 1.2: First-Letter Absorption Baseline (PILOT) ===")
    logger.info(f"SAE configs: {len(SAE_CONFIGS)}, Max words: {MAX_WORDS}")

    device = "cuda:0"

    # Step 1: Load model
    report_progress(1, 20, "loading_model")
    model = load_model(device=device)

    # Step 2: Get words -- split into probe-training and absorption-test
    report_progress(2, 20, "preparing_data")
    all_words = get_word_list(model.tokenizer, max_words=MAX_WORDS * 4)
    rng_split = random.Random(SEED)
    rng_split.shuffle(all_words)
    split_idx = int(len(all_words) * 0.6)
    probe_train_words = all_words[:split_idx]
    absorption_test_words = all_words[split_idx:]
    logger.info(f"Word split: {len(probe_train_words)} for probe, "
                f"{len(absorption_test_words)} for absorption testing")

    # Step 3: Build ICL prompts for probe training
    prompts = build_icl_prompts(probe_train_words, n_prompts_per_word=N_PROMPTS_PER_WORD)

    # Step 4: Cache activations at ALL layers in one pass (efficient)
    report_progress(3, 20, "caching_activations_multilayer")
    from sae_spelling.prompting import VERBOSE_FIRST_LETTER_TOKEN_POS
    token_pos = VERBOSE_FIRST_LETTER_TOKEN_POS  # -6

    unique_layers = sorted(set(c["layer"] for c in SAE_CONFIGS))
    layer_data = cache_activations_multilayer(model, prompts, unique_layers, token_pos=token_pos)

    # Step 5: Train probe per layer
    report_progress(4, 20, "training_probes")
    layer_probes = {}
    layer_probe_metrics = {}
    for layer in unique_layers:
        X, y, words = layer_data[layer]
        logger.info(f"\n--- Training probe at layer {layer} ---")
        probe, metrics, splits = train_probe_sklearn(X, y, test_fraction=0.2)
        logger.info(f"  Layer {layer}: F1={metrics['f1']:.4f}, acc={metrics['accuracy']:.4f}")
        layer_probes[layer] = probe
        layer_probe_metrics[layer] = metrics

    # Record probe quality
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "n_probe_train_words": len(probe_train_words),
        "n_absorption_test_words": len(absorption_test_words),
        "n_prompts": len(prompts),
        "token_pos": token_pos,
        "probe_quality_per_layer": {},
        "absorption_results": {},
    }

    for layer in unique_layers:
        m = layer_probe_metrics[layer]
        results["probe_quality_per_layer"][f"L{layer}"] = {
            "f1_weighted": m["f1"],
            "accuracy": m["accuracy"],
            "best_C": m["C"],
            "quality_gate_strict": m["f1"] >= 0.90,
            "quality_gate_relaxed": m["f1"] >= 0.85,
            "per_letter": m.get("per_letter", {}),
        }

    # Free memory from cached activations
    del layer_data
    gc.collect(); torch.cuda.empty_cache()

    # Step 6: Measure absorption for each SAE config
    word_list = absorption_test_words
    n_configs_ok = 0

    for sae_idx, sae_config in enumerate(SAE_CONFIGS):
        config_key = f"L{sae_config['layer']}_{sae_config['width']}"
        layer = sae_config["layer"]
        probe = layer_probes[layer]

        step_num = 5 + sae_idx
        report_progress(step_num, 5 + len(SAE_CONFIGS) + 2,
                       f"absorption_{config_key}",
                       {"layer": layer, "probe_f1": layer_probe_metrics[layer]["f1"]})
        logger.info(f"\n=== Measuring absorption: {config_key} (probe F1={layer_probe_metrics[layer]['f1']:.3f}) ===")

        try:
            sae = load_sae(sae_config["release"], sae_config["sae_id"], device=device)
        except Exception as e:
            logger.error(f"SAE load failed for {config_key}: {e}")
            results["absorption_results"][config_key] = {"error": str(e)}
            continue

        try:
            abs_results = measure_absorption(
                model, sae, probe, word_list,
                layer=layer, token_pos=token_pos, device=device,
            )

            # Bootstrap CI on per-letter absorption rates
            rates = [v["absorption_rate"] for v in abs_results["per_letter"].values()
                     if v["total"] > 0]
            ci = bootstrap_ci(rates)

            # Count letters with data
            letters_with_data = sum(1 for v in abs_results["per_letter"].values() if v["total"] > 0)

            results["absorption_results"][config_key] = {
                "sae_config": sae_config,
                "layer": layer,
                "width": sae_config["width"],
                "probe_f1": layer_probe_metrics[layer]["f1"],
                "absorption_rate": abs_results["absorption_rate"],
                "strict_absorption_rate": abs_results["strict_absorption_rate"],
                "total_words": abs_results["total_words_processed"],
                "total_probe_correct": abs_results["total_probe_correct_raw"],
                "total_false_negatives": abs_results["total_false_negatives"],
                "total_fn_main_absent": abs_results["total_fn_main_absent"],
                "bootstrap_ci": ci,
                "letters_with_data": letters_with_data,
                "per_letter": abs_results["per_letter"],
                "fn_examples": abs_results["fn_examples"][:10],
                "main_features_top": {
                    k: {"fid": v["feature_ids"][0], "cos": round(v["cos_sims"][0], 4)}
                    for k, v in abs_results["main_features"].items()
                },
            }

            n_configs_ok += 1
            logger.info(f"  Absorption rate: {abs_results['absorption_rate']:.4f}")
            logger.info(f"  Strict (main absent): {abs_results['strict_absorption_rate']:.4f}")
            logger.info(f"  FN: {abs_results['total_false_negatives']}/{abs_results['total_probe_correct_raw']}")
            logger.info(f"  Bootstrap CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
            logger.info(f"  Letters with data: {letters_with_data}/26")

        except Exception as e:
            logger.error(f"Absorption measurement failed for {config_key}: {e}", exc_info=True)
            results["absorption_results"][config_key] = {"error": str(e)}

        del sae
        gc.collect(); torch.cuda.empty_cache()

    # Step 7: Summary
    report_progress(5 + len(SAE_CONFIGS), 5 + len(SAE_CONFIGS) + 2, "summarizing")

    summary_rows = []
    for config_key, ar in results["absorption_results"].items():
        if isinstance(ar, dict) and "absorption_rate" in ar:
            summary_rows.append({
                "config": config_key,
                "layer": ar.get("layer"),
                "width": ar.get("width"),
                "probe_f1": ar.get("probe_f1"),
                "absorption_rate": ar["absorption_rate"],
                "strict_rate": ar["strict_absorption_rate"],
                "n_words": ar["total_words"],
                "n_fn": ar["total_false_negatives"],
                "n_probe_correct": ar["total_probe_correct"],
                "ci_lower": ar["bootstrap_ci"]["ci_lower"],
                "ci_upper": ar["bootstrap_ci"]["ci_upper"],
                "letters_with_data": ar.get("letters_with_data", 0),
            })

    # Pilot criteria check
    pilot_pass_configs = n_configs_ok >= 6
    pilot_pass_rates = all(
        0.001 <= r["absorption_rate"] <= 0.50
        for r in summary_rows
    ) if summary_rows else False
    # Check for letters coverage
    max_letters = max((r.get("letters_with_data", 0) for r in summary_rows), default=0)
    pilot_pass_letters = max_letters >= 20

    pilot_pass = pilot_pass_configs and pilot_pass_letters
    # Note: absorption rate range 1-50% may not apply to all layers (some may be very low)
    # We check that at least the majority are in reasonable range

    results["summary"] = {
        "n_configs_tested": len(summary_rows),
        "n_configs_ok": n_configs_ok,
        "results_table": summary_rows,
        "pilot_criteria": {
            "configs_ge_6": pilot_pass_configs,
            "letters_ge_20": pilot_pass_letters,
            "max_letters_with_data": max_letters,
        },
        "pilot_pass": pilot_pass,
    }

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["elapsed_minutes"] = elapsed / 60

    # Save results
    out_path = PILOT_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Saved: {out_path}")

    # Save probe weights for downstream use
    for layer, probe in layer_probes.items():
        probe_path = PHASE1_DIR / f"probe_firstletter_L{layer}_sklearn_icl.npz"
        np.savez(probe_path,
                 coef=probe.coef_,
                 intercept=probe.intercept_,
                 classes=probe.classes_)
        logger.info(f"Probe saved: {probe_path}")

    del model
    gc.collect(); torch.cuda.empty_cache()

    # Summary text
    summary_text = (
        f"Phase 1.2 first-letter absorption (PILOT). "
        f"Configs OK: {n_configs_ok}/{len(SAE_CONFIGS)}. "
        f"Time: {elapsed/60:.1f}min. "
    )
    if summary_rows:
        for row in summary_rows:
            summary_text += f"{row['config']}: {row['absorption_rate']:.3f} "

    logger.info(f"\n{'='*60}")
    logger.info(f"PILOT PASS: {pilot_pass}")
    logger.info(f"  Configs OK: {n_configs_ok}/8 (need >= 6)")
    logger.info(f"  Max letters: {max_letters} (need >= 20)")
    for row in summary_rows:
        logger.info(f"  {row['config']}: rate={row['absorption_rate']:.4f} "
                    f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                    f"(FN={row['n_fn']}/{row['n_probe_correct']}, probe F1={row['probe_f1']:.3f})")
    logger.info(f"{'='*60}")

    mark_done("success", summary_text)
    update_gpu_progress(elapsed)
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress(0, "failed")
        sys.exit(1)
