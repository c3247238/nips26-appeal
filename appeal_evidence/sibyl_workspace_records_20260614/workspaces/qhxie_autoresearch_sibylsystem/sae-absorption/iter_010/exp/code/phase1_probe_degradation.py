"""
Phase 1.1: Probe Degradation Ablation (H10 -- HIGHEST PRIORITY)
Iteration 10 - PILOT mode.

Resolves whether cross-domain absorption variation is genuine or a probe quality artifact.

Approach:
  1. Follow the EXACT iter_009 pipeline: train fresh probes, measure absorption.
  2. For control (F1=1.0): train probe normally, measure absorption. Must match iter_009.
  3. For degraded levels {0.70, 0.80, 0.85, 0.90}:
     - Add Gaussian noise to the trained probe weights to degrade F1.
     - Calibrate noise level using binary search on a held-out validation set.
     - Re-measure absorption with the degraded probe on THE SAME test words.
  4. Key comparison: at matched F1, does first-letter absorption match RAVEL rates?

Critical insight: absorption = FN rate = (probe correct on raw BUT wrong on SAE) / (probe correct on raw).
  When probe quality degrades, FEWER words are "probe correct on raw" (denominator shrinks),
  but the FN among those words may change. The question is whether the FN *rate* tracks probe quality.

Output: exp/results/phase1/probe_degradation.json
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
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats as scipy_stats

# ── Configuration ────────────────────────────────────────────────
TASK_ID = "phase1_probe_degradation"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ITER009_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_009")

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = "PILOT"

# SAE config: L24 16k
SAE_LAYER = 24
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"

# Degradation targets
TARGET_F1_LEVELS = [0.70, 0.80, 0.85, 0.90, 1.0]

# RAVEL reference points (from iter_009 absorption_crossdomain.json L24_16k)
RAVEL_REFERENCE = {
    "city-continent": {"absorption_rate": 0.3143, "probe_f1": 0.871},
    "city-country":   {"absorption_rate": 0.4510, "probe_f1": 0.726},
    "city-language":  {"absorption_rate": 0.1156, "probe_f1": 0.818},
}

# Iter_009 first-letter baseline (L24_16k)
ITER009_FIRSTLETTER_BASELINE = {
    "absorption_rate": 0.2707,
    "ci_lower": 0.2632,
    "ci_upper": 0.3473,
}

TOKEN_POS = -6
N_PROMPTS_PER_WORD = 3
PROBE_TRAIN_PROMPTS = 5
MAX_TEST_WORDS = 500
N_BOOTSTRAP = 10000
LETTERS = "abcdefghijklmnopqrstuvwxyz"

GPU_ID = int(os.environ.get("CUDA_GPU_ID", "0"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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
    done_data = json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    })
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    logger.info(f"DONE (status={status}): {summary}")


# ── SAE utility ──────────────────────────────────────────────────
def get_sae_hook_name(sae):
    if hasattr(sae.cfg, 'hook_name') and isinstance(sae.cfg.hook_name, str):
        return sae.cfg.hook_name
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        md = sae.cfg.metadata
        if isinstance(md, dict):
            return md.get('hook_name', md.get('hook_point'))
        if hasattr(md, 'hook_name'):
            return md.hook_name
        try:
            return md['hook_name']
        except Exception:
            pass
    raise ValueError(f"Cannot find hook_name in SAE config: {sae.cfg}")


# ── Model / SAE loading ─────────────────────────────────────────
def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    return model


def load_sae(device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}")
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    hook_name = get_sae_hook_name(sae)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}, hook={hook_name}")
    return sae


# ── Word list ────────────────────────────────────────────────────
def get_word_list(tokenizer, max_words=2000):
    """Balanced word list from tokenizer vocab (matches iter_009)."""
    try:
        from sae_spelling.vocab import get_common_word_tokens
        common_tokens = get_common_word_tokens(tokenizer)
        words = []
        for tok in common_tokens:
            w = tok.strip().lstrip(" ")
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
            w = tok.strip().lstrip(" ")
            if len(w) >= 3 and w.isascii() and w.isalpha():
                words.append(w.lower())
        words = sorted(set(words))

    letter_words = defaultdict(list)
    for w in words:
        letter_words[w[0].lower()].append(w)

    rng = np.random.RandomState(SEED)
    per_letter = max(10, max_words // 26 + 3)
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


# ── ICL prompt building ─────────────────────────────────────────
def build_icl_prompts(word_list, n_prompts_per_word=5, example_pool=None):
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    if example_pool is None:
        example_pool = word_list

    all_prompts = []
    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS:
            continue
        letter_idx = LETTERS.index(letter)
        for _ in range(n_prompts_per_word):
            try:
                sp = create_icl_prompt(
                    word=word, examples=example_pool,
                    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                    answer_formatter=formatter,
                    max_icl_examples=10, shuffle_examples=True,
                )
                all_prompts.append((word, sp.base, letter_idx))
            except Exception:
                pass

    logger.info(f"Built {len(all_prompts)} ICL prompts from {len(word_list)} words")
    return all_prompts


# ── Activation caching ──────────────────────────────────────────
def cache_activations(model, prompts, layer=24, token_pos=-6):
    hook_name = f"blocks.{layer}.hook_resid_post"
    acts = []
    labels = []
    words = []

    for i, (word, prompt_str, label) in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt_str, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name][0, token_pos, :].float().cpu().numpy()
            acts.append(act)
            labels.append(label)
            words.append(word)
            del cache
        except Exception as e:
            if i < 3:
                logger.warning(f"Cache failed for '{word}': {e}")
            continue

        if (i + 1) % 500 == 0:
            logger.info(f"  Cached {i+1}/{len(prompts)} activations")
            torch.cuda.empty_cache()

    X = np.array(acts)
    y = np.array(labels)
    logger.info(f"  Cached: {len(labels)} samples, X shape: {X.shape}")
    return X, y, words


# ── Probe training ──────────────────────────────────────────────
def train_probe(X, y):
    """Train sklearn probe with CV of C parameter."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    best_f1, best_probe, best_C = -1, None, None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=SEED)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_probe = clf
            best_C = C

    y_pred_final = best_probe.predict(X_val)
    f1_final = f1_score(y_val, y_pred_final, average="weighted")
    f1_macro = f1_score(y_val, y_pred_final, average="macro")
    acc_final = accuracy_score(y_val, y_pred_final)

    logger.info(f"  Probe trained: C={best_C}, F1_w={f1_final:.4f}, "
                f"F1_m={f1_macro:.4f}, Acc={acc_final:.4f}")
    return best_probe, {
        "best_C": best_C, "f1_weighted": float(f1_final),
        "f1_macro": float(f1_macro), "accuracy": float(acc_final),
        "n_train": len(y_train), "n_val": len(y_val),
    }


# ── Probe degradation ──────────────────────────────────────────
def degrade_probe_to_target_f1(original_probe, X_val, y_val, target_f1,
                                max_iter=200, tolerance=0.02):
    """
    Degrade probe by adding Gaussian noise to weights.
    Binary search on noise scale to hit target F1.
    """
    if target_f1 >= 0.99:
        y_pred = original_probe.predict(X_val)
        actual_f1 = f1_score(y_val, y_pred, average="weighted")
        return original_probe, actual_f1, 0.0, "none"

    lo_noise, hi_noise = 0.0, 50.0
    best_probe = None
    best_f1 = None
    best_noise = None
    rng = np.random.RandomState(SEED)

    for iteration in range(max_iter):
        mid_noise = (lo_noise + hi_noise) / 2.0

        degraded = LogisticRegression(max_iter=1)
        noise = rng.randn(*original_probe.coef_.shape) * mid_noise
        degraded.coef_ = original_probe.coef_ + noise
        degraded.intercept_ = original_probe.intercept_.copy()
        degraded.classes_ = original_probe.classes_.copy()

        y_pred = degraded.predict(X_val)
        actual_f1 = f1_score(y_val, y_pred, average="weighted")

        if abs(actual_f1 - target_f1) < tolerance:
            best_probe = degraded
            best_f1 = actual_f1
            best_noise = mid_noise
            break

        if actual_f1 > target_f1:
            lo_noise = mid_noise
        else:
            hi_noise = mid_noise

        if best_f1 is None or abs(actual_f1 - target_f1) < abs(best_f1 - target_f1):
            best_probe = degraded
            best_f1 = actual_f1
            best_noise = mid_noise

    if best_probe is None:
        best_probe = degraded
        best_f1 = actual_f1
        best_noise = mid_noise

    logger.info(f"  Degraded: target={target_f1:.2f}, actual={best_f1:.4f}, "
                f"noise={best_noise:.4f}, iter={min(iteration+1, max_iter)}")
    return best_probe, best_f1, best_noise, "weight_noise"


# ── Absorption measurement (matches iter_009 exactly) ───────────
def measure_absorption(model, sae, sklearn_probe, test_words,
                       layer=24, token_pos=-6, n_prompts_per_word=3,
                       example_pool=None, device="cuda:0"):
    """
    Measure first-letter absorption for one probe.
    Same pipeline as iter_009 phase1_absorption_firstletter_full.py.
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    if example_pool is None:
        example_pool = test_words
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute main SAE features per letter
    probe_coefs = torch.tensor(sklearn_probe.coef_, dtype=torch.float32)
    probe_classes = sklearn_probe.classes_
    W_dec = sae.W_dec.detach().float().cpu()

    letter_to_probe_row = {}
    for row_idx, cls in enumerate(probe_classes):
        cls_int = int(cls)
        if 0 <= cls_int < 26:
            letter_to_probe_row[cls_int] = row_idx

    main_features = {}
    for letter_idx, row_idx in letter_to_probe_row.items():
        letter = LETTERS[letter_idx]
        probe_dir = probe_coefs[row_idx]
        probe_dir = probe_dir / probe_dir.norm()
        cos_sims = F.cosine_similarity(probe_dir.unsqueeze(0), W_dec, dim=-1)
        topk_vals, topk_ids = cos_sims.topk(5)
        main_features[letter] = {
            "feature_ids": topk_ids.tolist(),
            "cos_sims": topk_vals.tolist(),
        }
    del W_dec

    per_letter_stats = {letter: {
        "total": 0,
        "probe_correct_raw": 0,
        "probe_correct_sae": 0,
        "false_negatives": 0,
        "fn_and_main_absent": 0,
        "fn_and_main_present": 0,
    } for letter in LETTERS}

    word_results = []
    fn_examples = []
    processed = 0

    for word in test_words:
        letter = word[0].lower()
        if letter not in LETTERS:
            continue
        letter_idx = LETTERS.index(letter)
        if letter_idx not in letter_to_probe_row:
            continue

        prompt_results = []
        for _ in range(n_prompts_per_word):
            try:
                sp = create_icl_prompt(
                    word=word, examples=example_pool,
                    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                    answer_formatter=formatter,
                    max_icl_examples=10, shuffle_examples=True,
                )
            except Exception:
                continue

            try:
                tokens = model.to_tokens(sp.base, prepend_bos=True)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

                raw_act = cache[tl_hook][0, token_pos, :].detach().float()
                raw_act_dev = raw_act.to(device)
                with torch.no_grad():
                    sae_features = sae.encode(raw_act_dev.unsqueeze(0))
                    sae_out = sae.decode(sae_features)

                raw_np = raw_act.cpu().numpy().reshape(1, -1)
                sae_np = sae_out[0].float().cpu().numpy().reshape(1, -1)

                raw_pred = int(sklearn_probe.predict(raw_np)[0])
                sae_pred = int(sklearn_probe.predict(sae_np)[0])

                probe_correct_raw = (raw_pred == letter_idx)
                probe_correct_sae = (sae_pred == letter_idx)

                if letter in main_features:
                    mfids = main_features[letter]["feature_ids"]
                    feat_acts = sae_features[0, mfids].detach().float().cpu()
                    any_main_fires = (feat_acts.abs() > 1e-6).any().item()
                else:
                    any_main_fires = False

                del cache

                prompt_results.append({
                    "probe_correct_raw": bool(probe_correct_raw),
                    "probe_correct_sae": bool(probe_correct_sae),
                    "is_fn": bool(probe_correct_raw and not probe_correct_sae),
                    "main_fires": bool(any_main_fires),
                })
            except Exception as e:
                if processed < 3:
                    logger.warning(f"Error '{word}': {e}")
                continue

        if not prompt_results:
            continue

        n_raw_correct = sum(1 for p in prompt_results if p["probe_correct_raw"])
        n_sae_correct = sum(1 for p in prompt_results if p["probe_correct_sae"])
        n_fn = sum(1 for p in prompt_results if p["is_fn"])
        n_fn_main_absent = sum(1 for p in prompt_results
                               if p["is_fn"] and not p["main_fires"])
        n_fn_main_present = sum(1 for p in prompt_results
                                if p["is_fn"] and p["main_fires"])
        n_prompts = len(prompt_results)

        stats = per_letter_stats[letter]
        stats["total"] += n_prompts
        stats["probe_correct_raw"] += n_raw_correct
        stats["probe_correct_sae"] += n_sae_correct
        stats["false_negatives"] += n_fn
        stats["fn_and_main_absent"] += n_fn_main_absent
        stats["fn_and_main_present"] += n_fn_main_present

        word_results.append({
            "word": word, "letter": letter,
            "n_prompts": n_prompts,
            "n_raw_correct": n_raw_correct,
            "n_sae_correct": n_sae_correct,
            "n_fn": n_fn,
            "fn_rate": float(n_fn / max(n_raw_correct, 1)),
            "any_raw_correct": n_raw_correct > 0,
            "any_fn": n_fn > 0,
        })

        if n_fn > 0 and len(fn_examples) < 10:
            fn_examples.append({"word": word, "letter": letter,
                                "n_fn": n_fn, "n_prompts": n_prompts})

        processed += 1
        if processed % 100 == 0:
            total_fn = sum(s["false_negatives"] for s in per_letter_stats.values())
            total_corr = sum(s["probe_correct_raw"] for s in per_letter_stats.values())
            rate = total_fn / max(total_corr, 1)
            logger.info(f"  [{processed}/{len(test_words)}] abs_rate={rate:.4f} "
                       f"({total_fn}/{total_corr})")
            torch.cuda.empty_cache()

    total_raw = sum(s["probe_correct_raw"] for s in per_letter_stats.values())
    total_sae = sum(s["probe_correct_sae"] for s in per_letter_stats.values())
    total_fn = sum(s["false_negatives"] for s in per_letter_stats.values())
    total_fn_abs = sum(s["fn_and_main_absent"] for s in per_letter_stats.values())
    total_tokens = sum(s["total"] for s in per_letter_stats.values())

    absorption_rate = total_fn / max(total_raw, 1)
    strict_rate = total_fn_abs / max(total_raw, 1)

    logger.info(f"  Final: abs_rate={absorption_rate:.4f} ({total_fn}/{total_raw}), "
                f"strict={strict_rate:.4f}, probe_acc={total_raw/max(total_tokens,1):.4f}")

    return {
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "total_words": total_tokens,
        "total_unique_words": processed,
        "total_probe_correct_raw": total_raw,
        "total_probe_correct_sae": total_sae,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_abs,
        "probe_raw_accuracy": total_raw / max(total_tokens, 1),
        "word_results": word_results,
        "fn_examples": fn_examples,
    }


# ── Bootstrap CI ─────────────────────────────────────────────────
def bootstrap_absorption_rate(word_results, n_bootstrap=10000):
    eligible = [w for w in word_results if w["any_raw_correct"]]
    if not eligible:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0, "n": 0}
    fn_rates = np.array([w["fn_rate"] for w in eligible])
    rng = np.random.RandomState(SEED)
    boot_rates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(fn_rates), size=len(fn_rates), replace=True)
        boot_rates.append(fn_rates[idx].mean())
    boot_rates = sorted(boot_rates)
    lo = boot_rates[int(0.025 * n_bootstrap)]
    hi = boot_rates[min(int(0.975 * n_bootstrap), len(boot_rates) - 1)]
    return {
        "mean": float(fn_rates.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(boot_rates)),
        "n": len(eligible),
    }


# ── GPU progress / state ────────────────────────────────────────
def update_gpu_progress(elapsed_seconds, status="completed"):
    import filelock
    path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}}
            key = "completed" if status == "completed" else "failed"
            if TASK_ID not in data.get(key, []):
                data.setdefault(key, []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 55, "actual_min": round(elapsed_seconds / 60, 1),
            }
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")


def update_experiment_state(status="completed"):
    import filelock
    path = WORKSPACE / "exp" / "experiment_state.json"
    lock_path = WORKSPACE / "exp" / "experiment_state.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {
                "schema_version": 1, "tasks": {}}
            if TASK_ID in data.get("tasks", {}):
                data["tasks"][TASK_ID]["status"] = status
                data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"experiment_state update failed: {e}")


# ── Figure generation ────────────────────────────────────────────
def generate_figures(degradation_results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    f1s = np.array([r["actual_probe_f1"] for r in degradation_results])
    rates = np.array([r["absorption_rate"] for r in degradation_results])
    ci_lows = np.array([r["bootstrap_ci"]["ci_lower"] for r in degradation_results])
    ci_highs = np.array([r["bootstrap_ci"]["ci_upper"] for r in degradation_results])

    sort_idx = np.argsort(f1s)
    f1s, rates = f1s[sort_idx], rates[sort_idx]
    ci_lows, ci_highs = ci_lows[sort_idx], ci_highs[sort_idx]

    ax.plot(f1s, rates, 'b-o', markersize=8, linewidth=2,
            label='First-letter (degraded probe)', zorder=5)
    ax.fill_between(f1s, ci_lows, ci_highs, alpha=0.2, color='blue')

    colors = {'city-continent': 'red', 'city-country': 'green', 'city-language': 'orange'}
    markers = {'city-continent': 's', 'city-country': 'D', 'city-language': '^'}
    for h, ref in RAVEL_REFERENCE.items():
        ax.scatter(ref["probe_f1"], ref["absorption_rate"],
                   c=colors[h], marker=markers[h], s=120, linewidth=2,
                   edgecolors='black', zorder=10, label=f'{h} (RAVEL)')

    ax.set_xlabel('Probe F1 (weighted)', fontsize=12)
    ax.set_ylabel('Absorption Rate', fontsize=12)
    ax.set_title('Probe Degradation Ablation (H10): Absorption vs Probe Quality', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(0.65, 1.05)
    ax.set_ylim(0, max(0.55, max(rates) * 1.2))
    ax.grid(True, alpha=0.3)

    fig_path = output_dir / "fig_probe_degradation.pdf"
    fig.savefig(str(fig_path), dpi=300, bbox_inches='tight')
    png_path = output_dir / "fig_probe_degradation.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figures: {fig_path}, {png_path}")
    return str(fig_path)


# ── Main ─────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 12, "starting")

    logger.info("=" * 70)
    logger.info("=== Phase 1.1: Probe Degradation Ablation (H10) -- PILOT ===")
    logger.info(f"Target F1 levels: {TARGET_F1_LEVELS}")
    logger.info(f"SAE: L{SAE_LAYER} 16k, token_pos={TOKEN_POS}")
    logger.info("=" * 70)

    device = f"cuda:{GPU_ID}"

    # Step 1: Load model
    report_progress(1, 12, "loading_model")
    model = load_model(device)

    # Step 2: Load SAE
    report_progress(2, 12, "loading_sae")
    sae = load_sae(device)

    # Step 3: Prepare word lists (same split as iter_009)
    report_progress(3, 12, "preparing_words")
    all_words = get_word_list(model.tokenizer, max_words=2000)
    rng_split = random.Random(SEED)
    rng_split.shuffle(all_words)
    split_idx = int(len(all_words) * 0.6)
    probe_train_words = all_words[:split_idx]
    test_words = all_words[split_idx:][:MAX_TEST_WORDS]
    logger.info(f"Split: {len(probe_train_words)} probe-train, {len(test_words)} absorption-test")

    # Step 4: Build ICL prompts and cache activations for probe training
    report_progress(4, 12, "caching_probe_activations")
    probe_prompts = build_icl_prompts(
        probe_train_words, n_prompts_per_word=PROBE_TRAIN_PROMPTS,
        example_pool=probe_train_words,
    )
    X_probe, y_probe, w_probe = cache_activations(
        model, probe_prompts, layer=SAE_LAYER, token_pos=TOKEN_POS,
    )

    # Step 5: Train probe (this is our F1=1.0 control probe)
    report_progress(5, 12, "training_probe")
    original_probe, probe_metrics = train_probe(X_probe, y_probe)
    logger.info(f"Control probe: F1_w={probe_metrics['f1_weighted']:.4f}")

    # Create validation set for degradation calibration
    # Use 20% of the probe training data as calibration set
    X_train_probe, X_val_deg, y_train_probe, y_val_deg = train_test_split(
        X_probe, y_probe, test_size=0.2, random_state=SEED + 100, stratify=y_probe,
    )

    # Verify control probe F1 on validation set
    y_pred_ctrl = original_probe.predict(X_val_deg)
    ctrl_val_f1 = f1_score(y_val_deg, y_pred_ctrl, average="weighted")
    logger.info(f"Control probe F1 on degradation val set: {ctrl_val_f1:.4f}")

    # Free probe training data
    del X_probe, y_probe, probe_prompts
    gc.collect(); torch.cuda.empty_cache()

    # Step 6-10: For each target F1, degrade probe and measure absorption
    degradation_results = []

    for fi, target_f1 in enumerate(TARGET_F1_LEVELS):
        step = 6 + fi
        report_progress(step, 12, f"degradation_f1={target_f1:.2f}")

        logger.info(f"\n{'='*60}")
        logger.info(f"=== Level {fi+1}/{len(TARGET_F1_LEVELS)}: Target F1={target_f1:.2f} ===")

        # Degrade probe (calibrated on validation set)
        degraded_probe, actual_f1, noise_scale, method = degrade_probe_to_target_f1(
            original_probe, X_val_deg, y_val_deg, target_f1,
        )

        # Full F1 metrics
        y_pred_deg = degraded_probe.predict(X_val_deg)
        f1_macro = f1_score(y_val_deg, y_pred_deg, average="macro")
        acc_deg = accuracy_score(y_val_deg, y_pred_deg)

        # Measure absorption with this probe
        logger.info(f"  Measuring absorption (F1={actual_f1:.4f})...")
        abs_result = measure_absorption(
            model, sae, degraded_probe, test_words,
            layer=SAE_LAYER, token_pos=TOKEN_POS,
            n_prompts_per_word=N_PROMPTS_PER_WORD,
            example_pool=all_words, device=device,
        )

        # Bootstrap CI
        boot_ci = bootstrap_absorption_rate(abs_result["word_results"], N_BOOTSTRAP)

        entry = {
            "target_f1": target_f1,
            "actual_probe_f1": actual_f1,
            "actual_probe_f1_macro": f1_macro,
            "actual_probe_accuracy": acc_deg,
            "noise_scale": noise_scale,
            "degradation_method": method,
            "absorption_rate": abs_result["absorption_rate"],
            "strict_absorption_rate": abs_result["strict_absorption_rate"],
            "probe_raw_accuracy": abs_result["probe_raw_accuracy"],
            "n_false_negatives": abs_result["total_false_negatives"],
            "n_probe_correct_raw": abs_result["total_probe_correct_raw"],
            "n_probe_correct_sae": abs_result["total_probe_correct_sae"],
            "n_total_tokens": abs_result["total_words"],
            "n_unique_words": abs_result["total_unique_words"],
            "bootstrap_ci": boot_ci,
        }
        degradation_results.append(entry)

        logger.info(f"  RESULT: F1={actual_f1:.4f} -> abs={abs_result['absorption_rate']:.4f} "
                    f"[{boot_ci['ci_lower']:.4f}, {boot_ci['ci_upper']:.4f}] "
                    f"FN={abs_result['total_false_negatives']}/{abs_result['total_probe_correct_raw']}")

        torch.cuda.empty_cache(); gc.collect()

    # Step 11: Statistical analysis
    report_progress(11, 12, "analysis")
    logger.info(f"\n{'='*60}")
    logger.info("=== Statistical Analysis ===")

    f1_vals = [r["actual_probe_f1"] for r in degradation_results]
    abs_vals = [r["absorption_rate"] for r in degradation_results]

    slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(f1_vals, abs_vals)
    logger.info(f"Linear: abs = {slope:.4f}*F1 + {intercept:.4f}, "
                f"R^2={r_val**2:.4f}, p={p_val:.4f}")

    spearman_rho, spearman_p = scipy_stats.spearmanr(f1_vals, abs_vals)
    logger.info(f"Spearman: rho={spearman_rho:.4f}, p={spearman_p:.4f}")

    # Extrapolate to RAVEL F1 levels
    extrapolation = {}
    for h, ref in RAVEL_REFERENCE.items():
        pred = slope * ref["probe_f1"] + intercept
        delta = ref["absorption_rate"] - pred
        extrapolation[h] = {
            "ravel_f1": ref["probe_f1"],
            "ravel_absorption": ref["absorption_rate"],
            "predicted_from_curve": round(pred, 4),
            "delta": round(delta, 4),
            "within_5pp": abs(delta) <= 0.05,
            "within_10pp": abs(delta) <= 0.10,
        }
        logger.info(f"  {h}: RAVEL={ref['absorption_rate']:.4f}, "
                    f"pred={pred:.4f}, delta={delta:+.4f}")

    # Verdict at F1~0.80
    f1_080 = next((r for r in degradation_results
                   if abs(r["actual_probe_f1"] - 0.80) < 0.05), None)

    verdict_details = {}
    if f1_080:
        fl_at_080 = f1_080["absorption_rate"]
        ravel_rates = [ref["absorption_rate"] for ref in RAVEL_REFERENCE.values()]
        ravel_min, ravel_max = min(ravel_rates), max(ravel_rates)
        ravel_mean = np.mean(ravel_rates)

        within_range = ravel_min <= fl_at_080 <= ravel_max
        within_5pp = abs(fl_at_080 - ravel_mean) <= 0.05

        if within_range or within_5pp:
            verdict = "PROBE_ARTIFACT"
            explanation = (
                f"At F1~0.80, first-letter absorption ({fl_at_080:.4f}) falls within "
                f"RAVEL range [{ravel_min:.4f}, {ravel_max:.4f}]. "
                f"Variation is primarily a probe quality artifact."
            )
        else:
            verdict = "GENUINE_HIERARCHY_EFFECT"
            explanation = (
                f"At F1~0.80, first-letter absorption ({fl_at_080:.4f}) is outside "
                f"RAVEL range [{ravel_min:.4f}, {ravel_max:.4f}]. "
                f"Variation is a genuine hierarchy-dependent effect."
            )

        verdict_details = {
            "verdict": verdict,
            "explanation": explanation,
            "firstletter_at_matched_f1": fl_at_080,
            "matched_f1": f1_080["actual_probe_f1"],
            "ravel_range": [ravel_min, ravel_max],
            "ravel_mean": float(ravel_mean),
            "within_range": within_range,
            "within_5pp_of_mean": within_5pp,
        }
        logger.info(f"\nVERDICT: {verdict}")
        logger.info(explanation)

    # Control check vs iter_009
    f1_100 = next((r for r in degradation_results if r["target_f1"] >= 0.99), None)
    control_check = {}
    if f1_100:
        ctrl_rate = f1_100["absorption_rate"]
        base_rate = ITER009_FIRSTLETTER_BASELINE["absorption_rate"]
        delta = ctrl_rate - base_rate
        within_ci = (ITER009_FIRSTLETTER_BASELINE["ci_lower"] <= ctrl_rate <=
                     ITER009_FIRSTLETTER_BASELINE["ci_upper"])
        control_check = {
            "control_rate": ctrl_rate, "iter009_baseline": base_rate,
            "delta": delta, "within_iter009_ci": within_ci,
        }
        logger.info(f"\nControl: rate={ctrl_rate:.4f}, iter009={base_rate:.4f}, "
                    f"delta={delta:+.4f}, within_CI={within_ci}")

    # Generate figures
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig_path = generate_figures(degradation_results, figures_dir)
    except Exception as e:
        logger.warning(f"Figure gen failed: {e}")
        fig_path = None

    # Summary table
    logger.info(f"\n{'='*110}")
    logger.info("PROBE DEGRADATION ABLATION SUMMARY")
    logger.info(f"{'F1_tgt':>8s} {'F1_act':>8s} {'AbsRate':>8s} {'Strict':>8s} "
                f"{'PrbAcc':>8s} {'CI_lo':>8s} {'CI_hi':>8s} "
                f"{'FN':>5s} {'Corr':>6s} {'Noise':>8s}")
    logger.info("-" * 110)
    for r in degradation_results:
        logger.info(
            f"{r['target_f1']:>8.2f} {r['actual_probe_f1']:>8.4f} "
            f"{r['absorption_rate']:>8.4f} {r['strict_absorption_rate']:>8.4f} "
            f"{r['probe_raw_accuracy']:>8.4f} "
            f"{r['bootstrap_ci']['ci_lower']:>8.4f} {r['bootstrap_ci']['ci_upper']:>8.4f} "
            f"{r['n_false_negatives']:>5d} {r['n_probe_correct_raw']:>6d} "
            f"{r['noise_scale']:>8.4f}"
        )
    logger.info("=" * 110)

    elapsed = time.time() - start_time

    # Compile final results
    results = {
        "task_id": TASK_ID, "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED, "model": "gemma-2-2b",
        "sae_config": {"layer": SAE_LAYER, "width": "16k",
                        "release": SAE_RELEASE, "sae_id": SAE_ID},
        "token_pos": TOKEN_POS,
        "n_prompts_per_word": N_PROMPTS_PER_WORD,
        "n_bootstrap": N_BOOTSTRAP,
        "probe_metrics": probe_metrics,
        "target_f1_levels": TARGET_F1_LEVELS,
        "degradation_results": degradation_results,
        "statistical_analysis": {
            "linear_regression": {
                "slope": slope, "intercept": intercept,
                "r_squared": r_val ** 2, "p_value": p_val, "std_err": std_err,
            },
            "spearman": {"rho": spearman_rho, "p_value": spearman_p},
            "extrapolation_to_ravel": extrapolation,
        },
        "verdict": verdict_details,
        "control_check": control_check,
        "ravel_reference": RAVEL_REFERENCE,
        "iter009_baseline": ITER009_FIRSTLETTER_BASELINE,
        "figure_path": fig_path,
        "elapsed_seconds": elapsed, "elapsed_minutes": elapsed / 60,
        "pass_criteria": {
            "all_levels_computed": len(degradation_results) == len(TARGET_F1_LEVELS),
            "control_matches_baseline": control_check.get("within_iter009_ci", False),
            "verdict_reached": bool(verdict_details),
        },
    }

    out_path = PHASE1_DIR / "probe_degradation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    del model, sae
    gc.collect(); torch.cuda.empty_cache()

    summary = (
        f"Phase 1.1 probe degradation H10. "
        f"Levels: {len(degradation_results)}/{len(TARGET_F1_LEVELS)}. "
    )
    if verdict_details:
        summary += f"Verdict: {verdict_details.get('verdict', '?')}. "
    if control_check:
        summary += f"Control delta={control_check.get('delta', 0):+.4f}. "
    summary += f"R^2={r_val**2:.4f}. Time: {elapsed/60:.1f}min."

    pass_all = all(results["pass_criteria"].values())
    mark_done("success" if pass_all else "partial", summary)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")

    report_progress(12, 12, "completed", {
        "verdict": verdict_details.get("verdict", "unknown"),
        "elapsed_min": round(elapsed / 60, 1),
    })

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE in {elapsed/60:.1f} min: {summary}")
    logger.info(f"{'='*70}")

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress(0, "failed")
        update_experiment_state("failed")
        sys.exit(1)
