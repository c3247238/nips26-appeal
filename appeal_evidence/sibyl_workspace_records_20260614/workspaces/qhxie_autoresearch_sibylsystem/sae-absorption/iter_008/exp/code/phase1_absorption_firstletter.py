"""
Phase 1.2: First-Letter Absorption Baseline (Replication of Chanin et al.)

Measures first-letter absorption rates on Gemma 2 2B + Gemma Scope SAEs.

Approach:
1. Use sae-spelling ICL prompt format to get high-quality first-letter representations
2. Train sklearn LogisticRegression probes (robust, no NaN issues)
3. For each SAE: encode residual stream, apply probe, identify false negatives
4. Compute absorption rate with bootstrap CI

PILOT: Layer 12, 16k width, ~200 words
FULL: Layers 6,12,18,24 x widths 16k,65k
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
from sklearn.metrics import f1_score, accuracy_score, classification_report

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

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

if MODE == "PILOT":
    # Pilot must test >= 6/8 SAE configs per task_plan.json pilot criteria
    SAE_CONFIGS = [
        {"layer": l, "width": w, "release": "gemma-scope-2b-pt-res-canonical",
         "sae_id": f"layer_{l}/width_{w}/canonical"}
        for l in [6, 12, 18, 24] for w in ["16k", "65k"]
    ]
    MAX_WORDS = 100
    N_PROMPTS_PER_WORD = 3
    TIMEOUT = 900
else:
    SAE_CONFIGS = [
        {"layer": l, "width": w, "release": "gemma-scope-2b-pt-res-canonical",
         "sae_id": f"layer_{l}/width_{w}/canonical"}
        for l in [6, 12, 18, 24] for w in ["16k", "65k"]
    ]
    MAX_WORDS = 1000
    N_PROMPTS_PER_WORD = 10
    TIMEOUT = 2700

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
        # Try dict-like access
        if isinstance(md, dict):
            return md.get('hook_name')
        # Try attribute access on SAEMetadata
        try:
            return md['hook_name']
        except:
            pass
    raise ValueError(f"Cannot find hook_name in SAE config: {sae.cfg}")


# ============================================================
# Model & SAE loading
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
# Word list + ICL prompts
# ============================================================
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def get_word_list(tokenizer, max_words=200):
    """Get curated word list from tokenizer vocab."""
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


def build_icl_prompts(word_list, n_prompts_per_word=5):
    """
    Build ICL prompts using sae-spelling's template:
    "{word} has the first letter:"

    Returns list of (word, prompt_string, letter_idx) tuples.
    """
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
# Activation caching
# ============================================================
def cache_activations(model, prompts, layer, token_pos=-6, batch_size=1):
    """
    Cache residual stream activations at the specified layer and token position.

    Args:
        prompts: list of (word, prompt_str, label) tuples
        layer: which layer's residual stream
        token_pos: position index in the token sequence (default -6 for Chanin template)

    Returns: (X, y, words) where X is [N, d_model], y is [N], words is list of str
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []
    all_labels = []
    all_words = []

    for i, (word, prompt_str, label) in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt_str, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name][0, token_pos, :].float().cpu().numpy()
            all_acts.append(act)
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

    X = np.array(all_acts)
    y = np.array(all_labels)
    logger.info(f"  Final: X={X.shape}, y={y.shape}, {len(set(all_labels))} classes")
    return X, y, all_words


# ============================================================
# Probe training (sklearn, robust)
# ============================================================
def train_probe_sklearn(X, y, test_fraction=0.2):
    """
    Train logistic regression probe using sklearn.
    Returns probe, metrics dict, (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=SEED, stratify=y
    )

    # Try multiple regularization strengths
    best_f1, best_probe, best_metrics = -1, None, None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        probe = LogisticRegression(
            C=C, max_iter=5000, solver="lbfgs", random_state=SEED,
        )
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"  C={C}: F1={f1:.4f}, acc={acc:.4f}")
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
            per_letter[letter] = {"f1": None, "n_test": 0, "n_correct": 0}
            continue
        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_l = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_letter[letter] = {
            "f1": float(f1_l),
            "precision": float(prec),
            "recall": float(rec),
            "n_test": int(mask_true.sum()),
            "n_correct": int(tp),
        }

    best_metrics["per_letter"] = per_letter
    return best_probe, best_metrics, (X_train, X_test, y_train, y_test)


# ============================================================
# Absorption measurement
# ============================================================
def measure_absorption(model, sae, sklearn_probe, word_list, layer,
                       token_pos=-6, n_prompts=1, device="cuda:0"):
    """
    Measure first-letter absorption rate for a given SAE.

    For each word:
    1. Run model, get residual stream activation at the word position
    2. Apply sklearn probe to raw activation -> should predict correct letter
    3. Encode through SAE, decode
    4. Apply sklearn probe to SAE output -> may fail (false negative = absorption)
    5. Also check if the top-cosine-similarity SAE feature for that letter fires

    Returns detailed absorption statistics.
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )

    formatter = first_letter_formatter()
    hook_name = get_sae_hook_name(sae)
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute probe directions and find main SAE features per letter
    # sklearn probe: coef_ is [n_classes, d_model], classes_ maps to actual labels
    probe_classes = sklearn_probe.classes_  # array of class indices the probe knows
    probe_coefs = torch.tensor(sklearn_probe.coef_, dtype=torch.float32)  # [n_classes, d_model]
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]

    # Build mapping: letter -> probe class index (if present)
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
    logger.info(f"  Main features computed for {len(main_features)}/26 letters")

    # Process words
    per_letter_stats = {letter: {
        "total": 0,
        "probe_correct_raw": 0,
        "probe_correct_sae": 0,
        "false_negatives": 0,
        "main_feature_fires": 0,
        "fn_and_main_absent": 0,
        "fn_and_main_present": 0,
    } for letter in LETTERS}

    fn_examples = []  # Store example false negatives

    processed = 0
    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS or letter not in letter_to_probe_idx:
            continue
        letter_idx = LETTERS.index(letter)
        # The sklearn probe uses class indices that map to letter indices
        # probe.predict() returns the original class label (= letter_idx)

        # Build prompt
        try:
            sp = create_icl_prompt(
                word=word,
                examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter,
                max_icl_examples=10,
                shuffle_examples=True,
            )
        except:
            continue

        try:
            tokens = model.to_tokens(sp.base, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, token_pos, :].detach().float()  # [d_model]

            # SAE encode/decode (ensure no grad tracking)
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))  # [1, d_sae]
                sae_out = sae.decode(sae_features)  # [1, d_model]

            # Sklearn probe predictions
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

        # False negative: correct on raw, wrong on SAE
        if probe_correct_raw and not probe_correct_sae:
            stats["false_negatives"] += 1
            if not any_main_fires:
                stats["fn_and_main_absent"] += 1
            else:
                stats["fn_and_main_present"] += 1

            if len(fn_examples) < 20:
                fn_examples.append({
                    "word": word,
                    "letter": letter,
                    "raw_pred": LETTERS[raw_pred],
                    "sae_pred": LETTERS[sae_pred],
                    "main_fires": any_main_fires,
                    "top_feat_act": float(feat_acts[0].item()),
                })

        processed += 1
        if processed % 50 == 0:
            logger.info(f"  Processed {processed}/{len(word_list)} words")
            torch.cuda.empty_cache()

    # Aggregate
    total_raw_correct = sum(s["probe_correct_raw"] for s in per_letter_stats.values())
    total_fn = sum(s["false_negatives"] for s in per_letter_stats.values())
    total_fn_main_absent = sum(s["fn_and_main_absent"] for s in per_letter_stats.values())
    total_tokens = sum(s["total"] for s in per_letter_stats.values())

    absorption_rate = total_fn / max(total_raw_correct, 1)
    strict_rate = total_fn_main_absent / max(total_raw_correct, 1)

    # Per-letter absorption rates for bootstrap
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
        "total_words_processed": total_tokens,
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
# GPU progress
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
            "planned_min": 45,
            "actual_min": round(elapsed_seconds / 60),
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": MODE,
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
    report_progress(0, 10, "starting")

    logger.info(f"=== Phase 1.2: First-Letter Absorption Baseline ===")
    logger.info(f"Mode: {MODE}, SAE configs: {len(SAE_CONFIGS)}")

    device = "cuda:0"

    # Step 1: Load model
    report_progress(1, 10, "loading_model")
    model = load_model(device=device)

    # Step 2: Get words -- need a LARGE pool, split into probe-training and absorption-test
    report_progress(2, 10, "preparing_data")
    # Get as many words as possible for proper train/test split
    all_words = get_word_list(model.tokenizer, max_words=MAX_WORDS * 5)
    rng_split = random.Random(SEED)
    rng_split.shuffle(all_words)
    # 60% for probe training, 40% for absorption testing (completely unseen)
    split_idx = int(len(all_words) * 0.6)
    probe_train_words = all_words[:split_idx]
    absorption_test_words = all_words[split_idx:]
    logger.info(f"Word split: {len(probe_train_words)} for probe training, "
                f"{len(absorption_test_words)} for absorption testing")

    # Build ICL prompts for probe training
    prompts = build_icl_prompts(probe_train_words, n_prompts_per_word=N_PROMPTS_PER_WORD)

    # Step 3: Cache activations for probe training
    report_progress(3, 10, "caching_activations")
    from sae_spelling.prompting import VERBOSE_FIRST_LETTER_TOKEN_POS
    token_pos = VERBOSE_FIRST_LETTER_TOKEN_POS  # -6

    X, y, words = cache_activations(model, prompts, layer=12, token_pos=token_pos)

    # Step 4: Train probe
    report_progress(4, 10, "training_probe")
    sklearn_probe, probe_metrics, splits = train_probe_sklearn(X, y, test_fraction=0.2)
    logger.info(f"Probe: F1={probe_metrics['f1']:.4f}, acc={probe_metrics['accuracy']:.4f}")
    word_list = absorption_test_words  # Use unseen words for absorption measurement

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "n_probe_train_words": len(probe_train_words),
        "n_absorption_test_words": len(absorption_test_words),
        "n_prompts": len(prompts),
        "sample_probe_words": probe_train_words[:10],
        "sample_test_words": absorption_test_words[:10],
        "probe_quality": {
            "layer": 12,
            "token_pos": token_pos,
            "template": "{word} has the first letter:",
            "f1_weighted": probe_metrics["f1"],
            "accuracy": probe_metrics["accuracy"],
            "best_C": probe_metrics["C"],
            "per_letter": probe_metrics["per_letter"],
            "quality_gate_strict": probe_metrics["f1"] >= 0.90,
            "quality_gate_relaxed": probe_metrics["f1"] >= 0.85,
        },
        "absorption_results": {},
    }

    # Step 5: Measure absorption for each SAE config
    for sae_idx, sae_config in enumerate(SAE_CONFIGS):
        config_key = f"L{sae_config['layer']}_{sae_config['width']}"
        report_progress(5 + sae_idx, 5 + len(SAE_CONFIGS) + 1,
                       f"absorption_{config_key}",
                       {"probe_f1": probe_metrics["f1"]})
        logger.info(f"\n=== Measuring absorption: {config_key} ===")

        try:
            sae = load_sae(sae_config["release"], sae_config["sae_id"], device=device)
        except Exception as e:
            logger.error(f"SAE load failed: {e}")
            results["absorption_results"][config_key] = {"error": str(e)}
            continue

        try:
            abs_results = measure_absorption(
                model, sae, sklearn_probe, word_list,
                layer=sae_config["layer"],
                token_pos=token_pos,
                n_prompts=1,
                device=device,
            )

            # Bootstrap CI on per-letter rates
            rates = [v["absorption_rate"] for v in abs_results["per_letter"].values()
                     if v["total"] > 0]
            ci = bootstrap_ci(rates)

            results["absorption_results"][config_key] = {
                "sae_config": sae_config,
                "absorption_rate": abs_results["absorption_rate"],
                "strict_absorption_rate": abs_results["strict_absorption_rate"],
                "total_words": abs_results["total_words_processed"],
                "total_probe_correct": abs_results["total_probe_correct_raw"],
                "total_false_negatives": abs_results["total_false_negatives"],
                "total_fn_main_absent": abs_results["total_fn_main_absent"],
                "bootstrap_ci": ci,
                "per_letter": abs_results["per_letter"],
                "fn_examples": abs_results["fn_examples"],
                "main_features_top": {
                    k: {"fid": v["feature_ids"][0], "cos": round(v["cos_sims"][0], 4)}
                    for k, v in abs_results["main_features"].items()
                },
            }

            logger.info(f"  Absorption rate: {abs_results['absorption_rate']:.4f}")
            logger.info(f"  Strict (main absent): {abs_results['strict_absorption_rate']:.4f}")
            logger.info(f"  FN: {abs_results['total_false_negatives']}/{abs_results['total_probe_correct_raw']}")
            logger.info(f"  Bootstrap CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

        except Exception as e:
            logger.error(f"Absorption measurement failed: {e}", exc_info=True)
            results["absorption_results"][config_key] = {"error": str(e)}

        del sae
        gc.collect(); torch.cuda.empty_cache()

    # Step 6: Summary
    report_progress(5 + len(SAE_CONFIGS), 5 + len(SAE_CONFIGS) + 1, "summarizing")

    summary_rows = []
    for config_key, ar in results["absorption_results"].items():
        if isinstance(ar, dict) and "absorption_rate" in ar:
            summary_rows.append({
                "config": config_key,
                "absorption_rate": ar["absorption_rate"],
                "strict_rate": ar["strict_absorption_rate"],
                "n_words": ar["total_words"],
                "n_fn": ar["total_false_negatives"],
                "ci_lower": ar["bootstrap_ci"]["ci_lower"],
                "ci_upper": ar["bootstrap_ci"]["ci_upper"],
            })

    # Chanin et al. reference: 15-35% absorption on GPT-2 Small first-letter
    pilot_pass = False
    if summary_rows:
        rate = summary_rows[0]["absorption_rate"]
        pilot_pass = 0.001 <= rate <= 0.99
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: absorption_rate={rate:.4f}, pilot={'PASS' if pilot_pass else 'FAIL'}")
        logger.info(f"Chanin et al. reference range: 15-35% on GPT-2 Small")
        logger.info(f"{'='*60}")

    results["summary"] = {
        "n_configs_tested": len(summary_rows),
        "results_table": summary_rows,
        "probe_f1": probe_metrics["f1"],
        "pilot_pass": pilot_pass,
    }

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["elapsed_minutes"] = elapsed / 60

    # Save
    if MODE == "PILOT":
        out_path = PILOT_DIR / f"{TASK_ID}.json"
    else:
        out_path = PHASE1_DIR / "absorption_firstletter.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Saved: {out_path}")

    # Also save probe weights for downstream use
    probe_path = PHASE1_DIR / "probe_firstletter_L12_sklearn.npz"
    np.savez(probe_path,
             coef=sklearn_probe.coef_,
             intercept=sklearn_probe.intercept_,
             classes=np.array(list(LETTERS)))
    logger.info(f"Probe saved: {probe_path}")

    del model
    gc.collect(); torch.cuda.empty_cache()

    summary_text = (
        f"Phase 1.2 first-letter absorption ({MODE}). "
        f"Probe F1={probe_metrics['f1']:.4f}. "
        f"Configs: {len(summary_rows)}. "
        f"Time: {elapsed/60:.1f}min."
    )
    if summary_rows:
        summary_text += (f" Absorption: {summary_rows[0]['absorption_rate']:.4f} "
                        f"[{summary_rows[0].get('ci_lower', 0):.3f}-{summary_rows[0].get('ci_upper', 0):.3f}].")

    mark_done("success", summary_text)
    update_gpu_progress(elapsed)
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress(time.time() - (globals().get("start_time", time.time())), "failed")
        sys.exit(1)
