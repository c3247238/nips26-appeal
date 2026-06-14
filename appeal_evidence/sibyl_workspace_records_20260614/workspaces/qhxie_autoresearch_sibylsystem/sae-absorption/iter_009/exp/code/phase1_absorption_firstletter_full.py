"""
Phase 1.2a: First-Letter Absorption Measurement (8 SAE configs) -- FULL MODE
Iteration 9.

Measures first-letter absorption rates at L6/L12/L18/L24 x 16k/65k JumpReLU SAEs.

Probe strategy:
  Train per-layer sklearn probes at position -6 (VERBOSE_FIRST_LETTER_TOKEN_POS = word token)
  using the same ICL prompt template as absorption measurement.
  60/40 train/test split with large word pool for high-quality probes.

Pipeline per SAE config:
  1. Load model + SAE at layer L
  2. For each test word x multiple contexts:
     - Get residual stream at word token position (-6)
     - Apply sklearn probe to raw activation -> correct letter? (clean prediction)
     - SAE encode/decode -> apply probe -> correct letter? (SAE prediction)
     - False negative = correct on raw, wrong on SAE (= absorption)
  3. Classify FNs: check if main letter feature fires (absorbed vs hedged)
  4. Bootstrap 95% CI on absorption rate
  5. Random direction baseline
  6. Compare with iter_008 baselines

Output: exp/results/phase1/absorption_firstletter.json
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

# ── Configuration ────────────────────────────────────────────
TASK_ID = "phase1_absorption_firstletter"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
FULL_DIR = RESULTS_DIR / "full"
for d in [PILOT_DIR, PHASE1_DIR, FULL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = "FULL"

# All 8 SAE configs (L6/L12/L18/L24 x 16k/65k)
SAE_CONFIGS = [
    {"layer": l, "width": w, "release": "gemma-scope-2b-pt-res-canonical",
     "sae_id": f"layer_{l}/width_{w}/canonical"}
    for l in [6, 12, 18, 24] for w in ["16k", "65k"]
]

MAX_TEST_WORDS = 500          # test words for absorption measurement
N_PROMPTS_PER_WORD = 3        # prompts per word for absorption (majority vote)
PROBE_TRAIN_PROMPTS = 5       # prompts per word for probe training (more data)
TOKEN_POS = -6                # word token position (VERBOSE_FIRST_LETTER_TOKEN_POS)
TIMEOUT = 5400                # 90 min timeout

LETTERS = "abcdefghijklmnopqrstuvwxyz"
GPU_ID = int(os.environ.get("CUDA_GPU_ID", "0"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ── Progress tracking ────────────────────────────────────────
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


# ── Utility ──────────────────────────────────────────────────
def get_sae_hook_name(sae):
    if hasattr(sae.cfg, 'hook_name'):
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


# ── Model loading ────────────────────────────────────────────
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
    gc.collect()
    torch.cuda.empty_cache()
    return model


def load_sae(release, sae_id, device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    hook_name = get_sae_hook_name(sae)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}, hook={hook_name}")
    return sae


# ── Word list ────────────────────────────────────────────────
def get_word_list(tokenizer, max_words=2000):
    """Get balanced word list from tokenizer vocab."""
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


# ── ICL prompt building ──────────────────────────────────────
def build_icl_prompts(word_list, n_prompts_per_word=5, example_pool=None):
    """Build ICL prompts using sae-spelling template."""
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
                    word=word,
                    examples=example_pool,
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


# ── Activation caching ──────────────────────────────────────
def cache_activations_multi_layer(model, prompts, layers, token_pos=-6):
    """Cache activations at multiple layers simultaneously for probe training."""
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    acts = {l: [] for l in layers}
    labels = []
    words = []

    for i, (word, prompt_str, label) in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt_str, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_names)
            for l, hn in zip(layers, hook_names):
                act = cache[hn][0, token_pos, :].float().cpu().numpy()
                acts[l].append(act)
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

    result = {}
    for l in layers:
        X = np.array(acts[l])
        result[l] = X
    y = np.array(labels)
    logger.info(f"  Cached: {len(labels)} samples, layers {layers}, "
                f"X shapes: {[result[l].shape for l in layers]}")
    return result, y, words


# ── Probe training ───────────────────────────────────────────
def train_probe(X, y, layer):
    """Train sklearn probe with cross-validation of C parameter."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    best_f1, best_probe, best_C = -1, None, None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            C=C, max_iter=5000, solver="lbfgs", random_state=SEED
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"    L{layer} C={C}: F1={f1:.4f}, Acc={acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_probe = clf
            best_C = C

    # Test set metrics
    y_pred_final = best_probe.predict(X_test)
    f1_final = f1_score(y_test, y_pred_final, average="weighted")
    f1_macro = f1_score(y_test, y_pred_final, average="macro")
    acc_final = accuracy_score(y_test, y_pred_final)

    # Per-letter F1
    per_letter_f1 = {}
    for i, letter in enumerate(LETTERS):
        mask = (y_test == i)
        if mask.sum() == 0:
            per_letter_f1[letter] = None
            continue
        mask_pred = (y_pred_final == i)
        tp = int((mask & mask_pred).sum())
        fp = int((~mask & mask_pred).sum())
        fn = int((mask & ~mask_pred).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_l = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_letter_f1[letter] = round(f1_l, 4)

    metrics = {
        "layer": layer,
        "position": TOKEN_POS,
        "best_C": best_C,
        "f1_weighted": float(f1_final),
        "f1_macro": float(f1_macro),
        "accuracy": float(acc_final),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(best_probe.classes_),
        "per_letter_f1": per_letter_f1,
        "quality_gate_strict": f1_final >= 0.90,
        "quality_gate_relaxed": f1_final >= 0.80,
    }
    logger.info(f"  Best L{layer}: C={best_C}, F1_w={f1_final:.4f}, "
                f"F1_m={f1_macro:.4f}, Acc={acc_final:.4f}, "
                f"strict={'PASS' if metrics['quality_gate_strict'] else 'FAIL'}")
    return best_probe, metrics


# ── Absorption measurement ───────────────────────────────────
def measure_absorption_for_sae(model, sae, sklearn_probe, word_list,
                                layer, token_pos=-6, n_prompts_per_word=3,
                                example_pool=None, device="cuda:0"):
    """
    Measure first-letter absorption for one SAE config.

    For each word x n_prompts_per_word contexts:
      1. Get raw residual stream at word token position
      2. Probe predicts letter on raw -> correct? (clean prediction)
      3. SAE encode/decode -> probe on SAE output -> correct? (SAE prediction)
      4. FN = correct on raw, wrong on SAE
      5. Check main letter feature firing for FN classification
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    if example_pool is None:
        example_pool = word_list
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute main SAE features per letter (most probe-aligned)
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
    logger.info(f"  Main features computed for {len(main_features)}/{len(probe_classes)} letters")

    # Per-letter stats
    per_letter_stats = {letter: {
        "total": 0,
        "probe_correct_raw": 0,
        "probe_correct_sae": 0,
        "false_negatives": 0,
        "main_feature_fires": 0,
        "fn_and_main_absent": 0,
        "fn_and_main_present": 0,
    } for letter in LETTERS}

    fn_examples = []
    word_results = []  # Per-word for bootstrap

    processed = 0
    skipped = 0

    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS:
            continue
        letter_idx = LETTERS.index(letter)
        if letter_idx not in letter_to_probe_row:
            skipped += 1
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
                    feat_acts = torch.zeros(5)

                del cache

                prompt_results.append({
                    "probe_correct_raw": bool(probe_correct_raw),
                    "probe_correct_sae": bool(probe_correct_sae),
                    "is_fn": bool(probe_correct_raw and not probe_correct_sae),
                    "main_fires": bool(any_main_fires),
                    "raw_pred": raw_pred,
                    "sae_pred": sae_pred,
                    "top_feat_act": float(feat_acts[0].item()) if len(feat_acts) > 0 else 0.0,
                })
            except Exception as e:
                if processed < 3:
                    logger.warning(f"Error processing '{word}': {e}")
                continue

        if not prompt_results:
            continue

        # Aggregate across prompts for this word
        n_raw_correct = sum(1 for p in prompt_results if p["probe_correct_raw"])
        n_sae_correct = sum(1 for p in prompt_results if p["probe_correct_sae"])
        n_fn = sum(1 for p in prompt_results if p["is_fn"])
        n_main_fires = sum(1 for p in prompt_results if p["main_fires"])
        n_fn_main_absent = sum(1 for p in prompt_results if p["is_fn"] and not p["main_fires"])
        n_fn_main_present = sum(1 for p in prompt_results if p["is_fn"] and p["main_fires"])
        n_prompts = len(prompt_results)

        stats = per_letter_stats[letter]
        stats["total"] += n_prompts
        stats["probe_correct_raw"] += n_raw_correct
        stats["probe_correct_sae"] += n_sae_correct
        stats["false_negatives"] += n_fn
        stats["main_feature_fires"] += n_main_fires
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

        if n_fn > 0 and len(fn_examples) < 30:
            fn_p = next((p for p in prompt_results if p["is_fn"]), None)
            if fn_p:
                fn_examples.append({
                    "word": word, "letter": letter,
                    "raw_pred": LETTERS[fn_p["raw_pred"]] if 0 <= fn_p["raw_pred"] < 26 else "?",
                    "sae_pred": LETTERS[fn_p["sae_pred"]] if 0 <= fn_p["sae_pred"] < 26 else "?",
                    "main_fires": fn_p["main_fires"],
                    "top_feat_act": fn_p["top_feat_act"],
                    "n_fn_of_n_prompts": f"{n_fn}/{n_prompts}",
                })

        processed += 1
        if processed % 100 == 0:
            total_fn = sum(s["false_negatives"] for s in per_letter_stats.values())
            total_corr = sum(s["probe_correct_raw"] for s in per_letter_stats.values())
            rate = total_fn / max(total_corr, 1)
            logger.info(f"  Processed {processed}/{len(word_list)} words, "
                       f"abs rate: {rate:.4f} ({total_fn}/{total_corr})")
            torch.cuda.empty_cache()

    if skipped > 0:
        logger.info(f"  Skipped {skipped} words (letter not in probe classes)")

    # Aggregate
    total_raw_correct = sum(s["probe_correct_raw"] for s in per_letter_stats.values())
    total_sae_correct = sum(s["probe_correct_sae"] for s in per_letter_stats.values())
    total_fn = sum(s["false_negatives"] for s in per_letter_stats.values())
    total_fn_main_absent = sum(s["fn_and_main_absent"] for s in per_letter_stats.values())
    total_tokens = sum(s["total"] for s in per_letter_stats.values())

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
        "total_words": total_tokens,
        "total_unique_words": processed,
        "total_probe_correct_raw": total_raw_correct,
        "total_probe_correct_sae": total_sae_correct,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_main_absent,
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "per_letter": per_letter_rates,
        "fn_examples": fn_examples,
        "word_results": word_results,
        "main_features": {k: v for k, v in main_features.items()},
    }


# ── Random baseline ─────────────────────────────────────────
def random_baseline_absorption(model, sae, word_list, layer, token_pos=-6,
                                device="cuda:0"):
    """Random probe should have near-chance accuracy, demonstrating
    that systematic absorption requires a real probe."""
    from sae_spelling.prompting import (
        create_icl_prompt, first_letter_formatter, VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    tl_hook = f"blocks.{layer}.hook_resid_post"

    rng = np.random.RandomState(SEED + 999)
    d_model = 2304
    random_coef = rng.randn(26, d_model).astype(np.float64)
    random_coef /= np.linalg.norm(random_coef, axis=1, keepdims=True)

    random_probe = LogisticRegression(max_iter=1)
    random_probe.coef_ = random_coef
    random_probe.intercept_ = np.zeros(26, dtype=np.float64)
    random_probe.classes_ = np.arange(26)

    total_correct_raw = 0
    total_correct_sae = 0
    total_fn = 0
    total_words = 0

    test_words = word_list[:min(100, len(word_list))]
    for word in test_words:
        letter_idx = LETTERS.index(word[0].lower())
        try:
            sp = create_icl_prompt(
                word=word, examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter,
                max_icl_examples=10, shuffle_examples=True,
            )
            tokens = model.to_tokens(sp.base, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])
            raw_act = cache[tl_hook][0, token_pos, :].float()
            raw_act_dev = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_dev.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            raw_np = raw_act.cpu().numpy().reshape(1, -1)
            sae_np = sae_out[0].float().cpu().numpy().reshape(1, -1)

            raw_pred = int(random_probe.predict(raw_np)[0])
            sae_pred = int(random_probe.predict(sae_np)[0])

            if raw_pred == letter_idx:
                total_correct_raw += 1
                if sae_pred != letter_idx:
                    total_fn += 1
            if sae_pred == letter_idx:
                total_correct_sae += 1
            total_words += 1
            del cache
        except Exception:
            continue

    random_accuracy = total_correct_raw / max(total_words, 1)
    logger.info(f"  Random baseline: acc={random_accuracy:.4f} (expected ~{1/26:.4f}), "
                f"FN={total_fn}/{total_correct_raw}")
    return {
        "random_probe_accuracy": float(random_accuracy),
        "random_sae_accuracy": float(total_correct_sae / max(total_words, 1)),
        "random_fn_rate": float(total_fn / max(total_correct_raw, 1)) if total_correct_raw > 0 else 0,
        "expected_chance": 1.0 / 26,
        "n_words_tested": total_words,
        "n_correct_raw": total_correct_raw,
        "n_fn": total_fn,
    }


# ── Bootstrap CI ─────────────────────────────────────────────
def bootstrap_ci(values, n_bootstrap=2000, ci=0.95):
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    rng = np.random.RandomState(SEED)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[min(int((1 - alpha) * n_bootstrap), len(boot_means) - 1)]
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(values)),
        "n": len(values),
    }


def bootstrap_absorption_rate(word_results, n_bootstrap=2000):
    """Bootstrap on word-level FN rate."""
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


# ── GPU progress ─────────────────────────────────────────────
def update_gpu_progress(elapsed_seconds, status="completed"):
    import filelock
    path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}}
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 50,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "n_sae_configs": len(SAE_CONFIGS),
                    "max_test_words": MAX_TEST_WORDS,
                    "n_prompts_per_word": N_PROMPTS_PER_WORD,
                    "token_pos": TOKEN_POS,
                },
            }
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")
        try:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}}
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


def update_experiment_state(status="completed"):
    import filelock
    path = WORKSPACE / "exp" / "experiment_state.json"
    lock_path = WORKSPACE / "exp" / "experiment_state.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "tasks": {}}
            if TASK_ID in data.get("tasks", {}):
                data["tasks"][TASK_ID]["status"] = status
                data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"experiment_state update failed: {e}")


# ── Main ─────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 16, "starting")

    logger.info(f"=== Phase 1.2a: First-Letter Absorption FULL ===")
    logger.info(f"SAE configs: {len(SAE_CONFIGS)}, test_words={MAX_TEST_WORDS}, "
                f"prompts/word={N_PROMPTS_PER_WORD}, token_pos={TOKEN_POS}")

    device = f"cuda:{GPU_ID}"

    # Step 1: Load model
    report_progress(1, 16, "loading_model")
    model = load_model(device)

    # Step 2: Get word pool and split into probe-train / absorption-test
    report_progress(2, 16, "preparing_words")
    all_words = get_word_list(model.tokenizer, max_words=2000)
    rng_split = random.Random(SEED)
    rng_split.shuffle(all_words)
    split_idx = int(len(all_words) * 0.6)
    probe_train_words = all_words[:split_idx]
    test_words = all_words[split_idx:][:MAX_TEST_WORDS]
    logger.info(f"Words: {len(probe_train_words)} probe-train, {len(test_words)} absorption-test")

    # Step 3: Build ICL prompts for probe training
    report_progress(3, 16, "building_prompts")
    probe_prompts = build_icl_prompts(
        probe_train_words, n_prompts_per_word=PROBE_TRAIN_PROMPTS,
        example_pool=probe_train_words
    )

    # Step 4: Cache activations at all 4 layers simultaneously
    report_progress(4, 16, "caching_activations")
    unique_layers = sorted(set(c["layer"] for c in SAE_CONFIGS))
    layer_acts, labels, words = cache_activations_multi_layer(
        model, probe_prompts, unique_layers, token_pos=TOKEN_POS
    )

    # Step 5: Train probes for each layer
    report_progress(5, 16, "training_probes")
    probes = {}
    probe_qualities = {}
    for layer in unique_layers:
        X = layer_acts[layer]
        logger.info(f"\n--- Training probe for layer {layer} ---")
        probe, metrics = train_probe(X, labels, layer)
        probes[layer] = probe
        probe_qualities[layer] = metrics

    # Free probe training data
    del layer_acts, labels, probe_prompts
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"\nProbe summary:")
    for layer, m in probe_qualities.items():
        logger.info(f"  L{layer}: F1_w={m['f1_weighted']:.4f}, F1_m={m['f1_macro']:.4f}, "
                    f"Acc={m['accuracy']:.4f}, strict={'PASS' if m['quality_gate_strict'] else 'FAIL'}")

    # Iter_008 baselines for comparison
    iter008_baselines = {
        "L6_16k": 0.0237, "L6_65k": 0.0241,
        "L12_16k": 0.0567, "L12_65k": 0.0922,
        "L18_16k": 0.0219, "L18_65k": 0.0452,
        "L24_16k": 0.3448, "L24_65k": 0.2553,
    }

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "token_pos": TOKEN_POS,
        "n_prompts_per_word": N_PROMPTS_PER_WORD,
        "probe_train_prompts_per_word": PROBE_TRAIN_PROMPTS,
        "n_probe_train_words": len(probe_train_words),
        "n_test_words": len(test_words),
        "probes": probe_qualities,
        "absorption_results": {},
        "iter008_baselines": iter008_baselines,
    }

    # Steps 6-13: Measure absorption for each SAE config
    last_sae_for_baseline = None
    last_sae_layer_for_baseline = None

    for sae_idx, sae_config in enumerate(SAE_CONFIGS):
        config_key = f"L{sae_config['layer']}_{sae_config['width']}"
        layer = sae_config["layer"]
        step = 6 + sae_idx
        report_progress(step, 16, f"absorption_{config_key}")
        logger.info(f"\n{'='*60}")
        logger.info(f"=== Absorption: {config_key} (SAE {sae_idx+1}/{len(SAE_CONFIGS)}) ===")

        if layer not in probes:
            logger.error(f"No probe for layer {layer}, skipping {config_key}")
            results["absorption_results"][config_key] = {"error": f"no probe for layer {layer}"}
            continue

        try:
            sae = load_sae(sae_config["release"], sae_config["sae_id"], device=device)
        except Exception as e:
            logger.error(f"SAE load failed for {config_key}: {e}")
            results["absorption_results"][config_key] = {"error": str(e)}
            continue

        try:
            abs_data = measure_absorption_for_sae(
                model, sae, probes[layer], test_words,
                layer=layer, token_pos=TOKEN_POS,
                n_prompts_per_word=N_PROMPTS_PER_WORD,
                example_pool=all_words, device=device,
            )

            boot_ci = bootstrap_absorption_rate(abs_data["word_results"])
            rates = [v["absorption_rate"] for v in abs_data["per_letter"].values()
                     if v["total"] > 0]
            per_letter_ci = bootstrap_ci(rates)

            baseline = iter008_baselines.get(config_key, None)
            delta = abs_data["absorption_rate"] - baseline if baseline is not None else None
            within_5pp = abs(delta) <= 0.05 if delta is not None else None
            within_10pp = abs(delta) <= 0.10 if delta is not None else None

            config_result = {
                "sae_config": sae_config,
                "absorption_rate": abs_data["absorption_rate"],
                "strict_absorption_rate": abs_data["strict_absorption_rate"],
                "total_words": abs_data["total_words"],
                "total_unique_words": abs_data["total_unique_words"],
                "total_probe_correct": abs_data["total_probe_correct_raw"],
                "total_probe_correct_sae": abs_data["total_probe_correct_sae"],
                "total_false_negatives": abs_data["total_false_negatives"],
                "total_fn_main_absent": abs_data["total_fn_main_absent"],
                "probe_raw_accuracy": abs_data["total_probe_correct_raw"] / max(abs_data["total_words"], 1),
                "bootstrap_ci_word": boot_ci,
                "bootstrap_ci_per_letter": per_letter_ci,
                "iter008_baseline": baseline,
                "delta_vs_iter008": float(delta) if delta is not None else None,
                "within_5pp_of_iter008": within_5pp,
                "within_10pp_of_iter008": within_10pp,
                "per_letter": abs_data["per_letter"],
                "fn_examples": abs_data["fn_examples"][:15],
                "main_features_top": {
                    k: {"fid": v["feature_ids"][0], "cos": round(v["cos_sims"][0], 4)}
                    for k, v in abs_data["main_features"].items()
                },
            }
            results["absorption_results"][config_key] = config_result

            probe_acc = abs_data["total_probe_correct_raw"] / max(abs_data["total_words"], 1)
            logger.info(f"  Probe raw accuracy: {probe_acc:.4f}")
            logger.info(f"  Absorption rate: {abs_data['absorption_rate']:.4f}")
            logger.info(f"  Strict (main absent): {abs_data['strict_absorption_rate']:.4f}")
            logger.info(f"  FN: {abs_data['total_false_negatives']}/{abs_data['total_probe_correct_raw']} "
                       f"(of {abs_data['total_words']} total probed)")
            logger.info(f"  Bootstrap CI: [{boot_ci['ci_lower']:.4f}, {boot_ci['ci_upper']:.4f}]")
            if baseline is not None:
                logger.info(f"  vs iter_008: delta={delta:+.4f}, within 5pp: {within_5pp}")

            # Keep L24_16k SAE for random baseline
            if config_key == "L24_16k":
                last_sae_for_baseline = sae
                last_sae_layer_for_baseline = layer
                sae = None

        except Exception as e:
            logger.error(f"Absorption failed for {config_key}: {e}", exc_info=True)
            results["absorption_results"][config_key] = {"error": str(e)}

        if sae is not None:
            del sae
        gc.collect()
        torch.cuda.empty_cache()

    # Step 14: Random baseline
    report_progress(14, 16, "random_baseline")
    logger.info("\n=== Random Baseline Control ===")
    try:
        if last_sae_for_baseline is not None:
            random_bl = random_baseline_absorption(
                model, last_sae_for_baseline, test_words,
                layer=last_sae_layer_for_baseline,
                token_pos=TOKEN_POS, device=device
            )
            results["random_baseline"] = random_bl
            del last_sae_for_baseline
        else:
            results["random_baseline"] = {"error": "no SAE available"}
    except Exception as e:
        logger.warning(f"Random baseline failed: {e}")
        results["random_baseline"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()

    # Step 15: Summary
    report_progress(15, 16, "summarizing")
    summary_rows = []
    for config_key, ar in results["absorption_results"].items():
        if isinstance(ar, dict) and "absorption_rate" in ar:
            summary_rows.append({
                "config": config_key,
                "layer": ar["sae_config"]["layer"],
                "width": ar["sae_config"]["width"],
                "absorption_rate": ar["absorption_rate"],
                "strict_rate": ar["strict_absorption_rate"],
                "n_unique_words": ar["total_unique_words"],
                "n_total_probed": ar["total_words"],
                "n_probe_correct": ar["total_probe_correct"],
                "probe_raw_accuracy": ar.get("probe_raw_accuracy", 0),
                "n_fn": ar["total_false_negatives"],
                "ci_lower": ar["bootstrap_ci_word"]["ci_lower"],
                "ci_upper": ar["bootstrap_ci_word"]["ci_upper"],
                "iter008_baseline": ar.get("iter008_baseline"),
                "delta": ar.get("delta_vs_iter008"),
                "within_5pp": ar.get("within_5pp_of_iter008"),
                "within_10pp": ar.get("within_10pp_of_iter008"),
            })

    configs_with_results = len(summary_rows)
    configs_within_10pp = sum(1 for r in summary_rows if r.get("within_10pp", False))
    configs_within_5pp = sum(1 for r in summary_rows if r.get("within_5pp", False))
    pass_criteria = configs_with_results >= 6

    results["summary"] = {
        "n_configs_tested": configs_with_results,
        "n_configs_within_10pp": configs_within_10pp,
        "n_configs_within_5pp": configs_within_5pp,
        "results_table": summary_rows,
        "pass": pass_criteria,
        "pass_criteria": ">=6 configs computed",
    }

    # Print summary table
    logger.info(f"\n{'='*110}")
    logger.info("FIRST-LETTER ABSORPTION RATE SUMMARY (FULL)")
    logger.info(f"{'Config':12s} {'Rate':>8s} {'Strict':>8s} {'PrbAcc':>7s} {'Words':>6s} "
                f"{'FN/Corr':>10s} {'CI':>16s} {'iter008':>8s} {'Delta':>8s} {'<5pp':>5s}")
    logger.info("-" * 110)
    for r in summary_rows:
        logger.info(
            f"{r['config']:12s} {r['absorption_rate']:8.4f} {r['strict_rate']:8.4f} "
            f"{r['probe_raw_accuracy']:7.4f} "
            f"{r['n_unique_words']:>6d} "
            f"{r['n_fn']:>4d}/{r['n_probe_correct']:<4d} "
            f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}] "
            f"{r.get('iter008_baseline', 0):8.4f} "
            f"{r.get('delta', 0):+8.4f} "
            f"{'Y' if r.get('within_5pp') else 'N':>5s}"
        )
    logger.info(f"{'='*110}")
    logger.info(f"Pass: {pass_criteria} ({configs_with_results}/8 configs, "
                f"{configs_within_10pp} within 10pp, {configs_within_5pp} within 5pp)")

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["elapsed_minutes"] = elapsed / 60

    # Save
    out_path = PHASE1_DIR / "absorption_firstletter.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Saved: {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 1.2a first-letter absorption FULL. "
        f"Configs: {configs_with_results}/{len(SAE_CONFIGS)}. "
        f"Within 10pp: {configs_within_10pp}. Within 5pp: {configs_within_5pp}. "
        f"Time: {elapsed/60:.1f}min. "
        f"Pass: {'YES' if pass_criteria else 'NO'}."
    )
    if summary_rows:
        l24_row = next((r for r in summary_rows if r["config"] == "L24_16k"), None)
        if l24_row:
            summary_text += (f" L24_16k: {l24_row['absorption_rate']:.4f} "
                           f"[{l24_row['ci_lower']:.3f},{l24_row['ci_upper']:.3f}]")

    # Report probe quality for key layers
    for layer in [6, 24]:
        if layer in probe_qualities:
            pq = probe_qualities[layer]
            summary_text += f" Probe_L{layer}_F1={pq['f1_weighted']:.3f}"

    mark_done("success" if pass_criteria else "partial", summary_text)
    update_gpu_progress(elapsed, "completed" if pass_criteria else "failed")
    update_experiment_state("completed" if pass_criteria else "failed")

    report_progress(16, 16, "completed", {
        "configs_tested": configs_with_results,
        "pass": pass_criteria,
        "elapsed_min": round(elapsed / 60, 1),
    })

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
