"""
Phase 1.2a: First-Letter Absorption Measurement (8 SAE configs)
Iteration 9, PILOT mode.

Measures first-letter absorption rates at L6/L12/L18/L24 x 16k/65k JumpReLU SAEs.
Uses pre-trained sklearn probe from phase1_probe_training (F1=0.928 at L24).

Pipeline per SAE config:
  1. Load model + SAE at layer L
  2. For each test word: build ICL prompt, get residual stream at probe position
  3. Apply sklearn probe to raw activation -> correct letter? (clean prediction)
  4. Encode through SAE, decode -> apply probe -> correct letter? (SAE prediction)
  5. False negative = correct on raw, wrong on SAE (= absorption)
  6. Classify FNs: check if main letter feature fires (absorbed vs hedged)
  7. Bootstrap 95% CI on absorption rate
  8. Random direction baseline: replace probe direction with random -> near-zero FN

Pilot: 100 words, 1 prompt/word, all 8 configs
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
for d in [PILOT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# All 8 SAE configs (L6/L12/L18/L24 x 16k/65k)
SAE_CONFIGS = [
    {"layer": l, "width": w, "release": "gemma-scope-2b-pt-res-canonical",
     "sae_id": f"layer_{l}/width_{w}/canonical"}
    for l in [6, 12, 18, 24] for w in ["16k", "65k"]
]

if MODE == "PILOT":
    MAX_WORDS = 100
    N_PROMPTS_PER_WORD = 1
    TIMEOUT = 900
else:
    MAX_WORDS = 500
    N_PROMPTS_PER_WORD = 5
    TIMEOUT = 2700

LETTERS = "abcdefghijklmnopqrstuvwxyz"
GPU_ID = 0  # CUDA_VISIBLE_DEVICES remaps

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


# ── Utility: SAE hook name ───────────────────────────────────
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
def get_word_list(tokenizer, max_words=200):
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


# ── ICL prompts ──────────────────────────────────────────────
def build_icl_prompts(word_list, n_prompts_per_word=1):
    """Build ICL prompts using sae-spelling template."""
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


# ── Probe training (self-contained, position-consistent) ─────
def train_probe_at_position(model, word_list, layer, token_pos=-6, device="cuda:0"):
    """
    Train sklearn probe at the EXACT position used for absorption measurement.
    This ensures position consistency between probe and downstream usage.
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    hook_name = f"blocks.{layer}.hook_resid_post"

    logger.info(f"Training probe at L{layer} pos={token_pos} on {len(word_list)} words...")
    acts = []
    labels = []

    for i, word in enumerate(word_list):
        letter = word[0].lower()
        if letter not in LETTERS:
            continue
        letter_idx = LETTERS.index(letter)
        try:
            sp = create_icl_prompt(
                word=word, examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter,
                max_icl_examples=10, shuffle_examples=True,
            )
            tokens = model.to_tokens(sp.base, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name][0, token_pos, :].float().cpu().numpy()
            acts.append(act)
            labels.append(letter_idx)
            del cache
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Cached {i+1}/{len(word_list)} training activations")
            torch.cuda.empty_cache()

    X = np.array(acts)
    y = np.array(labels)
    n_classes_seen = len(set(y))
    logger.info(f"  Training data: {X.shape[0]} samples, {n_classes_seen}/26 letters")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    best_f1, best_probe = -1, None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=SEED)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"    C={C}: F1={f1:.4f}, Acc={acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_probe = clf

    logger.info(f"  Best probe L{layer}: F1={best_f1:.4f}, n_classes={len(best_probe.classes_)}")
    return best_probe, float(best_f1)


# ── Absorption measurement ───────────────────────────────────
def measure_absorption_for_sae(model, sae, sklearn_probe, word_list,
                                layer, token_pos=-6, device="cuda:0"):
    """
    Measure first-letter absorption for one SAE config.

    For each word:
      1. Get raw residual stream at probe position
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
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute probe directions and top features per letter
    probe_coefs = torch.tensor(sklearn_probe.coef_, dtype=torch.float32)  # [n_classes, d_model]
    probe_classes = sklearn_probe.classes_  # actual class indices the probe knows
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]

    # Map letter_idx -> probe row index (handle probes with < 26 classes)
    letter_to_probe_row = {}
    for row_idx, cls in enumerate(probe_classes):
        if 0 <= cls < 26:
            letter_to_probe_row[cls] = row_idx

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

    # Stats per letter
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
    word_results = []  # Per-word detail for bootstrap

    processed = 0
    skipped_no_class = 0
    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS:
            continue
        letter_idx = LETTERS.index(letter)

        # Skip letters not known to the probe
        if letter_idx not in letter_to_probe_row:
            skipped_no_class += 1
            continue

        try:
            sp = create_icl_prompt(
                word=word, examples=word_list,
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

            # SAE encode/decode
            raw_act_dev = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_dev.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            # Probe predictions
            raw_np = raw_act.cpu().numpy().reshape(1, -1)
            sae_np = sae_out[0].float().cpu().numpy().reshape(1, -1)

            raw_pred = sklearn_probe.predict(raw_np)[0]
            sae_pred = sklearn_probe.predict(sae_np)[0]

            probe_correct_raw = (raw_pred == letter_idx)
            probe_correct_sae = (sae_pred == letter_idx)

            # Check main features (only for letters with computed features)
            if letter in main_features:
                mfids = main_features[letter]["feature_ids"]
                feat_acts = sae_features[0, mfids].detach().float().cpu()
                any_main_fires = (feat_acts.abs() > 1e-6).any().item()
            else:
                any_main_fires = False
                feat_acts = torch.zeros(5)

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

        # Record per-word result for bootstrap
        is_fn = probe_correct_raw and not probe_correct_sae
        word_results.append({
            "word": word,
            "letter": letter,
            "probe_correct_raw": bool(probe_correct_raw),
            "probe_correct_sae": bool(probe_correct_sae),
            "is_fn": is_fn,
            "main_fires": any_main_fires,
        })

        if is_fn:
            stats["false_negatives"] += 1
            if not any_main_fires:
                stats["fn_and_main_absent"] += 1
            else:
                stats["fn_and_main_present"] += 1

            if len(fn_examples) < 25:
                fn_examples.append({
                    "word": word, "letter": letter,
                    "raw_pred": LETTERS[raw_pred],
                    "sae_pred": LETTERS[sae_pred],
                    "main_fires": any_main_fires,
                    "top_feat_act": float(feat_acts[0].item()),
                })

        processed += 1
        if processed % 25 == 0:
            logger.info(f"  Processed {processed}/{len(word_list)} words")
            torch.cuda.empty_cache()

    if skipped_no_class > 0:
        logger.info(f"  Skipped {skipped_no_class} words (letter not in probe classes)")

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


# ── Random baseline control ─────────────────────────────────
def random_baseline_absorption(model, word_list, layer, token_pos=-6,
                                device="cuda:0", n_trials=3):
    """
    Control: use a random direction instead of the probe direction.
    Should show near-zero false negatives (not systematic).
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Create random probe
    rng = np.random.RandomState(SEED + 999)
    d_model = 2304  # Gemma 2 2B
    random_coef = rng.randn(26, d_model).astype(np.float32)
    random_coef /= np.linalg.norm(random_coef, axis=1, keepdims=True)

    random_probe = LogisticRegression(max_iter=1)
    random_probe.coef_ = random_coef
    random_probe.intercept_ = np.zeros(26, dtype=np.float32)
    random_probe.classes_ = np.arange(26)

    total_correct_raw = 0
    total_fn = 0
    total_words = 0

    # Only test a subset for speed
    test_words = word_list[:min(50, len(word_list))]
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
            raw_act = cache[tl_hook][0, token_pos, :].float().cpu().numpy().reshape(1, -1)
            del cache

            raw_pred = random_probe.predict(raw_act)[0]
            if raw_pred == letter_idx:
                total_correct_raw += 1
            total_words += 1
        except Exception:
            continue

    # Random probe should have ~1/26 accuracy, so FN rate is meaningless
    random_accuracy = total_correct_raw / max(total_words, 1)
    logger.info(f"  Random baseline: acc={random_accuracy:.4f} (expected ~{1/26:.4f})")
    return {
        "random_probe_accuracy": float(random_accuracy),
        "expected_chance": 1.0 / 26,
        "n_words_tested": total_words,
        "n_correct": total_correct_raw,
    }


# ── Bootstrap CI ─────────────────────────────────────────────
def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
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


def bootstrap_absorption_rate(word_results, n_bootstrap=1000):
    """Bootstrap on word-level absorption indicators."""
    # Filter to words where raw probe was correct
    eligible = [w for w in word_results if w["probe_correct_raw"]]
    if not eligible:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0, "n": 0}

    fn_flags = np.array([1 if w["is_fn"] else 0 for w in eligible])
    rng = np.random.RandomState(SEED)
    boot_rates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(fn_flags), size=len(fn_flags), replace=True)
        boot_rates.append(fn_flags[idx].mean())
    boot_rates = sorted(boot_rates)
    lo = boot_rates[int(0.025 * n_bootstrap)]
    hi = boot_rates[min(int(0.975 * n_bootstrap), len(boot_rates) - 1)]
    return {
        "mean": float(fn_flags.mean()),
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
                "planned_min": 15 if MODE == "PILOT" else 50,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "n_sae_configs": len(SAE_CONFIGS),
                    "max_words": MAX_WORDS,
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
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


# ── Main ─────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 12, "starting")

    logger.info(f"=== Phase 1.2a: First-Letter Absorption ({MODE}) ===")
    logger.info(f"SAE configs: {len(SAE_CONFIGS)}, max_words={MAX_WORDS}")

    device = f"cuda:{GPU_ID}"

    # Step 1: Load model
    report_progress(1, 12, "loading_model")
    model = load_model(device)

    # Step 2: Prepare word lists
    report_progress(2, 12, "preparing_words")
    from sae_spelling.prompting import VERBOSE_FIRST_LETTER_TOKEN_POS
    token_pos = VERBOSE_FIRST_LETTER_TOKEN_POS  # -6

    # Get word pool and split: 60% probe training, 40% absorption test
    all_words = get_word_list(model.tokenizer, max_words=MAX_WORDS * 3)
    rng_split = random.Random(SEED)
    rng_split.shuffle(all_words)
    split_idx = int(len(all_words) * 0.6)
    probe_train_words = all_words[:split_idx]
    test_words = all_words[split_idx:][:MAX_WORDS]
    logger.info(f"Words: {len(probe_train_words)} probe-train, {len(test_words)} absorption-test")

    # Step 3: Train probes at position token_pos (consistent with absorption measurement)
    # NOTE: We train fresh probes here instead of using pre-trained ones from phase1,
    # because phase1 probes were trained at position -2 (sae_spelling convention),
    # but absorption measurement uses VERBOSE_FIRST_LETTER_TOKEN_POS (-6).
    report_progress(3, 12, "training_probes")

    unique_layers = sorted(set(c["layer"] for c in SAE_CONFIGS))
    probes = {}
    probe_qualities = {}

    for layer in unique_layers:
        probe, f1 = train_probe_at_position(
            model, probe_train_words, layer, token_pos, device
        )
        if probe is not None:
            probes[layer] = probe
            probe_qualities[layer] = {
                "source": "trained_at_pos_neg6",
                "f1": float(f1),
                "n_classes": len(probe.classes_),
                "position": token_pos,
            }
        else:
            logger.error(f"Cannot train probe for layer {layer}")
            probe_qualities[layer] = {"source": "failed"}

    logger.info(f"Probes ready: {list(probes.keys())}")

    # Step 4: Measure absorption for each SAE config
    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "token_pos": token_pos,
        "n_probe_train_words": len(probe_train_words),
        "n_test_words": len(test_words),
        "probes": probe_qualities,
        "absorption_results": {},
        "iter008_baselines": {
            "L6_16k": 0.0237, "L6_65k": 0.0241,
            "L12_16k": 0.0567, "L12_65k": 0.0922,
            "L18_16k": 0.0219, "L18_65k": 0.0452,
            "L24_16k": 0.3448, "L24_65k": 0.2553,
        },
    }

    for sae_idx, sae_config in enumerate(SAE_CONFIGS):
        config_key = f"L{sae_config['layer']}_{sae_config['width']}"
        layer = sae_config["layer"]
        step = 4 + sae_idx
        report_progress(step, 12, f"absorption_{config_key}")
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
                layer=layer, token_pos=token_pos, device=device,
            )

            # Bootstrap CI on word-level results
            boot_ci = bootstrap_absorption_rate(abs_data["word_results"])

            # Per-letter rate bootstrap
            rates = [v["absorption_rate"] for v in abs_data["per_letter"].values()
                     if v["total"] > 0]
            per_letter_ci = bootstrap_ci(rates)

            # Comparison with iter_008
            baseline = results["iter008_baselines"].get(config_key, None)
            delta = abs_data["absorption_rate"] - baseline if baseline else None
            within_10pp = abs(delta) <= 0.10 if delta is not None else None

            config_result = {
                "sae_config": sae_config,
                "absorption_rate": abs_data["absorption_rate"],
                "strict_absorption_rate": abs_data["strict_absorption_rate"],
                "total_words": abs_data["total_words"],
                "total_probe_correct": abs_data["total_probe_correct_raw"],
                "total_probe_correct_sae": abs_data["total_probe_correct_sae"],
                "total_false_negatives": abs_data["total_false_negatives"],
                "total_fn_main_absent": abs_data["total_fn_main_absent"],
                "bootstrap_ci_word": boot_ci,
                "bootstrap_ci_per_letter": per_letter_ci,
                "iter008_baseline": baseline,
                "delta_vs_iter008": float(delta) if delta is not None else None,
                "within_10pp_of_iter008": within_10pp,
                "per_letter": abs_data["per_letter"],
                "fn_examples": abs_data["fn_examples"][:10],
                "main_features_top": {
                    k: {"fid": v["feature_ids"][0], "cos": round(v["cos_sims"][0], 4)}
                    for k, v in abs_data["main_features"].items()
                },
            }
            results["absorption_results"][config_key] = config_result

            logger.info(f"  Absorption rate: {abs_data['absorption_rate']:.4f}")
            logger.info(f"  Strict (main absent): {abs_data['strict_absorption_rate']:.4f}")
            logger.info(f"  FN: {abs_data['total_false_negatives']}/{abs_data['total_probe_correct_raw']}")
            logger.info(f"  Bootstrap CI: [{boot_ci['ci_lower']:.4f}, {boot_ci['ci_upper']:.4f}]")
            if baseline is not None:
                logger.info(f"  vs iter_008: delta={delta:+.4f}, within 10pp: {within_10pp}")

        except Exception as e:
            logger.error(f"Absorption failed for {config_key}: {e}", exc_info=True)
            results["absorption_results"][config_key] = {"error": str(e)}

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # Step 5: Random baseline control
    report_progress(11, 12, "random_baseline")
    logger.info("\n=== Random Baseline Control ===")
    try:
        random_bl = random_baseline_absorption(model, test_words, layer=24,
                                                token_pos=token_pos, device=device)
        results["random_baseline"] = random_bl
        logger.info(f"Random baseline accuracy: {random_bl['random_probe_accuracy']:.4f}")
    except Exception as e:
        logger.warning(f"Random baseline failed: {e}")
        results["random_baseline"] = {"error": str(e)}

    # Step 6: Summary
    report_progress(12, 12, "summarizing")
    summary_rows = []
    for config_key, ar in results["absorption_results"].items():
        if isinstance(ar, dict) and "absorption_rate" in ar:
            summary_rows.append({
                "config": config_key,
                "layer": ar["sae_config"]["layer"],
                "width": ar["sae_config"]["width"],
                "absorption_rate": ar["absorption_rate"],
                "strict_rate": ar["strict_absorption_rate"],
                "n_words": ar["total_words"],
                "n_probe_correct": ar["total_probe_correct"],
                "n_fn": ar["total_false_negatives"],
                "ci_lower": ar["bootstrap_ci_word"]["ci_lower"],
                "ci_upper": ar["bootstrap_ci_word"]["ci_upper"],
                "iter008_baseline": ar.get("iter008_baseline"),
                "delta": ar.get("delta_vs_iter008"),
                "within_10pp": ar.get("within_10pp_of_iter008"),
            })

    # Pilot pass criteria: rates computed for all configs, within 10pp of iter_008
    configs_with_results = len(summary_rows)
    configs_within_10pp = sum(1 for r in summary_rows if r.get("within_10pp", False))
    pilot_pass = configs_with_results >= 6 and configs_within_10pp >= 4

    results["summary"] = {
        "n_configs_tested": configs_with_results,
        "n_configs_within_10pp": configs_within_10pp,
        "results_table": summary_rows,
        "pilot_pass": pilot_pass,
        "pilot_criteria": ">=6 configs computed, >=4 within 10pp of iter_008",
    }

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("ABSORPTION RATE SUMMARY")
    logger.info(f"{'Config':12s} {'Rate':>8s} {'Strict':>8s} {'FN/Corr':>10s} "
                f"{'CI':>16s} {'iter008':>8s} {'Delta':>8s}")
    logger.info("-" * 80)
    for r in summary_rows:
        logger.info(
            f"{r['config']:12s} {r['absorption_rate']:8.4f} {r['strict_rate']:8.4f} "
            f"{r['n_fn']:>4d}/{r['n_probe_correct']:<4d} "
            f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}] "
            f"{r.get('iter008_baseline', 0):8.4f} "
            f"{r.get('delta', 0):+8.4f}"
        )
    logger.info(f"{'='*80}")
    logger.info(f"Pilot pass: {pilot_pass} ({configs_with_results} configs, "
                f"{configs_within_10pp} within 10pp)")

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

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 1.2a first-letter absorption ({MODE}). "
        f"Configs: {configs_with_results}/{len(SAE_CONFIGS)}. "
        f"Within 10pp: {configs_within_10pp}. "
        f"Time: {elapsed/60:.1f}min. "
        f"Pilot: {'PASS' if pilot_pass else 'FAIL'}."
    )
    if summary_rows:
        l24_row = next((r for r in summary_rows if r["config"] == "L24_16k"), None)
        if l24_row:
            summary_text += f" L24_16k: {l24_row['absorption_rate']:.4f}"

    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed" if pilot_pass else "failed")
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress(0, "failed")
        sys.exit(1)
