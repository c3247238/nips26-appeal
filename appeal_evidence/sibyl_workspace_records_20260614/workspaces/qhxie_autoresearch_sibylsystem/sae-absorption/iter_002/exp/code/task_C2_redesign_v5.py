"""
Task C.2-REDESIGN v5: Cross-Domain Absorption via Child-Feature Suppression (PILOT MODE)

CORRECT MEASUREMENT:
The previous attempts (v1-v4) all failed because:
- v1-v3: wrong parent latent identification
- v4: ratio = 1.0 because ICL suppresses ALL words equally

KEY INSIGHT: To get a valid ratio-to-null, we need:
  ratio = (absorption in concept context) / (absorption in CONTROL context)

For first_letter sanity check:
- "apple" in LETTER-A ICL context: child feature of "apple" is suppressed
- "apple" in LETTER-B ICL context (control): child feature is NOT suppressed (or less)
- Ratio >> 1.0 if suppression is specific to the matching letter

The null baseline: same concept words, but in a WRONG-LETTER ICL context.
If absorption is CATEGORY-SPECIFIC, concept words get more suppressed in their-letter
context than in a different-letter context.

For semantic hierarchies:
- Animate words in ANIMATE ICL context vs animate words in INANIMATE ICL context
- If absorption is animate-specific, animate words get more suppressed in animate context

This is the correct cross-domain absorption measurement:
  absorption_rate = P(child_suppressed | matching_category_ICL) / P(child_suppressed | non_matching_ICL)
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
PROBES_DIR = RESULTS_DIR / "probes"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_C2_redesign"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = FULL_DIR / "C2_child_suppression_absorption.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
sys.stdout.flush()


def report_progress(step, total_steps, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID, "step": step, "total_steps": total_steps,
        "elapsed_sec": elapsed, "note": note, "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {step}/{total_steps}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary=""):
    PID_FILE.unlink(missing_ok=True)
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(), "elapsed_sec": time.time() - start_time,
    }))


TOTAL_STEPS = 12
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v5: Category-specific suppression")

# ── Load model and SAE ────────────────────────────────────────────────────────

report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small + SAE L6 jb")

from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_spelling.prompting import create_icl_prompt, first_letter_formatter

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

sae, _, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=DEVICE
)
sae.eval()

W_enc = sae.W_enc.detach().float().cpu()  # [d_model, d_sae]
W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]
d_sae = W_dec.shape[0]
d_model = W_dec.shape[1]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

LAYER = 6
CACHE_KEY = f"blocks.{LAYER}.hook_resid_pre"
print(f"SAE: d_sae={d_sae}, d_model={d_model}")
sys.stdout.flush()

# Load probe weights
probe_fl = np.load(str(PROBES_DIR / "probe_first_letter_weights.npy"))  # [24, 768]
probe_fl_classes = np.load(str(PROBES_DIR / "probe_first_letter_classes.npy"))
print(f"Probe shapes: first_letter={probe_fl.shape}, classes={probe_fl_classes}")

# ── Utility functions ─────────────────────────────────────────────────────────

@torch.no_grad()
def get_sae_acts_pos(texts, position=-2, batch_size=32):
    """Get SAE activations at specific token position."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        if tokens.shape[1] <= abs(position):
            continue
        _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
        resid = cache[CACHE_KEY][:, position, :].float()
        acts = sae.encode(resid)
        all_acts.append(acts.cpu())
    if not all_acts:
        return None
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def get_sae_acts_last(texts, batch_size=64):
    """Get SAE activations at last token position."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
        resid = cache[CACHE_KEY][:, -1, :].float()
        acts = sae.encode(resid)
        all_acts.append(acts.cpu())
    return torch.cat(all_acts, dim=0)


def find_word_specific_features(word, reference_words, top_k=5, min_act=0.3):
    """
    Find SAE features specific to this word vs. reference words.
    Returns top-k features by differential activation (word - mean_reference).
    """
    word_acts = get_sae_acts_last([word])
    if word_acts is None:
        return []
    word_vec = word_acts[0].numpy()

    ref_acts = get_sae_acts_last(reference_words[:20])
    ref_mean = ref_acts.mean(dim=0).numpy()

    diff = word_vec - ref_mean
    # Features active for word but not for references
    candidate_mask = word_vec >= min_act
    diff_masked = np.where(candidate_mask, diff, -999)
    top_k_idx = np.argsort(diff_masked)[::-1][:top_k]

    features = []
    for j in top_k_idx:
        if word_vec[j] >= min_act:
            features.append({
                "latent_idx": int(j),
                "word_act": float(word_vec[j]),
                "ref_mean_act": float(ref_mean[j]),
                "differential": float(diff[j]),
            })

    return features


def measure_suppression_in_context(word, word_feats, context_text, icl_position=-2):
    """
    Measure suppression of word-specific features in a given context.
    Returns dict with:
    - n_child_features: number of word-specific features
    - n_suppressed: how many get suppressed >= 30%
    - absorption_rate: n_suppressed / n_child
    - mean_suppression: mean suppression ratio
    """
    if not word_feats:
        return None

    icl_acts = get_sae_acts_pos([context_text], position=icl_position)
    if icl_acts is None:
        return None
    icl_vec = icl_acts[0].numpy()

    suppressed_count = 0
    suppressions = []
    for feat in word_feats:
        j = feat["latent_idx"]
        iso_act = feat["word_act"]
        icl_act = float(icl_vec[j])
        suppression = (iso_act - icl_act) / (iso_act + 1e-8)
        suppressions.append(float(suppression))
        if suppression >= 0.30:
            suppressed_count += 1

    n = len(word_feats)
    return {
        "n_child_features": n,
        "n_suppressed": suppressed_count,
        "absorption_rate": suppressed_count / n if n > 0 else 0.0,
        "mean_suppression": float(np.mean(suppressions)) if suppressions else 0.0,
        "is_absorbed": suppressed_count > 0,
    }


# ── Step 2: FIRST_LETTER — same-letter ICL vs different-letter ICL ────────────

report_progress(2, TOTAL_STEPS, "First_letter: same-letter vs different-letter ICL context")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter")
print("Design: concept_words in same-letter ICL vs concept_words in diff-letter ICL")
print("="*70)
sys.stdout.flush()

from sae_spelling.vocab import get_alpha_tokens
vocab_alpha = get_alpha_tokens(tokenizer)

# Letters to test (from C1 passing probe)
test_letters = ["a", "b", "s", "m"]  # 4 letters
rng = random.Random(SEED)

# Test words: words starting with each letter (single-token preferred)
test_words_by_letter = {
    "a": ["apple", "angel", "arrow", "album", "actor", "adult", "agent", "ample"],
    "b": ["bird", "blood", "brain", "bread", "bridge", "brush", "beach", "brave"],
    "s": ["stone", "storm", "smile", "snake", "sheep", "shell", "spark", "sport"],
    "m": ["music", "metal", "magic", "model", "maple", "month", "moral", "marsh"],
}

# Reference words (unrelated, no specific letter pattern) for child feature identification
reference_words = [
    "the", "and", "for", "are", "but", "not", "you", "all",
    "table", "chair", "book", "car", "house", "door", "floor", "wall",
]

# Build ICL vocab from alpha tokens (for each letter)
# Collect single-token words starting with each letter
icl_vocab_by_letter = {}
for letter in test_letters:
    icl_words_for_letter = []
    for w in vocab_alpha:
        w_clean = w.strip().lower()
        if w_clean.startswith(letter) and w_clean.isalpha() and 3 <= len(w_clean) <= 7:
            icl_words_for_letter.append(w_clean)
    rng.shuffle(icl_words_for_letter)
    icl_vocab_by_letter[letter] = icl_words_for_letter[:60]
    print(f"  Letter '{letter}': {len(icl_words_for_letter)} ICL words available, using first 60")

# Also collect a pool for "wrong letter" ICL contexts
icl_vocab_wrong = {}
for letter in test_letters:
    wrong_pool = []
    for other_letter in test_letters:
        if other_letter != letter:
            wrong_pool.extend(icl_vocab_by_letter.get(other_letter, [])[:20])
    rng.shuffle(wrong_pool)
    icl_vocab_wrong[letter] = wrong_pool[:60]

sys.stdout.flush()

fl_results_by_letter = {}
fl_all_same_rates = []
fl_all_diff_rates = []

for letter in test_letters:
    words = test_words_by_letter[letter]
    same_letter_icl_vocab = [w for w in icl_vocab_by_letter.get(letter, [])
                              if w not in words]
    diff_letter_icl_vocab = icl_vocab_wrong.get(letter, [])

    letter_same = []
    letter_diff = []

    for word in words:
        # Find word-specific child features (vs. reference words)
        word_feats = find_word_specific_features(word, reference_words, top_k=8, min_act=0.3)
        if not word_feats:
            print(f"  '{word}' ('{letter}'): no specific features found")
            continue

        # --- SAME-LETTER ICL context ---
        icl_examples_same = [w for w in same_letter_icl_vocab if w != word]
        rng.shuffle(icl_examples_same)
        try:
            prompt_same = create_icl_prompt(
                word,
                examples=icl_examples_same,
                base_template="{word}:",
                answer_formatter=first_letter_formatter(),
                max_icl_examples=10,
                shuffle_examples=False,
            )
            same_text = prompt_same.base
        except Exception as e:
            print(f"  '{word}': same-letter prompt error: {e}")
            continue

        # --- DIFFERENT-LETTER ICL context ---
        icl_examples_diff = [w for w in diff_letter_icl_vocab if w != word]
        rng.shuffle(icl_examples_diff)
        try:
            prompt_diff = create_icl_prompt(
                word,
                examples=icl_examples_diff,
                base_template="{word}:",
                answer_formatter=first_letter_formatter(),
                max_icl_examples=10,
                shuffle_examples=False,
            )
            diff_text = prompt_diff.base
        except Exception as e:
            print(f"  '{word}': diff-letter prompt error: {e}")
            continue

        # Measure suppression in SAME-letter context
        res_same = measure_suppression_in_context(word, word_feats, same_text, icl_position=-2)
        # Measure suppression in DIFFERENT-letter context
        res_diff = measure_suppression_in_context(word, word_feats, diff_text, icl_position=-2)

        if res_same is None or res_diff is None:
            print(f"  '{word}': context measurement failed")
            continue

        print(f"  '{word}' ('{letter}'): "
              f"n_child={res_same['n_child_features']}, "
              f"same_rate={res_same['absorption_rate']:.3f}, "
              f"diff_rate={res_diff['absorption_rate']:.3f}, "
              f"same_supp={res_same['mean_suppression']:.3f}, "
              f"diff_supp={res_diff['mean_suppression']:.3f}")

        letter_same.append(res_same["absorption_rate"])
        letter_diff.append(res_diff["absorption_rate"])
        fl_all_same_rates.append(res_same["absorption_rate"])
        fl_all_diff_rates.append(res_diff["absorption_rate"])

    if letter_same and letter_diff:
        mean_same = float(np.mean(letter_same))
        mean_diff = float(np.mean(letter_diff))
        ratio = mean_same / mean_diff if mean_diff > 0 else (10.0 if mean_same > 0 else 1.0)
        fl_results_by_letter[letter] = {
            "mean_same_letter_absorption": mean_same,
            "mean_diff_letter_absorption": mean_diff,
            "ratio": ratio,
            "n_words": len(letter_same),
        }
        print(f"  Letter '{letter}': same={mean_same:.3f}, diff={mean_diff:.3f}, ratio={ratio:.3f}")

sys.stdout.flush()

# Aggregate
n_fl = min(len(fl_all_same_rates), len(fl_all_diff_rates))
fl_same_mean = float(np.mean(fl_all_same_rates)) if fl_all_same_rates else 0.0
fl_diff_mean = float(np.mean(fl_all_diff_rates)) if fl_all_diff_rates else 0.0
fl_ratio = fl_same_mean / fl_diff_mean if fl_diff_mean > 0 else (10.0 if fl_same_mean > 0 else 1.0)
fl_go_nogo = "GO" if fl_ratio >= 1.5 else ("WEAK" if fl_ratio >= 1.1 else "NO_GO")

# Statistical test: paired comparison (same vs diff per word)
from scipy import stats
if len(fl_all_same_rates) >= 5 and len(fl_all_diff_rates) >= 5:
    n_paired = min(len(fl_all_same_rates), len(fl_all_diff_rates))
    t_stat, p_val = stats.ttest_rel(
        fl_all_same_rates[:n_paired],
        fl_all_diff_rates[:n_paired]
    )
    print(f"\n  first_letter paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
else:
    t_stat, p_val = None, None

# Bootstrap CI on ratio
rng_boot = np.random.RandomState(SEED)
boot_ratios_fl = []
n_boot_words = len(fl_all_same_rates)
for _ in range(1000):
    if n_boot_words > 0:
        idx = rng_boot.choice(n_boot_words, n_boot_words, replace=True)
        s = float(np.mean([fl_all_same_rates[i] for i in idx]))
        d = float(np.mean([fl_all_diff_rates[i] for i in idx]))
        boot_ratios_fl.append(s / d if d > 0 else (10.0 if s > 0 else 1.0))
fl_ci_lower = float(np.percentile(boot_ratios_fl, 2.5)) if boot_ratios_fl else 0.0
fl_ci_upper = float(np.percentile(boot_ratios_fl, 97.5)) if boot_ratios_fl else 0.0

print(f"\n  [first_letter] AGGREGATE:")
print(f"    n_words={n_boot_words}, same_mean={fl_same_mean:.4f}, diff_mean={fl_diff_mean:.4f}")
print(f"    ratio_to_null={fl_ratio:.3f}, CI=[{fl_ci_lower:.3f}, {fl_ci_upper:.3f}]")
print(f"    go_nogo={fl_go_nogo}")
sys.stdout.flush()

result_first_letter = {
    "hierarchy": "first_letter",
    "measurement": "Concept words in same-letter ICL vs different-letter ICL (category-specific suppression)",
    "n_words": n_boot_words,
    "same_letter_absorption_rate": fl_same_mean,
    "diff_letter_absorption_rate": fl_diff_mean,
    "ratio_to_null": fl_ratio,
    "ci_lower": fl_ci_lower,
    "ci_upper": fl_ci_upper,
    "paired_t_stat": float(t_stat) if t_stat is not None else None,
    "paired_p_val": float(p_val) if p_val is not None else None,
    "results_by_letter": fl_results_by_letter,
    "go_nogo": fl_go_nogo,
}

# ── Step 3: ANIMATE_INANIMATE ─────────────────────────────────────────────────

report_progress(3, TOTAL_STEPS, "animate_inanimate: animate-ICL vs inanimate-ICL")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate")
print("Design: animate words in animate-ICL vs animate words in inanimate-ICL")
print("="*70)
sys.stdout.flush()

animate_words_test = [
    "dog", "cat", "horse", "wolf", "bear", "eagle", "lion",
    "tiger", "rabbit", "monkey", "elephant", "shark", "whale",
]
inanimate_words_pool = [
    "rock", "stone", "table", "chair", "book", "car", "house",
    "bridge", "bottle", "box", "ring", "cloud", "river", "lamp",
]

# ICL contexts for animate/inanimate classification
animate_icl_examples = [
    "dog: animate", "cat: animate", "wolf: animate", "eagle: animate",
    "rock: inanimate", "table: inanimate", "chair: inanimate", "book: inanimate",
]
inanimate_icl_examples = [
    "rock: inanimate", "stone: inanimate", "table: inanimate", "chair: inanimate",
    "dog: animate", "cat: animate", "wolf: animate", "eagle: animate",
]

# Control ICL: noun_proper format (completely different task)
proper_icl_examples = [
    "London: proper", "Alice: proper", "Paris: proper",
    "table: common", "chair: common", "river: common",
]


def make_category_prompt(word, examples, k=6):
    """Create a few-shot classification ICL prompt."""
    rng.shuffle(examples)
    selected = examples[:k]
    return "\n".join(selected) + "\n" + word + ":"


rng_ai = random.Random(SEED + 1)
ai_results_same = []
ai_results_diff = []

for word in animate_words_test:
    word_feats = find_word_specific_features(word, inanimate_words_pool, top_k=8, min_act=0.3)
    if not word_feats:
        print(f"  '{word}': no specific features found")
        continue

    # Animate ICL context (category-matching)
    examples_shuffled = animate_icl_examples.copy()
    rng_ai.shuffle(examples_shuffled)
    same_text = make_category_prompt(word, examples_shuffled, k=6)

    # Inanimate ICL context (wrong category control)
    examples_wrong = inanimate_icl_examples.copy()
    rng_ai.shuffle(examples_wrong)
    diff_text = make_category_prompt(word, examples_wrong, k=6)

    res_same = measure_suppression_in_context(word, word_feats, same_text, icl_position=-2)
    res_diff = measure_suppression_in_context(word, word_feats, diff_text, icl_position=-2)

    if res_same is None or res_diff is None:
        print(f"  '{word}': context measurement failed")
        continue

    print(f"  '{word}' (animate): "
          f"n_child={res_same['n_child_features']}, "
          f"same_rate={res_same['absorption_rate']:.3f}, "
          f"diff_rate={res_diff['absorption_rate']:.3f}, "
          f"diff={res_same['absorption_rate'] - res_diff['absorption_rate']:.3f}")

    ai_results_same.append(res_same["absorption_rate"])
    ai_results_diff.append(res_diff["absorption_rate"])

sys.stdout.flush()

n_ai = len(ai_results_same)
ai_same_mean = float(np.mean(ai_results_same)) if ai_results_same else 0.0
ai_diff_mean = float(np.mean(ai_results_diff)) if ai_results_diff else 0.0
ai_ratio = ai_same_mean / ai_diff_mean if ai_diff_mean > 0 else (10.0 if ai_same_mean > 0 else 1.0)
ai_go_nogo = "GO" if ai_ratio >= 1.5 else ("WEAK" if ai_ratio >= 1.1 else "NO_GO")

# Bootstrap CI on ratio
boot_ratios_ai = []
for _ in range(1000):
    if n_ai > 0:
        idx = rng_boot.choice(n_ai, n_ai, replace=True)
        s = float(np.mean([ai_results_same[i] for i in idx]))
        d = float(np.mean([ai_results_diff[i] for i in idx]))
        boot_ratios_ai.append(s / d if d > 0 else (10.0 if s > 0 else 1.0))
ai_ci_lower = float(np.percentile(boot_ratios_ai, 2.5)) if boot_ratios_ai else 0.0
ai_ci_upper = float(np.percentile(boot_ratios_ai, 97.5)) if boot_ratios_ai else 0.0

# Paired t-test
if n_ai >= 5:
    t_ai, p_ai = stats.ttest_rel(ai_results_same, ai_results_diff)
else:
    t_ai, p_ai = None, None

print(f"\n  [animate_inanimate] n_words={n_ai}, same={ai_same_mean:.4f}, diff={ai_diff_mean:.4f}")
print(f"    ratio_to_null={ai_ratio:.3f}, CI=[{ai_ci_lower:.3f}, {ai_ci_upper:.3f}], go={ai_go_nogo}")
sys.stdout.flush()

result_ai = {
    "hierarchy": "animate_inanimate",
    "measurement": "Animate words in animate-ICL vs inanimate-ICL (category-specific suppression)",
    "n_words": n_ai,
    "same_category_absorption_rate": ai_same_mean,
    "diff_category_absorption_rate": ai_diff_mean,
    "ratio_to_null": ai_ratio,
    "ci_lower": ai_ci_lower,
    "ci_upper": ai_ci_upper,
    "paired_t_stat": float(t_ai) if t_ai is not None else None,
    "paired_p_val": float(p_ai) if p_ai is not None else None,
    "go_nogo": ai_go_nogo,
}

# ── Step 4: NOUN_PROPER ───────────────────────────────────────────────────────

report_progress(4, TOTAL_STEPS, "noun_proper: proper-noun-ICL vs common-noun-ICL")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper")
print("Design: proper nouns in proper-ICL vs proper nouns in common-ICL")
print("="*70)
sys.stdout.flush()

proper_words_test = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney",
    "Alice", "Robert", "Michael", "Jennifer", "William",
]
common_words_pool = [
    "table", "chair", "river", "music", "stone", "flower",
    "bridge", "forest", "garden", "paper", "clock", "window",
]

proper_icl_ex = [
    "London: proper", "Paris: proper", "Alice: proper", "Berlin: proper",
    "table: common", "river: common", "chair: common", "flower: common",
]
common_icl_ex = [
    "table: common", "river: common", "chair: common", "flower: common",
    "London: proper", "Paris: proper", "Alice: proper", "Berlin: proper",
]

rng_np = random.Random(SEED + 2)
np_results_same = []
np_results_diff = []

for word in proper_words_test:
    word_feats = find_word_specific_features(word, common_words_pool, top_k=8, min_act=0.3)
    if not word_feats:
        print(f"  '{word}': no specific features found")
        continue

    # Proper-noun ICL context (category-matching)
    examples_p = proper_icl_ex.copy()
    rng_np.shuffle(examples_p)
    same_text = make_category_prompt(word, examples_p, k=6)

    # Common-noun ICL context (wrong category control)
    examples_c = common_icl_ex.copy()
    rng_np.shuffle(examples_c)
    diff_text = make_category_prompt(word, examples_c, k=6)

    res_same = measure_suppression_in_context(word, word_feats, same_text, icl_position=-2)
    res_diff = measure_suppression_in_context(word, word_feats, diff_text, icl_position=-2)

    if res_same is None or res_diff is None:
        print(f"  '{word}': context measurement failed")
        continue

    print(f"  '{word}' (proper): "
          f"n_child={res_same['n_child_features']}, "
          f"same_rate={res_same['absorption_rate']:.3f}, "
          f"diff_rate={res_diff['absorption_rate']:.3f}, "
          f"diff={res_same['absorption_rate'] - res_diff['absorption_rate']:.3f}")

    np_results_same.append(res_same["absorption_rate"])
    np_results_diff.append(res_diff["absorption_rate"])

sys.stdout.flush()

n_np = len(np_results_same)
np_same_mean = float(np.mean(np_results_same)) if np_results_same else 0.0
np_diff_mean = float(np.mean(np_results_diff)) if np_results_diff else 0.0
np_ratio = np_same_mean / np_diff_mean if np_diff_mean > 0 else (10.0 if np_same_mean > 0 else 1.0)
np_go_nogo = "GO" if np_ratio >= 1.5 else ("WEAK" if np_ratio >= 1.1 else "NO_GO")

# Bootstrap CI on ratio
boot_ratios_np = []
for _ in range(1000):
    if n_np > 0:
        idx = rng_boot.choice(n_np, n_np, replace=True)
        s = float(np.mean([np_results_same[i] for i in idx]))
        d = float(np.mean([np_results_diff[i] for i in idx]))
        boot_ratios_np.append(s / d if d > 0 else (10.0 if s > 0 else 1.0))
np_ci_lower = float(np.percentile(boot_ratios_np, 2.5)) if boot_ratios_np else 0.0
np_ci_upper = float(np.percentile(boot_ratios_np, 97.5)) if boot_ratios_np else 0.0

# Paired t-test
if n_np >= 5:
    t_np, p_np = stats.ttest_rel(np_results_same, np_results_diff)
else:
    t_np, p_np = None, None

print(f"\n  [noun_proper] n_words={n_np}, same={np_same_mean:.4f}, diff={np_diff_mean:.4f}")
print(f"    ratio_to_null={np_ratio:.3f}, CI=[{np_ci_lower:.3f}, {np_ci_upper:.3f}], go={np_go_nogo}")
sys.stdout.flush()

result_np = {
    "hierarchy": "noun_proper",
    "measurement": "Proper nouns in proper-ICL vs common-ICL (category-specific suppression)",
    "n_words": n_np,
    "same_category_absorption_rate": np_same_mean,
    "diff_category_absorption_rate": np_diff_mean,
    "ratio_to_null": np_ratio,
    "ci_lower": np_ci_lower,
    "ci_upper": np_ci_upper,
    "paired_t_stat": float(t_np) if t_np is not None else None,
    "paired_p_val": float(p_np) if p_np is not None else None,
    "go_nogo": np_go_nogo,
}

# ── Step 5: DIAGNOSTIC — Verify with corpus-based P(child|parent_fires) ───────

report_progress(5, TOTAL_STEPS, "Diagnostic: corpus-based parent/child activation analysis")

print("\n" + "="*70)
print("DIAGNOSTIC: Probe-aligned parent latents → child suppression check")
print("="*70)
sys.stdout.flush()

# Load probe weights for first_letter
probe_weights = torch.tensor(probe_fl).float()  # [24, 768]
probe_weights_norm = F.normalize(probe_weights, dim=-1)

# Find top-K SAE latents aligned with letter 'a' probe
letter_a_idx = list(probe_fl_classes).index('a')
probe_a = probe_weights_norm[letter_a_idx]  # [768]

# Cosine similarity with decoder columns
cos_sims_a = F.cosine_similarity(
    probe_a.unsqueeze(0).expand(d_sae, -1),
    W_dec.float(),  # [d_sae, d_model]
    dim=-1
)
top_a_latents = torch.argsort(cos_sims_a, descending=True)[:10]
print(f"  Letter 'a' top-10 SAE latents by probe-decoder cos_sim:")
for j in top_a_latents.tolist()[:5]:
    print(f"    Latent {j}: cos_sim={cos_sims_a[j].item():.4f}")

# Test: do these fire in ICL context for 'apple'?
test_word = "apple"
try:
    from sae_spelling.prompting import create_icl_prompt, first_letter_formatter
    sample_icl_vocab = [w for w in icl_vocab_by_letter.get('a', []) if w != test_word][:10]
    prompt_test = create_icl_prompt(
        test_word, examples=sample_icl_vocab,
        base_template="{word}:", answer_formatter=first_letter_formatter(),
        max_icl_examples=10
    )
    icl_test_text = prompt_test.base

    acts_icl = get_sae_acts_pos([icl_test_text], position=-2)
    acts_iso = get_sae_acts_last([test_word])

    if acts_icl is not None:
        icl_vec = acts_icl[0].numpy()
        iso_vec = acts_iso[0].numpy()

        print(f"\n  For '{test_word}':")
        print(f"    Top-5 letter-a latents in ICL context:")
        for j in top_a_latents.tolist()[:5]:
            print(f"      Latent {j}: ICL={icl_vec[j]:.3f}, ISO={iso_vec[j]:.3f}")

        # Check actual ICL-active features for 'apple'
        top_icl_feats = np.argsort(icl_vec)[::-1][:10]
        print(f"    Top-10 actually active features in ICL:")
        for j in top_icl_feats:
            if icl_vec[j] > 0.1:
                print(f"      Latent {j}: ICL={icl_vec[j]:.3f}, ISO={iso_vec[j]:.3f}, cos_sim_a={cos_sims_a[j].item():.4f}")
except Exception as e:
    print(f"  Diagnostic error: {e}")

sys.stdout.flush()

# ── Step 6: Summary ───────────────────────────────────────────────────────────

report_progress(6, TOTAL_STEPS, "Computing overall summary")


def safe_ratio_v(r):
    if r is None:
        return 0.0
    if isinstance(r, float) and (np.isnan(r) or np.isinf(r)):
        return 10.0
    return float(r)


fl_r = safe_ratio_v(result_first_letter["ratio_to_null"])
ai_r = safe_ratio_v(result_ai["ratio_to_null"])
np_r = safe_ratio_v(result_np["ratio_to_null"])

n_passing = sum(1 for r in [fl_r, ai_r, np_r] if r >= 1.5)
n_weak = sum(1 for r in [fl_r, ai_r, np_r] if 1.1 <= r < 1.5)

sanity_pass = fl_r >= 1.5
sanity_warn = fl_r < 1.0

overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if sanity_warn and n_passing == 0:
    overall = "INVESTIGATE"

bonferroni_alpha = 0.05 / 3.0

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v5)")
print("="*70)
print(f"  first_letter:      same={fl_same_mean:.4f}, diff={fl_diff_mean:.4f}, ratio={fl_r:.3f}, go={fl_go_nogo}")
print(f"  animate_inanimate: same={ai_same_mean:.4f}, diff={ai_diff_mean:.4f}, ratio={ai_r:.3f}, go={ai_go_nogo}")
print(f"  noun_proper:       same={np_same_mean:.4f}, diff={np_diff_mean:.4f}, ratio={np_r:.3f}, go={np_go_nogo}")
print(f"  Sanity check (first_letter ratio>=1.5): {'PASS' if sanity_pass else 'FAIL'}")
print(f"  Overall: {overall} (n_passing={n_passing}, n_weak={n_weak})")
sys.stdout.flush()

# ── Step 7: Save results ──────────────────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Saving results")

elapsed = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "task_name": "task_C2_redesign",
    "mode": "PILOT",
    "version": "redesign_v5_category_specific",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": LAYER,
        "d_sae": d_sae,
        "d_model": d_model,
        "seed": SEED,
        "child_min_act": 0.3,
        "suppression_threshold": 0.30,
        "n_bootstrap": 1000,
        "bonferroni_alpha": bonferroni_alpha,
        "measurement_approach": (
            "Category-specific suppression: P(child_suppressed | same-category ICL) / "
            "P(child_suppressed | different-category ICL). "
            "Child features identified as word-specific features (active for word but not for reference words). "
            "Null = same concept words but in WRONG category ICL context."
        ),
    },
    "hierarchies": {
        "first_letter": result_first_letter,
        "animate_inanimate": result_ai,
        "noun_proper": result_np,
    },
    "summary": {
        "first_letter_same_rate": fl_same_mean,
        "first_letter_diff_rate": fl_diff_mean,
        "first_letter_ratio": fl_r,
        "animate_same_rate": ai_same_mean,
        "animate_diff_rate": ai_diff_mean,
        "animate_ratio": ai_r,
        "noun_proper_same_rate": np_same_mean,
        "noun_proper_diff_rate": np_diff_mean,
        "noun_proper_ratio": np_r,
        "n_passing_ratio_1.5": n_passing,
        "n_weak_ratio_1.1": n_weak,
        "overall_go_nogo": overall,
        "bonferroni_alpha": bonferroni_alpha,
        "sanity_check": {
            "criteria": "first_letter ratio >= 1.5 (category-specific suppression)",
            "ratio": fl_r,
            "result": "PASS" if sanity_pass else ("INVESTIGATE" if sanity_warn else "FAIL"),
        },
    },
    "pilot_pass_criteria": {
        "criteria": "Ratio-to-null >= 1.5 for first_letter (sanity check). If < 1.0: investigate.",
        "first_letter_ratio": fl_r,
        "first_letter_result": fl_go_nogo,
        "overall": overall,
    },
    "key_diagnostics": {
        "v5_design": (
            "Correct category-specific suppression measurement. "
            "Same concept words run in SAME-category ICL vs DIFFERENT-category ICL. "
            "If ratio > 1.5: absorption is category-specific (not just general context shift). "
            "If ratio ≈ 1.0: ICL context causes equal suppression regardless of category match."
        ),
        "previous_failure": "v4 used concept-ICL for concept words vs control-ICL for control words. Since ALL words get suppressed by ANY ICL context, ratio=1.0.",
        "null_baseline": "Same concept words in WRONG-category ICL context (same ICL structure, different category)",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nSaved: {OUTPUT_FILE}")
sys.stdout.flush()

# ── Update gpu_progress.json ──────────────────────────────────────────────────

report_progress(8, TOTAL_STEPS, "Updating gpu_progress.json")

gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gp = json.loads(gpu_progress_file.read_text())
except Exception:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if TASK_ID not in gp.get("completed", []):
    gp.setdefault("completed", []).append(TASK_ID)
if TASK_ID in gp.get("running", {}):
    del gp["running"][TASK_ID]

gp.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 55,
    "actual_min": int(elapsed / 60),
    "start_time": datetime.fromtimestamp(start_time).isoformat(),
    "end_time": datetime.now().isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "method": "category-specific-suppression-v5",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mode": "PILOT",
    }
}
gpu_progress_file.write_text(json.dumps(gp, indent=2))

# ── Done ─────────────────────────────────────────────────────────────────────

report_progress(9, TOTAL_STEPS, "Writing DONE marker")

summary = (
    f"C2-REDESIGN v5 PILOT: {overall}. "
    f"first_letter same={fl_same_mean:.3f} diff={fl_diff_mean:.3f} ratio={fl_r:.3f} ({fl_go_nogo}). "
    f"animate same={ai_same_mean:.3f} diff={ai_diff_mean:.3f} ratio={ai_r:.3f} ({ai_go_nogo}). "
    f"noun_proper same={np_same_mean:.3f} diff={np_diff_mean:.3f} ratio={np_r:.3f} ({np_go_nogo}). "
    f"n_passing={n_passing}/3."
)
print(f"\n{'='*70}")
print(f"SUMMARY: {summary}")
print(f"{'='*70}")
sys.stdout.flush()

mark_done("success", summary)
report_progress(10, TOTAL_STEPS, "Complete")
