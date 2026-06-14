"""
Task C.2-REDESIGN v4: Cross-Domain Absorption via Child-Feature Suppression (PILOT MODE)

Based on diagnostic analysis of ICL context activations.

KEY FINDING: In ICL context for letter-words:
- Parent features (ICL-specific): fire in ICL context but NOT in isolation
- Child features (word-specific): fire in isolation but go SILENT in ICL context
- 23 features silent in ICL (absorption), 21 NEW features fire in ICL (absorbing parents)

CORRECT MEASUREMENT for first_letter:
1. For each concept word: get SAE acts in ICL context (pos -2) vs isolation
2. Child feature = one that fires in isolation but NOT in ICL context
3. Parent feature = one that fires in ICL but NOT in isolation (the absorbing feature)
4. Absorption rate = fraction of words where child features are suppressed in ICL vs isolation

For SEMANTIC HIERARCHIES:
- Use ICL-style task prompts designed for each hierarchy
- animate_inanimate: use category-classification ICL prompts
- noun_proper: use entity-type classification prompts
- Fallback to activation comparison if ICL doesn't produce absorption signal

OUTPUT: exp/results/full/C2_child_suppression_absorption.json
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


TOTAL_STEPS = 14
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v4: ICL direct absorption")

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

# ── Utility functions ─────────────────────────────────────────────────────────

def get_sae_acts_pos(texts, position, batch_size=16):
    """Get SAE activations at specific position."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        if tokens.shape[1] <= abs(position):
            continue
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
        resid = cache[CACHE_KEY][:, position, :].float()
        with torch.no_grad():
            acts = sae.encode(resid)
        all_acts.append(acts.cpu())
    if not all_acts:
        return None
    return torch.cat(all_acts, dim=0)


def get_sae_acts_last(texts, batch_size=64):
    """Get SAE activations at last token position."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
        resid = cache[CACHE_KEY][:, -1, :].float()
        with torch.no_grad():
            acts = sae.encode(resid)
        all_acts.append(acts.cpu())
    return torch.cat(all_acts, dim=0)


# ── Core measurement function ─────────────────────────────────────────────────

def measure_icl_absorption_for_word(
    word,
    icl_context,
    icl_position,   # position of word in ICL context (-2 for "word:" format)
    isolation_context=None,  # how to present word in isolation (None = word alone)
    isolation_position=-1,
    child_min_act=0.5,       # min activation in isolation to be candidate child feature
    suppression_threshold=0.30,  # fraction of activation that must drop
):
    """
    Measure child-feature suppression for one word.

    1. Get SAE acts at word position in ICL context
    2. Get SAE acts for word in isolation
    3. Find child features: fire in isolation but suppressed in ICL
    4. Return suppression statistics

    Returns: dict with suppression data, or None if no candidate child features
    """
    # ICL acts
    icl_acts = get_sae_acts_pos([icl_context], position=icl_position)
    if icl_acts is None:
        return None
    icl_vec = icl_acts[0].numpy()

    # Isolation acts
    if isolation_context is None:
        isolation_context = word
    isolation_acts = get_sae_acts_pos([isolation_context], position=isolation_position)
    if isolation_acts is None:
        isolation_acts = get_sae_acts_last([isolation_context])
    iso_vec = isolation_acts[0].numpy()

    # Find child features: active in isolation, suppressed in ICL
    child_candidates = []
    for j in range(d_sae):
        iso_act = float(iso_vec[j])
        icl_act = float(icl_vec[j])
        if iso_act >= child_min_act:
            suppression = (iso_act - icl_act) / iso_act
            if suppression >= suppression_threshold:
                child_candidates.append({
                    "latent_idx": j,
                    "iso_act": iso_act,
                    "icl_act": icl_act,
                    "suppression_ratio": float(suppression),
                    "is_suppressed": True,
                })
            else:
                child_candidates.append({
                    "latent_idx": j,
                    "iso_act": iso_act,
                    "icl_act": icl_act,
                    "suppression_ratio": float(suppression),
                    "is_suppressed": False,
                })

    if not child_candidates:
        return None

    # Find parent features: fire in ICL but NOT in isolation (absorbing features)
    parent_candidates = []
    for j in range(d_sae):
        icl_act = float(icl_vec[j])
        iso_act = float(iso_vec[j])
        if icl_act >= child_min_act and iso_act < 0.1:
            parent_candidates.append({
                "latent_idx": j,
                "iso_act": iso_act,
                "icl_act": icl_act,
                "enhancement": icl_act - iso_act,
            })

    # Sort by suppression ratio for child features
    child_candidates.sort(key=lambda x: -x["suppression_ratio"])

    n_suppressed = sum(1 for c in child_candidates if c["is_suppressed"])
    n_total = len(child_candidates)
    absorption_rate_per_word = n_suppressed / n_total if n_total > 0 else 0.0

    # Mean suppression across all child candidates
    mean_suppression = float(np.mean([c["suppression_ratio"] for c in child_candidates]))

    return {
        "word": word,
        "n_child_candidates": n_total,
        "n_suppressed": n_suppressed,
        "absorption_rate_per_word": absorption_rate_per_word,
        "mean_suppression_ratio": mean_suppression,
        "top_child_features": child_candidates[:3],
        "top_parent_features": sorted(parent_candidates, key=lambda x: -x["enhancement"])[:3],
        "is_absorbed": n_suppressed > 0,
    }


# ── Step 3: FIRST_LETTER measurement (ICL spelling context) ───────────────────

report_progress(2, TOTAL_STEPS, "First_letter: ICL spelling absorption")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter (ICL spelling context, direct suppression)")
print("="*70)

icl_words_fl = [
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "cat",
    "dog", "run", "sit", "top", "win", "big", "buy", "got", "job", "may",
]

# Test words: letters with F1 >= 0.80 from C1
test_words_by_letter = {
    "a": ["apple", "angel", "arrow", "album", "alarm", "anchor", "arena", "actor"],
    "b": ["bird", "brain", "bread", "bridge", "brush", "blood", "beach"],
    "s": ["stone", "storm", "smile", "snake", "sheep", "shell", "spark"],
    "m": ["music", "metal", "magic", "model", "maple", "month", "moral"],
    "n": ["night", "novel", "nurse", "north", "nerve", "noble", "noise"],
}

rng_fl = random.Random(SEED)

fl_word_results = []
fl_by_letter = {}

for letter, words in test_words_by_letter.items():
    letter_results = []

    for word in words:
        # Create ICL prompt for spelling task
        examples = [w for w in icl_words_fl if w.lower() != word.lower()]
        rng_fl.shuffle(examples)
        try:
            prompt = create_icl_prompt(
                word,
                examples=examples,
                base_template="{word}:",
                answer_formatter=first_letter_formatter(),
                max_icl_examples=10,
                shuffle_examples=False,  # already shuffled
            )
            icl_text = prompt.base
        except Exception as e:
            print(f"  Prompt error for '{word}': {e}")
            continue

        result = measure_icl_absorption_for_word(
            word=word,
            icl_context=icl_text,
            icl_position=-2,
            isolation_context=word,
            isolation_position=-1,
            child_min_act=0.5,
            suppression_threshold=0.30,
        )

        if result is None:
            print(f"  '{word}': no candidate child features")
            continue

        print(f"  '{word}' (letter '{letter}'): "
              f"n_child={result['n_child_candidates']}, "
              f"n_suppressed={result['n_suppressed']}, "
              f"absorption_rate={result['absorption_rate_per_word']:.3f}, "
              f"mean_suppression={result['mean_suppression_ratio']:.3f}")

        fl_word_results.append(result)
        letter_results.append(result)

    if letter_results:
        n_absorbed = sum(1 for r in letter_results if r["is_absorbed"])
        fl_by_letter[letter] = {
            "n_words": len(letter_results),
            "n_absorbed": n_absorbed,
            "absorption_rate": n_absorbed / len(letter_results),
            "mean_suppression": float(np.mean([r["mean_suppression_ratio"] for r in letter_results])),
        }
        print(f"  Letter '{letter}': {n_absorbed}/{len(letter_results)} absorbed, "
              f"mean_suppression={fl_by_letter[letter]['mean_suppression']:.3f}")

# Aggregate first_letter results
n_fl_total = len(fl_word_results)
n_fl_absorbed = sum(1 for r in fl_word_results if r["is_absorbed"])
fl_absorption_rate = n_fl_absorbed / n_fl_total if n_fl_total > 0 else 0.0

# Also compute mean suppression ratio across all child features
all_fl_suppressions = [r["mean_suppression_ratio"] for r in fl_word_results]
fl_mean_suppression = float(np.mean(all_fl_suppressions)) if all_fl_suppressions else None

# Bootstrap CI on absorption rate
rng_boot = np.random.RandomState(SEED)
boot_fl = []
fl_is_absorbed = [int(r["is_absorbed"]) for r in fl_word_results]
for _ in range(1000):
    idx = rng_boot.choice(n_fl_total, n_fl_total, replace=True) if n_fl_total > 0 else []
    boot_fl.append(float(np.mean([fl_is_absorbed[i] for i in idx])) if len(idx) > 0 else 0.0)
fl_ci_lower = float(np.percentile(boot_fl, 2.5))
fl_ci_upper = float(np.percentile(boot_fl, 97.5))

# Null: shuffle word → letter assignments (permutation of is_absorbed)
null_fl_rates = []
for _ in range(100):
    perm = rng_boot.permutation(n_fl_total)
    null_fl_rates.append(float(np.mean([fl_is_absorbed[i] for i in perm])) if n_fl_total > 0 else 0.0)
fl_null_mean = float(np.mean(null_fl_rates))

fl_ratio = fl_absorption_rate / fl_null_mean if fl_null_mean > 0 else (
    10.0 if fl_absorption_rate > 0 else 1.0
)
fl_go_nogo = "GO" if fl_ratio >= 1.5 else ("WEAK" if fl_ratio >= 1.1 else "NO_GO")

print(f"\n  [first_letter] AGGREGATE:")
print(f"    n_total={n_fl_total}, n_absorbed={n_fl_absorbed}, absorption_rate={fl_absorption_rate:.4f}")
print(f"    CI=[{fl_ci_lower:.4f}, {fl_ci_upper:.4f}]")
print(f"    null_mean={fl_null_mean:.4f}, ratio_to_null={fl_ratio:.3f}")
print(f"    mean_suppression={fl_mean_suppression}")
print(f"    go_nogo={fl_go_nogo}")
sys.stdout.flush()

# Find parent latents (empirical, using ICL context)
fl_all_parent_latents = {}
for letter, words in test_words_by_letter.items():
    # Aggregate parent features across words for this letter
    examples = [w for w in icl_words_fl]
    rng_fl.shuffle(examples)
    icl_texts = []
    for word in words[:3]:  # sample 3 words
        try:
            prompt = create_icl_prompt(word, examples=[w for w in examples if w != word],
                                        base_template="{word}:", answer_formatter=first_letter_formatter(),
                                        max_icl_examples=10)
            icl_texts.append(prompt.base)
        except Exception:
            pass

    if icl_texts:
        icl_mean_acts = get_sae_acts_pos(icl_texts, position=-2)
        if icl_mean_acts is not None:
            iso_acts = get_sae_acts_last(words[:3])
            diff = icl_mean_acts.mean(dim=0).numpy() - iso_acts.mean(dim=0).numpy()
            top_parent = np.argsort(diff)[::-1][:3]
            icl_mv = icl_mean_acts.mean(dim=0).numpy()
            iso_mv = iso_acts.mean(dim=0).numpy()
            fl_all_parent_latents[letter] = [
                {"latent_idx": int(j), "icl_mean": float(icl_mv[j]), "iso_mean": float(iso_mv[j]),
                 "enhancement": float(diff[j])}
                for j in top_parent
                if float(icl_mv[j]) >= 0.3
            ]

result_first_letter = {
    "hierarchy": "first_letter",
    "measurement": "ICL-based direct suppression: child features active in isolation but suppressed in ICL context",
    "n_words": n_fl_total,
    "n_absorbed": n_fl_absorbed,
    "absorption_rate": fl_absorption_rate,
    "mean_suppression_ratio": fl_mean_suppression,
    "bootstrap_ci_lower": fl_ci_lower,
    "bootstrap_ci_upper": fl_ci_upper,
    "ci_reliable": n_fl_total >= 10,
    "null_mean": fl_null_mean,
    "ratio_to_null": fl_ratio,
    "go_nogo": fl_go_nogo,
    "results_by_letter": fl_by_letter,
    "parent_latent_info": fl_all_parent_latents,
    "word_level_results": fl_word_results[:10],  # sample
}

# ── Step 4: ANIMATE_INANIMATE using ICL-style category classification ──────────

report_progress(3, TOTAL_STEPS, "animate_inanimate: ICL category classification")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate (ICL category classification)")
print("="*70)
sys.stdout.flush()

animate_words = [
    "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger",
    "wolf", "bear", "eagle", "monkey", "rabbit",
]
inanimate_words = [
    "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
    "bottle", "box", "ring", "cloud", "river",
]

# ICL examples for animate classification: "X: animate/inanimate"
animate_icl_examples = (
    [f"{w}: animate" for w in ["dog", "cat", "bird"][:2]] +
    [f"{w}: inanimate" for w in ["rock", "table", "car"][:2]]
)

def create_animate_icl_prompt(word, examples):
    """Create an ICL prompt for animate/inanimate classification."""
    example_str = "\n".join(examples)
    return f"{example_str}\n{word}:"


rng_ai = random.Random(SEED)
ai_word_results = []

for word in animate_words:
    examples_shuffled = animate_icl_examples.copy()
    rng_ai.shuffle(examples_shuffled)
    icl_text = create_animate_icl_prompt(word, examples_shuffled)

    result = measure_icl_absorption_for_word(
        word=word,
        icl_context=icl_text,
        icl_position=-2,
        isolation_context=word,
        isolation_position=-1,
        child_min_act=0.5,
        suppression_threshold=0.30,
    )

    if result is None:
        print(f"  '{word}': no candidate child features")
        continue

    print(f"  '{word}' (animate): "
          f"n_child={result['n_child_candidates']}, "
          f"n_suppressed={result['n_suppressed']}, "
          f"absorption_rate={result['absorption_rate_per_word']:.3f}, "
          f"mean_suppression={result['mean_suppression_ratio']:.3f}")
    ai_word_results.append(result)

n_ai_total = len(ai_word_results)
n_ai_absorbed = sum(1 for r in ai_word_results if r["is_absorbed"])
ai_absorption_rate = n_ai_absorbed / n_ai_total if n_ai_total > 0 else 0.0
ai_suppressions = [r["mean_suppression_ratio"] for r in ai_word_results]
ai_mean_suppression = float(np.mean(ai_suppressions)) if ai_suppressions else None

# Bootstrap CI
boot_ai = []
ai_is_absorbed = [int(r["is_absorbed"]) for r in ai_word_results]
for _ in range(1000):
    if n_ai_total > 0:
        idx = rng_boot.choice(n_ai_total, n_ai_total, replace=True)
        boot_ai.append(float(np.mean([ai_is_absorbed[i] for i in idx])))
    else:
        boot_ai.append(0.0)
ai_ci_lower = float(np.percentile(boot_ai, 2.5))
ai_ci_upper = float(np.percentile(boot_ai, 97.5))

# Null: inanimate words in the same ICL context (should NOT show absorption)
ai_null_results = []
for word in inanimate_words[:len(animate_words)]:
    examples_shuffled = animate_icl_examples.copy()
    rng_ai.shuffle(examples_shuffled)
    icl_text = create_animate_icl_prompt(word, examples_shuffled)

    result = measure_icl_absorption_for_word(
        word=word,
        icl_context=icl_text,
        icl_position=-2,
        isolation_context=word,
        isolation_position=-1,
        child_min_act=0.5,
        suppression_threshold=0.30,
    )
    if result is not None:
        ai_null_results.append(result)

ai_null_rate = float(np.mean([int(r["is_absorbed"]) for r in ai_null_results])) if ai_null_results else ai_absorption_rate
ai_ratio = ai_absorption_rate / ai_null_rate if ai_null_rate > 0 else (
    10.0 if ai_absorption_rate > 0 else 1.0
)
ai_go_nogo = "GO" if ai_ratio >= 1.5 else ("WEAK" if ai_ratio >= 1.1 else "NO_GO")

# Find parent latents for animate
ai_icl_texts = [create_animate_icl_prompt(w, animate_icl_examples) for w in animate_words[:5]]
ai_iso_acts = get_sae_acts_last(animate_words[:5])
ai_icl_acts = get_sae_acts_pos(ai_icl_texts, position=-2)
ai_parent_info_dict = {}
if ai_icl_acts is not None:
    ai_icl_mv = ai_icl_acts.mean(dim=0).numpy()
    ai_iso_mv = ai_iso_acts.mean(dim=0).numpy()
    ai_diff = ai_icl_mv - ai_iso_mv
    top_ai_parent = np.argsort(ai_diff)[::-1][:5]
    ai_parent_info_dict["animate"] = [
        {"latent_idx": int(j), "icl_mean": float(ai_icl_mv[j]), "iso_mean": float(ai_iso_mv[j]),
         "enhancement": float(ai_diff[j])}
        for j in top_ai_parent
        if float(ai_icl_mv[j]) >= 0.3
    ]

print(f"\n  [animate_inanimate] n_total={n_ai_total}, n_absorbed={n_ai_absorbed}, "
      f"absorption_rate={ai_absorption_rate:.4f}")
print(f"    null_rate={ai_null_rate:.4f}, ratio_to_null={ai_ratio:.3f}, go={ai_go_nogo}")
print(f"    mean_suppression={ai_mean_suppression}")
sys.stdout.flush()

result_ai = {
    "hierarchy": "animate_inanimate",
    "measurement": "ICL animate/inanimate classification context",
    "n_words": n_ai_total,
    "n_absorbed": n_ai_absorbed,
    "absorption_rate": ai_absorption_rate,
    "mean_suppression_ratio": ai_mean_suppression,
    "bootstrap_ci_lower": ai_ci_lower,
    "bootstrap_ci_upper": ai_ci_upper,
    "ci_reliable": n_ai_total >= 10,
    "null_rate": ai_null_rate,
    "ratio_to_null": ai_ratio,
    "go_nogo": ai_go_nogo,
    "parent_latent_info": ai_parent_info_dict,
    "word_level_results": ai_word_results[:10],
}

# ── Step 5: NOUN_PROPER using ICL-style entity classification ────────────────

report_progress(4, TOTAL_STEPS, "noun_proper: ICL entity classification")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper (ICL entity classification context)")
print("="*70)
sys.stdout.flush()

proper_words = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
    "Alice", "Robert", "Michael", "Jennifer", "William",
]
common_words = [
    "table", "chair", "river", "music", "stone", "flower", "kitchen",
    "bridge", "forest", "garden", "paper", "clock",
]

# ICL examples for proper noun classification
proper_icl_examples = (
    [f"{w}: proper" for w in ["London", "Alice", "Google"][:2]] +
    [f"{w}: common" for w in ["table", "chair", "river"][:2]]
)


def create_proper_icl_prompt(word, examples):
    example_str = "\n".join(examples)
    return f"{example_str}\n{word}:"


rng_np = random.Random(SEED)
np_word_results = []

for word in proper_words:
    examples_shuffled = proper_icl_examples.copy()
    rng_np.shuffle(examples_shuffled)
    icl_text = create_proper_icl_prompt(word, examples_shuffled)

    result = measure_icl_absorption_for_word(
        word=word,
        icl_context=icl_text,
        icl_position=-2,
        isolation_context=word,
        isolation_position=-1,
        child_min_act=0.5,
        suppression_threshold=0.30,
    )

    if result is None:
        print(f"  '{word}': no candidate child features")
        continue

    print(f"  '{word}' (proper): "
          f"n_child={result['n_child_candidates']}, "
          f"n_suppressed={result['n_suppressed']}, "
          f"absorption_rate={result['absorption_rate_per_word']:.3f}, "
          f"mean_suppression={result['mean_suppression_ratio']:.3f}")
    np_word_results.append(result)

n_np_total = len(np_word_results)
n_np_absorbed = sum(1 for r in np_word_results if r["is_absorbed"])
np_absorption_rate = n_np_absorbed / n_np_total if n_np_total > 0 else 0.0
np_suppressions = [r["mean_suppression_ratio"] for r in np_word_results]
np_mean_suppression = float(np.mean(np_suppressions)) if np_suppressions else None

# Bootstrap CI
boot_np = []
np_is_absorbed = [int(r["is_absorbed"]) for r in np_word_results]
for _ in range(1000):
    if n_np_total > 0:
        idx = rng_boot.choice(n_np_total, n_np_total, replace=True)
        boot_np.append(float(np.mean([np_is_absorbed[i] for i in idx])))
    else:
        boot_np.append(0.0)
np_ci_lower = float(np.percentile(boot_np, 2.5))
np_ci_upper = float(np.percentile(boot_np, 97.5))

# Null: common nouns in same ICL context
np_null_results = []
for word in common_words[:len(proper_words)]:
    examples_shuffled = proper_icl_examples.copy()
    rng_np.shuffle(examples_shuffled)
    icl_text = create_proper_icl_prompt(word, examples_shuffled)
    result = measure_icl_absorption_for_word(
        word=word, icl_context=icl_text, icl_position=-2,
        isolation_context=word, isolation_position=-1,
        child_min_act=0.5, suppression_threshold=0.30,
    )
    if result is not None:
        np_null_results.append(result)

np_null_rate = float(np.mean([int(r["is_absorbed"]) for r in np_null_results])) if np_null_results else np_absorption_rate
np_ratio = np_absorption_rate / np_null_rate if np_null_rate > 0 else (
    10.0 if np_absorption_rate > 0 else 1.0
)
np_go_nogo = "GO" if np_ratio >= 1.5 else ("WEAK" if np_ratio >= 1.1 else "NO_GO")

# Find parent latents for proper_noun
np_icl_texts = [create_proper_icl_prompt(w, proper_icl_examples) for w in proper_words[:5]]
np_iso_acts = get_sae_acts_last(proper_words[:5])
np_icl_acts = get_sae_acts_pos(np_icl_texts, position=-2)
np_parent_info_dict = {}
if np_icl_acts is not None:
    np_icl_mv = np_icl_acts.mean(dim=0).numpy()
    np_iso_mv = np_iso_acts.mean(dim=0).numpy()
    np_diff_mv = np_icl_mv - np_iso_mv
    top_np_parent = np.argsort(np_diff_mv)[::-1][:5]
    np_parent_info_dict["proper_noun"] = [
        {"latent_idx": int(j), "icl_mean": float(np_icl_mv[j]), "iso_mean": float(np_iso_mv[j]),
         "enhancement": float(np_diff_mv[j])}
        for j in top_np_parent
        if float(np_icl_mv[j]) >= 0.3
    ]

print(f"\n  [noun_proper] n_total={n_np_total}, n_absorbed={n_np_absorbed}, "
      f"absorption_rate={np_absorption_rate:.4f}")
print(f"    null_rate={np_null_rate:.4f}, ratio_to_null={np_ratio:.3f}, go={np_go_nogo}")
print(f"    mean_suppression={np_mean_suppression}")
sys.stdout.flush()

result_np = {
    "hierarchy": "noun_proper",
    "measurement": "ICL proper noun classification context",
    "n_words": n_np_total,
    "n_absorbed": n_np_absorbed,
    "absorption_rate": np_absorption_rate,
    "mean_suppression_ratio": np_mean_suppression,
    "bootstrap_ci_lower": np_ci_lower,
    "bootstrap_ci_upper": np_ci_upper,
    "ci_reliable": n_np_total >= 10,
    "null_rate": np_null_rate,
    "ratio_to_null": np_ratio,
    "go_nogo": np_go_nogo,
    "parent_latent_info": np_parent_info_dict,
    "word_level_results": np_word_results[:10],
}

# ── Step 6: EDA for parent latents ───────────────────────────────────────────

report_progress(5, TOTAL_STEPS, "Computing EDA for parent latents")


def compute_eda(latent_indices):
    edas = []
    for j in latent_indices:
        enc_j = W_enc[:, j].float()
        dec_j = W_dec[j, :].float()
        cos_sim = F.cosine_similarity(enc_j.unsqueeze(0), dec_j.unsqueeze(0)).item()
        edas.append(1.0 - float(cos_sim))
    return edas


eda_per_hierarchy = {}

# first_letter: aggregate parent latents from ICL
fl_parent_idx = []
for letter, latents in fl_all_parent_latents.items():
    fl_parent_idx.extend([lt["latent_idx"] for lt in latents])
fl_parent_idx = list(set(fl_parent_idx))
fl_edas = compute_eda(fl_parent_idx)
eda_per_hierarchy["first_letter"] = {
    "parent_latent_indices": fl_parent_idx,
    "eda_values": fl_edas,
    "mean_parent_eda": float(np.mean(fl_edas)) if fl_edas else None,
    "note": "ICL-context enhanced features (appear in ICL but not isolation)",
}

# animate
ai_parent_idx = [lt["latent_idx"] for lt in ai_parent_info_dict.get("animate", [])]
ai_edas = compute_eda(ai_parent_idx)
eda_per_hierarchy["animate_inanimate"] = {
    "parent_latent_indices": ai_parent_idx,
    "eda_values": ai_edas,
    "mean_parent_eda": float(np.mean(ai_edas)) if ai_edas else None,
}

# proper_noun
np_parent_idx = [lt["latent_idx"] for lt in np_parent_info_dict.get("proper_noun", [])]
np_edas = compute_eda(np_parent_idx)
eda_per_hierarchy["noun_proper"] = {
    "parent_latent_indices": np_parent_idx,
    "eda_values": np_edas,
    "mean_parent_eda": float(np.mean(np_edas)) if np_edas else None,
}

print("\nEDA per hierarchy:")
for h, ed in eda_per_hierarchy.items():
    print(f"  {h}: n_parents={len(ed['parent_latent_indices'])}, mean_EDA={ed.get('mean_parent_eda')}")
sys.stdout.flush()

# ── Step 7: AUPRC + summary ───────────────────────────────────────────────────

report_progress(6, TOTAL_STEPS, "Computing overall summary")


def safe_ratio(result):
    r = result.get("ratio_to_null")
    if r is None:
        return 0.0
    if isinstance(r, float) and (np.isnan(r) or np.isinf(r)):
        ar = result.get("absorption_rate", 0) or 0.0
        return 10.0 if ar > 0 else 1.0
    return float(r)


fl_ratio_s = safe_ratio(result_first_letter)
ai_ratio_s = safe_ratio(result_ai)
np_ratio_s = safe_ratio(result_np)

n_passing = sum(1 for r in [fl_ratio_s, ai_ratio_s, np_ratio_s] if r >= 1.5)
n_weak = sum(1 for r in [fl_ratio_s, ai_ratio_s, np_ratio_s] if 1.1 <= r < 1.5)

sanity_pass = fl_ratio_s >= 1.5
sanity_warn = fl_ratio_s < 1.0

overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if sanity_warn and n_passing == 0:
    overall = "PIPELINE_BUG_INVESTIGATE"

bonferroni_alpha = 0.05 / 3.0

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v4)")
print("="*70)
print(f"  first_letter:      absorption_rate={fl_absorption_rate:.4f}, ratio={fl_ratio_s:.3f}, go={fl_go_nogo}")
print(f"  animate_inanimate: absorption_rate={ai_absorption_rate:.4f}, ratio={ai_ratio_s:.3f}, go={ai_go_nogo}")
print(f"  noun_proper:       absorption_rate={np_absorption_rate:.4f}, ratio={np_ratio_s:.3f}, go={np_go_nogo}")
print(f"  Sanity check (first_letter ratio>=1.5): {'PASS' if sanity_pass else ('BUG?' if sanity_warn else 'FAIL')}")
print(f"  Overall: {overall} (n_passing={n_passing}, n_weak={n_weak})")
sys.stdout.flush()

# ── Step 8: Save results ──────────────────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Saving results")

elapsed = time.time() - start_time

results_all = {
    "first_letter": result_first_letter,
    "animate_inanimate": result_ai,
    "noun_proper": result_np,
}

output = {
    "task_id": TASK_ID,
    "task_name": "task_C2_redesign",
    "mode": "PILOT",
    "version": "redesign_v4_icl_direct",
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
        "child_min_act": 0.5,
        "suppression_threshold": 0.30,
        "parent_percentile_of_suppression": "N/A (using ICL context directly)",
        "n_bootstrap": 1000,
        "bonferroni_alpha": bonferroni_alpha,
        "measurement_approach": "ICL-based: compare SAE activations at word position in ICL context vs isolation. Child feature = active in isolation but suppressed (>=30%) in ICL context.",
    },
    "hierarchies": results_all,
    "eda_per_hierarchy": eda_per_hierarchy,
    "summary": {
        "first_letter_absorption_rate": fl_absorption_rate,
        "first_letter_ratio": fl_ratio_s,
        "animate_absorption_rate": ai_absorption_rate,
        "animate_ratio": ai_ratio_s,
        "noun_proper_absorption_rate": np_absorption_rate,
        "noun_proper_ratio": np_ratio_s,
        "n_passing_ratio_1.5": n_passing,
        "n_weak_ratio_1.1": n_weak,
        "overall_go_nogo": overall,
        "bonferroni_alpha": bonferroni_alpha,
        "sanity_check": {
            "criteria": "first_letter ratio >= 1.5",
            "ratio": fl_ratio_s,
            "result": "PASS" if sanity_pass else ("PIPELINE_BUG" if sanity_warn else "FAIL"),
        },
    },
    "pilot_pass_criteria": {
        "criteria": "Ratio-to-null >= 1.5 for first_letter (sanity check). If < 1.0: pipeline bug.",
        "first_letter_ratio": fl_ratio_s,
        "first_letter_result": fl_go_nogo,
        "overall": overall,
    },
    "key_diagnostics": {
        "v4_insight": "ICL context for letter 'a' words: ~23 features active in isolation but silent in ICL, ~21 new features appear in ICL (absorbing parents). This IS the absorption phenomenon.",
        "absorption_definition": "absorption_rate = fraction of words where at least 1 child feature (active in isolation) is suppressed >=30% in ICL context.",
        "suppression_meaning": "if absorption_rate=1.0 and ratio=1.0, null has same rate — context effect not category-specific.",
        "pilot_interpretation": "If ratio_to_null > 1.5 for first_letter, child suppression is category-specific (absorption). If ratio ≈ 1.0, context causes suppression equally for concept and control words.",
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
        "sae_release": "gpt2-small-res-jb",
        "layer": LAYER,
        "method": "ICL-direct-suppression-v4",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mode": "PILOT",
    }
}

gpu_progress_file.write_text(json.dumps(gp, indent=2))

# ── Done ─────────────────────────────────────────────────────────────────────

report_progress(9, TOTAL_STEPS, "Writing DONE marker")

summary = (
    f"C2-REDESIGN v4 PILOT: {overall}. "
    f"first_letter rate={fl_absorption_rate:.3f} ratio={fl_ratio_s:.3f} (sanity {'PASS' if sanity_pass else 'FAIL'}). "
    f"animate rate={ai_absorption_rate:.3f} ratio={ai_ratio_s:.3f}. "
    f"noun_proper rate={np_absorption_rate:.3f} ratio={np_ratio_s:.3f}. "
    f"n_passing={n_passing}/3."
)
print(f"\n{'='*70}")
print(f"SUMMARY: {summary}")
print(f"{'='*70}")
sys.stdout.flush()

mark_done("success", summary)
report_progress(10, TOTAL_STEPS, "Complete")
