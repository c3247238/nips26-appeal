"""
Task C.2-REDESIGN v3: Cross-Domain Absorption via Child-Feature Suppression (PILOT MODE)

CORRECT implementation based on Chanin et al. (sae-spelling) methodology.

Key difference from v1/v2: Uses TASK-RELEVANT CONTEXT (ICL prompts) for first_letter,
which is the domain where absorption is empirically known to occur.

For FIRST_LETTER (sanity check):
- Use ICL spelling prompts (like Chanin et al.)
- Main features = SAE features aligned with letter probe (per C1 probe weights)
- Absorption = main features do NOT fire for word in ICL context
- Sanity check: measured absorption rate should be > 0 (known phenomenon)

For SEMANTIC HIERARCHIES (animate_inanimate, noun_proper):
- Use ACTIVATION COMPARISON approach:
  - For each concept token, compute SAE acts in isolation and in parent-context
  - Child feature = most specifically active feature for token in isolation
  - Parent context = sentence context dominated by parent category words
  - Absorption proxy = child feature activation DECREASES in parent-dominant context
  - Ratio-to-null = ratio vs shuffled control pairing

Output: exp/results/full/C2_child_suppression_absorption.json
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
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v3: ICL+suppression")

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

W_enc = sae.W_enc.detach().float()  # [d_model, d_sae]
W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
d_sae = W_dec.shape[0]
d_model = W_dec.shape[1]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

LAYER = 6
CACHE_KEY = f"blocks.{LAYER}.hook_resid_pre"
print(f"SAE: d_sae={d_sae}, d_model={d_model}")
sys.stdout.flush()

# ── Utility: SAE activations ───────────────────────────────────────────────────

def get_sae_acts_at_pos(texts, position=-2, batch_size=16):
    """Get SAE activations at a specific position for list of texts."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        # Ensure all tokens are same length (needed for position indexing)
        if tokens.shape[1] < abs(position):
            continue
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
        resid = cache[CACHE_KEY][:, position, :].float()
        with torch.no_grad():
            acts = sae.encode(resid)
        all_acts.append(acts.cpu())
    if not all_acts:
        return None
    return torch.cat(all_acts, dim=0)  # [n, d_sae]


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


# ── Step 2: Load probe weights ────────────────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Loading C1 probe weights")

probe_weights_fl = np.load(PROBES_DIR / "probe_first_letter_weights.npy")   # [24, 768]
probe_classes_fl = np.load(PROBES_DIR / "probe_first_letter_classes.npy", allow_pickle=True)
probe_weights_ai = np.load(PROBES_DIR / "probe_animate_inanimate_weights.npy")  # [1, 768]
probe_weights_np = np.load(PROBES_DIR / "probe_noun_proper_weights.npy")    # [1, 768]

probe_weights_fl_t = torch.tensor(probe_weights_fl, dtype=torch.float32)  # [24, 768]
probe_weights_ai_t = torch.tensor(probe_weights_ai[0], dtype=torch.float32)  # [768]
probe_weights_np_t = torch.tensor(probe_weights_np[0], dtype=torch.float32)  # [768]

print(f"First-letter probe: {probe_weights_fl_t.shape}, classes: {probe_classes_fl.tolist()}")
print(f"Animate probe: {probe_weights_ai_t.shape}")
print(f"Proper noun probe: {probe_weights_np_t.shape}")
sys.stdout.flush()

# ── Step 3: Identify letter-specific SAE features (parent = letter probes) ────

report_progress(3, TOTAL_STEPS, "Identifying letter-specific SAE features via probe alignment")

# For each letter, find SAE features most aligned with the letter probe direction
# These are the "main features" in Chanin et al.'s terminology
# These are the PARENT features (letter features) for first_letter hierarchy

TOP_K_LETTER_FEATURES = 5

letter_parent_features = {}  # letter -> [latent_idx, ...]
for i, letter in enumerate(probe_classes_fl):
    probe_dir = probe_weights_fl_t[i]  # [768]
    probe_dir_norm = F.normalize(probe_dir.unsqueeze(0), dim=-1)  # [1, 768]

    # Cosine similarity with decoder rows (W_dec is on CPU)
    cos_sims = F.cosine_similarity(probe_dir_norm.cpu(), W_dec.cpu(), dim=-1)  # [d_sae]
    top_k_idx = cos_sims.topk(TOP_K_LETTER_FEATURES).indices.tolist()

    letter_parent_features[letter] = {
        "latent_indices": top_k_idx,
        "cos_sims": [float(cos_sims[j]) for j in top_k_idx],
    }

print(f"Sample parent features:")
for letter in list(letter_parent_features.keys())[:3]:
    print(f"  '{letter}': latents={letter_parent_features[letter]['latent_indices'][:3]}, "
          f"cos_sims={[f'{c:.3f}' for c in letter_parent_features[letter]['cos_sims'][:3]]}")
sys.stdout.flush()

# ── Step 4: Find child-specific features (token-specific = WORD features) ─────

report_progress(4, TOTAL_STEPS, "Finding word-specific (child) SAE features")


def get_word_specific_features(word, control_words, top_k=3, min_act=0.5):
    """
    Find SAE features that fire specifically for `word` vs. `control_words`.
    These are the CHILD features in the absorption framework.
    """
    word_acts = get_sae_acts_last([word])  # [1, d_sae]
    ctrl_acts = get_sae_acts_last(control_words)  # [n_ctrl, d_sae]

    word_vec = word_acts[0].numpy()
    ctrl_mean = ctrl_acts.mean(dim=0).numpy()
    diff = word_vec - ctrl_mean

    top_idx = np.argsort(diff)[::-1]

    children = []
    for idx in top_idx:
        if len(children) >= top_k:
            break
        if float(word_vec[idx]) >= min_act:
            children.append({
                "latent_idx": int(idx),
                "word_act": float(word_vec[idx]),
                "ctrl_mean_act": float(ctrl_mean[idx]),
                "differential": float(diff[idx]),
            })
    return children


# ── Step 5: FIRST_LETTER measurement (using ICL context) ─────────────────────

report_progress(5, TOTAL_STEPS, "Measuring first_letter absorption (ICL-based)")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter (ICL-based child-feature suppression)")
print("="*70)
sys.stdout.flush()

# ICL word list for spelling (Chanin et al. approach)
icl_word_list = [
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "cat",
    "dog", "run", "sit", "top", "win", "big", "buy", "got", "job", "may",
]

# Test words: starting with letters that have good F1 (>= 0.80)
# Letters: a, b, d, e, g, h, i, j, k, m, n, o, q, r, s, w, y
test_words_fl = {
    "a": ["apple", "angel", "arrow", "album", "alarm", "anchor", "arena", "actor"],
    "b": ["bird", "brain", "bread", "bridge", "brush", "blood", "beach"],
    "s": ["stone", "storm", "smile", "snake", "sheep", "shell", "spark"],
    "m": ["music", "metal", "magic", "model", "maple", "moral", "month"],
    "n": ["night", "novel", "nerve", "noble", "north", "nurse", "noise"],
}

# Control words (don't start with these letters)
control_words_fl = ["table", "chair", "river", "cloud", "flame", "clock", "ocean", "bread"]

def measure_first_letter_absorption_icl(letter, words, parent_feature_indices, icl_words, seed=42):
    """
    Measure absorption using ICL prompt context.

    For each word starting with `letter`:
    1. Find word-specific (child) SAE features
    2. Create ICL spelling prompts (word at position -2)
    3. Measure: do parent (letter) features fire? Do child (word) features fire?
    4. Absorption = parent fires BUT child does NOT fire

    Returns dict with absorption stats.
    """
    rng = random.Random(seed)

    results = []
    n_absorption = 0
    n_total = 0

    for word in words:
        # Find child features
        children = get_word_specific_features(word, control_words_fl, top_k=2, min_act=0.3)
        if not children:
            continue
        child_indices = [c["latent_idx"] for c in children]

        # Create ICL prompt with same template as Chanin et al.
        try:
            spelling_examples = [w for w in icl_words if w != word]
            rng.shuffle(spelling_examples)
            prompt = create_icl_prompt(
                word,
                examples=spelling_examples,
                base_template="{word}:",
                answer_formatter=first_letter_formatter(),
                max_icl_examples=10,
                shuffle_examples=True,
            )
            prompt_text = prompt.base
        except Exception as e:
            print(f"    ICL prompt error for '{word}': {e}")
            continue

        # Get SAE acts at word position (-2 in ICL format: "word: X\n...")
        try:
            acts = get_sae_acts_at_pos([prompt_text], position=-2, batch_size=1)
            if acts is None or acts.shape[0] == 0:
                continue
            acts_vec = acts[0].numpy()  # [d_sae]
        except Exception as e:
            print(f"    Acts error for '{word}': {e}")
            continue

        # Check parent fires
        parent_acts = [float(acts_vec[j]) for j in parent_feature_indices]
        parent_max = max(parent_acts) if parent_acts else 0.0
        parent_fires = parent_max > 0.1  # threshold from sae_spelling EPS

        # Check child fires
        child_acts_vals = [float(acts_vec[j]) for j in child_indices]
        child_max = max(child_acts_vals) if child_acts_vals else 0.0
        child_fires = child_max > 0.1

        # Also get baseline child act (without ICL context — just the word)
        baseline_acts = get_sae_acts_last([word])
        baseline_child_max = float(baseline_acts[0, child_indices].max()) if child_indices else 0.0

        # Absorption = parent fires AND child does NOT fire in ICL context
        # (but child would have fired in baseline)
        is_absorption = parent_fires and not child_fires and baseline_child_max > 0.1

        # Alternative: suppression ratio
        suppression_ratio = None
        if baseline_child_max > 0.01:
            suppression_ratio = (baseline_child_max - child_max) / baseline_child_max

        n_total += 1
        if is_absorption:
            n_absorption += 1

        results.append({
            "word": word,
            "letter": letter,
            "parent_max_act": float(parent_max),
            "parent_fires": bool(parent_fires),
            "child_max_act_icl": float(child_max),
            "child_max_act_baseline": float(baseline_child_max),
            "child_fires_icl": bool(child_fires),
            "is_absorption": bool(is_absorption),
            "suppression_ratio": float(suppression_ratio) if suppression_ratio is not None else None,
        })

    absorption_rate = n_absorption / n_total if n_total > 0 else 0.0
    mean_suppression = (
        float(np.mean([r["suppression_ratio"] for r in results if r["suppression_ratio"] is not None]))
        if any(r["suppression_ratio"] is not None for r in results)
        else None
    )

    return {
        "n_words": n_total,
        "n_absorption": n_absorption,
        "absorption_rate": absorption_rate,
        "mean_suppression_ratio": mean_suppression,
        "results": results,
    }


# Run for multiple letters
fl_results_by_letter = {}
all_fl_triples = []

for letter, words in test_words_fl.items():
    parent_indices = letter_parent_features.get(letter, {}).get("latent_indices", [])
    if not parent_indices:
        print(f"  No parent features for letter '{letter}' — skipping")
        continue

    print(f"\n  Letter '{letter}' ({len(words)} words, {len(parent_indices)} parent features):")
    sys.stdout.flush()

    res = measure_first_letter_absorption_icl(
        letter, words, parent_indices, icl_word_list, seed=SEED
    )

    fl_results_by_letter[letter] = res
    print(f"    n_words={res['n_words']}, absorption_rate={res['absorption_rate']:.3f}, "
          f"mean_suppression={res['mean_suppression_ratio']}")

    # Collect all results for aggregate stats
    all_fl_triples.extend(res["results"])

# Aggregate first_letter results
n_fl_total = sum(r["n_words"] for r in fl_results_by_letter.values())
n_fl_absorption = sum(r["n_absorption"] for r in fl_results_by_letter.values())
fl_absorption_rate = n_fl_absorption / n_fl_total if n_fl_total > 0 else 0.0

# Null: shuffle parent-child assignments
rng_null = np.random.RandomState(SEED)
null_rates_fl = []
fl_is_abs = [r["is_absorption"] for r in all_fl_triples]
for _ in range(100):
    perm = rng_null.permutation(len(fl_is_abs))
    null_rates_fl.append(float(np.mean([fl_is_abs[i] for i in perm])))
fl_null_mean = float(np.mean(null_rates_fl))
fl_ratio = fl_absorption_rate / fl_null_mean if fl_null_mean > 0 else (10.0 if fl_absorption_rate > 0 else 1.0)

# Bootstrap CI
boot_fl = []
for _ in range(1000):
    idx = rng_null.choice(len(fl_is_abs), len(fl_is_abs), replace=True)
    boot_fl.append(float(np.mean([fl_is_abs[i] for i in idx])))
fl_ci_lower = float(np.percentile(boot_fl, 2.5))
fl_ci_upper = float(np.percentile(boot_fl, 97.5))

fl_suppression_ratios = [r["suppression_ratio"] for r in all_fl_triples if r.get("suppression_ratio") is not None]
fl_mean_suppression = float(np.mean(fl_suppression_ratios)) if fl_suppression_ratios else None

fl_go_nogo = "GO" if fl_ratio >= 1.5 else ("WEAK" if fl_ratio >= 1.1 else "NO_GO")

print(f"\n  [first_letter] AGGREGATE:")
print(f"    n_total={n_fl_total}, n_absorbed={n_fl_absorption}, absorption_rate={fl_absorption_rate:.4f}")
print(f"    CI=[{fl_ci_lower:.4f}, {fl_ci_upper:.4f}]")
print(f"    null_mean={fl_null_mean:.4f}, ratio_to_null={fl_ratio:.3f}")
print(f"    mean_suppression_ratio={fl_mean_suppression}")
print(f"    go_nogo={fl_go_nogo}")
sys.stdout.flush()

result_first_letter = {
    "hierarchy": "first_letter",
    "measurement": "ICL-based: parent fires but child does NOT fire in ICL spelling context",
    "n_words": n_fl_total,
    "n_absorbed": n_fl_absorption,
    "absorption_rate": fl_absorption_rate,
    "mean_suppression_ratio": fl_mean_suppression,
    "bootstrap_ci_lower": fl_ci_lower,
    "bootstrap_ci_upper": fl_ci_upper,
    "ci_reliable": n_fl_total >= 10,
    "null_mean": fl_null_mean,
    "ratio_to_null": fl_ratio,
    "go_nogo": fl_go_nogo,
    "results_by_letter": fl_results_by_letter,
    "parent_feature_info": {letter: letter_parent_features[letter] for letter in test_words_fl},
}

# ── Step 6: ANIMATE_INANIMATE measurement (activation comparison) ─────────────

report_progress(6, TOTAL_STEPS, "Measuring animate_inanimate hierarchy")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate (activation comparison method)")
print("="*70)
sys.stdout.flush()

# For semantic hierarchies, we use a different approach:
# For each concept token, get its SAE activations in:
# (A) Isolation context: just the token
# (B) Parent-dominated context: sentence where parent category is implied
# Child suppression = child latent activation decreases from (A) to (B)

animate_concept_words = [
    "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger",
    "wolf", "bear", "eagle", "monkey", "rabbit", "cow", "deer",
]
animate_control_words = [
    "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
    "bottle", "box", "ring", "cloud", "river", "mountain",
]

# Parent-dominated contexts for animate_inanimate:
# These are sentences where "animate" category is strongly implied
animate_parent_contexts = [
    "The animal",
    "This creature",
    "A living being",
    "The organism",
    "This wildlife",
    "A living creature",
    "This animal species",
    "The living thing",
    "A wild creature",
    "The animal kingdom",
]

# Neutral/control contexts (no category implication)
neutral_contexts = [
    "The",
    "A",
    "This",
    "One",
    "That",
    "Here",
    "My",
    "Some",
    "Any",
    "The specific",
]

# Find parent latents for animate
print("  Finding parent latents for 'animate'...")
ai_parent_acts = get_sae_acts_last(animate_concept_words)   # [n_animate, d_sae]
ai_ctrl_acts = get_sae_acts_last(animate_control_words)     # [n_inanimate, d_sae]
ai_diff = ai_parent_acts.mean(dim=0).numpy() - ai_ctrl_acts.mean(dim=0).numpy()
ai_parent_idx = np.argsort(ai_diff)[::-1][:5].tolist()
ai_parent_info = [
    {"latent_idx": int(j), "diff": float(ai_diff[j]),
     "animate_mean": float(ai_parent_acts.mean(dim=0).numpy()[j]),
     "ctrl_mean": float(ai_ctrl_acts.mean(dim=0).numpy()[j])}
    for j in ai_parent_idx
    if float(ai_parent_acts.mean(dim=0).numpy()[j]) >= 0.3
]
print(f"  Found {len(ai_parent_info)} parent latents for animate")
for pl in ai_parent_info[:3]:
    print(f"    latent {pl['latent_idx']}: animate_mean={pl['animate_mean']:.3f}, diff={pl['diff']:.3f}")
sys.stdout.flush()


def measure_semantic_absorption(
    concept_words, control_words, parent_latent_indices,
    parent_context_prefix, neutral_prefix,
    suppression_threshold=0.30, n_bootstrap=1000, seed=42,
):
    """
    Measure absorption for semantic hierarchies using context comparison.

    For each concept word:
    1. Find child-specific latent (fires in isolation vs. control)
    2. Compare: child activation in isolation vs. in parent-dominant context
    3. Suppression = relative decrease >= 30% in parent context

    Null: use control words (no parent-child relationship) instead of concept words
    """
    rng = np.random.RandomState(seed)

    triples = []
    n_skipped_no_child = 0
    n_skipped_inactive = 0

    for word in concept_words:
        # Find child-specific latents
        children = get_word_specific_features(word, control_words, top_k=2, min_act=0.3)
        if not children:
            n_skipped_no_child += 1
            continue

        # Get SAE acts in isolation (baseline)
        baseline_acts = get_sae_acts_last([word])  # [1, d_sae]
        baseline_vec = baseline_acts[0].numpy()

        # Get SAE acts in parent-dominant contexts
        # These contexts end with the concept word
        parent_contexts = [f"{p} {word}" for p in parent_context_prefix]
        neutral_ctx = [f"{p} {word}" for p in neutral_prefix]

        try:
            parent_acts = get_sae_acts_last(parent_contexts)   # [n_ctx, d_sae]
            neutral_acts = get_sae_acts_last(neutral_ctx)      # [n_ctx, d_sae]
        except Exception as e:
            print(f"    Error for '{word}': {e}")
            continue

        for child_info in children:
            child_idx = child_info["latent_idx"]

            # Child activation in different contexts
            child_baseline = float(baseline_vec[child_idx])
            child_in_parent_ctx = float(parent_acts[:, child_idx].mean())
            child_in_neutral_ctx = float(neutral_acts[:, child_idx].mean())

            # Also check parent latent activation in parent vs. neutral contexts
            if parent_latent_indices:
                parent_in_parent_ctx = float(parent_acts[:, parent_latent_indices].max(dim=1).values.mean())
                parent_in_neutral_ctx = float(neutral_acts[:, parent_latent_indices].max(dim=1).values.mean())
            else:
                parent_in_parent_ctx = 0.0
                parent_in_neutral_ctx = 0.0

            # Suppression = child activation lower in parent context vs. neutral context
            reference = child_in_neutral_ctx
            if reference < 0.01:
                n_skipped_inactive += 1
                continue

            suppression_ratio = (reference - child_in_parent_ctx) / reference
            is_suppressed = bool(suppression_ratio >= suppression_threshold)

            triples.append({
                "word": word,
                "child_latent_idx": child_idx,
                "child_act_baseline": child_baseline,
                "child_act_parent_ctx": child_in_parent_ctx,
                "child_act_neutral_ctx": reference,
                "parent_act_parent_ctx": parent_in_parent_ctx,
                "parent_act_neutral_ctx": parent_in_neutral_ctx,
                "suppression_ratio": float(suppression_ratio),
                "is_suppressed": bool(is_suppressed),
            })

    print(f"  Triples: {len(triples)} valid, {n_skipped_no_child} no_child, {n_skipped_inactive} inactive")

    if not triples:
        return {
            "n_triples": 0, "absorption_rate": 0.0, "ratio_to_null": None,
            "go_nogo": "NO_GO", "error": "No valid triples"
        }

    suppressed = np.array([t["is_suppressed"] for t in triples], dtype=float)
    n_triples = len(triples)
    absorption_rate = float(suppressed.mean())
    mean_suppression = float(np.mean([t["suppression_ratio"] for t in triples]))

    # Bootstrap CI
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_triples, n_triples, replace=True)
        boot.append(float(suppressed[idx].mean()))
    ci_lower = float(np.percentile(boot, 2.5))
    ci_upper = float(np.percentile(boot, 97.5))

    # Null: use control words instead of concept words
    ctrl_triples = []
    for word in control_words[:len(concept_words)]:
        children = get_word_specific_features(word, concept_words[:5], top_k=1, min_act=0.3)
        if not children:
            continue
        child_idx = children[0]["latent_idx"]
        baseline_acts = get_sae_acts_last([word])
        child_baseline = float(baseline_acts[0, child_idx])

        parent_contexts = [f"{p} {word}" for p in parent_context_prefix[:5]]
        neutral_ctx = [f"{p} {word}" for p in neutral_prefix[:5]]
        try:
            parent_acts_c = get_sae_acts_last(parent_contexts)
            neutral_acts_c = get_sae_acts_last(neutral_ctx)
        except Exception:
            continue

        child_p = float(parent_acts_c[:, child_idx].mean())
        child_n = float(neutral_acts_c[:, child_idx].mean())
        if child_n < 0.01:
            continue

        sup = (child_n - child_p) / child_n
        ctrl_triples.append({"is_suppressed": bool(sup >= suppression_threshold)})

    if ctrl_triples:
        null_rate = float(np.mean([t["is_suppressed"] for t in ctrl_triples]))
    else:
        null_rate = absorption_rate  # fallback: same as observed (conservative null)

    ratio_to_null = absorption_rate / null_rate if null_rate > 0 else (
        10.0 if absorption_rate > 0 else 1.0
    )
    go_nogo = "GO" if ratio_to_null >= 1.5 else ("WEAK" if ratio_to_null >= 1.1 else "NO_GO")

    return {
        "n_triples": n_triples,
        "n_suppressed": int(suppressed.sum()),
        "absorption_rate": absorption_rate,
        "mean_suppression_ratio": mean_suppression,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "ci_reliable": n_triples >= 10,
        "null_rate": null_rate,
        "ratio_to_null": float(ratio_to_null),
        "go_nogo": go_nogo,
        "triples_sample": triples[:10],
    }


try:
    result_ai = measure_semantic_absorption(
        animate_concept_words, animate_control_words,
        parent_latent_indices=[p["latent_idx"] for p in ai_parent_info[:3]],
        parent_context_prefix=animate_parent_contexts,
        neutral_prefix=neutral_contexts,
        suppression_threshold=0.30, seed=SEED,
    )
    result_ai["hierarchy"] = "animate_inanimate"
    result_ai["parent_latent_info"] = {"animate": ai_parent_info}
except Exception as e:
    import traceback
    print(f"ERROR animate_inanimate: {e}")
    traceback.print_exc()
    result_ai = {"hierarchy": "animate_inanimate", "error": str(e), "go_nogo": "ERROR",
                 "ratio_to_null": None, "absorption_rate": None}

print(f"\n  [animate_inanimate] absorption_rate={result_ai.get('absorption_rate', 'N/A')}, "
      f"ratio_to_null={result_ai.get('ratio_to_null', 'N/A')}, go={result_ai.get('go_nogo', '?')}")
sys.stdout.flush()

# ── Step 7: NOUN_PROPER measurement ──────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Measuring noun_proper hierarchy")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper (activation comparison method)")
print("="*70)
sys.stdout.flush()

proper_concept_words = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
    "Alice", "Robert", "Michael", "Jennifer", "William", "Elizabeth",
]
proper_control_words = [
    "table", "chair", "river", "music", "stone", "flower", "kitchen",
    "bridge", "forest", "garden", "paper", "clock",
]

# Parent-dominated contexts for proper nouns:
# Contexts that strongly imply "named entity / proper noun" category
proper_parent_contexts = [
    "The well-known place called",
    "The famous person named",
    "The capital city known as",
    "The well-known name",
    "The proper name",
    "The named entity",
    "The famous landmark called",
    "The well-known person",
    "The notable place called",
    "The city known as",
]

# Find parent latents for proper_noun
print("  Finding parent latents for 'proper_noun'...")
np_parent_acts = get_sae_acts_last(proper_concept_words)
np_ctrl_acts = get_sae_acts_last(proper_control_words)
np_diff = np_parent_acts.mean(dim=0).numpy() - np_ctrl_acts.mean(dim=0).numpy()
np_parent_idx = np.argsort(np_diff)[::-1][:5].tolist()
np_parent_info = [
    {"latent_idx": int(j), "diff": float(np_diff[j]),
     "proper_mean": float(np_parent_acts.mean(dim=0).numpy()[j]),
     "ctrl_mean": float(np_ctrl_acts.mean(dim=0).numpy()[j])}
    for j in np_parent_idx
    if float(np_parent_acts.mean(dim=0).numpy()[j]) >= 0.3
]
print(f"  Found {len(np_parent_info)} parent latents")
for pl in np_parent_info[:3]:
    print(f"    latent {pl['latent_idx']}: proper_mean={pl['proper_mean']:.3f}, diff={pl['diff']:.3f}")
sys.stdout.flush()

try:
    result_np = measure_semantic_absorption(
        proper_concept_words, proper_control_words,
        parent_latent_indices=[p["latent_idx"] for p in np_parent_info[:3]],
        parent_context_prefix=proper_parent_contexts,
        neutral_prefix=neutral_contexts,
        suppression_threshold=0.30, seed=SEED,
    )
    result_np["hierarchy"] = "noun_proper"
    result_np["parent_latent_info"] = {"proper_noun": np_parent_info}
except Exception as e:
    import traceback
    print(f"ERROR noun_proper: {e}")
    traceback.print_exc()
    result_np = {"hierarchy": "noun_proper", "error": str(e), "go_nogo": "ERROR",
                 "ratio_to_null": None, "absorption_rate": None}

print(f"\n  [noun_proper] absorption_rate={result_np.get('absorption_rate', 'N/A')}, "
      f"ratio_to_null={result_np.get('ratio_to_null', 'N/A')}, go={result_np.get('go_nogo', '?')}")
sys.stdout.flush()

# ── Step 8: Compute EDA for parent latents ────────────────────────────────────

report_progress(8, TOTAL_STEPS, "Computing EDA for parent latents")

def compute_eda_for_latent_list(latent_indices):
    edas = []
    for j in latent_indices:
        enc_j = W_enc[:, j].float()
        dec_j = W_dec[j, :].float()
        cos_sim = F.cosine_similarity(enc_j.unsqueeze(0), dec_j.unsqueeze(0)).item()
        edas.append(1.0 - float(cos_sim))
    return edas


eda_per_hierarchy = {}

# first_letter: parent latents from letter_parent_features
fl_all_parent_idx = []
for letter in test_words_fl.keys():
    fl_all_parent_idx.extend(letter_parent_features.get(letter, {}).get("latent_indices", []))
fl_all_parent_idx = list(set(fl_all_parent_idx))
fl_edas = compute_eda_for_latent_list(fl_all_parent_idx)
eda_per_hierarchy["first_letter"] = {
    "parent_latent_indices": fl_all_parent_idx,
    "eda_values": fl_edas,
    "mean_parent_eda": float(np.mean(fl_edas)) if fl_edas else None,
}

# animate_inanimate
ai_all_idx = [p["latent_idx"] for p in ai_parent_info]
ai_edas = compute_eda_for_latent_list(ai_all_idx)
eda_per_hierarchy["animate_inanimate"] = {
    "parent_latent_indices": ai_all_idx,
    "eda_values": ai_edas,
    "mean_parent_eda": float(np.mean(ai_edas)) if ai_edas else None,
}

# noun_proper
np_all_idx = [p["latent_idx"] for p in np_parent_info]
np_edas = compute_eda_for_latent_list(np_all_idx)
eda_per_hierarchy["noun_proper"] = {
    "parent_latent_indices": np_all_idx,
    "eda_values": np_edas,
    "mean_parent_eda": float(np.mean(np_edas)) if np_edas else None,
}

print("\nEDA per hierarchy:")
for h, ed in eda_per_hierarchy.items():
    print(f"  {h}: n_parents={len(ed['parent_latent_indices'])}, mean_EDA={ed['mean_parent_eda']}")
sys.stdout.flush()

# ── Step 9: Summary ───────────────────────────────────────────────────────────

report_progress(9, TOTAL_STEPS, "Computing overall summary")


def safe_ratio(result):
    r = result.get("ratio_to_null")
    if r is None:
        return 0.0
    if isinstance(r, float) and (np.isnan(r) or np.isinf(r)):
        # inf means null=0 and absorbed>0 → strong signal
        ar = result.get("absorption_rate", 0) or 0.0
        return 10.0 if ar > 0 else 1.0
    return float(r)


fl_ratio_safe = safe_ratio(result_first_letter)
ai_ratio_safe = safe_ratio(result_ai)
np_ratio_safe = safe_ratio(result_np)

n_passing = sum(1 for r in [fl_ratio_safe, ai_ratio_safe, np_ratio_safe] if r >= 1.5)
n_weak = sum(1 for r in [fl_ratio_safe, ai_ratio_safe, np_ratio_safe] if 1.1 <= r < 1.5)

sanity_pass = fl_ratio_safe >= 1.5
sanity_warn = fl_ratio_safe < 1.0

overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if sanity_warn and n_passing == 0:
    overall = "PIPELINE_BUG_INVESTIGATE"

bonferroni_alpha = 0.05 / 3.0

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v3)")
print("="*70)
print(f"  first_letter:      absorption_rate={result_first_letter.get('absorption_rate', 'N/A')}, "
      f"ratio={fl_ratio_safe:.3f}, go={result_first_letter.get('go_nogo', '?')}")
print(f"  animate_inanimate: absorption_rate={result_ai.get('absorption_rate', 'N/A')}, "
      f"ratio={ai_ratio_safe:.3f}, go={result_ai.get('go_nogo', '?')}")
print(f"  noun_proper:       absorption_rate={result_np.get('absorption_rate', 'N/A')}, "
      f"ratio={np_ratio_safe:.3f}, go={result_np.get('go_nogo', '?')}")
print(f"  Sanity check: {'PASS' if sanity_pass else ('BUG?' if sanity_warn else 'FAIL')}")
print(f"  Overall: {overall}")
sys.stdout.flush()

# ── Step 10: AUPRC computation ────────────────────────────────────────────────

report_progress(10, TOTAL_STEPS, "Computing AUPRC")

from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auprc(triples_or_results, key_y="is_suppressed", key_score="suppression_ratio"):
    if not triples_or_results:
        return None, None
    y = [int(r.get(key_y, 0)) for r in triples_or_results]
    s = [r.get(key_score, 0) or 0.0 for r in triples_or_results]
    if len(set(y)) < 2:
        return None, None
    try:
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
        return auroc, auprc
    except Exception:
        return None, None


# ── Step 11: Save results ─────────────────────────────────────────────────────

report_progress(11, TOTAL_STEPS, "Saving results")

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
    "version": "redesign_v3_icl_suppression",
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
        "suppression_threshold": 0.30,
        "n_permutations": 100,
        "n_bootstrap": 1000,
        "bonferroni_alpha": bonferroni_alpha,
        "first_letter_method": "ICL-based: parent fires but child does NOT fire in ICL spelling context",
        "semantic_method": "Activation comparison: child activation in parent-dominated vs neutral context",
    },
    "hierarchies": results_all,
    "eda_per_hierarchy": eda_per_hierarchy,
    "summary": {
        "first_letter_ratio": fl_ratio_safe,
        "animate_inanimate_ratio": ai_ratio_safe,
        "noun_proper_ratio": np_ratio_safe,
        "n_passing_ratio_1.5": n_passing,
        "n_weak_ratio_1.1": n_weak,
        "overall_go_nogo": overall,
        "bonferroni_alpha": bonferroni_alpha,
        "sanity_check_first_letter": {
            "criteria": "ratio_to_null >= 1.5",
            "ratio": fl_ratio_safe,
            "result": "PASS" if sanity_pass else ("PIPELINE_BUG" if sanity_warn else "FAIL"),
        },
    },
    "pilot_pass_criteria": {
        "criteria": "Ratio-to-null >= 1.5 for first_letter (sanity). If < 1.0 for all: pipeline bug.",
        "first_letter_ratio": fl_ratio_safe,
        "first_letter_result": result_first_letter.get("go_nogo"),
        "overall": overall,
    },
    "design_notes": {
        "v1_v2_issues": "v1/v2 used prefix-template contexts; child latents inactive in these contexts → 0 valid triples for first_letter.",
        "v3_first_letter": "Uses ICL spelling prompts (Chanin et al. format). Parent = letter probe-aligned features. Child = word-specific SAE features. Absorption = parent fires, child doesn't.",
        "v3_semantic": "Uses natural-language contexts with parent-category implication vs. neutral. Suppression = child activation drops >= 30% in parent-implied context.",
        "limitation": "Semantic hierarchy measurement is a proxy (not true ICL format). Cross-domain absorption may require different task contexts specific to each hierarchy type.",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nSaved: {OUTPUT_FILE}")
sys.stdout.flush()

# ── Step 12: Done ─────────────────────────────────────────────────────────────

report_progress(12, TOTAL_STEPS, "Updating gpu_progress.json")

import datetime as dt_module

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
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mode": "PILOT",
    }
}

gpu_progress_file.write_text(json.dumps(gp, indent=2))

report_progress(13, TOTAL_STEPS, "Done")

summary = (
    f"C2-REDESIGN v3 PILOT: {overall}. "
    f"first_letter ratio={fl_ratio_safe:.3f} (sanity {'PASS' if sanity_pass else 'FAIL'}). "
    f"animate ratio={ai_ratio_safe:.3f}. noun_proper ratio={np_ratio_safe:.3f}. "
    f"n_passing={n_passing}/3."
)
print(f"\n{'='*70}")
print(f"SUMMARY: {summary}")
print(f"{'='*70}")
sys.stdout.flush()

mark_done("success", summary)
report_progress(14, TOTAL_STEPS, "Complete")
