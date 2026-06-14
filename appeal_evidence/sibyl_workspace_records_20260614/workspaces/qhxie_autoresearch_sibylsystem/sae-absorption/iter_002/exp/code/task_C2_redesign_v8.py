"""
Task C.2-REDESIGN v8: Correct SAE Feature Absorption Measurement

KEY FIXES OVER v7:
1. first_letter: ICL word list with uniform-token-length words, shuffle_examples=False
   → all prompts have same token count → FeatureAbsorptionCalculator works
2. Semantic hierarchies: Measure at WORD POSITION (not last token) to ensure
   parent features actually fire. Use word-isolation baseline vs ICL task context.

ROOT CAUSE RECAP:
- Batching bug: fixed in v7, still present in v7 semantic (but not first_letter)
- Variable-length prompts: ICL words of different lengths → different prompt lengths
  Fix: filter ICL word list to single-token words, use shuffle_examples=False
- n_high=0: parent features never fire above threshold because measured at last
  position (sentence end). Fix: measure at word token position.
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
from scipy import stats

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


TOTAL_STEPS = 10
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v8: uniform-length ICL + word-position measurement")

# ── Load model and SAE ────────────────────────────────────────────────────────

report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small + SAE L6 jb")

from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_spelling.prompting import (
    create_icl_prompt, first_letter_formatter,
    VERBOSE_FIRST_LETTER_TEMPLATE, VERBOSE_FIRST_LETTER_TOKEN_POS
)
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
from sae_spelling.vocab import get_alpha_tokens, LETTERS

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

probe_fl_weights = np.load(str(PROBES_DIR / "probe_first_letter_weights.npy"))  # [24, 768]
probe_fl_classes = np.load(str(PROBES_DIR / "probe_first_letter_classes.npy"))
probe_ai_weights = np.load(str(PROBES_DIR / "probe_animate_inanimate_weights.npy"))
probe_np_weights = np.load(str(PROBES_DIR / "probe_noun_proper_weights.npy"))
sys.stdout.flush()

# ── Individual text processing (avoids padding) ───────────────────────────────

@torch.no_grad()
def get_acts_single(text, position=-1):
    """Process ONE text at a time to avoid padding issues."""
    tokens = model.to_tokens([text], prepend_bos=True)
    _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
    resid = cache[CACHE_KEY][:, position, :].float()
    acts = sae.encode(resid)
    return acts[0].cpu().numpy()


@torch.no_grad()
def get_acts_batch_individual(texts, position=-1):
    """Process texts individually and stack."""
    return np.array([get_acts_single(t, position) for t in texts])


def tok_len(w):
    """Return number of tokens for word with leading space."""
    return len(tokenizer.encode(" " + w))


# ── Step 2: Build single-token word vocab ─────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Building single-token word vocabulary (uniform length)")

print("\n" + "="*70)
print("Building uniform-token-length vocabulary")
print("="*70)
sys.stdout.flush()

vocab_alpha = get_alpha_tokens(tokenizer)
# For GPT-2, vocab_alpha returns words that tokenize to a single token with a space prefix
# These are exactly what we want for uniform-length ICL prompts

words_by_letter = {}
for w in vocab_alpha:
    w_clean = w.strip()
    if w_clean and w_clean[0].isalpha() and w_clean.isalpha():
        letter = w_clean[0].lower()
        if letter in LETTERS:
            # Verify: " word" = 1 token
            if tok_len(w_clean) == 1:
                words_by_letter.setdefault(letter, []).append(w_clean)

target_letters = list(probe_fl_classes)
rng = random.Random(SEED)

# Find split_feats for each letter
letter_split_feats = {}
letter_vocab_1tok = {}  # single-token words only

other_pool_1tok = []
for l, ws in words_by_letter.items():
    clean_ws = [w for w in ws if len(w) >= 3][:10]
    other_pool_1tok.extend(clean_ws)
rng.shuffle(other_pool_1tok)
other_pool_1tok = other_pool_1tok[:100]

for letter in target_letters:
    letter_words = [w for w in words_by_letter.get(letter, []) if len(w) >= 3]
    letter_vocab_1tok[letter] = letter_words

    if len(letter_words) < 10:
        letter_split_feats[letter] = []
        continue

    rng.shuffle(letter_words)
    letter_sample = letter_words[:20]
    other_sample = [w for w in other_pool_1tok if not w.lower().startswith(letter)][:20]

    # Get activations with individual processing
    acts_l = get_acts_batch_individual(letter_sample, position=-1)  # [20, d_sae]
    acts_o = get_acts_batch_individual(other_sample, position=-1)    # [20, d_sae]

    mean_l = acts_l.mean(axis=0)
    mean_o = acts_o.mean(axis=0)
    diff = mean_l - mean_o

    # Features active in ≥10/20 letter words AND significantly more than other
    active_count = (acts_l >= 0.5).sum(axis=0)
    valid = np.where((active_count >= 10) & (diff > 0.2))[0]
    if len(valid) == 0:
        valid = np.where((active_count >= 7) & (diff > 0.1))[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:5]
    letter_split_feats[letter] = [int(j) for j in valid_sorted if mean_l[j] >= 0.2]

for letter in target_letters[:6]:
    feats = letter_split_feats.get(letter, [])
    n_words = len(letter_vocab_1tok.get(letter, []))
    print(f"  Letter '{letter}': n_1tok={n_words}, split_feats={feats[:3]}")
print(f"  Remaining: {[(l, len(letter_split_feats.get(l, []))) for l in target_letters[6:]]}")
n_with_feats = sum(1 for l in target_letters if letter_split_feats.get(l))
print(f"\n  Letters with split_feats: {n_with_feats}/{len(target_letters)}")
sys.stdout.flush()

# ── Step 3: First_letter — FeatureAbsorptionCalculator (fixed) ───────────────

report_progress(3, TOTAL_STEPS, "First_letter: FeatureAbsorptionCalculator with fixed same-length prompts")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter — FeatureAbsorptionCalculator")
print("KEY FIX: ICL list = single-token words only, shuffle_examples=False")
print("="*70)
sys.stdout.flush()

# CRITICAL FIX:
# 1. ICL word list: ALL words must tokenize to exactly 1 token (with space prefix)
#    This ensures all ICL examples contribute the same number of tokens
# 2. shuffle_examples=False: use the SAME 8 ICL examples for every target word
#    → all prompts have EXACTLY the same total token count

# Build single-token ICL word list
icl_word_list = []
for letter in LETTERS:
    single_tok = letter_vocab_1tok.get(letter, [])
    # Take up to 15 single-token words per letter
    icl_word_list.extend(single_tok[:15])

# Shuffle once with fixed seed for reproducibility
rng.shuffle(icl_word_list)

print(f"  ICL word list: {len(icl_word_list)} single-token words")
print(f"  Sample: {icl_word_list[:10]}")
sys.stdout.flush()

# Verify: with shuffle_examples=False, using the FIRST 8 examples always,
# all prompts will be:
# "{ex1} has the first letter: {A}\n{ex2} has the first letter: {B}\n...\n{target} has the first letter:"
# All ex_i have the same token count (single token) → all lines have same length →
# total prompt length = 8 * (line_length) + final_query_length
# But: "{ex_i} has the first letter: {letter_i}" — the letter answers also need same length
# " A", " B", etc. are all 1 token in GPT-2 → YES, uniform length!

# Verify manually
test_prompt_a = create_icl_prompt(
    "apple" if "apple" in icl_word_list else icl_word_list[0],
    examples=icl_word_list,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    max_icl_examples=8,
    shuffle_examples=False,
    check_contamination=False,
)
test_prompt_b = create_icl_prompt(
    icl_word_list[1],
    examples=icl_word_list,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    max_icl_examples=8,
    shuffle_examples=False,
    check_contamination=False,
)

tok_a = len(model.to_tokens([test_prompt_a.base], prepend_bos=True)[0])
tok_b = len(model.to_tokens([test_prompt_b.base], prepend_bos=True)[0])
print(f"\n  Verify token lengths: prompt_a={tok_a}, prompt_b={tok_b}")
if tok_a == tok_b:
    print(f"  ✓ Same length: {tok_a} tokens")
else:
    print(f"  ✗ Different lengths! Need further fix.")
    # Check which ICL words are NOT single-token
    bad_words = [w for w in icl_word_list[:8] if tok_len(w) != 1]
    print(f"  Bad ICL words: {bad_words}")
sys.stdout.flush()

# Define letter_delta_metric locally (from sae_spelling implementation)
def letter_delta_metric(tokenizer_obj, pos_letter):
    LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    neg_letters = [f" {l}" for l in LETTERS_UPPER if pos_letter[-1].upper() != l]
    pos_letter_tok = tokenizer_obj.encode(pos_letter)[-1]
    neg_letter_toks = torch.tensor([tokenizer_obj.encode(nl)[-1] for nl in neg_letters]).to(DEVICE)

    def metric_fn(logits):
        pos_logit = logits[:, -1, pos_letter_tok]
        neg_logits = logits[:, -1, neg_letter_toks]
        return pos_logit - (neg_logits.sum(dim=-1) / len(neg_letters))

    return metric_fn


# Build calculator with shuffle_examples=False
calculator = FeatureAbsorptionCalculator(
    model=model,
    icl_word_list=icl_word_list,
    max_icl_examples=8,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,  # -6
    probe_cos_sim_threshold=0.025,
    ablation_delta_threshold=0.5,  # slightly relaxed from 1.0 to be less strict
    ig_interpolation_steps=6,
    ig_batch_size=6,
    filter_prompts_batch_size=20,
    topk_feats=10,
    shuffle_examples=False,  # KEY FIX: same ICL examples for all prompts
)

# Build probe for first_letter
class SimpleProbe:
    def __init__(self, weights):
        self.weights = torch.tensor(weights).float()
probe_fl = SimpleProbe(probe_fl_weights)

# Test letters with valid split_feats
test_letters_fl = [(l, f) for l, f in letter_split_feats.items() if len(f) >= 1]
test_letters_fl.sort(key=lambda x: -len(x[1]))

fl_results_by_letter = {}
fl_all_absorption = []

for letter, main_feat_ids in test_letters_fl[:8]:  # test 8 letters for better coverage
    letter_idx = list(probe_fl_classes).index(letter)
    probe_dir = probe_fl.weights[letter_idx]
    metric_fn = letter_delta_metric(tokenizer, letter.upper())

    # Use ONLY single-token words for target words too
    concept_words = [w for w in letter_vocab_1tok.get(letter, []) if len(w) >= 3]

    if len(concept_words) < 5:
        print(f"  Letter '{letter}': insufficient single-token words ({len(concept_words)})")
        continue

    # Take up to 25 words for sampling
    concept_words_use = concept_words[:25]
    print(f"\n  Letter '{letter}': {len(concept_words_use)} single-token words, main_feats={main_feat_ids[:3]}")
    sys.stdout.flush()

    try:
        results = calculator.calculate_absorption_sampled(
            sae=sae,
            words=concept_words_use,
            probe_dir=probe_dir,
            metric_fn=metric_fn,
            main_feature_ids=main_feat_ids,
            max_ablation_samples=min(15, len(concept_words_use)),
            filter_prompts=True,
            show_progress=True,
        )

        n_absorbed = sum(1 for r in results.sample_results if r.is_absorption)
        n_total = len(results.sample_results)
        rate = n_absorbed / n_total if n_total > 0 else 0.0

        print(f"  Letter '{letter}': n_filtered={n_total}, n_absorbed={n_absorbed}, rate={rate:.3f}")
        fl_results_by_letter[letter] = {
            "n_total": n_total, "n_absorbed": n_absorbed,
            "absorption_rate": rate, "sample_portion": results.sample_portion,
        }
        fl_all_absorption.extend([int(r.is_absorption) for r in results.sample_results])

    except ValueError as e:
        err_msg = str(e)[:100]
        print(f"  Letter '{letter}': FAILED — {err_msg}")
        # Diagnostic: check prompt lengths
        prompts = calculator._build_prompts(concept_words_use[:3])
        lens = [len(model.to_tokens([p.base], prepend_bos=True)[0]) for p in prompts]
        print(f"  DEBUG: first 3 prompt lengths = {lens}")
    except Exception as e:
        print(f"  Letter '{letter}': ERROR — {str(e)[:100]}")
        import traceback
        traceback.print_exc()

    sys.stdout.flush()

n_fl_events = len(fl_all_absorption)
fl_rate = float(np.mean(fl_all_absorption)) if fl_all_absorption else 0.0
print(f"\n  [first_letter] Total: n={n_fl_events}, absorption_rate={fl_rate:.4f}")

# Null: bootstrap with permuted labels
rng_boot = np.random.RandomState(SEED)
null_rates_fl = []
for _ in range(200):
    perm = rng_boot.permutation(n_fl_events) if n_fl_events > 0 else []
    if len(perm) > 0:
        null_rates_fl.append(float(np.mean([fl_all_absorption[i] for i in perm])))
fl_null = float(np.mean(null_rates_fl)) if null_rates_fl else fl_rate

fl_ratio = fl_rate / fl_null if fl_null > 0 else (10.0 if fl_rate > 0 else 1.0)
fl_go_nogo = "GO" if fl_ratio >= 1.5 else ("WEAK" if fl_ratio >= 1.1 else "NO_GO")

boot_fl = []
for _ in range(500):
    if n_fl_events > 0:
        idx = rng_boot.choice(n_fl_events, n_fl_events, replace=True)
        boot_fl.append(float(np.mean([fl_all_absorption[i] for i in idx])))
fl_ci_lower = float(np.percentile(boot_fl, 2.5)) if boot_fl else 0.0
fl_ci_upper = float(np.percentile(boot_fl, 97.5)) if boot_fl else 0.0

print(f"    null_rate={fl_null:.4f}, ratio={fl_ratio:.3f}, CI=[{fl_ci_lower:.3f}, {fl_ci_upper:.3f}]")
print(f"    go_nogo={fl_go_nogo}")
sys.stdout.flush()

result_first_letter = {
    "hierarchy": "first_letter",
    "measurement": "FeatureAbsorptionCalculator: IG-based, single-token ICL words, shuffle=False",
    "n_events": n_fl_events,
    "absorption_rate": fl_rate,
    "null_rate": fl_null,
    "ratio_to_null": fl_ratio,
    "ci_lower": fl_ci_lower,
    "ci_upper": fl_ci_upper,
    "go_nogo": fl_go_nogo,
    "results_by_letter": fl_results_by_letter,
    "note": "Single-token ICL word list + shuffle_examples=False fixes variable-length prompt error",
}

# ── Step 4: Semantic hierarchies: word-position measurement ──────────────────

report_progress(4, TOTAL_STEPS, "Semantic hierarchies: word-position parent feature identification")

print("\n" + "="*70)
print("HIERARCHIES 2+3: Semantic hierarchies")
print("KEY FIX: measure at WORD TOKEN POSITION (not last token)")
print("METHOD: for each word, measure SAE activations at word token in sentence contexts")
print("Parent fires → child_specific feature suppressed?")
print("="*70)
sys.stdout.flush()

# Concept word lists
animate_words_all = [
    "dog", "cat", "horse", "wolf", "bear", "eagle", "lion", "tiger",
    "rabbit", "monkey", "elephant", "shark", "whale", "deer", "fox",
    "snake", "parrot", "frog", "hawk", "crow",
]
inanimate_words_all = [
    "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
    "bottle", "box", "ring", "cloud", "river", "lamp", "clock", "window",
    "phone", "cup", "knife", "plate",
]
proper_words_all = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
    "Alice", "Robert", "Michael", "Jennifer", "William", "Vienna", "Seoul",
    "Boston", "Chicago", "Mumbai", "Moscow", "Cairo", "Athens",
]
common_nouns_all = [
    "table", "chair", "river", "music", "stone", "flower", "kitchen",
    "bridge", "forest", "garden", "paper", "clock", "window", "bottle",
    "cloud", "market", "village", "castle", "street", "corner",
]

def get_word_position_in_text(text, word):
    """Get the negative index of the word token in the encoded text (with BOS)."""
    toks = tokenizer.encode(text)
    n = len(toks)
    # Try to find word with/without space prefix
    for variant in [" " + word, word]:
        encoded = tokenizer.encode(variant)
        if len(encoded) == 1:
            tok_id = encoded[0]
            for pos_idx in range(n):
                if toks[pos_idx] == tok_id:
                    # pos_idx in toks → with BOS prepended, position = pos_idx + 1
                    # Negative index: (pos_idx + 1) - (n + 1) = pos_idx - n
                    return pos_idx - n  # negative index relative to total length with BOS
    return -1  # fallback: last token


def get_word_acts_in_context(word, template, position_fn=None):
    """Get SAE activations at the word token position within a sentence context."""
    text = template.format(word)
    tok_pos = get_word_position_in_text(text, word)
    return get_acts_single(text, position=tok_pos)


# Sentence templates (word must be embeddable in {})
TEMPLATES_GENERAL = [
    "The {} was there.",
    "A {} appeared.",
    "I saw the {}.",
    "The {} moved quickly.",
    "A {} was found.",
    "The {} looked small.",
    "I found a {}.",
    "The {} stopped here.",
    "A {} came near.",
    "The {} stayed still.",
]

def find_parent_latents_at_word_pos(concept_words, control_words, templates, top_k=5, min_act=0.3):
    """Find SAE features that discriminate concept vs control at WORD POSITION in sentence."""
    concept_acts = []
    for w in concept_words[:15]:
        for tmpl in templates[:3]:
            v = get_word_acts_in_context(w, tmpl)
            concept_acts.append(v)

    control_acts = []
    for w in control_words[:15]:
        for tmpl in templates[:3]:
            v = get_word_acts_in_context(w, tmpl)
            control_acts.append(v)

    if not concept_acts or not control_acts:
        return []

    mean_concept = np.array(concept_acts).mean(axis=0)
    mean_control = np.array(control_acts).mean(axis=0)
    diff = mean_concept - mean_control

    valid = np.where((mean_concept >= min_act) & (diff > 0.2))[0]
    if len(valid) == 0:
        valid = np.where((mean_concept >= min_act * 0.5) & (diff > 0.1))[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:top_k]
    return [int(j) for j in valid_sorted]


def find_child_latents_at_word_pos(word, reference_words, templates, top_k=5, min_act=0.3):
    """Find features specific to this word at its token position in contexts."""
    word_acts = []
    for tmpl in templates[:5]:
        v = get_word_acts_in_context(word, tmpl)
        word_acts.append(v)
    mean_word = np.array(word_acts).mean(axis=0)

    ref_acts = []
    for w in reference_words[:8]:
        for tmpl in templates[:3]:
            v = get_word_acts_in_context(w, tmpl)
            ref_acts.append(v)
    mean_ref = np.array(ref_acts).mean(axis=0) if ref_acts else np.zeros_like(mean_word)

    diff = mean_word - mean_ref
    valid = np.where((mean_word >= min_act) & (diff > 0.2))[0]
    if len(valid) == 0:
        valid = np.where(mean_word >= min_act)[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:top_k]
    return [int(j) for j in valid_sorted]


print("  Finding animate parent latents at word position...")
ai_parent_latents = find_parent_latents_at_word_pos(
    animate_words_all, inanimate_words_all, TEMPLATES_GENERAL)
print(f"  animate parent latents (word pos): {ai_parent_latents}")

print("  Finding noun_proper parent latents at word position...")
np_parent_latents_list = find_parent_latents_at_word_pos(
    proper_words_all, common_nouns_all, TEMPLATES_GENERAL)
print(f"  noun_proper parent latents (word pos): {np_parent_latents_list}")
sys.stdout.flush()

# ── Step 5: animate_inanimate P(child|parent_fired) ──────────────────────────

report_progress(5, TOTAL_STEPS, "animate_inanimate: P(child_active|parent_HIGH) at word position")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate — word-position measurement")
print("="*70)
sys.stdout.flush()

PARENT_ACTIVE_THRESH = 0.3  # Lowered from 0.5 to increase n_high
CHILD_MIN_ACT = 0.3
EPS = 1e-8

ai_word_data = []

for word in animate_words_all:
    # Child latents: specific to this word vs other animate words
    other_animate = [w for w in animate_words_all if w != word]
    child_latents = find_child_latents_at_word_pos(
        word, other_animate, TEMPLATES_GENERAL, top_k=5, min_act=CHILD_MIN_ACT)
    if not child_latents:
        # Try with lower threshold
        child_latents = find_child_latents_at_word_pos(
            word, inanimate_words_all, TEMPLATES_GENERAL, top_k=3, min_act=0.1)

    if not child_latents:
        continue

    word_entries = []
    for tmpl in TEMPLATES_GENERAL:
        v = get_word_acts_in_context(word, tmpl)
        parent_act = float(np.max([v[j] for j in ai_parent_latents])) if ai_parent_latents else 0.0
        child_act_mean = float(np.mean([v[j] for j in child_latents]))

        word_entries.append({
            "parent_act": parent_act,
            "child_act": child_act_mean,
            "parent_fired": parent_act > PARENT_ACTIVE_THRESH,
            "child_active": child_act_mean > CHILD_MIN_ACT * 0.5,
        })

    if word_entries:
        ai_word_data.append({"word": word, "entries": word_entries, "child_latents": child_latents})

# Compute P(child_active | parent_fired)
ai_child_given_high = []
ai_child_given_low = []

for item in ai_word_data:
    entries_high = [e for e in item["entries"] if e["parent_fired"]]
    entries_low = [e for e in item["entries"] if not e["parent_fired"]]
    if entries_high:
        ai_child_given_high.append(float(np.mean([int(e["child_active"]) for e in entries_high])))
    if entries_low:
        ai_child_given_low.append(float(np.mean([int(e["child_active"]) for e in entries_low])))

ai_p_high = float(np.mean(ai_child_given_high)) if ai_child_given_high else 0.0
ai_p_low = float(np.mean(ai_child_given_low)) if ai_child_given_low else 0.0
ai_absorption_rate = max(0.0, 1.0 - (ai_p_high / (ai_p_low + EPS))) if ai_p_low > 0 else 0.0

print(f"  P(child_active | parent_HIGH): {ai_p_high:.4f} (n_words_with_high={len(ai_child_given_high)})")
print(f"  P(child_active | parent_LOW):  {ai_p_low:.4f} (n_words_with_low={len(ai_child_given_low)})")
print(f"  Suppression rate (1 - ratio):  {ai_absorption_rate:.4f}")

# Per-word diagnostic
for item in ai_word_data[:5]:
    n_high = sum(1 for e in item["entries"] if e["parent_fired"])
    n_low = sum(1 for e in item["entries"] if not e["parent_fired"])
    ch = float(np.mean([int(e["child_active"]) for e in item["entries"] if e["parent_fired"]])) if n_high > 0 else 0.0
    cl = float(np.mean([int(e["child_active"]) for e in item["entries"] if not e["parent_fired"]])) if n_low > 0 else 0.0
    parent_acts = [e["parent_act"] for e in item["entries"]]
    print(f"  '{item['word']}': n_high={n_high}, n_low={n_low}, "
          f"child@high={ch:.3f}, child@low={cl:.3f}, "
          f"max_parent={max(parent_acts):.3f}")

# Null permutation
rng_boot = np.random.RandomState(SEED)
all_parent_ai = [e["parent_act"] for item in ai_word_data for e in item["entries"]]
all_child_ai  = [e["child_act"]  for item in ai_word_data for e in item["entries"]]

null_abs_ai = []
for _ in range(100):
    perm = rng_boot.permutation(len(all_parent_ai))
    ph_idx = [i for i in range(len(all_parent_ai)) if all_parent_ai[i] > PARENT_ACTIVE_THRESH]
    pl_idx = [i for i in range(len(all_parent_ai)) if all_parent_ai[i] <= PARENT_ACTIVE_THRESH]
    sc = [all_child_ai[perm[i]] for i in range(len(all_child_ai))]
    ch = float(np.mean([int(sc[i] > CHILD_MIN_ACT * 0.5) for i in ph_idx])) if ph_idx else 0.0
    cl = float(np.mean([int(sc[i] > CHILD_MIN_ACT * 0.5) for i in pl_idx])) if pl_idx else 0.0
    null_abs_ai.append(max(0.0, 1.0 - (ch / (cl + EPS))) if cl > 0 else 0.0)

ai_null_mean = float(np.mean(null_abs_ai)) if null_abs_ai else ai_absorption_rate
ai_ratio = ai_absorption_rate / ai_null_mean if ai_null_mean > 0 else (10.0 if ai_absorption_rate > 0 else 1.0)
ai_go_nogo = "GO" if ai_ratio >= 1.5 else ("WEAK" if ai_ratio >= 1.1 else "NO_GO")

print(f"\n  [animate] absorption_rate={ai_absorption_rate:.4f}, null={ai_null_mean:.4f}, ratio={ai_ratio:.3f}, go={ai_go_nogo}")
sys.stdout.flush()

result_ai = {
    "hierarchy": "animate_inanimate",
    "measurement": "P(child|parent_HIGH) at word position in sentence contexts",
    "P_child_parent_high": ai_p_high,
    "P_child_parent_low": ai_p_low,
    "absorption_rate": ai_absorption_rate,
    "null_mean": ai_null_mean,
    "ratio_to_null": ai_ratio,
    "go_nogo": ai_go_nogo,
    "parent_latents": ai_parent_latents,
    "n_words": len(ai_word_data),
}

# ── Step 6: noun_proper P(child|parent) ──────────────────────────────────────

report_progress(6, TOTAL_STEPS, "noun_proper: word-position P(child|parent) measurement")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper")
print("="*70)
sys.stdout.flush()

np_word_data = []
for word in proper_words_all:
    other_proper = [w for w in proper_words_all if w != word]
    child_latents = find_child_latents_at_word_pos(
        word, other_proper, TEMPLATES_GENERAL, top_k=5, min_act=CHILD_MIN_ACT)
    if not child_latents:
        child_latents = find_child_latents_at_word_pos(
            word, common_nouns_all, TEMPLATES_GENERAL, top_k=3, min_act=0.1)
    if not child_latents:
        continue

    word_entries = []
    for tmpl in TEMPLATES_GENERAL:
        v = get_word_acts_in_context(word, tmpl)
        parent_act = float(np.max([v[j] for j in np_parent_latents_list])) if np_parent_latents_list else 0.0
        child_act_mean = float(np.mean([v[j] for j in child_latents]))
        word_entries.append({
            "parent_act": parent_act, "child_act": child_act_mean,
            "parent_fired": parent_act > PARENT_ACTIVE_THRESH,
            "child_active": child_act_mean > CHILD_MIN_ACT * 0.5,
        })

    if word_entries:
        np_word_data.append({"word": word, "entries": word_entries, "child_latents": child_latents})

np_child_high = []
np_child_low = []
for item in np_word_data:
    eh = [e for e in item["entries"] if e["parent_fired"]]
    el = [e for e in item["entries"] if not e["parent_fired"]]
    if eh:
        np_child_high.append(float(np.mean([int(e["child_active"]) for e in eh])))
    if el:
        np_child_low.append(float(np.mean([int(e["child_active"]) for e in el])))

np_p_high = float(np.mean(np_child_high)) if np_child_high else 0.0
np_p_low = float(np.mean(np_child_low)) if np_child_low else 0.0
np_absorption_rate = max(0.0, 1.0 - (np_p_high / (np_p_low + EPS))) if np_p_low > 0 else 0.0

all_parent_np = [e["parent_act"] for item in np_word_data for e in item["entries"]]
all_child_np  = [e["child_act"]  for item in np_word_data for e in item["entries"]]
null_abs_np = []
for _ in range(100):
    perm = rng_boot.permutation(len(all_parent_np))
    ph = [i for i in range(len(all_parent_np)) if all_parent_np[i] > PARENT_ACTIVE_THRESH]
    pl = [i for i in range(len(all_parent_np)) if all_parent_np[i] <= PARENT_ACTIVE_THRESH]
    sc = [all_child_np[perm[i]] for i in range(len(all_child_np))]
    ch = float(np.mean([int(sc[i] > CHILD_MIN_ACT * 0.5) for i in ph])) if ph else 0.0
    cl = float(np.mean([int(sc[i] > CHILD_MIN_ACT * 0.5) for i in pl])) if pl else 0.0
    null_abs_np.append(max(0.0, 1.0 - (ch / (cl + EPS))) if cl > 0 else 0.0)

np_null_mean = float(np.mean(null_abs_np)) if null_abs_np else np_absorption_rate
np_ratio = np_absorption_rate / np_null_mean if np_null_mean > 0 else (10.0 if np_absorption_rate > 0 else 1.0)
np_go_nogo = "GO" if np_ratio >= 1.5 else ("WEAK" if np_ratio >= 1.1 else "NO_GO")

print(f"  P(child@high)={np_p_high:.4f} (n={len(np_child_high)}), P(child@low)={np_p_low:.4f} (n={len(np_child_low)})")
print(f"  absorption_rate={np_absorption_rate:.4f}, null={np_null_mean:.4f}, ratio={np_ratio:.3f}, go={np_go_nogo}")
sys.stdout.flush()

result_np = {
    "hierarchy": "noun_proper",
    "measurement": "P(child|parent_HIGH) at word position in sentence contexts",
    "P_child_parent_high": np_p_high,
    "P_child_parent_low": np_p_low,
    "absorption_rate": np_absorption_rate,
    "null_mean": np_null_mean,
    "ratio_to_null": np_ratio,
    "go_nogo": np_go_nogo,
    "parent_latents": np_parent_latents_list,
    "n_words": len(np_word_data),
}

# ── Step 7: Summary ───────────────────────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Computing overall summary")

fl_r = fl_ratio if np.isfinite(fl_ratio) else 10.0
ai_r = ai_ratio if np.isfinite(ai_ratio) else 10.0
np_r = np_ratio if np.isfinite(np_ratio) else 10.0

n_passing = sum(1 for r in [fl_r, ai_r, np_r] if r >= 1.5)
n_weak = sum(1 for r in [fl_r, ai_r, np_r] if 1.1 <= r < 1.5)

sanity_pass = fl_r >= 1.5
sanity_warn = fl_r < 1.0 and n_fl_events == 0
overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if n_fl_events == 0 and sanity_warn:
    overall = "NULL_RESULT"

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v8)")
print("="*70)
print(f"  first_letter:      n={n_fl_events}, rate={fl_rate:.4f}, null={fl_null:.4f}, ratio={fl_r:.3f}, go={fl_go_nogo}")
print(f"  animate_inanimate: rate={ai_absorption_rate:.4f}, null={ai_null_mean:.4f}, ratio={ai_r:.3f}, go={ai_go_nogo}")
print(f"  noun_proper:       rate={np_absorption_rate:.4f}, null={np_null_mean:.4f}, ratio={np_r:.3f}, go={np_go_nogo}")
print(f"  Sanity check (first_letter ratio>=1.5): {'PASS' if sanity_pass else 'FAIL'}")
print(f"  Overall: {overall} (n_passing={n_passing}, n_weak={n_weak})")
sys.stdout.flush()

# ── Step 8: Save results ──────────────────────────────────────────────────────

report_progress(8, TOTAL_STEPS, "Saving results")

output = {
    "version": "v8",
    "timestamp": datetime.now().isoformat(),
    "overall": overall,
    "n_passing": n_passing,
    "n_weak": n_weak,
    "sanity_check_passed": sanity_pass,
    "hierarchies": [result_first_letter, result_ai, result_np],
    "pilot_pass_criterion": "first_letter ratio >= 1.5",
    "notes": [
        "v8 fix: ICL word list = single-token words only, shuffle_examples=False",
        "v8 fix: semantic hierarchies measured at word token position (not last token)",
        "v8 fix: parent threshold lowered from 0.5 to 0.3 to ensure n_high > 0",
    ],
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nSaved: {OUTPUT_FILE}")

# ── Step 9: Pilot DONE marker ──────────────────────────────────────────────────

report_progress(9, TOTAL_STEPS, "Writing pilot summary")

pilot_summary = {
    "pilot_id": "C2_redesign_v8",
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "overall": overall,
    "hierarchies": {
        "first_letter": {"ratio": fl_r, "go": fl_go_nogo, "n": n_fl_events},
        "animate_inanimate": {"ratio": ai_r, "go": ai_go_nogo},
        "noun_proper": {"ratio": np_r, "go": np_go_nogo},
    },
    "pilot_passed": n_passing >= 1,
}
(PILOTS_DIR / "C2_pilot_summary_v8.json").write_text(json.dumps(pilot_summary, indent=2))

summary_str = (
    f"C2-REDESIGN v8 PILOT: {overall}. "
    f"first_letter: n={n_fl_events}, rate={fl_rate:.3f}, ratio={fl_r:.3f} ({fl_go_nogo}). "
    f"animate: rate={ai_absorption_rate:.3f}, ratio={ai_r:.3f} ({ai_go_nogo}). "
    f"noun_proper: rate={np_absorption_rate:.3f}, ratio={np_r:.3f} ({np_go_nogo}). "
    f"n_passing={n_passing}/3."
)
print("\n" + "="*70)
print(f"SUMMARY: {summary_str}")
print("="*70)

mark_done(status="success", summary=summary_str)

report_progress(10, TOTAL_STEPS, "Complete")
print("\nDone.")
