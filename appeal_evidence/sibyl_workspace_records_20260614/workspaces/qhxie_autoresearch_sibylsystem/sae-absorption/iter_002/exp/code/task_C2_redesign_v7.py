"""
Task C.2-REDESIGN v7: Cross-Domain Absorption via Child-Feature Suppression (PILOT MODE)

CRITICAL BUG FIXED: model.to_tokens() with batch creates padding → position=-1 hits PAD token.
Must process each text individually.

FINDINGS FROM v1-v6 DIAGNOSTICS:
1. Letter-specific features (e.g., L11446 for 'a') DO fire in isolation
2. They get suppressed in ANY ICL context (general context shift, not letter-specific)
3. The true absorption signal requires IG to find when another probe-aligned feature takes over
4. FeatureAbsorptionCalculator requires same-length prompts

FINAL APPROACH:
A. For first_letter SANITY CHECK:
   - Find letter-specific split_feats (active in ≥7/10 words starting with that letter)
   - Use FeatureAbsorptionCalculator with same-length words (filter by token length)
   - This gives validated absorption rate comparable to Chanin et al.

B. For semantic hierarchies (cross-domain extension):
   - Use natural sentence contexts where the semantic parent feature fires strongly
   - Compare child (token-specific) activation when parent fires vs when it doesn't
   - This is the correct P(child_active|parent_fired) vs P(child_active|parent_not_fired) approach
   - Null: permute parent-child assignments (100 permutations)
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
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v7: corrected individual processing")

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

# ── CRITICAL FIX: Individual text processing ──────────────────────────────────

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
    """Process texts individually and stack — avoids padding bugs."""
    results = []
    for text in texts:
        v = get_acts_single(text, position)
        results.append(v)
    return np.array(results)


# ── Step 2: Find letter-specific split_feats (individual processing) ──────────

report_progress(2, TOTAL_STEPS, "Finding letter-specific split_feats (corrected)")

print("\n" + "="*70)
print("Finding SAE features specific to each letter (corrected individual processing)")
print("="*70)
sys.stdout.flush()

vocab_alpha = get_alpha_tokens(tokenizer)
words_by_letter = {}
for w in vocab_alpha:
    w_clean = w.strip()
    if w_clean and w_clean[0].isalpha():
        letter = w_clean[0].lower()
        if letter in LETTERS:
            words_by_letter.setdefault(letter, []).append(w_clean)

target_letters = list(probe_fl_classes)  # 24 letters
rng = random.Random(SEED)

# Find split_feats for each letter using individual processing
letter_split_feats = {}
letter_vocab = {}

other_pool = []
for l, ws in words_by_letter.items():
    clean_ws = [w for w in ws if w.isalpha() and 3 <= len(w) <= 7][:10]
    other_pool.extend(clean_ws)
rng.shuffle(other_pool)
other_pool = other_pool[:100]

for letter in target_letters:
    letter_words = [w for w in words_by_letter.get(letter, []) if w.isalpha() and 3 <= len(w) <= 7]
    if len(letter_words) < 10:
        letter_split_feats[letter] = []
        letter_vocab[letter] = letter_words
        continue

    rng.shuffle(letter_words)
    letter_sample = letter_words[:20]
    letter_vocab[letter] = letter_words[:200]

    other_sample = [w for w in other_pool if not w.lower().startswith(letter)][:20]

    # Individual processing
    acts_l = get_acts_batch_individual(letter_sample, position=-1)  # [20, d_sae]
    acts_o = get_acts_batch_individual(other_sample, position=-1)    # [20, d_sae]

    mean_l = acts_l.mean(axis=0)
    mean_o = acts_o.mean(axis=0)
    diff = mean_l - mean_o

    # Find consistently active features
    thresh = 0.5
    active_count = (acts_l >= thresh).sum(axis=0)  # [d_sae]

    # Features active in ≥10/20 letter words AND significantly more than other
    valid = np.where((active_count >= 10) & (diff > 0.2))[0]
    if len(valid) == 0:
        # Relax threshold
        valid = np.where((active_count >= 7) & (diff > 0.1))[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:5]
    letter_split_feats[letter] = [int(j) for j in valid_sorted if mean_l[j] >= 0.2]

sys.stdout.flush()

# Print split_feats
for letter in target_letters[:6]:
    feats = letter_split_feats.get(letter, [])
    print(f"  Letter '{letter}': split_feats={feats[:3]}")
print(f"  Remaining letters: {[(l, len(letter_split_feats.get(l, []))) for l in target_letters[6:]]}")
n_with_feats = sum(1 for l in target_letters if letter_split_feats.get(l))
print(f"\n  Total letters with split_feats: {n_with_feats}/{len(target_letters)}")
sys.stdout.flush()

# ── Step 3: FIRST_LETTER — FeatureAbsorptionCalculator with same-length filtering ──

report_progress(3, TOTAL_STEPS, "First_letter: FeatureAbsorptionCalculator (same-length prompts)")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter (FeatureAbsorptionCalculator)")
print("="*70)
sys.stdout.flush()

# Define letter_delta_metric locally
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


# Build ICL word list (for the calculator)
icl_word_list = []
for letter in LETTERS:
    icl_words = [w for w in letter_vocab.get(letter, []) if w.isalpha() and 2 <= len(w) <= 6][:15]
    icl_word_list.extend(icl_words)
rng.shuffle(icl_word_list)
print(f"  ICL word list: {len(icl_word_list)} words")

calculator = FeatureAbsorptionCalculator(
    model=model,
    icl_word_list=icl_word_list,
    max_icl_examples=8,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
    probe_cos_sim_threshold=0.025,
    ablation_delta_threshold=1.0,
    ig_interpolation_steps=6,
    ig_batch_size=6,
    filter_prompts_batch_size=20,
    topk_feats=10,
    shuffle_examples=True,
)

# Build probe for first_letter
class SimpleProbe:
    def __init__(self, weights):
        self.weights = torch.tensor(weights).float()
probe_fl = SimpleProbe(probe_fl_weights)

# Test letters with valid split_feats and run FeatureAbsorptionCalculator
test_letters = [(l, f) for l, f in letter_split_feats.items() if len(f) >= 1]
test_letters.sort(key=lambda x: -len(x[1]))

fl_results_by_letter = {}
fl_all_absorption = []

for letter, main_feat_ids in test_letters[:5]:  # test 5 letters
    letter_idx = list(probe_fl_classes).index(letter)
    probe_dir = probe_fl.weights[letter_idx]
    metric_fn = letter_delta_metric(tokenizer, letter.upper())

    concept_words = [w for w in letter_vocab.get(letter, []) if w.isalpha() and 2 <= len(w) <= 6]
    # Filter to single-token words only (avoids variable-length token issue)
    single_tok_words = [w for w in concept_words
                        if len(tokenizer.encode(w)) == 1 or
                           len(tokenizer.encode(" " + w)) == 1][:40]

    if len(single_tok_words) < 5:
        print(f"  Letter '{letter}': insufficient single-token words ({len(single_tok_words)})")
        continue

    print(f"\n  Letter '{letter}': {len(single_tok_words)} single-token words, main_feats={main_feat_ids[:3]}")
    sys.stdout.flush()

    try:
        results = calculator.calculate_absorption_sampled(
            sae=sae,
            words=single_tok_words,
            probe_dir=probe_dir,
            metric_fn=metric_fn,
            main_feature_ids=main_feat_ids,
            max_ablation_samples=min(15, len(single_tok_words)),
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
        print(f"  Letter '{letter}': Variable-length error — {str(e)[:80]}")
        # Try with words of fixed length
        fixed_len_words = [w for w in single_tok_words if len(tokenizer.encode(w)) == 1][:20]
        if len(fixed_len_words) >= 5:
            try:
                results2 = calculator.calculate_absorption_sampled(
                    sae=sae, words=fixed_len_words,
                    probe_dir=probe_dir, metric_fn=metric_fn,
                    main_feature_ids=main_feat_ids,
                    max_ablation_samples=min(10, len(fixed_len_words)),
                    filter_prompts=True, show_progress=False,
                )
                n_absorbed2 = sum(1 for r in results2.sample_results if r.is_absorption)
                n_total2 = len(results2.sample_results)
                rate2 = n_absorbed2 / n_total2 if n_total2 > 0 else 0.0
                print(f"  Letter '{letter}' (fixed): n={n_total2}, absorbed={n_absorbed2}, rate={rate2:.3f}")
                fl_results_by_letter[letter] = {
                    "n_total": n_total2, "n_absorbed": n_absorbed2,
                    "absorption_rate": rate2, "sample_portion": results2.sample_portion,
                }
                fl_all_absorption.extend([int(r.is_absorption) for r in results2.sample_results])
            except Exception as e2:
                print(f"  Letter '{letter}' (fixed): STILL ERROR — {str(e2)[:80]}")
    except Exception as e:
        print(f"  Letter '{letter}': ERROR — {str(e)[:100]}")
        import traceback
        traceback.print_exc()

    sys.stdout.flush()

n_fl_events = len(fl_all_absorption)
fl_rate = float(np.mean(fl_all_absorption)) if fl_all_absorption else 0.0
print(f"\n  [first_letter] Total: n={n_fl_events}, absorption_rate={fl_rate:.4f}")

# Null: shuffle letter assignments across all results
rng_boot = np.random.RandomState(SEED)
null_rates_fl = []
for _ in range(100):
    perm = rng_boot.permutation(n_fl_events) if n_fl_events > 0 else []
    if len(perm) > 0:
        null_rates_fl.append(float(np.mean([fl_all_absorption[i] for i in perm])))
fl_null = float(np.mean(null_rates_fl)) if null_rates_fl else fl_rate

fl_ratio = fl_rate / fl_null if fl_null > 0 else (10.0 if fl_rate > 0 else 1.0)
fl_go_nogo = "GO" if fl_ratio >= 1.5 else ("WEAK" if fl_ratio >= 1.1 else "NO_GO")

# Bootstrap CI on rate
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
    "measurement": "FeatureAbsorptionCalculator: IG-based absorption detection (corrected)",
    "n_events": n_fl_events,
    "absorption_rate": fl_rate,
    "null_rate": fl_null,
    "ratio_to_null": fl_ratio,
    "ci_lower": fl_ci_lower,
    "ci_upper": fl_ci_upper,
    "go_nogo": fl_go_nogo,
    "results_by_letter": fl_results_by_letter,
    "note": "Uses FeatureAbsorptionCalculator with empirical split_feats (corrected individual processing)",
}

# ── Step 4: SEMANTIC HIERARCHIES — corpus-based P(child|parent) ───────────────

report_progress(4, TOTAL_STEPS, "Semantic hierarchies: corpus-based conditional activation")

print("\n" + "="*70)
print("HIERARCHIES 2+3: Semantic hierarchies (corpus-based approach)")
print("P(child_active | parent_fired) vs P(child_active | parent_not_fired)")
print("="*70)
sys.stdout.flush()

# For semantic hierarchies, we use natural sentence contexts.
# We construct short sentences with the concept word and check:
# - Parent activation (category-level SAE feature)
# - Child activation (word-specific SAE feature)
# Corpus: simple sentence templates with concept words

# Find parent latents for animate/noun_proper empirically
probe_ai_dir = torch.tensor(probe_ai_weights[0]).float()  # [768]
probe_np_dir = torch.tensor(probe_np_weights[0]).float()  # [768]

# Sentence templates for concept words
TEMPLATES_ANIMATE = [
    "The {} ran across the field.",
    "A {} was seen in the garden.",
    "The {} looked up at the sky.",
    "She saw a {} near the house.",
    "The {} moved quietly through the forest.",
]
TEMPLATES_INANIMATE = [
    "The {} was left on the table.",
    "A {} was found near the door.",
    "The {} broke into pieces.",
    "She found a {} in the drawer.",
    "The {} was placed on the shelf.",
]
TEMPLATES_PROPER = [
    "{} is a city in Europe.",
    "{} is known for its culture.",
    "She visited {} last summer.",
    "He was born in {}.",
    "{} is a popular destination.",
]
TEMPLATES_COMMON = [
    "The {} was on the table.",
    "A {} was near the door.",
    "The {} broke yesterday.",
    "She found a {} in the drawer.",
    "The {} was heavy.",
]

animate_words_all = [
    "dog", "cat", "horse", "wolf", "bear", "eagle", "lion", "tiger",
    "rabbit", "monkey", "elephant", "shark", "whale", "deer", "fox", "owl",
    "snake", "fish", "parrot", "frog",
]
inanimate_words_all = [
    "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
    "bottle", "box", "ring", "cloud", "river", "lamp", "clock", "window",
    "phone", "cup", "knife", "plate",
]
proper_words_all = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
    "Alice", "Robert", "Michael", "Jennifer", "William", "Vienna", "Seoul",
    "Toronto", "Boston", "Chicago", "Mumbai", "Moscow", "Cairo",
]
common_words_all = [
    "table", "chair", "river", "music", "stone", "flower", "kitchen",
    "bridge", "forest", "garden", "paper", "clock", "window", "bottle",
    "cloud", "market", "garden", "forest", "village", "castle",
]


def find_word_parent_latent(concept_words, control_words, top_k=5, min_act=0.2):
    """
    Find SAE features that discriminate concept from control in natural contexts.
    Process individually to avoid padding bug.
    """
    all_concept_acts = []
    for w in concept_words[:15]:
        tmpl = TEMPLATES_ANIMATE[0]
        text = tmpl.format(w)
        v = get_acts_single(text, position=-1)  # last token (period)
        all_concept_acts.append(v)

    all_control_acts = []
    for w in control_words[:15]:
        tmpl = TEMPLATES_ANIMATE[0]
        text = tmpl.format(w)
        v = get_acts_single(text, position=-1)
        all_control_acts.append(v)

    if not all_concept_acts or not all_control_acts:
        return []

    mean_concept = np.array(all_concept_acts).mean(axis=0)
    mean_control = np.array(all_control_acts).mean(axis=0)
    diff = mean_concept - mean_control

    valid = np.where((mean_concept >= min_act) & (diff > 0.1))[0]
    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:top_k]
    return [int(j) for j in valid_sorted]


def find_word_child_latent(word, reference_words, top_k=5, min_act=0.3):
    """
    Find SAE features specific to this word (using individual processing).
    Process word in isolation and compare with reference words.
    """
    v_word = get_acts_single(word, position=-1)

    ref_acts = [get_acts_single(w, position=-1) for w in reference_words[:15]]
    mean_ref = np.array(ref_acts).mean(axis=0) if ref_acts else np.zeros_like(v_word)

    diff = v_word - mean_ref
    valid = np.where((v_word >= min_act) & (diff > 0.1))[0]
    if len(valid) == 0:
        valid = np.where(v_word >= min_act)[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:top_k]
    return [int(j) for j in valid_sorted]


# Find parent latents for animate and noun_proper
print("  Finding parent latents for animate_inanimate...")
ai_parent_latents = find_word_parent_latent(animate_words_all, inanimate_words_all)
print(f"  animate parent latents: {ai_parent_latents}")

print("  Finding parent latents for noun_proper...")
np_parent_latents_list = find_word_parent_latent(proper_words_all, common_words_all)
print(f"  noun_proper parent latents: {np_parent_latents_list}")
sys.stdout.flush()

# ── Step 5: Animate P(child|parent_fired) ────────────────────────────────────

report_progress(5, TOTAL_STEPS, "animate_inanimate: P(child_active|parent) computation")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate — P(child|parent_fired) vs P(child|parent_not_fired)")
print("="*70)
sys.stdout.flush()

EPS = 1e-8
CHILD_MIN_ACT = 0.3
PARENT_ACTIVE_THRESH = 0.5

ai_word_data = []  # For each word: list of (parent_act, child_act) over templates

for word in animate_words_all:
    # Find child latents for this specific word
    child_latents = find_word_child_latent(word, inanimate_words_all, top_k=5, min_act=CHILD_MIN_ACT)
    if not child_latents:
        continue

    word_entries = []
    templates_for_word = TEMPLATES_ANIMATE[:5] + TEMPLATES_INANIMATE[:5]  # mix for variability
    for tmpl in templates_for_word:
        text = tmpl.format(word)
        # Get activation at the WORD position (not last)
        toks = tokenizer.encode(text)
        # Find position of the word token
        word_tok = tokenizer.encode(" " + word)[-1]
        try:
            pos = toks.index(word_tok)
            # pos is 0-indexed in the encoded sequence, but model adds BOS
            # so actual position in model is pos+1 (with BOS at 0)
            # In terms of negative indexing: pos - (len(toks)) -- careful
            tok_pos = pos - len(toks)  # negative index (BOS not in toks)
        except ValueError:
            # word might tokenize differently
            tok_pos = -1  # fallback to last
            pos = len(toks) - 1

        v = get_acts_single(text, position=tok_pos)

        parent_act = float(np.mean([v[j] for j in ai_parent_latents])) if ai_parent_latents else 0.0
        child_act_mean = float(np.mean([v[j] for j in child_latents]))

        word_entries.append({
            "parent_act": parent_act,
            "child_act": child_act_mean,
            "parent_fired": parent_act > PARENT_ACTIVE_THRESH,
            "child_active": child_act_mean > CHILD_MIN_ACT * 0.5,
        })

    if word_entries:
        ai_word_data.append({"word": word, "entries": word_entries, "child_latents": child_latents})

# Compute P(child_active | parent_fired) and P(child_active | parent_not_fired)
ai_child_given_parent_high = []
ai_child_given_parent_low = []

for item in ai_word_data:
    entries_high = [e for e in item["entries"] if e["parent_fired"]]
    entries_low = [e for e in item["entries"] if not e["parent_fired"]]

    if entries_high:
        rate_high = float(np.mean([int(e["child_active"]) for e in entries_high]))
        ai_child_given_parent_high.append(rate_high)

    if entries_low:
        rate_low = float(np.mean([int(e["child_active"]) for e in entries_low]))
        ai_child_given_parent_low.append(rate_low)

ai_p_child_parent_high = float(np.mean(ai_child_given_parent_high)) if ai_child_given_parent_high else 0.0
ai_p_child_parent_low = float(np.mean(ai_child_given_parent_low)) if ai_child_given_parent_low else 0.0

# Absorption = child is suppressed when parent fires
ai_absorption_rate = 1.0 - (ai_p_child_parent_high / (ai_p_child_parent_low + EPS)) if ai_p_child_parent_low > 0 else 0.0
ai_absorption_rate = max(0.0, ai_absorption_rate)

print(f"  P(child_active | parent_HIGH): {ai_p_child_parent_high:.4f}")
print(f"  P(child_active | parent_LOW):  {ai_p_child_parent_low:.4f}")
print(f"  Suppression rate (1 - ratio):  {ai_absorption_rate:.4f}")

# Also print per-word summary
for item in ai_word_data[:5]:
    entries = item["entries"]
    n_high = sum(1 for e in entries if e["parent_fired"])
    n_low = sum(1 for e in entries if not e["parent_fired"])
    child_high = float(np.mean([int(e["child_active"]) for e in entries if e["parent_fired"]])) if n_high > 0 else 0.0
    child_low = float(np.mean([int(e["child_active"]) for e in entries if not e["parent_fired"]])) if n_low > 0 else 0.0
    print(f"  '{item['word']}': n_high={n_high}, n_low={n_low}, "
          f"child@high={child_high:.3f}, child@low={child_low:.3f}")

# Null: permute parent-child assignments
null_absorption_rates_ai = []
all_parent_acts = [e["parent_act"] for item in ai_word_data for e in item["entries"]]
all_child_acts = [e["child_act"] for item in ai_word_data for e in item["entries"]]

for _ in range(100):
    perm = rng_boot.permutation(len(all_parent_acts))
    parent_high_idx = [i for i in range(len(all_parent_acts)) if all_parent_acts[i] > PARENT_ACTIVE_THRESH]
    parent_low_idx = [i for i in range(len(all_parent_acts)) if all_parent_acts[i] <= PARENT_ACTIVE_THRESH]

    # Shuffle child assignments
    shuffled_child = [all_child_acts[perm[i]] for i in range(len(all_child_acts))]
    child_null_high = float(np.mean([int(shuffled_child[i] > CHILD_MIN_ACT * 0.5) for i in parent_high_idx])) if parent_high_idx else 0.0
    child_null_low = float(np.mean([int(shuffled_child[i] > CHILD_MIN_ACT * 0.5) for i in parent_low_idx])) if parent_low_idx else 0.0

    null_abso = max(0.0, 1.0 - (child_null_high / (child_null_low + EPS))) if child_null_low > 0 else 0.0
    null_absorption_rates_ai.append(null_abso)

ai_null_mean = float(np.mean(null_absorption_rates_ai)) if null_absorption_rates_ai else ai_absorption_rate

ai_ratio = ai_absorption_rate / ai_null_mean if ai_null_mean > 0 else (10.0 if ai_absorption_rate > 0 else 1.0)
ai_go_nogo = "GO" if ai_ratio >= 1.5 else ("WEAK" if ai_ratio >= 1.1 else "NO_GO")

print(f"\n  [animate_inanimate] absorption_rate={ai_absorption_rate:.4f}, null={ai_null_mean:.4f}, ratio={ai_ratio:.3f}, go={ai_go_nogo}")
sys.stdout.flush()

result_ai = {
    "hierarchy": "animate_inanimate",
    "measurement": "P(child_active|parent_HIGH) vs P(child_active|parent_LOW) in natural sentence contexts",
    "P_child_parent_high": ai_p_child_parent_high,
    "P_child_parent_low": ai_p_child_parent_low,
    "absorption_rate": ai_absorption_rate,
    "null_mean": ai_null_mean,
    "ratio_to_null": ai_ratio,
    "go_nogo": ai_go_nogo,
    "parent_latents": ai_parent_latents,
    "n_words": len(ai_word_data),
}

# ── Step 6: NOUN_PROPER P(child|parent) ──────────────────────────────────────

report_progress(6, TOTAL_STEPS, "noun_proper: P(child_active|parent) computation")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper")
print("="*70)
sys.stdout.flush()

np_word_data = []

for word in proper_words_all:
    child_latents = find_word_child_latent(word, common_words_all, top_k=5, min_act=CHILD_MIN_ACT)
    if not child_latents:
        continue

    word_entries = []
    templates_for_word = TEMPLATES_PROPER[:5] + TEMPLATES_COMMON[:5]
    for tmpl in templates_for_word:
        text = tmpl.format(word)
        toks = tokenizer.encode(text)
        # Find word position
        word_tok_candidates = [tokenizer.encode(" " + word)[-1], tokenizer.encode(word)[-1]]
        tok_pos = -1
        for word_tok in word_tok_candidates:
            try:
                pos = toks.index(word_tok)
                tok_pos = pos - len(toks)
                break
            except ValueError:
                continue

        v = get_acts_single(text, position=tok_pos)

        parent_act = float(np.mean([v[j] for j in np_parent_latents_list])) if np_parent_latents_list else 0.0
        child_act_mean = float(np.mean([v[j] for j in child_latents]))

        word_entries.append({
            "parent_act": parent_act,
            "child_act": child_act_mean,
            "parent_fired": parent_act > PARENT_ACTIVE_THRESH,
            "child_active": child_act_mean > CHILD_MIN_ACT * 0.5,
        })

    if word_entries:
        np_word_data.append({"word": word, "entries": word_entries, "child_latents": child_latents})

# Compute P(child_active | parent)
np_child_given_high = []
np_child_given_low = []

for item in np_word_data:
    entries_high = [e for e in item["entries"] if e["parent_fired"]]
    entries_low = [e for e in item["entries"] if not e["parent_fired"]]

    if entries_high:
        np_child_given_high.append(float(np.mean([int(e["child_active"]) for e in entries_high])))
    if entries_low:
        np_child_given_low.append(float(np.mean([int(e["child_active"]) for e in entries_low])))

np_p_high = float(np.mean(np_child_given_high)) if np_child_given_high else 0.0
np_p_low = float(np.mean(np_child_given_low)) if np_child_given_low else 0.0

np_absorption_rate = max(0.0, 1.0 - (np_p_high / (np_p_low + EPS))) if np_p_low > 0 else 0.0

# Null
all_parent_acts_np = [e["parent_act"] for item in np_word_data for e in item["entries"]]
all_child_acts_np = [e["child_act"] for item in np_word_data for e in item["entries"]]
null_abs_np = []
for _ in range(100):
    perm = rng_boot.permutation(len(all_parent_acts_np))
    parent_high_idx = [i for i in range(len(all_parent_acts_np)) if all_parent_acts_np[i] > PARENT_ACTIVE_THRESH]
    parent_low_idx = [i for i in range(len(all_parent_acts_np)) if all_parent_acts_np[i] <= PARENT_ACTIVE_THRESH]
    shuffled_child = [all_child_acts_np[perm[i]] for i in range(len(all_child_acts_np))]
    ch = float(np.mean([int(shuffled_child[i] > CHILD_MIN_ACT * 0.5) for i in parent_high_idx])) if parent_high_idx else 0.0
    cl = float(np.mean([int(shuffled_child[i] > CHILD_MIN_ACT * 0.5) for i in parent_low_idx])) if parent_low_idx else 0.0
    null_abs_np.append(max(0.0, 1.0 - (ch / (cl + EPS))) if cl > 0 else 0.0)

np_null_mean = float(np.mean(null_abs_np)) if null_abs_np else np_absorption_rate
np_ratio = np_absorption_rate / np_null_mean if np_null_mean > 0 else (10.0 if np_absorption_rate > 0 else 1.0)
np_go_nogo = "GO" if np_ratio >= 1.5 else ("WEAK" if np_ratio >= 1.1 else "NO_GO")

print(f"  P(child@high)={np_p_high:.4f}, P(child@low)={np_p_low:.4f}")
print(f"  absorption_rate={np_absorption_rate:.4f}, null={np_null_mean:.4f}, ratio={np_ratio:.3f}, go={np_go_nogo}")
sys.stdout.flush()

result_np = {
    "hierarchy": "noun_proper",
    "measurement": "P(child_active|parent_HIGH) vs P(child_active|parent_LOW) in natural contexts",
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

fl_r = fl_ratio
ai_r = ai_ratio
np_r = np_ratio

# Handle inf/nan
for r_name, r_val in [("fl", fl_r), ("ai", ai_r), ("np", np_r)]:
    if not np.isfinite(r_val):
        r_val = 10.0

n_passing = sum(1 for r in [fl_r, ai_r, np_r] if r >= 1.5)
n_weak = sum(1 for r in [fl_r, ai_r, np_r] if 1.1 <= r < 1.5)

sanity_pass = fl_r >= 1.5
sanity_warn = fl_r < 1.0
overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if sanity_warn and n_passing == 0:
    overall = "NULL_RESULT"  # scoped null result, not pipeline bug

bonferroni_alpha = 0.05 / 3.0

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v7)")
print("="*70)
print(f"  first_letter:      rate={fl_rate:.4f}, null={fl_null:.4f}, ratio={fl_r:.3f}, go={fl_go_nogo}")
print(f"  animate_inanimate: supp={ai_absorption_rate:.4f}, null={ai_null_mean:.4f}, ratio={ai_r:.3f}, go={ai_go_nogo}")
print(f"  noun_proper:       supp={np_absorption_rate:.4f}, null={np_null_mean:.4f}, ratio={np_r:.3f}, go={np_go_nogo}")
print(f"  Sanity check (first_letter ratio>=1.5): {'PASS' if sanity_pass else 'FAIL'}")
print(f"  Overall: {overall} (n_passing={n_passing}, n_weak={n_weak})")
sys.stdout.flush()

# ── Step 8: Save results ──────────────────────────────────────────────────────

report_progress(8, TOTAL_STEPS, "Saving results")
elapsed = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "task_name": "task_C2_redesign",
    "mode": "PILOT",
    "version": "redesign_v7_corrected",
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
        "bonferroni_alpha": bonferroni_alpha,
        "measurement_approach": (
            "first_letter: FeatureAbsorptionCalculator (IG-based, corrected individual processing). "
            "Semantic: P(child_active|parent_HIGH) vs P(child_active|parent_LOW) in natural sentence contexts."
        ),
        "critical_bug_fix": "model.to_tokens() batch padding causes position=-1 to hit PAD token. Fixed by processing texts individually.",
    },
    "hierarchies": {
        "first_letter": result_first_letter,
        "animate_inanimate": result_ai,
        "noun_proper": result_np,
    },
    "summary": {
        "first_letter_rate": fl_rate,
        "first_letter_null": fl_null,
        "first_letter_ratio": fl_r,
        "animate_suppression": ai_absorption_rate,
        "animate_null": ai_null_mean,
        "animate_ratio": ai_r,
        "noun_proper_suppression": np_absorption_rate,
        "noun_proper_null": np_null_mean,
        "noun_proper_ratio": np_r,
        "n_passing_ratio_1.5": n_passing,
        "n_weak_ratio_1.1": n_weak,
        "overall_go_nogo": overall,
        "sanity_check": {
            "criteria": "first_letter ratio >= 1.5",
            "ratio": fl_r,
            "result": "PASS" if sanity_pass else ("NULL" if sanity_warn else "FAIL"),
        },
    },
    "pilot_pass_criteria": {
        "criteria": "Ratio-to-null >= 1.5 for first_letter",
        "first_letter_ratio": fl_r,
        "first_letter_result": fl_go_nogo,
        "overall": overall,
    },
    "key_findings": {
        "batching_bug": "model.to_tokens(batch) pads sequences → position=-1 hits padding token. All v1-v5 results were measuring PAD token activations.",
        "icl_general_suppression": "Letter-specific features (e.g., L11446) are suppressed in ANY ICL context, not just same-letter ICL. Simple suppression ratio ≈ 1.0.",
        "ig_required": "IG ablation is required to identify which feature takes over as 'absorber'. Without IG, can't distinguish absorption from general context shift.",
        "cross_domain_approach": "For semantic hierarchies, corpus-based P(child|parent_fired) approach is more principled.",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nSaved: {OUTPUT_FILE}")
sys.stdout.flush()

# ── Update gpu_progress.json ──────────────────────────────────────────────────
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
    "planned_min": 55, "actual_min": int(elapsed / 60),
    "start_time": datetime.fromtimestamp(start_time).isoformat(),
    "end_time": datetime.now().isoformat(),
}
gpu_progress_file.write_text(json.dumps(gp, indent=2))

# ── Done ─────────────────────────────────────────────────────────────────────
summary = (
    f"C2-REDESIGN v7 PILOT: {overall}. "
    f"first_letter rate={fl_rate:.3f} null={fl_null:.3f} ratio={fl_r:.3f} ({fl_go_nogo}). "
    f"animate supp={ai_absorption_rate:.3f} null={ai_null_mean:.3f} ratio={ai_r:.3f} ({ai_go_nogo}). "
    f"noun_proper supp={np_absorption_rate:.3f} null={np_null_mean:.3f} ratio={np_r:.3f} ({np_go_nogo}). "
    f"n_passing={n_passing}/3."
)
print(f"\n{'='*70}")
print(f"SUMMARY: {summary}")
print(f"{'='*70}")
sys.stdout.flush()

mark_done("success", summary)
report_progress(9, TOTAL_STEPS, "Complete")
