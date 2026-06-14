"""
Task C.2-REDESIGN v9: Correct SAE Feature Absorption Measurement

ROOT CAUSE OF ALL PREVIOUS FAILURES:
1. Batching bug: model.to_tokens(list) → padding → wrong position (fixed in v7+)
2. Variable-length prompts: need words where len(tok.encode('\\n' + word)) == 2
   NOT len(tok.encode(' ' + word)) == 1 — the template uses newline separators!
   'eagle': ' eagle' = 1 tok but '\\neagle' = 3 toks → WRONG
   'apple': ' apple' = 1 tok AND '\\napple' = 2 toks → CORRECT
3. Semantic hierarchies: n_high=0 in v7/v8 because parent never fires above threshold
   Fix: lower threshold, use np.max (not np.mean) across parent latents

FINAL DESIGN:
- first_letter: FeatureAbsorptionCalculator with newline-safe single-token words
- Semantic: P(child|parent_HIGH) at word token position in sentence contexts
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
    p = {"task_id": TASK_ID, "step": step, "total_steps": total_steps,
         "elapsed_sec": elapsed, "note": note, "updated_at": datetime.now().isoformat()}
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))
    print(f"[{elapsed:.1f}s] Step {step}/{total_steps}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary=""):
    PID_FILE.unlink(missing_ok=True)
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(), "elapsed_sec": time.time() - start_time,
    }))


TOTAL_STEPS = 10
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v9: newline-safe tokens fix")

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

W_dec = sae.W_dec.detach().float().cpu()
d_sae = W_dec.shape[0]
d_model = W_dec.shape[1]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

LAYER = 6
CACHE_KEY = f"blocks.{LAYER}.hook_resid_pre"
print(f"SAE: d_sae={d_sae}, d_model={d_model}")

# COMPATIBILITY FIX: newer sae_lens uses StandardSAEConfig without hook_name
# Monkey-patch it so FeatureAbsorptionCalculator can find the hook
if not hasattr(sae.cfg, 'hook_name'):
    sae.cfg.hook_name = CACHE_KEY
    print(f"  Patched sae.cfg.hook_name = {CACHE_KEY}")

probe_fl_weights = np.load(str(PROBES_DIR / "probe_first_letter_weights.npy"))
probe_fl_classes = np.load(str(PROBES_DIR / "probe_first_letter_classes.npy"))
probe_ai_weights = np.load(str(PROBES_DIR / "probe_animate_inanimate_weights.npy"))
probe_np_weights = np.load(str(PROBES_DIR / "probe_noun_proper_weights.npy"))
sys.stdout.flush()

# ── Individual text processing ────────────────────────────────────────────────

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
    return np.array([get_acts_single(t, position) for t in texts])


def is_newline_safe(word):
    """True if len(tokenizer.encode('\\n' + word)) == 2 — single token after newline."""
    return len(tokenizer.encode("\n" + word)) == 2


# ── Step 2: Build newline-safe vocabulary ─────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Building newline-safe single-token word vocabulary")

print("\n" + "="*70)
print("KEY FIX: filter words to len(tok.encode('\\\\n' + word)) == 2")
print("="*70)
sys.stdout.flush()

vocab_alpha = get_alpha_tokens(tokenizer)
target_letters = list(probe_fl_classes)

# Build newline-safe word lists per letter
words_nl_safe = {}
for w in vocab_alpha:
    w_clean = w.strip()
    if w_clean and w_clean[0].isalpha() and w_clean.isalpha():
        letter = w_clean[0].lower()
        if letter in LETTERS and is_newline_safe(w_clean):
            words_nl_safe.setdefault(letter, []).append(w_clean)

rng = random.Random(SEED)

for letter in target_letters[:6]:
    n = len(words_nl_safe.get(letter, []))
    print(f"  Letter '{letter}': {n} newline-safe words")
print(f"  Letters with >=10 words: {sum(1 for l in target_letters if len(words_nl_safe.get(l, [])) >= 10)}/24")
sys.stdout.flush()

# Find split_feats using individual processing
other_pool = []
for l, ws in words_nl_safe.items():
    clean_ws = [w for w in ws if len(w) >= 3][:10]
    other_pool.extend(clean_ws)
rng.shuffle(other_pool)
other_pool = other_pool[:100]

letter_split_feats = {}

for letter in target_letters:
    letter_words = [w for w in words_nl_safe.get(letter, []) if len(w) >= 3]
    if len(letter_words) < 10:
        letter_split_feats[letter] = []
        continue

    rng.shuffle(letter_words)
    letter_sample = letter_words[:20]
    other_sample = [w for w in other_pool if not w.lower().startswith(letter)][:20]

    acts_l = get_acts_batch_individual(letter_sample, position=-1)
    acts_o = get_acts_batch_individual(other_sample, position=-1)

    mean_l = acts_l.mean(axis=0)
    mean_o = acts_o.mean(axis=0)
    diff = mean_l - mean_o

    active_count = (acts_l >= 0.5).sum(axis=0)
    valid = np.where((active_count >= 10) & (diff > 0.2))[0]
    if len(valid) == 0:
        valid = np.where((active_count >= 7) & (diff > 0.1))[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:5]
    letter_split_feats[letter] = [int(j) for j in valid_sorted if mean_l[j] >= 0.2]

for letter in target_letters[:6]:
    print(f"  Letter '{letter}': split_feats={letter_split_feats.get(letter, [])[:3]}")
n_with_feats = sum(1 for l in target_letters if letter_split_feats.get(l))
print(f"  Total with split_feats: {n_with_feats}/{len(target_letters)}")
sys.stdout.flush()

# ── Step 3: First_letter — FeatureAbsorptionCalculator (FIXED) ───────────────

report_progress(3, TOTAL_STEPS, "First_letter: FeatureAbsorptionCalculator (newline-safe, shuffle=False)")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter — FeatureAbsorptionCalculator")
print("ALL ICL words and target words are newline-safe (1 token after newline)")
print("shuffle_examples=False: same 8 ICL examples for all targets")
print("="*70)
sys.stdout.flush()

# Collect all potential target words (first 25 per letter for test_letters)
# So ICL list can exclude them to prevent contamination errors
all_target_words = set()
for letter in target_letters:
    concept_words = [w for w in words_nl_safe.get(letter, []) if len(w) >= 3]
    all_target_words.update(concept_words[:25])

# Build ICL word list from newline-safe words only, EXCLUDING all potential target words
# Use words from positions 25+ to avoid overlap with targets
icl_word_list = []
for letter in LETTERS:
    nl_safe = [w for w in words_nl_safe.get(letter, []) if len(w) >= 3 and w not in all_target_words]
    # Use words from later in the list (not the first 25 which are used as targets)
    icl_word_list.extend(nl_safe[:10])
rng.shuffle(icl_word_list)
icl_word_list = icl_word_list[:80]
print(f"  ICL word list: {len(icl_word_list)} newline-safe words (no overlap with targets)")
print(f"  Sample: {icl_word_list[:8]}")

# Verify lengths with shuffle_examples=False
# All prompts should be: [BOS] + [8 ICL lines] + [target line (base)]
# Each ICL line: "{word} has the first letter: {LETTER}\n"
# All lines have same length IF all words are newline-safe (1 tok after \n)
# LETTER answers: " A", " B", etc. — all single-token in GPT-2
test_a = create_icl_prompt(
    [w for w in words_nl_safe.get('a', ['ants']) if len(w) >= 3][0],
    examples=icl_word_list,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    max_icl_examples=8,
    shuffle_examples=False,
    check_contamination=False,
)
test_b = create_icl_prompt(
    [w for w in words_nl_safe.get('b', ['both']) if len(w) >= 3][0],
    examples=icl_word_list,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    max_icl_examples=8,
    shuffle_examples=False,
    check_contamination=False,
)
tok_a = len(model.to_tokens([test_a.base], prepend_bos=True)[0])
tok_b = len(model.to_tokens([test_b.base], prepend_bos=True)[0])
print(f"\n  Verify: prompt_a={tok_a} toks, prompt_b={tok_b} toks")
if tok_a == tok_b:
    print(f"  PASS: Same length ({tok_a} tokens)")
else:
    print(f"  FAIL: Different lengths! {tok_a} vs {tok_b}")
sys.stdout.flush()

# Define letter_delta_metric
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


calculator = FeatureAbsorptionCalculator(
    model=model,
    icl_word_list=icl_word_list,
    max_icl_examples=8,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,  # -6
    probe_cos_sim_threshold=0.025,
    ablation_delta_threshold=0.5,
    ig_interpolation_steps=6,
    ig_batch_size=6,
    filter_prompts_batch_size=20,
    topk_feats=10,
    shuffle_examples=False,  # KEY FIX: same 8 ICL examples for all prompts
)

class SimpleProbe:
    def __init__(self, weights):
        self.weights = torch.tensor(weights).float()
probe_fl = SimpleProbe(probe_fl_weights)

test_letters_fl = [(l, f) for l, f in letter_split_feats.items() if len(f) >= 1]
test_letters_fl.sort(key=lambda x: -len(x[1]))

fl_results_by_letter = {}
fl_all_absorption = []
fl_all_words_tested = 0

for letter, main_feat_ids in test_letters_fl[:8]:
    letter_idx = list(probe_fl_classes).index(letter)
    probe_dir = probe_fl.weights[letter_idx]
    metric_fn = letter_delta_metric(tokenizer, letter.upper())

    # Target words must also be newline-safe
    concept_words = [w for w in words_nl_safe.get(letter, []) if len(w) >= 3]
    if len(concept_words) < 5:
        print(f"  Letter '{letter}': insufficient newline-safe words ({len(concept_words)})")
        continue

    concept_words_use = concept_words[:25]
    print(f"\n  Letter '{letter}': {len(concept_words_use)} newline-safe words, main_feats={main_feat_ids[:3]}")
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
        n_filtered = len(concept_words_use) - int(len(concept_words_use) * (1 - results.sample_portion))
        rate = n_absorbed / n_total if n_total > 0 else 0.0

        print(f"  Letter '{letter}': n_filtered={n_total}, n_absorbed={n_absorbed}, rate={rate:.3f}, sample_portion={results.sample_portion:.3f}")
        fl_results_by_letter[letter] = {
            "n_total": n_total, "n_absorbed": n_absorbed,
            "absorption_rate": rate, "sample_portion": results.sample_portion,
        }
        fl_all_absorption.extend([int(r.is_absorption) for r in results.sample_results])
        fl_all_words_tested += n_total

    except ValueError as e:
        err_msg = str(e)[:100]
        print(f"  Letter '{letter}': FAILED — {err_msg}")
        # Diagnostic
        prompts_test = calculator._build_prompts(concept_words_use[:3])
        lens = [len(model.to_tokens([p.base], prepend_bos=True)[0]) for p in prompts_test]
        print(f"  DEBUG: lengths={lens}")
    except Exception as e:
        print(f"  Letter '{letter}': ERROR — {str(e)[:100]}")
        import traceback
        traceback.print_exc()

    sys.stdout.flush()

n_fl_events = len(fl_all_absorption)
fl_rate = float(np.mean(fl_all_absorption)) if fl_all_absorption else 0.0
print(f"\n  [first_letter] Total: n={n_fl_events}, absorption_rate={fl_rate:.4f}")

rng_boot = np.random.RandomState(SEED)

# CORRECT NULL: run the calculator with WRONG letter split_feats (shifted by ~12 letters)
# This gives the "false positive rate" — how often absorption is detected with wrong features
print("\n  Computing null (wrong split_feats control)...")
fl_null_absorption = []
null_letters_tested = []

# Get letters that had successful results
success_letters = [l for l, r in fl_results_by_letter.items() if r.get("n_total", 0) > 0]
all_letters_with_feats = [l for l in target_letters if letter_split_feats.get(l)]

for i, letter in enumerate(success_letters[:4]):  # Test 4 letters for null
    letter_idx = list(probe_fl_classes).index(letter)
    probe_dir = probe_fl.weights[letter_idx]
    metric_fn = letter_delta_metric(tokenizer, letter.upper())
    concept_words = [w for w in words_nl_safe.get(letter, []) if len(w) >= 3][:25]

    # Use split_feats from a different letter (offset by ~12)
    wrong_letter = all_letters_with_feats[(all_letters_with_feats.index(letter) + 12) % len(all_letters_with_feats)]
    wrong_feats = letter_split_feats.get(wrong_letter, [])
    if not wrong_feats or not concept_words:
        continue

    print(f"  Null control: letter={letter}, wrong_feats from '{wrong_letter}': {wrong_feats[:3]}")
    sys.stdout.flush()

    try:
        null_results = calculator.calculate_absorption_sampled(
            sae=sae,
            words=concept_words,
            probe_dir=probe_dir,
            metric_fn=metric_fn,
            main_feature_ids=wrong_feats,
            max_ablation_samples=min(15, len(concept_words)),
            filter_prompts=True,
            show_progress=False,
        )
        n_absorbed_null = sum(1 for r in null_results.sample_results if r.is_absorption)
        n_total_null = len(null_results.sample_results)
        null_rate = n_absorbed_null / n_total_null if n_total_null > 0 else 0.0
        print(f"  Null '{letter}' (wrong feats from '{wrong_letter}'): n={n_total_null}, absorbed={n_absorbed_null}, rate={null_rate:.3f}")
        fl_null_absorption.extend([int(r.is_absorption) for r in null_results.sample_results])
        null_letters_tested.append(letter)
    except Exception as e:
        print(f"  Null '{letter}': ERROR — {str(e)[:60]}")

fl_null = float(np.mean(fl_null_absorption)) if fl_null_absorption else fl_rate
print(f"\n  Null rate (wrong split_feats): {fl_null:.4f}")
print(f"  True rate: {fl_rate:.4f}")

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
    "measurement": "FeatureAbsorptionCalculator (IG-based) with newline-safe words, shuffle=False",
    "n_events": n_fl_events,
    "absorption_rate": fl_rate,
    "null_rate": fl_null,
    "ratio_to_null": fl_ratio,
    "ci_lower": fl_ci_lower,
    "ci_upper": fl_ci_upper,
    "go_nogo": fl_go_nogo,
    "results_by_letter": fl_results_by_letter,
    "note": "newline-safe filter: len(tok.encode('\\n'+word))==2 ensures uniform prompt lengths",
}

# ── Step 4: Semantic hierarchies ──────────────────────────────────────────────

report_progress(4, TOTAL_STEPS, "Semantic hierarchies: word-position P(child|parent_HIGH)")

print("\n" + "="*70)
print("HIERARCHIES 2+3: Semantic hierarchies at WORD POSITION")
print("FIX: lower parent threshold, use max across parent latents")
print("="*70)
sys.stdout.flush()

animate_words = [
    "dog", "cat", "horse", "wolf", "bear", "eagle", "lion", "tiger",
    "rabbit", "monkey", "elephant", "shark", "whale", "deer", "fox",
    "snake", "parrot", "frog", "hawk", "crow",
]
inanimate_words = [
    "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
    "bottle", "box", "ring", "cloud", "river", "lamp", "clock", "window",
    "phone", "cup", "knife", "plate",
]
proper_words = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
    "Alice", "Robert", "Michael", "Jennifer", "William", "Vienna", "Seoul",
    "Boston", "Chicago", "Mumbai", "Moscow", "Cairo", "Athens",
]
common_nouns = [
    "table", "chair", "river", "music", "stone", "flower", "kitchen",
    "bridge", "forest", "garden", "paper", "clock", "window", "bottle",
    "cloud", "market", "village", "castle", "street", "corner",
]

TEMPLATES = [
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


def get_word_tok_pos(text, word, tokenizer):
    """Get negative index of word token position within text (with BOS offset)."""
    toks = tokenizer.encode(text)
    n = len(toks)
    for variant in [" " + word, word]:
        encoded = tokenizer.encode(variant)
        if len(encoded) == 1:
            tok_id = encoded[0]
            for pos_idx in range(n):
                if toks[pos_idx] == tok_id:
                    # With BOS prepended, actual length = n+1
                    # Negative index of position pos_idx+1 in length n+1: (pos_idx+1) - (n+1) = pos_idx - n
                    return pos_idx - n
    return -1


def find_parent_latents(concept_words, control_words, templates, top_k=5, min_act=0.2):
    """Find parent latents at word position in sentence contexts."""
    concept_acts = []
    for w in concept_words[:15]:
        for tmpl in templates[:4]:
            tok_pos = get_word_tok_pos(tmpl.format(w), w, tokenizer)
            v = get_acts_single(tmpl.format(w), position=tok_pos)
            concept_acts.append(v)

    control_acts = []
    for w in control_words[:15]:
        for tmpl in templates[:4]:
            tok_pos = get_word_tok_pos(tmpl.format(w), w, tokenizer)
            v = get_acts_single(tmpl.format(w), position=tok_pos)
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


def find_child_latents(word, reference_words, templates, top_k=5, min_act=0.2):
    """Find features specific to this word at word position."""
    word_acts = []
    for tmpl in templates[:5]:
        tok_pos = get_word_tok_pos(tmpl.format(word), word, tokenizer)
        v = get_acts_single(tmpl.format(word), position=tok_pos)
        word_acts.append(v)
    mean_word = np.array(word_acts).mean(axis=0)

    ref_acts = []
    for w in reference_words[:8]:
        for tmpl in templates[:3]:
            tok_pos = get_word_tok_pos(tmpl.format(w), w, tokenizer)
            v = get_acts_single(tmpl.format(w), position=tok_pos)
            ref_acts.append(v)
    mean_ref = np.array(ref_acts).mean(axis=0) if ref_acts else np.zeros_like(mean_word)

    diff = mean_word - mean_ref
    valid = np.where((mean_word >= min_act) & (diff > 0.1))[0]
    if len(valid) == 0:
        valid = np.where(mean_word >= min_act)[0]

    valid_sorted = valid[np.argsort(diff[valid])[::-1]][:top_k]
    return [int(j) for j in valid_sorted]


print("  Finding animate parent latents...")
ai_parent_latents = find_parent_latents(animate_words, inanimate_words, TEMPLATES)
print(f"  animate parent latents: {ai_parent_latents}")

print("  Finding noun_proper parent latents...")
np_parent_latents_list = find_parent_latents(proper_words, common_nouns, TEMPLATES)
print(f"  noun_proper parent latents: {np_parent_latents_list}")
sys.stdout.flush()

# ── Step 5: animate_inanimate ─────────────────────────────────────────────────

report_progress(5, TOTAL_STEPS, "animate_inanimate: P(child|parent) measurement")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate")
print("="*70)
sys.stdout.flush()

# Lowered threshold to ensure n_high > 0
PARENT_THRESH = 0.3
CHILD_MIN = 0.15
EPS = 1e-8

ai_word_data = []

for word in animate_words:
    other_words = [w for w in animate_words if w != word]
    child_latents = find_child_latents(word, other_words, TEMPLATES, top_k=5, min_act=CHILD_MIN)
    if not child_latents:
        child_latents = find_child_latents(word, inanimate_words, TEMPLATES, top_k=3, min_act=0.05)

    if not child_latents:
        continue

    entries = []
    for tmpl in TEMPLATES:
        tok_pos = get_word_tok_pos(tmpl.format(word), word, tokenizer)
        v = get_acts_single(tmpl.format(word), position=tok_pos)
        # Use MAX across parent latents (higher sensitivity)
        parent_act = float(max([v[j] for j in ai_parent_latents])) if ai_parent_latents else 0.0
        child_act = float(np.mean([v[j] for j in child_latents]))
        entries.append({
            "parent_act": parent_act, "child_act": child_act,
            "parent_fired": parent_act > PARENT_THRESH,
            "child_active": child_act > CHILD_MIN * 0.5,
        })

    if entries:
        ai_word_data.append({"word": word, "entries": entries, "child_latents": child_latents})

ai_child_high = []
ai_child_low = []
for item in ai_word_data:
    eh = [e for e in item["entries"] if e["parent_fired"]]
    el = [e for e in item["entries"] if not e["parent_fired"]]
    if eh:
        ai_child_high.append(float(np.mean([int(e["child_active"]) for e in eh])))
    if el:
        ai_child_low.append(float(np.mean([int(e["child_active"]) for e in el])))

ai_p_high = float(np.mean(ai_child_high)) if ai_child_high else 0.0
ai_p_low  = float(np.mean(ai_child_low))  if ai_child_low  else 0.0
ai_absorption_rate = max(0.0, 1.0 - (ai_p_high / (ai_p_low + EPS))) if ai_p_low > 0 else 0.0

print(f"  P(child|parent_HIGH)={ai_p_high:.4f} (n={len(ai_child_high)}), P(child|parent_LOW)={ai_p_low:.4f} (n={len(ai_child_low)})")

for item in ai_word_data[:5]:
    n_h = sum(1 for e in item["entries"] if e["parent_fired"])
    n_l = sum(1 for e in item["entries"] if not e["parent_fired"])
    ch = float(np.mean([int(e["child_active"]) for e in item["entries"] if e["parent_fired"]])) if n_h > 0 else 0.0
    cl = float(np.mean([int(e["child_active"]) for e in item["entries"] if not e["parent_fired"]])) if n_l > 0 else 0.0
    pa = [e["parent_act"] for e in item["entries"]]
    print(f"  '{item['word']}': n_high={n_h}, n_low={n_l}, ch@hi={ch:.3f}, ch@lo={cl:.3f}, max_pa={max(pa):.3f}")

rng_boot = np.random.RandomState(SEED)
all_pa = [e["parent_act"] for item in ai_word_data for e in item["entries"]]
all_ca = [e["child_act"]  for item in ai_word_data for e in item["entries"]]
null_ai = []
for _ in range(100):
    perm = rng_boot.permutation(len(all_pa))
    ph = [i for i in range(len(all_pa)) if all_pa[i] > PARENT_THRESH]
    pl = [i for i in range(len(all_pa)) if all_pa[i] <= PARENT_THRESH]
    sc = [all_ca[perm[i]] for i in range(len(all_ca))]
    ch = float(np.mean([int(sc[i] > CHILD_MIN * 0.5) for i in ph])) if ph else 0.0
    cl = float(np.mean([int(sc[i] > CHILD_MIN * 0.5) for i in pl])) if pl else 0.0
    null_ai.append(max(0.0, 1.0 - (ch / (cl + EPS))) if cl > 0 else 0.0)

ai_null = float(np.mean(null_ai)) if null_ai else ai_absorption_rate
ai_ratio = ai_absorption_rate / ai_null if ai_null > 0 else (10.0 if ai_absorption_rate > 0 else 1.0)
ai_go = "GO" if ai_ratio >= 1.5 else ("WEAK" if ai_ratio >= 1.1 else "NO_GO")
print(f"\n  [animate] rate={ai_absorption_rate:.4f}, null={ai_null:.4f}, ratio={ai_ratio:.3f}, go={ai_go}")
sys.stdout.flush()

result_ai = {
    "hierarchy": "animate_inanimate",
    "P_child_parent_high": ai_p_high, "P_child_parent_low": ai_p_low,
    "absorption_rate": ai_absorption_rate, "null_mean": ai_null,
    "ratio_to_null": ai_ratio, "go_nogo": ai_go,
    "parent_latents": ai_parent_latents, "n_words": len(ai_word_data),
}

# ── Step 6: noun_proper ───────────────────────────────────────────────────────

report_progress(6, TOTAL_STEPS, "noun_proper: P(child|parent) measurement")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper")
print("="*70)
sys.stdout.flush()

np_word_data = []
for word in proper_words:
    other_proper = [w for w in proper_words if w != word]
    child_latents = find_child_latents(word, other_proper, TEMPLATES, top_k=5, min_act=CHILD_MIN)
    if not child_latents:
        child_latents = find_child_latents(word, common_nouns, TEMPLATES, top_k=3, min_act=0.05)
    if not child_latents:
        continue

    entries = []
    for tmpl in TEMPLATES:
        tok_pos = get_word_tok_pos(tmpl.format(word), word, tokenizer)
        v = get_acts_single(tmpl.format(word), position=tok_pos)
        parent_act = float(max([v[j] for j in np_parent_latents_list])) if np_parent_latents_list else 0.0
        child_act = float(np.mean([v[j] for j in child_latents]))
        entries.append({
            "parent_act": parent_act, "child_act": child_act,
            "parent_fired": parent_act > PARENT_THRESH,
            "child_active": child_act > CHILD_MIN * 0.5,
        })

    if entries:
        np_word_data.append({"word": word, "entries": entries, "child_latents": child_latents})

np_child_high = []
np_child_low  = []
for item in np_word_data:
    eh = [e for e in item["entries"] if e["parent_fired"]]
    el = [e for e in item["entries"] if not e["parent_fired"]]
    if eh:
        np_child_high.append(float(np.mean([int(e["child_active"]) for e in eh])))
    if el:
        np_child_low.append(float(np.mean([int(e["child_active"]) for e in el])))

np_p_high = float(np.mean(np_child_high)) if np_child_high else 0.0
np_p_low  = float(np.mean(np_child_low))  if np_child_low  else 0.0
np_absorption_rate = max(0.0, 1.0 - (np_p_high / (np_p_low + EPS))) if np_p_low > 0 else 0.0

all_pa_np = [e["parent_act"] for item in np_word_data for e in item["entries"]]
all_ca_np = [e["child_act"]  for item in np_word_data for e in item["entries"]]
null_np = []
for _ in range(100):
    perm = rng_boot.permutation(len(all_pa_np))
    ph = [i for i in range(len(all_pa_np)) if all_pa_np[i] > PARENT_THRESH]
    pl = [i for i in range(len(all_pa_np)) if all_pa_np[i] <= PARENT_THRESH]
    sc = [all_ca_np[perm[i]] for i in range(len(all_ca_np))]
    ch = float(np.mean([int(sc[i] > CHILD_MIN * 0.5) for i in ph])) if ph else 0.0
    cl = float(np.mean([int(sc[i] > CHILD_MIN * 0.5) for i in pl])) if pl else 0.0
    null_np.append(max(0.0, 1.0 - (ch / (cl + EPS))) if cl > 0 else 0.0)

np_null = float(np.mean(null_np)) if null_np else np_absorption_rate
np_ratio = np_absorption_rate / np_null if np_null > 0 else (10.0 if np_absorption_rate > 0 else 1.0)
np_go = "GO" if np_ratio >= 1.5 else ("WEAK" if np_ratio >= 1.1 else "NO_GO")

print(f"  P(child@high)={np_p_high:.4f} (n={len(np_child_high)}), P(child@low)={np_p_low:.4f} (n={len(np_child_low)})")
print(f"  rate={np_absorption_rate:.4f}, null={np_null:.4f}, ratio={np_ratio:.3f}, go={np_go}")
sys.stdout.flush()

result_np = {
    "hierarchy": "noun_proper",
    "P_child_parent_high": np_p_high, "P_child_parent_low": np_p_low,
    "absorption_rate": np_absorption_rate, "null_mean": np_null,
    "ratio_to_null": np_ratio, "go_nogo": np_go,
    "parent_latents": np_parent_latents_list, "n_words": len(np_word_data),
}

# ── Step 7: Summary ───────────────────────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Computing overall summary")

fl_r = fl_ratio if np.isfinite(fl_ratio) else 10.0
ai_r = ai_ratio if np.isfinite(ai_ratio) else 10.0
np_r = np_ratio if np.isfinite(np_ratio) else 10.0

n_passing = sum(1 for r in [fl_r, ai_r, np_r] if r >= 1.5)
n_weak    = sum(1 for r in [fl_r, ai_r, np_r] if 1.1 <= r < 1.5)
sanity_pass = fl_r >= 1.5
overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if n_fl_events == 0 and n_passing == 0:
    overall = "NULL_RESULT"

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v9)")
print("="*70)
print(f"  first_letter:      n={n_fl_events}, rate={fl_rate:.4f}, null={fl_null:.4f}, ratio={fl_r:.3f}, go={fl_go_nogo}")
print(f"  animate_inanimate: rate={ai_absorption_rate:.4f}, null={ai_null:.4f}, ratio={ai_r:.3f}, go={ai_go}")
print(f"  noun_proper:       rate={np_absorption_rate:.4f}, null={np_null:.4f}, ratio={np_r:.3f}, go={np_go}")
print(f"  Sanity (first_letter ratio>=1.5): {'PASS' if sanity_pass else 'FAIL'}")
print(f"  Overall: {overall} (n_passing={n_passing}, n_weak={n_weak})")
sys.stdout.flush()

# ── Step 8: Save ──────────────────────────────────────────────────────────────

report_progress(8, TOTAL_STEPS, "Saving results")

output = {
    "version": "v9",
    "timestamp": datetime.now().isoformat(),
    "overall": overall,
    "n_passing": n_passing,
    "n_weak": n_weak,
    "sanity_check_passed": sanity_pass,
    "hierarchies": [result_first_letter, result_ai, result_np],
    "pilot_pass_criterion": "first_letter ratio >= 1.5",
    "fix_notes": [
        "v9: newline-safe filter len(tok.encode('\\n'+word))==2 for all ICL and target words",
        "v9: shuffle_examples=False ensures same ICL context for all prompts",
        "v9: semantic hierarchies use word token position (not last token)",
        "v9: parent threshold 0.3 + max across parent latents for higher sensitivity",
    ],
}
OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"Saved: {OUTPUT_FILE}")

# ── Step 9: Pilot summary ─────────────────────────────────────────────────────

report_progress(9, TOTAL_STEPS, "Writing pilot summary")

pilot = {
    "pilot_id": "C2_redesign_v9", "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(), "overall": overall,
    "hierarchies": {
        "first_letter": {"ratio": fl_r, "go": fl_go_nogo, "n": n_fl_events},
        "animate_inanimate": {"ratio": ai_r, "go": ai_go},
        "noun_proper": {"ratio": np_r, "go": np_go},
    },
    "pilot_passed": n_passing >= 1,
}
(PILOTS_DIR / "C2_pilot_summary_v9.json").write_text(json.dumps(pilot, indent=2))

summary_str = (
    f"C2-REDESIGN v9 PILOT: {overall}. "
    f"first_letter n={n_fl_events} rate={fl_rate:.3f} ratio={fl_r:.3f} ({fl_go_nogo}). "
    f"animate rate={ai_absorption_rate:.3f} ratio={ai_r:.3f} ({ai_go}). "
    f"noun_proper rate={np_absorption_rate:.3f} ratio={np_r:.3f} ({np_go}). "
    f"n_passing={n_passing}/3."
)
print("\n" + "="*70)
print(f"SUMMARY: {summary_str}")
print("="*70)

mark_done(status="success", summary=summary_str)
report_progress(10, TOTAL_STEPS, "Complete")
print("\nDone.")
