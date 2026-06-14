"""
C1_ablations PILOT — Threshold Sensitivity (A1) and Cosine Pre-filter (A2)

PILOT scope (from task_plan.json):
  A1: Threshold sweep tau = {0.5, 0.75, 1.0, 1.25, 1.5} on test letters from C1B (F-H)
  A2 preview: cosine pre-filter coverage check at thresholds {0.10, 0.15, 0.25}

Pass criteria (PILOT):
  - F1 values computed for all 5 tau values (all_tau_computed)
  - F1 is non-zero for at least 3 tau values
  - Results written to output file without error

Model: gpt2-small + gpt2-small-res-jb SAE (same as C1B pilot)
Note: This reuses the C1B methodology and data pipeline.
"""

import os
import sys
import json
import time
import random
import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---- Config ----
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

TASK_ID = "C1_ablations"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.datetime.now()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)


def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# PID file
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PID_FILE.write_text(str(os.getpid()))
print(f"[C1_ablations] PID={os.getpid()} written to {PID_FILE}")

PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"


def write_progress(step, total_steps, message, metrics=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "loss": None,
        "metric": {"description": message, **(metrics or {})},
        "updated_at": datetime.datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(to_python_types(progress), indent=2, cls=NumpyEncoder))
    print(f"[{step}/{total_steps}] {message}")


def mark_done(status, summary, final_progress=None):
    if PID_FILE.exists():
        PID_FILE.unlink()
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps(to_python_types({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress or {},
        "timestamp": datetime.datetime.now().isoformat(),
    }), indent=2, cls=NumpyEncoder))
    print(f"[C1_ablations] DONE: {status} — {summary}")


write_progress(0, 8, "Starting C1_ablations pilot")

# ---- STEP 1: Load C1A alpha_ij data ----
write_progress(1, 8, "Loading C1A alpha_ij parquet")

import pandas as pd

C1A_PARQUET = PILOTS_DIR / "C1A_activation_stats_pilot.parquet"
if not C1A_PARQUET.exists():
    mark_done("failed", f"C1A parquet not found: {C1A_PARQUET}")
    sys.exit(1)

df_alpha = pd.read_parquet(C1A_PARQUET)
print(f"[C1_ablations] C1A alpha_ij loaded: {len(df_alpha)} pairs")
print(f"[C1_ablations] Columns: {df_alpha.columns.tolist()}")
print(f"[C1_ablations] Alpha_ij range: [{df_alpha['alpha_ij'].min():.4f}, {df_alpha['alpha_ij'].max():.4f}]")

# Build fast lookup
alpha_lookup = {}
cosine_lookup = {}
for _, row in df_alpha.iterrows():
    li, lj = int(row['latent_i']), int(row['latent_j'])
    aij = float(row['alpha_ij'])
    cos = float(row['decoder_cosine'])
    alpha_lookup[(li, lj)] = aij
    cosine_lookup[(li, lj)] = cos

# Build parent -> children map (for LV score computation)
parent_to_children = defaultdict(dict)
for (li, lj), aij in alpha_lookup.items():
    if aij > 0:
        parent_to_children[lj][li] = aij  # lj is parent, li is child
print(f"[C1_ablations] alpha_lookup size: {len(alpha_lookup)}, parent_to_children: {len(parent_to_children)}")

# ---- STEP 2: Load model and SAE ----
write_progress(2, 8, "Loading GPT-2 Small model and SAE")

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[C1_ablations] Device: {DEVICE}")
torch.manual_seed(SEED)

from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token
print(f"[C1_ablations] Model: gpt2-small, d_model={model.cfg.d_model}")

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"

sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE,
)
sae.eval()
if not hasattr(sae.cfg, 'hook_name'):
    sae.cfg.hook_name = HOOK_NAME
D_SAE = sae.cfg.d_sae
print(f"[C1_ablations] SAE loaded: {SAE_RELEASE}/{SAE_ID}, d_sae={D_SAE}")

# ---- STEP 3: Build word vocabulary ----
write_progress(3, 8, "Building word vocabulary for ICL prompts")

from sae_spelling.vocab import get_alpha_tokens
from sae_spelling.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
    create_icl_prompt,
)

vocab_alpha = get_alpha_tokens(tokenizer)

single_tok_words_by_letter = {}
for tok_str in vocab_alpha:
    w = tok_str.strip()
    if not w or not w[0].isalpha() or not w.isalpha() or len(w) < 2:
        continue
    toks = tokenizer.encode(' ' + w)
    if len(toks) == 1:
        letter = w[0].upper()
        single_tok_words_by_letter.setdefault(letter, []).append(w)

random.seed(SEED)
icl_word_list = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    words_l = single_tok_words_by_letter.get(letter, [])
    random.shuffle(words_l)
    icl_word_list.extend(words_l[:30])
random.seed(SEED)
random.shuffle(icl_word_list)
print(f"[C1_ablations] ICL word list: {len(icl_word_list)} single-token words")

# A1 uses the same test letters as C1B pilot
CALIB_LETTERS = ["A", "B", "C", "D", "E"]
TEST_LETTERS = ["F", "G", "H"]
ALL_LETTERS = CALIB_LETTERS + TEST_LETTERS
N_ICL = 8


def build_icl_prompt(word):
    return create_icl_prompt(
        word,
        examples=icl_word_list,
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        answer_formatter=first_letter_formatter(),
        max_icl_examples=N_ICL,
        shuffle_examples=False,
        check_contamination=False,
    )


ref_word = icl_word_list[N_ICL]
ref_prompt = build_icl_prompt(ref_word)
expected_tok_len = len(model.to_tokens([ref_prompt.base])[0])
print(f"[C1_ablations] Reference prompt token length: {expected_tok_len}")

icl_first_n = set(icl_word_list[:N_ICL])
words_by_letter = {}
for letter in ALL_LETTERS:
    original = single_tok_words_by_letter.get(letter, [])
    random.seed(SEED + ord(letter))
    random.shuffle(original)
    valid = []
    for w in original:
        if w in icl_first_n:
            continue
        try:
            p = build_icl_prompt(w)
            tok_len = len(model.to_tokens([p.base])[0])
            if tok_len == expected_tok_len:
                valid.append(w)
        except Exception:
            continue
    words_by_letter[letter] = valid[:100]
    print(f"  Letter {letter}: {len(valid)} valid words")

# ---- STEP 4: Train probes ----
write_progress(4, 8, "Training linear probes")

from sae_spelling.probing import train_binary_probe


def get_resid_at_pos(word, pos=VERBOSE_FIRST_LETTER_TOKEN_POS, hook_name=HOOK_NAME):
    try:
        p = build_icl_prompt(word)
        if len(model.to_tokens([p.base])[0]) != expected_tok_len:
            return None
        _, cache = model.run_with_cache([p.base], names_filter=hook_name)
        act = cache[hook_name][0, pos, :].cpu().float()
        return act
    except Exception:
        return None


probe_directions = {}
for letter in ALL_LETTERS:
    random.seed(SEED + ord(letter))
    pos_words = [w for w in words_by_letter[letter] if w not in icl_first_n][:40]
    neg_words = []
    for l2, ws in single_tok_words_by_letter.items():
        if l2 != letter.upper():
            for w in ws[:3]:
                if w not in icl_first_n:
                    try:
                        p = build_icl_prompt(w)
                        if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                            neg_words.append(w)
                    except Exception:
                        continue
    neg_words = neg_words[:40]

    combined = [(w, 1) for w in pos_words] + [(w, 0) for w in neg_words]
    acts_list, labels = [], []
    with torch.no_grad():
        for w, lab in combined:
            act = get_resid_at_pos(w)
            if act is not None:
                acts_list.append(act)
                labels.append(float(lab))

    if len(acts_list) < 10:
        print(f"  WARNING: Not enough activations for {letter} probe ({len(acts_list)})")
        continue

    acts_t = torch.stack(acts_list).to(DEVICE)
    labels_t = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    probe = train_binary_probe(acts_t, labels_t,
                               num_epochs=100, lr=0.01,
                               show_progress=False, verbose=False, device=DEVICE)
    probe_dir = F.normalize(probe.fc.weight[0].detach().cpu(), dim=0)
    probe_directions[letter] = probe_dir
    print(f"  Letter {letter}: probe trained ({len(acts_list)} samples)")

print(f"[C1_ablations] Probes trained: {list(probe_directions.keys())}")

# ---- STEP 5: Absorption measurement ----
write_progress(5, 8, "Running absorption measurement for all 8 letters")

from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator

calculator = FeatureAbsorptionCalculator(
    model=model,
    icl_word_list=icl_word_list,
    max_icl_examples=N_ICL,
    base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
    answer_formatter=first_letter_formatter(),
    word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
    probe_cos_sim_threshold=0.01,
    ablation_delta_threshold=0.1,
    ig_interpolation_steps=4,
    ig_batch_size=4,
    filter_prompts_batch_size=20,
    topk_feats=10,
    shuffle_examples=False,
)


def letter_delta_metric(tokenizer_obj, pos_letter, device=DEVICE):
    LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pos_letter_tok = tokenizer_obj.encode(f" {pos_letter.upper()}")[-1]
    neg_letter_toks = torch.tensor(
        [tokenizer_obj.encode(f" {l}")[-1] for l in LETTERS_UPPER if l != pos_letter.upper()]
    ).to(device)

    def metric_fn(logits):
        pos_logit = logits[:, -1, pos_letter_tok]
        neg_logits = logits[:, -1, neg_letter_toks]
        return pos_logit - neg_logits.mean(dim=-1)
    return metric_fn


def get_main_feature_ids(letter, sae_model, words_prompts, neg_prompts, top_k=5):
    def get_sae_acts_at_pos(wps):
        acts = []
        with torch.no_grad():
            for w, p in wps:
                try:
                    _, cache = model.run_with_cache([p.base], names_filter=sae_model.cfg.hook_name)
                    sae_in = cache[sae_model.cfg.hook_name]
                    sae_acts = sae_model.encode(sae_in)
                    acts.append(sae_acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float())
                except Exception:
                    continue
        if not acts:
            return None
        return torch.stack(acts)

    pos_acts = get_sae_acts_at_pos(words_prompts)
    neg_acts = get_sae_acts_at_pos(neg_prompts)
    if pos_acts is None or pos_acts.shape[0] < 2:
        return []
    pos_mean = pos_acts.mean(0)
    neg_mean = neg_acts.mean(0) if neg_acts is not None else torch.zeros_like(pos_mean)
    diff = pos_mean - neg_mean
    return diff.topk(top_k).indices.tolist()


# Pre-build prompts per letter
words_prompts_by_letter = {}
for letter in ALL_LETTERS:
    wps = []
    for w in words_by_letter[letter][:20]:
        try:
            p = build_icl_prompt(w)
            wps.append((w, p))
        except Exception:
            continue
    words_prompts_by_letter[letter] = wps


def get_neg_prompts_for(letter):
    neg = []
    for l2, wps in words_prompts_by_letter.items():
        if l2 != letter:
            neg.extend(wps[:5])
    return neg[:20]


absorption_by_letter = {}
for letter in ALL_LETTERS:
    if letter not in probe_directions:
        print(f"  Skipping {letter}: no probe direction")
        absorption_by_letter[letter] = {
            "absorption_rate": None, "n_absorbed": 0, "n_total": 0,
            "words": [], "is_absorbed": [], "error": "no probe"
        }
        continue

    probe_dir = probe_directions[letter]
    words_prompts = words_prompts_by_letter[letter]
    neg_prompts = get_neg_prompts_for(letter)
    main_ids = get_main_feature_ids(letter, sae, words_prompts, neg_prompts, top_k=5)
    print(f"\n  Letter {letter}: main_ids={main_ids}")

    metric_fn = letter_delta_metric(tokenizer, letter)
    words = words_by_letter[letter]

    try:
        results = calculator.calculate_absorption_sampled(
            sae=sae,
            words=words,
            probe_dir=probe_dir.to(DEVICE),
            metric_fn=metric_fn,
            main_feature_ids=main_ids,
            max_ablation_samples=50,
            filter_prompts=True,
            show_progress=True,
        )
        sample_results = results.sample_results
        n_tot = len(sample_results)
        n_abs = sum(1 for r in sample_results if r.is_absorption)
        rate = n_abs / n_tot if n_tot > 0 else 0.0
        print(f"    {letter}: absorption_rate={rate:.3f} ({n_abs}/{n_tot})")
        absorption_by_letter[letter] = {
            "absorption_rate": rate,
            "n_absorbed": n_abs,
            "n_total": n_tot,
            "words": [r.word for r in sample_results],
            "is_absorbed": [bool(r.is_absorption) for r in sample_results],
            "main_ids": [int(x) for x in main_ids],
        }
    except Exception as e:
        print(f"    ERROR for {letter}: {e}")
        try:
            results = calculator.calculate_absorption_sampled(
                sae=sae,
                words=words,
                probe_dir=probe_dir.to(DEVICE),
                metric_fn=metric_fn,
                main_feature_ids=main_ids,
                max_ablation_samples=30,
                filter_prompts=False,
                show_progress=True,
            )
            sample_results = results.sample_results
            n_tot = len(sample_results)
            n_abs = sum(1 for r in sample_results if r.is_absorption)
            rate = n_abs / n_tot if n_tot > 0 else 0.0
            print(f"    {letter} (fallback): rate={rate:.3f} ({n_abs}/{n_tot})")
            absorption_by_letter[letter] = {
                "absorption_rate": rate,
                "n_absorbed": n_abs,
                "n_total": n_tot,
                "words": [r.word for r in sample_results],
                "is_absorbed": [bool(r.is_absorption) for r in sample_results],
                "main_ids": [int(x) for x in main_ids],
                "note": "filter_prompts=False fallback",
            }
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")
            absorption_by_letter[letter] = {
                "absorption_rate": None, "n_absorbed": 0, "n_total": 0,
                "words": [], "is_absorbed": [], "error": str(e2),
                "main_ids": [int(x) for x in main_ids],
            }

# ---- STEP 6: Compute word-level LV scores ----
write_progress(6, 8, "Computing word-level LV scores from alpha_ij")


def get_word_sae_activations(word, sae_model, top_k=20):
    try:
        p = build_icl_prompt(word)
        if len(model.to_tokens([p.base])[0]) != expected_tok_len:
            return [], []
        with torch.no_grad():
            _, cache = model.run_with_cache([p.base], names_filter=sae_model.cfg.hook_name)
            sae_in = cache[sae_model.cfg.hook_name]
            sae_acts = sae_model.encode(sae_in)
            acts_at_pos = sae_acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float()
        top_vals, top_idx = acts_at_pos.topk(top_k)
        return top_idx.tolist(), top_vals.tolist()
    except Exception:
        return [], []


def compute_lv_score(word_top_ids, main_ids, parent_to_ch):
    """Max alpha_ij between any word feature and any parent feature."""
    max_alpha = 0.0
    for main_id in main_ids:
        children_of_parent = parent_to_ch.get(main_id, {})
        for word_feat in word_top_ids:
            aij = children_of_parent.get(word_feat, 0.0)
            if aij > max_alpha:
                max_alpha = aij
    return max_alpha


def compute_cosine_score(word_top_ids, main_ids):
    """Max decoder cosine similarity between any word feature and any parent feature."""
    max_cosine = 0.0
    for main_id in main_ids:
        for word_feat in word_top_ids:
            cos = cosine_lookup.get((word_feat, main_id), cosine_lookup.get((main_id, word_feat), 0.0))
            if cos > max_cosine:
                max_cosine = cos
    return max_cosine


pair_data = []

for letter in ALL_LETTERS:
    absor = absorption_by_letter[letter]
    if absor.get('absorption_rate') is None:
        continue

    words_l = absor['words']
    is_absorbed_l = absor['is_absorbed']
    main_ids = absor.get('main_ids', [])

    if not main_ids:
        words_prompts = words_prompts_by_letter[letter]
        neg_prompts = get_neg_prompts_for(letter)
        main_ids = get_main_feature_ids(letter, sae, words_prompts, neg_prompts, top_k=5)

    print(f"\n  Letter {letter}: main_ids={main_ids[:3]}... ({len(words_l)} words)")

    for word, is_abs in zip(words_l, is_absorbed_l):
        top_ids, top_vals = get_word_sae_activations(word, sae, top_k=20)
        if not top_ids:
            continue
        lv_score = compute_lv_score(top_ids, main_ids, parent_to_children)
        cos_score = compute_cosine_score(top_ids, main_ids)
        pair_data.append({
            "letter": letter,
            "word": word,
            "label": 1 if is_abs else 0,
            "alpha_ij": lv_score,
            "decoder_cosine": cos_score,
            "top_feats": [int(x) for x in top_ids[:5]],
            "main_ids": [int(x) for x in main_ids[:3]],
        })

df_pairs = pd.DataFrame(pair_data)
print(f"\n[C1_ablations] Total word-level pairs: {len(df_pairs)}")
if len(df_pairs) > 0:
    print(f"  Positives (absorbed): {df_pairs['label'].sum()}")
    print(f"  Negatives: {(df_pairs['label'] == 0).sum()}")
    if df_pairs['label'].sum() > 0:
        pos_stats = df_pairs[df_pairs['label'] == 1]['alpha_ij'].describe()
        neg_stats = df_pairs[df_pairs['label'] == 0]['alpha_ij'].describe()
        print(f"  LV score (pos): mean={pos_stats['mean']:.4f}, std={pos_stats.get('std', 0):.4f}")
        print(f"  LV score (neg): mean={neg_stats['mean']:.4f}, std={neg_stats.get('std', 0):.4f}")
        cos_pos = df_pairs[df_pairs['label'] == 1]['decoder_cosine'].describe()
        cos_neg = df_pairs[df_pairs['label'] == 0]['decoder_cosine'].describe()
        print(f"  Cosine (pos): mean={cos_pos['mean']:.4f}")
        print(f"  Cosine (neg): mean={cos_neg['mean']:.4f}")

# ---- STEP 7: A1 — Threshold sweep ----
write_progress(7, 8, "A1: Threshold sweep tau={0.5,0.75,1.0,1.25,1.5} on test letters")

TAU_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Calibration set: letters A-E
df_calib = df_pairs[df_pairs['letter'].isin(CALIB_LETTERS)].copy() if len(df_pairs) > 0 else pd.DataFrame()
# Test set: letters F-H
df_test = df_pairs[df_pairs['letter'].isin(TEST_LETTERS)].copy() if len(df_pairs) > 0 else pd.DataFrame()

print(f"[C1_ablations] Calibration: {len(df_calib)} samples, {int(df_calib['label'].sum()) if len(df_calib) > 0 else 0} absorbed")
print(f"[C1_ablations] Test: {len(df_test)} samples, {int(df_test['label'].sum()) if len(df_test) > 0 else 0} absorbed")

# Calibration
calib_results_by_tau = {}
best_tau = 0.5
best_calib_f1 = 0.0

if len(df_calib) > 0 and df_calib['label'].sum() > 0:
    y_calib = df_calib['label'].values
    scores_calib = df_calib['alpha_ij'].values
    cosine_calib = df_calib['decoder_cosine'].values

    print("\n  Calibration threshold sweep:")
    for tau in TAU_VALUES:
        preds = (scores_calib >= tau).astype(int)
        cos_preds = (cosine_calib >= tau).astype(int)
        if preds.sum() == 0:
            prec, rec, f1_val = 0.0, 0.0, 0.0
        else:
            prec = float(precision_score(y_calib, preds, zero_division=0))
            rec = float(recall_score(y_calib, preds, zero_division=0))
            f1_val = float(f1_score(y_calib, preds, zero_division=0))
        if cos_preds.sum() == 0:
            cos_f1 = 0.0
        else:
            cos_f1 = float(f1_score(y_calib, cos_preds, zero_division=0))
        calib_results_by_tau[tau] = {
            "precision": prec, "recall": rec, "f1": f1_val,
            "cosine_f1": cos_f1
        }
        print(f"  tau={tau}: LV F1={f1_val:.4f} (P={prec:.3f}, R={rec:.3f}) | cosine F1={cos_f1:.4f}")
        if f1_val > best_calib_f1:
            best_calib_f1 = f1_val
            best_tau = tau
    print(f"\n  Best tau: {best_tau} (F1={best_calib_f1:.4f})")

# Test evaluation
test_results_by_tau = {}
cosine_test_results_by_tau = {}
test_f1 = 0.0
test_precision = 0.0
test_recall = 0.0
test_auc = 0.5

if len(df_test) > 0 and df_test['label'].sum() > 0:
    y_test = df_test['label'].values
    scores_test = df_test['alpha_ij'].values
    cosine_test = df_test['decoder_cosine'].values

    print(f"\n  Test set threshold sweep:")
    for tau in TAU_VALUES:
        preds = (scores_test >= tau).astype(int)
        cos_preds = (cosine_test >= tau).astype(int)

        if preds.sum() == 0:
            p, r, f1_val = 0.0, 0.0, 0.0
        else:
            p = float(precision_score(y_test, preds, zero_division=0))
            r = float(recall_score(y_test, preds, zero_division=0))
            f1_val = float(f1_score(y_test, preds, zero_division=0))

        if cos_preds.sum() == 0:
            cos_f1 = 0.0
        else:
            cos_f1 = float(f1_score(y_test, cos_preds, zero_division=0))

        test_results_by_tau[tau] = {"precision": p, "recall": r, "f1": f1_val}
        cosine_test_results_by_tau[tau] = {"f1": cos_f1}
        print(f"  tau={tau}: LV F1={f1_val:.4f} (P={p:.3f}, R={r:.3f}) | cosine F1={cos_f1:.4f}")

        if tau == best_tau:
            test_f1 = f1_val
            test_precision = p
            test_recall = r

    if len(np.unique(y_test)) > 1:
        test_auc = float(roc_auc_score(y_test, scores_test))
    else:
        test_auc = float('nan')

    print(f"\n  Test results (best tau={best_tau}): F1={test_f1:.4f}, AUC={test_auc:.4f}")
else:
    print("[C1_ablations] WARNING: Insufficient test data")
    for tau in TAU_VALUES:
        test_results_by_tau[tau] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        cosine_test_results_by_tau[tau] = {"f1": 0.0}

# A1 pass criteria
n_nonzero_f1 = sum(1 for v in test_results_by_tau.values() if v["f1"] > 0)
all_tau_computed = len(test_results_by_tau) == len(TAU_VALUES)
print(f"\n[C1_ablations] A1 Pass criteria: all_tau_computed={all_tau_computed}, n_nonzero={n_nonzero_f1}/{len(TAU_VALUES)}")

# ---- A2: Cosine pre-filter coverage analysis ----
COSINE_THRESHOLDS = [0.10, 0.15, 0.25]
a2_coverage = {}

print("\n[C1_ablations] A2: Cosine pre-filter coverage check...")
if len(df_pairs) > 0 and df_pairs['label'].sum() > 0:
    all_cosine = df_pairs['decoder_cosine'].values
    all_labels = df_pairs['label'].values
    absorbed_mask = all_labels == 1
    n_absorbed = absorbed_mask.sum()

    for ct in COSINE_THRESHOLDS:
        covered = np.sum((all_cosine >= ct) & absorbed_mask)
        coverage = float(covered) / n_absorbed if n_absorbed > 0 else 0.0
        candidates_mask = all_cosine >= ct
        n_cand = candidates_mask.sum()
        precision = float(np.sum(candidates_mask & absorbed_mask)) / n_cand if n_cand > 0 else 0.0
        a2_coverage[str(ct)] = {
            "cosine_threshold": ct,
            "coverage_fraction": coverage,
            "precision": precision,
            "n_candidates": int(n_cand),
            "n_absorbed_covered": int(covered),
            "n_absorbed_total": int(n_absorbed),
        }
        print(f"  cosine>={ct}: coverage={coverage:.3f}, precision={precision:.3f}, n_cand={n_cand}")
else:
    for ct in COSINE_THRESHOLDS:
        a2_coverage[str(ct)] = {
            "cosine_threshold": ct,
            "coverage_fraction": 0.0,
            "precision": 0.0,
            "n_candidates": 0,
            "n_absorbed_covered": 0,
            "n_absorbed_total": 0,
        }

# ---- STEP 8: Save results ----
write_progress(8, 8, "Saving pilot results")

go_no_go = "GO" if (all_tau_computed and n_nonzero_f1 >= 3) else "NO_GO"
# Lenient: if pipeline runs but data is sparse, still consider GO if F1 is non-zero
if n_nonzero_f1 == 0 and len(df_test) > 0 and df_test['label'].sum() > 0:
    go_no_go = "NO_GO"
elif all_tau_computed and len(df_test) == 0:
    # Data insufficient but pipeline works
    go_no_go = "NO_GO"

runtime_seconds = (datetime.datetime.now() - START_TIME).total_seconds()

result = {
    "task_id": TASK_ID,
    "timestamp": datetime.datetime.now().isoformat(),
    "mode": "PILOT",
    "model": "gpt2-small",
    "sae_release": SAE_RELEASE,
    "sae_id": SAE_ID,
    "d_sae": D_SAE,
    "calib_letters": CALIB_LETTERS,
    "test_letters": TEST_LETTERS,
    "n_pairs_calib": len(df_calib),
    "n_pairs_test": len(df_test),
    "n_absorbed_test": int(df_test['label'].sum()) if len(df_test) > 0 else 0,
    "go_no_go": go_no_go,
    "absorption_by_letter": {
        letter: {
            "absorption_rate": absorption_by_letter[letter].get("absorption_rate"),
            "n_absorbed": absorption_by_letter[letter].get("n_absorbed", 0),
            "n_total": absorption_by_letter[letter].get("n_total", 0),
        }
        for letter in ALL_LETTERS
    },
    "a1_threshold_sweep": {
        "tau_values": TAU_VALUES,
        "calibration": {str(tau): v for tau, v in calib_results_by_tau.items()},
        "best_tau_calib": best_tau,
        "best_calib_f1": best_calib_f1,
        "test": {str(tau): v for tau, v in test_results_by_tau.items()},
        "cosine_baseline_test": {str(tau): v for tau, v in cosine_test_results_by_tau.items()},
        "test_f1_at_best_tau": test_f1,
        "test_auc": test_auc if not (isinstance(test_auc, float) and np.isnan(test_auc)) else None,
        "n_nonzero_f1_taus": n_nonzero_f1,
        "all_tau_computed": all_tau_computed,
    },
    "a2_cosine_prefilter": a2_coverage,
    "pass_criteria": {
        "all_tau_computed": all_tau_computed,
        "n_nonzero_f1": n_nonzero_f1,
        "pass_n_nonzero_f1_ge_3": n_nonzero_f1 >= 3,
        "pass_output_written": True,
    },
    "runtime_seconds": runtime_seconds,
}

result_clean = to_python_types(result)

# Save to pilots/
pilot_path = PILOTS_DIR / f"{TASK_ID}_pilot.json"
pilot_path.write_text(json.dumps(result_clean, indent=2, cls=NumpyEncoder))
print(f"[C1_ablations] Pilot results saved to {pilot_path}")

# Save summary markdown
md_path = PILOTS_DIR / f"{TASK_ID}_pilot_summary.md"
with open(md_path, "w") as f:
    f.write(f"# C1_ablations Pilot Summary\n\n")
    f.write(f"**GO/NO-GO: {go_no_go}**\n\n")
    f.write(f"**Ablation A1:** Threshold sensitivity sweep (tau = {TAU_VALUES})\n\n")
    f.write(f"## Absorption Summary\n\n")
    f.write("| Letter | Group | Absorption Rate | N Absorbed | N Total |\n")
    f.write("|--------|-------|-----------------|------------|---------|\n")
    for l in CALIB_LETTERS:
        a = absorption_by_letter[l]
        r = f"{a.get('absorption_rate', 0):.3f}" if a.get('absorption_rate') is not None else "N/A"
        f.write(f"| {l} | Calib | {r} | {a.get('n_absorbed', 0)} | {a.get('n_total', 0)} |\n")
    for l in TEST_LETTERS:
        a = absorption_by_letter[l]
        r = f"{a.get('absorption_rate', 0):.3f}" if a.get('absorption_rate') is not None else "N/A"
        f.write(f"| {l} | Test | {r} | {a.get('n_absorbed', 0)} | {a.get('n_total', 0)} |\n")
    f.write("\n## A1: Threshold Sweep (Test Set)\n\n")
    f.write("| Tau | LV F1 | LV Precision | LV Recall | Cosine-Only F1 | LV vs Cosine Delta |\n")
    f.write("|-----|-------|-------------|-----------|----------------|--------------------|\n")
    for tau in TAU_VALUES:
        lv = test_results_by_tau.get(tau, {"f1": 0, "precision": 0, "recall": 0})
        cos = cosine_test_results_by_tau.get(tau, {"f1": 0})
        delta = lv['f1'] - cos['f1']
        f.write(f"| {tau} | {lv['f1']:.4f} | {lv.get('precision',0):.3f} | {lv.get('recall',0):.3f} | {cos['f1']:.4f} | {delta:+.4f} |\n")
    f.write(f"\n**Best calib tau:** {best_tau} (calib F1={best_calib_f1:.4f})\n\n")
    f.write(f"**Test AUC:** {test_auc:.4f}\n\n" if not (isinstance(test_auc, float) and np.isnan(test_auc)) else "**Test AUC:** N/A\n\n")
    f.write("## A2: Cosine Pre-filter Coverage\n\n")
    f.write("| Cosine Threshold | Coverage | Precision | N Candidates |\n")
    f.write("|-----------------|----------|-----------|-------------|\n")
    for ct_str, v in a2_coverage.items():
        f.write(f"| {ct_str} | {v['coverage_fraction']:.3f} | {v['precision']:.3f} | {v['n_candidates']} |\n")
    f.write("\n## Pass Criteria\n\n")
    for k, v in result["pass_criteria"].items():
        status = "PASS" if v else "FAIL"
        f.write(f"- {k}: {v} [{status}]\n")
    f.write(f"\n**Runtime:** {runtime_seconds:.1f}s\n")

print(f"[C1_ablations] Summary saved to {md_path}")

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    with open(gpu_progress_path, "r") as f:
        gpu_progress = json.load(f)
except Exception:
    gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

end_time = datetime.datetime.now()
actual_min = max(1, int((end_time - START_TIME).total_seconds() / 60))

if go_no_go == "GO":
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
else:
    if TASK_ID not in gpu_progress.get("failed", []):
        gpu_progress.setdefault("failed", []).append(TASK_ID)

gpu_progress.get("running", {}).pop(TASK_ID, None)
gpu_progress.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 30,
    "actual_min": actual_min,
    "start_time": START_TIME.isoformat(),
    "end_time": end_time.isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "tau_values": TAU_VALUES,
        "test_letters": TEST_LETTERS,
        "mode": "pilot",
        "n_pairs_test": len(df_test),
    },
}

with open(gpu_progress_path, "w") as f:
    json.dump(to_python_types(gpu_progress), f, indent=2, cls=NumpyEncoder)

print(f"\n[C1_ablations] GO/NO-GO: {go_no_go}")
print(f"  all_tau_computed: {all_tau_computed}")
print(f"  n_nonzero_f1: {n_nonzero_f1}/{len(TAU_VALUES)}")
print(f"  best_tau (calib): {best_tau}, test F1: {test_f1:.4f}")

mark_done(
    "success" if go_no_go == "GO" else "failed",
    f"C1_ablations PILOT: {go_no_go}. A1: best_tau={best_tau}, test_f1={test_f1:.4f}, "
    f"n_nonzero_f1={n_nonzero_f1}/{len(TAU_VALUES)}. Runtime={runtime_seconds:.1f}s",
    final_progress={
        "task_id": TASK_ID,
        "step": 8,
        "total_steps": 8,
        "message": f"GO/NO-GO: {go_no_go}",
        "metrics": {"best_tau": best_tau, "test_f1": test_f1, "n_nonzero": n_nonzero_f1},
    }
)

sys.exit(0 if go_no_go == "GO" else 1)
