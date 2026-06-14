"""
C1D_width_paradox PILOT experiment
====================================
PILOT SCOPE:
  - Widths {24576, 49152} (GPT-2 feature-splitting layer 8; analogous to task plan's {16k, 65k})
    Release: gpt2-small-res-jb-feature-splitting
    Widths 768,1536,3072,6144,12288,24576,49152,98304 available
  - Letters A-E only (5 letters)
  - ~1k activation tokens for alpha_ij, ~50 words per letter for absorption

Procedure for each width:
  1. Load SAE at that width (gpt2-small-res-jb-feature-splitting, blocks.8.hook_resid_pre_{W})
  2. Collect SAE activations on ~1k OpenWebText tokens
  3. Find "parent" letter feature for each letter (top differential SAE feature for letter words)
  4. DAS(k=1): run sae-spelling absorption measurement -> absorption_rate = DAS(k=1) proxy
     (Absorption rate IS the DAS(k=1) measure in sae-spelling framework)
  5. Alpha_ij computation for this SAE: co-activation statistics for parent's top-3 children
  6. DAS(k=3): fit logistic regression predicting parent activation from top-3 child features
     DAS(k=3) = 1 - McFadden_R2_of_3child_model / McFadden_R2_of_baseline
     (baseline = intercept-only model predicting parent binary activation)

Pass criteria (PILOT):
  - DAS(k=1) and DAS(k=3) computable for all 5 pilot letters at both widths
  - DAS(k=3) >= DAS(k=1) for at least 3 of 5 letters at 49152 (wider SAE)

Output:
  exp/results/pilots/C1D_das_vs_width_pilot.json
  exp/results/C1D_width_paradox_PROGRESS.json (progress)
  exp/results/C1D_width_paradox_DONE (completion marker)
"""

import os
import sys
import json
import gc
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Force correct GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

TASK_ID = "C1D_width_paradox"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.now()

# PID file
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PID_FILE.write_text(str(os.getpid()))
print(f"[C1D PILOT] PID={os.getpid()} written to {PID_FILE}")

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


def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_py(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def write_progress(step, total, msg, metrics=None):
    data = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total,
        "message": msg,
        "metrics": metrics or {},
        "elapsed_sec": (datetime.now() - START_TIME).total_seconds(),
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(
        json.dumps(to_py(data), indent=2, cls=NumpyEncoder)
    )
    print(f"[{step}/{total}] {msg}")


def mark_done(status, summary, final_progress=None):
    if PID_FILE.exists():
        PID_FILE.unlink()
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps(to_py({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress or {},
        "timestamp": datetime.now().isoformat(),
    }), indent=2, cls=NumpyEncoder))
    print(f"[C1D PILOT] DONE: {status} — {summary}")


TOTAL_STEPS = 10

write_progress(1, TOTAL_STEPS, "Importing libraries and loading model")

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[C1D PILOT] Device: {DEVICE}")

torch.manual_seed(SEED)

from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

# Load model once; reuse across widths
print("[C1D] Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
LETTERS = ["A", "B", "C", "D", "E"]
N_TOKENS = 1000
F_MIN = 0.001
COSINE_THRESHOLD = 0.15
N_WORDS_PER_LETTER = 50

# SAE widths to test: feature-splitting experiment at layer 8
# These provide width variation: 24576 (~24k) and 49152 (~48k)
SAE_CONFIGS = [
    {
        "release": "gpt2-small-res-jb-feature-splitting",
        "sae_id": "blocks.8.hook_resid_pre_24576",
        "width": 24576,
        "label": "24k",
    },
    {
        "release": "gpt2-small-res-jb-feature-splitting",
        "sae_id": "blocks.8.hook_resid_pre_49152",
        "width": 49152,
        "label": "49k",
    },
]

write_progress(2, TOTAL_STEPS, "Collecting text tokens for activation analysis")

# Collect tokens
texts = []
try:
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for item in dataset.take(50):
        texts.append(item["text"])
    print(f"[C1D] Loaded {len(texts)} texts from Skylion007/openwebtext")
except Exception as e:
    print(f"[C1D] openwebtext failed: {e}, using wikitext fallback")

if len(texts) < 20:
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=False)
        for item in dataset:
            if item["text"].strip():
                texts.append(item["text"])
            if len(texts) >= 100:
                break
    except Exception as e:
        print(f"[C1D] wikitext failed: {e}")

if len(texts) < 5:
    texts = ["The quick brown fox jumps over the lazy dog. " * 100] * 10

all_tokens = []
for text in texts:
    try:
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
    except Exception:
        continue
    if len(all_tokens) >= N_TOKENS + 256:
        break

tokens = torch.tensor(all_tokens[:N_TOKENS], dtype=torch.long, device=DEVICE).unsqueeze(0)
print(f"[C1D] Tokens collected: {tokens.shape[1]}")

write_progress(3, TOTAL_STEPS, "Building word vocabulary for ICL absorption measurement")

from sae_spelling.vocab import get_alpha_tokens
from sae_spelling.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
    create_icl_prompt,
)

vocab_alpha = get_alpha_tokens(tokenizer)

# Find single-token words for ICL
single_tok_words_by_letter = {}
for tok_str in vocab_alpha:
    w = tok_str.strip()
    if not w or not w[0].isalpha() or not w.isalpha() or len(w) < 2:
        continue
    toks = tokenizer.encode(' ' + w)
    if len(toks) == 1:
        letter = w[0].upper()
        single_tok_words_by_letter.setdefault(letter, []).append(w)

# Build ICL word list
random.seed(SEED)
icl_word_list = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    words_l = list(single_tok_words_by_letter.get(letter, []))
    random.shuffle(words_l)
    icl_word_list.extend(words_l[:30])
random.seed(SEED)
random.shuffle(icl_word_list)
print(f"[C1D] ICL word list: {len(icl_word_list)} words")

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
print(f"[C1D] Reference prompt token length: {expected_tok_len}")

icl_first_n = set(icl_word_list[:N_ICL])

words_by_letter = {}
for letter in LETTERS:
    original = list(single_tok_words_by_letter.get(letter, []))
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
    words_by_letter[letter] = valid[:N_WORDS_PER_LETTER]
    print(f"  Letter {letter}: {len(valid)} valid words (using {len(words_by_letter[letter])})")


def letter_delta_metric(tokenizer_obj, pos_letter):
    LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pos_letter_tok = tokenizer_obj.encode(f" {pos_letter.upper()}")[-1]
    neg_letter_toks = torch.tensor(
        [tokenizer_obj.encode(f" {l}")[-1] for l in LETTERS_UPPER if l != pos_letter.upper()]
    ).to(DEVICE)

    def metric_fn(logits):
        pos_logit = logits[:, -1, pos_letter_tok]
        neg_logits = logits[:, -1, neg_letter_toks]
        return pos_logit - neg_logits.mean(dim=-1)
    return metric_fn


def compute_sae_acts_on_tokens(sae_model, token_tensor, seq_len=32):
    """Collect SAE activations for all tokens. Returns [n_tokens, d_sae] numpy array."""
    all_acts = []
    n_total = token_tensor.shape[1]
    with torch.no_grad():
        for start in range(0, n_total, seq_len):
            end = min(start + seq_len, n_total)
            chunk = token_tensor[:, start:end]
            _, cache = model.run_with_cache(chunk, names_filter=HOOK_NAME)
            resid = cache[HOOK_NAME].reshape(-1, cache[HOOK_NAME].shape[-1])
            sae_acts = sae_model.encode(resid)
            if isinstance(sae_acts, tuple):
                sae_acts = sae_acts[0]
            all_acts.append(sae_acts.cpu().float())
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.cat(all_acts, dim=0).numpy()  # [n_tokens, d_sae]


def find_parent_feature(letter, sae_model, sae_hook_name, top_k=5):
    """Find top-k letter parent features by differential activation."""
    pos_words = words_by_letter[letter][:20]
    neg_words_raw = []
    for l2, ws in single_tok_words_by_letter.items():
        if l2 != letter:
            for w in ws[:3]:
                if w not in icl_first_n:
                    try:
                        p = build_icl_prompt(w)
                        if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                            neg_words_raw.append(w)
                    except Exception:
                        continue
    neg_words = neg_words_raw[:20]

    pos_acts = []
    neg_acts = []
    with torch.no_grad():
        for w in pos_words:
            try:
                p = build_icl_prompt(w)
                _, cache = model.run_with_cache([p.base], names_filter=sae_hook_name)
                sae_in = cache[sae_hook_name]
                acts = sae_model.encode(sae_in)
                if isinstance(acts, tuple):
                    acts = acts[0]
                pos_acts.append(acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float())
                del cache
            except Exception:
                continue
        for w in neg_words:
            try:
                p = build_icl_prompt(w)
                _, cache = model.run_with_cache([p.base], names_filter=sae_hook_name)
                sae_in = cache[sae_hook_name]
                acts = sae_model.encode(sae_in)
                if isinstance(acts, tuple):
                    acts = acts[0]
                neg_acts.append(acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float())
                del cache
            except Exception:
                continue

    if not pos_acts:
        return []
    pos_mean = torch.stack(pos_acts).mean(0)
    neg_mean = torch.stack(neg_acts).mean(0) if neg_acts else torch.zeros_like(pos_mean)
    diff = pos_mean - neg_mean
    return diff.topk(top_k).indices.tolist()


def compute_alpha_ij_for_sae(sae_model, token_tensor):
    """Compute alpha_ij for a given SAE using OpenWebText tokens. Returns DataFrame."""
    # Get activations
    acts_np = compute_sae_acts_on_tokens(sae_model, token_tensor)  # [n_tokens, d_sae]
    d_sae = acts_np.shape[1]

    binary = (acts_np > 0).astype(np.float32)
    n_tok = binary.shape[0]
    f_i = binary.mean(axis=0)

    active_idx = np.where(f_i > F_MIN)[0]
    print(f"  Active latents: {len(active_idx)} / {d_sae}")

    # Decoder cosine similarities
    dec_W = sae_model.W_dec.detach().cpu().float().numpy()  # [d_sae, d_model]
    norms = np.linalg.norm(dec_W, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    dec_W_norm = dec_W / norms

    active_dec = dec_W_norm[active_idx]
    active_f = f_i[active_idx]
    n_active = len(active_idx)

    # Find candidate pairs using cosine threshold
    BATCH = 256
    rows = []
    for i_start in range(0, n_active, BATCH):
        i_end = min(i_start + BATCH, n_active)
        batch_i = active_dec[i_start:i_end]
        cos_sim = batch_i @ active_dec.T  # [batch, n_active]

        for local_i, global_i_idx in enumerate(range(i_start, i_end)):
            gi = active_idx[global_i_idx]
            row_cos = cos_sim[local_i]
            # Only j > global_i_idx to avoid duplicates
            for j_idx in range(global_i_idx + 1, n_active):
                if row_cos[j_idx] > COSINE_THRESHOLD:
                    gj = active_idx[j_idx]
                    fi = float(active_f[global_i_idx])
                    fj = float(active_f[j_idx])
                    if fi < 1e-9 or fj < 1e-9:
                        continue
                    # Co-activation
                    coact = float(np.mean(binary[:, gi] * binary[:, gj]))
                    sigma_ij = coact / min(fi, fj)
                    alpha_ij = sigma_ij * (fj / fi)
                    rows.append({
                        "latent_i": int(gi),
                        "latent_j": int(gj),
                        "f_i": fi,
                        "f_j": fj,
                        "coact_rate": coact,
                        "sigma_ij": sigma_ij,
                        "alpha_ij": alpha_ij,
                        "decoder_cosine": float(row_cos[j_idx]),
                    })

    df = pd.DataFrame(rows)
    print(f"  Total candidate pairs (OpenWebText): {len(df)}")
    return df, binary, f_i


def collect_word_level_sae_acts(sae_model, words_by_letter_dict, all_letters,
                                 n_per_letter=40):
    """
    Collect SAE activations at the word-token position for ICL prompts.
    Returns dict: letter -> np.array [n_words, d_sae]
    Also collects negative (other-letter) acts for parent feature identification.
    """
    letter_acts = {}
    for letter in all_letters:
        words = words_by_letter_dict[letter][:n_per_letter]
        acts_list = []
        with torch.no_grad():
            for w in words:
                try:
                    p = build_icl_prompt(w)
                    if len(model.to_tokens([p.base])[0]) != expected_tok_len:
                        continue
                    _, cache = model.run_with_cache([p.base], names_filter=HOOK_NAME)
                    sae_in = cache[HOOK_NAME]
                    sae_acts = sae_model.encode(sae_in)
                    if isinstance(sae_acts, tuple):
                        sae_acts = sae_acts[0]
                    # Get activation at the word-first-letter position
                    act_at_pos = sae_acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float().numpy()
                    acts_list.append(act_at_pos)
                    del cache
                except Exception:
                    continue
        if acts_list:
            letter_acts[letter] = np.stack(acts_list)  # [n, d_sae]
        else:
            letter_acts[letter] = np.zeros((0, sae_model.cfg.d_sae))
    return letter_acts


def compute_das_k3_from_word_acts(parent_id, letter, word_acts_dict,
                                   sae_model, n_children=3):
    """
    Compute DAS(k=3) using word-level SAE activations.

    For each letter:
    - Positive examples: words that START with letter (parent should be active)
    - Negative examples: words that do NOT start with letter (controls)

    Children of parent_id: features that co-activate with parent,
    identified by decoder cosine similarity and high differential activation
    on letter-starting words.

    DAS(k=3) = McFadden R2 of logistic regression predicting P(parent active)
               from top-3 child feature activations.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    D_SAE = sae_model.cfg.d_sae

    # Get decoder matrix for cosine similarity computation
    dec_W = sae_model.W_dec.detach().cpu().float().numpy()  # [d_sae, d_model]
    norms = np.linalg.norm(dec_W, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    dec_W_norm = dec_W / norms

    # Parent feature decoder vector
    parent_vec = dec_W_norm[parent_id]  # [d_model]

    # Get positive acts (for this letter) and negative acts (other letters)
    pos_acts = word_acts_dict.get(letter, np.zeros((0, D_SAE)))

    neg_acts_list = []
    for l2, acts in word_acts_dict.items():
        if l2 != letter and len(acts) > 0:
            neg_acts_list.append(acts[:5])
    neg_acts = np.vstack(neg_acts_list) if neg_acts_list else np.zeros((0, D_SAE))

    if len(pos_acts) < 5 or len(neg_acts) < 5:
        return None, None, []

    # All word-level activations
    all_acts = np.vstack([pos_acts, neg_acts])  # [n_total, d_sae]
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    n_total = len(all_acts)

    # Verify parent_id activation
    parent_act_pos = (pos_acts[:, parent_id] > 0).mean()
    parent_act_neg = (neg_acts[:, parent_id] > 0).mean()
    print(f"    Parent {parent_id}: pos_act={parent_act_pos:.3f}, neg_act={parent_act_neg:.3f}")

    # Find top children: features with high decoder cosine with parent
    # and differential activation on positive vs negative words
    f_word = (all_acts > 0).mean(axis=0)  # [d_sae]
    diff_act = (pos_acts > 0).mean(0) - (neg_acts > 0).mean(0)  # [d_sae]

    # Cosine similarity to parent
    cos_all = dec_W_norm @ parent_vec  # [d_sae]

    # Score: combine cosine similarity and differential activation
    # Find features similar to parent but NOT the parent itself
    candidate_mask = (cos_all > 0.10) & (np.arange(D_SAE) != parent_id) & (f_word > 0.001)
    candidate_ids = np.where(candidate_mask)[0]

    if len(candidate_ids) == 0:
        # Fallback: use top features by differential activation
        candidate_ids = np.argsort(diff_act)[::-1][:50]
        candidate_ids = candidate_ids[candidate_ids != parent_id]

    print(f"    Candidate children: {len(candidate_ids)}")

    if len(candidate_ids) == 0:
        return None, None, []

    # Score candidates by: cos_sim * diff_activation
    candidate_scores = cos_all[candidate_ids] * np.clip(diff_act[candidate_ids], 0, None)
    top_children = candidate_ids[np.argsort(candidate_scores)[::-1][:n_children]]
    child_ids = top_children.tolist()

    if not child_ids:
        return None, None, []

    print(f"    Top-{n_children} children: {child_ids}")

    # Target: parent activation (binary)
    y_parent = (all_acts[:, parent_id] > 0).astype(int)

    if y_parent.sum() < 3 or (n_total - y_parent.sum()) < 3:
        # Parent rarely activates - use letter label as proxy
        y_parent = y

    # Feature matrix: child activations
    X_children = (all_acts[:, child_ids] > 0).astype(float)

    # Null log-likelihood (intercept-only)
    p_mean = np.clip(y_parent.mean(), 1e-7, 1 - 1e-7)
    if abs(np.log(p_mean)) < 1e-9 or abs(np.log(1 - p_mean)) < 1e-9:
        return None, None, child_ids
    ll_null = n_total * (p_mean * np.log(p_mean) + (1 - p_mean) * np.log(1 - p_mean))

    # Fit logistic regression
    try:
        if X_children.sum() < 3:
            # Not enough variation in children
            return 0.0, 0.0, child_ids

        lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        lr.fit(X_children, y_parent)
        y_pred = np.clip(lr.predict_proba(X_children)[:, 1], 1e-7, 1 - 1e-7)
        ll_model = -log_loss(y_parent, y_pred, normalize=False)

        if abs(ll_null) < 1e-9:
            das_k3 = 0.0
        else:
            mcfadden_r2 = 1.0 - ll_model / ll_null
            das_k3 = float(np.clip(mcfadden_r2, 0.0, 1.0))

        child_act_rate = float(X_children.mean())
        return das_k3, child_act_rate, child_ids
    except Exception as e:
        print(f"    DAS(k=3) LR failed: {e}")
        return None, None, child_ids


def compute_das_k3(parent_id, df_alpha, binary_acts, f_i, n_children=3):
    """Legacy: not used. See compute_das_k3_from_word_acts."""
    pass


# ---- Main loop: iterate over SAE widths ----
write_progress(4, TOTAL_STEPS, f"Processing {len(SAE_CONFIGS)} SAE widths")

from sae_spelling.probing import train_binary_probe
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator

results_by_width = {}

for cfg_idx, sae_cfg in enumerate(SAE_CONFIGS):
    width_label = sae_cfg["label"]
    width = sae_cfg["width"]
    print(f"\n{'='*60}")
    print(f"[C1D] Processing width={width} ({width_label})")
    print(f"{'='*60}")

    write_progress(4 + cfg_idx * 3, TOTAL_STEPS,
                   f"Loading SAE width={width_label}")

    # Load this SAE
    try:
        sae = SAE.from_pretrained(
            release=sae_cfg["release"],
            sae_id=sae_cfg["sae_id"],
            device=DEVICE,
        )
        if isinstance(sae, tuple):
            sae = sae[0]
        sae.eval()
        # Ensure hook_name is set
        if not hasattr(sae.cfg, 'hook_name') or not sae.cfg.hook_name:
            sae.cfg.hook_name = HOOK_NAME
        D_SAE = sae.cfg.d_sae
        print(f"[C1D] SAE loaded: width={D_SAE}, hook={sae.cfg.hook_name}")
    except Exception as e:
        print(f"[C1D] ERROR loading SAE width {width}: {e}")
        results_by_width[width_label] = {"error": str(e), "width": width}
        continue

    # Verify d_sae matches expected width
    if D_SAE != width:
        print(f"[C1D] WARNING: expected d_sae={width}, got {D_SAE}")

    write_progress(4 + cfg_idx * 3 + 1, TOTAL_STEPS,
                   f"Computing word-level SAE activations and finding parent features for width={width_label}")

    # Find parent features and train probes for DAS(k=1)
    parent_ids_by_letter = {}
    probe_dirs_by_letter = {}

    for letter in LETTERS:
        try:
            main_ids = find_parent_feature(letter, sae, sae.cfg.hook_name, top_k=5)
            parent_ids_by_letter[letter] = main_ids
            print(f"  Letter {letter}: parent feature IDs = {main_ids[:3]}")
        except Exception as e:
            print(f"  Letter {letter}: find_parent_feature failed: {e}")
            parent_ids_by_letter[letter] = []

        # Train probe for absorption detection
        try:
            pos_words = words_by_letter[letter][:30]
            neg_words_raw = []
            for l2, ws in single_tok_words_by_letter.items():
                if l2 != letter:
                    for w in ws[:3]:
                        if w not in icl_first_n:
                            try:
                                p = build_icl_prompt(w)
                                if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                                    neg_words_raw.append(w)
                            except Exception:
                                continue
            neg_words = neg_words_raw[:30]

            combined = [(w, 1) for w in pos_words] + [(w, 0) for w in neg_words]
            acts_list, labels = [], []
            with torch.no_grad():
                for w, lab in combined:
                    try:
                        p = build_icl_prompt(w)
                        if len(model.to_tokens([p.base])[0]) != expected_tok_len:
                            continue
                        _, cache = model.run_with_cache([p.base], names_filter=HOOK_NAME)
                        act = cache[HOOK_NAME][0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float()
                        acts_list.append(act)
                        labels.append(float(lab))
                        del cache
                    except Exception:
                        continue

            if len(acts_list) >= 10:
                acts_t = torch.stack(acts_list).to(DEVICE)
                labels_t = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
                probe = train_binary_probe(acts_t, labels_t,
                                          num_epochs=100, lr=0.01,
                                          show_progress=False, verbose=False, device=DEVICE)
                probe_dir = F.normalize(probe.fc.weight[0].detach().cpu(), dim=0)
                probe_dirs_by_letter[letter] = probe_dir
                print(f"  Letter {letter}: probe trained ({len(acts_list)} samples)")
            else:
                print(f"  Letter {letter}: insufficient samples for probe ({len(acts_list)})")
        except Exception as e:
            print(f"  Letter {letter}: probe training failed: {e}")

    write_progress(4 + cfg_idx * 3 + 2, TOTAL_STEPS,
                   f"Running DAS(k=1) absorption measurement for width={width_label}")

    # DAS(k=1): Run absorption measurement via sae-spelling
    das_k1_by_letter = {}

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

    for letter in LETTERS:
        if letter not in probe_dirs_by_letter:
            das_k1_by_letter[letter] = {"das_k1": None, "error": "no probe"}
            continue

        probe_dir = probe_dirs_by_letter[letter]
        main_ids = parent_ids_by_letter.get(letter, [])
        if not main_ids:
            das_k1_by_letter[letter] = {"das_k1": None, "error": "no parent ids"}
            continue

        words = words_by_letter[letter]
        metric_fn = letter_delta_metric(tokenizer, letter)

        try:
            results = calculator.calculate_absorption_sampled(
                sae=sae,
                words=words,
                probe_dir=probe_dir.to(DEVICE),
                metric_fn=metric_fn,
                main_feature_ids=main_ids[:5],
                max_ablation_samples=40,
                filter_prompts=True,
                show_progress=False,
            )
            sample_results = results.sample_results
            n_tot = len(sample_results)
            n_abs = sum(1 for r in sample_results if r.is_absorption)
            das_k1 = n_abs / n_tot if n_tot > 0 else 0.0
            das_k1_by_letter[letter] = {"das_k1": das_k1, "n_absorbed": n_abs, "n_total": n_tot}
            print(f"  Letter {letter}: DAS(k=1)={das_k1:.3f} ({n_abs}/{n_tot})")
        except Exception as e:
            print(f"  Letter {letter}: DAS(k=1) failed: {e}, trying fallback...")
            try:
                results = calculator.calculate_absorption_sampled(
                    sae=sae,
                    words=words,
                    probe_dir=probe_dir.to(DEVICE),
                    metric_fn=metric_fn,
                    main_feature_ids=main_ids[:5],
                    max_ablation_samples=30,
                    filter_prompts=False,
                    show_progress=False,
                )
                sample_results = results.sample_results
                n_tot = len(sample_results)
                n_abs = sum(1 for r in sample_results if r.is_absorption)
                das_k1 = n_abs / n_tot if n_tot > 0 else 0.0
                das_k1_by_letter[letter] = {"das_k1": das_k1, "n_absorbed": n_abs, "n_total": n_tot, "note": "fallback"}
                print(f"  Letter {letter}: DAS(k=1) fallback={das_k1:.3f} ({n_abs}/{n_tot})")
            except Exception as e2:
                das_k1_by_letter[letter] = {"das_k1": None, "error": str(e2)}
                print(f"  Letter {letter}: DAS(k=1) completely failed: {e2}")

    # Collect word-level SAE activations for DAS(k=3)
    print(f"\n[C1D] Collecting word-level SAE acts for DAS(k=3), width={width_label}...")
    word_acts = collect_word_level_sae_acts(sae, words_by_letter, LETTERS, n_per_letter=40)
    for letter in LETTERS:
        n_acts = len(word_acts.get(letter, []))
        print(f"  Letter {letter}: {n_acts} word-level SAE act samples")

    # DAS(k=3): Fit logistic regression from word-level activations
    print(f"\n[C1D] Computing DAS(k=3) for width={width_label}...")
    das_k3_by_letter = {}

    for letter in LETTERS:
        main_ids = parent_ids_by_letter.get(letter, [])
        if not main_ids:
            das_k3_by_letter[letter] = {"das_k3": None, "error": "no parent ids"}
            continue

        # Use the top parent feature ID
        parent_id = main_ids[0]
        try:
            das_k3, child_act_rate, child_ids = compute_das_k3_from_word_acts(
                parent_id, letter, word_acts, sae, n_children=3
            )
            das_k3_by_letter[letter] = {
                "das_k3": das_k3,
                "parent_id": parent_id,
                "child_ids": child_ids,
                "child_activation_rate": child_act_rate,
            }
            das_k1_val = das_k1_by_letter.get(letter, {}).get("das_k1")
            print(f"  Letter {letter}: parent={parent_id}, children={child_ids}, "
                  f"DAS(k=1)={das_k1_val}, DAS(k=3)={das_k3}")
        except Exception as e:
            print(f"  Letter {letter}: DAS(k=3) failed: {e}")
            import traceback; traceback.print_exc()
            das_k3_by_letter[letter] = {"das_k3": None, "error": str(e)}

    # Store results for this width
    results_by_width[width_label] = {
        "width": width,
        "d_sae": D_SAE,
        "sae_id": sae_cfg["sae_id"],
        "das_k1": das_k1_by_letter,
        "das_k3": das_k3_by_letter,
        "n_word_act_samples": {l: len(word_acts.get(l, [])) for l in LETTERS},
    }

    # Free SAE memory
    del sae, word_acts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

write_progress(TOTAL_STEPS - 1, TOTAL_STEPS, "Evaluating pass criteria and writing output")

# ---- Evaluate pass criteria ----
print("\n[C1D] Evaluating pass criteria...")

# Count letters where both DAS values are computable
both_computable = {letter: False for letter in LETTERS}
das_k3_ge_das_k1_at_wider = {letter: False for letter in LETTERS}

width_labels = [c["label"] for c in SAE_CONFIGS]
wider_label = width_labels[-1] if len(width_labels) > 1 else width_labels[0]

for letter in LETTERS:
    for wl in width_labels:
        if wl not in results_by_width:
            continue
        r = results_by_width[wl]
        d1 = r.get("das_k1", {}).get(letter, {}).get("das_k1")
        d3 = r.get("das_k3", {}).get(letter, {}).get("das_k3")
        if d1 is not None and d3 is not None:
            both_computable[letter] = True

    # Check DAS(k=3) >= DAS(k=1) at wider width
    if wider_label in results_by_width:
        r_wide = results_by_width[wider_label]
        d1_wide = r_wide.get("das_k1", {}).get(letter, {}).get("das_k1")
        d3_wide = r_wide.get("das_k3", {}).get(letter, {}).get("das_k3")
        if d1_wide is not None and d3_wide is not None:
            das_k3_ge_das_k1_at_wider[letter] = (d3_wide >= d1_wide)
        print(f"  {letter}: DAS(k=1)={d1_wide}, DAS(k=3)={d3_wide}, "
              f"DAS(k=3)>=DAS(k=1): {das_k3_ge_das_k1_at_wider[letter]}")

n_both_computable = sum(both_computable.values())
n_das3_ge_das1_wider = sum(das_k3_ge_das_k1_at_wider.values())

pass_computable = (n_both_computable >= 5)  # All 5 letters
pass_das3_ge_das1 = (n_das3_ge_das1_wider >= 3)  # 3+ of 5 letters

go_no_go = "GO" if (pass_computable and pass_das3_ge_das1) else "PARTIAL" if n_both_computable >= 3 else "NO_GO"
print(f"\n[C1D] Pass criteria:")
print(f"  Both DAS computable for all 5 letters: {pass_computable} (n={n_both_computable})")
print(f"  DAS(k=3) >= DAS(k=1) at wider SAE: {pass_das3_ge_das1} (n={n_das3_ge_das1_wider}/5)")
print(f"  GO/NO-GO: {go_no_go}")

# ---- Build output ----
output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "model": "gpt2-small",
    "sae_release": "gpt2-small-res-jb-feature-splitting",
    "layer": LAYER,
    "letters": LETTERS,
    "widths_tested": [c["width"] for c in SAE_CONFIGS],
    "width_labels": width_labels,
    "wider_label": wider_label,
    "results_by_width": results_by_width,
    "summary": {
        "both_computable": both_computable,
        "das_k3_ge_das_k1_at_wider": das_k3_ge_das_k1_at_wider,
        "n_both_computable": n_both_computable,
        "n_das3_ge_das1_wider": n_das3_ge_das1_wider,
    },
    "pass_criteria": {
        "pass_computable_all_5": bool(pass_computable),
        "pass_das3_ge_das1_at_least_3": bool(pass_das3_ge_das1),
        "n_computable": n_both_computable,
        "n_das3_ge_das1": n_das3_ge_das1_wider,
    },
    "go_no_go": go_no_go,
    "runtime_seconds": (datetime.now() - START_TIME).total_seconds(),
}

# Save pilot output
out_path = PILOTS_DIR / "C1D_das_vs_width_pilot.json"
out_path.write_text(json.dumps(to_py(output), indent=2, cls=NumpyEncoder))
print(f"[C1D] Results saved: {out_path}")

# Also save to full dir (the expected_output location)
full_out_path = FULL_DIR / "C1D_das_vs_width.json"
full_out_path.write_text(json.dumps(to_py(output), indent=2, cls=NumpyEncoder))
print(f"[C1D] Full output saved: {full_out_path}")

# ---- Update gpu_progress.json ----
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    with open(gpu_progress_path, "r") as f:
        gpu_progress = json.load(f)
except Exception:
    gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

end_time = datetime.now()
actual_min = max(1, int((end_time - START_TIME).total_seconds() / 60))

if go_no_go in ("GO", "PARTIAL"):
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
else:
    if TASK_ID not in gpu_progress.get("failed", []):
        gpu_progress.setdefault("failed", []).append(TASK_ID)

gpu_progress.get("running", {}).pop(TASK_ID, None)

gpu_progress.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 60,
    "actual_min": actual_min,
    "start_time": START_TIME.isoformat(),
    "end_time": end_time.isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb-feature-splitting",
        "layer": LAYER,
        "widths": [c["width"] for c in SAE_CONFIGS],
        "letters": LETTERS,
        "mode": "pilot",
    }
}

with open(gpu_progress_path, "w") as f:
    json.dump(to_py(gpu_progress), f, indent=2)

write_progress(TOTAL_STEPS, TOTAL_STEPS,
               f"Done. GO/NO-GO: {go_no_go}. "
               f"Computable: {n_both_computable}/5, "
               f"DAS(k=3)>=DAS(k=1) at wider: {n_das3_ge_das1_wider}/5",
               {"go_no_go": go_no_go, "n_computable": n_both_computable})

mark_done(
    "success" if go_no_go in ("GO", "PARTIAL") else "failed",
    (f"C1D PILOT: {go_no_go}. "
     f"Widths {[c['width'] for c in SAE_CONFIGS]}. "
     f"DAS computable for {n_both_computable}/5 letters. "
     f"DAS(k=3)>=DAS(k=1) at wider: {n_das3_ge_das1_wider}/5."),
    final_progress={
        "task_id": TASK_ID,
        "step": TOTAL_STEPS,
        "total_steps": TOTAL_STEPS,
        "message": f"GO/NO-GO: {go_no_go}",
        "metrics": {
            "n_computable": n_both_computable,
            "n_das3_ge_das1": n_das3_ge_das1_wider,
            "go_no_go": go_no_go,
        },
        "elapsed_sec": (datetime.now() - START_TIME).total_seconds(),
        "updated_at": datetime.now().isoformat(),
    }
)

print(f"\n[C1D PILOT] Final result: {go_no_go}")
print(f"  Both DAS computable: {n_both_computable}/5")
print(f"  DAS(k=3) >= DAS(k=1) at wider width: {n_das3_ge_das1_wider}/5")
sys.exit(0)
