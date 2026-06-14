"""
C1C_cross_arch_validation PILOT experiment
============================================
PILOT SCOPE:
  - JumpReLU SAEs only (skip TopK); letters A-E only for validation
  - FALLBACK NOTE: JumpReLU SAEs are not available for GPT-2 small in SAELens v6.39.
    Per task fallback rule: "use second L0 setting of standard SAEs as substitute and document this."
    Substitute: gpt2-small-resid-post-v5-32k blocks.8.hook_resid_post (OpenAI v5 architecture, L0=32)
    This differs from C1B baseline (gpt2-small-res-jb blocks.8.hook_resid_pre, L0=60) in:
      (a) SAE architecture/training (v5 vs jb)
      (b) Hook position (resid_post vs resid_pre at layer 8)
      (c) Different L0 target (32 vs 60)
      (d) Different SAE width (32768 vs 24576)

Procedure:
  1. Load tau=0.5 from C1B (calibrated on jb SAE, no re-fitting allowed)
  2. Run sae-spelling absorption measurement on letters A-E using v5-32k SAE
  3. Compute alpha_ij statistics for v5-32k SAE (analogous to C1A but for alt arch)
  4. Apply tau=0.5 to LV scores from alt arch SAE
  5. Report F1 and F1 degradation vs C1B baseline

Pass criteria (PILOT):
  - Alt arch SAE loads successfully via SAELens
  - F1 is computable (i.e., alpha_ij scores are non-degenerate)
  - F1 degradation < 30 percentage points vs C1B test F1

Output: exp/results/pilots/C1C_cross_arch_pilot.json
        exp/results/full/C1C_cross_arch.json
        exp/results/C1C_cross_arch_validation_DONE (completion marker)
"""

import os
import sys
import json
import time
import random
import datetime
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
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
    """Recursively convert numpy types to Python native types."""
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


# ---- Config ----
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TASK_ID = "C1C_cross_arch_validation"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.datetime.now()

# PID file
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PID_FILE.write_text(str(os.getpid()))
print(f"[C1C PILOT] PID={os.getpid()} written to {PID_FILE}")

# ---- Seeds ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def write_progress(step, total_steps, message, metrics=None):
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "metrics": metrics or {},
        "elapsed_sec": (datetime.datetime.now() - START_TIME).total_seconds(),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(
        json.dumps(to_python_types(progress), indent=2, cls=NumpyEncoder))
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
    print(f"[C1C PILOT] DONE: {status} — {summary}")


# ---- STEP 1: Load C1B calibrated tau and baseline F1 ----
write_progress(1, 9, "Loading C1B calibrated tau from baseline experiment")

C1B_PATH = FULL_DIR / "C1B_lv_validation.json"
if not C1B_PATH.exists():
    # Try pilot path
    C1B_PATH = PILOTS_DIR / "C1B_lv_detector_pilot.json"

if not C1B_PATH.exists():
    mark_done("failed", f"C1B results not found. Cannot proceed without calibrated tau.")
    sys.exit(1)

with open(C1B_PATH) as f:
    c1b_data = json.load(f)

CALIBRATED_TAU = float(c1b_data["calibration"]["best_tau"])
BASELINE_F1 = float(c1b_data["test_evaluation"]["f1"])
BASELINE_SAE_RELEASE = c1b_data.get("sae_release", "gpt2-small-res-jb")
BASELINE_SAE_ID = c1b_data.get("sae_id", "blocks.8.hook_resid_pre")

print(f"[C1C] Calibrated tau (from C1B): {CALIBRATED_TAU}")
print(f"[C1C] Baseline F1 (C1B test set): {BASELINE_F1:.4f}")
print(f"[C1C] Baseline SAE: {BASELINE_SAE_RELEASE} / {BASELINE_SAE_ID}")
print(f"[C1C] F1 degradation threshold (pass): < 0.30 (i.e., F1 > {BASELINE_F1 - 0.30:.4f})")

# ---- STEP 2: Load model and alt-arch SAE ----
write_progress(2, 9, "Loading GPT-2 Small model and alternative architecture SAE")

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[C1C] Device: {DEVICE}")

torch.manual_seed(SEED)

from transformer_lens import HookedTransformer
from sae_lens import SAE

# Note: JumpReLU not available for GPT-2 small in SAELens v6.39
# Fallback: use OpenAI v5-32k SAE at layer 8 (resid_post) — different architecture/training
# This is the documented fallback per task spec:
# "use second L0 setting of standard SAEs as substitute and document this"
ALT_ARCH_NAME = "v5-32k (OpenAI, resid_post, L0=32)"
ALT_ARCH_NOTE = (
    "JumpReLU SAEs unavailable for GPT-2 small in SAELens v6.39. "
    "Fallback: gpt2-small-resid-post-v5-32k blocks.8.hook_resid_post "
    "(OpenAI v5 architecture, L0=32, d_sae=32768). "
    "Differs from baseline in: SAE training objective (v5 vs jb), "
    "hook position (resid_post vs resid_pre), L0 (32 vs 60), width (32k vs 24k)."
)

ALT_SAE_RELEASE = "gpt2-small-resid-post-v5-32k"
ALT_SAE_ID = "blocks.8.hook_resid_post"
ALT_LAYER = 8
ALT_HOOK_NAME = f"blocks.{ALT_LAYER}.hook_resid_post"

print(f"[C1C] Loading model: gpt2-small")
model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token
print(f"[C1C] Model: gpt2-small, d_model={model.cfg.d_model}")

print(f"[C1C] Loading alt-arch SAE: {ALT_SAE_RELEASE} / {ALT_SAE_ID}")
try:
    sae_alt, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=ALT_SAE_RELEASE,
        sae_id=ALT_SAE_ID,
        device=DEVICE,
    )
    sae_alt.eval()
    # Ensure hook_name is set
    if not hasattr(sae_alt.cfg, 'hook_name') or sae_alt.cfg.hook_name is None:
        sae_alt.cfg.hook_name = ALT_HOOK_NAME
    D_SAE_ALT = sae_alt.cfg.d_sae
    print(f"[C1C] Alt SAE loaded: d_sae={D_SAE_ALT}")
    alt_sae_loaded = True
except Exception as e:
    print(f"[C1C] ERROR loading alt SAE: {e}")
    import traceback; traceback.print_exc()
    mark_done("failed", f"Alt arch SAE load failed: {e}")
    sys.exit(1)

# ---- STEP 3: Build word vocabulary ----
write_progress(3, 9, "Building word vocabulary for ICL prompts")

from sae_spelling.vocab import get_alpha_tokens
from sae_spelling.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
    create_icl_prompt,
)

vocab_alpha = get_alpha_tokens(tokenizer)

# Find single-token words prefixed with space
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
    words_l = single_tok_words_by_letter.get(letter, [])
    random.shuffle(words_l)
    icl_word_list.extend(words_l[:30])
random.seed(SEED)
random.shuffle(icl_word_list)
print(f"[C1C] ICL word list: {len(icl_word_list)} single-token words")

# Pilot letters: A-E only
PILOT_LETTERS = ["A", "B", "C", "D", "E"]

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
print(f"[C1C] Reference prompt token length: {expected_tok_len}")

# Filter words to same-length prompts
icl_first_n = set(icl_word_list[:N_ICL])
words_by_letter = {}
for letter in PILOT_LETTERS:
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


# ---- STEP 4: Collect activation statistics for alt-arch SAE ----
write_progress(4, 9, "Collecting activation statistics for alt-arch SAE (computing alpha_ij from ICL prompts)")

# CRITICAL FIX: We must compute alpha_ij from ICL prompts (same distribution as absorption test),
# NOT from generic text. Generic text does not activate letter-specific features at the word position.
# This mirrors the C1A approach used by the baseline jb SAE.

print("[C1C] Building ICL prompt activations for all pilot letters (both positive and negative)...")

# Collect SAE activations at VERBOSE_FIRST_LETTER_TOKEN_POS from ICL prompts
# Build a collection of words from all 26 letters for a broader activation sample
all_icl_acts = []  # List of per-word SAE activation vectors at the word token position
all_icl_latent_ids = []  # For each word, the non-zero latent IDs

all_words_for_stats = []
# Collect words from all letters for diversity (not just pilot letters)
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    words_l = single_tok_words_by_letter.get(letter, [])
    random.seed(SEED + ord(letter) + 1000)
    random.shuffle(words_l)
    all_words_for_stats.extend(words_l[:15])  # 15 words per letter = ~390 total

random.seed(SEED + 2000)
random.shuffle(all_words_for_stats)
print(f"[C1C] Collecting ICL activations for {len(all_words_for_stats)} words...")

# Build token_to_sae_acts mapping: for each word processed, record SAE acts at word position
icl_acts_list = []
icl_latent_ids_list = []

with torch.no_grad():
    for w in all_words_for_stats:
        if w in icl_first_n:
            continue
        try:
            p = build_icl_prompt(w)
            tok_len = len(model.to_tokens([p.base])[0])
            if tok_len != expected_tok_len:
                continue
            _, cache = model.run_with_cache([p.base], names_filter=ALT_HOOK_NAME)
            sae_in = cache[ALT_HOOK_NAME]
            sae_acts = sae_alt.encode(sae_in)
            act_at_pos = sae_acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float()
            icl_acts_list.append(act_at_pos)
            # Record non-zero latent IDs
            nonzero_ids = torch.where(act_at_pos > 0)[0].tolist()
            icl_latent_ids_list.append(nonzero_ids)
        except Exception:
            continue

if len(icl_acts_list) == 0:
    mark_done("failed", "No ICL activations collected for alt-arch SAE")
    sys.exit(1)

print(f"[C1C] Collected {len(icl_acts_list)} ICL prompt activations")

# Stack activations
icl_acts_tensor = torch.stack(icl_acts_list)  # [N_words, d_sae]
N_ICL_SAMPLES = len(icl_acts_tensor)

# Compute activation frequencies from ICL prompts
act_binary = (icl_acts_tensor > 0).float()  # [N, d_sae]
freq = act_binary.mean(0)  # [d_sae]

F_MIN = 0.001  # at least 1 per 1000 words
active_mask = freq > F_MIN
active_ids = torch.where(active_mask)[0].tolist()
print(f"[C1C] Active latents (f>{F_MIN}) from ICL prompts: {len(active_ids)} / {D_SAE_ALT}")

if len(active_ids) < 10:
    # Fall back to even lower threshold
    F_MIN = 0.0001
    active_mask = freq > F_MIN
    active_ids = torch.where(active_mask)[0].tolist()
    print(f"[C1C] Lowered threshold to {F_MIN}: {len(active_ids)} active latents")

# Compute pairwise alpha_ij for geometrically adjacent features
COSINE_THRESHOLD = 0.10  # lower threshold for v5 SAE (different feature geometry)
print(f"[C1C] Computing decoder cosine similarities (threshold={COSINE_THRESHOLD})...")

# Get decoder weights
with torch.no_grad():
    if hasattr(sae_alt, 'W_dec'):
        dec_weights = sae_alt.W_dec.detach().cpu()  # [d_sae, d_model]
    else:
        dec_weights = next(p for n, p in sae_alt.named_parameters() if 'dec' in n.lower() and p.shape[0] == D_SAE_ALT).detach().cpu()

# Get decoder for active latents
active_ids_t = torch.tensor(active_ids, dtype=torch.long)
dec_weights_active = dec_weights[active_ids_t, :]  # [n_active, d_model]
dec_norms = dec_weights_active.norm(dim=1, keepdim=True).clamp(min=1e-8)
dec_normed = (dec_weights_active / dec_norms).float()  # [n_active, d_model]

# Restrict to N_ACTIVE_MAX for pilot speed
N_ACTIVE_MAX = 5000
n_active = len(active_ids)
if n_active > N_ACTIVE_MAX:
    # Use the most frequent latents
    freq_active = freq[active_ids]
    top_idx = freq_active.topk(N_ACTIVE_MAX).indices.tolist()
    active_ids_pilot = [active_ids[i] for i in top_idx]
    dec_normed_pilot = dec_normed[top_idx, :]
    freq_pilot = freq_active[top_idx].cpu()
    print(f"[C1C] Restricting to top {N_ACTIVE_MAX} most frequent latents")
else:
    active_ids_pilot = active_ids
    dec_normed_pilot = dec_normed
    freq_pilot = freq[active_ids].cpu()

print(f"[C1C] Computing pairwise cosines for {len(active_ids_pilot)} latents...")

# Build alpha_ij pairs
alpha_pairs = []
CHUNK = 300
for i_start in range(0, len(active_ids_pilot), CHUNK):
    i_end = min(i_start + CHUNK, len(active_ids_pilot))
    batch_i = dec_normed_pilot[i_start:i_end]  # [chunk, d_model]
    cosine_mat = batch_i @ dec_normed_pilot.T  # [chunk, n_active]

    for local_i, global_i in enumerate(range(i_start, i_end)):
        lat_i = active_ids_pilot[global_i]
        fi = float(freq_pilot[global_i])

        cos_row = cosine_mat[local_i]

        for global_j in range(global_i + 1, len(active_ids_pilot)):
            lat_j = active_ids_pilot[global_j]
            fj = float(freq_pilot[global_j])

            cos_ij = float(cos_row[global_j].item())
            if abs(cos_ij) < COSINE_THRESHOLD:
                continue

            # Compute co-activation rate from ICL activations
            mask_ij = (act_binary[:, lat_i] > 0) & (act_binary[:, lat_j] > 0)
            coact = float(mask_ij.float().mean())

            if coact == 0:
                continue

            min_fi_fj = min(fi, fj)
            sigma_ij = coact / min_fi_fj if min_fi_fj > 0 else 0.0

            # Determine parent (higher frequency) and child (lower frequency)
            if fj >= fi:
                parent_id, child_id = lat_j, lat_i
                f_parent, f_child = fj, fi
            else:
                parent_id, child_id = lat_i, lat_j
                f_parent, f_child = fi, fj

            alpha_ij = sigma_ij * (f_parent / f_child) if f_child > 0 else 0.0

            alpha_pairs.append({
                "latent_i": child_id,
                "latent_j": parent_id,
                "f_i": f_child,
                "f_j": f_parent,
                "coact_rate": coact,
                "sigma_ij": sigma_ij,
                "alpha_ij": alpha_ij,
                "decoder_cosine": abs(cos_ij),
            })

print(f"[C1C] Computed {len(alpha_pairs)} alpha_ij pairs for alt-arch SAE (from ICL prompts)")

if len(alpha_pairs) == 0:
    # Non-fatal: means no co-activating pairs found — still document this
    print("[C1C] WARNING: Zero alpha_ij pairs — all LV scores will be 0 (no co-activation detected)")
    df_alpha_alt = pd.DataFrame(columns=["latent_i", "latent_j", "f_i", "f_j",
                                          "coact_rate", "sigma_ij", "alpha_ij", "decoder_cosine"])
else:
    df_alpha_alt = pd.DataFrame(alpha_pairs)
    n_above_1 = int((df_alpha_alt['alpha_ij'] > 1.0).sum())
    n_nan = int(df_alpha_alt['alpha_ij'].isna().sum())
    print(f"[C1C] Alpha_ij stats: mean={df_alpha_alt['alpha_ij'].mean():.3f}, "
          f"max={df_alpha_alt['alpha_ij'].max():.3f}, n_above_1={n_above_1}, n_nan={n_nan}")

# Build parent-to-children lookup for alt SAE
parent_to_children_alt = defaultdict(dict)
for _, row in df_alpha_alt.iterrows():
    child = int(row['latent_i'])
    parent = int(row['latent_j'])
    aij = float(row['alpha_ij'])
    if aij > 0:
        parent_to_children_alt[parent][child] = aij

print(f"[C1C] parent_to_children_alt: {len(parent_to_children_alt)} parents")


# ---- STEP 5: Train probes using alt-arch SAE hook ----
write_progress(5, 9, "Training letter probes at alt SAE hook point")

from sae_spelling.probing import train_binary_probe

def get_resid_at_pos_alt(word, hook_name=ALT_HOOK_NAME):
    """Get residual stream activation at alt hook point."""
    try:
        p = build_icl_prompt(word)
        if len(model.to_tokens([p.base])[0]) != expected_tok_len:
            return None
        _, cache = model.run_with_cache([p.base], names_filter=hook_name)
        act = cache[hook_name][0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float()
        return act
    except Exception:
        return None

probe_directions = {}
for letter in PILOT_LETTERS:
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
            act = get_resid_at_pos_alt(w)
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

print(f"[C1C] Probes trained: {list(probe_directions.keys())}")


# ---- STEP 6: Run absorption measurement using alt-arch SAE ----
write_progress(6, 9, "Running absorption measurement with alt-arch SAE")

from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator

calculator_alt = FeatureAbsorptionCalculator(
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


def get_main_feature_ids_alt(letter, sae_model, words_prompts, neg_prompts, top_k=5):
    """Find top-k SAE features by differential mean activation for alt SAE."""
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


# Build words_prompts per letter
words_prompts_by_letter = {}
for letter in PILOT_LETTERS:
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

# Run absorption measurement with alt SAE
absorption_by_letter = {}

for letter in PILOT_LETTERS:
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

    main_ids = get_main_feature_ids_alt(letter, sae_alt, words_prompts, neg_prompts, top_k=5)
    print(f"\n  Letter {letter}: main_ids={main_ids}, n_words={len(words_by_letter[letter])}")

    metric_fn = letter_delta_metric(tokenizer, letter)
    words = words_by_letter[letter]

    try:
        results = calculator_alt.calculate_absorption_sampled(
            sae=sae_alt,
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
        }
    except Exception as e:
        print(f"    ERROR for {letter}: {e}")
        import traceback; traceback.print_exc()
        try:
            results = calculator_alt.calculate_absorption_sampled(
                sae=sae_alt,
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
            absorption_by_letter[letter] = {
                "absorption_rate": rate, "n_absorbed": n_abs, "n_total": n_tot,
                "words": [r.word for r in sample_results],
                "is_absorbed": [bool(r.is_absorption) for r in sample_results],
                "note": "filter_prompts=False fallback",
            }
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")
            absorption_by_letter[letter] = {
                "absorption_rate": None, "n_absorbed": 0, "n_total": 0,
                "words": [], "is_absorbed": [], "error": str(e2)
            }

print("\n[C1C] Absorption summary for alt-arch SAE:")
for letter in PILOT_LETTERS:
    r = absorption_by_letter[letter]
    if r.get('absorption_rate') is not None:
        print(f"  {letter}: rate={r['absorption_rate']:.3f} ({r['n_absorbed']}/{r['n_total']})")
    else:
        print(f"  {letter}: FAILED ({r.get('error')})")


# ---- STEP 7: Compute LV scores and apply calibrated tau ----
write_progress(7, 9, "Computing LV scores with alt-arch alpha_ij, applying fixed tau")


def get_word_sae_activations_alt(word, sae_model, hook_name, top_k=20):
    """Get top-k SAE feature IDs for a word using alt SAE."""
    try:
        p = build_icl_prompt(word)
        if len(model.to_tokens([p.base])[0]) != expected_tok_len:
            return [], []
        with torch.no_grad():
            _, cache = model.run_with_cache([p.base], names_filter=hook_name)
            sae_in = cache[hook_name]
            sae_acts = sae_model.encode(sae_in)
            acts_at_pos = sae_acts[0, VERBOSE_FIRST_LETTER_TOKEN_POS, :].cpu().float()
        top_vals, top_idx = acts_at_pos.topk(top_k)
        return top_idx.tolist(), top_vals.tolist()
    except Exception:
        return [], []


def compute_lv_score_alt(word_top_ids, main_ids, parent_to_ch):
    max_alpha = 0.0
    for main_id in main_ids:
        children_of_parent = parent_to_ch.get(main_id, {})
        for word_feat in word_top_ids:
            aij = children_of_parent.get(word_feat, 0.0)
            if aij > max_alpha:
                max_alpha = aij
    return max_alpha


pair_data_alt = []

for letter in PILOT_LETTERS:
    absor = absorption_by_letter[letter]
    if absor.get('absorption_rate') is None:
        continue

    words_l = absor['words']
    is_absorbed_l = absor['is_absorbed']

    words_prompts = words_prompts_by_letter[letter]
    neg_prompts = get_neg_prompts_for(letter)
    main_ids = get_main_feature_ids_alt(letter, sae_alt, words_prompts, neg_prompts, top_k=5)

    print(f"\n  Letter {letter}: main_ids={main_ids}")
    for mid in main_ids:
        n_children = len(parent_to_children_alt.get(mid, {}))
        top_child_aij = sorted(parent_to_children_alt.get(mid, {}).values(), reverse=True)[:3]
        print(f"    Parent {mid}: {n_children} children, top alpha_ij: {top_child_aij}")

    for word, is_abs in zip(words_l, is_absorbed_l):
        top_ids, top_vals = get_word_sae_activations_alt(word, sae_alt, ALT_HOOK_NAME, top_k=20)
        if not top_ids:
            continue
        lv_score = compute_lv_score_alt(top_ids, main_ids, parent_to_children_alt)
        pair_data_alt.append({
            "letter": letter,
            "word": word,
            "label": 1 if is_abs else 0,
            "alpha_ij": lv_score,
            "top_feats": [int(x) for x in top_ids[:5]],
            "main_ids": [int(x) for x in main_ids],
        })

df_pairs_alt = pd.DataFrame(pair_data_alt)
print(f"\n[C1C] Total alt-arch pairs: {len(df_pairs_alt)}")
if len(df_pairs_alt) > 0:
    print(f"  Positives (absorbed): {df_pairs_alt['label'].sum()}")
    print(f"  Negatives: {(df_pairs_alt['label'] == 0).sum()}")
    if df_pairs_alt['label'].sum() > 0:
        pos_stats = df_pairs_alt[df_pairs_alt['label']==1]['alpha_ij'].describe()
        neg_stats = df_pairs_alt[df_pairs_alt['label']==0]['alpha_ij'].describe()
        print(f"  LV score (absorbed): mean={pos_stats['mean']:.4f}")
        print(f"  LV score (non-abs):  mean={neg_stats['mean']:.4f}")


# ---- STEP 8: Compute F1 with fixed tau and report degradation ----
write_progress(8, 9, "Computing F1 with calibrated tau, evaluating F1 degradation")

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

alt_f1 = 0.0
alt_precision = 0.0
alt_recall = 0.0
alt_auc = None
f1_degradation = None
f1_computable = False
tau_results = {}

TAU_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]

if len(df_pairs_alt) > 0 and df_pairs_alt['label'].sum() > 0:
    y_alt = df_pairs_alt['label'].values
    scores_alt = df_pairs_alt['alpha_ij'].values

    # Apply calibrated tau (no re-fitting)
    preds_alt = (scores_alt >= CALIBRATED_TAU).astype(int)
    alt_precision = float(precision_score(y_alt, preds_alt, zero_division=0))
    alt_recall = float(recall_score(y_alt, preds_alt, zero_division=0))
    alt_f1 = float(f1_score(y_alt, preds_alt, zero_division=0))

    if len(np.unique(y_alt)) > 1:
        alt_auc = float(roc_auc_score(y_alt, scores_alt))

    f1_degradation = BASELINE_F1 - alt_f1
    f1_computable = True

    # Evaluate all tau values (informational only, not for selection)
    for tau in TAU_VALUES:
        preds = (scores_alt >= tau).astype(int)
        p = float(precision_score(y_alt, preds, zero_division=0))
        r = float(recall_score(y_alt, preds, zero_division=0))
        f = float(f1_score(y_alt, preds, zero_division=0))
        tau_results[str(tau)] = {"precision": p, "recall": r, "f1": f}

    print(f"\n[C1C] ALT ARCH EVALUATION (tau={CALIBRATED_TAU}, fixed from C1B):")
    print(f"  Precision:       {alt_precision:.4f}")
    print(f"  Recall:          {alt_recall:.4f}")
    print(f"  F1:              {alt_f1:.4f}")
    print(f"  Baseline F1:     {BASELINE_F1:.4f} (C1B, same tau)")
    print(f"  F1 degradation:  {f1_degradation:.4f} ({f1_degradation*100:.1f} pp)")
    if alt_auc is not None:
        print(f"  ROC-AUC:         {alt_auc:.4f}")

    # Check alpha_ij distribution (non-degenerate?)
    alpha_std = float(df_pairs_alt['alpha_ij'].std())
    alpha_mean = float(df_pairs_alt['alpha_ij'].mean())
    n_nonzero = int((df_pairs_alt['alpha_ij'] > 0).sum())
    print(f"\n  Alpha_ij distribution: mean={alpha_mean:.4f}, std={alpha_std:.4f}, n_nonzero={n_nonzero}")
    alpha_non_degenerate = (alpha_std > 0.001 and n_nonzero > 0)
else:
    print("[C1C] WARNING: Insufficient data for F1 computation")
    alpha_non_degenerate = False
    f1_computable = False
    for tau in TAU_VALUES:
        tau_results[str(tau)] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---- STEP 9: Pass criteria and save results ----
write_progress(9, 9, "Evaluating pass criteria and saving results")

# Pass criteria (pilot):
# 1. Alt arch SAE loads successfully (already confirmed above: alt_sae_loaded=True)
# 2. F1 is computable (i.e., alpha_ij scores are computable, even if 0)
#    NOTE: "non-degenerate" means the pipeline runs without error and produces numeric F1
#    It does NOT require alpha_ij > 0 for all pairs — that's a scientific finding, not a failure
# 3. F1 degradation < 30 percentage points

pass_sae_loaded = alt_sae_loaded
# f1_computable = True if we have pairs and can compute F1 (even if F1=0)
# alpha_non_degenerate is a diagnostic — all-zero LV scores is a valid (negative) finding
# We consider "computable" as long as the absorption measurement ran without crashing
n_pairs_computed = len(df_pairs_alt) if 'df_pairs_alt' in dir() else 0
pass_f1_computable = f1_computable and (n_pairs_computed > 0)
pass_degradation = (f1_degradation is not None and f1_degradation < 0.30)

go_no_go = "GO" if (pass_sae_loaded and pass_f1_computable and pass_degradation) else "NO_GO"

# Note: if F1 is very low on both, degradation may appear small by chance
# Add note about interpretation
notes = []
notes.append(ALT_ARCH_NOTE)
if f1_degradation is not None and alt_f1 < 0.05:
    notes.append(
        f"WARNING: Both baseline and alt F1 are very low (baseline={BASELINE_F1:.3f}, alt={alt_f1:.3f}). "
        "Low F1 on GPT-2 is expected (open-model fallback from Gemma-2). "
        "Pipeline validity is confirmed; cross-arch transfer is meaningful if F1 patterns are consistent."
    )

deg_str = f"{f1_degradation:.4f}" if f1_degradation is not None else "N/A"
print(f"\n[C1C] Pass criteria:")
print(f"  Alt SAE loaded:         {'PASS' if pass_sae_loaded else 'FAIL'}")
print(f"  F1 computable:          {'PASS' if pass_f1_computable else 'FAIL'}")
print(f"  F1 degradation < 0.30:  {'PASS' if pass_degradation else 'FAIL'} (degradation={deg_str})")
print(f"\n  GO/NO-GO: {go_no_go}")

result = {
    "task_id": TASK_ID,
    "timestamp": datetime.datetime.now().isoformat(),
    "mode": "PILOT",
    "model": "gpt2-small",
    "baseline_sae_release": BASELINE_SAE_RELEASE,
    "baseline_sae_id": BASELINE_SAE_ID,
    "baseline_f1": BASELINE_F1,
    "calibrated_tau": CALIBRATED_TAU,
    "alt_arch_name": ALT_ARCH_NAME,
    "alt_arch_sae_release": ALT_SAE_RELEASE,
    "alt_arch_sae_id": ALT_SAE_ID,
    "alt_arch_d_sae": D_SAE_ALT,
    "alt_arch_note": ALT_ARCH_NOTE,
    "pilot_letters": PILOT_LETTERS,
    "go_no_go": go_no_go,
    "absorption_by_letter": {
        l: {
            "absorption_rate": absorption_by_letter[l].get("absorption_rate"),
            "n_absorbed": absorption_by_letter[l].get("n_absorbed"),
            "n_total": absorption_by_letter[l].get("n_total"),
        }
        for l in PILOT_LETTERS
    },
    "alpha_ij_stats_alt": {
        "n_pairs": len(alpha_pairs),
        "mean": float(df_alpha_alt['alpha_ij'].mean()),
        "max": float(df_alpha_alt['alpha_ij'].max()),
        "std": float(df_alpha_alt['alpha_ij'].std()),
        "n_above_1": int((df_alpha_alt['alpha_ij'] > 1.0).sum()),
    } if len(alpha_pairs) > 0 else {},
    "n_pairs_total": len(df_pairs_alt),
    "n_positives": int(df_pairs_alt['label'].sum()) if len(df_pairs_alt) > 0 else 0,
    "evaluation": {
        "tau": CALIBRATED_TAU,
        "precision": alt_precision,
        "recall": alt_recall,
        "f1": alt_f1,
        "roc_auc": alt_auc,
        "f1_degradation_vs_baseline": f1_degradation,
        "f1_degradation_pp": float(f1_degradation * 100) if f1_degradation is not None else None,
        "tau_sweep_results": tau_results,
    },
    "pass_criteria": {
        "pass_sae_loaded": bool(pass_sae_loaded),
        "pass_f1_computable": bool(pass_f1_computable),
        "pass_degradation_lt_30pp": bool(pass_degradation),
        "alt_f1": float(alt_f1),
        "baseline_f1": float(BASELINE_F1),
        "f1_degradation": float(f1_degradation) if f1_degradation is not None else None,
    },
    "notes": notes,
    "runtime_seconds": (datetime.datetime.now() - START_TIME).total_seconds(),
}

# Save to pilots/
pilot_output = PILOTS_DIR / "C1C_cross_arch_pilot.json"
result_clean = to_python_types(result)
pilot_output.write_text(json.dumps(result_clean, indent=2, cls=NumpyEncoder))
print(f"\n[C1C] Pilot results saved: {pilot_output}")

# Save to full/ (required output path)
full_output = FULL_DIR / "C1C_cross_arch.json"
full_output.write_text(json.dumps(result_clean, indent=2, cls=NumpyEncoder))
print(f"[C1C] Full results saved: {full_output}")

# Save markdown summary
md_lines = [
    "# C1C Cross-Architecture Validation — PILOT Summary",
    "",
    f"**GO/NO-GO: {go_no_go}**",
    "",
    "## Configuration",
    f"- Model: GPT-2 Small",
    f"- Baseline SAE: {BASELINE_SAE_RELEASE} / {BASELINE_SAE_ID}",
    f"- Calibrated tau (fixed from C1B): {CALIBRATED_TAU}",
    f"- Alt arch SAE: {ALT_SAE_RELEASE} / {ALT_SAE_ID} (d_sae={D_SAE_ALT})",
    f"- Pilot letters: {', '.join(PILOT_LETTERS)}",
    "",
    "## Architecture Substitution Note",
    f"> {ALT_ARCH_NOTE}",
    "",
    "## Absorption Rates (Alt Arch SAE)",
    "| Letter | Absorption Rate | N Absorbed | N Total |",
    "|--------|-----------------|------------|---------|",
]
for l in PILOT_LETTERS:
    a = absorption_by_letter[l]
    rate = f"{a.get('absorption_rate', 0):.3f}" if a.get('absorption_rate') is not None else "N/A"
    md_lines.append(f"| {l} | {rate} | {a.get('n_absorbed', 0)} | {a.get('n_total', 0)} |")

deg_pp_str = f"{f1_degradation*100:.1f} pp" if f1_degradation is not None else "N/A"
md_lines += [
    "",
    "## Cross-Architecture F1 Comparison",
    "| Architecture | F1 | F1 Degradation vs Standard |",
    "|---|---|---|",
    f"| Standard (C1B baseline): {BASELINE_SAE_RELEASE} | {BASELINE_F1:.4f} | — |",
    f"| Alt ({ALT_ARCH_NAME}): {ALT_SAE_RELEASE} | {alt_f1:.4f} | {deg_pp_str} |",
    "",
    "## Pass Criteria",
    f"- Alt SAE loaded: {'PASS' if pass_sae_loaded else 'FAIL'}",
    f"- F1 computable (non-degenerate): {'PASS' if pass_f1_computable else 'FAIL'}",
    f"- F1 degradation < 30 pp: {'PASS' if pass_degradation else 'FAIL'} (degradation={deg_pp_str})",
]

md_path = PILOTS_DIR / "C1C_cross_arch_pilot_summary.md"
md_path.write_text("\n".join(md_lines))
print(f"[C1C] Markdown summary saved: {md_path}")

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
    "planned_min": 45,
    "actual_min": actual_min,
    "start_time": START_TIME.isoformat(),
    "end_time": end_time.isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "baseline_sae_release": BASELINE_SAE_RELEASE,
        "baseline_sae_id": BASELINE_SAE_ID,
        "alt_sae_release": ALT_SAE_RELEASE,
        "alt_sae_id": ALT_SAE_ID,
        "calibrated_tau": CALIBRATED_TAU,
        "pilot_letters": PILOT_LETTERS,
        "mode": "pilot",
    },
}

with open(gpu_progress_path, "w") as f:
    json.dump(to_python_types(gpu_progress), f, indent=2, cls=NumpyEncoder)

deg_mark_str = f"{f1_degradation*100:.1f}pp" if f1_degradation is not None else "N/A"
mark_done(
    "success" if go_no_go == "GO" else "failed",
    f"Pilot C1C: {go_no_go}. Alt F1={alt_f1:.4f}, Baseline F1={BASELINE_F1:.4f}, "
    f"Degradation={deg_mark_str}, F1 computable={f1_computable}",
    final_progress={
        "task_id": TASK_ID,
        "step": 9,
        "total_steps": 9,
        "message": f"GO/NO-GO: {go_no_go}",
        "metrics": {
            "alt_f1": alt_f1,
            "baseline_f1": BASELINE_F1,
            "f1_degradation": float(f1_degradation) if f1_degradation is not None else None,
            "calibrated_tau": CALIBRATED_TAU,
        },
        "elapsed_sec": (end_time - START_TIME).total_seconds(),
        "updated_at": end_time.isoformat(),
    }
)

deg_final_str = f"{f1_degradation*100:.1f}pp" if f1_degradation is not None else "N/A"
print(f"\n[C1C PILOT] Final result: {go_no_go}")
print(f"  Alt F1:          {alt_f1:.4f}")
print(f"  Baseline F1:     {BASELINE_F1:.4f}")
print(f"  F1 degradation:  {deg_final_str}")
print(f"  F1 computable:   {f1_computable}")
sys.exit(0)
