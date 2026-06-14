"""
C1C: Cross-Architecture Validation of LV Detector — FULL MODE

Apply the calibrated tau from C1B (no re-fitting) to two alternative SAE architectures
at GPT-2 Small layer 8:
  (1) gpt2-small-resid-post-v5-32k (OpenAI v5, 32k latents, resid_post hook)
  (2) gpt2-small-resid-post-v5-128k (OpenAI v5, 128k latents, resid_post hook)

For each architecture:
  - Collect activation stats on 10k tokens (same as C1A)
  - Compute alpha_ij for candidate pairs
  - Apply fixed tau from C1B
  - Compare predicted absorbed pairs against absorption ground truth labels
  - Report F1 and F1 degradation vs within-architecture calibration

Note: JumpReLU and TopK SAEs are not available for GPT-2 via SAELens.
We use v5 (different training objective) and v5-128k (wider) as cross-architecture substitutes.
"""

import json
import os
import sys
import time
import random
import gc
import traceback
from datetime import datetime
from pathlib import Path

# Set longer HuggingFace timeout
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ─── Configuration ─────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
C1A_DIR = FULL_DIR / "C1A_alpha_ij_stats"
C1B_FILE = FULL_DIR / "C1B_lv_validation.json"
OUTPUT_FILE = FULL_DIR / "C1C_cross_arch.json"

TASK_ID = "C1C_cross_arch_validation"
SEED = 42
DEVICE = "cuda:0"

# Baseline SAE (same as C1B)
BASELINE_SAE_RELEASE = "gpt2-small-res-jb"
BASELINE_SAE_ID = "blocks.8.hook_resid_pre"
BASELINE_HOOK = "blocks.8.hook_resid_pre"
MODEL_NAME = "gpt2-small"
LAYER = 8

# Alternative architectures to test
ALT_ARCHS = [
    {
        "name": "v5-32k (OpenAI v5, resid_post, d_sae=32768)",
        "release": "gpt2-small-resid-post-v5-32k",
        "sae_id": "blocks.8.hook_resid_post",
        "hook": "blocks.8.hook_resid_post",
        "alpha_parquet": "v5post_layer9_32k.parquet",  # closest available from C1A (layer 9)
        "note": "OpenAI v5 architecture; hook_resid_post (vs baseline hook_resid_pre); "
                "different training objective. C1A collected stats at layer 9 (closest available).",
    },
    {
        "name": "v5-128k (OpenAI v5, resid_post, d_sae=131072)",
        "release": "gpt2-small-resid-post-v5-128k",
        "sae_id": "blocks.8.hook_resid_post",
        "hook": "blocks.8.hook_resid_post",
        "alpha_parquet": None,  # Need to compute fresh
        "note": "OpenAI v5 architecture, 128k latents (4x wider than v5-32k). "
                "Tests whether LV detector generalizes to very wide SAEs with different architecture.",
    },
]

# Letters
ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CALIB_LETTERS = list("ABCDEFGHIJKLM")
TEST_LETTERS = list("NOPQRSTUVWXYZ")

# Activation stats parameters (same as C1A)
N_TOKENS = 10000
F_MIN = 0.001
COSINE_THRESHOLD = 0.15

# Tau sweep values (for within-architecture calibration comparison)
TAU_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]

# Absorption detection
N_PROMPTS_PER_TOKEN = 5
MAX_ABLATION_SAMPLES = 100
NUM_PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 4096
NEG_RATIO = 3


# ─── Helpers ───────────────────────────────────────────────────────────

def write_progress(step, total, message, metrics=None, start_time=None):
    elapsed = time.time() - start_time if start_time else 0
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total,
        "message": message,
        "metrics": metrics or {},
        "elapsed_sec": elapsed,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress, indent=2))


def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


def eval_threshold(df, tau, score_col='alpha_ij'):
    """Evaluate binary classification at threshold tau."""
    y_true = df['is_positive'].astype(int).values
    y_pred = (df[score_col] > tau).astype(int).values
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": p, "recall": r, "f1": f}


def collect_activation_stats(model, sae, hook_name, n_tokens, f_min, cosine_threshold, device):
    """Collect activation statistics and compute alpha_ij for all candidate pairs.

    Returns a DataFrame with columns:
      latent_i, latent_j, f_i, f_j, coact_rate, sigma_ij, alpha_ij, decoder_cosine
    """
    from datasets import load_dataset

    print(f"    Loading OpenWebText tokens (n={n_tokens})...")
    try:
        ds = load_dataset("stas/openwebtext-10k", split="train", trust_remote_code=True)
    except Exception:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                          trust_remote_code=True)
        ds = list(ds.take(500))

    # Tokenize and collect activations
    all_texts = [item['text'] for item in ds]
    tokenizer = model.tokenizer
    all_token_ids = []
    for text in all_texts:
        toks = tokenizer.encode(text)
        all_token_ids.extend(toks)
        if len(all_token_ids) >= n_tokens:
            break
    all_token_ids = all_token_ids[:n_tokens]
    print(f"    Collected {len(all_token_ids)} tokens")

    # Process in chunks
    chunk_size = 256
    d_sae = sae.cfg.d_sae

    # Accumulate: binary activation indicators per token
    freq_counts = torch.zeros(d_sae, device='cpu', dtype=torch.float32)
    n_processed = 0

    # For co-activation: need to know which features activate per token
    # Store sparse sets to avoid O(d_sae^2) memory
    token_active_sets = []

    print(f"    Computing activation frequencies (d_sae={d_sae})...")
    for start in range(0, len(all_token_ids), chunk_size):
        chunk_ids = all_token_ids[start:start + chunk_size]
        input_ids = torch.tensor([chunk_ids], device=device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(input_ids)
            sae_input = cache[hook_name]
            sae_acts = sae.encode(sae_input)  # (1, seq_len, d_sae)

        acts = sae_acts[0]  # (seq_len, d_sae)
        active_mask = (acts > 0).cpu()

        # Update frequency counts
        freq_counts += active_mask.float().sum(dim=0)

        # Store active sets for co-activation computation
        for t in range(acts.shape[0]):
            active_indices = active_mask[t].nonzero().squeeze(-1).tolist()
            token_active_sets.append(set(active_indices))

        n_processed += acts.shape[0]
        del cache, sae_input, sae_acts, acts, active_mask
        torch.cuda.empty_cache()

    # Compute frequencies
    freqs = freq_counts / n_processed  # (d_sae,)
    freq_dict = {}
    active_features = []
    for i in range(d_sae):
        f = freqs[i].item()
        if f >= f_min:
            freq_dict[i] = f
            active_features.append(i)

    print(f"    Active features (f >= {f_min}): {len(active_features)}")

    # Pre-filter by decoder cosine similarity
    W_dec = sae.W_dec.detach()  # (d_sae, d_model)
    active_idx = torch.tensor(active_features, device=device)
    active_decoders = W_dec[active_idx]  # (n_active, d_model)

    # Compute pairwise cosine similarities among active features
    print(f"    Computing pairwise decoder cosine similarities...")
    # Normalize
    norms = active_decoders.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    active_decoders_norm = active_decoders / norms

    # Compute cosine similarity matrix in chunks to avoid OOM
    candidate_pairs = []  # list of (i_idx, j_idx, cosine)
    chunk = 500
    for ci in range(0, len(active_features), chunk):
        batch_i = active_decoders_norm[ci:ci + chunk]
        for cj in range(ci, len(active_features), chunk):
            batch_j = active_decoders_norm[cj:cj + chunk]
            cos_mat = torch.mm(batch_i, batch_j.T).cpu()

            for li in range(cos_mat.shape[0]):
                for lj in range(cos_mat.shape[1]):
                    global_i = ci + li
                    global_j = cj + lj
                    if global_i >= global_j:
                        continue
                    cos_val = cos_mat[li, lj].item()
                    if cos_val >= cosine_threshold:
                        candidate_pairs.append((
                            active_features[global_i],
                            active_features[global_j],
                            cos_val
                        ))

    print(f"    Candidate pairs (cosine >= {cosine_threshold}): {len(candidate_pairs)}")

    if len(candidate_pairs) == 0:
        return pd.DataFrame(columns=[
            'latent_i', 'latent_j', 'f_i', 'f_j', 'coact_rate',
            'sigma_ij', 'alpha_ij', 'decoder_cosine'
        ])

    # Compute co-activation rates
    print(f"    Computing co-activation rates for {len(candidate_pairs)} pairs...")
    coact_counts = {}
    for i, j, _ in candidate_pairs:
        coact_counts[(i, j)] = 0

    for active_set in token_active_sets:
        for i, j, _ in candidate_pairs:
            if i in active_set and j in active_set:
                coact_counts[(i, j)] += 1

    # Build result DataFrame
    rows = []
    for i, j, cos_val in candidate_pairs:
        f_i = freq_dict.get(i, 0)
        f_j = freq_dict.get(j, 0)
        coact = coact_counts[(i, j)] / n_processed

        min_f = min(f_i, f_j)
        if min_f > 0:
            sigma_ij = coact / min_f
        else:
            sigma_ij = 0.0

        # Compute alpha_ij in both directions
        if f_i > 0:
            alpha_ij_fwd = sigma_ij * (f_j / f_i)
        else:
            alpha_ij_fwd = 0.0

        if f_j > 0:
            alpha_ij_rev = sigma_ij * (f_i / f_j)
        else:
            alpha_ij_rev = 0.0

        # Use max of both directions
        alpha_ij = max(alpha_ij_fwd, alpha_ij_rev)

        rows.append({
            'latent_i': i,
            'latent_j': j,
            'f_i': f_i,
            'f_j': f_j,
            'coact_rate': coact,
            'sigma_ij': sigma_ij,
            'alpha_ij': alpha_ij,
            'decoder_cosine': cos_val,
        })

    df = pd.DataFrame(rows)
    print(f"    Alpha_ij stats: mean={df['alpha_ij'].mean():.4f}, "
          f"max={df['alpha_ij'].max():.4f}, "
          f"n>1={int((df['alpha_ij'] > 1).sum())}")

    return df


def detect_absorption_for_sae(model, sae, hook_name, layer, clean_words, probe_dirs,
                               all_letters, max_samples, device):
    """Run fast activation-based absorption detection for a given SAE.

    Returns dict: letter -> {absorption_rate, n_absorbed, n_total, absorbed_feature_ids, main_feature_ids}
    """
    from sae_spelling.prompting import create_icl_prompt, first_letter_formatter
    from sae_spelling.vocab import LETTERS as VOCAB_LETTERS

    formatter = first_letter_formatter()

    W_dec = sae.W_dec.detach()

    # Identify main features per letter (by probe-decoder cosine)
    letter_main_features = {}
    probe_cosines_all = {}

    for letter in all_letters:
        probe_dir = probe_dirs[letter]
        cos_sims = torch.nn.functional.cosine_similarity(
            probe_dir.unsqueeze(0).to(device),
            W_dec,
            dim=-1
        ).cpu()
        topk = cos_sims.topk(5)
        letter_main_features[letter] = topk.indices.tolist()
        probe_cosines_all[letter] = cos_sims

    # ICL word list
    icl_words = [w for w in clean_words if 3 <= len(w) <= 8][:200]

    absorption_by_letter = {}

    for letter in all_letters:
        letter_lower = letter.lower()
        letter_words = [w for w in clean_words if w[0].lower() == letter_lower]
        if len(letter_words) == 0:
            absorption_by_letter[letter] = {
                "absorption_rate": 0.0, "n_absorbed": 0, "n_total": 0,
                "absorbed_feature_ids": [], "main_feature_ids": letter_main_features[letter],
            }
            continue

        random.shuffle(letter_words)
        letter_words = letter_words[:max_samples]

        main_fids = letter_main_features[letter]
        cos_sims = probe_cosines_all[letter]

        answer_token_id = model.tokenizer.encode(f" {letter}")[0]

        # Build prompts
        prompts = []
        for w in letter_words:
            try:
                p = create_icl_prompt(
                    w, examples=icl_words, base_template="{word}:",
                    answer_formatter=formatter, max_icl_examples=10,
                    shuffle_examples=True,
                )
                prompts.append((w, p))
            except Exception:
                continue

        if not prompts:
            absorption_by_letter[letter] = {
                "absorption_rate": 0.0, "n_absorbed": 0, "n_total": 0,
                "absorbed_feature_ids": [], "main_feature_ids": main_fids,
            }
            continue

        n_absorbed = 0
        n_total = 0
        absorbed_fids = set()

        for w, p in prompts:
            try:
                with torch.inference_mode():
                    logits, cache = model.run_with_cache(p.base)
                    sae_in = cache[hook_name]
                    sae_acts = sae.encode(sae_in)

                    pred_token = logits[0, -1].argmax().item()
                    if pred_token != answer_token_id:
                        continue

                    word_acts = sae_acts[0, -2]
                    main_active = any(word_acts[fid].item() > 1e-8 for fid in main_fids)
                    n_total += 1

                    if not main_active:
                        active_mask = word_acts > 1e-8
                        active_indices = active_mask.nonzero().squeeze(-1).cpu()

                        if len(active_indices) > 0:
                            active_cos = cos_sims[active_indices]
                            best_idx = active_cos.argmax()
                            best_cos = active_cos[best_idx].item()
                            best_fid = active_indices[best_idx].item()

                            if best_cos > 0.025:
                                n_absorbed += 1
                                absorbed_fids.add(best_fid)

            except Exception:
                continue

        rate = n_absorbed / max(n_total, 1)
        absorption_by_letter[letter] = {
            "absorption_rate": rate,
            "n_absorbed": n_absorbed,
            "n_total": n_total,
            "absorbed_feature_ids": list(absorbed_fids),
            "main_feature_ids": main_fids,
        }

    return absorption_by_letter


def build_pairs_dataset(absorption_by_letter, alpha_lookup, cosine_lookup,
                         latent_i_to_pairs, latent_j_to_pairs, W_dec,
                         all_letters, neg_ratio=3):
    """Build binary classification dataset of (parent, absorber) pairs."""
    all_pairs = []

    for letter in all_letters:
        info = absorption_by_letter[letter]
        main_fids = info["main_feature_ids"]
        absorbed_fids = info["absorbed_feature_ids"]

        if info["n_total"] == 0:
            continue

        absorbed_set = set(absorbed_fids)

        # Positive pairs
        for main_fid in main_fids:
            for abs_fid in absorbed_fids:
                key1 = (main_fid, abs_fid)
                key2 = (abs_fid, main_fid)
                if key1 in alpha_lookup:
                    all_pairs.append({
                        "letter": letter, "is_positive": True,
                        "alpha_ij": alpha_lookup[key1],
                        "decoder_cosine": cosine_lookup[key1],
                        "latent_i": main_fid, "latent_j": abs_fid,
                    })
                elif key2 in alpha_lookup:
                    all_pairs.append({
                        "letter": letter, "is_positive": True,
                        "alpha_ij": alpha_lookup[key2],
                        "decoder_cosine": cosine_lookup[key2],
                        "latent_i": abs_fid, "latent_j": main_fid,
                    })
                else:
                    with torch.no_grad():
                        cos = torch.nn.functional.cosine_similarity(
                            W_dec[main_fid].unsqueeze(0),
                            W_dec[abs_fid].unsqueeze(0),
                            dim=-1
                        ).item()
                    all_pairs.append({
                        "letter": letter, "is_positive": True,
                        "alpha_ij": 0.0, "decoder_cosine": cos,
                        "latent_i": main_fid, "latent_j": abs_fid,
                    })

        # Negative pairs
        n_positives = sum(1 for p in all_pairs if p["letter"] == letter and p["is_positive"])
        n_negatives_needed = max(n_positives * neg_ratio, 5)

        neg_candidates = []
        for main_fid in main_fids:
            if main_fid in latent_i_to_pairs:
                for li, lj, aij, cos in latent_i_to_pairs[main_fid]:
                    if lj not in absorbed_set:
                        neg_candidates.append({
                            "letter": letter, "is_positive": False,
                            "alpha_ij": aij, "decoder_cosine": cos,
                            "latent_i": li, "latent_j": lj,
                        })
            if main_fid in latent_j_to_pairs:
                for li, lj, aij, cos in latent_j_to_pairs[main_fid]:
                    if li not in absorbed_set:
                        neg_candidates.append({
                            "letter": letter, "is_positive": False,
                            "alpha_ij": aij, "decoder_cosine": cos,
                            "latent_i": li, "latent_j": lj,
                        })

        if len(neg_candidates) > n_negatives_needed:
            random.shuffle(neg_candidates)
            neg_candidates = neg_candidates[:n_negatives_needed]

        all_pairs.extend(neg_candidates)

    return pd.DataFrame(all_pairs)


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start_time = time.time()
    write_pid()
    total_steps = 12

    print(f"[C1C FULL] Starting Cross-Architecture Validation — {datetime.now().isoformat()}")
    print(f"  Baseline: {BASELINE_SAE_RELEASE}/{BASELINE_SAE_ID}")
    print(f"  Alt architectures: {len(ALT_ARCHS)}")
    print(f"  Device: {DEVICE}")
    print(f"  All 26 letters used for ground-truth comparison")

    # ── Step 1: Load C1B calibrated tau ───────────────────────────────
    write_progress(1, total_steps, "Loading C1B calibrated tau", start_time=start_time)
    print("\n[Step 1] Loading C1B calibrated tau...")

    c1b = json.loads(C1B_FILE.read_text())
    calibrated_tau = c1b["calibration"]["best_tau"]
    baseline_test_f1 = c1b["test_evaluation"]["f1"]
    baseline_test_auc = c1b["test_evaluation"]["roc_auc"]
    baseline_cosine_f1 = c1b["cosine_baseline"]["test_f1"]

    print(f"  Calibrated tau from C1B: {calibrated_tau}")
    print(f"  Baseline test F1 (LV): {baseline_test_f1:.4f}")
    print(f"  Baseline test AUC (LV): {baseline_test_auc:.4f}")
    print(f"  Baseline cosine F1: {baseline_cosine_f1:.4f}")

    # ── Step 2: Load model ────────────────────────────────────────────
    write_progress(2, total_steps, "Loading GPT-2 model", start_time=start_time)
    print("\n[Step 2] Loading GPT-2 Small...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    print(f"  Model loaded: {MODEL_NAME}")

    # Get vocab for absorption detection
    vocab = model.tokenizer.get_vocab()
    from sae_spelling.vocab import LETTERS as VOCAB_LETTERS

    clean_words = []
    for token_str, token_id in vocab.items():
        if token_str.startswith('\u0120') and len(token_str) > 2:
            word = token_str[1:]
            if word.isascii() and word.isalpha() and len(word) >= 2:
                first_char = word[0].lower()
                if first_char in VOCAB_LETTERS:
                    clean_words.append(word)

    print(f"  Clean vocab words: {len(clean_words)}")

    # ── Step 3: Train probe (same as C1B, for consistent ground truth) ──
    write_progress(3, total_steps, "Training letter probes", start_time=start_time)
    print("\n[Step 3] Training letter probes for ground-truth absorption detection...")

    from sae_spelling.probing import (
        create_dataset_probe_training,
        gen_and_save_df_acts_probing,
        train_linear_probe_for_task,
    )
    from sae_spelling.prompting import first_letter_formatter

    formatter = first_letter_formatter()

    def answer_class_fn(answer):
        letter = answer.strip().lower()
        if letter in VOCAB_LETTERS:
            return VOCAB_LETTERS.index(letter)
        raise ValueError(f"Invalid answer: {repr(answer)}")

    # Train probe using BASELINE hook (resid_pre)
    train_dataset, test_dataset = create_dataset_probe_training(
        vocab=clean_words,
        formatter=formatter,
        num_prompts_per_token=N_PROMPTS_PER_TOKEN,
        base_template="{word}:",
        max_icl_examples=10,
        train_test_fraction=0.8,
        answer_class_fn=answer_class_fn,
    )

    probe_path = RESULTS_DIR / "C1C_probe_data_resid_pre"
    probe_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df, train_acts, test_acts = gen_and_save_df_acts_probing(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        path=probe_path,
        hook_point=BASELINE_HOOK,
        layer=LAYER,
        batch_size=64,
        position_idx=-2,
    )

    probe_pre, _ = train_linear_probe_for_task(
        train_df=train_df, test_df=test_df,
        device=torch.device(DEVICE),
        train_activations=train_acts, test_activations=test_acts,
        num_classes=26,
        batch_size=PROBE_BATCH_SIZE,
        num_epochs=NUM_PROBE_EPOCHS,
    )

    probe_dirs_pre = {}
    for i, letter in enumerate(VOCAB_LETTERS):
        probe_dirs_pre[letter.upper()] = probe_pre.fc.weight[i].detach().clone()

    print(f"  Probe (resid_pre) trained: weight shape = {probe_pre.fc.weight.shape}")

    # Also train probe using resid_post (for alt architectures)
    probe_path_post = RESULTS_DIR / "C1C_probe_data_resid_post"
    probe_path_post.mkdir(parents=True, exist_ok=True)

    alt_hook = "blocks.8.hook_resid_post"
    train_df_post, test_df_post, train_acts_post, test_acts_post = gen_and_save_df_acts_probing(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        path=probe_path_post,
        hook_point=alt_hook,
        layer=LAYER,
        batch_size=64,
        position_idx=-2,
    )

    probe_post, _ = train_linear_probe_for_task(
        train_df=train_df_post, test_df=test_df_post,
        device=torch.device(DEVICE),
        train_activations=train_acts_post, test_activations=test_acts_post,
        num_classes=26,
        batch_size=PROBE_BATCH_SIZE,
        num_epochs=NUM_PROBE_EPOCHS,
    )

    probe_dirs_post = {}
    for i, letter in enumerate(VOCAB_LETTERS):
        probe_dirs_post[letter.upper()] = probe_post.fc.weight[i].detach().clone()

    print(f"  Probe (resid_post) trained: weight shape = {probe_post.fc.weight.shape}")

    # ── Step 4: Load baseline absorption from C1B ─────────────────────
    write_progress(4, total_steps, "Loading baseline absorption data from C1B", start_time=start_time)
    print("\n[Step 4] Loading baseline absorption data...")

    baseline_absorption = c1b.get("absorption_by_letter", {})
    baseline_overall = np.mean([
        v["absorption_rate"] for v in baseline_absorption.values()
        if v.get("n_total", 0) > 0
    ])
    print(f"  Baseline overall absorption rate: {baseline_overall:.4f}")

    # ── Step 5+: Process each alternative architecture ────────────────
    arch_results = []

    for arch_idx, alt in enumerate(ALT_ARCHS):
        step_base = 5 + arch_idx * 3
        arch_name = alt["name"]
        print(f"\n{'='*60}")
        print(f"  Processing architecture: {arch_name}")
        print(f"{'='*60}")

        # Step A: Load SAE
        write_progress(step_base, total_steps, f"Loading SAE: {arch_name}", start_time=start_time)
        print(f"\n[Step {step_base}] Loading SAE: {alt['release']}/{alt['sae_id']}...")

        try:
            alt_sae = SAE.from_pretrained(
                release=alt["release"],
                sae_id=alt["sae_id"],
                device=DEVICE,
            )
            alt_sae.eval()
            d_sae_alt = alt_sae.cfg.d_sae
            print(f"  SAE loaded: d_sae={d_sae_alt}")
        except Exception as e:
            print(f"  ERROR loading SAE: {e}")
            arch_results.append({
                "arch_name": arch_name,
                "release": alt["release"],
                "sae_id": alt["sae_id"],
                "d_sae": None,
                "status": "load_failed",
                "error": str(e),
            })
            continue

        # Step B: Collect activation stats and compute alpha_ij
        write_progress(step_base + 1, total_steps,
                      f"Computing alpha_ij for {arch_name}", start_time=start_time)
        print(f"\n[Step {step_base + 1}] Computing activation stats and alpha_ij...")

        alpha_df_alt = collect_activation_stats(
            model=model, sae=alt_sae, hook_name=alt["hook"],
            n_tokens=N_TOKENS, f_min=F_MIN,
            cosine_threshold=COSINE_THRESHOLD, device=DEVICE,
        )

        print(f"  Alpha_ij pairs computed: {len(alpha_df_alt)}")

        # Build lookups
        alpha_lookup_alt = {}
        cosine_lookup_alt = {}
        li_to_pairs_alt = {}
        lj_to_pairs_alt = {}

        if len(alpha_df_alt) > 0:
            for _, row in alpha_df_alt.iterrows():
                li, lj = int(row['latent_i']), int(row['latent_j'])
                aij, cos = float(row['alpha_ij']), float(row['decoder_cosine'])
                key = (li, lj)
                alpha_lookup_alt[key] = aij
                cosine_lookup_alt[key] = cos
                li_to_pairs_alt.setdefault(li, []).append((li, lj, aij, cos))
                lj_to_pairs_alt.setdefault(lj, []).append((li, lj, aij, cos))

        # Step C: Detect absorption and evaluate
        write_progress(step_base + 2, total_steps,
                      f"Absorption detection and evaluation for {arch_name}",
                      start_time=start_time)
        print(f"\n[Step {step_base + 2}] Running absorption detection for {arch_name}...")

        # Choose appropriate probe for this hook
        if "resid_post" in alt["hook"]:
            probe_dirs_use = probe_dirs_post
        else:
            probe_dirs_use = probe_dirs_pre

        alt_absorption = detect_absorption_for_sae(
            model=model, sae=alt_sae, hook_name=alt["hook"],
            layer=LAYER, clean_words=clean_words,
            probe_dirs=probe_dirs_use,
            all_letters=ALL_LETTERS, max_samples=MAX_ABLATION_SAMPLES,
            device=DEVICE,
        )

        alt_overall = np.mean([
            v["absorption_rate"] for v in alt_absorption.values()
            if v.get("n_total", 0) > 0
        ])
        print(f"  Alt overall absorption rate: {alt_overall:.4f}")

        # Print per-letter summary
        for letter in ['A', 'M', 'N', 'Z']:
            info = alt_absorption.get(letter, {})
            print(f"    {letter}: {info.get('n_absorbed', 0)}/{info.get('n_total', 0)} "
                  f"({info.get('absorption_rate', 0):.2%})")

        # Build pairs dataset
        W_dec_alt = alt_sae.W_dec.detach()
        pairs_df_alt = build_pairs_dataset(
            absorption_by_letter=alt_absorption,
            alpha_lookup=alpha_lookup_alt,
            cosine_lookup=cosine_lookup_alt,
            latent_i_to_pairs=li_to_pairs_alt,
            latent_j_to_pairs=lj_to_pairs_alt,
            W_dec=W_dec_alt,
            all_letters=ALL_LETTERS,
            neg_ratio=NEG_RATIO,
        )

        print(f"  Pairs dataset: {len(pairs_df_alt)} "
              f"({pairs_df_alt['is_positive'].sum() if len(pairs_df_alt) > 0 else 0} positive)")

        if len(pairs_df_alt) == 0 or pairs_df_alt['is_positive'].sum() == 0:
            print(f"  WARNING: No valid positive pairs for {arch_name}")
            arch_results.append({
                "arch_name": arch_name,
                "release": alt["release"],
                "sae_id": alt["sae_id"],
                "d_sae": d_sae_alt,
                "status": "no_pairs",
                "note": alt["note"],
                "absorption_rate": alt_overall,
                "n_pairs": len(pairs_df_alt),
                "n_positives": int(pairs_df_alt['is_positive'].sum()) if len(pairs_df_alt) > 0 else 0,
            })
            # Free SAE memory
            del alt_sae, W_dec_alt
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Split into calib/test
        calib_df_alt = pairs_df_alt[pairs_df_alt['letter'].isin(CALIB_LETTERS)].copy()
        test_df_alt = pairs_df_alt[pairs_df_alt['letter'].isin(TEST_LETTERS)].copy()

        # Evaluate with fixed tau from C1B (no re-fitting)
        fixed_tau_result = eval_threshold(pairs_df_alt, calibrated_tau)
        fixed_tau_calib = eval_threshold(calib_df_alt, calibrated_tau) if len(calib_df_alt) > 0 else {"f1": 0.0}
        fixed_tau_test = eval_threshold(test_df_alt, calibrated_tau) if len(test_df_alt) > 0 else {"f1": 0.0}

        # ROC-AUC
        y_true_all = pairs_df_alt['is_positive'].astype(int).values
        y_scores_all = pairs_df_alt['alpha_ij'].values
        try:
            auc_all = roc_auc_score(y_true_all, y_scores_all)
        except ValueError:
            auc_all = 0.5

        # Within-architecture calibration (sweep tau on calib, evaluate on test)
        within_arch_calib = {}
        for tau in TAU_VALUES:
            if len(calib_df_alt) > 0:
                within_arch_calib[str(tau)] = eval_threshold(calib_df_alt, tau)
            else:
                within_arch_calib[str(tau)] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        best_within_tau = max(TAU_VALUES,
                              key=lambda t: within_arch_calib[str(t)]['f1'])
        within_test_result = eval_threshold(test_df_alt, best_within_tau) if len(test_df_alt) > 0 else {"f1": 0.0}

        # Cosine baseline on alt arch
        cosine_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        cosine_results = {}
        for ct in cosine_thresholds:
            cosine_results[str(ct)] = eval_threshold(pairs_df_alt, ct, score_col='decoder_cosine')

        best_cosine_thresh = max(cosine_thresholds,
                                 key=lambda t: cosine_results[str(t)]['f1'])
        alt_cosine_f1 = cosine_results[str(best_cosine_thresh)]['f1']

        try:
            cosine_auc = roc_auc_score(y_true_all, pairs_df_alt['decoder_cosine'].values)
        except ValueError:
            cosine_auc = 0.5

        # F1 degradation
        f1_degradation_fixed = baseline_test_f1 - fixed_tau_result['f1']
        f1_degradation_within = baseline_test_f1 - within_test_result['f1']

        print(f"\n  === Results for {arch_name} ===")
        print(f"  Fixed tau={calibrated_tau}: P={fixed_tau_result['precision']:.4f} "
              f"R={fixed_tau_result['recall']:.4f} F1={fixed_tau_result['f1']:.4f}")
        print(f"  Fixed tau (test only): F1={fixed_tau_test['f1']:.4f}")
        print(f"  ROC-AUC: {auc_all:.4f}")
        print(f"  Within-arch best tau={best_within_tau}: "
              f"Test F1={within_test_result['f1']:.4f}")
        print(f"  Cosine baseline: F1={alt_cosine_f1:.4f} (thresh={best_cosine_thresh})")
        print(f"  F1 degradation (fixed tau vs baseline): {f1_degradation_fixed:+.4f}")
        print(f"  F1 degradation (within-arch vs baseline): {f1_degradation_within:+.4f}")

        # Collect all tau sweep results on test set
        test_tau_sweep = {}
        for tau in TAU_VALUES:
            if len(test_df_alt) > 0:
                test_tau_sweep[str(tau)] = eval_threshold(test_df_alt, tau)
            else:
                test_tau_sweep[str(tau)] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            r = test_tau_sweep[str(tau)]
            print(f"    tau={tau}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

        arch_results.append({
            "arch_name": arch_name,
            "release": alt["release"],
            "sae_id": alt["sae_id"],
            "d_sae": d_sae_alt,
            "status": "success",
            "note": alt["note"],
            "absorption_rate": float(alt_overall),
            "absorption_by_letter": {
                letter: {
                    "absorption_rate": v["absorption_rate"],
                    "n_absorbed": v["n_absorbed"],
                    "n_total": v["n_total"],
                }
                for letter, v in alt_absorption.items()
            },
            "alpha_ij_stats": {
                "n_pairs": len(alpha_df_alt),
                "mean": float(alpha_df_alt['alpha_ij'].mean()),
                "max": float(alpha_df_alt['alpha_ij'].max()),
                "std": float(alpha_df_alt['alpha_ij'].std()),
                "n_above_1": int((alpha_df_alt['alpha_ij'] > 1).sum()),
            },
            "n_pairs_total": len(pairs_df_alt),
            "n_positives": int(pairs_df_alt['is_positive'].sum()),
            "n_negatives": int((~pairs_df_alt['is_positive']).sum()),
            "n_pairs_calib": len(calib_df_alt),
            "n_pairs_test": len(test_df_alt),
            "fixed_tau_evaluation": {
                "tau": calibrated_tau,
                "all_data": fixed_tau_result,
                "calib_only": fixed_tau_calib,
                "test_only": fixed_tau_test,
                "roc_auc": auc_all,
            },
            "within_arch_calibration": {
                "tau_sweep_calib": within_arch_calib,
                "best_tau": best_within_tau,
                "test_evaluation": within_test_result,
                "test_tau_sweep": test_tau_sweep,
            },
            "cosine_baseline": {
                "best_threshold": best_cosine_thresh,
                "all_f1": alt_cosine_f1,
                "roc_auc": cosine_auc,
                "threshold_results": cosine_results,
            },
            "degradation": {
                "f1_degradation_fixed_tau": f1_degradation_fixed,
                "f1_degradation_within_arch": f1_degradation_within,
                "f1_degradation_pp": f1_degradation_fixed * 100,
                "passes_10pp_criterion": abs(f1_degradation_fixed) < 0.10,
                "passes_30pp_criterion": abs(f1_degradation_fixed) < 0.30,
            },
        })

        # Free SAE memory
        del alt_sae, W_dec_alt, alpha_df_alt
        gc.collect()
        torch.cuda.empty_cache()

    # ── Final step: Save results ──────────────────────────────────────
    write_progress(total_steps, total_steps, "Saving final results", start_time=start_time)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"C1C FULL RESULTS SUMMARY")
    print(f"{'='*60}")

    # Summary table
    print(f"\n  {'Architecture':<45} {'F1 (fixed)':<12} {'F1 (within)':<12} {'AUC':<8} {'Degrad.':<10}")
    print(f"  {'-'*45} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
    print(f"  {'Baseline (jb, 24k, resid_pre)':<45} "
          f"{baseline_test_f1:<12.4f} {baseline_test_f1:<12.4f} {baseline_test_auc:<8.4f} {'--':<10}")

    for res in arch_results:
        if res["status"] == "success":
            print(f"  {res['arch_name'][:45]:<45} "
                  f"{res['fixed_tau_evaluation']['all_data']['f1']:<12.4f} "
                  f"{res['within_arch_calibration']['test_evaluation']['f1']:<12.4f} "
                  f"{res['fixed_tau_evaluation']['roc_auc']:<8.4f} "
                  f"{res['degradation']['f1_degradation_pp']:+.1f}pp")
        else:
            print(f"  {res['arch_name'][:45]:<45} {'FAILED':<12} {'-':<12} {'-':<8} {'-':<10}")

    results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "FULL",
        "model": MODEL_NAME,
        "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
        "baseline": {
            "sae_release": BASELINE_SAE_RELEASE,
            "sae_id": BASELINE_SAE_ID,
            "d_sae": 24576,
            "test_f1": baseline_test_f1,
            "test_auc": baseline_test_auc,
            "cosine_f1": baseline_cosine_f1,
        },
        "calibrated_tau": calibrated_tau,
        "n_alt_architectures": len(ALT_ARCHS),
        "n_successful": sum(1 for r in arch_results if r["status"] == "success"),
        "architectures": arch_results,
        "summary": {
            "cross_arch_generalization": None,
            "mean_f1_degradation_pp": None,
            "all_pass_30pp": None,
            "all_pass_10pp": None,
        },
        "notes": [
            "JumpReLU and TopK SAEs not available for GPT-2 small in SAELens.",
            "Substituted with OpenAI v5 SAEs (different training objective, resid_post hook).",
            "v5-32k and v5-128k test both architecture and width generalization.",
            "Probe trained separately for resid_pre (baseline) and resid_post (alt) hooks.",
        ],
        "runtime_seconds": elapsed,
    }

    # Compute summary stats
    successful = [r for r in arch_results if r["status"] == "success"]
    if successful:
        degradations = [r["degradation"]["f1_degradation_pp"] for r in successful]
        results["summary"]["mean_f1_degradation_pp"] = float(np.mean(degradations))
        results["summary"]["all_pass_30pp"] = all(abs(d) < 30 for d in degradations)
        results["summary"]["all_pass_10pp"] = all(abs(d) < 10 for d in degradations)
        results["summary"]["cross_arch_generalization"] = (
            "strong" if results["summary"]["all_pass_10pp"]
            else "moderate" if results["summary"]["all_pass_30pp"]
            else "weak"
        )

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")

    # Update gpu_progress.json
    try:
        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        gp = json.loads(gp_path.read_text())

        if TASK_ID not in gp.get("completed", []):
            gp.setdefault("completed", []).append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 45,
            "actual_min": int(elapsed / 60),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": MODEL_NAME,
                "mode": "FULL",
                "n_alt_archs": len(ALT_ARCHS),
                "n_successful": len(successful),
                "n_letters": 26,
                "gpu": DEVICE,
            },
        }

        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"  gpu_progress.json updated")
    except Exception as e:
        print(f"  WARNING: Failed to update gpu_progress.json: {e}")

    summary_str = "; ".join([
        f"{r['arch_name'][:20]}: F1={r['fixed_tau_evaluation']['all_data']['f1']:.3f}"
        for r in successful
    ]) if successful else "no successful architectures"

    mark_done("success", summary_str)
    print(f"\n[C1C FULL] Complete. DONE marker written.")


if __name__ == "__main__":
    main()
