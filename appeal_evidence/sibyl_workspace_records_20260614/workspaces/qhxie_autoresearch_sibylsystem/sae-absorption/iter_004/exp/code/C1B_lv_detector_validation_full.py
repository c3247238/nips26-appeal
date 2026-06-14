"""
C1B: LV Detector Threshold Calibration and Validation (H1) — FULL MODE

Load sae-spelling ground-truth absorption labels for GPT-2 Small SAE at layer 8
(gpt2-small-res-jb, blocks.8.hook_resid_pre, 24k latents), all 26 letters.

Split: A-M calibration (13 letters), N-Z test (13 letters).
For each letter:
  1. Run sae-spelling absorption detection → get per-token absorption labels
  2. Identify which SAE latents absorb the letter's information
  3. Look up alpha_ij from C1A output for parent-child pairs
  4. Build binary classification dataset: absorbed vs non-absorbed pairs

Fit threshold tau via F1 maximization on calibration set.
Apply best tau to test set. Report precision, recall, F1, ROC-AUC.
Compute LV sharpness diagnostic: sigmoid vs linear fit on binned data.
Also compare against cosine-only baseline detector.
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

# Set longer HuggingFace timeout and prefer cached models
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)

# ─── Configuration ─────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
C1A_DIR = FULL_DIR / "C1A_alpha_ij_stats"
OUTPUT_FILE = FULL_DIR / "C1B_lv_validation.json"

TASK_ID = "C1B_lv_detector_validation"
SEED = 42
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=2 makes physical GPU 2 appear as cuda:0

# SAE config — GPT-2 Small layer 8
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
HOOK_NAME = "blocks.8.hook_resid_pre"
MODEL_NAME = "gpt2-small"
LAYER = 8
D_SAE = 24576

# Alpha_ij parquet from C1A
ALPHA_PARQUET = C1A_DIR / "jb_layer8_24k.parquet"

# Letters
ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CALIB_LETTERS = list("ABCDEFGHIJKLM")  # A-M (13)
TEST_LETTERS = list("NOPQRSTUVWXYZ")    # N-Z (13)

# Threshold sweep
TAU_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]

# sae-spelling parameters
N_PROMPTS_PER_TOKEN = 5  # prompts per vocab token for probing
MAX_ABLATION_SAMPLES = 100  # per letter
NUM_PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 4096

# Negative sampling ratio
NEG_RATIO = 3  # 3 negatives per positive

# ─── Helpers ───────────────────────────────────────────────────────────

def write_progress(step, total, message, metrics=None, start_time=None):
    """Write progress file for monitoring."""
    elapsed = time.time() - start_time if start_time else 0
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total,
        "message": message,
        "metrics": metrics or {},
        "elapsed_sec": elapsed,
        "updated_at": datetime.now().isoformat()
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress, indent=2))


def write_pid():
    """Write PID file for system monitoring."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def mark_done(status="success", summary=""):
    """Write DONE marker."""
    # Clean up PID
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    # Read final progress
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


def sigmoid(x, k, x0):
    """Sigmoid function for curve fitting."""
    return expit(k * (x - x0))


def linear(x, a, b):
    """Linear function for curve fitting."""
    return a * x + b


def compute_aic(n, rss, k_params):
    """Compute AIC given n observations, residual sum of squares, k parameters."""
    if rss <= 0 or n <= k_params:
        return float('inf')
    return n * np.log(rss / n) + 2 * k_params


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start_time = time.time()
    write_pid()
    total_steps = 10

    print(f"[C1B FULL] Starting LV Detector Validation — {datetime.now().isoformat()}")
    print(f"  Model: {MODEL_NAME}, SAE: {SAE_RELEASE}/{SAE_ID}")
    print(f"  Device: {DEVICE}")
    print(f"  Calibration letters: {CALIB_LETTERS}")
    print(f"  Test letters: {TEST_LETTERS}")

    # ── Step 1: Load alpha_ij data ────────────────────────────────────
    write_progress(1, total_steps, "Loading alpha_ij from C1A", start_time=start_time)
    print("\n[Step 1] Loading alpha_ij data from C1A...")

    alpha_df = pd.read_parquet(ALPHA_PARQUET)
    print(f"  Loaded {len(alpha_df)} pairs from {ALPHA_PARQUET.name}")
    print(f"  Alpha_ij range: [{alpha_df['alpha_ij'].min():.4f}, {alpha_df['alpha_ij'].max():.4f}]")
    print(f"  Pairs with alpha_ij > 1: {(alpha_df['alpha_ij'] > 1).sum()}")

    # Build lookup: (latent_i, latent_j) -> alpha_ij, decoder_cosine
    # Vectorized construction for speed (4.4M rows)
    li_arr = alpha_df['latent_i'].astype(int).values
    lj_arr = alpha_df['latent_j'].astype(int).values
    aij_arr = alpha_df['alpha_ij'].values
    cos_arr = alpha_df['decoder_cosine'].values

    alpha_lookup = {}
    cosine_lookup = {}
    latent_i_to_pairs = {}
    latent_j_to_pairs = {}

    for idx in range(len(li_arr)):
        li, lj = int(li_arr[idx]), int(lj_arr[idx])
        aij, cos = float(aij_arr[idx]), float(cos_arr[idx])
        key = (li, lj)
        alpha_lookup[key] = aij
        cosine_lookup[key] = cos
        latent_i_to_pairs.setdefault(li, []).append((li, lj, aij, cos))
        latent_j_to_pairs.setdefault(lj, []).append((li, lj, aij, cos))

    print(f"  Unique latent_i values: {len(latent_i_to_pairs)}")
    print(f"  Unique latent_j values: {len(latent_j_to_pairs)}")

    # ── Step 2: Load model and SAE ────────────────────────────────────
    write_progress(2, total_steps, "Loading model and SAE", start_time=start_time)
    print("\n[Step 2] Loading GPT-2 Small and SAE...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=DEVICE
    )
    sae.eval()
    print(f"  Model loaded: {MODEL_NAME}")
    print(f"  SAE loaded: d_sae={sae.cfg.d_sae}")

    # ── Step 3: Train letter probes ───────────────────────────────────
    write_progress(3, total_steps, "Training letter probes for absorption detection", start_time=start_time)
    print("\n[Step 3] Training letter probes...")

    from sae_spelling.probing import (
        create_dataset_probe_training,
        gen_and_save_df_acts_probing,
        train_linear_probe_for_task,
        LinearProbe,
    )
    from sae_spelling.vocab import LETTERS as VOCAB_LETTERS

    # Get vocabulary words that start with each letter
    vocab = model.tokenizer.get_vocab()
    # Filter for clean tokens starting with a letter (space-prefixed tokens in GPT-2)
    clean_words = []
    for token_str, token_id in vocab.items():
        # GPT-2 uses 'Ġ' prefix for space tokens; we want words
        if token_str.startswith('Ġ') and len(token_str) > 2:
            word = token_str[1:]  # remove Ġ prefix
            if word.isascii() and word.isalpha() and len(word) >= 2:
                # Ensure the first letter is a valid A-Z letter
                first_char = word[0].lower()
                if first_char in VOCAB_LETTERS:
                    clean_words.append(word)

    print(f"  Found {len(clean_words)} clean vocab words")

    # Create probe training dataset
    from sae_spelling.prompting import first_letter_formatter
    formatter = first_letter_formatter()

    # Define answer_class_fn that safely maps answers to letter indices
    def answer_class_fn(answer):
        """Map answer string (e.g., ' H') to letter index 0-25."""
        letter = answer.strip().lower()
        if letter in VOCAB_LETTERS:
            return VOCAB_LETTERS.index(letter)
        raise ValueError(f"Invalid answer: {repr(answer)}, letter={repr(letter)}")

    # Create dataset for probe training
    train_dataset, test_dataset = create_dataset_probe_training(
        vocab=clean_words,
        formatter=formatter,
        num_prompts_per_token=N_PROMPTS_PER_TOKEN,
        base_template="{word}:",
        max_icl_examples=10,
        train_test_fraction=0.8,
        answer_class_fn=answer_class_fn,
    )
    print(f"  Train dataset: {len(train_dataset)} prompts")
    print(f"  Test dataset: {len(test_dataset)} prompts")

    # Generate activations and train probe
    probe_path = RESULTS_DIR / "C1B_probe_data"
    probe_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df, train_acts, test_acts = gen_and_save_df_acts_probing(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        path=probe_path,
        hook_point=HOOK_NAME,
        layer=LAYER,
        batch_size=64,
        position_idx=-2,
    )

    probe, probe_data = train_linear_probe_for_task(
        train_df=train_df,
        test_df=test_df,
        device=torch.device(DEVICE),
        train_activations=train_acts,
        test_activations=test_acts,
        num_classes=26,
        batch_size=PROBE_BATCH_SIZE,
        num_epochs=NUM_PROBE_EPOCHS,
    )

    # Get probe directions for each letter
    probe_dirs = {}
    for i, letter in enumerate(VOCAB_LETTERS):
        probe_dirs[letter.upper()] = probe.fc.weight[i].detach().clone()

    print(f"  Probe trained. Probe weight shape: {probe.fc.weight.shape}")

    # ── Step 4: Identify main features for each letter ────────────────
    write_progress(4, total_steps, "Identifying main features per letter via probe-decoder cosine sim", start_time=start_time)
    print("\n[Step 4] Identifying main features per letter...")

    # For each letter, find SAE features whose decoder direction is most aligned with the probe
    W_dec = sae.W_dec.detach()  # (d_sae, d_model)

    letter_main_features = {}  # letter -> list of main feature IDs
    letter_probe_cosines = {}  # letter -> dict of feature_id -> cosine_sim

    for letter in ALL_LETTERS:
        probe_dir = probe_dirs[letter]
        # Compute cosine similarity between probe direction and all decoder directions
        cos_sims = torch.nn.functional.cosine_similarity(
            probe_dir.unsqueeze(0).to(DEVICE),
            W_dec,
            dim=-1
        ).cpu()

        # Main features are those with highest cosine similarity to probe
        # Use top-5 features as potential "main" features for this letter
        topk = cos_sims.topk(5)
        main_ids = topk.indices.tolist()
        main_sims = topk.values.tolist()

        letter_main_features[letter] = main_ids
        letter_probe_cosines[letter] = {fid: sim for fid, sim in zip(main_ids, main_sims)}

        if letter in ['A', 'N']:  # Print examples
            print(f"  Letter {letter}: main features = {main_ids}, cosines = {[f'{s:.4f}' for s in main_sims]}")

    # ── Step 5: Run fast absorption detection for all 26 letters ─────
    write_progress(5, total_steps, "Running fast activation-based absorption detection for 26 letters", start_time=start_time)
    print("\n[Step 5] Running fast activation-based absorption detection...")
    print("  Using activation-based detection (no IG) for speed: ")
    print("  Absorption = main features don't fire + model correct + probe-aligned feature fires")

    from sae_spelling.prompting import create_icl_prompt

    # Pre-compute decoder-probe cosine similarities for all features
    probe_cosines_all = {}  # letter -> tensor of shape (d_sae,)
    for letter in ALL_LETTERS:
        probe_dir = probe_dirs[letter]
        cos_sims = torch.nn.functional.cosine_similarity(
            probe_dir.unsqueeze(0).to(DEVICE),
            W_dec,
            dim=-1
        ).cpu()
        probe_cosines_all[letter] = cos_sims

    # ICL word list
    icl_words = [w for w in clean_words if 3 <= len(w) <= 8][:200]

    absorption_by_letter = {}

    for idx, letter in enumerate(ALL_LETTERS):
        letter_lower = letter.lower()
        letter_words = [w for w in clean_words if w[0].lower() == letter_lower]
        if len(letter_words) == 0:
            absorption_by_letter[letter] = {
                "absorption_rate": 0.0, "n_absorbed": 0, "n_total": 0,
                "absorbed_feature_ids": [], "main_feature_ids": letter_main_features[letter],
            }
            continue

        random.shuffle(letter_words)
        letter_words = letter_words[:MAX_ABLATION_SAMPLES]

        main_fids = letter_main_features[letter]
        main_fids_set = set(main_fids)
        letter_idx = ord(letter) - ord('A')
        answer_token_id = model.tokenizer.encode(f" {letter}")[0]
        cos_sims = probe_cosines_all[letter]

        # Build prompts in ICL format
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

        # Batch process: run model + SAE on all prompts
        n_absorbed = 0
        n_total = 0
        absorbed_fids = set()

        # Process in batches of 20
        batch_size = 20
        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start:batch_start + batch_size]

            # Check all prompts have same token length (required by model batching)
            try:
                token_lens = set()
                for _, p in batch:
                    toks = model.to_tokens(p.base)
                    token_lens.add(toks.shape[1])

                if len(token_lens) > 1:
                    # Process individually if variable length
                    for w, p in batch:
                        try:
                            with torch.inference_mode():
                                logits, cache = model.run_with_cache(p.base)
                                sae_in = cache[HOOK_NAME]
                                sae_acts = sae.encode(sae_in)

                                # Check if model predicts correctly
                                pred_token = logits[0, -1].argmax().item()
                                if pred_token != answer_token_id:
                                    continue  # model wrong, skip

                                # Check if main features fire at word position (-2)
                                word_acts = sae_acts[0, -2]
                                main_active = any(word_acts[fid].item() > 1e-8 for fid in main_fids)

                                n_total += 1

                                if not main_active:
                                    # Main features DON'T fire → check for absorption
                                    # Find top active features aligned with probe
                                    active_mask = word_acts > 1e-8
                                    active_indices = active_mask.nonzero().squeeze(-1).cpu()

                                    if len(active_indices) > 0:
                                        active_cos = cos_sims[active_indices]
                                        # Find if any active feature has high probe cosine
                                        best_idx = active_cos.argmax()
                                        best_cos = active_cos[best_idx].item()
                                        best_fid = active_indices[best_idx].item()

                                        if best_cos > 0.025:  # probe_cos_sim_threshold
                                            n_absorbed += 1
                                            absorbed_fids.add(best_fid)

                        except Exception:
                            continue
                else:
                    # Batch process
                    with torch.inference_mode():
                        texts = [p.base for _, p in batch]
                        logits, cache = model.run_with_cache(texts)
                        sae_in = cache[HOOK_NAME]
                        sae_acts = sae.encode(sae_in)

                        for bi, (w, p) in enumerate(batch):
                            pred_token = logits[bi, -1].argmax().item()
                            if pred_token != answer_token_id:
                                continue

                            word_acts = sae_acts[bi, -2]
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

            except Exception as e:
                # Fallback: process individually
                for w, p in batch:
                    try:
                        with torch.inference_mode():
                            logits, cache = model.run_with_cache(p.base)
                            sae_in = cache[HOOK_NAME]
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

        if (idx + 1) % 5 == 0 or letter in ['A', 'M', 'N', 'Z']:
            print(f"  Letter {letter}: {n_absorbed}/{n_total} absorbed ({rate:.2%}), absorbers: {list(absorbed_fids)[:3]}")

    write_progress(5, total_steps, "Absorption detection complete",
                   metrics={"letters_done": len(absorption_by_letter)}, start_time=start_time)

    # ── Step 6: Build paired dataset (parent→absorber pairs) ──────────
    write_progress(6, total_steps, "Building paired alpha_ij dataset", start_time=start_time)
    print("\n[Step 6] Building paired dataset for LV detector evaluation...")

    # For each letter, we need:
    # - Positive pairs: (main_feature, absorber_feature) where absorption occurred
    # - Negative pairs: (main_feature, random_feature) where no absorption

    # Build positive and negative examples with alpha_ij scores
    all_pairs = []  # list of (letter, is_positive, alpha_ij, decoder_cosine, latent_i, latent_j)

    for letter in ALL_LETTERS:
        info = absorption_by_letter[letter]
        main_fids = info["main_feature_ids"]
        absorbed_fids = info["absorbed_feature_ids"]

        if info["n_total"] == 0:
            continue

        # Positive pairs: main feature -> absorbed feature
        for main_fid in main_fids:
            for abs_fid in absorbed_fids:
                # Check both directions in alpha_ij lookup
                key1 = (main_fid, abs_fid)
                key2 = (abs_fid, main_fid)

                if key1 in alpha_lookup:
                    all_pairs.append({
                        "letter": letter,
                        "is_positive": True,
                        "alpha_ij": alpha_lookup[key1],
                        "decoder_cosine": cosine_lookup[key1],
                        "latent_i": main_fid,
                        "latent_j": abs_fid,
                    })
                elif key2 in alpha_lookup:
                    all_pairs.append({
                        "letter": letter,
                        "is_positive": True,
                        "alpha_ij": alpha_lookup[key2],
                        "decoder_cosine": cosine_lookup[key2],
                        "latent_i": abs_fid,
                        "latent_j": main_fid,
                    })
                # If pair not found in alpha_ij data, it means cosine < 0.15 threshold
                # Still record it with alpha_ij = 0
                else:
                    # Compute decoder cosine directly
                    with torch.no_grad():
                        cos = torch.nn.functional.cosine_similarity(
                            W_dec[main_fid].unsqueeze(0),
                            W_dec[abs_fid].unsqueeze(0),
                            dim=-1
                        ).item()
                    all_pairs.append({
                        "letter": letter,
                        "is_positive": True,
                        "alpha_ij": 0.0,  # below pre-filter threshold
                        "decoder_cosine": cos,
                        "latent_i": main_fid,
                        "latent_j": abs_fid,
                    })

        # Negative pairs: sample from non-absorbed features connected to main features
        n_positives = sum(1 for p in all_pairs if p["letter"] == letter and p["is_positive"])
        n_negatives_needed = max(n_positives * NEG_RATIO, 5)  # at least 5

        neg_candidates = []
        for main_fid in main_fids:
            # Get pairs involving this main feature from alpha_ij data
            if main_fid in latent_i_to_pairs:
                for li, lj, aij, cos in latent_i_to_pairs[main_fid]:
                    if lj not in absorbed_fids:
                        neg_candidates.append({
                            "letter": letter,
                            "is_positive": False,
                            "alpha_ij": aij,
                            "decoder_cosine": cos,
                            "latent_i": li,
                            "latent_j": lj,
                        })
            if main_fid in latent_j_to_pairs:
                for li, lj, aij, cos in latent_j_to_pairs[main_fid]:
                    if li not in absorbed_fids:
                        neg_candidates.append({
                            "letter": letter,
                            "is_positive": False,
                            "alpha_ij": aij,
                            "decoder_cosine": cos,
                            "latent_i": li,
                            "latent_j": lj,
                        })

        # Sample negatives
        if len(neg_candidates) > n_negatives_needed:
            random.shuffle(neg_candidates)
            neg_candidates = neg_candidates[:n_negatives_needed]

        all_pairs.extend(neg_candidates)

    pairs_df = pd.DataFrame(all_pairs)
    print(f"  Total pairs: {len(pairs_df)}")
    print(f"  Positive: {pairs_df['is_positive'].sum()}")
    print(f"  Negative: {(~pairs_df['is_positive']).sum()}")

    if len(pairs_df) == 0:
        print("  WARNING: No pairs found! Check absorption detection and alpha_ij matching.")
        mark_done("failed", "No valid pairs for LV detector evaluation")
        return

    # Split into calibration and test
    calib_df = pairs_df[pairs_df['letter'].isin(CALIB_LETTERS)].copy()
    test_df_eval = pairs_df[pairs_df['letter'].isin(TEST_LETTERS)].copy()

    print(f"  Calibration: {len(calib_df)} pairs ({calib_df['is_positive'].sum()} positive)")
    print(f"  Test: {len(test_df_eval)} pairs ({test_df_eval['is_positive'].sum()} positive)")

    # ── Step 7: Threshold calibration ─────────────────────────────────
    write_progress(7, total_steps, "Calibrating threshold tau on A-M letters", start_time=start_time)
    print("\n[Step 7] Threshold calibration on A-M...")

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

    # Sweep tau on calibration set
    calib_tau_results = {}
    for tau in TAU_VALUES:
        res = eval_threshold(calib_df, tau)
        calib_tau_results[str(tau)] = res
        print(f"  tau={tau}: P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f}")

    # Select best tau by F1
    best_tau = max(TAU_VALUES, key=lambda t: calib_tau_results[str(t)]['f1'])
    best_calib_f1 = calib_tau_results[str(best_tau)]['f1']
    print(f"  Best tau: {best_tau} (F1={best_calib_f1:.4f})")

    # ── Step 8: Test evaluation ───────────────────────────────────────
    write_progress(8, total_steps, "Evaluating on N-Z test set", start_time=start_time)
    print("\n[Step 8] Test evaluation on N-Z...")

    test_tau_results = {}
    for tau in TAU_VALUES:
        res = eval_threshold(test_df_eval, tau)
        test_tau_results[str(tau)] = res

    # Apply best tau to test
    test_result = test_tau_results[str(best_tau)]

    # Compute ROC-AUC on test set
    y_true_test = test_df_eval['is_positive'].astype(int).values
    y_scores_test = test_df_eval['alpha_ij'].values

    try:
        roc_auc = roc_auc_score(y_true_test, y_scores_test)
    except ValueError:
        roc_auc = 0.5  # degenerate case

    print(f"  Test @ tau={best_tau}: P={test_result['precision']:.4f} R={test_result['recall']:.4f} F1={test_result['f1']:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Also evaluate all tau on test
    for tau in TAU_VALUES:
        r = test_tau_results[str(tau)]
        print(f"    tau={tau}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    # ── Cosine-only baseline ──────────────────────────────────────────
    print("\n  Cosine-only baseline comparison:")
    cosine_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    cosine_baseline = {}
    for ct in cosine_thresholds:
        calib_res = eval_threshold(calib_df, ct, score_col='decoder_cosine')
        test_res = eval_threshold(test_df_eval, ct, score_col='decoder_cosine')
        cosine_baseline[str(ct)] = {
            "calib": calib_res,
            "test": test_res,
        }
        print(f"    cosine>{ct}: Calib F1={calib_res['f1']:.4f}, Test F1={test_res['f1']:.4f}")

    # Best cosine threshold
    best_cosine_tau = max(cosine_thresholds, key=lambda t: cosine_baseline[str(t)]['calib']['f1'])
    cosine_test_f1 = cosine_baseline[str(best_cosine_tau)]['test']['f1']
    print(f"  Best cosine threshold: {best_cosine_tau} (Test F1={cosine_test_f1:.4f})")

    # Cosine ROC-AUC
    try:
        cosine_roc_auc = roc_auc_score(y_true_test, test_df_eval['decoder_cosine'].values)
    except ValueError:
        cosine_roc_auc = 0.5
    print(f"  Cosine ROC-AUC: {cosine_roc_auc:.4f}")

    # ── Step 9: LV sharpness diagnostic ───────────────────────────────
    write_progress(9, total_steps, "Computing LV sharpness diagnostic", start_time=start_time)
    print("\n[Step 9] LV sharpness diagnostic...")

    # Use ALL pairs (both calib and test) for the sharpness analysis
    # Bin alpha_ij in bins of 0.1 over [0, 3]
    bin_edges = np.arange(0, 3.1, 0.1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_rates = []
    bin_counts = []

    for i in range(len(bin_centers)):
        mask = (pairs_df['alpha_ij'] >= bin_edges[i]) & (pairs_df['alpha_ij'] < bin_edges[i + 1])
        bin_data = pairs_df[mask]
        n = len(bin_data)
        bin_counts.append(n)
        if n >= 3:  # minimum count for meaningful rate
            rate = bin_data['is_positive'].mean()
            bin_rates.append(rate)
        else:
            bin_rates.append(None)

    # Fit sigmoid and linear on non-null bins
    valid_mask = [r is not None for r in bin_rates]
    x_valid = bin_centers[valid_mask]
    y_valid = np.array([r for r in bin_rates if r is not None])

    print(f"  Valid bins: {sum(valid_mask)} / {len(bin_centers)}")
    print(f"  Bin centers (valid): {x_valid}")
    print(f"  Bin rates (valid): {y_valid}")

    # Sigmoid fit
    sigmoid_converged = False
    sigmoid_params = {"k": 0.0, "x0": 1.0}
    try:
        popt, pcov = curve_fit(sigmoid, x_valid, y_valid, p0=[1.0, 1.0], maxfev=5000,
                               bounds=([-20, -5], [20, 10]))
        sigmoid_params = {"k": float(popt[0]), "x0": float(popt[1])}
        sigmoid_converged = True
        y_pred_sig = sigmoid(x_valid, *popt)
        rss_sigmoid = np.sum((y_valid - y_pred_sig) ** 2)
        print(f"  Sigmoid fit: k={popt[0]:.4f}, x0={popt[1]:.4f}")
    except Exception as e:
        print(f"  Sigmoid fit failed: {e}")
        rss_sigmoid = float('inf')

    # Linear fit
    try:
        popt_lin, _ = curve_fit(linear, x_valid, y_valid, p0=[0.1, 0.1])
        linear_params = {"a": float(popt_lin[0]), "b": float(popt_lin[1])}
        y_pred_lin = linear(x_valid, *popt_lin)
        rss_linear = np.sum((y_valid - y_pred_lin) ** 2)
        print(f"  Linear fit: a={popt_lin[0]:.4f}, b={popt_lin[1]:.4f}")
    except Exception as e:
        print(f"  Linear fit failed: {e}")
        linear_params = {"a": 0.0, "b": 0.0}
        rss_linear = float('inf')

    # AIC comparison
    n_valid = len(x_valid)
    aic_sigmoid = compute_aic(n_valid, rss_sigmoid, 2) if sigmoid_converged else float('inf')
    aic_linear = compute_aic(n_valid, rss_linear, 2)

    winner = "sigmoid" if aic_sigmoid < aic_linear else "linear"
    print(f"  AIC sigmoid: {aic_sigmoid:.4f}")
    print(f"  AIC linear: {aic_linear:.4f}")
    print(f"  Winner: {winner}")

    # ── Step 10: Save results ─────────────────────────────────────────
    write_progress(10, total_steps, "Saving results", start_time=start_time)
    print("\n[Step 10] Saving results...")

    elapsed = time.time() - start_time

    # Compute overall statistics
    overall_absorption_rate = np.mean([
        v['absorption_rate'] for v in absorption_by_letter.values()
        if v['n_total'] > 0
    ])

    results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "FULL",
        "model": MODEL_NAME,
        "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "d_sae": D_SAE,
        "calib_letters": CALIB_LETTERS,
        "test_letters": TEST_LETTERS,
        "go_no_go": "GO" if test_result['f1'] > 0.35 else "NO_GO",
        "overall_absorption_rate": float(overall_absorption_rate),
        "absorption_by_letter": {
            letter: {
                "absorption_rate": v["absorption_rate"],
                "n_absorbed": v["n_absorbed"],
                "n_total": v["n_total"],
                "absorbed_feature_ids": v.get("absorbed_feature_ids", []),
            }
            for letter, v in absorption_by_letter.items()
        },
        "n_pairs_total": len(pairs_df),
        "n_pairs_calib": len(calib_df),
        "n_pairs_test": len(test_df_eval),
        "n_positives_calib": int(calib_df['is_positive'].sum()),
        "n_positives_test": int(test_df_eval['is_positive'].sum()),
        "calibration": {
            "tau_results": calib_tau_results,
            "best_tau": best_tau,
            "best_calib_f1": best_calib_f1,
        },
        "test_evaluation": {
            "tau": best_tau,
            "precision": test_result['precision'],
            "recall": test_result['recall'],
            "f1": test_result['f1'],
            "roc_auc": roc_auc,
            "tau_results": test_tau_results,
        },
        "cosine_baseline": {
            "best_threshold": best_cosine_tau,
            "test_f1": cosine_test_f1,
            "test_roc_auc": cosine_roc_auc,
            "all_thresholds": cosine_baseline,
        },
        "f1_delta_vs_cosine": test_result['f1'] - cosine_test_f1,
        "auc_delta_vs_cosine": roc_auc - cosine_roc_auc,
        "lv_sharpness": {
            "sigmoid_converged": sigmoid_converged,
            "sigmoid_params": sigmoid_params,
            "linear_params": linear_params,
            "aic_sigmoid": aic_sigmoid if aic_sigmoid != float('inf') else None,
            "aic_linear": aic_linear if aic_linear != float('inf') else None,
            "winner": winner,
            "bin_data": {
                "bin_centers": bin_centers.tolist(),
                "bin_rates": [float(r) if r is not None else None for r in bin_rates],
                "bin_counts": [int(c) for c in bin_counts],
            }
        },
        "pass_criteria": {
            "pass_f1_gt_035": test_result['f1'] > 0.35,
            "pass_sigmoid_converged": sigmoid_converged,
            "pass_aic_finite": aic_sigmoid != float('inf') and aic_linear != float('inf'),
            "test_f1": test_result['f1'],
            "aic_sigmoid": aic_sigmoid if aic_sigmoid != float('inf') else None,
            "aic_linear": aic_linear if aic_linear != float('inf') else None,
        },
        "runtime_seconds": elapsed,
    }

    # Save main results
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Results saved to {OUTPUT_FILE}")

    # Save pairs dataset for downstream analysis
    pairs_output = FULL_DIR / "C1B_pairs_dataset.parquet"
    pairs_df.to_parquet(pairs_output, index=False)
    print(f"  Pairs dataset saved to {pairs_output}")

    # Summary
    print(f"\n{'='*60}")
    print(f"C1B FULL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Overall absorption rate: {overall_absorption_rate:.4f}")
    print(f"  Pairs: {len(pairs_df)} total ({pairs_df['is_positive'].sum()} positive)")
    print(f"  Best threshold (tau): {best_tau}")
    print(f"  Test F1 (LV detector): {test_result['f1']:.4f}")
    print(f"  Test ROC-AUC (LV detector): {roc_auc:.4f}")
    print(f"  Test F1 (cosine baseline): {cosine_test_f1:.4f}")
    print(f"  Test ROC-AUC (cosine baseline): {cosine_roc_auc:.4f}")
    print(f"  F1 delta (LV - cosine): {test_result['f1'] - cosine_test_f1:+.4f}")
    print(f"  AUC delta (LV - cosine): {roc_auc - cosine_roc_auc:+.4f}")
    print(f"  Sharpness winner: {winner}")
    print(f"  Go/No-Go: {results['go_no_go']}")
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
                "sae": f"{SAE_RELEASE}/{SAE_ID}",
                "d_sae": D_SAE,
                "n_letters": 26,
                "n_pairs": len(pairs_df),
                "gpu": DEVICE,
            }
        }

        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"  gpu_progress.json updated")
    except Exception as e:
        print(f"  WARNING: Failed to update gpu_progress.json: {e}")

    mark_done("success", f"F1={test_result['f1']:.4f}, AUC={roc_auc:.4f}, winner={winner}")
    print(f"\n[C1B FULL] Complete. DONE marker written.")


if __name__ == "__main__":
    main()
