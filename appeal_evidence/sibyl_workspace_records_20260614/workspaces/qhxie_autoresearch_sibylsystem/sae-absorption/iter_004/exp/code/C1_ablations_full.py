"""
C1_ablations: Component 1 Ablations — Threshold Sensitivity and Cosine Pre-filter (A1, A2)
FULL MODE

Ablation experiments for H1 detector:
  A1: Threshold sensitivity — re-evaluate H1 LV detector at tau values
      {0.5, 0.75, 1.0, 1.25, 1.5} on the 13-letter test set.
      Report F1 for each tau. Confirm F1 peak is within {0.75, 1.25}.
  A2: Cosine pre-filter threshold — re-run C1A activation pair filtering
      with cosine thresholds {0.10, 0.15, 0.25}. Compute coverage (fraction
      of known absorbed pairs captured at each threshold) and precision.
      Compare LV detector (alpha_ij > tau) vs cosine-only baseline
      (cosine > threshold) at matched precision. Report F1 delta.

Uses:
  - C1B_pairs_dataset.parquet (paired absorption dataset from C1B)
  - C1B_lv_validation.json (calibrated threshold and results)
  - C1A alpha_ij parquet (jb_layer8_24k.parquet for re-filtering)
  - GPT-2 Small model + SAE (for re-computing alpha_ij at cosine 0.10)
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

# Environment
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
)

# ─── Configuration ─────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
C1A_DIR = FULL_DIR / "C1A_alpha_ij_stats"
OUTPUT_FILE = FULL_DIR / "C1_ablations.json"

TASK_ID = "C1_ablations"
SEED = 42
DEVICE = "cuda:0"

# Model/SAE for A2 re-computation
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
HOOK_NAME = "blocks.8.hook_resid_pre"
MODEL_NAME = "gpt2-small"
LAYER = 8
D_SAE = 24576

# Alpha_ij parquet from C1A (pre-filtered at cosine >= 0.15)
ALPHA_PARQUET = C1A_DIR / "jb_layer8_24k.parquet"

# C1B outputs
PAIRS_PARQUET = FULL_DIR / "C1B_pairs_dataset.parquet"
C1B_RESULTS = FULL_DIR / "C1B_lv_validation.json"

# Letters
CALIB_LETTERS = list("ABCDEFGHIJKLM")
TEST_LETTERS = list("NOPQRSTUVWXYZ")

# A1: Threshold sweep values (same as C1B + finer grid)
TAU_VALUES_COARSE = [0.5, 0.75, 1.0, 1.25, 1.5]
TAU_VALUES_FINE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0]

# A2: Cosine pre-filter thresholds
COSINE_THRESHOLDS = [0.10, 0.15, 0.25]


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
    """Evaluate binary classification at given threshold."""
    y_true = df['is_positive'].astype(int).values
    y_pred = (df[score_col] > tau).astype(int).values

    if y_true.sum() == 0 or y_pred.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def compute_roc_auc(df, score_col='alpha_ij'):
    """Compute ROC-AUC for a score column."""
    y_true = df['is_positive'].astype(int).values
    y_scores = df[score_col].values
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return 0.5


def compute_avg_precision(df, score_col='alpha_ij'):
    """Compute Average Precision for a score column."""
    y_true = df['is_positive'].astype(int).values
    y_scores = df[score_col].values
    try:
        return float(average_precision_score(y_true, y_scores))
    except ValueError:
        return 0.0


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start_time = time.time()
    write_pid()
    total_steps = 8

    print(f"[C1_ablations FULL] Starting — {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Load existing data from C1B
    # ══════════════════════════════════════════════════════════════════════
    write_progress(1, total_steps, "Loading C1B pairs dataset and results", start_time=start_time)
    print("\n[Step 1] Loading C1B data...")

    pairs_df = pd.read_parquet(PAIRS_PARQUET)
    c1b_results = json.loads(C1B_RESULTS.read_text())

    print(f"  Pairs: {len(pairs_df)} total, {pairs_df['is_positive'].sum()} positive")
    print(f"  C1B best tau: {c1b_results['calibration']['best_tau']}")
    print(f"  C1B test F1: {c1b_results['test_evaluation']['f1']:.4f}")

    # Split into calibration and test
    calib_df = pairs_df[pairs_df['letter'].isin(CALIB_LETTERS)].copy()
    test_df = pairs_df[pairs_df['letter'].isin(TEST_LETTERS)].copy()

    print(f"  Calibration: {len(calib_df)} pairs ({calib_df['is_positive'].sum()} positive)")
    print(f"  Test: {len(test_df)} pairs ({test_df['is_positive'].sum()} positive)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: A1 — Threshold Sensitivity Analysis (COARSE)
    # ══════════════════════════════════════════════════════════════════════
    write_progress(2, total_steps, "A1: Threshold sensitivity — coarse sweep", start_time=start_time)
    print("\n[Step 2] A1: Threshold sensitivity (coarse sweep)...")

    a1_coarse = {}
    for tau in TAU_VALUES_COARSE:
        calib_res = eval_threshold(calib_df, tau)
        test_res = eval_threshold(test_df, tau)
        a1_coarse[str(tau)] = {
            "calib": calib_res,
            "test": test_res,
        }
        print(f"  tau={tau:.2f}: Calib F1={calib_res['f1']:.4f} | Test F1={test_res['f1']:.4f} "
              f"P={test_res['precision']:.4f} R={test_res['recall']:.4f}")

    # Find peak tau in calibration
    best_calib_tau = max(TAU_VALUES_COARSE,
                         key=lambda t: a1_coarse[str(t)]['calib']['f1'])
    best_test_tau = max(TAU_VALUES_COARSE,
                         key=lambda t: a1_coarse[str(t)]['test']['f1'])

    peak_in_075_125 = best_calib_tau >= 0.75 and best_calib_tau <= 1.25
    print(f"  Best calibration tau: {best_calib_tau} (F1={a1_coarse[str(best_calib_tau)]['calib']['f1']:.4f})")
    print(f"  Best test tau: {best_test_tau} (F1={a1_coarse[str(best_test_tau)]['test']['f1']:.4f})")
    print(f"  F1 peak within {{0.75, 1.25}}: {'YES' if peak_in_075_125 else 'NO'}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: A1 — Fine-grained threshold sweep
    # ══════════════════════════════════════════════════════════════════════
    write_progress(3, total_steps, "A1: Fine-grained threshold sweep", start_time=start_time)
    print("\n[Step 3] A1: Fine-grained threshold sweep...")

    a1_fine = {}
    for tau in TAU_VALUES_FINE:
        calib_res = eval_threshold(calib_df, tau)
        test_res = eval_threshold(test_df, tau)
        a1_fine[str(tau)] = {
            "calib": calib_res,
            "test": test_res,
        }

    # Print summary table
    print(f"  {'tau':>6s}  {'Calib P':>9s}  {'Calib R':>9s}  {'Calib F1':>9s}  "
          f"{'Test P':>9s}  {'Test R':>9s}  {'Test F1':>9s}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
    for tau in TAU_VALUES_FINE:
        c = a1_fine[str(tau)]['calib']
        t = a1_fine[str(tau)]['test']
        print(f"  {tau:6.2f}  {c['precision']:9.4f}  {c['recall']:9.4f}  {c['f1']:9.4f}  "
              f"{t['precision']:9.4f}  {t['recall']:9.4f}  {t['f1']:9.4f}")

    best_fine_calib_tau = max(TAU_VALUES_FINE,
                              key=lambda t: a1_fine[str(t)]['calib']['f1'])
    best_fine_test_tau = max(TAU_VALUES_FINE,
                              key=lambda t: a1_fine[str(t)]['test']['f1'])
    print(f"\n  Best fine calib tau: {best_fine_calib_tau} "
          f"(F1={a1_fine[str(best_fine_calib_tau)]['calib']['f1']:.4f})")
    print(f"  Best fine test tau: {best_fine_test_tau} "
          f"(F1={a1_fine[str(best_fine_test_tau)]['test']['f1']:.4f})")

    # Compute ROC-AUC and Average Precision for alpha_ij on test
    lv_test_auc = compute_roc_auc(test_df, 'alpha_ij')
    lv_test_ap = compute_avg_precision(test_df, 'alpha_ij')
    print(f"  Alpha_ij ROC-AUC on test: {lv_test_auc:.4f}")
    print(f"  Alpha_ij Avg Precision on test: {lv_test_ap:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: A1 — Sensitivity analysis: F1 stability across tau range
    # ══════════════════════════════════════════════════════════════════════
    write_progress(4, total_steps, "A1: Stability analysis", start_time=start_time)
    print("\n[Step 4] A1: Stability analysis...")

    # Compute how much F1 degrades when moving +/- 0.25 from the peak
    test_f1s_fine = {tau: a1_fine[str(tau)]['test']['f1'] for tau in TAU_VALUES_FINE}
    best_f1 = max(test_f1s_fine.values())
    # Range of tau values where F1 is within 90% of peak
    threshold_90pct = best_f1 * 0.90
    stable_range = [t for t in TAU_VALUES_FINE if test_f1s_fine[t] >= threshold_90pct]
    if stable_range:
        stability_range = (min(stable_range), max(stable_range))
        stability_width = stability_range[1] - stability_range[0]
    else:
        stability_range = (best_fine_test_tau, best_fine_test_tau)
        stability_width = 0.0

    print(f"  Peak test F1: {best_f1:.4f} at tau={best_fine_test_tau}")
    print(f"  90% stability range: [{stability_range[0]:.2f}, {stability_range[1]:.2f}] (width={stability_width:.2f})")

    # Per-letter F1 at the best tau (on test letters)
    letter_f1s = {}
    for letter in TEST_LETTERS:
        letter_df = test_df[test_df['letter'] == letter]
        if letter_df['is_positive'].sum() == 0:
            letter_f1s[letter] = None
            continue
        res = eval_threshold(letter_df, best_fine_test_tau)
        letter_f1s[letter] = res['f1']
        print(f"    Letter {letter}: F1={res['f1']:.4f} (n_pos={letter_df['is_positive'].sum()}, n={len(letter_df)})")

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: A2 — Load model+SAE for re-computing alpha_ij at cos=0.10
    # ══════════════════════════════════════════════════════════════════════
    write_progress(5, total_steps, "A2: Loading model and SAE for cosine threshold ablation", start_time=start_time)
    print("\n[Step 5] A2: Loading model and SAE for cosine threshold re-computation...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from datasets import load_dataset

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
    sae.eval()
    print(f"  Model: {MODEL_NAME}, SAE: d_sae={sae.cfg.d_sae}")

    # Load the existing alpha_ij data (pre-filtered at cosine >= 0.15)
    alpha_df_015 = pd.read_parquet(ALPHA_PARQUET)
    print(f"  Loaded existing alpha_ij at cos>=0.15: {len(alpha_df_015)} pairs")

    # ══════════════════════════════════════════════════════════════════════
    # Step 6: A2 — Compute alpha_ij at cosine threshold 0.10
    # ══════════════════════════════════════════════════════════════════════
    write_progress(6, total_steps, "A2: Re-computing alpha_ij at cosine threshold 0.10", start_time=start_time)
    print("\n[Step 6] A2: Computing alpha_ij at cosine threshold 0.10...")

    # We need to collect activation statistics and compute pairwise stats
    # for pairs with cosine >= 0.10 but < 0.15 (those not in existing data)
    # Strategy: run same activation collection pipeline as C1A but with cos_threshold=0.10

    W_dec = sae.W_dec.detach()  # (d_sae, d_model)
    n_latents = W_dec.shape[0]

    # Compute pairwise decoder cosine similarities to find additional pairs at cos >= 0.10
    print("  Computing decoder cosine similarity matrix (batched)...")

    # For efficiency, normalize decoder weights first
    W_dec_norm = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-8)

    # Find pairs with cosine >= 0.10 but < 0.15 to augment the existing data
    # We'll do this in blocks to avoid OOM
    BLOCK_SIZE = 2048
    new_pairs_data = []

    # Pre-compute frequency from activation data (use same 10k tokens as C1A)
    print("  Loading token data for activation frequency computation...")
    try:
        ds = load_dataset("stas/openwebtext-10k", split="train")
        texts = [ex['text'] for ex in ds]
    except Exception:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            texts.append(ex['text'])
            if i >= 200:
                break

    # Tokenize and collect 10k tokens
    all_tokens = []
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=False)
        all_tokens.append(tokens.squeeze(0))
        if sum(t.shape[0] for t in all_tokens) >= 10000:
            break

    token_tensor = torch.cat(all_tokens)[:10000].unsqueeze(0).to(DEVICE)  # (1, 10000)
    print(f"  Token tensor shape: {token_tensor.shape}")

    # Run through model + SAE in batches to get activations
    print("  Running activation collection...")
    BATCH_TOKENS = 512
    n_tokens = token_tensor.shape[1]
    activation_sum = torch.zeros(n_latents, device=DEVICE)
    activation_active = torch.zeros(n_latents, device=DEVICE)
    coactivation_accumulator = {}  # We'll collect this differently

    all_sae_acts_binary = []

    for start_idx in range(0, n_tokens, BATCH_TOKENS):
        end_idx = min(start_idx + BATCH_TOKENS, n_tokens)
        batch = token_tensor[:, start_idx:end_idx]

        with torch.inference_mode():
            _, cache = model.run_with_cache(batch)
            sae_in = cache[HOOK_NAME]  # (1, seq_len, d_model)
            sae_acts = sae.encode(sae_in)  # (1, seq_len, d_sae)

            # Binary activation (> 0)
            active = (sae_acts > 0).float().squeeze(0)  # (seq_len, d_sae)
            all_sae_acts_binary.append(active.cpu())

            # Frequency accumulation
            activation_active += active.sum(dim=0)

    # Compute frequencies
    freq = (activation_active / n_tokens).cpu()  # (d_sae,)
    print(f"  Frequency computed: mean={freq.mean():.4f}, "
          f"active latents (f>0.001): {(freq > 0.001).sum().item()}")

    # Combine all binary activations
    all_binary = torch.cat(all_sae_acts_binary, dim=0)  # (n_tokens, d_sae)

    # Find latents above frequency threshold
    f_min = 0.001
    active_mask = freq > f_min
    active_indices = active_mask.nonzero().squeeze(-1).numpy()
    n_active = len(active_indices)
    print(f"  Active latents (f > {f_min}): {n_active}")

    # For A2, we compute alpha_ij at three cosine thresholds
    # The key difference: at lower cosine threshold, we include more pairs
    a2_results = {}

    for cos_threshold in COSINE_THRESHOLDS:
        print(f"\n  === Cosine threshold: {cos_threshold} ===")

        if cos_threshold == 0.15:
            # Use existing data
            alpha_data = alpha_df_015.copy()
            print(f"    Using existing C1A data: {len(alpha_data)} pairs")
        elif cos_threshold == 0.25:
            # Filter existing data
            alpha_data = alpha_df_015[alpha_df_015['decoder_cosine'] >= 0.25].copy()
            print(f"    Filtered from existing C1A data: {len(alpha_data)} pairs")
        else:
            # cos_threshold == 0.10: Need to compute additional pairs
            # Strategy: for each active latent, find neighbors with cosine >= 0.10
            print(f"    Computing new pairs with cosine >= {cos_threshold}...")

            # Compute pairwise cosines in blocks for active latents
            W_active = W_dec_norm[active_indices]  # (n_active, d_model)
            binary_active = all_binary[:, active_indices]  # (n_tokens, n_active)
            freq_active = freq[active_indices]  # (n_active,)

            new_rows = []
            pair_count = 0
            total_checked = 0

            for block_start in range(0, n_active, BLOCK_SIZE):
                block_end = min(block_start + BLOCK_SIZE, n_active)
                block_W = W_active[block_start:block_end]  # (block_size, d_model)

                # Compute cosine similarities for this block against all active
                cos_sims = torch.mm(block_W.to(DEVICE), W_active.to(DEVICE).T)  # (block, n_active)

                # Find pairs above threshold
                for local_i in range(block_end - block_start):
                    global_i = block_start + local_i
                    latent_i = int(active_indices[global_i])
                    f_i_val = float(freq_active[global_i])

                    # Find j indices with cosine above threshold
                    row_cos = cos_sims[local_i].cpu()
                    valid_j_mask = (row_cos >= cos_threshold) & (torch.arange(n_active) > global_i)
                    valid_j_indices = valid_j_mask.nonzero().squeeze(-1)
                    total_checked += len(valid_j_indices)

                    for jj in valid_j_indices:
                        j_idx = int(jj.item())
                        latent_j = int(active_indices[j_idx])
                        f_j_val = float(freq_active[j_idx])
                        cos_val = float(row_cos[j_idx])

                        # Compute co-activation rate
                        coact = (binary_active[:, global_i] * binary_active[:, j_idx]).sum().item()
                        coact_rate = coact / n_tokens

                        # Compute sigma and alpha
                        min_f = min(f_i_val, f_j_val)
                        if min_f > 0:
                            sigma = coact_rate / min_f
                            alpha = sigma * (f_j_val / f_i_val) if f_i_val > 0 else 0.0
                        else:
                            sigma = 0.0
                            alpha = 0.0

                        new_rows.append({
                            'latent_i': latent_i,
                            'latent_j': latent_j,
                            'f_i': f_i_val,
                            'f_j': f_j_val,
                            'coact_rate': coact_rate,
                            'sigma_ij': sigma,
                            'alpha_ij': alpha,
                            'decoder_cosine': cos_val,
                        })
                        pair_count += 1

                if (block_start // BLOCK_SIZE + 1) % 5 == 0:
                    print(f"      Block {block_start//BLOCK_SIZE + 1}: "
                          f"{pair_count} pairs so far...")

                # Clean CUDA cache periodically
                if (block_start // BLOCK_SIZE + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                # Combine with existing data (all existing pairs have cos >= 0.15 >= 0.10)
                alpha_data = pd.concat([alpha_df_015, new_df], ignore_index=True)
                # Remove duplicates (prefer existing data)
                alpha_data = alpha_data.drop_duplicates(
                    subset=['latent_i', 'latent_j'], keep='first'
                )
                print(f"    New pairs at cos>={cos_threshold}: {len(new_df)}")
                print(f"    Total after merge: {len(alpha_data)} pairs")
            else:
                alpha_data = alpha_df_015.copy()
                print(f"    No new pairs found at cos>={cos_threshold}")

        # ── Build absorption detection dataset at this cosine threshold ──
        # For each known absorbed pair from C1B, check if it exists in alpha_data
        # This measures coverage

        c1b_absorption = c1b_results['absorption_by_letter']

        # Build lookup from alpha_data
        alpha_lookup = {}
        cosine_lookup = {}
        for _, row in alpha_data.iterrows():
            key = (int(row['latent_i']), int(row['latent_j']))
            alpha_lookup[key] = float(row['alpha_ij'])
            cosine_lookup[key] = float(row['decoder_cosine'])

        # Check coverage: how many positive pairs from C1B are in the filtered set?
        positive_pairs = pairs_df[pairs_df['is_positive']].copy()
        n_total_positive = len(positive_pairs)
        n_covered = 0
        n_covered_with_nonzero_alpha = 0

        for _, row in positive_pairs.iterrows():
            li, lj = int(row['latent_i']), int(row['latent_j'])
            key1 = (li, lj)
            key2 = (lj, li)
            if key1 in alpha_lookup or key2 in alpha_lookup:
                n_covered += 1
                a = alpha_lookup.get(key1, alpha_lookup.get(key2, 0.0))
                if a > 0:
                    n_covered_with_nonzero_alpha += 1

        coverage = n_covered / max(n_total_positive, 1)
        coverage_nonzero = n_covered_with_nonzero_alpha / max(n_total_positive, 1)

        # Now build updated pairs dataset using alpha_ij from this threshold's data
        # Re-assign alpha_ij to each pair based on the new data
        updated_pairs = pairs_df.copy()
        new_alpha_vals = []
        new_cosine_vals = []

        for _, row in updated_pairs.iterrows():
            li, lj = int(row['latent_i']), int(row['latent_j'])
            key1 = (li, lj)
            key2 = (lj, li)
            if key1 in alpha_lookup:
                new_alpha_vals.append(alpha_lookup[key1])
                new_cosine_vals.append(cosine_lookup[key1])
            elif key2 in alpha_lookup:
                new_alpha_vals.append(alpha_lookup[key2])
                new_cosine_vals.append(cosine_lookup[key2])
            else:
                new_alpha_vals.append(0.0)
                new_cosine_vals.append(float(row['decoder_cosine']))

        updated_pairs['alpha_ij_updated'] = new_alpha_vals
        updated_pairs['decoder_cosine_updated'] = new_cosine_vals

        # Evaluate LV detector at this cosine threshold
        test_updated = updated_pairs[updated_pairs['letter'].isin(TEST_LETTERS)].copy()
        calib_updated = updated_pairs[updated_pairs['letter'].isin(CALIB_LETTERS)].copy()

        # Best tau on updated calibration
        best_tau_updated = None
        best_f1_updated = -1
        tau_sweep_updated = {}
        for tau in TAU_VALUES_COARSE:
            calib_res = eval_threshold(calib_updated, tau, score_col='alpha_ij_updated')
            test_res = eval_threshold(test_updated, tau, score_col='alpha_ij_updated')
            tau_sweep_updated[str(tau)] = {"calib": calib_res, "test": test_res}
            if calib_res['f1'] > best_f1_updated:
                best_f1_updated = calib_res['f1']
                best_tau_updated = tau

        test_at_best = tau_sweep_updated[str(best_tau_updated)]['test']
        lv_roc_auc = compute_roc_auc(test_updated, 'alpha_ij_updated')
        lv_avg_prec = compute_avg_precision(test_updated, 'alpha_ij_updated')

        # Cosine-only baseline at this threshold
        cosine_thresholds_baseline = [cos_threshold, 0.15, 0.20, 0.25, 0.30]
        cosine_baseline_results = {}
        best_cosine_f1 = -1
        best_cosine_th = None
        for ct in cosine_thresholds_baseline:
            cr = eval_threshold(calib_updated, ct, score_col='decoder_cosine_updated')
            tr = eval_threshold(test_updated, ct, score_col='decoder_cosine_updated')
            cosine_baseline_results[str(ct)] = {"calib": cr, "test": tr}
            if cr['f1'] > best_cosine_f1:
                best_cosine_f1 = cr['f1']
                best_cosine_th = ct

        cosine_test_at_best = cosine_baseline_results[str(best_cosine_th)]['test']
        cosine_roc_auc = compute_roc_auc(test_updated, 'decoder_cosine_updated')

        f1_delta = test_at_best['f1'] - cosine_test_at_best['f1']

        a2_results[str(cos_threshold)] = {
            "n_pairs_in_alpha_data": len(alpha_data),
            "coverage_of_positive_pairs": float(coverage),
            "coverage_nonzero_alpha": float(coverage_nonzero),
            "n_positive_covered": n_covered,
            "n_positive_total": n_total_positive,
            "lv_detector": {
                "best_tau": best_tau_updated,
                "test_f1": test_at_best['f1'],
                "test_precision": test_at_best['precision'],
                "test_recall": test_at_best['recall'],
                "test_roc_auc": lv_roc_auc,
                "test_avg_precision": lv_avg_prec,
                "tau_sweep": tau_sweep_updated,
            },
            "cosine_baseline": {
                "best_threshold": best_cosine_th,
                "test_f1": cosine_test_at_best['f1'],
                "test_precision": cosine_test_at_best['precision'],
                "test_recall": cosine_test_at_best['recall'],
                "test_roc_auc": cosine_roc_auc,
                "all_thresholds": cosine_baseline_results,
            },
            "f1_delta_lv_minus_cosine": f1_delta,
        }

        print(f"    Coverage: {coverage:.4f} ({n_covered}/{n_total_positive})")
        print(f"    Coverage (nonzero alpha): {coverage_nonzero:.4f}")
        print(f"    LV detector: best_tau={best_tau_updated}, "
              f"Test F1={test_at_best['f1']:.4f}, AUC={lv_roc_auc:.4f}")
        print(f"    Cosine baseline: best_th={best_cosine_th}, "
              f"Test F1={cosine_test_at_best['f1']:.4f}, AUC={cosine_roc_auc:.4f}")
        print(f"    F1 delta (LV - cosine): {f1_delta:+.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7: A2 — Coverage summary and verification
    # ══════════════════════════════════════════════════════════════════════
    write_progress(7, total_steps, "A2: Generating coverage summary", start_time=start_time)
    print("\n[Step 7] A2: Coverage summary...")

    coverage_015 = a2_results['0.15']['coverage_of_positive_pairs']
    coverage_pass = coverage_015 >= 0.80
    print(f"  Coverage at cosine=0.15: {coverage_015:.4f} "
          f"({'PASS (>= 0.80)' if coverage_pass else 'FAIL (< 0.80)'})")

    # Compare coverage across thresholds
    print(f"\n  Cosine Threshold | Coverage | N_pairs | LV F1 | Cosine F1 | F1 delta")
    print(f"  {'─'*16} | {'─'*8} | {'─'*7} | {'─'*6} | {'─'*9} | {'─'*8}")
    for ct in COSINE_THRESHOLDS:
        r = a2_results[str(ct)]
        print(f"  {ct:16.2f} | {r['coverage_of_positive_pairs']:8.4f} | "
              f"{r['n_pairs_in_alpha_data']:7d} | "
              f"{r['lv_detector']['test_f1']:6.4f} | "
              f"{r['cosine_baseline']['test_f1']:9.4f} | "
              f"{r['f1_delta_lv_minus_cosine']:+8.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 8: Save results
    # ══════════════════════════════════════════════════════════════════════
    write_progress(8, total_steps, "Saving results", start_time=start_time)
    print("\n[Step 8] Saving results...")

    elapsed = time.time() - start_time

    # Compile full results
    results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "FULL",
        "model": MODEL_NAME,
        "model_note": "GPT-2 Small (open-model anchor)",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "d_sae": D_SAE,
        "A1_threshold_sensitivity": {
            "description": "Re-evaluate H1 LV detector at various tau values on 13-letter test set (N-Z)",
            "coarse_sweep": {
                "tau_values": TAU_VALUES_COARSE,
                "results": a1_coarse,
                "best_calib_tau": best_calib_tau,
                "best_test_tau": best_test_tau,
                "peak_in_075_125": peak_in_075_125,
            },
            "fine_sweep": {
                "tau_values": TAU_VALUES_FINE,
                "results": a1_fine,
                "best_calib_tau": best_fine_calib_tau,
                "best_test_tau": best_fine_test_tau,
                "test_roc_auc": lv_test_auc,
                "test_avg_precision": lv_test_ap,
            },
            "stability": {
                "peak_test_f1": best_f1,
                "peak_tau": best_fine_test_tau,
                "stability_range_90pct": {
                    "lower": stability_range[0],
                    "upper": stability_range[1],
                    "width": stability_width,
                },
                "per_letter_f1": {k: v for k, v in letter_f1s.items()},
            },
            "conclusion": (
                f"F1 peak at tau={best_fine_test_tau} (F1={best_f1:.4f}). "
                f"Peak {'is' if peak_in_075_125 else 'is NOT'} within {{0.75, 1.25}}. "
                f"90% stability width: {stability_width:.2f}."
            ),
        },
        "A2_cosine_prefilter": {
            "description": "Re-run C1A with cosine thresholds {0.10, 0.15, 0.25}; "
                          "compute coverage and precision; compare LV vs cosine-only",
            "cosine_thresholds": COSINE_THRESHOLDS,
            "results": a2_results,
            "coverage_at_015": coverage_015,
            "coverage_pass_80pct": coverage_pass,
            "conclusion": (
                f"Coverage at cos=0.15: {coverage_015:.4f} "
                f"({'PASS' if coverage_pass else 'FAIL'} >= 80% criterion). "
                f"Relaxing to cos=0.10 gives {a2_results['0.1']['coverage_of_positive_pairs']:.4f} coverage. "
                f"Tightening to cos=0.25 gives {a2_results['0.25']['coverage_of_positive_pairs']:.4f} coverage."
            ),
        },
        "pass_criteria": {
            "a1_f1_nonzero_at_3_taus": sum(
                1 for tau in TAU_VALUES_COARSE
                if a1_coarse[str(tau)]['test']['f1'] > 0
            ) >= 3,
            "a1_peak_in_075_125": peak_in_075_125,
            "a2_coverage_at_015_ge_80pct": coverage_pass,
            "all_results_written": True,
        },
        "runtime_seconds": elapsed,
    }

    # Save
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Results saved to {OUTPUT_FILE}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"C1_ABLATIONS FULL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  A1: Threshold Sensitivity")
    print(f"    Best tau (coarse): {best_calib_tau} (calib F1={a1_coarse[str(best_calib_tau)]['calib']['f1']:.4f})")
    print(f"    Best tau (fine):   {best_fine_test_tau} (test F1={best_f1:.4f})")
    print(f"    Peak in {{0.75, 1.25}}: {'YES' if peak_in_075_125 else 'NO'}")
    print(f"    90% stability range: [{stability_range[0]:.2f}, {stability_range[1]:.2f}]")
    print(f"    ROC-AUC: {lv_test_auc:.4f}")

    print(f"\n  A2: Cosine Pre-filter")
    for ct in COSINE_THRESHOLDS:
        r = a2_results[str(ct)]
        print(f"    cos >= {ct}: coverage={r['coverage_of_positive_pairs']:.4f}, "
              f"LV_F1={r['lv_detector']['test_f1']:.4f}, "
              f"cosine_F1={r['cosine_baseline']['test_f1']:.4f}, "
              f"delta={r['f1_delta_lv_minus_cosine']:+.4f}")
    print(f"    Coverage at 0.15 >= 80%: {'PASS' if coverage_pass else 'FAIL'}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    # ── Update gpu_progress.json ─────────────────────────────────────────
    try:
        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        gp = json.loads(gp_path.read_text())

        if TASK_ID not in gp.get("completed", []):
            gp.setdefault("completed", []).append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 30,
            "actual_min": int(elapsed / 60),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": MODEL_NAME,
                "sae": f"{SAE_RELEASE}/{SAE_ID}",
                "d_sae": D_SAE,
                "ablation_A1_tau_values": TAU_VALUES_COARSE,
                "ablation_A2_cosine_thresholds": COSINE_THRESHOLDS,
                "gpu": DEVICE,
                "gpu_count": 1,
            }
        }

        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"  gpu_progress.json updated")
    except Exception as e:
        print(f"  WARNING: Failed to update gpu_progress.json: {e}")

    mark_done("success",
              f"A1: best_tau={best_fine_test_tau}, F1={best_f1:.4f}, "
              f"peak_in_075_125={peak_in_075_125}; "
              f"A2: coverage@0.15={coverage_015:.4f}, pass={coverage_pass}")
    print(f"\n[C1_ablations FULL] Complete. DONE marker written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        mark_done("failed", str(e))
        sys.exit(1)
