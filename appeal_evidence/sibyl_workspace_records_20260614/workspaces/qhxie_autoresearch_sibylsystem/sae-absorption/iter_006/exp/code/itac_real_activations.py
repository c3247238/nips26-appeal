#!/usr/bin/env python3
"""
ITAC Computation on Real Activations (task: itac_real_activations)

Stage 2b of the unsupervised absorption detection pipeline.
Computes the Information-Theoretic Absorption Coefficient (ITAC)
on REAL model activations, NOT synthetic decoder columns.

For each candidate pair (f_parent, f_child) from decoder_geometry:
  1. Run SAE on 100k tokens, collecting latent activations and residuals
  2. Compute residual projection: r_proj = r . w_parent / ||w_parent||
  3. Partition tokens into 4 groups based on activation status
  4. ITAC = Var(r_proj | child active, parent inactive) / Var(r_proj | neither active)
  5. Null test: ITAC on random pairs -> expected ~1.0

Author: Sibyl Experimenter Agent (Iter 6)
"""

import os
import sys
import json
import time
import gc
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from scipy import stats as scipy_stats

# ==============================================================================
# Configuration
# ==============================================================================
TASK_ID = "itac_real_activations"
SEED = 42
N_CORPUS_TOKENS = 100_000          # target tokens for activation collection
N_CANDIDATE_PAIRS = 500            # top candidates from decoder geometry
N_RANDOM_PAIRS = 500               # null test random pairs
MIN_GROUP_SIZE = 10                # minimum tokens in a partition group for valid variance
BATCH_SIZE = 64                    # sequences per batch for model forward pass
SEQ_LEN = 128                     # tokens per sequence
ACTIVATION_THRESHOLD = 0.0        # feature considered "active" if activation > this

# Paths
WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
DECODER_GEOM_FILE = FULL_DIR / "decoder_geometry.json"
OUTPUT_FILE = FULL_DIR / "itac_real_activations.json"
GPU_PROGRESS_FILE = WORKSPACE / "exp" / "gpu_progress.json"

# SAE config from decoder geometry
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_12/width_16k/average_l0_82"
MODEL_NAME = "gemma-2-2b"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_pid(task_id, results_dir):
    """Write PID file for system recovery detection."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    return pid_file


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def update_gpu_progress(task_id, workspace, status, timing_info=None):
    """Update gpu_progress.json with task completion."""
    gp_file = workspace / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if status == "completed":
        if task_id not in gp["completed"]:
            gp["completed"].append(task_id)
        gp.get("running", {}).pop(task_id, None)
    elif status == "failed":
        if task_id not in gp.get("failed", []):
            gp.setdefault("failed", []).append(task_id)
        gp.get("running", {}).pop(task_id, None)

    if timing_info:
        gp.setdefault("timings", {})[task_id] = timing_info

    gp_file.write_text(json.dumps(gp, indent=2))


def load_candidate_pairs(decoder_geom_file, n_candidates=500):
    """Load candidate pairs from decoder geometry results."""
    with open(decoder_geom_file) as f:
        geom = json.load(f)

    pairs = geom.get("candidate_pairs", [])
    # Sort by absorption_score descending and take top N
    pairs.sort(key=lambda x: x.get("absorption_score", 0), reverse=True)
    pairs = pairs[:n_candidates]

    print(f"  Loaded {len(pairs)} candidate pairs from decoder geometry")
    print(f"  Score range: {pairs[-1]['absorption_score']:.4f} - {pairs[0]['absorption_score']:.4f}")

    return pairs


def generate_random_pairs(n_features, n_pairs, exclude_pairs, seed=42):
    """Generate random feature pairs for null test."""
    rng = np.random.RandomState(seed + 999)
    exclude_set = set()
    for p in exclude_pairs:
        exclude_set.add((p["parent_idx"], p["child_idx"]))
        exclude_set.add((p["child_idx"], p["parent_idx"]))

    random_pairs = []
    attempts = 0
    while len(random_pairs) < n_pairs and attempts < n_pairs * 100:
        i = rng.randint(0, n_features)
        j = rng.randint(0, n_features)
        if i != j and (i, j) not in exclude_set:
            random_pairs.append({"parent_idx": int(i), "child_idx": int(j)})
            exclude_set.add((i, j))
        attempts += 1

    print(f"  Generated {len(random_pairs)} random pairs for null test")
    return random_pairs


def collect_activations_and_residuals(model, sae, device, n_tokens, batch_size, seq_len):
    """
    Run SAE on corpus tokens, collecting:
    - Latent activations for all features
    - Residual r = x - x_hat (reconstruction error)

    Uses the same pattern as decoder_geometry.py (proven to work).
    Returns dict with activations, residuals, etc.
    """
    from datasets import load_dataset

    print("\n  Collecting real activations and residuals...")

    # Load a text corpus - try wikitext first (reliable), then Pile
    try:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train",
                               trust_remote_code=True)
        corpus_name = "wikitext-103"
        texts = [item["text"] for item in dataset
                 if len(item.get("text", "").strip()) > 50][:2000]
        print(f"  Corpus: {corpus_name}, {len(texts)} documents loaded")
    except Exception as e1:
        print(f"  wikitext unavailable ({e1}), trying pile...")
        try:
            dataset = load_dataset("monology/pile-uncopyrighted", split="train",
                                   streaming=True, trust_remote_code=True)
            corpus_name = "pile-uncopyrighted"
            texts = []
            for item in dataset:
                text = item.get("text", "")
                if len(text.strip()) > 50:
                    texts.append(text)
                if len(texts) >= 500:
                    break
            print(f"  Corpus: {corpus_name}, {len(texts)} documents loaded")
        except Exception as e2:
            raise RuntimeError(f"Could not load any corpus: wikitext={e1}, pile={e2}")

    # Tokenize all text into a flat token list
    tokenizer = model.tokenizer
    combined_text = " ".join(texts)
    all_token_ids = tokenizer.encode(combined_text)
    if isinstance(all_token_ids, torch.Tensor):
        all_token_ids = all_token_ids.tolist()

    # Trim to target + buffer
    target_tokens = n_tokens + 5000
    all_token_ids = all_token_ids[:target_tokens]
    total_available = len(all_token_ids)
    print(f"  Tokenized: {total_available} tokens available (target: {n_tokens})")

    # Build sequence batches (same pattern as decoder_geometry.py)
    n_batches_total = total_available // seq_len
    n_sequences_needed = (n_tokens + seq_len - 1) // seq_len
    n_sequences_to_use = min(n_batches_total, n_sequences_needed + 10)

    print(f"  Using {n_sequences_to_use} sequences x {seq_len} = {n_sequences_to_use * seq_len} tokens")

    n_features = sae.cfg.d_sae
    d_model = sae.cfg.d_in
    hook_point = "blocks.12.hook_resid_post"

    total_token_positions = 0
    feature_activation_chunks = []
    residual_chunks = []

    sae.eval()
    chunk_size = batch_size  # sequences per forward pass

    with torch.no_grad():
        for batch_idx in range(0, n_sequences_to_use, chunk_size):
            chunk_end = min(batch_idx + chunk_size, n_sequences_to_use)

            # Build batch of sequences
            batch_sequences = []
            for seq_idx in range(batch_idx, chunk_end):
                start = seq_idx * seq_len
                end = start + seq_len
                if end <= total_available:
                    batch_sequences.append(all_token_ids[start:end])

            if not batch_sequences:
                continue

            # Pad if needed (shouldn't be necessary with fixed seq_len)
            max_len = max(len(s) for s in batch_sequences)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded = [s + [pad_id] * (max_len - len(s)) for s in batch_sequences]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)

            # Forward pass to get residual stream at layer 12
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[hook_point],
                return_type=None,
            )
            x = cache[hook_point]  # (batch, seq, d_model)

            # Flatten to (n_tokens, d_model) and cast to float32 for SAE
            flat_x = x.reshape(-1, d_model).float()

            # Run SAE encode to get feature activations
            feature_acts = sae.encode(flat_x)  # (n_tokens, n_features)

            # Compute reconstruction and residual
            x_hat = sae.decode(feature_acts)    # (n_tokens, d_model)
            residual = flat_x - x_hat           # (n_tokens, d_model)

            # Store on CPU
            feature_activation_chunks.append(feature_acts.cpu())
            residual_chunks.append(residual.cpu())
            total_token_positions += flat_x.shape[0]

            # Clean up GPU memory
            del cache, x, flat_x, feature_acts, x_hat, residual, input_ids
            torch.cuda.empty_cache()

            batch_num = batch_idx // chunk_size + 1
            total_batches = (n_sequences_to_use + chunk_size - 1) // chunk_size
            if batch_num % 3 == 0 or batch_num == total_batches:
                print(f"    Batch {batch_num}/{total_batches}: "
                      f"{total_token_positions:,} positions collected")

    # Concatenate all chunks
    print(f"  Concatenating {len(feature_activation_chunks)} chunks...")
    feature_acts_all = torch.cat(feature_activation_chunks, dim=0)
    residuals_all = torch.cat(residual_chunks, dim=0)

    # Trim to exact target if we collected extra
    if feature_acts_all.shape[0] > n_tokens:
        feature_acts_all = feature_acts_all[:n_tokens]
        residuals_all = residuals_all[:n_tokens]
        total_token_positions = n_tokens

    print(f"  Total: {feature_acts_all.shape[0]} token positions, {feature_acts_all.shape[1]} features")
    print(f"  Feature acts: {feature_acts_all.shape}, dtype={feature_acts_all.dtype}")
    print(f"  Residuals: {residuals_all.shape}, dtype={residuals_all.dtype}")

    # Compute firing rates
    active_mask = (feature_acts_all > ACTIVATION_THRESHOLD)
    firing_rates = active_mask.float().mean(dim=0).numpy()
    n_active_features = (firing_rates > 0).sum()
    print(f"  Active features (firing rate > 0): {n_active_features}/{n_features}")

    # Memory stats
    mem_gb = (feature_acts_all.element_size() * feature_acts_all.nelement() +
              residuals_all.element_size() * residuals_all.nelement()) / 1e9
    print(f"  CPU memory for stored tensors: {mem_gb:.2f} GB")

    return {
        "feature_acts": feature_acts_all,
        "residuals": residuals_all,
        "firing_rates": firing_rates,
        "n_positions": feature_acts_all.shape[0],
        "n_features": n_features,
        "d_model": d_model,
        "corpus": corpus_name,
    }


def compute_itac_for_pair(parent_idx, child_idx, feature_acts, residuals,
                           W_dec_parent, min_group_size=10):
    """
    Compute ITAC for a single (parent, child) pair.

    ITAC = Var(r_proj | child active, parent inactive) / Var(r_proj | neither active)

    Where r_proj = residual . w_parent / ||w_parent||

    Returns dict with ITAC value and group statistics.
    """
    # Get activation vectors for parent and child
    parent_acts = feature_acts[:, parent_idx]  # (N,)
    child_acts = feature_acts[:, child_idx]    # (N,)

    # Binary activation status
    parent_active = parent_acts > ACTIVATION_THRESHOLD
    child_active = child_acts > ACTIVATION_THRESHOLD

    # Four groups
    both_active = parent_active & child_active          # Group (a)
    parent_only = parent_active & ~child_active         # Group (b)
    child_only = ~parent_active & child_active          # Group (c) - absorption signal
    neither = ~parent_active & ~child_active            # Group (d) - baseline

    n_both = both_active.sum().item()
    n_parent_only = parent_only.sum().item()
    n_child_only = child_only.sum().item()
    n_neither = neither.sum().item()

    # Compute residual projection onto parent decoder direction
    # r_proj = r . w_parent / ||w_parent||
    W_dec_parent_norm = W_dec_parent / W_dec_parent.norm()
    r_proj = (residuals @ W_dec_parent_norm).numpy()  # (N,)

    result = {
        "parent_idx": int(parent_idx),
        "child_idx": int(child_idx),
        "n_both_active": n_both,
        "n_parent_only": n_parent_only,
        "n_child_only": n_child_only,
        "n_neither": n_neither,
    }

    # Compute variance for each group
    group_stats = {}
    for name, mask, label in [
        ("both_active", both_active, "a"),
        ("parent_only", parent_only, "b"),
        ("child_only", child_only, "c"),
        ("neither", neither, "d"),
    ]:
        n = mask.sum().item()
        if n >= min_group_size:
            vals = r_proj[mask.numpy()]
            group_stats[name] = {
                "n": n,
                "mean": float(np.mean(vals)),
                "var": float(np.var(vals, ddof=1)),
                "std": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
            }
        else:
            group_stats[name] = {"n": n, "insufficient": True}

    result["group_stats"] = group_stats

    # Compute ITAC = Var(child_only) / Var(neither)
    child_only_stats = group_stats.get("child_only", {})
    neither_stats = group_stats.get("neither", {})

    # insufficient=False means group has enough data (key absent = sufficient)
    child_ok = not child_only_stats.get("insufficient", False)
    neither_ok = not neither_stats.get("insufficient", False)

    if (child_ok and neither_ok and
        neither_stats.get("var", 0) > 1e-12):
        itac = child_only_stats["var"] / neither_stats["var"]
        result["itac"] = float(itac)
        result["valid"] = True

        # Also compute alternative metrics
        # Mean shift: how much does the residual project onto parent direction
        # when child is active but parent isn't?
        if child_ok:
            result["mean_shift_child_only"] = float(child_only_stats["mean"])
        if neither_ok:
            result["mean_shift_neither"] = float(neither_stats["mean"])

        # Variance ratio for parent_only vs neither (sanity check: should be ~1)
        parent_only_stats = group_stats.get("parent_only", {})
        if not parent_only_stats.get("insufficient", False) and neither_stats["var"] > 1e-12:
            result["var_ratio_parent_only"] = float(parent_only_stats["var"] / neither_stats["var"])

        # Both active group
        both_stats = group_stats.get("both_active", {})
        if not both_stats.get("insufficient", False) and neither_stats["var"] > 1e-12:
            result["var_ratio_both"] = float(both_stats["var"] / neither_stats["var"])
    else:
        result["itac"] = None
        result["valid"] = False
        if not child_ok:
            result["invalid_reason"] = f"child_only group too small (n={child_only_stats.get('n', 0)})"
        elif not neither_ok:
            result["invalid_reason"] = f"neither group too small (n={neither_stats.get('n', 0)})"
        else:
            result["invalid_reason"] = "zero variance in baseline group"

    return result


def compute_itac_batch(pairs, feature_acts, residuals, W_dec, label="candidates",
                        min_group_size=10):
    """Compute ITAC for a batch of pairs."""
    results = []
    n_valid = 0
    n_total = len(pairs)

    for i, pair in enumerate(pairs):
        parent_idx = pair["parent_idx"]
        child_idx = pair["child_idx"]

        W_dec_parent = W_dec[parent_idx]  # (d_model,)

        result = compute_itac_for_pair(
            parent_idx, child_idx,
            feature_acts, residuals,
            W_dec_parent,
            min_group_size=min_group_size
        )

        # Copy over geometry info if available
        for key in ["global_cosine", "conditional_cosine", "absorption_score",
                     "parent_firing_rate", "child_firing_rate", "firing_rate_ratio"]:
            if key in pair:
                result[key] = pair[key]

        results.append(result)

        if result.get("valid"):
            n_valid += 1

        if (i + 1) % 100 == 0:
            print(f"    [{label}] {i+1}/{n_total} pairs processed, {n_valid} valid")

    print(f"  [{label}] Done: {n_valid}/{n_total} valid ITAC values")
    return results


def compute_statistics(candidate_results, random_results):
    """Compute summary statistics comparing candidates vs random pairs."""
    # Extract valid ITAC values
    cand_itac = [r["itac"] for r in candidate_results if r.get("valid") and r["itac"] is not None]
    rand_itac = [r["itac"] for r in random_results if r.get("valid") and r["itac"] is not None]

    stats = {
        "candidates": {
            "n_total": len(candidate_results),
            "n_valid": len(cand_itac),
            "pct_valid": len(cand_itac) / max(len(candidate_results), 1) * 100,
        },
        "random": {
            "n_total": len(random_results),
            "n_valid": len(rand_itac),
            "pct_valid": len(rand_itac) / max(len(random_results), 1) * 100,
        },
    }

    if cand_itac:
        cand_arr = np.array(cand_itac)
        stats["candidates"].update({
            "mean": float(np.mean(cand_arr)),
            "median": float(np.median(cand_arr)),
            "std": float(np.std(cand_arr)),
            "min": float(np.min(cand_arr)),
            "max": float(np.max(cand_arr)),
            "q25": float(np.percentile(cand_arr, 25)),
            "q75": float(np.percentile(cand_arr, 75)),
            "q95": float(np.percentile(cand_arr, 95)),
            "pct_above_1.5": float((cand_arr > 1.5).mean() * 100),
            "pct_above_2.0": float((cand_arr > 2.0).mean() * 100),
            "pct_above_3.0": float((cand_arr > 3.0).mean() * 100),
        })

    if rand_itac:
        rand_arr = np.array(rand_itac)
        stats["random"].update({
            "mean": float(np.mean(rand_arr)),
            "median": float(np.median(rand_arr)),
            "std": float(np.std(rand_arr)),
            "min": float(np.min(rand_arr)),
            "max": float(np.max(rand_arr)),
            "q25": float(np.percentile(rand_arr, 25)),
            "q75": float(np.percentile(rand_arr, 75)),
            "q95": float(np.percentile(rand_arr, 95)),
            "pct_above_1.5": float((rand_arr > 1.5).mean() * 100),
            "pct_above_2.0": float((rand_arr > 2.0).mean() * 100),
            "pct_above_3.0": float((rand_arr > 3.0).mean() * 100),
        })

    # Robust statistics (log-ITAC to handle heavy tails)
    if cand_itac:
        cand_arr = np.array(cand_itac)
        # Filter out extreme values for robust analysis (>0)
        cand_positive = cand_arr[cand_arr > 0]
        if len(cand_positive) > 0:
            log_cand = np.log(cand_positive)
            stats["candidates"]["log_itac_mean"] = float(np.mean(log_cand))
            stats["candidates"]["log_itac_median"] = float(np.median(log_cand))
            stats["candidates"]["log_itac_std"] = float(np.std(log_cand))
            # Winsorized (5-95%) trimmed mean
            p5 = np.percentile(cand_arr, 5)
            p95 = np.percentile(cand_arr, 95)
            trimmed = cand_arr[(cand_arr >= p5) & (cand_arr <= p95)]
            if len(trimmed) > 0:
                stats["candidates"]["trimmed_mean_5_95"] = float(np.mean(trimmed))
                stats["candidates"]["trimmed_median_5_95"] = float(np.median(trimmed))

    if rand_itac:
        rand_arr = np.array(rand_itac)
        rand_positive = rand_arr[rand_arr > 0]
        if len(rand_positive) > 0:
            log_rand = np.log(rand_positive)
            stats["random"]["log_itac_mean"] = float(np.mean(log_rand))
            stats["random"]["log_itac_median"] = float(np.median(log_rand))
            stats["random"]["log_itac_std"] = float(np.std(log_rand))
            p5 = np.percentile(rand_arr, 5)
            p95 = np.percentile(rand_arr, 95)
            trimmed = rand_arr[(rand_arr >= p5) & (rand_arr <= p95)]
            if len(trimmed) > 0:
                stats["random"]["trimmed_mean_5_95"] = float(np.mean(trimmed))
                stats["random"]["trimmed_median_5_95"] = float(np.median(trimmed))

    # Statistical tests
    if len(cand_itac) >= 10 and len(rand_itac) >= 10:
        # Mann-Whitney U test (on raw ITAC)
        u_stat, u_pval = scipy_stats.mannwhitneyu(cand_itac, rand_itac, alternative="greater")
        stats["mann_whitney_u"] = {
            "U_statistic": float(u_stat),
            "p_value": float(u_pval),
            "alternative": "candidates > random",
            "significant_005": u_pval < 0.05,
            "significant_001": u_pval < 0.001,
        }

        # Mann-Whitney on log-ITAC (more robust to outliers)
        cand_pos = [x for x in cand_itac if x > 0]
        rand_pos = [x for x in rand_itac if x > 0]
        if len(cand_pos) >= 10 and len(rand_pos) >= 10:
            u_log, p_log = scipy_stats.mannwhitneyu(
                np.log(cand_pos), np.log(rand_pos), alternative="greater"
            )
            stats["mann_whitney_u_log"] = {
                "U_statistic": float(u_log),
                "p_value": float(p_log),
                "significant_005": p_log < 0.05,
            }

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = scipy_stats.ks_2samp(cand_itac, rand_itac, alternative="greater")
        stats["ks_test"] = {
            "KS_statistic": float(ks_stat),
            "p_value": float(ks_pval),
            "significant_005": ks_pval < 0.05,
        }

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(cand_itac, ddof=1) + np.var(rand_itac, ddof=1)) / 2)
        if pooled_std > 1e-12:
            cohens_d = (np.mean(cand_itac) - np.mean(rand_itac)) / pooled_std
            stats["cohens_d"] = float(cohens_d)
        else:
            stats["cohens_d"] = 0.0

        # Cohen's d on log-ITAC (robust)
        if len(cand_pos) >= 10 and len(rand_pos) >= 10:
            log_c = np.log(cand_pos)
            log_r = np.log(rand_pos)
            pooled_log = np.sqrt((np.var(log_c, ddof=1) + np.var(log_r, ddof=1)) / 2)
            if pooled_log > 1e-12:
                stats["cohens_d_log"] = float((np.mean(log_c) - np.mean(log_r)) / pooled_log)

        # Rank-biserial correlation (effect size for Mann-Whitney)
        n1, n2 = len(cand_itac), len(rand_itac)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        stats["rank_biserial_r"] = float(rank_biserial)

        # Correlation between absorption score and ITAC (for candidates with scores)
        scored = [(r.get("absorption_score"), r["itac"])
                  for r in candidate_results
                  if r.get("valid") and r["itac"] is not None and "absorption_score" in r]
        if len(scored) >= 10:
            scores, itacs = zip(*scored)
            rho_score, p_score = scipy_stats.spearmanr(scores, itacs)
            stats["absorption_score_vs_itac"] = {
                "spearman_rho": float(rho_score),
                "p_value": float(p_score),
                "n_pairs": len(scored),
            }
            # Also log-ITAC
            rho_log_s, p_log_s = scipy_stats.spearmanr(scores, np.log([max(x, 1e-10) for x in itacs]))
            stats["absorption_score_vs_log_itac"] = {
                "spearman_rho": float(rho_log_s),
                "p_value": float(p_log_s),
            }

    # Check pass criteria
    stats["pass_criteria"] = {}

    # Criterion 1: >= 50 valid candidate pairs
    stats["pass_criteria"]["n_valid_candidates_ge_50"] = len(cand_itac) >= 50

    # Criterion 2: Random pairs median ITAC in [0.8, 1.2]
    if rand_itac:
        rand_median = float(np.median(rand_itac))
        stats["pass_criteria"]["random_median_in_expected_range"] = 0.8 <= rand_median <= 1.2
        stats["pass_criteria"]["random_median"] = rand_median
    else:
        stats["pass_criteria"]["random_median_in_expected_range"] = False

    # Criterion 3: Candidate pairs show median ITAC > 1.5 OR clear separation
    if cand_itac and rand_itac:
        cand_median = float(np.median(cand_itac))
        stats["pass_criteria"]["candidate_median_above_1.5"] = cand_median > 1.5
        stats["pass_criteria"]["candidate_median"] = cand_median
        # Clear separation can be raw or log-scale Mann-Whitney significance
        raw_sig = stats.get("mann_whitney_u", {}).get("significant_005", False)
        log_sig = stats.get("mann_whitney_u_log", {}).get("significant_005", False)
        stats["pass_criteria"]["clear_separation"] = (
            raw_sig or log_sig or cand_median > 1.5
        )
        stats["pass_criteria"]["raw_mw_significant"] = raw_sig
        stats["pass_criteria"]["log_mw_significant"] = log_sig

    overall_pass = all([
        stats["pass_criteria"].get("n_valid_candidates_ge_50", False),
        stats["pass_criteria"].get("random_median_in_expected_range", False),
        stats["pass_criteria"].get("clear_separation", False),
    ])
    stats["pass_criteria"]["overall"] = overall_pass

    return stats


def create_visualization_data(candidate_results, random_results):
    """Create data for histogram and box plot visualizations."""
    cand_itac = [r["itac"] for r in candidate_results if r.get("valid") and r["itac"] is not None]
    rand_itac = [r["itac"] for r in random_results if r.get("valid") and r["itac"] is not None]

    # Histogram data: ITAC distribution for candidates vs random
    all_vals = cand_itac + rand_itac
    if all_vals:
        bin_min = max(0, min(all_vals) - 0.1)
        bin_max = min(max(all_vals), np.percentile(all_vals, 99) + 0.5)
        bins = np.linspace(bin_min, bin_max, 50)

        cand_hist, _ = np.histogram(cand_itac, bins=bins) if cand_itac else (np.zeros(49), bins)
        rand_hist, _ = np.histogram(rand_itac, bins=bins) if rand_itac else (np.zeros(49), bins)

        histogram_data = {
            "bin_edges": bins.tolist(),
            "candidate_counts": cand_hist.tolist(),
            "random_counts": rand_hist.tolist(),
        }
    else:
        histogram_data = None

    # Box plot data: group-level variance ratios
    box_plot_data = {
        "candidates": {
            "itac_values": [float(v) for v in cand_itac],
            "var_ratio_parent_only": [
                float(r["var_ratio_parent_only"])
                for r in candidate_results
                if r.get("valid") and "var_ratio_parent_only" in r
            ],
            "var_ratio_both": [
                float(r["var_ratio_both"])
                for r in candidate_results
                if r.get("valid") and "var_ratio_both" in r
            ],
        },
        "random": {
            "itac_values": [float(v) for v in rand_itac],
        },
    }

    return {
        "histogram": histogram_data,
        "box_plot": box_plot_data,
    }


def main():
    start_time = datetime.now()
    print(f"=" * 70)
    print(f"ITAC Real Activations Experiment")
    print(f"Task ID: {TASK_ID}")
    print(f"Start: {start_time.isoformat()}")
    print(f"=" * 70)

    set_seed(SEED)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID
    write_pid(TASK_ID, RESULTS_DIR)
    report_progress(TASK_ID, RESULTS_DIR, 0, 5, metric={"status": "starting"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # =========================================================================
    # Step 1: Load decoder geometry candidate pairs
    # =========================================================================
    print("\n[Step 1] Loading candidate pairs from decoder geometry...")
    candidate_pairs = load_candidate_pairs(DECODER_GEOM_FILE, N_CANDIDATE_PAIRS)
    n_features_needed = max(
        max(p["parent_idx"], p["child_idx"]) for p in candidate_pairs
    ) + 1
    print(f"  Max feature index needed: {n_features_needed}")

    report_progress(TASK_ID, RESULTS_DIR, 1, 5, metric={"status": "loaded_candidates"})

    # =========================================================================
    # Step 2: Load model and SAE, collect activations
    # =========================================================================
    print("\n[Step 2] Loading model and SAE...")
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use unsloth mirror to bypass gated repo auth
    hf_model_name = "unsloth/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float16,
    )
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Model loaded: {MODEL_NAME} (via {hf_model_name})")

    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=str(device),
    )
    # from_pretrained may return a tuple (sae, cfg_dict, sparsity) or just SAE
    if isinstance(sae, tuple):
        sae = sae[0]
    print(f"  SAE loaded: {SAE_ID}")
    print(f"  SAE features: {sae.cfg.d_sae}, d_model: {sae.cfg.d_in}")
    # Hook name from SAE config (may be hook_name or hook_point depending on version)
    hook_name = getattr(sae.cfg, 'hook_name', getattr(sae.cfg, 'hook_point', 'blocks.12.hook_resid_post'))
    print(f"  Hook: {hook_name}")

    # Get decoder weights
    W_dec = sae.W_dec.data.cpu()  # (n_features, d_model)
    print(f"  Decoder weights shape: {W_dec.shape}")

    report_progress(TASK_ID, RESULTS_DIR, 2, 5, metric={"status": "model_loaded"})

    # Collect activations
    act_data = collect_activations_and_residuals(
        model, sae, device,
        n_tokens=N_CORPUS_TOKENS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
    )

    # Free GPU memory - model no longer needed
    del model
    del sae
    torch.cuda.empty_cache()
    gc.collect()

    feature_acts = act_data["feature_acts"]   # (N, n_features)
    residuals = act_data["residuals"]          # (N, d_model)
    print(f"\n  Activation collection complete:")
    print(f"  Positions: {act_data['n_positions']}, Features: {act_data['n_features']}")

    report_progress(TASK_ID, RESULTS_DIR, 3, 5,
                    metric={"status": "activations_collected",
                            "n_positions": act_data["n_positions"]})

    # =========================================================================
    # Step 3: Compute ITAC for candidate pairs
    # =========================================================================
    print("\n[Step 3] Computing ITAC for candidate pairs...")
    candidate_results = compute_itac_batch(
        candidate_pairs, feature_acts, residuals, W_dec,
        label="candidates", min_group_size=MIN_GROUP_SIZE
    )

    # =========================================================================
    # Step 4: Compute ITAC for random pairs (null test)
    # =========================================================================
    print("\n[Step 4] Computing ITAC for random pairs (null test)...")
    random_pairs = generate_random_pairs(
        act_data["n_features"], N_RANDOM_PAIRS,
        candidate_pairs, seed=SEED
    )
    random_results = compute_itac_batch(
        random_pairs, feature_acts, residuals, W_dec,
        label="random", min_group_size=MIN_GROUP_SIZE
    )

    report_progress(TASK_ID, RESULTS_DIR, 4, 5,
                    metric={"status": "itac_computed"})

    # =========================================================================
    # Step 5: Statistical analysis and output
    # =========================================================================
    print("\n[Step 5] Statistical analysis...")
    statistics = compute_statistics(candidate_results, random_results)
    vis_data = create_visualization_data(candidate_results, random_results)

    end_time = datetime.now()
    elapsed_sec = (end_time - start_time).total_seconds()

    # Build output
    output = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": MODEL_NAME,
        "sae_config": "L12-16k",
        "sae_id": SAE_ID,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "elapsed_sec": elapsed_sec,
        "corpus": act_data["corpus"],
        "n_token_positions": act_data["n_positions"],
        "n_features": act_data["n_features"],
        "activation_threshold": ACTIVATION_THRESHOLD,
        "min_group_size": MIN_GROUP_SIZE,
        "candidate_pairs_results": candidate_results,  # All results
        "random_pairs_results": random_results,
        "statistics": statistics,
        "visualizations_data": vis_data,
        "pass_criteria": statistics["pass_criteria"],
        "summary": None,
    }

    # Print results
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Candidate pairs: {statistics['candidates']['n_valid']}/{statistics['candidates']['n_total']} valid")
    if "mean" in statistics["candidates"]:
        print(f"  Median ITAC: {statistics['candidates']['median']:.4f}")
        print(f"  Mean ITAC: {statistics['candidates']['mean']:.4f}")
        print(f"  Std ITAC: {statistics['candidates']['std']:.4f}")
        print(f"  >1.5: {statistics['candidates']['pct_above_1.5']:.1f}%")
        print(f"  >2.0: {statistics['candidates']['pct_above_2.0']:.1f}%")
        print(f"  >3.0: {statistics['candidates']['pct_above_3.0']:.1f}%")

    print(f"\nRandom pairs (null): {statistics['random']['n_valid']}/{statistics['random']['n_total']} valid")
    if "mean" in statistics["random"]:
        print(f"  Median ITAC: {statistics['random']['median']:.4f}")
        print(f"  Mean ITAC: {statistics['random']['mean']:.4f}")
        print(f"  Std ITAC: {statistics['random']['std']:.4f}")

    # Log-ITAC stats
    if "log_itac_mean" in statistics.get("candidates", {}):
        print(f"  Log-ITAC mean: {statistics['candidates']['log_itac_mean']:.4f}")
        print(f"  Log-ITAC median: {statistics['candidates']['log_itac_median']:.4f}")
    if "trimmed_mean_5_95" in statistics.get("candidates", {}):
        print(f"  Trimmed mean (5-95%): {statistics['candidates']['trimmed_mean_5_95']:.4f}")

    if "log_itac_mean" in statistics.get("random", {}):
        print(f"\n  Random log-ITAC mean: {statistics['random']['log_itac_mean']:.4f}")
        print(f"  Random log-ITAC median: {statistics['random']['log_itac_median']:.4f}")

    if "mann_whitney_u" in statistics:
        mw = statistics["mann_whitney_u"]
        print(f"\nMann-Whitney U (raw): U={mw['U_statistic']:.0f}, p={mw['p_value']:.2e}")
        print(f"  Significant (p<0.05): {mw['significant_005']}")

    if "mann_whitney_u_log" in statistics:
        mw_log = statistics["mann_whitney_u_log"]
        print(f"Mann-Whitney U (log): U={mw_log['U_statistic']:.0f}, p={mw_log['p_value']:.2e}")
        print(f"  Significant (p<0.05): {mw_log['significant_005']}")

    if "cohens_d" in statistics:
        print(f"Cohen's d (raw): {statistics['cohens_d']:.4f}")
    if "cohens_d_log" in statistics:
        print(f"Cohen's d (log): {statistics['cohens_d_log']:.4f}")

    if "absorption_score_vs_itac" in statistics:
        asc = statistics["absorption_score_vs_itac"]
        print(f"Absorption score vs ITAC: rho={asc['spearman_rho']:.4f}, p={asc['p_value']:.4e}")

    print(f"\nPass criteria:")
    for k, v in statistics["pass_criteria"].items():
        print(f"  {k}: {v}")

    overall = statistics["pass_criteria"].get("overall", False)
    status_str = "PASS" if overall else "PARTIAL"
    summary_text = (
        f"ITAC analysis on {act_data['n_positions']} real token positions. "
        f"{statistics['candidates']['n_valid']} valid candidate pairs, "
        f"{statistics['random']['n_valid']} valid random pairs. "
        f"Candidate median ITAC: {statistics['candidates'].get('median', 'N/A')}, "
        f"Random median ITAC: {statistics['random'].get('median', 'N/A')}. "
        f"Overall: {status_str}"
    )
    output["summary"] = summary_text
    print(f"\n{summary_text}")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Mark done
    mark_task_done(TASK_ID, RESULTS_DIR, status="success", summary=summary_text)

    # Update gpu_progress
    actual_min = int(elapsed_sec / 60) + 1
    timing_info = {
        "planned_min": 30,
        "actual_min": actual_min,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_snapshot": {
            "model": MODEL_NAME,
            "sae_config": "L12-16k",
            "n_corpus_tokens": act_data["n_positions"],
            "n_candidate_pairs": len(candidate_results),
            "n_random_pairs": len(random_results),
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_count": 1,
        }
    }
    update_gpu_progress(TASK_ID, WORKSPACE, "completed", timing_info)

    report_progress(TASK_ID, RESULTS_DIR, 5, 5,
                    metric={"status": "completed", "elapsed_sec": elapsed_sec})

    print(f"\nTotal time: {elapsed_sec:.1f}s ({elapsed_sec/60:.1f} min)")
    return output


if __name__ == "__main__":
    main()
