#!/usr/bin/env python3
"""
decoder_geometry.py — Multi-Resolution Decoder Geometry Analysis (Pilot)

Task: Compute pairwise cosine similarity of Gemma Scope L12 16k decoder directions,
      identify candidate absorption pairs via conditional cosine similarity,
      apply firing rate asymmetry filter, and build hierarchical clustering.

Stage 2a of the unsupervised absorption detection pipeline.

GPU: Single GPU (CUDA_VISIBLE_DEVICES set externally)
Expected runtime: ~15-20 minutes (pilot: 100k tokens for firing rate)
"""

import json
import os
import sys
import time
import gc
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
FULL_RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "decoder_geometry"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID file (critical for system recovery) ──────────────────────────────────
pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(step, total_steps, description, extra=None):
    """Write progress file for system monitor."""
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "description": description,
        "metric": extra or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress.write_text(json.dumps(data, indent=2))


def mark_done(status="success", summary="", results=None):
    """Write DONE marker file."""
    if pid_file.exists():
        pid_file.unlink()
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "results": results or {},
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


def main():
    start_time = time.time()
    total_steps = 8
    np.random.seed(42)

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": 42,
        "model": "gemma-2-2b",
        "sae_config": "L12-16k",
        "sae_id": "layer_12/width_16k/average_l0_82",
        "timestamp_start": datetime.now().isoformat(),
        "steps": {},
        "candidate_pairs": [],
        "statistics": {},
        "visualizations_data": {},
        "pass_criteria": {},
        "warnings": [],
        "errors": [],
    }

    print("=" * 70)
    print("SAE Absorption Study — Decoder Geometry Analysis (Pilot)")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Step 1: Load SAE decoder weights ─────────────────────────────────
    report_progress(1, total_steps, "Loading SAE decoder weights")
    print("\n[Step 1/8] Loading SAE decoder weights...")
    t0 = time.time()

    import torch
    from sae_lens import SAE

    device = torch.device("cuda:0")

    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device=str(device),
    )
    n_features = sae.cfg.d_sae
    d_model = sae.cfg.d_in

    # Extract decoder weights: shape (n_features, d_model)
    W_dec = sae.W_dec.detach().clone()  # (16384, 2304)
    print(f"  SAE loaded: {n_features} features, d_model={d_model}")
    print(f"  W_dec shape: {W_dec.shape}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    results["steps"]["load_sae"] = {
        "n_features": n_features,
        "d_model": d_model,
        "time_sec": round(time.time() - t0, 1),
    }

    # ── Step 2: Compute pairwise cosine similarity matrix ────────────────
    report_progress(2, total_steps, "Computing pairwise cosine similarity matrix")
    print("\n[Step 2/8] Computing pairwise cosine similarity matrix...")
    print(f"  Matrix size: {n_features} x {n_features} = {n_features**2:,} pairs")
    t0 = time.time()

    # Normalize decoder directions
    W_dec_normed = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)

    # Compute full cosine similarity matrix in batches to avoid OOM
    # 16k x 16k float32 = ~1GB, should fit easily
    cos_sim_matrix = torch.mm(W_dec_normed, W_dec_normed.T)  # (16384, 16384)

    # Zero out diagonal (self-similarity = 1.0, not interesting)
    cos_sim_matrix.fill_diagonal_(0.0)

    cosine_time = time.time() - t0
    print(f"  Cosine matrix computed in {cosine_time:.1f}s")

    # Statistics on the cosine similarity distribution
    # Use sampled approach to avoid memory issues with 134M-element tensor
    # Compute basic stats directly on GPU matrix (mean/min/max are fine)
    triu_mask = torch.ones(n_features, n_features, dtype=torch.bool, device=device).triu(diagonal=1)
    n_total_pairs = int(triu_mask.sum().item())

    # For mean: sum all upper triangle values / count
    cos_sum = (cos_sim_matrix * triu_mask.float()).sum().item()
    cos_mean = cos_sum / n_total_pairs

    # For min/max: get from masked values
    cos_min = cos_sim_matrix[triu_mask].min().item()
    cos_max = cos_sim_matrix[triu_mask].max().item()

    # For quantiles: sample a large subset
    print("  Computing statistics via sampling...")
    sample_n = min(10_000_000, n_total_pairs)
    # Sample random indices from upper triangle
    triu_idx = triu_mask.nonzero(as_tuple=False)  # (N, 2) - this is large but manageable
    perm = torch.randperm(len(triu_idx), device=device)[:sample_n]
    sample_rows = triu_idx[perm, 0]
    sample_cols = triu_idx[perm, 1]
    sample_vals = cos_sim_matrix[sample_rows, sample_cols].cpu().numpy()

    del triu_mask, triu_idx, perm, sample_rows, sample_cols
    torch.cuda.empty_cache()

    cos_stats = {
        "mean": float(cos_mean),
        "std": float(np.std(sample_vals)),
        "median": float(np.median(sample_vals)),
        "min": float(cos_min),
        "max": float(cos_max),
        "q25": float(np.quantile(sample_vals, 0.25)),
        "q75": float(np.quantile(sample_vals, 0.75)),
        "q95": float(np.quantile(sample_vals, 0.95)),
        "q99": float(np.quantile(sample_vals, 0.99)),
        "n_pairs_total": n_total_pairs,
        "n_sampled_for_quantiles": sample_n,
    }

    # Count pairs above various thresholds (use the cos_sim_matrix directly)
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for tau in thresholds:
        n_above = int((cos_sim_matrix.triu(diagonal=1) > tau).sum().item())
        cos_stats[f"n_above_{tau}"] = n_above
        cos_stats[f"pct_above_{tau}"] = round(100.0 * n_above / n_total_pairs, 4)
        print(f"  Pairs with cosine > {tau}: {n_above:,} ({cos_stats[f'pct_above_{tau}']}%)")

    results["steps"]["cosine_matrix"] = {
        "time_sec": round(cosine_time, 1),
        "statistics": cos_stats,
    }
    results["statistics"]["global_cosine"] = cos_stats

    # Save cosine distribution histogram data (binned, for visualization)
    counts, bin_edges = np.histogram(sample_vals, bins=200, range=(-0.3, 1.0))
    results["visualizations_data"]["cosine_histogram"] = {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "n_sampled": int(sample_n),
        "description": "Global pairwise cosine similarity distribution (sampled)",
    }

    del sample_vals
    gc.collect()

    # ── Step 3: Identify candidate pairs (global cosine > 0.1) ───────────
    report_progress(3, total_steps, "Identifying high-cosine candidate pairs")
    print("\n[Step 3/8] Identifying candidate pairs with global cosine > 0.1...")
    t0 = time.time()

    GLOBAL_COS_THRESHOLD = 0.1

    # Find pairs above threshold (upper triangle only)
    high_cos_mask = cos_sim_matrix > GLOBAL_COS_THRESHOLD
    # Only upper triangle
    high_cos_mask = high_cos_mask.triu(diagonal=1)
    pair_indices = high_cos_mask.nonzero(as_tuple=False)  # (N_pairs, 2)

    n_candidate_pairs = len(pair_indices)
    print(f"  Candidate pairs with global cosine > {GLOBAL_COS_THRESHOLD}: {n_candidate_pairs:,}")

    if n_candidate_pairs == 0:
        print("  WARNING: No pairs found above threshold. Lowering to 0.05...")
        GLOBAL_COS_THRESHOLD = 0.05
        high_cos_mask = cos_sim_matrix > GLOBAL_COS_THRESHOLD
        high_cos_mask = high_cos_mask.triu(diagonal=1)
        pair_indices = high_cos_mask.nonzero(as_tuple=False)
        n_candidate_pairs = len(pair_indices)
        print(f"  Candidate pairs with global cosine > {GLOBAL_COS_THRESHOLD}: {n_candidate_pairs:,}")
        results["warnings"].append(f"Lowered global cosine threshold to {GLOBAL_COS_THRESHOLD}")

    results["steps"]["candidate_identification"] = {
        "threshold": GLOBAL_COS_THRESHOLD,
        "n_candidate_pairs": n_candidate_pairs,
        "time_sec": round(time.time() - t0, 1),
    }

    del high_cos_mask
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 4: Compute conditional cosine similarity ────────────────────
    report_progress(4, total_steps, "Computing conditional cosine similarity")
    print("\n[Step 4/8] Computing conditional cosine similarity for candidate pairs...")
    print(f"  Processing {n_candidate_pairs:,} pairs...")
    t0 = time.time()

    # For each pair (i, j) with high global cosine:
    # 1. Find shared subspace: top-K dims where both decoders have magnitude > threshold
    # 2. Compute cosine in that subspace
    # This detects pairs that are similar in specific subspaces (potential parent-child)

    # We process in batches for efficiency
    MAGNITUDE_THRESHOLD = 0.01  # decoder component magnitude threshold
    TOP_K_SHARED = 50  # top shared dimensions to consider

    # Limit to manageable number of pairs for pilot
    MAX_PAIRS_PILOT = 50000
    if n_candidate_pairs > MAX_PAIRS_PILOT:
        print(f"  Subsampling to {MAX_PAIRS_PILOT} pairs for pilot...")
        # Take top pairs by cosine similarity
        pair_cosines = cos_sim_matrix[pair_indices[:, 0], pair_indices[:, 1]]
        _, top_idx = pair_cosines.topk(min(MAX_PAIRS_PILOT, len(pair_cosines)))
        pair_indices = pair_indices[top_idx]
        n_candidate_pairs = len(pair_indices)
        del pair_cosines, top_idx

    # Compute conditional cosine for all candidate pairs
    batch_size = 5000
    conditional_cosines = []
    global_cosines_for_pairs = []
    shared_dims_counts = []

    for batch_start in range(0, n_candidate_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_candidate_pairs)
        batch_pairs = pair_indices[batch_start:batch_end]
        i_indices = batch_pairs[:, 0]
        j_indices = batch_pairs[:, 1]

        # Get decoder directions for these pairs
        w_i = W_dec[i_indices]  # (batch, d_model)
        w_j = W_dec[j_indices]  # (batch, d_model)

        # Global cosine for these pairs
        gc_vals = cos_sim_matrix[i_indices, j_indices]
        global_cosines_for_pairs.append(gc_vals.cpu())

        # Find shared subspace: dimensions where both have magnitude > threshold
        # Use absolute value of decoder components
        abs_i = w_i.abs()
        abs_j = w_j.abs()
        both_significant = (abs_i > MAGNITUDE_THRESHOLD) & (abs_j > MAGNITUDE_THRESHOLD)

        # For each pair, compute conditional cosine in shared subspace
        batch_cond_cos = []
        batch_shared_dims = []

        for k in range(len(batch_pairs)):
            shared_mask = both_significant[k]
            n_shared = int(shared_mask.sum())
            batch_shared_dims.append(n_shared)

            if n_shared < 5:
                # Too few shared dimensions; fall back to top-K by product of magnitudes
                product = abs_i[k] * abs_j[k]
                _, topk_idx = product.topk(min(TOP_K_SHARED, d_model))
                shared_mask = torch.zeros(d_model, dtype=torch.bool, device=device)
                shared_mask[topk_idx] = True
                n_shared = int(shared_mask.sum())

            # Compute cosine in shared subspace
            vi = w_i[k][shared_mask]
            vj = w_j[k][shared_mask]
            cond_cos = float(torch.dot(vi, vj) / (vi.norm() * vj.norm() + 1e-8))
            batch_cond_cos.append(cond_cos)

        conditional_cosines.extend(batch_cond_cos)
        shared_dims_counts.extend(batch_shared_dims)

        if (batch_start // batch_size) % 5 == 0:
            print(f"    Processed {batch_end}/{n_candidate_pairs} pairs...")

    conditional_cosines = np.array(conditional_cosines)
    global_cosines_for_pairs = torch.cat(global_cosines_for_pairs).numpy()
    shared_dims_counts = np.array(shared_dims_counts)

    cond_time = time.time() - t0
    print(f"  Conditional cosine computed in {cond_time:.1f}s")

    # Statistics
    cond_cos_stats = {
        "mean": float(np.mean(conditional_cosines)),
        "std": float(np.std(conditional_cosines)),
        "median": float(np.median(conditional_cosines)),
        "min": float(np.min(conditional_cosines)),
        "max": float(np.max(conditional_cosines)),
        "q25": float(np.quantile(conditional_cosines, 0.25)),
        "q75": float(np.quantile(conditional_cosines, 0.75)),
        "q95": float(np.quantile(conditional_cosines, 0.95)),
        "n_pairs": int(len(conditional_cosines)),
        "mean_shared_dims": float(np.mean(shared_dims_counts)),
    }
    print(f"  Conditional cosine stats: mean={cond_cos_stats['mean']:.4f}, "
          f"std={cond_cos_stats['std']:.4f}, "
          f"median={cond_cos_stats['median']:.4f}")

    results["steps"]["conditional_cosine"] = {
        "time_sec": round(cond_time, 1),
        "magnitude_threshold": MAGNITUDE_THRESHOLD,
        "top_k_fallback": TOP_K_SHARED,
        "statistics": cond_cos_stats,
    }
    results["statistics"]["conditional_cosine"] = cond_cos_stats

    # Save scatter data for visualization: global vs conditional cosine
    # Subsample for reasonable file size
    n_viz = min(10000, len(conditional_cosines))
    viz_idx = np.random.choice(len(conditional_cosines), n_viz, replace=False)
    results["visualizations_data"]["global_vs_conditional"] = {
        "global_cosine": global_cosines_for_pairs[viz_idx].tolist(),
        "conditional_cosine": conditional_cosines[viz_idx].tolist(),
        "shared_dims": shared_dims_counts[viz_idx].tolist(),
        "n_points": n_viz,
        "description": "Global vs conditional cosine similarity for candidate pairs",
    }

    # Conditional cosine histogram
    cc_counts, cc_edges = np.histogram(conditional_cosines, bins=100, range=(-0.5, 1.0))
    results["visualizations_data"]["conditional_cosine_histogram"] = {
        "counts": cc_counts.tolist(),
        "bin_edges": cc_edges.tolist(),
        "description": "Conditional cosine similarity distribution",
    }

    # ── Step 5: Load model and compute firing rates ──────────────────────
    report_progress(5, total_steps, "Computing firing rates on 100k tokens")
    print("\n[Step 5/8] Computing firing rates on 100k tokens...")
    t0 = time.time()

    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer and model
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
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Generate token corpus for firing rate computation
    # Use diverse text prompts for a representative sample
    corpus_texts = [
        "The capital of France is Paris, and it is known for the Eiffel Tower.",
        "Machine learning algorithms can be supervised, unsupervised, or reinforcement-based.",
        "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Quantum mechanics describes the behavior of particles at the atomic scale.",
        "The stock market crashed in 1929, leading to the Great Depression.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Mozart composed his first symphony at the age of eight.",
        "The Amazon rainforest produces about 20 percent of the world's oxygen.",
        "DNA contains the genetic instructions for all living organisms.",
        "The Industrial Revolution began in Britain in the late 18th century.",
        "Einstein published his theory of general relativity in 1915.",
        "The Nile is often considered the longest river in the world.",
        "Antibiotics were first discovered by Alexander Fleming in 1928.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Democracy originated in ancient Athens around the 5th century BCE.",
        "The speed of light is approximately 300,000 kilometers per second.",
        "The human brain contains roughly 86 billion neurons.",
        "Global warming is primarily caused by greenhouse gas emissions.",
        "The first computer program was written by Ada Lovelace in 1843.",
        "Tokyo is the most populous metropolitan area in the world.",
        "Water molecules consist of two hydrogen atoms and one oxygen atom.",
        "The Renaissance period saw a revival of art and learning in Europe.",
        "Coral reefs are among the most diverse ecosystems on the planet.",
        "The periodic table organizes chemical elements by atomic number.",
        "Beethoven continued to compose music even after becoming deaf.",
        "The Grand Canyon was formed by the Colorado River over millions of years.",
        "Artificial intelligence aims to create systems that can perform human-like tasks.",
        "The Sahara Desert covers most of North Africa.",
        "Vaccines work by stimulating the immune system to recognize pathogens.",
    ]

    # Build a larger corpus by repeating and combining texts
    # Target: ~100k tokens
    n_target_tokens = 100_000
    print(f"  Building corpus targeting ~{n_target_tokens:,} tokens...")

    # Tokenize all texts and see how many tokens we have
    all_tokens = []
    for text in corpus_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    # Repeat until we have enough tokens
    base_tokens = list(all_tokens)
    while len(all_tokens) < n_target_tokens:
        all_tokens.extend(base_tokens)
        # Also shuffle to avoid exact repetition patterns
        np.random.shuffle(base_tokens)

    all_tokens = all_tokens[:n_target_tokens]
    print(f"  Corpus: {len(all_tokens):,} tokens")

    # Process tokens through model + SAE in batches
    # Collect per-feature activation counts (binary: active or not)
    seq_len = 128
    activation_counts = torch.zeros(n_features, dtype=torch.long, device=device)
    total_token_positions = 0

    n_batches = (len(all_tokens) + seq_len - 1) // seq_len

    # Process in chunks of sequences
    chunk_size = 32  # sequences per forward pass
    sae.eval()

    with torch.no_grad():
        for batch_idx in range(0, n_batches, chunk_size):
            chunk_end = min(batch_idx + chunk_size, n_batches)
            # Build batch of sequences
            batch_sequences = []
            for seq_idx in range(batch_idx, chunk_end):
                start = seq_idx * seq_len
                end = min(start + seq_len, len(all_tokens))
                if end > start:
                    batch_sequences.append(all_tokens[start:end])

            if not batch_sequences:
                continue

            # Pad sequences to same length
            max_len = max(len(s) for s in batch_sequences)
            padded = [s + [tokenizer.pad_token_id or 0] * (max_len - len(s))
                      for s in batch_sequences]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)

            # Forward pass to get residual stream at layer 12
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=["blocks.12.hook_resid_post"],
                return_type=None,
            )
            residuals = cache["blocks.12.hook_resid_post"]  # (batch, seq, d_model)

            # Flatten to (n_tokens, d_model)
            flat_residuals = residuals.reshape(-1, d_model).float()

            # Run through SAE encoder to get activations
            # SAE.encode returns the latent activations
            sae_acts = sae.encode(flat_residuals)  # (n_tokens, n_features)

            # Count active features (activation > 0)
            active_mask = (sae_acts > 0).long()
            activation_counts += active_mask.sum(dim=0)
            total_token_positions += flat_residuals.shape[0]

            del cache, residuals, flat_residuals, sae_acts, active_mask, input_ids
            torch.cuda.empty_cache()

            if batch_idx % (chunk_size * 5) == 0:
                print(f"    Processed {min((batch_idx + chunk_size) * seq_len, len(all_tokens)):,}/{len(all_tokens):,} tokens...")

    # Compute firing rates
    firing_rates = (activation_counts.float() / total_token_positions).cpu().numpy()

    firing_rate_time = time.time() - t0
    print(f"  Firing rates computed in {firing_rate_time:.1f}s")
    print(f"  Total token positions: {total_token_positions:,}")

    fr_stats = {
        "total_token_positions": int(total_token_positions),
        "mean_firing_rate": float(np.mean(firing_rates)),
        "median_firing_rate": float(np.median(firing_rates)),
        "std_firing_rate": float(np.std(firing_rates)),
        "n_dead_features": int((firing_rates == 0).sum()),
        "n_very_rare": int((firing_rates < 0.001).sum()),
        "n_common": int((firing_rates > 0.1).sum()),
        "pct_dead": round(100.0 * (firing_rates == 0).sum() / n_features, 2),
        "time_sec": round(firing_rate_time, 1),
    }
    print(f"  Dead features: {fr_stats['n_dead_features']} ({fr_stats['pct_dead']}%)")
    print(f"  Mean firing rate: {fr_stats['mean_firing_rate']:.4f}")

    results["steps"]["firing_rates"] = fr_stats
    results["statistics"]["firing_rates"] = fr_stats

    # Firing rate histogram data
    fr_nonzero = firing_rates[firing_rates > 0]
    if len(fr_nonzero) > 0:
        fr_log = np.log10(fr_nonzero + 1e-10)
        fr_counts, fr_edges = np.histogram(fr_log, bins=100)
        results["visualizations_data"]["firing_rate_histogram"] = {
            "counts": fr_counts.tolist(),
            "bin_edges": fr_edges.tolist(),
            "description": "Log10 firing rate distribution (non-zero features only)",
        }

    # Free model memory - we don't need it anymore
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 6: Apply firing rate asymmetry filter ───────────────────────
    report_progress(6, total_steps, "Applying firing rate asymmetry filter")
    print("\n[Step 6/8] Applying firing rate asymmetry filter...")
    t0 = time.time()

    # For each candidate pair (i, j):
    # - The "child" is the one with LOWER firing rate (more specific)
    # - The "parent" is the one with HIGHER firing rate (more general)
    # - Require: freq(child) < 0.5 * freq(parent) (child fires more specifically)
    # - Also: both features must be non-dead

    FIRING_RATE_ASYMMETRY = 0.5  # child must fire less than 50% of parent rate

    # Build comprehensive pair data
    pair_data = []
    n_passed_fr_filter = 0
    n_both_dead = 0
    n_one_dead = 0
    n_symmetric = 0

    for idx in range(len(pair_indices)):
        i = int(pair_indices[idx, 0])
        j = int(pair_indices[idx, 1])

        fr_i = float(firing_rates[i])
        fr_j = float(firing_rates[j])
        gc_val = float(global_cosines_for_pairs[idx])
        cc_val = float(conditional_cosines[idx])
        n_shared = int(shared_dims_counts[idx])

        # Skip dead features
        if fr_i == 0 and fr_j == 0:
            n_both_dead += 1
            continue
        if fr_i == 0 or fr_j == 0:
            n_one_dead += 1
            continue

        # Determine parent (higher firing rate) and child (lower firing rate)
        if fr_i >= fr_j:
            parent_idx, child_idx = i, j
            parent_fr, child_fr = fr_i, fr_j
        else:
            parent_idx, child_idx = j, i
            parent_fr, child_fr = fr_j, fr_i

        # Check asymmetry filter
        passes_fr_filter = child_fr < FIRING_RATE_ASYMMETRY * parent_fr

        if passes_fr_filter:
            n_passed_fr_filter += 1
        else:
            n_symmetric += 1

        # Compute composite absorption score:
        # Higher conditional cosine + more asymmetric firing rate = more likely absorption
        fr_asymmetry_ratio = 1.0 - (child_fr / (parent_fr + 1e-10))
        absorption_score = cc_val * fr_asymmetry_ratio  # composite score

        pair_entry = {
            "parent_idx": parent_idx,
            "child_idx": child_idx,
            "global_cosine": round(gc_val, 5),
            "conditional_cosine": round(cc_val, 5),
            "parent_firing_rate": round(parent_fr, 6),
            "child_firing_rate": round(child_fr, 6),
            "firing_rate_ratio": round(child_fr / (parent_fr + 1e-10), 4),
            "firing_rate_asymmetry": round(fr_asymmetry_ratio, 4),
            "n_shared_dims": n_shared,
            "passes_fr_filter": passes_fr_filter,
            "absorption_score": round(absorption_score, 5),
        }
        pair_data.append(pair_entry)

    # Sort by absorption score (descending)
    pair_data.sort(key=lambda x: x["absorption_score"], reverse=True)

    fr_filter_time = time.time() - t0
    print(f"  Pairs after removing dead features: {len(pair_data):,}")
    print(f"  Pairs passing FR asymmetry filter: {n_passed_fr_filter:,}")
    print(f"  Both dead: {n_both_dead}, One dead: {n_one_dead}, Symmetric: {n_symmetric}")

    results["steps"]["firing_rate_filter"] = {
        "asymmetry_threshold": FIRING_RATE_ASYMMETRY,
        "n_total_after_dead_removal": len(pair_data),
        "n_passed_filter": n_passed_fr_filter,
        "n_both_dead": n_both_dead,
        "n_one_dead": n_one_dead,
        "n_symmetric": n_symmetric,
        "time_sec": round(fr_filter_time, 1),
    }

    # ── Step 7: Hierarchical agglomerative clustering (lightweight) ──────
    report_progress(7, total_steps, "Building hierarchical clustering dendrogram")
    print("\n[Step 7/8] Building hierarchical clustering dendrogram...")
    t0 = time.time()

    # For pilot, we cluster the top candidate features (not all 16k)
    # Collect unique feature indices from top candidate pairs
    TOP_FEATURES_FOR_CLUSTERING = 2000  # Cluster top features appearing in absorption candidates

    # Get features that appear in filtered candidates
    filtered_pairs = [p for p in pair_data if p["passes_fr_filter"]]
    feature_scores = {}
    for p in filtered_pairs:
        pi, ci = p["parent_idx"], p["child_idx"]
        feature_scores[pi] = max(feature_scores.get(pi, 0), p["absorption_score"])
        feature_scores[ci] = max(feature_scores.get(ci, 0), p["absorption_score"])

    # Take top features by max absorption score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    top_feature_indices = [f[0] for f in sorted_features[:TOP_FEATURES_FOR_CLUSTERING]]
    n_cluster_features = len(top_feature_indices)
    print(f"  Clustering {n_cluster_features} features involved in top absorption candidates")

    clustering_results = {}
    if n_cluster_features >= 10:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Build distance matrix for selected features
        feat_idx_tensor = torch.tensor(top_feature_indices, dtype=torch.long, device=device)
        W_subset = W_dec_normed[feat_idx_tensor]  # (n_cluster, d_model)
        cos_sub = torch.mm(W_subset, W_subset.T).cpu().numpy()
        dist_sub = 1.0 - cos_sub
        np.fill_diagonal(dist_sub, 0)
        # Ensure symmetry and non-negativity
        dist_sub = (dist_sub + dist_sub.T) / 2
        dist_sub = np.clip(dist_sub, 0, 2)

        # Convert to condensed form
        dist_condensed = squareform(dist_sub)

        # Hierarchical clustering with average linkage
        Z = linkage(dist_condensed, method='average')

        # Analyze at multiple resolution thresholds
        resolution_analysis = {}
        for tau in [0.3, 0.5, 0.7, 0.9]:
            try:
                clusters = fcluster(Z, t=tau, criterion='distance')
                n_clusters = len(set(clusters))
                # Find merge events: clusters that contain both parent and child features
                cluster_map = {}
                for i, c in enumerate(clusters):
                    if c not in cluster_map:
                        cluster_map[c] = []
                    cluster_map[c].append(top_feature_indices[i])

                # Count clusters with >1 member (potential absorption groups)
                multi_member = sum(1 for members in cluster_map.values() if len(members) > 1)
                max_cluster_size = max(len(members) for members in cluster_map.values())

                resolution_analysis[str(tau)] = {
                    "n_clusters": n_clusters,
                    "n_multi_member": multi_member,
                    "max_cluster_size": max_cluster_size,
                    "mean_cluster_size": round(n_cluster_features / n_clusters, 2),
                }
                print(f"    tau={tau}: {n_clusters} clusters, {multi_member} multi-member, max_size={max_cluster_size}")
            except Exception as e:
                resolution_analysis[str(tau)] = {"error": str(e)}

        clustering_results = {
            "n_features_clustered": n_cluster_features,
            "linkage_method": "average",
            "resolution_analysis": resolution_analysis,
            "time_sec": round(time.time() - t0, 1),
        }

        del W_subset, cos_sub, dist_sub, dist_condensed, Z
    else:
        clustering_results = {
            "skipped": True,
            "reason": f"Only {n_cluster_features} features available, need >= 10",
        }
        print(f"  Skipped clustering: only {n_cluster_features} features")

    results["steps"]["clustering"] = clustering_results
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 8: Compile final results ────────────────────────────────────
    report_progress(8, total_steps, "Compiling results and saving")
    print("\n[Step 8/8] Compiling final results...")

    # Top candidate pairs (save top 500 for downstream use)
    top_candidates = []
    for p in pair_data[:500]:
        if p["passes_fr_filter"]:
            top_candidates.append(p)
    # Also include some that fail FR filter for comparison
    non_fr_count = 0
    for p in pair_data[:500]:
        if not p["passes_fr_filter"] and non_fr_count < 50:
            top_candidates.append(p)
            non_fr_count += 1

    results["candidate_pairs"] = top_candidates
    results["n_total_candidates_fr_passed"] = n_passed_fr_filter
    results["n_total_candidates_all"] = len(pair_data)

    # Top 20 absorption candidate details for inspection
    top_20 = []
    for p in filtered_pairs[:20]:
        top_20.append({
            "rank": len(top_20) + 1,
            "parent_feature": p["parent_idx"],
            "child_feature": p["child_idx"],
            "global_cosine": p["global_cosine"],
            "conditional_cosine": p["conditional_cosine"],
            "parent_fr": p["parent_firing_rate"],
            "child_fr": p["child_firing_rate"],
            "fr_ratio": p["firing_rate_ratio"],
            "absorption_score": p["absorption_score"],
        })
    results["top_20_candidates"] = top_20

    # Compute absorption score distribution statistics
    if filtered_pairs:
        abs_scores = [p["absorption_score"] for p in filtered_pairs]
        results["statistics"]["absorption_score"] = {
            "mean": float(np.mean(abs_scores)),
            "std": float(np.std(abs_scores)),
            "median": float(np.median(abs_scores)),
            "min": float(np.min(abs_scores)),
            "max": float(np.max(abs_scores)),
            "q95": float(np.quantile(abs_scores, 0.95)),
            "n_scores": len(abs_scores),
        }

        # Absorption score histogram
        as_counts, as_edges = np.histogram(abs_scores, bins=50)
        results["visualizations_data"]["absorption_score_histogram"] = {
            "counts": as_counts.tolist(),
            "bin_edges": as_edges.tolist(),
            "description": "Absorption score distribution for FR-filtered candidates",
        }

    # Check bimodality of conditional cosine distribution
    # Simple test: compare conditional cosine distribution for FR-filtered vs non-filtered
    if len(filtered_pairs) > 10:
        filtered_cc = [p["conditional_cosine"] for p in pair_data if p["passes_fr_filter"]]
        non_filtered_cc = [p["conditional_cosine"] for p in pair_data if not p["passes_fr_filter"]]

        from scipy import stats
        if len(non_filtered_cc) > 10:
            ks_stat, ks_p = stats.ks_2samp(filtered_cc, non_filtered_cc)
            results["statistics"]["bimodality_test"] = {
                "test": "KS_2samp",
                "statistic": round(float(ks_stat), 4),
                "p_value": float(ks_p),
                "n_filtered": len(filtered_cc),
                "n_non_filtered": len(non_filtered_cc),
                "mean_filtered_cc": round(float(np.mean(filtered_cc)), 4),
                "mean_non_filtered_cc": round(float(np.mean(non_filtered_cc)), 4),
                "interpretation": "significant" if ks_p < 0.05 else "not_significant",
            }
            print(f"  KS test (filtered vs non-filtered cond. cosine): "
                  f"D={ks_stat:.4f}, p={ks_p:.2e}")

    # Pass criteria check
    elapsed = time.time() - start_time
    pass_criteria = {
        "cosine_matrix_computed": True,  # We got here
        "min_100_candidates": n_passed_fr_filter >= 100,
        "conditional_cosine_computed": len(conditional_cosines) > 0,
        "firing_rates_computed": total_token_positions > 0,
    }

    # Check for clear tail in conditional cosine distribution
    if len(conditional_cosines) > 0:
        q95_cc = float(np.quantile(conditional_cosines, 0.95))
        q50_cc = float(np.median(conditional_cosines))
        has_tail = (q95_cc - q50_cc) > 0.1  # clear separation between median and tail
        pass_criteria["conditional_cosine_has_tail"] = has_tail
    else:
        pass_criteria["conditional_cosine_has_tail"] = False

    pass_criteria["overall"] = all(pass_criteria.values())
    results["pass_criteria"] = pass_criteria

    # Timing
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    # Summary
    summary_lines = [
        f"Decoder geometry analysis completed in {elapsed:.1f}s",
        f"Global cosine matrix: {n_features}x{n_features}, threshold={GLOBAL_COS_THRESHOLD}",
        f"Candidate pairs (global cosine > {GLOBAL_COS_THRESHOLD}): {n_candidate_pairs:,}",
        f"After FR filter: {n_passed_fr_filter:,} pairs",
        f"Top absorption score: {top_20[0]['absorption_score']:.4f}" if top_20 else "No candidates",
        f"Pass criteria: {'PASS' if pass_criteria['overall'] else 'FAIL'}",
    ]
    results["summary"] = "\n".join(summary_lines)

    # Save results
    output_path_pilot = RESULTS_DIR / "decoder_geometry.json"
    output_path_full = FULL_RESULTS_DIR / "decoder_geometry.json"

    for path in [output_path_pilot, output_path_full]:
        path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Saved: {path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for line in summary_lines:
        print(f"  {line}")
    print(f"\n  Top 10 absorption candidates:")
    for entry in top_20[:10]:
        print(f"    #{entry['rank']}: parent={entry['parent_feature']}, "
              f"child={entry['child_feature']}, "
              f"gc={entry['global_cosine']:.3f}, cc={entry['conditional_cosine']:.3f}, "
              f"fr_ratio={entry['fr_ratio']:.3f}, score={entry['absorption_score']:.4f}")

    if pass_criteria["overall"]:
        print("\n  PILOT PASS: All criteria met")
    else:
        failed = [k for k, v in pass_criteria.items() if not v and k != "overall"]
        print(f"\n  PILOT CONCERNS: Failed criteria: {failed}")

    print("=" * 70)

    # Mark done
    mark_done(
        status="success" if pass_criteria["overall"] else "partial",
        summary=results["summary"],
        results={"pass_criteria": pass_criteria, "n_candidates": n_passed_fr_filter},
    )

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        mark_done("failed", f"Fatal error: {e}", {"traceback": tb})
        sys.exit(1)
