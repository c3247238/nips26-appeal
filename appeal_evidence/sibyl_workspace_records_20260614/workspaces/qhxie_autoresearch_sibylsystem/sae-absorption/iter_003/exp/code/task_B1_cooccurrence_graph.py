"""
task_B1_cooccurrence_graph.py
Build directed co-occurrence graph for ARS_v2.

PILOT MODE: uses 100 tokens and small subset
Full mode: 10,000 tokens from OpenWebText through GPT-2 with gpt2-small-res-jb L6 SAE.

Computes:
1. activation frequency f_j = P(latent j fires) for all j
2. conditional frequency P(j fires | i fires) for letter-involving pairs only
3. directed asymmetry A_cooccur(j) = max_{i: f_i > 3*f_j} [P(j|i) / max(P(i|j), 1e-6)]
4. Jaccard overlap O(i,j) for letter-involving pairs
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Paths
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR = Path(WORKSPACE) / "exp" / "results" / "pilots"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_B1_cooccurrence_graph"
OWT_CACHE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_002/exp/results/owt_tokens_cache.pt")

# Parse mode
MODE = "PILOT"
if "--full" in sys.argv:
    MODE = "FULL"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Letter feature IDs from iter_002 B3 (71 features, activation-proxy labeled)
LETTER_FEATURE_IDS = [368, 505, 765, 2014, 2392, 2406, 2788, 2820, 2888, 3226,
                       3898, 3943, 4091, 4176, 4703, 5019, 5200, 5768, 5783, 6083,
                       6565, 7371, 7474, 7791, 7803, 8243, 9722, 10156, 10959, 11074,
                       11138, 11270, 11414, 11821, 12341, 13159, 13531, 13916, 13936,
                       14080, 14109, 14399, 14700, 15509, 15837, 16655, 16664, 16825,
                       17402, 18363, 18598, 18890, 19123, 19158, 19264, 19350, 19521,
                       19695, 19756, 19819, 20084, 20349, 20460, 20946, 21441, 22815,
                       23195, 23321, 23903, 24262, 24466]

LETTER_FEATURE_SET = set(LETTER_FEATURE_IDS)

# Mode settings
if MODE == "PILOT":
    N_TOKENS = 100
    OUTPUT_FILE = PILOTS_DIR / "B1_cooccurrence_graph_pilot.json"
    print(f"[B1] PILOT MODE: {N_TOKENS} tokens")
else:
    N_TOKENS = 10000
    OUTPUT_FILE = RESULTS_DIR / "B1_cooccurrence_graph.json"
    print(f"[B1] FULL MODE: {N_TOKENS} tokens")

start_time = time.time()

# Write PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

print("[B1] Loading SAE and GPT-2...")
report_progress(0, 5, metric={"phase": "loading"})

try:
    from sae_lens import SAE
    import transformer_lens
    from transformer_lens import HookedTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[B1] Using device: {device}")

    # Load GPT-2 model
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    print("[B1] GPT-2 loaded")

    # Load SAE
    sae, cfg_dict, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
    )
    sae = sae.to(device)
    sae.eval()
    d_sae = sae.cfg.d_sae
    print(f"[B1] SAE loaded: d_sae={d_sae}")

    report_progress(1, 5, metric={"phase": "models_loaded"})

    # Load or generate tokens
    print("[B1] Loading tokens...")
    if OWT_CACHE.exists():
        print(f"[B1] Found cached tokens at {OWT_CACHE}")
        cached = torch.load(OWT_CACHE, weights_only=True)
        if isinstance(cached, torch.Tensor):
            all_tokens = cached
        elif isinstance(cached, dict):
            # Try common keys
            for k in ['tokens', 'input_ids', 'data']:
                if k in cached:
                    all_tokens = cached[k]
                    break
            else:
                all_tokens = list(cached.values())[0]
        print(f"[B1] Cached tokens shape: {all_tokens.shape}")
    else:
        print("[B1] No cache found, generating tokens from OpenWebText...")
        from datasets import load_dataset
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        tokenizer = model.tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        all_text = []
        for i, item in enumerate(dataset):
            all_text.append(item['text'])
            if i >= 200:
                break

        combined = " ".join(all_text)
        tokens = tokenizer(combined, return_tensors="pt", truncation=True, max_length=50000)['input_ids'][0]
        all_tokens = tokens
        print(f"[B1] Generated tokens: {all_tokens.shape}")

    # Use N_TOKENS tokens for processing
    # Process in batches to get activations
    BATCH_SIZE = 64  # tokens per batch (each batch = 64 token positions)
    SEQ_LEN = 128    # context window
    THRESHOLD = 0.01  # binary activation threshold

    # Flatten all tokens to a 1D sequence
    if all_tokens.dim() > 1:
        all_tokens_flat = all_tokens.flatten()
    else:
        all_tokens_flat = all_tokens

    # Select N_TOKENS positions for analysis
    # We'll process in chunks of SEQ_LEN to get proper context
    total_positions = min(N_TOKENS + SEQ_LEN, len(all_tokens_flat))
    tokens_subset = all_tokens_flat[:total_positions]

    print(f"[B1] Processing {N_TOKENS} positions from {total_positions} total tokens")
    report_progress(2, 5, metric={"phase": "running_activations", "n_tokens": N_TOKENS})

    # Initialize activation storage
    # binary_activations: shape (N_TOKENS, d_sae) - bool
    all_activations = []

    hook_name = "blocks.6.hook_resid_pre"

    # Process in chunks with context
    n_chunks = (N_TOKENS + SEQ_LEN - 1) // SEQ_LEN
    positions_processed = 0

    for chunk_idx in range(n_chunks):
        start_pos = chunk_idx * SEQ_LEN
        end_pos = min(start_pos + SEQ_LEN, N_TOKENS)

        # Get tokens with preceding context
        ctx_start = max(0, start_pos)
        ctx_end = min(start_pos + SEQ_LEN, len(tokens_subset))

        chunk_tokens = tokens_subset[ctx_start:ctx_end].unsqueeze(0).to(device)  # (1, seq_len)

        with torch.no_grad():
            _, cache = model.run_with_cache(chunk_tokens, names_filter=hook_name)
            residual = cache[hook_name][0]  # (seq_len, d_model)

            # Get SAE activations
            feature_acts = sae.encode(residual)  # (seq_len, d_sae)

            # Binary activation
            binary = (feature_acts > THRESHOLD).cpu()  # (seq_len, d_sae)
            all_activations.append(binary)

        positions_processed += (end_pos - start_pos)
        if chunk_idx % 5 == 0:
            print(f"[B1] Chunk {chunk_idx+1}/{n_chunks}, positions: {positions_processed}/{N_TOKENS}")

    # Concatenate all activations
    # Each chunk is (seq_len, d_sae), concatenate along dim 0
    activation_matrix = torch.cat(all_activations, dim=0)  # (total_positions, d_sae)
    # Trim to exactly N_TOKENS
    activation_matrix = activation_matrix[:N_TOKENS]
    print(f"[B1] Activation matrix shape: {activation_matrix.shape}")

    report_progress(3, 5, metric={"phase": "computing_statistics", "n_positions": activation_matrix.shape[0]})

    # Convert to float for computations
    act_float = activation_matrix.float()  # (N_TOKENS, d_sae)

    # 1. Activation frequency f_j = P(latent j fires) for all j
    activation_freq = act_float.mean(dim=0).numpy()  # (d_sae,)
    print(f"[B1] Activation freq computed. Mean freq: {activation_freq.mean():.4f}, Non-zero: {(activation_freq > 0).sum()}")

    # 2. Conditional frequencies for letter-involving pairs only
    # For each letter feature j (71 features), compute:
    #   P(j fires | i fires) for all i where f_i > 0.001
    # And P(i fires | j fires)

    letter_feature_ids = sorted(LETTER_FEATURE_IDS)
    n_letter = len(letter_feature_ids)

    # Get firing sets for all latents
    # act_bool: (N_TOKENS, d_sae)
    act_bool = activation_matrix.numpy().astype(bool)  # (N_TOKENS, d_sae)

    # For efficiency: precompute co-firing counts only for letter-involving pairs
    # P(j|i) = count(j fires AND i fires) / count(i fires)

    # Compute joint firing counts with letter features
    # act_letter: (N_TOKENS, n_letter) - firing of letter features
    letter_idx = np.array(letter_feature_ids)
    act_letter = act_bool[:, letter_idx]  # (N_TOKENS, n_letter)

    # Count how many tokens each latent fires
    fire_counts = act_bool.sum(axis=0)  # (d_sae,) - integer counts
    letter_fire_counts = fire_counts[letter_idx]  # (n_letter,)

    # For each letter feature j and each non-zero latent i:
    # co_count[i, j] = how many tokens both i and j fire
    # We compute co_count[i, letter_j] = act_bool[:, i] @ act_letter[:, j_idx]

    # This is: (N_TOKENS, d_sae)^T @ (N_TOKENS, n_letter) = (d_sae, n_letter)
    # But d_sae=24576, N_TOKENS up to 10000 - doable in batches

    print(f"[B1] Computing co-occurrence matrix...")

    # Batch over non-letter latents to save memory
    BATCH = 2048
    # co_counts[i, j_idx] = number of tokens where both latent i and letter_feature[j_idx] fire
    co_counts = np.zeros((d_sae, n_letter), dtype=np.float32)

    for batch_start in range(0, d_sae, BATCH):
        batch_end = min(batch_start + BATCH, d_sae)
        # (N_TOKENS, batch_size)
        act_batch = act_bool[:, batch_start:batch_end].astype(np.float32)
        # (batch_size, n_letter)
        batch_co = act_batch.T @ act_letter.astype(np.float32)
        co_counts[batch_start:batch_end, :] = batch_co

        if batch_start % (BATCH * 4) == 0:
            print(f"[B1]   Co-occurrence batch {batch_start}/{d_sae}...")

    print(f"[B1] Co-occurrence matrix computed: {co_counts.shape}")

    # 3. Compute conditional probabilities
    # P(j fires | i fires) = co_counts[i, j_idx] / fire_counts[i]
    # P(i fires | j fires) = co_counts[i, j_idx] / letter_fire_counts[j_idx]

    # Avoid division by zero
    fire_counts_safe = np.maximum(fire_counts, 1).astype(np.float32)  # (d_sae,)
    letter_fire_counts_safe = np.maximum(letter_fire_counts, 1).astype(np.float32)  # (n_letter,)

    # P(j|i): (d_sae, n_letter)
    p_j_given_i = co_counts / fire_counts_safe[:, None]  # divide each row by fire_counts[i]

    # P(i|j): (d_sae, n_letter)
    p_i_given_j = co_counts / letter_fire_counts_safe[None, :]  # divide each col by letter_fire_counts[j]

    # 4. Compute directed asymmetry A_cooccur(j) for each latent j (all d_sae)
    # For each latent j (indexed in full d_sae space):
    #   A_cooccur(j) = max_{i: f_i > 3*f_j} [P(j|i) / max(P(i|j), 1e-6)]
    # where j here is a LETTER feature

    # For non-letter features j, we use the reverse: find letter parents i
    # Actually per task_B1: A_cooccur(j) for ALL 24576 latents j, but:
    # "Only compute pairwise stats for pairs where at least one of {i,j} is in the letter feature set"
    # So A_cooccur is defined for letter features as target j, looking for parent i

    # Compute A_cooccur for each letter feature j_idx
    # f_j = activation_freq[j], f_i = activation_freq[i]
    # Condition: f_i > 3 * f_j (parent fires more than 3x than j)

    print(f"[B1] Computing A_cooccur for letter features...")

    A_cooccur_letter = np.zeros(n_letter, dtype=np.float32)
    A_cooccur_argmax = np.full(n_letter, -1, dtype=np.int32)  # which parent gives max

    for j_idx in range(n_letter):
        j_global = letter_feature_ids[j_idx]
        f_j = activation_freq[j_global]

        # Find parent latents: f_i > 3 * f_j
        if f_j == 0:
            A_cooccur_letter[j_idx] = 0.0
            continue

        parent_mask = activation_freq > 3 * f_j  # (d_sae,)

        if parent_mask.sum() == 0:
            A_cooccur_letter[j_idx] = 0.0
            continue

        # For parent latents i:
        # P(j|i) = p_j_given_i[i, j_idx]
        # P(i|j) = p_i_given_j[i, j_idx]
        # Asymmetry = P(j|i) / max(P(i|j), 1e-6)

        p_jgiven_parents = p_j_given_i[parent_mask, j_idx]  # (n_parents,)
        p_igiven_j_parents = p_i_given_j[parent_mask, j_idx]  # (n_parents,)

        asymmetry = p_jgiven_parents / np.maximum(p_igiven_j_parents, 1e-6)

        best_idx = np.argmax(asymmetry)
        A_cooccur_letter[j_idx] = asymmetry[best_idx]

        parent_indices = np.where(parent_mask)[0]
        A_cooccur_argmax[j_idx] = parent_indices[best_idx]

    print(f"[B1] A_cooccur computed for letter features.")
    print(f"[B1] A_cooccur stats: mean={A_cooccur_letter.mean():.3f}, max={A_cooccur_letter.max():.3f}")
    pairs_gt2 = (A_cooccur_letter > 2.0).sum()
    print(f"[B1] Pairs with A_cooccur > 2.0: {pairs_gt2}")

    # 5. Also compute A_cooccur for all d_sae latents (using letter features as potential parents)
    # A_cooccur_all[j] = max over letter parents i where f_i > 3*f_j
    print(f"[B1] Computing A_cooccur for all {d_sae} latents (letter parents)...")

    A_cooccur_all = np.zeros(d_sae, dtype=np.float32)

    # For each latent j, look for letter feature parents i where f_i > 3*f_j
    # p_j_given_i_from_letter[j, i_letter] = P(j fires | letter_i fires) = co_counts[j, i_letter] / fire_counts[j_ref]
    # Wait, let me reuse co_counts differently:
    # co_counts[i, j_idx] = P(i AND j_letter fire together)
    # For non-letter j as child, we need to check if letter feature is a parent
    # co_counts[j, j_letter_idx] ... but j might not be in letter features
    #
    # Actually co_counts is (d_sae, n_letter) where co_counts[i, k] = count of tokens where latent i fires AND letter_k fires
    # So for any latent j (could be letter or non-letter):
    #   P(j fires | letter_k fires) = co_counts[j, k] / letter_fire_counts[k]  = p_i_given_j[j, k]
    #   P(letter_k fires | j fires) = co_counts[j, k] / fire_counts[j]  = p_j_given_i[j, k]
    #
    # For A_cooccur(j) where j is any latent, and parents i are letter features:
    #   A_cooccur(j) = max_{k: letter_freq[k] > 3*f_j} [P(j|letter_k) / max(P(letter_k|j), 1e-6)]
    # = max_{k: letter_freq[k] > 3*f_j} [p_i_given_j[j, k] / max(p_j_given_i[j, k], 1e-6)]

    # This is well-defined for all j
    letter_freqs = activation_freq[letter_idx]  # (n_letter,)

    # Process in batches
    for batch_start in range(0, d_sae, BATCH):
        batch_end = min(batch_start + BATCH, d_sae)
        batch_freqs = activation_freq[batch_start:batch_end]  # (batch_size,)

        # For each j in batch, find letter parents where f_letter > 3*f_j
        # parent_mask[b, k] = (letter_freqs[k] > 3 * batch_freqs[b])
        parent_mask = letter_freqs[None, :] > 3 * batch_freqs[:, None]  # (batch_size, n_letter)

        # P(j|letter_k) = p_i_given_j[batch_start:batch_end, :]
        p_j_given_letter = p_i_given_j[batch_start:batch_end, :]  # (batch_size, n_letter)

        # P(letter_k|j) = p_j_given_i[batch_start:batch_end, :]
        p_letter_given_j = p_j_given_i[batch_start:batch_end, :]  # (batch_size, n_letter)

        # Asymmetry
        asymmetry = p_j_given_letter / np.maximum(p_letter_given_j, 1e-6)  # (batch_size, n_letter)

        # Mask out non-parents (set to 0)
        asymmetry_masked = asymmetry * parent_mask  # (batch_size, n_letter)

        # Max over letter parents
        A_cooccur_all[batch_start:batch_end] = asymmetry_masked.max(axis=1)

    print(f"[B1] A_cooccur_all computed. Top values for letter features:")
    for j_idx, j_global in enumerate(letter_feature_ids[:5]):
        print(f"  Letter feature {j_global}: A_cooccur={A_cooccur_letter[j_idx]:.3f}, A_cooccur_all={A_cooccur_all[j_global]:.3f}")

    # 6. Compute Jaccard overlap O(i,j) for letter-involving pairs
    print(f"[B1] Computing Jaccard overlap...")

    # Jaccard: O(i,j) = |S_i ∩ S_j| / |S_i ∪ S_j|
    # = co_count(i,j) / (fire_count_i + fire_count_j - co_count(i,j))

    # co_counts[i, j_idx] = co_firing count
    # union_counts[i, j_idx] = fire_counts[i] + letter_fire_counts[j_idx] - co_counts[i, j_idx]

    union_counts = (fire_counts[:, None] + letter_fire_counts_safe[None, :] - co_counts)
    # Avoid division by zero
    jaccard = co_counts / np.maximum(union_counts, 1.0)  # (d_sae, n_letter)

    # O_jaccard_letter[j_idx] = max over parents i where f_i > 3*f_j of O(i, j)
    O_jaccard_letter = np.zeros(n_letter, dtype=np.float32)

    for j_idx in range(n_letter):
        j_global = letter_feature_ids[j_idx]
        f_j = activation_freq[j_global]
        parent_mask_j = activation_freq > 3 * f_j
        if parent_mask_j.sum() == 0:
            O_jaccard_letter[j_idx] = 0.0
        else:
            O_jaccard_letter[j_idx] = jaccard[parent_mask_j, j_idx].max()

    print(f"[B1] Jaccard computed. Mean={O_jaccard_letter.mean():.4f}, Max={O_jaccard_letter.max():.4f}")

    report_progress(4, 5, metric={"phase": "saving_results"})

    # Compile results
    elapsed = time.time() - start_time

    # Top letter features by A_cooccur
    top_by_A = sorted(
        [(letter_feature_ids[i], float(A_cooccur_letter[i]), int(A_cooccur_argmax[i]))
         for i in range(n_letter)],
        key=lambda x: x[1], reverse=True
    )

    # Pilot pass criteria check
    pairs_gt2_count = int((A_cooccur_letter > 2.0).sum())
    pilot_pass = pairs_gt2_count >= 10

    # Save letter-feature-indexed results
    letter_results = {}
    for j_idx, j_global in enumerate(letter_feature_ids):
        letter_results[str(j_global)] = {
            "feature_id": j_global,
            "activation_freq": float(activation_freq[j_global]),
            "A_cooccur": float(A_cooccur_letter[j_idx]),
            "O_jaccard": float(O_jaccard_letter[j_idx]),
            "best_parent_id": int(A_cooccur_argmax[j_idx]),
            "best_parent_freq": float(activation_freq[A_cooccur_argmax[j_idx]]) if A_cooccur_argmax[j_idx] >= 0 else 0.0,
        }

    result = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "config": {
            "n_tokens": N_TOKENS,
            "activation_threshold": THRESHOLD,
            "sae": "gpt2-small-res-jb blocks.6.hook_resid_pre",
            "d_sae": int(d_sae),
            "n_letter_features": len(letter_feature_ids),
            "seed": SEED,
        },
        "activation_freq_summary": {
            "mean": float(activation_freq.mean()),
            "std": float(activation_freq.std()),
            "n_nonzero": int((activation_freq > 0).sum()),
            "n_rare": int((activation_freq > 0.001).sum()),
        },
        "A_cooccur_summary": {
            "letter_features": {
                "mean": float(A_cooccur_letter.mean()),
                "std": float(A_cooccur_letter.std()),
                "max": float(A_cooccur_letter.max()),
                "n_gt_1": int((A_cooccur_letter > 1.0).sum()),
                "n_gt_2": pairs_gt2_count,
                "n_gt_5": int((A_cooccur_letter > 5.0).sum()),
            },
            "all_latents": {
                "mean": float(A_cooccur_all.mean()),
                "max": float(A_cooccur_all.max()),
                "n_gt_2": int((A_cooccur_all > 2.0).sum()),
            }
        },
        "O_jaccard_summary": {
            "mean": float(O_jaccard_letter.mean()),
            "std": float(O_jaccard_letter.std()),
            "max": float(O_jaccard_letter.max()),
            "n_gt_0": int((O_jaccard_letter > 0).sum()),
        },
        "top_letter_features_by_A_cooccur": top_by_A[:20],
        "letter_feature_results": letter_results,
        "arrays": {
            # We save key arrays inline for B2 to use
            "activation_freq": activation_freq.tolist(),
            "A_cooccur_all": A_cooccur_all.tolist(),
            "letter_feature_ids": letter_feature_ids,
            "A_cooccur_letter": A_cooccur_letter.tolist(),
            "O_jaccard_letter": O_jaccard_letter.tolist(),
        },
        "pilot_pass_criteria": {
            "pass": pilot_pass,
            "pairs_gt2_found": pairs_gt2_count,
            "required": 10,
            "note": "A_cooccur(j) > 2.0 means j fires at least twice as often given parent i fires"
        }
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[B1] === RESULTS ===")
    print(f"[B1] Tokens processed: {N_TOKENS}")
    print(f"[B1] SAE d_sae: {d_sae}")
    print(f"[B1] Letter features analyzed: {n_letter}")
    print(f"[B1] Activation freq: mean={activation_freq.mean():.4f}, non-zero={int((activation_freq > 0).sum())}")
    print(f"[B1] A_cooccur for letter features: mean={A_cooccur_letter.mean():.3f}, max={A_cooccur_letter.max():.3f}")
    print(f"[B1] Pairs with A_cooccur > 2.0: {pairs_gt2_count} (need >= 10)")
    print(f"[B1] Pilot PASS: {pilot_pass}")
    print(f"[B1] Top letter features by A_cooccur:")
    for feat_id, a_val, parent_id in top_by_A[:10]:
        print(f"  Feature {feat_id}: A_cooccur={a_val:.3f}, best parent={parent_id} (freq={activation_freq[parent_id]:.4f})")
    print(f"\n[B1] Output saved to: {OUTPUT_FILE}")
    print(f"[B1] Elapsed: {elapsed:.1f}s")

    # Also save activation_freq and A_cooccur as numpy arrays for B2
    np.save(RESULTS_DIR.parent / "pilots" / "B1_activation_freq.npy", activation_freq)
    np.save(RESULTS_DIR.parent / "pilots" / "B1_A_cooccur_all.npy", A_cooccur_all)
    np.save(RESULTS_DIR.parent / "pilots" / "B1_A_cooccur_letter.npy", A_cooccur_letter)
    np.save(RESULTS_DIR.parent / "pilots" / "B1_O_jaccard_letter.npy", O_jaccard_letter)
    print(f"[B1] Arrays also saved as .npy files")

    if MODE == "FULL":
        # For full mode, also copy pilot npy to full directory
        np.save(RESULTS_DIR / "B1_activation_freq.npy", activation_freq)
        np.save(RESULTS_DIR / "B1_A_cooccur_all.npy", A_cooccur_all)
        np.save(RESULTS_DIR / "B1_A_cooccur_letter.npy", A_cooccur_letter)
        np.save(RESULTS_DIR / "B1_O_jaccard_letter.npy", O_jaccard_letter)

    report_progress(5, 5, metric={"phase": "done", "pilot_pass": pilot_pass})
    mark_done("success", f"A_cooccur computed for {d_sae} latents; {pairs_gt2_count} letter features with A>2; pilot_pass={pilot_pass}")

    print(f"\n[B1] DONE. Status: SUCCESS")

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(f"[B1] ERROR: {e}")
    print(tb)

    # Save error info
    error_result = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
        "status": "failed",
        "error": str(e),
        "traceback": tb,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(error_result, f, indent=2)

    mark_done("failed", f"Error: {str(e)[:200]}")
    sys.exit(1)
