"""
task_B1_cooccurrence_graph_full.py
Build directed co-occurrence graph for ARS_v2 — FULL MODE with token generation.

Uses 10,000 token positions from OpenWebText through GPT-2 with gpt2-small-res-jb L6 SAE.

Computes:
1. activation frequency f_j = P(latent j fires) for all j
2. conditional frequency P(j fires | i fires) for letter-involving pairs only
3. directed asymmetry A_cooccur(j) for each letter feature j
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

SEED = 42
N_TOKENS = 10000

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

start_time = time.time()

# Write PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, metric=None):
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
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

print("[B1-FULL] Loading SAE and GPT-2...")
report_progress(0, 6, metric={"phase": "loading"})

try:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[B1-FULL] Using device: {device}")

    # Load GPT-2 model
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    print("[B1-FULL] GPT-2 loaded")

    # Load SAE
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
    )
    sae = sae.to(device)
    sae.eval()
    d_sae = sae.cfg.d_sae
    print(f"[B1-FULL] SAE loaded: d_sae={d_sae}")

    report_progress(1, 6, metric={"phase": "models_loaded"})

    # Generate tokens from OpenWebText
    print("[B1-FULL] Generating tokens from OpenWebText...")
    from datasets import load_dataset
    from transformer_lens import utils as tl_utils

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # Use streaming to get enough text
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    all_text_parts = []
    target_chars = 200000  # enough for ~50k tokens
    total_chars = 0

    for i, item in enumerate(dataset):
        text = item['text']
        all_text_parts.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break

    print(f"[B1-FULL] Collected {len(all_text_parts)} documents, ~{total_chars} chars")

    combined_text = " ".join(all_text_parts)
    encoded = tokenizer(combined_text, return_tensors="pt",
                        truncation=True, max_length=80000)
    all_tokens_flat = encoded['input_ids'][0]
    print(f"[B1-FULL] Total tokens available: {len(all_tokens_flat)}")

    # Take the first N_TOKENS
    n_available = min(N_TOKENS + 512, len(all_tokens_flat))
    tokens_for_analysis = all_tokens_flat[:n_available]

    report_progress(2, 6, metric={"phase": "tokenizing_done", "n_tokens_available": len(all_tokens_flat)})

    # Process tokens in batches using sliding windows
    ACTIVATION_THRESHOLD = 0.01
    SEQ_LEN = 256  # context window per batch
    BATCH_SIZE = 4   # number of sequences per GPU batch

    print(f"[B1-FULL] Computing SAE activations for {N_TOKENS} positions...")

    all_activations = []  # list of (n, d_sae) bool tensors
    hook_name = "blocks.6.hook_resid_pre"

    # Compute how many chunks we need to get N_TOKENS positions
    n_chunks = (N_TOKENS + SEQ_LEN - 1) // SEQ_LEN
    positions_collected = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * SEQ_LEN
        end = min(start + SEQ_LEN, N_TOKENS, len(tokens_for_analysis))

        chunk_tokens = tokens_for_analysis[start:end].unsqueeze(0).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(chunk_tokens, names_filter=hook_name)
            residual = cache[hook_name][0]  # (seq_len, d_model)
            feature_acts = sae.encode(residual)  # (seq_len, d_sae)
            binary = (feature_acts > ACTIVATION_THRESHOLD).cpu().bool()
            all_activations.append(binary)

        positions_collected += (end - start)

        if chunk_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[B1-FULL]   Chunk {chunk_idx+1}/{n_chunks}, positions={positions_collected}/{N_TOKENS}, elapsed={elapsed:.0f}s")

        if positions_collected >= N_TOKENS:
            break

    # Concatenate
    activation_matrix = torch.cat(all_activations, dim=0)[:N_TOKENS]
    print(f"[B1-FULL] Activation matrix: {activation_matrix.shape}")

    report_progress(3, 6, metric={"phase": "activations_done", "positions": activation_matrix.shape[0]})

    # Convert to numpy for efficient computation
    act_bool = activation_matrix.numpy()  # (N_TOKENS, d_sae) bool
    act_float = act_bool.astype(np.float32)

    # 1. Activation frequency
    activation_freq = act_float.mean(axis=0)  # (d_sae,)
    n_active = (activation_freq > 0).sum()
    print(f"[B1-FULL] Activation freq: mean={activation_freq.mean():.4f}, n_active={n_active}")

    letter_idx = np.array(LETTER_FEATURE_IDS)
    n_letter = len(letter_idx)
    act_letter = act_bool[:, letter_idx]  # (N_TOKENS, n_letter)

    # Fire counts
    fire_counts = act_bool.sum(axis=0).astype(np.float32)  # (d_sae,)
    letter_fire_counts = fire_counts[letter_idx]  # (n_letter,)

    # 2. Co-occurrence matrix (d_sae, n_letter)
    print(f"[B1-FULL] Computing co-occurrence matrix ({d_sae} x {n_letter})...")
    BATCH = 2048
    co_counts = np.zeros((d_sae, n_letter), dtype=np.float32)

    for batch_start in range(0, d_sae, BATCH):
        batch_end = min(batch_start + BATCH, d_sae)
        act_batch = act_bool[:, batch_start:batch_end].astype(np.float32)
        co_counts[batch_start:batch_end, :] = act_batch.T @ act_float[:, letter_idx]
        if batch_start % (BATCH * 4) == 0:
            print(f"[B1-FULL]   Co-occ batch {batch_start}/{d_sae}...")

    print(f"[B1-FULL] Co-occurrence matrix done")
    report_progress(4, 6, metric={"phase": "cooccurrence_done"})

    # 3. Conditional probs
    fire_counts_safe = np.maximum(fire_counts, 1.0)
    letter_fire_safe = np.maximum(letter_fire_counts, 1.0)

    p_j_given_i = co_counts / fire_counts_safe[:, None]  # P(letter_j | latent_i fires)
    p_i_given_j = co_counts / letter_fire_safe[None, :]  # P(latent_i | letter_j fires)

    # 4. A_cooccur for each letter feature j
    print(f"[B1-FULL] Computing A_cooccur for letter features...")
    A_cooccur_letter = np.zeros(n_letter, dtype=np.float32)
    A_cooccur_argmax = np.full(n_letter, -1, dtype=np.int32)

    for j_idx in range(n_letter):
        j_global = LETTER_FEATURE_IDS[j_idx]
        f_j = activation_freq[j_global]
        if f_j == 0:
            continue
        parent_mask = activation_freq > 3 * f_j
        if parent_mask.sum() == 0:
            continue
        p_jgiven_parents = p_j_given_i[parent_mask, j_idx]
        p_parents_given_j = p_i_given_j[parent_mask, j_idx]
        asymmetry = p_jgiven_parents / np.maximum(p_parents_given_j, 1e-6)
        best = np.argmax(asymmetry)
        A_cooccur_letter[j_idx] = asymmetry[best]
        A_cooccur_argmax[j_idx] = np.where(parent_mask)[0][best]

    # 5. A_cooccur for all latents (using letter parents)
    print(f"[B1-FULL] Computing A_cooccur for all {d_sae} latents...")
    A_cooccur_all = np.zeros(d_sae, dtype=np.float32)
    letter_freqs = activation_freq[letter_idx]

    for batch_start in range(0, d_sae, BATCH):
        batch_end = min(batch_start + BATCH, d_sae)
        batch_freqs = activation_freq[batch_start:batch_end]
        parent_mask = letter_freqs[None, :] > 3 * batch_freqs[:, None]
        p_j_given_letter = p_i_given_j[batch_start:batch_end, :]
        p_letter_given_j = p_j_given_i[batch_start:batch_end, :]
        asymmetry = p_j_given_letter / np.maximum(p_letter_given_j, 1e-6)
        asymmetry_masked = asymmetry * parent_mask
        A_cooccur_all[batch_start:batch_end] = asymmetry_masked.max(axis=1)

    # 6. Jaccard overlap
    print(f"[B1-FULL] Computing Jaccard overlap...")
    union = fire_counts[:, None] + letter_fire_safe[None, :] - co_counts
    jaccard = co_counts / np.maximum(union, 1.0)

    O_jaccard_letter = np.zeros(n_letter, dtype=np.float32)
    for j_idx in range(n_letter):
        j_global = LETTER_FEATURE_IDS[j_idx]
        f_j = activation_freq[j_global]
        if f_j == 0:
            continue
        parent_mask = activation_freq > 3 * f_j
        if parent_mask.sum() > 0:
            O_jaccard_letter[j_idx] = jaccard[parent_mask, j_idx].max()

    report_progress(5, 6, metric={"phase": "computing_done"})

    elapsed = time.time() - start_time
    pairs_gt2 = int((A_cooccur_letter > 2.0).sum())
    pilot_pass = pairs_gt2 >= 10

    print(f"\n[B1-FULL] === RESULTS ===")
    print(f"[B1-FULL] N_TOKENS: {N_TOKENS}")
    print(f"[B1-FULL] d_sae: {d_sae}")
    print(f"[B1-FULL] Letter features: {n_letter}")
    print(f"[B1-FULL] Active latents: {n_active}/{d_sae}")
    print(f"[B1-FULL] A_cooccur(letter): mean={A_cooccur_letter.mean():.3f}, max={A_cooccur_letter.max():.3f}")
    print(f"[B1-FULL] Pairs with A_cooccur > 2.0: {pairs_gt2}")
    print(f"[B1-FULL] O_jaccard(letter): mean={O_jaccard_letter.mean():.4f}, max={O_jaccard_letter.max():.4f}")

    top_by_A = sorted(
        [(LETTER_FEATURE_IDS[i], float(A_cooccur_letter[i]), int(A_cooccur_argmax[i]))
         for i in range(n_letter)],
        key=lambda x: x[1], reverse=True
    )
    print(f"[B1-FULL] Top letter features by A_cooccur:")
    for feat_id, a_val, parent_id in top_by_A[:10]:
        parent_freq = float(activation_freq[parent_id]) if parent_id >= 0 else 0.0
        print(f"  Feature {feat_id}: A={a_val:.3f}, parent={parent_id} (freq={parent_freq:.4f})")

    # Build per-feature results
    letter_results = {}
    for j_idx, j_global in enumerate(LETTER_FEATURE_IDS):
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
        "mode": "FULL",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "config": {
            "n_tokens": N_TOKENS,
            "activation_threshold": ACTIVATION_THRESHOLD,
            "sae": "gpt2-small-res-jb blocks.6.hook_resid_pre",
            "d_sae": int(d_sae),
            "n_letter_features": n_letter,
            "seed": SEED,
        },
        "activation_freq_summary": {
            "mean": float(activation_freq.mean()),
            "std": float(activation_freq.std()),
            "n_active": int(n_active),
            "n_rare": int((activation_freq > 0.001).sum()),
        },
        "A_cooccur_summary": {
            "letter_features": {
                "mean": float(A_cooccur_letter.mean()),
                "std": float(A_cooccur_letter.std()),
                "max": float(A_cooccur_letter.max()),
                "n_gt_1": int((A_cooccur_letter > 1.0).sum()),
                "n_gt_2": pairs_gt2,
                "n_gt_5": int((A_cooccur_letter > 5.0).sum()),
                "n_gt_10": int((A_cooccur_letter > 10.0).sum()),
            },
            "all_latents": {
                "mean": float(A_cooccur_all.mean()),
                "max": float(A_cooccur_all.max()),
                "n_gt_2": int((A_cooccur_all > 2.0).sum()),
            }
        },
        "O_jaccard_summary": {
            "mean": float(O_jaccard_letter.mean()),
            "max": float(O_jaccard_letter.max()),
            "n_gt_0": int((O_jaccard_letter > 0).sum()),
        },
        "top_letter_features_by_A_cooccur": top_by_A[:20],
        "letter_feature_results": letter_results,
        "arrays": {
            "activation_freq": activation_freq.tolist(),
            "A_cooccur_all": A_cooccur_all.tolist(),
            "letter_feature_ids": LETTER_FEATURE_IDS,
            "A_cooccur_letter": A_cooccur_letter.tolist(),
            "O_jaccard_letter": O_jaccard_letter.tolist(),
        },
        "pilot_pass_criteria": {
            "pass": pilot_pass,
            "pairs_gt2_found": pairs_gt2,
            "required": 10,
            "note": "A_cooccur > 2 means j fires at least twice as often given parent i fires"
        }
    }

    out_file = RESULTS_DIR / "B1_cooccurrence_graph.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Save numpy arrays
    np.save(RESULTS_DIR / "B1_activation_freq.npy", activation_freq)
    np.save(RESULTS_DIR / "B1_A_cooccur_all.npy", A_cooccur_all)
    np.save(RESULTS_DIR / "B1_A_cooccur_letter.npy", A_cooccur_letter)
    np.save(RESULTS_DIR / "B1_O_jaccard_letter.npy", O_jaccard_letter)
    # Also save to pilots for B2 to find
    np.save(PILOTS_DIR / "B1_activation_freq.npy", activation_freq)
    np.save(PILOTS_DIR / "B1_A_cooccur_all.npy", A_cooccur_all)
    np.save(PILOTS_DIR / "B1_A_cooccur_letter.npy", A_cooccur_letter)
    np.save(PILOTS_DIR / "B1_O_jaccard_letter.npy", O_jaccard_letter)

    print(f"\n[B1-FULL] Output: {out_file}")
    print(f"[B1-FULL] Pilot PASS: {pilot_pass}")
    print(f"[B1-FULL] Elapsed: {elapsed:.1f}s")

    # Also write pilot output for consistency
    pilot_out = PILOTS_DIR / "B1_cooccurrence_graph_pilot.json"
    with open(pilot_out, 'w') as f:
        json.dump(result, f, indent=2)

    report_progress(6, 6, metric={"phase": "done", "pilot_pass": pilot_pass})
    mark_done("success", f"A_cooccur for {d_sae} latents; {pairs_gt2} letter features with A>2; pass={pilot_pass}")

    # Update gpu_progress.json
    import fcntl
    gpu_progress_file = Path(WORKSPACE) / "exp" / "gpu_progress.json"
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
    if gpu_progress_file.exists():
        try:
            with open(gpu_progress_file) as f:
                gp = json.load(f)
        except:
            pass
    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 15,
        "actual_min": round(elapsed / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "n_tokens": N_TOKENS,
            "d_sae": int(d_sae),
            "n_letter_features": n_letter,
            "gpu": "RTX PRO 6000",
        }
    }
    with open(gpu_progress_file, 'w') as f:
        json.dump(gp, f, indent=2)

    print("[B1-FULL] gpu_progress.json updated")

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(f"[B1-FULL] ERROR: {e}")
    print(tb)
    mark_done("failed", f"Error: {str(e)[:200]}")
    sys.exit(1)
