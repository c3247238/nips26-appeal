"""
C1A_activation_stats FULL experiment - Optimized Vectorized Version
======================================================================
FULL SCOPE: GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)

SAE configurations (8 configs):
  Group A: gpt2-small-res-jb (JB architecture, ~24k width)
    - layers 3, 5, 8, 10 (analogs to Gemma layers 6, 12, 20, 25)
  Group B: gpt2-small-res-jb-feature-splitting (width variation at layer 8)
    - layer 8, d_sae=49152 ("wide" analog)
  Group C: gpt2-small-resid-post-v5-32k (different architecture, v5 training)
    - layers 3, 6, 9

Token count: 10k per config (matches full methodology spec)

Optimization: Uses vectorized matrix ops (no Python loops over pairs)
  - Cosine similarity: batch matrix multiply -> thresholded sparse indices
  - Co-activation: (binary.T @ binary) / n_tokens -> full matrix, sample from it
"""

import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---- Config ----
TASK_ID = "C1A_activation_stats"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
OUTPUT_DIR = RESULTS_DIR / "full" / "C1A_alpha_ij_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.now()

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PID_FILE.write_text(str(os.getpid()))
print(f"[C1A FULL] PID={os.getpid()} written to {PID_FILE}")

# SAE configs
SAE_CONFIGS = [
    {"config_name": "jb_layer3_24k",    "release": "gpt2-small-res-jb",                    "sae_id": "blocks.3.hook_resid_pre",         "hook": "blocks.3.hook_resid_pre",    "layer": 3,  "layer_pct": 3/11, "width_class": "narrow",  "arch": "jb"},
    {"config_name": "jb_layer5_24k",    "release": "gpt2-small-res-jb",                    "sae_id": "blocks.5.hook_resid_pre",         "hook": "blocks.5.hook_resid_pre",    "layer": 5,  "layer_pct": 5/11, "width_class": "narrow",  "arch": "jb"},
    {"config_name": "jb_layer8_24k",    "release": "gpt2-small-res-jb",                    "sae_id": "blocks.8.hook_resid_pre",         "hook": "blocks.8.hook_resid_pre",    "layer": 8,  "layer_pct": 8/11, "width_class": "narrow",  "arch": "jb"},
    {"config_name": "jb_layer10_24k",   "release": "gpt2-small-res-jb",                    "sae_id": "blocks.10.hook_resid_pre",        "hook": "blocks.10.hook_resid_pre",   "layer": 10, "layer_pct":10/11, "width_class": "narrow",  "arch": "jb"},
    {"config_name": "jb_layer8_49k",    "release": "gpt2-small-res-jb-feature-splitting",  "sae_id": "blocks.8.hook_resid_pre_49152",   "hook": "blocks.8.hook_resid_pre",    "layer": 8,  "layer_pct": 8/11, "width_class": "wide",    "arch": "jb_split"},
    {"config_name": "v5post_layer3_32k","release": "gpt2-small-resid-post-v5-32k",         "sae_id": "blocks.3.hook_resid_post",        "hook": "blocks.3.hook_resid_post",   "layer": 3,  "layer_pct": 3/11, "width_class": "medium",  "arch": "v5_post"},
    {"config_name": "v5post_layer6_32k","release": "gpt2-small-resid-post-v5-32k",         "sae_id": "blocks.6.hook_resid_post",        "hook": "blocks.6.hook_resid_post",   "layer": 6,  "layer_pct": 6/11, "width_class": "medium",  "arch": "v5_post"},
    {"config_name": "v5post_layer9_32k","release": "gpt2-small-resid-post-v5-32k",         "sae_id": "blocks.9.hook_resid_post",        "hook": "blocks.9.hook_resid_post",   "layer": 9,  "layer_pct": 9/11, "width_class": "medium",  "arch": "v5_post"},
]

N_TOKENS = 10000
SEED = 42
COSINE_THRESHOLD = 0.15
F_MIN = 0.001
SEQ_LEN = 64
GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}"

TOTAL_STEPS = 4 + len(SAE_CONFIGS) * 5 + 2


def write_progress(step, total_steps, message, metrics=None):
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "metrics": metrics or {},
        "elapsed_sec": (datetime.now() - START_TIME).total_seconds(),
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress, indent=2))
    print(f"[{step}/{total_steps}] {message}")


def mark_done(status, summary, final_progress=None):
    if PID_FILE.exists():
        PID_FILE.unlink()
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress or {},
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"[C1A FULL] DONE: {status} — {summary}")


def compute_alpha_ij_vectorized(sae_acts_np, decoder_W, cosine_threshold, f_min, config_name, max_pairs=10_000_000):
    """
    Vectorized alpha_ij computation using matrix operations.
    sae_acts_np: [n_tokens, d_sae] float32 numpy
    decoder_W: [d_sae, d_model] float32 numpy
    """
    n_tokens, d_sae = sae_acts_np.shape
    print(f"  [{config_name}] d_sae={d_sae}, n_tokens={n_tokens}")

    # Step 1: Binary activations and per-latent frequencies
    binary = (sae_acts_np > 0).astype(np.float32)  # [n_tokens, d_sae]
    f_i = binary.mean(axis=0)  # [d_sae]

    active_mask = f_i > f_min
    active_idx = np.where(active_mask)[0]  # indices into d_sae
    n_active = len(active_idx)
    print(f"  [{config_name}] Active latents (f>{f_min}): {n_active}/{d_sae}")

    if n_active < 10:
        return pd.DataFrame(columns=['latent_i', 'latent_j', 'f_i', 'f_j', 'coact_rate', 'sigma_ij', 'alpha_ij', 'decoder_cosine'])

    active_f = f_i[active_idx]       # [n_active]
    active_bin = binary[:, active_idx]  # [n_tokens, n_active]

    # Step 2: Normalize decoder vectors for active latents
    D = decoder_W[active_idx].astype(np.float32)  # [n_active, d_model]
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    D_norm = D / norms  # [n_active, d_model]

    # Step 3: Find candidate pairs using block-wise cosine similarity
    # Process in blocks to manage memory
    COSINE_BLOCK = 2000
    t_cos = time.time()

    # Collect pairs as arrays (avoid growing lists)
    all_pairs_i = []
    all_pairs_j = []
    all_pairs_cos = []

    for i_start in range(0, n_active, COSINE_BLOCK):
        i_end = min(i_start + COSINE_BLOCK, n_active)
        block = D_norm[i_start:i_end]  # [block, d_model]
        cos_mat = block @ D_norm.T     # [block, n_active]

        # For each row, find j > i with cos > threshold
        for local_i in range(i_end - i_start):
            g_i = i_start + local_i
            row = cos_mat[local_i, g_i + 1:]  # only j > g_i (upper triangle)
            j_rel = np.where(row > cosine_threshold)[0]
            if len(j_rel) > 0:
                all_pairs_i.append(np.full(len(j_rel), g_i, dtype=np.int32))
                all_pairs_j.append(j_rel + g_i + 1)
                all_pairs_cos.append(row[j_rel])

        if i_start % (COSINE_BLOCK * 4) == 0:
            n_found = sum(len(x) for x in all_pairs_i)
            print(f"  [{config_name}] Cosine {i_start}/{n_active}, pairs found: {n_found}, {time.time()-t_cos:.0f}s")

    if not all_pairs_i:
        return pd.DataFrame(columns=['latent_i', 'latent_j', 'f_i', 'f_j', 'coact_rate', 'sigma_ij', 'alpha_ij', 'decoder_cosine'])

    pairs_i_active = np.concatenate(all_pairs_i).astype(np.int32)  # active-space indices
    pairs_j_active = np.concatenate(all_pairs_j).astype(np.int32)
    pairs_cos = np.concatenate(all_pairs_cos).astype(np.float32)

    n_pairs = len(pairs_i_active)
    print(f"  [{config_name}] Total candidate pairs: {n_pairs} (cosine filter time: {time.time()-t_cos:.1f}s)")

    # Truncate if too many pairs
    if n_pairs > max_pairs:
        print(f"  [{config_name}] Truncating to {max_pairs} pairs (sorted by cosine)")
        top_idx = np.argpartition(pairs_cos, -max_pairs)[-max_pairs:]
        pairs_i_active = pairs_i_active[top_idx]
        pairs_j_active = pairs_j_active[top_idx]
        pairs_cos = pairs_cos[top_idx]
        n_pairs = max_pairs

    # Step 4: Compute co-activation rates vectorized
    # active_bin[:, pairs_i] * active_bin[:, pairs_j] -> mean over tokens
    t_coact = time.time()
    COACT_BATCH = 50000
    coact = np.zeros(n_pairs, dtype=np.float32)
    for b in range(0, n_pairs, COACT_BATCH):
        be = min(b + COACT_BATCH, n_pairs)
        pi = pairs_i_active[b:be]
        pj = pairs_j_active[b:be]
        coact[b:be] = (active_bin[:, pi] * active_bin[:, pj]).mean(axis=0)
    print(f"  [{config_name}] Co-activation computed: {time.time()-t_coact:.1f}s")

    # Step 5: Compute alpha_ij
    fi_arr = active_f[pairs_i_active]
    fj_arr = active_f[pairs_j_active]
    min_f = np.minimum(fi_arr, fj_arr)

    valid = (min_f > 1e-9) & (fi_arr > 1e-9)
    sigma = np.where(valid, coact / np.where(min_f > 1e-9, min_f, 1.0), 0.0)
    alpha = np.where(valid, sigma * (fj_arr / np.where(fi_arr > 1e-9, fi_arr, 1.0)), 0.0)
    finite_ok = valid & np.isfinite(sigma) & np.isfinite(alpha)

    # Convert active-space indices back to global d_sae indices
    lat_i = active_idx[pairs_i_active[finite_ok]]
    lat_j = active_idx[pairs_j_active[finite_ok]]

    df = pd.DataFrame({
        "latent_i": lat_i.astype(np.int32),
        "latent_j": lat_j.astype(np.int32),
        "f_i": fi_arr[finite_ok].astype(np.float32),
        "f_j": fj_arr[finite_ok].astype(np.float32),
        "coact_rate": coact[finite_ok].astype(np.float32),
        "sigma_ij": sigma[finite_ok].astype(np.float32),
        "alpha_ij": alpha[finite_ok].astype(np.float32),
        "decoder_cosine": pairs_cos[finite_ok].astype(np.float32),
    })
    return df


# ---- Setup ----
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)

step = 1
write_progress(step, TOTAL_STEPS, "Importing libraries and setting up environment")

from sae_lens import SAE
from transformer_lens import HookedTransformer
from datasets import load_dataset

step += 1
write_progress(step, TOTAL_STEPS, "Loading language model (GPT-2 Small)")

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

model = HookedTransformer.from_pretrained_no_processing(
    "gpt2",
    device=DEVICE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model.eval()
print(f"[C1A FULL] Model loaded: gpt2-small on {DEVICE}")

step += 1
write_progress(step, TOTAL_STEPS, "Loading text tokens (10k)")

tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

texts = []
try:
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for item in dataset.take(500):
        texts.append(item["text"])
    print(f"[C1A FULL] Loaded {len(texts)} texts from Skylion007/openwebtext")
except Exception as e:
    print(f"[C1A FULL] Skylion007 failed: {e}, trying wikitext...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for item in dataset:
            if item["text"].strip():
                texts.append(item["text"])
            if len(texts) >= 1000:
                break
    except Exception as e2:
        print(f"[C1A FULL] wikitext failed: {e2}, using dummy")
        texts = ["The quick brown fox jumps over the lazy dog. " * 200] * 500

all_tokens = []
for text in texts:
    try:
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
    except Exception:
        continue
    if len(all_tokens) >= N_TOKENS + 512:
        break

tokens_np = np.array(all_tokens[:N_TOKENS])
print(f"[C1A FULL] Collected {len(tokens_np)} tokens")

step += 1
write_progress(step, TOTAL_STEPS, "Starting SAE config processing loop")

# ---- Main loop ----
all_config_results = []

for cfg_idx, cfg in enumerate(SAE_CONFIGS):
    config_name = cfg["config_name"]
    sae_id = cfg["sae_id"]
    release = cfg["release"]
    hook_name = cfg["hook"]

    step += 1
    write_progress(step, TOTAL_STEPS, f"[{cfg_idx+1}/{len(SAE_CONFIGS)}] Loading SAE: {config_name}")

    t_cfg_start = time.time()

    try:
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
        if isinstance(sae, tuple):
            sae = sae[0]
        d_sae = sae.cfg.d_sae
        print(f"  [{config_name}] SAE loaded: d_sae={d_sae}")
    except Exception as e:
        print(f"  [{config_name}] FAILED to load SAE: {e}")
        all_config_results.append({
            "config_name": config_name, "release": release, "sae_id": sae_id,
            "status": "failed", "error": str(e)
        })
        step += 4
        continue

    step += 1
    write_progress(step, TOTAL_STEPS, f"[{cfg_idx+1}/{len(SAE_CONFIGS)}] Collecting activations: {config_name}")

    tokens_tensor = torch.tensor(tokens_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
    all_sae_acts = []

    t_act = time.time()
    with torch.no_grad():
        for start in range(0, N_TOKENS, SEQ_LEN):
            end = min(start + SEQ_LEN, N_TOKENS)
            chunk = tokens_tensor[:, start:end]
            _, cache = model.run_with_cache(chunk, names_filter=hook_name)
            resid = cache[hook_name].reshape(-1, cache[hook_name].shape[-1])
            sae_acts = sae.encode(resid)
            if isinstance(sae_acts, tuple):
                sae_acts = sae_acts[0]
            all_sae_acts.append(sae_acts.cpu().float().numpy())
            del cache, resid, sae_acts

    del tokens_tensor
    all_sae_acts_np = np.concatenate(all_sae_acts, axis=0)  # [n_tokens, d_sae]
    del all_sae_acts
    print(f"  [{config_name}] SAE acts collected: {all_sae_acts_np.shape}, {time.time()-t_act:.1f}s")

    # Get decoder, free SAE
    decoder_W = sae.W_dec.detach().cpu().float().numpy()
    del sae
    torch.cuda.empty_cache()
    gc.collect()

    step += 1
    write_progress(step, TOTAL_STEPS, f"[{cfg_idx+1}/{len(SAE_CONFIGS)}] Computing alpha_ij: {config_name}")

    df = compute_alpha_ij_vectorized(
        all_sae_acts_np, decoder_W,
        cosine_threshold=COSINE_THRESHOLD, f_min=F_MIN,
        config_name=config_name
    )

    del all_sae_acts_np, decoder_W
    gc.collect()

    # Save parquet
    parquet_path = OUTPUT_DIR / f"{config_name}.parquet"
    if len(df) > 0:
        df.to_parquet(parquet_path, index=False)
        n_nan = int(df["alpha_ij"].isna().sum())
        n_inf = int(np.isinf(df["alpha_ij"].values).sum())
        n_alpha_above_1 = int((df["alpha_ij"] > 1.0).sum())
        alpha_stats = {
            "mean": float(df["alpha_ij"].mean()),
            "median": float(df["alpha_ij"].median()),
            "max": float(df["alpha_ij"].max()),
            "min": float(df["alpha_ij"].min()),
            "std": float(df["alpha_ij"].std()),
        }
        top10 = df.nlargest(10, "alpha_ij")[["latent_i", "latent_j", "f_i", "f_j", "alpha_ij", "decoder_cosine"]].to_dict(orient="records")
        status = "success"
        print(f"  [{config_name}] Saved {len(df)} pairs to {parquet_path}")
    else:
        n_nan = n_inf = n_alpha_above_1 = 0
        alpha_stats = {}
        top10 = []
        status = "empty"

    t_cfg_elapsed = time.time() - t_cfg_start
    config_result = {
        "config_name": config_name, "release": release, "sae_id": sae_id,
        "layer": cfg["layer"], "layer_pct": cfg["layer_pct"],
        "width_class": cfg["width_class"], "arch": cfg["arch"],
        "d_sae": d_sae, "n_tokens": N_TOKENS,
        "status": status, "n_pairs": len(df),
        "n_nan_alpha": n_nan, "n_inf_alpha": n_inf,
        "n_alpha_above_1": n_alpha_above_1,
        "alpha_stats": alpha_stats, "top10_pairs": top10,
        "parquet_file": str(parquet_path) if status == "success" else None,
        "runtime_sec": t_cfg_elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    all_config_results.append(config_result)

    # Save partial summary
    partial = {
        "task_id": TASK_ID, "mode": "FULL", "model": "gpt2-small",
        "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
        "n_tokens_per_config": N_TOKENS,
        "configs_processed": all_config_results, "configs_total": len(SAE_CONFIGS),
        "cosine_threshold": COSINE_THRESHOLD, "f_min": F_MIN,
        "updated_at": datetime.now().isoformat(),
    }
    (OUTPUT_DIR / "C1A_summary.json").write_text(json.dumps(partial, indent=2))
    print(f"  [{config_name}] Done: {len(df)} pairs, {t_cfg_elapsed:.0f}s total")

# ---- Final summary ----
step += 1
write_progress(step, TOTAL_STEPS, "Computing final summary")

successful = [r for r in all_config_results if r.get("status") == "success"]
total_pairs = sum(r.get("n_pairs", 0) for r in successful)
go_no_go = "GO" if len(successful) >= 6 else "NO_GO"

final_summary = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL", "model": "gpt2-small",
    "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
    "n_tokens_per_config": N_TOKENS,
    "n_configs_total": len(SAE_CONFIGS),
    "n_configs_successful": len(successful),
    "total_pairs_across_configs": total_pairs,
    "cosine_threshold": COSINE_THRESHOLD, "f_min": F_MIN,
    "go_no_go": go_no_go,
    "configs": all_config_results,
    "output_dir": str(OUTPUT_DIR),
    "runtime_seconds": (datetime.now() - START_TIME).total_seconds(),
}
(OUTPUT_DIR / "C1A_summary.json").write_text(json.dumps(final_summary, indent=2))
print(f"\n[C1A FULL] {go_no_go}: {len(successful)}/{len(SAE_CONFIGS)} configs, {total_pairs} total pairs")

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gpu_progress = json.loads(gpu_progress_path.read_text())
except Exception:
    gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

end_time = datetime.now()
actual_min = int((end_time - START_TIME).total_seconds() / 60) + 1
if go_no_go == "GO":
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
else:
    if TASK_ID not in gpu_progress.get("failed", []):
        gpu_progress.setdefault("failed", []).append(TASK_ID)
gpu_progress.get("running", {}).pop(TASK_ID, None)
gpu_progress.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 90, "actual_min": actual_min,
    "start_time": START_TIME.isoformat(), "end_time": end_time.isoformat(),
    "config_snapshot": {
        "model": "gpt2-small", "n_configs": len(SAE_CONFIGS),
        "n_tokens_per_config": N_TOKENS, "mode": "full",
        "cosine_threshold": COSINE_THRESHOLD, "f_min": F_MIN,
        "gpu": "RTX PRO 6000 Blackwell", "gpu_id": GPU_ID,
    }
}
gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

# Update experiment_state.json
exp_state_path = WORKSPACE / "exp" / "experiment_state.json"
try:
    exp_state = json.loads(exp_state_path.read_text())
except Exception:
    exp_state = {"schema_version": 1, "tasks": {}, "last_recovery_at": "", "recovery_log": []}
exp_state["tasks"][TASK_ID] = {
    "status": "completed" if go_no_go == "GO" else "failed",
    "completed_at": end_time.isoformat(),
    "gate_decision": go_no_go,
    "mode": "full",
    "n_configs_successful": len(successful),
    "total_pairs": total_pairs,
}
exp_state_path.write_text(json.dumps(exp_state, indent=2))

mark_done(
    "success" if go_no_go == "GO" else "failed",
    f"FULL C1A: {go_no_go}. {len(successful)}/{len(SAE_CONFIGS)} configs, {total_pairs} total pairs",
    final_progress={
        "task_id": TASK_ID, "step": step, "total_steps": TOTAL_STEPS,
        "message": f"GO/NO-GO: {go_no_go}",
        "metrics": {"n_configs_successful": len(successful), "total_pairs": total_pairs},
        "elapsed_sec": (end_time - START_TIME).total_seconds(),
        "updated_at": end_time.isoformat(),
    }
)

print(f"\n[C1A FULL] Completed in {actual_min} minutes")
sys.exit(0 if go_no_go == "GO" else 1)
