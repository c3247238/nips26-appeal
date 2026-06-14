"""
C1A_activation_stats PILOT experiment
======================================
PILOT SCOPE: layer 8 (GPT-2 Small equivalent of "layer 12"), width ~24k (gpt2-small-res-jb),
             1k tokens instead of 10k.

Computes:
  - Per-latent mean activation frequency f_i
  - Pairwise co-activation rate P(a_i>0, a_j>0) for candidate pairs
  - sigma_ij = co_act / min(f_i, f_j)
  - alpha_ij = sigma_ij * (f_j / f_i)

Pass criteria (PILOT):
  - alpha_ij parquet file written with >= 500 valid pairs
  - no NaN/inf values in alpha_ij column
  - at least 10 pairs with alpha_ij > 1.0

Output: exp/results/pilots/C1A_activation_stats_pilot.parquet
        exp/results/pilots/C1A_activation_stats_pilot_summary.json
        exp/results/C1A_activation_stats_PROGRESS.json (progress)
        exp/results/C1A_activation_stats_DONE (completion marker)
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
PILOTS_DIR = RESULTS_DIR / "pilots"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.now()

# PID file
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PID_FILE.write_text(str(os.getpid()))
print(f"[C1A PILOT] PID={os.getpid()} written to {PID_FILE}")


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
    # Clean up PID
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
    print(f"[C1A PILOT] DONE: {status} — {summary}")


# ---- Setup ----
write_progress(1, 8, "Setting up environment and importing libraries")

import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[C1A PILOT] Device: {DEVICE}")

# ---- Parameters ----
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
N_TOKENS = 1000   # PILOT: 1k tokens
SEED = 42
COSINE_THRESHOLD = 0.15  # Pre-filter for pairwise candidates
F_MIN = 0.001            # Minimum activation frequency to include latent
BATCH_SIZE = 64
MIN_PAIRS = 500          # Pass criterion
MIN_ALPHA_ABOVE_1 = 10   # Pass criterion

torch.manual_seed(SEED)
np.random.seed(SEED)

write_progress(2, 8, "Loading SAE model")

from sae_lens import SAE
from transformer_lens import HookedTransformer

# Load SAE
sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE
)
# SAE from_pretrained may now return only the SAE object
if isinstance(sae, tuple):
    sae = sae[0]

D_SAE = sae.cfg.d_sae
print(f"[C1A PILOT] SAE loaded: release={SAE_RELEASE}, id={SAE_ID}, d_sae={D_SAE}")

write_progress(3, 8, "Loading language model (GPT-2 Small)")

# Load GPT-2 Small
model = HookedTransformer.from_pretrained_no_processing(
    "gpt2",
    device=DEVICE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model.eval()
print(f"[C1A PILOT] Model loaded: gpt2-small")

write_progress(4, 8, "Loading text tokens")

# Load tokens — try multiple sources with fallback
from datasets import load_dataset
import glob

tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

texts = []
# Strategy 1: Skylion007 openwebtext streaming
try:
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for item in dataset.take(50):
        texts.append(item["text"])
    print(f"[C1A PILOT] Loaded {len(texts)} texts from Skylion007/openwebtext")
except Exception as e:
    print(f"[C1A PILOT] Skylion007 failed: {e}")
    texts = []

# Strategy 2: wikitext (cached)
if len(texts) == 0:
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=False)
        for i, item in enumerate(dataset):
            if item["text"].strip():
                texts.append(item["text"])
            if len(texts) >= 200:
                break
        print(f"[C1A PILOT] Loaded {len(texts)} texts from wikitext")
    except Exception as e:
        print(f"[C1A PILOT] wikitext failed: {e}")
        texts = []

# Fallback: dummy text
if len(texts) == 0:
    print("[C1A PILOT] Using dummy text fallback")
    dummy = "The quick brown fox jumps over the lazy dog. " * 50
    texts = [dummy] * 50

# Collect tokens
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
print(f"[C1A PILOT] Collected {tokens.shape[1]} tokens")

write_progress(5, 8, f"Running {N_TOKENS} tokens through SAE to collect activations")

# Run tokens in batches through model → extract hook activations → encode with SAE
HOOK_NAME = "blocks.8.hook_resid_pre"
SEQ_LEN = 32  # Process in chunks

# Collect SAE activations
all_sae_acts = []  # shape: [n_tokens, d_sae]

with torch.no_grad():
    n_total = tokens.shape[1]
    for start in range(0, n_total, SEQ_LEN):
        end = min(start + SEQ_LEN, n_total)
        chunk = tokens[:, start:end]  # [1, chunk_len]

        # Get hook activation
        _, cache = model.run_with_cache(chunk, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME]  # [1, chunk_len, d_model]
        resid_flat = resid.reshape(-1, resid.shape[-1])  # [chunk_len, d_model]

        # Encode with SAE
        sae_acts = sae.encode(resid_flat)  # [chunk_len, d_sae]
        # Handle different return types
        if isinstance(sae_acts, tuple):
            sae_acts = sae_acts[0]

        all_sae_acts.append(sae_acts.cpu().float())

        del cache, resid, resid_flat
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

all_sae_acts = torch.cat(all_sae_acts, dim=0)  # [n_tokens, d_sae]
print(f"[C1A PILOT] SAE activations shape: {all_sae_acts.shape}")

# Free model memory
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

write_progress(6, 8, "Computing activation statistics (f_i, co-activation rates)")

# Compute per-latent activation frequency f_i
# Binary: latent > 0
binary_acts = (all_sae_acts > 0).float()  # [n_tokens, d_sae]
n_tokens_actual = binary_acts.shape[0]

f_i = binary_acts.mean(dim=0).numpy()  # [d_sae]

# Filter: only latents with f_i > F_MIN
active_latents = np.where(f_i > F_MIN)[0]
print(f"[C1A PILOT] Active latents (f_i > {F_MIN}): {len(active_latents)} / {D_SAE}")

# Compute decoder cosine similarities for pre-filtering
# Get decoder weight matrix: [d_sae, d_model]
decoder_W = sae.W_dec.detach().cpu().float().numpy()  # [d_sae, d_model]

# Normalize decoder vectors
norms = np.linalg.norm(decoder_W, axis=1, keepdims=True)
norms = np.where(norms < 1e-8, 1.0, norms)
decoder_W_norm = decoder_W / norms  # [d_sae, d_model]

# Restrict to active latents for efficiency
active_decoder = decoder_W_norm[active_latents]  # [n_active, d_model]
active_f = f_i[active_latents]                    # [n_active]

n_active = len(active_latents)
print(f"[C1A PILOT] Computing cosine similarities for {n_active} active latents...")

# Compute cosine similarity in batches to find candidate pairs
COSINE_BATCH = 256
candidate_pairs = []

for i_batch_start in range(0, n_active, COSINE_BATCH):
    i_batch_end = min(i_batch_start + COSINE_BATCH, n_active)
    batch_i = active_decoder[i_batch_start:i_batch_end]  # [batch, d_model]

    # Cosine similarity with all active latents
    cos_sim = batch_i @ active_decoder.T  # [batch, n_active]

    for local_i, global_i_idx in enumerate(range(i_batch_start, i_batch_end)):
        global_i = active_latents[global_i_idx]
        row = cos_sim[local_i]

        # Find j where cos_sim > threshold, j > global_i_idx (avoid duplicates), f_j > F_MIN
        for j_idx in range(global_i_idx + 1, n_active):
            if row[j_idx] > COSINE_THRESHOLD:
                global_j = active_latents[j_idx]
                candidate_pairs.append((global_i, global_j, float(row[j_idx])))

    if i_batch_start % (COSINE_BATCH * 4) == 0:
        print(f"  Cosine batch {i_batch_start}/{n_active}, found {len(candidate_pairs)} pairs so far")

print(f"[C1A PILOT] Candidate pairs after cosine filter: {len(candidate_pairs)}")

if len(candidate_pairs) == 0:
    mark_done("failed", "No candidate pairs found after cosine filter")
    sys.exit(1)

write_progress(7, 8, f"Computing alpha_ij for {len(candidate_pairs)} candidate pairs")

# Compute co-activation rates and alpha_ij for candidate pairs
binary_acts_np = binary_acts.numpy()  # [n_tokens, d_sae]

rows = []
for (lat_i, lat_j, cos_ij) in candidate_pairs:
    act_i = binary_acts_np[:, lat_i]
    act_j = binary_acts_np[:, lat_j]

    fi = float(f_i[lat_i])
    fj = float(f_i[lat_j])

    coact = float(np.mean(act_i * act_j))

    if min(fi, fj) < 1e-9:
        continue

    sigma_ij = coact / min(fi, fj)

    if fi < 1e-9:
        continue

    alpha_ij = sigma_ij * (fj / fi)

    rows.append({
        "latent_i": int(lat_i),
        "latent_j": int(lat_j),
        "f_i": fi,
        "f_j": fj,
        "coact_rate": coact,
        "sigma_ij": sigma_ij,
        "alpha_ij": alpha_ij,
        "decoder_cosine": cos_ij,
    })

df = pd.DataFrame(rows)
print(f"[C1A PILOT] Valid pairs computed: {len(df)}")

# Check for NaN/inf
if len(df) > 0:
    n_nan = df["alpha_ij"].isna().sum()
    n_inf = np.isinf(df["alpha_ij"].values).sum()
    n_above_1 = (df["alpha_ij"] > 1.0).sum()
    print(f"  NaN in alpha_ij: {n_nan}")
    print(f"  Inf in alpha_ij: {n_inf}")
    print(f"  Pairs with alpha_ij > 1.0: {n_above_1}")

    # Show top-10
    top10 = df.nlargest(10, "alpha_ij")[["latent_i", "latent_j", "f_i", "f_j", "alpha_ij", "decoder_cosine"]]
    print("\nTop-10 alpha_ij pairs:")
    print(top10.to_string(index=False))
else:
    n_nan = 0
    n_inf = 0
    n_above_1 = 0

# Save parquet
parquet_path = PILOTS_DIR / "C1A_activation_stats_pilot.parquet"
df.to_parquet(parquet_path, index=False)
print(f"[C1A PILOT] Saved parquet: {parquet_path}")

# Evaluate pass criteria
pass_pairs = len(df) >= MIN_PAIRS
pass_no_nan = (n_nan == 0) and (n_inf == 0)
pass_alpha_above_1 = n_above_1 >= MIN_ALPHA_ABOVE_1

go_no_go = "GO" if (pass_pairs and pass_no_nan and pass_alpha_above_1) else "NO_GO"

# Save summary
summary = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "model": "gpt2-small",
    "sae_release": SAE_RELEASE,
    "sae_id": SAE_ID,
    "d_sae": D_SAE,
    "n_tokens": n_tokens_actual,
    "active_latents": int(len(active_latents)),
    "total_candidate_pairs": len(candidate_pairs),
    "valid_pairs": len(df),
    "cosine_threshold": COSINE_THRESHOLD,
    "f_min": F_MIN,
    "go_no_go": go_no_go,
    "pass_criteria": {
        "pass_pairs_ge_500": bool(pass_pairs),
        "pass_no_nan_inf": bool(pass_no_nan),
        "pass_alpha_above_1_ge_10": bool(pass_alpha_above_1),
        "n_pairs": len(df),
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
        "n_alpha_above_1": int(n_above_1),
    },
    "alpha_ij_stats": {
        "mean": float(df["alpha_ij"].mean()) if len(df) > 0 else None,
        "median": float(df["alpha_ij"].median()) if len(df) > 0 else None,
        "max": float(df["alpha_ij"].max()) if len(df) > 0 else None,
        "min": float(df["alpha_ij"].min()) if len(df) > 0 else None,
        "std": float(df["alpha_ij"].std()) if len(df) > 0 else None,
    },
    "top10_pairs": top10.to_dict(orient="records") if len(df) > 0 else [],
    "runtime_seconds": (datetime.now() - START_TIME).total_seconds(),
}

summary_path = PILOTS_DIR / "C1A_activation_stats_pilot_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"[C1A PILOT] Summary saved: {summary_path}")

# Write summary markdown
md_lines = [
    "# C1A Activation Stats — PILOT Summary",
    "",
    f"**GO/NO-GO: {go_no_go}**",
    "",
    "## Configuration",
    f"- Model: GPT-2 Small (open-model anchor; Gemma-2-2b requires gated access)",
    f"- SAE: {SAE_RELEASE} / {SAE_ID} (d_sae={D_SAE})",
    f"- Tokens: {n_tokens_actual} (pilot: 1k)",
    f"- Cosine threshold: {COSINE_THRESHOLD}",
    f"- f_min: {F_MIN}",
    "",
    "## Pass Criteria Results",
    f"- Valid pairs >= 500: {'PASS' if pass_pairs else 'FAIL'} (n={len(df)})",
    f"- No NaN/inf in alpha_ij: {'PASS' if pass_no_nan else 'FAIL'} (nan={n_nan}, inf={n_inf})",
    f"- Pairs with alpha_ij > 1.0 >= 10: {'PASS' if pass_alpha_above_1 else 'FAIL'} (n={n_above_1})",
    "",
    "## Alpha_ij Distribution",
    f"- Mean: {summary['alpha_ij_stats']['mean']:.4f}" if summary['alpha_ij_stats']['mean'] is not None else "- Mean: N/A",
    f"- Median: {summary['alpha_ij_stats']['median']:.4f}" if summary['alpha_ij_stats']['median'] is not None else "- Median: N/A",
    f"- Max: {summary['alpha_ij_stats']['max']:.4f}" if summary['alpha_ij_stats']['max'] is not None else "- Max: N/A",
]
if len(df) > 0:
    md_lines += [
        "",
        "## Top-10 Pairs by alpha_ij",
        "| latent_i | latent_j | f_i | f_j | alpha_ij | decoder_cosine |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in top10.iterrows():
        md_lines.append(f"| {row['latent_i']} | {row['latent_j']} | {row['f_i']:.4f} | {row['f_j']:.4f} | {row['alpha_ij']:.4f} | {row['decoder_cosine']:.4f} |")

md_path = PILOTS_DIR / "C1A_activation_stats_pilot_summary.md"
md_path.write_text("\n".join(md_lines))

write_progress(8, 8, f"Done. GO/NO-GO: {go_no_go}", {
    "go_no_go": go_no_go,
    "valid_pairs": len(df),
    "n_alpha_above_1": int(n_above_1),
})

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    with open(gpu_progress_path, "r") as f:
        gpu_progress = json.load(f)
except:
    gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

end_time = datetime.now()
actual_min = int((end_time - START_TIME).total_seconds() / 60) + 1

if go_no_go == "GO":
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
else:
    if TASK_ID not in gpu_progress.get("failed", []):
        gpu_progress.setdefault("failed", []).append(TASK_ID)

# Remove from running
gpu_progress.get("running", {}).pop(TASK_ID, None)

gpu_progress.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 90,
    "actual_min": actual_min,
    "start_time": START_TIME.isoformat(),
    "end_time": end_time.isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "d_sae": D_SAE,
        "n_tokens": n_tokens_actual,
        "mode": "pilot",
        "cosine_threshold": COSINE_THRESHOLD,
        "f_min": F_MIN,
    }
}

with open(gpu_progress_path, "w") as f:
    json.dump(gpu_progress, f, indent=2)

mark_done(
    "success" if go_no_go == "GO" else "failed",
    f"Pilot C1A: {go_no_go}. Valid pairs={len(df)}, alpha>1: {n_above_1}, nan={n_nan}",
    final_progress={
        "task_id": TASK_ID,
        "step": 8,
        "total_steps": 8,
        "message": f"GO/NO-GO: {go_no_go}",
        "metrics": {
            "valid_pairs": len(df),
            "n_alpha_above_1": int(n_above_1),
        },
        "elapsed_sec": (datetime.now() - START_TIME).total_seconds(),
        "updated_at": datetime.now().isoformat(),
    }
)

print(f"\n[C1A PILOT] Final result: {go_no_go}")
print(f"  Valid pairs: {len(df)}")
print(f"  Pairs with alpha_ij > 1.0: {n_above_1}")
print(f"  NaN/inf: {n_nan}/{n_inf}")
sys.exit(0)
