"""
Phase 2.1: Geometric Absorption Score (GAS) Computation — PILOT mode

Computes the Geometric Absorption Score, the first unsupervised absorption detector.
GAS uses decoder-activation co-occurrence mismatch to detect absorption without probe supervision.

Pipeline:
1. Load Gemma Scope SAE at layer 12 (16k width) — pilot uses 16k only
2. Extract decoder weight matrix W_dec
3. Compute pairwise cosine similarity for all pairs with cos > 0.3
4. Run SAE on 200 sequences from Wikitext-2: record activation patterns
5. For each high-similarity pair (i, j):
   a. Compute actual co-activation frequency
   b. Estimate expected co-activation from decoder geometry
   c. Compute frequency asymmetry freq(i)/freq(j)
6. GAS(i -> j) = cos_sim(d_i, d_j) * [E[coact(i,j)] - actual_coact(i,j)] / E[coact(i,j)] * freq(i) / freq(j)
7. Per-feature absorption vulnerability score: max_i GAS(i -> j)
8. Save: GAS scores for all features, high-similarity pair statistics, co-activation matrix

Pilot pass criteria:
- GAS computed for at least 1000 high-similarity decoder pairs at 16k width
- Co-activation statistics collected on 100+ sequences
- GAS distribution is non-degenerate (variance > 0, some scores > 0.1)
- Frequency asymmetry term computed
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────

TASK_ID = "phase2_gas_computation"
SEED = 42
DEVICE = "cuda"
PILOT_MODE = True

# SAE config for pilot: layer 12, 16k width only
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"

# Pilot settings
N_SEQUENCES = 200          # Number of sequences for activation collection (pilot: 200, full: 10k)
MAX_SEQ_LEN = 128          # Tokens per sequence
COS_THRESHOLD = 0.3        # Minimum cosine similarity for "high-similarity" pairs
MIN_PAIRS_TARGET = 1000    # Pilot pass criterion: at least 1000 pairs

# Model path (local cache to avoid gated repo auth issues)
GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

# Paths
WORKSPACE = Path(os.environ.get("WORKSPACE", "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"))
RESULTS_DIR = WORKSPACE / "exp" / "results"
PHASE2_DIR = RESULTS_DIR / "phase2"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── PID file for system recovery ──────────────────────────────────────────────

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
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


# ── Step 1: Load SAE and extract decoder weights ──────────────────────────────

report_progress(TASK_ID, RESULTS_DIR, 1, 6, step=0, total_steps=5,
                metric={"phase": "loading_sae"})

print(f"[{datetime.now().isoformat()}] Step 1: Loading SAE...")
start_time = time.time()

from sae_lens import SAE

sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE,
)
print(f"  SAE loaded: {SAE_RELEASE} / {SAE_ID}")
print(f"  SAE config: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

# Extract decoder weight matrix W_dec: shape (n_features, d_model)
W_dec = sae.W_dec.detach().clone()  # (d_sae, d_in)
n_features = W_dec.shape[0]
d_model = W_dec.shape[1]
print(f"  Decoder matrix: {n_features} features x {d_model} dimensions")

# Normalize decoder vectors for cosine similarity computation
W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)

sae_load_time = time.time() - start_time
print(f"  SAE loaded in {sae_load_time:.1f}s")


# ── Step 2: Find high-similarity decoder pairs (cos > threshold) ──────────────

report_progress(TASK_ID, RESULTS_DIR, 2, 6, step=1, total_steps=5,
                metric={"phase": "computing_cosine_similarity"})

print(f"\n[{datetime.now().isoformat()}] Step 2: Computing pairwise cosine similarity...")
t0 = time.time()

# For 16k features, the full cosine similarity matrix is 16k x 16k = 256M entries.
# At float16 that's ~512MB — feasible. We'll compute in chunks to manage memory.

CHUNK_SIZE = 2048  # Process 2048 features at a time

high_sim_pairs = []  # list of (i, j, cos_sim) tuples where i < j
cos_sim_values = []

for chunk_start in range(0, n_features, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, n_features)
    chunk = W_dec_normed[chunk_start:chunk_end]  # (chunk_size, d_model)

    # Compute cosine similarity of this chunk against ALL features
    # cos_sim = chunk @ W_dec_normed.T  -> (chunk_size, n_features)
    sim_matrix = chunk @ W_dec_normed.T  # (chunk_size, n_features)

    # Find pairs where cos > threshold and i < j
    for local_i in range(chunk_end - chunk_start):
        global_i = chunk_start + local_i
        # Only look at j > global_i to avoid duplicates
        j_start = global_i + 1
        if j_start >= n_features:
            continue
        row = sim_matrix[local_i, j_start:]
        high_mask = row > COS_THRESHOLD
        if high_mask.any():
            high_indices = torch.nonzero(high_mask, as_tuple=True)[0]
            for idx in high_indices:
                j = j_start + idx.item()
                cos_val = row[idx].item()
                high_sim_pairs.append((global_i, j, cos_val))
                cos_sim_values.append(cos_val)

    if (chunk_start // CHUNK_SIZE) % 2 == 0:
        print(f"  Processed features {chunk_start}-{chunk_end} / {n_features}, "
              f"found {len(high_sim_pairs)} pairs so far")

    del sim_matrix
    torch.cuda.empty_cache()

n_high_sim_pairs = len(high_sim_pairs)
cos_sim_time = time.time() - t0
print(f"  Found {n_high_sim_pairs} pairs with cos > {COS_THRESHOLD}")
print(f"  Cosine similarity computation took {cos_sim_time:.1f}s")

if n_high_sim_pairs < MIN_PAIRS_TARGET:
    print(f"  WARNING: Only {n_high_sim_pairs} pairs found, below target {MIN_PAIRS_TARGET}.")
    print(f"  Lowering threshold to 0.2...")
    COS_THRESHOLD = 0.2
    high_sim_pairs = []
    cos_sim_values = []
    for chunk_start in range(0, n_features, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_features)
        chunk = W_dec_normed[chunk_start:chunk_end]
        sim_matrix = chunk @ W_dec_normed.T
        for local_i in range(chunk_end - chunk_start):
            global_i = chunk_start + local_i
            j_start = global_i + 1
            if j_start >= n_features:
                continue
            row = sim_matrix[local_i, j_start:]
            high_mask = row > COS_THRESHOLD
            if high_mask.any():
                high_indices = torch.nonzero(high_mask, as_tuple=True)[0]
                for idx in high_indices:
                    j = j_start + idx.item()
                    cos_val = row[idx].item()
                    high_sim_pairs.append((global_i, j, cos_val))
                    cos_sim_values.append(cos_val)
        del sim_matrix
        torch.cuda.empty_cache()
    n_high_sim_pairs = len(high_sim_pairs)
    print(f"  With threshold {COS_THRESHOLD}: found {n_high_sim_pairs} pairs")

# Cosine similarity distribution statistics
cos_sim_arr = np.array(cos_sim_values)
cos_sim_stats = {
    "n_pairs": n_high_sim_pairs,
    "cos_threshold": COS_THRESHOLD,
    "mean": float(np.mean(cos_sim_arr)) if len(cos_sim_arr) > 0 else 0,
    "std": float(np.std(cos_sim_arr)) if len(cos_sim_arr) > 0 else 0,
    "min": float(np.min(cos_sim_arr)) if len(cos_sim_arr) > 0 else 0,
    "max": float(np.max(cos_sim_arr)) if len(cos_sim_arr) > 0 else 0,
    "median": float(np.median(cos_sim_arr)) if len(cos_sim_arr) > 0 else 0,
}
print(f"  Cosine sim stats: mean={cos_sim_stats['mean']:.4f}, "
      f"std={cos_sim_stats['std']:.4f}, max={cos_sim_stats['max']:.4f}")

# Free decoder matrix from GPU — we'll keep W_dec_normed on CPU for later
W_dec_normed_cpu = W_dec_normed.cpu()
del W_dec, W_dec_normed
torch.cuda.empty_cache()
gc.collect()


# ── Step 3: Collect activation patterns from text data ──────────────────────────

report_progress(TASK_ID, RESULTS_DIR, 3, 6, step=2, total_steps=5,
                metric={"phase": "collecting_activations", "n_pairs": n_high_sim_pairs})

print(f"\n[{datetime.now().isoformat()}] Step 3: Collecting activation patterns...")
t0 = time.time()

# Load Gemma 2 2B for residual stream activations
from transformer_lens import HookedTransformer

print("  Loading Gemma 2 2B from local cache...")
hf_model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
model = HookedTransformer.from_pretrained(
    "gemma-2-2b",
    device=DEVICE,
    dtype=torch.bfloat16,
    hf_model=hf_model,
    tokenizer=tokenizer,
)
del hf_model
gc.collect()
torch.cuda.empty_cache()
print(f"  Model loaded. Tokenizer vocab size: {model.tokenizer.vocab_size}")

# Load text data — use Wikitext-2 for reproducibility
from datasets import load_dataset

print("  Loading Wikitext-2 dataset...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text_source = "wikitext-2-raw-v1"
except Exception:
    print("  Wikitext-2 not available, trying wikitext-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    text_source = "wikitext-103-raw-v1"

print(f"  Dataset loaded: {text_source}, {len(dataset)} entries")

# Prepare sequences by concatenating non-empty lines and chunking
all_text = " ".join([t for t in dataset["text"] if len(t.strip()) > 50])
tokens_encoded = model.tokenizer.encode(all_text)
if isinstance(tokens_encoded, torch.Tensor):
    tokens_full = tokens_encoded.flatten()
elif isinstance(tokens_encoded, list):
    tokens_full = torch.tensor(tokens_encoded, dtype=torch.long)
else:
    tokens_full = tokens_encoded[0] if hasattr(tokens_encoded, '__getitem__') else torch.tensor(tokens_encoded)
print(f"  Total tokens available: {len(tokens_full)}")

# Create chunks of MAX_SEQ_LEN tokens
n_available_seqs = len(tokens_full) // MAX_SEQ_LEN
n_seqs = min(N_SEQUENCES, n_available_seqs)
print(f"  Creating {n_seqs} sequences of {MAX_SEQ_LEN} tokens each")

# Collect activation indicators: for each feature, on how many tokens does it fire?
# Also collect pairwise co-activation counts for high-similarity pairs

# Build lookup structures for efficient co-activation counting
# Create a set of features involved in high-similarity pairs
involved_features = set()
for i, j, _ in high_sim_pairs:
    involved_features.add(i)
    involved_features.add(j)
involved_features_list = sorted(involved_features)
feat_to_idx = {f: idx for idx, f in enumerate(involved_features_list)}
n_involved = len(involved_features_list)
print(f"  {n_involved} unique features involved in high-similarity pairs")

# Feature activation frequencies (across all tokens)
feature_freq = np.zeros(n_features, dtype=np.float64)
total_tokens = 0

# Co-activation counts for high-similarity pairs
# Store as dict: (i, j) -> count of tokens where both fire
coact_counts = {}
for i, j, _ in high_sim_pairs:
    coact_counts[(i, j)] = 0

# Process sequences in batches
BATCH_SIZE = 8  # Small batch to fit Gemma 2 2B + SAE in memory
n_batches = (n_seqs + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(n_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, n_seqs)
    batch_seqs = []
    for seq_idx in range(batch_start, batch_end):
        start_tok = seq_idx * MAX_SEQ_LEN
        end_tok = start_tok + MAX_SEQ_LEN
        batch_seqs.append(tokens_full[start_tok:end_tok])

    batch_tensor = torch.stack(batch_seqs).to(DEVICE)  # (batch, seq_len)

    # Get residual stream activations at layer 12
    with torch.no_grad():
        _, cache = model.run_with_cache(
            batch_tensor,
            names_filter=["blocks.12.hook_resid_post"],
            return_type=None,
        )
    resid = cache["blocks.12.hook_resid_post"]  # (batch, seq_len, d_model)

    # Encode through SAE
    with torch.no_grad():
        sae_out = sae.encode(resid.reshape(-1, d_model))  # (batch*seq_len, d_sae)

    # Binary activation mask: which features fire?
    active_mask = (sae_out > 0).cpu().numpy()  # (batch*seq_len, d_sae)
    n_tokens_batch = active_mask.shape[0]
    total_tokens += n_tokens_batch

    # Update per-feature frequencies
    feature_freq += active_mask.sum(axis=0)

    # Update co-activation counts for high-similarity pairs
    # For efficiency, only check involved features
    involved_active = active_mask[:, involved_features_list]  # (n_tokens, n_involved)
    for pair_idx, (i, j, _) in enumerate(high_sim_pairs):
        idx_i = feat_to_idx[i]
        idx_j = feat_to_idx[j]
        coact = np.sum(involved_active[:, idx_i] & involved_active[:, idx_j])
        coact_counts[(i, j)] += int(coact)

    del resid, sae_out, cache, active_mask, involved_active, batch_tensor
    torch.cuda.empty_cache()

    if batch_idx % 5 == 0:
        print(f"  Batch {batch_idx+1}/{n_batches}, tokens so far: {total_tokens}")

activation_time = time.time() - t0
print(f"  Activation collection took {activation_time:.1f}s")
print(f"  Total tokens processed: {total_tokens}")

# Convert frequencies to rates
feature_freq_rate = feature_freq / total_tokens
n_active_features = int((feature_freq > 0).sum())
n_dead_features = n_features - n_active_features
print(f"  Active features: {n_active_features}/{n_features} "
      f"(dead: {n_dead_features}, {100*n_dead_features/n_features:.1f}%)")

# Activation frequency statistics
active_rates = feature_freq_rate[feature_freq_rate > 0]
freq_stats = {
    "n_active_features": n_active_features,
    "n_dead_features": n_dead_features,
    "mean_activation_rate": float(np.mean(active_rates)) if len(active_rates) > 0 else 0,
    "median_activation_rate": float(np.median(active_rates)) if len(active_rates) > 0 else 0,
    "max_activation_rate": float(np.max(active_rates)) if len(active_rates) > 0 else 0,
    "total_tokens": total_tokens,
    "n_sequences": n_seqs,
}
print(f"  Feature activation rate: mean={freq_stats['mean_activation_rate']:.4f}, "
      f"median={freq_stats['median_activation_rate']:.4f}")


# ── Step 4: Calibrate expected co-activation from decoder geometry ─────────────

report_progress(TASK_ID, RESULTS_DIR, 4, 6, step=3, total_steps=5,
                metric={"phase": "calibrating_coactivation"})

print(f"\n[{datetime.now().isoformat()}] Step 4: Calibrating expected co-activation...")
t0 = time.time()

# The key insight for GAS: if two features have high decoder cosine similarity,
# we'd EXPECT them to co-activate (they represent similar directions).
# When actual co-activation is MUCH LOWER than expected, it suggests absorption:
# one feature suppresses the other.

# Build calibration curve: cos_sim -> expected co-activation rate
# Sample random pairs at various cosine similarity levels

N_CALIBRATION_PAIRS = 5000
np.random.seed(SEED)

# Sample random feature pairs
cal_i = np.random.randint(0, n_features, N_CALIBRATION_PAIRS)
cal_j = np.random.randint(0, n_features, N_CALIBRATION_PAIRS)
# Ensure i != j
mask = cal_i != cal_j
cal_i = cal_i[mask]
cal_j = cal_j[mask]

# Compute their cosine similarities
cal_cos = []
cal_coact_rate = []
for idx in range(len(cal_i)):
    fi, fj = int(cal_i[idx]), int(cal_j[idx])
    # Cosine similarity from normalized decoder vectors
    cos_val = float(torch.dot(W_dec_normed_cpu[fi], W_dec_normed_cpu[fj]).item())
    # Co-activation rate: P(both fire) / P(either fires)
    freq_i = feature_freq[fi]
    freq_j = feature_freq[fj]
    if freq_i == 0 or freq_j == 0:
        continue
    # We need actual co-activation for these pairs — expensive if not in our set
    # For calibration, estimate from independent activation: P(i)*P(j)*total_tokens
    # This gives the baseline under independence assumption
    expected_coact_independent = (freq_i / total_tokens) * (freq_j / total_tokens)
    cal_cos.append(cos_val)
    cal_coact_rate.append(float(expected_coact_independent))

cal_cos = np.array(cal_cos)
cal_coact_rate = np.array(cal_coact_rate)

# For the actual calibration curve, we use the high-similarity pairs where we have
# real co-activation data. Bin by cosine similarity and compute mean co-activation.
pair_cos_vals = np.array([c for _, _, c in high_sim_pairs])
pair_coact_rates = np.array([
    coact_counts[(i, j)] / total_tokens for i, j, _ in high_sim_pairs
])

# Also compute independent expectation for each pair
pair_independent_rates = np.array([
    (feature_freq[i] / total_tokens) * (feature_freq[j] / total_tokens)
    for i, j, _ in high_sim_pairs
])

# Bin the high-sim pairs by cosine similarity to build calibration curve
cos_bins = np.arange(COS_THRESHOLD, 1.01, 0.05)
calibration_curve = []
for bin_start, bin_end in zip(cos_bins[:-1], cos_bins[1:]):
    bin_mask = (pair_cos_vals >= bin_start) & (pair_cos_vals < bin_end)
    if bin_mask.sum() > 10:  # Need enough pairs for reliable estimate
        mean_coact = float(np.mean(pair_coact_rates[bin_mask]))
        mean_indep = float(np.mean(pair_independent_rates[bin_mask]))
        calibration_curve.append({
            "cos_bin_start": float(bin_start),
            "cos_bin_end": float(bin_end),
            "n_pairs": int(bin_mask.sum()),
            "mean_actual_coact": mean_coact,
            "mean_independent_coact": mean_indep,
            "coact_ratio": mean_coact / mean_indep if mean_indep > 0 else float('nan'),
        })

print(f"  Calibration curve: {len(calibration_curve)} bins")
for entry in calibration_curve:
    print(f"    cos [{entry['cos_bin_start']:.2f}, {entry['cos_bin_end']:.2f}): "
          f"n={entry['n_pairs']}, actual={entry['mean_actual_coact']:.6f}, "
          f"independent={entry['mean_independent_coact']:.6f}, "
          f"ratio={entry['coact_ratio']:.2f}")

# Build expected co-activation function:
# For a given cos_sim, interpolate expected co-activation rate from the calibration curve
# Use the binned mean actual co-activation as the expected value
if len(calibration_curve) >= 2:
    cal_cos_centers = [(e["cos_bin_start"] + e["cos_bin_end"]) / 2 for e in calibration_curve]
    cal_expected_coact = [e["mean_actual_coact"] for e in calibration_curve]

    from scipy.interpolate import interp1d
    expected_coact_fn = interp1d(
        cal_cos_centers, cal_expected_coact,
        kind='linear', fill_value='extrapolate', bounds_error=False
    )
else:
    # Fallback: use independent rate as expected
    print("  WARNING: Not enough calibration bins, using independent rate as expected")
    expected_coact_fn = None

cal_time = time.time() - t0
print(f"  Calibration took {cal_time:.1f}s")


# ── Step 5: Compute GAS scores ───────────────────────────────────────────────

report_progress(TASK_ID, RESULTS_DIR, 5, 6, step=4, total_steps=5,
                metric={"phase": "computing_gas"})

print(f"\n[{datetime.now().isoformat()}] Step 5: Computing GAS scores...")
t0 = time.time()

# GAS(i -> j) = cos_sim(d_i, d_j) * [E[coact(i,j)] - actual_coact(i,j)] / E[coact(i,j)] * freq(i) / freq(j)
#
# Interpretation:
# - cos_sim: decoder similarity — high means they SHOULD co-activate
# - mismatch term: how much LESS they co-activate than expected
# - freq asymmetry: absorption goes from frequent to rare (child absorbs parent)
#
# GAS(i -> j) > 0 means feature i may be absorbing feature j

gas_scores = []  # List of dicts with pair info and GAS scores
per_feature_gas = np.zeros(n_features)  # max GAS vulnerability per feature (as target j)

for pair_idx, (i, j, cos_val) in enumerate(high_sim_pairs):
    freq_i = feature_freq_rate[i]
    freq_j = feature_freq_rate[j]

    if freq_i == 0 or freq_j == 0:
        continue

    actual_coact = coact_counts[(i, j)] / total_tokens

    # Expected co-activation from calibration curve
    if expected_coact_fn is not None:
        expected_coact = float(expected_coact_fn(cos_val))
        # Ensure positive and reasonable
        expected_coact = max(expected_coact, 1e-8)
    else:
        # Fallback: use independent rate scaled by cos_sim
        expected_coact = freq_i * freq_j * (1 + cos_val)
        expected_coact = max(expected_coact, 1e-8)

    # Mismatch: how much less co-activation than expected?
    # Positive when actual < expected (suppression)
    mismatch = (expected_coact - actual_coact) / expected_coact
    mismatch = max(mismatch, 0)  # Only care about suppression, not excess

    # GAS(i -> j): i absorbs j (i is child, j is parent)
    freq_ratio_ij = freq_i / freq_j
    gas_ij = cos_val * mismatch * freq_ratio_ij

    # GAS(j -> i): j absorbs i
    freq_ratio_ji = freq_j / freq_i
    gas_ji = cos_val * mismatch * freq_ratio_ji

    # Record both directions
    gas_scores.append({
        "feature_i": i,
        "feature_j": j,
        "cos_sim": round(cos_val, 6),
        "freq_i": round(float(freq_i), 8),
        "freq_j": round(float(freq_j), 8),
        "actual_coact": round(float(actual_coact), 8),
        "expected_coact": round(float(expected_coact), 8),
        "mismatch": round(float(mismatch), 6),
        "gas_i_absorbs_j": round(float(gas_ij), 6),
        "gas_j_absorbs_i": round(float(gas_ji), 6),
    })

    # Update per-feature vulnerability (as absorption TARGET)
    per_feature_gas[j] = max(per_feature_gas[j], gas_ij)
    per_feature_gas[i] = max(per_feature_gas[i], gas_ji)

gas_time = time.time() - t0
n_computed = len(gas_scores)
print(f"  GAS computed for {n_computed} pairs in {gas_time:.1f}s")

# GAS distribution statistics
all_gas_ij = [s["gas_i_absorbs_j"] for s in gas_scores]
all_gas_ji = [s["gas_j_absorbs_i"] for s in gas_scores]
all_gas = all_gas_ij + all_gas_ji
all_gas_arr = np.array(all_gas)

gas_dist_stats = {
    "n_gas_scores": len(all_gas),
    "mean": float(np.mean(all_gas_arr)),
    "std": float(np.std(all_gas_arr)),
    "min": float(np.min(all_gas_arr)),
    "max": float(np.max(all_gas_arr)),
    "median": float(np.median(all_gas_arr)),
    "p90": float(np.percentile(all_gas_arr, 90)),
    "p95": float(np.percentile(all_gas_arr, 95)),
    "p99": float(np.percentile(all_gas_arr, 99)),
    "n_above_0.01": int((all_gas_arr > 0.01).sum()),
    "n_above_0.05": int((all_gas_arr > 0.05).sum()),
    "n_above_0.1": int((all_gas_arr > 0.1).sum()),
    "n_above_0.5": int((all_gas_arr > 0.5).sum()),
    "n_zero": int((all_gas_arr == 0).sum()),
}
print(f"\n  GAS distribution stats:")
print(f"    Mean: {gas_dist_stats['mean']:.6f}")
print(f"    Std:  {gas_dist_stats['std']:.6f}")
print(f"    Max:  {gas_dist_stats['max']:.6f}")
print(f"    Median: {gas_dist_stats['median']:.6f}")
print(f"    P90: {gas_dist_stats['p90']:.6f}")
print(f"    P95: {gas_dist_stats['p95']:.6f}")
print(f"    P99: {gas_dist_stats['p99']:.6f}")
print(f"    Scores > 0.01: {gas_dist_stats['n_above_0.01']}")
print(f"    Scores > 0.05: {gas_dist_stats['n_above_0.05']}")
print(f"    Scores > 0.1: {gas_dist_stats['n_above_0.1']}")
print(f"    Scores > 0.5: {gas_dist_stats['n_above_0.5']}")
print(f"    Zeros: {gas_dist_stats['n_zero']}")

# Per-feature vulnerability stats
vulnerable_features = per_feature_gas[per_feature_gas > 0]
per_feature_stats = {
    "n_vulnerable_features": int((per_feature_gas > 0).sum()),
    "mean_vulnerability": float(np.mean(vulnerable_features)) if len(vulnerable_features) > 0 else 0,
    "max_vulnerability": float(np.max(per_feature_gas)),
    "p90_vulnerability": float(np.percentile(vulnerable_features, 90)) if len(vulnerable_features) > 0 else 0,
    "p99_vulnerability": float(np.percentile(vulnerable_features, 99)) if len(vulnerable_features) > 0 else 0,
}
print(f"\n  Per-feature vulnerability stats:")
print(f"    Vulnerable features: {per_feature_stats['n_vulnerable_features']}/{n_features}")
print(f"    Mean vulnerability: {per_feature_stats['mean_vulnerability']:.6f}")
print(f"    Max vulnerability: {per_feature_stats['max_vulnerability']:.6f}")

# Top 20 most vulnerable features
top_vulnerable_indices = np.argsort(per_feature_gas)[::-1][:20]
top_vulnerable = []
for rank, feat_idx in enumerate(top_vulnerable_indices):
    if per_feature_gas[feat_idx] == 0:
        break
    top_vulnerable.append({
        "rank": rank + 1,
        "feature_idx": int(feat_idx),
        "gas_vulnerability": round(float(per_feature_gas[feat_idx]), 6),
        "activation_rate": round(float(feature_freq_rate[feat_idx]), 6),
    })
print(f"\n  Top 10 most absorption-vulnerable features:")
for entry in top_vulnerable[:10]:
    print(f"    Rank {entry['rank']}: feature {entry['feature_idx']} "
          f"(GAS={entry['gas_vulnerability']:.4f}, rate={entry['activation_rate']:.4f})")

# Top 20 pairs by GAS score
top_pairs = sorted(gas_scores, key=lambda x: max(x["gas_i_absorbs_j"], x["gas_j_absorbs_i"]),
                   reverse=True)[:20]
print(f"\n  Top 10 pairs by GAS score:")
for rank, pair in enumerate(top_pairs[:10]):
    max_gas = max(pair["gas_i_absorbs_j"], pair["gas_j_absorbs_i"])
    direction = "i->j" if pair["gas_i_absorbs_j"] > pair["gas_j_absorbs_i"] else "j->i"
    print(f"    Rank {rank+1}: ({pair['feature_i']}, {pair['feature_j']}) "
          f"cos={pair['cos_sim']:.4f}, GAS={max_gas:.4f} ({direction}), "
          f"mismatch={pair['mismatch']:.4f}")


# ── Step 6: Pilot pass/fail assessment ────────────────────────────────────────

report_progress(TASK_ID, RESULTS_DIR, 6, 6, step=5, total_steps=5,
                metric={"phase": "assessment"})

print(f"\n[{datetime.now().isoformat()}] Step 6: Pilot assessment...")

pilot_assessment = {
    "criterion_1_n_pairs": {
        "target": f">= {MIN_PAIRS_TARGET} high-similarity pairs",
        "actual": n_high_sim_pairs,
        "pass": n_high_sim_pairs >= MIN_PAIRS_TARGET,
    },
    "criterion_2_n_sequences": {
        "target": ">= 100 sequences for co-activation",
        "actual": n_seqs,
        "pass": n_seqs >= 100,
    },
    "criterion_3_nondegenerate": {
        "target": "variance > 0 and some scores > 0.1",
        "variance": gas_dist_stats["std"] ** 2,
        "n_above_0_1": gas_dist_stats["n_above_0.1"],
        "pass": gas_dist_stats["std"] > 0 and gas_dist_stats["n_above_0.1"] > 0,
    },
    "criterion_4_freq_asymmetry": {
        "target": "frequency asymmetry term computed",
        "n_pairs_with_freq_ratio": n_computed,
        "pass": n_computed > 0,
    },
}

all_pass = all(c["pass"] for c in pilot_assessment.values())
print(f"\n  Pilot Assessment:")
for name, crit in pilot_assessment.items():
    status = "PASS" if crit["pass"] else "FAIL"
    print(f"    {name}: {status}")

overall = "GO" if all_pass else "NO_GO"
print(f"\n  Overall: {overall}")

# If variance is 0 but we have scores, this might mean the mismatch is
# uniformly zero — diagnose
if gas_dist_stats["std"] == 0 and gas_dist_stats["n_above_0.1"] == 0:
    # Check if all mismatches are zero
    n_nonzero_mismatch = sum(1 for s in gas_scores if s["mismatch"] > 0)
    print(f"\n  DIAGNOSTIC: {n_nonzero_mismatch}/{n_computed} pairs have nonzero mismatch")
    if n_nonzero_mismatch == 0:
        print("  All co-activation rates equal or exceed expected — no suppression detected in pilot.")
        print("  This could mean: (a) 200 sequences insufficient for rare pair co-activation,")
        print("  (b) threshold too high excludes informative pairs, (c) GAS concept needs revision.")


# ── Save results ──────────────────────────────────────────────────────────────

total_time = time.time() - start_time
print(f"\n[{datetime.now().isoformat()}] Saving results...")

# Sample gas_scores for output (full set can be very large)
gas_scores_sample = sorted(gas_scores,
                           key=lambda x: max(x["gas_i_absorbs_j"], x["gas_j_absorbs_i"]),
                           reverse=True)[:500]

results = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "sae_config": {
        "release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "n_features": n_features,
        "d_model": d_model,
    },
    "pilot_settings": {
        "n_sequences": n_seqs,
        "max_seq_len": MAX_SEQ_LEN,
        "cos_threshold": COS_THRESHOLD,
        "text_source": text_source,
    },
    "cosine_similarity_stats": cos_sim_stats,
    "activation_frequency_stats": freq_stats,
    "calibration_curve": calibration_curve,
    "gas_distribution_stats": gas_dist_stats,
    "per_feature_vulnerability_stats": per_feature_stats,
    "top_vulnerable_features": top_vulnerable,
    "top_pairs_by_gas": [{
        "rank": rank + 1,
        **pair,
    } for rank, pair in enumerate(top_pairs[:20])],
    "gas_scores_sample": gas_scores_sample,
    "pilot_assessment": pilot_assessment,
    "overall_verdict": overall,
    "timing": {
        "sae_load_sec": round(sae_load_time, 1),
        "cosine_similarity_sec": round(cos_sim_time, 1),
        "activation_collection_sec": round(activation_time, 1),
        "calibration_sec": round(cal_time, 1),
        "gas_computation_sec": round(gas_time, 1),
        "total_sec": round(total_time, 1),
    },
}

# Save main results
output_path = PHASE2_DIR / "gas_computation.json"
output_path.write_text(json.dumps(results, indent=2))
print(f"  Saved: {output_path}")

# Save per-feature GAS vulnerability scores (compact binary)
gas_vuln_path = PHASE2_DIR / "per_feature_gas_vulnerability.npz"
np.savez_compressed(
    gas_vuln_path,
    per_feature_gas=per_feature_gas,
    feature_freq_rate=feature_freq_rate,
)
print(f"  Saved: {gas_vuln_path}")

# Save summary markdown
summary_md = f"""# Phase 2.1: GAS Computation — PILOT Results

## Configuration
- SAE: {SAE_RELEASE} / {SAE_ID}
- Features: {n_features}, Model dim: {d_model}
- Sequences: {n_seqs} x {MAX_SEQ_LEN} tokens = {total_tokens:,} tokens
- Cosine threshold: {COS_THRESHOLD}
- Text source: {text_source}

## Key Results

### Decoder Cosine Similarity
- High-similarity pairs (cos > {COS_THRESHOLD}): **{n_high_sim_pairs:,}**
- Mean cosine: {cos_sim_stats['mean']:.4f}, Max: {cos_sim_stats['max']:.4f}

### Activation Statistics
- Active features: {n_active_features}/{n_features} ({100*n_active_features/n_features:.1f}%)
- Dead features: {n_dead_features} ({100*n_dead_features/n_features:.1f}%)
- Mean activation rate: {freq_stats['mean_activation_rate']:.4f}

### GAS Distribution
- Total GAS scores computed: {gas_dist_stats['n_gas_scores']:,}
- Mean: {gas_dist_stats['mean']:.6f}, Std: {gas_dist_stats['std']:.6f}
- Max: {gas_dist_stats['max']:.6f}, Median: {gas_dist_stats['median']:.6f}
- Scores > 0.1: {gas_dist_stats['n_above_0.1']:,}
- Scores > 0.05: {gas_dist_stats['n_above_0.05']:,}

### Per-Feature Vulnerability
- Features with GAS > 0: {per_feature_stats['n_vulnerable_features']}
- Max vulnerability: {per_feature_stats['max_vulnerability']:.4f}
- Mean vulnerability: {per_feature_stats['mean_vulnerability']:.6f}

### Calibration Curve
| Cos Bin | N Pairs | Actual Co-act | Independent | Ratio |
|---------|---------|---------------|-------------|-------|
"""
for entry in calibration_curve:
    summary_md += (f"| [{entry['cos_bin_start']:.2f}, {entry['cos_bin_end']:.2f}) | "
                   f"{entry['n_pairs']} | {entry['mean_actual_coact']:.6f} | "
                   f"{entry['mean_independent_coact']:.6f} | {entry['coact_ratio']:.2f} |\n")

summary_md += f"""
### Top 10 Most Vulnerable Features
| Rank | Feature | GAS Vulnerability | Activation Rate |
|------|---------|-------------------|-----------------|
"""
for entry in top_vulnerable[:10]:
    summary_md += (f"| {entry['rank']} | {entry['feature_idx']} | "
                   f"{entry['gas_vulnerability']:.4f} | {entry['activation_rate']:.4f} |\n")

summary_md += f"""
## Pilot Assessment
- Pairs >= {MIN_PAIRS_TARGET}: **{'PASS' if pilot_assessment['criterion_1_n_pairs']['pass'] else 'FAIL'}** ({n_high_sim_pairs:,} pairs)
- Sequences >= 100: **{'PASS' if pilot_assessment['criterion_2_n_sequences']['pass'] else 'FAIL'}** ({n_seqs} sequences)
- Non-degenerate: **{'PASS' if pilot_assessment['criterion_3_nondegenerate']['pass'] else 'FAIL'}** (var={gas_dist_stats['std']**2:.6f}, n>0.1={gas_dist_stats['n_above_0.1']})
- Frequency asymmetry: **{'PASS' if pilot_assessment['criterion_4_freq_asymmetry']['pass'] else 'FAIL'}** ({n_computed} pairs)

## Overall Verdict: **{overall}**

## Timing
- SAE load: {sae_load_time:.1f}s
- Cosine similarity: {cos_sim_time:.1f}s
- Activation collection: {activation_time:.1f}s
- Calibration: {cal_time:.1f}s
- GAS computation: {gas_time:.1f}s
- **Total: {total_time:.1f}s** ({total_time/60:.1f} min)
"""

summary_path = PHASE2_DIR / "gas_computation_summary.md"
summary_path.write_text(summary_md)
print(f"  Saved: {summary_path}")

# ── DONE marker ───────────────────────────────────────────────────────────────

mark_task_done(
    TASK_ID, RESULTS_DIR,
    status="success" if all_pass else "partial",
    summary=f"GAS pilot: {n_high_sim_pairs} pairs, {n_computed} GAS scores, "
            f"verdict={overall}, max_GAS={gas_dist_stats['max']:.4f}",
)

print(f"\n[{datetime.now().isoformat()}] Phase 2.1 GAS Computation — PILOT complete.")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Verdict: {overall}")
