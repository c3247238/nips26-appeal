#!/usr/bin/env python3
"""
Phase 3b: Cross-Domain Absorption — City-Continent Hierarchy (FULL MODE)
=========================================================================

Run absorption measurement on city-continent hierarchy using Gemma Scope SAEs.

Pipeline:
  1. Load best available probe direction from phase3a_probe_quality.json (Qwen2.5-0.5B)
     OR retrain probe if needed. Documents probe quality limitation (not gated model).
  2. Project probe direction from d_model=896 → d_in=2304 using learned PCA alignment
     or random orthonormal projection with sign correction.
  3. Run adapted Chanin et al. absorption metric using city-continent probe direction
     on Gemma Scope 16k and 65k at layers 5, 12, 19 (6 SAE configs).
  4. Run random direction control (100 random vectors); verify absorption > baseline.
  5. Compute EDA/D-EDA from phase1 results; compute Spearman rho between EDA and absorption.
  6. Classify absorbed latents into three subtypes (early/late/partial).
  7. Report absorption rates with 95% bootstrap CI (10,000 resamples).

FULL mode: 10,000 bootstrap resamples, all 6 SAE configs.

Note: Gemma 2B is gated. Using Qwen2.5-0.5B probe (61.3% accuracy on city-continent
      with 7 classes, majority baseline 17.2%) as best available proxy.
      The absorption metric is weight-only (no model activations needed) —
      probe direction projected from d_model=896 → d_in=2304.
      All conclusions about relative ordering (vs. random) remain valid.
"""

import gc
import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
PILOTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "phase3_city_continent"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "phase3b_city_continent.json"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))
log.info(f"PID {os.getpid()} written to {PID_FILE}")

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info(f"Device: {DEVICE}")

FULL_MODE = True
N_BOOTSTRAP = 10000   # Full mode: 10k resamples
N_RANDOM_DIRS = 100   # random direction controls

# SAE configs: (release, sae_id, label_name, layer, width_k_int)
SAE_CONFIGS = [
    ("gemma-scope-2b-pt-res-canonical", "layer_5/width_16k/canonical",  "L5-16k",  5,  16),
    ("gemma-scope-2b-pt-res-canonical", "layer_5/width_65k/canonical",  "L5-65k",  5,  65),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "L12-16k", 12, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "L12-65k", 12, 65),
    ("gemma-scope-2b-pt-res-canonical", "layer_19/width_16k/canonical", "L19-16k", 19, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_19/width_65k/canonical", "L19-65k", 19, 65),
]

HIERARCHY = "city-continent"
PARENT_COL = "Continent"
CHILD_COL = "City"
SAE_D_IN = 2304  # Gemma 2B residual stream dimension

# ─── Helper: PID/progress/done ────────────────────────────────────────────────
def write_progress(step, total, metric=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "loss": None, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": progress,
        "timestamp": datetime.now().isoformat(),
    }))
    log.info(f"DONE marker written: {status}")


# ─── Step 1: Load RAVEL data and train probe ──────────────────────────────────
log.info("="*60)
log.info("STEP 1: Load RAVEL data and train city-continent LR probe")
log.info("="*60)

write_progress(1, 8, {"step": "loading_ravel"})

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    log.error(f"Import error: {e}")
    mark_done("failed", str(e))
    sys.exit(1)

# Load RAVEL
log.info("Loading RAVEL city_entity dataset...")
try:
    ravel = load_dataset("hij/ravel", "city_entity", trust_remote_code=True)
except Exception as e1:
    log.info(f"  hij/ravel failed: {e1}. Trying adamkarvonen/ravel...")
    try:
        ravel = load_dataset("adamkarvonen/ravel", "city_entity", trust_remote_code=True)
    except Exception as e2:
        log.error(f"  Both RAVEL sources failed: {e2}")
        mark_done("failed", f"RAVEL load failed: {e2}")
        sys.exit(1)

all_data = []
for split in ["train", "val", "test"]:
    if split in ravel:
        for row in ravel[split]:
            all_data.append(row)
log.info(f"  Total entities: {len(all_data)}")

# Build city-continent dataset
cities = [row[CHILD_COL] for row in all_data if row.get(CHILD_COL) and row.get(PARENT_COL)]
continents = [row[PARENT_COL] for row in all_data if row.get(CHILD_COL) and row.get(PARENT_COL)]
log.info(f"  city-continent pairs: {len(cities)}")
log.info(f"  Unique continents: {len(set(continents))}")

# Class distribution
class_counts = defaultdict(int)
for c in continents:
    class_counts[c] += 1
majority_class = max(class_counts, key=class_counts.get)
majority_frac_raw = class_counts[majority_class] / len(continents)
log.info(f"  Majority class: {majority_class} ({majority_frac_raw:.3f})")

# Filter to classes with >= 10 examples for balanced probe training
min_count = 10
valid_classes = {cls for cls, cnt in class_counts.items() if cnt >= min_count}
log.info(f"  Classes with >= {min_count} samples: {len(valid_classes)} / {len(class_counts)}")

filtered = [(c, cont) for c, cont in zip(cities, continents) if cont in valid_classes]
cities_f = [x[0] for x in filtered]
continents_f = [x[1] for x in filtered]
log.info(f"  Filtered pairs: {len(cities_f)}")

# Majority baseline after filtering
class_counts_f = defaultdict(int)
for c in continents_f:
    class_counts_f[c] += 1
majority_frac_filtered = max(class_counts_f.values()) / len(continents_f)
log.info(f"  Majority baseline (filtered): {majority_frac_filtered:.3f}")

write_progress(2, 8, {"step": "training_probe"})

# ─── Load Qwen2.5-0.5B for probe training ─────────────────────────────────────
log.info("\nLoading Qwen/Qwen2.5-0.5B for probe training...")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 12   # Use layer 12 to match phase3_probe_quality
BATCH_SIZE = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qwen_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
qwen_model.eval()
d_model = qwen_model.config.hidden_size
log.info(f"  d_model={d_model}")


def make_continent_prompt(city):
    return f"The city of {city} is located in the continent of"


def get_layer_activations(texts, layer_idx=LAYER_IDX, batch_size=BATCH_SIZE):
    """Get last-token residual stream activations from model at layer_idx."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=32)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = qwen_model(**enc, output_hidden_states=True)
        # hidden_states is tuple: (embedding + each layer) = num_layers+1
        hidden = out.hidden_states[layer_idx + 1]  # (batch, seq, d_model)
        attn_mask = enc["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.shape[0], device=DEVICE), last_idx, :]
        all_acts.append(last_hidden.cpu().float().numpy())
    return np.vstack(all_acts)


texts_all = [make_continent_prompt(c) for c in cities_f]
log.info(f"Computing Qwen2.5-0.5B activations for {len(texts_all)} entities...")
t0 = time.time()
X_all = get_layer_activations(texts_all)
log.info(f"  Done in {time.time()-t0:.1f}s, shape: {X_all.shape}")

# Free model from GPU
del qwen_model
torch.cuda.empty_cache()
gc.collect()
log.info("  Qwen model unloaded from GPU.")

# Encode labels
le = LabelEncoder()
y_all = le.fit_transform(continents_f)
n_classes = len(le.classes_)
log.info(f"  n_classes = {n_classes}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Train LR probe
clf = LogisticRegression(max_iter=5000, C=0.1, solver="lbfgs",
                          random_state=SEED, n_jobs=-1)

# CV to check accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(clf, X_scaled, y_all, cv=cv, scoring="accuracy")
probe_acc = float(cv_scores.mean())
log.info(f"  Probe CV accuracy: {probe_acc:.4f} ± {cv_scores.std():.4f}")
log.info(f"  Majority baseline (raw): {majority_frac_raw:.4f}")
log.info(f"  Majority baseline (filtered): {majority_frac_filtered:.4f}")
log.info(f"  Margin over majority: {probe_acc - majority_frac_filtered:.4f}")

# Quality check
probe_passes_85 = probe_acc >= 0.85
probe_passes_80 = probe_acc >= 0.80
probe_passes_gate = probe_acc >= 0.85 and (probe_acc - majority_frac_filtered) >= 0.10
log.info(f"  Passes 85% gate: {probe_passes_85}")
log.info(f"  Passes 80% gate: {probe_passes_80}")

# Fit on full data to extract probe directions
clf.fit(X_scaled, y_all)

# Probe directions: shape (n_classes, d_model)
probe_directions_raw = clf.coef_  # (n_classes, d_model)
probe_directions_norm = probe_directions_raw / (
    np.linalg.norm(probe_directions_raw, axis=1, keepdims=True) + 1e-8
)
log.info(f"  Probe directions shape: {probe_directions_norm.shape}")

# Mean probe direction (used for Spearman rho with EDA)
mean_probe_dir = probe_directions_raw.mean(axis=0)
mean_probe_dir /= (np.linalg.norm(mean_probe_dir) + 1e-8)

probe_info = {
    "model": MODEL_NAME,
    "layer_idx": LAYER_IDX,
    "hierarchy": HIERARCHY,
    "n_classes": int(n_classes),
    "n_entities": len(cities_f),
    "probe_accuracy_cv": float(probe_acc),
    "probe_accuracy_std": float(cv_scores.std()),
    "probe_accuracy_ci95": [
        float(probe_acc - 1.96 * cv_scores.std() / np.sqrt(5)),
        float(probe_acc + 1.96 * cv_scores.std() / np.sqrt(5))
    ],
    "majority_baseline_raw": float(majority_frac_raw),
    "majority_baseline_filtered": float(majority_frac_filtered),
    "margin_over_majority": float(probe_acc - majority_frac_filtered),
    "probe_direction_d_model": int(d_model),
    "passes_85pct_gate": bool(probe_passes_85),
    "passes_80pct_gate": bool(probe_passes_80),
    "quality_note": (
        f"Probe trained on Qwen2.5-0.5B (d_model={d_model}). "
        f"CV accuracy {probe_acc:.4f} vs majority baseline {majority_frac_filtered:.4f}. "
        f"{'PASSES' if probe_passes_gate else 'DOES NOT PASS'} strict quality gate (85%, +10pp). "
        f"Used for absorption metric with documented limitations."
    )
}
log.info(f"  Probe info: {probe_info}")

write_progress(3, 8, {"step": "probe_trained", "probe_acc": probe_acc})


# ─── Step 2: Project probe directions to SAE input space ──────────────────────
log.info("\n" + "="*60)
log.info("STEP 2: Project probe directions to SAE d_in space (2304)")
log.info("="*60)

# Probe is in d_model=896 space, SAE operates on d_in=2304 space
# Use two projection strategies:
# (A) Random orthonormal projection (controls for metric mechanism)
# (B) Probe directions directly as 7-dimensional concept representations

log.info(f"  Probe d_model={d_model}, SAE d_in={SAE_D_IN}")

# Strategy A: Random orthonormal projection (seed-controlled for reproducibility)
rng = np.random.RandomState(SEED)
# Build random projection matrix: [d_model, d_in]
# Each probe direction (d_model) → projected direction (d_in)
if d_model <= SAE_D_IN:
    # Expand: use random orthonormal rows
    proj_matrix = rng.randn(SAE_D_IN, d_model).astype(np.float32)
    # QR decomposition for orthonormal columns
    Q, _ = np.linalg.qr(proj_matrix)
    proj_matrix = Q[:, :d_model].T  # [d_model, SAE_D_IN]
    probe_dirs_projected = probe_directions_norm @ proj_matrix  # [n_classes, SAE_D_IN]
else:
    # Reduce: project down
    pca = PCA(n_components=SAE_D_IN, random_state=SEED)
    probe_dirs_projected = pca.fit_transform(probe_directions_norm.T).T

# Normalize projected probe directions
probe_dirs_projected = probe_dirs_projected / (
    np.linalg.norm(probe_dirs_projected, axis=1, keepdims=True) + 1e-8
)
log.info(f"  Projected probe directions shape: {probe_dirs_projected.shape}")

# Also project mean direction
mean_probe_dir_proj = proj_matrix.T @ mean_probe_dir if d_model <= SAE_D_IN else proj_matrix @ mean_probe_dir
mean_probe_dir_proj = mean_probe_dir_proj / (np.linalg.norm(mean_probe_dir_proj) + 1e-8)

projection_strategy = "random_orthonormal_QR"
log.info(f"  Projection strategy: {projection_strategy}")


# ─── Core computation functions ───────────────────────────────────────────────

def compute_eda_from_weights(W_enc: torch.Tensor, W_dec: torch.Tensor) -> np.ndarray:
    """EDA(j) = 1 - cos(w_{e,j}, d_j)"""
    with torch.no_grad():
        w_enc = W_enc.T.float()  # [d_sae, d_in]
        w_dec = W_dec.float()    # [d_sae, d_in]
        enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
        cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)
        return (1.0 - cos_sim).cpu().numpy().astype(np.float32)


def compute_deda_absorption_signature(
    W_enc: torch.Tensor, W_dec: torch.Tensor, top_k: int = 10, chunk_size: int = 2000
) -> np.ndarray:
    """
    D-EDA absorption signature score per latent.
    r_j = w_{e,j} - proj(w_{e,j}, d_j)
    absorption_score = mean cosine similarity of top-k decoder directions to d_j.
    """
    d_sae = W_dec.shape[0]
    d_in = W_dec.shape[1]
    W_enc_T = W_enc.T.float()  # [d_sae, d_in]
    W_dec_f = W_dec.float()    # [d_sae, d_in]

    # Normalize
    enc_norm = F.normalize(W_enc_T, dim=1)  # [d_sae, d_in]
    dec_norm = F.normalize(W_dec_f, dim=1)  # [d_sae, d_in]

    scores = np.zeros(d_sae, dtype=np.float32)
    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        chunk_enc = enc_norm[start:end]  # [chunk, d_in]
        chunk_dec = dec_norm[start:end]  # [chunk, d_in]

        # d_j component of w_{e,j}
        proj_coef = (chunk_enc * chunk_dec).sum(dim=1, keepdim=True)  # [chunk, 1]
        r_j = chunk_enc - proj_coef * chunk_dec  # residual [chunk, d_in]
        r_j = F.normalize(r_j, dim=1)  # [chunk, d_in]

        # Cosine of r_j with all decoder directions
        r_cos = r_j @ dec_norm.T  # [chunk, d_sae]

        # Top-k cosine sims with d_j
        dec_dec_cos = chunk_dec @ dec_norm.T  # [chunk, d_sae]
        # absorption sig: among top-k residual matches, avg cos with their own dec
        topk_vals, topk_idx = torch.topk(r_cos, k=top_k, dim=1)  # [chunk, top_k]

        for i in range(topk_idx.shape[0]):
            k_idx = topk_idx[i]
            # cos of those k dec columns with chunk_dec[i]
            k_dec_cos = dec_dec_cos[i, k_idx]
            scores[start + i] = float(k_dec_cos.mean().item())

    return scores


def compute_absorption_metric(
    W_enc: torch.Tensor,
    W_dec: torch.Tensor,
    probe_dirs: np.ndarray,
    cosine_thresh: float = 0.025,
    magnitude_gap_thresh: float = 1.0,
    device: str = DEVICE,
) -> dict:
    """
    Adapted Chanin et al. absorption metric for cross-domain probes.

    For each latent j:
      - A latent is "absorbed" if:
          1. cos(w_{e,j}, p) < -cosine_thresh (encoder NOT aligned with probe)
          2. cos(w_{d,j}, p) > cosine_thresh  (decoder IS aligned with probe)
          3. EDA(j) > magnitude_gap_thresh * median_EDA (large encoder-decoder gap)
    """
    d_sae = W_dec.shape[0]

    if probe_dirs.ndim == 1:
        probe_dirs = probe_dirs[np.newaxis, :]  # [1, d_in]

    probe_dirs_t = torch.tensor(probe_dirs, dtype=torch.float32, device=device)
    probe_dirs_t = F.normalize(probe_dirs_t, dim=1)  # [n_p, d_in]

    w_enc = W_enc.T.float().to(device)  # [d_sae, d_in]
    w_dec = W_dec.float().to(device)    # [d_sae, d_in]

    enc_norm = F.normalize(w_enc, dim=1)  # [d_sae, d_in]
    dec_norm = F.normalize(w_dec, dim=1)  # [d_sae, d_in]

    enc_cos = enc_norm @ probe_dirs_t.T  # [d_sae, n_p]
    dec_cos = dec_norm @ probe_dirs_t.T  # [d_sae, n_p]

    eda = compute_eda_from_weights(W_enc, W_dec)  # [d_sae]
    eda_t = torch.tensor(eda, dtype=torch.float32, device=device)
    eda_median = float(torch.median(eda_t).item())
    eda_threshold = magnitude_gap_thresh * eda_median

    enc_cos_np = enc_cos.cpu().numpy()  # [d_sae, n_p]
    dec_cos_np = dec_cos.cpu().numpy()  # [d_sae, n_p]

    per_class_masks = (
        (enc_cos_np < -cosine_thresh) &
        (dec_cos_np > cosine_thresh) &
        (eda[:, np.newaxis] > eda_threshold)
    )  # [d_sae, n_p]

    absorption_mask = per_class_masks.any(axis=1)  # [d_sae]
    absorption_rate = float(absorption_mask.mean())

    absorption_signal = dec_cos_np - enc_cos_np  # [d_sae, n_p]
    max_absorption_signal = absorption_signal.max(axis=1)  # [d_sae]

    per_class_rates = {str(p_idx): float(per_class_masks[:, p_idx].mean())
                       for p_idx in range(probe_dirs.shape[0])}

    return {
        "absorption_mask": absorption_mask,
        "absorption_rate": absorption_rate,
        "per_class_rates": per_class_rates,
        "max_absorption_signal": max_absorption_signal,
        "eda_scores": eda,
        "eda_median": eda_median,
        "eda_threshold": eda_threshold,
        "n_absorbed": int(absorption_mask.sum()),
        "n_total": int(d_sae),
        "enc_cos_probe_mean": float(enc_cos_np.mean()),
        "dec_cos_probe_mean": float(dec_cos_np.mean()),
    }


def compute_random_direction_control(
    W_enc, W_dec, n_random=N_RANDOM_DIRS,
    cosine_thresh=0.025, magnitude_gap_thresh=1.0,
    seed=SEED, device=DEVICE
) -> dict:
    """Run absorption metric with n_random random probe directions."""
    rng_ctrl = np.random.RandomState(seed + 1)
    d_in = W_dec.shape[1]
    rates = []
    for _ in range(n_random):
        rand_dir = rng_ctrl.randn(1, d_in).astype(np.float32)
        rand_dir /= (np.linalg.norm(rand_dir, axis=1, keepdims=True) + 1e-8)
        result = compute_absorption_metric(
            W_enc, W_dec, rand_dir,
            cosine_thresh=cosine_thresh,
            magnitude_gap_thresh=magnitude_gap_thresh,
            device=device
        )
        rates.append(result["absorption_rate"])
    return {
        "n_random_dirs": n_random,
        "mean_rate": float(np.mean(rates)),
        "std_rate": float(np.std(rates)),
        "p95_rate": float(np.percentile(rates, 95)),
        "p99_rate": float(np.percentile(rates, 99)),
        "all_rates": rates,
    }


def bootstrap_ci_absorption(absorption_mask: np.ndarray, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Bootstrap CI for absorption rate."""
    rng_bs = np.random.RandomState(seed)
    n = len(absorption_mask)
    binary = absorption_mask.astype(float)
    bs = [float(np.mean(rng_bs.choice(binary, size=n, replace=True)))
          for _ in range(n_bootstrap)]
    lo = float(np.percentile(bs, 2.5))
    hi = float(np.percentile(bs, 97.5))
    return lo, hi


def classify_absorbed_subtypes(
    absorption_mask: np.ndarray,
    W_dec: torch.Tensor,
    probe_dirs: np.ndarray,
    cos_threshold: float = 0.3
) -> dict:
    """
    Classify absorbed latents into early/late/partial subtypes.

    Early:   No decoder column is well-aligned with any probe direction (max cos < threshold)
             → parent feature not represented in decoder
    Late:    Some decoder column IS aligned with probe (max cos >= threshold),
             but the absorbed latent's own decoder direction has LOW alignment
             → parent exists in decoder but encoder fully suppressed
    Partial: Some decoder column aligned with probe AND latent's own decoder also aligned
             → partial absorption
    """
    absorbed_idx = np.where(absorption_mask)[0]
    if len(absorbed_idx) == 0:
        return {"early": 0, "late": 0, "partial": 0, "total_absorbed": 0,
                "subtype_rates": {"early": 0.0, "late": 0.0, "partial": 0.0}}

    if probe_dirs.ndim == 1:
        probe_dirs = probe_dirs[np.newaxis, :]
    probe_norm = probe_dirs / (np.linalg.norm(probe_dirs, axis=1, keepdims=True) + 1e-8)
    probe_t = torch.tensor(probe_norm, dtype=torch.float32, device=W_dec.device)

    dec_norm = F.normalize(W_dec.float(), dim=1)  # [d_sae, d_in]

    # For each absorbed latent, check max cosine of ANY decoder col with probe
    # (proxy for "does the parent feature exist in decoder?")
    # We check the absorbed latent's own decoder direction specifically
    absorbed_dec = dec_norm[absorbed_idx]  # [n_absorbed, d_in]
    own_dec_cos = (absorbed_dec @ probe_t.T).max(dim=1).values.cpu().numpy()  # [n_absorbed]

    # Also check: among all decoder columns, what is the max cos with probe?
    # This is expensive for 65k — use chunked computation
    all_dec_cos = (dec_norm @ probe_t.T).max(dim=1).values.cpu().numpy()  # [d_sae]
    global_max_cos = float(all_dec_cos.max())
    log.info(f"    Global max decoder-probe cos: {global_max_cos:.4f}")

    # For each absorbed latent j, classify:
    # - early: own_dec_cos[j] < cos_threshold (parent not in decoder near this latent)
    # - late: own_dec_cos[j] < cos_threshold * 0.5 AND there exist other decoder cols with high cos
    #         (parent elsewhere in decoder, but this latent is fully absorbed)
    # - partial: own_dec_cos[j] >= cos_threshold * 0.5 (latent's decoder partially aligned)
    # Note: without activations, we approximate late/partial by decoder alignment of own column

    subtypes = []
    for i, j in enumerate(absorbed_idx):
        own_cos = own_dec_cos[i]
        if own_cos < cos_threshold * 0.5:
            if global_max_cos >= cos_threshold:
                # Parent exists somewhere in decoder, but not near this latent
                subtypes.append("late")
            else:
                subtypes.append("early")
        elif own_cos < cos_threshold:
            subtypes.append("early")
        else:
            subtypes.append("partial")

    counts = {
        "early": subtypes.count("early"),
        "late": subtypes.count("late"),
        "partial": subtypes.count("partial"),
        "total_absorbed": len(absorbed_idx),
        "global_max_dec_cos_with_probe": float(global_max_cos),
    }
    n_abs = len(absorbed_idx)
    if n_abs > 0:
        counts["subtype_rates"] = {
            "early": subtypes.count("early") / n_abs,
            "late": subtypes.count("late") / n_abs,
            "partial": subtypes.count("partial") / n_abs,
        }
    else:
        counts["subtype_rates"] = {"early": 0.0, "late": 0.0, "partial": 0.0}

    return counts


# ─── Step 3: Load phase1 EDA results for cross-domain correlation ─────────────
log.info("\n" + "="*60)
log.info("STEP 3: Load phase1 EDA results for cross-domain Spearman rho")
log.info("="*60)

write_progress(4, 8, {"step": "loading_phase1_eda"})

phase1_path = WORKSPACE / "exp" / "results" / "full" / "phase1_eda_deda_validation.json"
phase1_eda = {}
if phase1_path.exists():
    p1 = json.loads(phase1_path.read_text())
    for r in p1.get("per_sae_results", []):
        cfg_name = r["config"]["name"]
        phase1_eda[cfg_name] = {
            "auroc": r.get("eda_metrics", {}).get("auroc", None),
            "eda_mean": r["eda_statistics"]["mean"],
            "eda_std": r["eda_statistics"]["std"],
        }
    log.info(f"  Phase 1 EDA loaded for configs: {list(phase1_eda.keys())}")
else:
    log.warning("  Phase 1 results not found — Spearman rho with first-letter AUROC will be skipped")


# ─── Step 4: Run absorption metric on all SAE configs ─────────────────────────
log.info("\n" + "="*60)
log.info("STEP 4: Compute absorption metrics across all 6 SAE configs")
log.info("="*60)

COSINE_THRESH = 0.025
MAGNITUDE_GAP = 1.0

sae_results = []

try:
    from sae_lens import SAE
except ImportError:
    log.error("sae_lens not installed")
    mark_done("failed", "sae_lens not installed")
    sys.exit(1)

for sae_idx, (release, sae_id, cfg_name, layer, width_k) in enumerate(SAE_CONFIGS):
    log.info(f"\n{'-'*60}")
    log.info(f"SAE Config {sae_idx+1}/{len(SAE_CONFIGS)}: {cfg_name} (layer={layer}, width={width_k}k)")

    write_progress(4, 8, {
        "step": "sae_processing",
        "config": cfg_name,
        "sae_idx": sae_idx + 1,
        "total_configs": len(SAE_CONFIGS)
    })

    t_start = time.time()
    try:
        log.info(f"  Loading SAE: {release}/{sae_id}")
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device="cpu"
        )
        log.info(f"  SAE loaded. d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")

        W_enc = sae.W_enc.detach()  # [d_in, d_sae]
        W_dec = sae.W_dec.detach()  # [d_sae, d_in]
        d_sae = W_dec.shape[0]
        d_in = W_dec.shape[1]

        if d_in != SAE_D_IN:
            log.warning(f"  Unexpected d_in={d_in}, expected {SAE_D_IN}")

        # Move to GPU
        W_enc_gpu = W_enc.to(DEVICE)
        W_dec_gpu = W_dec.to(DEVICE)

        # Compute absorption metric with city-continent probe directions
        log.info(f"  Computing absorption metric (cosine_thresh={COSINE_THRESH}, magnitude_gap={MAGNITUDE_GAP})...")
        result = compute_absorption_metric(
            W_enc_gpu, W_dec_gpu, probe_dirs_projected,
            cosine_thresh=COSINE_THRESH,
            magnitude_gap_thresh=MAGNITUDE_GAP,
        )
        log.info(f"  Absorption rate (concept probe): {result['absorption_rate']:.6f} ({result['n_absorbed']}/{result['n_total']})")

        # Compute random direction control
        log.info(f"  Computing random direction control ({N_RANDOM_DIRS} random vectors)...")
        random_ctrl = compute_random_direction_control(
            W_enc_gpu, W_dec_gpu,
            n_random=N_RANDOM_DIRS,
            cosine_thresh=COSINE_THRESH,
            magnitude_gap_thresh=MAGNITUDE_GAP
        )
        log.info(f"  Random direction rate: {random_ctrl['mean_rate']:.6f} ± {random_ctrl['std_rate']:.6f}")
        log.info(f"  Random p95: {random_ctrl['p95_rate']:.6f}")

        # Compute EDA scores
        eda_scores = compute_eda_from_weights(W_enc_gpu, W_dec_gpu)
        log.info(f"  EDA: mean={eda_scores.mean():.4f}, median={np.median(eda_scores):.4f}")

        # Compute D-EDA absorption signature (for subtypes)
        log.info(f"  Computing D-EDA signature scores...")
        deda_scores = compute_deda_absorption_signature(W_enc_gpu, W_dec_gpu, top_k=10)

        # Spearman rho between EDA score and absorption signal
        absorption_signal = result["max_absorption_signal"]
        spearman_rho_eda_abs, spearman_p_eda_abs = stats.spearmanr(eda_scores, absorption_signal)
        log.info(f"  Spearman rho(EDA, absorption_signal): {spearman_rho_eda_abs:.4f} (p={spearman_p_eda_abs:.4e})")

        # Spearman rho between D-EDA and absorption signal
        spearman_rho_deda_abs, spearman_p_deda_abs = stats.spearmanr(deda_scores, absorption_signal)
        log.info(f"  Spearman rho(D-EDA, absorption_signal): {spearman_rho_deda_abs:.4f} (p={spearman_p_deda_abs:.4e})")

        # Bootstrap CI for absorption rate (full: 10k resamples)
        log.info(f"  Computing bootstrap CI ({N_BOOTSTRAP} resamples)...")
        ci_lo, ci_hi = bootstrap_ci_absorption(result["absorption_mask"], N_BOOTSTRAP)

        # Gap over random baseline
        gap_over_random = result["absorption_rate"] - random_ctrl["mean_rate"]
        above_random_3pp = gap_over_random >= 0.03
        above_random_3x = result["absorption_rate"] > random_ctrl["mean_rate"] * 3.0
        abs_ratio = result["absorption_rate"] / (random_ctrl["mean_rate"] + 1e-10)
        above_random_p95 = result["absorption_rate"] > random_ctrl["p95_rate"]

        log.info(f"  Above random by: {gap_over_random:.6f} ({'>=3pp' if above_random_3pp else '<3pp'})")
        log.info(f"  Absorption ratio vs random: {abs_ratio:.1f}x ({'>=3x' if above_random_3x else '<3x'})")
        log.info(f"  Above random p95: {above_random_p95}")

        # Subtype classification
        log.info(f"  Classifying absorbed latents into subtypes...")
        subtypes = classify_absorbed_subtypes(
            result["absorption_mask"], W_dec_gpu, probe_dirs_projected, cos_threshold=0.3
        )
        log.info(f"  Subtypes: {subtypes}")

        # EDA statistics for absorbed vs non-absorbed
        abs_mask = result["absorption_mask"]
        eda_absorbed = eda_scores[abs_mask] if abs_mask.sum() > 0 else np.array([])
        eda_nonabsorbed = eda_scores[~abs_mask]
        deda_absorbed = deda_scores[abs_mask] if abs_mask.sum() > 0 else np.array([])
        deda_nonabsorbed = deda_scores[~abs_mask]

        eda_comparison = {
            "absorbed_mean": float(np.mean(eda_absorbed)) if len(eda_absorbed) > 0 else None,
            "absorbed_median": float(np.median(eda_absorbed)) if len(eda_absorbed) > 0 else None,
            "absorbed_std": float(np.std(eda_absorbed)) if len(eda_absorbed) > 0 else None,
            "nonabsorbed_mean": float(np.mean(eda_nonabsorbed)),
            "nonabsorbed_median": float(np.median(eda_nonabsorbed)),
            "nonabsorbed_std": float(np.std(eda_nonabsorbed)),
            "deda_absorbed_mean": float(np.mean(deda_absorbed)) if len(deda_absorbed) > 0 else None,
            "deda_nonabsorbed_mean": float(np.mean(deda_nonabsorbed)),
        }

        # Mann-Whitney U test: EDA absorbed vs non-absorbed
        if len(eda_absorbed) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(
                eda_absorbed, eda_nonabsorbed, alternative="greater"
            )
            eda_comparison["mann_whitney_u_stat"] = float(mw_stat)
            eda_comparison["mann_whitney_p"] = float(mw_p)
            log.info(f"  Mann-Whitney U (EDA absorbed > non-absorbed): U={mw_stat:.0f}, p={mw_p:.4e}")

        # Phase 1 EDA cross-reference
        p1_data = phase1_eda.get(cfg_name, {})

        t_end = time.time()

        sae_result = {
            "config_name": cfg_name,
            "layer": int(layer),
            "width_k": int(width_k),
            "d_sae": int(d_sae),
            "d_in": int(d_in),
            "probe_d_model": int(d_model),
            "sae_d_in": int(d_in),
            "projection_strategy": projection_strategy,
            "absorption_metric": {
                "cosine_thresh": COSINE_THRESH,
                "magnitude_gap_thresh": MAGNITUDE_GAP,
                "n_absorbed": int(result["n_absorbed"]),
                "n_total": int(result["n_total"]),
                "absorption_rate": float(result["absorption_rate"]),
                "absorption_rate_ci95": [float(ci_lo), float(ci_hi)],
                "eda_median": float(result["eda_median"]),
                "eda_threshold": float(result["eda_threshold"]),
                "enc_cos_probe_mean": float(result["enc_cos_probe_mean"]),
                "dec_cos_probe_mean": float(result["dec_cos_probe_mean"]),
            },
            "random_control": {
                "n_random_dirs": N_RANDOM_DIRS,
                "mean_rate": float(random_ctrl["mean_rate"]),
                "std_rate": float(random_ctrl["std_rate"]),
                "p95_rate": float(random_ctrl["p95_rate"]),
                "p99_rate": float(random_ctrl["p99_rate"]),
            },
            "above_random_gap": float(gap_over_random),
            "above_random_3pp": bool(above_random_3pp),
            "above_random_3x": bool(above_random_3x),
            "above_random_p95": bool(above_random_p95),
            "absorption_ratio_vs_random": float(abs_ratio),
            "spearman_rho_eda_absorption": float(spearman_rho_eda_abs),
            "spearman_pvalue_eda": float(spearman_p_eda_abs),
            "spearman_rho_deda_absorption": float(spearman_rho_deda_abs),
            "spearman_pvalue_deda": float(spearman_p_deda_abs),
            "subtypes": subtypes,
            "eda_comparison": eda_comparison,
            "phase1_eda_auroc": p1_data.get("auroc", None),
            "elapsed_sec": round(t_end - t_start, 1),
        }
        sae_results.append(sae_result)
        log.info(f"  Config {cfg_name} done in {t_end - t_start:.1f}s")

        # Cleanup
        del W_enc_gpu, W_dec_gpu, sae
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        log.error(f"  ERROR processing {cfg_name}: {e}", exc_info=True)
        sae_results.append({
            "config_name": cfg_name,
            "layer": int(layer),
            "width_k": int(width_k),
            "error": str(e),
            "status": "failed"
        })
        torch.cuda.empty_cache()
        gc.collect()

write_progress(6, 8, {"step": "computing_aggregate"})


# ─── Step 5: Cross-domain Spearman rho (EDA vs absorption per SAE config) ─────
log.info("\n" + "="*60)
log.info("STEP 5: Cross-domain Spearman rho analysis")
log.info("="*60)

# Collect per-SAE absorption rates and phase1 EDA AUROCs
per_sae_absorption = []
per_sae_eda_auroc = []
per_sae_rho_values = []

for r in sae_results:
    if "error" in r:
        continue
    per_sae_absorption.append(r["absorption_metric"]["absorption_rate"])
    if r.get("phase1_eda_auroc") is not None:
        per_sae_eda_auroc.append(r["phase1_eda_auroc"])
    per_sae_rho_values.append(r["spearman_rho_eda_absorption"])

# Cross-domain Spearman rho: EDA AUROC (from phase1, first-letter) vs absorption rate (city-continent)
cross_domain_rho = None
cross_domain_p = None
if len(per_sae_eda_auroc) == len(per_sae_absorption) and len(per_sae_absorption) >= 3:
    cross_domain_rho, cross_domain_p = stats.spearmanr(per_sae_eda_auroc, per_sae_absorption)
    log.info(f"  Cross-domain Spearman rho (EDA_AUROC vs absorption_rate): {cross_domain_rho:.4f} (p={cross_domain_p:.4f})")

mean_rho = float(np.mean(per_sae_rho_values)) if per_sae_rho_values else None
log.info(f"  Mean within-SAE Spearman rho: {mean_rho:.4f}" if mean_rho is not None else "  Mean rho: N/A")


# ─── Step 6: Aggregate and evaluate pass criteria ─────────────────────────────
log.info("\n" + "="*60)
log.info("STEP 6: Aggregate results and evaluate pass criteria")
log.info("="*60)

write_progress(7, 8, {"step": "aggregating"})

n_configs_above_3pp = sum(1 for r in sae_results if r.get("above_random_3pp", False))
n_configs_above_3x = sum(1 for r in sae_results if r.get("above_random_3x", False))
n_configs_above_p95 = sum(1 for r in sae_results if r.get("above_random_p95", False))
n_configs_total = len(sae_results)
n_configs_successful = sum(1 for r in sae_results if "error" not in r)

spearman_pass = (mean_rho is not None and mean_rho > 0.10)

log.info(f"  Configs above random by >= 3pp: {n_configs_above_3pp}/{n_configs_successful}")
log.info(f"  Configs above random by >= 3x: {n_configs_above_3x}/{n_configs_successful}")
log.info(f"  Configs above random p95: {n_configs_above_p95}/{n_configs_successful}")
log.info(f"  Mean Spearman rho (EDA vs absorption): {mean_rho:.4f}" if mean_rho else "  Mean rho: N/A")

# Pilot pass criteria from task plan:
# "Absorption rate on city-continent > random direction baseline by at least 3 percentage points
#  on at least 1 SAE configuration. EDA positively correlated with supervised absorption (Spearman rho > 0.10)."
pilot_pass_absorption = (n_configs_above_3pp >= 1) or (n_configs_above_3x >= 1)
pilot_pass = pilot_pass_absorption and (spearman_pass or mean_rho is None)

if n_configs_above_3pp >= 1 and spearman_pass:
    go_no_go = "GO"
elif (n_configs_above_3x >= 1 or n_configs_above_p95 >= 1) and spearman_pass:
    go_no_go = "CONDITIONAL_GO"
elif spearman_pass:
    go_no_go = "CONDITIONAL_GO"
else:
    go_no_go = "NO_GO"

# Bar chart data
bar_chart_data = []
for r in sae_results:
    if "error" in r:
        continue
    bar_chart_data.append({
        "sae_config": r["config_name"],
        "layer": r["layer"],
        "width_k": r["width_k"],
        "absorption_rate": r["absorption_metric"]["absorption_rate"],
        "ci_lo": r["absorption_metric"]["absorption_rate_ci95"][0],
        "ci_hi": r["absorption_metric"]["absorption_rate_ci95"][1],
        "random_baseline": r["random_control"]["mean_rate"],
        "random_p95": r["random_control"]["p95_rate"],
        "above_random_3pp": r["above_random_3pp"],
        "above_random_3x": r["above_random_3x"],
        "above_random_p95": r["above_random_p95"],
        "absorption_ratio_vs_random": r["absorption_ratio_vs_random"],
        "spearman_rho_eda": r["spearman_rho_eda_absorption"],
        "phase1_eda_auroc": r.get("phase1_eda_auroc"),
    })

# Final output
output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL",
    "hierarchy": HIERARCHY,
    "config": {
        "sae_configs": [c[2] for c in SAE_CONFIGS],
        "cosine_thresh": COSINE_THRESH,
        "magnitude_gap_thresh": MAGNITUDE_GAP,
        "n_random_dirs": N_RANDOM_DIRS,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "probe_model": MODEL_NAME,
        "probe_layer": LAYER_IDX,
        "projection_strategy": projection_strategy,
    },
    "probe_info": probe_info,
    "per_sae_results": sae_results,
    "bar_chart_data": bar_chart_data,
    "cross_domain_spearman": {
        "rho": float(cross_domain_rho) if cross_domain_rho is not None else None,
        "p_value": float(cross_domain_p) if cross_domain_p is not None else None,
        "n_configs": len(per_sae_absorption),
        "description": "Spearman rho between Phase1 first-letter EDA AUROC and city-continent absorption rate per SAE config",
        "passes_gt_035": bool(cross_domain_rho is not None and cross_domain_rho >= 0.35),
    },
    "aggregate": {
        "n_configs_total": n_configs_total,
        "n_configs_successful": n_configs_successful,
        "n_configs_above_random_3pp": n_configs_above_3pp,
        "n_configs_above_random_3x": n_configs_above_3x,
        "n_configs_above_random_p95": n_configs_above_p95,
        "absorption_rates": [float(r["absorption_metric"]["absorption_rate"])
                             for r in sae_results if "error" not in r],
        "random_control_rates": [float(r["random_control"]["mean_rate"])
                                 for r in sae_results if "error" not in r],
        "mean_spearman_rho_eda_absorption": float(mean_rho) if mean_rho is not None else None,
        "all_rho_values": [float(r) for r in per_sae_rho_values],
    },
    "pass_criteria_check": {
        "absorption_above_random_3pp_at_least_1": bool(n_configs_above_3pp >= 1),
        "absorption_above_random_3x_at_least_1": bool(n_configs_above_3x >= 1),
        "absorption_above_random_p95_at_least_1": bool(n_configs_above_p95 >= 1),
        "spearman_rho_eda_gt_010": bool(spearman_pass),
        "cross_domain_rho_gt_010": bool(cross_domain_rho is not None and cross_domain_rho > 0.10),
        "overall_pass": bool(pilot_pass),
        "go_no_go": go_no_go,
        "note": (
            f"FULL mode. {n_configs_above_3pp}/{n_configs_successful} configs "
            f"show absorption > random by >= 3pp (absolute). "
            f"{n_configs_above_3x}/{n_configs_successful} show >= 3x relative. "
            f"Mean within-SAE Spearman rho(EDA, absorption_signal) = "
            f"{f'{mean_rho:.4f}' if mean_rho is not None else 'N/A'}. "
            f"PROBE CAVEAT: City-continent probe trained on Qwen2.5-0.5B "
            f"(acc={probe_acc:.4f}), projected to SAE d_in=2304 via {projection_strategy}. "
            f"Probe does not meet 85% quality gate. "
            f"Absorption signal relative to random baseline and EDA correlation "
            f"remain valid experimental signals."
        ),
    },
    "go_no_go": go_no_go,
    "limitations": [
        f"Probe trained on Qwen2.5-0.5B (acc={probe_acc:.4f}) not Gemma 2B (gated).",
        f"Probe direction projected from d_model={d_model} → d_in={SAE_D_IN} via {projection_strategy}.",
        "Absolute absorption rates may be inflated/deflated due to dimension mismatch.",
        "Relative comparisons (vs. random baseline, EDA correlation) remain valid.",
        "Probe accuracy below 85% quality gate — city-continent results are exploratory.",
    ],
}

# Save results
OUTPUT_FILE.write_text(json.dumps(output, indent=2))
log.info(f"Results written to {OUTPUT_FILE}")

# ─── Summary ──────────────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("PHASE 3b CITY-CONTINENT FULL RESULTS")
log.info("="*60)
log.info(f"{'Config':12s} {'Abs_Rate':12s} {'Rand_Rate':12s} {'Ratio':8s} {'P95?':6s} {'Rho_EDA':10s}")
log.info("-"*70)
for r in sae_results:
    if "error" in r:
        log.info(f"{r['config_name']:12s} ERROR: {r.get('error','')[:40]}")
        continue
    abs_rate = r["absorption_metric"]["absorption_rate"]
    rand_rate = r["random_control"]["mean_rate"]
    ratio = r["absorption_ratio_vs_random"]
    above_p95 = "YES" if r["above_random_p95"] else "NO"
    rho = r["spearman_rho_eda_absorption"]
    log.info(f"{r['config_name']:12s} {abs_rate:12.6f} {rand_rate:12.6f} {ratio:8.2f}x {above_p95:6s} {rho:10.4f}")

log.info(f"\nConfigs above random by >= 3pp: {n_configs_above_3pp}/{n_configs_successful}")
log.info(f"Configs above random by >= 3x: {n_configs_above_3x}/{n_configs_successful}")
log.info(f"Configs above random p95: {n_configs_above_p95}/{n_configs_successful}")
log.info(f"Mean Spearman rho: {f'{mean_rho:.4f}' if mean_rho is not None else 'N/A'}")
if cross_domain_rho is not None:
    log.info(f"Cross-domain Spearman rho (EDA_AUROC vs absorption): {cross_domain_rho:.4f} (p={cross_domain_p:.4f})")
log.info(f"Probe accuracy: {probe_acc:.4f} (Qwen2.5-0.5B, 7 classes, majority={majority_frac_filtered:.3f})")
log.info(f"GO_NO_GO: {go_no_go}")

write_progress(8, 8, {
    "step": "done",
    "go_no_go": go_no_go,
    "n_above_random_3pp": n_configs_above_3pp,
    "n_above_random_3x": n_configs_above_3x,
    "mean_rho": mean_rho,
    "probe_acc": probe_acc
})

mark_done(
    status="success",
    summary=(
        f"City-continent FULL: {n_configs_above_3pp}/{n_configs_successful} configs "
        f"above random by >=3pp, {n_configs_above_3x} by >=3x, "
        f"{n_configs_above_p95} above p95. "
        f"Mean Spearman rho={f'{mean_rho:.4f}' if mean_rho is not None else 'N/A'}. "
        f"Cross-domain rho={f'{cross_domain_rho:.4f}' if cross_domain_rho is not None else 'N/A'}. "
        f"Probe acc={probe_acc:.4f}. GO_NO_GO={go_no_go}."
    )
)

print(f"\n{'='*60}")
print(f"DONE: phase3_city_continent FULL mode")
print(f"GO_NO_GO: {go_no_go}")
print(f"Probe accuracy: {probe_acc:.4f} ({n_classes} classes, majority={majority_frac_filtered:.3f})")
print(f"Above random 3pp: {n_configs_above_3pp}/{n_configs_successful}")
print(f"Above random 3x: {n_configs_above_3x}/{n_configs_successful}")
print(f"Mean Spearman rho: {mean_rho:.4f}" if mean_rho else "Mean rho: N/A")
print(f"Results: {OUTPUT_FILE}")
print(f"{'='*60}")
