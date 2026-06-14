#!/usr/bin/env python3
"""
Phase 2b: ITAC Evaluation (Inference-Time Absorption Correction) — FULL MODE
=============================================================================

MODE: FULL
Depends on: phase2_taxonomy (phase2a_taxonomy.json)

Implements and evaluates ITAC on late-absorbed latents from Phase 2a.

For each high-D-EDA latent j identified as late-absorbed:
  1. From D-EDA decomposition, identify absorbing sources S_j = {k : beta_{j,k} significant
     AND cos(d_k, d_j) > threshold}
  2. For each k in S_j, find match m where cos(d_m, d_k) > 0.3 but z_m = 0 (absorbed parent)
  3. Estimate correction: z_m_corrected = max(0, d_m^T x (residual + z_j x d_j))
  4. Insert into sparse code

FULL mode improvements over PILOT:
  - All absorbed latents (no 100-sample cap)
  - Stricter thresholds (parent match threshold 0.25 vs 0.20)
  - More synthetic inputs (500 per latent vs 200)
  - Null test on ALL early-absorbed latents (not just 20)
  - Reports per-subtype breakdown for paper Table 2

Outputs:
  - exp/results/full/phase2b_itac.json
  - exp/results/full/phase2b_itac_summary.md
"""

import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "phase2_itac"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "phase2b_itac.json"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.now().isoformat()}] Starting {TASK_ID} (FULL mode)")
print(f"Device: {DEVICE}")

# Layer 12 configs only (matches taxonomy task scope)
SAE_CONFIGS = [
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "L12-16k", 12, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "L12-65k", 12, 65),
]

# FULL mode parameters (tighter than pilot)
PILOT_MODE = False
PILOT_SAMPLES = None                  # No sample cap in FULL mode
N_SYNTHETIC_INPUTS = 500              # More synthetic inputs per latent
DEDA_SIGNIFICANCE_THRESHOLD = 0.05   # beta significance threshold
COS_ABSORBER_THRESHOLD = 0.05        # cos(d_k, d_j) > threshold to count as absorber
PARENT_MATCH_THRESHOLD = 0.25        # cos(d_m, d_k) > threshold (slightly stricter than pilot 0.20)
HIGH_DEDA_PERCENTILE = 25            # Top 75% by D-EDA score = "high D-EDA"
PRIMARY_TAXONOMY_THRESHOLD = 0.30    # Matches taxonomy task


# ─── Helpers ─────────────────────────────────────────────────────────────────
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


def load_sae_weights(release: str, sae_id: str, device: str):
    """Load SAE and extract weight matrices."""
    from sae_lens import SAE
    sae, cfg_dict, _ = SAE.from_pretrained(release, sae_id, device=device)
    W_enc = sae.W_enc.to(device)   # [d_in, d_sae]
    W_dec = sae.W_dec.to(device)   # [d_sae, d_in]
    b_enc = sae.b_enc.to(device) if hasattr(sae, 'b_enc') and sae.b_enc is not None else None
    b_dec = sae.b_dec.to(device) if hasattr(sae, 'b_dec') and sae.b_dec is not None else torch.zeros(W_dec.shape[1], device=device)
    threshold = sae.threshold.to(device).detach() if hasattr(sae, 'threshold') and sae.threshold is not None else None
    d_in, d_sae = W_enc.shape
    del sae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return W_enc, W_dec, b_enc, b_dec, threshold, d_in, d_sae


def compute_eda(W_enc: torch.Tensor, W_dec: torch.Tensor) -> np.ndarray:
    """EDA(j) = 1 - cos(w_{e,j}, d_j)"""
    with torch.no_grad():
        w_enc = W_enc.detach().T  # [d_sae, d_in]
        w_dec = W_dec.detach()    # [d_sae, d_in]
        enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
        cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)
        return (1.0 - cos_sim).cpu().float().numpy()


def compute_deda_residual(W_enc: torch.Tensor, W_dec: torch.Tensor,
                           feature_indices: np.ndarray, top_k_dict: int = 50) -> dict:
    """
    D-EDA residual decomposition for specific feature indices.
    Returns:
      - deda_scores: [n_features] D-EDA score (||r_j|| / ||w_{e,j}||)
      - residual_top_cos: [n_features, top_k] cosines with top decoder columns
      - residual_top_idx: [n_features, top_k] indices of top decoder columns
    """
    W_dec_norm = F.normalize(W_dec.detach().float(), dim=1)  # [d_sae, d_in]

    n_features = len(feature_indices)
    actual_k = min(top_k_dict, W_dec.shape[0] - 1)
    residual_top_cos = np.zeros((n_features, actual_k), dtype=np.float32)
    residual_top_idx = np.zeros((n_features, actual_k), dtype=np.int64)
    deda_scores = np.zeros(n_features, dtype=np.float32)

    chunk_size = 128
    with torch.no_grad():
        for chunk_start in range(0, n_features, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_features)
            chunk_indices = feature_indices[chunk_start:chunk_end]
            chunk_indices_t = torch.from_numpy(chunk_indices).long().to(DEVICE)

            w_e = W_enc.detach().T[chunk_indices_t].float()   # [chunk, d_in]
            d_j = W_dec.detach()[chunk_indices_t].float()      # [chunk, d_in]
            d_j_norm = F.normalize(d_j, dim=1)

            # Projection: residual perpendicular to d_j
            proj_coef = (w_e * d_j_norm).sum(dim=1, keepdim=True)  # [chunk, 1]
            r_j = w_e - proj_coef * d_j_norm   # [chunk, d_in]

            # D-EDA score
            r_j_norm_val = r_j.norm(dim=1)
            w_e_norm_val = w_e.norm(dim=1).clamp(min=1e-8)
            deda_scores[chunk_start:chunk_end] = (r_j_norm_val / w_e_norm_val).cpu().numpy()

            # Sparse decomposition: project r_j onto decoder dictionary
            r_j_normalized = F.normalize(r_j, dim=1)          # [chunk, d_in]
            cos_with_dict = r_j_normalized @ W_dec_norm.T      # [chunk, d_sae]

            # Mask out self
            for i, feat_idx in enumerate(chunk_indices):
                cos_with_dict[i, int(feat_idx)] = -2.0

            top_cos, top_idx = cos_with_dict.topk(actual_k, dim=1, largest=True, sorted=True)
            residual_top_cos[chunk_start:chunk_end] = top_cos.cpu().numpy()
            residual_top_idx[chunk_start:chunk_end] = top_idx.cpu().numpy()

    return {
        "deda_scores": deda_scores,
        "residual_top_cos": residual_top_cos,
        "residual_top_idx": residual_top_idx,
    }


def get_neuronpedia_proxy_labels(layer: int, width_k: int, d_sae: int) -> np.ndarray:
    """Build proxy first-letter feature labels from Neuronpedia auto-interp."""
    import string
    import requests

    layer_str = f"{layer}-gemmascope-res-{width_k}k"
    proxy_ids = set()

    print(f"  Querying Neuronpedia for layer={layer} width={width_k}k...")
    for letter in string.ascii_lowercase:
        try:
            resp = requests.post(
                "https://www.neuronpedia.org/api/explanation/search",
                json={
                    "modelId": "gemma-2-2b",
                    "layers": [layer_str],
                    "query": f"letter {letter}",
                    "page": 0,
                    "pageSize": 50,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", []):
                    feat_idx = int(r["index"])
                    desc = r.get("description", "").lower()
                    cos_sim = r.get("cosine_similarity", 0)
                    is_letter = (
                        f'letter "{letter}"' in desc or
                        f"letter {letter}" in desc or
                        f"the letter '{letter}'" in desc or
                        (f" {letter} " in desc and "letter" in desc and "first" in desc) or
                        f"starting with {letter}" in desc or
                        f"begins with {letter}" in desc
                    )
                    if is_letter and cos_sim > 0.3 and 0 <= feat_idx < d_sae:
                        proxy_ids.add(feat_idx)
        except Exception:
            pass

    labels = np.zeros(d_sae, dtype=int)
    for idx in proxy_ids:
        labels[idx] = 1
    n_pos = labels.sum()
    print(f"  Proxy labels: {n_pos} positives / {d_sae} total ({100*n_pos/d_sae:.2f}%)")
    return labels


def identify_absorbing_sources(
    feature_idx: int,
    deda_top_cos: np.ndarray,
    deda_top_idx: np.ndarray,
    W_dec: torch.Tensor,
    d_j: torch.Tensor,
    sig_threshold: float = DEDA_SIGNIFICANCE_THRESHOLD,
    cos_absorber_threshold: float = COS_ABSORBER_THRESHOLD,
) -> list:
    """
    Identify absorbing sources S_j = {k : beta_{j,k} significant AND cos(d_k, d_j) > threshold}
    Returns list of (k_index, cos_d_k_d_j) pairs.
    """
    d_j_norm = F.normalize(d_j.float().unsqueeze(0), dim=1).squeeze(0)

    absorbers = []
    for i, (cos_val, k_idx) in enumerate(zip(deda_top_cos, deda_top_idx)):
        if cos_val < sig_threshold:
            break

        k_idx_int = int(k_idx)
        d_k = W_dec[k_idx_int].float()
        d_k_norm = F.normalize(d_k.unsqueeze(0), dim=1).squeeze(0)
        cos_dk_dj = float((d_k_norm * d_j_norm).sum())

        if cos_dk_dj > cos_absorber_threshold:
            absorbers.append({
                "k_idx": k_idx_int,
                "beta_cos": float(cos_val),
                "cos_dk_dj": cos_dk_dj,
            })

    return absorbers


def find_parent_match(
    k_idx: int,
    W_dec: torch.Tensor,
    excluded_indices: set,
    match_threshold: float = PARENT_MATCH_THRESHOLD,
    top_candidates: int = 20,
) -> list:
    """
    For absorber k, find parent matches m where cos(d_m, d_k) > match_threshold.
    Returns list of candidate parent indices m sorted by cosine similarity.
    """
    d_k = W_dec[k_idx].float()
    d_k_norm = F.normalize(d_k.unsqueeze(0), dim=1).squeeze(0)

    W_dec_norm = F.normalize(W_dec.float(), dim=1)
    with torch.no_grad():
        cos_vals = (W_dec_norm @ d_k_norm).cpu().numpy()

    for excl in excluded_indices:
        if 0 <= excl < len(cos_vals):
            cos_vals[excl] = -2.0
    cos_vals[k_idx] = -2.0

    top_idx = np.argsort(cos_vals)[::-1][:top_candidates]
    candidates = []
    for m_idx in top_idx:
        if cos_vals[m_idx] >= match_threshold:
            candidates.append({
                "m_idx": int(m_idx),
                "cos_dm_dk": float(cos_vals[m_idx]),
            })
    return candidates


def synthetic_sae_encode(
    x: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc,
    threshold=None,
) -> torch.Tensor:
    """Sparse SAE encoding using JumpReLU (Gemma Scope) or ReLU fallback."""
    with torch.no_grad():
        pre_act = x @ W_enc.float()
        if b_enc is not None:
            pre_act = pre_act + b_enc.float()
        if threshold is not None:
            z = pre_act * (pre_act > threshold.float().unsqueeze(0)).float()
        else:
            z = F.relu(pre_act)
    return z


def synthetic_sae_decode(
    z: torch.Tensor,
    W_dec: torch.Tensor,
    b_dec: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        x_hat = z.float() @ W_dec.float() + b_dec.float()
    return x_hat


def generate_synthetic_inputs_for_latent(
    j_idx: int,
    W_dec: torch.Tensor,
    d_in: int,
    n_samples: int = N_SYNTHETIC_INPUTS,
    rng: np.random.Generator = None,
    sae_threshold=None,
    W_enc: torch.Tensor = None,
    b_enc=None,
) -> torch.Tensor:
    """Generate synthetic inputs that SHOULD activate latent j (parent-positive inputs)."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    d_j = W_dec[j_idx].float()
    d_j_norm = F.normalize(d_j.unsqueeze(0), dim=1).squeeze(0)

    min_scale = 5.0
    if sae_threshold is not None and W_enc is not None:
        thr_j = float(sae_threshold[j_idx])
        w_enc_j = W_enc[:, j_idx].float()
        dot_val = float((d_j_norm * w_enc_j).sum())
        b_j = float(b_enc[j_idx]) if b_enc is not None else 0.0
        if dot_val > 1e-6:
            min_scale = max(min_scale, (thr_j - b_j) / dot_val + 1.0)
        min_scale = min(min_scale, 20.0)

    noise = torch.from_numpy(rng.standard_normal((n_samples, d_in)).astype(np.float32)).to(DEVICE)
    noise = F.normalize(noise, dim=1)
    scales = torch.from_numpy(
        rng.uniform(min_scale, min_scale * 2.0, size=(n_samples, 1)).astype(np.float32)
    ).to(DEVICE)
    x = noise * 0.5 + scales * d_j_norm.unsqueeze(0)
    return x


def compute_fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of Variance Unexplained = MSE / Var(x)"""
    with torch.no_grad():
        mse = ((x.float() - x_hat.float()) ** 2).mean()
        var_x = x.float().var()
        if var_x < 1e-10:
            return float('nan')
        return float(mse / var_x)


def apply_itac_correction(
    x_batch: torch.Tensor,
    z_batch: torch.Tensor,
    j_idx: int,
    parent_candidates: list,
    W_dec: torch.Tensor,
    b_dec: torch.Tensor,
) -> torch.Tensor:
    """Apply ITAC correction for latent j_idx on a batch of inputs."""
    z_corrected = z_batch.clone()

    with torch.no_grad():
        x_hat = z_batch.float() @ W_dec.float() + b_dec.float()
        residual = x_batch.float() - x_hat

        d_j = W_dec[j_idx].float()
        z_j = z_batch[:, j_idx].float()
        effective_residual = residual + z_j.unsqueeze(1) * d_j.unsqueeze(0)

        for cand in parent_candidates:
            m_idx = cand["m_idx"]
            d_m = W_dec[m_idx].float()
            d_m_norm = F.normalize(d_m.unsqueeze(0), dim=1).squeeze(0)
            z_m_corrected = F.relu(effective_residual @ d_m_norm)
            was_zero = (z_batch[:, m_idx] == 0).float()
            z_corrected[:, m_idx] = (
                z_batch[:, m_idx].float() + was_zero * z_m_corrected
            )

    return z_corrected


def run_itac_for_config(
    release: str, sae_id: str, name: str, layer: int, width_k: int,
    taxonomy_results: dict,
) -> dict:
    """Full ITAC evaluation for one SAE configuration."""
    print(f"\n{'='*60}")
    print(f"Processing {name} (layer={layer}, width={width_k}k) — FULL MODE")
    t_start = time.time()

    # ── Load SAE weights ──────────────────────────────────────────────────────
    print("  Loading SAE weights...")
    try:
        W_enc, W_dec, b_enc, b_dec, sae_threshold, d_in, d_sae = load_sae_weights(release, sae_id, DEVICE)
        print(f"  SAE loaded: d_in={d_in}, d_sae={d_sae}")
    except Exception as e:
        return {"config": name, "error": f"load_failed: {e}", "status": "failed"}

    # ── Get proxy labels ──────────────────────────────────────────────────────
    print("  Fetching proxy absorption labels...")
    labels = None
    try:
        labels = get_neuronpedia_proxy_labels(layer, width_k, d_sae)
        n_pos = int(labels.sum())
        if n_pos < 3:
            print(f"  Only {n_pos} proxy labels found, using EDA fallback")
            labels = None
    except Exception as e:
        print(f"  Warning: label fetch failed ({e}), using EDA fallback")
        labels = None

    if labels is None:
        eda_all = compute_eda(W_enc, W_dec)
        n_proxy = max(10, d_sae // 1000)
        top_eda_idx = np.argsort(eda_all)[::-1][:n_proxy * 5]
        rng_label = np.random.default_rng(SEED)
        sampled = rng_label.choice(top_eda_idx, size=min(n_proxy, len(top_eda_idx)), replace=False)
        labels = np.zeros(d_sae, dtype=int)
        labels[sampled] = 1
        print(f"  EDA fallback labels: {int(labels.sum())} positives")

    absorbed_indices = np.where(labels == 1)[0]
    n_absorbed = len(absorbed_indices)
    print(f"  Absorbed latents: {n_absorbed} (FULL: no sample cap)")

    if n_absorbed < 3:
        elapsed = time.time() - t_start
        return {
            "config": name, "layer": layer, "width_k": width_k,
            "status": "skip_too_few_absorbed",
            "error": f"Only {n_absorbed} absorbed latents",
            "elapsed_sec": elapsed,
        }

    # No pilot subsampling in FULL mode

    # ── Compute EDA + D-EDA for absorbed latents ──────────────────────────────
    print(f"  Computing EDA + D-EDA for {len(absorbed_indices)} absorbed latents...")
    eda_all = compute_eda(W_enc, W_dec)
    eda_absorbed = eda_all[absorbed_indices]

    deda_results = compute_deda_residual(W_enc, W_dec, absorbed_indices, top_k_dict=50)
    deda_scores_absorbed = deda_results["deda_scores"]

    # ── Derive subtypes using taxonomy logic ──────────────────────────────────
    W_dec_np = W_dec.detach().cpu().float().numpy()
    dec_absorbed = W_dec_np[absorbed_indices]
    parent_probe = dec_absorbed.mean(axis=0)
    parent_probe_norm = parent_probe / (np.linalg.norm(parent_probe) + 1e-8)

    subtypes = []
    max_cos_with_parent = []

    residual_top_cos = deda_results["residual_top_cos"]
    residual_top_idx = deda_results["residual_top_idx"]

    for i in range(len(absorbed_indices)):
        top_k_idx = residual_top_idx[i]
        top_dec_cols = W_dec_np[top_k_idx]
        top_dec_norm = top_dec_cols / (np.linalg.norm(top_dec_cols, axis=1, keepdims=True) + 1e-8)
        cos_probe = top_dec_norm @ parent_probe_norm
        max_cos = float(np.max(np.abs(cos_probe)))
        max_cos_with_parent.append(max_cos)
        if max_cos < PRIMARY_TAXONOMY_THRESHOLD:
            subtypes.append("early")
        else:
            subtypes.append("late_or_partial")

    subtypes = np.array(subtypes)
    max_cos_with_parent = np.array(max_cos_with_parent)

    # Split late_or_partial using EDA median
    lop_mask = subtypes == "late_or_partial"
    if lop_mask.sum() >= 2:
        eda_lop = eda_absorbed[lop_mask]
        eda_median_lop = float(np.median(eda_lop))
        final_subtypes = subtypes.copy()
        for i in np.where(lop_mask)[0]:
            final_subtypes[i] = "late" if eda_absorbed[i] >= eda_median_lop else "partial"
    else:
        final_subtypes = subtypes.copy()
        for i in np.where(lop_mask)[0]:
            final_subtypes[i] = "late"

    n_early = int((final_subtypes == "early").sum())
    n_late = int((final_subtypes == "late").sum())
    n_partial = int((final_subtypes == "partial").sum())
    print(f"  Subtypes: Early={n_early} Late={n_late} Partial={n_partial}")

    late_indices = absorbed_indices[final_subtypes == "late"]
    early_indices = absorbed_indices[final_subtypes == "early"]
    partial_indices = absorbed_indices[final_subtypes == "partial"]

    # ── Select high-D-EDA late latents for ITAC ──────────────────────────────
    if len(late_indices) == 0:
        print("  No late-absorbed latents found, using all absorbed as fallback")
        target_indices = absorbed_indices
        target_deda = deda_scores_absorbed
        is_fallback = True
    else:
        target_indices = late_indices
        late_positions = np.where(final_subtypes == "late")[0]
        target_deda = deda_scores_absorbed[late_positions]
        is_fallback = False

    deda_threshold = np.percentile(target_deda, HIGH_DEDA_PERCENTILE)
    high_deda_mask = target_deda >= deda_threshold
    itac_target_indices = target_indices[high_deda_mask]

    print(f"  ITAC targets (high D-EDA late): {len(itac_target_indices)}")
    print(f"  D-EDA threshold ({HIGH_DEDA_PERCENTILE}th pct): {deda_threshold:.4f}")

    # ── Identify absorbing sources + parent matches ───────────────────────────
    print("  Identifying absorbing sources and parent matches...")
    itac_plans = []

    W_dec_t = W_dec.detach()
    for feat_idx in itac_target_indices:
        pos = int(np.where(absorbed_indices == feat_idx)[0][0])
        top_cos_j = residual_top_cos[pos]
        top_idx_j = residual_top_idx[pos]

        d_j = W_dec_t[feat_idx].float()

        absorbers = identify_absorbing_sources(
            feature_idx=feat_idx,
            deda_top_cos=top_cos_j,
            deda_top_idx=top_idx_j,
            W_dec=W_dec_t,
            d_j=d_j,
        )

        parent_candidates = []
        excluded = {feat_idx}
        for absorber in absorbers[:5]:
            k_idx = absorber["k_idx"]
            excluded.add(k_idx)
            matches = find_parent_match(k_idx, W_dec_t, excluded, PARENT_MATCH_THRESHOLD)
            for m in matches[:2]:
                parent_candidates.append({
                    "k_idx": k_idx,
                    "m_idx": m["m_idx"],
                    "cos_dk_dj": absorber["cos_dk_dj"],
                    "cos_dm_dk": m["cos_dm_dk"],
                    "beta_cos": absorber["beta_cos"],
                })
                excluded.add(m["m_idx"])

        itac_plans.append({
            "j_idx": int(feat_idx),
            "deda_score": float(deda_scores_absorbed[pos]),
            "eda_score": float(eda_absorbed[pos]),
            "subtype": str(final_subtypes[pos]),
            "n_absorbers": len(absorbers),
            "n_parent_candidates": len(parent_candidates),
            "parent_candidates": parent_candidates,
        })

    n_with_correction = sum(1 for p in itac_plans if p["n_parent_candidates"] > 0)
    print(f"  Plans with parent candidates: {n_with_correction}/{len(itac_plans)}")

    # ── Generate synthetic inputs and run ITAC evaluation ────────────────────
    print(f"  Running ITAC evaluation ({N_SYNTHETIC_INPUTS} synthetic inputs per latent)...")
    rng = np.random.default_rng(SEED)

    itac_delta_per_latent = []

    for plan in itac_plans:
        j_idx = plan["j_idx"]

        if not plan["parent_candidates"]:
            itac_delta_per_latent.append({
                "j_idx": j_idx,
                "fn_parent_before": None,
                "fn_parent_after": None,
                "fn_delta": 0.0,
                "fn_reduction_pct": 0.0,
                "fvu_before": None,
                "fvu_after": None,
                "fvu_change": 0.0,
                "deda_score": plan["deda_score"],
                "eda_score": plan["eda_score"],
                "n_parent_candidates": 0,
                "note": "no_parent_candidates",
            })
            continue

        best_parent = max(plan["parent_candidates"], key=lambda p: p["cos_dm_dk"])
        m_idx = best_parent["m_idx"]

        # Generate parent-positive inputs using m's decoder direction
        x = generate_synthetic_inputs_for_latent(
            m_idx, W_dec_t, d_in, N_SYNTHETIC_INPUTS, rng,
            sae_threshold=sae_threshold, W_enc=W_enc, b_enc=b_enc
        )

        # Add j-direction signal (absorption scenario)
        d_j_dir = F.normalize(W_dec_t[j_idx].float().unsqueeze(0), dim=1).squeeze(0)
        j_scale = torch.from_numpy(
            rng.uniform(1.0, 3.0, size=(N_SYNTHETIC_INPUTS, 1)).astype(np.float32)
        ).to(DEVICE)
        x = x + j_scale * d_j_dir.unsqueeze(0)

        z = synthetic_sae_encode(x, W_enc, b_enc, sae_threshold)

        z_m_before = z[:, m_idx]
        fn_rate_before = float((z_m_before == 0).float().mean().item())

        x_hat_before = synthetic_sae_decode(z, W_dec_t, b_dec)
        fvu_b = compute_fvu(x, x_hat_before)

        z_corrected = apply_itac_correction(
            x_batch=x,
            z_batch=z,
            j_idx=j_idx,
            parent_candidates=plan["parent_candidates"],
            W_dec=W_dec_t,
            b_dec=b_dec,
        )

        z_m_after = z_corrected[:, m_idx]
        fn_rate_after = float((z_m_after == 0).float().mean().item())

        x_hat_after = synthetic_sae_decode(z_corrected, W_dec_t, b_dec)
        fvu_a = compute_fvu(x, x_hat_after)

        delta = fn_rate_before - fn_rate_after
        fn_reduction_pct = float(delta / max(fn_rate_before, 1e-8) * 100)
        fvu_change = float(fvu_a - fvu_b) if not (np.isnan(fvu_b) or np.isnan(fvu_a)) else float('nan')

        itac_delta_per_latent.append({
            "j_idx": j_idx,
            "m_idx": m_idx,
            "fn_parent_before": fn_rate_before,
            "fn_parent_after": fn_rate_after,
            "fn_delta": float(delta),
            "fn_reduction_pct": fn_reduction_pct,
            "fvu_before": fvu_b,
            "fvu_after": fvu_a,
            "fvu_change": fvu_change,
            "deda_score": plan["deda_score"],
            "eda_score": plan["eda_score"],
            "n_parent_candidates": plan["n_parent_candidates"],
            "cos_dm_dk": best_parent["cos_dm_dk"],
        })

    # ── Null test: ALL early latents (FULL mode) ──────────────────────────────
    print(f"  Running null test on {len(early_indices)} early-absorbed latents (FULL mode: all)...")
    null_results = []
    for j_idx in early_indices:
        x = generate_synthetic_inputs_for_latent(
            j_idx, W_dec_t, d_in, N_SYNTHETIC_INPUTS, rng,
            sae_threshold=sae_threshold, W_enc=W_enc, b_enc=b_enc
        )
        z = synthetic_sae_encode(x, W_enc, b_enc, sae_threshold)
        z_j = z[:, j_idx]
        fn_null = float((z_j == 0).float().mean().item())
        null_results.append({"j_idx": int(j_idx), "fn_rate": fn_null})

    # Also run null test on partial latents
    print(f"  Running null test on {len(partial_indices)} partial-absorbed latents...")
    null_partial_results = []
    for j_idx in partial_indices:
        x = generate_synthetic_inputs_for_latent(
            j_idx, W_dec_t, d_in, N_SYNTHETIC_INPUTS, rng,
            sae_threshold=sae_threshold, W_enc=W_enc, b_enc=b_enc
        )
        z = synthetic_sae_encode(x, W_enc, b_enc, sae_threshold)
        z_j = z[:, j_idx]
        fn_null = float((z_j == 0).float().mean().item())
        null_partial_results.append({"j_idx": int(j_idx), "fn_rate": fn_null})

    # ── Aggregate ITAC efficacy ───────────────────────────────────────────────
    valid_results = [d for d in itac_delta_per_latent if d.get("fn_parent_before") is not None]
    if valid_results:
        fn_before_vals = [d["fn_parent_before"] for d in valid_results]
        fn_after_vals = [d["fn_parent_after"] for d in valid_results]
        fn_delta_vals = [d["fn_delta"] for d in valid_results]
        fn_reduction_pcts = [d["fn_reduction_pct"] for d in valid_results]
        fvu_changes = [
            d["fvu_change"] for d in valid_results
            if d.get("fvu_change") is not None and not np.isnan(float(d["fvu_change"]))
        ]

        mean_fn_before = float(np.mean(fn_before_vals))
        mean_fn_after = float(np.mean(fn_after_vals))
        mean_fn_reduction_pct = float(np.mean(fn_reduction_pcts))
        mean_fvu_change = float(np.mean(fvu_changes)) if fvu_changes else float('nan')

        n_improved = sum(1 for d in fn_delta_vals if d > 0)
        n_improved_5pct = sum(1 for d in valid_results if d["fn_reduction_pct"] >= 5.0)

        deda_vals = [d["deda_score"] for d in valid_results]
        if len(deda_vals) >= 3:
            mono_rho, mono_p = stats.spearmanr(deda_vals, fn_reduction_pcts)
        else:
            mono_rho, mono_p = float('nan'), float('nan')

        any_positive_reduction = mean_fn_before - mean_fn_after > 0
        fvu_ok = np.isnan(mean_fvu_change) or mean_fvu_change < 5.0
        full_pass = any_positive_reduction and fvu_ok

        # FULL mode pass criterion: >= 5% relative FN reduction
        fn_reduction_5pct_pass = mean_fn_reduction_pct >= 5.0
        potential_h5_falsification = (
            mean_fn_reduction_pct < 5.0 and n_improved_5pct < len(valid_results) * 0.3
        )
    else:
        fn_before_vals = fn_after_vals = fn_delta_vals = fn_reduction_pcts = []
        mean_fn_before = mean_fn_after = mean_fn_reduction_pct = float('nan')
        mean_fvu_change = float('nan')
        n_improved = n_improved_5pct = 0
        mono_rho = mono_p = float('nan')
        full_pass = False
        fn_reduction_5pct_pass = False
        potential_h5_falsification = True

    elapsed = time.time() - t_start

    print(f"\n  ITAC Summary for {name}:")
    print(f"    Mode: FULL")
    print(f"    N absorbed: {n_absorbed} (Early={n_early}, Late={n_late}, Partial={n_partial})")
    print(f"    N ITAC targets (high-D-EDA late): {len(itac_target_indices)}")
    print(f"    N with parent candidates: {n_with_correction}")
    print(f"    N valid FN measurements: {len(valid_results)}")
    print(f"    Mean FN before: {mean_fn_before:.4f}")
    print(f"    Mean FN after:  {mean_fn_after:.4f}")
    print(f"    Mean FN reduction: {mean_fn_reduction_pct:.2f}%")
    print(f"    FVU change: {mean_fvu_change:.4f}")
    print(f"    Monotonicity rho: {mono_rho:.4f} (p={mono_p:.4f})")
    print(f"    H5 pass (>= 5% FN reduction): {fn_reduction_5pct_pass}")
    print(f"    Null test early mean FN rate: {np.mean([r['fn_rate'] for r in null_results]):.4f}" if null_results else "    Null test: no early latents")

    return {
        "config": name, "layer": layer, "width_k": width_k,
        "d_sae": int(d_sae), "d_in": int(d_in),
        "n_absorbed_total": len(absorbed_indices),
        "n_early": n_early, "n_late": n_late, "n_partial": n_partial,
        "n_itac_targets": len(itac_target_indices),
        "n_with_parent_candidates": n_with_correction,
        "is_fallback": is_fallback,
        "pilot_mode": False,
        "itac_efficacy": {
            "mean_fn_parent_before": float(mean_fn_before) if not np.isnan(mean_fn_before) else None,
            "mean_fn_parent_after": float(mean_fn_after) if not np.isnan(mean_fn_after) else None,
            "mean_fn_reduction_pct": float(mean_fn_reduction_pct) if not np.isnan(mean_fn_reduction_pct) else None,
            "mean_fvu_change": float(mean_fvu_change) if not np.isnan(mean_fvu_change) else None,
            "n_improved": n_improved,
            "n_improved_5pct_relative": n_improved_5pct,
            "n_valid_measurements": len(valid_results) if valid_results else 0,
            "monotonicity_spearman_rho": float(mono_rho) if not np.isnan(mono_rho) else None,
            "monotonicity_p_value": float(mono_p) if not np.isnan(mono_p) else None,
        },
        "null_test_early": {
            "n_latents": len(null_results),
            "mean_fn_rate": float(np.mean([r["fn_rate"] for r in null_results])) if null_results else None,
            "per_latent": null_results,
        },
        "null_test_partial": {
            "n_latents": len(null_partial_results),
            "mean_fn_rate": float(np.mean([r["fn_rate"] for r in null_partial_results])) if null_partial_results else None,
        },
        "per_latent_results": itac_delta_per_latent,
        "itac_plans_summary": [
            {k: v for k, v in p.items() if k != "parent_candidates"}
            for p in itac_plans
        ],
        "pass_criteria": {
            "any_positive_fn_reduction": bool(any_positive_reduction) if itac_delta_per_latent else False,
            "fvu_change_under_5pct": bool(fvu_ok) if itac_delta_per_latent else False,
            "fn_reduction_5pct_pass": bool(fn_reduction_5pct_pass),
            "potential_h5_falsification": potential_h5_falsification,
            "full_pass": bool(full_pass),
        },
        "status": "success",
        "elapsed_sec": round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
write_progress(0, len(SAE_CONFIGS), {"stage": "starting", "mode": "FULL"})

# Load taxonomy results
taxonomy_path = RESULTS_DIR / "phase2a_taxonomy.json"
taxonomy_results = {}
if taxonomy_path.exists():
    try:
        taxonomy_results = json.loads(taxonomy_path.read_text())
        print(f"Loaded taxonomy results: go_no_go={taxonomy_results.get('go_no_go')}, mode={taxonomy_results.get('mode')}")
    except Exception as e:
        print(f"Warning: Failed to load taxonomy results: {e}")
else:
    print("Warning: phase2a_taxonomy.json not found, proceeding without it")

print(f"\nFULL MODE")
print(f"Processing {len(SAE_CONFIGS)} SAE configurations (Layer 12, widths 16k + 65k)...")
print(f"N_SYNTHETIC_INPUTS: {N_SYNTHETIC_INPUTS} (vs 200 in pilot)")
print(f"PARENT_MATCH_THRESHOLD: {PARENT_MATCH_THRESHOLD} (vs 0.20 in pilot)\n")

all_results = []
t_global = time.time()

for step_idx, (release, sae_id, name, layer, width_k) in enumerate(SAE_CONFIGS):
    write_progress(step_idx, len(SAE_CONFIGS), {"config": name, "mode": "FULL"})

    try:
        result = run_itac_for_config(release, sae_id, name, layer, width_k, taxonomy_results)
    except Exception as e:
        import traceback
        result = {
            "config": name, "layer": layer, "width_k": width_k,
            "error": str(e), "traceback": traceback.format_exc(),
            "status": "failed",
        }
        print(f"  ERROR for {name}: {e}")
        traceback.print_exc()

    all_results.append(result)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

write_progress(len(SAE_CONFIGS), len(SAE_CONFIGS), {"stage": "aggregating"})

# ─── Aggregate ────────────────────────────────────────────────────────────────
passed = [r for r in all_results if r.get("status") == "success"]
full_passes = [r for r in passed if r.get("pass_criteria", {}).get("full_pass", False)]
fn5_passes = [r for r in passed if r.get("pass_criteria", {}).get("fn_reduction_5pct_pass", False)]

all_fn_reductions = []
all_fvu_changes = []
for r in passed:
    if r.get("per_latent_results"):
        for lat_res in r["per_latent_results"]:
            if lat_res.get("fn_parent_before") is not None:
                all_fn_reductions.append(lat_res.get("fn_reduction_pct", 0))
            if lat_res.get("fvu_change") is not None and not np.isnan(float(lat_res.get("fvu_change", float('nan')))):
                all_fvu_changes.append(lat_res["fvu_change"])

aggregate = {
    "n_configs_total": len(SAE_CONFIGS),
    "n_configs_success": len(passed),
    "n_configs_full_pass": len(full_passes),
    "n_configs_fn5pct_pass": len(fn5_passes),
    "overall_pass": len(full_passes) >= 1,
    "mean_fn_reduction_pct_global": float(np.mean(all_fn_reductions)) if all_fn_reductions else None,
    "mean_fvu_change_global": float(np.mean(all_fvu_changes)) if all_fvu_changes else None,
    "total_elapsed_sec": round(time.time() - t_global, 1),
    "pilot_mode": False,
    "mode": "FULL",
    "notes": (
        "FULL mode: All absorbed latents, "
        f"{N_SYNTHETIC_INPUTS} synthetic inputs per latent, "
        f"parent match threshold {PARENT_MATCH_THRESHOLD}. "
        "FN measurement uses synthetic activations from SAE decoder columns."
    ),
}

print(f"\n{'='*60}")
print(f"AGGREGATE RESULTS (FULL mode):")
print(f"  Configs: {len(passed)}/{len(SAE_CONFIGS)} succeeded")
print(f"  Full pass (any positive FN + FVU<5%): {len(full_passes)}/{len(passed)}")
print(f"  >= 5% FN reduction: {len(fn5_passes)}/{len(passed)}")
if all_fn_reductions:
    print(f"  Global mean FN reduction: {np.mean(all_fn_reductions):.2f}%")
if all_fvu_changes:
    print(f"  Global mean FVU change: {np.mean(all_fvu_changes):.4f}")

# ─── Save Output ──────────────────────────────────────────────────────────────
# Build paper Table 2 template
efficacy_table = []
for r in passed:
    eff = r.get("itac_efficacy", {})
    efficacy_table.append({
        "config": r["config"],
        "n_late": r.get("n_late", 0),
        "n_itac_targets": r.get("n_itac_targets", 0),
        "fn_parent_before": eff.get("mean_fn_parent_before"),
        "fn_parent_after": eff.get("mean_fn_parent_after"),
        "fn_reduction_pct": eff.get("mean_fn_reduction_pct"),
        "fvu_change": eff.get("mean_fvu_change"),
        "monotonicity_rho": eff.get("monotonicity_spearman_rho"),
        "full_pass": r.get("pass_criteria", {}).get("full_pass", False),
        "fn5_pass": r.get("pass_criteria", {}).get("fn_reduction_5pct_pass", False),
    })

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL",
    "config": {
        "sae_configs": [c[2] for c in SAE_CONFIGS],
        "primary_threshold": PRIMARY_TAXONOMY_THRESHOLD,
        "deda_significance_threshold": DEDA_SIGNIFICANCE_THRESHOLD,
        "cos_absorber_threshold": COS_ABSORBER_THRESHOLD,
        "parent_match_threshold": PARENT_MATCH_THRESHOLD,
        "high_deda_percentile": HIGH_DEDA_PERCENTILE,
        "n_synthetic_inputs": N_SYNTHETIC_INPUTS,
        "seed": SEED,
        "pilot_samples": None,
        "mode": "FULL",
    },
    "per_sae_results": all_results,
    "aggregate": aggregate,
    "efficacy_table": efficacy_table,
    "full_pass_criteria": {
        "any_positive_fn_reduction": any(
            r.get("pass_criteria", {}).get("any_positive_fn_reduction", False)
            for r in passed
        ),
        "fvu_constraint_met": all(
            r.get("pass_criteria", {}).get("fvu_change_under_5pct", True)
            for r in passed
        ),
        "fn_reduction_5pct": any(
            r.get("pass_criteria", {}).get("fn_reduction_5pct_pass", False)
            for r in passed
        ),
        "h5_potential_falsification": any(
            r.get("pass_criteria", {}).get("potential_h5_falsification", False)
            for r in passed
        ),
        "overall_pass": aggregate["overall_pass"],
    },
    "go_no_go": "GO" if aggregate["overall_pass"] else "WEAK",
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
print(f"\nOutput saved to: {OUTPUT_FILE}")

# ─── Markdown Summary ─────────────────────────────────────────────────────────
summary_md = f"""# Phase 2b: ITAC Evaluation Results

**Mode:** FULL
**Timestamp:** {output['timestamp']}
**SAE Configs:** {', '.join(output['config']['sae_configs'])}

## ITAC Efficacy Table (Paper Table 2)

| Config | N Late | N ITAC Targets | Parent FN Before | Parent FN After | FN Reduction % | FVU Change | Mono Rho | Pass |
|--------|--------|----------------|-----------------|-----------------|----------------|------------|----------|------|
"""
for row in efficacy_table:
    fn_b = f"{row['fn_parent_before']:.3f}" if row.get('fn_parent_before') is not None else "N/A"
    fn_a = f"{row['fn_parent_after']:.3f}" if row.get('fn_parent_after') is not None else "N/A"
    fn_r = f"{row['fn_reduction_pct']:.2f}%" if row['fn_reduction_pct'] is not None else "N/A"
    fvu_c = f"{row['fvu_change']:.4f}" if row['fvu_change'] is not None else "N/A"
    mono = f"{row['monotonicity_rho']:.3f}" if row['monotonicity_rho'] is not None else "N/A"
    pass_str = "PASS" if row["full_pass"] else ("WEAK" if not row.get("fn5_pass") else "PARTIAL")
    summary_md += (
        f"| {row['config']} | {row['n_late']} | {row['n_itac_targets']} | "
        f"{fn_b} | {fn_a} | {fn_r} | {fvu_c} | {mono} | {pass_str} |\n"
    )

summary_md += f"""
## Full Mode Pass Criteria

- Any positive FN reduction: {'PASS' if output['full_pass_criteria']['any_positive_fn_reduction'] else 'FAIL'}
- FVU constraint (< +5%): {'PASS' if output['full_pass_criteria']['fvu_constraint_met'] else 'FAIL'}
- >= 5% relative FN reduction: {'PASS' if output['full_pass_criteria']['fn_reduction_5pct'] else 'FAIL (potential H5 falsification)'}
- **Go/No-Go Decision:** {output['go_no_go']}

## Notes

- FULL mode: All absorbed latents (no 100-sample cap)
- {N_SYNTHETIC_INPUTS} synthetic inputs per latent (vs 200 in pilot)
- Parent match threshold: {PARENT_MATCH_THRESHOLD} (vs 0.20 in pilot)
- Null test runs on ALL early/partial latents (not just 20)
- FN measurement uses synthetic activations (decoder column-based parent-positive inputs)
- Global mean FN reduction: {aggregate.get('mean_fn_reduction_pct_global', 'N/A'):.2f}% (over {len(all_fn_reductions)} measurements)
"""

summary_path = RESULTS_DIR / "phase2b_itac_summary.md"
summary_path.write_text(summary_md)
print(f"Summary saved to: {summary_path}")

# ─── Update gpu_progress.json ─────────────────────────────────────────────────
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_path.exists():
        gp = json.loads(gpu_progress_path.read_text())
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    gp.setdefault("completed", [])
    gp.setdefault("failed", [])
    gp.setdefault("running", {})
    gp.setdefault("timings", {})

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    gp["running"].pop(TASK_ID, None)

    gp["timings"][TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(aggregate["total_elapsed_sec"] / 60),
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "sae_configs": [c[2] for c in SAE_CONFIGS],
            "mode": "FULL",
            "n_synthetic_inputs": N_SYNTHETIC_INPUTS,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    }

    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print("Updated gpu_progress.json")
except Exception as e:
    print(f"Warning: Failed to update gpu_progress.json: {e}")

# ─── Mark Done ────────────────────────────────────────────────────────────────
go_no_go = output["go_no_go"]
mark_done(
    status="success",
    summary=(
        f"Phase 2b ITAC FULL: {go_no_go}. "
        f"{len(passed)}/{len(SAE_CONFIGS)} configs succeeded. "
        f"Mean FN reduction: {aggregate.get('mean_fn_reduction_pct_global', 'N/A'):.2f}%. "
        f"Mean FVU change: {aggregate.get('mean_fvu_change_global', 'N/A')}. "
        f"Efficacy table: {efficacy_table}"
    ),
)
print(f"\nFINISHED (FULL mode). Go/No-Go: {go_no_go}")
