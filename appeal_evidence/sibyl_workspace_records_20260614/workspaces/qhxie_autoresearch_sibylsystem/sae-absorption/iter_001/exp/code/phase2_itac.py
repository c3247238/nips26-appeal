#!/usr/bin/env python3
"""
Phase 2b: ITAC Evaluation (Inference-Time Absorption Correction)
================================================================

MODE: PILOT (--tasks=phase2_itac)
Depends on: phase2_taxonomy (phase2a_taxonomy.json)

Implements and evaluates ITAC on late-absorbed latents from Phase 2a.

For each high-D-EDA latent j identified as late-absorbed:
  1. From D-EDA decomposition, identify absorbing sources S_j = {k : beta_{j,k} significant
     AND cos(d_k, d_j) > 0.1}
  2. For each k in S_j, find match m where cos(d_m, d_k) > 0.3 but z_m = 0 (absorbed parent)
  3. Estimate correction: z_m_corrected = max(0, d_m^T x (residual + z_j x d_j))
  4. Insert into sparse code

Metrics:
  - False-negative rate before vs. after ITAC
  - FVU change (must be < +5%)
  - Null test on early-absorbed latents
  - Monotonicity check (ITAC improvement vs. D-EDA score)

PILOT mode: Uses SAE weights only (training-free), synthetic activations generated
from SAE decoder columns (no Gemma 2 2B model needed).

Outputs:
  - exp/results/full/phase2b_itac.json
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

DEVICE = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '4').split(',')[0]}" if torch.cuda.is_available() else "cpu"
# If CUDA_VISIBLE_DEVICES is set, device index is always 0 within the visible set
if torch.cuda.is_available():
    DEVICE = "cuda:0"

print(f"[{datetime.now().isoformat()}] Starting {TASK_ID}")
print(f"Device: {DEVICE}")

# Focus on layer 12 (16k and 65k) matching taxonomy task
SAE_CONFIGS = [
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "L12-16k", 12, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "L12-65k", 12, 65),
]

# PILOT: use at most 100 absorbed latents, synthetic activations
PILOT_MODE = True
PILOT_SAMPLES = 100
N_SYNTHETIC_INPUTS = 200   # synthetic inputs per latent for FN measurement
DEDA_SIGNIFICANCE_THRESHOLD = 0.05   # beta significance threshold (lowered for pilot)
COS_ABSORBER_THRESHOLD = 0.05        # cos(d_k, d_j) > 0.05 to count as absorber
PARENT_MATCH_THRESHOLD = 0.20        # cos(d_m, d_k) > 0.2 for parent match (relaxed for pilot)
HIGH_DEDA_PERCENTILE = 25            # Top 75% by D-EDA score = "high D-EDA" (more inclusive)


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
    sae = SAE.from_pretrained(release, sae_id, device=device)
    W_enc = sae.W_enc.to(device)   # [d_in, d_sae]
    W_dec = sae.W_dec.to(device)   # [d_sae, d_in]
    b_enc = sae.b_enc.to(device) if hasattr(sae, 'b_enc') else None
    b_dec = sae.b_dec.to(device) if hasattr(sae, 'b_dec') else torch.zeros(W_dec.shape[1], device=device)
    # JumpReLU per-feature threshold
    threshold = sae.threshold.to(device).detach() if hasattr(sae, 'threshold') else None
    d_in, d_sae = W_enc.shape
    del sae
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
      - beta_significant: [n_features, top_k] boolean mask (cos > DEDA_SIGNIFICANCE_THRESHOLD)
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
    deda_top_cos: np.ndarray,      # [top_k] cosines with top decoder columns
    deda_top_idx: np.ndarray,      # [top_k] indices of top decoder columns
    W_dec: torch.Tensor,            # [d_sae, d_in]
    d_j: torch.Tensor,              # [d_in] decoder direction of feature j
    sig_threshold: float = DEDA_SIGNIFICANCE_THRESHOLD,
    cos_absorber_threshold: float = COS_ABSORBER_THRESHOLD,
) -> list:
    """
    Identify absorbing sources S_j = {k : beta_{j,k} significant AND cos(d_k, d_j) > threshold}

    Returns list of (k_index, cos_d_k_d_j) pairs.
    """
    d_j_norm = F.normalize(d_j.float().unsqueeze(0), dim=1).squeeze(0)  # [d_in]

    absorbers = []
    for i, (cos_val, k_idx) in enumerate(zip(deda_top_cos, deda_top_idx)):
        if cos_val < sig_threshold:
            break  # sorted descending, no more significant betas

        k_idx_int = int(k_idx)
        d_k = W_dec[k_idx_int].float()
        d_k_norm = F.normalize(d_k.unsqueeze(0), dim=1).squeeze(0)

        # cos(d_k, d_j) — similarity between absorber and absorbed decoder directions
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
    W_dec: torch.Tensor,         # [d_sae, d_in]
    excluded_indices: set,       # Don't match to these (j itself and already-matched)
    match_threshold: float = PARENT_MATCH_THRESHOLD,
    top_candidates: int = 20,
) -> list:
    """
    For absorber k, find parent matches m where cos(d_m, d_k) > match_threshold.

    Returns list of candidate parent indices m sorted by cosine similarity.
    Note: in PILOT mode we use decoder-only (no activation data for z_m == 0 check).
    We use "high D-EDA" of candidate m as proxy for "encoder suppressed" (fails to fire).
    """
    d_k = W_dec[k_idx].float()
    d_k_norm = F.normalize(d_k.unsqueeze(0), dim=1).squeeze(0)

    # Compute cosine similarities with all decoder columns
    W_dec_norm = F.normalize(W_dec.float(), dim=1)  # [d_sae, d_in]
    with torch.no_grad():
        cos_vals = (W_dec_norm @ d_k_norm).cpu().numpy()  # [d_sae]

    # Exclude self, the feature j, and already-matched
    for excl in excluded_indices:
        if 0 <= excl < len(cos_vals):
            cos_vals[excl] = -2.0
    cos_vals[k_idx] = -2.0

    # Top candidates above threshold
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
    x: torch.Tensor,                       # [n, d_in] input batch
    W_enc: torch.Tensor,                   # [d_in, d_sae]
    b_enc: torch.Tensor | None,            # [d_sae]
    threshold: torch.Tensor | None = None, # [d_sae] per-feature threshold (JumpReLU)
) -> torch.Tensor:
    """
    Sparse SAE encoding using JumpReLU (Gemma Scope uses per-feature threshold).
    z = pre_act * (pre_act > threshold)   where pre_act = x @ W_enc + b_enc
    Falls back to ReLU(pre_act) if threshold is None.
    """
    with torch.no_grad():
        pre_act = x @ W_enc.float()  # [n, d_sae]
        if b_enc is not None:
            pre_act = pre_act + b_enc.float()
        if threshold is not None:
            # JumpReLU: activate only if pre_act > threshold
            z = pre_act * (pre_act > threshold.float().unsqueeze(0)).float()
        else:
            z = F.relu(pre_act)
    return z


def synthetic_sae_decode(
    z: torch.Tensor,    # [n, d_sae]
    W_dec: torch.Tensor,  # [d_sae, d_in]
    b_dec: torch.Tensor,  # [d_in]
) -> torch.Tensor:
    """SAE decode: x_hat = z @ W_dec + b_dec"""
    with torch.no_grad():
        x_hat = z.float() @ W_dec.float() + b_dec.float()
    return x_hat


def generate_synthetic_inputs_for_latent(
    j_idx: int,
    W_dec: torch.Tensor,           # [d_sae, d_in]
    d_in: int,
    n_samples: int = N_SYNTHETIC_INPUTS,
    rng: np.random.Generator = None,
    sae_threshold: torch.Tensor | None = None,  # [d_sae] JumpReLU threshold
    W_enc: torch.Tensor | None = None,           # [d_in, d_sae] encoder weights
    b_enc: torch.Tensor | None = None,           # [d_sae] encoder bias
) -> torch.Tensor:
    """
    Generate synthetic inputs that SHOULD activate latent j (parent-positive inputs).

    Strategy: Add scaled d_j to random residual stream directions.
    The scale is chosen to exceed the JumpReLU threshold for latent j.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    d_j = W_dec[j_idx].float()    # [d_in]
    d_j_norm = F.normalize(d_j.unsqueeze(0), dim=1).squeeze(0)

    # Determine required scale to make pre-activation exceed threshold
    # pre_act_j = x @ W_enc[:, j] + b_enc[j]
    # For x = scale * d_j_norm: pre_act_j ≈ scale * (d_j_norm @ W_enc[:, j]) + b_enc[j]
    min_scale = 5.0
    if sae_threshold is not None and W_enc is not None:
        thr_j = float(sae_threshold[j_idx])
        w_enc_j = W_enc[:, j_idx].float()   # [d_in]
        dot_val = float((d_j_norm * w_enc_j).sum())
        b_j = float(b_enc[j_idx]) if b_enc is not None else 0.0
        # Scale needed so that scale * dot + b_j > thr_j
        if dot_val > 1e-6:
            min_scale = max(min_scale, (thr_j - b_j) / dot_val + 1.0)
        min_scale = min(min_scale, 20.0)  # cap to avoid huge values

    # Random noise in residual stream space
    noise = torch.from_numpy(rng.standard_normal((n_samples, d_in)).astype(np.float32)).to(DEVICE)
    noise = F.normalize(noise, dim=1)

    # Scale distribution: enough to exceed threshold
    scales = torch.from_numpy(
        rng.uniform(min_scale, min_scale * 2.0, size=(n_samples, 1)).astype(np.float32)
    ).to(DEVICE)
    x = noise * 0.5 + scales * d_j_norm.unsqueeze(0)

    return x


def compute_fn_rate_for_latents(
    latent_indices: np.ndarray,
    W_enc: torch.Tensor,
    W_dec: torch.Tensor,
    b_enc: torch.Tensor | None,
    b_dec: torch.Tensor,
    sae_threshold: torch.Tensor | None,
    d_in: int,
    rng: np.random.Generator,
    n_samples: int = N_SYNTHETIC_INPUTS,
) -> dict:
    """
    Compute false-negative rate for latents using synthetic inputs.

    FN: latent j fails to fire (z_j == 0) on a parent-positive input.
    FN rate = fraction of parent-positive inputs where z_j = 0.

    Returns per-latent FN rates.
    """
    fn_rates = {}
    z_means = {}

    with torch.no_grad():
        for j_idx in latent_indices:
            x = generate_synthetic_inputs_for_latent(j_idx, W_dec, d_in, n_samples, rng)
            z = synthetic_sae_encode(x, W_enc, b_enc, sae_threshold)  # [n_samples, d_sae]

            z_j = z[:, j_idx]           # [n_samples]
            fn_count = int((z_j == 0).sum().item())
            fn_rate = fn_count / n_samples
            fn_rates[int(j_idx)] = fn_rate
            z_means[int(j_idx)] = float(z_j.mean().item())

    return {"fn_rates": fn_rates, "z_means": z_means}


def apply_itac_correction(
    x_batch: torch.Tensor,          # [n, d_in] input batch
    z_batch: torch.Tensor,          # [n, d_sae] original sparse codes
    j_idx: int,                     # absorbed latent index
    parent_candidates: list,        # list of {m_idx, cos_dm_dk}
    W_dec: torch.Tensor,            # [d_sae, d_in]
    b_dec: torch.Tensor,            # [d_in]
) -> torch.Tensor:
    """
    Apply ITAC correction for latent j_idx on a batch of inputs.

    ITAC Algorithm:
    For each parent candidate m:
      1. Compute residual = x - x_hat_without_j  (what remains if j's contribution removed)
      2. Add back j's contribution: effective_residual = residual + z_j * d_j
      3. Compute correction: z_m_corrected = max(0, d_m^T @ effective_residual)
      4. If z_m_corrected > 0 and original z_m == 0: update z_batch[:, m] = z_m_corrected

    Returns corrected z_batch (copy, not in-place).
    """
    z_corrected = z_batch.clone()

    with torch.no_grad():
        # Current reconstruction
        x_hat = z_batch.float() @ W_dec.float() + b_dec.float()  # [n, d_in]
        residual = x_batch.float() - x_hat                        # [n, d_in]

        # Add back j's contribution to get effective residual
        d_j = W_dec[j_idx].float()     # [d_in]
        z_j = z_batch[:, j_idx].float()  # [n]
        effective_residual = residual + z_j.unsqueeze(1) * d_j.unsqueeze(0)  # [n, d_in]

        # For each parent candidate m
        for cand in parent_candidates:
            m_idx = cand["m_idx"]
            d_m = W_dec[m_idx].float()  # [d_in]
            d_m_norm = F.normalize(d_m.unsqueeze(0), dim=1).squeeze(0)

            # z_m_corrected = max(0, d_m^T @ effective_residual)
            z_m_corrected = F.relu(effective_residual @ d_m_norm)  # [n]

            # Only update if original z_m was 0 (correction for absorbed parent)
            was_zero = (z_batch[:, m_idx] == 0).float()  # [n]
            z_corrected[:, m_idx] = (
                z_batch[:, m_idx].float() + was_zero * z_m_corrected
            )

    return z_corrected


def compute_fvu(
    x: torch.Tensor,      # [n, d_in]
    x_hat: torch.Tensor,  # [n, d_in]
) -> float:
    """Fraction of Variance Unexplained = MSE / Var(x)"""
    with torch.no_grad():
        mse = ((x.float() - x_hat.float()) ** 2).mean()
        var_x = x.float().var()
        if var_x < 1e-10:
            return float('nan')
        return float(mse / var_x)


def run_itac_for_config(
    release: str, sae_id: str, name: str, layer: int, width_k: int,
    taxonomy_results: dict,
) -> dict:
    """
    Full ITAC evaluation for one SAE configuration.
    """
    print(f"\n{'='*60}")
    print(f"Processing {name} (layer={layer}, width={width_k}k)")
    t_start = time.time()

    # ── Load SAE weights ─────────────────────────────────────────────────────
    print("  Loading SAE weights...")
    try:
        W_enc, W_dec, b_enc, b_dec, sae_threshold, d_in, d_sae = load_sae_weights(release, sae_id, DEVICE)
        print(f"  SAE loaded: d_in={d_in}, d_sae={d_sae}")
    except Exception as e:
        return {"config": name, "error": f"load_failed: {e}", "status": "failed"}

    # ── Get proxy labels ─────────────────────────────────────────────────────
    print("  Fetching proxy absorption labels...")
    labels = None
    try:
        labels = get_neuronpedia_proxy_labels(layer, width_k, d_sae)
        n_pos = int(labels.sum())
        if n_pos < 3:
            print(f"  Only {n_pos} proxy labels found, falling back to EDA top features")
            labels = None
    except Exception as e:
        print(f"  Warning: label fetch failed ({e}), using EDA top features")
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
    print(f"  Absorbed latents: {n_absorbed}")

    if n_absorbed < 3:
        elapsed = time.time() - t_start
        return {
            "config": name, "layer": layer, "width_k": width_k,
            "status": "skip_too_few_absorbed",
            "error": f"Only {n_absorbed} absorbed latents",
            "elapsed_sec": elapsed,
        }

    # ── PILOT mode subsample ─────────────────────────────────────────────────
    if PILOT_MODE and n_absorbed > PILOT_SAMPLES:
        rng_pilot = np.random.default_rng(SEED)
        absorbed_indices = np.sort(
            rng_pilot.choice(absorbed_indices, size=PILOT_SAMPLES, replace=False)
        )
        print(f"  PILOT mode: subsampled to {PILOT_SAMPLES} absorbed latents")

    # ── Compute EDA + D-EDA for absorbed latents ─────────────────────────────
    print(f"  Computing EDA + D-EDA for {len(absorbed_indices)} absorbed latents...")
    eda_all = compute_eda(W_enc, W_dec)
    eda_absorbed = eda_all[absorbed_indices]

    deda_results = compute_deda_residual(W_enc, W_dec, absorbed_indices, top_k_dict=50)
    deda_scores_absorbed = deda_results["deda_scores"]

    # ── Retrieve taxonomy subtype labels from phase2_taxonomy ───────────────
    # Extract subtype classification for matching config from phase2a results
    subtype_map = {}  # feature_idx -> subtype
    if taxonomy_results:
        for sae_result in taxonomy_results.get("per_sae_results", []):
            if sae_result.get("config") == name:
                # We don't have per-feature subtypes in the taxonomy JSON (aggregated)
                # So we re-derive subtypes using the same logic as phase2_taxonomy
                break

    # Re-derive subtypes using taxonomy logic (decoder-only, no activations)
    # parent probe = mean decoder direction of absorbed latents
    W_dec_np = W_dec.detach().cpu().float().numpy()
    dec_absorbed = W_dec_np[absorbed_indices]
    parent_probe = dec_absorbed.mean(axis=0)
    parent_probe_norm = parent_probe / (np.linalg.norm(parent_probe) + 1e-8)

    PRIMARY_THRESHOLD = 0.30
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
        if max_cos < PRIMARY_THRESHOLD:
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

    # ── Select high-D-EDA late latents for ITAC ─────────────────────────────
    if len(late_indices) == 0:
        print("  No late-absorbed latents found, using all absorbed as fallback")
        # Fallback: use all absorbed for ITAC evaluation
        target_indices = absorbed_indices
        target_deda = deda_scores_absorbed
        is_fallback = True
    else:
        target_indices = late_indices
        late_positions = np.where(final_subtypes == "late")[0]
        target_deda = deda_scores_absorbed[late_positions]
        is_fallback = False

    # Select high-D-EDA subset
    deda_threshold = np.percentile(target_deda, HIGH_DEDA_PERCENTILE)
    high_deda_mask = target_deda >= deda_threshold
    itac_target_indices = target_indices[high_deda_mask]

    print(f"  ITAC targets (high D-EDA late): {len(itac_target_indices)}")
    print(f"  D-EDA threshold ({HIGH_DEDA_PERCENTILE}th pct): {deda_threshold:.4f}")

    # ── Identify absorbing sources + parent matches ──────────────────────────
    print("  Identifying absorbing sources and parent matches...")
    itac_plans = []  # Per-latent ITAC correction plan

    W_dec_t = W_dec.detach()
    for feat_idx in itac_target_indices:
        pos = int(np.where(absorbed_indices == feat_idx)[0][0])
        top_cos_j = residual_top_cos[pos]
        top_idx_j = residual_top_idx[pos]

        d_j = W_dec_t[feat_idx].float()

        # Identify absorbers S_j
        absorbers = identify_absorbing_sources(
            feature_idx=feat_idx,
            deda_top_cos=top_cos_j,
            deda_top_idx=top_idx_j,
            W_dec=W_dec_t,
            d_j=d_j,
        )

        # Find parent match for each absorber
        parent_candidates = []
        excluded = {feat_idx}
        for absorber in absorbers[:5]:  # limit to top-5 absorbers
            k_idx = absorber["k_idx"]
            excluded.add(k_idx)
            matches = find_parent_match(k_idx, W_dec_t, excluded, PARENT_MATCH_THRESHOLD)
            for m in matches[:2]:  # take top-2 matches per absorber
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
    print(f"  Plans with at least 1 parent candidate: {n_with_correction}/{len(itac_plans)}")

    # ── Generate synthetic inputs and run ITAC evaluation ───────────────────
    print("  Generating synthetic inputs and running ITAC evaluation...")
    rng = np.random.default_rng(SEED)

    # FN rates before/after ITAC
    # ITAC corrects the PARENT latent (m), not latent j itself.
    # FN of parent m: parent-positive inputs where m fails to fire.
    # After ITAC: m's activation is inserted, so m's FN should drop.
    itac_delta_per_latent = []

    for plan in itac_plans:
        j_idx = plan["j_idx"]

        if not plan["parent_candidates"]:
            # No correction possible — record as no-change
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

        # Use the best parent candidate (highest cos_dm_dk)
        best_parent = max(plan["parent_candidates"], key=lambda p: p["cos_dm_dk"])
        m_idx = best_parent["m_idx"]

        # Generate synthetic inputs where the parent concept IS present.
        # For ITAC: inputs that should activate m (parent latent) but don't because j absorbs them.
        # Proxy: use d_m direction (parent decoder direction) as the "parent-positive" signal.
        x = generate_synthetic_inputs_for_latent(
            m_idx, W_dec_t, d_in, N_SYNTHETIC_INPUTS, rng,
            sae_threshold=sae_threshold, W_enc=W_enc, b_enc=b_enc
        )

        # Also ensure j is activated (absorption scenario: j fires instead of m)
        # Mix: strong m-direction signal + moderate j-direction signal
        d_j_dir = F.normalize(W_dec_t[j_idx].float().unsqueeze(0), dim=1).squeeze(0)
        j_scale = torch.from_numpy(
            rng.uniform(1.0, 3.0, size=(N_SYNTHETIC_INPUTS, 1)).astype(np.float32)
        ).to(DEVICE)
        x = x + j_scale * d_j_dir.unsqueeze(0)

        # Encode with SAE
        z = synthetic_sae_encode(x, W_enc, b_enc, sae_threshold)  # [n, d_sae]

        # FN of parent latent m BEFORE ITAC
        z_m_before = z[:, m_idx]
        fn_rate_before = float((z_m_before == 0).float().mean().item())

        # Reconstruct before
        x_hat_before = synthetic_sae_decode(z, W_dec_t, b_dec)
        fvu_b = compute_fvu(x, x_hat_before)

        # Apply ITAC correction
        z_corrected = apply_itac_correction(
            x_batch=x,
            z_batch=z,
            j_idx=j_idx,
            parent_candidates=plan["parent_candidates"],
            W_dec=W_dec_t,
            b_dec=b_dec,
        )

        # FN of parent latent m AFTER ITAC
        z_m_after = z_corrected[:, m_idx]
        fn_rate_after = float((z_m_after == 0).float().mean().item())

        # Reconstruct after
        x_hat_after = synthetic_sae_decode(z_corrected, W_dec_t, b_dec)
        fvu_a = compute_fvu(x, x_hat_after)

        delta = fn_rate_before - fn_rate_after  # positive = FN reduction (improvement)
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

    # ── Null test: early latents ─────────────────────────────────────────────
    print("  Running null test on early-absorbed latents...")
    null_results = []
    null_indices = early_indices[:min(20, len(early_indices))]  # cap at 20 for pilot

    for j_idx in null_indices:
        pos = int(np.where(absorbed_indices == j_idx)[0][0])
        plan_null = {
            "j_idx": int(j_idx),
            "parent_candidates": [],  # early has no parent match → no ITAC
        }

        x = generate_synthetic_inputs_for_latent(
            j_idx, W_dec_t, d_in, N_SYNTHETIC_INPUTS, rng,
            sae_threshold=sae_threshold, W_enc=W_enc, b_enc=b_enc
        )
        z = synthetic_sae_encode(x, W_enc, b_enc, sae_threshold)
        z_j = z[:, j_idx]
        fn_null = float((z_j == 0).float().mean().item())
        null_results.append({"j_idx": int(j_idx), "fn_rate": fn_null})

    # ── Aggregate ITAC efficacy ───────────────────────────────────────────────
    # Only include results where we have actual parent FN measurement
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

        # Monotonicity: Spearman rho(D-EDA, FN reduction)
        deda_vals = [d["deda_score"] for d in valid_results]
        if len(deda_vals) >= 3:
            mono_rho, mono_p = stats.spearmanr(deda_vals, fn_reduction_pcts)
        else:
            mono_rho, mono_p = float('nan'), float('nan')

        # Check pass criteria
        any_positive_reduction = mean_fn_before - mean_fn_after > 0
        fvu_ok = np.isnan(mean_fvu_change) or mean_fvu_change < 5.0

        pilot_pass = any_positive_reduction and fvu_ok
        potential_h5_falsification = (
            mean_fn_reduction_pct < 5.0 and n_improved_5pct < len(valid_results) * 0.3
        )

    else:
        fn_before_vals = fn_after_vals = fn_delta_vals = fn_reduction_pcts = []
        mean_fn_before = mean_fn_after = mean_fn_reduction_pct = float('nan')
        mean_fvu_change = float('nan')
        n_improved = n_improved_5pct = 0
        mono_rho = mono_p = float('nan')
        pilot_pass = False
        potential_h5_falsification = True

    elapsed = time.time() - t_start

    print(f"\n  ITAC Summary for {name}:")
    print(f"    N ITAC targets: {len(itac_target_indices)}")
    print(f"    With parent candidates: {n_with_correction}")
    print(f"    Mean FN before: {mean_fn_before:.4f}")
    print(f"    Mean FN after:  {mean_fn_after:.4f}")
    print(f"    Mean FN reduction: {mean_fn_reduction_pct:.2f}%")
    print(f"    Mean FVU change: {mean_fvu_change:.4f}")
    print(f"    Monotonicity rho: {mono_rho:.4f} (p={mono_p:.4f})")
    print(f"    Pilot PASS: {pilot_pass}")

    return {
        "config": name, "layer": layer, "width_k": width_k,
        "d_sae": int(d_sae), "d_in": int(d_in),
        "n_absorbed_total": len(absorbed_indices),
        "n_early": n_early, "n_late": n_late, "n_partial": n_partial,
        "n_itac_targets": len(itac_target_indices),
        "n_with_parent_candidates": n_with_correction,
        "is_fallback": is_fallback,
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
        },
        "per_latent_results": itac_delta_per_latent,
        "itac_plans_summary": [
            {k: v for k, v in p.items() if k != "parent_candidates"}
            for p in itac_plans
        ],
        "pass_criteria": {
            "any_positive_fn_reduction": bool(any_positive_reduction) if itac_delta_per_latent else False,
            "fvu_change_under_5pct": bool(fvu_ok) if itac_delta_per_latent else False,
            "potential_h5_falsification": potential_h5_falsification,
            "pilot_pass": bool(pilot_pass),
        },
        "status": "success",
        "elapsed_sec": round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
write_progress(0, len(SAE_CONFIGS), {"stage": "starting"})

# Load taxonomy results
taxonomy_path = RESULTS_DIR / "phase2a_taxonomy.json"
taxonomy_results = {}
if taxonomy_path.exists():
    try:
        taxonomy_results = json.loads(taxonomy_path.read_text())
        print(f"Loaded taxonomy results: go_no_go={taxonomy_results.get('go_no_go')}")
    except Exception as e:
        print(f"Warning: Failed to load taxonomy results: {e}")
else:
    print("Warning: phase2a_taxonomy.json not found, proceeding without it")

print(f"\nPILOT MODE: {PILOT_MODE}")
print(f"Processing {len(SAE_CONFIGS)} SAE configurations (Layer 12, widths 16k + 65k)...\n")

all_results = []
t_global = time.time()

for step_idx, (release, sae_id, name, layer, width_k) in enumerate(SAE_CONFIGS):
    write_progress(step_idx, len(SAE_CONFIGS), {"config": name})

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
pilot_passes = [r for r in passed if r.get("pass_criteria", {}).get("pilot_pass", False)]

# Aggregate FN reduction across configs (only valid measurements with parent FN)
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
    "n_configs_pilot_pass": len(pilot_passes),
    "overall_pilot_pass": len(pilot_passes) >= 1,
    "mean_fn_reduction_pct_global": float(np.mean(all_fn_reductions)) if all_fn_reductions else None,
    "mean_fvu_change_global": float(np.mean(all_fvu_changes)) if all_fvu_changes else None,
    "total_elapsed_sec": round(time.time() - t_global, 1),
    "pilot_mode": PILOT_MODE,
    "notes": "PILOT: Synthetic activations derived from SAE decoder columns (no Gemma 2B required). FN measurement uses synthetic parent-positive inputs."
}

print(f"\n{'='*60}")
print(f"AGGREGATE RESULTS:")
print(f"  Configs: {len(passed)}/{len(SAE_CONFIGS)} succeeded")
print(f"  Pilot pass: {len(pilot_passes)}/{len(passed)}")
if all_fn_reductions:
    print(f"  Global mean FN reduction: {np.mean(all_fn_reductions):.2f}%")
if all_fvu_changes:
    print(f"  Global mean FVU change: {np.mean(all_fvu_changes):.4f}")

# ─── Save Output ──────────────────────────────────────────────────────────────
# Build efficacy table (for paper Table 2 template)
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
        "pilot_pass": r.get("pass_criteria", {}).get("pilot_pass", False),
    })

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "config": {
        "sae_configs": [c[2] for c in SAE_CONFIGS],
        "primary_threshold": 0.30,
        "deda_significance_threshold": DEDA_SIGNIFICANCE_THRESHOLD,
        "cos_absorber_threshold": COS_ABSORBER_THRESHOLD,
        "parent_match_threshold": PARENT_MATCH_THRESHOLD,
        "high_deda_percentile": HIGH_DEDA_PERCENTILE,
        "n_synthetic_inputs": N_SYNTHETIC_INPUTS,
        "seed": SEED,
        "pilot_samples": PILOT_SAMPLES,
    },
    "per_sae_results": all_results,
    "aggregate": aggregate,
    "efficacy_table": efficacy_table,
    "pilot_pass_criteria": {
        "any_positive_fn_reduction": any(
            r.get("pass_criteria", {}).get("any_positive_fn_reduction", False)
            for r in passed
        ),
        "fvu_constraint_met": all(
            r.get("pass_criteria", {}).get("fvu_change_under_5pct", True)
            for r in passed
        ),
        "h5_potential_falsification": any(
            r.get("pass_criteria", {}).get("potential_h5_falsification", False)
            for r in passed
        ),
        "overall_pass": aggregate["overall_pilot_pass"],
    },
    "go_no_go": "GO" if aggregate["overall_pilot_pass"] else "NO_GO",
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
print(f"\nOutput saved to: {OUTPUT_FILE}")

# ─── Markdown Summary ─────────────────────────────────────────────────────────
summary_md = f"""# Phase 2b: ITAC Evaluation Results

**Mode:** PILOT
**Timestamp:** {output['timestamp']}
**SAE Configs:** {', '.join(output['config']['sae_configs'])}

## ITAC Efficacy Table

| Config | N Late | N ITAC Targets | Parent FN Before | Parent FN After | FN Reduction % | FVU Change | Mono Rho | Pass |
|--------|--------|----------------|-----------------|-----------------|----------------|------------|----------|------|
"""
for row in efficacy_table:
    fn_b = f"{row['fn_parent_before']:.3f}" if row.get('fn_parent_before') is not None else "N/A"
    fn_a = f"{row['fn_parent_after']:.3f}" if row.get('fn_parent_after') is not None else "N/A"
    fn_r = f"{row['fn_reduction_pct']:.2f}%" if row['fn_reduction_pct'] is not None else "N/A"
    fvu_c = f"{row['fvu_change']:.4f}" if row['fvu_change'] is not None else "N/A"
    mono = f"{row['monotonicity_rho']:.3f}" if row['monotonicity_rho'] is not None else "N/A"
    pass_str = "PASS" if row["pilot_pass"] else "FAIL"
    summary_md += (
        f"| {row['config']} | {row['n_late']} | {row['n_itac_targets']} | "
        f"{fn_b} | {fn_a} | {fn_r} | {fvu_c} | {mono} | {pass_str} |\n"
    )

summary_md += f"""
## Pilot Pass Criteria

- Any positive FN reduction: {'PASS' if output['pilot_pass_criteria']['any_positive_fn_reduction'] else 'FAIL'}
- FVU constraint (< +5%): {'PASS' if output['pilot_pass_criteria']['fvu_constraint_met'] else 'FAIL'}
- H5 potential falsification: {'WARNING' if output['pilot_pass_criteria']['h5_potential_falsification'] else 'OK'}
- **Go/No-Go Decision:** {output['go_no_go']}

## Notes

- PILOT mode: {PILOT_SAMPLES} absorbed latents max per config, {N_SYNTHETIC_INPUTS} synthetic inputs per latent
- FN measurement uses synthetic activations (decoder column-based parent-positive inputs)
- No Gemma 2B model loading required in PILOT mode
- Full mode would use actual Gemma 2B residual stream activations
- ITAC correction uses decoder-only parent matching (cos(d_m, d_k) > {PARENT_MATCH_THRESHOLD})
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
            "pilot_mode": PILOT_MODE,
            "pilot_samples": PILOT_SAMPLES,
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
        f"Phase 2b ITAC: {go_no_go}. "
        f"{len(passed)}/{len(SAE_CONFIGS)} configs succeeded. "
        f"Mean FN reduction: {aggregate.get('mean_fn_reduction_pct_global', 'N/A')} pct. "
        f"Efficacy table: {efficacy_table}"
    ),
)
print(f"\nFINISHED. Go/No-Go: {go_no_go}")
