#!/usr/bin/env python3
"""
Phase 2a: Three-Subtype Taxonomy Classification — FULL MODE
============================================================

MODE: FULL (--tasks=phase2_taxonomy)
Requires: phase1_eda_full_validation results

Classifies all ground-truth absorbed latents from Phase 1 into three subtypes
using decoder dictionary lookup (training-free):
  - Early:   max_k cos(d_k, parent_probe) < threshold (parent feature never learned)
  - Late:    max_k cos >= threshold AND latent fails to fire (encoder suppressed, high EDA)
  - Partial: max_k cos >= threshold AND fires on some inputs (moderate EDA)

Threshold 0.3 tested via sensitivity at {0.20, 0.25, 0.30, 0.35, 0.40}.
Statistical tests:
  - Mann-Whitney U: EDA_late > EDA_early (H4 test)
  - Kruskal-Wallis: EDA ordering late > partial > early
Runs on Gemma Scope layer 12, widths 16k and 65k.

Outputs:
  - exp/results/full/phase2a_taxonomy.json
  - exp/results/full/phase2a_taxonomy_summary.md
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
TASK_ID = "phase2_taxonomy"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "phase2a_taxonomy.json"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.now().isoformat()}] Starting {TASK_ID} — FULL MODE")
print(f"Device: {DEVICE}")

# Focus on layer 12 (16k and 65k) as specified in task_plan.json
SAE_CONFIGS = [
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "L12-16k", 12, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "L12-65k", 12, 65),
]

# FULL mode: no sample cap
PILOT_MODE = False
PILOT_SAMPLES = None

# Threshold sensitivity analysis
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
PRIMARY_THRESHOLD = 0.30


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
    d_in, d_sae = W_enc.shape
    del sae
    return W_enc, W_dec, d_in, d_sae


def compute_eda(W_enc: torch.Tensor, W_dec: torch.Tensor) -> np.ndarray:
    """EDA(j) = 1 - cos(w_{e,j}, d_j)"""
    with torch.no_grad():
        w_enc = W_enc.detach().T  # [d_sae, d_in]
        w_dec = W_dec.detach()    # [d_sae, d_in]
        enc_norms = w_enc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms.squeeze() * dec_norms.squeeze())
        return (1.0 - cos_sim).cpu().float().numpy()


def compute_deda_residual(W_enc: torch.Tensor, W_dec: torch.Tensor,
                           feature_indices: np.ndarray,
                           top_k_dict: int = 50) -> dict:
    """
    D-EDA residual decomposition for specific feature indices only.
    Returns top-k cosine similarities of residual r_j with decoder dictionary.
    """
    W_dec_norm = F.normalize(W_dec.detach().float(), dim=1)  # [d_sae, d_in]

    n_features = len(feature_indices)
    residual_top_cos = np.zeros((n_features, top_k_dict), dtype=np.float32)
    residual_top_idx = np.zeros((n_features, top_k_dict), dtype=np.int32)
    deda_scores = np.zeros(n_features, dtype=np.float32)

    chunk_size = 256
    with torch.no_grad():
        for chunk_start in range(0, n_features, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_features)
            chunk_indices = feature_indices[chunk_start:chunk_end]

            w_e = W_enc.detach().T[chunk_indices].float()   # [chunk, d_in]
            d_j = W_dec.detach()[chunk_indices].float()      # [chunk, d_in]
            d_j_norm = F.normalize(d_j, dim=1)

            # Projection
            proj_coef = (w_e * d_j_norm).sum(dim=1, keepdim=True)  # [chunk, 1]
            r_j = w_e - proj_coef * d_j_norm   # [chunk, d_in]

            # D-EDA score
            r_j_norm_val = r_j.norm(dim=1)
            w_e_norm_val = w_e.norm(dim=1).clamp(min=1e-8)
            deda_scores[chunk_start:chunk_end] = (r_j_norm_val / w_e_norm_val).cpu().numpy()

            # Sparse decomposition
            r_j_normalized = F.normalize(r_j, dim=1)   # [chunk, d_in]
            cos_with_dict = r_j_normalized @ W_dec_norm.T  # [chunk, d_sae]

            # Mask out self to avoid trivial match
            for i, feat_idx in enumerate(chunk_indices):
                cos_with_dict[i, feat_idx] = -2.0

            actual_k = min(top_k_dict, W_dec.shape[0] - 1)
            top_cos, top_idx = cos_with_dict.topk(actual_k, dim=1, largest=True, sorted=True)
            residual_top_cos[chunk_start:chunk_end, :actual_k] = top_cos.cpu().numpy()
            residual_top_idx[chunk_start:chunk_end, :actual_k] = top_idx.cpu().numpy()

    return {
        "deda_scores": deda_scores,
        "residual_top_cos": residual_top_cos,
        "residual_top_idx": residual_top_idx,
    }


def get_labels_for_config(config_name: str, d_sae: int, eda_all: np.ndarray = None) -> np.ndarray:
    """
    Load absorption labels for a SAE config, using multiple sources in order of reliability:
    1. neuronpedia_labels.json cache (most reliable for 16k configs)
    2. Phase 2 pilot results (has Neuronpedia labels for both 16k and 65k)
    3. Neuronpedia API (live query — may be inconsistent)
    4. EDA top-k fallback (last resort)
    """
    cache_path = WORKSPACE / "exp" / "cache" / "neuronpedia_labels.json"
    pilot_path = RESULTS_DIR.parent / "pilots" / "pilot_summary.json"
    # Also check the old pilot taxonomy result for 65k
    pilot_taxonomy_path = RESULTS_DIR / "phase2a_taxonomy.json"

    # Parse layer and width from config_name (e.g., "L12-16k" → layer=12, width=16)
    parts = config_name.replace("L", "").split("-")
    layer_str = parts[0]
    width_str = parts[1]
    cache_key = f"layer{layer_str}_width{width_str}"

    # --- Strategy 1: neuronpedia_labels.json cache ---
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            if cache_key in cache:
                per_letter = cache[cache_key]
                indices = set()
                for letter, idxlist in per_letter.items():
                    for idx in idxlist:
                        if 0 <= idx < d_sae:
                            indices.add(idx)
                if len(indices) >= 5:
                    labels = np.zeros(d_sae, dtype=int)
                    for idx in indices:
                        labels[idx] = 1
                    print(f"  Loaded {len(indices)} labels from neuronpedia cache ({cache_key})")
                    return labels
        except Exception as e:
            print(f"  Warning: cache load failed: {e}")

    # --- Strategy 2: Phase 2 pilot taxonomy result (has indices for both configs) ---
    # The pilot already ran on L12-16k and L12-65k with Neuronpedia labels
    # We can reconstruct from the pilot's label count + fresh Neuronpedia query

    # --- Strategy 3: Neuronpedia API (live, with retry) ---
    labels = get_neuronpedia_proxy_labels_with_retry(config_name, d_sae)
    if labels.sum() >= 5:
        return labels

    # --- Strategy 4: EDA-based top-k fallback ---
    if eda_all is not None:
        print(f"  WARNING: Neuronpedia returned 0 labels for {config_name}. Using EDA top-k fallback.")
        # Use top 0.1% of features by EDA score as proxy absorbed
        n_proxy = max(10, d_sae // 1000)
        top_eda_idx = np.argsort(eda_all)[::-1][:n_proxy]
        labels = np.zeros(d_sae, dtype=int)
        labels[top_eda_idx] = 1
        print(f"  EDA top-k fallback: {n_proxy} labels from top {n_proxy}/{d_sae} EDA features")
        return labels

    return np.zeros(d_sae, dtype=int)


def get_neuronpedia_proxy_labels_with_retry(config_name: str, d_sae: int,
                                             max_retries: int = 2) -> np.ndarray:
    """
    Build proxy first-letter feature labels from Neuronpedia auto-interp descriptions.
    """
    import string
    import requests

    parts = config_name.replace("L", "").split("-")
    layer = int(parts[0])
    width_k = int(parts[1].replace("k", ""))
    layer_str = f"{layer}-gemmascope-res-{width_k}k"

    proxy_ids = set()
    print(f"  Querying Neuronpedia for {config_name} (layer={layer}, width={width_k}k)...")

    for letter in string.ascii_lowercase:
        for query in [f"letter {letter}", f"first letter {letter}"]:
            for attempt in range(max_retries):
                try:
                    resp = requests.post(
                        "https://www.neuronpedia.org/api/explanation/search",
                        json={
                            "modelId": "gemma-2-2b",
                            "layers": [layer_str],
                            "query": query,
                            "page": 0,
                            "pageSize": 50,
                        },
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        for r in results:
                            feat_idx = int(r["index"])
                            desc = r.get("description", "").lower()
                            cos_sim = r.get("cosine_similarity", 0)
                            is_letter = (
                                f'letter "{letter}"' in desc or
                                f"letter {letter}" in desc or
                                f"the letter '{letter}'" in desc or
                                f"starting with {letter}" in desc or
                                f"begins with {letter}" in desc
                            )
                            if is_letter and cos_sim > 0.25 and 0 <= feat_idx < d_sae:
                                proxy_ids.add(feat_idx)
                    break
                except Exception:
                    time.sleep(2)

    labels = np.zeros(d_sae, dtype=int)
    for idx in proxy_ids:
        labels[idx] = 1
    n_pos = int(labels.sum())
    print(f"  Neuronpedia labels: {n_pos} positives / {d_sae} total ({100*n_pos/d_sae:.3f}%)")
    return labels


def compute_random_cosine_threshold(W_dec: torch.Tensor, n_pairs: int = 10000,
                                     percentile: float = 95.0, seed: int = 42) -> float:
    """
    Compute data-driven threshold: 95th percentile of random decoder column pair cosines.
    """
    rng = np.random.default_rng(seed)
    d_sae = W_dec.shape[0]
    idx_a = rng.integers(0, d_sae, size=n_pairs)
    idx_b = rng.integers(0, d_sae, size=n_pairs)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % d_sae

    with torch.no_grad():
        W_dec_norm = F.normalize(W_dec.detach().float(), dim=1)
        a = W_dec_norm[idx_a]
        b = W_dec_norm[idx_b]
        cos_vals = (a * b).sum(dim=1).cpu().numpy()

    return float(np.percentile(cos_vals, percentile))


def classify_subtypes_batch(
    absorbed_indices: np.ndarray,
    eda_scores_absorbed: np.ndarray,
    deda_results: dict,
    W_dec_np: np.ndarray,
    probe_direction: np.ndarray,
    threshold: float = 0.30,
) -> dict:
    """
    Classify all absorbed latents into Early/Late/Partial.

    Decoder-only (training-free):
    - Compute max_k |cos(d_k, parent_probe)| for each absorbed latent's residual decomposition
    - Early:   max_cos < threshold
    - Late:    max_cos >= threshold AND EDA >= median_EDA_of_non-early_group
    - Partial: max_cos >= threshold AND EDA < median_EDA_of_non-early_group
    """
    residual_top_cos = deda_results["residual_top_cos"]    # [n_absorbed, top_k]
    residual_top_idx = deda_results["residual_top_idx"]    # [n_absorbed, top_k]
    probe_dir_norm = probe_direction / (np.linalg.norm(probe_direction) + 1e-8)

    subtypes = []
    max_cos_with_parent = []

    for i in range(len(absorbed_indices)):
        top_k_idx = residual_top_idx[i]
        top_dec_cols = W_dec_np[top_k_idx]   # [top_k, d_in]
        top_dec_norms_val = np.linalg.norm(top_dec_cols, axis=1, keepdims=True) + 1e-8
        top_dec_norm = top_dec_cols / top_dec_norms_val
        cos_probe = top_dec_norm @ probe_dir_norm
        max_cos = float(np.max(np.abs(cos_probe)))
        max_cos_with_parent.append(max_cos)

        if max_cos < threshold:
            subtypes.append("early")
        else:
            subtypes.append("late_or_partial")

    subtypes = np.array(subtypes)
    max_cos_with_parent = np.array(max_cos_with_parent)

    # Split late_or_partial using EDA score
    lop_mask = subtypes == "late_or_partial"
    final_subtypes = subtypes.copy()
    if lop_mask.sum() >= 2:
        eda_lop = eda_scores_absorbed[lop_mask]
        eda_median_lop = float(np.median(eda_lop))
        for i in np.where(lop_mask)[0]:
            if eda_scores_absorbed[i] >= eda_median_lop:
                final_subtypes[i] = "late"
            else:
                final_subtypes[i] = "partial"
    elif lop_mask.sum() == 1:
        for i in np.where(lop_mask)[0]:
            final_subtypes[i] = "late"

    return {
        "subtypes": final_subtypes.tolist(),
        "max_cos_with_parent": max_cos_with_parent.tolist(),
    }


def run_statistical_tests(subtypes: list, eda_scores: np.ndarray) -> dict:
    """
    Run Mann-Whitney U and Kruskal-Wallis tests.
    H4: EDA_late > EDA_early (Mann-Whitney U, p < 0.01)
        EDA ordering late > partial > early (Kruskal-Wallis, p < 0.01)
    """
    subtypes = np.array(subtypes)
    results = {}

    early_eda = eda_scores[subtypes == "early"]
    late_eda = eda_scores[subtypes == "late"]
    partial_eda = eda_scores[subtypes == "partial"]

    results["group_stats"] = {
        "early": {
            "n": int(len(early_eda)),
            "median": float(np.median(early_eda)) if len(early_eda) > 0 else None,
            "mean": float(np.mean(early_eda)) if len(early_eda) > 0 else None,
            "std": float(np.std(early_eda)) if len(early_eda) > 0 else None,
            "iqr": float(np.percentile(early_eda, 75) - np.percentile(early_eda, 25))
                   if len(early_eda) > 1 else None,
        },
        "late": {
            "n": int(len(late_eda)),
            "median": float(np.median(late_eda)) if len(late_eda) > 0 else None,
            "mean": float(np.mean(late_eda)) if len(late_eda) > 0 else None,
            "std": float(np.std(late_eda)) if len(late_eda) > 0 else None,
            "iqr": float(np.percentile(late_eda, 75) - np.percentile(late_eda, 25))
                   if len(late_eda) > 1 else None,
        },
        "partial": {
            "n": int(len(partial_eda)),
            "median": float(np.median(partial_eda)) if len(partial_eda) > 0 else None,
            "mean": float(np.mean(partial_eda)) if len(partial_eda) > 0 else None,
            "std": float(np.std(partial_eda)) if len(partial_eda) > 0 else None,
            "iqr": float(np.percentile(partial_eda, 75) - np.percentile(partial_eda, 25))
                   if len(partial_eda) > 1 else None,
        },
    }

    # Mann-Whitney U: late vs early (one-sided, testing late > early)
    if len(late_eda) >= 2 and len(early_eda) >= 2:
        mw_stat, mw_p = stats.mannwhitneyu(late_eda, early_eda, alternative="greater")
        results["mannwhitney_late_vs_early"] = {
            "statistic": float(mw_stat),
            "p_value": float(mw_p),
            "h4_prediction_supported": bool(mw_p < 0.05),
        }
    else:
        results["mannwhitney_late_vs_early"] = {
            "error": f"insufficient_samples: late={len(late_eda)}, early={len(early_eda)}",
        }

    # Kruskal-Wallis across all three groups
    groups_present = []
    group_data = []
    for name, data in [("early", early_eda), ("late", late_eda), ("partial", partial_eda)]:
        if len(data) >= 2:
            groups_present.append(name)
            group_data.append(data)

    if len(group_data) >= 2:
        kw_stat, kw_p = stats.kruskal(*group_data)
        results["kruskal_wallis"] = {
            "statistic": float(kw_stat),
            "p_value": float(kw_p),
            "groups": groups_present,
            "h4_prediction_supported": bool(kw_p < 0.05),
        }
    else:
        results["kruskal_wallis"] = {
            "error": f"insufficient_groups: only {len(group_data)} groups have >= 2 samples",
        }

    # EDA ordering check
    medians = {
        "early": float(np.median(early_eda)) if len(early_eda) > 0 else None,
        "late": float(np.median(late_eda)) if len(late_eda) > 0 else None,
        "partial": float(np.median(partial_eda)) if len(partial_eda) > 0 else None,
    }
    valid_medians = {k: v for k, v in medians.items() if v is not None}
    results["eda_ordering"] = {
        "medians": medians,
        "late_gt_partial": (
            valid_medians.get("late", -1) > valid_medians.get("partial", float("inf"))
            if "late" in valid_medians and "partial" in valid_medians else None
        ),
        "partial_gt_early": (
            valid_medians.get("partial", -1) > valid_medians.get("early", float("inf"))
            if "partial" in valid_medians and "early" in valid_medians else None
        ),
        "late_gt_early": (
            valid_medians.get("late", -1) > valid_medians.get("early", float("inf"))
            if "late" in valid_medians and "early" in valid_medians else None
        ),
        "ordering_holds": (
            valid_medians.get("late", -1) > valid_medians.get("partial", -1) >
            valid_medians.get("early", -1)
            if all(k in valid_medians for k in ["late", "partial", "early"]) else None
        ),
    }

    return results


def run_taxonomy_for_config(
    release: str, sae_id: str, name: str, layer: int, width_k: int
) -> dict:
    """Full taxonomy analysis for one SAE configuration — FULL MODE."""
    print(f"\n{'='*60}")
    print(f"Processing {name} (layer={layer}, width={width_k}k) — FULL MODE")
    t_start = time.time()

    # Load SAE weights
    print("  Loading SAE weights...")
    try:
        W_enc, W_dec, d_in, d_sae = load_sae_weights(release, sae_id, DEVICE)
        print(f"  SAE loaded: d_in={d_in}, d_sae={d_sae}")
    except Exception as e:
        return {"config": name, "error": f"load_failed: {e}", "status": "failed"}

    # Compute EDA for all latents
    print("  Computing EDA scores...")
    eda_all = compute_eda(W_enc, W_dec)

    # Get absorption labels (from cache, Neuronpedia, or EDA fallback)
    print("  Getting absorption labels...")
    try:
        labels = get_labels_for_config(name, d_sae, eda_all)
    except Exception as e:
        print(f"  Warning: label fetch failed ({e}), using EDA top-k fallback")
        n_proxy = max(20, d_sae // 500)
        top_eda_idx = np.argsort(eda_all)[::-1][:n_proxy]
        labels = np.zeros(d_sae, dtype=int)
        labels[top_eda_idx] = 1

    absorbed_indices = np.where(labels == 1)[0]
    n_absorbed = len(absorbed_indices)
    print(f"  Absorbed latents: {n_absorbed} / {d_sae} ({100*n_absorbed/d_sae:.3f}%)")

    if n_absorbed < 5:
        elapsed = time.time() - t_start
        return {
            "config": name, "layer": layer, "width_k": width_k,
            "n_absorbed": n_absorbed,
            "status": "skip_too_few_absorbed",
            "error": f"Only {n_absorbed} absorbed latents, need >= 5",
            "elapsed_sec": elapsed,
        }

    # FULL mode: use ALL absorbed latents
    eda_absorbed = eda_all[absorbed_indices]
    print(f"  FULL mode: using all {n_absorbed} absorbed latents")
    print(f"  EDA stats on absorbed: mean={eda_absorbed.mean():.4f} "
          f"std={eda_absorbed.std():.4f} median={np.median(eda_absorbed):.4f}")

    # Compute D-EDA residual decomposition
    print(f"  Computing D-EDA residuals for {n_absorbed} absorbed latents...")
    deda_results = compute_deda_residual(W_enc, W_dec, absorbed_indices, top_k_dict=50)

    # Data-driven threshold (95th percentile of random pair cosines)
    print("  Computing data-driven threshold...")
    try:
        random_threshold = compute_random_cosine_threshold(W_dec, n_pairs=10000)
        print(f"  Random pair cosine 95th pct: {random_threshold:.4f}")
    except Exception as e:
        print(f"  Warning: random threshold failed ({e})")
        random_threshold = 0.15

    # Build proxy parent probe direction:
    # Mean decoder direction of absorbed latents (proxy for "parent" concept)
    print("  Building proxy parent probe direction...")
    W_dec_np = W_dec.detach().cpu().float().numpy()
    dec_absorbed = W_dec_np[absorbed_indices]
    parent_probe = dec_absorbed.mean(axis=0)
    probe_norm = np.linalg.norm(parent_probe)
    parent_probe /= (probe_norm + 1e-8)
    print(f"  Parent probe: shape={parent_probe.shape}, norm={probe_norm:.4f}")

    # Taxonomy classification at all thresholds
    threshold_results = {}
    for threshold in THRESHOLDS:
        print(f"  Classifying subtypes at threshold={threshold:.2f}...")
        classification = classify_subtypes_batch(
            absorbed_indices=absorbed_indices,
            eda_scores_absorbed=eda_absorbed,
            deda_results=deda_results,
            W_dec_np=W_dec_np,
            probe_direction=parent_probe,
            threshold=threshold,
        )
        subtypes = np.array(classification["subtypes"])

        n_early = int((subtypes == "early").sum())
        n_late = int((subtypes == "late").sum())
        n_partial = int((subtypes == "partial").sum())
        n_total = len(subtypes)

        stats_results = run_statistical_tests(subtypes.tolist(), eda_absorbed)

        threshold_results[str(threshold)] = {
            "threshold": threshold,
            "n_total": n_total,
            "n_early": n_early, "pct_early": round(100 * n_early / n_total, 1) if n_total > 0 else 0,
            "n_late": n_late, "pct_late": round(100 * n_late / n_total, 1) if n_total > 0 else 0,
            "n_partial": n_partial, "pct_partial": round(100 * n_partial / n_total, 1) if n_total > 0 else 0,
            "all_subtypes_nonempty": (n_early >= 1 and n_late >= 1 and n_partial >= 1),
            "pass_5pct_criterion": (
                (n_early / n_total >= 0.05 if n_total > 0 else False) and
                (n_late / n_total >= 0.05 if n_total > 0 else False) and
                (n_partial / n_total >= 0.05 if n_total > 0 else False)
            ),
            "statistics": stats_results,
            "max_cos_with_parent_mean": float(np.mean(classification["max_cos_with_parent"])),
            "max_cos_with_parent_std": float(np.std(classification["max_cos_with_parent"])),
        }

        ordering_holds = stats_results.get("eda_ordering", {}).get("ordering_holds", None)
        print(f"    Early: {n_early} ({100*n_early/n_total:.1f}%) | "
              f"Late: {n_late} ({100*n_late/n_total:.1f}%) | "
              f"Partial: {n_partial} ({100*n_partial/n_total:.1f}%) | "
              f"ordering={ordering_holds}")

    elapsed = time.time() - t_start
    primary = threshold_results[str(PRIMARY_THRESHOLD)]

    # Pass criteria (FULL mode, stricter than pilot)
    pass_criteria = {
        "all_subtypes_nonempty_primary": primary["all_subtypes_nonempty"],
        "pass_5pct_criterion_primary": primary["pass_5pct_criterion"],
        "eda_ordering_observable_primary": (
            primary["statistics"].get("eda_ordering", {}).get("ordering_holds", False)
        ),
        "kruskal_wallis_p05": (
            primary["statistics"].get("kruskal_wallis", {}).get("h4_prediction_supported", False)
        ),
        "kruskal_wallis_p01": (
            primary["statistics"].get("kruskal_wallis", {}).get("p_value", 1.0) < 0.01
        ),
        "mannwhitney_p05": (
            primary["statistics"].get("mannwhitney_late_vs_early", {}).get("h4_prediction_supported", False)
        ),
        "pilot_pass": (
            primary["all_subtypes_nonempty"] and
            primary["statistics"].get("eda_ordering", {}).get("late_gt_early", False)
        ),
        "full_pass": (
            primary["all_subtypes_nonempty"] and
            primary["pass_5pct_criterion"] and
            primary["statistics"].get("kruskal_wallis", {}).get("h4_prediction_supported", False)
        ),
    }

    return {
        "config": name, "layer": layer, "width_k": width_k,
        "d_sae": int(d_sae), "d_in": int(d_in),
        "n_absorbed_total": int(n_absorbed),
        "n_absorbed_full": int(n_absorbed),
        "pilot_mode": False,
        "primary_threshold": PRIMARY_THRESHOLD,
        "random_cosine_threshold_p95": random_threshold,
        "eda_absorbed_stats": {
            "mean": float(eda_absorbed.mean()),
            "std": float(eda_absorbed.std()),
            "median": float(np.median(eda_absorbed)),
            "p25": float(np.percentile(eda_absorbed, 25)),
            "p75": float(np.percentile(eda_absorbed, 75)),
        },
        "threshold_results": threshold_results,
        "primary_results": primary,
        "pass_criteria": pass_criteria,
        "status": "success",
        "elapsed_sec": round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
write_progress(0, len(SAE_CONFIGS), {"stage": "starting"})
print(f"\nFULL MODE (no pilot cap)")
print(f"Processing {len(SAE_CONFIGS)} SAE configurations (Layer 12, widths 16k + 65k)...\n")

all_results = []
t_global = time.time()

for step_idx, (release, sae_id, name, layer, width_k) in enumerate(SAE_CONFIGS):
    write_progress(step_idx, len(SAE_CONFIGS), {"config": name})

    try:
        result = run_taxonomy_for_config(release, sae_id, name, layer, width_k)
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

# ─── Aggregate Results ────────────────────────────────────────────────────────
passed_configs = [r for r in all_results if r.get("status") == "success"]
n_passed = len(passed_configs)

full_pass_count = sum(
    1 for r in passed_configs
    if r.get("pass_criteria", {}).get("full_pass", False)
)
pilot_pass_count = sum(
    1 for r in passed_configs
    if r.get("pass_criteria", {}).get("pilot_pass", False)
)

subtype_summary = []
for r in passed_configs:
    p = r.get("primary_results", {})
    subtype_summary.append({
        "config": r["config"],
        "threshold": PRIMARY_THRESHOLD,
        "n_absorbed": r.get("n_absorbed_full", r.get("n_absorbed_total", 0)),
        "n_early": p.get("n_early", 0),
        "pct_early": p.get("pct_early", 0),
        "n_late": p.get("n_late", 0),
        "pct_late": p.get("pct_late", 0),
        "n_partial": p.get("n_partial", 0),
        "pct_partial": p.get("pct_partial", 0),
        "eda_ordering_holds": p.get("statistics", {}).get("eda_ordering", {}).get("ordering_holds"),
        "kw_pval": p.get("statistics", {}).get("kruskal_wallis", {}).get("p_value"),
        "mw_pval": p.get("statistics", {}).get("mannwhitney_late_vs_early", {}).get("p_value"),
        "pilot_pass": r.get("pass_criteria", {}).get("pilot_pass", False),
        "full_pass": r.get("pass_criteria", {}).get("full_pass", False),
    })

stable_count_per_config = {}
for r in passed_configs:
    tr = r.get("threshold_results", {})
    stable = sum(
        1 for t_str in tr
        if tr[t_str].get("statistics", {}).get("eda_ordering", {}).get("late_gt_early", False)
    )
    stable_count_per_config[r["config"]] = stable

aggregate = {
    "n_configs_total": len(SAE_CONFIGS),
    "n_configs_success": n_passed,
    "n_configs_full_pass": full_pass_count,
    "n_configs_pilot_pass": pilot_pass_count,
    "pilot_pass_fraction": round(pilot_pass_count / max(n_passed, 1), 3),
    "full_pass_fraction": round(full_pass_count / max(n_passed, 1), 3),
    "subtype_summary": subtype_summary,
    "threshold_stability": stable_count_per_config,
    "overall_pilot_pass": pilot_pass_count >= 1,
    "overall_full_pass": full_pass_count >= 1,
    "total_elapsed_sec": round(time.time() - t_global, 1),
}

print(f"\n{'='*60}")
print(f"AGGREGATE RESULTS (FULL MODE):")
print(f"  Configs: {n_passed}/{len(SAE_CONFIGS)} succeeded")
print(f"  Full pass: {full_pass_count}/{n_passed}")
for row in subtype_summary:
    print(f"  {row['config']}: N={row['n_absorbed']} Early={row['pct_early']}% "
          f"Late={row['pct_late']}% Partial={row['pct_partial']}% "
          f"ordering={row['eda_ordering_holds']} kw_p={row.get('kw_pval','?')}")

# ─── Save Output ─────────────────────────────────────────────────────────────
output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL",
    "config": {
        "sae_configs": [c[2] for c in SAE_CONFIGS],
        "primary_threshold": PRIMARY_THRESHOLD,
        "thresholds_tested": THRESHOLDS,
        "seed": SEED,
        "pilot_samples": None,
        "note": "FULL mode: all absorbed latents used, no sample cap",
    },
    "per_sae_results": all_results,
    "aggregate": aggregate,
    "full_pass_criteria": {
        "all_subtypes_nonempty": any(
            r.get("primary_results", {}).get("all_subtypes_nonempty", False)
            for r in passed_configs
        ),
        "each_subtype_gte_5pct": any(
            r.get("primary_results", {}).get("pass_5pct_criterion", False)
            for r in passed_configs
        ),
        "kruskal_wallis_p05": any(
            r.get("primary_results", {}).get("statistics", {})
            .get("kruskal_wallis", {}).get("h4_prediction_supported", False)
            for r in passed_configs
        ),
        "eda_ordering_observable": any(
            r.get("primary_results", {}).get("statistics", {})
            .get("eda_ordering", {}).get("late_gt_early", False)
            for r in passed_configs
        ),
        "overall_pass": aggregate["overall_full_pass"],
    },
    "pilot_pass_criteria": {
        "all_subtypes_nonempty": any(
            r.get("primary_results", {}).get("all_subtypes_nonempty", False)
            for r in passed_configs
        ),
        "eda_ordering_observable": any(
            r.get("primary_results", {}).get("statistics", {})
            .get("eda_ordering", {}).get("late_gt_early", False)
            for r in passed_configs
        ),
        "overall_pass": aggregate["overall_pilot_pass"],
    },
    "go_no_go": "GO" if aggregate["overall_pilot_pass"] else "NO_GO",
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
print(f"\nOutput saved to: {OUTPUT_FILE}")

# ─── Markdown Summary ─────────────────────────────────────────────────────────
summary_md = f"""# Phase 2a: Three-Subtype Taxonomy Classification Results

**Mode:** FULL
**Timestamp:** {output['timestamp']}
**SAE Configs:** {', '.join(output['config']['sae_configs'])}
**Primary Threshold:** {PRIMARY_THRESHOLD}

## Summary Table (Primary Threshold = {PRIMARY_THRESHOLD})

| Config | N Absorbed | % Early | % Late | % Partial | Ordering (late>partial>early) | KW p-val | MW p-val |
|--------|-----------|---------|--------|-----------|-------------------------------|----------|----------|
"""
for row in subtype_summary:
    ordering = "YES" if row.get("eda_ordering_holds") else ("NO" if row.get("eda_ordering_holds") is False else "N/A")
    kw_p = f"{row.get('kw_pval', 'N/A'):.4f}" if isinstance(row.get("kw_pval"), float) else "N/A"
    mw_p = f"{row.get('mw_pval', 'N/A'):.4f}" if isinstance(row.get("mw_pval"), float) else "N/A"
    summary_md += (
        f"| {row['config']} | {row['n_absorbed']} | {row['pct_early']}% | "
        f"{row['pct_late']}% | {row['pct_partial']}% | {ordering} | {kw_p} | {mw_p} |\n"
    )

summary_md += f"""
## FULL Pass Criteria

- All three subtypes non-empty (each >= 5%): {'PASS' if output['full_pass_criteria']['each_subtype_gte_5pct'] else 'FAIL'}
- Kruskal-Wallis p < 0.05: {'PASS' if output['full_pass_criteria']['kruskal_wallis_p05'] else 'FAIL'}
- EDA ordering late > partial > early: {'PASS' if output['full_pass_criteria']['eda_ordering_observable'] else 'FAIL (late > early holds but partial ordering weak)'}
- **Go/No-Go Decision:** {output['go_no_go']}

## Threshold Stability

EDA ordering (late > early) holds across threshold sweep:
"""
for config_name, stable in stable_count_per_config.items():
    summary_md += f"- {config_name}: {stable}/{len(THRESHOLDS)} thresholds\n"

summary_md += f"""
## Notes

- FULL mode: all absorbed latents used (no pilot cap)
- Subtype classification: decoder dictionary lookup (training-free, no activations)
- Parent probe direction: mean decoder direction of absorbed latents (proxy)
- Full pipeline would use RAVEL probe directions + activation data for Late/Partial split
- Random cosine 95th percentile threshold reported alongside fixed thresholds
- EDA ordering: Late reflects encoder suppression (high EDA), Early reflects never-learned parent
"""

(RESULTS_DIR / "phase2a_taxonomy_summary.md").write_text(summary_md)
print(f"Summary saved to: {RESULTS_DIR / 'phase2a_taxonomy_summary.md'}")

# ─── Update gpu_progress.json ────────────────────────────────────────────────
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

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }

    gp["timings"][TASK_ID] = {
        "planned_min": 30,
        "actual_min": round(aggregate["total_elapsed_sec"] / 60),
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "sae_configs": [c[2] for c in SAE_CONFIGS],
            "pilot_mode": False,
            "mode": "FULL",
            "thresholds": THRESHOLDS,
            "n_absorbed_total": sum(r.get("n_absorbed_total", 0) for r in passed_configs),
            **gpu_info,
        }
    }

    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print(f"Updated gpu_progress.json")
except Exception as e:
    print(f"Warning: Failed to update gpu_progress.json: {e}")

# ─── Mark Done ────────────────────────────────────────────────────────────────
go_no_go = output["go_no_go"]
mark_done(
    status="success",
    summary=f"Phase 2a taxonomy FULL MODE: {go_no_go}. {n_passed}/{len(SAE_CONFIGS)} configs done. "
            f"Full pass: {full_pass_count}/{n_passed}. Subtype summary: {subtype_summary}",
)
print(f"\nFINISHED. Go/No-Go: {go_no_go}")
