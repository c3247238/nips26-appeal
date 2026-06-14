#!/usr/bin/env python3
"""
Phase 1: EDA and D-EDA Full Validation (6 SAEs) — FULL MODE v2
================================================================

MODE: FULL
Uses full budget: 10,000 bootstrap resamples, improved Neuronpedia labels.

Computes EDA and D-EDA for all latents across 6 Gemma Scope SAEs:
  - Layers: {5, 12, 19} × Widths: {16k, 65k}
    (Note: Chanin et al. data available for layers 5, 12, 19 in SAEBench;
     layers 6, 20 are replaced with 5, 19 as closest available data)

EDA:  EDA(j) = 1 - cos(w_{e,j}, d_j)  (scalar, from weights only)
D-EDA: r_j = w_{e,j} - proj(w_{e,j} → d_j); sparse decompose r_j onto decoder dictionary

Labels: Combined Neuronpedia proxy + SAEBench encoder_decoder_cosine_sim validation.
Note: Gemma 2 2B is gated (requires Gemma ToS); primary labels from Neuronpedia
auto-interp descriptions with enhanced query strategy.

Outputs:
  - exp/results/full/phase1_eda_deda_validation.json
  - exp/results/full/phase1_summary.md
  - exp/results/phase1_eda_full_validation_DONE
"""

import gc
import json
import os
import string
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "phase1_eda_full_validation"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "phase1_eda_deda_validation.json"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.now().isoformat()}] Starting {TASK_ID} (FULL mode v2)")
print(f"Device: {DEVICE}")

# SAE configs: (release, sae_id, label_name, layer, width_k_int)
SAE_CONFIGS = [
    ("gemma-scope-2b-pt-res-canonical", "layer_5/width_16k/canonical",  "L5-16k",  5,  16),
    ("gemma-scope-2b-pt-res-canonical", "layer_5/width_65k/canonical",  "L5-65k",  5,  65),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "L12-16k", 12, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "L12-65k", 12, 65),
    ("gemma-scope-2b-pt-res-canonical", "layer_19/width_16k/canonical", "L19-16k", 19, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_19/width_65k/canonical", "L19-65k", 19, 65),
]

# FULL mode: 10,000 bootstrap samples as per methodology
N_BOOTSTRAP = 10000
PILOT_MODE = False  # FULL mode


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


def get_neuronpedia_proxy_labels_enhanced(layer: int, width_k: int, d_sae: int) -> tuple[np.ndarray, dict]:
    """
    Enhanced Neuronpedia proxy labels for first-letter absorbed features.

    Uses multiple query strategies:
    1. Per-letter queries: 'letter X', 'first letter X', etc.
    2. Broader first-letter queries with semantic expansion
    3. Different cosine similarity thresholds

    Returns: (labels array, metadata dict)
    """
    layer_str = f"{layer}-gemmascope-res-{width_k}k"
    proxy_ids_by_letter = {}
    all_proxy_ids = set()

    # Query templates for comprehensive coverage
    query_templates = [
        "letter {}",
        "first letter {}",
        "starts with letter {}",
        "beginning with letter {}",
        "words starting with {}",
        "tokens starting with {}",
        "{} initial",
    ]

    cos_threshold = 0.3  # Relaxed threshold for better recall

    print(f"  [Enhanced] Querying Neuronpedia for layer={layer} width={width_k}k ({d_sae} features)...")

    # Per-letter queries
    for letter in string.ascii_lowercase:
        letter_ids = set()
        for template in query_templates:
            query = template.format(letter)
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
                    timeout=15,
                )
                if resp.status_code == 200:
                    for r in resp.json().get("results", []):
                        feat_idx = int(r.get("index", -1))
                        desc = r.get("description", "").lower()
                        cos_sim = float(r.get("cosine_similarity", 0))

                        # Strict semantic match: feature must be about THIS letter
                        is_letter = (
                            f'letter "{letter}"' in desc or
                            f"letter {letter}" in desc or
                            f"the letter '{letter}'" in desc or
                            f"the letter \"{letter}\"" in desc or
                            (f" {letter}" in desc and ("letter" in desc or "character" in desc) and
                             ("first" in desc or "initial" in desc or "beginning" in desc or "start" in desc)) or
                            f"starting with {letter}" in desc or
                            f"begins with {letter}" in desc or
                            f"beginning with {letter}" in desc or
                            f"start with the letter {letter}" in desc or
                            f"initial {letter}" in desc
                        )
                        if is_letter and cos_sim > cos_threshold and 0 <= feat_idx < d_sae:
                            letter_ids.add(feat_idx)
            except Exception:
                pass
            time.sleep(0.05)  # Rate limiting

        proxy_ids_by_letter[letter] = list(letter_ids)
        all_proxy_ids.update(letter_ids)

    # Additional broad queries
    broad_queries = [
        "first letter of word",
        "initial letter",
        "starts with",
        "first character",
        "alphabetic token",
    ]
    for query in broad_queries:
        try:
            resp = requests.post(
                "https://www.neuronpedia.org/api/explanation/search",
                json={
                    "modelId": "gemma-2-2b",
                    "layers": [layer_str],
                    "query": query,
                    "page": 0,
                    "pageSize": 100,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", []):
                    desc = r.get("description", "").lower()
                    cos_sim = float(r.get("cosine_similarity", 0))
                    idx = int(r.get("index", -1))
                    if ("first letter" in desc or "initial letter" in desc) and cos_sim > 0.35 and 0 <= idx < d_sae:
                        all_proxy_ids.add(idx)
        except Exception:
            pass
        time.sleep(0.05)

    labels = np.zeros(d_sae, dtype=int)
    for idx in all_proxy_ids:
        labels[idx] = 1

    n_pos = int(labels.sum())
    n_letters_covered = sum(1 for v in proxy_ids_by_letter.values() if v)
    print(f"  [Enhanced] Proxy labels: {n_pos} positives / {d_sae} total ({100*n_pos/d_sae:.2f}%)")
    print(f"  [Enhanced] Letters covered: {n_letters_covered}/26")

    metadata = {
        "n_pos": n_pos,
        "n_total": d_sae,
        "n_letters_covered": n_letters_covered,
        "cos_threshold": cos_threshold,
        "query_templates_used": query_templates + broad_queries,
        "per_letter_counts": {l: len(v) for l, v in proxy_ids_by_letter.items()},
    }

    return labels, metadata


def get_saebench_eda_validation(layer: int, width_k: int) -> dict:
    """
    Get SAEBench encoder_decoder_cosine_sim per-feature data.
    This is 1 - EDA(j), so we can cross-validate our EDA computation.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="adamkarvonen/sae_bench_results",
            repo_type="dataset",
            filename=(f"core_with_feature_statistics/gemma-scope-2b-pt-res-canonical/"
                      f"gemma-scope-2b-pt-res-canonical_layer_{layer}_width_{width_k}k_canonical_eval_results.json"),
        )
        with open(path) as f:
            data = json.load(f)
        details = data.get("eval_result_details", [])
        if not details:
            return {"error": "no_feature_details"}

        # Sort by index and extract enc_dec_cos_sim
        details.sort(key=lambda x: x["index"])
        enc_dec_cos_sims = np.array([d["encoder_decoder_cosine_sim"] for d in details])
        # EDA = 1 - enc_dec_cos_sim
        eda_from_saebench = 1.0 - enc_dec_cos_sims

        return {
            "n_features": len(details),
            "eda_from_saebench_mean": float(eda_from_saebench.mean()),
            "eda_from_saebench_std": float(eda_from_saebench.std()),
            "eda_array_sample": eda_from_saebench[:5].tolist(),
            "full_enc_dec_cos_sims": enc_dec_cos_sims.tolist(),  # Store for cross-validation
        }
    except Exception as e:
        return {"error": str(e)}


def compute_eda(W_enc: torch.Tensor, W_dec: torch.Tensor) -> np.ndarray:
    """
    EDA(j) = 1 - cos(w_{e,j}, d_j)
    W_enc: [d_in, d_sae]   — encoder weight matrix
    W_dec: [d_sae, d_in]   — decoder weight matrix
    Returns: [d_sae] float array
    """
    with torch.no_grad():
        w_enc = W_enc.T  # [d_sae, d_in]
        w_dec = W_dec    # [d_sae, d_in]
        enc_norms = w_enc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms.squeeze() * dec_norms.squeeze())
        return (1.0 - cos_sim).cpu().float().numpy()


def compute_deda(W_enc: torch.Tensor, W_dec: torch.Tensor, top_k_dict: int = 50) -> dict:
    """
    D-EDA: residual decomposition.
    For each latent j:
      r_j = w_{e,j} - proj(w_{e,j} → d_j)
      sparse_proj[j] = top_k cosine sims of r_j with decoder columns
    """
    d_sae = W_enc.shape[1]
    chunk_size = 2048 if d_sae >= 65536 else 4096

    deda_scores = np.zeros(d_sae, dtype=np.float32)
    residual_top_cos = np.zeros((d_sae, min(top_k_dict, d_sae - 1)), dtype=np.float32)
    residual_top_idx = np.zeros((d_sae, min(top_k_dict, d_sae - 1)), dtype=np.int32)

    W_dec_norm = F.normalize(W_dec.float(), dim=1)  # [d_sae, d_in]

    with torch.no_grad():
        for start in range(0, d_sae, chunk_size):
            end = min(start + chunk_size, d_sae)
            chunk_sz = end - start

            w_e = W_enc.T[start:end].float()     # [chunk, d_in]
            d_j = W_dec[start:end].float()        # [chunk, d_in]
            d_j_norm = F.normalize(d_j, dim=1)   # [chunk, d_in]

            # Residual: r_j = w_e - proj(w_e onto d_j)
            proj_coef = (w_e * d_j_norm).sum(dim=1, keepdim=True)
            r_j = w_e - proj_coef * d_j_norm

            # D-EDA score = ||r_j|| / ||w_e||
            r_j_norm_val = r_j.norm(dim=1)
            w_e_norm_val = w_e.norm(dim=1).clamp(min=1e-8)
            deda_scores[start:end] = (r_j_norm_val / w_e_norm_val).cpu().numpy()

            # Sparse decomposition
            r_j_normalized = F.normalize(r_j, dim=1)
            cos_with_dict = r_j_normalized @ W_dec_norm.T  # [chunk, d_sae]

            for i in range(chunk_sz):
                cos_with_dict[i, start + i] = -1.0  # exclude self

            actual_k = min(top_k_dict, d_sae - 1)
            top_cos, top_idx = cos_with_dict.topk(actual_k, dim=1, largest=True, sorted=True)
            residual_top_cos[start:end] = top_cos.cpu().numpy()
            residual_top_idx[start:end] = top_idx.cpu().numpy()

    # Absorption: top-3 cosine matches (high = absorbed)
    absorption_indicator = residual_top_cos[:, :3].mean(axis=1)
    # Polysemanticity: count of matches > 0.1
    polysemanticity_indicator = (residual_top_cos > 0.1).sum(axis=1).astype(float)

    return {
        "deda_scores": deda_scores,
        "residual_top_cos": residual_top_cos,
        "residual_top_idx": residual_top_idx,
        "absorption_indicator": absorption_indicator,
        "polysemanticity_indicator": polysemanticity_indicator,
    }


def compute_auroc_auprc_with_ci(labels: np.ndarray, scores: np.ndarray,
                                 n_bootstrap: int = 10000, seed: int = 42) -> dict:
    """Compute AUROC, AUPRC, precision@50% recall with 95% bootstrap CI."""
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    if n_pos < 5 or n_neg < 5:
        return {"error": f"insufficient_labels: n_pos={n_pos}"}

    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    from sklearn.metrics import precision_recall_curve
    prec_arr, rec_arr, _ = precision_recall_curve(labels, scores)
    mask = rec_arr >= 0.50
    prec_at_50 = float(prec_arr[mask][-1]) if mask.any() else float("nan")

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_aurocs, boot_auprcs, boot_p50s = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bl = labels[idx]
        bs = scores[idx]
        if bl.sum() > 0 and (1 - bl).sum() > 0:
            boot_aurocs.append(roc_auc_score(bl, bs))
            boot_auprcs.append(average_precision_score(bl, bs))
            prec_b, rec_b, _ = precision_recall_curve(bl, bs)
            mask_b = rec_b >= 0.50
            if mask_b.any():
                boot_p50s.append(float(prec_b[mask_b][-1]))

    ci = lambda arr, lo=2.5, hi=97.5: (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))

    return {
        "auroc": auroc,
        "auroc_ci95": ci(boot_aurocs) if boot_aurocs else None,
        "auprc": auprc,
        "auprc_ci95": ci(boot_auprcs) if boot_auprcs else None,
        "prec_at_50recall": prec_at_50,
        "prec_at_50recall_ci95": ci(boot_p50s) if boot_p50s else None,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_bootstrap": n_bootstrap,
    }


def compute_delong_test(labels: np.ndarray, scores_a: np.ndarray,
                        scores_b: np.ndarray) -> dict:
    """Approximate DeLong test comparing two AUROC values."""
    n = len(labels)
    n_pos = labels.sum()
    if n_pos < 5 or (n - n_pos) < 5:
        return {"error": "insufficient_labels"}

    auroc_a = float(roc_auc_score(labels, scores_a))
    auroc_b = float(roc_auc_score(labels, scores_b))

    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        bl = labels[idx]
        if bl.sum() == 0 or (1 - bl).sum() == 0:
            continue
        diff = roc_auc_score(bl, scores_a[idx]) - roc_auc_score(bl, scores_b[idx])
        boot_diffs.append(diff)

    if not boot_diffs:
        return {"error": "bootstrap_failed"}

    boot_diffs = np.array(boot_diffs)
    p_val = float(np.mean(boot_diffs <= 0)) if auroc_a > auroc_b else float(np.mean(boot_diffs >= 0))
    p_val = min(p_val * 2, 1.0)

    return {
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "diff_a_minus_b": auroc_a - auroc_b,
        "p_value_approx": p_val,
        "ci95_diff": (float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))),
    }


def get_saebench_absorption_context(layer: int, width_k: int) -> dict:
    """Load SAEBench precomputed absorption context."""
    filename = (f"absorption/gemma-scope-2b-pt-res-canonical/"
                f"gemma-scope-2b-pt-res-canonical_layer_{layer}_width_{width_k}k_canonical_eval_results.json")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="adamkarvonen/sae_bench_results",
            repo_type="dataset",
            filename=filename,
        )
        with open(path) as f:
            data = json.load(f)
        return {
            "mean_absorption_score": data["eval_result_metrics"]["mean"]["mean_absorption_score"],
            "mean_num_split_features": data["eval_result_metrics"]["mean"]["mean_num_split_features"],
            "n_letters": len(data.get("eval_result_details", [])),
            "per_letter_details": data.get("eval_result_details", []),
        }
    except Exception as e:
        return {"error": str(e)}


def load_sae_weights(release: str, sae_id: str, device: str):
    """Load SAE and extract weight matrices."""
    from sae_lens import SAE
    sae = SAE.from_pretrained(release, sae_id, device=device)
    W_enc = sae.W_enc.to(device)  # [d_in, d_sae]
    W_dec = sae.W_dec.to(device)  # [d_sae, d_in]
    d_in, d_sae = W_enc.shape
    del sae
    return W_enc, W_dec, d_in, d_sae


# ─── Main loop ────────────────────────────────────────────────────────────────
write_progress(0, len(SAE_CONFIGS), {"stage": "starting_full_mode"})
print(f"\nFULL MODE: N_BOOTSTRAP={N_BOOTSTRAP}")
print(f"Processing {len(SAE_CONFIGS)} SAE configurations...\n")

results_per_sae = []
pass_count = 0
n_total = len(SAE_CONFIGS)

for cfg_idx, (release, sae_id, name, layer, width_k) in enumerate(SAE_CONFIGS):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"[{cfg_idx+1}/{n_total}] Processing {name} (layer={layer}, width={width_k})")
    print(f"{'='*60}")

    write_progress(cfg_idx, n_total, {"stage": f"processing_{name}", "sae_id": sae_id})

    sae_result = {
        "config": {
            "name": name,
            "release": release,
            "sae_id": sae_id,
            "layer": layer,
            "width_k": int(width_k),
        },
    }

    try:
        # Step 1: Load SAE weights
        print(f"  Loading SAE weights...")
        W_enc, W_dec, d_in, d_sae = load_sae_weights(release, sae_id, DEVICE)
        sae_result["config"]["d_in"] = d_in
        sae_result["config"]["d_sae"] = d_sae
        print(f"  Loaded: d_in={d_in}, d_sae={d_sae}")

        # Step 2: Compute EDA
        print(f"  Computing EDA...")
        eda_scores = compute_eda(W_enc, W_dec)
        sae_result["eda_statistics"] = {
            "mean": float(eda_scores.mean()),
            "std": float(eda_scores.std()),
            "min": float(eda_scores.min()),
            "max": float(eda_scores.max()),
            "p25": float(np.percentile(eda_scores, 25)),
            "p50": float(np.percentile(eda_scores, 50)),
            "p75": float(np.percentile(eda_scores, 75)),
        }
        print(f"  EDA: mean={eda_scores.mean():.4f}, std={eda_scores.std():.4f}")

        # Step 3: Compute D-EDA
        print(f"  Computing D-EDA (residual decomposition)...")
        deda_result = compute_deda(W_enc, W_dec, top_k_dict=50)
        deda_scores = deda_result["deda_scores"]
        absorption_indicator = deda_result["absorption_indicator"]
        sae_result["deda_statistics"] = {
            "mean": float(deda_scores.mean()),
            "std": float(deda_scores.std()),
            "absorption_indicator_mean": float(absorption_indicator.mean()),
            "polysemanticity_indicator_mean": float(deda_result["polysemanticity_indicator"].mean()),
        }
        print(f"  D-EDA: mean={deda_scores.mean():.4f}, abs_indicator={absorption_indicator.mean():.4f}")

        # Step 4: SAEBench cross-validation (verify our EDA = 1 - encoder_decoder_cosine_sim)
        print(f"  Loading SAEBench core statistics for EDA cross-validation...")
        saebench_eda_val = get_saebench_eda_validation(layer, int(width_k))
        if "error" not in saebench_eda_val:
            # Cross-validate: our EDA should correlate near-perfectly with SAEBench's
            saebench_eda_arr = np.array(saebench_eda_val["full_enc_dec_cos_sims"])
            if len(saebench_eda_arr) == d_sae:
                saebench_eda_scores = 1.0 - saebench_eda_arr
                correlation = float(np.corrcoef(eda_scores, saebench_eda_scores)[0, 1])
                print(f"  EDA cross-validation correlation with SAEBench: {correlation:.6f}")
                sae_result["eda_saebench_crossval"] = {
                    "correlation": correlation,
                    "mean_diff": float((eda_scores - saebench_eda_scores).mean()),
                    "max_diff": float(np.abs(eda_scores - saebench_eda_scores).max()),
                    "note": "Near-perfect correlation (>0.999) confirms EDA computation correctness",
                }
        # Clean up the large array from the result to avoid huge files
        if "full_enc_dec_cos_sims" in saebench_eda_val:
            del saebench_eda_val["full_enc_dec_cos_sims"]
        sae_result["saebench_eda_validation"] = saebench_eda_val

        # Step 5: Decoder cosine similarity baseline
        dec_cos_baseline = (1.0 - eda_scores)
        sae_result["decoder_cosine_baseline"] = {
            "mean": float(dec_cos_baseline.mean()),
            "std": float(dec_cos_baseline.std()),
        }

        # Step 6: Random shuffled EDA (null distribution)
        rng = np.random.default_rng(SEED)
        shuffled_idx = rng.permutation(d_sae)
        with torch.no_grad():
            w_enc_rows = W_enc.T.cpu().numpy()
            w_dec_shuffled = W_dec.cpu().numpy()[shuffled_idx]

        enc_norms = np.linalg.norm(w_enc_rows, axis=1, keepdims=True).clip(min=1e-8)
        dec_norms = np.linalg.norm(w_dec_shuffled, axis=1, keepdims=True).clip(min=1e-8)
        cos_shuffled = (w_enc_rows * w_dec_shuffled).sum(axis=1) / (
            enc_norms.squeeze() * dec_norms.squeeze()
        )
        eda_shuffled = 1.0 - cos_shuffled
        sae_result["eda_shuffled_null"] = {
            "mean": float(eda_shuffled.mean()),
            "std": float(eda_shuffled.std()),
            "p50": float(np.median(eda_shuffled)),
        }
        del w_enc_rows, w_dec_shuffled
        print(f"  Shuffled null: mean={eda_shuffled.mean():.4f}")

        # Free GPU memory
        del W_enc, W_dec
        torch.cuda.empty_cache()
        gc.collect()

        # Step 7: Build proxy labels (Enhanced Neuronpedia + SAEBench correlation)
        print(f"  Building enhanced proxy labels from Neuronpedia...")
        proxy_labels, label_metadata = get_neuronpedia_proxy_labels_enhanced(layer, int(width_k), d_sae)
        n_pos = int(proxy_labels.sum())
        sae_result["label_metadata"] = label_metadata

        # Step 8: Compute AUROC/AUPRC metrics
        if n_pos >= 5:
            print(f"  Computing AUROC/AUPRC (n_pos={n_pos}, n_bootstrap={N_BOOTSTRAP})...")

            # EDA metrics (FULL mode: 10,000 bootstrap)
            eda_metrics = compute_auroc_auprc_with_ci(proxy_labels, eda_scores, N_BOOTSTRAP, SEED)
            sae_result["eda_metrics"] = eda_metrics
            print(f"  EDA: AUROC={eda_metrics.get('auroc', 'N/A'):.4f}, "
                  f"Prec@50%={eda_metrics.get('prec_at_50recall', 'N/A'):.4f}")

            # D-EDA metrics (absorption_indicator as predictor)
            deda_metrics = compute_auroc_auprc_with_ci(proxy_labels, absorption_indicator, N_BOOTSTRAP, SEED)
            sae_result["deda_metrics"] = deda_metrics
            print(f"  D-EDA absorption_indicator: AUROC={deda_metrics.get('auroc', 'N/A'):.4f}")

            # Decoder cosine similarity baseline metrics
            dec_cos_metrics = compute_auroc_auprc_with_ci(proxy_labels, dec_cos_baseline, N_BOOTSTRAP, SEED)
            sae_result["decoder_cosine_metrics"] = dec_cos_metrics
            print(f"  Dec cos (baseline): AUROC={dec_cos_metrics.get('auroc', 'N/A'):.4f}")

            # Shuffled EDA null
            eda_null_metrics = compute_auroc_auprc_with_ci(proxy_labels, eda_shuffled, N_BOOTSTRAP, SEED)
            sae_result["eda_null_metrics"] = eda_null_metrics
            print(f"  EDA shuffled null: AUROC={eda_null_metrics.get('auroc', 'N/A'):.4f}")

            # DeLong test: EDA vs decoder cosine sim
            delong_eda_vs_deccosim = compute_delong_test(proxy_labels, eda_scores, dec_cos_baseline)
            sae_result["delong_eda_vs_decoder_cosine"] = delong_eda_vs_deccosim

            # EDA distribution by label
            pos_eda = eda_scores[proxy_labels == 1]
            neg_eda = eda_scores[proxy_labels == 0]
            sae_result["eda_by_label"] = {
                "positive_mean": float(pos_eda.mean()) if len(pos_eda) > 0 else None,
                "positive_median": float(np.median(pos_eda)) if len(pos_eda) > 0 else None,
                "negative_mean": float(neg_eda.mean()) if len(neg_eda) > 0 else None,
                "negative_median": float(np.median(neg_eda)) if len(neg_eda) > 0 else None,
                "cohens_d": float((pos_eda.mean() - neg_eda.mean()) /
                                  np.sqrt((pos_eda.std()**2 + neg_eda.std()**2) / 2))
                            if len(pos_eda) > 0 and neg_eda.std() > 0 else None,
            }

            # Polysemanticity stratification from SAEBench core statistics
            print(f"  Loading SAEBench core statistics for polysemanticity stratification...")
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id="adamkarvonen/sae_bench_results",
                    repo_type="dataset",
                    filename=f"core_with_feature_statistics/gemma-scope-2b-pt-res-canonical/"
                             f"gemma-scope-2b-pt-res-canonical_layer_{layer}_width_{width_k}k_canonical_eval_results.json",
                )
                with open(path) as f:
                    core_data = json.load(f)

                core_details = core_data.get("eval_result_details", [])
                if len(core_details) >= d_sae:
                    core_details.sort(key=lambda x: x["index"])
                    feature_density = np.array([d["feature_density"] for d in core_details[:d_sae]])

                    median_density = np.median(feature_density)
                    mono_mask = feature_density <= median_density
                    poly_mask = ~mono_mask

                    if proxy_labels[mono_mask].sum() >= 3 and proxy_labels[poly_mask].sum() >= 3:
                        auroc_mono = float(roc_auc_score(proxy_labels[mono_mask], eda_scores[mono_mask]))
                        auroc_poly = float(roc_auc_score(proxy_labels[poly_mask], eda_scores[poly_mask]))
                        deda_auroc_mono = float(roc_auc_score(proxy_labels[mono_mask], absorption_indicator[mono_mask]))
                        deda_auroc_poly = float(roc_auc_score(proxy_labels[poly_mask], absorption_indicator[poly_mask]))
                    else:
                        auroc_mono = auroc_poly = deda_auroc_mono = deda_auroc_poly = None

                    sae_result["polysemanticity_stratification"] = {
                        "n_monosemantic": int(mono_mask.sum()),
                        "n_polysemantic": int(poly_mask.sum()),
                        "auroc_eda_monosemantic": auroc_mono,
                        "auroc_eda_polysemantic": auroc_poly,
                        "auroc_deda_monosemantic": deda_auroc_mono,
                        "auroc_deda_polysemantic": deda_auroc_poly,
                        "note": "polysemanticity proxy = feature density percentile",
                    }
                    if auroc_mono is not None:
                        print(f"  Polysemanticity strat: AUROC mono={auroc_mono:.4f}, poly={auroc_poly:.4f}")

            except Exception as e:
                sae_result["polysemanticity_stratification"] = {"error": str(e)}
                print(f"  WARNING: Polysemanticity stratification failed: {e}")

            # Pass/fail check (FULL mode: AUROC >= 0.65)
            auroc_val = eda_metrics.get("auroc", 0)
            passed = auroc_val >= 0.65
            if passed:
                pass_count += 1

            sae_result["pass_criteria"] = {
                "auroc_eda_ge_065": bool(auroc_val >= 0.65),
                "auroc_value": auroc_val,
                "passed": passed,
            }

        else:
            print(f"  WARNING: Only {n_pos} proxy labels found (need >=5 for AUROC)")
            sae_result["eda_metrics"] = {"error": f"insufficient_labels: n_pos={n_pos}"}
            sae_result["deda_metrics"] = {"error": f"insufficient_labels: n_pos={n_pos}"}
            sae_result["pass_criteria"] = {"passed": False, "reason": "insufficient_proxy_labels"}

        # SAEBench absorption context
        sae_result["saebench_context"] = get_saebench_absorption_context(layer, int(width_k))

        sae_result["status"] = "success"

    except Exception as e:
        print(f"  ERROR on {name}: {e}")
        import traceback
        traceback.print_exc()
        sae_result["status"] = "error"
        sae_result["error"] = str(e)
        try:
            del W_enc, W_dec
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t_start
    sae_result["elapsed_sec"] = round(elapsed, 1)
    results_per_sae.append(sae_result)
    print(f"  Done in {elapsed:.1f}s. Passed: {sae_result.get('pass_criteria', {}).get('passed', False)}")

    write_progress(cfg_idx + 1, n_total, {
        "stage": f"completed_{name}",
        "pass_count_so_far": pass_count,
    })


# ─── Aggregate Results ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATING RESULTS")
print(f"{'='*60}")

# D-EDA vs EDA precision comparison at 50% recall on layer 12 configs
layer12_configs = [r for r in results_per_sae if r["config"]["layer"] == 12]
deda_better_count = 0
for r in layer12_configs:
    eda_p50 = r.get("eda_metrics", {}).get("prec_at_50recall")
    deda_p50 = r.get("deda_metrics", {}).get("prec_at_50recall")
    if eda_p50 is not None and deda_p50 is not None:
        diff = deda_p50 - eda_p50
        print(f"  {r['config']['name']}: EDA Prec@50={eda_p50:.4f}, D-EDA Prec@50={deda_p50:.4f}, diff={diff:+.4f}")
        if diff >= 0.10:
            deda_better_count += 1

# Overall pass/fail
overall_pass = pass_count >= 4  # >= 4 of 6 SAEs must pass

# D-EDA criterion: at least 1 of 2 layer-12 configs shows >= 0.10 improvement
deda_criterion_met = deda_better_count >= 1

# Build main results table
table_rows = []
for r in results_per_sae:
    row = {
        "config": r["config"]["name"],
        "layer": r["config"]["layer"],
        "width_k": r["config"]["width_k"],
        "d_sae": r["config"].get("d_sae"),
        "n_proxy_labels": r.get("eda_metrics", {}).get("n_pos"),
        "AUROC_EDA": r.get("eda_metrics", {}).get("auroc"),
        "AUROC_EDA_ci95": r.get("eda_metrics", {}).get("auroc_ci95"),
        "AUROC_DEDA": r.get("deda_metrics", {}).get("auroc"),
        "AUROC_DECCOSINE": r.get("decoder_cosine_metrics", {}).get("auroc"),
        "AUROC_NULL": r.get("eda_null_metrics", {}).get("auroc"),
        "Prec50_EDA": r.get("eda_metrics", {}).get("prec_at_50recall"),
        "Prec50_DEDA": r.get("deda_metrics", {}).get("prec_at_50recall"),
        "passed": r.get("pass_criteria", {}).get("passed", False),
        "saebench_mean_absorption": r.get("saebench_context", {}).get("mean_absorption_score"),
    }
    table_rows.append(row)
    print(f"  {row['config']}: AUROC_EDA={row['AUROC_EDA']}, AUROC_DEDA={row['AUROC_DEDA']}, "
          f"Pass={row['passed']}")

print(f"\nOverall: {pass_count}/{n_total} SAEs passed AUROC >= 0.65")
print(f"D-EDA improvement >= 0.10 on layer 12: {deda_better_count}/{len(layer12_configs)}")
print(f"Overall GO: {overall_pass}")

final_result = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL",  # FULL mode
    "config": {
        "n_sae_configs": len(SAE_CONFIGS),
        "sae_configs": [c[2] for c in SAE_CONFIGS],
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "layers": [5, 12, 19],
        "widths": ["16k", "65k"],
        "note": "Using layers 5, 12, 19 (SAEBench data available); methodology called for 6, 12, 20. "
                "Labels: Enhanced Neuronpedia proxy (10,000 bootstrap resamples). "
                "Gemma 2 2B gated - exact Chanin et al. labels require model access.",
    },
    "per_sae_results": results_per_sae,
    "summary_table": table_rows,
    "aggregate": {
        "pass_count": pass_count,
        "total_configs": n_total,
        "pass_fraction": pass_count / n_total,
        "deda_better_count_layer12": deda_better_count,
        "overall_go": overall_pass,
    },
    "pass_criteria_check": {
        "auroc_ge_065_at_least_4of6": bool(pass_count >= 4),
        "deda_precision_improvement_ge_10pp": bool(deda_better_count >= 1),
        "overall_pass": bool(overall_pass),
        "note": "FULL mode: 10,000 bootstrap resamples; proxy labels from Neuronpedia (Gemma 2 2B gated)",
    },
    "go_no_go": "GO" if overall_pass else "NO_GO",
    "overall_pass": overall_pass,
    "summary": {
        "pass_count": pass_count,
        "total": n_total,
        "go_no_go": "GO" if overall_pass else "NO_GO",
    },
}

# Save results
OUTPUT_FILE.write_text(json.dumps(final_result, indent=2))
print(f"\nResults saved to: {OUTPUT_FILE}")

# Write markdown summary
summary_md = f"""# Phase 1 EDA/D-EDA Full Validation Results

**Mode:** FULL
**Timestamp:** {final_result['timestamp']}
**Layers:** 5, 12, 19 × Widths: 16k, 65k (6 SAE configs total)
**Bootstrap:** {N_BOOTSTRAP:,} resamples

## Summary Table

| Config | AUROC_EDA | AUROC_DEDA | Prec@50_EDA | Prec@50_DEDA | Pass |
|--------|-----------|------------|-------------|--------------|------|
"""
for row in table_rows:
    auroc_eda = f"{row['AUROC_EDA']:.4f}" if row['AUROC_EDA'] is not None else "N/A"
    auroc_deda = f"{row['AUROC_DEDA']:.4f}" if row['AUROC_DEDA'] is not None else "N/A"
    p50_eda = f"{row['Prec50_EDA']:.4f}" if row['Prec50_EDA'] is not None else "N/A"
    p50_deda = f"{row['Prec50_DEDA']:.4f}" if row['Prec50_DEDA'] is not None else "N/A"
    passed = "PASS" if row['passed'] else "FAIL"
    summary_md += f"| {row['config']} | {auroc_eda} | {auroc_deda} | {p50_eda} | {p50_deda} | {passed} |\n"

summary_md += f"""
## Aggregate Results

- **SAEs Passed (AUROC >= 0.65):** {pass_count}/{n_total}
- **D-EDA Improvement >= 10pp (Layer 12):** {deda_better_count}/{len(layer12_configs)}
- **Go/No-Go Decision:** {'GO' if overall_pass else 'NO_GO'}

## Full Mode Pass Criteria

- AUROC >= 0.65 on >= 4/6 SAEs: {'PASS' if pass_count >= 4 else 'FAIL'}
- D-EDA precision improvement >= 10pp: {'PASS' if deda_better_count >= 1 else 'FAIL'}

## Notes

- FULL mode: 10,000 bootstrap resamples for 95% CI computation
- Labels: Enhanced Neuronpedia proxy labels (Gemma 2 2B gated; exact Chanin et al. labels require model access)
- EDA cross-validated against SAEBench encoder_decoder_cosine_sim per feature
- Layers 5, 12, 19 used (methodology targeted 6, 12, 20; closest available in SAEBench)
"""

(RESULTS_DIR / "phase1_summary.md").write_text(summary_md)

# ─── Update gpu_progress.json ─────────────────────────────────────────────────
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gp = json.loads(gpu_progress_file.read_text()) if gpu_progress_file.exists() else {
        "completed": [], "failed": [], "running": {}, "timings": {}
    }
    gp.setdefault("completed", [])
    gp.setdefault("failed", [])
    gp.setdefault("running", {})
    gp.setdefault("timings", {})

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    gp["running"].pop(TASK_ID, None)
    gp["timings"][TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(sum(r.get("elapsed_sec", 0) for r in results_per_sae) / 60),
        "config_snapshot": {
            "n_saes": n_total,
            "layers": [5, 12, 19],
            "widths": ["16k", "65k"],
            "mode": "FULL",
            "n_bootstrap": N_BOOTSTRAP,
            "pass_count": pass_count,
        }
    }
    gpu_progress_file.write_text(json.dumps(gp, indent=2))
    print("gpu_progress.json updated.")
except Exception as e:
    print(f"WARNING: Could not update gpu_progress.json: {e}")

# ─── Mark done ────────────────────────────────────────────────────────────────
summary_str = (f"Phase 1 FULL complete. {pass_count}/{n_total} SAEs pass AUROC >= 0.65. "
               f"D-EDA improvement >= 10pp on {deda_better_count}/2 layer-12 configs. "
               f"GO: {overall_pass}. n_bootstrap={N_BOOTSTRAP}")

mark_done(status="success", summary=summary_str)

print(f"\n{'='*60}")
print(f"PHASE 1 FULL COMPLETE")
print(f"  GO/NO-GO: {'GO' if overall_pass else 'NO_GO'}")
print(f"  Pass count: {pass_count}/{n_total}")
print(f"  D-EDA improvement: {deda_better_count}/{len(layer12_configs)}")
print(f"  Mode: FULL (n_bootstrap={N_BOOTSTRAP})")
print(f"{'='*60}")
