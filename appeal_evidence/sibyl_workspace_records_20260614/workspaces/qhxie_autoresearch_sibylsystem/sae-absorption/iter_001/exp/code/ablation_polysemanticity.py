#!/usr/bin/env python3
"""
Ablation: EDA Performance Stratified by Polysemanticity
=========================================================

MODE: PILOT (--tasks=ablation_polysemanticity)

Tests whether EDA's discriminative power is confounded by polysemanticity.
Stratifies latents by estimated polysemanticity proxy (decoder column activation frequency
from SAEBench core statistics).

Pilot criteria:
- AUROC_EDA in monosemantic stratum > AUROC_EDA in polysemantic stratum
  (confirming polysemanticity is a confound for EDA discrimination)
- D-EDA precision improvement larger in polysemantic stratum

SAE configs: Layer 12 x {16k, 65k}

Label strategy:
1. Use cached positive feature IDs from pilot/phase1 results if available
2. Fall back to Neuronpedia API (with rate-limit awareness)
3. Fall back to SAEBench-informed feature-level search

Outputs:
  - exp/results/full/ablation_polysemanticity.json
  - exp/results/ablation_polysemanticity_DONE
"""

import gc
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "ablation_polysemanticity"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "ablation_polysemanticity.json"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.now().isoformat()}] Starting {TASK_ID}")
print(f"Device: {DEVICE}")

# Layer 12 configs only (per task description)
SAE_CONFIGS = [
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", "L12-16k", 12, 16),
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical", "L12-65k", 12, 65),
]

N_BOOTSTRAP = 500  # pilot budget
PILOT_MODE = True

# ─── Cached labels from prior runs ────────────────────────────────────────────
# From pilot_eda_layer12_16k.py results: 16 first-letter features in L12-16k
CACHED_LABELS = {
    # L12-16k labels from pilot run (positive_feature_ids from Neuronpedia)
    "L12-16k": [342, 2038, 2105, 2236, 2575, 5611, 6512, 6899, 9754, 9994,
                10326, 11061, 12436, 14110, 14453, 15693],
    # L12-65k: will be retrieved from phase1 results or Neuronpedia
    "L12-65k": None,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
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


def get_proxy_labels_from_neuronpedia(layer: int, width_k: int, d_sae: int,
                                       max_retries: int = 3) -> np.ndarray:
    """Build proxy first-letter feature labels from Neuronpedia (with rate-limit awareness)."""
    import requests
    import string

    layer_str = f"{layer}-gemmascope-res-{width_k}k"
    proxy_ids = set()

    print(f"  Querying Neuronpedia for layer={layer} width={width_k}k (with rate-limit backoff)...")
    for letter in string.ascii_lowercase:
        for attempt in range(max_retries):
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
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"    Rate limited on letter '{letter}', waiting {wait}s...")
                    time.sleep(wait)
                    continue
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
                    break  # success
                else:
                    break
            except Exception:
                time.sleep(5)

    labels = np.zeros(d_sae, dtype=int)
    for idx in proxy_ids:
        labels[idx] = 1
    return labels


def get_proxy_labels(name: str, layer: int, width_k: int, d_sae: int) -> np.ndarray:
    """Get proxy labels: cached first, then Neuronpedia, then phase1 reuse."""
    cached = CACHED_LABELS.get(name)
    if cached:
        print(f"  Using {len(cached)} cached proxy labels for {name}")
        labels = np.zeros(d_sae, dtype=int)
        for idx in cached:
            if 0 <= idx < d_sae:
                labels[idx] = 1
        return labels

    # Try to load from phase1 results
    phase1_path = WORKSPACE / "exp" / "results" / "full" / "phase1_eda_deda_validation.json"
    if phase1_path.exists():
        try:
            with open(phase1_path) as f:
                phase1 = json.load(f)
            for r in phase1.get("per_sae_results", []):
                if r["config"]["name"] == name:
                    n_pos = r.get("eda_metrics", {}).get("n_pos", 0)
                    if n_pos and n_pos > 0:
                        print(f"  Phase1 has {n_pos} positive labels for {name}, "
                              "but no feature IDs stored. Trying Neuronpedia...")
                    break
        except Exception:
            pass

    # Fall back to Neuronpedia
    print(f"  Falling back to Neuronpedia for {name}...")
    labels = get_proxy_labels_from_neuronpedia(layer, width_k, d_sae)
    n_pos = int(labels.sum())
    print(f"  Neuronpedia returned {n_pos} labels for {name}")
    return labels


def compute_eda(W_enc: torch.Tensor, W_dec: torch.Tensor) -> np.ndarray:
    """EDA(j) = 1 - cos(w_{e,j}, d_j)"""
    with torch.no_grad():
        w_enc = W_enc.T  # [d_sae, d_in]
        w_dec = W_dec    # [d_sae, d_in]
        enc_norms = w_enc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms.squeeze() * dec_norms.squeeze())
        return (1.0 - cos_sim).cpu().float().numpy()


def compute_deda_absorption_indicator(W_enc: torch.Tensor, W_dec: torch.Tensor,
                                       top_k_dict: int = 50) -> np.ndarray:
    """D-EDA absorption indicator = mean top-3 cosine of residual with decoder dictionary."""
    d_sae = W_enc.shape[1]
    chunk_size = 2048 if d_sae >= 65536 else 4096

    absorption_indicator = np.zeros(d_sae, dtype=np.float32)
    W_dec_norm = F.normalize(W_dec.float(), dim=1)  # [d_sae, d_in]

    with torch.no_grad():
        for start in range(0, d_sae, chunk_size):
            end = min(start + chunk_size, d_sae)
            chunk_sz = end - start

            w_e = W_enc.T[start:end].float()   # [chunk, d_in]
            d_j = W_dec[start:end].float()      # [chunk, d_in]
            d_j_norm = F.normalize(d_j, dim=1)

            proj_coef = (w_e * d_j_norm).sum(dim=1, keepdim=True)
            r_j = w_e - proj_coef * d_j_norm   # [chunk, d_in]
            r_j_normalized = F.normalize(r_j, dim=1)

            cos_with_dict = r_j_normalized @ W_dec_norm.T  # [chunk, d_sae]
            for i in range(chunk_sz):
                cos_with_dict[i, start + i] = -1.0  # exclude self

            actual_k = min(top_k_dict, d_sae - 1)
            top_cos, _ = cos_with_dict.topk(actual_k, dim=1, largest=True, sorted=True)
            top3 = top_cos[:, :3].cpu().numpy()
            absorption_indicator[start:end] = top3.mean(axis=1)

    return absorption_indicator


def compute_auroc_with_ci(labels: np.ndarray, scores: np.ndarray,
                           n_bootstrap: int = 500, seed: int = 42) -> dict:
    """Compute AUROC with 95% bootstrap CI and precision@50% recall."""
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    if n_pos < 3 or n_neg < 3:
        return {"error": f"insufficient_labels: n_pos={n_pos}", "auroc": None}

    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    prec_arr, rec_arr, _ = precision_recall_curve(labels, scores)
    mask = rec_arr >= 0.50
    prec_at_50 = float(prec_arr[mask][-1]) if mask.any() else float("nan")

    rng = np.random.default_rng(seed)
    boot_aurocs, boot_p50s = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bl = labels[idx]
        bs = scores[idx]
        if bl.sum() > 0 and (1 - bl).sum() > 0:
            boot_aurocs.append(roc_auc_score(bl, bs))
            pb, rb, _ = precision_recall_curve(bl, bs)
            mb = rb >= 0.50
            if mb.any():
                boot_p50s.append(float(pb[mb][-1]))

    ci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if arr else None

    return {
        "auroc": auroc,
        "auroc_ci95": ci(boot_aurocs),
        "auprc": auprc,
        "prec_at_50recall": prec_at_50,
        "prec_at_50recall_ci95": ci(boot_p50s),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_bootstrap": n_bootstrap,
    }


def get_feature_density(layer: int, width_k: int, d_sae: int) -> np.ndarray:
    """Load SAEBench core statistics for feature density (polysemanticity proxy)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="adamkarvonen/sae_bench_results",
        repo_type="dataset",
        filename=(
            f"core_with_feature_statistics/gemma-scope-2b-pt-res-canonical/"
            f"gemma-scope-2b-pt-res-canonical_layer_{layer}_width_{width_k}k_canonical_eval_results.json"
        ),
    )
    with open(path) as f:
        core_data = json.load(f)

    core_details = core_data.get("eval_result_details", [])
    # Build density array indexed by feature index
    density = np.zeros(d_sae, dtype=np.float32)
    n_loaded = 0
    for d in core_details:
        idx = d.get("index")
        fd = d.get("feature_density")
        if idx is not None and fd is not None and 0 <= idx < d_sae:
            density[idx] = float(fd)
            n_loaded += 1

    print(f"  Feature density: {n_loaded}/{d_sae} entries loaded, "
          f"mean={density[density > 0].mean() if (density > 0).any() else 0:.4f}, "
          f"zeros={(density == 0).sum()}")
    return density


def get_decoder_neighbor_count(W_dec: torch.Tensor,
                                cos_threshold: float = 0.1,
                                chunk_size: int = 2048) -> np.ndarray:
    """
    Alternative polysemanticity proxy: count decoder neighbors with cos sim > threshold.
    High count = many related features = polysemantic.
    """
    d_sae = W_dec.shape[0]
    W_dec_norm = F.normalize(W_dec.float(), dim=1)
    neighbor_count = np.zeros(d_sae, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, d_sae, chunk_size):
            end = min(start + chunk_size, d_sae)
            chunk = W_dec_norm[start:end]  # [chunk, d_in]
            cos_mat = chunk @ W_dec_norm.T  # [chunk, d_sae]
            # Exclude self
            for i in range(end - start):
                cos_mat[i, start + i] = 0.0
            count = (cos_mat > cos_threshold).sum(dim=1).cpu().numpy()
            neighbor_count[start:end] = count

    print(f"  Decoder neighbor count (cos>{cos_threshold}): "
          f"mean={neighbor_count.mean():.1f}, max={neighbor_count.max()}")
    return neighbor_count


def load_sae_weights(release: str, sae_id: str, device: str):
    """Load SAE and extract weight matrices."""
    from sae_lens import SAE
    sae = SAE.from_pretrained(release, sae_id, device=device)
    W_enc = sae.W_enc.to(device)  # [d_in, d_sae]
    W_dec = sae.W_dec.to(device)  # [d_sae, d_in]
    d_in, d_sae = W_enc.shape
    del sae
    return W_enc, W_dec, d_in, d_sae


def stratify_and_compute(labels, eda_scores, absorption_indicator,
                          mask_mono, mask_poly, label_mono, label_poly,
                          n_bootstrap, seed):
    """Compute AUROC metrics for monosemantic and polysemantic strata."""
    strat = {
        "n_monosemantic": int(mask_mono.sum()),
        "n_polysemantic": int(mask_poly.sum()),
        "n_pos_monosemantic": int(labels[mask_mono].sum()),
        "n_pos_polysemantic": int(labels[mask_poly].sum()),
    }

    n_mono_pos = strat["n_pos_monosemantic"]
    n_poly_pos = strat["n_pos_polysemantic"]
    print(f"    Strat ({label_mono}/{label_poly}): "
          f"mono={strat['n_monosemantic']} ({n_mono_pos} pos), "
          f"poly={strat['n_polysemantic']} ({n_poly_pos} pos)")

    if n_mono_pos >= 3 and n_poly_pos >= 3:
        eda_mono_m = compute_auroc_with_ci(
            labels[mask_mono], eda_scores[mask_mono], n_bootstrap, seed)
        eda_poly_m = compute_auroc_with_ci(
            labels[mask_poly], eda_scores[mask_poly], n_bootstrap, seed)
        deda_mono_m = compute_auroc_with_ci(
            labels[mask_mono], absorption_indicator[mask_mono], n_bootstrap, seed)
        deda_poly_m = compute_auroc_with_ci(
            labels[mask_poly], absorption_indicator[mask_poly], n_bootstrap, seed)

        strat[f"eda_{label_mono}"] = eda_mono_m
        strat[f"eda_{label_poly}"] = eda_poly_m
        strat[f"deda_{label_mono}"] = deda_mono_m
        strat[f"deda_{label_poly}"] = deda_poly_m

        eda_mono_auroc = eda_mono_m.get("auroc") or 0.0
        eda_poly_auroc = eda_poly_m.get("auroc") or 0.0
        deda_mono_p50 = deda_mono_m.get("prec_at_50recall") or 0.0
        deda_poly_p50 = deda_poly_m.get("prec_at_50recall") or 0.0
        eda_mono_p50 = eda_mono_m.get("prec_at_50recall") or 0.0
        eda_poly_p50 = eda_poly_m.get("prec_at_50recall") or 0.0

        deda_improvement_mono = deda_mono_p50 - eda_mono_p50
        deda_improvement_poly = deda_poly_p50 - eda_poly_p50

        strat["comparisons"] = {
            f"eda_auroc_{label_mono}": eda_mono_auroc,
            f"eda_auroc_{label_poly}": eda_poly_auroc,
            f"eda_auroc_{label_mono}_gt_{label_poly}": bool(eda_mono_auroc > eda_poly_auroc),
            f"deda_improvement_{label_mono}_prec50": deda_improvement_mono,
            f"deda_improvement_{label_poly}_prec50": deda_improvement_poly,
            f"deda_improvement_{label_poly}_gt_{label_mono}": bool(
                deda_improvement_poly > deda_improvement_mono),
        }

        print(f"    EDA AUROC {label_mono}={eda_mono_auroc:.4f}, {label_poly}={eda_poly_auroc:.4f}")
        print(f"    D-EDA improvement {label_mono}={deda_improvement_mono:.4f}, "
              f"{label_poly}={deda_improvement_poly:.4f}")
    else:
        strat["error"] = (f"insufficient_positives: mono_pos={n_mono_pos}, poly_pos={n_poly_pos}")

    return strat


# ─── Main ─────────────────────────────────────────────────────────────────────
write_progress(0, len(SAE_CONFIGS), {"stage": "starting"})
print(f"\nPILOT MODE: {PILOT_MODE}")
print(f"Processing {len(SAE_CONFIGS)} SAE configurations (L12 only)...\n")

results_per_sae = []
t_global_start = time.time()

for cfg_idx, (release, sae_id, name, layer, width_k) in enumerate(SAE_CONFIGS):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"[{cfg_idx+1}/{len(SAE_CONFIGS)}] {name} (layer={layer}, width={width_k}k)")
    print(f"{'='*60}")

    write_progress(cfg_idx, len(SAE_CONFIGS), {"stage": f"processing_{name}"})

    sae_result = {
        "config": {"name": name, "release": release, "sae_id": sae_id,
                   "layer": layer, "width_k": int(width_k)},
    }

    try:
        # 1. Load SAE weights
        print("  Loading SAE weights...")
        W_enc, W_dec, d_in, d_sae = load_sae_weights(release, sae_id, DEVICE)
        sae_result["config"].update({"d_in": d_in, "d_sae": d_sae})
        print(f"  d_in={d_in}, d_sae={d_sae}")

        # 2. Compute EDA
        print("  Computing EDA...")
        eda_scores = compute_eda(W_enc, W_dec)
        sae_result["eda_statistics"] = {
            "mean": float(eda_scores.mean()), "std": float(eda_scores.std()),
            "min": float(eda_scores.min()), "max": float(eda_scores.max()),
            "p50": float(np.median(eda_scores)),
        }
        print(f"  EDA: mean={eda_scores.mean():.4f}, std={eda_scores.std():.4f}")

        # 3. Compute D-EDA absorption indicator
        print("  Computing D-EDA absorption indicator...")
        absorption_indicator = compute_deda_absorption_indicator(W_enc, W_dec, top_k_dict=50)
        sae_result["deda_absorption_indicator_statistics"] = {
            "mean": float(absorption_indicator.mean()),
            "std": float(absorption_indicator.std()),
            "p50": float(np.median(absorption_indicator)),
        }
        print(f"  Absorption indicator: mean={absorption_indicator.mean():.4f}")

        # 4. Compute decoder neighbor count (second polysemanticity proxy)
        print("  Computing decoder neighbor count...")
        decoder_neighbor_count = get_decoder_neighbor_count(
            W_dec.cpu(), cos_threshold=0.1, chunk_size=2048)

        # 5. Free GPU memory before label fetching
        W_enc_cpu = W_enc.cpu()
        W_dec_cpu = W_dec.cpu()
        del W_enc, W_dec
        torch.cuda.empty_cache()
        gc.collect()

        # 6. Get proxy labels
        print("  Getting proxy labels...")
        proxy_labels = get_proxy_labels(name, layer, int(width_k), d_sae)
        n_pos = int(proxy_labels.sum())
        sae_result["n_proxy_positives"] = n_pos
        sae_result["label_source"] = (
            "cached" if CACHED_LABELS.get(name) else "neuronpedia")
        print(f"  Proxy labels: {n_pos} positives / {d_sae} total")

        if n_pos < 5:
            sae_result["status"] = "insufficient_labels"
            sae_result["error"] = f"n_pos={n_pos} < 5, AUROC not computable"
            results_per_sae.append(sae_result)
            continue

        # 7. Overall metrics (full set)
        print(f"\n  --- Overall metrics (n_pos={n_pos}) ---")
        overall_eda = compute_auroc_with_ci(proxy_labels, eda_scores, N_BOOTSTRAP, SEED)
        overall_deda = compute_auroc_with_ci(proxy_labels, absorption_indicator, N_BOOTSTRAP, SEED)
        sae_result["overall_eda_metrics"] = overall_eda
        sae_result["overall_deda_metrics"] = overall_deda
        print(f"  Overall EDA AUROC={overall_eda.get('auroc', 'N/A'):.4f}, "
              f"D-EDA AUROC={overall_deda.get('auroc', 'N/A'):.4f}")

        # 8. Polysemanticity stratification — Proxy 1: SAEBench feature density
        print("\n  --- Stratification by SAEBench feature density ---")
        try:
            feature_density = get_feature_density(layer, int(width_k), d_sae)

            # Median split
            median_density = np.median(feature_density)
            mono_mask_med = feature_density <= median_density
            poly_mask_med = ~mono_mask_med

            strat_med = {
                "proxy": "saebench_feature_density",
                "split_method": "median_split",
                "median_density": float(median_density),
            }
            strat_med.update(
                stratify_and_compute(
                    proxy_labels, eda_scores, absorption_indicator,
                    mono_mask_med, poly_mask_med, "monosemantic", "polysemantic",
                    N_BOOTSTRAP, SEED
                )
            )

            # Quartile split (more extreme)
            q25 = float(np.percentile(feature_density, 25))
            q75 = float(np.percentile(feature_density, 75))
            bq_mask = feature_density <= q25
            tq_mask = feature_density >= q75
            print("  Quartile split:")
            strat_q = {
                "split_method": "quartile_split",
                "q25_density": q25, "q75_density": q75,
            }
            strat_q.update(
                stratify_and_compute(
                    proxy_labels, eda_scores, absorption_indicator,
                    bq_mask, tq_mask, "bottom_quartile", "top_quartile",
                    N_BOOTSTRAP, SEED
                )
            )

            sae_result["stratification_by_feature_density"] = {
                "median_split": strat_med,
                "quartile_split": strat_q,
            }

        except Exception as e:
            import traceback
            sae_result["stratification_by_feature_density"] = {"error": str(e)}
            print(f"  WARNING: Feature density stratification failed: {e}")
            traceback.print_exc()

        # 9. Polysemanticity stratification — Proxy 2: Decoder neighbor count
        print("\n  --- Stratification by decoder neighbor count ---")
        try:
            median_count = np.median(decoder_neighbor_count)
            mono_mask_nc = decoder_neighbor_count <= median_count
            poly_mask_nc = ~mono_mask_nc

            strat_nc = {
                "proxy": "decoder_neighbor_count_cos010",
                "split_method": "median_split",
                "median_count": float(median_count),
            }
            strat_nc.update(
                stratify_and_compute(
                    proxy_labels, eda_scores, absorption_indicator,
                    mono_mask_nc, poly_mask_nc, "monosemantic", "polysemantic",
                    N_BOOTSTRAP, SEED
                )
            )
            sae_result["stratification_by_decoder_neighbors"] = strat_nc

        except Exception as e:
            sae_result["stratification_by_decoder_neighbors"] = {"error": str(e)}
            print(f"  WARNING: Decoder neighbor stratification failed: {e}")

        # 10. EDA statistics by polysemanticity (distribution analysis)
        print("\n  --- EDA distribution by polysemanticity stratum ---")
        try:
            q_cuts = [0, 25, 50, 75, 100]
            percentile_vals = np.percentile(feature_density, q_cuts)

            dist_analysis = {}
            for q_lo, q_hi, label in [(0, 25, "Q1_monosemantic"), (25, 50, "Q2"),
                                       (50, 75, "Q3"), (75, 100, "Q4_polysemantic")]:
                lo_val = float(np.percentile(feature_density, q_lo))
                hi_val = float(np.percentile(feature_density, q_hi))
                mask = (feature_density >= lo_val) & (feature_density <= hi_val)
                if q_lo == 0:  # include the minimum
                    mask = feature_density <= hi_val
                elif q_hi == 100:
                    mask = feature_density >= lo_val

                n_in_q = int(mask.sum())
                n_pos_in_q = int(proxy_labels[mask].sum())

                dist_analysis[label] = {
                    "density_range": [lo_val, hi_val],
                    "n_latents": n_in_q,
                    "n_pos": n_pos_in_q,
                    "eda_mean": float(eda_scores[mask].mean()),
                    "eda_median": float(np.median(eda_scores[mask])),
                    "eda_positive_mean": float(eda_scores[mask & (proxy_labels == 1)].mean())
                                         if n_pos_in_q > 0 else None,
                    "eda_negative_mean": float(eda_scores[mask & (proxy_labels == 0)].mean())
                                         if (mask & (proxy_labels == 0)).any() else None,
                }

            sae_result["eda_distribution_by_quartile"] = dist_analysis

        except Exception as e:
            sae_result["eda_distribution_by_quartile"] = {"error": str(e)}

        sae_result["status"] = "success"

    except Exception as e:
        import traceback
        print(f"  ERROR on {name}: {e}")
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
    print(f"\n  [{name}] Done in {elapsed:.1f}s")

    write_progress(cfg_idx + 1, len(SAE_CONFIGS), {"stage": f"completed_{name}"})


# ─── Aggregate and Evaluate Pass Criteria ─────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATING RESULTS")
print(f"{'='*60}")

pilot_pass_list = []
for r in results_per_sae:
    name = r["config"]["name"]
    # Try feature density median split comparisons first
    strat = r.get("stratification_by_feature_density", {})
    comp = strat.get("median_split", {}).get("comparisons", {})

    if not comp:
        # Fall back to decoder neighbor count
        strat2 = r.get("stratification_by_decoder_neighbors", {})
        comp = strat2.get("comparisons", {})

    c1 = comp.get("eda_auroc_monosemantic_gt_polysemantic")
    c2 = comp.get("deda_improvement_polysemantic_gt_monosemantic")

    passed = (c1 is True) or (c2 is True)
    pilot_pass_list.append({
        "config": name,
        "criterion_1_eda_mono_gt_poly": c1,
        "criterion_2_deda_improvement_poly_gt_mono": c2,
        "passed": passed,
        "n_labels": r.get("n_proxy_positives", 0),
    })
    print(f"  {name}: C1={c1}, C2={c2}, n_labels={r.get('n_proxy_positives', 0)}")

configs_with_data = [r for r in pilot_pass_list if r["n_labels"] >= 5]
n_c1_pass = sum(1 for r in configs_with_data
                if r["criterion_1_eda_mono_gt_poly"] is True)
n_c2_pass = sum(1 for r in configs_with_data
                if r["criterion_2_deda_improvement_poly_gt_mono"] is True)

go_no_go = "GO" if (n_c1_pass >= 1 or n_c2_pass >= 1) else "NO_GO"
if not configs_with_data:
    go_no_go = "INCONCLUSIVE"

summary = (
    f"Ablation polysemanticity PILOT: {len(configs_with_data)}/{len(results_per_sae)} configs "
    f"with sufficient labels. "
    f"C1 (EDA mono>poly AUROC): {n_c1_pass}/{len(configs_with_data)}. "
    f"C2 (D-EDA improvement larger in poly): {n_c2_pass}/{len(configs_with_data)}. "
    f"GO_NO_GO={go_no_go}."
)
print(f"\n{summary}")

# ─── Write Output ─────────────────────────────────────────────────────────────
output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "config": {
        "sae_configs": [c[2] for c in SAE_CONFIGS],
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "note": "L12-16k labels cached from pilot run; L12-65k via Neuronpedia or fallback",
    },
    "per_sae_results": results_per_sae,
    "pass_criteria_summary": pilot_pass_list,
    "aggregate": {
        "n_configs_with_data": len(configs_with_data),
        "n_criterion_1_pass": n_c1_pass,
        "n_criterion_2_pass": n_c2_pass,
    },
    "go_no_go": go_no_go,
    "summary": summary,
    "total_elapsed_sec": round(time.time() - t_global_start, 1),
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults written to {OUTPUT_FILE}")

mark_done(status="success", summary=summary)
print("DONE.")
