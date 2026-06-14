"""
pilot_eda_layer12_16k_full.py

EDA Pilot (FULL MODE): Complete validation of EDA on Gemma Scope Layer 12 16k
Task: pilot_eda_layer12_16k (FULL run)

Full mode differences vs. pilot:
- 10,000 bootstrap resamples (vs 1,000 in pilot)
- Full ROC curve computation and saving
- Enhanced statistical analysis
- Proper DONE marker creation
- GPU progress update

CONSTRAINT NOTE: Gemma 2 2B is HF-gated (requires model access). Exact Chanin et al.
activation-based labels are not available. This script uses:
  1. Neuronpedia auto-interp labels (features labeled as "letter X") as proxy labels
  2. SAEBench pre-computed absorption rates for context/calibration
  3. Internal consistency checks (EDA distribution analysis)
  4. SAE-Spelling / SAEBench per-feature absorption data where available
"""

import json
import os
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "pilot_eda_layer12_16k"
START_TIME = datetime.now()

# ── Random seed ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
DEVICE = f"cuda:5" if torch.cuda.is_available() and torch.cuda.device_count() > 5 else "cuda:0"
N_BOOTSTRAP = 10000  # FULL mode: 10,000 resamples

print(f"[{datetime.now().isoformat()}] Starting {TASK_ID} (FULL MODE)")
print(f"Device: {DEVICE}")
print(f"Bootstrap resamples: {N_BOOTSTRAP}")

# ── Write PID ──────────────────────────────────────────────────────────────────
pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    prog = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    prog.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if pid_file.exists():
        pid_file.unlink()
    prog_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    final_prog = {}
    if prog_file.exists():
        try:
            final_prog = json.loads(prog_file.read_text())
        except Exception:
            pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_prog,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"  DONE marker written: {marker}")


# ── Step 1: Load SAE ───────────────────────────────────────────────────────────
print("\n[1/7] Loading Gemma Scope Layer 12 16k SAE...")
report_progress(0, 7, metric={"stage": "loading_sae", "mode": "FULL"})

import warnings
from sae_lens import SAE

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=DEVICE,
    )

W_enc = sae.W_enc.to(DEVICE)  # [d_in, d_sae]
W_dec = sae.W_dec.to(DEVICE)  # [d_sae, d_in]
b_enc = sae.b_enc.to(DEVICE)  # [d_sae]
d_in, d_sae = W_enc.shape
print(f"  W_enc: {W_enc.shape}, W_dec: {W_dec.shape}")
print(f"  d_in={d_in}, d_sae={d_sae}")

report_progress(1, 7, metric={"stage": "sae_loaded", "d_sae": d_sae})


# ── Step 2: Compute EDA for all latents ───────────────────────────────────────
print("\n[2/7] Computing EDA(j) = 1 - cos(w_{e,j}, d_j) for all latents...")

with torch.no_grad():
    # Encoder rows: w_{e,j} = W_enc[:, j]  (d_in-dimensional)
    w_enc = W_enc.T  # [d_sae, d_in]
    # Decoder rows: d_j = W_dec[j, :]  (d_in-dimensional)
    w_dec = W_dec  # [d_sae, d_in]

    # Cosine similarity: cos(w_{e,j}, d_j)
    enc_norms = torch.norm(w_enc, dim=1, keepdim=True).clamp(min=1e-8)
    dec_norms = torch.norm(w_dec, dim=1, keepdim=True).clamp(min=1e-8)

    cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms.squeeze() * dec_norms.squeeze())
    eda_scores = (1 - cos_sim).cpu().numpy()  # [d_sae]

print(f"  EDA computed for {d_sae} latents")
print(f"  EDA stats: min={eda_scores.min():.4f}, max={eda_scores.max():.4f}, "
      f"mean={eda_scores.mean():.4f}, std={eda_scores.std():.4f}")
print(f"  EDA percentiles: p25={np.percentile(eda_scores, 25):.4f}, "
      f"p50={np.percentile(eda_scores, 50):.4f}, p75={np.percentile(eda_scores, 75):.4f}")

report_progress(2, 7, metric={"stage": "eda_computed", "eda_mean": float(eda_scores.mean())})


# ── Step 3: Load SAEBench pre-computed absorption data ────────────────────────
print("\n[3/7] Loading SAEBench pre-computed absorption data...")

saebench_absorption = {}
absorption_labels_from_saebench = None

try:
    from huggingface_hub import hf_hub_download

    # Load absorption results
    path = hf_hub_download(
        repo_id="adamkarvonen/sae_bench_results",
        repo_type="dataset",
        filename="absorption/gemma-scope-2b-pt-res-canonical/gemma-scope-2b-pt-res-canonical_layer_12_width_16k_canonical_eval_results.json",
    )
    with open(path) as f:
        abs_data = json.load(f)

    saebench_absorption = {
        "mean_absorption_score": abs_data["eval_result_metrics"]["mean"]["mean_absorption_score"],
        "mean_num_split_features": abs_data["eval_result_metrics"]["mean"]["mean_num_split_features"],
        "per_letter_details": abs_data.get("eval_result_details", []),
    }
    print(f"  SAEBench absorption score: {saebench_absorption['mean_absorption_score']:.4f}")
    print(f"  Mean split features per letter: {saebench_absorption['mean_num_split_features']:.2f}")
    print(f"  Per-letter details: {len(saebench_absorption['per_letter_details'])} letters")

    # Extract split (absorbed) feature IDs from per_letter_details
    split_feature_ids = set()
    for letter_detail in saebench_absorption["per_letter_details"]:
        # SAEBench stores the split feature IDs for each letter
        if "split_features" in letter_detail:
            for sf in letter_detail["split_features"]:
                if isinstance(sf, int):
                    split_feature_ids.add(sf)
                elif isinstance(sf, dict) and "feature_id" in sf:
                    split_feature_ids.add(sf["feature_id"])
        # Also check "absorption_features" key
        if "absorption_features" in letter_detail:
            for sf in letter_detail["absorption_features"]:
                if isinstance(sf, int):
                    split_feature_ids.add(sf)
                elif isinstance(sf, dict) and "feature_id" in sf:
                    split_feature_ids.add(sf["feature_id"])

    if split_feature_ids:
        print(f"  Extracted {len(split_feature_ids)} split/absorbed feature IDs from SAEBench")
        absorption_labels_from_saebench = np.zeros(d_sae, dtype=int)
        for fid in split_feature_ids:
            if 0 <= fid < d_sae:
                absorption_labels_from_saebench[fid] = 1
        n_sae_pos = absorption_labels_from_saebench.sum()
        print(f"  SAEBench absorption labels: {n_sae_pos} positives / {d_sae} total ({n_sae_pos/d_sae*100:.2f}%)")
    else:
        print("  NOTE: No per-feature split IDs found in SAEBench; will use Neuronpedia proxy labels")

    # Load core statistics for EDA validation
    try:
        core_path = hf_hub_download(
            repo_id="adamkarvonen/sae_bench_results",
            repo_type="dataset",
            filename="core_with_feature_statistics/gemma-scope-2b-pt-res-canonical/gemma-scope-2b-pt-res-canonical_layer_12_width_16k_canonical_eval_results.json",
        )
        with open(core_path) as f:
            core_data = json.load(f)
        core_details = core_data.get("eval_result_details", [])
        if core_details:
            enc_dec_cos_from_core = np.array([d["encoder_decoder_cosine_sim"] for d in core_details])
            eda_from_core = 1 - enc_dec_cos_from_core
            corr = np.corrcoef(eda_scores, eda_from_core)[0, 1]
            max_diff = float(np.abs(eda_scores - eda_from_core).max())
            print(f"  EDA vs SAEBench validation: corr={corr:.6f}, max_diff={max_diff:.6f}")
            saebench_eda_validation = {"correlation": float(corr), "max_diff": max_diff, "n_features": len(core_details)}
        else:
            saebench_eda_validation = {"note": "No core details available"}
    except Exception as e:
        print(f"  WARNING: Could not load core SAEBench data: {e}")
        saebench_eda_validation = {"error": str(e)}

except Exception as e:
    print(f"  WARNING: Could not load SAEBench results: {e}")
    saebench_absorption = {"error": str(e)}
    saebench_eda_validation = {"error": str(e)}

report_progress(3, 7, metric={"stage": "saebench_loaded"})


# ── Step 4: Build proxy labels from Neuronpedia ───────────────────────────────
print("\n[4/7] Constructing proxy first-letter labels from Neuronpedia...")
print("  NOTE: Gemma 2 2B is HF-gated; using Neuronpedia auto-interp labels as proxy.")

import requests

def query_neuronpedia_letter_features(letter, pagesize=100):
    """Query Neuronpedia for features related to a specific letter."""
    try:
        resp = requests.post(
            "https://www.neuronpedia.org/api/explanation/search",
            json={
                "modelId": "gemma-2-2b",
                "layers": ["12-gemmascope-res-16k"],
                "query": f"letter {letter}",
                "page": 0,
                "pageSize": pagesize,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
    except Exception as e:
        print(f"    Error querying letter {letter}: {e}")
    return []

proxy_positive_feature_ids = set()
proxy_feature_details = {}

print("  Querying Neuronpedia for each letter a-z...")
for letter in string.ascii_lowercase:
    results = query_neuronpedia_letter_features(letter, pagesize=50)
    letter_feats = []
    for r in results:
        feat_idx = int(r["index"])
        desc = r.get("description", "").lower()
        cos_sim_val = r.get("cosine_similarity", 0)

        is_letter_feature = (
            f'letter "{letter}"' in desc or
            f"letter {letter}" in desc or
            f"the letter '{letter}'" in desc or
            (f" {letter} " in desc and "letter" in desc and "first" in desc) or
            (f"starting with {letter}" in desc) or
            (f"begins with {letter}" in desc)
        )

        if is_letter_feature and cos_sim_val > 0.3:
            proxy_positive_feature_ids.add(feat_idx)
            letter_feats.append((feat_idx, desc, cos_sim_val))

    if letter_feats:
        print(f"  Letter '{letter}': {len(letter_feats)} features -> {[idx for idx, _, _ in letter_feats[:3]]}")
    proxy_feature_details[letter] = letter_feats

print(f"\n  Total unique proxy positive features: {len(proxy_positive_feature_ids)}")

# Broad query
try:
    resp = requests.post(
        "https://www.neuronpedia.org/api/explanation/search",
        json={
            "modelId": "gemma-2-2b",
            "layers": ["12-gemmascope-res-16k"],
            "query": "first letter of word",
            "page": 0,
            "pageSize": 100,
        },
        timeout=15,
    )
    if resp.status_code == 200:
        data = resp.json()
        for r in data.get("results", []):
            desc = r.get("description", "").lower()
            cos_sim_val = r.get("cosine_similarity", 0)
            if ("first letter" in desc or "initial letter" in desc) and cos_sim_val > 0.35:
                proxy_positive_feature_ids.add(int(r["index"]))
except Exception as e:
    print(f"  WARNING: Broad query failed: {e}")

print(f"  After broad query: {len(proxy_positive_feature_ids)} proxy positive features")

# Build binary labels array
proxy_labels = np.zeros(d_sae, dtype=int)
for feat_id in proxy_positive_feature_ids:
    if 0 <= feat_id < d_sae:
        proxy_labels[feat_id] = 1

n_positives = proxy_labels.sum()
n_negatives = d_sae - n_positives
print(f"  Binary labels: {n_positives} positives, {n_negatives} negatives ({n_positives/d_sae*100:.2f}%)")

report_progress(4, 7, metric={"stage": "proxy_labels_built", "n_positives": int(n_positives)})


# ── Step 5: Compute AUROC, ROC curve, and bootstrap CI ───────────────────────
print(f"\n[5/7] Computing AUROC/AUPRC with {N_BOOTSTRAP} bootstrap resamples (FULL MODE)...")

auroc_results = {}

def compute_auroc_metrics(labels, scores, label_type, n_bootstrap=N_BOOTSTRAP):
    """Compute AUROC, AUPRC, ROC curve, and bootstrap CI."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos < 5 or n_neg < 5:
        return {"error": f"Insufficient labels: {n_pos} pos, {n_neg} neg", "n_pos": int(n_pos), "n_neg": int(n_neg)}

    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    # ROC curve (sampled for storage)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Downsample to 100 points
    step = max(1, len(fpr) // 100)
    roc_sampled = {
        "fpr": fpr[::step].tolist(),
        "tpr": tpr[::step].tolist(),
        "thresholds": thresholds[::step].tolist() if len(thresholds) > 0 else [],
    }

    # Precision@k thresholds
    prec_at_recall = {}
    pr_prec, pr_rec, pr_thr = precision_recall_curve(labels, scores)
    for recall_threshold in [0.25, 0.50, 0.75]:
        mask = pr_rec >= recall_threshold
        if mask.any():
            prec_at_recall[f"prec_at_{int(recall_threshold*100)}_recall"] = float(pr_prec[mask][-1])

    # Bootstrap CI
    print(f"  Computing {n_bootstrap} bootstrap samples for {label_type}...")
    rng = np.random.default_rng(SEED)
    bootstrap_aurocs = []
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(labels), size=len(labels))
        bl = labels[idx]
        bs = scores[idx]
        if bl.sum() > 0 and (1 - bl).sum() > 0:
            bootstrap_aurocs.append(roc_auc_score(bl, bs))
        if (i + 1) % 2000 == 0:
            print(f"    Progress: {i+1}/{n_bootstrap} resamples...")

    ci_lower = float(np.percentile(bootstrap_aurocs, 2.5))
    ci_upper = float(np.percentile(bootstrap_aurocs, 97.5))
    print(f"  {label_type} AUROC: {auroc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"  {label_type} AUPRC: {auprc:.4f}")

    # EDA distributions by label
    pos_eda = scores[labels == 1]
    neg_eda = scores[labels == 0]
    pooled_std = np.sqrt((pos_eda.std()**2 + neg_eda.std()**2) / 2)
    cohens_d = float((pos_eda.mean() - neg_eda.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "auroc": auroc,
        "auprc": auprc,
        "bootstrap_ci_95_lower": ci_lower,
        "bootstrap_ci_95_upper": ci_upper,
        "n_bootstrap": len(bootstrap_aurocs),
        "precision_at_recall": prec_at_recall,
        "roc_curve_sampled": roc_sampled,
        "eda_distribution": {
            "positive": {
                "mean": float(pos_eda.mean()),
                "median": float(np.median(pos_eda)),
                "std": float(pos_eda.std()),
                "p25": float(np.percentile(pos_eda, 25)),
                "p75": float(np.percentile(pos_eda, 75)),
            },
            "negative": {
                "mean": float(neg_eda.mean()),
                "median": float(np.median(neg_eda)),
                "std": float(neg_eda.std()),
                "p25": float(np.percentile(neg_eda, 25)),
                "p75": float(np.percentile(neg_eda, 75)),
            },
            "cohens_d": cohens_d,
        },
    }


# AUROC with proxy labels (primary)
auroc_proxy = compute_auroc_metrics(proxy_labels, eda_scores, "Proxy (Neuronpedia)")
auroc_results["proxy_labels"] = auroc_proxy
report_progress(5, 7, metric={"stage": "proxy_auroc_done",
                               "auroc_proxy": auroc_proxy.get("auroc"),
                               "ci_lower": auroc_proxy.get("bootstrap_ci_95_lower")})

# AUROC with SAEBench absorption labels (if available)
if absorption_labels_from_saebench is not None and absorption_labels_from_saebench.sum() > 5:
    print("\n  Also computing AUROC with SAEBench absorption labels...")
    auroc_saebench = compute_auroc_metrics(absorption_labels_from_saebench, eda_scores, "SAEBench absorption", n_bootstrap=N_BOOTSTRAP)
    auroc_results["saebench_labels"] = auroc_saebench
else:
    auroc_results["saebench_labels"] = {"note": "No SAEBench per-feature absorption labels extracted"}

report_progress(6, 7, metric={"stage": "all_auroc_done"})


# ── Step 6: Decision and analysis ─────────────────────────────────────────────
print("\n[6/7] Making GO/NO-GO decision...")

# Use best available AUROC (prefer SAEBench if available and good)
primary_auroc = auroc_results.get("saebench_labels", {}).get("auroc") or auroc_results.get("proxy_labels", {}).get("auroc")
primary_ci_lower = auroc_results.get("saebench_labels", {}).get("bootstrap_ci_95_lower") or auroc_results.get("proxy_labels", {}).get("bootstrap_ci_95_lower")
primary_label_type = "saebench" if auroc_results.get("saebench_labels", {}).get("auroc") else "proxy"

if primary_auroc is None:
    decision = "INSUFFICIENT_LABELS"
    decision_note = "No valid labels available for AUROC computation."
elif primary_auroc > 0.65:
    decision = "GO"
    decision_note = f"AUROC={primary_auroc:.3f} > 0.65 threshold (labels: {primary_label_type}). Proceed to Phase 1 full validation."
elif primary_auroc < 0.55:
    decision = "INVESTIGATE"
    decision_note = f"AUROC={primary_auroc:.3f} < 0.55 (labels: {primary_label_type}). Investigate root cause."
else:
    decision = "PROCEED_WITH_CAUTION"
    decision_note = f"AUROC={primary_auroc:.3f} in [0.55, 0.65] gray zone (labels: {primary_label_type}). Add polysemanticity stratification."

print(f"  Primary AUROC ({primary_label_type}): {primary_auroc:.4f}")
if primary_ci_lower:
    print(f"  95% CI lower bound: {primary_ci_lower:.4f}")
print(f"  Decision: {decision}")
print(f"  Note: {decision_note}")

# Pass criteria check
pass_criteria = {
    "auroc_gt_065": primary_auroc is not None and primary_auroc > 0.65,
    "ci_lower_gt_055": primary_ci_lower is not None and primary_ci_lower > 0.55,
    "proxy_auroc_gt_065": auroc_results.get("proxy_labels", {}).get("auroc", 0) > 0.65,
    "labels_source": primary_label_type,
    "eda_validation_with_saebench": saebench_eda_validation.get("correlation", 0) > 0.99 if "correlation" in saebench_eda_validation else None,
    "note": "PROXY/SAEBENCH LABELS - exact Chanin et al. validation blocked by gated model access",
}

# Internal consistency
print("\n  Internal consistency checks:")
print(f"  - EDA range valid: {eda_scores.min():.4f} to {eda_scores.max():.4f} (should be ~[0, 2])")
print(f"  - Positive features have higher EDA than negative: {(auroc_results.get('proxy_labels', {}).get('auroc', 0.5) > 0.5)}")

# Per-letter proxy stats
per_letter_stats = {}
for letter, feats in proxy_feature_details.items():
    if feats:
        feat_ids = [idx for idx, _, _ in feats]
        feat_edas = [float(eda_scores[idx]) for idx in feat_ids if idx < d_sae]
        per_letter_stats[letter] = {
            "n_proxy_features": len(feats),
            "feature_ids": feat_ids,
            "eda_mean": float(np.mean(feat_edas)) if feat_edas else None,
            "eda_median": float(np.median(feat_edas)) if feat_edas else None,
        }

# Top/bottom EDA features
top_eda_indices = np.argsort(eda_scores)[::-1][:10]
top_eda_samples = [
    {"feature_id": int(idx), "eda": float(eda_scores[idx]),
     "is_proxy_positive": int(proxy_labels[idx])}
    for idx in top_eda_indices
]

report_progress(7, 7, metric={"stage": "analysis_done", "decision": decision})


# ── Step 7: Save results and mark done ────────────────────────────────────────
print("\n[7/7] Saving FULL mode results and writing DONE marker...")

result = {
    "task_id": TASK_ID,
    "mode": "FULL",
    "timestamp": datetime.now().isoformat(),
    "config": {
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "d_in": d_in,
        "d_sae": d_sae,
        "seed": SEED,
        "device": DEVICE,
        "n_bootstrap": N_BOOTSTRAP,
        "label_sources": ["neuronpedia_proxy", "saebench_absorption"],
        "label_source_note": (
            "Gemma 2 2B is HF-gated; primary labels from Neuronpedia auto-interp "
            "and SAEBench absorption benchmark. Exact Chanin et al. activation-based "
            "labels require model access."
        ),
    },
    "eda_statistics": {
        "all_latents": {
            "min": float(eda_scores.min()),
            "max": float(eda_scores.max()),
            "mean": float(eda_scores.mean()),
            "std": float(eda_scores.std()),
            "p5": float(np.percentile(eda_scores, 5)),
            "p25": float(np.percentile(eda_scores, 25)),
            "p50": float(np.percentile(eda_scores, 50)),
            "p75": float(np.percentile(eda_scores, 75)),
            "p95": float(np.percentile(eda_scores, 95)),
        },
    },
    "saebench_absorption_context": saebench_absorption,
    "saebench_eda_validation": saebench_eda_validation,
    "auroc_results": auroc_results,
    "primary_auroc": primary_auroc,
    "primary_label_type": primary_label_type,
    "proxy_labels": {
        "n_positive": int(n_positives),
        "n_negative": int(n_negatives),
        "positive_feature_ids": sorted([int(x) for x in proxy_positive_feature_ids]),
        "per_letter": per_letter_stats,
    },
    "top_eda_features": top_eda_samples,
    "decision": decision,
    "decision_note": decision_note,
    "pass_criteria_check": pass_criteria,
    "go_no_go": decision,
    "confidence": 0.7 if primary_auroc is not None and primary_auroc > 0.65 and primary_label_type == "saebench" else
                  0.6 if primary_auroc is not None and primary_auroc > 0.65 else 0.4,
    "limitations": [
        "Gemma 2 2B is HF-gated; exact Chanin et al. first-letter activation-based labels not accessible",
        "Using Neuronpedia proxy and SAEBench aggregate labels; may underestimate true AUROC",
        "SAEBench provides aggregate absorption rates; per-feature labels may not be perfectly aligned",
        "Full exact validation (Phase 1) should use Chanin et al. pipeline with model access",
    ],
    "next_steps": {
        "go_path": "Proceed with Phase 1 full validation using GPT-2 Small (non-gated) for exact Chanin labels",
        "alternative_path": "Use GPT-2 Small for EDA validation with exact Chanin et al. labels (non-gated)",
    },
    "runtime_seconds": (datetime.now() - START_TIME).total_seconds(),
}

# Save to pilots directory (updated result)
output_path = RESULTS_DIR / f"{TASK_ID}.json"
output_path.write_text(json.dumps(result, indent=2))
print(f"  Results saved to: {output_path}")

# Also save ROC curve data separately for visualization
roc_path = RESULTS_DIR / f"{TASK_ID}_roc_curve.json"
roc_data = {
    "task_id": TASK_ID,
    "mode": "FULL",
    "proxy_roc": auroc_results.get("proxy_labels", {}).get("roc_curve_sampled"),
    "saebench_roc": auroc_results.get("saebench_labels", {}).get("roc_curve_sampled"),
}
roc_path.write_text(json.dumps(roc_data, indent=2))
print(f"  ROC curve data saved to: {roc_path}")

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_path.exists():
        gp = json.loads(gpu_progress_path.read_text())
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp.get("completed", []):
        gp.setdefault("completed", []).append(TASK_ID)

    # Remove from running
    gp.setdefault("running", {}).pop(TASK_ID, None)

    # Record timing
    elapsed_min = (datetime.now() - START_TIME).total_seconds() / 60.0
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 15,
        "actual_min": round(elapsed_min),
        "start_time": START_TIME.isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "sae": f"{SAE_RELEASE}/{SAE_ID}",
            "d_sae": d_sae,
            "n_bootstrap": N_BOOTSTRAP,
            "mode": "FULL",
            "decision": decision,
            "primary_auroc": primary_auroc,
        },
    }
    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print(f"  gpu_progress.json updated: {TASK_ID} marked completed")
except Exception as e:
    print(f"  WARNING: Could not update gpu_progress.json: {e}")

# Write DONE marker
auroc_str = f"{primary_auroc:.3f}" if primary_auroc is not None else "N/A"
ci_str = f"{primary_ci_lower:.3f}" if primary_ci_lower is not None else "N/A"
mark_done(
    status="success",
    summary=(
        f"EDA pilot FULL mode completed. "
        f"Primary AUROC ({primary_label_type})={auroc_str}. "
        f"Decision: {decision}. "
        f"Bootstrap CI lower: {ci_str}. "
        f"Note: proxy labels used due to gated Gemma model."
    ),
)

print(f"\n{'='*60}")
print(f"FULL MODE COMPLETE: {TASK_ID}")
print(f"Decision: {decision}")
auroc_disp = f"{primary_auroc:.4f}" if primary_auroc is not None else "N/A"
print(f"Primary AUROC ({primary_label_type}): {auroc_disp}")
if primary_ci_lower:
    print(f"95% CI lower: {primary_ci_lower:.4f}")
print(f"Runtime: {(datetime.now() - START_TIME).total_seconds():.1f}s")
print(f"{'='*60}")
