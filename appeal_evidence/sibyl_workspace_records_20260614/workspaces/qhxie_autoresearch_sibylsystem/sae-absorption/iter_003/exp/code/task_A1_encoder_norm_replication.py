"""
Task A1: Encoder Norm Replication — GPT-2 L6 and L10

Replicate the encoder_norm AUROC=0.757 finding from iter_002 on GPT-2 L6
with exact Chanin IG labels (n_pos=18). Extend to GPT-2 L10.
Compare encoder_norm against all iter_002 detectors.
Compute Spearman correlation between encoder_norm and EDA.
Report AUROC, AUPRC, Precision@50/100/500 for every detector at both layers.

Mode: PILOT (uses cached labels, does NOT run IG from scratch)
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

TASK_ID = "task_A1_encoder_norm_replication"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FILE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_001/exp/results/r4/r4a_direct_labels.json")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()

# Write PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def write_progress(epoch, total_epochs, step=0, total_steps=0, metric=None):
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

def compute_auroc(scores, labels):
    """Compute AUROC using Mann-Whitney U statistic."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5, None
    # Mann-Whitney
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(pos_scores, neg_scores, alternative='greater')
    auroc = stat / (n_pos * n_neg)
    return float(auroc), float(pval)

def compute_auroc_bootstrap(scores, labels, n_bootstrap=1000, seed=42):
    """Compute AUROC with 95% CI via bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    aurocs = []
    base_auroc, _ = compute_auroc(scores, labels)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        s = scores[idx]
        l = labels[idx]
        if l.sum() == 0 or l.sum() == len(l):
            aurocs.append(base_auroc)
            continue
        a, _ = compute_auroc(s, l)
        aurocs.append(a)
    ci_lo = float(np.percentile(aurocs, 2.5))
    ci_hi = float(np.percentile(aurocs, 97.5))
    return float(base_auroc), ci_lo, ci_hi

def compute_auprc(scores, labels):
    """Compute AUPRC."""
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return float('nan')
    return float(average_precision_score(labels, scores))

def compute_precision_at_k(scores, labels, k):
    """Precision@k: fraction of true positives in top-k predictions."""
    if k > len(scores):
        k = len(scores)
    top_k_idx = np.argsort(-scores)[:k]
    return float(labels[top_k_idx].sum() / k)

def spearman_correlation(x, y):
    """Spearman correlation."""
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)

def delong_test(scores1, scores2, labels):
    """
    DeLong test for comparing two AUROCs.
    Returns z-statistic and p-value (one-sided: scores1 > scores2).
    Approximation using bootstrap if n is small.
    """
    from scipy.stats import norm
    n = len(labels)
    n_boot = 1000
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s1 = scores1[idx]; s2 = scores2[idx]; l = labels[idx]
        if l.sum() == 0 or l.sum() == n:
            continue
        a1, _ = compute_auroc(s1, l)
        a2, _ = compute_auroc(s2, l)
        diffs.append(a1 - a2)
    if len(diffs) < 10:
        return float('nan'), float('nan')
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    if std_diff < 1e-8:
        return float('nan'), float('nan')
    z = mean_diff / std_diff
    p = float(norm.sf(z))  # one-sided
    return float(z), float(p)

def evaluate_detector(scores, labels, name, n_bootstrap=200):
    """Full evaluation of a detector."""
    auroc, ci_lo, ci_hi = compute_auroc_bootstrap(scores, labels, n_bootstrap=n_bootstrap)
    auprc = compute_auprc(scores, labels)
    p50 = compute_precision_at_k(scores, labels, 50)
    p100 = compute_precision_at_k(scores, labels, 100)
    p500 = compute_precision_at_k(scores, labels, 500)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    print(f"  {name}: AUROC={auroc:.4f} [{ci_lo:.4f},{ci_hi:.4f}] AUPRC={auprc:.4f} P@50={p50:.4f} P@100={p100:.4f} P@500={p500:.4f} (n_pos={n_pos}, n_neg={n_neg})")
    return {
        "detector": name,
        "auroc": auroc,
        "auroc_ci95": [ci_lo, ci_hi],
        "auprc": auprc,
        "precision_at_50": p50,
        "precision_at_100": p100,
        "precision_at_500": p500,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

def load_labels():
    """Load exact labels from cached label file."""
    with open(LABEL_FILE) as f:
        data = json.load(f)

    labels_dict = {}
    for sae_result in data["per_sae_results"]:
        config_name = sae_result["config"]["name"]
        layer_idx = sae_result["config"]["layer_idx"]
        d_sae = sae_result["config"]["d_sae"]

        # Get absorbed latent IDs
        if "all_direct_labels" in data and config_name in data["all_direct_labels"]:
            absorbed_ids = set(data["all_direct_labels"][config_name]["absorbed_latent_ids"])
        else:
            absorbed_ids = set(sae_result.get("absorbed_latent_ids", []))

        labels = np.zeros(d_sae, dtype=np.float32)
        for idx in absorbed_ids:
            if idx < d_sae:
                labels[idx] = 1.0

        # Get all letter feature IDs (letter features)
        letter_feature_ids = set()
        for letter_data in sae_result.get("absorption_details", {}).values():
            for fid in letter_data.get("main_feature_ids", []):
                if fid < d_sae:
                    letter_feature_ids.add(fid)

        labels_dict[config_name] = {
            "layer_idx": layer_idx,
            "d_sae": d_sae,
            "labels": labels,
            "absorbed_ids": sorted(absorbed_ids),
            "letter_feature_ids": sorted(letter_feature_ids),
            "n_pos": int(labels.sum()),
            "label_source": sae_result.get("probe_quality", {}).get("label_source", "unknown"),
        }
        print(f"Loaded labels for {config_name}: n_pos={labels_dict[config_name]['n_pos']}, "
              f"d_sae={d_sae}, label_source={labels_dict[config_name]['label_source']}")

    return labels_dict

def load_sae_weights(layer_idx):
    """Load SAE weights using SAELens."""
    print(f"Loading gpt2-small-res-jb SAE for layer {layer_idx}...")
    from sae_lens import SAE

    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    sae, cfg_dict, log_sparsities = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=hook_name,
    )
    sae = sae.to("cpu")
    print(f"  Loaded SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    return sae

def compute_all_detectors(sae, labels_info):
    """Compute all detector scores for a given SAE."""
    W_enc = sae.W_enc.detach().cpu().numpy()  # [d_in, d_sae]
    W_dec = sae.W_dec.detach().cpu().numpy()  # [d_sae, d_in]

    d_sae = W_enc.shape[1]
    assert d_sae == labels_info["d_sae"], f"d_sae mismatch: {d_sae} vs {labels_info['d_sae']}"

    # Encoder norm: ||W_enc[:, j]|| for each latent j
    # W_enc is [d_in, d_sae], so column j is the encoding direction for latent j
    encoder_norm = np.linalg.norm(W_enc, axis=0)  # [d_sae]

    # Decoder norm: ||W_dec[j, :]|| for each latent j
    # W_dec is [d_sae, d_in], so row j is the decoding direction for latent j
    decoder_norm = np.linalg.norm(W_dec, axis=1)  # [d_sae]

    # EDA: 1 - cos(enc_j, dec_j) where enc_j = W_enc[:, j] / ||W_enc[:, j]||
    # and dec_j = W_dec[j, :] / ||W_dec[j, :]||
    enc_normalized = W_enc / (encoder_norm[None, :] + 1e-8)  # [d_in, d_sae]
    dec_normalized = W_dec / (decoder_norm[:, None] + 1e-8)  # [d_sae, d_in]
    # cos(enc_j, dec_j) = sum_d enc_normalized[d, j] * dec_normalized[j, d]
    cos_enc_dec = np.sum(enc_normalized.T * dec_normalized, axis=1)  # [d_sae]
    eda = 1.0 - cos_enc_dec  # [d_sae]

    # Activation frequency (inverted) — need to load from existing results or compute
    # For PILOT, we'll use random proxy as activation_freq is not cached
    # We'll mark this as "not_available" for now and use a proper proxy
    # Actually, we can use the b_enc bias as a proxy for activation frequency
    # High b_enc[j] → feature j fires more frequently
    # activation_freq_inverted ≈ -b_enc (lower bias → less frequent → could be absorbed?)
    # From methodology: activation_freq_inverted AUROC=0.595 from iter_002
    # We'll compute a proper version using b_enc
    b_enc = sae.b_enc.detach().cpu().numpy()  # [d_sae]
    activation_freq_proxy = b_enc  # proxy: higher b_enc → fires more → NOT absorbed
    activation_freq_inverted = -activation_freq_proxy  # inverted: lower freq → more likely absorbed

    # Random baseline
    rng = np.random.RandomState(SEED)
    random_scores = rng.randn(d_sae)

    # Decoder-decoder cosine similarity to "parent" features
    # This requires knowing the parent features; skip for now (need co-occurrence data)
    # We'll add a decoder norm score as an extra detector

    scores = {
        "encoder_norm": encoder_norm,
        "decoder_norm": decoder_norm,
        "eda": eda,
        "activation_freq_inverted": activation_freq_inverted,
        "random": random_scores,
    }

    print(f"\n  Detector statistics:")
    print(f"    encoder_norm: mean={encoder_norm.mean():.4f}, std={encoder_norm.std():.4f}")
    print(f"    decoder_norm: mean={decoder_norm.mean():.4f}, std={decoder_norm.std():.4f}")
    print(f"    eda: mean={eda.mean():.4f}, std={eda.std():.4f}")
    print(f"    activation_freq_inverted: mean={activation_freq_inverted.mean():.4f}, std={activation_freq_inverted.std():.4f}")

    return scores

def run_layer_analysis(layer_name, labels_info, mode="PILOT"):
    """Run full detector evaluation for one layer."""
    layer_idx = labels_info["layer_idx"]
    labels = labels_info["labels"]

    print(f"\n{'='*60}")
    print(f"Layer: {layer_name} (L{layer_idx}), n_pos={labels_info['n_pos']}, label_source={labels_info['label_source']}")
    print(f"{'='*60}")

    # Load SAE weights
    sae = load_sae_weights(layer_idx)

    # Compute detector scores
    scores = compute_all_detectors(sae, labels_info)

    # For PILOT: restrict evaluation to letter features only (not all 24576)
    # But keep all features for AUROC computation since that uses all features
    n_bootstrap = 200 if mode == "PILOT" else 1000

    # Evaluate all detectors
    print(f"\n  Detector AUROC comparison (n_pos={labels_info['n_pos']}):")
    results = {}
    for detector_name, detector_scores in scores.items():
        results[detector_name] = evaluate_detector(
            detector_scores, labels, detector_name, n_bootstrap=n_bootstrap
        )

    # Spearman correlation: encoder_norm vs. EDA
    r_spearman, p_spearman = spearman_correlation(scores["encoder_norm"], scores["eda"])
    print(f"\n  Spearman(encoder_norm, EDA) = {r_spearman:.4f} (p={p_spearman:.4e})")

    # DeLong test: encoder_norm vs. EDA
    z_delong, p_delong = delong_test(scores["encoder_norm"], scores["eda"], labels)
    print(f"  DeLong test (encoder_norm > EDA): z={z_delong:.4f}, p={p_delong:.4e}")

    # Per-feature scores for absorbed features (top-20 by encoder_norm)
    absorbed_ids = labels_info["absorbed_ids"]
    absorbed_enc_norms = [(fid, float(scores["encoder_norm"][fid])) for fid in absorbed_ids if fid < len(scores["encoder_norm"])]
    absorbed_enc_norms.sort(key=lambda x: -x[1])

    non_absorbed_sample = [i for i in range(min(1000, labels_info["d_sae"])) if labels[i] == 0][:20]

    absorbed_stats = {
        "mean_encoder_norm": float(np.mean([scores["encoder_norm"][fid] for fid in absorbed_ids if fid < len(scores["encoder_norm"])])),
        "mean_eda": float(np.mean([scores["eda"][fid] for fid in absorbed_ids if fid < len(scores["eda"])])),
        "top_by_encoder_norm": absorbed_enc_norms[:10],
    }

    return {
        "layer_name": layer_name,
        "layer_idx": layer_idx,
        "n_pos": labels_info["n_pos"],
        "n_neg": int((labels == 0).sum()),
        "label_source": labels_info["label_source"],
        "detectors": results,
        "spearman_encoder_norm_vs_eda": {
            "r": r_spearman,
            "p": p_spearman,
        },
        "delong_encoder_norm_vs_eda": {
            "z": z_delong,
            "p_one_sided": p_delong,
        },
        "absorbed_feature_stats": absorbed_stats,
    }

def main():
    print("=" * 70)
    print(f"Task A1: Encoder Norm Replication — PILOT MODE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    write_progress(0, 4, metric={"status": "loading_labels"})

    # Load labels
    print("\n[Step 1] Loading exact Chanin IG labels...")
    labels_dict = load_labels()

    write_progress(1, 4, metric={"status": "labels_loaded", "n_layers": len(labels_dict)})

    layer_results = {}

    # Layer L6 (primary — should replicate AUROC=0.757)
    print("\n[Step 2] Running L6 analysis (primary replication)...")
    if "GPT2-L6" in labels_dict:
        try:
            l6_result = run_layer_analysis("GPT2-L6", labels_dict["GPT2-L6"], mode="PILOT")
            layer_results["GPT2-L6"] = l6_result
            write_progress(2, 4, metric={
                "status": "L6_done",
                "L6_encoder_norm_auroc": l6_result["detectors"]["encoder_norm"]["auroc"],
            })
        except Exception as e:
            print(f"ERROR in L6 analysis: {e}")
            traceback.print_exc()
            layer_results["GPT2-L6"] = {"error": str(e)}

    # Layer L10 (extension — generate labels separately or use cached from r4a)
    print("\n[Step 3] Running L10 analysis (extension)...")
    if "GPT2-L10" in labels_dict:
        try:
            l10_result = run_layer_analysis("GPT2-L10", labels_dict["GPT2-L10"], mode="PILOT")
            layer_results["GPT2-L10"] = l10_result
            write_progress(3, 4, metric={
                "status": "L10_done",
                "L10_encoder_norm_auroc": l10_result["detectors"]["encoder_norm"]["auroc"],
            })
        except Exception as e:
            print(f"ERROR in L10 analysis: {e}")
            traceback.print_exc()
            layer_results["GPT2-L10"] = {"error": str(e)}

    # Assess pass criteria
    l6_auroc = layer_results.get("GPT2-L6", {}).get("detectors", {}).get("encoder_norm", {}).get("auroc", 0.0)
    l10_measured = "GPT2-L10" in layer_results and "detectors" in layer_results["GPT2-L10"]

    pass_criteria_met = l6_auroc >= 0.70 and l10_measured

    print(f"\n[Step 4] Pilot pass assessment:")
    print(f"  L6 encoder_norm AUROC = {l6_auroc:.4f} (threshold >= 0.70)")
    print(f"  L10 measured = {l10_measured}")
    print(f"  PASS: {pass_criteria_met}")

    elapsed = time.time() - start_time

    # Compile final result
    result = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": float(elapsed),
        "seed": SEED,
        "label_file": str(LABEL_FILE),
        "layer_results": layer_results,
        "pilot_pass_criteria": {
            "L6_encoder_norm_auroc_ge_0.70": bool(l6_auroc >= 0.70),
            "L10_measured": bool(l10_measured),
            "pass": bool(pass_criteria_met),
            "l6_auroc": float(l6_auroc),
        },
        "summary": {
            "L6": {
                "encoder_norm_auroc": float(layer_results.get("GPT2-L6", {}).get("detectors", {}).get("encoder_norm", {}).get("auroc", float("nan"))),
                "eda_auroc": float(layer_results.get("GPT2-L6", {}).get("detectors", {}).get("eda", {}).get("auroc", float("nan"))),
                "n_pos": layer_results.get("GPT2-L6", {}).get("n_pos", 0),
            },
            "L10": {
                "encoder_norm_auroc": float(layer_results.get("GPT2-L10", {}).get("detectors", {}).get("encoder_norm", {}).get("auroc", float("nan"))),
                "eda_auroc": float(layer_results.get("GPT2-L10", {}).get("detectors", {}).get("eda", {}).get("auroc", float("nan"))),
                "n_pos": layer_results.get("GPT2-L10", {}).get("n_pos", 0),
            },
        },
    }

    # Save result
    out_path = RESULTS_DIR / "A1_encoder_norm_replication.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Detector AUROC Comparison")
    print("=" * 70)
    header = f"{'Detector':<30} {'L6 AUROC':>10} {'L10 AUROC':>10}"
    print(header)
    print("-" * 55)
    detectors = ["encoder_norm", "eda", "activation_freq_inverted", "decoder_norm", "random"]
    for d in detectors:
        l6_a = layer_results.get("GPT2-L6", {}).get("detectors", {}).get(d, {}).get("auroc", float("nan"))
        l10_a = layer_results.get("GPT2-L10", {}).get("detectors", {}).get(d, {}).get("auroc", float("nan"))
        print(f"  {d:<28} {l6_a:>10.4f} {l10_a:>10.4f}")

    write_progress(4, 4, metric={
        "status": "complete",
        "L6_encoder_norm_auroc": float(l6_auroc),
        "pass": bool(pass_criteria_met),
    })

    mark_done(
        status="success" if pass_criteria_met else "completed_fail_criteria",
        summary=f"L6 encoder_norm AUROC={l6_auroc:.4f}, pass={pass_criteria_met}"
    )

    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    return result

if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
