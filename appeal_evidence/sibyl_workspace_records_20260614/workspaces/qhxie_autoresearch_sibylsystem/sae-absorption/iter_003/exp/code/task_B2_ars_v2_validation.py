"""
task_B2_ars_v2_validation.py
ARS_v2 Validation Against Chanin Labels — PILOT MODE

Computes 7 ARS formulations against exact Chanin IG labels (n_pos=18) for GPT-2 L6:
1. encoder_norm alone
2. A_cooccur(j) alone
3. O_jaccard(j) alone
4. ARS_v2(j) = encoder_norm_j * A_cooccur(j)
5. ARS_original(j) = O_jaccard(j) * A_cooccur(j) * cos2(dec_i, dec_j)
6. ARS_full(j) = encoder_norm_j * A_cooccur(j) * O_jaccard(j)
7. EDA (baseline from iter_002)

For each formulation: AUROC, AUPRC, Precision@50/100/500
DeLong test: ARS_v2 vs encoder_norm alone
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR = Path(WORKSPACE) / "exp" / "results" / "pilots"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_B2_ars_v2_validation"
LABEL_FILE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_001/exp/results/r4/r4a_direct_labels.json"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()

# Write PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, metric=None):
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": epoch, "total_steps": total_epochs,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def compute_auroc_auprc(scores, labels):
    """Compute AUROC and AUPRC using manual trapezoid rule."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5, float(labels.mean()), []
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return auroc, auprc


def compute_precision_at_k(scores, labels, k):
    """Precision@k: among top-k scored items, fraction that are positive."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    return labels[top_k_idx].sum() / k


def bootstrap_auroc_ci(scores, labels, n_bootstrap=1000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for AUROC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    n = len(labels)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if labels[idx].sum() == 0 or labels[idx].sum() == n:
            continue
        try:
            aurocs.append(roc_auc_score(labels[idx], scores[idx]))
        except:
            pass
    if len(aurocs) < 10:
        return [None, None]
    aurocs.sort()
    lo = aurocs[int(alpha / 2 * len(aurocs))]
    hi = aurocs[int((1 - alpha / 2) * len(aurocs))]
    return [lo, hi]


def delong_test(scores_a, scores_b, labels):
    """
    DeLong's test for comparing two AUROCs.
    Returns z-statistic and one-sided p-value (H1: AUROC_a > AUROC_b).
    Implementation based on DeLong et al. (1988), vectorized for performance.
    """
    from scipy import stats

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.5

    def _psi_matrix(pos_scores, neg_scores):
        """
        Compute kernel matrix: psi[i,j] = P(pos_i > neg_j) + 0.5*P(pos_i == neg_j).
        pos_scores: (n_pos,), neg_scores: (n_neg,)
        Returns: (n_pos, n_neg) matrix
        """
        diff = pos_scores[:, None] - neg_scores[None, :]  # (n_pos, n_neg)
        return (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)

    def structural_components(scores):
        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]
        psi = _psi_matrix(pos_scores, neg_scores)  # (n_pos, n_neg)
        V10 = psi.mean(axis=1)   # (n_pos,): for each pos, mean over neg
        V01 = psi.mean(axis=0)   # (n_neg,): for each neg, mean over pos
        return V10, V01

    V10_a, V01_a = structural_components(scores_a)
    V10_b, V01_b = structural_components(scores_b)

    auroc_a = float(V10_a.mean())
    auroc_b = float(V10_b.mean())

    # Covariance matrix of structural components
    S10 = np.cov(np.stack([V10_a, V10_b]))  # (2,2)
    S01 = np.cov(np.stack([V01_a, V01_b]))  # (2,2)

    # Variance of AUROC difference
    var_diff = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / n_pos + \
               (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n_neg

    if var_diff <= 0:
        return 0.0, 0.5 if auroc_a == auroc_b else (float('inf') if auroc_a > auroc_b else float('-inf'))

    z = (auroc_a - auroc_b) / np.sqrt(var_diff)
    p_one_sided = 1 - stats.norm.cdf(z)
    return float(z), float(p_one_sided)


print("[B2] Starting ARS_v2 Validation")
report_progress(0, 7, {"phase": "loading"})

try:
    # =========================================================
    # Step 1: Load Labels
    # =========================================================
    print("[B2] Loading Chanin IG labels...")
    with open(LABEL_FILE) as f:
        label_data = json.load(f)

    per_sae = label_data['per_sae_results']
    # Find L6 SAE
    l6_data = None
    for item in per_sae:
        if item['config']['layer_idx'] == 6:
            l6_data = item
            break

    if l6_data is None:
        raise ValueError("L6 SAE data not found in label file")

    absorbed_ids = l6_data['absorbed_latent_ids']
    n_absorbed = l6_data['n_absorbed']
    n_total = l6_data['n_absorbed'] + l6_data['n_non_absorbed']
    d_sae = 24576

    print(f"[B2] Labels loaded: n_pos={n_absorbed}, n_total={n_total} (d_sae={d_sae})")

    # Build binary label array
    labels = np.zeros(d_sae, dtype=np.int32)
    for fid in absorbed_ids:
        labels[fid] = 1

    n_pos = labels.sum()
    n_neg = d_sae - n_pos
    print(f"[B2] n_pos={n_pos}, n_neg={n_neg}")

    # =========================================================
    # Step 2: Load SAE weights for encoder_norm, EDA
    # =========================================================
    print("[B2] Loading SAE weights...")
    report_progress(1, 7, {"phase": "loading_sae"})

    from sae_lens import SAE

    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
    )
    sae.eval()

    with torch.no_grad():
        W_enc = sae.W_enc.detach()  # (d_in, d_sae)
        W_dec = sae.W_dec.detach()  # (d_sae, d_in)

        # Encoder norm
        encoder_norms = torch.norm(W_enc, dim=0).cpu().numpy()  # (d_sae,)

        # Decoder norm
        decoder_norms = torch.norm(W_dec, dim=1).cpu().numpy()  # (d_sae,)

        # EDA = 1 - cosine(enc_j, dec_j)
        enc_normalized = W_enc / (torch.norm(W_enc, dim=0, keepdim=True) + 1e-8)
        dec_normalized = W_dec / (torch.norm(W_dec, dim=1, keepdim=True) + 1e-8)
        cosine_vals = (enc_normalized * dec_normalized.T).sum(dim=0).cpu().numpy()
        eda = 1 - cosine_vals

        # Decoder-decoder cosine similarity for cos2 term in ARS_original
        # For each feature j: cos2(dec_i, dec_j) where i is the best parent
        # We'll load this from B1 later (parent IDs are stored there)

    print(f"[B2] SAE loaded. encoder_norms range: {encoder_norms.min():.3f} - {encoder_norms.max():.3f}")

    # =========================================================
    # Step 3: Load B1 co-occurrence arrays
    # =========================================================
    print("[B2] Loading B1 co-occurrence arrays...")
    report_progress(2, 7, {"phase": "loading_b1"})

    A_cooccur_all = np.load(RESULTS_DIR / "B1_A_cooccur_all.npy")  # (d_sae,)
    act_freq = np.load(RESULTS_DIR / "B1_activation_freq.npy")     # (d_sae,)

    # For O_jaccard: we need per-feature max jaccard with any parent
    # This is computed over letter parents in B1, but we need it for all absorbed features
    # Load B1 json for detailed per-feature results
    with open(RESULTS_DIR / "B1_cooccurrence_graph.json") as f:
        b1_data = json.load(f)

    # Build O_jaccard for all 24576 features
    # B1 computed O_jaccard_letter for letter features only
    # We need to extend this: for each absorbed feature, find its best parent jaccard
    # O_jaccard_all: use A_cooccur_argmax from B1
    letter_feature_results = b1_data.get("letter_feature_results", {})
    letter_feature_ids = b1_data.get("arrays", {}).get("letter_feature_ids", [])
    O_jaccard_letter = b1_data.get("arrays", {}).get("O_jaccard_letter", [])

    # Build per-feature O_jaccard for all d_sae features
    # For features not in the letter set, use 0 (no parent analysis done)
    # For letter features, use the stored value
    O_jaccard_all = np.zeros(d_sae, dtype=np.float32)
    for j_idx, j_global in enumerate(letter_feature_ids):
        O_jaccard_all[j_global] = O_jaccard_letter[j_idx]

    # Note: absorbed features not in letter set will have O_jaccard = 0
    # This is a limitation of the current B1 implementation

    print(f"[B2] A_cooccur_all: mean={A_cooccur_all.mean():.4f}, max={A_cooccur_all.max():.4f}")
    print(f"[B2] O_jaccard_all (letter features only): n_nonzero={int((O_jaccard_all > 0).sum())}")

    # =========================================================
    # Step 4: Compute ARS_original cos2 component
    # =========================================================
    # ARS_original needs cos2(dec_i, dec_j) where i is the best parent
    # Best parent is stored in B1 A_cooccur_argmax
    # We need to reconstruct this for all features

    print("[B2] Computing cos2 component for ARS_original...")
    report_progress(3, 7, {"phase": "computing_cos2"})

    # Reload B1 cooccurrence to get argmax parents
    A_cooccur_argmax_all = np.full(d_sae, -1, dtype=np.int32)
    for j_global_str, feat_data in letter_feature_results.items():
        j_global = int(j_global_str)
        parent_id = feat_data.get("best_parent_id", -1)
        A_cooccur_argmax_all[j_global] = parent_id

    # Compute cos2(dec_j, dec_parent) for all letter features
    with torch.no_grad():
        dec_normed = (W_dec / (torch.norm(W_dec, dim=1, keepdim=True) + 1e-8)).cpu().numpy()

    cos2_all = np.zeros(d_sae, dtype=np.float32)
    for j_idx, j_global in enumerate(letter_feature_ids):
        parent_id = int(A_cooccur_argmax_all[j_global])
        if parent_id >= 0:
            cos_val = float(np.dot(dec_normed[j_global], dec_normed[parent_id]))
            cos2_all[j_global] = cos_val ** 2
        # else stays 0

    print(f"[B2] cos2 computed: mean={cos2_all[cos2_all > 0].mean():.4f} (for letter features with parents)")

    # =========================================================
    # Step 5: Activation freq inverted (lower freq = higher score)
    # =========================================================
    act_freq_safe = np.maximum(act_freq, 1e-10)
    act_freq_inverted = 1.0 / act_freq_safe

    # =========================================================
    # Step 6: Compute all 7 formulations
    # =========================================================
    print("[B2] Computing all 7 ARS formulations...")
    report_progress(4, 7, {"phase": "computing_formulations"})

    # Normalize encoder_norms to [0,1] for combining with bounded co-occurrence scores
    # But keep raw encoder_norms for encoder_norm alone detector
    enc_norm_raw = encoder_norms.copy()

    # All formulations (higher score = more likely absorbed)
    formulations = {
        "encoder_norm": enc_norm_raw,
        "A_cooccur": A_cooccur_all,
        "O_jaccard": O_jaccard_all,
        "ARS_v2": enc_norm_raw * A_cooccur_all,
        "ARS_original": O_jaccard_all * A_cooccur_all * cos2_all,
        "ARS_full": enc_norm_raw * A_cooccur_all * O_jaccard_all,
        "EDA": eda,
    }

    # =========================================================
    # Step 7: Evaluate all formulations
    # =========================================================
    print("[B2] Evaluating all formulations against Chanin labels...")
    report_progress(5, 7, {"phase": "evaluating"})

    results_by_formulation = {}

    for name, scores in formulations.items():
        print(f"[B2]   Computing {name}...")
        auroc, auprc = compute_auroc_auprc(scores, labels)
        auroc_ci = bootstrap_auroc_ci(scores, labels, n_bootstrap=2000, seed=SEED)
        p50 = compute_precision_at_k(scores, labels, k=50)
        p100 = compute_precision_at_k(scores, labels, k=100)
        p500 = compute_precision_at_k(scores, labels, k=500)

        # Top-20 features by this formulation
        top20_idx = np.argsort(scores)[::-1][:20]
        top20 = [(int(i), float(scores[i]), int(labels[i])) for i in top20_idx]

        # Absorbed features scores
        absorbed_scores = [(int(fid), float(scores[fid])) for fid in absorbed_ids]
        absorbed_scores_sorted = sorted(absorbed_scores, key=lambda x: -x[1])

        results_by_formulation[name] = {
            "formulation": name,
            "auroc": float(auroc),
            "auroc_ci95": auroc_ci,
            "auprc": float(auprc),
            "precision_at_50": float(p50),
            "precision_at_100": float(p100),
            "precision_at_500": float(p500),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "score_stats": {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "absorbed_mean": float(scores[labels == 1].mean()),
                "non_absorbed_mean": float(scores[labels == 0].mean()),
            },
            "top20_features": top20,
            "absorbed_feature_scores": absorbed_scores_sorted,
        }

        print(f"[B2]   {name}: AUROC={auroc:.4f} ({auroc_ci[0]:.3f}-{auroc_ci[1]:.3f}), AUPRC={auprc:.6f}, P@50={p50:.4f}, P@100={p100:.4f}")

    # =========================================================
    # Step 8: DeLong tests
    # =========================================================
    print("[B2] Running DeLong tests...")
    report_progress(6, 7, {"phase": "delong_tests"})

    delong_results = {}

    # Primary: ARS_v2 vs encoder_norm alone
    z, p = delong_test(formulations["ARS_v2"], formulations["encoder_norm"], labels)
    delong_results["ARS_v2_vs_encoder_norm"] = {
        "z": z, "p_one_sided": p,
        "interpretation": "ARS_v2 AUROC significantly > encoder_norm" if p < 0.05 else "ARS_v2 does NOT significantly improve over encoder_norm"
    }

    # Secondary: ARS_v2 vs ARS_original
    z2, p2 = delong_test(formulations["ARS_v2"], formulations["ARS_original"], labels)
    delong_results["ARS_v2_vs_ARS_original"] = {
        "z": z2, "p_one_sided": p2,
        "interpretation": "ARS_v2 significantly > ARS_original" if p2 < 0.05 else "ARS_v2 NOT significantly > ARS_original"
    }

    # ARS_full vs encoder_norm
    z3, p3 = delong_test(formulations["ARS_full"], formulations["encoder_norm"], labels)
    delong_results["ARS_full_vs_encoder_norm"] = {
        "z": z3, "p_one_sided": p3,
        "interpretation": "ARS_full significantly > encoder_norm" if p3 < 0.05 else "ARS_full NOT significantly > encoder_norm"
    }

    # encoder_norm vs EDA
    z4, p4 = delong_test(formulations["encoder_norm"], formulations["EDA"], labels)
    delong_results["encoder_norm_vs_EDA"] = {
        "z": z4, "p_one_sided": p4,
        "interpretation": "encoder_norm significantly > EDA" if p4 < 0.05 else "encoder_norm NOT significantly > EDA"
    }

    print(f"[B2] DeLong results:")
    for test_name, test_result in delong_results.items():
        print(f"  {test_name}: z={test_result['z']:.3f}, p={test_result['p_one_sided']:.4f} — {test_result['interpretation']}")

    # =========================================================
    # Step 9: Spearman correlations between formulations
    # =========================================================
    print("[B2] Computing Spearman correlations...")

    spearman_results = {}
    formula_names = list(formulations.keys())
    for i, name_a in enumerate(formula_names):
        for name_b in formula_names[i+1:]:
            r, p_val = spearmanr(formulations[name_a], formulations[name_b])
            spearman_results[f"{name_a}_vs_{name_b}"] = {
                "r": float(r), "p": float(p_val)
            }

    # =========================================================
    # Step 10: Pilot pass criteria check
    # =========================================================
    auroc_enc_norm = results_by_formulation["encoder_norm"]["auroc"]
    auroc_ars_v2 = results_by_formulation["ARS_v2"]["auroc"]
    delong_ars_v2_vs_enc = delong_results["ARS_v2_vs_encoder_norm"]

    all_computed = all(
        name in results_by_formulation for name in
        ["encoder_norm", "A_cooccur", "O_jaccard", "ARS_v2", "ARS_original", "ARS_full", "EDA"]
    )

    pilot_pass = all_computed and "ARS_v2_vs_encoder_norm" in delong_results

    # =========================================================
    # Step 11: Summary ranking
    # =========================================================
    ranking = sorted(results_by_formulation.items(), key=lambda x: -x[1]["auroc"])
    print("\n[B2] === RANKING BY AUROC ===")
    for rank, (name, res) in enumerate(ranking, 1):
        ci = res.get("auroc_ci95", [None, None])
        ci_lo = f"{ci[0]:.3f}" if ci[0] is not None else "N/A"
        ci_hi = f"{ci[1]:.3f}" if ci[1] is not None else "N/A"
        print(f"  {rank}. {name}: AUROC={res['auroc']:.4f} ({ci_lo}-{ci_hi}), AUPRC={res['auprc']:.6f}, P@50={res['precision_at_50']:.4f}, P@100={res['precision_at_100']:.4f}, P@500={res['precision_at_500']:.4f}")

    elapsed = time.time() - start_time

    # =========================================================
    # Final Result JSON
    # =========================================================
    result = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "config": {
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "d_sae": d_sae,
            "label_file": LABEL_FILE,
            "sae": "gpt2-small-res-jb blocks.6.hook_resid_pre",
            "b1_source": "exp/results/full/B1_cooccurrence_graph.json",
            "n_tokens_for_cooccurrence": b1_data["config"]["n_tokens"],
            "seed": SEED,
        },
        "formulation_results": results_by_formulation,
        "auroc_ranking": [
            {
                "rank": i + 1,
                "formulation": name,
                "auroc": res["auroc"],
                "auprc": res["auprc"],
                "precision_at_50": res["precision_at_50"],
                "precision_at_100": res["precision_at_100"],
                "precision_at_500": res["precision_at_500"],
            }
            for i, (name, res) in enumerate(ranking)
        ],
        "delong_tests": delong_results,
        "spearman_correlations": spearman_results,
        "notes": {
            "A_cooccur_bounded": "A_cooccur values are bounded at ~0.33 (= f_j/f_i < 1/3 by construction when f_i > 3*f_j). This mathematically limits the discriminative power of A_cooccur alone and combined formulations.",
            "O_jaccard_coverage": "O_jaccard is only non-zero for features in the 71 letter feature set. Absorbed features NOT in letter set (9 of 18) will have O_jaccard=0, limiting ARS_original and ARS_full.",
            "absorbed_not_in_letter_set": [fid for fid in absorbed_ids if fid not in set(letter_feature_ids)],
            "cos2_parent_dependency": "cos2 component requires a parent feature; features with no identified parent get 0."
        },
        "pilot_pass_criteria": {
            "all_7_formulations_computed": all_computed,
            "delong_test_run": "ARS_v2_vs_encoder_norm" in delong_results,
            "pass": pilot_pass,
            "encoder_norm_auroc": auroc_enc_norm,
            "ars_v2_auroc": auroc_ars_v2,
        },
        "primary_finding": {
            "best_formulation": ranking[0][0],
            "best_auroc": ranking[0][1]["auroc"],
            "encoder_norm_auroc": auroc_enc_norm,
            "ars_v2_auroc": auroc_ars_v2,
            "ars_v2_improves_encoder_norm": auroc_ars_v2 > auroc_enc_norm,
            "delong_ars_v2_vs_enc": delong_ars_v2_vs_enc,
        }
    }

    out_file = RESULTS_DIR / "B2_ars_v2_validation.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[B2] Output saved: {out_file}")
    print(f"[B2] Pilot PASS: {pilot_pass}")
    print(f"[B2] Best formulation: {ranking[0][0]} (AUROC={ranking[0][1]['auroc']:.4f})")
    print(f"[B2] ARS_v2 AUROC={auroc_ars_v2:.4f} vs encoder_norm AUROC={auroc_enc_norm:.4f}")
    print(f"[B2] Elapsed: {elapsed:.1f}s")

    report_progress(7, 7, {"phase": "done", "pilot_pass": pilot_pass})
    mark_done("success", f"7 formulations evaluated; best={ranking[0][0]} (AUROC={ranking[0][1]['auroc']:.4f}); ARS_v2 AUROC={auroc_ars_v2:.4f}")

    # Update gpu_progress.json
    gpu_progress_file = Path(WORKSPACE) / "exp" / "gpu_progress.json"
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
    if gpu_progress_file.exists():
        try:
            with open(gpu_progress_file) as f:
                gp = json.load(f)
        except:
            pass
    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 15,
        "actual_min": round(elapsed / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "d_sae": d_sae,
            "n_formulations": 7,
            "gpu": "RTX PRO 6000 Blackwell",
            "mode": "PILOT",
        }
    }
    with open(gpu_progress_file, 'w') as f:
        json.dump(gp, f, indent=2)
    print("[B2] gpu_progress.json updated")

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(f"[B2] ERROR: {e}")
    print(tb)
    mark_done("failed", f"Error: {str(e)[:200]}")
    # Update gpu_progress as failed
    try:
        gpu_progress_file = Path(WORKSPACE) / "exp" / "gpu_progress.json"
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if gpu_progress_file.exists():
            with open(gpu_progress_file) as f:
                gp = json.load(f)
        if TASK_ID not in gp["failed"]:
            gp["failed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        with open(gpu_progress_file, 'w') as f:
            json.dump(gp, f, indent=2)
    except:
        pass
    sys.exit(1)
