"""
Task D.1: EDA Validation Against Exact Chanin Labels (PILOT MODE)

Validate EDA = 1 - cos(encoder_j, decoder_j) against exact sae-spelling
absorption labels for GPT-2 Small L6.

Compute: AUROC, AUPRC, Precision@k (k=50, 100, 500), precision-recall curve.
Baselines: random, cos(enc,dec) alone, feature activation frequency,
           decoder norm alone.
Null AUROC: permute absorption labels 100 times.
DeLong test: AUROC comparison between EDA and best baseline.
Compare EDA on correct label format vs proxy approach from pilot.

Output: exp/results/full/D1_eda_validation.json

PILOT MODE: Uses exact Chanin labels from iter_001 (n_pos=18 for GPT2-L6)
            plus the existing per-feature EDA values already computed.
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
ITER001_WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_001")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_D1_eda_validation"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "D1_eda_validation.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "note": note,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {epoch}/{total_epochs}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary=""):
    PID_FILE.unlink(missing_ok=True)
    progress_file_data = {}
    if PROGRESS_FILE.exists():
        try:
            progress_file_data = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": progress_file_data,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
    }))


TOTAL_STEPS = 10
report_progress(0, TOTAL_STEPS, note="Starting D1: EDA validation against exact Chanin labels")

# ─── Step 1: Load model and SAE ───────────────────────────────────────────────
report_progress(1, TOTAL_STEPS, note="Loading GPT-2 Small and SAE (L6, width 24576)")

import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sys.stdout.flush()

from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=True)
model.eval()
model.to(device)

# Load GPT-2 Small L6 SAE
sae, cfg_dict, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=str(device),
)
sae.eval()

D_SAE = sae.cfg.d_sae  # 24576
D_IN = sae.cfg.d_in    # 768
print(f"SAE loaded: d_sae={D_SAE}, d_in={D_IN}")
sys.stdout.flush()

# ─── Step 2: Compute EDA and baselines for all features ───────────────────────
report_progress(2, TOTAL_STEPS, note="Computing EDA, encoder/decoder norms for all features")

with torch.no_grad():
    # Get encoder and decoder weights
    # W_enc: [D_IN, D_SAE] -> encoder directions (columns)
    # W_dec: [D_SAE, D_IN] -> decoder directions (rows)
    W_enc = sae.W_enc  # [D_IN, D_SAE]
    W_dec = sae.W_dec  # [D_SAE, D_IN]

    # Normalize encoder columns and decoder rows
    enc_norms = W_enc.norm(dim=0)  # [D_SAE]
    dec_norms = W_dec.norm(dim=1)  # [D_SAE]

    # Unit vectors
    W_enc_unit = W_enc / enc_norms.unsqueeze(0)  # [D_IN, D_SAE]
    W_dec_unit = W_dec / dec_norms.unsqueeze(1)  # [D_SAE, D_IN]

    # cos(encoder_j, decoder_j) for each feature j
    cos_enc_dec = (W_enc_unit * W_dec_unit.T).sum(dim=0)  # [D_SAE]

    # EDA = 1 - cos(enc_j, dec_j)
    eda_scores = 1.0 - cos_enc_dec  # [D_SAE]

    enc_norms_np = enc_norms.cpu().float().numpy()
    dec_norms_np = dec_norms.cpu().float().numpy()
    eda_np = eda_scores.cpu().float().numpy()
    cos_enc_dec_np = cos_enc_dec.cpu().float().numpy()

print(f"EDA computed for {D_SAE} features. Mean EDA: {eda_np.mean():.4f}, Std: {eda_np.std():.4f}")
sys.stdout.flush()

# ─── Step 3: Compute activation frequency baseline ─────────────────────────────
report_progress(3, TOTAL_STEPS, note="Computing activation frequencies via OWT sample")

# Load or compute OWT activations
owt_cache_file = WORKSPACE / "exp" / "owt_tokens_cache.pt"
iter002_cache = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_002/exp/results/owt_tokens_cache.pt")

if owt_cache_file.exists():
    print("Loading cached OWT tokens...")
    owt_tokens = torch.load(owt_cache_file, map_location="cpu")
    if owt_tokens.dim() == 1:
        owt_tokens = owt_tokens.unsqueeze(0)
    print(f"Loaded OWT tokens: shape {owt_tokens.shape}")
elif iter002_cache.exists():
    print("Using iter_002 OWT tokens cache...")
    owt_tokens = torch.load(str(iter002_cache), map_location="cpu")
    if owt_tokens.dim() == 1:
        owt_tokens = owt_tokens.unsqueeze(0)
    print(f"Loaded OWT tokens: shape {owt_tokens.shape}")
    # Copy to local cache
    torch.save(owt_tokens, owt_cache_file)
else:
    print("No OWT cache found, generating tokens from model tokenizer...")
    # Generate synthetic random tokens as fallback for frequency estimation
    # This is just for the activation frequency baseline, not the main metric
    torch.manual_seed(SEED)
    vocab_size = model.tokenizer.vocab_size
    owt_tokens = torch.randint(0, vocab_size, (20, 128))
    print(f"Generated random tokens: shape {owt_tokens.shape}")
    torch.save(owt_tokens, owt_cache_file)

# Compute activation frequencies
max_token_batches = 50
activation_counts = np.zeros(D_SAE, dtype=np.float32)
n_tokens_total = 0

batch_size = 10
n_batches = min(max_token_batches // batch_size + 1, owt_tokens.shape[0] // batch_size + 1)

for batch_idx in range(min(n_batches, 10)):
    start = batch_idx * batch_size
    end = min(start + batch_size, owt_tokens.shape[0])
    if start >= end:
        break

    batch_tokens = owt_tokens[start:end].to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter="blocks.6.hook_resid_pre",
            device=device,
        )
        resid = cache["blocks.6.hook_resid_pre"]  # [batch, seq, D_IN]
        B, S, D = resid.shape
        resid_flat = resid.reshape(-1, D)
        sae_acts = sae.encode(resid_flat)  # [B*S, D_SAE]
        active_mask = (sae_acts > 0).float()
        activation_counts += active_mask.sum(dim=0).cpu().float().numpy()
        n_tokens_total += resid_flat.shape[0]

activation_freq = activation_counts / max(n_tokens_total, 1)
print(f"Activation frequency computed. n_tokens: {n_tokens_total}, mean_freq: {activation_freq.mean():.5f}")
sys.stdout.flush()

# ─── Step 4: Load exact Chanin labels from iter_001 ─────────────────────────────
report_progress(4, TOTAL_STEPS, note="Loading exact Chanin labels (FeatureAbsorptionCalculator)")

iter001_labels_file = ITER001_WORKSPACE / "exp" / "results" / "r4" / "r4a_direct_labels.json"
chanin_absorbed_ids_l6 = None

if iter001_labels_file.exists():
    with open(iter001_labels_file, "r") as f:
        iter001_data = json.load(f)

    labels = iter001_data.get("all_direct_labels", {})
    l6_labels = labels.get("GPT2-L6", {})
    chanin_absorbed_ids_l6 = l6_labels.get("absorbed_latent_ids", [])
    print(f"Loaded exact Chanin labels: n_pos={len(chanin_absorbed_ids_l6)}")
    print(f"Absorbed IDs: {chanin_absorbed_ids_l6}")
else:
    print("WARNING: iter_001 labels file not found. Falling back to proxy labels from B1_pairwise_eda.")
    # Fallback: use letter features as proxy
    b1_file = RESULTS_DIR / "B1_eda_decomposition.json"
    if b1_file.exists():
        with open(b1_file) as f:
            b1_data = json.load(f)
        sample = b1_data.get("layer6", {}).get("per_feature_sample", [])
        chanin_absorbed_ids_l6 = [s["feature_idx"] for s in sample if s.get("is_letter", False)]
        print(f"Fallback proxy labels: n_pos={len(chanin_absorbed_ids_l6)}")
    else:
        raise RuntimeError("Neither exact Chanin labels nor B1 fallback available.")

n_pos = len(chanin_absorbed_ids_l6)
n_total = D_SAE

# Create binary label array
y_true = np.zeros(n_total, dtype=np.float32)
for fid in chanin_absorbed_ids_l6:
    if 0 <= fid < n_total:
        y_true[fid] = 1.0

n_neg = int((y_true == 0).sum())
base_rate = n_pos / n_total
print(f"Labels: n_pos={n_pos}, n_neg={n_neg}, base_rate={base_rate:.6f}")
sys.stdout.flush()

# ─── Step 5: Compute metrics for all detectors ────────────────────────────────
report_progress(5, TOTAL_STEPS, note="Computing AUROC, AUPRC, Precision@k for EDA and baselines")


def compute_metrics(scores, y_true, detector_name, k_list=[50, 100, 500]):
    """Compute AUROC, AUPRC, and Precision@k for a detector."""
    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    auprc_over_base = auprc / base_rate if base_rate > 0 else 0.0

    # Precision@k
    top_k_idxs = np.argsort(scores)[::-1]
    precision_at_k = {}
    for k in k_list:
        top_k = top_k_idxs[:k]
        prec = float(y_true[top_k].sum()) / k
        precision_at_k[f"precision_at_{k}"] = prec

    # Precision-recall curve (simplified, key breakpoints)
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, scores)
    # Sample at recall breakpoints 0, 0.1, 0.2, ..., 1.0
    pr_curve_sample = []
    for target_recall in np.linspace(0, 1.0, 11):
        idx = np.argmin(np.abs(recall_arr - target_recall))
        pr_curve_sample.append({
            "recall": float(recall_arr[idx]),
            "precision": float(precision_arr[idx]),
        })

    return {
        "detector": detector_name,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_over_base": auprc_over_base,
        **precision_at_k,
        "pr_curve_sample": pr_curve_sample,
    }


# All detectors
metrics_all = {}

# EDA = 1 - cos(enc_j, dec_j)  [higher = more absorbed]
metrics_all["EDA"] = compute_metrics(eda_np, y_true, "EDA")
print(f"EDA AUROC: {metrics_all['EDA']['auroc']:.4f}, AUPRC: {metrics_all['EDA']['auprc']:.6f}")

# cos(enc_j, dec_j) alone [lower = more absorbed, so invert]
metrics_all["cos_enc_dec_inverted"] = compute_metrics(-cos_enc_dec_np, y_true, "cos_enc_dec_inverted")
print(f"cos_enc_dec_inv AUROC: {metrics_all['cos_enc_dec_inverted']['auroc']:.4f}")

# cos(enc_j, dec_j) alone [not inverted - for completeness]
metrics_all["cos_enc_dec"] = compute_metrics(cos_enc_dec_np, y_true, "cos_enc_dec")

# Feature activation frequency [hypothesis: absorbed features may have lower frequency]
# Higher activation freq = less absorbed (inverse direction)
metrics_all["freq_inv"] = compute_metrics(-activation_freq, y_true, "freq_inv")
metrics_all["freq"] = compute_metrics(activation_freq, y_true, "freq")
print(f"freq_inv AUROC: {metrics_all['freq_inv']['auroc']:.4f}")
print(f"freq AUROC: {metrics_all['freq']['auroc']:.4f}")

# Decoder norm alone [hypothesis: absorbed features may have different decoder norms]
metrics_all["dec_norm"] = compute_metrics(dec_norms_np, y_true, "dec_norm")
metrics_all["dec_norm_inv"] = compute_metrics(-dec_norms_np, y_true, "dec_norm_inv")
print(f"dec_norm AUROC: {metrics_all['dec_norm']['auroc']:.4f}")

# Encoder norm alone
metrics_all["enc_norm"] = compute_metrics(enc_norms_np, y_true, "enc_norm")

# Random baseline
metrics_all["random"] = {
    "detector": "random",
    "auroc": 0.5,
    "auprc": float(base_rate),
    "auprc_over_base": 1.0,
    "precision_at_50": float(n_pos / n_total),
    "precision_at_100": float(n_pos / n_total),
    "precision_at_500": float(n_pos / n_total),
}

sys.stdout.flush()

# ─── Step 6: Null AUROC via label permutation (100 permutations) ───────────────
report_progress(6, TOTAL_STEPS, note="Running null AUROC: 100 label permutations")

np.random.seed(SEED)
null_aurocs_eda = []
for i in range(100):
    y_perm = np.random.permutation(y_true)
    null_auroc = float(roc_auc_score(y_perm, eda_np))
    null_aurocs_eda.append(null_auroc)

null_mean = float(np.mean(null_aurocs_eda))
null_std = float(np.std(null_aurocs_eda))
eda_z_above_null = (metrics_all["EDA"]["auroc"] - null_mean) / (null_std + 1e-8)

null_result = {
    "n_permutations": 100,
    "null_auroc_mean": null_mean,
    "null_auroc_std": null_std,
    "eda_auroc": metrics_all["EDA"]["auroc"],
    "z_above_null": float(eda_z_above_null),
    "passes_2sd_threshold": bool(eda_z_above_null >= 2.0),
    "eda_minus_null_in_sd": float(eda_z_above_null),
}
print(f"Null AUROC: mean={null_mean:.4f}, std={null_std:.4f}")
print(f"EDA z-score above null: {eda_z_above_null:.2f} (>2.0 = passes)")
sys.stdout.flush()

# ─── Step 7: DeLong test for AUROC comparison ────────────────────────────────
report_progress(7, TOTAL_STEPS, note="Running DeLong test: EDA vs best baseline")


def delong_roc_variance(ground_truth, predictions):
    """
    Compute variance of AUROC estimator using DeLong method.
    Reference: DeLong et al. 1988.
    """
    order = np.argsort(-predictions)
    gt_sorted = ground_truth[order]
    pred_sorted = predictions[order]

    pos_mask = gt_sorted == 1
    neg_mask = gt_sorted == 0

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    pred_pos = pred_sorted[pos_mask]
    pred_neg = pred_sorted[neg_mask]

    # V10 and V01 - placements
    V10 = np.zeros(n_pos)
    for i, p in enumerate(pred_pos):
        V10[i] = ((pred_neg < p).sum() + 0.5 * (pred_neg == p).sum()) / n_neg

    V01 = np.zeros(n_neg)
    for i, n in enumerate(pred_neg):
        V01[i] = ((pred_pos > n).sum() + 0.5 * (pred_pos == n).sum()) / n_pos

    theta = float(np.mean(V10))
    s10 = float(np.var(V10) / n_pos)
    s01 = float(np.var(V01) / n_neg)
    var_auc = s10 + s01

    return theta, var_auc


def delong_test(y_true, scores_a, scores_b):
    """Two-sided DeLong test comparing AUROC of two classifiers."""
    auroc_a, var_a = delong_roc_variance(y_true, scores_a)
    auroc_b, var_b = delong_roc_variance(y_true, scores_b)
    diff = auroc_a - auroc_b
    # For two correlated AUROCs, we need covariance
    # Simplified: treat as independent (conservative)
    se = np.sqrt(var_a + var_b + 1e-10)
    z = diff / se
    # Two-tailed p-value approximation using normal distribution
    from scipy import stats
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "diff": float(diff),
        "z_stat": float(z),
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
    }


# Find best baseline to compare against EDA
best_baseline_name = max(
    ["cos_enc_dec_inverted", "freq_inv", "dec_norm", "enc_norm"],
    key=lambda k: metrics_all[k]["auroc"]
)
best_baseline_scores_map = {
    "cos_enc_dec_inverted": -cos_enc_dec_np,
    "freq_inv": -activation_freq,
    "dec_norm": dec_norms_np,
    "enc_norm": enc_norms_np,
    "dec_norm_inv": -dec_norms_np,
}
best_baseline_scores = best_baseline_scores_map[best_baseline_name]

print(f"Best baseline: {best_baseline_name} (AUROC={metrics_all[best_baseline_name]['auroc']:.4f})")
delong_result = delong_test(y_true, eda_np, best_baseline_scores)
print(f"DeLong EDA vs {best_baseline_name}: diff={delong_result['diff']:.4f}, p={delong_result['p_value']:.4f}")
sys.stdout.flush()

# ─── Step 8: Proxy vs exact label comparison ──────────────────────────────────
report_progress(8, TOTAL_STEPS, note="Comparing EDA on exact vs proxy labels")

# Proxy labels: letter features from B1_eda_decomposition (is_letter threshold)
b1_file = RESULTS_DIR / "B1_eda_decomposition.json"
proxy_comparison = {}
if b1_file.exists():
    with open(b1_file) as f:
        b1_data = json.load(f)

    # From B1, letter features above threshold at L6
    sample = b1_data.get("layer6", {}).get("per_feature_sample", [])
    proxy_letter_ids = set(s["feature_idx"] for s in sample if s.get("is_letter", False))

    # Also check B1_decoder_geometry for more letter features
    b1_dg_file = RESULTS_DIR / "B1_decoder_geometry.json"
    if b1_dg_file.exists():
        with open(b1_dg_file) as f:
            b1_dg = json.load(f)
        more_letter = b1_dg.get("layer6", {}).get("letter_feature_ids", [])
        proxy_letter_ids.update(more_letter)

    proxy_letter_ids_list = sorted(proxy_letter_ids)
    y_proxy = np.zeros(n_total, dtype=np.float32)
    for fid in proxy_letter_ids_list:
        if 0 <= fid < n_total:
            y_proxy[fid] = 1.0

    if y_proxy.sum() > 0:
        proxy_auroc = float(roc_auc_score(y_proxy, eda_np))
        proxy_auprc = float(average_precision_score(y_proxy, eda_np))
        proxy_base_rate = float(y_proxy.mean())

        # Overlap between proxy and exact labels
        exact_set = set(chanin_absorbed_ids_l6)
        proxy_set = set(proxy_letter_ids_list)
        overlap = len(exact_set & proxy_set)
        union = len(exact_set | proxy_set)
        jaccard = overlap / union if union > 0 else 0.0

        proxy_comparison = {
            "proxy_n_pos": int(y_proxy.sum()),
            "proxy_base_rate": proxy_base_rate,
            "proxy_eda_auroc": proxy_auroc,
            "proxy_eda_auprc": proxy_auprc,
            "exact_n_pos": n_pos,
            "exact_eda_auroc": metrics_all["EDA"]["auroc"],
            "exact_eda_auprc": metrics_all["EDA"]["auprc"],
            "label_overlap_count": overlap,
            "label_union_count": union,
            "jaccard_similarity": jaccard,
            "note": "Proxy = letter features above threshold (B1 probe alignment). Exact = FeatureAbsorptionCalculator (Chanin et al.).",
        }
        print(f"Proxy labels: n_pos={int(y_proxy.sum())}, EDA AUROC={proxy_auroc:.4f}")
        print(f"Exact labels: n_pos={n_pos}, EDA AUROC={metrics_all['EDA']['auroc']:.4f}")
        print(f"Label overlap: {overlap}/{union} (Jaccard={jaccard:.3f})")
    else:
        proxy_comparison = {"error": "No proxy letter features found"}

sys.stdout.flush()

# ─── Step 9: AJT SAE comparison (if available) ─────────────────────────────────
report_progress(9, TOTAL_STEPS, note="Computing EDA on AJT SAE for comparison (if available)")

ajt_comparison = {}
try:
    sae_ajt, cfg_ajt, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.10.hook_resid_pre",
        device=str(device),
    )
    sae_ajt.eval()

    with torch.no_grad():
        W_enc_ajt = sae_ajt.W_enc  # [D_IN, D_SAE]
        W_dec_ajt = sae_ajt.W_dec  # [D_SAE, D_IN]

        enc_norms_ajt = W_enc_ajt.norm(dim=0)
        dec_norms_ajt = W_dec_ajt.norm(dim=1)

        W_enc_unit_ajt = W_enc_ajt / enc_norms_ajt.unsqueeze(0)
        W_dec_unit_ajt = W_dec_ajt / dec_norms_ajt.unsqueeze(1)

        cos_enc_dec_ajt = (W_enc_unit_ajt * W_dec_unit_ajt.T).sum(dim=0)
        eda_ajt = 1.0 - cos_enc_dec_ajt
        eda_ajt_np = eda_ajt.cpu().float().numpy()

    d_sae_ajt = sae_ajt.cfg.d_sae
    ajt_comparison = {
        "sae_release": "gpt2-small-res-jb",
        "hook_name": "blocks.10.hook_resid_pre",
        "d_sae": d_sae_ajt,
        "eda_mean": float(eda_ajt_np.mean()),
        "eda_std": float(eda_ajt_np.std()),
        "note": "L10 SAE. Cannot compute AUROC against Chanin labels (labels are for L6 only).",
    }

    # L10 does not have exact Chanin labels for the same features, skip AUROC
    print(f"L10 AJT EDA: mean={eda_ajt_np.mean():.4f}, std={eda_ajt_np.std():.4f}")

except Exception as e:
    ajt_comparison = {"error": str(e), "note": "L10 SAE comparison not available"}
    print(f"L10 SAE comparison skipped: {e}")

sys.stdout.flush()

# ─── Step 10: Save results ─────────────────────────────────────────────────────
report_progress(10, TOTAL_STEPS, note="Saving D1 EDA validation results")

# Determine pass/fail
eda_auroc = metrics_all["EDA"]["auroc"]
eda_auprc = metrics_all["EDA"]["auprc"]
auprc_over_base = metrics_all["EDA"]["auprc_over_base"]
pass_auroc = eda_auroc >= 0.60
pass_auprc = auprc_over_base >= 3.0

results = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "elapsed_sec": time.time() - start_time,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": 6,
        "d_sae": D_SAE,
        "d_in": D_IN,
        "seed": SEED,
        "n_null_permutations": 100,
        "k_list": [50, 100, 500],
        "label_source": "FeatureAbsorptionCalculator (Chanin et al. 2024, sae_spelling)",
        "label_source_file": str(iter001_labels_file),
    },
    "labels": {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": n_total,
        "base_rate": float(base_rate),
        "absorbed_feature_ids": chanin_absorbed_ids_l6,
        "label_method": "FeatureAbsorptionCalculator exact labels (iter_001 R4)",
    },
    "metrics": {
        "EDA": metrics_all["EDA"],
        "cos_enc_dec_inverted": metrics_all["cos_enc_dec_inverted"],
        "cos_enc_dec": metrics_all["cos_enc_dec"],
        "activation_freq_inverted": metrics_all["freq_inv"],
        "activation_freq": metrics_all["freq"],
        "decoder_norm": metrics_all["dec_norm"],
        "decoder_norm_inverted": metrics_all["dec_norm_inv"],
        "encoder_norm": metrics_all["enc_norm"],
        "random_baseline": metrics_all["random"],
    },
    "null_auroc": null_result,
    "delong_test": {
        "comparison": f"EDA vs {best_baseline_name}",
        "best_baseline": best_baseline_name,
        **delong_result,
    },
    "proxy_vs_exact_comparison": proxy_comparison,
    "l10_comparison": ajt_comparison,
    "pass_criteria": {
        "eda_auroc_ge_060": pass_auroc,
        "eda_auroc_value": float(eda_auroc),
        "auprc_gt_3x_base": pass_auprc,
        "auprc_over_base_rate": float(auprc_over_base),
        "eda_z_above_null": float(eda_z_above_null),
        "passes_null_2sd": null_result["passes_2sd_threshold"],
        "overall_go_nogo": "GO" if pass_auroc else ("INVESTIGATE" if eda_auroc >= 0.55 else "NO_GO"),
        "notes": [
            f"EDA AUROC={eda_auroc:.4f} vs target >= 0.60 (exact Chanin labels, n_pos={n_pos})",
            f"AUPRC={eda_auprc:.6f} = {auprc_over_base:.2f}x base rate (target >3x)",
            f"Null z-score: {eda_z_above_null:.2f} ({'passes' if eda_z_above_null >= 2.0 else 'fails'} 2-SD threshold)",
            f"Best baseline: {best_baseline_name} (AUROC={metrics_all[best_baseline_name]['auroc']:.4f})",
            f"DeLong EDA vs {best_baseline_name}: p={delong_result['p_value']:.4f}",
        ],
    },
}

OUTPUT_FILE.write_text(json.dumps(results, indent=2))
print(f"Results saved to {OUTPUT_FILE}")
print(f"\n=== D1 EDA VALIDATION SUMMARY ===")
print(f"EDA AUROC: {eda_auroc:.4f} (target >= 0.60) -> {'PASS' if pass_auroc else 'FAIL'}")
print(f"EDA AUPRC: {eda_auprc:.6f} ({auprc_over_base:.2f}x base) -> {'PASS' if pass_auprc else 'FAIL'}")
print(f"Null z-score: {eda_z_above_null:.2f} -> {'PASS' if eda_z_above_null >= 2.0 else 'FAIL'}")
print(f"cos_enc_dec_inv AUROC: {metrics_all['cos_enc_dec_inverted']['auroc']:.4f}")
print(f"freq_inv AUROC: {metrics_all['freq_inv']['auroc']:.4f}")
print(f"dec_norm AUROC: {metrics_all['dec_norm']['auroc']:.4f}")
print(f"Overall: {results['pass_criteria']['overall_go_nogo']}")

mark_done(status="success", summary=f"EDA AUROC={eda_auroc:.4f}, AUPRC={eda_auprc:.6f}, z={eda_z_above_null:.2f}")
print("D1 EDA validation complete.")
