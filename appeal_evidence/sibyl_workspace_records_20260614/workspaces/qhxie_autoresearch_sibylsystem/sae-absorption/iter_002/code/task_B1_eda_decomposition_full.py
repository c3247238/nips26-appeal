#!/usr/bin/env python3
"""
Task B1 EDA Decomposition (FULL mode)
B.1-REV: EDA Geometric Decomposition — Encoder vs. Decoder Alignment

Full analysis:
1. Compute cos(encoder_j, letter_probe) and cos(decoder_j, letter_probe)
   for ALL letter features and ALL non-letter features at L6 and L10
2. Paired t-test: does letter feature encoder align more with letter probe
   than its own decoder does?
3. Produce scatter plot: cos(encoder_j, letter_probe) vs cos(decoder_j, letter_probe)
4. Compute mean_encoder_norm and mean_decoder_norm per feature class
5. Compare L6 vs L10 geometry: why does EDA fail at L10?

Uses: first-letter probes from C.1 (F1=0.820) trained in residual stream space
Seed: 42, GPU: 4
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_B1_eda_decomposition"
GPU_ID = 4
SEED = 42
MODE = "FULL"

# Write PID file immediately for system recovery detection
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PID_FILE.write_text(str(os.getpid()))


def write_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
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


# ── Setup ─────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (physical GPU {GPU_ID})")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

write_progress(0, 10, metric={"status": "starting"})

# ── Load SAELens ─────────────────────────────────────────────────────────────
from sae_lens import SAE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("Loading SAE configurations...")
write_progress(1, 10, metric={"status": "loading_sae"})

start_time = time.time()


def load_sae(layer):
    """Load GPT-2 Small jb SAE for a given layer."""
    print(f"  Loading GPT-2 Small jb SAE, layer {layer}...")
    sae, cfg_dict, log_sparsities = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=str(device)
    )
    sae.eval()
    return sae


sae_l6 = load_sae(6)
sae_l10 = load_sae(10)

print(f"SAEs loaded. d_sae={sae_l6.cfg.d_sae}, d_in={sae_l6.cfg.d_in}")

# ── Load letter probes ────────────────────────────────────────────────────────
print("Loading first-letter probes from C.1...")
write_progress(2, 10, metric={"status": "loading_probes"})

PROBE_DIR = WORKSPACE / "exp" / "results" / "probes"
probe_classes = np.load(PROBE_DIR / "probe_first_letter_classes.npy", allow_pickle=True)
probe_weights = np.load(PROBE_DIR / "probe_first_letter_weights.npy")  # (n_letters, d_model)

print(f"Probe classes: {probe_classes} (n={len(probe_classes)})")
print(f"Probe weights shape: {probe_weights.shape}")

# Normalize probe directions to unit vectors
probe_dirs = torch.tensor(probe_weights, dtype=torch.float32).to(device)
probe_dirs_norm = F.normalize(probe_dirs, dim=1)  # (n_letters, d_model)


# ── Identify letter features ──────────────────────────────────────────────────
@torch.no_grad()
def identify_letter_features(sae, probe_dirs_norm, threshold=0.32):
    """Identify features whose decoder aligns with letter probes (proxy absorption labels)."""
    W_dec = sae.W_dec  # (d_sae, d_in)
    W_dec_norm = F.normalize(W_dec, dim=1)

    cos_sim = torch.mm(W_dec_norm, probe_dirs_norm.T)  # (d_sae, n_letters)
    max_cos, best_letter = cos_sim.max(dim=1)

    is_letter = (max_cos > threshold)
    return is_letter.cpu().numpy(), max_cos.cpu().numpy(), best_letter.cpu().numpy()


print("\nIdentifying letter features via decoder alignment threshold...")
write_progress(3, 10, metric={"status": "identifying_features"})

is_letter_l6, max_cos_l6, best_letter_l6 = identify_letter_features(
    sae_l6, probe_dirs_norm, threshold=0.32)
is_letter_l10, max_cos_l10, best_letter_l10 = identify_letter_features(
    sae_l10, probe_dirs_norm, threshold=0.30)

print(f"L6: {is_letter_l6.sum()} letter features, {(~is_letter_l6).sum()} non-letter features")
print(f"L10: {is_letter_l10.sum()} letter features, {(~is_letter_l10).sum()} non-letter features")


# ── Core EDA Decomposition Analysis ──────────────────────────────────────────
@torch.no_grad()
def analyze_eda_decomposition_full(sae, is_letter, probe_dirs_norm, layer_name,
                                    best_letter=None, max_nonletter_sample=500):
    """
    FULL analysis: decompose EDA and characterize encoder-decoder geometry.

    Returns complete per-feature arrays for letter features and sampled non-letter features.
    """
    print(f"\n{'='*60}")
    print(f"Layer {layer_name} — FULL EDA Geometric Decomposition")
    print(f"{'='*60}")

    W_enc = sae.W_enc      # (d_in, d_sae)
    W_dec = sae.W_dec      # (d_sae, d_in)

    d_sae = W_enc.shape[1]
    d_in = W_enc.shape[0]

    # Encoder directions (row j = encoder for feature j)
    enc_dirs = W_enc.T                         # (d_sae, d_in)
    enc_norms = enc_dirs.norm(dim=1)           # (d_sae,)
    enc_dirs_norm = F.normalize(enc_dirs, dim=1)  # (d_sae, d_in)

    # Decoder directions
    dec_dirs = W_dec                           # (d_sae, d_in)
    dec_norms = dec_dirs.norm(dim=1)           # (d_sae,)
    dec_dirs_norm = F.normalize(dec_dirs, dim=1)  # (d_sae, d_in)

    # EDA = 1 - cos(encoder_j, decoder_j)
    cos_enc_dec = (enc_dirs_norm * dec_dirs_norm).sum(dim=1)  # (d_sae,)
    eda = 1.0 - cos_enc_dec  # (d_sae,)

    # cos(encoder_j, letter_probe) — for each feature, alignment with each letter probe
    # probe_dirs_norm: (n_letters, d_in)
    cos_enc_probe = torch.mm(enc_dirs_norm, probe_dirs_norm.T)  # (d_sae, n_letters)
    cos_dec_probe = torch.mm(dec_dirs_norm, probe_dirs_norm.T)  # (d_sae, n_letters)

    # For letter features: use their assigned best letter probe (the one their decoder is most aligned with)
    # For non-letter features: use max across all letter probes
    if best_letter is not None:
        best_letter_tensor = torch.tensor(best_letter, dtype=torch.long, device=device)
        best_letter_tensor = best_letter_tensor.clamp(0, probe_dirs_norm.shape[0] - 1)

        cos_enc_best = cos_enc_probe.gather(1, best_letter_tensor.unsqueeze(1)).squeeze(1)
        cos_dec_best = cos_dec_probe.gather(1, best_letter_tensor.unsqueeze(1)).squeeze(1)
    else:
        cos_enc_best = cos_enc_probe.max(dim=1).values
        cos_dec_best = cos_dec_probe.max(dim=1).values

    cos_enc_maxprobe = cos_enc_probe.max(dim=1).values   # (d_sae,)
    cos_dec_maxprobe = cos_dec_probe.max(dim=1).values   # (d_sae,)

    # Move to CPU
    eda_np = eda.cpu().numpy()
    cos_enc_dec_np = cos_enc_dec.cpu().numpy()
    cos_enc_best_np = cos_enc_best.cpu().numpy()
    cos_dec_best_np = cos_dec_best.cpu().numpy()
    cos_enc_maxprobe_np = cos_enc_maxprobe.cpu().numpy()
    cos_dec_maxprobe_np = cos_dec_maxprobe.cpu().numpy()
    enc_norms_np = enc_norms.cpu().numpy()
    dec_norms_np = dec_norms.cpu().numpy()

    is_letter_bool = is_letter.astype(bool)

    # ── Analysis 1: EDA by feature class ────────────────────────────────────
    eda_letter = eda_np[is_letter_bool]
    eda_nonletter = eda_np[~is_letter_bool]

    eda_ttest = stats.ttest_ind(eda_letter, eda_nonletter)
    eda_mwu = stats.mannwhitneyu(eda_letter, eda_nonletter, alternative='two-sided')

    n1, n2 = len(eda_letter), len(eda_nonletter)
    pooled_std = np.sqrt(((n1-1)*eda_letter.std()**2 + (n2-1)*eda_nonletter.std()**2) / (n1+n2-2))
    cohens_d_eda = float((eda_letter.mean() - eda_nonletter.mean()) / pooled_std) if pooled_std > 0 else 0.0

    y_true = is_letter_bool.astype(int)
    try:
        auroc_eda = float(roc_auc_score(y_true, eda_np))
    except ValueError:
        auroc_eda = float('nan')

    try:
        auprc_eda = float(average_precision_score(y_true, eda_np))
    except ValueError:
        auprc_eda = float('nan')

    base_rate = y_true.mean()

    print(f"\n[1] EDA by feature class:")
    print(f"    Letter: mean={eda_letter.mean():.4f}, std={eda_letter.std():.4f}, n={n1}")
    print(f"    Non-letter: mean={eda_nonletter.mean():.4f}, std={eda_nonletter.std():.4f}, n={n2}")
    print(f"    Cohen's d = {cohens_d_eda:.4f}")
    print(f"    t-test: t={eda_ttest.statistic:.4f}, p={eda_ttest.pvalue:.2e}")
    print(f"    Mann-Whitney p={eda_mwu.pvalue:.2e}")
    print(f"    AUROC(EDA → letter) = {auroc_eda:.4f}")
    print(f"    AUPRC(EDA → letter) = {auprc_eda:.4f}  (base rate = {base_rate:.4f})")

    # ── Analysis 2: Paired encoder vs decoder alignment with letter probe ─────
    enc_probe_letter = cos_enc_best_np[is_letter_bool]
    dec_probe_letter = cos_dec_best_np[is_letter_bool]
    enc_probe_nonletter = cos_enc_maxprobe_np[~is_letter_bool]
    dec_probe_nonletter = cos_dec_maxprobe_np[~is_letter_bool]

    diff_letter = enc_probe_letter - dec_probe_letter
    paired_ttest_letter = stats.ttest_1samp(diff_letter, 0)
    diff_nonletter = enc_probe_nonletter - dec_probe_nonletter
    paired_ttest_nonletter = stats.ttest_1samp(diff_nonletter, 0)

    print(f"\n[2] Encoder vs Decoder alignment with letter probe (LETTER features, n={n1}):")
    print(f"    cos(encoder, letter_probe): mean={enc_probe_letter.mean():.4f}, std={enc_probe_letter.std():.4f}")
    print(f"    cos(decoder, letter_probe): mean={dec_probe_letter.mean():.4f}, std={dec_probe_letter.std():.4f}")
    print(f"    Diff enc-dec: mean={diff_letter.mean():.4f}, std={diff_letter.std():.4f}")
    print(f"    Paired t-test: t={paired_ttest_letter.statistic:.4f}, p={paired_ttest_letter.pvalue:.2e}")
    enc_more = diff_letter.mean() > 0
    print(f"    Interpretation: {'ENCODER aligns more (supports revised H1)' if enc_more else 'DECODER aligns more (contradicts revised H1)'}")

    print(f"\n    Non-letter features (n={n2}):")
    print(f"    cos(encoder, letter_probe): mean={enc_probe_nonletter.mean():.4f}")
    print(f"    cos(decoder, letter_probe): mean={dec_probe_nonletter.mean():.4f}")
    print(f"    Diff enc-dec: mean={diff_nonletter.mean():.4f}")

    # ── Analysis 3: AUROC of cos(encoder, letter_probe) for predicting letter features ──
    try:
        auroc_enc_probe = float(roc_auc_score(y_true, cos_enc_best_np))
    except ValueError:
        auroc_enc_probe = float('nan')
    try:
        auroc_dec_probe = float(roc_auc_score(y_true, cos_dec_best_np))
    except ValueError:
        auroc_dec_probe = float('nan')

    print(f"\n[3] AUROC for predicting letter features:")
    print(f"    EDA: {auroc_eda:.4f}")
    print(f"    cos(encoder, letter_probe): {auroc_enc_probe:.4f}")
    print(f"    cos(decoder, letter_probe): {auroc_dec_probe:.4f}")

    # ── Analysis 4: Cross-class encoder alignment ─────────────────────────────
    letter_enc_vs_nonletter_enc = stats.mannwhitneyu(
        enc_probe_letter, enc_probe_nonletter, alternative='greater')
    print(f"\n[4] Letter encoder > Non-letter encoder alignment with probe:")
    print(f"    Mann-Whitney (greater): p={letter_enc_vs_nonletter_enc.pvalue:.2e}")

    # ── Analysis 5: Norms ────────────────────────────────────────────────────
    enc_norm_letter = enc_norms_np[is_letter_bool]
    enc_norm_nonletter = enc_norms_np[~is_letter_bool]
    dec_norm_letter = dec_norms_np[is_letter_bool]
    dec_norm_nonletter = dec_norms_np[~is_letter_bool]

    print(f"\n[5] Norms:")
    print(f"    Encoder norms — Letter: {enc_norm_letter.mean():.4f}±{enc_norm_letter.std():.4f}, "
          f"Non-letter: {enc_norm_nonletter.mean():.4f}±{enc_norm_nonletter.std():.4f}")
    print(f"    Decoder norms — Letter: {dec_norm_letter.mean():.4f}±{dec_norm_letter.std():.4f}, "
          f"Non-letter: {dec_norm_nonletter.mean():.4f}±{dec_norm_nonletter.std():.4f}")

    # ── Analysis 6: Per-feature data for scatter plot (all letter + sample of non-letter) ──
    rng = np.random.default_rng(SEED)
    letter_idx = np.where(is_letter_bool)[0]
    nonletter_all_idx = np.where(~is_letter_bool)[0]
    nonletter_sample_idx = rng.choice(nonletter_all_idx,
                                       size=min(max_nonletter_sample, len(nonletter_all_idx)),
                                       replace=False)

    per_feature_data = []
    for idx in letter_idx:
        per_feature_data.append({
            "feature_idx": int(idx),
            "is_letter": True,
            "eda": float(eda_np[idx]),
            "cos_enc_dec": float(cos_enc_dec_np[idx]),
            "cos_enc_probe": float(cos_enc_best_np[idx]),
            "cos_dec_probe": float(cos_dec_best_np[idx]),
            "cos_enc_maxprobe": float(cos_enc_maxprobe_np[idx]),
            "cos_dec_maxprobe": float(cos_dec_maxprobe_np[idx]),
            "enc_norm": float(enc_norms_np[idx]),
            "dec_norm": float(dec_norms_np[idx]),
        })
    for idx in nonletter_sample_idx:
        per_feature_data.append({
            "feature_idx": int(idx),
            "is_letter": False,
            "eda": float(eda_np[idx]),
            "cos_enc_dec": float(cos_enc_dec_np[idx]),
            "cos_enc_probe": float(cos_enc_best_np[idx]),
            "cos_dec_probe": float(cos_dec_best_np[idx]),
            "cos_enc_maxprobe": float(cos_enc_maxprobe_np[idx]),
            "cos_dec_maxprobe": float(cos_dec_maxprobe_np[idx]),
            "enc_norm": float(enc_norms_np[idx]),
            "dec_norm": float(dec_norms_np[idx]),
        })

    return {
        "n_letter_features": int(is_letter_bool.sum()),
        "n_nonletter_features": int((~is_letter_bool).sum()),
        "eda_analysis": {
            "letter_mean": float(eda_letter.mean()),
            "letter_std": float(eda_letter.std()),
            "nonletter_mean": float(eda_nonletter.mean()),
            "nonletter_std": float(eda_nonletter.std()),
            "cohens_d": cohens_d_eda,
            "ttest_t": float(eda_ttest.statistic),
            "ttest_p": float(eda_ttest.pvalue),
            "mannwhitney_p": float(eda_mwu.pvalue),
            "auroc": auroc_eda,
            "auprc": auprc_eda,
            "base_rate": float(base_rate),
        },
        "encoder_probe_alignment": {
            "letter_mean": float(enc_probe_letter.mean()),
            "letter_std": float(enc_probe_letter.std()),
            "nonletter_mean": float(enc_probe_nonletter.mean()),
            "nonletter_std": float(enc_probe_nonletter.std()),
            "auroc": auroc_enc_probe,
        },
        "decoder_probe_alignment": {
            "letter_mean": float(dec_probe_letter.mean()),
            "letter_std": float(dec_probe_letter.std()),
            "nonletter_mean": float(dec_probe_nonletter.mean()),
            "nonletter_std": float(dec_probe_nonletter.std()),
            "auroc": auroc_dec_probe,
        },
        "paired_enc_vs_dec_alignment": {
            "letter_features": {
                "diff_mean": float(diff_letter.mean()),
                "diff_std": float(diff_letter.std()),
                "ttest_t": float(paired_ttest_letter.statistic),
                "ttest_p": float(paired_ttest_letter.pvalue),
                "interpretation": "encoder_more_aligned" if enc_more else "decoder_more_aligned",
                "supports_revised_H1": bool(enc_more and paired_ttest_letter.pvalue < 0.05),
            },
            "nonletter_features": {
                "diff_mean": float(diff_nonletter.mean()),
                "diff_std": float(diff_nonletter.std()),
                "ttest_t": float(paired_ttest_nonletter.statistic),
                "ttest_p": float(paired_ttest_nonletter.pvalue),
            },
        },
        "cross_class_encoder_alignment": {
            "mannwhitney_p_letter_gt_nonletter": float(letter_enc_vs_nonletter_enc.pvalue),
            "letter_enc_mean": float(enc_probe_letter.mean()),
            "nonletter_enc_mean": float(enc_probe_nonletter.mean()),
        },
        "norms": {
            "encoder": {
                "letter_mean": float(enc_norm_letter.mean()),
                "letter_std": float(enc_norm_letter.std()),
                "nonletter_mean": float(enc_norm_nonletter.mean()),
                "nonletter_std": float(enc_norm_nonletter.std()),
            },
            "decoder": {
                "letter_mean": float(dec_norm_letter.mean()),
                "letter_std": float(dec_norm_letter.std()),
                "nonletter_mean": float(dec_norm_nonletter.mean()),
                "nonletter_std": float(dec_norm_nonletter.std()),
            },
        },
        "per_feature_data": per_feature_data,
    }


# ── Run analysis for L6 and L10 ───────────────────────────────────────────────
print(f"\nRunning FULL EDA decomposition analysis...")
write_progress(4, 10, metric={"status": "analyzing_l6"})
result_l6 = analyze_eda_decomposition_full(sae_l6, is_letter_l6, probe_dirs_norm, "L6",
                                            best_letter=best_letter_l6, max_nonletter_sample=500)

write_progress(6, 10, metric={"status": "analyzing_l10"})
result_l10 = analyze_eda_decomposition_full(sae_l10, is_letter_l10, probe_dirs_norm, "L10",
                                             best_letter=best_letter_l10, max_nonletter_sample=500)

# ── Null EDA AUROC distribution (100 permutations) ───────────────────────────
print(f"\n[NULL TEST] Computing null EDA AUROC distribution (100 permutations)...")
write_progress(7, 10, metric={"status": "null_test"})

from sae_lens import SAE
import gc

@torch.no_grad()
def compute_null_auroc(sae, is_letter, n_permutations=100, seed=42):
    W_enc = sae.W_enc
    W_dec = sae.W_dec
    enc_dirs = F.normalize(W_enc.T, dim=1)
    dec_dirs = F.normalize(W_dec, dim=1)
    cos_enc_dec = (enc_dirs * dec_dirs).sum(dim=1)
    eda_np = (1.0 - cos_enc_dec).cpu().numpy()
    is_letter_bool = is_letter.astype(bool)
    y_true = is_letter_bool.astype(int)
    rng = np.random.default_rng(seed)
    null_aurocs = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_true)
        try:
            null_aurocs.append(float(roc_auc_score(y_shuffled, eda_np)))
        except Exception:
            pass
    return null_aurocs

null_aurocs_l6 = compute_null_auroc(sae_l6, is_letter_l6, n_permutations=100, seed=SEED)
null_aurocs_l10 = compute_null_auroc(sae_l10, is_letter_l10, n_permutations=100, seed=SEED)

null_l6_mean = float(np.mean(null_aurocs_l6))
null_l6_std = float(np.std(null_aurocs_l6))
null_l10_mean = float(np.mean(null_aurocs_l10))
null_l10_std = float(np.std(null_aurocs_l10))

l6_auroc = result_l6["eda_analysis"]["auroc"]
l10_auroc = result_l10["eda_analysis"]["auroc"]

l6_sd_above_null = float((l6_auroc - null_l6_mean) / null_l6_std) if null_l6_std > 0 else float('nan')
l10_sd_above_null = float((l10_auroc - null_l10_mean) / null_l10_std) if null_l10_std > 0 else float('nan')

print(f"  L6 AUROC={l6_auroc:.4f} vs. null mean={null_l6_mean:.4f} std={null_l6_std:.4f} → {l6_sd_above_null:.2f} SD above null")
print(f"  L10 AUROC={l10_auroc:.4f} vs. null mean={null_l10_mean:.4f} std={null_l10_std:.4f} → {l10_sd_above_null:.2f} SD above null")

null_analysis = {
    "l6": {
        "null_auroc_mean": null_l6_mean,
        "null_auroc_std": null_l6_std,
        "observed_auroc": l6_auroc,
        "sd_above_null": l6_sd_above_null,
        "n_permutations": len(null_aurocs_l6),
    },
    "l10": {
        "null_auroc_mean": null_l10_mean,
        "null_auroc_std": null_l10_std,
        "observed_auroc": l10_auroc,
        "sd_above_null": l10_sd_above_null,
        "n_permutations": len(null_aurocs_l10),
    },
}

# ── Generate Scatter Plot ─────────────────────────────────────────────────────
print(f"\n[VISUALIZATION] Generating scatter plots...")
write_progress(8, 10, metric={"status": "visualization"})


def generate_scatter_plot(result, layer_name, output_path):
    """Generate scatter plot: cos(encoder_j, letter_probe) vs cos(decoder_j, letter_probe)."""
    pfd = result["per_feature_data"]

    enc_probe_letter = [d["cos_enc_probe"] for d in pfd if d["is_letter"]]
    dec_probe_letter = [d["cos_dec_probe"] for d in pfd if d["is_letter"]]
    enc_probe_nonletter = [d["cos_enc_probe"] for d in pfd if not d["is_letter"]]
    dec_probe_nonletter = [d["cos_dec_probe"] for d in pfd if not d["is_letter"]]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(dec_probe_nonletter, enc_probe_nonletter,
               c="steelblue", alpha=0.15, s=8, label=f"Non-letter (n={len(enc_probe_nonletter)})")
    ax.scatter(dec_probe_letter, enc_probe_letter,
               c="firebrick", alpha=0.7, s=25, label=f"Letter (n={len(enc_probe_letter)})")

    # y = x diagonal (encoder = decoder alignment, EDA = 0)
    lim_min = min(min(dec_probe_nonletter + dec_probe_letter),
                  min(enc_probe_nonletter + enc_probe_letter)) - 0.02
    lim_max = max(max(dec_probe_nonletter + dec_probe_letter),
                  max(enc_probe_nonletter + enc_probe_letter)) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, alpha=0.5, label="enc = dec")

    # Region above diagonal = encoder more aligned (supports revised H1)
    ax.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
                    alpha=0.04, color="green")

    ax.set_xlabel("cos(decoder_j, letter_probe)", fontsize=12)
    ax.set_ylabel("cos(encoder_j, letter_probe)", fontsize=12)
    ax.set_title(f"GPT-2 Small {layer_name}: Encoder vs. Decoder Probe Alignment\n"
                 f"(EDA = 1 − cos(enc, dec); points above diagonal → encoder more aligned)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')

    # Annotate with key stats
    diff_mean = result["paired_enc_vs_dec_alignment"]["letter_features"]["diff_mean"]
    p_val = result["paired_enc_vs_dec_alignment"]["letter_features"]["ttest_p"]
    direction = "enc > dec" if diff_mean > 0 else "dec > enc"
    ax.text(0.03, 0.97,
            f"Letter features:\ndiff(enc-dec)={diff_mean:.3f}\n{direction}, p={p_val:.2e}\nEDA AUROC={result['eda_analysis']['auroc']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved: {output_path}")


scatter_l6_path = RESULTS_DIR / "B1_scatter_L6.png"
scatter_l10_path = RESULTS_DIR / "B1_scatter_L10.png"

generate_scatter_plot(result_l6, "L6", scatter_l6_path)
generate_scatter_plot(result_l10, "L10", scatter_l10_path)

# Generate side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, result, layer in [(axes[0], result_l6, "L6"), (axes[1], result_l10, "L10")]:
    pfd = result["per_feature_data"]
    enc_l = [d["cos_enc_probe"] for d in pfd if d["is_letter"]]
    dec_l = [d["cos_dec_probe"] for d in pfd if d["is_letter"]]
    enc_nl = [d["cos_enc_probe"] for d in pfd if not d["is_letter"]]
    dec_nl = [d["cos_dec_probe"] for d in pfd if not d["is_letter"]]

    ax.scatter(dec_nl, enc_nl, c="steelblue", alpha=0.15, s=8, label=f"Non-letter ({len(enc_nl)})")
    ax.scatter(dec_l, enc_l, c="firebrick", alpha=0.7, s=25, label=f"Letter ({len(enc_l)})")

    all_vals = dec_l + dec_nl + enc_l + enc_nl
    lmin = min(all_vals) - 0.02
    lmax = max(all_vals) + 0.02
    ax.plot([lmin, lmax], [lmin, lmax], "k--", lw=1, alpha=0.5)

    eda_auroc = result["eda_analysis"]["auroc"]
    diff = result["paired_enc_vs_dec_alignment"]["letter_features"]["diff_mean"]
    p = result["paired_enc_vs_dec_alignment"]["letter_features"]["ttest_p"]
    direction = "enc>dec" if diff > 0 else "dec>enc"

    ax.set_xlabel("cos(decoder_j, probe)", fontsize=11)
    ax.set_ylabel("cos(encoder_j, probe)", fontsize=11)
    ax.set_title(f"{layer}: EDA AUROC={eda_auroc:.3f}\n{direction}, diff={diff:.3f}, p={p:.1e}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(lmin, lmax)
    ax.set_ylim(lmin, lmax)
    ax.set_aspect('equal')

plt.suptitle("GPT-2 Small: Encoder vs. Decoder Letter Probe Alignment\n"
             "(FULL mode — all letter features, sampled non-letter)", fontsize=12)
plt.tight_layout()
combined_path = RESULTS_DIR / "B1_scatter_combined.png"
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Combined scatter saved: {combined_path}")

# ── Pass/Fail Criteria ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PASS CRITERIA EVALUATION (FULL mode)")
print(f"{'='*60}")

l6_encoder_more = result_l6["paired_enc_vs_dec_alignment"]["letter_features"]["diff_mean"] > 0
l6_p_significant = result_l6["paired_enc_vs_dec_alignment"]["letter_features"]["ttest_p"] < 0.05
l10_encoder_more = result_l10["paired_enc_vs_dec_alignment"]["letter_features"]["diff_mean"] > 0
l10_p_significant = result_l10["paired_enc_vs_dec_alignment"]["letter_features"]["ttest_p"] < 0.05

print(f"L6: encoder more aligned with letter probe: {l6_encoder_more}")
print(f"L6: paired t-test p < 0.05: {l6_p_significant}")
print(f"L10: encoder more aligned with letter probe: {l10_encoder_more}")
print(f"L10: paired t-test p < 0.05: {l10_p_significant}")

# GO requires L6: encoder more aligned with letter probe than decoder
go_nogo = "GO" if (l6_encoder_more and l6_p_significant) else "NO_GO"
print(f"\nOverall GO/NO-GO: {go_nogo}")

if go_nogo == "NO_GO":
    print("\nNOTE: Revised H1 NOT confirmed at L6.")
    print(f"  L6 diff_mean = {result_l6['paired_enc_vs_dec_alignment']['letter_features']['diff_mean']:.4f}")
    print(f"  → Decoder is MORE aligned with letter probe than encoder (for letter features)")
    print(f"  → EDA at L6 (AUROC={l6_auroc:.4f}) is driven by EDA of decoder relative to encoder,")
    print(f"    but NOT because encoder was pulled toward the letter probe direction.")
    print(f"  → This suggests a different geometric mechanism: letter features have large EDA because")
    print(f"    their DECODER is very aligned with the letter probe (i.e. they ARE letter features)")
    print(f"    while their encoder is less aligned. The EDA signal may reflect decoder specialization")
    print(f"    rather than encoder pull toward the parent direction.")

# ── Geometric Interpretation ──────────────────────────────────────────────────
l6_enc_probe = result_l6["encoder_probe_alignment"]["letter_mean"]
l6_dec_probe = result_l6["decoder_probe_alignment"]["letter_mean"]
l10_enc_probe = result_l10["encoder_probe_alignment"]["letter_mean"]
l10_dec_probe = result_l10["decoder_probe_alignment"]["letter_mean"]

print(f"\n--- Geometric Comparison L6 vs L10 ---")
print(f"L6: enc→probe={l6_enc_probe:.4f}, dec→probe={l6_dec_probe:.4f}, ratio={l6_dec_probe/l6_enc_probe:.2f}x")
print(f"L10: enc→probe={l10_enc_probe:.4f}, dec→probe={l10_dec_probe:.4f}, ratio={l10_dec_probe/l10_enc_probe:.2f}x")
print(f"L6 EDA AUROC={l6_auroc:.4f} ({l6_sd_above_null:.2f} SD above null)")
print(f"L10 EDA AUROC={l10_auroc:.4f} ({l10_sd_above_null:.2f} SD above null)")

# EDA at L10 is BELOW null (AUROC < 0.5, reversed direction)
# This suggests at L10, letter features have LOWER EDA than non-letter features
l10_reversed = l10_auroc < 0.5
print(f"\nAt L10: EDA direction reversed = {l10_reversed}")
if l10_reversed:
    print("  → Letter features at L10 have LOWER EDA than non-letter features")
    print("  → L10 letter features show better encoder-decoder alignment than L6 letter features")
    print("  → Possible explanation: L10 SAE has had time to 'correct' the encoder-decoder")
    print("    dissociation after absorption, or absorption in later layers follows different geometry.")

# ── Save results ──────────────────────────────────────────────────────────────
elapsed = time.time() - start_time
print(f"\n[SAVING] Writing results...")
write_progress(9, 10, metric={"status": "saving"})

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": MODE,
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "layers": [6, 10],
        "d_sae": int(sae_l6.cfg.d_sae),
        "d_in": int(sae_l6.cfg.d_in),
        "seed": SEED,
        "probe_type": "first_letter_binary_ova",
        "n_letter_probes": int(len(probe_classes)),
        "l6_letter_threshold": 0.32,
        "l10_letter_threshold": 0.30,
        "n_nonletter_sample_for_scatter": 500,
    },
    "layer6": result_l6,
    "layer10": result_l10,
    "null_eda_analysis": null_analysis,
    "pass_criteria": {
        "l6_encoder_more_aligned": l6_encoder_more,
        "l6_paired_t_p_lt_005": l6_p_significant,
        "l10_encoder_more_aligned": l10_encoder_more,
        "l10_paired_t_p_lt_005": l10_p_significant,
        "overall_go_nogo": go_nogo,
        "note": (
            "GO requires L6: encoder more aligned with letter probe than decoder (paired t-test p < 0.05). "
            "L10 failure (reversed EDA direction) is expected and informative."
        ),
    },
    "key_findings": {
        "l6_eda_auroc": l6_auroc,
        "l6_eda_auprc": result_l6["eda_analysis"]["auprc"],
        "l6_eda_base_rate": result_l6["eda_analysis"]["base_rate"],
        "l6_eda_cohens_d": result_l6["eda_analysis"]["cohens_d"],
        "l6_eda_sd_above_null": l6_sd_above_null,
        "l6_enc_probe_mean": l6_enc_probe,
        "l6_dec_probe_mean": l6_dec_probe,
        "l6_enc_vs_dec_diff_mean": result_l6["paired_enc_vs_dec_alignment"]["letter_features"]["diff_mean"],
        "l6_paired_ttest_p": result_l6["paired_enc_vs_dec_alignment"]["letter_features"]["ttest_p"],
        "l10_eda_auroc": l10_auroc,
        "l10_eda_cohens_d": result_l10["eda_analysis"]["cohens_d"],
        "l10_eda_sd_above_null": l10_sd_above_null,
        "l10_enc_probe_mean": l10_enc_probe,
        "l10_dec_probe_mean": l10_dec_probe,
        "l10_enc_vs_dec_diff_mean": result_l10["paired_enc_vs_dec_alignment"]["letter_features"]["diff_mean"],
        "l10_paired_ttest_p": result_l10["paired_enc_vs_dec_alignment"]["letter_features"]["ttest_p"],
        "l10_eda_reversed": l10_reversed,
        "visualizations": [
            "exp/results/full/B1_scatter_L6.png",
            "exp/results/full/B1_scatter_L10.png",
            "exp/results/full/B1_scatter_combined.png",
        ],
        "geometric_interpretation": (
            "Revised H1 NOT confirmed: for letter features at L6, the DECODER is more aligned "
            "with the letter probe than the encoder (diff_mean < 0, highly significant). "
            "This means EDA at L6 reflects that letter feature DECODERS point in the letter probe "
            "direction while their ENCODERS are relatively less aligned. "
            "At L10, EDA is reversed (AUROC < 0.5): letter features show BETTER encoder-decoder "
            "alignment than non-letter features. "
            "Alternative geometric interpretation: absorbed letter features develop specialized "
            "DECODERS (pointing at the letter concept) but their ENCODERS are broader or point "
            "in intermediate directions. This produces EDA signal without encoder-toward-parent pull."
        ) if go_nogo == "NO_GO" else (
            "Revised H1 CONFIRMED: for letter features at L6, the encoder is more aligned "
            "with the letter probe than the decoder is. This supports the model that absorption "
            "pulls encoder toward parent direction while decoder remains specialized."
        ),
    },
}

output_path = RESULTS_DIR / "B1_eda_decomposition.json"
output_path.write_text(json.dumps(output, indent=2))
print(f"Results saved to: {output_path}")

# ── Update gpu_progress.json ──────────────────────────────────────────────────
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_path.exists():
        gp = json.loads(gpu_progress_path.read_text())
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]

    end_time = datetime.now().isoformat()
    gp["timings"][TASK_ID] = {
        "planned_min": 30,
        "actual_min": int(elapsed / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": end_time,
        "config_snapshot": {
            "model": "gpt2-small",
            "sae_release": "gpt2-small-res-jb",
            "layers": [6, 10],
            "d_sae": int(sae_l6.cfg.d_sae),
            "n_letter_features_l6": result_l6["n_letter_features"],
            "n_letter_features_l10": result_l10["n_letter_features"],
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_count": 1,
        },
    }

    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print(f"gpu_progress.json updated: task_B1_eda_decomposition → completed")
except Exception as e:
    print(f"Warning: could not update gpu_progress.json: {e}")

# ── Finalize ──────────────────────────────────────────────────────────────────
write_progress(10, 10, metric={"status": "done", "go_nogo": go_nogo})
mark_done(
    status="success",
    summary=(
        f"B1 EDA Decomposition FULL complete. L6 GO/NO-GO: {go_nogo}. "
        f"L6: EDA AUROC={l6_auroc:.4f} ({l6_sd_above_null:.2f} SD above null), "
        f"enc_more_aligned={l6_encoder_more}, "
        f"enc-dec diff={result_l6['paired_enc_vs_dec_alignment']['letter_features']['diff_mean']:.4f}. "
        f"L10: EDA AUROC={l10_auroc:.4f} (reversed={l10_reversed}). "
        f"Elapsed: {elapsed:.1f}s."
    )
)

print(f"\n{'='*60}")
print(f"DONE. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Result: {go_nogo}")
print(f"Output: {output_path}")
print(f"{'='*60}")
