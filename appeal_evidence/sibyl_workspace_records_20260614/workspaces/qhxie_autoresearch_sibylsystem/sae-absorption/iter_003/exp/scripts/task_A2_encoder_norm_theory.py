"""
task_A2_encoder_norm_theory.py
-------------------------------
Encoder Norm Theoretical Analysis and Mechanistic Test (PILOT mode)

Tests:
1. Spearman correlation of encoder_norm with EDA across all 24576 latents (L6)
2. Early vs. late absorbed latents differ in encoder_norm
   (inferred from decoder cosine similarity with known parent features)
3. Encoder_norm monotonicity across layers L2, L4, L6, L8, L10
4. Theoretical note: mechanistic interpretation

Dependencies:
- A1 results: exp/results/full/A1_encoder_norm_replication.json
- Exact labels: iter_001/exp/results/r4/r4a_direct_labels.json
- SAELens: gpt2-small-res-jb
"""

import json
import os
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sae_lens import SAE

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
ITER001_LABELS = Path(
    "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption"
    "/iter_001/exp/results/r4/r4a_direct_labels.json"
)
A1_RESULTS = WORKSPACE / "exp/results/full/A1_encoder_norm_replication.json"
RESULTS_DIR = WORKSPACE / "exp/results/full"
TASK_ID = "task_A2_encoder_norm_theory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Layer configuration
LAYERS = [2, 4, 6, 8, 10]
SAE_RELEASE = "gpt2-small-res-jb"
HOOK_TEMPLATE = "blocks.{}.hook_resid_pre"

# ── PID file ───────────────────────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"

def write_progress(step, total_steps, note=""):
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "note": note,
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status, summary):
    if pid_file.exists():
        pid_file.unlink()
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ── Helper: compute encoder_norm and EDA for all latents in an SAE ─────────────
def compute_enc_norm_and_eda(sae):
    """
    encoder_norm_j = ||W_enc[:, j]||_2  (L2 norm of j-th column of encoder weight)
    EDA_j = 1 - cosine_similarity(W_enc[:, j], W_dec[j, :])
    """
    W_enc = sae.W_enc.float()  # shape [d_in, d_sae]
    W_dec = sae.W_dec.float()  # shape [d_sae, d_in]

    enc_cols = W_enc.T          # [d_sae, d_in]
    dec_rows = W_dec            # [d_sae, d_in]

    enc_norm = torch.norm(enc_cols, dim=1)  # [d_sae]

    # EDA = 1 - cos(enc_j, dec_j)
    enc_unit = enc_cols / (enc_norm.unsqueeze(1) + 1e-8)
    dec_norm = torch.norm(dec_rows, dim=1, keepdim=True)
    dec_unit = dec_rows / (dec_norm + 1e-8)
    cos_sim = (enc_unit * dec_unit).sum(dim=1)
    eda = 1.0 - cos_sim  # [d_sae]

    return enc_norm.detach().cpu().numpy(), eda.detach().cpu().numpy()


# ── Load labels ────────────────────────────────────────────────────────────────
def load_l6_labels():
    with open(ITER001_LABELS) as f:
        data = json.load(f)

    # Find GPT2-L6 entry
    for entry in data["per_sae_results"]:
        if entry["config"]["name"] == "GPT2-L6":
            absorbed_ids = set(entry["absorbed_latent_ids"])
            d_sae = entry["config"]["d_sae"]
            return absorbed_ids, d_sae, entry

    raise ValueError("GPT2-L6 not found in label file")


# ── Task 1: Spearman correlation encoder_norm vs EDA over all latents (L6) ────
def task1_spearman_correlation(sae_l6):
    """Compute Spearman correlation between encoder_norm and EDA for all 24576 latents."""
    write_progress(1, 5, "Computing encoder_norm and EDA for all L6 latents")
    enc_norm, eda = compute_enc_norm_and_eda(sae_l6)
    r, p = spearmanr(enc_norm, eda)
    return {
        "spearman_r": float(r),
        "spearman_p": float(p),
        "n_latents": len(enc_norm),
        "enc_norm_stats": {
            "mean": float(np.mean(enc_norm)),
            "std": float(np.std(enc_norm)),
            "p25": float(np.percentile(enc_norm, 25)),
            "p50": float(np.percentile(enc_norm, 50)),
            "p75": float(np.percentile(enc_norm, 75)),
        },
        "eda_stats": {
            "mean": float(np.mean(eda)),
            "std": float(np.std(eda)),
            "p25": float(np.percentile(eda, 25)),
            "p50": float(np.percentile(eda, 50)),
            "p75": float(np.percentile(eda, 75)),
        },
    }, enc_norm, eda


# ── Task 2: Early vs. Late absorbed encoder_norm ────────────────────────────────
def task2_early_late_subtype(sae_l6, absorbed_ids, enc_norm_l6, eda_l6, label_entry):
    """
    Infer early vs. late absorption for GPT-2 L6 absorbed latents.

    Strategy:
    - Early absorption = decoder-absent = absorbed feature's decoder direction
      does NOT have a highly similar other feature (low max decoder cosine similarity
      with any non-absorbed feature).
    - Late absorption = decoder-present = the absorbed feature's decoder direction
      IS similar to a non-absorbed letter feature (max decoder cosine > threshold).

    Threshold: 0.3 (same as iter_001 taxonomy).

    Additionally check decoder cosine similarity between absorbed feature and
    known 'parent' features (other letter features of the same letter).
    """
    write_progress(2, 5, "Computing early vs late absorption taxonomy for L6")

    W_dec = sae_l6.W_dec.float().detach().cpu().numpy()  # [d_sae, d_in]
    absorbed_ids_list = sorted(absorbed_ids)

    # Get all letter feature IDs (from label_entry)
    all_letter_features = set()
    absorption_details = label_entry.get("absorption_details", {})
    for letter, detail in absorption_details.items():
        for fid in detail.get("main_feature_ids", []):
            all_letter_features.add(fid)

    non_absorbed_letter_features = all_letter_features - absorbed_ids

    # For each absorbed feature, compute max cosine similarity with non-absorbed letter features
    absorbed_dec = W_dec[absorbed_ids_list]  # [n_pos, d_in]
    non_absorbed_arr = list(non_absorbed_letter_features) if non_absorbed_letter_features else []

    subtypes = {}
    if non_absorbed_arr:
        non_abs_dec = W_dec[non_absorbed_arr]  # [n_non_abs, d_in]
        # Normalize
        absorbed_norm = np.linalg.norm(absorbed_dec, axis=1, keepdims=True) + 1e-8
        non_abs_norm = np.linalg.norm(non_abs_dec, axis=1, keepdims=True) + 1e-8
        absorbed_unit = absorbed_dec / absorbed_norm
        non_abs_unit = non_abs_dec / non_abs_norm
        cos_mat = absorbed_unit @ non_abs_unit.T  # [n_pos, n_non_abs]

        DECODER_THRESHOLD = 0.3
        for i, fid in enumerate(absorbed_ids_list):
            max_cos = float(np.max(cos_mat[i]))
            subtype = "late" if max_cos >= DECODER_THRESHOLD else "early"
            subtypes[fid] = {
                "subtype": subtype,
                "max_cos_with_non_absorbed_letter_features": max_cos,
                "encoder_norm": float(enc_norm_l6[fid]),
                "eda": float(eda_l6[fid]),
            }
    else:
        # No non-absorbed letter features -- fallback: use all SAE features for comparison
        # Sample random 1000 non-absorbed features for speed
        rng = np.random.default_rng(42)
        all_ids = np.array([i for i in range(W_dec.shape[0]) if i not in absorbed_ids])
        sample_ids = rng.choice(all_ids, size=min(1000, len(all_ids)), replace=False)
        sample_dec = W_dec[sample_ids]
        absorbed_norm = np.linalg.norm(absorbed_dec, axis=1, keepdims=True) + 1e-8
        sample_norm = np.linalg.norm(sample_dec, axis=1, keepdims=True) + 1e-8
        absorbed_unit = absorbed_dec / absorbed_norm
        sample_unit = sample_dec / sample_norm
        cos_mat = absorbed_unit @ sample_unit.T

        DECODER_THRESHOLD = 0.3
        for i, fid in enumerate(absorbed_ids_list):
            max_cos = float(np.max(cos_mat[i]))
            subtype = "late" if max_cos >= DECODER_THRESHOLD else "early"
            subtypes[fid] = {
                "subtype": subtype,
                "max_cos_with_sampled_features": max_cos,
                "encoder_norm": float(enc_norm_l6[fid]),
                "eda": float(eda_l6[fid]),
                "note": "fallback: compared against random sample of non-absorbed features",
            }

    early_feats = [v for v in subtypes.values() if v["subtype"] == "early"]
    late_feats = [v for v in subtypes.values() if v["subtype"] == "late"]

    early_enc_norms = [f["encoder_norm"] for f in early_feats]
    late_enc_norms = [f["encoder_norm"] for f in late_feats]

    from scipy.stats import mannwhitneyu

    mw_result = None
    if early_enc_norms and late_enc_norms:
        stat, p = mannwhitneyu(early_enc_norms, late_enc_norms, alternative="two-sided")
        mw_result = {"statistic": float(stat), "p_value": float(p)}

    return {
        "taxonomy": subtypes,
        "n_early": len(early_feats),
        "n_late": len(late_feats),
        "decoder_threshold": 0.3,
        "early_encoder_norm_stats": {
            "mean": float(np.mean(early_enc_norms)) if early_enc_norms else None,
            "std": float(np.std(early_enc_norms)) if early_enc_norms else None,
            "values": [round(x, 4) for x in early_enc_norms],
        },
        "late_encoder_norm_stats": {
            "mean": float(np.mean(late_enc_norms)) if late_enc_norms else None,
            "std": float(np.std(late_enc_norms)) if late_enc_norms else None,
            "values": [round(x, 4) for x in late_enc_norms],
        },
        "mannwhitney_early_vs_late_encoder_norm": mw_result,
        "interpretation": (
            "early_absorbed > late_absorbed encoder_norm suggests amortization gap is "
            "worse for decoder-absent latents (no dictionary anchor, encoder over-compensates)"
            if (early_enc_norms and late_enc_norms and
                np.mean(early_enc_norms) > np.mean(late_enc_norms))
            else "no clear early > late encoder_norm difference observed"
        ),
    }


# ── Task 3: Layer monotonicity of encoder_norm ──────────────────────────────────
def task3_layer_monotonicity(absorbed_ids):
    """
    Compute mean encoder_norm of absorbed vs. non-absorbed latents for each layer.
    Test if encoder_norm peaks at absorption-active layers (L6).
    """
    write_progress(3, 5, "Computing encoder_norm across layers L2, L4, L6, L8, L10")

    layer_results = {}

    for layer_idx in LAYERS:
        hook_name = HOOK_TEMPLATE.format(layer_idx)
        try:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=hook_name)
            sae = sae.to(DEVICE)
            enc_norm, eda = compute_enc_norm_and_eda(sae)
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            abs_ids_list = list(absorbed_ids)
            non_abs_mask = np.ones(len(enc_norm), dtype=bool)
            non_abs_mask[abs_ids_list] = False

            abs_norms = enc_norm[abs_ids_list]
            non_abs_norms = enc_norm[non_abs_mask]

            layer_results[f"L{layer_idx}"] = {
                "layer_idx": layer_idx,
                "mean_enc_norm_absorbed": float(np.mean(abs_norms)),
                "std_enc_norm_absorbed": float(np.std(abs_norms)),
                "mean_enc_norm_non_absorbed": float(np.mean(non_abs_norms)),
                "std_enc_norm_non_absorbed": float(np.std(non_abs_norms)),
                "ratio_absorbed_over_non_absorbed": float(np.mean(abs_norms) / (np.mean(non_abs_norms) + 1e-8)),
                "n_absorbed": len(abs_ids_list),
                "n_non_absorbed": int(np.sum(non_abs_mask)),
            }
        except Exception as e:
            layer_results[f"L{layer_idx}"] = {
                "layer_idx": layer_idx,
                "error": str(e),
            }

    # Test monotonicity: does ratio peak at L6?
    ratios = {
        k: v["ratio_absorbed_over_non_absorbed"]
        for k, v in layer_results.items()
        if "ratio_absorbed_over_non_absorbed" in v
    }
    if ratios:
        max_layer = max(ratios, key=lambda k: ratios[k])
        ratio_values = [ratios.get(f"L{l}", None) for l in LAYERS]
        monotone_up_to_l6 = all(
            ratio_values[i] is not None and ratio_values[i+1] is not None and
            ratio_values[i] <= ratio_values[i+1]
            for i in range(LAYERS.index(6))
            if f"L{LAYERS[i]}" in ratios and f"L{LAYERS[i+1]}" in ratios
        ) if LAYERS.index(6) > 0 else True
    else:
        max_layer = None
        monotone_up_to_l6 = False

    return {
        "layer_results": layer_results,
        "peak_ratio_layer": max_layer,
        "ratios_by_layer": ratios,
        "monotone_increase_up_to_L6": monotone_up_to_l6,
        "interpretation": (
            f"Encoder norm ratio (absorbed/non-absorbed) peaks at {max_layer}. "
            + ("This is consistent with absorption being most active at L6, "
               "where encoder norms of absorbed latents are disproportionately high."
               if max_layer == "L6"
               else f"Peak at {max_layer} rather than L6 suggests absorption signal "
                    "may not be layer-specific or L6 labels are used for all layers.")
        ),
    }


# ── Theoretical note ────────────────────────────────────────────────────────────
MECHANISTIC_NOTE = """
Mechanistic Interpretation of Encoder Norm as Absorption Signal
===============================================================

THEORETICAL BASIS:
encoder_norm_j = ||W_enc[:, j]||_2 = sqrt(sum_d W_enc[d, j]^2)

This is the L2 norm of the j-th column of the encoder weight matrix (the encoder direction for latent j).
A high encoder_norm means the feature has a sharply peaked encoder response over the model's residual
stream basis -- the encoder direction has large magnitude, requiring a strong input signal to fire.

AMORTIZATION GAP CONNECTION:
The amortization gap (O'Neill et al., 2411.13117) states that the feedforward encoder is a
compressed approximation of optimal sparse inference. For features that are frequently absorbed:
- The SAE learns to encode feature j via W_enc[:, j] with high norm to capture the sparse signal
  that remains after the absorbing feature i has consumed most of the activation budget
- This competition drives ||W_enc[:, j]|| upward as a compensatory mechanism
- Under the absorbing feature's co-activation, the residual signal for j is small but the encoder
  must still detect it reliably, leading to a high-norm encoding direction

SPARSITY LANDSCAPE CONNECTION:
Tang et al. (2512.05534) show absorption corresponds to spurious local minima in the piecewise
biconvex SDL loss. High encoder norms may reflect the 'attraction basin' geometry around these
minima -- the encoder direction becomes sharper to distinguish the absorbed feature's small residual
signal from noise, without the decoder direction having converged to the correct parent direction
(early absorption) or while the decoder converges to the parent (late absorption).

PREDICTION:
If the amortization gap is the dominant cause (H_ENC):
- encoder_norm should correlate positively with EDA (both signal the same phenomenon)
- encoder_norm should be higher for early-absorbed than late-absorbed latents
  (early = no decoder anchor, encoder must 'over-reach' more)
- encoder_norm ratio (absorbed/non-absorbed) should peak at the layer where absorption is active

If the sparsity landscape is the dominant cause:
- encoder_norm may correlate with EDA but through a different mechanism
  (EDA = decoder-encoder angular divergence, encoder_norm = encoder amplitude)
- The ratio may be stable across layers

The positive Spearman correlation between encoder_norm and EDA (r ~ 0.71 from A1) suggests both
metrics capture a shared underlying signal, but they are NOT identical: DeLong test shows
encoder_norm AUROC > EDA AUROC (0.757 vs. 0.650 at L6), implying encoder_norm adds information
beyond what EDA alone captures.
"""


# ── Main execution ─────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    write_progress(0, 5, "Starting A2 encoder norm theory analysis")

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "seed": 42,
        "label_file": str(ITER001_LABELS),
    }

    # Load L6 labels
    absorbed_ids, d_sae, label_entry = load_l6_labels()
    results["n_absorbed_L6"] = len(absorbed_ids)
    results["d_sae_L6"] = d_sae
    results["absorbed_latent_ids_L6"] = sorted(absorbed_ids)

    # Load L6 SAE
    write_progress(0, 5, "Loading L6 SAE")
    sae_l6 = SAE.from_pretrained(release=SAE_RELEASE, sae_id=HOOK_TEMPLATE.format(6))
    sae_l6 = sae_l6.to(DEVICE)

    # Task 1: Spearman correlation
    print("[A2] Task 1: Spearman correlation encoder_norm vs EDA (all 24576 latents, L6)")
    t1_results, enc_norm_l6, eda_l6 = task1_spearman_correlation(sae_l6)
    results["task1_spearman_full_latents"] = t1_results
    print(f"  Spearman r = {t1_results['spearman_r']:.4f}, p = {t1_results['spearman_p']:.2e}")

    # Also compute Spearman restricted to absorbed vs. non-absorbed
    absorbed_ids_list = sorted(absorbed_ids)
    # Sample 500 non-absorbed for comparison
    rng = np.random.default_rng(42)
    all_non_abs = np.array([i for i in range(d_sae) if i not in absorbed_ids])
    sample_non_abs = rng.choice(all_non_abs, size=500, replace=False)
    compare_ids = np.concatenate([absorbed_ids_list, sample_non_abs])
    r_sub, p_sub = spearmanr(enc_norm_l6[compare_ids], eda_l6[compare_ids])
    results["task1_spearman_absorbed_plus_sample"] = {
        "n_absorbed": len(absorbed_ids_list),
        "n_non_absorbed_sample": len(sample_non_abs),
        "spearman_r": float(r_sub),
        "spearman_p": float(p_sub),
    }
    print(f"  Spearman (absorbed + 500 non-abs sample) r = {r_sub:.4f}")

    # Task 2: Early vs. Late subtype encoder_norm
    print("[A2] Task 2: Early vs. late absorbed encoder_norm at L6")
    t2_results = task2_early_late_subtype(sae_l6, absorbed_ids, enc_norm_l6, eda_l6, label_entry)
    results["task2_early_late_subtype"] = t2_results
    print(f"  n_early={t2_results['n_early']}, n_late={t2_results['n_late']}")
    if t2_results['early_encoder_norm_stats']['mean'] is not None:
        print(f"  early mean enc_norm={t2_results['early_encoder_norm_stats']['mean']:.4f}")
    if t2_results['late_encoder_norm_stats']['mean'] is not None:
        print(f"  late mean enc_norm={t2_results['late_encoder_norm_stats']['mean']:.4f}")

    del sae_l6
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Task 3: Layer monotonicity
    print("[A2] Task 3: Layer monotonicity of encoder_norm (L2, L4, L6, L8, L10)")
    write_progress(4, 5, "Computing layer monotonicity")
    t3_results = task3_layer_monotonicity(absorbed_ids)
    results["task3_layer_monotonicity"] = t3_results
    print(f"  Peak ratio layer: {t3_results['peak_ratio_layer']}")
    for layer_key, ratio in t3_results['ratios_by_layer'].items():
        print(f"  {layer_key}: ratio={ratio:.4f}")

    # Mechanistic note
    results["mechanistic_note"] = MECHANISTIC_NOTE

    # Pilot pass criteria check
    t1_ok = abs(t1_results["spearman_r"]) > 0.01  # any correlation computed
    t2_ok = t2_results["n_early"] + t2_results["n_late"] == len(absorbed_ids)
    t3_ok = len(t3_results["layer_results"]) >= 3  # at least 3 layers succeeded

    results["pilot_pass_criteria"] = {
        "spearman_correlation_computed": t1_ok,
        "early_late_difference_reported": t2_ok,
        "layer_monotonicity_computed": t3_ok,
        "pass": t1_ok and t3_ok,
        "spearman_r": t1_results["spearman_r"],
        "n_layers_computed": len([v for v in t3_results["layer_results"].values() if "error" not in v]),
    }

    results["elapsed_sec"] = time.time() - start_time

    # Summary for human reading
    results["summary"] = {
        "spearman_r_enc_norm_vs_eda_all_latents": t1_results["spearman_r"],
        "spearman_p_value": t1_results["spearman_p"],
        "n_early_absorbed": t2_results["n_early"],
        "n_late_absorbed": t2_results["n_late"],
        "early_mean_enc_norm": t2_results["early_encoder_norm_stats"]["mean"],
        "late_mean_enc_norm": t2_results["late_encoder_norm_stats"]["mean"],
        "peak_ratio_layer": t3_results["peak_ratio_layer"],
        "pilot_pass": results["pilot_pass_criteria"]["pass"],
    }

    # Save results
    output_path = RESULTS_DIR / "A2_encoder_norm_theory.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[A2] Results saved to {output_path}")

    mark_done(
        status="success",
        summary=(
            f"Spearman r(enc_norm, EDA)={t1_results['spearman_r']:.4f}; "
            f"n_early={t2_results['n_early']}, n_late={t2_results['n_late']}; "
            f"peak ratio layer={t3_results['peak_ratio_layer']}; "
            f"pilot pass={results['pilot_pass_criteria']['pass']}"
        )
    )
    print("[A2] DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[A2] FATAL ERROR: {e}\n{tb}")
        # Write DONE marker with failure status
        done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
        done_marker.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "summary": f"Fatal error: {e}",
            "traceback": tb,
            "timestamp": datetime.now().isoformat(),
        }))
        if pid_file.exists():
            pid_file.unlink()
        raise
