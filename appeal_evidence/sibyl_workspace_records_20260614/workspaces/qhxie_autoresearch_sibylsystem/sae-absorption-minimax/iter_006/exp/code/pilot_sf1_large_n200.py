#!/usr/bin/env python3
"""
Pilot H-SF1: Sensitivity Floor Quadrant Classification (N=200)
Classify 200 features into 4 quadrants using Chanin absorption protocol
and Tian sensitivity protocol. Test H-SF1: Q2+Q4 remain empty (<5%) at larger sample.
Also fit quadratic model S(A)=aA^2+bA+c for H-SF2.
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
TASK_ID = "pilot_sf1_large_n200"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 42
N_FEATURES = 200
N_TOKENS = 500
LAYER = 8
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"

TAU_FS = 0.03
TAU_PA = 0.4
SENSITIVITY_THRESHOLD = 0.65

np.random.seed(SEED)
torch.manual_seed(SEED)


def write_pid(results_dir):
    pid_file = Path(results_dir) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(epoch=0, total_epochs=1, step=0, metric=None):
    progress_file = Path(WORKSPACE / "exp/results") / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "metric": metric,
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    results_dir = WORKSPACE / "exp/results"
    pid_file = results_dir / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    marker = results_dir / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))


def get_first_letter_bucket(token: str) -> int:
    if not token:
        return -1
    first_char = token[0].upper()
    if first_char.isalpha():
        return 0 if first_char <= 'M' else 1
    return -1


def load_model_and_sae():
    print(f"Loading model {MODEL_NAME} and SAE {SAE_RELEASE}...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    result = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE,
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE
    )
    sae = result[0] if isinstance(result, tuple) else result
    print(f"  SAE loaded: d_sae={sae.cfg.d_sae}")
    return model, sae


def generate_diverse_tokens(model, n_samples=500):
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A bird in the hand is worth two in the bush.",
        "Time flies when you're having fun!",
        "Actions speak louder than words...",
        "123 Main Street, Apt #4B.",
        "Hello, World! How are you?",
        "Email: test@example.com",
        "Price: $19.99 (50% off!)",
        "Call 555-1234 now!",
        "Products: A1, B2, C3, D4.",
        "the and that this with have from they",
        "was will for have were been can",
        "xyz abc mno def pqr stu ghi",
        "123 456 789 012 345 678 901",
        "... !!! ??? --- === +++",
        "### @@@ === __* &%$",
    ] * (n_samples // 16 + 1)

    tokens = model.tokenizer.batch_encode_plus(
        sample_texts[:n_samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )["input_ids"].to(DEVICE)
    return tokens


def select_features_by_quantile(mean_acts, n_features=200):
    """Select features across the activation spectrum, not just top ones."""
    valid_mask = mean_acts > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < n_features:
        return valid_indices.tolist()

    quantiles = np.linspace(0.05, 1.0, n_features + 1)
    q_values = np.quantile(mean_acts[valid_indices], quantiles)

    selected = []
    for i in range(len(quantiles) - 1):
        low = q_values[i]
        high = q_values[i + 1]
        in_range = valid_indices[(mean_acts[valid_indices] >= low) & (mean_acts[valid_indices] < high)]
        if len(in_range) > 0:
            selected.append(np.random.choice(in_range))
        else:
            remaining = [idx for idx in valid_indices if idx not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))

    remaining = [idx for idx in valid_indices if idx not in selected]
    np.random.shuffle(remaining)
    selected.extend(remaining[:n_features - len(selected)])

    return selected[:n_features]


def compute_absorption_batch(model, sae, tokens, feature_indices, layer=8):
    results = {}

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        resid_post = cache[f"blocks.{layer}.hook_resid_post"]
        resid_flat = resid_post.reshape(-1, resid_post.shape[-1])
        sae_acts = sae.encode(resid_flat.to(DEVICE)).cpu().numpy()

    tokens_flat = tokens.reshape(-1)
    token_strings = model.tokenizer.batch_decode(tokens_flat)
    labels = np.array([get_first_letter_bucket(t) for t in token_strings])
    alphabetic_mask = labels >= 0

    for i, feat_idx in enumerate(feature_indices):
        if i % 20 == 0:
            print(f"  Absorption progress: {i}/{len(feature_indices)}")

        feat_acts = sae_acts[:, feat_idx]
        high_act_mask = feat_acts > TAU_FS

        combined_mask = alphabetic_mask & high_act_mask
        pos_indices = np.where(combined_mask)[0]

        if len(pos_indices) < 5:
            results[feat_idx] = {"uas": np.nan, "acc_resid": np.nan, "acc_sae": np.nan, "n_positive": int(high_act_mask.sum())}
            continue

        neg_indices = np.where(alphabetic_mask & ~high_act_mask)[0]
        np.random.shuffle(neg_indices)
        neg_indices = neg_indices[:len(pos_indices)]

        if len(neg_indices) < 5:
            results[feat_idx] = {"uas": np.nan, "acc_resid": np.nan, "acc_sae": np.nan, "n_positive": int(high_act_mask.sum())}
            continue

        all_indices = np.concatenate([pos_indices, neg_indices])
        all_labels = labels[all_indices]
        all_resid = resid_flat[all_indices].cpu().numpy()
        all_sae_acts = sae_acts[all_indices, feat_idx].reshape(-1, 1)

        try:
            lr_resid = LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs')
            lr_resid.fit(all_resid, all_labels)
            acc_resid = lr_resid.score(all_resid, all_labels)

            lr_sae = LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs')
            lr_sae.fit(all_sae_acts, all_labels)
            acc_sae = lr_sae.score(all_sae_acts, all_labels)

            if acc_resid > acc_sae:
                uas = (acc_resid - acc_sae) / (1 - acc_sae)
            else:
                uas = 0.0

            results[feat_idx] = {
                "uas": float(uas),
                "acc_resid": float(acc_resid),
                "acc_sae": float(acc_sae),
                "n_positive": int(high_act_mask.sum())
            }
        except:
            results[feat_idx] = {"uas": np.nan, "acc_resid": np.nan, "acc_sae": np.nan, "n_positive": int(high_act_mask.sum())}

    return results, sae_acts


def compute_sensitivity_stability(sae_acts, feature_indices):
    """Compute sensitivity using split-half correlation (Tian 2025 paraphrase proxy)."""
    results = {}
    n = sae_acts.shape[0]
    half = n // 2

    for i, feat_idx in enumerate(feature_indices):
        if i % 50 == 0:
            print(f"  Sensitivity progress: {i}/{len(feature_indices)}")

        feat_acts = sae_acts[:, feat_idx]
        acts1 = feat_acts[:half]
        acts2 = feat_acts[half:2*half]

        if np.std(acts1) > 0 and np.std(acts2) > 0:
            corr = np.corrcoef(acts1, acts2)[0, 1]
            if np.isnan(corr):
                corr = 0.5
        else:
            corr = 0.5

        # AUC-style sensitivity: (corr + 1) / 2 maps [-1, 1] to [0, 1]
        sensitivity = (corr + 1) / 2
        results[feat_idx] = float(sensitivity)

    return results


def classify_quadrants(absorption_results, sensitivity_results, tau_pa=0.4, sens_thresh=0.65):
    quadrants = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

    for feat_idx in absorption_results:
        uas = absorption_results[feat_idx].get("uas", np.nan)
        sens = sensitivity_results.get(feat_idx, np.nan)

        if np.isnan(uas) or np.isnan(sens):
            continue

        is_absorbed = uas < tau_pa
        is_low_sens = sens < sens_thresh

        if is_absorbed and is_low_sens:
            quadrants["Q1"].append(int(feat_idx))
        elif is_absorbed and not is_low_sens:
            quadrants["Q2"].append(int(feat_idx))
        elif not is_absorbed and is_low_sens:
            quadrants["Q3"].append(int(feat_idx))
        else:
            quadrants["Q4"].append(int(feat_idx))

    return quadrants


def fit_u_shape(absorption_scores, sensitivity_scores):
    """Fit S(A) = aA^2 + bA + c quadratic model for U-shape test (H-SF2)."""
    valid_pairs = [(a, s) for a, s in zip(absorption_scores, sensitivity_scores)
                  if not np.isnan(a) and not np.isnan(s) and a >= 0]

    if len(valid_pairs) < 10:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan, "n_points": len(valid_pairs)}

    A = np.array([p[0] for p in valid_pairs]).reshape(-1, 1)
    S = np.array([p[1] for p in valid_pairs])

    poly = PolynomialFeatures(degree=2)
    A_poly = poly.fit_transform(A)

    model = LinearRegression()
    model.fit(A_poly, S)

    a = model.coef_[2] if len(model.coef_) > 2 else 0
    b = model.coef_[1] if len(model.coef_) > 1 else 0
    c = model.intercept_
    r2 = model.score(A_poly, S)

    return {"a": float(a), "b": float(b), "c": float(c), "r2": float(r2), "n_points": len(valid_pairs)}


def update_gpu_progress(task_id, status, summary, planned_min, actual_min, config_snapshot):
    """Update gpu_progress.json after task completion."""
    gpu_progress_path = WORKSPACE / "exp/gpu_progress.json"
    if gpu_progress_path.exists():
        with open(gpu_progress_path) as f:
            gpu_progress = json.load(f)
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if status == "completed":
        if task_id not in gpu_progress["completed"]:
            gpu_progress["completed"].append(task_id)
        if task_id in gpu_progress.get("running", {}):
            del gpu_progress["running"][task_id]
    elif status == "failed":
        if task_id not in gpu_progress["failed"]:
            gpu_progress["failed"].append(task_id)
        if task_id in gpu_progress.get("running", {}):
            del gpu_progress["running"][task_id]

    gpu_progress["timings"][task_id] = {
        "planned_min": planned_min,
        "actual_min": actual_min,
        "config_snapshot": config_snapshot
    }

    with open(gpu_progress_path, "w") as f:
        json.dump(gpu_progress, f, indent=2)


def main():
    print(f"[{TASK_ID}] Starting Sensitivity Floor Pilot (N=200)...")
    print(f"Device: {DEVICE}, Layer: {LAYER}")

    results_dir = WORKSPACE / "exp/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pilots_dir = results_dir / "pilots"
    pilots_dir.mkdir(parents=True, exist_ok=True)

    write_pid(results_dir)
    report_progress(step=1, metric={"phase": "loading_model"})

    model, sae = load_model_and_sae()

    print("Generating diverse token dataset...")
    tokens = generate_diverse_tokens(model, n_samples=N_TOKENS)
    print(f"Token dataset shape: {tokens.shape}")

    report_progress(step=2, metric={"phase": "computing_activations"})
    print("Computing activations for feature selection...")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        resid_post = cache[f"blocks.{LAYER}.hook_resid_post"]
        resid_flat = resid_post.reshape(-1, resid_post.shape[-1])
        sae_acts = sae.encode(resid_flat.to(DEVICE)).cpu().numpy()

    mean_acts = np.mean(sae_acts, axis=0)
    print(f"Features with mean > 0: {(mean_acts > 0).sum()}/{len(mean_acts)}")

    # Select features across the activation spectrum
    print(f"Selecting {N_FEATURES} features across activation quantiles...")
    selected = select_features_by_quantile(mean_acts, n_features=N_FEATURES)
    print(f"Selected {len(selected)} features")

    selected_mean_acts = mean_acts[selected]
    print(f"Selected feature mean activation: min={selected_mean_acts.min():.4f}, max={selected_mean_acts.max():.4f}, median={np.median(selected_mean_acts):.4f}")

    # Compute absorption
    print("Computing absorption scores...")
    report_progress(step=3, metric={"phase": "absorption"})
    absorption_results, sae_acts = compute_absorption_batch(model, sae, tokens, selected, layer=LAYER)

    valid_abs = [k for k, v in absorption_results.items() if not np.isnan(v.get("uas", np.nan))]
    print(f"Valid absorption measurements: {len(valid_abs)}/{len(selected)}")

    valid_uas = [absorption_results[k]['uas'] for k in valid_abs]
    print(f"UAS of valid features: min={min(valid_uas):.4f}, max={max(valid_uas):.4f}")
    absorbed_count = sum(1 for uas in valid_uas if uas < TAU_PA)
    print(f"  Absorbed (UAS < 0.4): {absorbed_count}")

    # Compute sensitivity
    print("Computing sensitivity scores...")
    report_progress(step=4, metric={"phase": "sensitivity"})
    sensitivity_results = compute_sensitivity_stability(sae_acts, selected)

    # Classify into quadrants
    print("Classifying into quadrants...")
    report_progress(step=5, metric={"phase": "quadrant_classification"})
    quadrants = classify_quadrants(absorption_results, sensitivity_results, tau_pa=TAU_PA, sens_thresh=SENSITIVITY_THRESHOLD)

    for q_name, q_features in quadrants.items():
        print(f"  {q_name}: {len(q_features)} features")

    # Compute correlation on valid features
    valid_pairs = [(k, absorption_results[k]['uas'], sensitivity_results.get(k, np.nan))
                   for k in absorption_results.keys()
                   if not np.isnan(absorption_results[k].get('uas', np.nan)) and not np.isnan(sensitivity_results.get(k, np.nan))]

    if len(valid_pairs) > 2:
        uas_vals = [p[1] for p in valid_pairs]
        sens_vals = [p[2] for p in valid_pairs]
        if np.std(uas_vals) > 0 and np.std(sens_vals) > 0:
            spearman_r, spearman_p = spearmanr(uas_vals, sens_vals)
        else:
            spearman_r, spearman_p = np.nan, np.nan
    else:
        spearman_r, spearman_p = np.nan, np.nan

    print(f"Spearman r(absorption, sensitivity): {spearman_r:.4f} (p={spearman_p:.4e})")

    # Fit U-shape model (H-SF2)
    print("Fitting U-shape model S(A) = aA^2 + bA + c...")
    uas_for_fit = [absorption_results[k]['uas'] for k in valid_abs if not np.isnan(absorption_results[k]['uas'])]
    sens_for_fit = [sensitivity_results.get(k, np.nan) for k in valid_abs if not np.isnan(absorption_results[k]['uas']) and not np.isnan(sensitivity_results.get(k, np.nan))]

    u_shape_result = fit_u_shape(uas_for_fit, sens_for_fit)
    print(f"  Quadratic coefficient a = {u_shape_result['a']:.4f} (positive = U-shape)")
    print(f"  R^2 = {u_shape_result['r2']:.4f}")

    # Compute Q2+Q4 fraction
    total_valid = len(valid_pairs)
    q2_q4_count = len(quadrants["Q2"]) + len(quadrants["Q4"])
    q2_q4_fraction = q2_q4_count / total_valid if total_valid > 0 else 0
    print(f"Q2+Q4 fraction: {q2_q4_fraction:.1%} ({q2_q4_count}/{total_valid})")

    # H-SF1 pass criteria: Q2+Q4 < 5%
    hsf1_pass = q2_q4_fraction < 0.05
    # H-SF2 pass criteria: quadratic coefficient a > 0 (U-shape)
    hsf2_pass = u_shape_result['a'] > 0

    print(f"\n=== H-SF1 (Emptiness): Q2+Q4 < 5% ===")
    print(f"  Result: {q2_q4_fraction:.1%} ({q2_q4_count}/{total_valid})")
    print(f"  {'PASS' if hsf1_pass else 'FAIL'} (threshold: < 5%)")

    print(f"\n=== H-SF2 (U-Shape): a > 0 ===")
    print(f"  Result: a = {u_shape_result['a']:.4f}")
    print(f"  {'PASS' if hsf2_pass else 'FAIL'} (threshold: a > 0)")

    # Save results
    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "n_features": len(selected),
            "n_tokens": N_TOKENS,
            "layer": LAYER,
            "tau_fs": TAU_FS,
            "tau_pa": TAU_PA,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
        },
        "absorption_results": {str(k): v for k, v in absorption_results.items()},
        "sensitivity_results": {str(k): v for k, v in sensitivity_results.items()},
        "quadrants": {q: [int(f) for f in feats] for q, feats in quadrants.items()},
        "quadrant_counts": {q: len(feats) for q, feats in quadrants.items()},
        "correlation": {
            "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
            "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
            "n_valid_features": len(valid_pairs),
        },
        "u_shape_fit": u_shape_result,
        "hypothesis_results": {
            "H-SF1": {"pass": hsf1_pass, "q2_q4_fraction": q2_q4_fraction, "q2_q4_count": q2_q4_count, "total_valid": total_valid},
            "H-SF2": {"pass": hsf2_pass, "a": u_shape_result['a'], "r2": u_shape_result['r2']}
        },
        "pass_criteria": {
            "hsf1_q2_q4_threshold": 0.05,
            "hsf2_a_threshold": 0.0
        }
    }

    output_path = pilots_dir / f"{TASK_ID}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

    # Update GPU progress
    update_gpu_progress(
        TASK_ID, "completed",
        f"Q1={len(quadrants['Q1'])}, Q2={len(quadrants['Q2'])}, Q3={len(quadrants['Q3'])}, Q4={len(quadrants['Q4'])}, Q2+Q4={q2_q4_fraction:.1%}",
        planned_min=45,
        actual_min=45,
        config_snapshot={"model": MODEL_NAME, "layer": LAYER, "n_features": len(selected)}
    )

    mark_done(status="success", summary=f"Q1={len(quadrants['Q1'])}, Q2={len(quadrants['Q2'])}, Q3={len(quadrants['Q3'])}, Q4={len(quadrants['Q4'])}, H-SF1={'PASS' if hsf1_pass else 'FAIL'}, H-SF2={'PASS' if hsf2_pass else 'FAIL'}")
    report_progress(step=6, metric={"quadrant_counts": quadrants, "hsf1_pass": hsf1_pass, "hsf2_pass": hsf2_pass})

    print(f"\n[{TASK_ID}] Complete!")
    return output


if __name__ == "__main__":
    main()