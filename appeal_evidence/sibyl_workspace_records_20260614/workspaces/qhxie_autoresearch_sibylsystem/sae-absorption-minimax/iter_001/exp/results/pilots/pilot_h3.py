#!/usr/bin/env python3
"""
Pilot H3: Steering sensitivity — absorbed vs. non-absorbed features.

Load GPT-2 layer 8 SAE. Select features by highest mean activation, compute UAS
for those, then identify 10 high-absorption and 10 low-absorption features.
For each feature:
  1. Apply steering intervention (add k * feature_direction to residual stream)
  2. Measure logit change at output
  3. Compare effect magnitude between groups.

Pass criteria: Low-absorption features show >30% larger steering effect than
high-absorption features.
"""

import json
import os
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from scipy.stats import spearmanr, skew

# ── Project paths ────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_H3_DIR = RESULTS_DIR / "pilots"
PILOT_H3_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_FEATURES_FOR_UAS = 300
N_HIGH = 10
N_LOW = 10
ALPHA_VALUES = [1, 3, 5, 10]
TEST_PROMPTS = [
    "The scientist conducted an experiment to test",
    "I love chocolate cake because it is",
    "The code compiles successfully and runs",
    "Paris is beautiful in spring when",
    "The economy improved after the government",
    "The doctor recommended that the patient",
    "Technology has changed the way we",
    "The book was written by a famous",
    "Music can express emotions that words",
    "The weather forecast predicts heavy",
]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_sae():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"[{datetime.now()}] Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"[{datetime.now()}] Loading SAE layer 8...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=DEVICE
    )
    print(f"[{datetime.now()}] SAE loaded. d_sae={sae.cfg.d_sae}, d_model={sae.cfg.d_in}")
    return model, sae


def get_top_activated_features(model, sae, n_tokens=500, batch_size=16):
    print(f"\n[{datetime.now()}] Identifying top-{N_FEATURES_FOR_UAS} activated features...")

    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED)

    activations_list = []
    tokens_collected = 0

    for batch in ds.iter(batch_size=batch_size):
        texts = batch["text"]
        for text in texts:
            try:
                tok = model.to_tokens([text], truncate=False)
                if tok.shape[1] < 2 or tok.shape[1] > 256:
                    continue
                _, cache = model.run_with_cache(
                    tok, names_filter="blocks.8.hook_resid_pre"
                )
                acts = cache["blocks.8.hook_resid_pre"][0]
                activations_list.append(acts.cpu())
                tokens_collected += acts.shape[0]
                if tokens_collected >= n_tokens:
                    break
            except Exception:
                pass
        if tokens_collected >= n_tokens:
            break

    all_acts = torch.cat(activations_list, dim=0)[:n_tokens].to(DEVICE)
    print(f"[{datetime.now()}] Collected {all_acts.shape[0]} token activations")

    with torch.no_grad():
        sae_out = sae.encode(all_acts)

    mean_acts = sae_out.mean(dim=0).cpu()
    top_indices = mean_acts.topk(N_FEATURES_FOR_UAS).indices.tolist()

    print(f"[{datetime.now()}] Top {N_FEATURES_FOR_UAS} features selected "
          f"(mean act range: {mean_acts[top_indices].min():.4f} - {mean_acts[top_indices].max():.4f})")

    return top_indices, sae_out.cpu()


def compute_uas_batch(sae, sae_acts_cpu, feature_indices):
    print(f"\n[{datetime.now()}] Computing UAS for {len(feature_indices)} features on GPU...")

    d_sae = sae.cfg.d_sae
    W_dec = sae.W_dec.detach().float().to(DEVICE)

    # Normalize decoder directions
    norms = W_dec.norm(dim=1, keepdim=True)
    W_dec_norm = W_dec / (norms + 1e-10)

    # Cosine similarity matrix on GPU: [d_sae, d_sae]
    print(f"[{datetime.now()}] Computing {d_sae}x{d_sae} cosine similarity matrix...")
    cos_sim = W_dec_norm @ W_dec_norm.T
    cos_sim_var = cos_sim.var(dim=1).cpu().numpy()
    del cos_sim
    gc.collect()
    torch.cuda.empty_cache()

    # freq_skewness
    sae_acts = sae_acts_cpu.numpy()
    freq_skewness = np.zeros(d_sae)
    for feat_idx in feature_indices:
        binary = (sae_acts[:, feat_idx] > 0).astype(float)
        if binary.std() > 0:
            freq_skewness[feat_idx] = skew(binary)
        else:
            freq_skewness[feat_idx] = 0.0

    # Normalize and compute UAS
    uas_scores = {}
    cos_vals = cos_sim_var[feature_indices]
    skew_vals = np.abs(freq_skewness[feature_indices])
    cos_max = max(cos_vals.max(), 1e-9)
    skew_max = max(skew_vals.max(), 1e-9)

    for feat_idx in feature_indices:
        norm_cos = min(cos_sim_var[feat_idx] / cos_max, 1.0)
        norm_skew = min(np.abs(freq_skewness[feat_idx]) / skew_max, 1.0)
        uas_scores[feat_idx] = float(norm_cos + norm_skew)

    del W_dec, W_dec_norm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{datetime.now()}] UAS computed.")
    return uas_scores


def select_features(uas_scores, n_high=N_HIGH, n_low=N_LOW):
    sorted_features = sorted(uas_scores.items(), key=lambda x: x[1])
    low_features = sorted_features[:n_low]
    high_features = sorted_features[-n_high:]

    print(f"\nSelected {len(low_features)} LOW-absorption features:")
    for idx, uas in low_features:
        print(f"  feat {idx}: UAS={uas:.4f}")

    print(f"\nSelected {len(high_features)} HIGH-absorption features:")
    for idx, uas in high_features:
        print(f"  feat {idx}: UAS={uas:.4f}")

    return low_features, high_features


def compute_steering_effect(model, prompt, feature_direction, alpha, layer=8):
    hook_name = f"blocks.{layer}.hook_resid_pre"

    def steering_hook(activation, hook):
        activation = activation + alpha * feature_direction.unsqueeze(0).unsqueeze(0)
        return activation

    try:
        tokens = model.to_tokens([prompt], truncate=False)
    except Exception:
        tokens = model.to_tokens(prompt)
    if tokens.shape[1] > 50:
        tokens = tokens[:, :50]

    with torch.no_grad():
        baseline_logits, _ = model.run_with_cache(tokens, return_type="logits")
        baseline_logits = baseline_logits.float()
        steered_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, steering_hook)],
            return_type="logits",
        )
        steered_logits = steered_logits.float()

    final_pos = tokens.shape[1] - 1
    logit_change = (steered_logits - baseline_logits)[0, final_pos]
    max_change = logit_change.abs().max().item()
    torch.cuda.empty_cache()
    return max_change


def run_steering_experiment(model, sae, features, alpha_values, task_id="pilot_h3"):
    results = []
    for feat_idx, uas in features:
        feat_dir = sae.W_dec[int(feat_idx)].to(DEVICE)
        for alpha in alpha_values:
            effect_per_prompt = []
            for prompt in TEST_PROMPTS:
                try:
                    max_change = compute_steering_effect(
                        model, prompt, feat_dir, alpha, layer=8
                    )
                    effect_per_prompt.append(max_change)
                except Exception as e:
                    print(f"  [WARN] Feature {feat_idx} alpha={alpha}: {e}")
                    effect_per_prompt.append(0.0)

            results.append({
                "feature_idx": int(feat_idx),
                "uas": float(uas),
                "alpha": alpha,
                "mean_effect": float(np.mean(effect_per_prompt)),
                "std_effect": float(np.std(effect_per_prompt)),
                "per_prompt_effects": [float(e) for e in effect_per_prompt],
            })
        print(f"  Feature {feat_idx} (UAS={uas:.4f}) steering done.")
    return results


def aggregate_results(high_results, low_results, alpha_values):
    agg = {"high_absorption": {}, "low_absorption": {}, "low_vs_high_ratio": {}}
    for alpha in alpha_values:
        high_effects = [r["mean_effect"] for r in high_results if r["alpha"] == alpha]
        low_effects = [r["mean_effect"] for r in low_results if r["alpha"] == alpha]
        high_mean = float(np.mean(high_effects))
        low_mean = float(np.mean(low_effects))
        agg["high_absorption"][alpha] = {"mean": high_mean, "std": float(np.std(high_effects)), "n": len(high_effects)}
        agg["low_absorption"][alpha] = {"mean": low_mean, "std": float(np.std(low_effects)), "n": len(low_effects)}
        agg["low_vs_high_ratio"][alpha] = float(low_mean / max(high_mean, 1e-9))
    return agg


def compute_correlation(high_results, low_results):
    all_results = high_results + low_results
    uas_vals = [r["uas"] for r in all_results]
    eff_vals = [r["mean_effect"] for r in all_results]
    r_val, p_val = spearmanr(uas_vals, eff_vals)
    return float(r_val), float(p_val)


def main():
    print(f"\n{'='*60}")
    print("PILOT H3: Steering Sensitivity — Absorbed vs Non-Absorbed")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Time: {datetime.now()}")
    set_seed(SEED)
    task_id = "pilot_h3"

    model, sae = load_model_and_sae()
    top_feature_indices, sae_acts = get_top_activated_features(model, sae, n_tokens=500)
    uas_scores = compute_uas_batch(sae, sae_acts, top_feature_indices)
    del sae_acts
    gc.collect()
    torch.cuda.empty_cache()

    low_features, high_features = select_features(uas_scores, N_LOW, N_HIGH)

    print(f"\n[{datetime.now()}] Steering on {len(high_features)} HIGH-absorption features...")
    high_results = run_steering_experiment(model, sae, high_features, ALPHA_VALUES, task_id)
    print(f"[{datetime.now()}] High-absorption done.")

    print(f"\n[{datetime.now()}] Steering on {len(low_features)} LOW-absorption features...")
    low_results = run_steering_experiment(model, sae, low_features, ALPHA_VALUES, task_id)
    print(f"[{datetime.now()}] Low-absorption done.")

    agg = aggregate_results(high_results, low_results, ALPHA_VALUES)
    spearman_r, spearman_p = compute_correlation(high_results, low_results)

    primary_alpha = 5
    ratio = agg["low_vs_high_ratio"].get(primary_alpha, 1.0)
    low_mean = agg["low_absorption"][primary_alpha]["mean"]
    high_mean = agg["high_absorption"][primary_alpha]["mean"]
    improvement_pct = (low_mean - high_mean) / max(high_mean, 1e-9) * 100
    pass_threshold = 1.30
    pass_h3 = ratio >= pass_threshold

    print(f"\n{'='*60}")
    print(f"PASS CRITERIA (alpha={primary_alpha})")
    print(f"  Low-absorption mean effect:  {low_mean:.4f}")
    print(f"  High-absorption mean effect: {high_mean:.4f}")
    print(f"  Ratio (low/high): {ratio:.4f} >= {pass_threshold}? {pass_h3}")
    print(f"  Improvement: {improvement_pct:.1f}%")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")
    print(f"  H3 PASS: {pass_h3}")
    print(f"{'='*60}")

    # PID
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    # Progress
    (RESULTS_DIR / f"{task_id}_PROGRESS.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "timestamp": datetime.now().isoformat()
    }))

    # Results
    output = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_features_for_uas": N_FEATURES_FOR_UAS,
            "n_high_absorption": N_HIGH,
            "n_low_absorption": N_LOW,
            "alpha_values": ALPHA_VALUES,
            "n_test_prompts": len(TEST_PROMPTS),
            "seed": SEED,
            "device": DEVICE,
        },
        "uas_scores": uas_scores,
        "high_absorption_features": [{"feature_idx": int(f), "uas": float(u)} for f, u in high_features],
        "low_absorption_features": [{"feature_idx": int(f), "uas": float(u)} for f, u in low_features],
        "high_absorption_results": high_results,
        "low_absorption_results": low_results,
        "aggregated": agg,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pass_criteria": {
            "threshold_ratio": pass_threshold,
            "alpha_used": primary_alpha,
            "achieved_ratio": ratio,
            "improvement_pct": improvement_pct,
            "pass_h3": pass_h3,
        },
    }

    out_path = PILOT_H3_DIR / f"{task_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    # DONE marker
    status = "success" if pass_h3 else "partial"
    summary = (f"H3 {'PASS' if pass_h3 else 'PARTIAL'}: "
               f"Low-abs={low_mean:.4f}, High-abs={high_mean:.4f}, "
               f"ratio={ratio:.3f} (threshold 1.30). "
               f"Spearman r={spearman_r:.3f}.")
    marker = Path(RESULTS_DIR) / f"{task_id}_DONE"
    marker.write_text(json.dumps({"task_id": task_id, "status": status,
                                  "summary": summary, "timestamp": datetime.now().isoformat()}))
    if pid_file.exists():
        pid_file.unlink()

    print(f"\n[{datetime.now()}] pilot_h3 complete. Status: {status}")
    return output


if __name__ == "__main__":
    main()
