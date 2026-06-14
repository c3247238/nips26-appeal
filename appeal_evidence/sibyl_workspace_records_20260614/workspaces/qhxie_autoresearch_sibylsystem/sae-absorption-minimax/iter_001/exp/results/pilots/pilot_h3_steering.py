#!/usr/bin/env python3
"""
Pilot H3: Steering experiment (reuse UAS scores from previous run).
Only re-runs the steering experiment — UAS computation already done.
"""

import json
import os
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from scipy.stats import spearmanr

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_H3_DIR = RESULTS_DIR / "pilots"
PILOT_H3_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
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


def load_previous_results():
    path = PILOT_H3_DIR / "pilot_h3.json"
    with open(path) as f:
        data = json.load(f)
    uas_scores = {int(k): v for k, v in data["uas_scores"].items()}
    high_features = [(f["feature_idx"], f["uas"]) for f in data["high_absorption_features"]]
    low_features = [(f["feature_idx"], f["uas"]) for f in data["low_absorption_features"]]
    return uas_scores, high_features, low_features


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
    print(f"[{datetime.now()}] SAE loaded. d_sae={sae.cfg.d_sae}")
    return model, sae


def compute_steering_effect(model, prompt, feature_direction, alpha, layer=8):
    hook_name = f"blocks.{layer}.hook_resid_pre"

    def steering_hook(value, hook):
        return value + alpha * feature_direction.unsqueeze(0).unsqueeze(0)

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
    torch.cuda.empty_cache()
    return logit_change.abs().max().item()


def run_steering_experiment(model, sae, features, alpha_values, task_id):
    results = []
    for feat_idx, uas in features:
        feat_dir = sae.W_dec[int(feat_idx)].to(DEVICE)
        for alpha in alpha_values:
            effect_per_prompt = []
            for prompt in TEST_PROMPTS:
                try:
                    effect = compute_steering_effect(model, prompt, feat_dir, alpha, layer=8)
                    effect_per_prompt.append(effect)
                except Exception as e:
                    print(f"  [WARN] feat={feat_idx} alpha={alpha}: {e}")
                    effect_per_prompt.append(0.0)
            results.append({
                "feature_idx": int(feat_idx),
                "uas": float(uas),
                "alpha": alpha,
                "mean_effect": float(np.mean(effect_per_prompt)),
                "std_effect": float(np.std(effect_per_prompt)),
                "per_prompt_effects": [float(e) for e in effect_per_prompt],
            })
        print(f"  Feature {feat_idx} (UAS={uas:.4f}) done.")
    return results


def aggregate_results(high_results, low_results, alpha_values):
    agg = {"high_absorption": {}, "low_absorption": {}, "low_vs_high_ratio": {}}
    for alpha in alpha_values:
        high_effs = [r["mean_effect"] for r in high_results if r["alpha"] == alpha]
        low_effs = [r["mean_effect"] for r in low_results if r["alpha"] == alpha]
        high_mean = float(np.mean(high_effs))
        low_mean = float(np.mean(low_effs))
        agg["high_absorption"][alpha] = {"mean": high_mean, "std": float(np.std(high_effs)), "n": len(high_effs)}
        agg["low_absorption"][alpha] = {"mean": low_mean, "std": float(np.std(low_effs)), "n": len(low_effs)}
        agg["low_vs_high_ratio"][alpha] = float(low_mean / max(high_mean, 1e-9))
    return agg


def main():
    print(f"\n{'='*60}")
    print("PILOT H3: Steering Sensitivity — Absorbed vs Non-Absorbed")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Time: {datetime.now()}")
    set_seed(SEED)
    task_id = "pilot_h3"

    uas_scores, high_features, low_features = load_previous_results()
    print(f"Loaded {len(uas_scores)} UAS scores")
    print(f"High features: {high_features}")
    print(f"Low features: {low_features}")

    model, sae = load_model_and_sae()

    print(f"\n[{datetime.now()}] Steering on {len(high_features)} HIGH-absorption features...")
    high_results = run_steering_experiment(model, sae, high_features, ALPHA_VALUES, task_id)
    print(f"[{datetime.now()}] High done.")

    print(f"\n[{datetime.now()}] Steering on {len(low_features)} LOW-absorption features...")
    low_results = run_steering_experiment(model, sae, low_features, ALPHA_VALUES, task_id)
    print(f"[{datetime.now()}] Low done.")

    agg = aggregate_results(high_results, low_results, ALPHA_VALUES)
    all_results = high_results + low_results
    uas_vals = [r["uas"] for r in all_results]
    eff_vals = [r["mean_effect"] for r in all_results]
    r_val, p_val = spearmanr(uas_vals, eff_vals)
    spearman_r = float(r_val)
    spearman_p = float(p_val)

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

    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    (RESULTS_DIR / f"{task_id}_PROGRESS.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "timestamp": datetime.now().isoformat()
    }))

    output = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_high_absorption": N_HIGH,
            "n_low_absorption": N_LOW,
            "alpha_values": ALPHA_VALUES,
            "n_test_prompts": len(TEST_PROMPTS),
            "seed": SEED,
            "device": DEVICE,
            "reuse_uas_from": str(PILOT_H3_DIR / "pilot_h3.json"),
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
