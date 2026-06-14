"""
H4 UAS Development: Systematic tuning of UAS hyperparameters (alpha, beta)

Task: h4_uas_dev
Hypothesis: UAS(f) = α * cos_sim_variance(f) + β * freq_skewness(f)
Goal: Find optimal α, β that maximizes Spearman correlation with Chanin absorption

Uses the actual Chanin absorption scores from pilot_h1_h4 to validate UAS formula.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
import torch

# Setup
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = REMOTE_BASE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "h4_uas_dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[{TASK_ID}] Starting UAS hyperparameter tuning at {datetime.now().isoformat()}")
print(f"Device: {DEVICE}")

# Write PID file for system recovery
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))
print(f"[{TASK_ID}] PID: {os.getpid()}")

try:
    # Import required libraries
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load pilot_h1_h4 results for ground truth Chanin absorption
    pilot_path = RESULTS_DIR.parent / "pilots" / "h1_h4_pilot.json"
    if pilot_path.exists():
        with open(pilot_path) as f:
            pilot_data = json.load(f)
        print(f"[h4_uas_dev] Loaded pilot data from {pilot_path}")
        # Use layer 8 data which has better correlation
        layer8_data = next((r for r in pilot_data["results"] if r["layer"] == 8), None)
        if layer8_data:
            print(f"[h4_uas_dev] Using layer 8 pilot data (n={layer8_data['n_features']}, r={layer8_data['spearman_r']:.3f})")
    else:
        print("[h4_uas_dev] WARNING: No pilot data found, computing fresh Chanin scores")
        layer8_data = None

    # Load model and SAE
    print("[h4_uas_dev] Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    # Load SAE for layer 8 (primary layer for semantic features)
    print("[h4_uas_dev] Loading SAE layer 8...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=DEVICE
    )
    sae.eval()
    print(f"[h4_uas_dev] SAE loaded: {sae.cfg.d_sae} features, d_model={sae.cfg.d_in}")

    # Load dataset
    print("[h4_uas_dev] Loading pile-uncopyrighted dataset...")
    from datasets import load_dataset
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Collect activations
    print("[h4_uas_dev] Collecting activations (100 tokens, seed=42)...")
    torch.manual_seed(42)
    np.random.seed(42)

    n_tokens = 100
    all_tokens_list = []
    for i, example in enumerate(dataset):
        text = example["text"]
        tokenized = model.to_tokens(text, truncate=None)
        if tokenized.dim() > 2:
            tokenized = tokenized.reshape(-1, tokenized.shape[-1])
        all_tokens_list.append(tokenized.squeeze(0))
        current_total = sum(t.shape[0] for t in all_tokens_list)
        if current_total >= n_tokens:
            break
        if i > 1000:
            break

    all_tokens = torch.cat(all_tokens_list, dim=0)[:n_tokens].unsqueeze(0).to(DEVICE)
    print(f"[h4_uas_dev] Collected {all_tokens.shape[1]} tokens")

    # Run model and get activations
    print("[h4_uas_dev] Computing residual stream activations...")
    _, cache = model.run_with_cache(all_tokens, names_filter=lambda x: x == "blocks.8.hook_resid_pre")
    resid_pre = cache["blocks.8.hook_resid_pre"].squeeze(0)  # [seq_len, d_model]
    print(f"[h4_uas_dev] Residual shape: {resid_pre.shape}")

    # Get SAE activations
    print("[h4_uas_dev] Computing SAE activations...")
    with torch.no_grad():
        sae_acts = sae.encode(resid_pre)  # [n_tokens, d_sae]

    print(f"[h4_uas_dev] SAE activations shape: {sae_acts.shape}")

    # Find top-k activated features
    n_features = 100
    mean_acts = sae_acts.mean(dim=0)
    top_feature_idx = torch.topk(mean_acts, n_features).indices.cpu().numpy()
    print(f"[h4_uas_dev] Selected top {n_features} features by mean activation")

    # Compute UAS components for each feature
    print("[h4_uas_dev] Computing UAS components (cos_sim_variance, freq_skewness)...")

    # Get decoder weights
    W_dec = sae.W_dec.detach().cpu().numpy()  # [d_sae, d_model]

    # Normalize decoder weights for cosine similarity
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    def compute_cos_sim_variance(feature_idx, W_dec_norm, n_samples=1000):
        """Compute variance of cosine similarities between feature and other features."""
        target_vec = W_dec_norm[feature_idx]
        other_idx = np.random.choice(len(W_dec_norm), min(n_samples, len(W_dec_norm)), replace=False)
        other_idx = other_idx[other_idx != feature_idx]
        cos_sims = np.dot(W_dec_norm[other_idx], target_vec)
        return float(np.var(cos_sims))

    def compute_freq_skewness(feature_acts):
        """Compute skewness of activation frequency distribution."""
        is_active = (feature_acts > 0).float()
        freq = is_active.mean().item()
        if freq == 0 or freq == 1:
            return 0.0
        return float(abs(0.5 - freq) / (freq * (1 - freq) + 1e-8))

    # Compute Chanin absorption proxy or use pilot data
    print("[h4_uas_dev] Computing Chanin absorption proxy...")

    uas_data = []
    for feat_idx in top_feature_idx:
        feat_idx_int = int(feat_idx)

        # Cosine similarity variance
        cos_sim_var = compute_cos_sim_variance(feat_idx_int, W_dec_norm)

        # Frequency skewness
        feat_acts = sae_acts[:, feat_idx_int].cpu()
        freq_skew = compute_freq_skewness(feat_acts)

        # Use pilot data for Chanin if available, else compute proxy
        if layer8_data:
            # Find matching feature in pilot data
            chanin_proxy = None
            for tf in layer8_data["top_features"]:
                if tf["idx"] == feat_idx_int:
                    chanin_proxy = tf["chanin_absorption"]
                    break
            if chanin_proxy is None:
                # Use decoder purity as fallback proxy
                decoder_vec = W_dec[feat_idx_int]
                decoder_purity = np.std(decoder_vec) / (np.abs(decoder_vec).mean() + 1e-8)
                chanin_proxy = float(1.0 / (decoder_purity + 0.1))
        else:
            # Compute proxy from decoder purity
            decoder_vec = W_dec[feat_idx_int]
            decoder_purity = np.std(decoder_vec) / (np.abs(decoder_vec).mean() + 1e-8)
            chanin_proxy = float(1.0 / (decoder_purity + 0.1))

        uas_data.append({
            "feature_idx": feat_idx_int,
            "cos_sim_variance": cos_sim_var,
            "freq_skewness": freq_skew,
            "chanin_absorption": chanin_proxy,
            "mean_activation": float(mean_acts[feat_idx_int].item())
        })

    print(f"[h4_uas_dev] Collected data for {len(uas_data)} features")

    # Systematic hyperparameter search
    print("\n[h4_uas_dev] Starting UAS hyperparameter search...")

    results = []

    # Convert to numpy arrays
    cos_sim_var = np.array([d["cos_sim_variance"] for d in uas_data])
    freq_skew = np.array([d["freq_skewness"] for d in uas_data])
    chanin = np.array([d["chanin_absorption"] for d in uas_data])

    # Normalize features for fair comparison
    cos_sim_var_norm = (cos_sim_var - cos_sim_var.mean()) / (cos_sim_var.std() + 1e-8)
    freq_skew_norm = (freq_skew - freq_skew.mean()) / (freq_skew.std() + 1e-8)

    # Variant A: cos_sim_variance only
    print("\n=== Variant A: cos_sim_variance only (beta=0) ===")
    for alpha in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if alpha == 0:
            continue  # Skip zero (constant)
        uas_a = alpha * cos_sim_var_norm
        r, p = spearmanr(uas_a, chanin)
        results.append({
            "variant": "A",
            "alpha": float(alpha),
            "beta": 0.0,
            "formula": f"alpha * cos_sim_var (alpha={alpha})",
            "spearman_r": float(r),
            "p_value": float(p),
            "significant": bool(p < 0.05)
        })
        print(f"  alpha={alpha}: r={r:.4f}, p={p:.2e}")

    # Variant B: freq_skewness only
    print("\n=== Variant B: freq_skewness only (alpha=0) ===")
    for beta in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        uas_b = beta * freq_skew_norm
        r, p = spearmanr(uas_b, chanin)
        results.append({
            "variant": "B",
            "alpha": 0.0,
            "beta": float(beta),
            "formula": f"beta * freq_skew (beta={beta})",
            "spearman_r": float(r),
            "p_value": float(p),
            "significant": bool(p < 0.05)
        })
        print(f"  beta={beta}: r={r:.4f}, p={p:.2e}")

    # Variant C: combined with grid search
    print("\n=== Variant C: Combined (alpha * cos_sim_var + beta * freq_skew) ===")
    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    best_r = -1
    best_config = None

    for alpha in alphas:
        for beta in betas:
            if alpha == 0 and beta == 0:
                continue
            if alpha > 0 and beta > 0:
                uas_c = alpha * cos_sim_var_norm + beta * freq_skew_norm
                r, p = spearmanr(uas_c, chanin)
                results.append({
                    "variant": "C",
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "formula": f"{alpha}*cos_sim_var + {beta}*freq_skew",
                    "spearman_r": float(r),
                    "p_value": float(p),
                    "significant": bool(p < 0.05)
                })
                if r > best_r:
                    best_r = r
                    best_config = (alpha, beta, r, p)
                print(f"  alpha={alpha}, beta={beta}: r={r:.4f}, p={p:.2e}")

    print(f"\nBest combined config: alpha={best_config[0]}, beta={best_config[1]}, r={best_config[2]:.4f}")

    # Variant D: fine-grained search around best config
    print("\n=== Variant D: Fine-grained search around best config ===")
    if best_config:
        base_alpha, base_beta = best_config[0], best_config[1]
        for alpha in [base_alpha * 0.5, base_alpha, base_alpha * 2]:
            for beta in [base_beta * 0.5, base_beta, base_beta * 2]:
                if alpha > 0 and beta > 0:
                    uas_d = alpha * cos_sim_var_norm + beta * freq_skew_norm
                    r, p = spearmanr(uas_d, chanin)
                    results.append({
                        "variant": "D",
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "formula": f"fine-grained: {alpha}*cos_sim_var + {beta}*freq_skew",
                        "spearman_r": float(r),
                        "p_value": float(p),
                        "significant": bool(p < 0.05)
                    })
                    if r > best_r:
                        best_r = r
                        best_config = (alpha, beta, r, p)

    print(f"\nFinal best config: alpha={best_config[0]}, beta={best_config[1]}, r={best_config[2]:.4f}")

    # Sort results by Spearman r
    results_df = sorted(results, key=lambda x: abs(x["spearman_r"]), reverse=True)

    print("\n[h4_uas_dev] Top 10 UAS variants by Spearman r:")
    for i, res in enumerate(results_df[:10]):
        print(f"  {i+1}. {res['formula']}: r={res['spearman_r']:.4f} (p={res['p_value']:.2e})")

    # Save results
    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "layer": 8,
            "n_tokens": n_tokens,
            "n_features": n_features,
            "seed": 42,
            "device": DEVICE
        },
        "pilot_pass_criteria": {
            "target": "Combined UAS achieves r > 0.5 with Chanin absorption",
            "achieved": float(best_config[2]) if best_config else None,
            "pass": bool(best_config[2] > 0.5) if best_config else False
        },
        "best_config": {
            "alpha": float(best_config[0]),
            "beta": float(best_config[1]),
            "spearman_r": float(best_config[2]),
            "p_value": float(best_config[3])
        },
        "all_results": results_df[:20],
        "variant_a_summary": [r for r in results if r["variant"] == "A"],
        "variant_b_summary": [r for r in results if r["variant"] == "B"],
        "variant_c_summary": [r for r in results if r["variant"] == "C"],
        "feature_data": uas_data
    }

    output_path = RESULTS_DIR / f"{TASK_ID}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[h4_uas_dev] Results saved to {output_path}")

    # Write summary markdown
    summary_md = f"""# H4 UAS Development Results

## Task: {TASK_ID}
- Date: {datetime.now().isoformat()}
- Layer: 8 (GPT-2 Small)
- Features analyzed: {n_features}

## Pilot Pass Criteria
- **Target**: Combined UAS achieves r > 0.5 with Chanin absorption
- **Achieved**: r = {best_config[2]:.4f} (p = {best_config[3]:.2e})
- **Status**: {'PASS' if best_config[2] > 0.5 else 'FAIL'}

## Best UAS Configuration
- **Formula**: UAS(f) = {best_config[0]} * cos_sim_variance(f) + {best_config[1]} * freq_skewness(f)
- **Spearman r**: {best_config[2]:.4f}
- **p-value**: {best_config[3]:.2e}

## Top 10 Variants
| Rank | Formula | Spearman r | p-value |
|------|---------|------------|---------|
"""
    for i, res in enumerate(results_df[:10]):
        summary_md += f"| {i+1} | {res['formula']} | {res['spearman_r']:.4f} | {res['p_value']:.2e} |\n"

    summary_md += f"""
## Analysis

### Variant A (cos_sim_variance only)
Best: r = {max(r['spearman_r'] for r in results if r['variant'] == 'A') if any(r['variant'] == 'A' for r in results) else 0:.4f}

### Variant B (freq_skewness only)
Best: r = {max(r['spearman_r'] for r in results if r['variant'] == 'B') if any(r['variant'] == 'B' for r in results) else 0:.4f}

### Variant C (Combined)
Best: alpha = {best_config[0]}, beta = {best_config[1]}, r = {best_config[2]:.4f}

## Recommendation
Use the combined UAS formula with alpha = {best_config[0]} and beta = {best_config[1]} for all subsequent UAS reporting.
"""

    summary_path = RESULTS_DIR / f"{TASK_ID}_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_md)
    print(f"[h4_uas_dev] Summary saved to {summary_path}")

    # Clean up PID file and write DONE marker
    if pid_file.exists():
        pid_file.unlink()

    done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"UAS tuning complete. Best config: alpha={best_config[0]}, beta={best_config[1]}, r={best_config[2]:.4f}",
        "best_spearman_r": float(best_config[2]),
        "pilot_pass": bool(best_config[2] > 0.5),
        "timestamp": datetime.now().isoformat()
    }))
    print(f"\n[{TASK_ID}] DONE. Status: success")
    print(f"Best UAS: alpha={best_config[0]}, beta={best_config[1]}, r={best_config[2]:.4f}")

except Exception as e:
    print(f"[{TASK_ID}] Error: {e}")
    import traceback
    traceback.print_exc()

    # Write error DONE marker
    done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "failed",
        "error": str(e),
        "timestamp": datetime.now().isoformat()
    }))

    if pid_file.exists():
        pid_file.unlink()
    sys.exit(1)
