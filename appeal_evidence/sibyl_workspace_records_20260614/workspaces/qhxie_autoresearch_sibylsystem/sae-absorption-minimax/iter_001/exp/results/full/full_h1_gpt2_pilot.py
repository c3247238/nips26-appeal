"""
Full H1 (PILOT): GPT-2 absorption atlas across 5 layers.
Tests H1: absorption severity peaks in middle layers (layer 6-8).

Pilot mode: 200 tokens, 200 features, 3 layers (2, 6, 10) + reuse pilot data from layers 4, 8.
Pass criteria: non-monotonic pattern AND peak at layer 6-8.
"""

import json
import os
import sys
import gc
import warnings
import numpy as np
import torch
from scipy.stats import spearmanr, skew
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
REMOTE_BASE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = REMOTE_BASE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "full_h1_gpt2"
DEVICE = "cuda"
N_TOKENS = 200       # pilot budget per layer
N_FEATURES = 200     # features per layer
SEED = 42

# UAS hyperparameters (calibrated from pilot_h1_h4: alpha=1.0, beta=0.5)
UAS_ALPHA = 1.0
UAS_BETA = 0.5

# New layers to test in this pilot (layers 2, 6, 10 to complement pilot layers 4, 8)
NEW_LAYERS = [2, 6, 10]
ALL_LAYERS = [2, 4, 6, 8, 10]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_uas(sae, feature_acts, topk_indices):
    """Compute Unsupervised Absorption Score (UAS)."""
    W_dec = sae.W_dec.data.cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    uas_scores = {}
    cos_sim_variances = {}
    freq_skewness_dict = {}

    for feat_idx in topk_indices:
        other_indices = [i for i in topk_indices if i != feat_idx]
        if len(other_indices) < 2:
            cos_sim_variances[feat_idx] = 0.0
            freq_skewness_dict[feat_idx] = 0.0
            uas_scores[feat_idx] = 0.0
            continue

        cos_sims = np.dot(W_dec_norm[feat_idx], W_dec_norm[other_indices].T)
        cos_sim_variances[feat_idx] = float(np.var(cos_sims))

        feat_act_values = feature_acts[0, :, feat_idx].cpu().numpy()
        non_zero_acts = feat_act_values[feat_act_values > 0]
        if len(non_zero_acts) > 2:
            freq_skewness_dict[feat_idx] = float(abs(skew(non_zero_acts)))
        else:
            freq_skewness_dict[feat_idx] = 0.0

        uas_scores[feat_idx] = (
            UAS_ALPHA * cos_sim_variances[feat_idx] + UAS_BETA * freq_skewness_dict[feat_idx]
        )

    return uas_scores, cos_sim_variances, freq_skewness_dict


def compute_gini_absorption(feature_acts, topk_indices):
    """Absorption proxy via Gini concentration (1 - Gini_coefficient)."""
    absorption_scores = {}
    for feat_idx in topk_indices:
        acts = np.abs(feature_acts[0, :, feat_idx].cpu().numpy())
        if np.sum(acts) == 0:
            absorption_scores[feat_idx] = 0.0
            continue
        sorted_acts = np.sort(acts)
        n = len(sorted_acts)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_acts)) / (n * np.sum(sorted_acts)) - (n + 1) / n
        absorption_scores[feat_idx] = 1.0 - max(0.0, min(1.0, gini))
    return absorption_scores


def collect_tokens(model, n_tokens: int, seed: int = 42):
    """Collect tokens from pile-uncopyrighted dataset."""
    from datasets import load_dataset
    print(f"  Collecting {n_tokens} tokens (seed={seed})...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed)
    tokens_list = []
    total_tokens = 0
    for example in dataset:
        text = example["text"]
        tokens = model.to_tokens(text, truncate=True)
        tokens_list.append(tokens)
        total_tokens += tokens.shape[1]
        if total_tokens >= n_tokens:
            break
    all_tokens = torch.cat(tokens_list, dim=0)[:, :n_tokens].to(model.cfg.device)
    print(f"  Collected {all_tokens.shape[0]} sequences, {all_tokens.shape[1]} tokens")
    return all_tokens


def run_layer_experiment(layer: int, n_tokens: int, n_features: int, reuse_pilot: bool = False):
    """Run absorption experiment for a single layer."""
    print(f"\n{'='*60}")
    print(f"Layer {layer} (reuse_pilot={reuse_pilot})")
    print(f"{'='*60}")

    set_seed(SEED)
    torch.cuda.empty_cache()
    gc.collect()

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load model
    print(f"  Loading gpt2-small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    model.eval()

    # Load SAE
    print(f"  Loading SAE: gpt2-small-res-jb layer {layer}")
    sae_id = f"blocks.{layer}.hook_resid_pre"
    act_sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=DEVICE,
    )
    act_sae.eval()
    print(f"  SAE: d_sae={act_sae.cfg.d_sae}, d_model={act_sae.cfg.d_in}")

    # Collect tokens
    tokens = collect_tokens(model, n_tokens, SEED)

    # Get residual activations
    print(f"  Getting residual stream activations at layer {layer}...")
    _, cache = model.run_with_cache(
        tokens,
        names_filter=[f"blocks.{layer}.hook_resid_pre"],
        return_type="loss",
    )
    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]

    # Encode through SAE
    print("  Encoding through SAE...")
    with torch.no_grad():
        feature_acts = act_sae.encode(resid_pre)
    sparsity = (feature_acts > 0).float().mean().item()
    print(f"  Feature acts: {feature_acts.shape}, sparsity={sparsity:.4f}")

    # Select top-k features
    total_act = feature_acts[0].sum(dim=0).cpu().numpy()
    topk_indices = np.argsort(total_act)[-n_features:][::-1]
    print(f"  Top {n_features} features: {list(topk_indices[:5])}...")

    # Compute metrics
    print("  Computing UAS scores...")
    uas_scores, cos_sim_var, freq_skew = compute_uas(act_sae, feature_acts, topk_indices)

    print("  Computing Gini absorption...")
    absorption_scores = compute_gini_absorption(feature_acts, topk_indices)

    # Reconstruction loss
    with torch.no_grad():
        reconstructed = act_sae.decode(feature_acts)
        recon_mse = torch.nn.functional.mse_loss(resid_pre, reconstructed).item()
    print(f"  Reconstruction MSE: {recon_mse:.4f}")

    # Summary stats per layer
    uas_vals = np.array([uas_scores.get(i, 0.0) for i in topk_indices])
    absorption_vals = np.array([absorption_scores.get(i, 0.0) for i in topk_indices])
    mean_uas = float(np.mean(uas_vals))
    mean_absorption = float(np.mean(absorption_vals))

    print(f"  Mean UAS: {mean_uas:.4f}")
    print(f"  Mean Gini absorption: {mean_absorption:.4f}")

    # Top 10 features
    top_features = [
        {
            "feature_idx": int(idx),
            "uas": float(uas_scores.get(idx, 0.0)),
            "chanin": float(absorption_scores.get(idx, 0.0)),
            "cos_sim_var": float(cos_sim_var.get(idx, 0.0)),
            "freq_skew": float(freq_skew.get(idx, 0.0)),
        }
        for idx in topk_indices[:10]
    ]

    del model, act_sae, feature_acts, resid_pre, tokens, cache
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "layer": layer,
        "n_features": n_features,
        "n_tokens": n_tokens,
        "sparsity": sparsity,
        "reconstruction_mse": recon_mse,
        "mean_uas": mean_uas,
        "mean_gini_absorption": mean_absorption,
        "std_gini_absorption": float(np.std(absorption_vals)),
        "top_features": top_features,
        "uas_alpha": UAS_ALPHA,
        "uas_beta": UAS_BETA,
        "source": "new",
    }


def load_pilot_layer_results(layer: int):
    """Load existing pilot data for a layer from pilot_h1_h4 results."""
    pilot_file = WORKSPACE / "exp/results/pilots/h1_h4_pilot.json"
    if not pilot_file.exists():
        return None
    with open(pilot_file) as f:
        data = json.load(f)
    for r in data.get("results", []):
        if r.get("layer") == layer:
            return {
                "layer": layer,
                "n_features": r.get("n_features", 100),
                "n_tokens": r.get("n_tokens", 100),
                "sparsity": r.get("sparsity"),
                "reconstruction_mse": r.get("reconstruction_ce_loss"),
                "mean_uas": r.get("mean_uas"),
                "mean_gini_absorption": r.get("mean_chanin_absorption"),
                "std_gini_absorption": 0.0,
                "top_features": r.get("top_features", []),
                "uas_alpha": r.get("uas_alpha", UAS_ALPHA),
                "uas_beta": r.get("uas_beta", UAS_BETA),
                "source": "pilot_h1_h4",
            }
    return None


def check_h1_pattern(results_with_layer):
    """
    Check if absorption shows non-monotonic pattern with peak at layer 6-8.
    Returns pass/fail and analysis details.
    """
    sorted_results = sorted(results_with_layer, key=lambda x: x["layer"])
    layers = [r["layer"] for r in sorted_results]
    absorption_vals = [r["mean_gini_absorption"] for r in sorted_results]

    # Find peak layer
    peak_idx = int(np.argmax(absorption_vals))
    peak_layer = layers[peak_idx]
    peak_val = absorption_vals[peak_idx]

    # Non-monotonic check: not strictly increasing or strictly decreasing
    is_increasing = all(absorption_vals[i] <= absorption_vals[i+1] for i in range(len(absorption_vals)-1))
    is_decreasing = all(absorption_vals[i] >= absorption_vals[i+1] for i in range(len(absorption_vals)-1))
    is_non_monotonic = not is_increasing and not is_decreasing

    # Peak at layer 6-8 check
    peak_at_68 = 6 <= peak_layer <= 8

    # Early > Late check (middle layers have higher absorption)
    if len(layers) >= 3:
        early_layers = [l for l in layers if l <= 4]
        mid_layers = [l for l in layers if 5 <= l <= 8]
        late_layers = [l for l in layers if l >= 9]
        early_vals = [absorption_vals[layers.index(l)] for l in early_layers if l in layers]
        mid_vals = [absorption_vals[layers.index(l)] for l in mid_layers if l in layers]
        late_vals = [absorption_vals[layers.index(l)] for l in late_layers if l in layers]
        early_mean = float(np.mean(early_vals)) if early_vals else 0.0
        mid_mean = float(np.mean(mid_vals)) if mid_vals else 0.0
        late_mean = float(np.mean(late_vals)) if late_vals else 0.0
    else:
        early_mean = mid_mean = late_mean = 0.0

    return {
        "layers": layers,
        "absorption_vals": absorption_vals,
        "peak_layer": peak_layer,
        "peak_val": peak_val,
        "is_non_monotonic": is_non_monotonic,
        "peak_at_68": peak_at_68,
        "early_mean": early_mean,
        "mid_mean": mid_mean,
        "late_mean": late_mean,
        "early_vs_mid": early_mean < mid_mean if early_mean > 0 and mid_mean > 0 else None,
        "mid_vs_late": mid_mean > late_mean if mid_mean > 0 and late_mean > 0 else None,
    }


def make_ascii_plot(results_with_layer):
    """Generate ASCII plot of absorption vs layer."""
    sorted_results = sorted(results_with_layer, key=lambda x: x["layer"])
    layers = [r["layer"] for r in sorted_results]
    vals = [r["mean_gini_absorption"] for r in sorted_results]

    if not vals:
        return ""

    max_val = max(vals)
    min_val = min(vals)
    scale = max_val - min_val if max_val > min_val else 1.0

    lines = []
    lines.append("\n  Absorption vs. Layer (ASCII plot)")
    lines.append("  " + "-" * 40)
    for r in sorted_results:
        bar_len = int(30 * (r["mean_gini_absorption"] - min_val) / scale) if scale > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  L{r['layer']:2d} | {bar:<30} {r['mean_gini_absorption']:.4f} [pilot={r['source']}]")
    lines.append("  " + "-" * 40)
    return "\n".join(lines)


def main():
    print(f"\n{'#'*60}")
    print(f"Full H1 (PILOT): GPT-2 Absorption Atlas")
    print(f"{'#'*60}")
    print(f"Task: {TASK_ID}")
    print(f"GPU: {DEVICE}")
    print(f"N tokens per layer: {N_TOKENS}")
    print(f"N features per layer: {N_FEATURES}")
    print(f"UAS: alpha={UAS_ALPHA}, beta={UAS_BETA}")
    print(f"New layers: {NEW_LAYERS}")
    print(f"Existing pilot layers: [4, 8] (from pilot_h1_h4)")
    print(f"Started: {datetime.now().isoformat()}")

    # Write PID
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # Build combined results: pilot layers + new layers
    all_results = {}  # layer -> result dict

    # Load pilot data for layers 4 and 8
    for layer in [4, 8]:
        p = load_pilot_layer_results(layer)
        if p:
            all_results[layer] = p
            print(f"\nLoaded pilot data for layer {layer}: absorption={p['mean_gini_absorption']:.4f} (n={p['n_features']})")

    # Run new layers
    for layer in NEW_LAYERS:
        try:
            result = run_layer_experiment(layer, N_TOKENS, N_FEATURES)
            all_results[layer] = result
        except Exception as e:
            print(f"ERROR on layer {layer}: {e}")
            import traceback
            traceback.print_exc()

    # Sort by layer
    sorted_results = sorted(all_results.values(), key=lambda x: x["layer"])

    # H1 pattern analysis
    pattern = check_h1_pattern(sorted_results)
    print(f"\n{'='*60}")
    print(f"H1 PATTERN ANALYSIS")
    print(f"{'='*60}")
    print(f"  Layers tested: {pattern['layers']}")
    print(f"  Absorption values: {[f'{v:.4f}' for v in pattern['absorption_vals']]}")
    print(f"  Peak layer: {pattern['peak_layer']} (value={pattern['peak_val']:.4f})")
    print(f"  Non-monotonic: {pattern['is_non_monotonic']}")
    print(f"  Peak at layer 6-8: {pattern['peak_at_68']}")
    print(f"  Early mean: {pattern['early_mean']:.4f}")
    print(f"  Mid mean: {pattern['mid_mean']:.4f}")
    print(f"  Late mean: {pattern['late_mean']:.4f}")
    print(f"  Early < Mid: {pattern['early_vs_mid']}")
    print(f"  Mid > Late: {pattern['mid_vs_late']}")

    # Pass criteria
    non_mono_pass = pattern["is_non_monotonic"]
    peak_pass = pattern["peak_at_68"]
    pilot_pass = non_mono_pass and peak_pass

    print(f"\n  PASS criteria:")
    print(f"    Non-monotonic pattern: {'PASS' if non_mono_pass else 'FAIL'}")
    print(f"    Peak at layer 6-8: {'PASS' if peak_pass else 'FAIL'}")
    print(f"    Overall: {'PASS' if pilot_pass else 'FAIL'}")

    # ASCII plot
    ascii_plot = make_ascii_plot(sorted_results)
    print(ascii_plot)

    # Compute Spearman correlation between layer and absorption
    if len(pattern["layers"]) >= 3:
        layers_arr = np.array(pattern["layers"])
        abs_arr = np.array(pattern["absorption_vals"])
        layer_corr, layer_p = spearmanr(layers_arr, abs_arr)
        layer_corr_val = float(layer_corr) if not np.isnan(layer_corr) else None
        layer_p_val = float(layer_p) if not np.isnan(layer_p) else None
        print(f"\n  Layer-absorption Spearman r: {layer_corr_val:.4f} (p={layer_p_val:.3f})")
    else:
        layer_corr = None
        layer_p = None
        layer_corr_val = None
        layer_p_val = None

    # Build summary
    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_tokens_per_layer": N_TOKENS,
            "n_features_per_layer": N_FEATURES,
            "seed": SEED,
            "uas_alpha": UAS_ALPHA,
            "uas_beta": UAS_BETA,
            "pilot_layers": [4, 8],
            "new_layers": NEW_LAYERS,
        },
        "layer_results": sorted_results,
        "h1_pattern": pattern,
        "h1_pattern_analysis": {
            "non_monotonic": pattern["is_non_monotonic"],
            "peak_at_68": pattern["peak_at_68"],
            "peak_layer": pattern["peak_layer"],
            "peak_val": pattern["peak_val"],
            "early_vs_mid": pattern["early_vs_mid"],
            "mid_vs_late": pattern["mid_vs_late"],
            "layer_corr": layer_corr_val,
            "layer_p": layer_p_val,
        },
        "pass_criteria": {
            "non_monotonic_required": True,
            "peak_at_68_required": True,
            "non_monotonic_pass": non_mono_pass,
            "peak_pass": peak_pass,
            "overall_pass": pilot_pass,
        },
        "pilot_pass": pilot_pass,
    }

    # Save JSON
    out_file = RESULTS_DIR / f"{TASK_ID}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Write DONE
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if pilot_pass else "failed",
        "summary": f"full_h1_gpt2 pilot: peak_layer={pattern['peak_layer']}, non_mono={pattern['is_non_monotonic']}, pilot_pass={pilot_pass}",
        "timestamp": datetime.now().isoformat(),
    }))

    # Write PROGRESS
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "completed",
        "peak_layer": pattern["peak_layer"],
        "pilot_pass": pilot_pass,
        "layers": pattern["layers"],
        "absorption_vals": [float(v) for v in pattern["absorption_vals"]],
        "updated_at": datetime.now().isoformat(),
    }))

    # Write summary markdown
    non_mono_pass_str = "PASS" if non_mono_pass else "FAIL"
    peak_pass_str = "PASS" if peak_pass else "FAIL"
    pilot_pass_str = "PASS" if pilot_pass else "FAIL"
    layer_corr_display = f"r={layer_corr_val:.4f} (p={layer_p_val:.3f})" if layer_corr_val is not None else "N/A"

    md_file = RESULTS_DIR / f"{TASK_ID}_summary.md"
    md_content = f"""# Full H1 (PILOT): GPT-2 Absorption Atlas

## Configuration
- N tokens per layer: {N_TOKENS}
- N features per layer: {N_FEATURES}
- UAS: alpha={UAS_ALPHA}, beta={UAS_BETA}
- Pilot data: layers 4, 8 from pilot_h1_h4 (n=100 features each)
- New data: layers {NEW_LAYERS} (n={N_FEATURES} features each)

## Results by Layer

| Layer | Mean Absorption | Mean UAS | Sparsity | Reconstruction MSE | Source |
|-------|----------------|----------|----------|--------------------|--------|
"""
    for r in sorted_results:
        md_content += f"| {r['layer']} | {r['mean_gini_absorption']:.4f} | {r['mean_uas']:.4f} | {r.get('sparsity', 'N/A')} | {r.get('reconstruction_mse', 'N/A')} | {r['source']} |\n"

    md_content += f"""
## H1 Pattern Analysis

- **Peak layer**: {pattern['peak_layer']} (absorption={pattern['peak_val']:.4f})
- **Non-monotonic**: {pattern['is_non_monotonic']}
- **Peak at layer 6-8**: {pattern['peak_at_68']}
- **Early mean**: {pattern['early_mean']:.4f} | **Mid mean**: {pattern['mid_mean']:.4f} | **Late mean**: {pattern['late_mean']:.4f}
- **Early < Mid**: {pattern['early_vs_mid']}
- **Mid > Late**: {pattern['mid_vs_late']}
- **Layer-absorption correlation**: {layer_corr_display}

## Pass Criteria

| Criterion | Result |
|-----------|--------|
| Non-monotonic pattern | {non_mono_pass_str} |
| Peak at layer 6-8 | {peak_pass_str} |
| **Overall** | **{pilot_pass_str}** |

## Pilot Pass: {pilot_pass}

"""
    with open(md_file, "w") as f:
        f.write(md_content)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Layers tested: {pattern['layers']}")
    print(f"  Absorption: {[f'{v:.4f}' for v in pattern['absorption_vals']]}")
    print(f"  Peak layer: {pattern['peak_layer']}")
    print(f"  Non-monotonic: {pattern['is_non_monotonic']}")
    print(f"  Peak at 6-8: {pattern['peak_at_68']}")
    print(f"  PILOT PASS: {pilot_pass}")
    print(f"  Results: {out_file}")
    print(f"  Summary: {md_file}")
    print(f"  Done: {datetime.now().isoformat()}")

    return summary


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
        done_file.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }))
        sys.exit(1)
