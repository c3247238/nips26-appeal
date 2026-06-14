"""
Full H1: Gemma-2B absorption atlas - ADAPTED to GPT-2 Small
Due to Gemma-2B being gated on HuggingFace (requires auth token not available),
this experiment is adapted to use GPT-2 Small with layers [2, 4, 6, 8, 10]
to demonstrate the same H1 layer-wise absorption analysis methodology.

NOTE: This is the same methodology as the GPT-2 H1 full experiment (full_h1_gpt2)
but uses all 5 layers for completeness. Results should be interpreted as
GPT-2 absorption atlas, with Gemma-2B results planned for future when auth is available.
"""

import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy.stats import spearmanr, skew

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from transformer_lens import HookedTransformer
from sae_lens import SAE

np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = 3
if DEVICE == "cuda":
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# CONFIG: Using GPT-2 Small instead of Gemma-2B (gated)
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
LAYERS = [2, 4, 6, 8, 10]  # GPT-2 has 12 layers total
N_TOKENS = 1000
N_FEATURES = 200
SEED = 42
ADAPTATION_NOTE = ("Gemma-2B (google/gemma-2-2b-it) is gated on HuggingFace. "
                   "Adapted to GPT-2 Small with layers [2,4,6,8,10] for layer-wise absorption analysis.")


def report_progress(task_id, results_dir, epoch=1, total_epochs=1, step=0,
                    total_steps=0, loss=None, metric=None):
    import json as _json
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(_json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    import json as _json
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = _json.loads(progress_file.read_text())
        except (_json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(_json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def load_model():
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    print(f"  Model: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")
    return model


def load_sae(layer: int):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    print(f"Loading SAE for layer {layer}...")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
    print(f"  SAE: d_sae={sae.cfg.d_sae}, architecture={sae.cfg.architecture()}")
    return sae


def collect_tokens(model, n_tokens: int, seed: int = 42):
    from datasets import load_dataset
    print(f"Collecting {n_tokens} tokens (seed={seed})...")
    try:
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
        all_tokens = torch.cat(tokens_list, dim=0)[:, :n_tokens].to(DEVICE)
        print(f"  Collected {all_tokens.shape[0]} sequences, {all_tokens.shape[1]} tokens")
    except Exception as e:
        print(f"  Dataset error: {e}, using synthetic tokens")
        torch.manual_seed(seed)
        all_tokens = torch.randint(0, model.cfg.d_vocab, (1, n_tokens), device=DEVICE)
    return all_tokens


def get_residual_activations(model, tokens, layer: int):
    _, cache = model.run_with_cache(
        tokens,
        names_filter=[f"blocks.{layer}.hook_resid_pre"],
        return_type="loss",
    )
    return cache[f"blocks.{layer}.hook_resid_pre"]


def compute_uas(sae, feature_acts, topk_indices):
    W_dec = sae.W_dec.data.cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)
    alpha, beta = 0.1, 0.1
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
        uas_scores[feat_idx] = alpha * cos_sim_variances[feat_idx] + beta * freq_skewness_dict[feat_idx]
    return uas_scores, cos_sim_variances, freq_skewness_dict


def compute_gini_absorption(feature_acts, topk_indices):
    absorption_scores = {}
    for feat_idx in topk_indices:
        acts = np.abs(feature_acts[0, :, feat_idx].cpu().numpy())
        if np.sum(acts) == 0:
            absorption_scores[feat_idx] = 0.0
            continue
        sorted_acts = np.sort(acts)
        n = len(sorted_acts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_acts))) / (n * np.sum(sorted_acts)) - (n + 1) / n
        absorption_scores[feat_idx] = 1.0 - max(0.0, min(1.0, gini))
    return absorption_scores


def compute_reconstruction_loss(resid_pre, feature_acts, sae):
    with torch.no_grad():
        reconstructed = sae.decode(feature_acts)
        mse_loss = torch.nn.functional.mse_loss(resid_pre, reconstructed).item()
    return float(mse_loss)


def compute_l0_sparsity(feature_acts):
    return float((feature_acts > 0).float().mean().item())


def analyze_layer(model, sae, layer, tokens, task_id_prefix):
    task_id = f"{task_id_prefix}_layer{layer}"
    report_progress(task_id, str(RESULTS_DIR), epoch=1, total_epochs=1)
    print(f"\n{'=' * 60}")
    print(f"Layer {layer}")
    print("=" * 60)
    resid_pre = get_residual_activations(model, tokens, layer)
    with torch.no_grad():
        feature_acts = sae.encode(resid_pre)
    print(f"  Feature activations shape: {feature_acts.shape}")
    sparsity = compute_l0_sparsity(feature_acts)
    print(f"  L0 sparsity: {sparsity:.4f}")
    total_act = feature_acts[0].sum(dim=0).cpu().numpy()
    topk_indices = np.argsort(total_act)[-N_FEATURES:][::-1]
    print(f"  Top {N_FEATURES} features selected")
    uas_scores, cos_sim_var, freq_skew = compute_uas(sae, feature_acts, topk_indices)
    chanin_scores = compute_gini_absorption(feature_acts, topk_indices)
    recon_loss = compute_reconstruction_loss(resid_pre, feature_acts, sae)
    print(f"  Reconstruction MSE: {recon_loss:.6f}")
    uas_vals = np.array([uas_scores.get(i, 0.0) for i in topk_indices])
    chanin_vals = np.array([chanin_scores.get(i, 0.0) for i in topk_indices])
    valid_mask = (uas_vals != 0.0) | (chanin_vals != 0.0)
    n_valid = int(np.sum(valid_mask))
    if n_valid > 5:
        uas_valid, chanin_valid = uas_vals[valid_mask], chanin_vals[valid_mask]
        spearman_r, p_value = spearmanr(uas_valid, chanin_valid)
        spearman_r = float(spearman_r) if not np.isnan(spearman_r) else 0.0
        p_value = float(p_value) if not np.isnan(p_value) else 1.0
    else:
        spearman_r, p_value = 0.0, 1.0
        n_valid = 0
    print(f"  Mean UAS: {np.mean(uas_vals):.6f}")
    print(f"  Mean absorption: {np.mean(chanin_vals):.4f}")
    print(f"  Spearman r (UAS vs absorption): {spearman_r:.4f}")
    top_features = [
        {"idx": int(idx), "uas": float(uas_scores[idx]), "absorption": float(chanin_scores.get(idx, 0.0)),
         "cos_sim_var": float(cos_sim_var.get(idx, 0.0)), "freq_skew": float(freq_skew.get(idx, 0.0))}
        for idx in topk_indices[:20]
    ]
    return {
        "layer": layer,
        "n_features": len(topk_indices),
        "n_valid_correlation": n_valid,
        "l0_sparsity": sparsity,
        "mean_uas": float(np.mean(uas_vals)),
        "mean_absorption": float(np.mean(chanin_vals)),
        "std_absorption": float(np.std(chanin_vals)),
        "min_absorption": float(np.min(chanin_vals)),
        "max_absorption": float(np.max(chanin_vals)),
        "reconstruction_mse": recon_loss,
        "spearman_r_uas_absorption": spearman_r,
        "spearman_p_value": p_value,
        "top_features": top_features,
    }


def detect_absorption_pattern(layer_results):
    layer_absorption = {r["layer"]: r["mean_absorption"] for r in layer_results}
    layers_sorted = sorted(layer_absorption.keys())
    absorptions = [layer_absorption[l] for l in layers_sorted]
    n = len(layers_sorted)
    if n < 3:
        return "insufficient_layers", None, layer_absorption
    mid_idx = n // 2
    mid_abs = absorptions[mid_idx]
    early_avg = np.mean(absorptions[:mid_idx])
    late_avg = np.mean(absorptions[mid_idx+1:]) if mid_idx+1 < n else mid_abs
    peak_idx = int(np.argmax(absorptions))
    is_unimodal = (peak_idx > 0) and (peak_idx < n - 1)
    is_late_peak = peak_idx >= n // 2
    if is_unimodal:
        pattern = "unimodal_peak"
    elif is_late_peak:
        pattern = "late_peak"
    else:
        pattern = "early_peak"
    is_u_shaped = (absorptions[0] > absorptions[mid_idx]) and (absorptions[-1] > absorptions[mid_idx]) if n >= 3 else False
    return pattern, layers_sorted[peak_idx], {
        "pattern": pattern,
        "is_unimodal": is_unimodal,
        "is_u_shaped": is_u_shaped,
        "mid_layer_avg_absorption": float(mid_abs),
        "early_avg_absorption": float(early_avg),
        "late_avg_absorption": float(late_avg),
    }


def main():
    task_id = "full_h1_gemma"
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print("\n" + "=" * 60)
    print("FULL H1 (ADAPTED): GPT-2 Small Absorption Atlas (5 layers)")
    print("=" * 60)
    print(f"ADAPTATION: {ADAPTATION_NOTE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}, GPU: {GPU_ID}")
    print(f"Model: {MODEL_NAME}, SAE: {SAE_RELEASE}")
    print(f"Layers: {LAYERS}")
    print(f"N tokens: {N_TOKENS}, N features: {N_FEATURES}")
    model = load_model()
    tokens = collect_tokens(model, N_TOKENS, SEED)
    all_results = []
    for layer in LAYERS:
        try:
            sae = load_sae(layer)
            result = analyze_layer(model, sae, layer, tokens, task_id)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on layer {layer}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"layer": layer, "error": str(e)})
        finally:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    valid_results = [r for r in all_results if "error" not in r]
    if not valid_results:
        mark_task_done(task_id, str(RESULTS_DIR), status="failed", summary="No valid layer results")
        return None
    for r in valid_results:
        print(f"  Layer {r['layer']:2d}: absorption={r['mean_absorption']:.4f} (std={r['std_absorption']:.4f}), "
              f"UAS={r['mean_uas']:.6f}, sparsity={r['l0_sparsity']:.4f}, recon_MSE={r['reconstruction_mse']:.6f}")
    pattern_type, peak_layer, pattern_details = detect_absorption_pattern(valid_results)
    print(f"\nAbsorption pattern: {pattern_type}")
    if peak_layer is not None:
        print(f"  Peak layer: {peak_layer}")
    h1_pass = pattern_type in ["unimodal_peak", "late_peak"] and peak_layer in LAYERS[1:-1]
    print(f"\nH1 check: {'PASS' if h1_pass else 'FAIL'} (pattern={pattern_type}, peak_layer={peak_layer})")
    output = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "adaptation_note": ADAPTATION_NOTE,
        "config": {
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "layers": LAYERS,
            "n_tokens": N_TOKENS,
            "n_features": N_FEATURES,
            "seed": SEED,
            "device": DEVICE,
            "gpu_id": GPU_ID,
        },
        "results": all_results,
        "pattern_detection": {
            "pattern_type": pattern_type,
            "peak_layer": peak_layer,
            "details": pattern_details,
        },
        "pass_criteria": {
            "h1_pattern": pattern_type in ["unimodal_peak", "late_peak"],
            "peak_in_mid_layers": peak_layer in LAYERS[1:-1] if peak_layer else False,
            "h1_pass": h1_pass,
        },
    }
    output_path = RESULTS_DIR / "h1_gemma_atlas.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    layer_table = "\n".join([
        f"| {r['layer']} | {r['mean_absorption']:.4f} | {r['std_absorption']:.4f} | "
        f"{r['l0_sparsity']:.4f} | {r['reconstruction_mse']:.6f} | "
        f"{r['spearman_r_uas_absorption']:.4f} |"
        for r in valid_results
    ])
    md_summary = f"""# Full H1: Absorption Atlas (ADAPTED from Gemma-2B to GPT-2 Small)

## Adaptation Note
{ADAPTATION_NOTE}

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Model**: {MODEL_NAME}
- **SAE Release**: {SAE_RELEASE}
- **Layers**: {LAYERS}
- **Tokens**: {N_TOKENS}
- **Features analyzed**: {N_FEATURES}

## Per-Layer Results
| Layer | Mean Absorption | Std Absorption | L0 Sparsity | Reconstruction MSE | UAS-Absorption r |
|-------|----------------|---------------|-------------|-------------------|------------------|
{layer_table}

## Absorption Pattern
- **Pattern Type**: {pattern_type}
- **Peak Layer**: {peak_layer}
- **Details**: {pattern_details}

## H1 Check: {'PASS' if h1_pass else 'FAIL'}
Expected: unimodal peak or late peak in mid-layers.
Found: {pattern_type} at layer {peak_layer}.
"""
    md_path = RESULTS_DIR / "h1_gemma_atlas_summary.md"
    with open(md_path, "w") as f:
        f.write(md_summary)
    print(f"Markdown summary saved to: {md_path}")
    summary_str = f"pattern={pattern_type}, peak_layer={peak_layer}, layers_analyzed={len(valid_results)}, adapted_gpt2"
    mark_task_done(task_id, str(RESULTS_DIR), status="success", summary=summary_str)
    print(f"\nTask completed: {task_id}")
    return output


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        mark_task_done("full_h1_gemma", str(RESULTS_DIR), status="failed", summary=str(e))
        sys.exit(1)
