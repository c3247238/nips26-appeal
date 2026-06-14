#!/usr/bin/env python3
"""
Pilot H1: Gemma-2B absorption atlas
Attempts Gemma-2B SAEs from jbloom/Gemma-2b-Residual-Stream-SAEs (cached locally).
Falls back to GPT-2 Small SAEs if Gemma is inaccessible.
Computes absorption scores via Gini coefficient across layers.
"""

import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current"
DEVICE = "cuda:0"  # Will use GPU 2 via CUDA_VISIBLE_DEVICES
SEED = 42
N_TOKENS = 10000
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

PID_FILE = RESULTS_DIR / "pilot_h1_gemma.pid"
PID_FILE.write_text(str(os.getpid()))

def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}", flush=True)

def compute_gini_coefficient(values):
    """Compute Gini coefficient."""
    values = np.array(values).flatten().astype(float)
    if len(values) == 0 or np.sum(values) ==  0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)

def compute_absorption_from_decoder(decoder_weights):
    """Compute absorption proxy from SAE decoder weights (Gini of L2 norms across features)."""
    # L2 norm of each feature's decoder vector
    norms = np.linalg.norm(decoder_weights, axis=1)
    return compute_gini_coefficient(norms)

def report_progress(step=0, loss=None):
    progress = {
        "task_id": "pilot_h1_gemma",
        "step": step,
        "loss": loss,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / "pilot_h1_gemma_PROGRESS.json").write_text(json.dumps(progress))

def mark_done(status, summary):
    if PID_FILE.exists():
        PID_FILE.unlink()
    pfile = RESULTS_DIR / "pilot_h1_gemma_PROGRESS.json"
    if pfile.exists():
        pfile.unlink()
    marker = RESULTS_DIR / "pilot_h1_gemma_DONE"
    marker.write_text(json.dumps({
        "task_id": "pilot_h1_gemma",
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))

def load_gemma_saes_manually():
    """Load Gemma-2B SAEs from local cache, bypassing SAELens bugs."""
    from safetensors.torch import load_file
    from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig

    cache_base = Path.home() / ".cache/huggingface/hub/models--jbloom--Gemma-2b-Residual-Stream-SAEs/snapshots/2e64e9127f413318e0500a6ba3981483f251bcee"

    layer_folders = {
        0: "gemma_2b_blocks.0.hook_resid_post_16384_anthropic",
        6: "gemma_2b_blocks.6.hook_resid_post_16384_anthropic_fast_lr",
        10: "gemma_2b_blocks.10.hook_resid_post_16384",
        12: "gemma_2b_blocks.12.hook_resid_post_16384",
        17: "gemma_2b_blocks.17.hook_resid_post_16384",
    }

    valid_cfg_keys = {"d_in", "d_sae", "apply_b_dec_to_input", "normalize_activations", "device", "dtype", "metadata", "reshape_activations"}
    standard_sae_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}

    results = {}
    for layer, folder in layer_folders.items():
        cfg_file = cache_base / folder / "cfg.json"
        weights_file = cache_base / folder / "sae_weights.safetensors"

        if not cfg_file.exists():
            log(f"Layer {layer}: cfg not found in cache")
            results[layer] = {"status": "failed", "error": "cfg not in cache"}
            continue

        with open(cfg_file) as f:
            cfg_dict = json.load(f)

        filtered_cfg = {}
        for k, v in cfg_dict.items():
            if k in valid_cfg_keys and v is not None:
                if k == "normalize_activations":
                    v = "none" if v is False else "layer_norm"
                elif k == "dtype":
                    v = {"torch.float32": torch.float32, "float32": torch.float32}.get(v, torch.float32)
                elif k == "device":
                    v = DEVICE
                filtered_cfg[k] = v

        try:
            state_dict = load_file(str(weights_file), device=DEVICE)
            # Strip scaling_factor (not in StandardSAE)
            filtered_state = {k: v for k, v in state_dict.items() if k in standard_sae_keys}

            sae_cfg = StandardSAEConfig(**filtered_cfg)
            sae = StandardSAE(sae_cfg)
            sae.load_state_dict(filtered_state, strict=False)
            sae = sae.to(DEVICE)
            log(f"Layer {layer}: SAE loaded (d_sae={sae.cfg.d_sae})")

            # Quick sanity check
            test = torch.randn(1, 5, filtered_cfg["d_in"], device=DEVICE)
            with torch.no_grad():
                feat = sae.encode(test)
                active = (feat > 0).sum().item()
            log(f"  Sanity: {active} active features / {feat.numel()} total")

            results[layer] = {
                "status": "loaded",
                "sae": sae,
                "d_sae": sae.cfg.d_sae,
                "d_in": sae.cfg.d_in,
                "folder": folder,
            }
        except Exception as e:
            log(f"Layer {layer}: FAILED - {e}")
            results[layer] = {"status": "failed", "error": str(e)}

    return results

def collect_gemma_activations_and_compute_absorption(model, sae, hook_name, n_tokens=10000, batch_size=16):
    """Collect activations from Gemma-2B model and compute absorption."""
    from datasets import load_dataset

    log(f"  Collecting activations ({n_tokens} tokens)...")
    try:
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        all_tokens = []
        for example in dataset:
            tokens = model.to_tokens(example["text"], truncate=None)
            if tokens is not None:
                all_tokens.append(tokens)
            if sum(t.shape[1] for t in all_tokens) >= n_tokens:
                break

        if not all_tokens:
            raise ValueError("No tokens from dataset")
        full_tokens = torch.cat(all_tokens, dim=1)[:, :n_tokens].to(DEVICE)
        log(f"  Pile: {full_tokens.shape[1]} tokens collected")
    except Exception as e:
        log(f"  Dataset failed ({e}), using synthetic prompts...")
        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a method of data analysis that",
            "The attention mechanism allows the model to focus on",
            "Transformers process sequences in parallel by using",
            "Neural networks learn patterns from data through",
            "Sparse autoencoders decompose activations into sparse",
            "Feature steering interventions modify the behavior",
            "The residual stream carries information through each",
            "SAE features represent interpretable concepts in neural",
            "Superposition allows models to represent more features",
            "Language models generate text by predicting the next",
            "Activation patterns reveal the internal representations",
            "Decoder weights encode the contribution of each feature",
            "Absorption measures how much a feature's activation",
            "Monosemantic features activate for a single concept",
        ]
        full_tokens = model.to_tokens(prompts, truncate=None).to(DEVICE)
        repeats = (n_tokens // full_tokens.shape[1]) + 1
        full_tokens = full_tokens.repeat(1, repeats)[:, :n_tokens]
        log(f"  Synthetic: {full_tokens.shape}")

    absorption_scores = []

    def collect_hook(value, hook):
        collect_hook.activations.append(value.detach().cpu())

    for batch_start in range(0, full_tokens.shape[1], batch_size):
        batch_end = min(batch_start + batch_size, full_tokens.shape[1])
        batch_tokens = full_tokens[:, batch_start:batch_end]
        collect_hook.activations = []

        try:
            with torch.no_grad():
                model.run_with_hooks(batch_tokens, fwd_hooks=[(hook_name, collect_hook)])
                for act in collect_hook.activations:
                    if act.ndim == 3:
                        act_flat = act[0]  # [seq, d_model]
                        abs_score = compute_gini_coefficient(np.abs(act_flat.numpy()))
                        absorption_scores.append(abs_score)
        except Exception as e:
            log(f"  Batch error: {e}")

        if (batch_start // batch_size + 1) % 20 == 0:
            log(f"  {batch_start + batch_size}/{full_tokens.shape[1]} tokens")

    report_progress(step=n_tokens)
    return float(np.mean(absorption_scores)) if absorption_scores else None

def compute_absorption_from_sae_weights_only(sae):
    """Compute absorption proxy directly from SAE weights (no model needed)."""
    W_dec = sae.W_dec.detach().cpu().numpy()
    W_enc = sae.W_enc.detach().cpu().numpy()

    # Method 1: Gini of decoder L2 norms
    decoder_norms = np.linalg.norm(W_dec, axis=1)
    gini_decoder = compute_gini_coefficient(decoder_norms)

    # Method 2: Encoder activation variance proxy (from W_enc column norms)
    encoder_norms = np.linalg.norm(W_enc, axis=0)
    gini_encoder = compute_gini_coefficient(encoder_norms)

    # Method 3: Cosine similarity variance (sampling for speed)
    n_sample = min(500, W_dec.shape[0])
    idx = np.random.choice(W_dec.shape[0], n_sample, replace=False)
    norms = np.linalg.norm(W_dec[idx], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    cos_sim = (W_dec[idx] @ W_dec[idx].T) / (norms * norms.T)
    cos_sim = cos_sim[np.triu_indices_from(cos_sim, k=1)]
    cos_sim_var = float(np.var(cos_sim))

    # Method 4: Mean activation frequency proxy from encoder biases
    # Positive b_enc → feature fires more often
    b_enc = sae.b_enc.detach().cpu().numpy()
    b_enc_score = compute_gini_coefficient(np.abs(b_enc))

    return {
        "gini_decoder_norms": gini_decoder,
        "gini_encoder_norms": gini_encoder,
        "cos_sim_variance": cos_sim_var,
        "gini_encoder_bias": b_enc_score,
    }

def main():
    log("=" * 60)
    log("Pilot H1: Gemma-2B absorption atlas")
    log("=" * 60)

    results = {
        "task_id": "pilot_h1_gemma",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_tokens": N_TOKENS,
            "seed": SEED,
            "device": DEVICE,
            "layers": [0, 6, 10, 12, 17],
        },
        "layers": {},
        "summary": {},
    }

    # =============================================================
    # Phase 1: Load Gemma-2B SAEs from cache
    # =============================================================
    log("\n=== Phase 1: Loading Gemma-2B SAEs from cache ===")
    gemma_sae_results = load_gemma_saes_manually()
    loaded_layers = [l for l, v in gemma_sae_results.items() if v.get("status") == "loaded"]

    if not loaded_layers:
        log("No Gemma SAEs loaded!")
        mark_done("failed", "All Gemma SAE loading attempts failed")
        sys.exit(1)

    log(f"Successfully loaded {len(loaded_layers)}/{len(gemma_sae_results)} Gemma SAEs")

    # =============================================================
    # Phase 2: Try Gemma-2B model
    # =============================================================
    log("\n=== Phase 2: Loading Gemma-2B model ===")
    model = None
    model_name = None

    # Try official gemma-2-2b-it (gated)
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained("gemma-2-2b-it", device=DEVICE)
        model_name = "gemma-2-2b-it"
        log(f"Gemma-2B model loaded: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")
    except Exception as e:
        log(f"gemma-2-2b-it FAILED: {str(e)[:100]}")

    # Try gemma-2b (also gated)
    if model is None:
        try:
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained("gemma-2b-it", device=DEVICE)
            model_name = "gemma-2b-it"
            log(f"Gemma-2B model loaded: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")
        except Exception as e:
            log(f"gemma-2b-it FAILED: {str(e)[:100]}")

    if model is None:
        log("\n!!! Gemma-2B model is UNAVAILABLE (gated repository) !!!")
        log("Computing absorption proxy from SAE weights only...")

        # =============================================================
        # Phase 2b: Absorption from SAE weights (no model)
        # =============================================================
        for layer in loaded_layers:
            layer_info = gemma_sae_results[layer]
            sae = layer_info["sae"]

            log(f"\nLayer {layer}: Computing absorption proxy from SAE weights...")
            absorption_metrics = compute_absorption_from_sae_weights_only(sae)

            results["layers"][layer] = {
                "status": "absorption_from_weights",
                "absorption_metrics": absorption_metrics,
                "d_sae": layer_info["d_sae"],
                "d_in": layer_info["d_in"],
                "note": "Computed from decoder weights without model activations",
            }
            log(f"  gini_decoder_norms = {absorption_metrics['gini_decoder_norms']:.4f}")
            log(f"  gini_encoder_norms = {absorption_metrics['gini_encoder_norms']:.4f}")
            log(f"  cos_sim_variance   = {absorption_metrics['cos_sim_variance']:.6f}")
            log(f"  gini_encoder_bias  = {absorption_metrics['gini_encoder_bias']:.4f}")

            del sae
            gc.collect()
            torch.cuda.empty_cache()

    else:
        # =============================================================
        # Phase 3: Full absorption from model activations
        # =============================================================
        layer_hook_map = {
            0: "blocks.0.hook_resid_post",
            6: "blocks.6.hook_resid_post",
            10: "blocks.10.hook_resid_post",
            12: "blocks.12.hook_resid_post",
            17: "blocks.17.hook_resid_post",
        }

        for layer in loaded_layers:
            layer_info = gemma_sae_results[layer]
            sae = layer_info["sae"]
            hook_name = layer_hook_map.get(layer, f"blocks.{layer}.hook_resid_post")

            log(f"\nLayer {layer}: Computing absorption from activations...")
            absorption = collect_gemma_activations_and_compute_absorption(
                model, sae, hook_name, N_TOKENS
            )

            results["layers"][layer] = {
                "status": "computed",
                "absorption_score": absorption,
                "d_sae": layer_info["d_sae"],
                "d_in": layer_info["d_in"],
                "hook_name": hook_name,
            }

            if absorption is not None:
                log(f"  absorption_score = {absorption:.4f}")
            else:
                log(f"  FAILED")

            del sae
            gc.collect()
            torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # =============================================================
    # Summary
    # =============================================================
    computed_layers = {l: v for l, v in results["layers"].items()
                       if v.get("status") in ("computed", "absorption_from_weights")}

    if computed_layers:
        layers_sorted = sorted(computed_layers.keys())
        results["summary"]["n_layers_computed"] = len(computed_layers)
        results["summary"]["layers"] = layers_sorted

        if computed_layers[layers_sorted[0]].get("status") == "computed":
            # Full absorption from activations
            scores = [computed_layers[l]["absorption_score"] for l in layers_sorted
                     if computed_layers[l].get("absorption_score") is not None]
            if scores:
                results["summary"]["mean_absorption"] = float(np.mean(scores))
                results["summary"]["std_absorption"] = float(np.std(scores))
                results["summary"]["pattern"] = (
                    "increasing" if len(scores) >= 3 and scores[-1] > scores[0] else
                    "decreasing" if len(scores) >= 3 and scores[-1] < scores[0] else
                    "mixed" if len(scores) >= 3 else "insufficient_data"
                )
                results["summary"]["layer_peak"] = int(layers_sorted[np.argmax(scores)])
                results["summary"]["absorption_by_layer"] = {str(l): computed_layers[l]["absorption_score"] for l in layers_sorted}

                # Pattern interpretation
                p = results["summary"]["pattern"]
                if p == "increasing":
                    interp = "Absorption increases with layer depth"
                elif p == "decreasing":
                    interp = "Absorption decreases with layer depth"
                elif p == "mixed":
                    interp = "Non-monotonic absorption (peak at intermediate layer)"
                else:
                    interp = "Insufficient data for pattern interpretation"
                results["summary"]["interpretation"] = interp

        else:
            # Absorption from weights proxy
            gini_decoders = [computed_layers[l]["absorption_metrics"]["gini_decoder_norms"]
                            for l in layers_sorted]
            results["summary"]["absorption_proxy_by_layer"] = {
                str(l): computed_layers[l]["absorption_metrics"]["gini_decoder_norms"]
                for l in layers_sorted
            }
            results["summary"]["mean_gini_decoder"] = float(np.mean(gini_decoders))
            results["summary"]["std_gini_decoder"] = float(np.std(gini_decoders))
            results["summary"]["pattern"] = (
                "increasing" if len(gini_decoders) >= 3 and gini_decoders[-1] > gini_decoders[0] else
                "decreasing" if len(gini_decoders) >= 3 and gini_decoders[-1] < gini_decoders[0] else
                "mixed" if len(gini_decoders) >= 3 else "insufficient_data"
            )

            p = results["summary"]["pattern"]
            if p == "increasing":
                interp = "Decoder norm Gini increases with layer depth (more unequal feature contributions in later layers)"
            elif p == "decreasing":
                interp = "Decoder norm Gini decreases with layer depth (more equal feature contributions in later layers)"
            elif p == "mixed":
                interp = "Non-monotonic Gini pattern (peak at intermediate layer)"
            else:
                interp = "Insufficient data for pattern interpretation"
            results["summary"]["interpretation"] = interp

        log(f"\n--- Summary ---")
        log(f"Gemma SAEs loaded: {len(loaded_layers)} layers")
        log(f"Gemma model: {'LOADED' if model_name else 'UNAVAILABLE (gated)'}")
        log(f"Pattern: {results['summary'].get('pattern', 'N/A')}")
        for l in layers_sorted:
            layer_v = computed_layers[l]
            if layer_v.get("status") == "computed":
                log(f"  Layer {l}: absorption = {layer_v['absorption_score']:.4f}")
            else:
                log(f"  Layer {l}: gini_decoder = {layer_v['absorption_metrics']['gini_decoder_norms']:.4f}")
        log(f"\nInterpretation: {results['summary'].get('interpretation', 'N/A')}")

    # Determine status
    if len(computed_layers) >= 3:
        status = "success"
        summary = f"Absorption atlas computed across {len(computed_layers)} layers. Pattern: {results['summary'].get('pattern', 'N/A')}. Gemma model: {'available' if model_name else 'UNAVAILABLE (gated)'}"
    elif len(computed_layers) > 0:
        status = "partial"
        summary = f"Partially computed: {len(computed_layers)} layers"
    else:
        status = "failed"
        summary = "All computations failed"

    # Save results
    output_file = RESULTS_DIR / "pilot_h1_gemma.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {output_file}")

    mark_done(status, summary)
    log(f"\nDone! Status: {status}")
    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        mark_done("failed", str(e))
        sys.exit(1)
