#!/usr/bin/env python3
"""
setup_env task for iteration 6: Environment Setup and Model Download
- Verify all dependencies are installed
- Load Gemma 2 2B model via TransformerLens (using unsloth/gemma-2-2b for ungated access)
- Load Gemma Scope SAEs: L10/L12/L20 at 16k/65k widths
- Load MULTIPLE L0 configs for L12 16k: L0={22, 41, 82, 176}
- Smoke test: encode 100 tokens, verify non-zero activations, measure VRAM
- Write PID, PROGRESS, DONE markers
"""

import json
import os
import sys
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

# === Configuration ===
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results"
TASK_ID = "setup_env"
GPU_ID = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# === PID file ===
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(step, total_steps, description, metric=None):
    """Write progress file for system monitor."""
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "description": description,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary="", results=None):
    """Write DONE marker."""
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "results": results or {},
        "timestamp": datetime.now().isoformat(),
    }))

def safe_serialize(obj):
    """Make object JSON-safe by converting non-serializable types."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

TOTAL_STEPS = 8
results = {
    "dependencies": {},
    "model_loaded": False,
    "model_info": {},
    "sae_configs_loaded": {},
    "smoke_test": {},
    "vram_usage": {},
    "all_checks_passed": False,
    "available_sae_ids": {},
}

try:
    # === Step 1: Verify dependencies ===
    report_progress(1, TOTAL_STEPS, "Verifying dependencies")
    print("=" * 60)
    print("Step 1: Verifying dependencies")
    print("=" * 60)

    import torch
    import numpy as np
    import sklearn
    import scipy

    try:
        import pygam
        pygam_ver = pygam.__version__
    except ImportError:
        pygam_ver = "NOT INSTALLED"

    deps = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pygam": pygam_ver,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2) if torch.cuda.is_available() else 0,
    }
    results["dependencies"] = deps
    print(f"  torch: {deps['torch']}")
    print(f"  CUDA: {deps['cuda_available']} ({deps['cuda_version']})")
    print(f"  GPU: {deps['gpu_name']} ({deps['gpu_vram_total_mb']} MB)")
    print(f"  numpy: {deps['numpy']}, sklearn: {deps['sklearn']}, scipy: {deps['scipy']}")
    print(f"  pygam: {deps['pygam']}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    # === Step 2: Import SAELens and TransformerLens ===
    report_progress(2, TOTAL_STEPS, "Importing SAELens and TransformerLens")
    print("\n" + "=" * 60)
    print("Step 2: Importing SAELens and TransformerLens")
    print("=" * 60)

    import sae_lens
    from sae_lens import SAE
    import transformer_lens
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from importlib.metadata import version as pkg_version
        tl_version = pkg_version("transformer-lens")
    except Exception:
        tl_version = "unknown"

    deps["sae_lens"] = sae_lens.__version__
    deps["transformer_lens"] = tl_version
    print(f"  sae_lens: {sae_lens.__version__}")
    print(f"  transformer_lens: {tl_version}")

    # === Step 3: Load Gemma 2 2B model ===
    report_progress(3, TOTAL_STEPS, "Loading Gemma 2 2B model")
    print("\n" + "=" * 60)
    print("Step 3: Loading Gemma 2 2B model via TransformerLens")
    print("=" * 60)

    t0 = time.time()
    hf_model_name = "unsloth/gemma-2-2b"
    print(f"  Loading tokenizer from {hf_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    print(f"  Loading HF model weights from {hf_model_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
    print("  Converting to TransformerLens HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device="cuda",
        dtype=torch.float16,
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    model_load_time = time.time() - t0
    results["model_loaded"] = True
    results["model_load_time_sec"] = round(model_load_time, 1)
    results["model_info"] = {
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "vocab_size": model.cfg.d_vocab,
        "hf_source": hf_model_name,
    }

    vram_after_model = torch.cuda.memory_allocated(0) / 1024**2
    results["vram_usage"]["after_model_mb"] = round(vram_after_model)
    print(f"  Model loaded in {model_load_time:.1f}s")
    print(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}, vocab: {model.cfg.d_vocab}")
    print(f"  VRAM after model: {vram_after_model:.0f} MB")

    # === Step 4: Load L12 16k SAE at multiple L0 values ===
    report_progress(4, TOTAL_STEPS, "Loading L12 16k SAEs (multiple L0)")
    print("\n" + "=" * 60)
    print("Step 4: Loading L12 16k SAEs at multiple L0 values")
    print("=" * 60)

    # Available L0 configs from SAELens directory: 22, 41, 82, 176, 445
    # Task plan asks for L0={22, 42, 82, 163} -- closest available: 22, 41, 82, 176
    l0_configs = {
        "L0_22": "layer_12/width_16k/average_l0_22",
        "L0_41": "layer_12/width_16k/average_l0_41",
        "L0_82": "layer_12/width_16k/average_l0_82",
        "L0_176": "layer_12/width_16k/average_l0_176",
    }

    loaded_saes = {}
    for l0_name, sae_id in l0_configs.items():
        t0 = time.time()
        try:
            sae = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res",
                sae_id=sae_id,
                device="cuda",
            )
            load_time = time.time() - t0
            # Get hook_name from metadata
            hook_name = sae.cfg.metadata.get("hook_name", "unknown") if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata else "unknown"
            loaded_saes[l0_name] = sae
            results["sae_configs_loaded"][l0_name] = {
                "sae_id": sae_id,
                "loaded": True,
                "load_time_sec": round(load_time, 1),
                "d_sae": sae.cfg.d_sae,
                "d_in": sae.cfg.d_in,
                "hook_name": hook_name,
                "architecture": getattr(sae.cfg, 'architecture', 'unknown'),
            }
            print(f"  {l0_name} ({sae_id}): OK in {load_time:.1f}s [d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}, hook={hook_name}]")
            # Free to save memory (keep L0_22 for smoke test)
            if l0_name != "L0_22":
                del sae
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as e:
            results["sae_configs_loaded"][l0_name] = {
                "sae_id": sae_id,
                "loaded": False,
                "error": str(e),
            }
            print(f"  {l0_name} ({sae_id}): FAILED - {e}")

    primary_sae = loaded_saes.get("L0_22", None)

    # === Step 5: Load other layer/width SAEs ===
    report_progress(5, TOTAL_STEPS, "Loading other SAE configs (L10, L12-65k, L20)")
    print("\n" + "=" * 60)
    print("Step 5: Loading additional SAE configs")
    print("=" * 60)

    other_configs = {
        "L12_65k": "layer_12/width_65k/average_l0_21",
        "L10_16k": "layer_10/width_16k/average_l0_21",
        "L20_16k": "layer_20/width_16k/average_l0_22",
    }

    for cfg_name, sae_id in other_configs.items():
        t0 = time.time()
        try:
            sae_other = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res",
                sae_id=sae_id,
                device="cuda",
            )
            load_time = time.time() - t0
            hook_name = sae_other.cfg.metadata.get("hook_name", "unknown") if hasattr(sae_other.cfg, 'metadata') and sae_other.cfg.metadata else "unknown"
            results["sae_configs_loaded"][cfg_name] = {
                "sae_id": sae_id,
                "loaded": True,
                "load_time_sec": round(load_time, 1),
                "d_sae": sae_other.cfg.d_sae,
                "d_in": sae_other.cfg.d_in,
                "hook_name": hook_name,
                "architecture": getattr(sae_other.cfg, 'architecture', 'unknown'),
            }
            print(f"  {cfg_name} ({sae_id}): OK in {load_time:.1f}s [d_sae={sae_other.cfg.d_sae}]")
            del sae_other
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            results["sae_configs_loaded"][cfg_name] = {
                "sae_id": sae_id,
                "loaded": False,
                "error": str(e),
            }
            print(f"  {cfg_name} ({sae_id}): FAILED - {e}")

    # === Step 6: Smoke test - encode tokens ===
    report_progress(6, TOTAL_STEPS, "Smoke test: encoding tokens through model + SAE")
    print("\n" + "=" * 60)
    print("Step 6: Smoke test - encode tokens through model + SAE")
    print("=" * 60)

    if primary_sae is not None:
        # Get the hook name from SAE metadata
        hook_name = primary_sae.cfg.metadata.get("hook_name", "blocks.12.hook_resid_post")

        # Generate test text with more tokens
        test_text = (
            "The capital of France is Paris. London is the capital of England. "
            "Tokyo is in Japan. Berlin is the capital of Germany. Madrid is in Spain. "
            "Rome is the capital of Italy. Beijing is in China. Moscow is in Russia. "
            "Cairo is in Egypt. Sydney is in Australia."
        )
        tokens = model.to_tokens(test_text)
        n_tokens = tokens.shape[1]
        print(f"  Test text tokenized: {n_tokens} tokens")

        with torch.no_grad():
            # Get residual stream activations at layer 12
            _, cache = model.run_with_cache(
                tokens,
                names_filter=hook_name,
                stop_at_layer=13,
            )
            activations = cache[hook_name]
            print(f"  Activations shape: {activations.shape}")
            print(f"  Activations dtype: {activations.dtype}")

            # Encode through SAE
            sae_input = activations.to(primary_sae.W_enc.dtype)
            sae_features = primary_sae.encode(sae_input)
            print(f"  SAE features shape: {sae_features.shape}")

            # Check non-zero activations
            nonzero_per_token = (sae_features > 0).sum(dim=-1).float()
            nonzero_pct = (nonzero_per_token > 0).float().mean().item() * 100
            mean_active = nonzero_per_token.mean().item()
            max_active = nonzero_per_token.max().item()
            min_active = nonzero_per_token.min().item()

            # Total active features across all tokens
            total_active_features = (sae_features.sum(dim=(0, 1)) > 0).sum().item()
            total_features = sae_features.shape[-1]

            # Also test reconstruction quality
            sae_output = primary_sae.decode(sae_features)
            recon_error = (activations.to(sae_output.dtype) - sae_output).norm(dim=-1)
            orig_norm = activations.to(sae_output.dtype).norm(dim=-1)
            relative_error = (recon_error / (orig_norm + 1e-8)).mean().item()

            smoke_results = {
                "n_tokens": n_tokens,
                "activations_shape": list(activations.shape),
                "sae_features_shape": list(sae_features.shape),
                "pct_tokens_with_nonzero": round(nonzero_pct, 1),
                "mean_active_features_per_token": round(mean_active, 1),
                "max_active_features_per_token": int(max_active),
                "min_active_features_per_token": int(min_active),
                "total_active_features": total_active_features,
                "total_features": total_features,
                "pct_live_features": round(total_active_features / total_features * 100, 2),
                "mean_relative_reconstruction_error": round(relative_error, 4),
                "hook_name": hook_name,
            }
            results["smoke_test"] = smoke_results

            print(f"  Tokens with non-zero SAE features: {nonzero_pct:.1f}%")
            print(f"  Mean active features per token: {mean_active:.1f}")
            print(f"  Active features (out of {total_features}): {total_active_features} ({total_active_features/total_features*100:.1f}%)")
            print(f"  Mean relative reconstruction error: {relative_error:.4f}")

            # Show top activating features
            top_acts = sae_features[0].max(dim=0).values
            top_k = 10
            top_feature_vals, top_feature_ids = top_acts.topk(top_k)
            print(f"\n  Top {top_k} most active features:")
            for i in range(top_k):
                print(f"    Feature {top_feature_ids[i].item()}: max activation = {top_feature_vals[i].item():.4f}")

        del cache, activations, sae_features, sae_input, sae_output
        gc.collect()
        torch.cuda.empty_cache()
    else:
        results["smoke_test"] = {"error": "Primary SAE (L0_22) not loaded"}
        print("  SKIPPED: Primary SAE not loaded")

    # === Step 7: VRAM measurement ===
    report_progress(7, TOTAL_STEPS, "Measuring VRAM usage")
    print("\n" + "=" * 60)
    print("Step 7: VRAM usage measurement")
    print("=" * 60)

    vram_allocated = torch.cuda.memory_allocated(0) / 1024**2
    vram_reserved = torch.cuda.memory_reserved(0) / 1024**2
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    results["vram_usage"] = {
        "allocated_mb": round(vram_allocated),
        "reserved_mb": round(vram_reserved),
        "total_mb": round(vram_total),
        "utilization_pct": round(vram_allocated / vram_total * 100, 1),
        "within_budget": vram_allocated < 20000,
    }
    print(f"  Allocated: {vram_allocated:.0f} MB")
    print(f"  Reserved: {vram_reserved:.0f} MB")
    print(f"  Total: {vram_total:.0f} MB")
    print(f"  Utilization: {vram_allocated/vram_total*100:.1f}%")
    print(f"  Within 20GB budget: {vram_allocated < 20000}")

    # === Step 8: Catalog available SAE IDs for scaling surface ===
    report_progress(8, TOTAL_STEPS, "Cataloging all available SAE configs")
    print("\n" + "=" * 60)
    print("Step 8: Catalog available SAE configs and write results")
    print("=" * 60)

    from sae_lens.saes.sae import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    release = d.get("gemma-scope-2b-pt-res")
    if release:
        all_ids = sorted(release.saes_map.keys())
        # Group by layer
        layer_catalog = {}
        for sid in all_ids:
            parts = sid.split("/")
            if len(parts) >= 2 and parts[0].startswith("layer_"):
                layer = parts[0]
                if layer not in layer_catalog:
                    layer_catalog[layer] = []
                layer_catalog[layer].append(sid)
        results["available_sae_ids"] = layer_catalog
        total_layer_saes = sum(len(v) for v in layer_catalog.values())
        print(f"  Total available layer SAE configs: {total_layer_saes}")
        for layer in sorted(layer_catalog.keys(), key=lambda x: int(x.split("_")[1])):
            print(f"    {layer}: {len(layer_catalog[layer])} configs")

    # === Final validation ===
    l0_loaded = sum(1 for k, v in results["sae_configs_loaded"].items()
                    if k.startswith("L0_") and v.get("loaded", False))
    all_loaded = sum(1 for v in results["sae_configs_loaded"].values() if v.get("loaded", False))
    total_configs = len(results["sae_configs_loaded"])

    smoke_ok = results["smoke_test"].get("pct_tokens_with_nonzero", 0) >= 90
    vram_ok = results["vram_usage"].get("within_budget", False)

    results["all_checks_passed"] = (
        results["model_loaded"]
        and l0_loaded >= 4
        and smoke_ok
        and vram_ok
    )
    results["summary"] = {
        "model_loaded": results["model_loaded"],
        "l0_configs_loaded": f"{l0_loaded}/4",
        "total_sae_configs_loaded": f"{all_loaded}/{total_configs}",
        "smoke_test_passed": smoke_ok,
        "vram_within_budget": vram_ok,
        "overall": "PASS" if results["all_checks_passed"] else "FAIL",
    }

    print(f"\n  {'='*40}")
    print(f"  Model loaded: {results['model_loaded']}")
    print(f"  L0 configs loaded: {l0_loaded}/4")
    print(f"  Total SAE configs loaded: {all_loaded}/{total_configs}")
    print(f"  Smoke test passed: {smoke_ok}")
    print(f"  VRAM within budget: {vram_ok}")
    print(f"  ===== OVERALL: {results['summary']['overall']} =====")

    # Write setup_verification.json (with safe serialization)
    output_path = RESULTS_DIR / "setup_verification.json"
    output_path.write_text(json.dumps(safe_serialize(results), indent=2))
    print(f"\n  Results written to {output_path}")

    mark_done(
        status="success",
        summary=f"Setup complete. {all_loaded}/{total_configs} SAE configs loaded. Smoke test {'PASS' if smoke_ok else 'FAIL'}. VRAM {results['vram_usage']['allocated_mb']}MB.",
        results=results["summary"],
    )
    print("  DONE marker written.")

except Exception as e:
    tb = traceback.format_exc()
    print(f"\n!!! FATAL ERROR: {e}")
    print(tb)
    results["error"] = str(e)
    results["traceback"] = tb

    output_path = RESULTS_DIR / "setup_verification.json"
    output_path.write_text(json.dumps(safe_serialize(results), indent=2))

    mark_done(
        status="failed",
        summary=f"Setup failed: {e}",
        results={"error": str(e)},
    )
    sys.exit(1)
finally:
    if pid_file.exists():
        pid_file.unlink()
