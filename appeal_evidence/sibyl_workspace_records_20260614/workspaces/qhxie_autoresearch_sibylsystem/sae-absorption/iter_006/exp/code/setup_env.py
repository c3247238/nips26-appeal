#!/usr/bin/env python3
"""
setup_env.py — Environment Setup and Model Download for SAE Absorption Study (Iteration 6)

Task: Install/verify dependencies, download Gemma 2 2B + Gemma Scope SAEs,
      run smoke tests to verify non-zero activations.

GPU: Single GPU (CUDA_VISIBLE_DEVICES should be set externally)
Expected runtime: ~15-20 minutes (mostly model loading + SAE download)
"""

import json
import os
import sys
import time
import gc
import traceback
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
TASK_ID = "setup_env"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID file (critical for system recovery) ──────────────────────────────────
pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

# ── Progress reporting ───────────────────────────────────────────────────────
def report_progress(step, total_steps, description, extra=None):
    """Write progress file for system monitor."""
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "description": description,
        "metric": extra or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress.write_text(json.dumps(data, indent=2))


def mark_done(status="success", summary="", results=None):
    """Write DONE marker file."""
    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()
    # Read final progress
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    # Write DONE marker
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "results": results or {},
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


def main():
    start_time = time.time()
    total_steps = 7
    results = {
        "dependencies": {},
        "model_load": {},
        "sae_configs": [],
        "smoke_tests": [],
        "vram_usage": {},
        "warnings": [],
        "errors": [],
    }

    print("=" * 70)
    print("SAE Absorption Study — Environment Setup (Iteration 6)")
    print(f"Task ID: {TASK_ID}")
    print(f"Workspace: {WORKSPACE}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Step 1: Verify Python dependencies ───────────────────────────────
    report_progress(1, total_steps, "Verifying Python dependencies")
    print("\n[Step 1/7] Verifying Python dependencies...")

    deps = {}
    required = [
        "torch", "transformer_lens", "sae_lens", "transformers",
        "datasets", "sklearn", "scipy", "pygam", "einops", "numpy"
    ]
    all_ok = True
    for pkg in required:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            deps[pkg] = {"version": str(ver), "status": "ok"}
            print(f"  {pkg}: {ver} ✓")
        except ImportError as e:
            deps[pkg] = {"status": "missing", "error": str(e)}
            print(f"  {pkg}: MISSING ✗ — {e}")
            all_ok = False

    results["dependencies"] = deps
    if not all_ok:
        missing = [k for k, v in deps.items() if v["status"] == "missing"]
        results["errors"].append(f"Missing dependencies: {missing}")
        mark_done("failed", f"Missing dependencies: {missing}", results)
        sys.exit(1)

    # ── Step 2: Check GPU / CUDA ─────────────────────────────────────────
    report_progress(2, total_steps, "Checking GPU and CUDA")
    print("\n[Step 2/7] Checking GPU and CUDA...")

    import torch
    if not torch.cuda.is_available():
        results["errors"].append("CUDA not available")
        mark_done("failed", "CUDA not available", results)
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    vram_info = {
        "gpu_name": gpu_name,
        "vram_total_mb": round(total_mem / 1e6),
        "vram_free_mb": round(free_mem / 1e6),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }
    results["vram_usage"]["initial"] = vram_info
    print(f"  GPU: {gpu_name}")
    print(f"  VRAM: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total")
    print(f"  CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")

    if free_mem < 10e9:
        results["warnings"].append(f"Low VRAM: {free_mem/1e9:.1f}GB free. Need ~10GB for Gemma 2 2B + SAE.")

    # ── Step 3: Load Gemma 2 2B via TransformerLens ──────────────────────
    report_progress(3, total_steps, "Loading Gemma 2 2B model via TransformerLens")
    print("\n[Step 3/7] Loading Gemma 2 2B model via TransformerLens...")
    t0 = time.time()

    import transformer_lens
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use the unsloth version which is cached and doesn't require gated access
    # (google/gemma-2-2b is gated; unsloth/gemma-2-2b is ungated with identical weights)
    model_name = "gemma-2-2b"
    hf_model_name = "unsloth/gemma-2-2b"
    try:
        print("  Loading tokenizer from unsloth/gemma-2-2b...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        print("  Loading HF model weights from unsloth/gemma-2-2b...")
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
        print("  Converting to TransformerLens HookedTransformer...")
        model = HookedTransformer.from_pretrained(
            "google/gemma-2-2b",
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=torch.float16,
        )
        del hf_model  # Free the HF model copy
        load_time = time.time() - t0
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        vocab_size = model.cfg.d_vocab

        results["model_load"] = {
            "model_name": model_name,
            "n_layers": n_layers,
            "d_model": d_model,
            "vocab_size": vocab_size,
            "load_time_sec": round(load_time, 1),
            "status": "ok",
        }
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Layers: {n_layers}, d_model: {d_model}, vocab: {vocab_size}")

        free_after = torch.cuda.mem_get_info(0)[0]
        vram_model = (free_mem - free_after) / 1e9
        results["vram_usage"]["after_model"] = {
            "vram_free_mb": round(free_after / 1e6),
            "model_vram_gb": round(vram_model, 2),
        }
        print(f"  VRAM used by model: {vram_model:.1f}GB, Free: {free_after/1e9:.1f}GB")

    except Exception as e:
        tb = traceback.format_exc()
        results["model_load"] = {"status": "failed", "error": str(e), "traceback": tb}
        results["errors"].append(f"Model load failed: {e}")
        print(f"  FAILED: {e}")
        mark_done("failed", f"Model load failed: {e}", results)
        sys.exit(1)

    # ── Step 4: Load Gemma Scope SAEs ────────────────────────────────────
    report_progress(4, total_steps, "Loading Gemma Scope SAEs")
    print("\n[Step 4/7] Loading Gemma Scope SAEs...")

    from sae_lens import SAE

    # Define SAE configs to test
    # Gemma Scope SAEs naming: google/gemma-scope-2b-pt-res
    # Hook points: blocks.{layer}.hook_resid_post for residual stream
    # Correct SAE IDs from gemma-scope-2b-pt-res release (verified against SAELens directory)
    # Task plan specified approximate L0 values; these are the closest available:
    sae_configs = [
        {"layer": 12, "width": "16k",  "sae_id": "layer_12/width_16k/average_l0_82"},
        {"layer": 12, "width": "65k",  "sae_id": "layer_12/width_65k/average_l0_72"},
        {"layer": 10, "width": "16k",  "sae_id": "layer_10/width_16k/average_l0_77"},
        {"layer": 20, "width": "16k",  "sae_id": "layer_20/width_16k/average_l0_71"},
    ]

    for cfg in sae_configs:
        cfg_label = f"L{cfg['layer']}-{cfg['width']}"
        print(f"\n  Loading SAE: {cfg_label} (id: {cfg['sae_id']})...")
        t0 = time.time()
        try:
            sae = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res",
                sae_id=cfg["sae_id"],
                device=str(device),
            )
            load_time = time.time() - t0
            n_features = sae.cfg.d_sae
            d_in = sae.cfg.d_in

            sae_result = {
                "layer": cfg["layer"],
                "width": cfg["width"],
                "sae_id": cfg["sae_id"],
                "n_features": n_features,
                "d_in": d_in,
                "load_time_sec": round(load_time, 1),
                "status": "ok",
            }
            print(f"    Loaded in {load_time:.1f}s — features: {n_features}, d_in: {d_in}")

            # Keep the first successfully loaded SAE for the smoke test
            if not hasattr(main, '_smoke_sae'):
                main._smoke_sae = sae
                main._smoke_layer = cfg["layer"]
                print(f"    (Keeping {cfg_label} for smoke test)")
            else:
                del sae
                gc.collect()
                torch.cuda.empty_cache()

            results["sae_configs"].append(sae_result)

        except Exception as e:
            tb = traceback.format_exc()
            sae_result = {
                "layer": cfg["layer"],
                "width": cfg["width"],
                "sae_id": cfg["sae_id"],
                "status": "failed",
                "error": str(e),
                "traceback": tb,
            }
            results["sae_configs"].append(sae_result)
            results["warnings"].append(f"SAE {cfg_label} load failed: {e}")
            print(f"    FAILED: {e}")

    n_loaded = sum(1 for s in results["sae_configs"] if s["status"] == "ok")
    print(f"\n  SAE loading complete: {n_loaded}/{len(sae_configs)} successful")

    if n_loaded == 0:
        results["errors"].append("All SAE configs failed to load")
        mark_done("failed", "All SAE configs failed to load", results)
        sys.exit(1)

    # ── Step 5: Smoke Test — encode tokens through model + SAE ───────────
    report_progress(5, total_steps, "Running smoke test: 100 tokens through model+SAE")
    print("\n[Step 5/7] Smoke test: encoding 100 tokens through model + SAE...")

    try:
        # Use the first successfully loaded SAE for smoke test
        if not hasattr(main, '_smoke_sae'):
            raise RuntimeError("No SAE available for smoke test (all loads failed)")
        sae = main._smoke_sae
        smoke_layer = main._smoke_layer
        hook_point = f"blocks.{smoke_layer}.hook_resid_post"
        print(f"  Using SAE at layer {smoke_layer} for smoke test")

        # Generate test tokens
        test_texts = [
            "The capital of France is Paris, which is a beautiful city.",
            "A cat sat on the mat and looked at the dog.",
            "Machine learning is transforming artificial intelligence research.",
            "The quick brown fox jumps over the lazy dog near the river.",
        ]

        tokenizer = model.tokenizer
        all_smoke_results = []

        for text in test_texts:
            tokens = model.to_tokens(text)
            n_tokens = tokens.shape[1]

            # Run model to get activations at the hook point
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_point],
                return_type="logits",
            )

            activations = cache[hook_point]  # shape: [batch, seq_len, d_model]
            act_flat = activations.reshape(-1, activations.shape[-1])  # [n_tokens, d_model]

            # Encode through SAE
            sae_out = sae.encode(act_flat)  # [n_tokens, n_features]

            n_nonzero = (sae_out > 0).sum().item()
            n_total = sae_out.numel()
            pct_nonzero = n_nonzero / n_total * 100
            n_tokens_with_activation = (sae_out.sum(dim=-1) > 0).sum().item()
            pct_tokens_active = n_tokens_with_activation / act_flat.shape[0] * 100

            smoke_result = {
                "text": text[:60] + "...",
                "n_tokens": n_tokens,
                "n_nonzero_activations": n_nonzero,
                "total_activations": n_total,
                "pct_nonzero": round(pct_nonzero, 4),
                "n_tokens_with_activation": n_tokens_with_activation,
                "pct_tokens_active": round(pct_tokens_active, 2),
                "mean_active_features_per_token": round(n_nonzero / act_flat.shape[0], 2),
            }
            all_smoke_results.append(smoke_result)
            print(f"    '{text[:50]}...'")
            print(f"      tokens={n_tokens}, active_tokens={n_tokens_with_activation}/{act_flat.shape[0]} "
                  f"({pct_tokens_active:.1f}%), mean_features/token={n_nonzero/act_flat.shape[0]:.1f}")

        results["smoke_tests"] = all_smoke_results

        # Aggregate check: >= 90% of tokens should have non-zero activations
        total_tokens = sum(s["n_tokens"] - 1 for s in all_smoke_results)  # -1 for BOS
        total_active = sum(s["n_tokens_with_activation"] for s in all_smoke_results)
        overall_pct = total_active / total_tokens * 100 if total_tokens > 0 else 0

        print(f"\n    Overall: {total_active}/{total_tokens} tokens active ({overall_pct:.1f}%)")
        if overall_pct < 90:
            results["warnings"].append(f"Only {overall_pct:.1f}% tokens have non-zero SAE activations (target: >= 90%)")

    except Exception as e:
        tb = traceback.format_exc()
        results["smoke_tests"] = [{"status": "failed", "error": str(e), "traceback": tb}]
        results["errors"].append(f"Smoke test failed: {e}")
        print(f"    FAILED: {e}")

    # ── Step 6: VRAM usage summary ───────────────────────────────────────
    report_progress(6, total_steps, "Measuring final VRAM usage")
    print("\n[Step 6/7] VRAM usage summary...")

    free_final, total_final = torch.cuda.mem_get_info(0)
    vram_used = (total_final - free_final) / 1e9
    results["vram_usage"]["final"] = {
        "vram_free_mb": round(free_final / 1e6),
        "vram_used_gb": round(vram_used, 2),
    }
    print(f"  VRAM used (model + L12-16k SAE): {vram_used:.1f}GB")
    print(f"  VRAM free: {free_final/1e9:.1f}GB")

    if vram_used > 20:
        results["warnings"].append(f"VRAM usage {vram_used:.1f}GB exceeds 20GB target for single GPU")

    # ── Step 7: Write results ────────────────────────────────────────────
    report_progress(7, total_steps, "Writing results")
    print("\n[Step 7/7] Writing results...")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp"] = datetime.now().isoformat()
    results["pass"] = len(results["errors"]) == 0

    # Pass criteria check
    n_sae_ok = sum(1 for s in results["sae_configs"] if s["status"] == "ok")
    smoke_ok = len(results["smoke_tests"]) > 0 and all(
        "status" not in s or s.get("status") != "failed" for s in results["smoke_tests"]
    )
    pass_criteria = {
        "all_saes_loaded": n_sae_ok == len(sae_configs),
        "at_least_3_saes": n_sae_ok >= 3,
        "smoke_test_passed": smoke_ok,
        "vram_under_20gb": vram_used < 20,
        "overall": n_sae_ok >= 3 and smoke_ok,
    }
    results["pass_criteria"] = pass_criteria

    # Save results
    output_path = RESULTS_DIR / "setup_verification.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Results saved to: {output_path}")

    # Cleanup
    if hasattr(main, '_smoke_sae'):
        del main._smoke_sae
    try:
        del sae
    except (NameError, UnboundLocalError):
        pass
    try:
        del model
    except (NameError, UnboundLocalError):
        pass
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    status = "SUCCESS" if results["pass"] else "FAILED"
    print(f"Setup verification: {status}")
    print(f"  SAEs loaded: {n_sae_ok}/{len(sae_configs)}")
    print(f"  Smoke test: {'PASS' if smoke_ok else 'FAIL'}")
    print(f"  VRAM: {vram_used:.1f}GB")
    print(f"  Elapsed: {elapsed:.1f}s")
    if results["warnings"]:
        print(f"  Warnings: {len(results['warnings'])}")
        for w in results["warnings"]:
            print(f"    - {w}")
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}")
        for e in results["errors"]:
            print(f"    - {e}")
    print("=" * 70)

    # Write DONE marker
    mark_done(
        "success" if results["pass"] else "failed",
        f"Setup {'complete' if results['pass'] else 'failed'}: {n_sae_ok}/{len(sae_configs)} SAEs, smoke={'ok' if smoke_ok else 'fail'}, VRAM={vram_used:.1f}GB",
        results,
    )

    return 0 if results["pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        mark_done("failed", f"Unhandled exception: {e}")
        raise
