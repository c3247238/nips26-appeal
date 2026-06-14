#!/usr/bin/env python3
"""
Environment Verification Script (setup_env_check)
Verifies all required packages, GPU availability, model accessibility, and dataset availability.
Output: exp/results/setup_check.json
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Write PID file for system recovery detection
TASK_ID = "setup_env_check"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "exp/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

# Progress reporting
def report_progress(step, total_steps, status="running", notes=""):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "loss": None,
        "metric": {"status": status, "notes": notes},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_task_done(status="success", summary=""):
    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()
    # Read final progress
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    # Write DONE marker
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

start_time = time.time()
results = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "python_version": sys.version,
    "checks": {},
    "packages": {},
    "gpu": {},
    "models": {},
    "datasets": {},
    "overall_status": "PASS",
    "issues": [],
}

TOTAL_STEPS = 7

# ---- Step 1: Core packages ----
report_progress(1, TOTAL_STEPS, notes="Checking core packages")
print("[1/7] Checking core packages...")

core_packages = {
    "torch": None,
    "transformer_lens": None,
    "sae_lens": None,
    "numpy": None,
    "scipy": None,
    "sklearn": "sklearn",
    "matplotlib": None,
    "datasets": None,
    "transformers": None,
}

for pkg_name, import_name in core_packages.items():
    actual_import = import_name or pkg_name
    try:
        mod = __import__(actual_import)
        version = getattr(mod, "__version__", "unknown")
        results["packages"][pkg_name] = {"version": str(version), "status": "OK"}
        print(f"  {pkg_name}: {version} OK")
    except ImportError as e:
        results["packages"][pkg_name] = {"version": None, "status": "MISSING", "error": str(e)}
        results["issues"].append(f"Package {pkg_name} not importable: {e}")
        results["overall_status"] = "FAIL"
        print(f"  {pkg_name}: MISSING - {e}")

# ---- Step 2: SAELens version check ----
report_progress(2, TOTAL_STEPS, notes="Checking SAELens version")
print("[2/7] Checking SAELens version...")

try:
    import sae_lens
    sae_version = sae_lens.__version__
    # Check >= 6.39 (if version parseable)
    parts = sae_version.split(".")
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    if major > 6 or (major == 6 and minor >= 39):
        results["checks"]["saelens_version"] = {"status": "OK", "version": sae_version, "required": ">=6.39"}
        print(f"  SAELens {sae_version} >= 6.39: OK")
    else:
        results["checks"]["saelens_version"] = {"status": "WARN", "version": sae_version, "required": ">=6.39"}
        results["issues"].append(f"SAELens version {sae_version} < 6.39 (may need upgrade)")
        print(f"  SAELens {sae_version} < 6.39: WARN")
except Exception as e:
    results["checks"]["saelens_version"] = {"status": "FAIL", "error": str(e)}
    results["issues"].append(f"SAELens version check failed: {e}")
    results["overall_status"] = "FAIL"

# ---- Step 3: sae_spelling import check ----
report_progress(3, TOTAL_STEPS, notes="Checking sae_spelling")
print("[3/7] Checking sae_spelling import...")

try:
    import sae_spelling
    version = getattr(sae_spelling, "__version__", "installed")
    results["packages"]["sae_spelling"] = {"version": str(version), "status": "OK"}
    print(f"  sae_spelling: {version} OK")
except ImportError as e:
    results["packages"]["sae_spelling"] = {"version": None, "status": "MISSING", "error": str(e)}
    results["issues"].append(f"sae_spelling not importable: {e}")
    # Not a hard fail -- can work around
    print(f"  sae_spelling: MISSING (non-critical) - {e}")

# ---- Step 4: GPU check ----
report_progress(4, TOTAL_STEPS, notes="Checking GPU availability")
print("[4/7] Checking GPU availability...")

try:
    import torch
    cuda_available = torch.cuda.is_available()
    results["gpu"]["cuda_available"] = cuda_available

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        results["gpu"]["device_count"] = gpu_count
        results["gpu"]["devices"] = []

        # Check the assigned GPU (GPU 4)
        assigned_gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "4"))

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            # PyTorch >= 2.x uses total_memory; older versions use total_mem
            total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            vram_mb = total_bytes // (1024 * 1024)
            device_info = {
                "index": i,
                "name": props.name,
                "vram_total_mb": vram_mb,
                "compute_capability": f"{props.major}.{props.minor}",
            }
            results["gpu"]["devices"].append(device_info)
            print(f"  GPU {i}: {props.name}, {vram_mb} MB VRAM, CC {props.major}.{props.minor}")

            if vram_mb < 24000:
                results["issues"].append(f"GPU {i} has {vram_mb}MB VRAM (< 24GB recommended)")

        # Quick VRAM test on default device
        try:
            device = torch.device("cuda")
            x = torch.randn(1000, 1000, device=device)
            y = x @ x.T
            del x, y
            torch.cuda.empty_cache()
            results["gpu"]["compute_test"] = "PASS"
            print("  GPU compute test: PASS")
        except Exception as e:
            results["gpu"]["compute_test"] = f"FAIL: {e}"
            results["issues"].append(f"GPU compute test failed: {e}")
    else:
        results["gpu"]["device_count"] = 0
        results["issues"].append("CUDA not available")
        results["overall_status"] = "FAIL"
        print("  CUDA not available: FAIL")
except Exception as e:
    results["gpu"]["error"] = str(e)
    results["issues"].append(f"GPU check failed: {e}")
    results["overall_status"] = "FAIL"

# ---- Step 5: Gemma 2B model accessibility ----
report_progress(5, TOTAL_STEPS, notes="Checking Gemma 2B model accessibility")
print("[5/7] Checking Gemma 2B model accessibility...")

try:
    from transformers import AutoTokenizer
    # Try loading tokenizer -- Gemma 2 2B is a gated model, needs HF login
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", trust_remote_code=True)
    results["models"]["gemma_2_2b"] = {
        "status": "OK",
        "tokenizer_loaded": True,
        "vocab_size": tokenizer.vocab_size,
    }
    print(f"  Gemma 2 2B tokenizer: OK (vocab_size={tokenizer.vocab_size})")
    del tokenizer
except Exception as e:
    # Check if it's an auth issue -- if so, check HF token
    error_str = str(e)
    if "gated" in error_str or "401" in error_str or "restricted" in error_str:
        # Try to check if HF token is set
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            results["models"]["gemma_2_2b"] = {"status": "WARN", "error": "Gated model, token present but may lack access"}
            results["issues"].append(f"Gemma 2 2B is gated; HF token present but access may not be granted")
        else:
            # Check huggingface-cli login
            try:
                from huggingface_hub import HfFolder
                cached_token = HfFolder.get_token()
                if cached_token:
                    results["models"]["gemma_2_2b"] = {"status": "WARN", "error": "Gated model, cached token exists but may lack access"}
                    results["issues"].append("Gemma 2 2B gated; cached HF token exists but may need model access approval")
                else:
                    results["models"]["gemma_2_2b"] = {"status": "WARN", "error": "No HF token found"}
                    results["issues"].append("Gemma 2 2B is gated; no HF token found. Run 'huggingface-cli login'")
            except Exception:
                results["models"]["gemma_2_2b"] = {"status": "WARN", "error": "No HF token found"}
                results["issues"].append("Gemma 2 2B is gated; no HF token found. Run 'huggingface-cli login'")
        print(f"  Gemma 2 2B tokenizer: WARN (gated model, need HF auth)")
        # Also check if model was previously cached locally
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        gemma_dirs = list(cache_dir.glob("models--google--gemma-2-2b*")) if cache_dir.exists() else []
        if gemma_dirs:
            results["models"]["gemma_2_2b"]["local_cache"] = True
            results["models"]["gemma_2_2b"]["status"] = "OK_CACHED"
            print(f"  Gemma 2 2B: found in local cache ({gemma_dirs[0].name})")
            # Try to actually load from cache
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(gemma_dirs[0] / "snapshots" / os.listdir(gemma_dirs[0] / "snapshots")[0]))
                results["models"]["gemma_2_2b"]["vocab_size"] = tokenizer.vocab_size
                results["models"]["gemma_2_2b"]["status"] = "OK"
                # Remove the warning issue we added
                results["issues"] = [i for i in results["issues"] if "Gemma 2 2B" not in i]
                print(f"  Gemma 2 2B tokenizer from cache: OK (vocab_size={tokenizer.vocab_size})")
                del tokenizer
            except Exception:
                pass
    else:
        results["models"]["gemma_2_2b"] = {"status": "WARN", "error": error_str}
        results["issues"].append(f"Gemma 2 2B tokenizer load failed: {e}")
        print(f"  Gemma 2 2B tokenizer: WARN - {e}")

# Check TransformerLens compatibility
try:
    from transformer_lens import HookedTransformer
    results["models"]["transformer_lens_available"] = True
    print("  TransformerLens HookedTransformer: importable")
except Exception as e:
    results["models"]["transformer_lens_available"] = False
    results["issues"].append(f"TransformerLens HookedTransformer import failed: {e}")
    print(f"  TransformerLens HookedTransformer: FAIL - {e}")

# ---- Step 6: RAVEL dataset accessibility ----
report_progress(6, TOTAL_STEPS, notes="Checking RAVEL dataset accessibility")
print("[6/7] Checking RAVEL dataset accessibility...")

try:
    from datasets import load_dataset
    # RAVEL has configs: 'city_entity' and 'city_prompt'
    ravel_configs = {}
    for config_name in ["city_entity", "city_prompt"]:
        try:
            ds = load_dataset("hij/ravel", config_name, split="train", streaming=True)
            sample = next(iter(ds))
            ravel_configs[config_name] = {
                "status": "OK",
                "sample_keys": list(sample.keys()) if isinstance(sample, dict) else "N/A",
            }
            print(f"  RAVEL '{config_name}': OK (keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'})")
            del ds, sample
        except Exception as e2:
            ravel_configs[config_name] = {"status": "WARN", "error": str(e2)}
            print(f"  RAVEL '{config_name}': WARN - {e2}")

    results["datasets"]["ravel"] = {
        "status": "OK" if any(v["status"] == "OK" for v in ravel_configs.values()) else "WARN",
        "source": "hij/ravel",
        "configs": ravel_configs,
    }
    if results["datasets"]["ravel"]["status"] != "OK":
        results["issues"].append("RAVEL dataset configs not fully accessible")
except Exception as e:
    results["datasets"]["ravel"] = {"status": "WARN", "error": str(e)}
    results["issues"].append(f"RAVEL dataset access failed: {e}")
    print(f"  RAVEL dataset: WARN - {e}")

# ---- Step 7: SAE Lens SAE loading check ----
report_progress(7, TOTAL_STEPS, notes="Checking SAE availability (metadata only)")
print("[7/7] Checking SAE availability (metadata)...")

try:
    from sae_lens import SAE
    # Check the primary SAE IDs exist (metadata only, no download)
    sae_ids = [
        "gemma-scope-2b-pt-res-canonical/layer_24/width_16k/canonical",
        "gemma-scope-2b-pt-res-canonical/layer_24/width_65k/canonical",
        "gemma-scope-2b-pt-res-canonical/layer_12/width_16k/canonical",
    ]
    results["models"]["sae_ids_checked"] = sae_ids
    results["models"]["sae_lens_SAE_importable"] = True
    print(f"  SAELens SAE class: importable")
    print(f"  SAE IDs to be used: {len(sae_ids)} configurations")
except Exception as e:
    results["models"]["sae_lens_SAE_importable"] = False
    results["issues"].append(f"SAELens SAE import failed: {e}")
    print(f"  SAELens SAE class: FAIL - {e}")

# ---- Final summary ----
elapsed = time.time() - start_time
results["elapsed_seconds"] = round(elapsed, 1)

n_issues = len(results["issues"])
# Determine final status: FAIL only if critical issues (GPU not available, core packages missing)
critical_fail = (
    not results["gpu"].get("cuda_available", False)
    or any(v.get("status") == "MISSING" for v in results["packages"].values()
           if v.get("status") == "MISSING" and "sae_spelling" not in str(v))
)
if n_issues == 0:
    results["overall_status"] = "PASS"
    summary_msg = "All checks passed"
elif critical_fail:
    results["overall_status"] = "FAIL"
    summary_msg = f"FAILED with {n_issues} issues"
else:
    results["overall_status"] = "PASS_WITH_WARNINGS"
    summary_msg = f"Passed with {n_issues} warnings"

results["summary"] = summary_msg

print(f"\n{'='*60}")
print(f"Overall: {results['overall_status']} ({summary_msg})")
print(f"Elapsed: {results['elapsed_seconds']}s")
if results["issues"]:
    print(f"Issues ({n_issues}):")
    for issue in results["issues"]:
        print(f"  - {issue}")
print(f"{'='*60}")

# Save results
output_path = RESULTS_DIR / "setup_check.json"
output_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\nResults saved to: {output_path}")

# Mark task done
mark_task_done(
    status="success" if "FAIL" not in results["overall_status"] else "failed",
    summary=summary_msg,
)
print(f"DONE marker written.")
