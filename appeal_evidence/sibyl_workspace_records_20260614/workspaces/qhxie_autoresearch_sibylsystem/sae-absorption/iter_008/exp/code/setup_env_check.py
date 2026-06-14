#!/usr/bin/env python3
"""
Environment Verification for Iteration 9 (FULL MODE).
Quick sanity check: SAELens, TransformerLens, GPU, Gemma 2B, RAVEL, sae_spelling.
All confirmed working in iter_008 pilot -- this is a fast sanity check.
Output: exp/results/setup_check.json
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---- Config ----
TASK_ID = "setup_env_check"
WORKSPACE = Path(os.environ.get("WORKSPACE", "."))
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Write PID file for recovery detection
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

start_time = datetime.now()

def report_progress(step, total_steps, status="running", detail=""):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "loss": None,
        "metric": {"status": status, "detail": detail},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ---- Begin Checks ----
checks = {}
total_checks = 10
current_check = 0

print("=" * 70)
print("  ENVIRONMENT VERIFICATION -- Iteration 9 (Full Mode)")
print("=" * 70)

# Check 1: Python version
current_check += 1
report_progress(current_check, total_checks, detail="Checking Python version")
try:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python_version"] = {"status": "OK", "version": py_ver}
    print(f"[{current_check}/{total_checks}] Python version: {py_ver} ✓")
except Exception as e:
    checks["python_version"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] Python version: FAIL - {e}")

# Check 2: PyTorch + CUDA
current_check += 1
report_progress(current_check, total_checks, detail="Checking PyTorch + CUDA")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if cuda_available else 0
    torch_ver = torch.__version__
    checks["pytorch_cuda"] = {
        "status": "OK" if cuda_available else "FAIL",
        "torch_version": torch_ver,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "cuda_version": torch.version.cuda if cuda_available else "N/A",
    }
    print(f"[{current_check}/{total_checks}] PyTorch {torch_ver}, CUDA: {cuda_available}, GPU: {gpu_name}, VRAM: {vram_gb}GB ✓")
except Exception as e:
    checks["pytorch_cuda"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] PyTorch/CUDA: FAIL - {e}")

# Check 3: TransformerLens
current_check += 1
report_progress(current_check, total_checks, detail="Checking TransformerLens")
try:
    import transformer_lens
    tl_ver = transformer_lens.__version__ if hasattr(transformer_lens, '__version__') else "unknown"
    checks["transformer_lens"] = {"status": "OK", "version": tl_ver}
    print(f"[{current_check}/{total_checks}] TransformerLens {tl_ver} ✓")
except Exception as e:
    checks["transformer_lens"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] TransformerLens: FAIL - {e}")

# Check 4: SAELens
current_check += 1
report_progress(current_check, total_checks, detail="Checking SAELens")
try:
    import sae_lens
    sl_ver = sae_lens.__version__ if hasattr(sae_lens, '__version__') else "unknown"
    checks["sae_lens"] = {"status": "OK", "version": sl_ver}
    print(f"[{current_check}/{total_checks}] SAELens {sl_ver} ✓")
except Exception as e:
    checks["sae_lens"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] SAELens: FAIL - {e}")

# Check 5: sae_spelling
current_check += 1
report_progress(current_check, total_checks, detail="Checking sae_spelling")
try:
    import sae_spelling
    ss_ver = sae_spelling.__version__ if hasattr(sae_spelling, '__version__') else "unknown"
    checks["sae_spelling"] = {"status": "OK", "version": ss_ver}
    print(f"[{current_check}/{total_checks}] sae_spelling {ss_ver} ✓")
except Exception as e:
    checks["sae_spelling"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] sae_spelling: FAIL - {e}")

# Check 6: scikit-learn + scipy + statsmodels
current_check += 1
report_progress(current_check, total_checks, detail="Checking sklearn/scipy/statsmodels")
try:
    import sklearn
    import scipy
    import statsmodels
    checks["stats_packages"] = {
        "status": "OK",
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "statsmodels_version": statsmodels.__version__,
    }
    print(f"[{current_check}/{total_checks}] sklearn {sklearn.__version__}, scipy {scipy.__version__}, statsmodels {statsmodels.__version__} ✓")
except Exception as e:
    checks["stats_packages"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] Stats packages: FAIL - {e}")

# Check 7: datasets (HuggingFace)
current_check += 1
report_progress(current_check, total_checks, detail="Checking HuggingFace datasets")
try:
    import datasets
    ds_ver = datasets.__version__
    checks["hf_datasets"] = {"status": "OK", "version": ds_ver}
    print(f"[{current_check}/{total_checks}] HuggingFace datasets {ds_ver} ✓")
except Exception as e:
    checks["hf_datasets"] = {"status": "FAIL", "error": str(e)}
    print(f"[{current_check}/{total_checks}] HuggingFace datasets: FAIL - {e}")

# Check 8: Gemma 2 2B model accessibility
current_check += 1
report_progress(current_check, total_checks, detail="Checking Gemma 2 2B model")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Check if unsloth/gemma-2-2b is cached or accessible
    # We just check the tokenizer to avoid loading the full model (fast check)
    model_name = "unsloth/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    checks["gemma_2b_model"] = {
        "status": "OK",
        "model_name": model_name,
        "vocab_size": vocab_size,
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    print(f"[{current_check}/{total_checks}] Gemma 2 2B tokenizer loaded (vocab: {vocab_size}) ✓")
    del tokenizer
except Exception as e:
    checks["gemma_2b_model"] = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}
    print(f"[{current_check}/{total_checks}] Gemma 2 2B: FAIL - {e}")

# Check 9: RAVEL dataset accessibility
current_check += 1
report_progress(current_check, total_checks, detail="Checking RAVEL dataset")
try:
    from datasets import load_dataset, get_dataset_config_names
    # Check available configs first
    configs = get_dataset_config_names("hij/ravel")
    # Load the city_entity config (used for entity-attribute hierarchies)
    ravel_entity = load_dataset("hij/ravel", "city_entity", split="train[:10]")
    entity_cols = ravel_entity.column_names
    n_entity_rows = len(ravel_entity)
    # Also load prompts config
    ravel_prompt = load_dataset("hij/ravel", "city_prompt", split="train[:10]")
    prompt_cols = ravel_prompt.column_names
    checks["ravel_dataset"] = {
        "status": "OK",
        "source": "hij/ravel",
        "available_configs": configs,
        "city_entity_columns": entity_cols,
        "city_prompt_columns": prompt_cols,
        "sample_entity": {k: str(v)[:100] for k, v in ravel_entity[0].items()} if n_entity_rows > 0 else {},
        "sample_prompt": {k: str(v)[:100] for k, v in ravel_prompt[0].items()} if len(ravel_prompt) > 0 else {},
    }
    print(f"[{current_check}/{total_checks}] RAVEL dataset loaded (configs: {configs}, entity cols: {entity_cols}) ✓")
    del ravel_entity, ravel_prompt
except Exception as e:
    checks["ravel_dataset"] = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}
    print(f"[{current_check}/{total_checks}] RAVEL dataset: FAIL - {e}")

# Check 10: Gemma Scope SAE availability via SAELens
current_check += 1
report_progress(current_check, total_checks, detail="Checking Gemma Scope SAE configs")
try:
    # Verify SAELens can resolve Gemma Scope release IDs
    from sae_lens import SAE
    # Just verify the SAE class is importable and list expected configs
    expected_configs = [
        "gemma-scope-2b-pt-res-canonical/layer_12/width_16k/canonical",
        "gemma-scope-2b-pt-res-canonical/layer_12/width_65k/canonical",
    ]
    checks["gemma_scope_sae"] = {
        "status": "OK",
        "sae_class": "SAE",
        "expected_configs": expected_configs,
        "note": "Full SAE loading deferred to experiment tasks to avoid unnecessary VRAM usage.",
    }
    print(f"[{current_check}/{total_checks}] Gemma Scope SAE class importable, {len(expected_configs)} target configs ✓")
except Exception as e:
    checks["gemma_scope_sae"] = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}
    print(f"[{current_check}/{total_checks}] Gemma Scope SAE: FAIL - {e}")

# ---- Summarize ----
end_time = datetime.now()
elapsed = (end_time - start_time).total_seconds()

n_ok = sum(1 for c in checks.values() if c.get("status") == "OK")
n_fail = sum(1 for c in checks.values() if c.get("status") == "FAIL")
overall = "PASS" if n_fail == 0 else "FAIL"

# Check for critical failures (packages needed for experiments)
critical_checks = ["pytorch_cuda", "transformer_lens", "sae_lens", "sae_spelling", "gemma_2b_model"]
critical_fails = [k for k in critical_checks if checks.get(k, {}).get("status") != "OK"]

if critical_fails:
    overall = "CRITICAL_FAIL"
    print(f"\n*** CRITICAL FAILURES: {critical_fails} ***")

result = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "overall_status": overall,
    "checks_passed": n_ok,
    "checks_failed": n_fail,
    "total_checks": total_checks,
    "critical_failures": critical_fails,
    "elapsed_seconds": round(elapsed, 1),
    "checks": checks,
    "environment": {
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "workspace": str(WORKSPACE),
    },
    "pass_criteria": "All critical packages importable, GPU available, Gemma 2B loadable, RAVEL accessible, sae_spelling importable.",
}

# Write results
output_path = RESULTS_DIR / "setup_check.json"
output_path.write_text(json.dumps(result, indent=2))

print(f"\n{'=' * 70}")
print(f"  RESULT: {overall} ({n_ok}/{total_checks} checks passed)")
print(f"  Elapsed: {elapsed:.1f}s")
print(f"  Output: {output_path}")
print(f"{'=' * 70}")

# Write done marker
mark_done(
    status="success" if overall != "CRITICAL_FAIL" else "failed",
    summary=f"{overall}: {n_ok}/{total_checks} checks passed in {elapsed:.1f}s. Critical fails: {critical_fails or 'none'}"
)

# Exit with appropriate code
sys.exit(0 if overall != "CRITICAL_FAIL" else 1)
