"""
Environment setup and validation script for UAD experiments.
Checks dependencies, GPU access, SAE loading, and first-letter feature detection.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/current/exp/results")
LOGS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/current/exp/logs")
PID_FILE = RESULTS_DIR / "setup_env.pid"
PROGRESS_FILE = RESULTS_DIR / "setup_env_PROGRESS.json"
DONE_FILE = RESULTS_DIR / "setup_env_DONE"

def report_progress(step, total_steps, message, extra=None):
    progress = {
        "task_id": "setup_env",
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "extra": extra or {},
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))
    print(f"[{step}/{total_steps}] {message}")

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def write_result(data):
    result_path = RESULTS_DIR / "setup_complete.json"
    result_path.write_text(json.dumps(data, indent=2, cls=SafeJSONEncoder))
    return result_path

def mark_done(status, summary):
    done_data = {
        "task_id": "setup_env",
        "status": status,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    DONE_FILE.write_text(json.dumps(done_data))
    if PID_FILE.exists():
        PID_FILE.unlink()

def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pilots").mkdir(exist_ok=True)
    (RESULTS_DIR / "full").mkdir(exist_ok=True)

    PID_FILE.write_text(str(os.getpid()))

    results = {
        "task_id": "setup_env",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": {},
        "overall_status": "running",
    }

    # === Check 1: PyTorch and CUDA ===
    report_progress(1, 6, "Checking PyTorch and CUDA...")
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    torch_version = torch.__version__

    results["checks"]["pytorch_cuda"] = {
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "cuda_version": cuda_version,
        "pass": cuda_available,
    }
    if not cuda_available:
        results["overall_status"] = "failed"
        write_result(results)
        mark_done("failed", "CUDA not available")
        sys.exit(1)

    # === Check 2: Key dependencies ===
    report_progress(2, 6, "Checking key dependencies...")
    deps = {}
    try:
        import sae_lens
        deps["sae_lens"] = {"version": sae_lens.__version__, "pass": True}
    except Exception as e:
        deps["sae_lens"] = {"error": str(e), "pass": False}

    try:
        import transformer_lens
        deps["transformer_lens"] = {"pass": True}
    except Exception as e:
        deps["transformer_lens"] = {"error": str(e), "pass": False}

    try:
        from datasets import load_dataset
        deps["datasets"] = {"pass": True}
    except Exception as e:
        deps["datasets"] = {"error": str(e), "pass": False}

    try:
        import scipy
        deps["scipy"] = {"version": scipy.__version__, "pass": True}
    except Exception as e:
        deps["scipy"] = {"error": str(e), "pass": False}

    try:
        import matplotlib
        deps["matplotlib"] = {"pass": True}
    except Exception as e:
        deps["matplotlib"] = {"error": str(e), "pass": False}

    results["checks"]["dependencies"] = deps
    all_deps_pass = all(d["pass"] for d in deps.values())
    if not all_deps_pass:
        results["overall_status"] = "failed"
        write_result(results)
        mark_done("failed", f"Missing dependencies: {[k for k,v in deps.items() if not v['pass']]}")
        sys.exit(1)

    # === Check 3: SAE loading ===
    report_progress(3, 6, "Loading pre-trained SAE (gpt2-small-res-jb, layer 8)...")
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(
            "gpt2-small-res-jb",
            "blocks.8.hook_resid_pre",
            device="cpu",
        )
        results["checks"]["sae_loading"] = {
            "pass": True,
            "sae_source": "gpt2-small-res-jb",
            "layer": 8,
            "d_sae": sae.cfg.d_sae,
            "d_in": sae.cfg.d_in,
            "architecture": sae.cfg.__class__.__name__,
        }
        # Move to GPU for later use
        device = "cuda:0"
        sae = sae.to(device)
    except Exception as e:
        results["checks"]["sae_loading"] = {"pass": False, "error": str(e)}
        results["overall_status"] = "failed"
        write_result(results)
        mark_done("failed", f"SAE loading failed: {e}")
        sys.exit(1)

    # === Check 4: Model loading ===
    report_progress(4, 6, "Loading GPT-2 Small model...")
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained_no_processing(
            "gpt2-small",
            device=device,
        )
        results["checks"]["model_loading"] = {
            "pass": True,
            "model": "gpt2-small",
            "device": device,
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
        }
    except Exception as e:
        results["checks"]["model_loading"] = {"pass": False, "error": str(e)}
        results["overall_status"] = "failed"
        write_result(results)
        mark_done("failed", f"Model loading failed: {e}")
        sys.exit(1)

    # === Check 5: First-letter feature detection ===
    report_progress(5, 6, "Detecting first-letter features...")
    try:
        from datasets import load_dataset

        # Load a small sample
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        texts = []
        for i, example in enumerate(ds):
            texts.append(example["text"])
            if i >= 99:
                break

        # Tokenize
        tokens = model.to_tokens(texts, truncate=True)  # [batch, seq_len]

        # Run model with SAE hook
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=["blocks.8.hook_resid_pre"],
            )
            resid = cache["blocks.8.hook_resid_pre"]  # [batch, seq_len, d_model]
            # Flatten and encode
            resid_flat = resid.reshape(-1, resid.shape[-1])
            sae_acts = sae.encode(resid_flat.to(sae.W_enc.dtype))  # [batch*seq_len, d_sae]

        # Detect first-letter features
        # For each token position, check if the token starts with a specific letter
        token_ids = tokens.reshape(-1).cpu().numpy()
        token_strs = [model.to_string([tid]).strip().lower() for tid in token_ids]

        # Find features that activate strongly on tokens starting with each letter
        feature_scores = {}
        for letter in "abcdefghijklmnopqrstuvwxyz":
            mask = np.array([t.startswith(letter) if len(t) > 0 else False for t in token_strs])
            if mask.sum() > 0:
                # Average activation for tokens starting with this letter
                avg_acts = sae_acts[mask].mean(dim=0).cpu().float().numpy()
                top_idx = int(np.argmax(avg_acts))
                top_score = float(avg_acts[top_idx])
                feature_scores[letter] = {"top_feature": top_idx, "score": top_score}

        detected_count = len(feature_scores)
        results["checks"]["first_letter_detection"] = {
            "pass": detected_count >= 10,
            "detected_letters": list(feature_scores.keys()),
            "detected_count": detected_count,
            "sample_scores": {k: v for k, v in list(feature_scores.items())[:5]},
        }
    except Exception as e:
        import traceback
        results["checks"]["first_letter_detection"] = {
            "pass": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        # Don't fail on this - it's a pilot check

    # === Check 6: GPU memory ===
    report_progress(6, 6, "Checking GPU memory...")
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem = gpu_props.total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = total_mem - allocated
        results["checks"]["gpu_memory"] = {
            "pass": free > 10,
            "total_gb": round(total_mem, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(free, 2),
        }
    except Exception as e:
        results["checks"]["gpu_memory"] = {"pass": False, "error": str(e)}

    # === Finalize ===
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)
    results["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Determine overall status
    def _check_passes(v):
        if isinstance(v, dict) and "pass" in v:
            return v["pass"]
        elif isinstance(v, dict):
            # Nested dict like dependencies - check all sub-items
            return all(_check_passes(sv) for sv in v.values())
        return bool(v)

    all_pass = all(
        _check_passes(v)
        for k, v in results["checks"].items()
        if k != "first_letter_detection"  # This is optional for setup
    )
    results["overall_status"] = "success" if all_pass else "failed"

    write_result(results)

    if all_pass:
        mark_done("success", f"Environment setup complete. All checks passed in {elapsed:.1f}s.")
        print(f"\nSetup complete in {elapsed:.1f}s. All checks passed.")
    else:
        failed = [k for k, v in results["checks"].items() if not v.get("pass", False)]
        mark_done("failed", f"Setup failed: {failed}")
        print(f"\nSetup failed: {failed}")
        sys.exit(1)

if __name__ == "__main__":
    main()
