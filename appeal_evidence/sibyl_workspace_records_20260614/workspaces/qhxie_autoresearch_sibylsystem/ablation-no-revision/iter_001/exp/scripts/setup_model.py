#!/usr/bin/env python3
"""
Setup script: Download and verify Qwen2.5-Math-7B-Instruct on Blackwell GPU.
Task: setup_model
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_PATH = f"{WORKSPACE}/shared/models/Qwen2.5-Math-7B-Instruct"
REQUESTED_GPU = 3  # Requested GPU from orchestrator
RESULTS_DIR = f"{WORKSPACE}/exp/results"
TASK_ID = "setup_model"

# Handle CUDA_VISIBLE_DEVICES mapping: when set, single GPU becomes index 0
if "CUDA_VISIBLE_DEVICES" in os.environ:
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) == 1:
        GPU_ID = 0  # Single GPU, use index 0
    else:
        GPU_ID = int(visible[0])  # First visible GPU
else:
    GPU_ID = REQUESTED_GPU

def setup():
    """Download and verify the model."""
    results = {
        "task_id": TASK_ID,
        "status": "running",
        "model_name": MODEL_NAME,
        "start_time": datetime.now().isoformat(),
        "requested_gpu_id": REQUESTED_GPU,
        "actual_gpu_id": GPU_ID,
        "gpu_name": torch.cuda.get_device_name(GPU_ID) if torch.cuda.is_available() else "N/A",
        "vram_total_mb": torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**2 if torch.cuda.is_available() else 0,
    }

    print(f"[setup_model] Starting at {datetime.now().isoformat()}")
    print(f"[setup_model] GPU: {torch.cuda.get_device_name(GPU_ID) if torch.cuda.is_available() else 'N/A'}")
    print(f"[setup_model] Model: {MODEL_NAME}")
    print(f"[setup_model] Target path: {MODEL_PATH}")

    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Write PID file
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[setup_model] PID: {os.getpid()}")

    try:
        # Check if model already exists
        if os.path.exists(MODEL_PATH):
            print(f"[setup_model] Model directory exists, checking contents...")
            config_file = Path(MODEL_PATH) / "config.json"
            if config_file.exists():
                print(f"[setup_model] Model already downloaded, skipping download.")
                results["download_skipped"] = True
            else:
                print(f"[setup_model] Incomplete download, will re-download.")
        else:
            print(f"[setup_model] Downloading model...")

        # Set CUDA device
        torch.cuda.set_device(GPU_ID)

        # Download and cache tokenizer
        print(f"[setup_model] Loading tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "config.json")) else None
        )
        tokenizer_time = time.time() - tokenizer_start
        results["tokenizer_load_time_sec"] = tokenizer_time
        print(f"[setup_model] Tokenizer loaded in {tokenizer_time:.1f}s")

        # Download and cache model weights
        print(f"[setup_model] Loading model weights...")
        model_start = time.time()

        # First try to load from cache, then download if needed
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{GPU_ID}",
                trust_remote_code=True,
                cache_dir=None  # Let HF use default cache
            )
        except OSError as e:
            # If download fails, try with explicit cache dir
            print(f"[setup_model] Initial load failed, retrying with explicit cache_dir...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{GPU_ID}",
                trust_remote_code=True,
                cache_dir=MODEL_PATH
            )

        model_time = time.time() - model_start
        results["model_load_time_sec"] = model_time
        print(f"[setup_model] Model loaded in {model_time:.1f}s")

        # Get VRAM usage
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated(GPU_ID) / 1024**2
        vram_total = torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**2
        results["vram_used_mb"] = vram_used
        results["vram_total_mb"] = vram_total
        results["vram_utilization_pct"] = 100 * vram_used / vram_total
        print(f"[setup_model] VRAM: {vram_used:.0f}MB / {vram_total:.0f}MB ({results['vram_utilization_pct']:.1f}%)")

        # Generate test output
        print(f"[setup_model] Running test generation...")
        test_prompt = "Solve: 2 + 2 = ?"
        messages = [{"role": "user", "content": test_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(f"cuda:{GPU_ID}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results["test_output"] = response
        results["test_passed"] = len(response) > 0
        print(f"[setup_model] Test output: {response[:100]}...")
        print(f"[setup_model] Test passed: {results['test_passed']}")

        # GPU profiling
        results["gpu_profile"] = {
            "gpu_name": torch.cuda.get_device_name(GPU_ID),
            "vram_total_mb": vram_total,
            "vram_used_mb": vram_used,
            "vram_free_mb": vram_total - vram_used,
            "utilization_pct": results["vram_utilization_pct"]
        }

        # Mark success
        results["status"] = "success"
        results["end_time"] = datetime.now().isoformat()
        results["total_time_sec"] = model_time + tokenizer_time

        # Write progress
        progress = {
            "task_id": TASK_ID,
            "status": "completed",
            "test_passed": results["test_passed"],
            "updated_at": datetime.now().isoformat(),
        }
        progress_file = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
        progress_file.write_text(json.dumps(progress))

        # Write DONE marker
        done_marker = Path(RESULTS_DIR) / f"{TASK_ID}_DONE"
        done_marker.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "success",
            "summary": f"Model loaded successfully, VRAM: {vram_used:.0f}MB",
            "timestamp": datetime.now().isoformat(),
        }))

        # Cleanup PID file
        if pid_file.exists():
            pid_file.unlink()

        print(f"[setup_model] SUCCESS: Model verified and ready")
        return results

    except Exception as e:
        print(f"[setup_model] ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        results["end_time"] = datetime.now().isoformat()

        # Write DONE marker for failure
        done_marker = Path(RESULTS_DIR) / f"{TASK_ID}_DONE"
        done_marker.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "summary": str(e),
            "timestamp": datetime.now().isoformat(),
        }))

        # Cleanup PID file
        pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
        if pid_file.exists():
            pid_file.unlink()

        return results

if __name__ == "__main__":
    results = setup()

    # Write results
    results_file = Path(RESULTS_DIR) / f"{TASK_ID}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[setup_model] Results saved to {results_file}")

    # Exit with appropriate code
    sys.exit(0 if results["status"] == "success" else 1)
