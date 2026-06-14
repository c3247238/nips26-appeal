#!/usr/bin/env python3
"""Setup environment verification for CARD experiments (new iteration).

Checks:
1. GPU availability and VRAM on assigned GPU
2. Required packages with versions
3. Model loading (LLaDA-8B-Instruct) with dtype and attention backend
4. Dataset availability (GSM8K)
5. Single-sample forward pass + generation test
6. VRAM profiling for batch size estimation
7. Writes setup_verification.json
"""

import json
import os
import sys
import time
import gc
from pathlib import Path
from datetime import datetime

# Configuration
GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "5"))
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = "/home/ccwang/sibyl_system/projects/dlm-improve/exp/results"
VERIFICATION_PATH = "/home/ccwang/sibyl_system/projects/dlm-improve/exp/setup_verification.json"

results = {
    "task_id": "setup_env",
    "iteration": "current",
    "timestamp": datetime.now().isoformat(),
    "status": "running",
    "checks": {},
}

def check_packages():
    """Verify all required packages are installed."""
    packages = {}
    for pkg_name, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("scikit-learn", "sklearn"),
        ("accelerate", "accelerate"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            mod = __import__(import_name)
            packages[pkg_name] = mod.__version__
        except ImportError:
            packages[pkg_name] = "NOT_INSTALLED"

    # Check optional performance packages
    for pkg_name, import_name in [
        ("flash-attn", "flash_attn"),
        ("triton", "triton"),
    ]:
        try:
            mod = __import__(import_name)
            packages[pkg_name] = mod.__version__
        except ImportError:
            packages[pkg_name] = "NOT_INSTALLED (optional)"

    return packages


def check_gpu():
    """Check GPU availability and VRAM."""
    import torch
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "assigned_gpu": GPU_ID,
    }

    if torch.cuda.is_available():
        # Use device 0 since CUDA_VISIBLE_DEVICES remaps
        props = torch.cuda.get_device_properties(0)
        gpu_info.update({
            "device_name": props.name,
            "vram_total_mb": round(props.total_memory / 1024**2),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })

        # Check attention backends
        gpu_info["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        try:
            from transformers.utils import is_flash_attn_2_available
            gpu_info["flash_attn_2_available"] = is_flash_attn_2_available()
        except ImportError:
            gpu_info["flash_attn_2_available"] = False

    return gpu_info


def check_model():
    """Load model and run a single forward pass."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_info = {"path": MODEL_PATH}

    # Check model files exist
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        model_info["error"] = "Model path does not exist"
        return model_info, None, None

    safetensor_files = list(model_path.glob("*.safetensors"))
    model_info["num_safetensor_files"] = len(safetensor_files)

    # Load tokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model_info["tokenizer_load_sec"] = round(time.time() - t0, 2)
    model_info["vocab_size"] = tokenizer.vocab_size
    model_info["mask_token_id"] = tokenizer.mask_token_id if hasattr(tokenizer, "mask_token_id") else None

    # Check for special mask token (LLaDA uses 126336)
    if model_info["mask_token_id"] is None:
        # LLaDA-specific: mask token is usually at the end of vocab
        try:
            config = json.load(open(model_path / "config.json"))
            model_info["mask_token_id"] = config.get("mask_token_id", 126336)
        except:
            model_info["mask_token_id"] = 126336

    # Load model with best available attention
    t0 = time.time()
    attn_impl = "sdpa"  # PyTorch native, no extra install needed

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).to("cuda:0")
        model_info["attn_implementation"] = attn_impl
    except Exception as e:
        # Fallback: load without specifying attn
        print(f"SDPA load failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda:0")
        model_info["attn_implementation"] = "eager"

    model.eval()
    model_info["load_time_sec"] = round(time.time() - t0, 2)

    # VRAM after load
    torch.cuda.synchronize()
    vram_after = torch.cuda.memory_allocated(0) / 1024**2
    model_info["vram_after_load_mb"] = round(vram_after)
    model_info["dtype"] = str(model.dtype) if hasattr(model, "dtype") else "bfloat16"
    model_info["num_params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)

    return model_info, model, tokenizer


def check_single_sample(model, tokenizer):
    """Run single sample generation test."""
    import torch

    test_prompt = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
    prompt_len = inputs["input_ids"].shape[1]
    gen_len = 128

    # Create masked input for generation (DLM style)
    mask_token_id = 126336  # LLaDA mask token
    input_ids = inputs["input_ids"]

    # Append masked tokens for generation
    masked_gen = torch.full((1, gen_len), mask_token_id, dtype=torch.long, device="cuda:0")
    full_input = torch.cat([input_ids, masked_gen], dim=1)

    # Single forward pass timing
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model(full_input)
    torch.cuda.synchronize()
    forward_time = time.time() - t0

    # Get logits at masked positions
    logits = outputs.logits[:, prompt_len:, :]  # (1, gen_len, vocab)

    # Get predictions and confidence
    probs = torch.softmax(logits, dim=-1)
    top_conf, top_ids = probs.max(dim=-1)  # (1, gen_len)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (1, gen_len)

    # Decode predictions
    predicted_tokens = tokenizer.decode(top_ids[0], skip_special_tokens=True)

    # VRAM peak
    vram_peak = torch.cuda.max_memory_allocated(0) / 1024**2

    sample_test = {
        "prompt_tokens": prompt_len,
        "gen_length": gen_len,
        "total_seq_len": full_input.shape[1],
        "forward_pass_time_sec": round(forward_time, 3),
        "vram_peak_mb": round(vram_peak),
        "mean_confidence": round(top_conf.mean().item(), 4),
        "mean_entropy": round(entropy.mean().item(), 4),
        "predicted_text_sample": predicted_tokens[:100],
        "logits_shape": list(outputs.logits.shape),
    }

    return sample_test


def check_batch_capacity(model, tokenizer):
    """Estimate max batch size for typical workload."""
    import torch

    mask_token_id = 126336
    seq_len = 256  # typical for GSM8K

    # Get available VRAM
    torch.cuda.empty_cache()
    gc.collect()
    vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**2

    batch_profile = {
        "seq_len": seq_len,
        "vram_free_mb": round(vram_free),
    }

    # Binary search for max batch size
    lo, hi, best = 1, 128, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            dummy = torch.full((mid, seq_len), mask_token_id, dtype=torch.long, device="cuda:0")
            with torch.no_grad():
                _ = model(dummy)
            torch.cuda.synchronize()
            best = mid
            lo = mid + 1
            del dummy
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            batch_profile["error"] = str(e)
            break

    vram_peak = torch.cuda.max_memory_allocated(0) / 1024**2
    batch_profile["max_batch_size"] = best
    batch_profile["vram_peak_at_max_batch_mb"] = round(vram_peak)
    batch_profile["utilization_pct"] = round(vram_peak / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100, 1)

    return batch_profile


def check_dataset():
    """Verify GSM8K dataset is available."""
    from datasets import load_from_disk

    dataset_path = "/home/ccwang/sibyl_system/shared/datasets/gsm8k"
    ds_info = {"path": dataset_path}

    try:
        ds = load_from_disk(dataset_path)
        ds_info["num_samples"] = len(ds)
        ds_info["columns"] = list(ds.column_names)
        ds_info["sample_0_question"] = ds[0]["question"][:100] + "..."
        ds_info["status"] = "ok"
    except Exception as e:
        ds_info["error"] = str(e)
        ds_info["status"] = "error"

    return ds_info


def main():
    print("=" * 60)
    print("CARD Experiment Environment Verification")
    print("=" * 60)

    # 1. Packages
    print("\n[1/5] Checking packages...")
    results["checks"]["packages"] = check_packages()
    print(f"  Packages: {json.dumps(results['checks']['packages'], indent=2)}")

    # 2. GPU
    print("\n[2/5] Checking GPU...")
    results["checks"]["gpu"] = check_gpu()
    print(f"  GPU: {results['checks']['gpu'].get('device_name', 'N/A')}")
    print(f"  VRAM: {results['checks']['gpu'].get('vram_total_mb', 'N/A')} MB")
    print(f"  SDPA: {results['checks']['gpu'].get('sdpa_available', False)}")
    print(f"  Flash Attn 2: {results['checks']['gpu'].get('flash_attn_2_available', False)}")

    # 3. Model
    print("\n[3/5] Loading model...")
    model_info, model, tokenizer = check_model()
    results["checks"]["model"] = model_info
    if model is not None:
        print(f"  Model: {model_info['num_params_b']}B params, loaded in {model_info['load_time_sec']}s")
        print(f"  VRAM after load: {model_info['vram_after_load_mb']} MB")
        print(f"  Attention: {model_info.get('attn_implementation', 'unknown')}")
    else:
        print(f"  ERROR: {model_info.get('error', 'unknown')}")

    # 4. Single sample test
    if model is not None:
        print("\n[4/5] Running single sample test...")
        sample_test = check_single_sample(model, tokenizer)
        results["checks"]["single_sample"] = sample_test
        print(f"  Forward pass: {sample_test['forward_pass_time_sec']}s")
        print(f"  VRAM peak: {sample_test['vram_peak_mb']} MB")
        print(f"  Mean confidence: {sample_test['mean_confidence']}")
        print(f"  Mean entropy: {sample_test['mean_entropy']}")

        # 4b. Batch capacity
        print("\n[4b/5] Estimating max batch size (seq_len=256)...")
        batch_profile = check_batch_capacity(model, tokenizer)
        results["checks"]["batch_profile"] = batch_profile
        print(f"  Max batch size: {batch_profile['max_batch_size']}")
        print(f"  VRAM utilization: {batch_profile.get('utilization_pct', 'N/A')}%")

        # Cleanup
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Dataset
    print("\n[5/5] Checking dataset...")
    results["checks"]["dataset"] = check_dataset()
    print(f"  GSM8K: {results['checks']['dataset'].get('num_samples', 'N/A')} samples")

    # Summary
    all_pass = (
        results["checks"]["gpu"].get("cuda_available", False)
        and results["checks"]["model"].get("load_time_sec") is not None
        and results["checks"].get("single_sample", {}).get("forward_pass_time_sec") is not None
        and results["checks"]["dataset"].get("status") == "ok"
    )

    results["status"] = "success" if all_pass else "failed"
    results["pass"] = all_pass
    results["summary"] = (
        f"Environment verified. "
        f"GPU: {results['checks']['gpu'].get('device_name', 'N/A')} "
        f"({results['checks']['gpu'].get('vram_total_mb', 'N/A')}MB). "
        f"Model: {results['checks']['model'].get('num_params_b', 'N/A')}B params. "
        f"Attn: {results['checks']['model'].get('attn_implementation', 'N/A')}. "
        f"Max batch (256 tokens): {results['checks'].get('batch_profile', {}).get('max_batch_size', 'N/A')}. "
        f"GSM8K: {results['checks']['dataset'].get('num_samples', 'N/A')} samples. "
        f"{'PASS' if all_pass else 'FAIL'}"
    )

    # Write results
    os.makedirs(os.path.dirname(VERIFICATION_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(VERIFICATION_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Also write DONE marker
    done_marker = Path(RESULTS_DIR) / "setup_env_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "setup_env",
        "status": "success" if all_pass else "failed",
        "summary": results["summary"],
        "timestamp": datetime.now().isoformat(),
    }))

    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if all_pass else 'FAIL'}")
    print(results["summary"])
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
