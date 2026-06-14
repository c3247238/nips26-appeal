"""
Setup verification script for setup_env task.
Verifies:
1. LLaDA-8B-Instruct model loads successfully
2. Baseline generates valid text on 5 GSM8K examples
3. No CUDA OOM errors
4. Conda environment has all required packages
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
CODE_DIR = WORKSPACE / "exp/code"
RESULTS_DIR = WORKSPACE / "exp/results"
MODEL_PATH = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared/checkpoints/llada-8b-instruct")

# Add code dir to path
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

verification_log = {
    "task_id": "setup_env",
    "timestamp": datetime.now().isoformat(),
    "checks": {},
    "status": "pending",
}

def check_imports():
    """Check all required packages are importable."""
    required = ["torch", "transformers", "datasets", "numpy", "tqdm"]
    results = {}
    for pkg in required:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            results[pkg] = {"status": "ok", "version": version}
        except ImportError as e:
            results[pkg] = {"status": "error", "error": str(e)}
    return results

def check_cuda():
    """Check CUDA availability."""
    import torch
    return {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2) if torch.cuda.is_available() else None,
    }

def check_model_loads():
    """Try loading LLaDA-8B-Instruct tokenizer and model."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    t_tok = time.time() - t0
    print(f"  Tokenizer loaded in {t_tok:.1f}s")

    t0 = time.time()
    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    t_model = time.time() - t0
    print(f"  Model loaded in {t_model:.1f}s")

    vram_used = torch.cuda.memory_allocated(0) / 1024**2
    vram_reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"  VRAM used: {vram_used:.0f} MB, reserved: {vram_reserved:.0f} MB")

    return {
        "status": "ok",
        "tokenizer_load_time_s": round(t_tok, 2),
        "model_load_time_s": round(t_model, 2),
        "vram_used_mb": round(vram_used),
        "vram_reserved_mb": round(vram_reserved),
        "model": model,
        "tokenizer": tokenizer,
    }

def run_baseline_on_5_gsm8k(model, tokenizer):
    """Run baseline inference on 5 GSM8K samples."""
    import torch
    import sys

    # 5 sample GSM8K problems
    problems = [
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
    ]

    expected_answers = ["6", "5", "39", "8", "9"]

    results = []
    for i, (problem, expected) in enumerate(zip(problems, expected_answers)):
        print(f"  Running sample {i+1}/5...")
        prompt = f"Q: {problem}\nA: Let's think step by step."

        try:
            # Use LLaDA generate
            sys.path.insert(0, str(CODE_DIR / "LLaDA"))
            from generate import generate

            tokenizer_result = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenizer_result["input_ids"].to("cuda:0")

            # Use a short generation for verification
            gen_length = 128
            # Pad input for masked diffusion
            input_len = input_ids.shape[1]

            with torch.no_grad():
                t0 = time.time()
                out = generate(
                    model,
                    input_ids,
                    steps=64,
                    gen_length=gen_length,
                    block_length=32,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                )
                elapsed = time.time() - t0

            generated_ids = out[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Try to extract answer
            import re
            num_match = re.findall(r'\d+', generated_text)
            predicted = num_match[-1] if num_match else "?"
            correct = predicted == expected

            print(f"    Generated: {generated_text[:80]}...")
            print(f"    Expected: {expected}, Predicted: {predicted}, Correct: {correct}")

            results.append({
                "problem_idx": i,
                "generated_text": generated_text[:200],
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "elapsed_s": round(elapsed, 2),
                "status": "ok",
            })

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                "problem_idx": i,
                "error": str(e),
                "traceback": traceback.format_exc()[-500:],
                "status": "error",
            })

    n_correct = sum(1 for r in results if r.get("correct", False))
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n  Results: {n_correct}/{len(problems)} correct, {n_ok}/{len(problems)} ran successfully")

    return {
        "n_samples": len(problems),
        "n_successful": n_ok,
        "n_correct": n_correct,
        "accuracy": n_correct / len(problems),
        "samples": results,
        "status": "ok" if n_ok >= 4 else "partial",
    }

def main():
    print("=" * 60)
    print("Setup Environment Verification")
    print("=" * 60)

    # Step 1: Check imports
    print("\n[1] Checking required packages...")
    import_check = check_imports()
    verification_log["checks"]["imports"] = import_check
    failed_imports = [k for k, v in import_check.items() if v["status"] != "ok"]
    if failed_imports:
        print(f"  FAILED imports: {failed_imports}")
    else:
        print(f"  All packages OK: {list(import_check.keys())}")

    # Step 2: Check CUDA
    print("\n[2] Checking CUDA...")
    cuda_info = check_cuda()
    verification_log["checks"]["cuda"] = cuda_info
    if not cuda_info["available"]:
        print("  WARNING: CUDA not available!")
    else:
        print(f"  GPU: {cuda_info['device_name']}, VRAM: {cuda_info['vram_total_mb']} MB")

    # Step 3: Check model loads
    print("\n[3] Loading LLaDA-8B-Instruct...")
    try:
        model_info = check_model_loads()
        model = model_info.pop("model")
        tokenizer = model_info.pop("tokenizer")
        verification_log["checks"]["model_load"] = model_info
        print(f"  Model loaded successfully!")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        verification_log["checks"]["model_load"] = {"status": "error", "error": str(e)}
        verification_log["status"] = "failed"
        # Save what we have and exit
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "setup_verification.json").write_text(
            json.dumps(verification_log, indent=2)
        )
        return False

    # Step 4: Run baseline on 5 GSM8K samples
    print("\n[4] Running baseline on 5 GSM8K samples...")
    try:
        gsm8k_results = run_baseline_on_5_gsm8k(model, tokenizer)
        verification_log["checks"]["gsm8k_pilot"] = gsm8k_results
    except Exception as e:
        print(f"  ERROR: {e}")
        verification_log["checks"]["gsm8k_pilot"] = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[-1000:],
        }

    # Step 5: Summary
    print("\n[5] Summary")
    all_ok = (
        not failed_imports
        and cuda_info.get("available", False)
        and verification_log["checks"].get("model_load", {}).get("status") == "ok"
        and verification_log["checks"].get("gsm8k_pilot", {}).get("n_successful", 0) >= 4
    )

    verification_log["status"] = "success" if all_ok else "partial"
    verification_log["files_verified"] = {
        "eval_harness.py": (CODE_DIR / "eval_harness.py").exists(),
        "inference_wrapper.py": (CODE_DIR / "inference_wrapper.py").exists(),
        "LLaDA/generate.py": (CODE_DIR / "LLaDA/generate.py").exists(),
        "llada-8b-instruct": MODEL_PATH.exists(),
    }

    # Save verification log
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "setup_verification.json"
    out_path.write_text(json.dumps(verification_log, indent=2))
    print(f"\nVerification log saved to: {out_path}")
    print(f"Status: {verification_log['status'].upper()}")

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
