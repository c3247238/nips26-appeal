#!/usr/bin/env python3
"""
G0 Baseline: Single-pass evaluation of Qwen2.5-Math-7B on MATH test set.

Pilot: 100 samples, seed 42
Pass criteria: Accuracy > 40% OR consistent with published results
"""

import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
REMOTE_BASE = WORKSPACE
SHARED_BASE = REMOTE_BASE / "shared"
MODEL_PATH = SHARED_BASE / "models" / "Qwen2.5-Math-7B-Instruct"
DATASET_PATH = SHARED_BASE / "datasets" / "math_test_200"
RESULTS_DIR = REMOTE_BASE / "exp" / "results"
TASK_ID = "g0_qwen_baseline"

# Hyperparameters
N_PILOT_SAMPLES = 100
SEED = 42
TEMPERATURE = 0.7
MAX_TOKENS = 1024
GPU_ID = 2

# Set random seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def write_pid_file():
    """Write PID file for system monitor."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] Written to {pid_file}: {os.getpid()}")


def report_progress(epoch=1, total_epochs=1, step=0, total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    """Write DONE marker file for system monitor."""
    # Clean up PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    # Write DONE marker
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[DONE] Marker written: {status}")


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    # Match \boxed{...} or boxed{...}
    match = re.search(r'\\?boxed\s*\{([^}]+)\}', text)
    if match:
        answer = match.group(1).strip()
        # Extract final numerical or expression answer
        # Remove any leading/trailing LaTeX
        answer = answer.strip('$')
        return answer
    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    # Remove spaces, lowercase
    normalized = re.sub(r'\s+', '', str(answer).lower())
    # Remove \\, \quad, etc.
    normalized = re.sub(r'\\+|quad|qquad', '', normalized)
    return normalized


def check_answer(model_answer, ground_truth):
    """Check if model answer matches ground truth."""
    model_norm = normalize_answer(model_answer)
    truth_norm = normalize_answer(ground_truth)

    if model_norm is None or truth_norm is None:
        return False

    # Exact match after normalization
    if model_norm == truth_norm:
        return True

    # Try numeric comparison for numbers
    try:
        model_num = float(model_norm)
        truth_num = float(truth_norm)
        return abs(model_num - truth_num) < 1e-6
    except ValueError:
        pass

    return False


def load_model():
    """Load Qwen2.5-Math-7B model and tokenizer."""
    print(f"[MODEL] Loading from {MODEL_PATH}")

    # Check if model exists locally
    if MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists():
        print("[MODEL] Found locally, loading from cache...")
    else:
        print("[MODEL] Will download from HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        local_files_only=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        local_files_only=False,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # CUDA_VISIBLE_DEVICES=2 makes it appear as device 0
    )

    model.eval()
    print(f"[MODEL] Loaded successfully on GPU {GPU_ID}")
    return model, tokenizer


def extract_final_answer(solution_text):
    """Extract the final answer from the solution text."""
    if solution_text is None:
        return None
    # The solution often ends with "The answer is X" or "= X" or similar
    # Try to extract from the end of the solution
    lines = solution_text.strip().split('\n')
    # Look for answer patterns in the last few lines
    for line in reversed(lines):
        # Look for boxed answer
        boxed = re.search(r'\\?boxed\s*\{([^}]+)\}', line)
        if boxed:
            return boxed.group(1).strip()
        # Look for "= X" patterns
        eq_match = re.search(r'=\s*([^=\\]+)$', line.strip())
        if eq_match:
            return eq_match.group(1).strip()
    # Fallback: return the last line as answer
    if lines:
        return lines[-1].strip()
    return solution_text.strip()


def load_dataset_samples():
    """Load MATH test dataset."""
    print("[DATASET] Loading MATH dataset...")

    # Load from HuggingFace
    math_ds = load_dataset("HuggingFaceH4/MATH", split="test")

    # Subsample for pilot
    samples = []
    for i, item in enumerate(math_ds):
        if i >= N_PILOT_SAMPLES:
            break
        # Extract answer from solution
        solution = item["solution"]
        answer = extract_final_answer(solution)

        samples.append({
            "problem": item["problem"],
            "solution": solution,
            "answer": answer,
            "level": item["level"],
            "type": item["type"],
            "idx": i,
        })

    print(f"[DATASET] Loaded {len(samples)} samples for pilot")
    return samples


def generate_answer(model, tokenizer, problem, level):
    """Generate answer for a single problem."""
    # Format prompt for Qwen2.5-Math
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{problem}\nPlease provide your final answer in \\boxed{{}} format.<|im_end|>\n<|im_start|>assistant\n"

    messages = [
        {"role": "user", "content": f"{problem}\nPlease provide your final answer in \\boxed{{}} format."}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    """Main experiment loop."""
    print("=" * 60)
    print("G0 Baseline: Single-pass evaluation")
    print(f"Task ID: {TASK_ID}")
    print(f"Pilot samples: {N_PILOT_SAMPLES}")
    print(f"Seed: {SEED}")
    print(f"GPU: {GPU_ID}")
    print("=" * 60)

    # Write PID file
    write_pid_file()

    # Initialize results
    results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "Qwen/Qwen2.5-Math-7B-Instruct",
            "n_samples": N_PILOT_SAMPLES,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "gpu_id": GPU_ID,
        },
        "samples": [],
        "metrics": {},
    }

    start_time = time.time()

    try:
        # Load model
        model, tokenizer = load_model()
        report_progress(step=1, total_steps=5, metric={"status": "model_loaded"})

        # Load dataset
        samples = load_dataset_samples()
        report_progress(step=2, total_steps=5, metric={"status": "dataset_loaded"})

        # Run inference
        correct = 0
        total = len(samples)
        token_counts = []

        print(f"\n[EVAL] Running inference on {total} problems...")

        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            # Generate answer
            response = generate_answer(model, tokenizer, sample["problem"], sample["level"])

            # Extract boxed answer
            extracted = extract_boxed_answer(response)
            model_answer = extracted if extracted else response

            # Check correctness
            is_correct = check_answer(model_answer, sample["answer"])

            if is_correct:
                correct += 1

            # Count tokens (rough estimate)
            token_count = len(response.split()) * 1.3  # Rough estimate
            token_counts.append(token_count)

            # Store result
            results["samples"].append({
                "idx": sample["idx"],
                "level": sample["level"],
                "type": sample["type"],
                "problem": sample["problem"][:200] + "..." if len(sample["problem"]) > 200 else sample["problem"],
                "model_response": response[:500] + "..." if len(response) > 500 else response,
                "extracted_answer": model_answer,
                "ground_truth": sample["answer"],
                "is_correct": is_correct,
                "token_count": token_count,
            })

            # Report progress
            current_acc = correct / (i + 1)
            report_progress(
                step=i + 3,
                total_steps=total + 3,
                metric={"accuracy": current_acc, "correct": correct, "total": i + 1}
            )

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate metrics
        accuracy = correct / total
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        results["metrics"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_token_count": avg_tokens,
            "total_tokens": sum(token_counts),
            "elapsed_seconds": elapsed,
            "problems_per_second": total / elapsed if elapsed > 0 else 0,
        }

        # Per-level accuracy
        level_correct = {}
        level_total = {}
        for sample in results["samples"]:
            level = sample["level"]
            level_total[level] = level_total.get(level, 0) + 1
            if sample["is_correct"]:
                level_correct[level] = level_correct.get(level, 0) + 1

        results["metrics"]["per_level_accuracy"] = {
            level: level_correct.get(level, 0) / level_total[level]
            for level in level_total
        }

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Per-level accuracy:")
        for level in sorted(level_total.keys()):
            lvl_acc = level_correct.get(level, 0) / level_total[level]
            print(f"  Level {level}: {lvl_acc:.4f} ({level_correct.get(level, 0)}/{level_total[level]})")
        print(f"Average tokens: {avg_tokens:.1f}")
        print(f"Elapsed time: {elapsed:.1f}s")
        print(f"Throughput: {total / elapsed:.2f} problems/s")
        print("=" * 60)

        # Pass/Fail check
        pass_threshold = 0.40
        if accuracy >= pass_threshold:
            print(f"\n[PASS] Accuracy {accuracy:.4f} >= threshold {pass_threshold}")
            results["pass"] = True
            results["pass_criteria"] = f"Accuracy {accuracy:.4f} >= {pass_threshold}"
        else:
            print(f"\n[FAIL] Accuracy {accuracy:.4f} < threshold {pass_threshold}")
            results["pass"] = False
            results["pass_criteria"] = f"Accuracy {accuracy:.4f} < {pass_threshold}"

        # Save results
        results_file = RESULTS_DIR / f"{TASK_ID}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Results saved to {results_file}")

        # Mark done
        mark_done(
            status="success",
            summary=f"Accuracy: {accuracy:.4f}, Correct: {correct}/{total}"
        )

        # Print sample outputs
        print("\n[SAMPLES] Sample outputs:")
        for i, sample in enumerate(results["samples"][:5]):
            print(f"\n--- Sample {i+1} (Level {sample['level']}) ---")
            print(f"Problem: {sample['problem'][:150]}...")
            print(f"Model answer: {sample['extracted_answer'][:100]}...")
            print(f"Ground truth: {sample['ground_truth'][:100]}...")
            print(f"Correct: {sample['is_correct']}")

    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
        results["pass"] = False

        # Save partial results
        results_file = RESULTS_DIR / f"{TASK_ID}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        mark_done(status="failed", summary=str(e))
        raise


if __name__ == "__main__":
    main()
