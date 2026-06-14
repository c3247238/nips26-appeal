#!/usr/bin/env python3
"""
G0 Baseline Evaluation: DeepSeek-Math-7B-Instruct on MATH Pilot Set
Task: eval_g0_baseline

This script evaluates the baseline performance of DeepSeek-Math-7B-Instruct on MATH problems
with CoT prompting. It records accuracy, per-problem confidence scores, and token counts
to establish the H1 falsification baseline.

Pass criterion: MATH accuracy >= 0.40 (H1 pass criterion)
"""

import os
import sys
import json
import re
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = WORKSPACE / "exp/results/pilots"
SCRIPTS_DIR = WORKSPACE / "exp/scripts"

MODEL_NAME = "deepseek-ai/DeepSeek-Math-7B-Instruct"
DATASET_NAME = "HuggingFaceH4/MATH"
N_PILOT_SAMPLES = 100
SEED = 42
TASK_ID = "eval_g0_baseline"
GPU_ID = 1  # Using GPU 1 (free, 0 MiB used)

# System prompt for CoT
SYSTEM_PROMPT = """You are a mathematical reasoning assistant. For each problem:
1. Provide step-by-step reasoning (CoT)
2. End with your final answer in the format: \\boxed{answer}
3. Rate your confidence in your final answer (0.0 to 1.0)

Example format:
Step 1: ...
Step 2: ...
...
\\boxed{42}

Confidence: 0.85
"""


def setup_environment():
    """Setup CUDA device and clear cache."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU {GPU_ID}: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available!")


def load_model_and_tokenizer():
    """Load DeepSeek-Math-7B-Instruct with 4-bit quantization."""
    print(f"\nLoading model: {MODEL_NAME}")

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def parse_answer(response: str) -> Tuple[Optional[str], float]:
    """Extract boxed answer and confidence from model response."""
    # Extract boxed answer
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    answer = boxed_match.group(1) if boxed_match else None

    # Extract confidence
    conf_match = re.search(r'[Cc]onfidence:\s*([\d.]+)', response)
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    return answer, confidence


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove spaces, lowercase
    ans = answer.strip()
    ans = re.sub(r'\s+', '', ans)
    ans = ans.lower()
    # Keep only alphanumeric and common math symbols
    ans = re.sub(r'[^a-z0-9.\-,]', '', ans)
    return ans


def extract_ground_truth_answer(ground_truth: str) -> Optional[str]:
    """Extract the boxed answer from ground truth solution."""
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', ground_truth)
    if boxed_match:
        return boxed_match.group(1)
    return None


def is_correct(model_answer: str, ground_truth: str) -> bool:
    """Check if model answer matches ground truth."""
    if model_answer is None:
        return False
    # Extract the boxed answer from ground truth
    gt_answer = extract_ground_truth_answer(ground_truth)
    if gt_answer is None:
        # Fallback: compare against normalized full ground truth
        return False
    return normalize_answer(model_answer) == normalize_answer(gt_answer)


def compute_ece(results: List[Dict], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not results:
        return 0.0

    # Bin by confidence
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins

    for r in results:
        conf = r.get("confidence", 0.5)
        correct = r.get("is_correct", False)

        # Find bin
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_sums[bin_idx] += conf
        bin_counts[bin_idx] += 1
        if correct:
            bin_correct[bin_idx] += 1

    # Compute ECE
    ece = 0.0
    total = len(results)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_confidence = bin_sums[i] / bin_counts[i]
            accuracy = bin_correct[i] / bin_counts[i]
            ece += (bin_counts[i] / total) * abs(avg_confidence - accuracy)

    return ece


def evaluate_problem(
    model, tokenizer, problem: str, level: str, ground_truth: str
) -> Dict:
    """Evaluate a single MATH problem."""
    # Format prompt with system
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_len = inputs.input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda(),
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Parse answer and confidence
    model_answer, confidence = parse_answer(response)
    correct = is_correct(model_answer, ground_truth)

    return {
        "problem": problem[:100] + "..." if len(problem) > 100 else problem,
        "level": level,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "confidence": confidence,
        "is_correct": correct,
        "response": response[:500],  # Truncate for storage
        "input_tokens": input_len,
        "output_tokens": outputs.shape[1] - input_len,
    }


def run_evaluation():
    """Main evaluation loop."""
    print(f"\n{'='*60}")
    print(f"G0 Baseline Evaluation: {TASK_ID}")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Samples: {N_PILOT_SAMPLES}")
    print(f"GPU: {GPU_ID}")
    print(f"{'='*60}\n")

    # Setup
    setup_environment()

    # Write PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"PID: {os.getpid()}")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load MATH dataset
    print(f"\nLoading MATH dataset...")
    dataset = load_dataset(DATASET_NAME, split="test")
    print(f"Total MATH test samples: {len(dataset)}")

    # Shuffle and take pilot samples
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:N_PILOT_SAMPLES].tolist()
    pilot_data = [dataset[i] for i in indices]

    print(f"Evaluating {N_PILOT_SAMPLES} pilot samples...")

    results = []
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"

    for i, item in enumerate(pilot_data):
        problem = item["problem"]
        level = item.get("level", "unknown")
        ground_truth = item["solution"]  # 'solution' field in MATH dataset

        # Progress update
        print(f"[{i+1}/{N_PILOT_SAMPLES}] Level: {level}", end=" ")

        # Evaluate
        result = evaluate_problem(model, tokenizer, problem, level, ground_truth)
        results.append(result)

        # Report progress
        progress = {
            "task_id": TASK_ID,
            "epoch": 1,
            "total_epochs": 1,
            "step": i + 1,
            "total_steps": N_PILOT_SAMPLES,
            "loss": None,
            "metric": {
                "correct_count": sum(1 for r in results if r["is_correct"]),
                "total_count": len(results)
            },
            "updated_at": datetime.now().isoformat(),
        }
        progress_file.write_text(json.dumps(progress))

        # Print result
        status = "PASS" if result["is_correct"] else "FAIL"
        print(f"-> {status} (conf={result['confidence']:.2f})")

    # Compute metrics
    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / len(results)
    ece = compute_ece(results)
    avg_tokens = sum(r["output_tokens"] for r in results) / len(results)

    # Per-level accuracy
    level_stats = {}
    for level in set(r["level"] for r in results):
        level_results = [r for r in results if r["level"] == level]
        level_acc = sum(1 for r in level_results if r["is_correct"]) / len(level_results)
        level_stats[level] = {"accuracy": level_acc, "count": len(level_results)}

    # Save results
    output_file = RESULTS_DIR / f"{TASK_ID}.json"
    with open(output_file, "w") as f:
        json.dump({
            "task_id": TASK_ID,
            "model": MODEL_NAME,
            "n_samples": len(results),
            "accuracy": accuracy,
            "ece": ece,
            "avg_output_tokens": avg_tokens,
            "level_stats": level_stats,
            "h1_pass_criterion": 0.40,
            "h1_passed": accuracy >= 0.40,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Write DONE marker
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"Accuracy: {accuracy:.2%}, ECE: {ece:.3f}, Avg Tokens: {avg_tokens:.0f}",
        "final_progress": progress,
        "timestamp": datetime.now().isoformat(),
    }))

    # Print summary
    print(f"\n{'='*60}")
    print(f"G0 BASELINE EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Accuracy:    {accuracy:.2%} (H1 criterion: >= 40%)")
    print(f"ECE:         {ece:.3f}")
    print(f"Avg Tokens:  {avg_tokens:.0f}")
    print(f"H1 Status:   {'PASS' if accuracy >= 0.40 else 'FAIL'}")
    print(f"\nPer-level accuracy:")
    for level, stats in sorted(level_stats.items()):
        print(f"  {level}: {stats['accuracy']:.2%} ({stats['count']} samples)")
    print(f"{'='*60}")

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    return accuracy, ece, avg_tokens


if __name__ == "__main__":
    try:
        accuracy, ece, avg_tokens = run_evaluation()

        # Exit code based on H1 pass criterion
        if accuracy >= 0.40:
            sys.exit(0)
        else:
            print(f"\nWARNING: H1 criterion not met (accuracy {accuracy:.2%} < 40%)")
            sys.exit(0)  # Still exit 0, as task completed successfully

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Write failure marker
        done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
        done_file.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "summary": str(e),
            "timestamp": datetime.now().isoformat(),
        }))

        sys.exit(1)
