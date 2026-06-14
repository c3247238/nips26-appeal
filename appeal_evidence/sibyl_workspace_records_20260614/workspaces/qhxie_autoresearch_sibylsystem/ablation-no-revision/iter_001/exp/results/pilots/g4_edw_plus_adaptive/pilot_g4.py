"""
Pilot G4: EDW-Step-DPO + adaptive inference
Tests training-inference co-optimization by applying adaptive inference on G2 (EDW-Step-DPO) model.
Combines G2 training with G3 adaptive routing on the same test set.
"""

import json
import os
import sys
import random
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import re

# Workspace paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/iter_001")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = WORKSPACE / "exp/results/pilots/g4_edw_plus_adaptive"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
G2_ADAPTER_PATH = WORKSPACE / "exp/results/pilots/g2_edw_stepdpo"
DATASET_SPLIT = 100  # Use 100 problems for pilot

# Task tracking
TASK_ID = "pilot_g4"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Difficulty thresholds
EASY_THRESHOLD = 0.7
HARD_THRESHOLD = 0.3

def report_progress(step=0, total_epochs=80, metric=None):
    """Write progress file for system monitor."""
    progress = {
        "task_id": TASK_ID,
        "epoch": 0,
        "total_epochs": total_epochs,
        "step": step,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    if step % 10 == 0:
        print(f"[PROGRESS] step={step}/{total_epochs}, metric={metric}")

def mark_done(status="success", summary=""):
    """Write DONE marker file for system monitor."""
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    marker = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": progress,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker, indent=2))
    if PID_FILE.exists():
        PID_FILE.unlink()
    print(f"[DONE] Status: {status}, Summary: {summary}")

def setup_model():
    """Load G2 trained model (base + LoRA adapter)."""
    print("[SETUP] Loading G2 trained model (EDW-Step-DPO + LoRA)...")

    # Get base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load G2 adapter on top
    model = PeftModel.from_pretrained(
        base_model,
        str(G2_ADAPTER_PATH),
        device_map="auto",
    )
    model.eval()

    print(f"[SETUP] Model loaded successfully. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model, tokenizer

def extract_surface_features(problem: str) -> dict:
    """Extract surface-level features from problem text for difficulty estimation."""
    features = {}
    tokens = problem.split()
    features["token_count"] = len(tokens)
    numbers = re.findall(r'\d+\.?\d*', problem)
    features["numeric_count"] = len(numbers)
    keywords = ["find", "prove", "minimize", "maximize", "calculate", "determine",
                "show", "evaluate", "solve", "compute", "express"]
    features["keyword_count"] = sum(1 for kw in keywords if kw.lower() in problem.lower())
    features["char_count"] = len(problem)
    features["digit_ratio"] = sum(c.isdigit() for c in problem) / max(len(problem), 1)
    features["has_question"] = 1.0 if "?" in problem else 0.0
    features["operator_count"] = sum(1 for op in ['+', '-', '*', '/', '=', '^'] if op in problem)
    return features

def single_shot_cot(problem: str, model, tokenizer, max_tokens=500) -> tuple[str, int]:
    """Run single-shot chain-of-thought inference."""
    prompt = f"<|user|>\n{problem}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    token_count = outputs.shape[1] - inputs.input_ids.shape[1]
    return response, token_count

def self_correct_cot(problem: str, model, tokenizer, num_rounds: int = 1, max_tokens=500) -> tuple[str, int]:
    """Run self-correction CoT inference for specified rounds."""
    prompt = f"<|user|>\n{problem}\n<|assistant|>"
    total_tokens = 0
    current_solution = ""

    for round_idx in range(num_rounds + 1):  # +1 for initial attempt
        if round_idx == 0:
            context = prompt
        else:
            context = f"{prompt}{current_solution}\n\n[Self-Correction Round {round_idx}] Please review your previous solution and correct any errors.\n"

        inputs = tokenizer(context, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        current_solution = response
        total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]

    return current_solution, total_tokens

def extract_final_answer(text: str) -> str:
    """Extract the final answer from the response text."""
    # Look for boxed answer first
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    # Look for "Answer: X" pattern
    answer_match = re.findall(r'[Aa]nswer[:\s]+([^\n]+)', text)
    if answer_match:
        return answer_match[-1].strip()
    # Use the last line as fallback
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()

def is_correct(model_answer: str, ground_truth: str) -> bool:
    """Check if model's answer matches ground truth."""
    model_final = extract_final_answer(model_answer)
    truth_final = extract_final_answer(ground_truth)

    # Normalize for comparison
    model_clean = re.sub(r'[\s,]+', '', model_final.lower())
    truth_clean = re.sub(r'[\s,]+', '', truth_final.lower())

    # Try numeric comparison first
    try:
        model_nums = re.findall(r'-?\d+\.?\d*', model_clean)
        truth_nums = re.findall(r'-?\d+\.?\d*', truth_clean)
        if model_nums and truth_nums:
            model_num = float(model_nums[-1])
            truth_num = float(truth_nums[-1])
            return abs(model_num - truth_num) < 1e-6
    except (AttributeError, ValueError, IndexError):
        pass

    # Fall back to string comparison
    return model_clean == truth_clean

def estimate_difficulty(problem: str) -> str:
    """Simple heuristic-based difficulty estimation."""
    tokens = len(problem.split())
    numbers = len(re.findall(r'\d+\.?\d*', problem))
    operators = sum(1 for op in ['+', '-', '*', '/', '=', '^'] if op in problem)

    # Simple heuristic
    complexity = tokens + numbers * 2 + operators * 3

    if complexity < 40:
        return 'easy'
    elif complexity < 80:
        return 'medium'
    else:
        return 'hard'

def run_evaluation(model, tokenizer, test_problems):
    """
    Run evaluation comparing G4 (EDW + adaptive) vs G2 (EDW single-shot).
    """
    print(f"\n[EVAL] Running evaluation on {len(test_problems)} test problems...")

    results = {
        'g4_edw_adaptive': {'correct': 0, 'total': 0, 'total_tokens': 0},
        'g2_edw_single': {'correct': 0, 'total': 0, 'total_tokens': 0},
        'g4_edw_fixed2': {'correct': 0, 'total': 0, 'total_tokens': 0},
    }
    routing_stats = {'easy': 0, 'medium': 0, 'hard': 0}

    for idx, item in enumerate(test_problems):
        problem = item['problem']
        solution = item['solution']

        # Estimate difficulty
        difficulty = estimate_difficulty(problem)
        routing_stats[difficulty] += 1

        # G4: EDW + adaptive routing
        if difficulty == 'easy':
            ad_response, ad_tokens = single_shot_cot(problem, model, tokenizer)
        elif difficulty == 'medium':
            ad_response, ad_tokens = self_correct_cot(problem, model, tokenizer, num_rounds=1)
        else:  # hard
            ad_response, ad_tokens = self_correct_cot(problem, model, tokenizer, num_rounds=2)

        ad_correct = is_correct(ad_response, solution)
        results['g4_edw_adaptive']['correct'] += int(ad_correct)
        results['g4_edw_adaptive']['total'] += 1
        results['g4_edw_adaptive']['total_tokens'] += ad_tokens

        # G2: EDW single-shot
        g2_response, g2_tokens = single_shot_cot(problem, model, tokenizer)
        g2_correct = is_correct(g2_response, solution)
        results['g2_edw_single']['correct'] += int(g2_correct)
        results['g2_edw_single']['total'] += 1
        results['g2_edw_single']['total_tokens'] += g2_tokens

        # G4 fixed 2 rounds
        g4f_response, g4f_tokens = self_correct_cot(problem, model, tokenizer, num_rounds=2)
        g4f_correct = is_correct(g4f_response, solution)
        results['g4_edw_fixed2']['correct'] += int(g4f_correct)
        results['g4_edw_fixed2']['total'] += 1
        results['g4_edw_fixed2']['total_tokens'] += g4f_tokens

        if (idx + 1) % 10 == 0:
            report_progress(
                step=idx + 1,
                total_epochs=len(test_problems),
                metric={'accuracy_so_far': results['g4_edw_adaptive']['correct'] / max(1, results['g4_edw_adaptive']['total'])}
            )

    return results, routing_stats

def compute_metrics(results):
    """Compute metrics from results."""
    metrics = {}
    for group, data in results.items():
        total = data['total']
        correct = data['correct']
        total_tokens = data['total_tokens']
        if total > 0:
            metrics[group] = {
                'accuracy': correct / total,
                'total_correct': correct,
                'total_evaluated': total,
                'avg_tokens': total_tokens / total,
            }
    return metrics

def main():
    """Main execution."""
    print("=" * 60)
    print("PILOT G4: EDW-Step-DPO + Adaptive Inference")
    print("=" * 60)

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))
    print(f"[PID] {os.getpid()}")
    report_progress(step=0)

    try:
        # Load MATH dataset (same as G3 for consistency)
        print("\n[DATA] Loading MATH dataset...")
        dataset = load_dataset("HuggingFaceH4/MATH", split=f"train[:{DATASET_SPLIT}]")
        problems = list(dataset)

        # Shuffle and split: 20% calibration, 80% test
        random.shuffle(problems)
        split_idx = int(len(problems) * 0.2)
        calibration_set = problems[:split_idx]
        test_set = problems[split_idx:]

        print(f"[DATA] Split: {len(calibration_set)} calibration, {len(test_set)} test problems")

        # Setup model
        print("\n[SETUP] Loading G2 model (EDW-Step-DPO trained)...")
        model, tokenizer = setup_model()

        # Run evaluation
        print("\n[EVAL] Starting evaluation...")
        results, routing_stats = run_evaluation(model, tokenizer, test_set)

        # Compute metrics
        metrics = compute_metrics(results)

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        for group, m in metrics.items():
            print(f"\n{group}:")
            print(f"  Accuracy: {m['accuracy']:.2%}")
            print(f"  Correct/Total: {m['total_correct']}/{m['total_evaluated']}")
            print(f"  Avg Tokens: {m['avg_tokens']:.1f}")

        print(f"\nRouting Stats: {routing_stats}")

        # Superadditivity check
        # H3: G4 > G2 + (G3 - G0)
        # We compare G4 adaptive vs G2 single and vs G4 fixed 2-rounds
        g4_acc = metrics.get('g4_edw_adaptive', {}).get('accuracy', 0)
        g2_acc = metrics.get('g2_edw_single', {}).get('accuracy', 0)
        g4f_acc = metrics.get('g4_edw_fixed2', {}).get('accuracy', 0)
        g4_tokens = metrics.get('g4_edw_adaptive', {}).get('avg_tokens', 0)
        g2_tokens = metrics.get('g2_edw_single', {}).get('avg_tokens', 0)
        g4f_tokens = metrics.get('g4_edw_fixed2', {}).get('avg_tokens', 0)

        superadditivity = {
            'g4_vs_g2_accuracy_delta': g4_acc - g2_acc,
            'g4_vs_g2_token_delta': g4_tokens - g2_tokens,
            'g4_vs_fixed2_accuracy_delta': g4_acc - g4f_acc,
            'g4_vs_fixed2_token_reduction_pct': 100 * (g4f_tokens - g4_tokens) / max(g4f_tokens, 1),
        }

        print(f"\nSuperadditivity Check: {superadditivity}")

        # Pass criteria: G4 shows either better accuracy than G2 or better efficiency
        pass_criteria_met = (
            g4_acc > g2_acc or  # Training + adaptive provides accuracy benefit
            g4_tokens < g2_tokens  # Adaptive saves tokens
        )

        # Save results
        final_summary = {
            "task_id": TASK_ID,
            "status": "success",
            "pass_criteria_met": pass_criteria_met,
            "key_metrics": metrics,
            "routing_stats": routing_stats,
            "superadditivity": superadditivity,
            "model": MODEL_NAME,
            "adapter": str(G2_ADAPTER_PATH),
            "test_size": len(test_set),
        }

        final_summary_path = RESULTS_DIR / "pilot_summary.json"
        final_summary_path.write_text(json.dumps(final_summary, indent=2))
        print(f"\n[SAVE] Summary saved to {final_summary_path}")

        # Also save a copy of the key metrics
        summary_path = RESULTS_DIR / f"{TASK_ID}_summary.json"
        summary_path.write_text(json.dumps({
            "metrics": metrics,
            "routing_stats": routing_stats,
            "superadditivity": superadditivity,
        }, indent=2))

        report_progress(step=len(test_set), metric={'final_accuracy': g4_acc})

        mark_done(
            status="success",
            summary=f"G4 acc: {g4_acc:.2%}, G2 acc: {g2_acc:.2%}, tokens: {g4_tokens:.0f} vs {g2_tokens:.0f}, routing: {routing_stats}"
        )

        print("\n[DONE] Pilot G4 completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=f"Error: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
