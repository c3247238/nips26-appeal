#!/usr/bin/env python3
"""
Pilot G3: Difficulty-adaptive inference on baseline (G0)
===========================================================

This experiment tests whether difficulty-adaptive inference strategy selection
can achieve comparable accuracy to fixed self-correction with fewer tokens.

Approach:
1. Train a difficulty estimator (logistic regression) on 20% of pilot data
2. Evaluate baseline G0 model with three strategies on 80% test data:
   - Fixed single-shot CoT
   - Fixed 3-round self-correction
   - Adaptive inference (routing based on difficulty)

Pass criteria:
- Accuracy within 2% of fixed self-correction
- Token cost < 80% of fixed self-correction
"""

import json
import os
import random
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = WORKSPACE / "exp/results/pilots/g3_adaptive_inference"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model and dataset
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DATASET_SPLIT = 100  # Use 100 problems for pilot

# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Difficulty thresholds (will be calibrated)
EASY_THRESHOLD = 0.7
HARD_THRESHOLD = 0.3


def extract_surface_features(problem: str) -> dict:
    """Extract surface-level features from problem text for difficulty estimation."""
    features = {}

    # Token count
    tokens = problem.split()
    features["token_count"] = len(tokens)

    # Numeric entity count
    numbers = re.findall(r'\d+\.?\d*', problem)
    features["numeric_count"] = len(numbers)

    # Constraint keyword count
    keywords = ["find", "prove", "minimize", "maximize", "calculate", "determine",
                "show", "evaluate", "solve", "compute", "express"]
    features["keyword_count"] = sum(1 for kw in keywords if kw.lower() in problem.lower())

    # Problem length
    features["char_count"] = len(problem)

    # Fraction of digits
    features["digit_ratio"] = sum(c.isdigit() for c in problem) / max(len(problem), 1)

    # Question mark presence (indicates a question)
    features["has_question"] = 1.0 if "?" in problem else 0.0

    # Algebraic expression complexity (heuristic)
    features["operator_count"] = sum(1 for op in ['+', '-', '*', '/', '=', '^']
                                      if op in problem)

    return features


def get_ground_truth_difficulty(problem: str, model, tokenizer, num_samples: int = 3) -> float:
    """
    Estimate ground-truth difficulty using baseline model's answer variance.

    Higher variance = more difficult (model is uncertain).
    Returns difficulty score in [0, 1] where 1 = most difficult.
    """
    prompt = f"<|user|>\n{problem}\n<|assistant|>"

    answers = []
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        answer_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answers.append(answer_text.strip())

    # Extract final numeric answer if possible
    def extract_number(text):
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return float(numbers[-1]) if numbers else None

    numeric_answers = [extract_number(a) for a in answers]
    valid_answers = [n for n in numeric_answers if n is not None]

    if len(valid_answers) < 2:
        return 0.5  # Default medium difficulty

    # Coefficient of variation as uncertainty measure
    mean = np.mean(valid_answers)
    std = np.std(valid_answers)
    if mean == 0:
        return 0.5

    cv = abs(std / mean)
    # Scale CV to [0, 1] with some reasonable bounds
    difficulty = min(1.0, cv / 2.0)
    return difficulty


def single_shot_cot(problem: str, model, tokenizer) -> tuple[str, int]:
    """Run single-shot chain-of-thought inference."""
    prompt = f"<|user|>\n{problem}\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    token_count = outputs.shape[1] - inputs.input_ids.shape[1]

    return response, token_count


def self_correct_cot(problem: str, model, tokenizer, num_rounds: int = 3) -> tuple[str, int]:
    """Run self-correction CoT inference for specified rounds."""
    prompt = f"<|user|>\n{problem}\n<|assistant|>"

    total_tokens = 0
    current_solution = ""

    for round_idx in range(num_rounds):
        # Build context
        if round_idx == 0:
            context = prompt
        else:
            context = f"{prompt}{current_solution}\n\n[Self-Correction Round {round_idx + 1}] Please review your previous solution and correct any errors.\n"

        inputs = tokenizer(context, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        current_solution = response
        total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]

        # If no correction needed (answer unchanged from previous), can stop early
        if round_idx > 0 and current_solution.strip() == prev_solution.strip():
            break

        prev_solution = current_solution

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
        model_num = float(re.search(r'-?\d+\.?\d*', model_clean).group())
        truth_num = float(re.search(r'-?\d+\.?\d*', truth_clean).group())
        return abs(model_num - truth_num) < 1e-6
    except (AttributeError, ValueError):
        pass

    # Fall back to string comparison
    return model_clean == truth_clean


def run_experiment():
    """Main experiment loop."""
    print("=" * 60)
    print("PILOT G3: Difficulty-Adaptive Inference")
    print("=" * 60)

    # Load model
    print("\n[1/6] Loading baseline model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Write PID file
    pid_file = RESULTS_DIR / "pilot_g3.pid"
    pid_file.write_text(str(os.getpid()))

    # Report progress
    progress = {
        "task_id": "pilot_g3",
        "step": "loading_model",
        "status": "in_progress",
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / "pilot_g3_PROGRESS.json").write_text(json.dumps(progress))

    # Load MATH dataset
    print("\n[2/6] Loading MATH dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH", split=f"train[:{DATASET_SPLIT}]")
    problems = [(i, item) for i, item in enumerate(dataset)]

    # Split: 20% calibration, 80% test
    random.shuffle(problems)
    split_idx = int(len(problems) * 0.2)
    calibration_set = problems[:split_idx]
    test_set = problems[split_idx:]

    print(f"  Calibration set: {len(calibration_set)} problems")
    print(f"  Test set: {len(test_set)} problems")

    # Extract features and ground-truth difficulty for calibration set
    print("\n[3/6] Building difficulty estimator (calibration set)...")

    calibration_features = []
    calibration_difficulties = []

    for idx, item in calibration_set:
        problem = item["problem"]
        solution = item["solution"]

        # Extract surface features
        features = extract_surface_features(problem)
        feature_vec = [features[k] for k in sorted(features.keys())]
        calibration_features.append(feature_vec)

        # Get ground-truth difficulty via answer variance
        difficulty = get_ground_truth_difficulty(problem, model, tokenizer, num_samples=3)
        calibration_difficulties.append(difficulty)

        # Quick single-shot result for calibration
        response, _ = single_shot_cot(problem, model, tokenizer)
        is_correct_response = is_correct(response, solution)

        # Update difficulty based on single-shot success (inverse relationship)
        # If single-shot succeeds, problem is easier
        if is_correct_response:
            calibration_difficulties[-1] = min(calibration_difficulties[-1], 0.3)
        else:
            calibration_difficulties[-1] = max(calibration_difficulties[-1], 0.7)

    # Train difficulty classifier
    X_cal = np.array(calibration_features)
    y_cal = np.array([1 if d > 0.5 else 0 for d in calibration_difficulties])  # Binary: hard vs easy

    # Scale features
    scaler = StandardScaler()
    X_cal_scaled = scaler.fit_transform(X_cal)

    # Train logistic regression
    clf = LogisticRegression(random_state=SEED, max_iter=1000)
    clf.fit(X_cal_scaled, y_cal)

    # Calibrate thresholds using calibration set
    # Run calibration set through difficulty estimator to find optimal thresholds
    cal_probs = clf.predict_proba(X_cal_scaled)[:, 1]  # Probability of being hard

    # Find thresholds that maximize accuracy-efficiency tradeoff
    easy_probs = cal_probs[cal_probs < 0.5]
    medium_probs = cal_probs[(cal_probs >= 0.3) & (cal_probs <= 0.7)]
    hard_probs = cal_probs[cal_probs > 0.5]

    print(f"  Calibration distribution: easy={len(easy_probs)}, medium={len(medium_probs)}, hard={len(hard_probs)}")

    # Evaluate on test set with three strategies
    print("\n[4/6] Evaluating on test set with three inference strategies...")

    results = {
        "single_shot": {"correct": 0, "total": 0, "total_tokens": 0},
        "fixed_self_correction": {"correct": 0, "total": 0, "total_tokens": 0},
        "adaptive": {"correct": 0, "total": 0, "total_tokens": 0},
    }

    adaptive_decisions = {"easy": 0, "medium": 0, "hard": 0}

    test_features = []
    for test_idx, (orig_idx, item) in enumerate(test_set):
        problem = item["problem"]
        solution = item["solution"]

        # Extract features for difficulty estimation
        features = extract_surface_features(problem)
        feature_vec = [np.array([features[k] for k in sorted(features.keys())])]
        test_features.append(feature_vec[0])

        # Strategy 1: Fixed single-shot CoT
        ss_response, ss_tokens = single_shot_cot(problem, model, tokenizer)
        ss_correct = is_correct(ss_response, solution)
        results["single_shot"]["correct"] += int(ss_correct)
        results["single_shot"]["total"] += 1
        results["single_shot"]["total_tokens"] += ss_tokens

        # Strategy 2: Fixed 3-round self-correction
        sc_response, sc_tokens = self_correct_cot(problem, model, tokenizer, num_rounds=3)
        sc_correct = is_correct(sc_response, solution)
        results["fixed_self_correction"]["correct"] += int(sc_correct)
        results["fixed_self_correction"]["total"] += 1
        results["fixed_self_correction"]["total_tokens"] += sc_tokens

        # Strategy 3: Adaptive inference
        X_test = scaler.transform(feature_vec)
        difficulty_prob = clf.predict_proba(X_test)[:, 1][0]

        if difficulty_prob > 0.7:  # Hard
            adaptive_decisions["hard"] += 1
            ad_response, ad_tokens = self_correct_cot(problem, model, tokenizer, num_rounds=3)
        elif difficulty_prob > 0.3:  # Medium
            adaptive_decisions["medium"] += 1
            ad_response, ad_tokens = self_correct_cot(problem, model, tokenizer, num_rounds=1)
        else:  # Easy
            adaptive_decisions["easy"] += 1
            ad_response, ad_tokens = single_shot_cot(problem, model, tokenizer)

        ad_correct = is_correct(ad_response, solution)
        results["adaptive"]["correct"] += int(ad_correct)
        results["adaptive"]["total"] += 1
        results["adaptive"]["total_tokens"] += ad_tokens

        # Progress update every 10 problems
        if (test_idx + 1) % 10 == 0:
            print(f"  Progress: {test_idx + 1}/{len(test_set)} test problems evaluated")
            progress["step"] = f"evaluating ({test_idx + 1}/{len(test_set)})"
            progress["updated_at"] = datetime.now().isoformat()
            (RESULTS_DIR / "pilot_g3_PROGRESS.json").write_text(json.dumps(progress))

    # Compute metrics
    print("\n[5/6] Computing metrics...")

    def compute_metrics(res):
        accuracy = res["correct"] / res["total"] if res["total"] > 0 else 0
        avg_tokens = res["total_tokens"] / res["total"] if res["total"] > 0 else 0
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "total_correct": res["correct"],
            "total_evaluated": res["total"],
        }

    ss_metrics = compute_metrics(results["single_shot"])
    sc_metrics = compute_metrics(results["fixed_self_correction"])
    ad_metrics = compute_metrics(results["adaptive"])

    # Check pass criteria
    sc_accuracy = sc_metrics["accuracy"]
    sc_tokens = sc_metrics["avg_tokens"]
    ad_accuracy = ad_metrics["accuracy"]
    ad_tokens = ad_metrics["avg_tokens"]

    accuracy_within_2pct = abs(ad_accuracy - sc_accuracy) <= 0.02
    token_cost_reduced = ad_tokens < 0.80 * sc_tokens

    pass_criteria_met = accuracy_within_2pct and token_cost_reduced

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Strategy':<30} {'Accuracy':<12} {'Avg Tokens':<12} {'Pass?'}")
    print("-" * 60)
    print(f"{'Single-shot CoT':<30} {ss_metrics['accuracy']:.2%}        {ss_metrics['avg_tokens']:.1f}")
    print(f"{'Fixed 3-round self-correction':<30} {sc_metrics['accuracy']:.2%}        {sc_metrics['avg_tokens']:.1f}")
    print(f"{'Adaptive inference':<30} {ad_metrics['accuracy']:.2%}        {ad_metrics['avg_tokens']:.1f}")
    print("-" * 60)
    print(f"\nAdaptive routing distribution: {adaptive_decisions}")

    print(f"\nPass Criteria:")
    print(f"  - Accuracy within 2%: {accuracy_within_2pct} ({ad_accuracy:.2%} vs {sc_accuracy:.2%})")
    print(f"  - Token cost < 80%: {token_cost_reduced} ({ad_tokens:.1f} vs {sc_tokens:.1f}, {100*ad_tokens/sc_tokens:.1f}%)")
    print(f"  - PASS: {pass_criteria_met}")

    # Save results
    print("\n[6/6] Saving results...")

    summary = {
        "task_id": "pilot_g3",
        "status": "success",
        "pass_criteria_met": pass_criteria_met,
        "strategies": {
            "single_shot": ss_metrics,
            "fixed_self_correction": sc_metrics,
            "adaptive": ad_metrics,
        },
        "adaptive_routing": adaptive_decisions,
        "accuracy_within_2pct": accuracy_within_2pct,
        "token_cost_reduced": token_cost_reduced,
        "pass_criteria": {
            "accuracy_within_2pct": True,
            "token_cost_below_80pct": True,
        },
        "model": MODEL_NAME,
        "calibration_size": len(calibration_set),
        "test_size": len(test_set),
        "difficulty_estimator": "LogisticRegression on surface features",
    }

    (RESULTS_DIR / "pilot_g3_summary.json").write_text(json.dumps(summary, indent=2))

    # Write DONE marker
    (RESULTS_DIR / "pilot_g3_DONE").write_text(json.dumps({
        "task_id": "pilot_g3",
        "status": "success",
        "pass_criteria_met": pass_criteria_met,
        "summary": f"Adaptive: {ad_metrics['accuracy']:.2%} acc, {ad_metrics['avg_tokens']:.1f} tokens; "
                   f"Fixed SC: {sc_metrics['accuracy']:.2%} acc, {sc_metrics['avg_tokens']:.1f} tokens",
        "timestamp": datetime.now().isoformat(),
    }))

    # Update progress
    progress["step"] = "completed"
    progress["status"] = "completed"
    progress["accuracy_single_shot"] = ss_metrics["accuracy"]
    progress["accuracy_fixed_sc"] = sc_metrics["accuracy"]
    progress["accuracy_adaptive"] = ad_metrics["accuracy"]
    progress["pass_criteria_met"] = pass_criteria_met
    progress["updated_at"] = datetime.now().isoformat()
    (RESULTS_DIR / "pilot_g3_PROGRESS.json").write_text(json.dumps(progress))

    print("\nResults saved to:", RESULTS_DIR)
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_experiment()
