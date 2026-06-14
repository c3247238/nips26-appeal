#!/usr/bin/env python3
"""
Generate calibration dataset for MATH benchmark with DeepSeek-Math-7B-Instruct.

This script:
1. Loads MATH dataset from HuggingFace
2. Generates CoT responses with confidence scores for each problem
3. Computes correctness by parsing final answers
4. Splits into train/calibration/test sets
5. Computes baseline ECE metric
"""

import json
import re
import os
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-Math-7B-Instruct"
PILOT_SAMPLES = 100  # For pilot mode
SEED = 42
OUTPUT_PATH = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results/pilots/calibration_dataset_200.jsonl"
RESULTS_DIR = Path(OUTPUT_PATH).parent

# Generate response generation function
COT_SYSTEM_PROMPT = """You are a helpful assistant that solves mathematical problems step by step.
Think carefully about each problem. When you provide your answer, clearly state the final answer.
Also, at the end of your response, state your confidence in your answer on a scale from 0.0 to 1.0.
Format: Answer: <final answer> | Confidence: <0.0-1.0>"""


def setup_model():
    """Load DeepSeek-Math-7B-Instruct model with quantization."""
    print(f"Loading model: {MODEL_NAME}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def parse_final_answer(text: str) -> Optional[str]:
    """Extract the final answer from model response."""
    # Try multiple patterns for answer extraction
    patterns = [
        # Answer: followed by content until | or end
        r"Answer:\s*(.+?)(?:\||\n|$)",
        # boxed{} in LaTeX
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        # Final boxed at end
        r"boxed\{(.+?)\}",
        # Confidence marker - take everything before it as answer
        r"(?:Confidence|conf):",
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            if i == 3:  # Confidence pattern - take content before it
                answer = text[:match.start()].strip()
            else:
                answer = match.group(1 if i < 2 else 0).strip()

            # Clean up the answer
            answer = re.sub(r'\s+', ' ', answer)

            # If answer looks incomplete (starts with partial LaTeX), try to get more
            if answer.startswith('\\frac') or answer.startswith('\\begin') or answer.startswith('\\text'):
                # Try to find complete content
                full_match = re.search(r'(?:Answer|boxed)[:\s]*(.+?)(?:\||Confidence|$)', text, re.DOTALL | re.IGNORECASE)
                if full_match:
                    answer = full_match.group(1).strip()

            return answer

    # Fallback: take last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip() and not re.match(r'.*Confidence', l, re.I)]
    if lines:
        # Try to extract the last line that looks like an answer
        for line in reversed(lines):
            if re.search(r'\d|\\boxed|Answer', line, re.I):
                return line

    return lines[-1] if lines else None


def parse_confidence(text: str) -> Optional[float]:
    """Extract confidence score from model response."""
    patterns = [
        r"Confidence:\s*([01]?\.?\d+)",
        r"confidence[:\s]+([01]?\.?\d+)",
        r"([01]?\.?\d+)\s*(?:confidence|conf)[\s.]",
        r"\\boxed\{([01]?\.?\d+)\}",
        r"confidence is\s+\$?\\?boxed\{?([01]?\.?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                conf = float(match.group(1))
                if 0.0 <= conf <= 1.0:
                    return conf
            except ValueError:
                continue

    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} in solution text."""
    # Match \boxed{...} pattern
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(pred: str, truth: str) -> bool:
    """Check if predicted answer matches ground truth (normalized)."""
    # First, try to extract boxed answer from ground truth
    truth_answer = extract_boxed_answer(truth)

    # If no boxed answer, use the last line / numeric content
    if not truth_answer:
        # Try to extract numeric answer
        numbers = re.findall(r'-?\d+(?:\.\d+)?', truth)
        if numbers:
            truth_answer = numbers[-1]  # Usually the last number is the answer

    if not truth_answer:
        # Fall back to comparing normalized full text
        truth_answer = truth

    # Normalize both
    pred_clean = re.sub(r'[^\w\s]', '', pred.lower())
    truth_clean = re.sub(r'[^\w\s]', '', truth_answer.lower())

    # Remove extra whitespace
    pred_clean = ' '.join(pred_clean.split())
    truth_clean = ' '.join(truth_clean.split())

    return pred_clean == truth_clean


def generate_response(model, tokenizer, problem: str, max_new_tokens: int = 768) -> tuple[str, Optional[float]]:
    """Generate CoT response with confidence score."""
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    confidence = parse_confidence(response)

    return response, confidence


def compute_ece(confidences: list, correct: list, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences = [c if c is not None else 0.5 for c in confidences]
    confidences = [min(1.0, max(0.0, c)) for c in confidences]

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = [
            (conf, corr) for conf, corr in zip(confidences, correct)
            if bin_lower <= conf < bin_upper
        ]

        if in_bin:
            avg_confidence = sum(c for c, _ in in_bin) / len(in_bin)
            accuracy = sum(corr for _, corr in in_bin) / len(in_bin)
            ece += (len(in_bin) / len(confidences)) * abs(avg_confidence - accuracy)

    return ece


def main():
    print(f"[{datetime.now().isoformat()}] Starting calibration dataset generation")
    print(f"Model: {MODEL_NAME}")
    print(f"Pilot samples: {PILOT_SAMPLES}")
    print(f"Output: {OUTPUT_PATH}")

    # Write PID file for system monitoring
    pid_file = RESULTS_DIR / "gen_calibration_dataset.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"PID: {os.getpid()}")

    # Setup model
    model, tokenizer = setup_model()
    print("Model loaded successfully")

    # Load MATH dataset
    print("Loading MATH dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH", split="test")
    dataset = dataset.shuffle(seed=SEED).select(range(PILOT_SAMPLES))
    print(f"Loaded {len(dataset)} MATH problems")

    # Generate responses
    results = []
    valid_count = 0
    total = len(dataset)

    for idx, item in enumerate(dataset):
        problem = item['problem']
        ground_truth = item['solution']
        level = item.get('level', 'unknown')
        type_ = item.get('type', 'unknown')

        # Generate CoT response
        response, confidence = generate_response(model, tokenizer, problem)

        # Parse predicted answer
        predicted_answer = parse_final_answer(response)

        # Check correctness
        is_correct = False
        if predicted_answer and ground_truth:
            is_correct = normalize_answer(predicted_answer, ground_truth)

        # Record result
        record = {
            "idx": idx,
            "problem": problem,
            "ground_truth": ground_truth,
            "model_response": response,
            "predicted_answer": predicted_answer,
            "confidence": confidence,
            "is_correct": is_correct,
            "level": level,
            "type": type_,
        }
        results.append(record)

        if predicted_answer is not None:
            valid_count += 1

        # Progress update every 10 items
        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{total}] Valid: {valid_count}, Correct: {sum(r['is_correct'] for r in results)}")

        # Report progress for system monitor
        progress = {
            "task_id": "gen_calibration_dataset",
            "epoch": 1,
            "total_epochs": 1,
            "step": idx + 1,
            "total_steps": total,
            "loss": None,
            "metric": {
                "valid_count": valid_count,
                "correct_count": sum(r['is_correct'] for r in results)
            },
            "updated_at": datetime.now().isoformat(),
        }
        progress_file = RESULTS_DIR / "gen_calibration_dataset_PROGRESS.json"
        progress_file.write_text(json.dumps(progress))

        # Clear cache periodically
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Compute ECE
    confidences = [r['confidence'] for r in results]
    correct = [1 if r['is_correct'] else 0 for r in results]
    ece = compute_ece(confidences, correct)
    accuracy = sum(correct) / len(correct)

    print(f"\nGeneration complete:")
    print(f"  Valid problems: {valid_count}/{total}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ECE: {ece:.4f}")

    # Save results as JSONL
    with open(OUTPUT_PATH, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')
    print(f"Results saved to {OUTPUT_PATH}")

    # Compute summary statistics
    summary = {
        "task": "gen_calibration_dataset",
        "model": MODEL_NAME,
        "n_samples": total,
        "valid_count": valid_count,
        "accuracy": accuracy,
        "ece": ece,
        "pass_criteria_met": valid_count >= 80,
        "pass_criteria": ">=80 problems with valid (prompt, answer, correctness) triples",
        "results": {
            "correct_by_confidence": {
                "high_conf": sum(1 for r in results if r['confidence'] and r['confidence'] > 0.8 and r['is_correct']),
                "high_conf_total": sum(1 for r in results if r['confidence'] and r['confidence'] > 0.8),
                "low_conf_correct": sum(1 for r in results if r['confidence'] and r['confidence'] < 0.4 and r['is_correct']),
                "low_conf_total": sum(1 for r in results if r['confidence'] and r['confidence'] < 0.4),
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = RESULTS_DIR / "gen_calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Write DONE marker
    done_marker = RESULTS_DIR / "gen_calibration_dataset_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "gen_calibration_dataset",
        "status": "success" if valid_count >= 80 else "partial",
        "summary": f"Generated {valid_count}/{total} valid problems, ECE={ece:.4f}",
        "timestamp": datetime.now().isoformat(),
    }))
    print("Done marker written")

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    return summary


if __name__ == "__main__":
    main()
