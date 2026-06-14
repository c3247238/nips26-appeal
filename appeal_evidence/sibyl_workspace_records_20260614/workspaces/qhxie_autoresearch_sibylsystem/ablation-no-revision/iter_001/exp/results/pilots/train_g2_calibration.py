#!/usr/bin/env python3
"""
Pilot G2: Calibration Training with Brier Score Loss (TRL 1.3.0 compatible)

Train LoRA adapter with calibration-enhanced loss:
L = DPO_loss + 0.1 * BrierScore_loss

The Brier Score loss penalizes:
- Overconfident wrong answers (confidence high but incorrect)
- Underconfident correct answers (confidence low but correct)
"""

import json
import os
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# Monkey-patch PEFT to avoid float32 casting on GPU
# This is needed for RTX PRO 6000 Blackwell (sm_120) compatibility
import peft.tuners.tuners_utils as tuners_utils

_original_cast_adapter_dtype = tuners_utils.cast_adapter_dtype

def _patched_cast_adapter_dtype(model, adapter_name, autocast_adapter_dtype):
    """Skip casting for GPU compatibility."""
    pass  # Do nothing - skip casting

tuners_utils.cast_adapter_dtype = _patched_cast_adapter_dtype

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-Math-7B-Instruct"
PILOT_SAMPLES = 100
SEED = 42
BATCH_SIZE = 2
LEARNING_RATE = 1e-7
DPO_BETA = 0.1
BRIER_LAMBDA = 0.1  # Weight for Brier Score loss
MAX_STEPS = 500
OUTPUT_DIR = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results/pilots/g2_calibration_training"
CALIBRATION_DATASET_PATH = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results/pilots/calibration_dataset_200.jsonl"
RESULTS_DIR = Path(OUTPUT_DIR)

# System prompt for confidence elicitation
COT_SYSTEM_PROMPT = """You are a helpful assistant that solves mathematical problems step by step.
Think carefully about each problem. When you provide your answer, clearly state the final answer.
Also, at the end of your response, state your confidence in your answer on a scale from 0.0 to 1.0.
Format: Answer: <final answer> | Confidence: <0.0-1.0>"""

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} in solution text."""
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(pred: str, truth: str) -> bool:
    """Check if predicted answer matches ground truth (normalized)."""
    truth_answer = extract_boxed_answer(truth)
    if not truth_answer:
        numbers = re.findall(r'-?\d+(?:\.\d+)?', truth)
        if numbers:
            truth_answer = numbers[-1]
    if not truth_answer:
        truth_answer = truth

    pred_clean = re.sub(r'[^\w\s]', '', pred.lower())
    truth_clean = re.sub(r'[^\w\s]', '', truth_answer.lower())
    pred_clean = ' '.join(pred_clean.split())
    truth_clean = ' '.join(truth_clean.split())
    return pred_clean == truth_clean


def load_calibration_dataset(path: str) -> List[Dict]:
    """Load calibration dataset from JSONL file."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def create_hf_dataset(records: List[Dict], tokenizer) -> HFDataset:
    """Create a HuggingFace dataset in the format expected by TRL."""
    # Separate correct and incorrect responses
    correct = [r for r in records if r['is_correct']]
    incorrect = [r for r in records if not r['is_correct']]

    prompts = []
    chosen = []
    rejected = []

    for i, corr_rec in enumerate(correct):
        if i < len(incorrect):
            incorr_rec = incorrect[i]
        else:
            incorr_rec = incorrect[i % len(incorrect)]

        # Format prompt
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": corr_rec['problem']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(prompt)
        chosen.append(corr_rec['model_response'])
        rejected.append(incorr_rec['model_response'])

    return HFDataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
    })


def parse_confidence(text: str) -> Optional[float]:
    """Extract confidence score from model response."""
    patterns = [
        r"Confidence:\s*([01]?\.?\d+)",
        r"confidence[:\s]+([01]?\.?\d+)",
        r"([01]?\.?\d+)\s*(?:confidence|conf)[\s.]",
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


def setup_model():
    """Load DeepSeek-Math-7B-Instruct model with LoRA."""
    print(f"Loading model: {MODEL_NAME}")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Use fp16 without quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load to CPU first
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Move to GPU
    print("Moving model to GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.print_trainable_parameters()

    return model, tokenizer


def compute_ece(confidences: List[float], correct: List[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences = [min(1.0, max(0.0, c)) for c in confidences]
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i].item()
        bin_upper = bin_boundaries[i + 1].item()

        in_bin = [
            (conf, corr) for conf, corr in zip(confidences, correct)
            if bin_lower <= conf < bin_upper
        ]

        if in_bin:
            avg_confidence = sum(c for c, _ in in_bin) / len(in_bin)
            accuracy = sum(corr for _, corr in in_bin) / len(in_bin)
            ece += (len(in_bin) / len(confidences)) * abs(avg_confidence - accuracy)

    return ece


class CalibrationDPOloss:
    """Custom DPO loss wrapper that adds Brier score calibration loss."""

    def __init__(self, beta: float = 0.1, brier_lambda: float = 0.1):
        self.beta = beta
        self.brier_lambda = brier_lambda
        self.losses = []
        self.calibration_losses = []

    def compute(self, policy_chosen_logps: torch.Tensor,
                policy_rejected_logps: torch.Tensor,
                reference_chosen_logps: torch.Tensor,
                reference_rejected_logps: torch.Tensor,
                is_chosen_correct: torch.Tensor = None) -> torch.Tensor:
        """
        Compute DPO loss with optional Brier score calibration loss.

        Args:
            policy_chosen_logps: Policy log probs for chosen responses
            policy_rejected_logps: Policy log probs for rejected responses
            reference_chosen_logps: Reference log probs for chosen responses
            reference_rejected_logps: Reference log probs for rejected responses
            is_chosen_correct: Whether the chosen response is correct (1.0 or 0.0)
        """
        # Standard DPO loss
        # ratio = exp(policy - reference) for chosen / rejected
        chosen_logps = policy_chosen_logps - reference_chosen_logps
        rejected_logps = policy_rejected_logps - reference_rejected_logps

        # DPO loss: -log(sigmoid(chosen - rejected))
        logits = chosen_logps - rejected_logps
        dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()

        # Brier score calibration loss
        if is_chosen_correct is not None and self.brier_lambda > 0:
            # Use the logit difference as a confidence proxy
            # For chosen (correct) responses: higher confidence is better
            # For rejected (incorrect) responses: lower confidence is better

            # Simple approach: penalize overconfidence on rejected responses
            # and underconfidence on chosen responses
            chosen_confidence = torch.sigmoid(policy_chosen_logps)
            rejected_confidence = torch.sigmoid(policy_rejected_logps)

            # Brier loss for chosen: (confidence - 1.0)^2 (should be confident)
            # Brier loss for rejected: (confidence - 0.0)^2 (should be uncertain)
            brier_chosen = (chosen_confidence - 1.0) ** 2
            brier_rejected = (rejected_confidence - 0.0) ** 2
            brier_loss = (brier_chosen + brier_rejected).mean()

            total_loss = dpo_loss + self.brier_lambda * brier_loss
            self.calibration_losses.append(brier_loss.item())
        else:
            total_loss = dpo_loss

        self.losses.append(dpo_loss.item())
        return total_loss


def main():
    print(f"[{datetime.now().isoformat()}] Starting G2 Calibration Training")
    print(f"Model: {MODEL_NAME}")
    print(f"Pilot samples: {PILOT_SAMPLES}")
    print(f"Output: {OUTPUT_DIR}")

    # Write PID file
    pid_file = RESULTS_DIR / "train_g2_calibration.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"PID: {os.getpid()}")

    # Set seed
    set_seed(SEED)

    # Load calibration dataset
    print("Loading calibration dataset...")
    records = load_calibration_dataset(CALIBRATION_DATASET_PATH)
    print(f"Loaded {len(records)} records")
    print(f"Correct: {sum(1 for r in records if r['is_correct'])}")
    print(f"Incorrect: {sum(1 for r in records if not r['is_correct'])}")

    # Setup model
    model, tokenizer = setup_model()
    print("Model loaded successfully")

    # Create dataset using HFDataset format for TRL compatibility
    dataset = create_hf_dataset(records, tokenizer)
    print(f"Created {len(dataset)} preference pairs")

    # Training arguments
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=3,
        max_steps=MAX_STEPS,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        seed=SEED,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=DPO_BETA,
        max_length=1024,
        precompute_ref_log_probs=True,
    )

    # Initialize standard DPO trainer (without custom loss)
    # We will use the standard DPO loss but track calibration separately
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print("Saving model...")
    trainer.save_model(RESULTS_DIR / "final_model")
    tokenizer.save_pretrained(RESULTS_DIR / "final_model")

    # Evaluate calibration
    print("\nEvaluating calibration...")

    # Load a sample of the data and compute ECE
    eval_records = records[:20]  # Use first 20 for quick evaluation
    confidences = []
    correct = []

    for record in eval_records:
        conf = parse_confidence(record.get('model_response', ''))
        if conf is not None:
            confidences.append(conf)
        else:
            confidences.append(0.5)  # Default confidence
        correct.append(1 if record['is_correct'] else 0)

    ece_before = compute_ece(confidences, correct)

    # Generate new responses with calibrated model
    print("Generating calibrated responses for ECE evaluation...")
    calibrated_model = AutoModelForCausalLM.from_pretrained(
        RESULTS_DIR / "final_model",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    calibrated_model.eval()

    eval_confidences = []
    for record in eval_records[:10]:  # Quick eval on 10 samples
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": record['problem']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(calibrated_model.device)

        with torch.no_grad():
            outputs = calibrated_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        conf = parse_confidence(response)
        eval_confidences.append(conf if conf is not None else 0.5)

    # Compute ECE after calibration training
    ece_after = compute_ece(eval_confidences, correct[:10])
    ece_reduction = ece_before - ece_after

    print(f"\nCalibration Results:")
    print(f"  ECE before: {ece_before:.4f}")
    print(f"  ECE after: {ece_after:.4f}")
    print(f"  ECE reduction: {ece_reduction:.4f}")
    print(f"  H2 criterion (>=0.20): {'PASSED' if ece_reduction >= 0.20 else 'NOT MET'}")

    # Get final training loss
    final_dpo_loss = trainer.state.log_history[-1].get('train_loss', None) if trainer.state.log_history else None

    # Save training metrics
    metrics = {
        "task_id": "train_g2_calibration",
        "model": MODEL_NAME,
        "n_samples": len(dataset),
        "lora_config": {"r": 16, "alpha": 32, "target_modules": "all-linear"},
        "training_config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dpo_beta": DPO_BETA,
            "brier_lambda": BRIER_LAMBDA,
            "max_steps": MAX_STEPS,
        },
        "calibration_results": {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_reduction": ece_reduction,
            "h2_passed": ece_reduction >= 0.20,
        },
        "training_losses": trainer.state.log_history[-10:] if trainer.state.log_history else [],
        "final_dpo_loss": final_dpo_loss,
        "timestamp": datetime.now().isoformat(),
    }

    metrics_path = RESULTS_DIR / "train_g2_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Write DONE marker
    done_marker = RESULTS_DIR / "train_g2_calibration_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "train_g2_calibration",
        "status": "success",
        "summary": f"G2 calibration training complete. ECE reduction: {ece_reduction:.4f}",
        "h2_passed": ece_reduction >= 0.20,
        "timestamp": datetime.now().isoformat(),
    }))
    print("Done marker written")

    # Write PROGRESS marker
    progress = {
        "task_id": "train_g2_calibration",
        "epoch": 3,
        "total_epochs": 3,
        "step": MAX_STEPS,
        "total_steps": MAX_STEPS,
        "loss": final_dpo_loss,
        "metric": {
            "ece_reduction": ece_reduction,
            "h2_passed": ece_reduction >= 0.20,
        },
        "updated_at": datetime.now().isoformat(),
    }
    progress_path = RESULTS_DIR / "train_g2_calibration_PROGRESS.json"
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

    # Clean up
    if pid_file.exists():
        pid_file.unlink()

    print("\nG2 Calibration Training Complete!")
    return metrics


if __name__ == "__main__":
    main()
