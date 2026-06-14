#!/usr/bin/env python3
"""
Pilot G1: Standard Step-DPO training on DeepSeek-Math-7B-Instruct
Train LoRA adapter (r=16, alpha=32) with standard DPO loss on 100 step-level preference pairs.

Pass criteria:
- Training completes without OOM
- Final loss < 0.5
- Accuracy >= G0 baseline
"""

import os
import sys
import json
import gc
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from datasets import Dataset

# Experiment configuration
TASK_ID = "train_g1_stepdpo"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
RESULTS_DIR = WORKSPACE / "exp/results/pilots/g1_uniform_stepdpo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-Math-7B-Instruct"
OUTPUT_DIR = RESULTS_DIR / "checkpoint"

# Training configuration (from task_plan.json)
LORA_RANK = 16
LORA_ALPHA = 32
BATCH_SIZE = 2
LEARNING_RATE = 1e-7
BETA = 0.1  # DPO KL penalty coefficient
MAX_STEPS = 500
SEED = 42
PILOT_SAMPLES = 100

# PID file
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"

def write_pid_file():
    """Write PID file for system monitor."""
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def report_progress(step=0, total_steps=MAX_STEPS, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_task_done(status="success", summary=""):
    """Write DONE marker file for system monitor."""
    # Clean up PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    # Write DONE marker
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"Task {TASK_ID} marked as DONE: {status}")

def load_preference_dataset():
    """Load and format preference dataset for DPO training."""
    print(f"Loading preference dataset (pilot: {PILOT_SAMPLES} samples)...")

    dataset_path = WORKSPACE / "exp/results/pilots/dataset_preference_100.jsonl"
    if not dataset_path.exists():
        # Try to find any available preference dataset
        for n in [80, 60, 40, 20]:
            dataset_path = WORKSPACE / f"exp/results/pilots/dataset_preference_{n}.jsonl"
            if dataset_path.exists():
                print(f"Using fallback dataset: {dataset_path}")
                break

    if not dataset_path.exists():
        raise FileNotFoundError(f"No preference dataset found in {WORKSPACE}/exp/results/pilots/")

    # Load JSONL dataset
    with open(dataset_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    # Convert to DPO format (prompt, chosen, rejected)
    prompts = []
    chosen = []
    rejected = []

    for item in raw_data[:PILOT_SAMPLES]:
        for pair in item.get("preference_pairs", [])[:3]:  # Limit pairs per problem
            prompts.append(pair["prompt"])
            chosen.append(pair["chosen"])
            rejected.append(pair["rejected"])

    print(f"Loaded {len(prompts)} DPO samples from {len(raw_data)} problems")

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
    })

    return dataset

def main():
    print("=" * 60)
    print(f"Starting {TASK_ID}: Standard Step-DPO training")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, beta: {BETA}")
    print(f"Max steps: {MAX_STEPS}, Pilot samples: {PILOT_SAMPLES}")
    print()

    # Write PID file immediately
    write_pid_file()
    report_progress(step=0, loss=None)

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Import training libraries
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig

    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for training
    model.config.use_cache = False

    # Setup LoRA
    print(f"Setting up LoRA (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load preference dataset
    dataset = load_preference_dataset()
    print(f"Dataset size: {len(dataset)}")

    # Create reference model (copy of original for DPO)
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()

    # DPO training configuration
    output_dir = str(OUTPUT_DIR)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        fp16=False,
        max_steps=MAX_STEPS,
        seed=SEED,
        remove_unused_columns=False,
        beta=BETA,
        max_length=512,
        report_to=[],  # Disable wandb/tensorboard
    )

    # Initialize DPO trainer
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    dpo_trainer.train()

    # Get final metrics
    train_history = dpo_trainer.state.log_history
    losses = [log.get("train_loss", None) for log in train_history if "train_loss" in log]
    final_loss = losses[-1] if losses else None
    if final_loss is None and losses:
        final_loss = np.mean(losses[-10:])

    print(f"\nTraining completed!")
    if final_loss is not None:
        print(f"Final loss: {final_loss:.4f}")
    else:
        print("Final loss not available")

    # Save model
    print(f"Saving model to {OUTPUT_DIR}...")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training metrics
    metrics = {
        "task_id": TASK_ID,
        "final_loss": float(final_loss) if final_loss is not None else None,
        "total_steps": dpo_trainer.state.global_step,
        "total_samples": len(dataset),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
    }

    metrics_file = RESULTS_DIR / f"{TASK_ID}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Determine pass/fail
    pass_criteria = {
        "training_completed": dpo_trainer.state.global_step > 0,
        "final_loss_lt_0.5": final_loss is not None and final_loss < 0.5,
        "no_oom": True,
    }

    passed = all(pass_criteria.values())
    status = "success" if passed else "partial"
    summary = f"Final loss: {final_loss:.4f if final_loss else 'N/A'}, Steps: {dpo_trainer.state.global_step}"
    if not passed:
        failed = [k for k, v in pass_criteria.items() if not v]
        summary += f", Failed: {failed}"

    # Mark task done
    mark_task_done(status=status, summary=summary)

    print("\n" + "=" * 60)
    print(f"Training {status.upper()}")
    print(f"Final loss: {final_loss:.4f if final_loss else 'N/A'} (target: < 0.5)")
    print(f"Total steps: {dpo_trainer.state.global_step}/{MAX_STEPS}")
    print("=" * 60)

    # Cleanup
    del model, ref_model, dpo_trainer
    torch.cuda.empty_cache()
    gc.collect()

    return 0 if passed else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        mark_task_done(status="failed", summary=str(e))
        sys.exit(1)
