#!/usr/bin/env python3
"""
EDW-Step-DPO L3-Only Ablation (pilot_g6)
Ablation: train EDW-Step-DPO using only Level-3 (conceptual) error steps.
Tests whether deep error training alone drives EDW-Step-DPO improvement.
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# ========== Configuration ==========
TASK_ID = "pilot_g6"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = REMOTE_BASE / "exp" / "results" / "pilots" / "g6_edw_l3_only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_PATH = REMOTE_BASE / "exp" / "results" / "pilots" / "dataset_preference_100.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"

# Training config
SEED = 42
NUM_TRAINING_STEPS = 200  # Same as G2 for fair comparison
LEARNING_RATE = 5e-7
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 1024
BETA = 0.1  # DPO KL penalty

# Depth weights: linear scheme (same as G2)
DEPTH_WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.0}

def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def write_pid():
    """Write PID file for system monitor."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] Written to {pid_file}: {os.getpid()}")

def report_progress(epoch, step, total_steps, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress_file.write_text(json.dumps(progress))

def mark_done(status="success", summary=""):
    """Write DONE marker file for system monitor."""
    # Clean up PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    # Merge final progress
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    # Write DONE marker
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[DONE] Written to {marker}: status={status}")

def load_dataset_l3_only():
    """Load and filter the preference dataset for DPO - L3 samples only."""
    print(f"[DATA] Loading L3-only dataset from {DATASET_PATH}")

    data = []
    depth_counts = {1: 0, 2: 0, 3: 0}
    total_pairs = 0

    with open(DATASET_PATH, 'r') as f:
        for line in f:
            entry = json.loads(line)

            # Flatten preference pairs to DPO format
            for pair in entry.get('preference_pairs', []):
                error_depth = pair.get('error_depth', 3)
                total_pairs += 1
                depth_counts[error_depth] = depth_counts.get(error_depth, 0) + 1

                # KEY DIFFERENCE FROM G2: Filter to L3 only
                if error_depth != 3:
                    continue

                step_weight = DEPTH_WEIGHTS.get(error_depth, 3.0)

                dpo_entry = {
                    "prompt": pair['prompt'],
                    "chosen": pair['chosen'],
                    "rejected": pair['rejected'],
                    "error_depth": error_depth,
                    "step_weight": step_weight,
                }
                data.append(dpo_entry)

    print(f"[DATA] Depth distribution in original dataset: {depth_counts}")
    print(f"[DATA] Total pairs in original: {total_pairs}")
    print(f"[DATA] L3 pairs selected: {len(data)}")

    # Filter out pairs with very short chosen/rejected
    filtered_data = [d for d in data if len(d["chosen"]) > 10 and len(d["rejected"]) > 10]
    print(f"[DATA] After filtering short pairs: {len(filtered_data)} pairs")

    # Save depth distribution info
    depth_info = {
        "original_depth_distribution": depth_counts,
        "total_original_pairs": total_pairs,
        "l3_pairs_selected": len(data),
        "l3_pairs_after_filtering": len(filtered_data),
    }
    with open(RESULTS_DIR / "depth_info.json", 'w') as f:
        json.dump(depth_info, f, indent=2)

    return filtered_data


def main():
    print("=" * 60)
    print(f"[EDW-Step-DPO L3-Only] Starting pilot_g6")
    print(f"[EDW-Step-DPO L3-Only] Ablation: Only Level-3 (conceptual) errors")
    print(f"[EDW-Step-DPO L3-Only] Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Set seed
    set_seed(SEED)

    # Write PID
    write_pid()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using {device}")

    # Check VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] Total: {vram_gb:.1f} GB")

    # Load tokenizer
    print(f"[MODEL] Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("[MODEL] Tokenizer loaded")

    # Load model with QLoRA config
    print(f"[MODEL] Loading model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for kbit training
    model.config.use_cache = False
    print("[MODEL] Base model loaded with QLoRA")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("[LORA] LoRA adapter applied")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load reference model (copy for DPO)
    print("[MODEL] Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_model = get_peft_model(ref_model, lora_config)
    ref_model.eval()

    # Load dataset (L3 only)
    raw_data = load_dataset_l3_only()

    # Check if we have enough samples
    if len(raw_data) < 5:
        print(f"[WARNING] Very few L3 samples ({len(raw_data)}). Training may be limited.")

    dataset = Dataset.from_list(raw_data)
    print(f"[DATA] Dataset created with {len(dataset)} samples")

    # Report initial progress
    report_progress(epoch=0, step=0, total_steps=NUM_TRAINING_STEPS, loss=None)

    # Configure DPO
    dpo_config = DPOConfig(
        output_dir=str(RESULTS_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        max_steps=NUM_TRAINING_STEPS,
        logging_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
        dataloader_drop_last=True,
    )

    # Create trainer
    print("[TRAIN] Creating DPO Trainer...")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("[TRAIN] Starting training...")
    train_result = trainer.train()

    # Save model
    print("[SAVE] Saving adapter...")
    trainer.save_model()
    trainer.save_state()

    # Final metrics
    final_loss = train_result.training_loss
    print(f"[TRAIN] Final training loss: {final_loss:.4f}")

    # Update progress
    report_progress(
        epoch=1,
        step=NUM_TRAINING_STEPS,
        total_steps=NUM_TRAINING_STEPS,
        loss=final_loss,
    )

    # Get depth distribution from dataset
    depth_distribution = {"L1": 0, "L2": 0, "L3": len(raw_data)}

    # Write summary
    summary = {
        "task_id": TASK_ID,
        "final_loss": float(final_loss),
        "num_steps": NUM_TRAINING_STEPS,
        "model": MODEL_NAME,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "depth_weights": DEPTH_WEIGHTS,
        "dataset_size": len(dataset),
        "depth_distribution": depth_distribution,
        "ablation_type": "L3_only",
        "status": "success",
        "note": "L3-only ablation: trained only on Level-3 (conceptual) error pairs. Comparing against G2 (all depths) to determine if deep errors alone drive improvement."
    }

    with open(RESULTS_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Mark done
    mark_done(status="success", summary=f"L3-only EDW-Step-DPO training complete. Final loss: {final_loss:.4f}, L3 samples: {len(raw_data)}")

    print("=" * 60)
    print(f"[EDW-Step-DPO L3-Only] Training complete!")
    print(f"[EDW-Step-DPO L3-Only] L3 samples used: {len(raw_data)}")
    print(f"[EDW-Step-DPO L3-Only] Model saved to {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[ERROR] Training failed: {e}")
        traceback.print_exc()

        # Mark as failed
        marker = RESULTS_DIR / f"{TASK_ID}_DONE"
        marker.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }))
        sys.exit(1)
