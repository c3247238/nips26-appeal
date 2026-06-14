#!/usr/bin/env python3
"""
Pilot: Absorption Detection using SAEBench official evaluation
Model: GPT-2 Small, Layer 8, 32K dictionary
Uses SAEBench's official absorption metric pipeline
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/current/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Task identification
TASK_ID = "pilot_absorption"
PID_FILE = RESULTS_DIR.parent / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR.parent / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR.parent / f"{TASK_ID}_DONE"

# Write PID file immediately
PID_FILE.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))

def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    marker = {
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))

# JSON encoder that handles numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[{TASK_ID}] Starting SAEBench absorption detection pilot")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")
print(f"  Seed: {SEED}")

# Imports
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_bench.evals.absorption.main import (
    run_feature_absortion_experiment,
    run_k_sparse_probing_experiment,
)

# Model and SAE configuration
MODEL_NAME = "gpt2"
SAE_RELEASE = "gpt2-small-resid-post-v5-32k"
SAE_ID = "blocks.8.hook_resid_post"
SAE_LAYER = 8
SAE_DICT_SIZE = 32768

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 5, metric={"stage": "loading_model"})

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=DEVICE,
    dtype=torch.float32,
)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")

print(f"\nLoading SAE: {SAE_RELEASE}/{SAE_ID}")
report_progress(1, 5, metric={"stage": "loading_sae"})

sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE,
)
print(f"  SAE loaded: d_sae={sae.cfg.d_sae}")

# Set up experiment directories
EXPERIMENT_DIR = RESULTS_DIR / "saebench_absorption"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
K_SPARSE_DIR = EXPERIMENT_DIR / "k_sparse_probing"
PROBES_DIR = EXPERIMENT_DIR / "probes"
FEATURE_ABS_DIR = EXPERIMENT_DIR / "feature_absorption"

SAE_NAME = "gpt2_small_layer8_32k"
PROMPT_TEMPLATE = "{word}:"
PROMPT_TOKEN_POS = -2
MAX_K_VALUE = 10

# Step 1: Run k-sparse probing
print(f"\nStep 1: Running k-sparse probing experiment...")
report_progress(2, 5, metric={"stage": "k_sparse_probing"})

try:
    k_sparse_df = run_k_sparse_probing_experiment(
        model=model,
        sae=sae,
        layer=SAE_LAYER,
        sae_name=SAE_NAME,
        max_k_value=MAX_K_VALUE,
        prompt_template=PROMPT_TEMPLATE,
        prompt_token_pos=PROMPT_TOKEN_POS,
        device=DEVICE,
        experiment_dir=K_SPARSE_DIR,
        probes_dir=PROBES_DIR,
        force=True,
        k_sparse_probe_num_epochs=50,
        eval_batch_size=24,
        verbose=True,
    )
    print(f"  K-sparse probing complete: {len(k_sparse_df)} rows")
    print(f"  Columns: {list(k_sparse_df.columns)}")
    print(f"  Sample:\n{k_sparse_df.head()}")
except Exception as e:
    print(f"  ERROR in k-sparse probing: {e}")
    import traceback
    traceback.print_exc()
    mark_done("failed", f"k_sparse_probing failed: {e}")
    sys.exit(1)

# Step 2: Run feature absorption
print(f"\nStep 2: Running feature absorption experiment...")
report_progress(4, 5, metric={"stage": "feature_absorption"})

try:
    absorption_df = run_feature_absortion_experiment(
        model=model,
        sae=sae,
        layer=SAE_LAYER,
        sae_name=SAE_NAME,
        max_k_value=MAX_K_VALUE,
        prompt_template=PROMPT_TEMPLATE,
        prompt_token_pos=PROMPT_TOKEN_POS,
        device=DEVICE,
        experiment_dir=FEATURE_ABS_DIR,
        sparse_probing_experiment_dir=K_SPARSE_DIR,
        probes_dir=PROBES_DIR,
        force=True,
        batch_size=10,
    )
    print(f"  Feature absorption complete: {len(absorption_df)} rows")
    print(f"  Columns: {list(absorption_df.columns)}")
    print(f"  Sample:\n{absorption_df.head()}")
except Exception as e:
    print(f"  ERROR in feature absorption: {e}")
    import traceback
    traceback.print_exc()
    mark_done("failed", f"feature_absorption failed: {e}")
    sys.exit(1)

# Step 3: Analyze results
print(f"\nStep 3: Analyzing absorption results...")
report_progress(5, 5, metric={"stage": "analyzing_results"})

# Extract absorption metrics per letter
# The absorption_df should have columns like:
# - letter
# - absorption_rate
# - full_absorption_rate
# - absorption_fraction
# etc.

# Check available columns
print(f"  Available columns: {list(absorption_df.columns)}")

# Try to extract per-letter absorption rates
letter_absorption = {}
if 'letter' in absorption_df.columns:
    for _, row in absorption_df.iterrows():
        letter = row['letter']
        letter_absorption[letter] = {
            col: float(row[col]) if pd.notna(row[col]) else 0.0
            for col in absorption_df.columns if col != 'letter'
        }
else:
    # The df might be indexed differently
    print(f"  DataFrame index: {absorption_df.index}")
    print(f"  Full sample:\n{absorption_df}")

# Summary
if letter_absorption:
    absorption_rates = []
    for letter, metrics in letter_absorption.items():
        # Try different possible column names for absorption rate
        rate = metrics.get('absorption_rate',
                  metrics.get('full_absorption_rate',
                    metrics.get('absorption_fraction', 0.0)))
        absorption_rates.append(rate)

    print(f"\n{'='*60}")
    print(f"ABSORPTION DETECTION RESULTS (SAEBench)")
    print(f"{'='*60}")
    print(f"Letters analyzed: {len(letter_absorption)}")
    if absorption_rates:
        print(f"Mean absorption rate: {np.mean(absorption_rates):.4f}")
        print(f"Std absorption rate: {np.std(absorption_rates):.4f}")
        print(f"Min absorption rate: {np.min(absorption_rates):.4f}")
        print(f"Max absorption rate: {np.max(absorption_rates):.4f}")

    print(f"\nPer-letter results:")
    for letter in sorted(letter_absorption.keys()):
        metrics = letter_absorption[letter]
        rate = metrics.get('absorption_rate',
                  metrics.get('full_absorption_rate',
                    metrics.get('absorption_fraction', 0.0)))
        print(f"  {letter}: {rate:.4f}")

# Save results
output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "sae_release": SAE_RELEASE,
    "sae_id": SAE_ID,
    "layer": SAE_LAYER,
    "dictionary_size": SAE_DICT_SIZE,
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "timestamp": datetime.now().isoformat(),
    "method": "SAEBench official absorption evaluation",
    "letter_absorption": letter_absorption,
    "k_sparse_summary": {
        "n_rows": len(k_sparse_df),
        "columns": list(k_sparse_df.columns),
    },
    "absorption_summary": {
        "n_rows": len(absorption_df),
        "columns": list(absorption_df.columns),
    },
}

output_file = RESULTS_DIR / "absorption_layer8_16k.json"
output_file.write_text(json.dumps(output, cls=NumpyEncoder, indent=2))
print(f"\nResults saved to: {output_file}")

# Also save raw dataframes
k_sparse_df.to_csv(RESULTS_DIR / "k_sparse_probing.csv", index=False)
absorption_df.to_csv(RESULTS_DIR / "absorption_results.csv", index=False)
print(f"Raw data saved to CSV")

# Update progress and mark done
summary_text = f"SAEBench absorption detection complete. "
summary_text += f"K-sparse: {len(k_sparse_df)} rows, Absorption: {len(absorption_df)} rows"

mark_done(status="success", summary=summary_text)
print(f"\n{'='*60}")
print(f"Pilot absorption detection COMPLETE")
print(f"{'='*60}")

# Clean up
if PID_FILE.exists():
    PID_FILE.unlink()
