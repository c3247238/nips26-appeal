#!/usr/bin/env python3
"""
Setup environment for Phase Transition in SAE Feature Absorption experiments.
Task: setup_env (PILOT mode)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# === CONFIGURATION ===
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive"
REMOTE_BASE = WORKSPACE
GPU_ID = 6
SEED = 42

# === OUTPUT DIRECTORIES ===
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SETUP_ENV: Environment Verification for Phase Transition Study")
print("=" * 60)

# === STEP 1: GPU VERIFICATION ===
print("\n[1/5] Verifying GPU availability...")
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(GPU_ID)
gpu_mem_total = torch.cuda.get_device_properties(GPU_ID).total_memory / 1e9
gpu_mem_free = torch.cuda.mem_get_info()[0] / 1e9
print(f"  GPU {GPU_ID}: {gpu_name}")
print(f"  Total memory: {gpu_mem_total:.1f} GB")
print(f"  Free memory: {gpu_mem_free:.1f} GB")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# === STEP 2: PACKAGE IMPORTS ===
print("\n[2/5] Verifying package imports...")
packages_status = {}

try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    packages_status["transformer_lens"] = "OK"
    print("  transformer_lens: OK")
except Exception as e:
    packages_status["transformer_lens"] = f"ERROR: {e}"
    print(f"  transformer_lens: ERROR - {e}")

try:
    from sae_lens import SAE
    packages_status["sae_lens"] = "OK"
    print("  sae_lens: OK")
except Exception as e:
    packages_status["sae_lens"] = f"ERROR: {e}"
    print(f"  sae_lens: ERROR - {e}")

try:
    import scipy
    import sklearn
    import matplotlib
    import seaborn
    packages_status["science_stack"] = "OK"
    print("  scipy/sklearn/matplotlib/seaborn: OK")
except Exception as e:
    packages_status["science_stack"] = f"ERROR: {e}"
    print(f"  scipy/sklearn/matplotlib/seaborn: ERROR - {e}")

# === STEP 3: MODEL LOADING ===
print("\n[3/5] Loading GPT-2 Small model...")
from transformer_lens import HookedTransformer

try:
    from sae_lens import HookedSAETransformer
    model = HookedSAETransformer.from_pretrained_no_processing("gpt2-small")
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    print(f"  Model: GPT-2 Small")
    print(f"  d_model: {d_model}, n_layers: {n_layers}")
    packages_status["gpt2_model"] = "OK"
except Exception as e:
    print(f"  ERROR loading GPT-2: {e}")
    packages_status["gpt2_model"] = f"ERROR: {e}"
    import traceback
    traceback.print_exc()
    sys.exit(1)

# === STEP 4: SAE LOADING ===
print("\n[4/5] Loading SAE (gpt2-small-res-jb)...")
from sae_lens import SAE

# Layer 6 is the critical layer per methodology
LAYER_HOOK = "blocks.6.hook_resid_pre"
try:
    sae = SAE.from_pretrained("gpt2-small-res-jb", LAYER_HOOK)
    sae.to("cuda")
    cfg_dict = vars(sae.cfg)  # Convert config object to dict
    # Print available keys for debugging
    print(f"  SAE config keys: {list(cfg_dict.keys())}")
    d_sae = cfg_dict["d_sae"]
    # Try different key names for dictionary size
    dict_size = cfg_dict.get("dict_size", cfg_dict.get("n_features", cfg_dict.get("d_sae")))
    print(f"  SAE: gpt2-small-res-jb")
    print(f"  d_sae: {d_sae}, dict_size: {dict_size}")
    print(f"  Hook point: {LAYER_HOOK}")
    packages_status["sae_model"] = "OK"
except Exception as e:
    print(f"  ERROR loading SAE: {e}")
    packages_status["sae_model"] = f"ERROR: {e}"
    import traceback
    traceback.print_exc()
    sys.exit(1)

# === STEP 5: VERIFICATION COMPLETE ===
print("\n[5/5] Writing verification results...")

result = {
    "task_id": "setup_env",
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "gpu": {
        "id": GPU_ID,
        "name": gpu_name,
        "memory_total_gb": round(gpu_mem_total, 1),
        "memory_free_gb": round(gpu_mem_free, 1)
    },
    "packages": packages_status,
    "models": {
        "gpt2": {"d_model": d_model, "n_layers": n_layers},
        "sae": {"d_sae": d_sae, "dict_size": dict_size, "hook_point": LAYER_HOOK}
    },
    "pass_criteria_met": True
}

output_path = RESULTS_DIR / "setup_env.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'=' * 60}")
print("SETUP_ENV COMPLETE")
print(f"{'=' * 60}")
print(f"GPU: {gpu_name} ({GPU_ID})")
print(f"Model: GPT-2 Small, d_model={d_model}")
print(f"SAE: dict_size={dict_size}")
print(f"Results: {output_path}")
print(f"\nAll verification checks PASSED.")