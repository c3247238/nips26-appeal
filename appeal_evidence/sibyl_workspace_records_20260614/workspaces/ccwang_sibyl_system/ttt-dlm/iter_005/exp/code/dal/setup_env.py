#!/usr/bin/env python3
"""
setup_env.py — Environment Setup & Model/Dataset Download for DaL (Denoising-as-Learning)

Task: setup_env
Mode: PILOT (verify loading with single forward pass)

Steps:
1. Install missing pip packages (einops, etc.)
2. Download LLaDA-8B-Instruct checkpoint
3. Download all required datasets (GSM8K, MATH500, HumanEval, MBPP, ARC-Challenge, Countdown, OpenWebText subset)
4. Create project directory structure
5. Verify model loading + single forward pass for Dream-7B and LLaDA-8B
6. Write setup_complete.json
"""

import json
import os
import sys
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# === Configuration ===
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT = "ttt-dlm"
PROJECT_DIR = f"{REMOTE_BASE}/projects/{PROJECT}"
SHARED_DIR = f"{REMOTE_BASE}/shared"
MODELS_DIR = f"{REMOTE_BASE}/models"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
LOGS_DIR = f"{PROJECT_DIR}/exp/logs"
TASK_ID = "setup_env"
SEED = 42

# PID file for system recovery
pid_file = Path(LOGS_DIR) / f"{TASK_ID}.pid"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/pilots", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/full", exist_ok=True)
pid_file.write_text(str(os.getpid()))

# === Progress reporting ===
def report_progress(step, total_steps, description, status="running"):
    progress = Path(LOGS_DIR) / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "loss": None,
        "metric": {"description": description, "status": status},
        "updated_at": datetime.now().isoformat(),
    }))
    print(f"[{step}/{total_steps}] {description} ({status})")

def mark_done(status="success", summary=""):
    pid_f = Path(LOGS_DIR) / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = Path(LOGS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(LOGS_DIR) / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

TOTAL_STEPS = 7

# === Step 1: Install missing packages ===
report_progress(1, TOTAL_STEPS, "Installing missing pip packages")
try:
    missing = []
    for pkg in ["einops"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
        print(f"  Installed: {missing}")
    else:
        print("  All packages already installed.")
except Exception as e:
    print(f"  WARNING: pip install failed: {e}")

# === Step 2: Download LLaDA-8B-Instruct ===
report_progress(2, TOTAL_STEPS, "Checking/downloading LLaDA-8B-Instruct")

llada_dir = f"{MODELS_DIR}/LLaDA-8B-Instruct"
llada_shared = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"

if os.path.exists(llada_dir) and any(f.endswith(".safetensors") for f in os.listdir(llada_dir)):
    print(f"  LLaDA-8B-Instruct already exists at {llada_dir}")
else:
    print(f"  Downloading LLaDA-8B-Instruct from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="GSAI-ML/LLaDA-8B-Instruct",
            local_dir=llada_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded to {llada_dir}")
    except Exception as e:
        print(f"  ERROR downloading LLaDA-8B-Instruct: {e}")
        traceback.print_exc()

# Create shared symlink if not exists
if os.path.exists(llada_dir) and not os.path.exists(llada_shared):
    os.makedirs(os.path.dirname(llada_shared), exist_ok=True)
    os.symlink(llada_dir, llada_shared)
    print(f"  Created symlink: {llada_shared} -> {llada_dir}")

# === Step 3: Download datasets ===
report_progress(3, TOTAL_STEPS, "Downloading datasets")

import datasets as hf_datasets

dataset_configs = [
    {"name": "gsm8k", "hf_path": "openai/gsm8k", "hf_split": "test", "hf_name": "main"},
    {"name": "math500", "hf_path": "HuggingFaceH4/MATH-500", "hf_split": "test", "hf_name": None},
    {"name": "humaneval", "hf_path": "openai/openai_humaneval", "hf_split": "test", "hf_name": None},
    {"name": "mbpp", "hf_path": "google-research-datasets/mbpp", "hf_split": "test", "hf_name": None},
    {"name": "arc_challenge", "hf_path": "allenai/ai2_arc", "hf_split": "test", "hf_name": "ARC-Challenge"},
    {"name": "openwebtext_10k", "hf_path": "Skylion007/openwebtext", "hf_split": "train", "hf_name": None, "max_samples": 10000},
]

datasets_dir = f"{SHARED_DIR}/datasets"
os.makedirs(datasets_dir, exist_ok=True)

for dc in dataset_configs:
    ds_path = f"{datasets_dir}/{dc['name']}"
    if os.path.exists(ds_path) and os.listdir(ds_path):
        print(f"  {dc['name']}: already cached at {ds_path}")
        continue
    print(f"  {dc['name']}: downloading from {dc['hf_path']}...")
    try:
        kwargs = {"path": dc["hf_path"], "split": dc["hf_split"]}
        if dc.get("hf_name"):
            kwargs["name"] = dc["hf_name"]
        ds = hf_datasets.load_dataset(**kwargs)
        if dc.get("max_samples") and len(ds) > dc["max_samples"]:
            ds = ds.select(range(dc["max_samples"]))
        ds.save_to_disk(ds_path)
        print(f"  {dc['name']}: saved {len(ds)} samples to {ds_path}")
    except Exception as e:
        print(f"  ERROR downloading {dc['name']}: {e}")
        traceback.print_exc()

# Countdown dataset — try Dream repo or generate synthetic
countdown_path = f"{datasets_dir}/countdown"
if os.path.exists(countdown_path) and os.listdir(countdown_path):
    print(f"  countdown: already cached at {countdown_path}")
else:
    print(f"  countdown: generating synthetic dataset...")
    try:
        import random
        random.seed(SEED)
        os.makedirs(countdown_path, exist_ok=True)
        samples = []
        for i in range(1000):
            target = random.randint(100, 999)
            nums = sorted(random.sample(range(1, 100), 6))
            samples.append({
                "id": f"countdown_{i}",
                "target": target,
                "numbers": nums,
                "prompt": f"Use the numbers {nums} with +, -, *, / to make {target}. Show your work step by step.",
            })
        with open(f"{countdown_path}/data.json", "w") as f:
            json.dump(samples, f, indent=2)
        print(f"  countdown: generated {len(samples)} samples")
    except Exception as e:
        print(f"  ERROR generating countdown: {e}")

# === Step 4: Create project directory structure ===
report_progress(4, TOTAL_STEPS, "Creating project directory structure")

dirs_to_create = [
    f"{PROJECT_DIR}/exp/code/dal",
    f"{PROJECT_DIR}/exp/code/baselines",
    f"{PROJECT_DIR}/exp/code/analysis",
    f"{PROJECT_DIR}/exp/code/configs",
    f"{PROJECT_DIR}/exp/results/pilots",
    f"{PROJECT_DIR}/exp/results/full",
    f"{PROJECT_DIR}/exp/logs",
]
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"  Created: {d}")

# === Step 5: Verify Dream-7B loading + forward pass ===
report_progress(5, TOTAL_STEPS, "Verifying Dream-7B loading + forward pass")

import torch
torch.manual_seed(SEED)

dream_dir = f"{MODELS_DIR}/Dream-v0-Instruct-7B"
dream_result = {"model": "Dream-7B-Instruct", "status": "unknown"}

try:
    from transformers import AutoTokenizer, AutoConfig
    print(f"  Loading Dream-7B tokenizer from {dream_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(dream_dir, trust_remote_code=True)
    print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    config = AutoConfig.from_pretrained(dream_dir, trust_remote_code=True)
    print(f"  Config loaded. Hidden size: {config.hidden_size}, Num layers: {config.num_hidden_layers}")

    from transformers import AutoModelForCausalLM
    print(f"  Loading Dream-7B model (this may take a minute)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        dream_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    print(f"  Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Single forward pass
    test_input = tokenizer("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**test_input)
    logits_shape = output.logits.shape if hasattr(output, 'logits') else "N/A"
    print(f"  Forward pass OK. Output logits shape: {logits_shape}")

    dream_result = {
        "model": "Dream-7B-Instruct",
        "status": "success",
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "vocab_size": tokenizer.vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "logits_shape": str(logits_shape),
        "device": str(device),
        "dtype": "bfloat16",
    }

    # Free memory
    del model, output, test_input
    torch.cuda.empty_cache()
    import gc; gc.collect()

except Exception as e:
    dream_result = {"model": "Dream-7B-Instruct", "status": "error", "error": str(e)}
    print(f"  ERROR: {e}")
    traceback.print_exc()

# === Step 6: Verify LLaDA-8B loading + forward pass ===
report_progress(6, TOTAL_STEPS, "Verifying LLaDA-8B loading + forward pass")

llada_result = {"model": "LLaDA-8B-Instruct", "status": "unknown"}

if os.path.exists(llada_dir) and any(f.endswith(".safetensors") for f in os.listdir(llada_dir)):
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
        print(f"  Loading LLaDA-8B tokenizer from {llada_dir}...")
        tokenizer_llada = AutoTokenizer.from_pretrained(llada_dir, trust_remote_code=True)
        print(f"  Tokenizer loaded. Vocab size: {tokenizer_llada.vocab_size}")

        config_llada = AutoConfig.from_pretrained(llada_dir, trust_remote_code=True)
        print(f"  Config loaded. Hidden size: {config_llada.hidden_size}, Num layers: {config_llada.num_hidden_layers}")

        print(f"  Loading LLaDA-8B model (this may take a minute)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # LLaDA uses AutoModelForCausalLM with trust_remote_code
        model_llada = AutoModelForCausalLM.from_pretrained(
            llada_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        print(f"  Model loaded on {device}. Parameters: {sum(p.numel() for p in model_llada.parameters()):,}")

        # Single forward pass
        test_input = tokenizer_llada("Hello, world!", return_tensors="pt").to(device)
        with torch.no_grad():
            output = model_llada(**test_input)
        logits_shape = output.logits.shape if hasattr(output, 'logits') else "N/A"
        print(f"  Forward pass OK. Output logits shape: {logits_shape}")

        llada_result = {
            "model": "LLaDA-8B-Instruct",
            "status": "success",
            "hidden_size": config_llada.hidden_size,
            "num_layers": config_llada.num_hidden_layers,
            "vocab_size": tokenizer_llada.vocab_size,
            "parameters": sum(p.numel() for p in model_llada.parameters()),
            "logits_shape": str(logits_shape),
            "device": str(device),
            "dtype": "bfloat16",
        }

        del model_llada, output, test_input
        torch.cuda.empty_cache()
        import gc; gc.collect()

    except Exception as e:
        llada_result = {"model": "LLaDA-8B-Instruct", "status": "error", "error": str(e)}
        print(f"  ERROR: {e}")
        traceback.print_exc()
else:
    llada_result = {"model": "LLaDA-8B-Instruct", "status": "not_downloaded", "error": "Model files not found"}
    print(f"  WARNING: LLaDA-8B-Instruct not found at {llada_dir}")

# === Step 7: Write setup_complete.json ===
report_progress(7, TOTAL_STEPS, "Writing setup_complete.json")

# Verify datasets
dataset_status = {}
for dc in dataset_configs + [{"name": "countdown"}]:
    ds_path = f"{datasets_dir}/{dc['name']}"
    if os.path.exists(ds_path):
        if dc["name"] == "countdown":
            data_file = f"{ds_path}/data.json"
            if os.path.exists(data_file):
                with open(data_file) as f:
                    n_samples = len(json.load(f))
                dataset_status[dc["name"]] = {"status": "ok", "samples": n_samples, "path": ds_path}
            else:
                dataset_status[dc["name"]] = {"status": "ok", "path": ds_path}
        else:
            # Check if saved dataset has arrow files
            files = os.listdir(ds_path)
            dataset_status[dc["name"]] = {"status": "ok", "files": len(files), "path": ds_path}
    else:
        dataset_status[dc["name"]] = {"status": "missing", "path": ds_path}

# GPU info
gpu_info = {}
if torch.cuda.is_available():
    gpu_info = {
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024*1024),
    }

setup_result = {
    "task_id": TASK_ID,
    "status": "success" if (dream_result["status"] == "success") else "partial",
    "timestamp": datetime.now().isoformat(),
    "environment": {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        **gpu_info,
    },
    "models": {
        "dream_7b": dream_result,
        "llada_8b": llada_result,
    },
    "datasets": dataset_status,
    "directories": {d: os.path.exists(d) for d in dirs_to_create},
    "pass_criteria": "Model loads successfully, single forward pass completes without error",
    "pass_result": dream_result["status"] == "success",
}

# Also check LLaDA pass
if llada_result["status"] == "success":
    setup_result["pass_result"] = True
    setup_result["status"] = "success"

# Write results
setup_file = f"{LOGS_DIR}/setup_complete.json"
with open(setup_file, "w") as f:
    json.dump(setup_result, f, indent=2)
print(f"\n=== Setup complete. Results written to {setup_file} ===")
print(f"Dream-7B: {dream_result['status']}")
print(f"LLaDA-8B: {llada_result['status']}")
print(f"Datasets: {sum(1 for v in dataset_status.values() if v['status'] == 'ok')}/{len(dataset_status)} OK")
print(f"Overall: {'PASS' if setup_result['pass_result'] else 'FAIL'}")

# Update shared registry
registry_path = f"{SHARED_DIR}/registry.json"
try:
    with open(registry_path) as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {"checkpoints": {}}

if "datasets" not in registry:
    registry["datasets"] = {}

# Add LLaDA to registry if downloaded
if llada_result["status"] == "success":
    registry["checkpoints"]["llada_8b_instruct"] = {
        "type": "checkpoint",
        "name": "LLaDA-8B-Instruct",
        "path": "shared/checkpoints/LLaDA-8B-Instruct",
        "target": llada_dir,
        "source": "huggingface:GSAI-ML/LLaDA-8B-Instruct",
        "project": PROJECT,
    }

# Add Dream to registry
registry["checkpoints"]["dream_7b_instruct"] = {
    "type": "checkpoint",
    "name": "Dream-v0-Instruct-7B",
    "path": "shared/checkpoints/Dream-v0-Instruct-7B",
    "target": f"{MODELS_DIR}/Dream-v0-Instruct-7B",
    "source": "huggingface:Dream-org/Dream-v0-Instruct-7B",
    "project": PROJECT,
}

# Add datasets to registry
for dc_name, ds_info in dataset_status.items():
    if ds_info["status"] == "ok":
        registry["datasets"][dc_name] = {
            "type": "dataset",
            "name": dc_name,
            "path": f"shared/datasets/{dc_name}",
            "project": PROJECT,
        }

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

# Mark task done
mark_done(
    status="success" if setup_result["pass_result"] else "partial",
    summary=f"Dream-7B: {dream_result['status']}, LLaDA-8B: {llada_result['status']}, "
            f"Datasets: {sum(1 for v in dataset_status.values() if v['status'] == 'ok')}/{len(dataset_status)} OK"
)

print("\n=== DONE ===")
