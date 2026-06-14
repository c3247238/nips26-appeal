#!/usr/bin/env python3
"""
setup_env_p2.py — Phase 2: Download remaining datasets + verify models.
OpenWebText subset, Countdown, then model verification.
"""

import json, os, sys, time, traceback, gc
from pathlib import Path
from datetime import datetime

REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT = "ttt-dlm"
PROJECT_DIR = f"{REMOTE_BASE}/projects/{PROJECT}"
SHARED_DIR = f"{REMOTE_BASE}/shared"
MODELS_DIR = f"{REMOTE_BASE}/models"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
LOGS_DIR = f"{PROJECT_DIR}/exp/logs"
TASK_ID = "setup_env"
SEED = 42
DATASETS_DIR = f"{SHARED_DIR}/datasets"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/pilots", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/full", exist_ok=True)

# PID
Path(f"{LOGS_DIR}/{TASK_ID}.pid").write_text(str(os.getpid()))

print("=" * 60)
print("Phase 2: Remaining datasets + model verification")
print("=" * 60)

# === 1. OpenWebText 10K subset ===
owt_path = f"{DATASETS_DIR}/openwebtext_10k"
if os.path.exists(owt_path) and os.listdir(owt_path):
    print(f"[1/5] openwebtext_10k: already cached")
else:
    print(f"[1/5] openwebtext_10k: downloading 10K subset via streaming...")
    try:
        import datasets as hf_datasets
        # Use streaming to avoid downloading entire dataset
        ds_stream = hf_datasets.load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True,
        )
        samples = []
        for i, example in enumerate(ds_stream):
            if i >= 10000:
                break
            samples.append(example)
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i+1}/10000 samples...")

        # Convert to HF dataset and save
        ds = hf_datasets.Dataset.from_list(samples)
        os.makedirs(owt_path, exist_ok=True)
        ds.save_to_disk(owt_path)
        print(f"  Saved {len(ds)} samples to {owt_path}")
        del ds, samples, ds_stream
        gc.collect()
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

# === 2. Countdown dataset ===
countdown_path = f"{DATASETS_DIR}/countdown"
if os.path.exists(countdown_path) and os.listdir(countdown_path):
    print(f"[2/5] countdown: already cached")
else:
    print(f"[2/5] countdown: generating synthetic dataset...")
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
    print(f"  Generated {len(samples)} samples")

# === 3. Verify Dream-7B ===
print(f"[3/5] Verifying Dream-7B loading + forward pass...")
import torch
torch.manual_seed(SEED)

dream_dir = f"{MODELS_DIR}/Dream-v0-Instruct-7B"
dream_result = {"model": "Dream-7B-Instruct", "status": "unknown"}

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(dream_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(dream_dir, trust_remote_code=True)
    print(f"  Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        dream_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Parameters: {n_params:,}")

    test_input = tokenizer("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**test_input)
    logits_shape = list(output.logits.shape) if hasattr(output, 'logits') else "N/A"
    print(f"  Forward pass OK. Logits shape: {logits_shape}")

    dream_result = {
        "model": "Dream-7B-Instruct", "status": "success",
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "vocab_size": tokenizer.vocab_size,
        "parameters": n_params,
        "logits_shape": str(logits_shape),
    }
    del model, output, test_input; torch.cuda.empty_cache(); gc.collect()
except Exception as e:
    dream_result = {"model": "Dream-7B-Instruct", "status": "error", "error": str(e)}
    print(f"  ERROR: {e}")
    traceback.print_exc()
    torch.cuda.empty_cache(); gc.collect()

# === 4. Verify LLaDA-8B ===
print(f"[4/5] Verifying LLaDA-8B loading + forward pass...")
llada_dir = f"{MODELS_DIR}/LLaDA-8B-Instruct"
llada_result = {"model": "LLaDA-8B-Instruct", "status": "unknown"}

if os.path.exists(llada_dir) and any(f.endswith(".safetensors") for f in os.listdir(llada_dir)):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        tokenizer_l = AutoTokenizer.from_pretrained(llada_dir, trust_remote_code=True)
        config_l = AutoConfig.from_pretrained(llada_dir, trust_remote_code=True)
        print(f"  Config: hidden_size={config_l.hidden_size}, layers={config_l.num_hidden_layers}")

        device = "cuda:0"
        model_l = AutoModelForCausalLM.from_pretrained(
            llada_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        n_params_l = sum(p.numel() for p in model_l.parameters())
        print(f"  Loaded. Parameters: {n_params_l:,}")

        test_input = tokenizer_l("Hello, world!", return_tensors="pt").to(device)
        with torch.no_grad():
            output = model_l(**test_input)
        logits_shape = list(output.logits.shape) if hasattr(output, 'logits') else "N/A"
        print(f"  Forward pass OK. Logits shape: {logits_shape}")

        llada_result = {
            "model": "LLaDA-8B-Instruct", "status": "success",
            "hidden_size": config_l.hidden_size,
            "num_layers": config_l.num_hidden_layers,
            "vocab_size": tokenizer_l.vocab_size,
            "parameters": n_params_l,
            "logits_shape": str(logits_shape),
        }
        del model_l, output, test_input; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        llada_result = {"model": "LLaDA-8B-Instruct", "status": "error", "error": str(e)}
        print(f"  ERROR: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache(); gc.collect()
else:
    llada_result = {"model": "LLaDA-8B-Instruct", "status": "not_downloaded"}
    print(f"  Not found at {llada_dir}")

# === 5. Write results ===
print(f"[5/5] Writing setup_complete.json...")

# Dataset status
dataset_status = {}
for name in ["gsm8k", "math500", "humaneval", "mbpp", "arc_challenge", "openwebtext_10k", "countdown"]:
    ds_path = f"{DATASETS_DIR}/{name}"
    if os.path.exists(ds_path) and os.listdir(ds_path):
        dataset_status[name] = {"status": "ok", "path": ds_path}
    else:
        dataset_status[name] = {"status": "missing", "path": ds_path}

gpu_info = {}
if torch.cuda.is_available():
    gpu_info = {
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024*1024),
    }

overall_pass = dream_result["status"] == "success"  # Dream is primary backbone

setup_result = {
    "task_id": TASK_ID,
    "status": "success" if overall_pass else "partial",
    "timestamp": datetime.now().isoformat(),
    "environment": {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        **gpu_info,
    },
    "models": {"dream_7b": dream_result, "llada_8b": llada_result},
    "datasets": dataset_status,
    "pass_criteria": "Model loads successfully, single forward pass completes without error",
    "pass_result": overall_pass,
}

setup_file = f"{LOGS_DIR}/setup_complete.json"
with open(setup_file, "w") as f:
    json.dump(setup_result, f, indent=2)

# Update registry
registry_path = f"{SHARED_DIR}/registry.json"
try:
    with open(registry_path) as f:
        registry = json.load(f)
except:
    registry = {"checkpoints": {}}
if "datasets" not in registry:
    registry["datasets"] = {}

if llada_result["status"] == "success":
    registry["checkpoints"]["llada_8b_instruct"] = {
        "type": "checkpoint", "name": "LLaDA-8B-Instruct",
        "path": "shared/checkpoints/LLaDA-8B-Instruct",
        "target": llada_dir, "source": "huggingface:GSAI-ML/LLaDA-8B-Instruct",
    }
registry["checkpoints"]["dream_7b_instruct"] = {
    "type": "checkpoint", "name": "Dream-v0-Instruct-7B",
    "path": "shared/checkpoints/Dream-v0-Instruct-7B",
    "target": f"{MODELS_DIR}/Dream-v0-Instruct-7B",
    "source": "huggingface:Dream-org/Dream-v0-Instruct-7B",
}
for name, info in dataset_status.items():
    if info["status"] == "ok":
        registry["datasets"][name] = {"type": "dataset", "name": name, "path": f"shared/datasets/{name}"}
with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

# DONE marker
marker = Path(f"{LOGS_DIR}/{TASK_ID}_DONE")
marker.write_text(json.dumps({
    "task_id": TASK_ID, "status": "success" if overall_pass else "partial",
    "summary": f"Dream-7B: {dream_result['status']}, LLaDA-8B: {llada_result['status']}, "
               f"Datasets: {sum(1 for v in dataset_status.values() if v['status']=='ok')}/{len(dataset_status)}",
    "timestamp": datetime.now().isoformat(),
}))

# Clean up PID
pid_f = Path(f"{LOGS_DIR}/{TASK_ID}.pid")
if pid_f.exists(): pid_f.unlink()

print(f"\n{'='*60}")
print(f"Dream-7B: {dream_result['status']}")
print(f"LLaDA-8B: {llada_result['status']}")
print(f"Datasets: {sum(1 for v in dataset_status.values() if v['status']=='ok')}/{len(dataset_status)} OK")
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
print(f"{'='*60}")
