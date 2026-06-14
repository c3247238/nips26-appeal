#!/usr/bin/env python3
"""
AR Baseline Honest Comparison: Qwen2.5-7B-Instruct with vLLM.

Re-runs the AR baseline comparison using vLLM (v0.19.0) for proper
throughput measurement. Includes speculative decoding with Qwen2.5-0.5B.

Configurations:
  1. Qwen2.5-7B-Instruct greedy (batch=1)  [vLLM]
  2. Qwen2.5-7B-Instruct greedy (batch=8)  [vLLM]
  3. Qwen2.5-7B-Instruct + Qwen2.5-0.5B speculative (batch=1)  [vLLM]
  4. Qwen2.5-7B-Instruct + Qwen2.5-0.5B speculative (batch=8)  [vLLM]

Benchmarks: 200 GSM8K + 100 MATH500, seed=42

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n sibyl_dlm-acceleration python ar_baseline_vllm.py
"""

import os
import sys
import re
import gc
import json
import time
import random
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import torch
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
TASK_ID = "ar_baseline_comparison"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "ar_comparison"
SHARED = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")

MODEL_PATH = str(SHARED / "checkpoints" / "qwen2.5-7b-instruct")
DRAFT_MODEL_PATH = str(SHARED / "checkpoints" / "qwen2.5-0.5b")

GSM8K_PATH = str(SHARED / "datasets" / "gsm8k" / "test.json")
MATH500_PATH = str(SHARED / "datasets" / "math500" / "test.json")

# LLaDA baseline reference from iter_001 (3 seeds mean)
LLADA_BASELINE = {
    "gsm8k": {"accuracy": 0.7122, "tps": 31.01},
    "math500": {"accuracy": 0.1107, "tps": 79.22},
}

SAMPLES_GSM8K = 200   # Full mode as specified in task plan
SAMPLES_MATH500 = 100
SEED = 42
N_WARMUP = 5
MAX_NEW_TOKENS = 512


# ── PID / Progress / Done Markers ──────────────────────────────────────────────

def write_pid(task_id: str, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{task_id}.pid").write_text(str(os.getpid()))


def report_progress(task_id: str, results_dir: Path, step: int, total: int,
                    metric: dict = None):
    progress = results_dir / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "loss": None,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(task_id: str, results_dir: Path, status: str = "success",
              summary: str = ""):
    pid_file = results_dir / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = results_dir / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = results_dir / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Dataset Loaders ────────────────────────────────────────────────────────────

def load_gsm8k(n_samples: int, seed: int = 42) -> List[Dict]:
    with open(GSM8K_PATH) as f:
        data = json.load(f)
    if n_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n_samples)
    return data


def load_math500(n_samples: int, seed: int = 42) -> List[Dict]:
    with open(MATH500_PATH) as f:
        data = json.load(f)
    if n_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n_samples)
    return data


def build_gsm8k_chat(question: str) -> List[Dict]:
    return [
        {"role": "system", "content": (
            "You are a helpful math tutor. Solve the problem step by step. "
            "End your answer with 'The answer is X.' where X is the final "
            "numerical answer."
        )},
        {"role": "user", "content": question},
    ]


def build_math500_chat(problem: str) -> List[Dict]:
    return [
        {"role": "system", "content": (
            "You are a helpful math tutor. Solve the problem step by step. "
            "Put your final answer in \\boxed{}."
        )},
        {"role": "user", "content": problem},
    ]


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def gsm8k_exact_match(pred: str, gold: str) -> bool:
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


def extract_math500_answer(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    match = re.search(r"[Tt]he answer is\s+(.+?)(?:\.|$)", text)
    if match:
        return match.group(1).strip()
    return None


def math500_exact_match(pred: str, gold: str) -> bool:
    p = extract_math500_answer(pred)
    g = gold.strip()
    if p is None:
        return False
    p = p.replace(" ", "").replace("\\,", "").strip()
    g = g.replace(" ", "").replace("\\,", "").strip()
    g_match = re.match(r"\\boxed\{(.+)\}", g)
    if g_match:
        g = g_match.group(1).strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ── QAS Computation ────────────────────────────────────────────────────────────

def compute_qas(speedup: float, accuracy_retention: float) -> float:
    return speedup * accuracy_retention


def compute_speedup(tps: float, baseline_tps: float) -> float:
    if baseline_tps == 0:
        return 0
    return tps / baseline_tps


# ── GPU Profiling ──────────────────────────────────────────────────────────────

def get_gpu_profile() -> Dict:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": total_mem // 1024**2,
            "vram_used_mb": torch.cuda.memory_allocated(0) // 1024**2,
            "vram_reserved_mb": torch.cuda.memory_reserved(0) // 1024**2,
        }
    return {}


# ── vLLM Evaluation ───────────────────────────────────────────────────────────

def create_vllm_engine(model_path: str, speculative: bool = False,
                       draft_model_path: str = None):
    """Create a vLLM LLM engine."""
    from vllm import LLM

    kwargs = {
        "model": model_path,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "seed": SEED,
        "enforce_eager": False,
    }

    if speculative and draft_model_path:
        print(f"[vLLM] Creating engine with speculative decoding:")
        print(f"  Target: {model_path}")
        print(f"  Draft:  {draft_model_path}")
        print(f"  Num speculative tokens: 5")
        # Pass speculative config through kwargs
        kwargs["speculative_config"] = {
            "model": draft_model_path,
            "num_speculative_tokens": 5,
            "method": "draft_model",
        }
    else:
        print(f"[vLLM] Creating engine (greedy, no speculation):")
        print(f"  Model: {model_path}")

    engine = LLM(**kwargs)
    print("[vLLM] Engine created successfully")
    return engine


def evaluate_vllm(engine, dataset: List[Dict], dataset_name: str,
                  batch_size: int, tokenizer) -> Dict:
    """Evaluate vLLM engine on a dataset.

    For batch_size=1: process one prompt at a time for per-sample TPS.
    For batch_size>1: batch prompts for throughput measurement.
    """
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    total = len(dataset)
    correct = 0
    tps_list = []
    all_results = []
    total_tokens = 0
    total_gen_time = 0.0

    print(f"\n[{dataset_name.upper()}] Evaluating {total} samples, batch_size={batch_size}")

    # Build all prompts
    all_prompts = []
    for item in dataset:
        if dataset_name == "gsm8k":
            messages = build_gsm8k_chat(item["question"])
        else:
            messages = build_math500_chat(item["problem"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_data = dataset[batch_start:batch_end]
        actual_bs = len(batch_prompts)

        try:
            t0 = time.time()
            outputs = engine.generate(batch_prompts, sampling)
            elapsed = time.time() - t0

            batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            batch_tps = batch_tokens / elapsed if elapsed > 0 else 0

            for i, (output, item) in enumerate(zip(outputs, batch_data)):
                idx = batch_start + i
                text = output.outputs[0].text
                n_tokens = len(output.outputs[0].token_ids)

                if dataset_name == "gsm8k":
                    is_correct = gsm8k_exact_match(text, item["answer"])
                    gold = extract_gsm8k_answer(item["answer"])
                    pred = extract_gsm8k_answer(text)
                    key_field = ("question", item["question"][:100])
                else:
                    is_correct = math500_exact_match(text, item["answer"])
                    gold = item["answer"]
                    pred = extract_math500_answer(text)
                    key_field = ("problem", item["problem"][:100])

                if is_correct:
                    correct += 1
                total_tokens += n_tokens
                total_gen_time += elapsed / actual_bs

                # Per-sample TPS: for batch=1 it's exact; for batch>1 it's
                # the overall batch throughput (shared among samples)
                sample_tps = batch_tps
                if idx >= N_WARMUP:
                    tps_list.append(sample_tps)

                result = {
                    "id": idx,
                    key_field[0]: key_field[1],
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "correct": is_correct,
                    "tps": round(sample_tps, 2),
                    "n_tokens": n_tokens,
                    "prediction_snippet": text[:200],
                }
                all_results.append(result)

        except Exception as e:
            print(f"  [ERROR] Batch {batch_start}-{batch_end}: {e}")
            traceback.print_exc()
            for i, item in enumerate(batch_data):
                key = "question" if dataset_name == "gsm8k" else "problem"
                all_results.append({
                    "id": batch_start + i,
                    key: (item.get("question") or item.get("problem", ""))[:100],
                    "correct": False,
                    "error": str(e),
                })

        if batch_end % 20 == 0 or batch_end == total:
            acc = correct / batch_end if batch_end > 0 else 0
            avg_tps = np.mean(tps_list) if tps_list else 0
            print(f"  [{batch_end}/{total}] acc={acc:.3f}, avg_tps={avg_tps:.1f}")
            report_progress(TASK_ID, RESULTS_DIR,
                            step=batch_end, total=total * 2,
                            metric={f"{dataset_name}_acc": acc,
                                    "tps": float(avg_tps)})

    exact_match = correct / total if total > 0 else 0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0
    tps_std = float(np.std(tps_list)) if len(tps_list) > 1 else 0

    return {
        "dataset": dataset_name,
        "n_samples": total,
        "correct": correct,
        "exact_match": exact_match,
        "avg_tps": avg_tps,
        "tps_std": tps_std,
        "total_tokens": total_tokens,
        "total_gen_time_s": round(total_gen_time, 2),
        "batch_size": batch_size,
        "samples": all_results[:10],  # Save first 10 as examples
    }


def cleanup_vllm():
    """Clean up vLLM engine resources."""
    gc.collect()
    torch.cuda.empty_cache()
    # vLLM 0.19.0 may use ray internally
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass
    time.sleep(3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    print("=" * 70)
    print("AR Baseline Honest Comparison (Qwen2.5-7B-Instruct + vLLM)")
    print(f"Task ID: {TASK_ID}")
    print(f"Start: {start_time.isoformat()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"vLLM version: ", end="")
    import vllm
    print(vllm.__version__)
    print("=" * 70)

    write_pid(TASK_ID, RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load datasets
    print("\n[1/7] Loading datasets...")
    gsm8k_data = load_gsm8k(n_samples=SAMPLES_GSM8K, seed=SEED)
    math500_data = load_math500(n_samples=SAMPLES_MATH500, seed=SEED)
    print(f"  GSM8K: {len(gsm8k_data)} samples")
    print(f"  MATH500: {len(math500_data)} samples")

    all_configs = []
    gpu_profiles = {}

    # ── Config 1: Greedy batch=1 ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("[2/7] Config 1: Qwen2.5-7B greedy, batch=1 (vLLM)")
    print("=" * 50)

    engine = create_vllm_engine(MODEL_PATH, speculative=False)
    gpu_profiles["after_greedy_load"] = get_gpu_profile()

    gsm8k_b1 = evaluate_vllm(engine, gsm8k_data, "gsm8k", batch_size=1,
                              tokenizer=tokenizer)
    math500_b1 = evaluate_vllm(engine, math500_data, "math500", batch_size=1,
                               tokenizer=tokenizer)

    config1 = {
        "config_name": "qwen7b_greedy_b1",
        "model": "Qwen2.5-7B-Instruct",
        "speculative": False,
        "batch_size": 1,
        "engine": "vllm",
        "gsm8k": gsm8k_b1,
        "math500": math500_b1,
    }
    all_configs.append(config1)
    print(f"\n  Results: GSM8K={gsm8k_b1['exact_match']:.3f} @ "
          f"{gsm8k_b1['avg_tps']:.1f} TPS")
    print(f"           MATH500={math500_b1['exact_match']:.3f} @ "
          f"{math500_b1['avg_tps']:.1f} TPS")

    # ── Config 2: Greedy batch=8 ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("[3/7] Config 2: Qwen2.5-7B greedy, batch=8 (vLLM)")
    print("=" * 50)

    gsm8k_b8 = evaluate_vllm(engine, gsm8k_data, "gsm8k", batch_size=8,
                              tokenizer=tokenizer)
    math500_b8 = evaluate_vllm(engine, math500_data, "math500", batch_size=8,
                               tokenizer=tokenizer)

    config2 = {
        "config_name": "qwen7b_greedy_b8",
        "model": "Qwen2.5-7B-Instruct",
        "speculative": False,
        "batch_size": 8,
        "engine": "vllm",
        "gsm8k": gsm8k_b8,
        "math500": math500_b8,
    }
    all_configs.append(config2)
    print(f"\n  Results: GSM8K={gsm8k_b8['exact_match']:.3f} @ "
          f"{gsm8k_b8['avg_tps']:.1f} TPS")
    print(f"           MATH500={math500_b8['exact_match']:.3f} @ "
          f"{math500_b8['avg_tps']:.1f} TPS")

    # Clean up greedy engine
    del engine
    cleanup_vllm()

    # ── Config 3: Speculative batch=1 ────────────────────────────────────
    print("\n" + "=" * 50)
    print("[4/7] Config 3: Qwen2.5-7B + 0.5B speculative, batch=1 (vLLM)")
    print("=" * 50)

    try:
        engine_spec = create_vllm_engine(
            MODEL_PATH, speculative=True, draft_model_path=DRAFT_MODEL_PATH
        )
        spec_engine_available = True
        gpu_profiles["after_speculative_load"] = get_gpu_profile()
    except Exception as e:
        print(f"  [WARNING] Speculative engine creation failed: {e}")
        print("  Falling back to HuggingFace assisted generation for spec configs")
        spec_engine_available = False

    if spec_engine_available:
        gsm8k_spec_b1 = evaluate_vllm(engine_spec, gsm8k_data, "gsm8k",
                                       batch_size=1, tokenizer=tokenizer)
        math500_spec_b1 = evaluate_vllm(engine_spec, math500_data, "math500",
                                         batch_size=1, tokenizer=tokenizer)
        spec_engine_type = "vllm"
    else:
        # Fallback: HF assisted generation for batch=1 only
        gsm8k_spec_b1, math500_spec_b1, spec_engine_type = \
            _hf_speculative_fallback(gsm8k_data, math500_data, batch_size=1,
                                     tokenizer=tokenizer)

    config3 = {
        "config_name": "qwen7b_speculative_b1",
        "model": "Qwen2.5-7B-Instruct",
        "draft_model": "Qwen2.5-0.5B",
        "speculative": True,
        "batch_size": 1,
        "engine": spec_engine_type,
        "gsm8k": gsm8k_spec_b1,
        "math500": math500_spec_b1,
    }
    all_configs.append(config3)
    print(f"\n  Results: GSM8K={gsm8k_spec_b1['exact_match']:.3f} @ "
          f"{gsm8k_spec_b1['avg_tps']:.1f} TPS")
    print(f"           MATH500={math500_spec_b1['exact_match']:.3f} @ "
          f"{math500_spec_b1['avg_tps']:.1f} TPS")

    # ── Config 4: Speculative batch=8 ────────────────────────────────────
    print("\n" + "=" * 50)
    print("[5/7] Config 4: Qwen2.5-7B + 0.5B speculative, batch=8 (vLLM)")
    print("=" * 50)

    if spec_engine_available:
        gsm8k_spec_b8 = evaluate_vllm(engine_spec, gsm8k_data, "gsm8k",
                                       batch_size=8, tokenizer=tokenizer)
        math500_spec_b8 = evaluate_vllm(engine_spec, math500_data, "math500",
                                         batch_size=8, tokenizer=tokenizer)
        spec_b8_engine = "vllm"
    else:
        # HF speculative does not support batch>1
        gsm8k_spec_b8 = _empty_result("gsm8k", SAMPLES_GSM8K, 8,
                                       "Speculative decoding batch>1 not supported without vLLM")
        math500_spec_b8 = _empty_result("math500", SAMPLES_MATH500, 8,
                                         "Speculative decoding batch>1 not supported without vLLM")
        spec_b8_engine = "none"

    config4 = {
        "config_name": "qwen7b_speculative_b8",
        "model": "Qwen2.5-7B-Instruct",
        "draft_model": "Qwen2.5-0.5B",
        "speculative": True,
        "batch_size": 8,
        "engine": spec_b8_engine if not spec_engine_available else "vllm",
        "gsm8k": gsm8k_spec_b8,
        "math500": math500_spec_b8,
    }
    all_configs.append(config4)
    print(f"\n  Results: GSM8K={gsm8k_spec_b8['exact_match']:.3f} @ "
          f"{gsm8k_spec_b8['avg_tps']:.1f} TPS")
    print(f"           MATH500={math500_spec_b8['exact_match']:.3f} @ "
          f"{math500_spec_b8['avg_tps']:.1f} TPS")

    # Clean up
    if spec_engine_available:
        del engine_spec
        cleanup_vllm()

    # ── Compute Comparison Metrics ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[6/7] COMPARISON WITH LLaDA-8B-Instruct BASELINE")
    print("=" * 70)

    comparison_table = []
    for cfg in all_configs:
        gsm8k_tps = cfg["gsm8k"]["avg_tps"]
        math_tps = cfg["math500"]["avg_tps"]
        gsm8k_acc = cfg["gsm8k"]["exact_match"]
        math_acc = cfg["math500"]["exact_match"]

        gsm8k_speedup = compute_speedup(gsm8k_tps, LLADA_BASELINE["gsm8k"]["tps"])
        gsm8k_acc_ret = gsm8k_acc / LLADA_BASELINE["gsm8k"]["accuracy"] \
            if LLADA_BASELINE["gsm8k"]["accuracy"] > 0 else 0
        gsm8k_qas = compute_qas(gsm8k_speedup, gsm8k_acc_ret)

        math_speedup = compute_speedup(math_tps, LLADA_BASELINE["math500"]["tps"])
        math_acc_ret = math_acc / LLADA_BASELINE["math500"]["accuracy"] \
            if LLADA_BASELINE["math500"]["accuracy"] > 0 else 0
        math_qas = compute_qas(math_speedup, math_acc_ret)

        # Combined metric: 0.7*GSM8K + 0.3*MATH500
        combined_acc_ret = 0.7 * gsm8k_acc_ret + 0.3 * math_acc_ret
        combined_speedup = 0.7 * gsm8k_speedup + 0.3 * math_speedup
        combined_qas = compute_qas(combined_speedup, combined_acc_ret)

        row = {
            "config_name": cfg["config_name"],
            "system": f"{'AR+SpecDec' if cfg.get('speculative') else 'AR'} (Qwen7B)",
            "batch_size": cfg["batch_size"],
            "engine": cfg["engine"],
            "gsm8k_acc": gsm8k_acc,
            "gsm8k_tps": round(gsm8k_tps, 1),
            "gsm8k_speedup_vs_llada": round(gsm8k_speedup, 2),
            "gsm8k_acc_retention": round(gsm8k_acc_ret, 3),
            "gsm8k_qas": round(gsm8k_qas, 3),
            "math500_acc": math_acc,
            "math500_tps": round(math_tps, 1),
            "math500_speedup_vs_llada": round(math_speedup, 2),
            "math500_acc_retention": round(math_acc_ret, 3),
            "math500_qas": round(math_qas, 3),
            "combined_acc_retention": round(combined_acc_ret, 3),
            "combined_speedup": round(combined_speedup, 2),
            "combined_qas": round(combined_qas, 3),
        }
        comparison_table.append(row)

        print(f"\n  {row['config_name']} ({row['engine']}):")
        print(f"    GSM8K:   acc={row['gsm8k_acc']:.3f}, TPS={row['gsm8k_tps']:.1f}, "
              f"speedup={row['gsm8k_speedup_vs_llada']:.2f}x, "
              f"AccRet={row['gsm8k_acc_retention']:.3f}, QAS={row['gsm8k_qas']:.3f}")
        print(f"    MATH500: acc={row['math500_acc']:.3f}, TPS={row['math500_tps']:.1f}, "
              f"speedup={row['math500_speedup_vs_llada']:.2f}x, "
              f"AccRet={row['math500_acc_retention']:.3f}, QAS={row['math500_qas']:.3f}")
        print(f"    Combined: AccRet={row['combined_acc_retention']:.3f}, "
              f"Speedup={row['combined_speedup']:.2f}x, QAS={row['combined_qas']:.3f}")

    # LLaDA reference row
    llada_ref = {
        "config_name": "llada8b_baseline",
        "system": "DLM (LLaDA-8B)",
        "batch_size": 8,
        "engine": "custom",
        "gsm8k_acc": LLADA_BASELINE["gsm8k"]["accuracy"],
        "gsm8k_tps": LLADA_BASELINE["gsm8k"]["tps"],
        "gsm8k_speedup_vs_llada": 1.0,
        "gsm8k_acc_retention": 1.0,
        "gsm8k_qas": 1.0,
        "math500_acc": LLADA_BASELINE["math500"]["accuracy"],
        "math500_tps": LLADA_BASELINE["math500"]["tps"],
        "math500_speedup_vs_llada": 1.0,
        "math500_acc_retention": 1.0,
        "math500_qas": 1.0,
        "combined_acc_retention": 1.0,
        "combined_speedup": 1.0,
        "combined_qas": 1.0,
        "note": "Reference from iter_001 full baseline (3 seeds mean)",
    }

    # ── Save Results ─────────────────────────────────────────────────────
    end_time = datetime.now()
    duration_min = (end_time - start_time).total_seconds() / 60

    print(f"\n[7/7] Saving results...")

    final_results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "Qwen2.5-7B-Instruct",
        "draft_model": "Qwen2.5-0.5B",
        "engine_used": "vllm",
        "vllm_version": vllm.__version__,
        "speculative_engine_used": spec_engine_type if spec_engine_available else "hf_fallback",
        "benchmarks": {
            "gsm8k_n": SAMPLES_GSM8K,
            "math500_n": SAMPLES_MATH500,
        },
        "llada_baseline_reference": LLADA_BASELINE,
        "configs": all_configs,
        "comparison_table": comparison_table,
        "llada_reference_row": llada_ref,
        "gpu_profile": gpu_profiles,
        "timing": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_min": round(duration_min, 1),
        },
        "timestamp": end_time.isoformat(),
    }

    output_path = RESULTS_DIR / "ar_baseline.json"
    output_path.write_text(json.dumps(final_results, indent=2, default=str))
    print(f"  [SAVED] {output_path}")

    # Save summary markdown
    md_lines = [
        "# AR Baseline Comparison - Full Results (vLLM)",
        "",
        f"**Task**: {TASK_ID}",
        f"**Date**: {end_time.strftime('%Y-%m-%d %H:%M')}",
        f"**Duration**: {duration_min:.1f} min",
        f"**Engine**: vLLM {vllm.__version__}",
        f"**Benchmarks**: {SAMPLES_GSM8K} GSM8K + {SAMPLES_MATH500} MATH500",
        "",
        "## Results Table",
        "",
        "| System | Batch | Engine | GSM8K Acc | GSM8K TPS | Speedup | "
        "MATH500 Acc | MATH500 TPS | Combined QAS |",
        "|--------|-------|--------|-----------|-----------|---------|"
        "-------------|-------------|--------------|",
    ]
    for row in comparison_table:
        md_lines.append(
            f"| {row['system']} | {row['batch_size']} | {row['engine']} | "
            f"{row['gsm8k_acc']:.3f} | {row['gsm8k_tps']:.1f} | "
            f"{row['gsm8k_speedup_vs_llada']:.2f}x | "
            f"{row['math500_acc']:.3f} | {row['math500_tps']:.1f} | "
            f"{row['combined_qas']:.3f} |"
        )
    md_lines.append(
        f"| DLM (LLaDA-8B) | 8 | custom | "
        f"{LLADA_BASELINE['gsm8k']['accuracy']:.3f} | "
        f"{LLADA_BASELINE['gsm8k']['tps']:.1f} | 1.00x | "
        f"{LLADA_BASELINE['math500']['accuracy']:.3f} | "
        f"{LLADA_BASELINE['math500']['tps']:.1f} | 1.000 |"
    )
    md_lines.extend([
        "",
        "## Key Findings",
        "",
        f"- AR greedy (b=1): GSM8K {comparison_table[0]['gsm8k_acc']:.3f} @ "
        f"{comparison_table[0]['gsm8k_tps']:.1f} TPS "
        f"({comparison_table[0]['gsm8k_speedup_vs_llada']:.2f}x vs LLaDA)",
        f"- AR greedy (b=8): GSM8K {comparison_table[1]['gsm8k_acc']:.3f} @ "
        f"{comparison_table[1]['gsm8k_tps']:.1f} TPS "
        f"({comparison_table[1]['gsm8k_speedup_vs_llada']:.2f}x vs LLaDA)",
        f"- AR+SpecDec (b=1): GSM8K {comparison_table[2]['gsm8k_acc']:.3f} @ "
        f"{comparison_table[2]['gsm8k_tps']:.1f} TPS "
        f"({comparison_table[2]['gsm8k_speedup_vs_llada']:.2f}x vs LLaDA)",
        f"- AR+SpecDec (b=8): GSM8K {comparison_table[3]['gsm8k_acc']:.3f} @ "
        f"{comparison_table[3]['gsm8k_tps']:.1f} TPS "
        f"({comparison_table[3]['gsm8k_speedup_vs_llada']:.2f}x vs LLaDA)",
        "",
        "## Notes",
        "",
        "- LLaDA baseline is batch=8 (MDM naturally processes all tokens in parallel)",
        "- AR greedy at batch=1 is the fair comparison for interactive use",
        "- Speculative decoding speedup over greedy: "
        f"{comparison_table[2]['gsm8k_tps'] / max(comparison_table[0]['gsm8k_tps'], 0.01):.2f}x (b=1)",
        "- Qwen2.5-7B has higher accuracy than LLaDA-8B (expected for a stronger AR model)",
        "- QAS comparison normalizes for accuracy differences (AccRet = AR_acc / LLaDA_acc)",
    ])

    md_path = RESULTS_DIR / "ar_baseline_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"  [SAVED] {md_path}")

    # ── Update gpu_progress.json ─────────────────────────────────────────
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gpu_progress = json.loads(gpu_progress_path.read_text())
    except Exception:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
    gpu_progress.get("running", {}).pop(TASK_ID, None)
    gpu_progress.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(duration_min),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_snapshot": {
            "model": "Qwen2.5-7B-Instruct",
            "draft_model": "Qwen2.5-0.5B",
            "task": TASK_ID,
            "engine": "vllm",
            "vllm_version": vllm.__version__,
            "benchmarks": ["gsm8k", "math500"],
            "samples": {"gsm8k": SAMPLES_GSM8K, "math500": SAMPLES_MATH500},
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))
    print(f"  [UPDATED] {gpu_progress_path}")

    # Mark done
    summary_str = (
        f"AR baseline (vLLM {vllm.__version__}): "
        f"greedy_b1={comparison_table[0]['gsm8k_acc']:.3f}@{comparison_table[0]['gsm8k_tps']:.1f}TPS, "
        f"greedy_b8={comparison_table[1]['gsm8k_acc']:.3f}@{comparison_table[1]['gsm8k_tps']:.1f}TPS, "
        f"spec_b1={comparison_table[2]['gsm8k_acc']:.3f}@{comparison_table[2]['gsm8k_tps']:.1f}TPS, "
        f"spec_b8={comparison_table[3]['gsm8k_acc']:.3f}@{comparison_table[3]['gsm8k_tps']:.1f}TPS. "
        f"Duration={duration_min:.1f}min"
    )
    mark_done(TASK_ID, RESULTS_DIR, status="success", summary=summary_str)

    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {duration_min:.1f} minutes")
    print(f"{'=' * 70}")

    return final_results


def _empty_result(dataset: str, n_samples: int, batch_size: int,
                  error_msg: str) -> Dict:
    """Create an empty result dict for failed configs."""
    return {
        "dataset": dataset,
        "n_samples": n_samples,
        "correct": 0,
        "exact_match": 0.0,
        "avg_tps": 0,
        "tps_std": 0,
        "total_tokens": 0,
        "total_gen_time_s": 0.0,
        "batch_size": batch_size,
        "samples": [{"error": error_msg}],
    }


def _hf_speculative_fallback(gsm8k_data, math500_data, batch_size, tokenizer):
    """Fallback to HuggingFace assisted generation for speculative decoding."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[HF Fallback] Loading model + draft for assisted generation...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True
    )
    draft_model.eval()
    draft_tokenizer = AutoTokenizer.from_pretrained(
        DRAFT_MODEL_PATH, trust_remote_code=True
    )

    def gen_single(prompt_text):
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=3584).to("cuda")
        input_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                assistant_model=draft_model,
                tokenizer=tokenizer,
                assistant_tokenizer=draft_tokenizer,
            )
        torch.cuda.synchronize()
        latency = time.time() - t0
        new_tokens = out[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, len(new_tokens), len(new_tokens) / latency if latency > 0 else 0, latency

    gsm8k_results = {"dataset": "gsm8k", "n_samples": len(gsm8k_data),
                     "correct": 0, "samples": []}
    tps_list = []
    for i, item in enumerate(gsm8k_data):
        msgs = build_gsm8k_chat(item["question"])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        text, n_tok, tps, lat = gen_single(prompt)
        is_correct = gsm8k_exact_match(text, item["answer"])
        if is_correct:
            gsm8k_results["correct"] += 1
        if i >= N_WARMUP:
            tps_list.append(tps)
        if i < 10:
            gsm8k_results["samples"].append({
                "id": i, "question": item["question"][:100],
                "gold_answer": extract_gsm8k_answer(item["answer"]),
                "predicted_answer": extract_gsm8k_answer(text),
                "correct": is_correct, "tps": round(tps, 2),
                "n_tokens": n_tok,
            })
        if (i + 1) % 20 == 0:
            print(f"  GSM8K [{i+1}/{len(gsm8k_data)}] "
                  f"acc={gsm8k_results['correct']/(i+1):.3f}")

    gsm8k_results["exact_match"] = gsm8k_results["correct"] / len(gsm8k_data)
    gsm8k_results["avg_tps"] = float(np.mean(tps_list)) if tps_list else 0
    gsm8k_results["tps_std"] = float(np.std(tps_list)) if len(tps_list) > 1 else 0
    gsm8k_results["batch_size"] = 1
    gsm8k_results["total_tokens"] = 0
    gsm8k_results["total_gen_time_s"] = 0.0

    math_results = {"dataset": "math500", "n_samples": len(math500_data),
                    "correct": 0, "samples": []}
    tps_list2 = []
    for i, item in enumerate(math500_data):
        msgs = build_math500_chat(item["problem"])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        text, n_tok, tps, lat = gen_single(prompt)
        is_correct = math500_exact_match(text, item["answer"])
        if is_correct:
            math_results["correct"] += 1
        if i >= N_WARMUP:
            tps_list2.append(tps)
        if i < 10:
            math_results["samples"].append({
                "id": i, "problem": item["problem"][:100],
                "gold_answer": item["answer"],
                "predicted_answer": extract_math500_answer(text),
                "correct": is_correct, "tps": round(tps, 2),
                "n_tokens": n_tok,
            })
        if (i + 1) % 20 == 0:
            print(f"  MATH500 [{i+1}/{len(math500_data)}] "
                  f"acc={math_results['correct']/(i+1):.3f}")

    math_results["exact_match"] = math_results["correct"] / len(math500_data)
    math_results["avg_tps"] = float(np.mean(tps_list2)) if tps_list2 else 0
    math_results["tps_std"] = float(np.std(tps_list2)) if len(tps_list2) > 1 else 0
    math_results["batch_size"] = 1
    math_results["total_tokens"] = 0
    math_results["total_gen_time_s"] = 0.0

    del model, draft_model
    gc.collect()
    torch.cuda.empty_cache()

    return gsm8k_results, math_results, "hf"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        mark_done(TASK_ID, RESULTS_DIR, status="failed", summary=str(e))
        gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
        try:
            gpu_progress = json.loads(gpu_progress_path.read_text())
            if TASK_ID not in gpu_progress.get("failed", []):
                gpu_progress.setdefault("failed", []).append(TASK_ID)
            gpu_progress.get("running", {}).pop(TASK_ID, None)
            gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))
        except Exception:
            pass
        sys.exit(1)
