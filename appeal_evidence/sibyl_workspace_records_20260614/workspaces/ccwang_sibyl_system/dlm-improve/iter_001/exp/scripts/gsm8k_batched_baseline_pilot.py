#!/usr/bin/env python3
"""Batched GSM8K pilot runner for reasoning baselines."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from batched_dlm_utils import BatchedDLMInference


TASK_ID = os.environ.get("TASK_ID", "baseline_standard_dnb")
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct")
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))
GEN_LENGTH = int(os.environ.get("GEN_LENGTH", "256"))
USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"
METHOD_SET = os.environ.get("METHOD_SET", "standard_dnb")
SEED = 42


def write_pid() -> None:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")


def report_progress(step: int, total_steps: int, metric: dict[str, Any]) -> None:
    payload = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "loss": None,
        "metric": metric,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mark_done(status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            final_progress = {}
    payload = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def normalize_answer(ans: str) -> str:
    ans = str(ans).strip().replace(",", "")
    if not ans:
        return ""
    try:
        value = float(ans)
    except ValueError:
        return ans
    if value.is_integer():
        return str(int(value))
    return ans


def get_gsm8k_samples(tokenizer, num_samples: int) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split="test")
    total = min(num_samples, len(dataset))
    samples = []
    for idx in range(total):
        question = dataset[idx]["question"]
        answer = dataset[idx]["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        samples.append(
            {
                "idx": idx,
                "prompt_ids": prompt_ids,
                "final_answer": final_answer,
                "question": question[:100],
            }
        )
    return samples


def decode_and_score(tokenizer, sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    gen_tokens = result["input_ids"][result["gen_start"]:result["gen_end"]].cpu().tolist()
    gen_tokens = [token for token in gen_tokens if token != 126336]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    predicted = normalize_answer(extract_gsm8k_answer(generated_text))
    reference = normalize_answer(sample["final_answer"])
    correct = bool(predicted and predicted == reference)
    return {
        "idx": sample["idx"],
        "correct": correct,
        "nfe": result["nfe"],
        "predicted_answer": predicted,
        "reference_answer": reference,
        "generated_text": generated_text[:400],
    }


def method_configs() -> list[tuple[str, str, dict[str, Any]]]:
    if METHOD_SET == "standard_dnb":
        return [
            ("Standard-64", "batched_standard_denoising", {"num_steps": 64}),
            ("DNB-84", "batched_standard_denoising", {"num_steps": 84}),
        ]
    if METHOD_SET == "prophet":
        return [
            ("Prophet-64", "batched_prophet", {"num_steps": 64, "confidence_threshold": 0.95}),
        ]
    raise ValueError(f"Unsupported METHOD_SET={METHOD_SET}")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()
    task_start = time.time()

    engine = BatchedDLMInference(
        model_path=MODEL_PATH,
        device="cuda",
        use_flash_attn=True,
        use_compile=USE_COMPILE,
    )
    samples = get_gsm8k_samples(engine.tokenizer, NUM_SAMPLES)
    prompt_max_len = max(len(sample["prompt_ids"]) for sample in samples)
    safe_batch_size = engine.find_max_batch_size(
        gen_length=GEN_LENGTH,
        prompt_max_len=min(prompt_max_len, 512),
        lo=1,
        hi=128,
    )

    gpu_profile = {
        "task_id": TASK_ID,
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
        "max_batch_size": safe_batch_size,
        "prompt_max_len": prompt_max_len,
        "gen_length": GEN_LENGTH,
        "compile_enabled": USE_COMPILE,
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2), encoding="utf-8")

    prompt_ids_list = [sample["prompt_ids"] for sample in samples]
    all_method_results = []
    method_specs = method_configs()
    for idx, (method_name, fn_name, kwargs) in enumerate(method_specs, start=1):
        method_start = time.time()
        fn = getattr(engine, fn_name)
        outputs = fn(prompt_ids_list, gen_length=GEN_LENGTH, batch_size=safe_batch_size, seed=SEED, **kwargs)
        per_sample = [decode_and_score(engine.tokenizer, sample, output) for sample, output in zip(samples, outputs, strict=True)]
        correct = sum(int(entry["correct"]) for entry in per_sample)
        avg_nfe = sum(entry["nfe"] for entry in per_sample) / max(len(per_sample), 1)
        latency = time.time() - method_start
        total_generated = len(samples) * GEN_LENGTH
        method_payload = {
            "method": method_name,
            "accuracy": round(correct / max(len(per_sample), 1), 4),
            "correct": correct,
            "total": len(per_sample),
            "avg_nfe": round(avg_nfe, 2),
            "latency_sec": round(latency, 3),
            "tokens_per_sec": round(total_generated / max(latency, 1e-6), 2),
            "batch_size": safe_batch_size,
            "attention_backend": getattr(engine.model.config, "_attn_implementation", "unknown"),
            "compile_enabled": USE_COMPILE,
            "per_sample": per_sample[:10],
        }
        all_method_results.append(method_payload)
        report_progress(
            idx,
            len(method_specs),
            {
                "method": method_name,
                "accuracy": method_payload["accuracy"],
                "avg_nfe": method_payload["avg_nfe"],
                "latency_sec": method_payload["latency_sec"],
                "batch_size": safe_batch_size,
            },
        )

    payload = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(samples),
        "safe_batch_size": safe_batch_size,
        "prompt_max_len": prompt_max_len,
        "methods": all_method_results,
        "wall_clock_sec": round(time.time() - task_start, 3),
    }
    (RESULTS_DIR / f"{TASK_ID}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = ", ".join(f"{item['method']}={item['accuracy']:.3f}" for item in all_method_results)
    mark_done("success", f"Completed {TASK_ID}: {summary}; batch={safe_batch_size}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive remote logging
        error_payload = {
            "task_id": TASK_ID,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (RESULTS_DIR / f"{TASK_ID}.json").write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        mark_done("failed", f"{type(exc).__name__}: {exc}")
        raise
