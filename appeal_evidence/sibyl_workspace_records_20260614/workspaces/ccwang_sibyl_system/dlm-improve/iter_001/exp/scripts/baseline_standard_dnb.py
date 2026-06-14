#!/usr/bin/env python3
"""Pilot task: baseline_standard_dnb.

Runs the reasoning baselines required for the TIGER pilot:
- Standard-64
- DNB-84

Uses batched inference to avoid the old batch_size=1 failure mode.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from batched_dlm_utils import BatchedDLMInference
from full_llada_gsm8k import check_gsm8k_correct, get_gsm8k_samples

TASK_ID = "baseline_standard_dnb"
PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
RESULTS_PATH = RESULTS_DIR / f"{TASK_ID}.json"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
SEED = 42
NUM_SAMPLES = 100
GEN_LENGTH = 256


def report_progress(epoch: int, total_epochs: int, metric: dict | None = None) -> None:
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": 0,
                "total_steps": 0,
                "loss": None,
                "metric": metric or {},
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def mark_done(status: str, summary: str) -> None:
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress: dict = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            final_progress = {}
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": status,
                "summary": summary,
                "final_progress": final_progress,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def decode_generation(tokenizer, result: dict) -> str:
    gen_ids = result["input_ids"][result["gen_start"]:result["gen_end"]].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_method(engine: BatchedDLMInference, tokenizer, samples: list[dict], steps: int, batch_size: int) -> dict:
    prompt_ids = [sample["prompt_ids"] for sample in samples]
    started = time.time()
    outputs = engine.batched_standard_denoising(
        prompt_ids_list=prompt_ids,
        gen_length=GEN_LENGTH,
        num_steps=steps,
        batch_size=batch_size,
        seed=SEED,
    )
    elapsed = time.time() - started

    correct = 0
    total_nfe = 0
    qualitative = []
    for idx, (sample, result) in enumerate(zip(samples, outputs)):
        generated_text = decode_generation(tokenizer, result)
        is_correct = check_gsm8k_correct(generated_text, sample["final_answer"])
        correct += int(is_correct)
        total_nfe += int(result["nfe"])
        if idx < 5:
            qualitative.append(
                {
                    "idx": sample["idx"],
                    "question": sample["question"][:120],
                    "prediction": generated_text[:240],
                    "gold": sample["final_answer"],
                    "correct": bool(is_correct),
                }
            )

    total_generated_tokens = len(samples) * GEN_LENGTH
    return {
        "sample_count": len(samples),
        "accuracy": round(correct / max(1, len(samples)), 4),
        "correct_count": correct,
        "actual_nfe_avg": round(total_nfe / max(1, len(samples)), 2),
        "actual_nfe_total": total_nfe,
        "latency_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_generated_tokens / max(elapsed, 1e-6), 2),
        "batch_size": batch_size,
        "qualitative_examples": qualitative,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        samples = get_gsm8k_samples(tokenizer, num_samples=NUM_SAMPLES)

        report_progress(0, 3, {"phase": "load_engine"})
        engine = BatchedDLMInference(
            model_path=MODEL_PATH,
            device="cuda",
            use_flash_attn=True,
            use_compile=True,
        )

        prompt_max_len = min(512, max(len(sample["prompt_ids"]) for sample in samples))
        safe_batch = engine.find_max_batch_size(
            gen_length=GEN_LENGTH,
            prompt_max_len=prompt_max_len,
            lo=1,
            hi=128,
            safety_margin=0.9,
        )

        result = {
            "task_id": TASK_ID,
            "timestamp": datetime.now().isoformat(),
            "status": "running",
            "model": "LLaDA-8B-Instruct",
            "sample_count": NUM_SAMPLES,
            "gen_length": GEN_LENGTH,
            "attention_backend": getattr(getattr(engine.model, "config", None), "_attn_implementation", "eager"),
            "compile_enabled": True,
            "safe_batch_size": safe_batch,
            "methods": {},
        }

        report_progress(1, 3, {"phase": "standard_64", "safe_batch_size": safe_batch})
        result["methods"]["standard_64"] = evaluate_method(engine, tokenizer, samples, steps=64, batch_size=safe_batch)

        gc.collect()
        report_progress(2, 3, {"phase": "dnb_84", "safe_batch_size": safe_batch})
        result["methods"]["dnb_84"] = evaluate_method(engine, tokenizer, samples, steps=84, batch_size=safe_batch)

        result["status"] = "success"
        result["pass"] = result["methods"]["standard_64"]["accuracy"] >= 0.30
        result["summary"] = (
            f"standard_64_acc={result['methods']['standard_64']['accuracy']}, "
            f"dnb_84_acc={result['methods']['dnb_84']['accuracy']}, "
            f"batch={safe_batch}, backend={result['attention_backend']}"
        )
        RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_progress(
            3,
            3,
            {
                "pass": result["pass"],
                "standard_64_acc": result["methods"]["standard_64"]["accuracy"],
                "dnb_84_acc": result["methods"]["dnb_84"]["accuracy"],
                "batch_size": safe_batch,
            },
        )
        mark_done("success" if result["pass"] else "failed", result["summary"])
        return 0 if result["pass"] else 1
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": TASK_ID,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": repr(exc),
        }
        RESULTS_PATH.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        report_progress(3, 3, {"pass": False, "error": repr(exc)})
        mark_done("failed", f"baseline_standard_dnb failed: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
