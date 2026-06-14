#!/usr/bin/env python3
"""Optional batched DCD-lite baseline for the TIGER pilot queue."""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime

import torch

from batched_dlm_utils import BatchedDLMInference
from batched_gsm8k_baselines import (
    GEN_LENGTH,
    RESULTS_DIR,
    SEED,
    check_gsm8k_correct,
    extract_gsm8k_answer,
    load_samples,
    mark_done,
    read_safe_batch_size,
    report_progress,
    write_json,
)


TASK_ID = "baseline_dcd_optional"


def evaluate_dcd(
    engine: BatchedDLMInference,
    samples: list[dict],
    batch_size: int,
) -> dict:
    total_correct = 0
    total_nfe = 0
    total_deferred = 0
    total_committed = 0
    per_sample = []
    method_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(engine.device)

    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        prompts = [sample["prompt_ids"] for sample in chunk]
        chunk_results = engine.batched_dcd(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            num_steps=64,
            base_confidence_threshold=0.90,
            final_confidence_threshold=0.55,
            window_radius=2,
            late_commit_start=0.75,
            batch_size=batch_size,
            seed=SEED,
        )
        chunk_texts = engine.decode_results(chunk_results)

        for sample, result, generated_text in zip(chunk, chunk_results, chunk_texts, strict=True):
            is_correct = check_gsm8k_correct(generated_text, sample["final_answer"])
            total_correct += int(is_correct)
            total_nfe += int(result["nfe"])
            total_deferred += int(result.get("deferred_count", 0))
            total_committed += int(result.get("committed_count", 0))
            per_sample.append(
                {
                    "idx": sample["idx"],
                    "correct": bool(is_correct),
                    "nfe": int(result["nfe"]),
                    "deferred_count": int(result.get("deferred_count", 0)),
                    "committed_count": int(result.get("committed_count", 0)),
                    "predicted_answer": extract_gsm8k_answer(generated_text),
                    "reference_answer": sample["final_answer"],
                    "generated_text": generated_text[:500],
                }
            )

        processed = min(chunk_start + len(chunk), len(samples))
        elapsed = time.perf_counter() - method_start
        report_progress(
            TASK_ID,
            1,
            1,
            processed,
            len(samples),
            {
                "method": "DCD-lite-64",
                "accuracy_so_far": round(total_correct / max(1, processed), 4),
                "avg_nfe": round(total_nfe / max(1, processed), 2),
                "avg_deferred": round(total_deferred / max(1, processed), 2),
                "batch_size": batch_size,
                "elapsed_sec": round(elapsed, 2),
            },
        )

    latency_sec = time.perf_counter() - method_start
    accuracy = total_correct / max(1, len(samples))
    avg_nfe = total_nfe / max(1, len(samples))
    avg_deferred = total_deferred / max(1, len(samples))
    avg_committed = total_committed / max(1, len(samples))
    total_generated_tokens = len(samples) * GEN_LENGTH
    tokens_per_sec = total_generated_tokens / max(latency_sec, 1e-6)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
    attention_backend = getattr(getattr(engine.model, "config", None), "_attn_implementation", None) or "eager_or_default"
    compile_enabled = hasattr(engine.model, "_orig_mod") or hasattr(torch, "compile")

    return {
        "method": "DCD-lite-64",
        "proxy_note": "Low-cost deferred-commit proxy implemented with batched confidence-aware windows.",
        "num_samples": len(samples),
        "accuracy": round(accuracy, 4),
        "actual_nfe": round(avg_nfe, 2),
        "latency_sec": round(latency_sec, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "batch_size": batch_size,
        "peak_vram_mb": peak_vram_mb,
        "attention_backend": attention_backend,
        "compile_enabled": compile_enabled,
        "avg_deferred_count": round(avg_deferred, 2),
        "avg_committed_count": round(avg_committed, 2),
        "per_sample": per_sample[:10],
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    engine = None
    try:
        batch_size = read_safe_batch_size()
        engine = BatchedDLMInference(
            device="cuda",
            use_flash_attn=True,
            use_compile=True,
        )
        batch_size = min(batch_size, 100)
        samples = load_samples(engine.tokenizer, 100)
        batch_size = min(batch_size, len(samples))
        report_progress(TASK_ID, 0, 1, 0, len(samples), {"phase": "loaded"})

        method_result = evaluate_dcd(engine, samples, batch_size)
        output = {
            "task_id": TASK_ID,
            "summary_name": "DCD 可选补充基线",
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "gen_length": GEN_LENGTH,
            "config": {
                "batch_size": batch_size,
                "seed": SEED,
                "proxy": True,
            },
            "methods": [method_result],
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", output)
        write_json(
            RESULTS_DIR / f"{TASK_ID}_gpu_profile.json",
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "used_batch_size": batch_size,
                "peak_vram_mb": method_result["peak_vram_mb"],
                "attention_backend": method_result["attention_backend"],
                "compile_enabled": method_result["compile_enabled"],
                "proxy": True,
            },
        )
        mark_done(
            TASK_ID,
            "success",
            f"DCD-lite-64 acc={method_result['accuracy']:.4f} nfe={method_result['actual_nfe']:.2f} tps={method_result['tokens_per_sec']:.1f}",
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {
                "task_id": TASK_ID,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": repr(exc),
            },
        )
        mark_done(TASK_ID, "failed", repr(exc))
        return 1
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
