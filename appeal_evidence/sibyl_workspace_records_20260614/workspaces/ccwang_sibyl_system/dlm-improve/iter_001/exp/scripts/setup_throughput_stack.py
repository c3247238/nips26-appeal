#!/usr/bin/env python3
"""Pilot task: setup_throughput_stack.

Verifies the remote runtime stack for high-throughput LLaDA inference:
- package and GPU availability
- attention backend choice (flash_attention_2 -> sdpa -> eager)
- single-sample forward sanity
- batch-size probing with BatchedDLMInference
- lightweight throughput benchmark
- optional torch.compile smoke test

Writes the Sibyl experimenter contract files:
- exp/results/setup_throughput_stack.pid
- exp/results/setup_throughput_stack_PROGRESS.json
- exp/results/setup_throughput_stack_DONE
- exp/results/setup_throughput_stack_gpu_profile.json
- exp/results/setup_throughput_verification.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from batched_dlm_utils import BatchedDLMInference
from setup_env_verify import (
    check_dataset,
    check_gpu,
    check_model,
    check_packages,
    check_single_sample,
)

TASK_ID = "setup_throughput_stack"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
RESULTS_PATH = RESULTS_DIR / "setup_throughput_verification.json"
GPU_PROFILE_PATH = RESULTS_DIR / f"{TASK_ID}_gpu_profile.json"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"


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


def benchmark_forward(engine: BatchedDLMInference, batch_size: int, seq_len: int, repeats: int = 3) -> dict:
    dummy = torch.full(
        (batch_size, seq_len),
        engine.pad_token_id,
        dtype=torch.long,
        device=engine.device,
    )
    attn = torch.ones_like(dummy, dtype=torch.long)

    # Warmup
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        engine.model(input_ids=dummy, attention_mask=attn)
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            engine.model(input_ids=dummy, attention_mask=attn)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)

    mean_sec = sum(elapsed) / len(elapsed)
    tokens = batch_size * seq_len
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mean_forward_sec": round(mean_sec, 4),
        "tokens_per_sec": round(tokens / mean_sec, 2),
    }


def run_engine_check(use_compile: bool) -> dict:
    started = time.time()
    engine = BatchedDLMInference(
        model_path=MODEL_PATH,
        device="cuda",
        use_flash_attn=True,
        use_compile=use_compile,
    )
    safe_batch = engine.find_max_batch_size(gen_length=256, prompt_max_len=512, lo=1, hi=64)
    # Benchmark near the safe maximum so the result reflects throughput-oriented execution.
    bench_batch = max(1, safe_batch)
    bench = benchmark_forward(engine, batch_size=bench_batch, seq_len=768)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
    backend = getattr(getattr(engine.model, "config", None), "_attn_implementation", None)
    if backend is None:
        backend = "eager_or_default"
    result = {
        "compile_enabled": use_compile,
        "attn_backend": backend,
        "safe_batch_size": safe_batch,
        "peak_vram_mb": peak_vram_mb,
        "model_load_and_probe_sec": round(time.time() - started, 2),
        "throughput": bench,
    }
    del engine
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    results: dict = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "gpu_id": GPU_ID,
        "status": "running",
        "checks": {},
    }

    try:
        report_progress(0, 5, {"phase": "packages"})
        results["checks"]["packages"] = check_packages()

        report_progress(1, 5, {"phase": "gpu"})
        results["checks"]["gpu"] = check_gpu()

        report_progress(2, 5, {"phase": "model_smoke"})
        model_info, model, tokenizer = check_model()
        results["checks"]["model"] = model_info
        if model is None or tokenizer is None:
            raise RuntimeError(model_info.get("error", "model load failed"))

        results["checks"]["single_sample"] = check_single_sample(model, tokenizer)
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        report_progress(3, 5, {"phase": "dataset"})
        results["checks"]["dataset"] = check_dataset()

        report_progress(4, 5, {"phase": "throughput_probe"})
        compile_off = run_engine_check(use_compile=False)
        results["checks"]["throughput_compile_off"] = compile_off

        compile_on: dict | None = None
        compile_error: str | None = None
        if hasattr(torch, "compile"):
            try:
                compile_on = run_engine_check(use_compile=True)
            except Exception as exc:  # noqa: BLE001
                compile_error = repr(exc)
                torch.cuda.empty_cache()
                gc.collect()
        results["checks"]["throughput_compile_on"] = compile_on
        if compile_error:
            results["checks"]["compile_error"] = compile_error

        gpu_profile = {
            "gpu_name": results["checks"]["gpu"].get("device_name"),
            "vram_total_mb": results["checks"]["gpu"].get("vram_total_mb"),
            "max_batch_size": compile_off["safe_batch_size"],
            "vram_used_mb": compile_off["peak_vram_mb"],
            "utilization_pct": round(
                100.0 * compile_off["peak_vram_mb"] / max(1, results["checks"]["gpu"].get("vram_total_mb", 1)),
                1,
            ),
            "attention_backend": compile_off["attn_backend"],
            "compile_tested": compile_on is not None,
        }
        GPU_PROFILE_PATH.write_text(json.dumps(gpu_profile, indent=2), encoding="utf-8")

        success = (
            results["checks"]["gpu"].get("cuda_available", False)
            and results["checks"]["dataset"].get("status") == "ok"
            and compile_off["safe_batch_size"] >= 1
        )
        results["status"] = "success" if success else "failed"
        results["pass"] = success
        results["summary"] = (
            f"backend={compile_off['attn_backend']}, safe_batch={compile_off['safe_batch_size']}, "
            f"compile={'ok' if compile_on else 'skip_or_fail'}, "
            f"tps={compile_off['throughput']['tokens_per_sec']}, "
            f"peak_vram_mb={compile_off['peak_vram_mb']}"
        )

        report_progress(
            5,
            5,
            {
                "safe_batch_size": compile_off["safe_batch_size"],
                "tokens_per_sec": compile_off["throughput"]["tokens_per_sec"],
                "attn_backend": compile_off["attn_backend"],
                "compile_ok": compile_on is not None,
            },
        )
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        mark_done("success" if success else "failed", results["summary"])
        return 0 if success else 1
    except Exception as exc:  # noqa: BLE001
        results["status"] = "failed"
        results["pass"] = False
        results["error"] = repr(exc)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        report_progress(5, 5, {"error": repr(exc)})
        mark_done("failed", repr(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
