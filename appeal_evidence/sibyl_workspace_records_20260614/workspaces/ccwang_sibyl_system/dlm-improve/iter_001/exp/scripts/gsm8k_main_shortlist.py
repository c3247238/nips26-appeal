#!/usr/bin/env python3
"""Main GSM8K shortlist: merge strong baselines with fresh entropy/TIGER runs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from batched_dlm_utils import BatchedDLMInference
from batched_gsm8k_baselines import (
    GEN_LENGTH,
    RESULTS_DIR,
    SEED,
    check_gsm8k_correct,
    extract_gsm8k_answer,
    load_samples,
    read_safe_batch_size,
    write_json,
)
from tiger_signal_screen import batched_instability_revise


TASK_ID = "gsm8k_main_shortlist"
REVISION_FRACTION = 0.10
REVISION_STEPS = 3
MAX_INITIAL_BATCH_SIZE = 32
WORKER_MODES = ("entropy", "tiger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="")
    parser.add_argument("--worker", choices=WORKER_MODES)
    parser.add_argument("--use-compile", action="store_true")
    return parser.parse_args()


def report_progress(metric: dict[str, Any]) -> None:
    write_json(
        RESULTS_DIR / f"{TASK_ID}_PROGRESS.json",
        {
            "task_id": TASK_ID,
            "epoch": 0,
            "total_epochs": 1,
            "step": 0,
            "total_steps": 1,
            "loss": None,
            "metric": metric,
            "updated_at": datetime.now().isoformat(),
        },
    )


def mark_done(status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            final_progress = {}
    write_json(
        RESULTS_DIR / f"{TASK_ID}_DONE",
        {
            "task_id": TASK_ID,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": datetime.now().isoformat(),
        },
    )


def worker_progress_path(mode: str) -> Path:
    return RESULTS_DIR / f"{TASK_ID}_{mode}_worker_PROGRESS.json"


def worker_result_path(mode: str) -> Path:
    return RESULTS_DIR / f"{TASK_ID}_{mode}_worker.json"


def worker_log_path(mode: str) -> Path:
    return RESULTS_DIR / f"{TASK_ID}_{mode}.log"


def engine_attention_backend(engine: BatchedDLMInference) -> str:
    return getattr(getattr(engine.model, "config", None), "_attn_implementation", None) or "eager_or_default"


def engine_compile_enabled(engine: BatchedDLMInference) -> bool:
    return hasattr(engine.model, "_orig_mod")


def is_retryable_runtime_error(exc: RuntimeError) -> bool:
    text = repr(exc).lower()
    markers = (
        "cuda error",
        "cublas_status_invalid_value",
        "cuda out of memory",
        "out of memory",
        "cudnn",
        "triton",
    )
    return any(marker in text for marker in markers)


def worker_report(mode: str, metric: dict[str, Any]) -> None:
    write_json(
        worker_progress_path(mode),
        {
            "task_id": TASK_ID,
            "worker": mode,
            "metric": metric,
            "updated_at": datetime.now().isoformat(),
        },
    )


def evaluate_worker_method(
    engine: BatchedDLMInference,
    mode: str,
    samples: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    total_correct = 0
    total_nfe = 0
    total_tokens_changed = 0
    signal_scores: list[float] = []
    per_sample = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(engine.device)

    method_name = "Entropy-Revise-64+3" if mode == "entropy" else "TIGER-Instability-64+3"

    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        prompts = [sample["prompt_ids"] for sample in chunk]

        if mode == "entropy":
            chunk_results = engine.batched_entropy_revise(
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=64,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=batch_size,
                seed=SEED,
            )
        elif mode == "tiger":
            chunk_results = batched_instability_revise(
                engine=engine,
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=64,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=batch_size,
                seed=SEED,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported worker mode: {mode}")

        chunk_texts = engine.decode_results(chunk_results)
        for sample, result, generated_text in zip(chunk, chunk_results, chunk_texts, strict=True):
            correct = check_gsm8k_correct(generated_text, sample["final_answer"])
            total_correct += int(correct)
            total_nfe += int(result["nfe"])
            total_tokens_changed += int(result.get("tokens_changed", 0))
            stats = result.get("instability_stats") or result.get("entropy_stats") or {}
            if stats:
                signal_scores.append(
                    float(
                        stats.get(
                            "revision_mean_instability",
                            stats.get("revision_mean_entropy", stats.get("mean_entropy", 0.0)),
                        ),
                    ),
                )
            if len(per_sample) < 10:
                per_sample.append(
                    {
                        "idx": sample["idx"],
                        "correct": bool(correct),
                        "nfe": int(result["nfe"]),
                        "tokens_changed": int(result.get("tokens_changed", 0)),
                        "signal_stats": stats,
                        "predicted_answer": extract_gsm8k_answer(generated_text),
                        "reference_answer": sample["final_answer"],
                        "generated_text": generated_text[:500],
                    }
                )

        processed = min(chunk_start + len(chunk), len(samples))
        worker_report(
            mode,
            {
                "phase": "running",
                "method": method_name,
                "processed": processed,
                "total": len(samples),
                "accuracy_so_far": round(total_correct / max(1, processed), 4),
                "avg_nfe": round(total_nfe / max(1, processed), 2),
                "avg_tokens_changed": round(total_tokens_changed / max(1, processed), 2),
                "batch_size": batch_size,
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "elapsed_sec": round(time.perf_counter() - start, 2),
            },
        )

    latency_sec = time.perf_counter() - start
    return {
        "method": method_name,
        "num_samples": len(samples),
        "accuracy": round(total_correct / max(1, len(samples)), 4),
        "actual_nfe": round(total_nfe / max(1, len(samples)), 2),
        "latency_sec": round(latency_sec, 2),
        "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(latency_sec, 1e-6), 2),
        "batch_size": batch_size,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
        "attention_backend": engine_attention_backend(engine),
        "compile_enabled": engine_compile_enabled(engine),
        "avg_tokens_changed": round(total_tokens_changed / max(1, len(samples)), 2),
        "avg_signal_value": round(sum(signal_scores) / max(1, len(signal_scores)), 4),
        "per_sample": per_sample,
    }


def run_worker_with_backoff(
    mode: str,
    samples: list[dict[str, Any]],
    requested_batch_size: int,
    use_compile: bool,
) -> dict[str, Any]:
    current_batch_size = max(1, min(requested_batch_size, MAX_INITIAL_BATCH_SIZE, len(samples)))
    last_error: RuntimeError | None = None

    while current_batch_size >= 1:
        engine: BatchedDLMInference | None = None
        try:
            engine = BatchedDLMInference(
                device="cuda",
                use_flash_attn=True,
                use_compile=use_compile,
            )
            worker_report(
                mode,
                {
                    "phase": "loaded",
                    "batch_size": current_batch_size,
                    "attention_backend": engine_attention_backend(engine),
                    "compile_enabled": engine_compile_enabled(engine),
                },
            )
            result = evaluate_worker_method(engine, mode, samples, current_batch_size)
            result["worker_mode"] = mode
            result["requested_batch_size"] = requested_batch_size
            return result
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_runtime_error(exc) or current_batch_size == 1:
                raise
            next_batch_size = max(1, current_batch_size // 2)
            worker_report(
                mode,
                {
                    "phase": "batch_backoff",
                    "batch_size": current_batch_size,
                    "retry_batch_size": next_batch_size,
                    "error": repr(exc)[:240],
                },
            )
            current_batch_size = next_batch_size
        finally:
            if engine is not None:
                del engine
            torch.cuda.empty_cache()
            gc.collect()

    if last_error is not None:  # pragma: no cover
        raise last_error
    raise RuntimeError(f"{mode} worker exited without result")


def parse_gpu_ids(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return [int(token) for token in tokens]


def load_baseline_methods() -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []

    std_path = RESULTS_DIR / "baseline_standard_dnb.json"
    prophet_path = RESULTS_DIR / "baseline_prophet.json"
    core_path = RESULTS_DIR / "baseline_core.json"

    standard = json.loads(std_path.read_text(encoding="utf-8"))
    methods.append(
        {
            "method": "Standard-64",
            "num_samples": standard["sample_count"],
            "accuracy": standard["methods"]["standard_64"]["accuracy"],
            "actual_nfe": standard["methods"]["standard_64"]["actual_nfe_avg"],
            "latency_sec": standard["methods"]["standard_64"]["latency_sec"],
            "tokens_per_sec": standard["methods"]["standard_64"]["tokens_per_sec"],
            "batch_size": standard["methods"]["standard_64"]["batch_size"],
            "peak_vram_mb": None,
            "attention_backend": standard["attention_backend"],
            "compile_enabled": standard["compile_enabled"],
        }
    )
    methods.append(
        {
            "method": "DNB-84",
            "num_samples": standard["sample_count"],
            "accuracy": standard["methods"]["dnb_84"]["accuracy"],
            "actual_nfe": standard["methods"]["dnb_84"]["actual_nfe_avg"],
            "latency_sec": standard["methods"]["dnb_84"]["latency_sec"],
            "tokens_per_sec": standard["methods"]["dnb_84"]["tokens_per_sec"],
            "batch_size": standard["methods"]["dnb_84"]["batch_size"],
            "peak_vram_mb": None,
            "attention_backend": standard["attention_backend"],
            "compile_enabled": standard["compile_enabled"],
        }
    )

    prophet = json.loads(prophet_path.read_text(encoding="utf-8"))
    methods.extend(prophet["methods"])

    core = json.loads(core_path.read_text(encoding="utf-8"))
    methods.append(
        {
            "method": core["method"],
            "num_samples": core["num_samples"],
            "accuracy": core["accuracy"],
            "actual_nfe": core["actual_nfe"],
            "latency_sec": core["latency_sec"],
            "tokens_per_sec": core["tokens_per_sec"],
            "batch_size": core["batch_size"],
            "peak_vram_mb": core["peak_vram_mb"],
            "attention_backend": core["attention_backend"],
            "compile_enabled": core["compile_enabled"],
            "proxy_note": core.get("proxy_note"),
        }
    )
    return methods


def launch_worker(mode: str, gpu_id: int, args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--samples",
        str(args.samples),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.use_compile:
        cmd.append("--use-compile")
    log_handle = open(worker_log_path(mode), "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parent,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def collect_gpu_info(gpu_ids: list[int]) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={','.join(str(i) for i in gpu_ids)}",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []

    infos = []
    for line in output.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            continue
        infos.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
            }
        )
    return infos


def run_worker_entry(args: argparse.Namespace) -> int:
    mode = args.worker
    assert mode is not None
    random_seed = SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct",
        trust_remote_code=True,
    )
    requested_batch_size = args.batch_size if args.batch_size > 0 else read_safe_batch_size()
    samples = load_samples(tokenizer, args.samples)
    result = run_worker_with_backoff(mode, samples, requested_batch_size, args.use_compile)
    write_json(worker_result_path(mode), result)
    worker_report(mode, {"phase": "done", "result_path": str(worker_result_path(mode))})
    return 0


def run_master(args: argparse.Namespace) -> int:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    requested_batch_size = args.batch_size if args.batch_size > 0 else read_safe_batch_size()
    raw_gpu_ids = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = parse_gpu_ids(raw_gpu_ids) if raw_gpu_ids else [0, 1]
    if len(gpu_ids) == 1:
        gpu_ids = [gpu_ids[0], gpu_ids[0]]

    for mode in WORKER_MODES:
        for path in (worker_progress_path(mode), worker_result_path(mode)):
            if path.exists():
                path.unlink()

    procs = {
        "entropy": launch_worker("entropy", gpu_ids[0], args),
        "tiger": launch_worker("tiger", gpu_ids[1], args),
    }
    start = time.time()

    try:
        while True:
            all_done = True
            worker_states = {}
            for mode, proc in procs.items():
                progress = {}
                if worker_progress_path(mode).exists():
                    try:
                        progress = json.loads(worker_progress_path(mode).read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        progress = {}
                exit_code = proc.poll()
                worker_states[mode] = {
                    "gpu_id": gpu_ids[0] if mode == "entropy" else gpu_ids[1],
                    "exit_code": exit_code,
                    "progress": progress,
                }
                if exit_code is None:
                    all_done = False

            report_progress(
                {
                    "phase": "waiting_workers",
                    "elapsed_sec": round(time.time() - start, 2),
                    "requested_batch_size": requested_batch_size,
                    "gpu_ids": gpu_ids,
                    "workers": worker_states,
                }
            )
            if all_done:
                break
            time.sleep(10)

        failures = {mode: proc.returncode for mode, proc in procs.items() if proc.returncode != 0}
        if failures:
            raise RuntimeError(f"workers failed: {failures}")

        methods = load_baseline_methods()
        entropy_result = json.loads(worker_result_path("entropy").read_text(encoding="utf-8"))
        tiger_result = json.loads(worker_result_path("tiger").read_text(encoding="utf-8"))
        methods.extend([entropy_result, tiger_result])

        by_name = {item["method"]: item for item in methods}
        tiger_acc = by_name["TIGER-Instability-64+3"]["accuracy"]
        entropy_acc = by_name["Entropy-Revise-64+3"]["accuracy"]
        prophet_acc = by_name["Prophet-64"]["accuracy"]
        dnb_acc = by_name["DNB-84"]["accuracy"]
        core_acc = by_name["CORE-proxy-64"]["accuracy"]

        output = {
            "task_id": TASK_ID,
            "summary_name": "GSM8K 主结果：强基线 + TIGER",
            "timestamp": datetime.now().isoformat(),
            "num_samples": args.samples,
            "gen_length": GEN_LENGTH,
            "config": {
                "requested_batch_size": requested_batch_size,
                "worker_gpu_ids": {
                    "entropy": gpu_ids[0],
                    "tiger": gpu_ids[1],
                },
                "use_compile": args.use_compile,
                "seed": SEED,
                "baseline_sources": [
                    "baseline_standard_dnb.json",
                    "baseline_prophet.json",
                    "baseline_core.json",
                ],
            },
            "methods": methods,
            "decision": {
                "tiger_acc": tiger_acc,
                "entropy_acc": entropy_acc,
                "delta_vs_entropy": round(tiger_acc - entropy_acc, 4),
                "delta_vs_prophet": round(tiger_acc - prophet_acc, 4),
                "delta_vs_dnb": round(tiger_acc - dnb_acc, 4),
                "delta_vs_core": round(tiger_acc - core_acc, 4),
                "beats_at_least_one_strong_baseline": bool(
                    any(tiger_acc > baseline for baseline in (prophet_acc, dnb_acc, core_acc))
                ),
                "pilot_passes_gate": bool(tiger_acc > entropy_acc and (tiger_acc > prophet_acc or tiger_acc > dnb_acc)),
            },
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", output)
        write_json(
            RESULTS_DIR / f"{TASK_ID}_gpu_profile.json",
            {
                "task_id": TASK_ID,
                "gpus": collect_gpu_info(sorted(set(gpu_ids))),
                "peak_vram_mb": {
                    "entropy": entropy_result["peak_vram_mb"],
                    "tiger": tiger_result["peak_vram_mb"],
                },
            },
        )
        summary = (
            f"TIGER acc={tiger_acc:.4f}, Entropy acc={entropy_acc:.4f}, "
            f"Prophet acc={prophet_acc:.4f}, DNB acc={dnb_acc:.4f}, CORE acc={core_acc:.4f}"
        )
        mark_done("success", summary)
        return 0
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()


def main() -> int:
    args = parse_args()
    try:
        if args.worker:
            return run_worker_entry(args)
        return run_master(args)
    except Exception as exc:  # noqa: BLE001
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {
                "task_id": TASK_ID,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": repr(exc),
            },
        )
        mark_done("failed", repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
