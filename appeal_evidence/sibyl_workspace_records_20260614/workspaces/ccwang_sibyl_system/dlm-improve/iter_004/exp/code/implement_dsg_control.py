#!/usr/bin/env python3
"""Smoke-run the DSG control candidate on one audited GSM8K sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

TASK_ID = "implement_dsg_control"
CANDIDATE_ID = "cand_dsg"
DEFAULT_GATE_THRESHOLD = 0.25
PROJECT_ROOT = Path(
    os.environ.get(
        "SIBYL_REMOTE_PROJECT_ROOT",
        "/home/ccwang/sibyl_system/projects/dlm-improve",
    )
)
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
SCRIPTS_DIR = PROJECT_ROOT / "exp" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from batched_dlm_utils import BatchedDLMInference  # noqa: E402
from batched_gsm8k_baselines import (  # noqa: E402
    GEN_LENGTH,
    SEED,
    check_gsm8k_correct,
    extract_gsm8k_answer,
    load_samples,
)
from tiger_signal_screen import batched_instability_revise  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pid_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}.pid"


def progress_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"


def done_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}_DONE"


def result_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}.json"


def gpu_profile_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}_gpu_profile.json"


def log_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}.log"


def log(message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with log_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def report_progress(epoch: int, total_epochs: int, metric: dict[str, Any]) -> None:
    write_json(
        progress_path(),
        {
            "task_id": TASK_ID,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "metric": metric,
            "updated_at": now_iso(),
        },
    )


def mark_done(status: str, summary: str, extra: dict[str, Any] | None = None) -> None:
    if pid_path().exists():
        pid_path().unlink()
    final_progress: dict[str, Any] = {}
    if progress_path().exists():
        try:
            final_progress = json.loads(progress_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            final_progress = {}
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": now_iso(),
    }
    if extra:
        payload.update(extra)
    write_json(done_path(), payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def read_runtime_contract() -> dict[str, Any]:
    gpu_profile = read_json(RESULTS_DIR / "shared_runtime_probe_gpu_profile.json")
    progress = read_json(RESULTS_DIR / "shared_runtime_probe_PROGRESS.json")
    metric = progress.get("metric") or {}
    max_batch_size = gpu_profile.get("max_batch_size") or metric.get("safe_batch_size") or 32
    return {
        "source": "shared_runtime_probe",
        "gpu_name": gpu_profile.get("gpu_name", "unknown"),
        "vram_total_mb": gpu_profile.get("vram_total_mb"),
        "max_batch_size": int(max_batch_size),
        "attention_backend": gpu_profile.get(
            "attention_backend",
            metric.get("attn_backend", "eager"),
        ),
        "compile_tested": bool(
            gpu_profile.get("compile_tested", metric.get("compile_ok", False))
        ),
        "tokens_per_sec_reference": metric.get("tokens_per_sec"),
    }


def read_gate_threshold() -> float:
    payload = read_json(RESULTS_DIR / "tiger_signal_screen.json")
    methods = payload.get("methods") or []
    for method in methods:
        if method.get("method") == "TIGER-Instability-64+3":
            value = method.get("avg_signal_value")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    break
    return DEFAULT_GATE_THRESHOLD


def build_engine(prefer_compile: bool) -> tuple[BatchedDLMInference, bool]:
    compile_enabled = prefer_compile
    try:
        engine = BatchedDLMInference(
            model_path="/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct",
            device="cuda",
            use_flash_attn=True,
            use_compile=prefer_compile,
        )
        return engine, compile_enabled and hasattr(engine.model, "_orig_mod")
    except Exception as exc:
        if not prefer_compile:
            raise
        log(f"compile 初始化失败，回退到 eager/未编译: {exc!r}")
        torch.cuda.empty_cache()
        engine = BatchedDLMInference(
            model_path="/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct",
            device="cuda",
            use_flash_attn=True,
            use_compile=False,
        )
        return engine, False


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_path().write_text(str(os.getpid()), encoding="utf-8")
    report_progress(
        0,
        3,
        {
            "phase": "boot",
            "candidate_id": CANDIDATE_ID,
            "samples_requested": args.samples,
        },
    )

    engine: BatchedDLMInference | None = None
    try:
        runtime_contract = read_runtime_contract()
        gate_threshold = read_gate_threshold()
        log(
            "启动 DSG control smoke；"
            f"runtime_source={runtime_contract['source']} "
            f"safe_batch={runtime_contract['max_batch_size']} "
            f"gate_threshold={gate_threshold:.4f}"
        )

        engine, compile_enabled = build_engine(
            prefer_compile=runtime_contract.get("compile_tested", False),
        )
        samples = load_samples(engine.tokenizer, args.samples)
        batch_hint = args.batch_size if args.batch_size > 0 else runtime_contract["max_batch_size"]
        batch_size = max(1, min(batch_hint, len(samples)))
        report_progress(
            1,
            3,
            {
                "phase": "engine_ready",
                "runtime_contract_source": runtime_contract["source"],
                "requested_batch_size": int(batch_hint),
                "effective_batch_size": batch_size,
                "attention_backend": getattr(
                    getattr(engine.model, "config", None),
                    "_attn_implementation",
                    None,
                )
                or runtime_contract["attention_backend"],
                "compile_enabled": compile_enabled,
            },
        )

        torch.cuda.reset_peak_memory_stats(engine.device)
        start_time = time.perf_counter()
        dsg_results = batched_instability_revise(
            engine=engine,
            prompt_ids_list=[sample["prompt_ids"] for sample in samples],
            gen_length=GEN_LENGTH,
            num_draft_steps=64,
            revision_fraction=0.10,
            revision_steps=3,
            batch_size=batch_size,
            seed=SEED,
        )
        generated_texts = engine.decode_results(dsg_results)
        latency_sec = time.perf_counter() - start_time

        result = dsg_results[0]
        sample = samples[0]
        generated_text = generated_texts[0]
        instability_stats = result.get("instability_stats") or {}
        gate_open = bool(
            float(instability_stats.get("revision_mean_instability", 0.0)) >= gate_threshold
            and int(result.get("tokens_changed", 0)) > 0
        )
        is_correct = check_gsm8k_correct(generated_text, sample["final_answer"])
        peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
        attention_backend = getattr(
            getattr(engine.model, "config", None),
            "_attn_implementation",
            None,
        ) or runtime_contract["attention_backend"]

        payload = {
            "task_id": TASK_ID,
            "candidate_id": CANDIDATE_ID,
            "mode": "PILOT",
            "status": "success",
            "go_no_go": "GO",
            "notes": "DSG control 已完成 smoke sample；这里只验证运行路径与 drift-span diagnostics，不宣称方法已成立。",
            "runtime_contract": {
                **runtime_contract,
                "attention_backend": attention_backend,
                "compile_enabled": compile_enabled,
                "effective_batch_size": batch_size,
                "peak_vram_mb": peak_vram_mb,
            },
            "method": {
                "name": "DSG-control-64+3",
                "num_samples": len(samples),
                "actual_nfe": int(result["nfe"]),
                "latency_sec": round(latency_sec, 2),
                "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(latency_sec, 1e-6), 2),
                "tokens_changed": int(result.get("tokens_changed", 0)),
                "gate_threshold": round(gate_threshold, 4),
                "gate_open": gate_open,
                "attention_backend": attention_backend,
                "compile_enabled": compile_enabled,
            },
            "diagnostics": {
                "mean_instability": round(
                    float(instability_stats.get("mean_instability", 0.0)),
                    4,
                ),
                "max_instability": round(
                    float(instability_stats.get("max_instability", 0.0)),
                    4,
                ),
                "revision_mean_instability": round(
                    float(instability_stats.get("revision_mean_instability", 0.0)),
                    4,
                ),
                "num_revised": int(instability_stats.get("num_revised", 0)),
                "tokens_changed": int(instability_stats.get("tokens_changed", 0)),
            },
            "sample": {
                "idx": int(sample["idx"]),
                "question": sample["question"],
                "predicted_answer": extract_gsm8k_answer(generated_text),
                "reference_answer": sample["final_answer"],
                "correct": bool(is_correct),
                "generated_text": generated_text[:800],
            },
            "timestamp": now_iso(),
        }
        write_json(result_path(), payload)
        write_json(
            gpu_profile_path(),
            {
                "gpu_name": runtime_contract.get("gpu_name", "unknown"),
                "vram_total_mb": runtime_contract.get("vram_total_mb"),
                "max_batch_size": runtime_contract.get("max_batch_size"),
                "vram_used_mb": peak_vram_mb,
                "utilization_pct": None,
                "attention_backend": attention_backend,
                "compile_tested": compile_enabled,
            },
        )
        report_progress(
            2,
            3,
            {
                "phase": "smoke_complete",
                "actual_nfe": int(result["nfe"]),
                "gate_open": gate_open,
                "tokens_changed": int(result.get("tokens_changed", 0)),
                "correct": bool(is_correct),
                "peak_vram_mb": peak_vram_mb,
            },
        )
        summary = (
            "DSG control smoke 完成；"
            f"gate_open={gate_open}, "
            f"tokens_changed={int(result.get('tokens_changed', 0))}, "
            f"actual_nfe={int(result['nfe'])}"
        )
        mark_done("success", summary, {"result_path": str(result_path())})
        log(summary)
        return 0
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}"
        tb = traceback.format_exc(limit=20)
        log(f"任务失败: {error_text}\n{tb}")
        report_progress(3, 3, {"phase": "failed", "error": error_text})
        mark_done(
            "failed",
            f"DSG control smoke 失败: {error_text}",
            {"traceback": tb[-4000:]},
        )
        return 1
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
