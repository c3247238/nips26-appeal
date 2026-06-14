#!/usr/bin/env python3
"""Boundary validation for ungated versus gated TIGER."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

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
from full_llada_humaneval import (
    check_humaneval_correct,
    extract_code_completion,
    get_humaneval_samples,
)
from tiger_signal_screen import batched_instability_revise


TASK_ID = "tiger_gating_boundary"
REVISION_FRACTION = 0.10
REVISION_STEPS = 3
CODE_GEN_LENGTH = 384
DEFAULT_HUMANEVAL_SAMPLES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--humaneval-samples", type=int, default=DEFAULT_HUMANEVAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=0)
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


def engine_attention_backend(engine: BatchedDLMInference) -> str:
    return getattr(getattr(engine.model, "config", None), "_attn_implementation", None) or "eager_or_default"


def engine_compile_enabled(engine: BatchedDLMInference) -> bool:
    return hasattr(engine.model, "_orig_mod")


def parse_ok(prompt: str, completion: str) -> bool:
    try:
        ast.parse(prompt + completion)
        return True
    except SyntaxError:
        return False


def classify_failure(parse_success: bool, passed: bool, error: str | None) -> tuple[bool, bool]:
    if passed:
        return False, False
    if not parse_success:
        return True, False
    if error and ("syntaxerror" in error.lower() or "indentationerror" in error.lower()):
        return True, False
    return False, True


def read_gate_threshold() -> float:
    screen_path = RESULTS_DIR / "tiger_signal_screen.json"
    if not screen_path.exists():
        return 0.25
    try:
        payload = json.loads(screen_path.read_text(encoding="utf-8"))
        for method in payload.get("methods", []):
            if method.get("method") == "TIGER-Instability-64+3":
                value = method.get("avg_signal_value")
                if value is not None:
                    return float(value)
    except (OSError, ValueError, TypeError):
        return 0.25
    return 0.25


def summarize_ratio(numer: int, denom: int) -> float:
    return round(numer / max(1, denom), 4)


def run_gsm8k_reasoning(
    engine: BatchedDLMInference,
    samples: list[dict[str, Any]],
    batch_size: int,
    gate_threshold: float,
) -> dict[str, Any]:
    draft_correct = 0
    tiger_correct = 0
    gated_correct = 0
    tiger_nfe = 0
    gated_nfe = 0
    gates_open = 0
    tokens_changed = 0
    signal_values: list[float] = []
    per_sample = []
    start = time.perf_counter()

    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        prompts = [sample["prompt_ids"] for sample in chunk]
        draft_results = engine.batched_standard_denoising(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            num_steps=64,
            batch_size=batch_size,
            seed=SEED,
        )
        tiger_results = batched_instability_revise(
            engine=engine,
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            num_draft_steps=64,
            revision_fraction=REVISION_FRACTION,
            revision_steps=REVISION_STEPS,
            batch_size=batch_size,
            seed=SEED,
        )

        draft_texts = engine.decode_results(draft_results)
        tiger_texts = engine.decode_results(tiger_results)

        for sample, draft_result, tiger_result, draft_text, tiger_text in zip(
            chunk,
            draft_results,
            tiger_results,
            draft_texts,
            tiger_texts,
            strict=True,
        ):
            draft_is_correct = check_gsm8k_correct(draft_text, sample["final_answer"])
            tiger_is_correct = check_gsm8k_correct(tiger_text, sample["final_answer"])
            draft_correct += int(draft_is_correct)
            tiger_correct += int(tiger_is_correct)

            stats = tiger_result.get("instability_stats") or {}
            signal_value = float(stats.get("revision_mean_instability", 0.0))
            signal_values.append(signal_value)
            gate_open = signal_value >= gate_threshold and int(tiger_result.get("tokens_changed", 0)) > 0
            selected_text = tiger_text if gate_open else draft_text
            selected_result = tiger_result if gate_open else draft_result
            gates_open += int(gate_open)
            gated_is_correct = check_gsm8k_correct(selected_text, sample["final_answer"])
            gated_correct += int(gated_is_correct)
            tiger_nfe += int(tiger_result["nfe"])
            gated_nfe += int(selected_result["nfe"])
            tokens_changed += int(tiger_result.get("tokens_changed", 0))

            if len(per_sample) < 10:
                per_sample.append(
                    {
                        "idx": sample["idx"],
                        "signal_value": round(signal_value, 4),
                        "gate_open": gate_open,
                        "draft_correct": bool(draft_is_correct),
                        "tiger_correct": bool(tiger_is_correct),
                        "gated_correct": bool(gated_is_correct),
                        "predicted_answer_draft": extract_gsm8k_answer(draft_text),
                        "predicted_answer_tiger": extract_gsm8k_answer(tiger_text),
                        "reference_answer": sample["final_answer"],
                    }
                )

        processed = min(chunk_start + len(chunk), len(samples))
        report_progress(
            {
                "phase": "gsm8k_reasoning",
                "processed": processed,
                "total": len(samples),
                "draft_acc_so_far": round(draft_correct / max(1, processed), 4),
                "tiger_acc_so_far": round(tiger_correct / max(1, processed), 4),
                "gated_acc_so_far": round(gated_correct / max(1, processed), 4),
                "gate_open_rate": round(gates_open / max(1, processed), 4),
                "gate_threshold": gate_threshold,
            }
        )

    latency_sec = time.perf_counter() - start
    return {
        "task": "gsm8k",
        "num_samples": len(samples),
        "draft_accuracy": summarize_ratio(draft_correct, len(samples)),
        "ungated_tiger_accuracy": summarize_ratio(tiger_correct, len(samples)),
        "gated_tiger_accuracy": summarize_ratio(gated_correct, len(samples)),
        "ungated_tiger_actual_nfe": round(tiger_nfe / max(1, len(samples)), 2),
        "gated_tiger_actual_nfe": round(gated_nfe / max(1, len(samples)), 2),
        "gate_open_rate": summarize_ratio(gates_open, len(samples)),
        "avg_signal_value": round(sum(signal_values) / max(1, len(signal_values)), 4),
        "avg_tokens_changed": round(tokens_changed / max(1, len(samples)), 2),
        "latency_sec": round(latency_sec, 2),
        "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(latency_sec, 1e-6), 2),
        "per_sample": per_sample,
    }


def run_humaneval_boundary(
    engine: BatchedDLMInference,
    samples: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    metrics = {
        "standard": {"pass": 0, "syntax": 0, "runtime": 0},
        "ungated_tiger": {"pass": 0, "syntax": 0, "runtime": 0},
        "gated_tiger": {"pass": 0, "syntax": 0, "runtime": 0},
    }
    parse_guard_calls = 0
    parse_guard_sec = 0.0
    per_sample = []
    start = time.perf_counter()

    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        prompts = [sample["prompt_ids"] for sample in chunk]
        draft_results = engine.batched_standard_denoising(
            prompt_ids_list=prompts,
            gen_length=CODE_GEN_LENGTH,
            num_steps=64,
            batch_size=batch_size,
            seed=SEED,
        )
        tiger_results = batched_instability_revise(
            engine=engine,
            prompt_ids_list=prompts,
            gen_length=CODE_GEN_LENGTH,
            num_draft_steps=64,
            revision_fraction=REVISION_FRACTION,
            revision_steps=REVISION_STEPS,
            batch_size=batch_size,
            seed=SEED,
        )

        draft_texts = engine.decode_results(draft_results)
        tiger_texts = engine.decode_results(tiger_results)

        for sample, draft_text, tiger_text in zip(chunk, draft_texts, tiger_texts, strict=True):
            completion_standard = extract_code_completion(draft_text, sample["prompt"])
            completion_tiger = extract_code_completion(tiger_text, sample["prompt"])

            parse_start = time.perf_counter()
            standard_parse = parse_ok(sample["prompt"], completion_standard)
            tiger_parse = parse_ok(sample["prompt"], completion_tiger)
            parse_guard_sec += time.perf_counter() - parse_start
            parse_guard_calls += 2

            gated_completion = completion_tiger if tiger_parse else completion_standard
            gated_parse = tiger_parse if tiger_parse else standard_parse

            standard_pass, standard_error = check_humaneval_correct(
                sample["prompt"],
                completion_standard,
                sample["test"],
                sample["entry_point"],
            )
            tiger_pass, tiger_error = check_humaneval_correct(
                sample["prompt"],
                completion_tiger,
                sample["test"],
                sample["entry_point"],
            )
            gated_pass, gated_error = check_humaneval_correct(
                sample["prompt"],
                gated_completion,
                sample["test"],
                sample["entry_point"],
            )

            for key, passed, error, parse_success in (
                ("standard", standard_pass, standard_error, standard_parse),
                ("ungated_tiger", tiger_pass, tiger_error, tiger_parse),
                ("gated_tiger", gated_pass, gated_error, gated_parse),
            ):
                syntax_fail, runtime_fail = classify_failure(parse_success, passed, error)
                metrics[key]["pass"] += int(passed)
                metrics[key]["syntax"] += int(syntax_fail)
                metrics[key]["runtime"] += int(runtime_fail)

            if len(per_sample) < 10:
                per_sample.append(
                    {
                        "task_id": sample["task_id"],
                        "standard_parse": standard_parse,
                        "tiger_parse": tiger_parse,
                        "gate_reverted_to_standard": not tiger_parse,
                        "standard_pass": bool(standard_pass),
                        "ungated_tiger_pass": bool(tiger_pass),
                        "gated_tiger_pass": bool(gated_pass),
                    }
                )

        processed = min(chunk_start + len(chunk), len(samples))
        report_progress(
            {
                "phase": "humaneval_boundary",
                "processed": processed,
                "total": len(samples),
                "standard_pass_so_far": round(metrics["standard"]["pass"] / max(1, processed), 4),
                "ungated_tiger_pass_so_far": round(metrics["ungated_tiger"]["pass"] / max(1, processed), 4),
                "gated_tiger_pass_so_far": round(metrics["gated_tiger"]["pass"] / max(1, processed), 4),
            }
        )

    latency_sec = time.perf_counter() - start
    return {
        "task": "humaneval_boundary",
        "num_samples": len(samples),
        "standard": {
            "pass_at_1": summarize_ratio(metrics["standard"]["pass"], len(samples)),
            "syntax_failure_rate": summarize_ratio(metrics["standard"]["syntax"], len(samples)),
            "runtime_failure_rate": summarize_ratio(metrics["standard"]["runtime"], len(samples)),
        },
        "ungated_tiger": {
            "pass_at_1": summarize_ratio(metrics["ungated_tiger"]["pass"], len(samples)),
            "syntax_failure_rate": summarize_ratio(metrics["ungated_tiger"]["syntax"], len(samples)),
            "runtime_failure_rate": summarize_ratio(metrics["ungated_tiger"]["runtime"], len(samples)),
        },
        "gated_tiger": {
            "pass_at_1": summarize_ratio(metrics["gated_tiger"]["pass"], len(samples)),
            "syntax_failure_rate": summarize_ratio(metrics["gated_tiger"]["syntax"], len(samples)),
            "runtime_failure_rate": summarize_ratio(metrics["gated_tiger"]["runtime"], len(samples)),
        },
        "syntax_guard_avg_ms": round((parse_guard_sec / max(1, parse_guard_calls)) * 1000, 3),
        "latency_sec": round(latency_sec, 2),
        "per_sample": per_sample,
    }


def main() -> int:
    args = parse_args()
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    gate_threshold = read_gate_threshold()
    requested_batch_size = args.batch_size if args.batch_size > 0 else read_safe_batch_size()
    reasoning_batch_size = max(1, min(requested_batch_size, 16, args.samples))
    code_batch_size = max(1, min(max(1, requested_batch_size // 4), 8, args.humaneval_samples))

    engine = None
    try:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        engine = BatchedDLMInference(
            device="cuda",
            use_flash_attn=True,
            use_compile=False,
        )
        gsm8k_samples = load_samples(engine.tokenizer, args.samples)
        humaneval_samples = get_humaneval_samples(engine.tokenizer, num_samples=args.humaneval_samples)

        report_progress(
            {
                "phase": "loaded",
                "gsm8k_samples": len(gsm8k_samples),
                "humaneval_samples": len(humaneval_samples),
                "reasoning_batch_size": reasoning_batch_size,
                "code_batch_size": code_batch_size,
                "gate_threshold": gate_threshold,
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
            }
        )

        torch.cuda.reset_peak_memory_stats(engine.device)
        gsm8k_payload = run_gsm8k_reasoning(engine, gsm8k_samples, reasoning_batch_size, gate_threshold)
        humaneval_payload = run_humaneval_boundary(engine, humaneval_samples, code_batch_size)

        output = {
            "task_id": TASK_ID,
            "summary_name": "TIGER task gating 与 code 边界验证",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "gate_threshold": gate_threshold,
                "revision_fraction": REVISION_FRACTION,
                "revision_steps": REVISION_STEPS,
                "gsm8k_batch_size": reasoning_batch_size,
                "humaneval_batch_size": code_batch_size,
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "code_gen_length": CODE_GEN_LENGTH,
            },
            "gsm8k_reasoning": gsm8k_payload,
            "humaneval_boundary": humaneval_payload,
            "decision": {
                "gsm8k_delta_gated_vs_ungated": round(
                    gsm8k_payload["gated_tiger_accuracy"] - gsm8k_payload["ungated_tiger_accuracy"],
                    4,
                ),
                "humaneval_parse_improvement": round(
                    humaneval_payload["ungated_tiger"]["syntax_failure_rate"]
                    - humaneval_payload["gated_tiger"]["syntax_failure_rate"],
                    4,
                ),
                "humaneval_runtime_improvement": round(
                    humaneval_payload["ungated_tiger"]["runtime_failure_rate"]
                    - humaneval_payload["gated_tiger"]["runtime_failure_rate"],
                    4,
                ),
                "humaneval_pass_improvement": round(
                    humaneval_payload["gated_tiger"]["pass_at_1"]
                    - humaneval_payload["ungated_tiger"]["pass_at_1"],
                    4,
                ),
                "passes_gate": bool(
                    gsm8k_payload["gated_tiger_accuracy"] >= gsm8k_payload["ungated_tiger_accuracy"] - 0.01
                    and (
                        humaneval_payload["gated_tiger"]["syntax_failure_rate"]
                        <= humaneval_payload["ungated_tiger"]["syntax_failure_rate"] * 0.8
                        or humaneval_payload["gated_tiger"]["runtime_failure_rate"]
                        <= humaneval_payload["ungated_tiger"]["runtime_failure_rate"] * 0.8
                        or humaneval_payload["gated_tiger"]["pass_at_1"]
                        >= humaneval_payload["ungated_tiger"]["pass_at_1"] + 0.03
                    )
                ),
            },
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", output)
        write_json(
            RESULTS_DIR / f"{TASK_ID}_gpu_profile.json",
            {
                "task_id": TASK_ID,
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
            },
        )
        summary = (
            f"gsm8k gated={gsm8k_payload['gated_tiger_accuracy']:.4f} vs ungated={gsm8k_payload['ungated_tiger_accuracy']:.4f}; "
            f"humaneval gated pass@1={humaneval_payload['gated_tiger']['pass_at_1']:.4f}"
        )
        mark_done("success", summary)
        return 0
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
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
