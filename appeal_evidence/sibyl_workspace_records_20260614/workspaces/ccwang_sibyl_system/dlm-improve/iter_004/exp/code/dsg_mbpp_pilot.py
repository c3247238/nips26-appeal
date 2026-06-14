#!/usr/bin/env python3
"""Run the iteration-4 DSG MBPP pilot with task-specific telemetry."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from pilot_revision_arm import DRAFT_STEPS, REVISION_FRACTION, REVISION_STEPS
from pilot_standard_arm import (
    ARMS_DIR,
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    SEED,
    SETUP_DIR,
    aggregate_metrics,
    append_jsonl,
    check_mbpp_correct,
    dataset_gen_length,
    engine_attention_backend,
    engine_compile_enabled,
    extract_code_from_generation,
    load_json,
    manifest_rows,
    mark_done,
    now_iso,
    pick_safe_batch_size,
    report_progress,
    write_json,
)
from batched_dlm_utils import BatchedDLMInference


TASK_ID = "dsg_mbpp_pilot"
ARM_NAME = "dsg84"
DATASET_NAME = "mbpp"
BASELINE_ARMS = ("card84", "rand84")
GEN_LENGTH = dataset_gen_length(DATASET_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--arm-name", default=ARM_NAME)
    parser.add_argument("--mode", default="entropy", choices=("entropy",))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--probe-hi", type=int, default=192)
    parser.add_argument("--probe-margin", type=float, default=0.95)
    return parser.parse_args()


def build_engine(runtime_contract: dict[str, Any], model_path: str) -> tuple[BatchedDLMInference, dict[str, Any]]:
    runtime_meta = runtime_contract.get("runtime", {})
    requested_compile = bool(runtime_meta.get("torch_compile_enabled", False))
    requested_flash = bool(runtime_meta.get("flash_attention_available", False))
    compile_error: str | None = None

    try:
        engine = BatchedDLMInference(
            model_path=model_path,
            device="cuda",
            use_flash_attn=requested_flash,
            use_compile=requested_compile,
        )
    except Exception as exc:
        if not requested_compile:
            raise
        compile_error = repr(exc)
        torch.cuda.empty_cache()
        gc.collect()
        engine = BatchedDLMInference(
            model_path=model_path,
            device="cuda",
            use_flash_attn=requested_flash,
            use_compile=False,
        )

    return engine, {
        "requested_compile": requested_compile,
        "requested_flash_attention": requested_flash,
        "compile_error": compile_error,
    }


def load_baseline_scores(arm_name: str) -> dict[str, dict[str, Any]]:
    path = ARMS_DIR / arm_name / f"seed{SEED}" / "per_sample_scores.jsonl"
    scores: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == DATASET_NAME:
                scores[row["sample_id"]] = row
    return scores


def compare_against_baseline(
    candidate_scores: list[dict[str, Any]],
    baseline_scores: dict[str, dict[str, Any]],
) -> dict[str, int]:
    summary = {
        "fixed": 0,
        "harmed": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "missing_baseline": 0,
    }
    for row in candidate_scores:
        baseline = baseline_scores.get(row["sample_id"])
        if baseline is None:
            summary["missing_baseline"] += 1
            continue
        cand_correct = bool(row["correct"])
        base_correct = bool(baseline.get("correct"))
        if cand_correct and not base_correct:
            summary["fixed"] += 1
        elif not cand_correct and base_correct:
            summary["harmed"] += 1
        elif cand_correct and base_correct:
            summary["unchanged_correct"] += 1
        else:
            summary["unchanged_wrong"] += 1
    summary["net_repaired"] = summary["fixed"] - summary["harmed"]
    return summary


def qualitative_examples(
    candidate_scores: list[dict[str, Any]],
    baseline_maps: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    priority = [
        ("correct", True),
        ("correct", False),
    ]
    for field, target in priority:
        for row in candidate_scores:
            if bool(row.get(field)) != target:
                continue
            entry = {
                "sample_id": row["sample_id"],
                "difficulty_bucket": row.get("difficulty_bucket"),
                "correct": bool(row["correct"]),
                "tokens_changed": int(row.get("tokens_changed", 0)),
                "nfe": int(row.get("nfe", 0)),
                "entropy_stats": row.get("entropy_stats", {}),
                "card84_correct": bool(
                    baseline_maps["card84"].get(row["sample_id"], {}).get("correct", False)
                ),
                "rand84_correct": bool(
                    baseline_maps["rand84"].get(row["sample_id"], {}).get("correct", False)
                ),
                "error": row.get("error"),
            }
            examples.append(entry)
            if len(examples) >= 6:
                return examples
    return examples


def write_gpu_profile(
    task_id: str,
    engine: BatchedDLMInference,
    prompt_max_len: int,
    inherited_batch_size: int,
    probed_batch_size: int,
    effective_batch_size: int,
    runtime_meta: dict[str, Any],
    oom_retries: int,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(engine.device)
    profile = {
        "task_id": task_id,
        "dataset": DATASET_NAME,
        "gpu_name": props.name,
        "vram_total_mb": round(props.total_memory / 1024**2),
        "attention_backend": engine_attention_backend(engine),
        "compile_enabled": engine_compile_enabled(engine),
        "requested_compile": runtime_meta["requested_compile"],
        "requested_flash_attention": runtime_meta["requested_flash_attention"],
        "compile_error": runtime_meta["compile_error"],
        "prompt_max_len": prompt_max_len,
        "gen_length": GEN_LENGTH,
        "inherited_safe_batch_size": inherited_batch_size,
        "probed_safe_batch_size": probed_batch_size,
        "effective_batch_size": effective_batch_size,
        "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
        "oom_retries": oom_retries,
        "timestamp": now_iso(),
    }
    write_json(RESULTS_DIR / f"{task_id}_gpu_profile.json", profile)
    return profile


def main() -> int:
    args = parse_args()
    task_id = args.task_id
    arm_name = args.arm_name
    arm_dir = ARMS_DIR / arm_name / f"seed{SEED}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(__import__("os").getpid()), encoding="utf-8")

    predictions_path = arm_dir / "predictions.jsonl"
    scores_path = arm_dir / "per_sample_scores.jsonl"
    revision_trace_path = arm_dir / "revision_trace.jsonl"
    for path in (predictions_path, scores_path, revision_trace_path):
        if path.exists():
            path.unlink()

    started_at = time.time()
    try:
        manifest_payload = load_json(SETUP_DIR / "sample_manifest.json")
        runtime_contract = load_json(SETUP_DIR / "runtime_contract.json")
        batch_probe = load_json(SETUP_DIR / "batch_probe.json")
        manifest = [row for row in manifest_rows(manifest_payload) if row["dataset"] == DATASET_NAME]
        if not manifest:
            raise ValueError(f"no {DATASET_NAME} rows found in manifest")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        prompt_ids_list = [
            tokenizer.encode(row["prompt"], add_special_tokens=False)
            for row in manifest
        ]
        prompt_max_len = max(len(ids) for ids in prompt_ids_list)
        inherited_batch_size = pick_safe_batch_size(batch_probe)

        report_progress(
            task_id,
            {
                "phase": "load_engine",
                "dataset": DATASET_NAME,
                "mode": args.mode,
                "samples": len(manifest),
                "inherited_safe_batch_size": inherited_batch_size,
            },
        )
        engine, runtime_meta = build_engine(runtime_contract, args.model_path)

        report_progress(
            task_id,
            {
                "phase": "probe_batch_size",
                "dataset": DATASET_NAME,
                "prompt_max_len": prompt_max_len,
                "gen_length": GEN_LENGTH,
                "probe_hi": args.probe_hi,
                "probe_margin": args.probe_margin,
            },
        )
        probed_batch_size = engine.find_max_batch_size(
            gen_length=GEN_LENGTH,
            prompt_max_len=prompt_max_len,
            lo=1,
            hi=max(args.probe_hi, inherited_batch_size),
            safety_margin=args.probe_margin,
        )
        effective_batch_size = max(1, min(len(manifest), probed_batch_size))
        torch.cuda.reset_peak_memory_stats(engine.device)
        gc.collect()
        torch.cuda.empty_cache()

        candidate_scores: list[dict[str, Any]] = []
        total_processed = 0
        oom_retries = 0
        loop_started_at = time.time()
        current_batch_size = effective_batch_size

        chunk_start = 0
        while chunk_start < len(manifest):
            chunk_end = min(chunk_start + current_batch_size, len(manifest))
            chunk_rows = manifest[chunk_start:chunk_end]
            chunk_prompt_ids = prompt_ids_list[chunk_start:chunk_end]
            report_progress(
                task_id,
                {
                    "phase": "mbpp_running",
                    "processed": total_processed,
                    "dataset_total": len(manifest),
                    "batch_size": current_batch_size,
                    "attention_backend": engine_attention_backend(engine),
                    "compile_enabled": engine_compile_enabled(engine),
                    "draft_steps": DRAFT_STEPS,
                    "revision_steps": REVISION_STEPS,
                },
            )
            try:
                outputs = engine.batched_entropy_revise(
                    prompt_ids_list=chunk_prompt_ids,
                    gen_length=GEN_LENGTH,
                    num_draft_steps=DRAFT_STEPS,
                    revision_fraction=REVISION_FRACTION,
                    revision_steps=REVISION_STEPS,
                    batch_size=len(chunk_prompt_ids),
                    seed=SEED,
                )
            except torch.cuda.OutOfMemoryError:
                oom_retries += 1
                torch.cuda.empty_cache()
                gc.collect()
                if current_batch_size <= 1:
                    raise
                current_batch_size = max(1, current_batch_size - max(1, current_batch_size // 8))
                continue

            generated_texts = engine.decode_results(outputs)
            prediction_rows: list[dict[str, Any]] = []
            score_rows: list[dict[str, Any]] = []
            revision_rows: list[dict[str, Any]] = []
            for sample, result, text in zip(
                chunk_rows, outputs, generated_texts, strict=True
            ):
                generated_text = text[:4000]
                extracted_code = extract_code_from_generation(generated_text)
                passed, error = check_mbpp_correct(extracted_code, sample.get("test_list", []))
                entropy_stats = result.get("entropy_stats", {}) or {}
                score_row = {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "source_index": sample["source_index"],
                    "prompt_length": sample["prompt_length"],
                    "difficulty_bucket": sample["difficulty_bucket"],
                    "nfe": int(result["nfe"]),
                    "tokens_changed": int(result.get("tokens_changed", 0)),
                    "generated_length": int(result["gen_end"] - result["gen_start"]),
                    "correct": bool(passed),
                    "error": error,
                    "reference_code": sample["reference"],
                    "entropy_stats": entropy_stats,
                }
                prediction_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "generated_text": generated_text,
                        "extracted_code": extracted_code[:2000],
                    }
                )
                revision_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "arm_name": arm_name,
                        "method": "dsg",
                        "mode": args.mode,
                        "draft_steps": DRAFT_STEPS,
                        "revision_fraction": REVISION_FRACTION,
                        "revision_steps": REVISION_STEPS,
                        "nfe": int(result["nfe"]),
                        "tokens_changed": int(result.get("tokens_changed", 0)),
                        "entropy_stats": entropy_stats,
                    }
                )
                score_rows.append(score_row)

            append_jsonl(predictions_path, prediction_rows)
            append_jsonl(scores_path, score_rows)
            append_jsonl(revision_trace_path, revision_rows)
            candidate_scores.extend(score_rows)
            total_processed += len(chunk_rows)
            chunk_start = chunk_end
            gc.collect()
            torch.cuda.empty_cache()

        dataset_metrics = aggregate_metrics(candidate_scores)
        dataset_metrics.update(
            {
                "dataset": DATASET_NAME,
                "latency_sec": round(time.time() - loop_started_at, 2),
                "batch_size": current_batch_size,
                "gen_length": GEN_LENGTH,
                "sample_count": len(candidate_scores),
                "draft_steps": DRAFT_STEPS,
                "revision_steps": REVISION_STEPS,
                "revision_fraction": REVISION_FRACTION,
                "mode": args.mode,
            }
        )

        baseline_maps = {arm: load_baseline_scores(arm) for arm in BASELINE_ARMS}
        comparisons = {
            f"dsg_vs_{arm}": compare_against_baseline(candidate_scores, baseline_maps[arm])
            for arm in BASELINE_ARMS
        }
        gpu_profile = write_gpu_profile(
            task_id=task_id,
            engine=engine,
            prompt_max_len=prompt_max_len,
            inherited_batch_size=inherited_batch_size,
            probed_batch_size=probed_batch_size,
            effective_batch_size=current_batch_size,
            runtime_meta=runtime_meta,
            oom_retries=oom_retries,
        )

        metrics_payload = {
            "task_id": task_id,
            "candidate_id": "cand_dsg",
            "arm_name": arm_name,
            "dataset": DATASET_NAME,
            "status": "success",
            "seed": SEED,
            "sample_count": len(candidate_scores),
            "overall": dataset_metrics,
            "by_dataset": {DATASET_NAME: dataset_metrics},
            "runtime": {
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                "batch_size": current_batch_size,
                "probed_safe_batch_size": probed_batch_size,
                "inherited_safe_batch_size": inherited_batch_size,
            },
            "dsg_config": {
                "mode": args.mode,
                "draft_steps": DRAFT_STEPS,
                "revision_fraction": REVISION_FRACTION,
                "revision_steps": REVISION_STEPS,
            },
            "comparisons": comparisons,
            "qualitative_examples": qualitative_examples(candidate_scores, baseline_maps),
            "gpu_profile_path": str((RESULTS_DIR / f"{task_id}_gpu_profile.json")),
            "elapsed_min": round((time.time() - started_at) / 60, 2),
            "timestamp": now_iso(),
        }

        write_json(arm_dir / "metrics.json", metrics_payload)
        write_json(
            arm_dir / "gpu_progress.json",
            {
                "task_id": task_id,
                "arm_name": arm_name,
                "mode": args.mode,
                "seed": SEED,
                "batch_size": current_batch_size,
                "probed_safe_batch_size": probed_batch_size,
                "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                "updated_at": now_iso(),
            },
        )
        write_json(RESULTS_DIR / f"{task_id}.json", metrics_payload)

        summary = (
            f"{arm_name} {DATASET_NAME}: acc={dataset_metrics['accuracy']} "
            f"vs_card84_net={comparisons['dsg_vs_card84']['net_repaired']} "
            f"vs_rand84_net={comparisons['dsg_vs_rand84']['net_repaired']} "
            f"batch={current_batch_size} prompt_max={prompt_max_len}"
        )
        report_progress(
            task_id,
            {
                "phase": "done",
                "summary": summary,
                "batch_size": current_batch_size,
                "probed_safe_batch_size": probed_batch_size,
                "peak_vram_mb": gpu_profile["peak_vram_mb"],
            },
        )
        mark_done(task_id, "success", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": task_id,
            "candidate_id": "cand_dsg",
            "arm_name": arm_name,
            "dataset": DATASET_NAME,
            "status": "failed",
            "error": repr(exc),
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{task_id}.json", failure)
        report_progress(task_id, {"phase": "failed", "error": repr(exc), "dataset": DATASET_NAME})
        mark_done(task_id, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
