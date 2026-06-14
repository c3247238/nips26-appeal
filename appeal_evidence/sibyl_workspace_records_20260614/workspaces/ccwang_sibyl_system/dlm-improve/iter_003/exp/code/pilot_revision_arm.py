#!/usr/bin/env python3
"""Run a revision-based pilot arm from the fixed iteration-3 manifest."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from pilot_standard_arm import (
    ARMS_DIR,
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    SEED,
    SETUP_DIR,
    aggregate_metrics,
    append_jsonl,
    check_gsm8k_correct,
    check_mbpp_correct,
    dataset_gen_length,
    engine_attention_backend,
    engine_compile_enabled,
    extract_code_from_generation,
    extract_gsm8k_answer,
    load_json,
    manifest_rows,
    mark_done,
    now_iso,
    pick_safe_batch_size,
    report_progress,
    write_json,
)
from batched_dlm_utils import BatchedDLMInference


DRAFT_STEPS = 64
REVISION_FRACTION = 0.10
REVISION_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--mode", choices=("entropy", "random"), required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def run_dataset_group(
    engine: BatchedDLMInference,
    tokenizer: Any,
    task_id: str,
    arm_name: str,
    mode: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    predictions_path: Path,
    scores_path: Path,
    revision_trace_path: Path,
) -> dict[str, Any]:
    dataset_name = rows[0]["dataset"]
    gen_length = dataset_gen_length(dataset_name)
    scores: list[dict[str, Any]] = []
    started = __import__("time").time()
    processed = 0

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompt_ids = [
            tokenizer.encode(sample["prompt"], add_special_tokens=False)
            for sample in chunk
        ]
        if mode == "entropy":
            outputs = engine.batched_entropy_revise(
                prompt_ids_list=prompt_ids,
                gen_length=gen_length,
                num_draft_steps=DRAFT_STEPS,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=len(chunk),
                seed=SEED,
            )
        else:
            outputs = engine.batched_random_revise(
                prompt_ids_list=prompt_ids,
                gen_length=gen_length,
                num_draft_steps=DRAFT_STEPS,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=len(chunk),
                seed=SEED,
            )
        texts = engine.decode_results(outputs)

        chunk_predictions: list[dict[str, Any]] = []
        chunk_scores: list[dict[str, Any]] = []
        chunk_revision_rows: list[dict[str, Any]] = []
        for sample, result, text in zip(chunk, outputs, texts, strict=True):
            generated_text = text[:4000]
            entropy_stats = result.get("entropy_stats", {}) or {}
            base_score = {
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "source_index": sample["source_index"],
                "prompt_length": sample["prompt_length"],
                "difficulty_bucket": sample["difficulty_bucket"],
                "nfe": int(result["nfe"]),
                "tokens_changed": int(result.get("tokens_changed", 0)),
                "generated_length": int(result["gen_end"] - result["gen_start"]),
            }
            chunk_predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "generated_text": generated_text,
                }
            )

            if dataset_name == "gsm8k":
                predicted_answer = extract_gsm8k_answer(generated_text)
                correct = check_gsm8k_correct(generated_text, sample["reference"])
                base_score.update(
                    {
                        "correct": bool(correct),
                        "predicted_answer": predicted_answer,
                        "reference_answer": sample["reference"],
                    }
                )
            else:
                extracted_code = extract_code_from_generation(generated_text)
                passed, error = check_mbpp_correct(
                    extracted_code,
                    sample.get("test_list", []),
                )
                base_score.update(
                    {
                        "correct": bool(passed),
                        "error": error,
                        "reference_code": sample["reference"],
                    }
                )
            chunk_scores.append(base_score)
            chunk_revision_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "arm_name": arm_name,
                    "mode": mode,
                    "draft_steps": DRAFT_STEPS,
                    "revision_fraction": REVISION_FRACTION,
                    "revision_steps": REVISION_STEPS,
                    "nfe": int(result["nfe"]),
                    "tokens_changed": int(result.get("tokens_changed", 0)),
                    "entropy_stats": entropy_stats,
                }
            )

        append_jsonl(predictions_path, chunk_predictions)
        append_jsonl(scores_path, chunk_scores)
        append_jsonl(revision_trace_path, chunk_revision_rows)
        scores.extend(chunk_scores)
        processed += len(chunk)
        report_progress(
            task_id,
            {
                "phase": f"{dataset_name}_running",
                "processed": processed,
                "dataset_total": len(rows),
                "batch_size": batch_size,
                "draft_steps": DRAFT_STEPS,
                "revision_steps": REVISION_STEPS,
                "mode": mode,
            },
        )
        gc.collect()

    metrics = aggregate_metrics(scores)
    metrics.update(
        {
            "dataset": dataset_name,
            "gen_length": gen_length,
            "latency_sec": round(__import__("time").time() - started, 2),
            "batch_size": batch_size,
            "draft_steps": DRAFT_STEPS,
            "revision_fraction": REVISION_FRACTION,
            "revision_steps": REVISION_STEPS,
            "mode": mode,
        }
    )
    return metrics


def main() -> int:
    args = parse_args()
    task_id = args.task_id
    arm_name = args.arm_name
    mode = args.mode

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arm_dir = ARMS_DIR / arm_name / f"seed{SEED}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(__import__("os").getpid()), encoding="utf-8")

    predictions_path = arm_dir / "predictions.jsonl"
    scores_path = arm_dir / "per_sample_scores.jsonl"
    revision_trace_path = arm_dir / "revision_trace.jsonl"
    for path in (predictions_path, scores_path, revision_trace_path):
        if path.exists():
            path.unlink()

    try:
        manifest_payload = load_json(SETUP_DIR / "sample_manifest.json")
        manifest = manifest_rows(manifest_payload)
        runtime_contract = load_json(SETUP_DIR / "runtime_contract.json")
        batch_probe = load_json(SETUP_DIR / "batch_probe.json")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        report_progress(task_id, {"phase": "load_engine", "mode": mode})
        engine = BatchedDLMInference(
            model_path=args.model_path,
            device="cuda",
            use_flash_attn=runtime_contract.get("flash_attention_2_available", False),
            use_compile=False,
        )
        batch_size = pick_safe_batch_size(batch_probe)
        dataset_metrics: dict[str, Any] = {}
        all_scores: list[dict[str, Any]] = []

        for dataset_name in ("gsm8k", "mbpp"):
            rows = [row for row in manifest if row["dataset"] == dataset_name]
            if not rows:
                continue
            metrics = run_dataset_group(
                engine=engine,
                tokenizer=tokenizer,
                task_id=task_id,
                arm_name=arm_name,
                mode=mode,
                rows=rows,
                batch_size=batch_size,
                predictions_path=predictions_path,
                scores_path=scores_path,
                revision_trace_path=revision_trace_path,
            )
            dataset_metrics[dataset_name] = metrics

        with scores_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    all_scores.append(json.loads(line))

        overall_metrics = aggregate_metrics(all_scores)
        metrics_payload = {
            "task_id": task_id,
            "arm_name": arm_name,
            "mode": mode,
            "draft_steps": DRAFT_STEPS,
            "revision_fraction": REVISION_FRACTION,
            "revision_steps": REVISION_STEPS,
            "seed": SEED,
            "sample_count": len(manifest),
            "overall": overall_metrics,
            "by_dataset": dataset_metrics,
            "runtime": {
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "batch_size": batch_size,
                "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        }
        write_json(arm_dir / "metrics.json", metrics_payload)
        write_json(
            arm_dir / "gpu_progress.json",
            {
                "task_id": task_id,
                "arm_name": arm_name,
                "mode": mode,
                "seed": SEED,
                "batch_size": batch_size,
                "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                "updated_at": now_iso(),
            },
        )
        write_json(RESULTS_DIR / f"{task_id}.json", metrics_payload)

        summary = (
            f"{arm_name}: overall_acc={overall_metrics['accuracy']} "
            f"gsm8k={dataset_metrics.get('gsm8k', {}).get('accuracy')} "
            f"mbpp={dataset_metrics.get('mbpp', {}).get('accuracy')} "
            f"batch={batch_size} draft={DRAFT_STEPS} rev={REVISION_STEPS}"
        )
        report_progress(task_id, {"phase": "done", "summary": summary})
        mark_done(task_id, "success", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": task_id,
            "arm_name": arm_name,
            "mode": mode,
            "status": "failed",
            "error": repr(exc),
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{task_id}.json", failure)
        report_progress(task_id, {"phase": "failed", "error": repr(exc), "mode": mode})
        mark_done(task_id, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
