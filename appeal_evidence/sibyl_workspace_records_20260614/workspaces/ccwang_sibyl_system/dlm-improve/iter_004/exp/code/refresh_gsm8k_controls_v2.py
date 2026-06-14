#!/usr/bin/env python3
"""Refresh the GSM8K shared control bundle under the iteration-4 runtime contract."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from pilot_revision_arm import DRAFT_STEPS, REVISION_FRACTION, REVISION_STEPS, run_dataset_group
from pilot_standard_arm import (
    ARMS_DIR,
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    SEED,
    SETUP_DIR,
    engine_attention_backend,
    engine_compile_enabled,
    load_json,
    manifest_rows,
    mark_done,
    now_iso,
    pick_safe_batch_size,
    report_progress,
    write_json,
)
from batched_dlm_utils import BatchedDLMInference


TASK_ID = os.environ.get("SIBYL_TASK_ID", "refresh_gsm8k_controls_v2")
RESULT_FILE = os.environ.get("SIBYL_RESULT_FILE", "gsm8k_controls_refresh_v2.json")
DATASET_NAME = "gsm8k"
ARM_CONFIGS: tuple[tuple[str, str], ...] = (
    ("card84", "entropy"),
    ("rand84", "random"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def build_engine(
    runtime_contract: dict[str, Any],
    model_path: str,
) -> tuple[BatchedDLMInference, dict[str, Any]]:
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
    except Exception as exc:  # noqa: BLE001
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


def load_scores(arm_name: str) -> list[dict[str, Any]]:
    path = ARMS_DIR / arm_name / f"seed{SEED}" / "per_sample_scores.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == DATASET_NAME:
                rows.append(row)
    return rows


def index_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in rows}


def compare_scores(
    primary_scores: list[dict[str, Any]],
    reference_map: dict[str, dict[str, Any]],
) -> dict[str, int]:
    summary = {
        "fixed": 0,
        "harmed": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "missing_reference": 0,
    }
    for row in primary_scores:
        reference = reference_map.get(row["sample_id"])
        if reference is None:
            summary["missing_reference"] += 1
            continue
        primary_correct = bool(row.get("correct"))
        reference_correct = bool(reference.get("correct"))
        if primary_correct and not reference_correct:
            summary["fixed"] += 1
        elif not primary_correct and reference_correct:
            summary["harmed"] += 1
        elif primary_correct and reference_correct:
            summary["unchanged_correct"] += 1
        else:
            summary["unchanged_wrong"] += 1
    summary["net_repaired"] = summary["fixed"] - summary["harmed"]
    return summary


def top_examples(
    primary_scores: list[dict[str, Any]],
    reference_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in ("fixed", "harmed"):
        count = 0
        for row in primary_scores:
            reference = reference_map.get(row["sample_id"])
            if reference is None:
                continue
            primary_correct = bool(row.get("correct"))
            reference_correct = bool(reference.get("correct"))
            comparison = ""
            if primary_correct and not reference_correct:
                comparison = "fixed"
            elif not primary_correct and reference_correct:
                comparison = "harmed"
            if comparison != label:
                continue
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "comparison": comparison,
                    "difficulty_bucket": row.get("difficulty_bucket"),
                    "predicted_answer": row.get("predicted_answer", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "tokens_changed": int(row.get("tokens_changed", 0)),
                    "nfe": int(row.get("nfe", 0)),
                }
            )
            count += 1
            if count >= 3:
                break
    return examples


def score_runtime_summary(
    rows: list[dict[str, Any]],
    latency_sec: float,
) -> dict[str, Any]:
    total_generated_tokens = sum(int(row.get("generated_length", 0)) for row in rows)
    total_nfe = sum(int(row.get("nfe", 0)) for row in rows)
    return {
        "sample_count": len(rows),
        "avg_nfe": round(total_nfe / max(1, len(rows)), 2),
        "generated_tokens": total_generated_tokens,
        "tokens_per_sec": round(total_generated_tokens / max(latency_sec, 1e-6), 2),
    }


def clear_arm_outputs(arm_name: str) -> tuple[Path, Path, Path]:
    arm_dir = ARMS_DIR / arm_name / f"seed{SEED}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = arm_dir / "predictions.jsonl"
    scores_path = arm_dir / "per_sample_scores.jsonl"
    revision_trace_path = arm_dir / "revision_trace.jsonl"
    for path in (predictions_path, scores_path, revision_trace_path):
        if path.exists():
            path.unlink()
    return predictions_path, scores_path, revision_trace_path


def write_gpu_profile(
    engine: BatchedDLMInference,
    runtime_meta: dict[str, Any],
    batch_size: int,
    manifest_count: int,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(engine.device)
    profile = {
        "task_id": TASK_ID,
        "dataset": DATASET_NAME,
        "gpu_name": props.name,
        "vram_total_mb": round(props.total_memory / 1024**2),
        "attention_backend": engine_attention_backend(engine),
        "compile_enabled": engine_compile_enabled(engine),
        "requested_compile": runtime_meta["requested_compile"],
        "requested_flash_attention": runtime_meta["requested_flash_attention"],
        "compile_error": runtime_meta["compile_error"],
        "effective_batch_size": batch_size,
        "sample_count": manifest_count,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
        "timestamp": now_iso(),
    }
    write_json(RESULTS_DIR / f"{TASK_ID}_gpu_profile.json", profile)
    return profile


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    try:
        manifest_payload = load_json(SETUP_DIR / "sample_manifest.json")
        runtime_contract = load_json(SETUP_DIR / "runtime_contract.json")
        batch_probe = load_json(SETUP_DIR / "batch_probe.json")
        manifest = [row for row in manifest_rows(manifest_payload) if row["dataset"] == DATASET_NAME]
        if not manifest:
            raise ValueError(f"no {DATASET_NAME} rows found in sample_manifest")

        report_progress(
            TASK_ID,
            {
                "phase": "load_engine",
                "dataset": DATASET_NAME,
                "samples": len(manifest),
                "draft_steps": DRAFT_STEPS,
                "revision_steps": REVISION_STEPS,
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        engine, runtime_meta = build_engine(runtime_contract, args.model_path)
        inherited_batch_size = pick_safe_batch_size(batch_probe)
        contract_batch = int(runtime_contract.get("runtime", {}).get("safe_batch_size", inherited_batch_size))
        preferred_batch_size = contract_batch if contract_batch > 0 else inherited_batch_size
        batch_size = max(1, min(len(manifest), preferred_batch_size))

        arm_metrics: dict[str, dict[str, Any]] = {}
        arm_scores: dict[str, list[dict[str, Any]]] = {}
        for idx, (arm_name, mode) in enumerate(ARM_CONFIGS, start=1):
            report_progress(
                TASK_ID,
                {
                    "phase": f"run_{arm_name}",
                    "mode": mode,
                    "batch_size": batch_size,
                    "arm_index": idx,
                    "arm_total": len(ARM_CONFIGS),
                },
            )
            predictions_path, scores_path, revision_trace_path = clear_arm_outputs(arm_name)
            metrics = run_dataset_group(
                engine=engine,
                tokenizer=tokenizer,
                task_id=TASK_ID,
                arm_name=arm_name,
                mode=mode,
                rows=manifest,
                batch_size=batch_size,
                predictions_path=predictions_path,
                scores_path=scores_path,
                revision_trace_path=revision_trace_path,
            )
            scores = load_scores(arm_name)
            arm_scores[arm_name] = scores
            arm_metrics[arm_name] = {
                **metrics,
                "mode": mode,
                **score_runtime_summary(scores, float(metrics.get("latency_sec", 0.0) or 0.0)),
            }

        card_scores = arm_scores["card84"]
        rand_scores = arm_scores["rand84"]
        card_vs_rand = compare_scores(card_scores, index_by_sample(rand_scores))
        rand_vs_card = compare_scores(rand_scores, index_by_sample(card_scores))
        gpu_profile = write_gpu_profile(engine, runtime_meta, batch_size, len(manifest))

        payload = {
            "task_id": TASK_ID,
            "status": "success",
            "dataset": DATASET_NAME,
            "checked_at": now_iso(),
            "sample_count": len(manifest),
            "summary": (
                "Refreshed GSM8K shared controls under runtime contract v2: "
                f"CARD-84 acc={arm_metrics['card84']['accuracy']}, "
                f"RAND-84 acc={arm_metrics['rand84']['accuracy']}, "
                f"CARD-84 net_repaired_vs_rand={card_vs_rand['net_repaired']}."
            ),
            "runtime_contract_path": str(SETUP_DIR / "runtime_contract.json"),
            "sample_manifest_path": str(SETUP_DIR / "sample_manifest.json"),
            "batch_probe_path": str(SETUP_DIR / "batch_probe.json"),
            "runtime": {
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "requested_compile": runtime_meta["requested_compile"],
                "requested_flash_attention": runtime_meta["requested_flash_attention"],
                "compile_error": runtime_meta["compile_error"],
                "effective_batch_size": batch_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "peak_vram_mb": gpu_profile["peak_vram_mb"],
            },
            "arms": arm_metrics,
            "comparisons": {
                "card84_vs_rand84": card_vs_rand,
                "rand84_vs_card84": rand_vs_card,
            },
            "shared_controls_table": [
                {
                    "method": "CARD-84",
                    "score": arm_metrics["card84"]["accuracy"],
                    "repair": card_vs_rand["fixed"],
                    "harm": card_vs_rand["harmed"],
                    "wall_clock_sec": arm_metrics["card84"]["latency_sec"],
                },
                {
                    "method": "RAND-84",
                    "score": arm_metrics["rand84"]["accuracy"],
                    "repair": rand_vs_card["fixed"],
                    "harm": rand_vs_card["harmed"],
                    "wall_clock_sec": arm_metrics["rand84"]["latency_sec"],
                },
            ],
            "evidence_examples": {
                "card84_vs_rand84": top_examples(card_scores, index_by_sample(rand_scores)),
                "rand84_vs_card84": top_examples(rand_scores, index_by_sample(card_scores)),
            },
            "contract": {
                "draft_steps": DRAFT_STEPS,
                "revision_fraction": REVISION_FRACTION,
                "revision_steps": REVISION_STEPS,
                "card84_mode": "entropy",
                "rand84_mode": "random",
            },
        }

        write_json(RESULTS_DIR / RESULT_FILE, payload)
        report_progress(
            TASK_ID,
            {
                "phase": "done",
                "card84_accuracy": arm_metrics["card84"]["accuracy"],
                "rand84_accuracy": arm_metrics["rand84"]["accuracy"],
                "card84_net_repaired_vs_rand": card_vs_rand["net_repaired"],
                "batch_size": batch_size,
            },
        )
        mark_done(TASK_ID, "success", payload["summary"])
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": TASK_ID,
            "status": "failed",
            "error": repr(exc),
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / RESULT_FILE, failure)
        report_progress(TASK_ID, {"phase": "failed", "error": repr(exc)})
        mark_done(TASK_ID, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
