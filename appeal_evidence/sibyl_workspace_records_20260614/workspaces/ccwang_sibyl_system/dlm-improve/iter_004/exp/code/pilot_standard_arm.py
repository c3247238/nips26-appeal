#!/usr/bin/env python3
"""Run a single standard-denoising pilot arm from the fixed iteration-3 manifest."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(
    os.environ.get("SIBYL_REMOTE_PROJECT_ROOT", "/home/ccwang/sibyl_system/projects/dlm-improve")
)
RESULTS_DIR = Path(os.environ.get("SIBYL_RESULTS_DIR", str(PROJECT_ROOT / "exp" / "results")))
SETUP_DIR = Path(
    os.environ.get(
        "SIBYL_SETUP_DIR",
        str(PROJECT_ROOT / "current" / "exp" / "pilot_evidence_closure_v1" / "setup"),
    )
)
ARMS_DIR = Path(
    os.environ.get(
        "SIBYL_ARMS_DIR",
        str(PROJECT_ROOT / "current" / "exp" / "pilot_evidence_closure_v1" / "arms"),
    )
)
SEED = 42
DEFAULT_MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
DEFAULT_GSM8K_GEN_LENGTH = 256
DEFAULT_MBPP_GEN_LENGTH = 512
MBPP_TIMEOUT = 10

sys.path.insert(0, str(PROJECT_ROOT / "exp" / "code"))

from batched_dlm_utils import BatchedDLMInference  # noqa: E402


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_progress(task_id: str, metric: dict[str, Any]) -> None:
    write_json(
        RESULTS_DIR / f"{task_id}_PROGRESS.json",
        {
            "task_id": task_id,
            "epoch": 0,
            "total_epochs": 1,
            "step": 0,
            "total_steps": 1,
            "loss": None,
            "metric": metric,
            "updated_at": now_iso(),
        },
    )


def mark_done(task_id: str, status: str, summary: str) -> None:
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            final_progress = {}
    write_json(
        RESULTS_DIR / f"{task_id}_DONE",
        {
            "task_id": task_id,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": now_iso(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def engine_attention_backend(engine: BatchedDLMInference) -> str:
    model = getattr(engine, "model", None)
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    return getattr(config, "_attn_implementation", None) or "eager_or_default"


def engine_compile_enabled(engine: BatchedDLMInference) -> bool:
    model = getattr(engine, "model", None)
    if hasattr(model, "_orig_mod"):
        return True
    if hasattr(model, "module") and hasattr(model.module, "_orig_mod"):
        return True
    return False


def decode_generation(tokenizer: Any, result: dict[str, Any]) -> str:
    gen_ids = result["input_ids"][result["gen_start"]:result["gen_end"]].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


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


def check_gsm8k_correct(prediction: str, gold: str) -> bool:
    gold_value = gold.split("####")[-1].strip() if "####" in gold else gold
    return normalize_answer(extract_gsm8k_answer(prediction)) == normalize_answer(gold_value)


def extract_code_from_generation(generated_text: str) -> str:
    text = generated_text.replace("<|endoftext|>", "").replace("<|end|>", "").replace("[MASK]", "")
    block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if block:
        return block.group(1).strip()

    lines = text.split("\n")
    code_lines: list[str] = []
    in_function = False
    function_indent = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            in_function = True
            function_indent = len(line) - len(line.lstrip())
            code_lines.append(line)
            continue
        if in_function:
            if stripped == "":
                code_lines.append("")
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent > function_indent:
                code_lines.append(line)
            elif stripped.startswith("def "):
                code_lines.append(line)
                function_indent = current_indent
            elif stripped.startswith("import ") or stripped.startswith("from "):
                code_lines.insert(0, line)
            else:
                break
    return "\n".join(code_lines).strip() if code_lines else text.strip()


def _run_test_in_process(code_str: str, result_dict: Any) -> None:
    try:
        exec_globals: dict[str, Any] = {}
        exec(code_str, exec_globals)
        result_dict["passed"] = True
        result_dict["error"] = None
    except Exception as exc:  # noqa: BLE001
        result_dict["passed"] = False
        result_dict["error"] = f"{type(exc).__name__}: {exc}"


def check_mbpp_correct(generated_code: str, test_list: list[str], timeout: int = MBPP_TIMEOUT) -> tuple[bool, str | None]:
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    result_dict["passed"] = False
    result_dict["error"] = None
    full_code = f"{generated_code}\n\n" + "\n".join(test_list) + "\n"
    proc = multiprocessing.Process(target=_run_test_in_process, args=(full_code, result_dict))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, "TimeoutError: execution exceeded time limit"
    return bool(result_dict.get("passed", False)), result_dict.get("error")


def pick_safe_batch_size(batch_probe: dict[str, Any]) -> int:
    safe_sizes = [
        bucket["safe_batch_size"]
        for bucket in batch_probe.get("buckets", {}).values()
        if bucket.get("safe_batch_size")
    ]
    return max(1, min(safe_sizes) if safe_sizes else 8)


def dataset_gen_length(dataset_name: str) -> int:
    return DEFAULT_MBPP_GEN_LENGTH if dataset_name == "mbpp" else DEFAULT_GSM8K_GEN_LENGTH


def manifest_rows(manifest_payload: Any) -> list[dict[str, Any]]:
    if isinstance(manifest_payload, list):
        return manifest_payload
    if isinstance(manifest_payload, dict):
        rows = manifest_payload.get("rows")
        if isinstance(rows, list):
            return rows
    raise TypeError(f"Unsupported manifest payload type: {type(manifest_payload)!r}")


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["correct"]) for row in rows)
    total_nfe = sum(int(row["nfe"]) for row in rows)
    total_changed = sum(int(row["tokens_changed"]) for row in rows)
    return {
        "count": len(rows),
        "accuracy": round(correct / max(1, len(rows)), 4),
        "correct_count": correct,
        "avg_nfe": round(total_nfe / max(1, len(rows)), 2),
        "avg_tokens_changed": round(total_changed / max(1, len(rows)), 2),
    }


def run_dataset_group(
    engine: BatchedDLMInference,
    tokenizer: Any,
    task_id: str,
    rows: list[dict[str, Any]],
    steps: int,
    batch_size: int,
    predictions_path: Path,
    scores_path: Path,
) -> dict[str, Any]:
    dataset_name = rows[0]["dataset"]
    gen_length = dataset_gen_length(dataset_name)
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    started = time.time()
    processed = 0

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompt_ids = [
            tokenizer.encode(sample["prompt"], add_special_tokens=False)
            for sample in chunk
        ]
        outputs = engine.batched_standard_denoising(
            prompt_ids_list=prompt_ids,
            gen_length=gen_length,
            num_steps=steps,
            batch_size=len(chunk),
            seed=SEED,
        )
        texts = engine.decode_results(outputs)

        chunk_predictions: list[dict[str, Any]] = []
        chunk_scores: list[dict[str, Any]] = []
        for sample, result, text in zip(chunk, outputs, texts, strict=True):
            generated_text = text[:4000]
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

        append_jsonl(predictions_path, chunk_predictions)
        append_jsonl(scores_path, chunk_scores)
        predictions.extend(chunk_predictions)
        scores.extend(chunk_scores)
        processed += len(chunk)
        report_progress(
            task_id,
            {
                "phase": f"{dataset_name}_running",
                "processed": processed,
                "dataset_total": len(rows),
                "batch_size": batch_size,
                "steps": steps,
            },
        )
        torch.cuda.empty_cache()
        gc.collect()

    metrics = aggregate_metrics(scores)
    metrics.update(
        {
            "dataset": dataset_name,
            "gen_length": gen_length,
            "latency_sec": round(time.time() - started, 2),
            "batch_size": batch_size,
        }
    )
    return metrics


def main() -> int:
    args = parse_args()
    task_id = args.task_id
    arm_name = args.arm_name
    steps = args.steps

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arm_dir = ARMS_DIR / arm_name / f"seed{SEED}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(os.getpid()), encoding="utf-8")

    predictions_path = arm_dir / "predictions.jsonl"
    scores_path = arm_dir / "per_sample_scores.jsonl"
    for path in (predictions_path, scores_path):
        if path.exists():
            path.unlink()

    try:
        manifest_payload = load_json(SETUP_DIR / "sample_manifest.json")
        manifest = manifest_rows(manifest_payload)
        runtime_contract = load_json(SETUP_DIR / "runtime_contract.json")
        batch_probe = load_json(SETUP_DIR / "batch_probe.json")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        report_progress(task_id, {"phase": "load_engine", "steps": steps})
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
                rows=rows,
                steps=steps,
                batch_size=batch_size,
                predictions_path=predictions_path,
                scores_path=scores_path,
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
            "steps": steps,
            "seed": SEED,
            "sample_count": len(manifest),
            "overall": overall_metrics,
            "by_dataset": dataset_metrics,
            "runtime": {
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "batch_size": batch_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        }
        write_json(arm_dir / "metrics.json", metrics_payload)
        write_json(
            arm_dir / "gpu_progress.json",
            {
                "task_id": task_id,
                "arm_name": arm_name,
                "seed": SEED,
                "batch_size": batch_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "updated_at": now_iso(),
            },
        )
        write_json(RESULTS_DIR / f"{task_id}.json", metrics_payload)

        summary = (
            f"{arm_name}: overall_acc={overall_metrics['accuracy']} "
            f"gsm8k={dataset_metrics.get('gsm8k', {}).get('accuracy')} "
            f"mbpp={dataset_metrics.get('mbpp', {}).get('accuracy')} "
            f"batch={batch_size} steps={steps}"
        )
        report_progress(task_id, {"phase": "done", "summary": summary})
        mark_done(task_id, "success", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": task_id,
            "arm_name": arm_name,
            "status": "failed",
            "error": repr(exc),
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{task_id}.json", failure)
        report_progress(task_id, {"phase": "failed", "error": repr(exc)})
        mark_done(task_id, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
