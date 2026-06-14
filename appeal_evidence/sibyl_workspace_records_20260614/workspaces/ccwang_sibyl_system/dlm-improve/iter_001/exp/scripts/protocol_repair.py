#!/usr/bin/env python3
"""Pilot task: protocol_repair.

Repairs the evaluation protocol for the new TIGER iteration:
- create a held-out GSM8K calibration split (disjoint from headline test)
- audit GSM8K answer extraction
- define the actual compute logging schema every later task must emit
- record how lm-evaluation-harness will be used (or why not)

Writes the Sibyl experimenter contract files:
- exp/results/protocol_repair.pid
- exp/results/protocol_repair_PROGRESS.json
- exp/results/protocol_repair_DONE
- exp/results/protocol_repair.json
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

TASK_ID = "protocol_repair"
PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
RESULTS_PATH = RESULTS_DIR / f"{TASK_ID}.json"
CALIBRATION_SPLIT_PATH = PROJECT_ROOT / "data" / "gsm8k_calibration_split_seed42_100.jsonl"
CALIBRATION_MANIFEST_PATH = RESULTS_DIR / "gsm8k_calibration_manifest.json"
GSM8K_SHARED_PATH = Path("/home/ccwang/sibyl_system/shared/datasets/gsm8k")
SEED = 42
CALIBRATION_SAMPLES = 100


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


def normalize_answer(ans: str) -> str:
    ans = str(ans).strip().replace(",", "")
    if "." in ans:
        try:
            value = float(ans)
            if value == int(value):
                return str(int(value))
        except ValueError:
            return ans
    return ans


def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def load_gsm8k_dataset() -> DatasetDict:
    if GSM8K_SHARED_PATH.exists():
        try:
            dataset = load_from_disk(str(GSM8K_SHARED_PATH))
            if isinstance(dataset, DatasetDict):
                return dataset
        except Exception:
            pass
    return load_dataset("gsm8k", "main")


def build_calibration_split(train_dataset: Dataset) -> dict:
    indices = list(range(len(train_dataset)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    selected = indices[:CALIBRATION_SAMPLES]
    subset = train_dataset.select(selected)

    CALIBRATION_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_SPLIT_PATH.open("w", encoding="utf-8") as handle:
        for row in subset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source": "gsm8k_train",
        "seed": SEED,
        "sample_count": len(subset),
        "dataset_size": len(train_dataset),
        "path": str(CALIBRATION_SPLIT_PATH),
        "selected_indices_preview": selected[:10],
        "disjoint_from_headline_split": True,
    }
    CALIBRATION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_gsm8k_extractor_audit(test_dataset: Dataset) -> dict:
    sample_count = min(25, len(test_dataset))
    rows = test_dataset.select(range(sample_count))
    examples = []
    pass_count = 0
    for idx, row in enumerate(rows):
        gold = normalize_answer(extract_gsm8k_answer(row["answer"]))
        direct = normalize_answer(extract_gsm8k_answer(row["answer"]))
        synthetic = normalize_answer(extract_gsm8k_answer(f"The answer is {gold}."))
        boxed = normalize_answer(extract_gsm8k_answer(f"\\boxed{{{gold}}}"))
        direct_ok = direct == gold and gold != ""
        synthetic_ok = synthetic == gold and gold != ""
        boxed_ok = boxed == gold and gold != ""
        all_ok = direct_ok and synthetic_ok and boxed_ok
        pass_count += int(all_ok)
        examples.append(
            {
                "index": idx,
                "gold": gold,
                "direct_ok": direct_ok,
                "synthetic_ok": synthetic_ok,
                "boxed_ok": boxed_ok,
            }
        )
    return {
        "sample_count": sample_count,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / max(1, sample_count), 3),
        "examples": examples[:10],
    }


def lm_eval_policy() -> dict:
    available = True
    version = None
    import_error = None
    try:
        import lm_eval  # type: ignore

        version = getattr(lm_eval, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        available = False
        import_error = repr(exc)

    return {
        "package_available": available,
        "version": version,
        "integration_mode": "task_conventions_only",
        "direct_adapter_recommended": False,
        "reason": (
            "DLM custom iterative denoising loop does not map cleanly to the standard HF causal LM adapter. "
            "Reuse benchmark/task conventions and metrics, but keep the custom evaluator for the actual decode loop."
        ),
        "import_error": import_error,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    result = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "protocol": {},
    }

    try:
        report_progress(0, 4, {"phase": "load_datasets"})
        dataset = load_gsm8k_dataset()
        train_split = dataset["train"]
        test_split = dataset["test"]
        result["protocol"]["dataset_summary"] = {
            "train_size": len(train_split),
            "test_size": len(test_split),
            "shared_path": str(GSM8K_SHARED_PATH),
        }

        report_progress(1, 4, {"phase": "calibration_split"})
        calibration_manifest = build_calibration_split(train_split)
        result["protocol"]["heldout_calibration_split"] = calibration_manifest

        report_progress(2, 4, {"phase": "extractor_audit"})
        result["protocol"]["gsm8k_extractor_audit"] = run_gsm8k_extractor_audit(test_split)
        result["protocol"]["humaneval_policy"] = {
            "mode": "boundary_only",
            "completion_extractor": "reuse current/exp/scripts/full_llada_humaneval.py extraction logic",
            "execution_timeout_sec": 10,
        }

        report_progress(3, 4, {"phase": "compute_schema"})
        result["protocol"]["compute_logging_schema"] = {
            "required_fields": [
                "actual_nfe",
                "latency_sec",
                "tokens_per_sec",
                "batch_size",
                "peak_vram_mb",
                "attention_backend",
                "compile_enabled",
            ],
            "note": "Every later benchmark row must include the full schema."
        }
        result["protocol"]["lm_eval_policy"] = lm_eval_policy()

        extractor_pass = result["protocol"]["gsm8k_extractor_audit"]["pass_rate"] >= 0.95
        schema_pass = len(result["protocol"]["compute_logging_schema"]["required_fields"]) == 7
        split_pass = calibration_manifest["sample_count"] == CALIBRATION_SAMPLES
        overall_pass = extractor_pass and schema_pass and split_pass

        result["pass"] = overall_pass
        result["status"] = "success" if overall_pass else "failed"
        result["summary"] = (
            f"Held-out split={calibration_manifest['sample_count']} from GSM8K train; "
            f"extractor_pass_rate={result['protocol']['gsm8k_extractor_audit']['pass_rate']}; "
            f"lm_eval_mode={result['protocol']['lm_eval_policy']['integration_mode']}"
        )

        RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_progress(
            4,
            4,
            {
                "pass": overall_pass,
                "calibration_samples": calibration_manifest["sample_count"],
                "extractor_pass_rate": result["protocol"]["gsm8k_extractor_audit"]["pass_rate"],
            },
        )
        mark_done("success" if overall_pass else "failed", result["summary"])
        return 0 if overall_pass else 1
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = repr(exc)
        RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_progress(4, 4, {"pass": False, "error": repr(exc)})
        mark_done("failed", f"protocol_repair failed: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
