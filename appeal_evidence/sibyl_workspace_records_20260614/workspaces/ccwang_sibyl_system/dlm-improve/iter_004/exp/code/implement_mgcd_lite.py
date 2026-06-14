#!/usr/bin/env python3
"""Smoke-test the MGCD-lite inference path for iteration 4."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


TASK_ID = "implement_mgcd_lite"
PROJECT_ROOT = Path(
    os.environ.get(
        "SIBYL_REMOTE_PROJECT_ROOT",
        "/home/ccwang/sibyl_system/projects/dlm-improve",
    )
)
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
SETUP_DIR = PROJECT_ROOT / "current" / "exp" / "pilot_evidence_closure_v1" / "setup"
CODE_DIR = PROJECT_ROOT / "current" / "exp" / "code"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
SEED = 42
DEFAULT_GSM8K_GEN_LENGTH = 256
DEFAULT_MBPP_GEN_LENGTH = 512
MBPP_TIMEOUT = 10
MGCD_CONFIG = {
    "num_draft_steps": 12,
    "revision_steps": 58,
    "revision_fraction": 0.08,
    "bridge_radius": 1,
    "disagreement_weight": 0.7,
    "entropy_weight": 0.3,
    "support_threshold": 0.6,
    "entropy_margin": 0.03,
}

sys.path.insert(0, str(CODE_DIR))

from batched_dlm_utils_mgcd import BatchedDLMInference  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log(message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with (RESULTS_DIR / f"{TASK_ID}.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def report_progress(epoch: int, total_epochs: int, metric: dict[str, Any]) -> None:
    write_json(
        RESULTS_DIR / f"{TASK_ID}_PROGRESS.json",
        {
            "task_id": TASK_ID,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": epoch,
            "total_steps": total_epochs,
            "loss": None,
            "metric": metric,
            "updated_at": now_iso(),
        },
    )


def mark_done(status: str, summary: str, extra: dict[str, Any] | None = None) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()

    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress: dict[str, Any] = {}
    if progress_path.exists():
        try:
            final_progress = load_json(progress_path)
        except (OSError, ValueError, json.JSONDecodeError):
            final_progress = {}

    payload = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": now_iso(),
    }
    if extra:
        payload.update(extra)
    write_json(RESULTS_DIR / f"{TASK_ID}_DONE", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--sample-id", default="")
    return parser.parse_args()


def dataset_gen_length(dataset_name: str) -> int:
    return DEFAULT_MBPP_GEN_LENGTH if dataset_name == "mbpp" else DEFAULT_GSM8K_GEN_LENGTH


def safe_batch_from_probe(batch_probe: dict[str, Any]) -> int:
    rows = batch_probe.get("rows", [])
    safe_sizes = [
        int(row.get("safe_batch_size", 0))
        for row in rows
        if int(row.get("safe_batch_size", 0)) > 0
    ]
    return max(1, min(safe_sizes) if safe_sizes else 1)


def attention_backend(engine: BatchedDLMInference) -> str:
    config = getattr(getattr(engine, "model", None), "config", None)
    if config is None and hasattr(getattr(engine, "model", None), "module"):
        config = getattr(engine.model.module, "config", None)
    return getattr(config, "_attn_implementation", None) or "eager_or_default"


def compile_enabled(engine: BatchedDLMInference) -> bool:
    model = getattr(engine, "model", None)
    return hasattr(model, "_orig_mod") or (
        hasattr(model, "module") and hasattr(model.module, "_orig_mod")
    )


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().replace(",", "")
    if not answer:
        return ""
    try:
        value = float(answer)
    except ValueError:
        return answer
    if value.is_integer():
        return str(int(value))
    return answer


def extract_gsm8k_answer(text: str) -> str:
    patterns = [
        r"####\s*(-?[\d,]+\.?\d*)",
        r"(?:the\s+)?answer\s+is\s*[:\s]*(-?[\d,]+\.?\d*)",
        r"\\boxed\{([^}]+)\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else ""


def check_gsm8k_correct(prediction: str, gold: str) -> bool:
    gold_value = gold.split("####")[-1].strip() if "####" in gold else gold
    return normalize_answer(extract_gsm8k_answer(prediction)) == normalize_answer(gold_value)


def extract_code_from_generation(generated_text: str) -> str:
    cleaned = (
        generated_text.replace("<|endoftext|>", "")
        .replace("<|end|>", "")
        .replace("[MASK]", "")
    )
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return cleaned.strip()


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


def select_smoke_row(rows: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    if sample_id:
        for row in rows:
            if row["sample_id"] == sample_id:
                return row
        raise ValueError(f"sample_id not found in manifest: {sample_id}")

    gsm8k_rows = [row for row in rows if row.get("dataset") == "gsm8k"]
    if gsm8k_rows:
        high_entropy = [
            row for row in gsm8k_rows if row.get("difficulty_bucket") == "high_entropy"
        ]
        return high_entropy[0] if high_entropy else gsm8k_rows[0]
    return rows[0]


def build_gpu_profile(
    *,
    inherited_safe_batch_size: int,
    probed_safe_batch_size: int,
    gen_length: int,
    prompt_length: int,
    engine: BatchedDLMInference,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(engine.device)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
    return {
        "task_id": TASK_ID,
        "gpu_name": props.name,
        "gpu_id": torch.cuda.current_device(),
        "vram_total_mb": round(props.total_memory / 1024**2),
        "prompt_length": prompt_length,
        "gen_length": gen_length,
        "inherited_safe_batch_size": inherited_safe_batch_size,
        "probed_safe_batch_size": probed_safe_batch_size,
        "actual_batch_size": 1,
        "peak_vram_mb": peak_vram_mb,
        "attention_backend": attention_backend(engine),
        "compile_enabled": compile_enabled(engine),
        "probe_note": "Smoke task probes headroom on the single assigned GPU and executes one sample end-to-end.",
        "timestamp": now_iso(),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    torch.manual_seed(SEED)

    total_epochs = 5
    try:
        report_progress(1, total_epochs, {"phase": "load_inputs"})
        manifest_payload = load_json(SETUP_DIR / "sample_manifest.json")
        manifest_rows = manifest_payload.get("rows", manifest_payload)
        runtime_contract = load_json(SETUP_DIR / "runtime_contract.json")
        batch_probe = load_json(SETUP_DIR / "batch_probe.json")
        smoke_row = select_smoke_row(manifest_rows, args.sample_id)
        gen_length = dataset_gen_length(smoke_row["dataset"])
        inherited_safe_batch_size = safe_batch_from_probe(batch_probe)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        prompt_ids = tokenizer.encode(smoke_row["prompt"], add_special_tokens=False)
        log(f"selected smoke sample {smoke_row['sample_id']} ({smoke_row['dataset']})")

        report_progress(
            2,
            total_epochs,
            {
                "phase": "load_engine",
                "sample_id": smoke_row["sample_id"],
                "dataset": smoke_row["dataset"],
            },
        )
        runtime_meta = runtime_contract.get("runtime", {})
        engine = BatchedDLMInference(
            model_path=args.model_path,
            device="cuda",
            use_flash_attn=bool(runtime_meta.get("flash_attention_available", False)),
            use_compile=bool(runtime_meta.get("torch_compile_enabled", False)),
        )

        report_progress(
            3,
            total_epochs,
            {
                "phase": "probe_batch_size",
                "sample_id": smoke_row["sample_id"],
                "inherited_safe_batch_size": inherited_safe_batch_size,
            },
        )
        probed_safe_batch_size = engine.find_max_batch_size(
            gen_length=gen_length,
            prompt_max_len=min(512, max(1, len(prompt_ids))),
            lo=1,
            hi=max(8, min(32, inherited_safe_batch_size + 4)),
            safety_margin=0.9,
        )
        gpu_profile = build_gpu_profile(
            inherited_safe_batch_size=inherited_safe_batch_size,
            probed_safe_batch_size=probed_safe_batch_size,
            gen_length=gen_length,
            prompt_length=len(prompt_ids),
            engine=engine,
        )
        write_json(RESULTS_DIR / f"{TASK_ID}_gpu_profile.json", gpu_profile)
        torch.cuda.reset_peak_memory_stats(engine.device)
        gc.collect()
        torch.cuda.empty_cache()

        report_progress(
            4,
            total_epochs,
            {
                "phase": "run_smoke",
                "sample_id": smoke_row["sample_id"],
                "attention_backend": attention_backend(engine),
                "compile_enabled": compile_enabled(engine),
                "probed_safe_batch_size": probed_safe_batch_size,
            },
        )
        started = time.time()
        outputs = engine.batched_mgcd(
            prompt_ids_list=[prompt_ids],
            gen_length=gen_length,
            batch_size=1,
            seed=SEED,
            **MGCD_CONFIG,
        )
        latency_sec = round(time.time() - started, 2)
        output = outputs[0]
        generated_text = engine.decode_results(outputs)[0][:4000]
        mgcd_stats = output.get("mgcd_stats", {}) or {}
        entropy_stats = output.get("entropy_stats", {}) or {}

        evaluation: dict[str, Any]
        if smoke_row["dataset"] == "gsm8k":
            predicted_answer = extract_gsm8k_answer(generated_text)
            evaluation = {
                "correct": bool(check_gsm8k_correct(generated_text, smoke_row["reference"])),
                "predicted_answer": predicted_answer,
                "reference_answer": smoke_row["reference"],
            }
        else:
            extracted_code = extract_code_from_generation(generated_text)
            passed, error = check_mbpp_correct(extracted_code, smoke_row.get("test_list", []))
            evaluation = {
                "correct": bool(passed),
                "error": error,
                "extracted_code": extracted_code[:2000],
                "reference_code": smoke_row["reference"],
            }

        result_payload = {
            "task_id": TASK_ID,
            "candidate_id": "cand_mgcd",
            "mode": "PILOT",
            "status": "success",
            "seed": SEED,
            "sample_count": 1,
            "runtime_contract": {
                "source": str(SETUP_DIR / "runtime_contract.json"),
                "attention_backend": attention_backend(engine),
                "compile_enabled": compile_enabled(engine),
                "flash_attention_available": bool(runtime_meta.get("flash_attention_available", False)),
                "prompt_max_tokens": runtime_contract.get("prompt_max_tokens"),
                "generation_tokens": runtime_contract.get("generation_tokens"),
                "inherited_safe_batch_size": inherited_safe_batch_size,
                "probed_safe_batch_size": probed_safe_batch_size,
                "actual_batch_size": 1,
            },
            "mgcd_config": MGCD_CONFIG,
            "sample": {
                "sample_id": smoke_row["sample_id"],
                "dataset": smoke_row["dataset"],
                "source_index": smoke_row["source_index"],
                "difficulty_bucket": smoke_row.get("difficulty_bucket"),
                "prompt_length": smoke_row["prompt_length"],
                "latency_sec": latency_sec,
                "nfe": int(output["nfe"]),
                "tokens_changed": int(output.get("tokens_changed", 0)),
                "generated_length": int(output["gen_end"] - output["gen_start"]),
                "generated_text": generated_text,
                "entropy_stats": entropy_stats,
                "mgcd_stats": mgcd_stats,
                "evaluation": evaluation,
            },
            "pass_criteria_check": {
                "end_to_end_smoke": True,
                "logged_contested_islands": "contested_island_count" in mgcd_stats,
                "logged_accepted_updates": "accepted_updates" in mgcd_stats,
                "logged_runtime_contract": True,
            },
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", result_payload)

        summary = (
            f"smoke sample {smoke_row['sample_id']} completed on {smoke_row['dataset']}; "
            f"islands={mgcd_stats.get('contested_island_count', 0)}, "
            f"accepted={mgcd_stats.get('accepted_updates', 0)}, "
            f"rejected={mgcd_stats.get('rejected_updates', 0)}, "
            f"attn={attention_backend(engine)}, compile={compile_enabled(engine)}"
        )
        report_progress(
            5,
            total_epochs,
            {
                "phase": "done",
                "summary": summary,
                "sample_id": smoke_row["sample_id"],
            },
        )
        mark_done("success", summary, extra={"result_path": str(RESULTS_DIR / f"{TASK_ID}.json")})
        log(summary)
        return 0
    except Exception as exc:
        failure = f"{TASK_ID} failed: {exc!r}"
        report_progress(5, total_epochs, {"phase": "failed", "error": repr(exc)})
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {
                "task_id": TASK_ID,
                "status": "failed",
                "error": repr(exc),
                "timestamp": now_iso(),
            },
        )
        mark_done("failed", failure)
        log(failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
