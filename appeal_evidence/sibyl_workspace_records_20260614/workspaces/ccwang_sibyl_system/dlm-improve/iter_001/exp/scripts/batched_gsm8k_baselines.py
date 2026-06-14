#!/usr/bin/env python3
"""Batched GSM8K pilot baselines for the TIGER queue.

Implements the ready-to-run pilot tasks:
- baseline_standard_dnb
- baseline_prophet

The script keeps the Sibyl task contract:
- exp/results/<task_id>.pid
- exp/results/<task_id>_PROGRESS.json
- exp/results/<task_id>_DONE
- exp/results/<task_id>_gpu_profile.json
- exp/results/<task_id>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk

from batched_dlm_utils import BatchedDLMInference

TASK_CONFIGS = {
    "pilot_mgcd_gsm8k": {
        "methods": [
            ("Standard-64", "standard", {"num_steps": 64}),
            ("DNB-84", "standard", {"num_steps": 84}),
            ("RAND-84", "random_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            ("CARD-84", "entropy_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            ("MGCD-84", "mgcd", {"num_draft_steps": 12, "revision_fraction": 0.10, "revision_steps": 58, "bridge_radius": 1}),
        ],
        "output_name": "pilot_mgcd_gsm8k.json",
        "summary_name": "MGCD GSM8K pilot（DNB / RAND / CARD / MGCD）",
    },
    "pilot_mgcd_v2_gsm8k": {
        "methods": [
            ("Standard-64", "standard", {"num_steps": 64}),
            ("DNB-84", "standard", {"num_steps": 84}),
            ("RAND-84", "random_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            ("CARD-84", "entropy_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            (
                "MGCDv2-84",
                "mgcd",
                {
                    "num_draft_steps": 12,
                    "revision_fraction": 0.08,
                    "revision_steps": 58,
                    "bridge_radius": 1,
                    "disagreement_weight": 0.7,
                    "entropy_weight": 0.3,
                    "support_threshold": 0.6,
                    "entropy_margin": 0.03,
                },
            ),
        ],
        "output_name": "pilot_mgcd_v2_gsm8k.json",
        "summary_name": "MGCD v2 GSM8K pilot（DNB / RAND / CARD / MGCDv2）",
    },
    "pilot_mgcd_anchor_gsm8k": {
        "methods": [
            ("Standard-64", "standard", {"num_steps": 64}),
            ("DNB-84", "standard", {"num_steps": 84}),
            ("RAND-84", "random_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            ("CARD-84", "entropy_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            (
                "MGCD-anchor-84",
                "mgcd",
                {
                    "main_draft_steps": 64,
                    "scout_steps": 8,
                    "revision_fraction": 0.08,
                    "revision_steps": 10,
                    "bridge_radius": 1,
                    "disagreement_weight": 0.75,
                    "entropy_weight": 0.25,
                    "support_threshold": 0.55,
                    "entropy_margin": 0.02,
                },
            ),
        ],
        "output_name": "pilot_mgcd_anchor_gsm8k.json",
        "summary_name": "MGCD anchor GSM8K pilot（DNB / RAND / CARD / MGCD-anchor）",
    },
    "pilot_mgcd_hybrid_gsm8k": {
        "methods": [
            ("Standard-64", "standard", {"num_steps": 64}),
            ("DNB-84", "standard", {"num_steps": 84}),
            ("RAND-84", "random_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            ("CARD-84", "entropy_revise", {"num_draft_steps": 64, "revision_fraction": 0.15, "revision_steps": 6}),
            (
                "MGCD-hybrid-84",
                "mgcd",
                {
                    "main_draft_steps": 64,
                    "scout_steps": 4,
                    "revision_fraction": 0.10,
                    "revision_steps": 14,
                    "bridge_radius": 1,
                    "disagreement_weight": 0.5,
                    "entropy_weight": 0.5,
                    "support_threshold": 0.55,
                    "entropy_margin": 0.01,
                },
            ),
        ],
        "output_name": "pilot_mgcd_hybrid_gsm8k.json",
        "summary_name": "MGCD hybrid GSM8K pilot（DNB / RAND / CARD / MGCD-hybrid）",
    },
    "baseline_standard_dnb": {
        "methods": [
            ("Standard-64", "standard", {"num_steps": 64}),
            ("DNB-84", "standard", {"num_steps": 84}),
        ],
        "output_name": "baseline_standard_dnb.json",
        "summary_name": "Standard 与 DNB 基线复现",
    },
    "baseline_prophet": {
        "methods": [
            ("Prophet-64", "prophet", {"num_steps": 64, "confidence_threshold": 0.95}),
        ],
        "output_name": "baseline_prophet.json",
        "summary_name": "Prophet 强基线复现",
    },
    "baseline_dcd_optional": {
        "methods": [
            (
                "DCD-lite-64",
                "dcd",
                {
                    "num_steps": 64,
                    "base_confidence_threshold": 0.90,
                    "final_confidence_threshold": 0.55,
                    "window_radius": 2,
                    "late_commit_start": 0.75,
                },
            ),
        ],
        "output_name": "baseline_dcd_optional.json",
        "summary_name": "DCD 可选补充基线（低成本近似 pilot）",
        "implementation_note": "Low-cost approximate DCD pilot implemented via confidence-aware deferred local commitment windows in the shared batched runner.",
    },
}

PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
PILOT_RESULTS_DIR = RESULTS_DIR / "pilots"
SETUP_PATH = RESULTS_DIR / "setup_throughput_verification.json"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
GSM8K_TEST_PATH = "/home/ccwang/sibyl_system/shared/datasets/gsm8k"
MASK_TOKEN_ID = 126336
GEN_LENGTH = 256
SEED = 42


def cosine_schedule(t: int, total_steps: int) -> float:
    return 1.0 - math.cos(math.pi * t / (2 * total_steps))


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


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().replace(",", "")
    if "." in answer:
        try:
            value = float(answer)
        except ValueError:
            return answer
        if value == int(value):
            return str(int(value))
    return answer


def check_gsm8k_correct(predicted_text: str, reference_answer: str) -> bool:
    predicted = normalize_answer(extract_gsm8k_answer(predicted_text))
    reference = normalize_answer(reference_answer)
    return bool(predicted and predicted == reference)


def read_safe_batch_size() -> int:
    if not SETUP_PATH.exists():
        return 32
    try:
        payload = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
        compile_on = payload["checks"].get("throughput_compile_on") or {}
        compile_off = payload["checks"].get("throughput_compile_off") or {}
        safe_batch = compile_on.get("safe_batch_size") or compile_off.get("safe_batch_size") or 32
        return max(1, int(safe_batch))
    except (OSError, ValueError, KeyError, TypeError):
        return 32


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def report_progress(task_id: str, epoch: int, total_epochs: int, step: int, total_steps: int, metric: dict) -> None:
    write_json(
        RESULTS_DIR / f"{task_id}_PROGRESS.json",
        {
            "task_id": task_id,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "loss": None,
            "metric": metric,
            "updated_at": datetime.now().isoformat(),
        },
    )


def mark_done(task_id: str, status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{task_id}.pid"
    if pid_path.exists():
        pid_path.unlink()
    progress_path = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            final_progress = {}
    write_json(
        RESULTS_DIR / f"{task_id}_DONE",
        {
            "task_id": task_id,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": datetime.now().isoformat(),
        },
    )


def load_samples(tokenizer, num_samples: int) -> list[dict]:
    dataset = load_from_disk(GSM8K_TEST_PATH)
    count = min(num_samples, len(dataset))
    samples = []
    for idx in range(count):
        row = dataset[idx]
        answer = row["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {row['question']}\n\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        samples.append(
            {
                "idx": idx,
                "prompt_ids": prompt_ids,
                "question": row["question"],
                "final_answer": final_answer,
            }
        )
    return samples


def run_batch(engine: BatchedDLMInference, mode: str, prompts: list[list[int]], kwargs: dict, batch_size: int) -> list[dict]:
    if mode == "standard":
        return engine.batched_standard_denoising(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            batch_size=batch_size,
            seed=SEED,
            **kwargs,
        )
    if mode == "prophet":
        return engine.batched_prophet(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            batch_size=batch_size,
            seed=SEED,
            **kwargs,
        )
    if mode == "entropy_revise":
        return engine.batched_entropy_revise(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            batch_size=batch_size,
            seed=SEED,
            **kwargs,
        )
    if mode == "random_revise":
        return engine.batched_random_revise(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            batch_size=batch_size,
            seed=SEED,
            **kwargs,
        )
    if mode == "mgcd":
        return engine.batched_mgcd(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            batch_size=batch_size,
            seed=SEED,
            **kwargs,
        )
    if mode == "dcd":
        return engine.batched_dcd(
            prompt_ids_list=prompts,
            gen_length=GEN_LENGTH,
            batch_size=batch_size,
            seed=SEED,
            **kwargs,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_method(
    engine: BatchedDLMInference,
    task_id: str,
    method_index: int,
    total_methods: int,
    method_name: str,
    mode: str,
    method_kwargs: dict,
    samples: list[dict],
    batch_size: int,
) -> dict:
    total_correct = 0
    total_nfe = 0
    total_deferred_count = 0
    total_committed_count = 0
    total_tokens_changed = 0
    total_locked_ratio = 0.0
    total_contested_island_count = 0
    total_mean_contested_span_length = 0.0
    total_accepted_updates = 0
    total_rejected_updates = 0
    per_sample = []
    method_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(engine.device)

    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        prompts = [sample["prompt_ids"] for sample in chunk]
        chunk_results = run_batch(engine, mode, prompts, method_kwargs, batch_size=batch_size)
        chunk_texts = engine.decode_results(chunk_results)

        for sample, result, generated_text in zip(chunk, chunk_results, chunk_texts, strict=True):
            is_correct = check_gsm8k_correct(generated_text, sample["final_answer"])
            total_correct += int(is_correct)
            total_nfe += result["nfe"]
            total_deferred_count += int(result.get("deferred_count", 0))
            total_committed_count += int(result.get("committed_count", 0))
            total_tokens_changed += int(result.get("tokens_changed", 0))
            mgcd_stats = result.get("mgcd_stats") or {}
            total_locked_ratio += float(mgcd_stats.get("locked_token_ratio", 0.0))
            total_contested_island_count += int(mgcd_stats.get("contested_island_count", 0))
            total_mean_contested_span_length += float(mgcd_stats.get("mean_contested_span_length", 0.0))
            total_accepted_updates += int(mgcd_stats.get("accepted_updates", 0))
            total_rejected_updates += int(mgcd_stats.get("rejected_updates", 0))
            per_sample.append(
                {
                    "idx": sample["idx"],
                    "correct": is_correct,
                    "nfe": int(result["nfe"]),
                    "tokens_changed": int(result.get("tokens_changed", 0)),
                    "predicted_answer": extract_gsm8k_answer(generated_text),
                    "reference_answer": sample["final_answer"],
                    "generated_text": generated_text[:500],
                    "mgcd_stats": mgcd_stats if mgcd_stats else None,
                }
            )

        processed = min(chunk_start + len(chunk), len(samples))
        elapsed = time.perf_counter() - method_start
        avg_nfe = total_nfe / max(1, processed)
        report_progress(
            task_id,
            method_index,
            total_methods,
            processed,
            len(samples),
            {
                "method": method_name,
                "accuracy_so_far": round(total_correct / max(1, processed), 4),
                "avg_nfe": round(avg_nfe, 2),
                "batch_size": batch_size,
                "elapsed_sec": round(elapsed, 2),
            },
        )

    latency_sec = time.perf_counter() - method_start
    accuracy = total_correct / max(1, len(samples))
    avg_nfe = total_nfe / max(1, len(samples))
    total_generated_tokens = len(samples) * GEN_LENGTH
    tokens_per_sec = total_generated_tokens / max(latency_sec, 1e-6)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
    attention_backend = getattr(getattr(engine.model, "config", None), "_attn_implementation", None) or "eager_or_default"
    compile_enabled = hasattr(engine.model, "_orig_mod")

    method_result = {
        "method": method_name,
        "num_samples": len(samples),
        "accuracy": round(accuracy, 4),
        "actual_nfe": round(avg_nfe, 2),
        "latency_sec": round(latency_sec, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "batch_size": batch_size,
        "peak_vram_mb": peak_vram_mb,
        "attention_backend": attention_backend,
        "compile_enabled": compile_enabled,
        "per_sample": per_sample,
    }
    if mode == "prophet":
        method_result["stop_ratio"] = round(1.0 - (avg_nfe / method_kwargs["num_steps"]), 4)
    if mode == "dcd":
        method_result["implementation_fidelity"] = "approximate_pilot"
        method_result["deferred_count_mean"] = round(
            total_deferred_count / max(1, len(samples)),
            2,
        )
        method_result["committed_count_mean"] = round(
            total_committed_count / max(1, len(samples)),
            2,
        )
        method_result["window_radius"] = method_kwargs["window_radius"]
        method_result["late_commit_start"] = method_kwargs["late_commit_start"]
    if mode in {"entropy_revise", "random_revise", "mgcd"}:
        method_result["tokens_changed_mean"] = round(
            total_tokens_changed / max(1, len(samples)),
            2,
        )
    if mode == "mgcd":
        method_result["locked_token_ratio_mean"] = round(
            total_locked_ratio / max(1, len(samples)),
            4,
        )
        method_result["contested_island_count_mean"] = round(
            total_contested_island_count / max(1, len(samples)),
            2,
        )
        method_result["mean_contested_span_length"] = round(
            total_mean_contested_span_length / max(1, len(samples)),
            2,
        )
        method_result["accepted_updates_mean"] = round(
            total_accepted_updates / max(1, len(samples)),
            2,
        )
        method_result["rejected_updates_mean"] = round(
            total_rejected_updates / max(1, len(samples)),
            2,
        )
        method_result["candidate_support_ratio_mean"] = round(
            sum(
                float((sample.get("mgcd_stats") or {}).get("candidate_support_ratio", 0.0))
                for sample in per_sample
            ) / max(1, len(samples)),
            4,
        )
    return method_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_id = args.task
    task_config = TASK_CONFIGS[task_id]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(os.getpid()), encoding="utf-8")

    engine = None
    try:
        requested_batch_size = args.batch_size if args.batch_size > 0 else read_safe_batch_size()
        engine = BatchedDLMInference(
            model_path=MODEL_PATH,
            device="cuda",
            use_flash_attn=True,
            use_compile=True,
        )
        samples = load_samples(engine.tokenizer, args.samples)
        batch_size = min(requested_batch_size, len(samples))

        report_progress(task_id, 0, len(task_config["methods"]), 0, len(samples), {"phase": "loaded"})
        methods = []
        for method_index, (method_name, mode, method_kwargs) in enumerate(task_config["methods"], start=1):
            methods.append(
                evaluate_method(
                    engine=engine,
                    task_id=task_id,
                    method_index=method_index,
                    total_methods=len(task_config["methods"]),
                    method_name=method_name,
                    mode=mode,
                    method_kwargs=method_kwargs,
                    samples=samples,
                    batch_size=batch_size,
                )
            )

        output = {
            "task_id": task_id,
            "summary_name": task_config["summary_name"],
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "gen_length": GEN_LENGTH,
            "config": {
                "model_path": MODEL_PATH,
                "dataset_path": GSM8K_TEST_PATH,
                "batch_size": batch_size,
                "safe_batch_size_reference": requested_batch_size,
                "compile_enabled": True,
                "seed": SEED,
            },
            "methods": methods,
        }
        if "implementation_note" in task_config:
            output["implementation_note"] = task_config["implementation_note"]

        output_path = RESULTS_DIR / task_config["output_name"]
        write_json(output_path, output)
        write_json(PILOT_RESULTS_DIR / task_config["output_name"], output)
        write_json(
            RESULTS_DIR / f"{task_id}_gpu_profile.json",
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "used_batch_size": batch_size,
                "peak_vram_mb": max(method["peak_vram_mb"] for method in methods),
                "attention_backend": methods[0]["attention_backend"],
                "compile_enabled": True,
            },
        )

        best_line = ", ".join(
            f"{method['method']} acc={method['accuracy']:.4f} nfe={method['actual_nfe']:.2f} tps={method['tokens_per_sec']:.1f}"
            for method in methods
        )
        mark_done(task_id, "success", best_line)
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            RESULTS_DIR / task_config["output_name"],
            {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": repr(exc),
            },
        )
        mark_done(task_id, "failed", repr(exc))
        return 1
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
