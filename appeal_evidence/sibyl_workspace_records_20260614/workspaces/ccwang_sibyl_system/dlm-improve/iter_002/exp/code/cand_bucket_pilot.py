#!/usr/bin/env python3
"""Pilot benefit-bucket audit for the GSM8K headline pair on a matched runtime path."""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset


PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
TASK_ID = "cand_bucket_pilot"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
GEN_LENGTH = 256
NUM_SAMPLES = 100
SEED = 42
REVISION_FRACTION = 0.10
REVISION_STEPS = 3
PAIR_NAME = "Standard-64 vs Entropy-Revise-64+3"
RUNTIME_PROBE_PATH = RESULTS_DIR / "runtime_probe_iter2.json"

sys.path.insert(0, str(PROJECT_ROOT / "exp" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "exp" / "code"))

from batched_dlm_utils import BatchedDLMInference  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_pid() -> None:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")


def report_progress(phase: str, metric: dict[str, Any]) -> None:
    payload = {
        "task_id": TASK_ID,
        "epoch": 0,
        "total_epochs": 1,
        "step": 0,
        "total_steps": 1,
        "loss": None,
        "metric": {"phase": phase, **metric},
        "updated_at": datetime.now().isoformat(),
    }
    write_json(RESULTS_DIR / f"{TASK_ID}_PROGRESS.json", payload)


def mark_done(status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress: dict[str, Any] = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
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
    return normalize_answer(extract_gsm8k_answer(prediction)) == normalize_answer(gold)


def load_samples(tokenizer, num_samples: int) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split="test")
    total = min(num_samples, len(dataset))
    samples: list[dict[str, Any]] = []
    for idx in range(total):
        question = dataset[idx]["question"]
        answer = dataset[idx]["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        samples.append(
            {
                "idx": idx,
                "question": question,
                "prompt_ids": prompt_ids,
                "final_answer": final_answer,
            }
        )
    return samples


def runtime_contract() -> dict[str, Any]:
    if not RUNTIME_PROBE_PATH.exists():
        raise FileNotFoundError(f"Runtime probe missing: {RUNTIME_PROBE_PATH}")
    payload = json.loads(RUNTIME_PROBE_PATH.read_text(encoding="utf-8"))
    compile_on = payload["checks"].get("throughput_compile_on", {})
    gpu = payload["checks"].get("gpu", {})
    return {
        "source": str(RUNTIME_PROBE_PATH),
        "compile_enabled": True,
        "requested_batch_size": int(compile_on.get("safe_batch_size", 32)),
        "attention_backend": compile_on.get("attn_backend", payload["checks"]["model"].get("attn_implementation", "eager")),
        "flash_attention_2_available": bool(gpu.get("flash_attn_2_available", False)),
        "probe_gpu_name": gpu.get("device_name"),
    }


def engine_attention_backend(engine: BatchedDLMInference) -> str:
    return getattr(getattr(engine.model, "config", None), "_attn_implementation", None) or "eager_or_default"


def run_method(
    engine: BatchedDLMInference,
    mode: str,
    samples: list[dict[str, Any]],
    requested_batch_size: int,
) -> dict[str, Any]:
    current_batch_size = max(1, min(requested_batch_size, len(samples)))
    method_name = "Standard-64" if mode == "standard" else "Entropy-Revise-64+3"
    prompt_ids_list = [sample["prompt_ids"] for sample in samples]
    last_exc: RuntimeError | None = None

    while current_batch_size >= 1:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(engine.device)
            start = time.perf_counter()
            if mode == "standard":
                outputs = engine.batched_standard_denoising(
                    prompt_ids_list=prompt_ids_list,
                    gen_length=GEN_LENGTH,
                    num_steps=64,
                    batch_size=current_batch_size,
                    seed=SEED,
                )
            elif mode == "entropy":
                outputs = engine.batched_entropy_revise(
                    prompt_ids_list=prompt_ids_list,
                    gen_length=GEN_LENGTH,
                    num_draft_steps=64,
                    revision_fraction=REVISION_FRACTION,
                    revision_steps=REVISION_STEPS,
                    batch_size=current_batch_size,
                    seed=SEED,
                )
            else:  # pragma: no cover
                raise ValueError(f"Unsupported mode={mode}")
            latency = time.perf_counter() - start
            texts = engine.decode_results(outputs)

            per_sample: list[dict[str, Any]] = []
            total_correct = 0
            total_nfe = 0
            total_tokens_changed = 0
            signal_values: list[float] = []
            for sample, output, generated_text in zip(samples, outputs, texts, strict=True):
                stats = output.get("entropy_stats") or {}
                predicted = normalize_answer(extract_gsm8k_answer(generated_text))
                reference = normalize_answer(sample["final_answer"])
                correct = bool(predicted and predicted == reference)
                total_correct += int(correct)
                total_nfe += int(output["nfe"])
                tokens_changed = int(output.get("tokens_changed", 0))
                total_tokens_changed += tokens_changed
                if mode == "entropy":
                    signal_values.append(
                        float(
                            stats.get(
                                "revision_mean_entropy",
                                stats.get("mean_entropy", 0.0),
                            )
                        )
                    )
                per_sample.append(
                    {
                        "idx": sample["idx"],
                        "question": sample["question"],
                        "correct": correct,
                        "nfe": int(output["nfe"]),
                        "tokens_changed": tokens_changed,
                        "signal_value": signal_values[-1] if mode == "entropy" and signal_values else 0.0,
                        "signal_stats": stats,
                        "predicted_answer": predicted,
                        "reference_answer": reference,
                        "generated_text": generated_text[:700],
                    }
                )

            return {
                "method": method_name,
                "mode": mode,
                "num_samples": len(samples),
                "accuracy": round(total_correct / max(len(samples), 1), 4),
                "correct_count": total_correct,
                "actual_nfe": round(total_nfe / max(len(samples), 1), 2),
                "latency_sec": round(latency, 2),
                "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(latency, 1e-6), 2),
                "batch_size": current_batch_size,
                "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": hasattr(engine.model, "_orig_mod"),
                "avg_tokens_changed": round(total_tokens_changed / max(len(samples), 1), 2),
                "avg_signal_value": round(sum(signal_values) / max(len(signal_values), 1), 4) if signal_values else 0.0,
                "per_sample": per_sample,
            }
        except RuntimeError as exc:
            last_exc = exc
            text = repr(exc).lower()
            retryable = any(marker in text for marker in ("out of memory", "cuda error", "cublas", "triton", "cudnn"))
            if not retryable or current_batch_size == 1:
                raise
            next_batch = max(1, current_batch_size // 2)
            report_progress(
                f"{mode}_batch_backoff",
                {
                    "method": method_name,
                    "failed_batch_size": current_batch_size,
                    "retry_batch_size": next_batch,
                    "error": repr(exc)[:240],
                },
            )
            current_batch_size = next_batch

    if last_exc is not None:  # pragma: no cover
        raise last_exc
    raise RuntimeError(f"{method_name} exited without producing results")


def build_bucket_audit(standard: dict[str, Any], entropy: dict[str, Any], runtime_meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    standard_rows = {row["idx"]: row for row in standard["per_sample"]}
    entropy_rows = {row["idx"]: row for row in entropy["per_sample"]}
    joined = []
    bucket_examples: dict[str, list[dict[str, Any]]] = {"fixed": [], "harmed": [], "no_effect": []}
    counts = {"fixed": 0, "harmed": 0, "no_effect": 0, "same_correct": 0, "same_wrong": 0}

    for idx in sorted(standard_rows):
        if idx not in entropy_rows:
            continue
        std = standard_rows[idx]
        ent = entropy_rows[idx]
        if (not std["correct"]) and ent["correct"]:
            bucket = "fixed"
        elif std["correct"] and (not ent["correct"]):
            bucket = "harmed"
        else:
            bucket = "no_effect"
        counts[bucket] += 1
        if std["correct"] and ent["correct"]:
            counts["same_correct"] += 1
        elif (not std["correct"]) and (not ent["correct"]):
            counts["same_wrong"] += 1

        row = {
            "idx": idx,
            "bucket": bucket,
            "subtype": "same_correct" if std["correct"] and ent["correct"] else ("same_wrong" if (not std["correct"]) and (not ent["correct"]) else bucket),
            "standard_correct": std["correct"],
            "entropy_correct": ent["correct"],
            "standard_prediction": std["predicted_answer"],
            "entropy_prediction": ent["predicted_answer"],
            "reference_answer": std["reference_answer"],
            "entropy_tokens_changed": ent["tokens_changed"],
            "entropy_signal_value": ent["signal_value"],
            "standard_generated_text": std["generated_text"],
            "entropy_generated_text": ent["generated_text"],
        }
        joined.append(row)
        if len(bucket_examples[bucket]) < 5:
            bucket_examples[bucket].append(row)

    total = len(joined)
    coverage = round(total / max(len(standard_rows), 1), 4)
    accuracy_delta = round(entropy["accuracy"] - standard["accuracy"], 4)
    implied_delta = round((counts["fixed"] - counts["harmed"]) / max(total, 1), 4)

    audit = {
        "task_id": TASK_ID,
        "candidate_id": "cand_bucket",
        "pair_name": PAIR_NAME,
        "timestamp": datetime.now().isoformat(),
        "num_samples": total,
        "coverage": coverage,
        "status": "success",
        "pass": bool(coverage >= 0.95),
        "go_no_go": "GO" if coverage >= 0.95 else "NO_GO",
        "aggregate": {
            "standard_accuracy": standard["accuracy"],
            "entropy_accuracy": entropy["accuracy"],
            "accuracy_delta": accuracy_delta,
            "fixed_minus_harmed_delta": implied_delta,
            "delta_matches_bucket_counts": bool(abs(accuracy_delta - implied_delta) < 1e-6),
        },
        "bucket_counts": counts,
        "bucket_proportions": {key: round(value / max(total, 1), 4) for key, value in counts.items()},
        "methods": {
            "standard": {
                "method": standard["method"],
                "actual_nfe": standard["actual_nfe"],
                "latency_sec": standard["latency_sec"],
                "batch_size": standard["batch_size"],
                "compile_enabled": standard["compile_enabled"],
                "attention_backend": standard["attention_backend"],
                "peak_vram_mb": standard["peak_vram_mb"],
            },
            "entropy": {
                "method": entropy["method"],
                "actual_nfe": entropy["actual_nfe"],
                "latency_sec": entropy["latency_sec"],
                "batch_size": entropy["batch_size"],
                "compile_enabled": entropy["compile_enabled"],
                "attention_backend": entropy["attention_backend"],
                "peak_vram_mb": entropy["peak_vram_mb"],
                "avg_tokens_changed": entropy["avg_tokens_changed"],
                "avg_signal_value": entropy["avg_signal_value"],
            },
        },
        "runtime_contract": runtime_meta,
        "summary": (
            f"coverage={coverage:.2%}, standard_acc={standard['accuracy']:.4f}, "
            f"entropy_acc={entropy['accuracy']:.4f}, fixed={counts['fixed']}, "
            f"harmed={counts['harmed']}, no_effect={counts['no_effect']}"
        ),
        "per_sample": joined,
    }
    return audit, bucket_examples


def write_pilot_summary(audit: dict[str, Any]) -> None:
    summary_json = {
        "overall_recommendation": "GO" if audit["pass"] else "REFINE",
        "selected_candidate_id": "cand_bucket" if audit["pass"] else None,
        "candidates": [
            {
                "candidate_id": "cand_bucket",
                "go_no_go": audit["go_no_go"],
                "confidence": 0.78 if audit["pass"] else 0.42,
                "supported_hypotheses": ["bucket_decomposes_aggregate_gain"] if audit["pass"] else [],
                "failed_assumptions": [] if audit["pass"] else ["coverage_below_threshold"],
                "key_metrics": {
                    "coverage": audit["coverage"],
                    "accuracy_delta": audit["aggregate"]["accuracy_delta"],
                    "fixed": audit["bucket_counts"]["fixed"],
                    "harmed": audit["bucket_counts"]["harmed"],
                    "no_effect": audit["bucket_counts"]["no_effect"],
                },
                "notes": audit["summary"],
            }
        ],
    }
    write_json(RESULTS_DIR / "pilot_summary.json", summary_json)
    md = "\n".join(
        [
            "# Pilot Summary",
            "",
            f"- overall_recommendation: {summary_json['overall_recommendation']}",
            "- candidate_id: cand_bucket",
            f"- coverage: {audit['coverage']:.2%}",
            f"- accuracy_delta: {audit['aggregate']['accuracy_delta']:.4f}",
            f"- fixed/harmed/no_effect: {audit['bucket_counts']['fixed']}/{audit['bucket_counts']['harmed']}/{audit['bucket_counts']['no_effect']}",
            f"- notes: {audit['summary']}",
        ]
    )
    (RESULTS_DIR / "pilot_summary.md").write_text(md, encoding="utf-8")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()
    task_start = time.time()
    runtime_meta = runtime_contract()
    report_progress("loading_runtime_contract", runtime_meta)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    engine = BatchedDLMInference(
        model_path=MODEL_PATH,
        device="cuda",
        use_flash_attn=True,
        use_compile=bool(runtime_meta["compile_enabled"]),
    )
    samples = load_samples(engine.tokenizer, NUM_SAMPLES)
    prompt_max_len = max(len(sample["prompt_ids"]) for sample in samples)
    gpu_profile = {
        "task_id": TASK_ID,
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
        "prompt_max_len": prompt_max_len,
        "runtime_probe_safe_batch_size": runtime_meta["requested_batch_size"],
        "compile_enabled": runtime_meta["compile_enabled"],
        "attention_backend": engine_attention_backend(engine),
    }
    write_json(RESULTS_DIR / f"{TASK_ID}_gpu_profile.json", gpu_profile)

    report_progress(
        "running_standard",
        {
            "requested_batch_size": runtime_meta["requested_batch_size"],
            "prompt_max_len": prompt_max_len,
            "compile_enabled": runtime_meta["compile_enabled"],
        },
    )
    standard = run_method(engine, "standard", samples, runtime_meta["requested_batch_size"])
    write_json(RESULTS_DIR / f"{TASK_ID}_standard_worker.json", standard)

    report_progress(
        "running_entropy",
        {
            "standard_accuracy": standard["accuracy"],
            "standard_batch_size": standard["batch_size"],
            "requested_batch_size": runtime_meta["requested_batch_size"],
        },
    )
    entropy = run_method(engine, "entropy", samples, runtime_meta["requested_batch_size"])
    write_json(RESULTS_DIR / f"{TASK_ID}_entropy_worker.json", entropy)

    audit, examples = build_bucket_audit(standard, entropy, runtime_meta)
    audit["wall_clock_sec"] = round(time.time() - task_start, 2)
    write_json(RESULTS_DIR / "benefit_bucket_audit_pilot.json", audit)
    write_json(RESULTS_DIR / "benefit_bucket_examples_pilot.json", examples)
    write_json(RESULTS_DIR / f"{TASK_ID}.json", {"standard": standard, "entropy": entropy, "audit_path": "exp/results/benefit_bucket_audit_pilot.json"})
    write_pilot_summary(audit)

    report_progress(
        "completed",
        {
            "coverage": audit["coverage"],
            "accuracy_delta": audit["aggregate"]["accuracy_delta"],
            "fixed": audit["bucket_counts"]["fixed"],
            "harmed": audit["bucket_counts"]["harmed"],
            "no_effect": audit["bucket_counts"]["no_effect"],
            "standard_batch_size": standard["batch_size"],
            "entropy_batch_size": entropy["batch_size"],
        },
    )
    mark_done(
        "success",
        (
            f"{PAIR_NAME}: coverage={audit['coverage']:.2%}, "
            f"delta={audit['aggregate']['accuracy_delta']:.4f}, "
            f"fixed={audit['bucket_counts']['fixed']}, harmed={audit['bucket_counts']['harmed']}"
        ),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - remote defensive logging
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {
                "task_id": TASK_ID,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        mark_done("failed", f"{type(exc).__name__}: {exc}")
        raise
