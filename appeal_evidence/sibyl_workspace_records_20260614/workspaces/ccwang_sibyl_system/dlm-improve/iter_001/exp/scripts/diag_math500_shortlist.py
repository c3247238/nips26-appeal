#!/usr/bin/env python3
"""MATH500 shortlist transfer for the cand_diag pilot."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk

from baseline_core_proxy import prompt_perturbation_scores, revise_selected_positions
from batched_dlm_utils import BatchedDLMInference
from batched_gsm8k_baselines import read_safe_batch_size, write_json
from tiger_signal_screen import batched_instability_revise


TASK_ID = "diag_math500_shortlist"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
MATH500_PATH = Path("/home/ccwang/sibyl_system/shared/datasets/math500")
GEN_LENGTH = 256
SEED = 42
REVISION_FRACTION = 0.10
REVISION_STEPS = 3
MAX_INITIAL_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--use-compile", action="store_true")
    return parser.parse_args()


def write_pid() -> None:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")


def clear_pid() -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()


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
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress: dict[str, Any] = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            final_progress = {}
    clear_pid()
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


def extract_math_answer(text: str) -> str:
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"answer\s+is\s*[:\s]*([^\n\.]+)",
        r"final\s+answer\s*[:\s]*([^\n\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def normalize_math_answer(answer: str) -> str:
    text = str(answer).strip()
    text = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text)
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "")
    text = text.replace(" ", "")
    text = text.rstrip(".")
    text = re.sub(r"^\{|\}$", "", text)
    return text


def check_math500_correct(predicted_text: str, reference_answer: str) -> bool:
    predicted = normalize_math_answer(extract_math_answer(predicted_text))
    reference = normalize_math_answer(reference_answer)
    return bool(predicted and predicted == reference)


def load_samples(tokenizer, num_samples: int) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(MATH500_PATH))
    count = min(num_samples, len(dataset))
    samples = []
    for idx in range(count):
        row = dataset[idx]
        prompt = (
            "Solve the following competition math problem carefully. "
            "Show concise reasoning and put the final answer in \\boxed{...}.\n\n"
            f"Problem: {row['problem']}\n\nSolution:"
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        samples.append(
            {
                "idx": idx,
                "prompt_ids": prompt_ids,
                "problem": row["problem"],
                "final_answer": row["answer"],
                "subject": row.get("subject"),
                "level": row.get("level"),
                "unique_id": row.get("unique_id"),
            }
        )
    return samples


def evaluate_standard_like(
    engine: BatchedDLMInference,
    method_name: str,
    mode: str,
    samples: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    prompt_ids = [sample["prompt_ids"] for sample in samples]
    if mode == "standard":
        results = engine.batched_standard_denoising(
            prompt_ids_list=prompt_ids,
            gen_length=GEN_LENGTH,
            num_steps=64,
            batch_size=batch_size,
            seed=SEED,
        )
    elif mode == "entropy":
        results = engine.batched_entropy_revise(
            prompt_ids_list=prompt_ids,
            gen_length=GEN_LENGTH,
            num_draft_steps=64,
            revision_fraction=REVISION_FRACTION,
            revision_steps=REVISION_STEPS,
            batch_size=batch_size,
            seed=SEED,
        )
    elif mode == "tiger":
        results = batched_instability_revise(
            engine=engine,
            prompt_ids_list=prompt_ids,
            gen_length=GEN_LENGTH,
            num_draft_steps=64,
            revision_fraction=REVISION_FRACTION,
            revision_steps=REVISION_STEPS,
            batch_size=batch_size,
            seed=SEED,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported mode: {mode}")

    texts = engine.decode_results(results)
    total_correct = 0
    total_nfe = 0
    per_sample = []
    total_tokens_changed = 0
    signal_scores: list[float] = []

    for sample, result, text in zip(samples, results, texts, strict=True):
        correct = check_math500_correct(text, sample["final_answer"])
        total_correct += int(correct)
        total_nfe += int(result["nfe"])
        total_tokens_changed += int(result.get("tokens_changed", 0))
        stats = result.get("entropy_stats") or result.get("instability_stats") or {}
        if stats:
            signal_scores.append(
                float(
                    stats.get(
                        "revision_mean_entropy",
                        stats.get("revision_mean_instability", stats.get("mean_entropy", 0.0)),
                    )
                )
            )
        if len(per_sample) < 20:
            per_sample.append(
                {
                    "idx": sample["idx"],
                    "correct": bool(correct),
                    "nfe": int(result["nfe"]),
                    "predicted_answer": extract_math_answer(text),
                    "reference_answer": sample["final_answer"],
                    "subject": sample["subject"],
                    "level": sample["level"],
                    "generated_text": text[:500],
                }
            )

    return {
        "method": method_name,
        "num_samples": len(samples),
        "accuracy": round(total_correct / max(len(samples), 1), 4),
        "actual_nfe": round(total_nfe / max(len(samples), 1), 2),
        "latency_sec": None,
        "tokens_per_sec": None,
        "batch_size": batch_size,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
        "attention_backend": engine_attention_backend(engine),
        "compile_enabled": engine_compile_enabled(engine),
        "avg_tokens_changed": round(total_tokens_changed / max(len(samples), 1), 2),
        "avg_signal_value": round(sum(signal_scores) / max(len(signal_scores), 1), 4),
        "per_sample": per_sample,
    }


def evaluate_core_proxy(
    engine: BatchedDLMInference,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    tokenizer = engine.tokenizer
    total_correct = 0
    total_nfe = 0
    per_sample = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(engine.device)

    for idx, sample in enumerate(samples, start=1):
        draft = engine.batched_standard_denoising(
            prompt_ids_list=[sample["prompt_ids"]],
            gen_length=GEN_LENGTH,
            num_steps=64,
            batch_size=1,
            seed=SEED + idx,
        )[0]
        prompt_len = len(sample["prompt_ids"])
        brittleness = prompt_perturbation_scores(
            engine.model,
            draft["input_ids"].unsqueeze(0),
            prompt_len,
            draft["gen_start"],
            draft["gen_end"],
        )
        k = max(1, int(round(0.10 * brittleness.numel())))
        selected = brittleness.topk(k).indices.tolist()
        revised, rev_nfe = revise_selected_positions(
            engine.model,
            draft["input_ids"].unsqueeze(0),
            draft["gen_start"],
            selected,
            revision_steps=3,
        )
        generated = tokenizer.decode(
            [token for token in revised[0, draft["gen_start"]:draft["gen_end"]].cpu().tolist() if token != 126336],
            skip_special_tokens=True,
        )
        correct = check_math500_correct(generated, sample["final_answer"])
        total_correct += int(correct)
        total_nfe += int(draft["nfe"]) + 2 + int(rev_nfe)
        if len(per_sample) < 20:
            per_sample.append(
                {
                    "idx": sample["idx"],
                    "correct": bool(correct),
                    "nfe": int(draft["nfe"]) + 2 + int(rev_nfe),
                    "selected_positions": selected[:10],
                    "brittleness_mean": round(float(brittleness.mean().item()), 4),
                    "predicted_answer": extract_math_answer(generated),
                    "reference_answer": sample["final_answer"],
                    "subject": sample["subject"],
                    "level": sample["level"],
                    "generated_text": generated[:500],
                }
            )
        if idx % 10 == 0 or idx == len(samples):
            report_progress(
                {
                    "phase": "core_proxy",
                    "processed": idx,
                    "total": len(samples),
                    "accuracy_so_far": round(total_correct / idx, 4),
                    "avg_nfe": round(total_nfe / idx, 2),
                }
            )

    elapsed = time.perf_counter() - start
    return {
        "method": "CORE-proxy-64",
        "num_samples": len(samples),
        "accuracy": round(total_correct / max(len(samples), 1), 4),
        "actual_nfe": round(total_nfe / max(len(samples), 1), 2),
        "latency_sec": round(elapsed, 2),
        "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(elapsed, 1e-6), 2),
        "batch_size": 1,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
        "attention_backend": engine_attention_backend(engine),
        "compile_enabled": engine_compile_enabled(engine),
        "per_sample": per_sample,
    }


def enrich_latency(rows: list[dict[str, Any]], total_elapsed_sec: float) -> None:
    known_latency = sum(row["latency_sec"] for row in rows if row["latency_sec"] is not None)
    remaining = max(total_elapsed_sec - known_latency, 0.0)
    missing = [row for row in rows if row["latency_sec"] is None]
    if missing:
        share = remaining / len(missing) if remaining > 0 else 0.0
        for row in missing:
            row["latency_sec"] = round(share, 2)
            row["tokens_per_sec"] = round((row["num_samples"] * GEN_LENGTH) / max(share, 1e-6), 2)


def load_gsm8k_reference_order() -> list[str]:
    payload = json.loads((RESULTS_DIR / "gsm8k_main_shortlist.json").read_text(encoding="utf-8"))
    shortlist = {method["method"]: float(method["accuracy"]) for method in payload["methods"]}
    shortlist["CORE-proxy-64"] = float(json.loads((RESULTS_DIR / "baseline_core.json").read_text(encoding="utf-8"))["accuracy"])
    methods = ["Standard-64", "CORE-proxy-64", "Entropy-Revise-64+3", "TIGER-Instability-64+3"]
    return sorted(methods, key=lambda name: (-shortlist[name], name))


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()
    start = time.perf_counter()
    engine = BatchedDLMInference(device="cuda", use_flash_attn=True, use_compile=args.use_compile)
    batch_size = max(1, min(args.batch_size or read_safe_batch_size(), MAX_INITIAL_BATCH_SIZE, args.samples))
    samples = load_samples(engine.tokenizer, args.samples)

    report_progress({"phase": "init", "samples": len(samples), "batch_size": batch_size})
    rows = []
    for method_name, mode in [
        ("Standard-64", "standard"),
        ("Entropy-Revise-64+3", "entropy"),
        ("TIGER-Instability-64+3", "tiger"),
    ]:
        torch.cuda.reset_peak_memory_stats(engine.device)
        method_start = time.perf_counter()
        row = evaluate_standard_like(engine, method_name, mode, samples, batch_size=batch_size)
        elapsed = time.perf_counter() - method_start
        row["latency_sec"] = round(elapsed, 2)
        row["tokens_per_sec"] = round((len(samples) * GEN_LENGTH) / max(elapsed, 1e-6), 2)
        rows.append(row)
        report_progress(
            {
                "phase": "method_complete",
                "method": method_name,
                "accuracy": row["accuracy"],
                "actual_nfe": row["actual_nfe"],
                "latency_sec": row["latency_sec"],
            }
        )
        gc.collect()
        torch.cuda.empty_cache()

    rows.append(evaluate_core_proxy(engine, samples))
    enrich_latency(rows, time.perf_counter() - start)
    rows.sort(key=lambda item: (-item["accuracy"], item["latency_sec"]))
    math500_order = [row["method"] for row in rows]
    gsm8k_order = load_gsm8k_reference_order()
    ranking_changed = math500_order != gsm8k_order
    clustered = (rows[0]["accuracy"] - rows[-1]["accuracy"]) <= 0.01
    same_story_replicates = rows[0]["method"] == "CORE-proxy-64" and rows[1]["accuracy"] >= rows[2]["accuracy"]
    pass_gate = ranking_changed or same_story_replicates
    summary = (
        f"MATH500 order={math500_order}; GSM8K order={gsm8k_order}; "
        f"ranking_changed={ranking_changed}, clustered={clustered}."
    )
    payload = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(samples),
        "dataset": "MATH500",
        "methods": rows,
        "gsm8k_reference_order": gsm8k_order,
        "math500_order": math500_order,
        "ranking_changed": ranking_changed,
        "clustered_within_1pp": clustered,
        "same_story_replicates": same_story_replicates,
        "pass": pass_gate,
        "summary": summary,
    }
    write_json(RESULTS_DIR / f"{TASK_ID}.json", payload)
    mark_done("success" if pass_gate else "failed", summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        error_payload = {
            "task_id": TASK_ID,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", error_payload)
        mark_done("failed", f"{type(exc).__name__}: {exc}")
        raise
