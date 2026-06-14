#!/usr/bin/env python3
"""Implement the frozen cand_espd pilot on GSM8K."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
    dataset_gen_length,
    engine_attention_backend,
    engine_compile_enabled,
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


TASK_ID = "implement_espd"
ARM_NAME = "espd"
DATASET_NAME = "gsm8k"
GEN_LENGTH = dataset_gen_length(DATASET_NAME)
DRAFT_STEPS = 64
MASK_TOKEN_ID = 126336

ESPD_CONTRACT = {
    "variant": "cand_espd_v1",
    "active_frontier_scoring_rule": {
        "definition": "Normalized token entropy after the shared 64-step draft",
        "source": "draft_entropy",
    },
    "frontier_retention_ratio": {
        "definition": "Retain the top 12% highest-entropy draft positions as the active frontier",
        "ratio": 0.12,
    },
    "stopping_threshold": {
        "definition": "Stop frontier revision once the mean masked-frontier entropy falls to <= 85% of the draft-frontier entropy",
        "entropy_ratio": 0.85,
        "minimum_steps": 1,
    },
    "maximum_retained_steps": {
        "definition": "Run at most 3 frontier-only revision steps after the draft",
        "steps": 3,
    },
    "runtime_ledger_policy": {
        "definition": "Count frontier selection, stopping checks, and extra forwards separately from the shared draft runtime",
        "extra_forward_passes": "explicitly_logged",
        "auxiliary_overhead_sec": "explicitly_logged",
    },
}

SUMMARY_BASENAME = os.environ.get("SIBYL_SUMMARY_BASENAME", "")
FORCE_BATCH_SIZE = int(os.environ.get("SIBYL_FORCE_BATCH_SIZE", "0") or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--arm-name", default=ARM_NAME)
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


def normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    lo = values.min()
    hi = values.max()
    if float((hi - lo).item()) <= 1e-6:
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo)


def load_baseline_scores(arm_name: str) -> dict[str, dict[str, Any]]:
    path = ARMS_DIR / arm_name / f"seed{SEED}" / "per_sample_scores.jsonl"
    if not path.exists():
        return {}

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
        cand_correct = bool(row.get("correct"))
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
    for target in (True, False):
        for row in candidate_scores:
            if bool(row.get("correct")) != target:
                continue
            espd_stats = row.get("espd_stats", {}) or {}
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "difficulty_bucket": row.get("difficulty_bucket"),
                    "correct": bool(row["correct"]),
                    "predicted_answer": row.get("predicted_answer", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "tokens_changed": int(row.get("tokens_changed", 0)),
                    "active_frontier_ratio": espd_stats.get("active_frontier_ratio"),
                    "stopped_step": espd_stats.get("stopped_step"),
                    "stop_reason": espd_stats.get("stop_reason"),
                    "card84_correct": bool(
                        baseline_maps["card84"].get(row["sample_id"], {}).get("correct", False)
                    ),
                    "rand84_correct": bool(
                        baseline_maps["rand84"].get(row["sample_id"], {}).get("correct", False)
                    ),
                }
            )
            if len(examples) >= 6:
                return examples
    return examples


def write_gpu_profile(
    task_id: str,
    engine: BatchedDLMInference,
    prompt_max_len: int,
    effective_batch_size: int,
    runtime_meta: dict[str, Any],
    manifest_count: int,
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
        "effective_batch_size": effective_batch_size,
        "sample_count": manifest_count,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(engine.device) / 1024**2),
        "timestamp": now_iso(),
    }
    write_json(RESULTS_DIR / f"{task_id}_gpu_profile.json", profile)
    return profile


def run_espd_batch(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    *,
    gen_length: int,
    batch_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    del batch_seed  # deterministic once the draft is fixed

    frontier_ratio = float(ESPD_CONTRACT["frontier_retention_ratio"]["ratio"])
    stopping_ratio = float(ESPD_CONTRACT["stopping_threshold"]["entropy_ratio"])
    minimum_steps = int(ESPD_CONTRACT["stopping_threshold"]["minimum_steps"])
    max_steps = int(ESPD_CONTRACT["maximum_retained_steps"]["steps"])

    frontier_selection_start = time.perf_counter()
    draft = engine._draft_batch(prompt_ids_list, gen_length, DRAFT_STEPS, "confidence")
    device = engine.device
    draft_input = torch.stack([row["input_ids"] for row in draft], dim=0)
    draft_attn = torch.stack([row["attn_mask"] for row in draft], dim=0).long()
    gen_start = draft[0]["gen_start"]
    gen_end = draft[0]["gen_end"]
    batch_size = draft_input.shape[0]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        draft_logits = engine.model(input_ids=draft_input, attention_mask=draft_attn).logits

    draft_probs = F.softmax(draft_logits[:, gen_start:gen_end, :], dim=-1)
    draft_entropy = -(draft_probs * draft_probs.clamp_min(1e-12).log()).sum(dim=-1)
    frontier_selection_sec = time.perf_counter() - frontier_selection_start

    revised_input = draft_input.clone()
    revised_attn = draft_attn.clone()
    frontier_masks: list[torch.Tensor] = []
    baseline_frontier_entropy: list[float] = []
    stats_list: list[dict[str, Any]] = []
    nfe_per_sample = torch.tensor([row["nfe"] + 1 for row in draft], dtype=torch.long, device=device)
    stopping_overhead_sec = 0.0

    for i in range(batch_size):
        entropy = draft_entropy[i]
        score = normalize_tensor(entropy)
        frontier_size = max(1, int(round(frontier_ratio * gen_length)))
        _, top_idx = score.topk(frontier_size)
        frontier_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
        frontier_mask[top_idx] = True
        frontier_masks.append(frontier_mask)

        frontier_entropy_mean = float(entropy[frontier_mask].mean().item())
        baseline_frontier_entropy.append(frontier_entropy_mean)
        revised_input[i, gen_start:gen_end][frontier_mask] = MASK_TOKEN_ID
        stats_list.append(
            {
                "active_frontier_ratio": float(frontier_mask.float().mean().item()),
                "frontier_token_count": int(frontier_mask.sum().item()),
                "stopping_threshold": frontier_entropy_mean * stopping_ratio,
                "stopped_step": max_steps,
                "stop_reason": "max_steps",
                "retained_steps": max_steps,
                "extra_forward_passes": 0,
                "frontier_selection_overhead_sec": 0.0,
                "stopping_overhead_sec": 0.0,
            }
        )

    total_frontier_targets = torch.tensor(
        [int(mask.sum().item()) for mask in frontier_masks],
        dtype=torch.long,
        device=device,
    )
    sample_done = total_frontier_targets == 0
    current_frontier_entropy = torch.tensor(baseline_frontier_entropy, dtype=torch.float32, device=device)

    for rev_step in range(max_steps):
        if sample_done.all():
            break

        gen_region = revised_input[:, gen_start:gen_end]
        masked = gen_region == MASK_TOKEN_ID
        active = (~sample_done) & masked.any(dim=1)
        if not active.any():
            break

        active_indices = torch.where(active)[0]
        active_input = revised_input[active_indices]
        active_attn = revised_attn[active_indices]

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = engine.model(input_ids=active_input, attention_mask=active_attn).logits

        nfe_per_sample[active_indices] += 1
        for gi in active_indices.tolist():
            stats_list[gi]["extra_forward_passes"] += 1

        active_logits = outputs[:, gen_start:gen_end, :]
        active_masked = masked[active_indices]
        stopping_check_start = time.perf_counter()

        for ai, gi_t in enumerate(active_indices):
            gi = gi_t.item()
            masked_positions = torch.where(active_masked[ai])[0]
            if masked_positions.numel() == 0:
                sample_done[gi] = True
                continue

            masked_logits = active_logits[ai, masked_positions]
            probs = F.softmax(masked_logits, dim=-1)
            top1_conf, top1_token = probs.max(dim=-1)
            masked_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            mean_masked_entropy = float(masked_entropy.mean().item())
            current_frontier_entropy[gi] = mean_masked_entropy

            stop_threshold = baseline_frontier_entropy[gi] * stopping_ratio
            if rev_step + 1 >= minimum_steps and mean_masked_entropy <= stop_threshold:
                revised_input[gi, gen_start + masked_positions] = top1_token
                stats_list[gi]["stopped_step"] = rev_step + 1
                stats_list[gi]["stop_reason"] = "entropy_threshold"
                stats_list[gi]["retained_steps"] = rev_step + 1
                sample_done[gi] = True
                continue

            target_unmasked = int(round((rev_step + 1) / max_steps * total_frontier_targets[gi].item()))
            already_unmasked = total_frontier_targets[gi].item() - masked_positions.numel()
            num_to_unmask = max(1, target_unmasked - already_unmasked)
            num_to_unmask = min(num_to_unmask, masked_positions.numel())
            _, topk_idx = top1_conf.topk(num_to_unmask)
            selected_pos = masked_positions[topk_idx]
            revised_input[gi, gen_start + selected_pos] = top1_token[topk_idx]

        stopping_overhead_sec += time.perf_counter() - stopping_check_start

    remaining_masked = revised_input[:, gen_start:gen_end] == MASK_TOKEN_ID
    if remaining_masked.any():
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = engine.model(input_ids=revised_input, attention_mask=revised_attn).logits
        nfe_per_sample += 1
        for i in range(batch_size):
            masked_pos = torch.where(remaining_masked[i])[0]
            if masked_pos.numel() == 0:
                continue
            pos_logits = outputs[i, gen_start + masked_pos]
            probs = F.softmax(pos_logits, dim=-1)
            tokens = probs.argmax(dim=-1)
            revised_input[i, gen_start + masked_pos] = tokens

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        revised_logits = engine.model(input_ids=revised_input, attention_mask=revised_attn).logits
    nfe_per_sample += 1
    revised_probs = F.softmax(revised_logits[:, gen_start:gen_end, :], dim=-1)
    revised_entropy = -(revised_probs * revised_probs.clamp_min(1e-12).log()).sum(dim=-1)

    results: list[dict[str, Any]] = []
    for i in range(batch_size):
        baseline_tokens = draft_input[i, gen_start:gen_end]
        final_tokens = revised_input[i, gen_start:gen_end]
        frontier_mask = frontier_masks[i]
        frontier_entropy_after = float(revised_entropy[i][frontier_mask].mean().item())
        tokens_changed = int((final_tokens[frontier_mask] != baseline_tokens[frontier_mask]).sum().item())

        stats = stats_list[i]
        stats["frontier_selection_overhead_sec"] = frontier_selection_sec / max(1, batch_size)
        stats["stopping_overhead_sec"] = stopping_overhead_sec / max(1, batch_size)
        stats["auxiliary_overhead_sec"] = (
            stats["frontier_selection_overhead_sec"] + stats["stopping_overhead_sec"]
        )
        stats["draft_frontier_entropy"] = baseline_frontier_entropy[i]
        stats["final_frontier_entropy"] = frontier_entropy_after
        stats["extra_forward_passes"] += 1

        results.append(
            {
                "input_ids": revised_input[i].clone(),
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": int(nfe_per_sample[i].item()),
                "tokens_changed": tokens_changed,
                "entropy_stats": {
                    "draft_mean_entropy": float(draft_entropy[i].mean().item()),
                    "revised_mean_entropy": float(revised_entropy[i].mean().item()),
                    "draft_frontier_entropy": baseline_frontier_entropy[i],
                    "final_frontier_entropy": frontier_entropy_after,
                },
                "espd_stats": stats,
                "prompt_pad_offset": draft[i]["prompt_pad_offset"],
            }
        )

    ledger = {
        "frontier_selection_overhead_sec": frontier_selection_sec,
        "stopping_overhead_sec": stopping_overhead_sec,
        "auxiliary_overhead_sec": frontier_selection_sec + stopping_overhead_sec,
    }
    return results, ledger


def main() -> int:
    args = parse_args()
    task_id = args.task_id
    arm_name = args.arm_name
    arm_dir = ARMS_DIR / arm_name / f"seed{SEED}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(os.getpid()), encoding="utf-8")

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
        contract_batch = int(runtime_contract.get("runtime", {}).get("safe_batch_size", inherited_batch_size))
        preferred_batch_size = contract_batch if contract_batch > 0 else inherited_batch_size
        if FORCE_BATCH_SIZE > 0:
            preferred_batch_size = min(preferred_batch_size, FORCE_BATCH_SIZE)
        effective_batch_size = max(1, min(len(manifest), preferred_batch_size))

        report_progress(
            task_id,
            {
                "phase": "load_engine",
                "dataset": DATASET_NAME,
                "samples": len(manifest),
                "effective_batch_size": effective_batch_size,
                "draft_steps": DRAFT_STEPS,
                "max_retained_steps": ESPD_CONTRACT["maximum_retained_steps"]["steps"],
            },
        )

        engine, runtime_meta = build_engine(runtime_contract, args.model_path)
        gpu_profile = write_gpu_profile(
            task_id,
            engine,
            prompt_max_len,
            effective_batch_size,
            runtime_meta,
            len(manifest),
        )

        candidate_scores: list[dict[str, Any]] = []
        frontier_ratios: list[float] = []
        stop_steps: list[int] = []
        ledger_totals = Counter()

        for start in range(0, len(manifest), effective_batch_size):
            chunk = manifest[start:start + effective_batch_size]
            chunk_prompts = prompt_ids_list[start:start + effective_batch_size]
            report_progress(
                task_id,
                {
                    "phase": "espd_running",
                    "processed": start,
                    "dataset_total": len(manifest),
                    "batch_size": len(chunk),
                    "frontier_ratio": ESPD_CONTRACT["frontier_retention_ratio"]["ratio"],
                },
            )
            results, batch_ledger = run_espd_batch(
                engine,
                chunk_prompts,
                gen_length=GEN_LENGTH,
                batch_seed=SEED + start,
            )

            chunk_predictions: list[dict[str, Any]] = []
            chunk_scores: list[dict[str, Any]] = []
            chunk_trace: list[dict[str, Any]] = []
            for sample, result in zip(chunk, results, strict=True):
                generated_ids = result["input_ids"][result["gen_start"]:result["gen_end"]].tolist()
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                predicted_answer = extract_gsm8k_answer(generated_text)
                correct = check_gsm8k_correct(generated_text, sample["reference"])
                espd_stats = result.get("espd_stats", {}) or {}

                score_row = {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "source_index": sample["source_index"],
                    "prompt_length": sample["prompt_length"],
                    "difficulty_bucket": sample["difficulty_bucket"],
                    "nfe": int(result["nfe"]),
                    "tokens_changed": int(result["tokens_changed"]),
                    "generated_length": int(result["gen_end"] - result["gen_start"]),
                    "correct": bool(correct),
                    "predicted_answer": predicted_answer,
                    "reference_answer": sample["reference"],
                    "espd_stats": espd_stats,
                }
                chunk_predictions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "generated_text": generated_text[:4000],
                    }
                )
                chunk_scores.append(score_row)
                chunk_trace.append(
                    {
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "arm_name": arm_name,
                        "draft_steps": DRAFT_STEPS,
                        "frontier_retention_ratio": ESPD_CONTRACT["frontier_retention_ratio"]["ratio"],
                        "stopping_threshold_ratio": ESPD_CONTRACT["stopping_threshold"]["entropy_ratio"],
                        "max_retained_steps": ESPD_CONTRACT["maximum_retained_steps"]["steps"],
                        "nfe": int(result["nfe"]),
                        "tokens_changed": int(result["tokens_changed"]),
                        "espd_stats": espd_stats,
                    }
                )
                frontier_ratios.append(float(espd_stats.get("active_frontier_ratio", 0.0)))
                stop_steps.append(int(espd_stats.get("stopped_step", ESPD_CONTRACT["maximum_retained_steps"]["steps"])))

            append_jsonl(predictions_path, chunk_predictions)
            append_jsonl(scores_path, chunk_scores)
            append_jsonl(revision_trace_path, chunk_trace)
            candidate_scores.extend(chunk_scores)
            ledger_totals.update(batch_ledger)
            gc.collect()

        baseline_maps = {
            "card84": load_baseline_scores("card84"),
            "rand84": load_baseline_scores("rand84"),
        }
        comparisons = {
            "espd_vs_card84": compare_against_baseline(candidate_scores, baseline_maps["card84"]),
            "espd_vs_rand84": compare_against_baseline(candidate_scores, baseline_maps["rand84"]),
        }

        metrics = aggregate_metrics(candidate_scores)
        metrics.update(
            {
                "dataset": DATASET_NAME,
                "latency_sec": round(time.time() - started_at, 2),
                "batch_size": effective_batch_size,
                "draft_steps": DRAFT_STEPS,
                "active_frontier_ratio": round(sum(frontier_ratios) / max(1, len(frontier_ratios)), 4),
                "stopped_step_histogram": {
                    str(step): count for step, count in sorted(Counter(stop_steps).items())
                },
                "equal_compute_quality": round(metrics["accuracy"], 4),
                "equal_quality_speed": round(
                    sum(int(row["generated_length"]) for row in candidate_scores)
                    / max(1e-6, time.time() - started_at),
                    2,
                ),
            }
        )

        payload = {
            "task_id": task_id,
            "candidate_id": "cand_espd",
            "arm_name": arm_name,
            "dataset": DATASET_NAME,
            "status": "success",
            "metrics": metrics,
            "comparisons": comparisons,
            "summary": (
                f"cand_espd completed on GSM8K with accuracy={metrics['accuracy']}, "
                f"frontier_ratio={metrics['active_frontier_ratio']}, "
                f"speed={metrics['equal_quality_speed']} tok/s."
            ),
            "frozen_contract": ESPD_CONTRACT,
            "runtime": {
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "requested_compile": runtime_meta["requested_compile"],
                "requested_flash_attention": runtime_meta["requested_flash_attention"],
                "compile_error": runtime_meta["compile_error"],
                "effective_batch_size": effective_batch_size,
                "peak_vram_mb": gpu_profile["peak_vram_mb"],
            },
            "runtime_ledger": {
                "extra_forward_passes": sum(
                    int((row.get("espd_stats") or {}).get("extra_forward_passes", 0))
                    for row in candidate_scores
                ),
                "frontier_selection_overhead_sec": round(ledger_totals["frontier_selection_overhead_sec"], 4),
                "stopping_overhead_sec": round(ledger_totals["stopping_overhead_sec"], 4),
                "auxiliary_overhead_sec": round(ledger_totals["auxiliary_overhead_sec"], 4),
            },
            "examples": qualitative_examples(candidate_scores, baseline_maps),
            "timestamp": now_iso(),
        }

        summary_name = SUMMARY_BASENAME or f"{arm_name}_{DATASET_NAME}_pilot"
        write_json(RESULTS_DIR / f"{summary_name}.json", payload)
        write_json(RESULTS_DIR / f"{task_id}.json", payload)
        report_progress(
            task_id,
            {
                "phase": "done",
                "accuracy": metrics["accuracy"],
                "active_frontier_ratio": metrics["active_frontier_ratio"],
                "equal_quality_speed": metrics["equal_quality_speed"],
            },
        )
        mark_done(task_id, "success", payload["summary"])
        return 0
    except Exception as exc:  # noqa: BLE001
        failure_payload = {
            "task_id": task_id,
            "candidate_id": "cand_espd",
            "arm_name": arm_name,
            "dataset": DATASET_NAME,
            "status": "failed",
            "error": repr(exc),
            "frozen_contract": ESPD_CONTRACT,
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{task_id}.json", failure_payload)
        report_progress(
            task_id,
            {"phase": "failed", "error": repr(exc), "dataset": DATASET_NAME},
        )
        mark_done(task_id, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
