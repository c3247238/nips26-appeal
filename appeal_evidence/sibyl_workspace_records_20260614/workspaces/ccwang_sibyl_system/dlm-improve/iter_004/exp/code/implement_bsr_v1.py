#!/usr/bin/env python3
"""Implement the frozen cand_bsr_v1 pilot on GSM8K."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from batched_dlm_utils import BatchedDLMInference
from batched_dlm_utils_mgcd import _contiguous_islands, _expand_positions


TASK_ID = "implement_bsr_v1"
ARM_NAME = "bsr_v1"
DATASET_NAME = "gsm8k"
BASELINE_ARMS = ("card84", "rand84")
GEN_LENGTH = dataset_gen_length(DATASET_NAME)
MASK_TOKEN_ID = 126336
DRAFT_STEPS = 64

BSR_CONTRACT = {
    "variant": "cand_bsr_v1",
    "island_score": {
        "definition": "0.75 * normalized entropy + 0.25 * normalized boundary tension",
        "entropy_weight": 0.75,
        "boundary_weight": 0.25,
        "target_fraction": 0.10,
    },
    "span_merge_rule": {
        "definition": "Merge contiguous high-score tokens when the gap between islands is <= 1 token",
        "gap_tolerance": 1,
    },
    "boundary_lock_width": {
        "definition": "Lock a 1-token halo on both sides of each island; revise only the interior positions",
        "width": 1,
    },
    "max_revision_steps": {
        "definition": "Run at most 3 local revision steps after the draft",
        "steps": 3,
    },
    "accept_reject_rule": {
        "definition": "Accept an island iff interior mean entropy improves by at least 0.02; otherwise revert to the draft tokens",
        "min_entropy_drop": 0.02,
    },
    "forbidden_auxiliary_compute": [
        "extra draft branch",
        "external verifier",
        "side search",
        "uplift estimator",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--arm-name", default=ARM_NAME)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--probe-hi", type=int, default=192)
    parser.add_argument("--probe-margin", type=float, default=0.95)
    return parser.parse_args()


def source_root_candidates() -> list[Path]:
    project_root = RESULTS_DIR.parents[1]
    candidates = [
        project_root / "current" / "exp" / "pilot_evidence_closure_v1",
        project_root / "exp" / "pilot_evidence_closure_v1",
        project_root / "iter_004" / "exp" / "pilot_evidence_closure_v1",
        project_root / "iter_003" / "exp" / "pilot_evidence_closure_v1",
    ]
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_manifest_payload() -> tuple[Any, Path]:
    manifest_path = first_existing(
        [root / "setup" / "sample_manifest.json" for root in source_root_candidates()]
    )
    if manifest_path is None:
        raise FileNotFoundError("sample_manifest.json not found in any pilot_evidence_closure_v1/setup root")
    return load_json(manifest_path), manifest_path


def load_baseline_scores(arm_name: str) -> dict[str, dict[str, Any]]:
    score_path = first_existing(
        [root / "arms" / arm_name / f"seed{SEED}" / "per_sample_scores.jsonl" for root in source_root_candidates()]
    )
    if score_path is None:
        raise FileNotFoundError(f"baseline scores missing for {arm_name}")

    scores: dict[str, dict[str, Any]] = {}
    with score_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == DATASET_NAME:
                scores[row["sample_id"]] = row
    return scores


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


def merge_islands(islands: list[tuple[int, int]], gap_tolerance: int) -> list[tuple[int, int]]:
    if not islands:
        return []
    merged: list[list[int]] = [[islands[0][0], islands[0][1]]]
    for start, end in islands[1:]:
        prev = merged[-1]
        gap = start - prev[1] - 1
        if gap <= gap_tolerance:
            prev[1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_commit_mask(gen_length: int, merged_islands: list[tuple[int, int]], boundary_width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    revision_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
    commit_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
    for start, end in merged_islands:
        revision_mask[start:end + 1] = True
        interior_start = min(end, start + boundary_width)
        interior_end = max(start, end - boundary_width)
        if interior_start <= interior_end:
            commit_mask[interior_start:interior_end + 1] = True
    return revision_mask, commit_mask


def compare_against_baseline(candidate_scores: list[dict[str, Any]], baseline_scores: dict[str, dict[str, Any]]) -> dict[str, int]:
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


def qualitative_examples(candidate_scores: list[dict[str, Any]], baseline_maps: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for target in (True, False):
        for row in candidate_scores:
            if bool(row.get("correct")) != target:
                continue
            bsr_stats = row.get("bsr_stats", {}) or {}
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "difficulty_bucket": row.get("difficulty_bucket"),
                    "correct": bool(row["correct"]),
                    "predicted_answer": row.get("predicted_answer", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "tokens_changed": int(row.get("tokens_changed", 0)),
                    "mean_span_length": bsr_stats.get("mean_span_length"),
                    "accepted_islands": bsr_stats.get("accepted_island_count"),
                    "rejected_islands": bsr_stats.get("rejected_island_count"),
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
        "draft_steps": DRAFT_STEPS,
        "timestamp": now_iso(),
    }
    write_json(RESULTS_DIR / f"{task_id}_gpu_profile.json", profile)
    return profile


def run_bsr_batch(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    *,
    gen_length: int,
    draft_steps: int,
    revision_steps: int,
    batch_seed: int,
) -> list[dict[str, Any]]:
    del batch_seed  # deterministic draft path, no extra stochastic branch

    draft = engine._draft_batch(prompt_ids_list, gen_length, draft_steps, "confidence")
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

    revised_input = draft_input.clone()
    revised_attn = draft_attn.clone()
    revision_masks: list[torch.Tensor] = []
    commit_masks: list[torch.Tensor] = []
    merged_island_lists: list[list[tuple[int, int]]] = []
    stats_list: list[dict[str, Any]] = []
    nfe_per_sample = torch.tensor([row["nfe"] + 1 for row in draft], dtype=torch.long, device=device)

    for i in range(batch_size):
        entropy = draft_entropy[i]
        left = torch.roll(entropy, shifts=1)
        right = torch.roll(entropy, shifts=-1)
        left[0] = entropy[0]
        right[-1] = entropy[-1]
        boundary_tension = (entropy - 0.5 * (left + right)).abs()

        entropy_norm = normalize_tensor(entropy)
        boundary_norm = normalize_tensor(boundary_tension)
        island_score = (
            BSR_CONTRACT["island_score"]["entropy_weight"] * entropy_norm
            + BSR_CONTRACT["island_score"]["boundary_weight"] * boundary_norm
        )

        target_fraction = float(BSR_CONTRACT["island_score"]["target_fraction"])
        num_core = max(1, int(round(target_fraction * gen_length)))
        _, top_idx = island_score.topk(num_core)
        core_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
        core_mask[top_idx] = True

        raw_islands = _contiguous_islands(core_mask.detach().cpu())
        merged_islands = merge_islands(raw_islands, BSR_CONTRACT["span_merge_rule"]["gap_tolerance"])
        revision_mask, commit_mask = build_commit_mask(
            gen_length,
            merged_islands,
            BSR_CONTRACT["boundary_lock_width"]["width"],
            device,
        )

        revision_masks.append(revision_mask)
        commit_masks.append(commit_mask)
        merged_island_lists.append(merged_islands)

        revised_input[i, gen_start:gen_end][commit_mask] = MASK_TOKEN_ID
        span_lengths = [end - start + 1 for start, end in merged_islands]
        locked_mask = _expand_positions(revision_mask, BSR_CONTRACT["boundary_lock_width"]["width"]) & (~commit_mask)
        stats_list.append(
            {
                "island_score": {
                    "target_fraction": target_fraction,
                    "mean_score": float(island_score.mean().item()),
                    "max_score": float(island_score.max().item()),
                },
                "span_merge_rule": {
                    "gap_tolerance": BSR_CONTRACT["span_merge_rule"]["gap_tolerance"],
                    "raw_island_count": len(raw_islands),
                    "merged_island_count": len(merged_islands),
                },
                "boundary_lock_width": BSR_CONTRACT["boundary_lock_width"]["width"],
                "max_revision_steps": revision_steps,
                "accept_reject_rule": {
                    "min_entropy_drop": BSR_CONTRACT["accept_reject_rule"]["min_entropy_drop"],
                },
                "mean_span_length": float(sum(span_lengths) / max(1, len(span_lengths))),
                "touched_token_ratio": float(commit_mask.float().mean().item()),
                "locked_token_ratio": float(locked_mask.float().mean().item()),
                "accepted_island_count": 0,
                "rejected_island_count": 0,
                "harmed_stable_token_proxy": 0,
            }
        )

    total_revision_targets = torch.tensor(
        [int(mask.sum().item()) for mask in commit_masks],
        dtype=torch.long,
        device=device,
    )
    sample_done = total_revision_targets == 0

    for rev_step in range(revision_steps):
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
        active_logits = outputs[:, gen_start:gen_end, :]
        active_masked = masked[active_indices]

        for ai, gi_t in enumerate(active_indices):
            gi = gi_t.item()
            masked_positions = torch.where(active_masked[ai])[0]
            if masked_positions.numel() == 0:
                sample_done[gi] = True
                continue
            masked_logits = active_logits[ai, masked_positions]
            probs = F.softmax(masked_logits, dim=-1)
            top1_conf, top1_token = probs.max(dim=-1)
            target_unmasked = int(round((rev_step + 1) / max(1, revision_steps) * total_revision_targets[gi].item()))
            already_unmasked = total_revision_targets[gi].item() - masked_positions.numel()
            num_to_unmask = max(1, target_unmasked - already_unmasked)
            num_to_unmask = min(num_to_unmask, masked_positions.numel())
            _, topk_idx = top1_conf.topk(num_to_unmask)
            selected_pos = masked_positions[topk_idx]
            revised_input[gi, gen_start + selected_pos] = top1_token[topk_idx]

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
            tokens = F.softmax(pos_logits, dim=-1).argmax(dim=-1)
            revised_input[i, gen_start + masked_pos] = tokens

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        revised_logits = engine.model(input_ids=revised_input, attention_mask=revised_attn).logits
    nfe_per_sample += 1
    revised_probs = F.softmax(revised_logits[:, gen_start:gen_end, :], dim=-1)
    revised_entropy = -(revised_probs * revised_probs.clamp_min(1e-12).log()).sum(dim=-1)

    results: list[dict[str, Any]] = []
    entropy_drop_threshold = float(BSR_CONTRACT["accept_reject_rule"]["min_entropy_drop"])
    for i in range(batch_size):
        baseline_tokens = draft_input[i, gen_start:gen_end].clone()
        candidate_tokens = revised_input[i, gen_start:gen_end]
        final_tokens = baseline_tokens.clone()
        commit_mask = commit_masks[i]
        revision_mask = revision_masks[i]
        merged_islands = merged_island_lists[i]
        accepted = 0
        rejected = 0

        stable_mask = (~revision_mask).clone()
        harmed_stable = int((candidate_tokens[stable_mask] != baseline_tokens[stable_mask]).sum().item())

        for start, end in merged_islands:
            island_commit_mask = commit_mask[start:end + 1]
            if not island_commit_mask.any():
                continue
            baseline_slice = draft_entropy[i, start:end + 1][island_commit_mask]
            revised_slice = revised_entropy[i, start:end + 1][island_commit_mask]
            entropy_drop = float((baseline_slice.mean() - revised_slice.mean()).item())
            if entropy_drop >= entropy_drop_threshold:
                positions = torch.arange(start, end + 1, device=device)[island_commit_mask]
                final_tokens[positions] = candidate_tokens[positions]
                accepted += 1
            else:
                rejected += 1

        tokens_changed = int((final_tokens[commit_mask] != baseline_tokens[commit_mask]).sum().item())
        final_input = draft_input[i].clone()
        final_input[gen_start:gen_end] = final_tokens

        stats = stats_list[i]
        stats["accepted_island_count"] = accepted
        stats["rejected_island_count"] = rejected
        stats["harmed_stable_token_proxy"] = harmed_stable
        stats["revision_target_count"] = int(commit_mask.sum().item())
        stats["full_revision_region_count"] = int(revision_mask.sum().item())

        results.append(
            {
                "input_ids": final_input,
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": int(nfe_per_sample[i].item()),
                "tokens_changed": tokens_changed,
                "entropy_stats": {
                    "draft_mean_entropy": float(draft_entropy[i].mean().item()),
                    "revised_mean_entropy": float(revised_entropy[i].mean().item()),
                    "entropy_drop_threshold": entropy_drop_threshold,
                },
                "bsr_stats": stats,
                "prompt_pad_offset": draft[i]["prompt_pad_offset"],
            }
        )

    return results


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
    contract_path = arm_dir / "frozen_contract.json"
    for path in (predictions_path, scores_path, revision_trace_path):
        if path.exists():
            path.unlink()

    started_at = time.time()
    try:
        manifest_payload, manifest_path = load_manifest_payload()
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
        write_json(contract_path, BSR_CONTRACT)

        report_progress(
            task_id,
            {
                "phase": "load_engine",
                "dataset": DATASET_NAME,
                "samples": len(manifest),
                "inherited_safe_batch_size": inherited_batch_size,
                "contract_variant": BSR_CONTRACT["variant"],
            },
        )
        engine, runtime_meta = build_engine(runtime_contract, args.model_path)

        report_progress(
            task_id,
            {
                "phase": "probe_batch_size",
                "dataset": DATASET_NAME,
                "samples": len(manifest),
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
        current_batch_size = max(1, min(len(manifest), probed_batch_size))
        torch.cuda.reset_peak_memory_stats(engine.device)
        gc.collect()
        torch.cuda.empty_cache()

        candidate_scores: list[dict[str, Any]] = []
        total_processed = 0
        oom_retries = 0
        loop_started_at = time.time()
        chunk_start = 0

        while chunk_start < len(manifest):
            chunk_end = min(chunk_start + current_batch_size, len(manifest))
            chunk_rows = manifest[chunk_start:chunk_end]
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            report_progress(
                task_id,
                {
                    "phase": "gsm8k_running",
                    "processed": total_processed,
                    "dataset_total": len(manifest),
                    "batch_size": current_batch_size,
                    "attention_backend": engine_attention_backend(engine),
                    "compile_enabled": engine_compile_enabled(engine),
                },
            )
            try:
                outputs = run_bsr_batch(
                    engine,
                    chunk_prompts,
                    gen_length=GEN_LENGTH,
                    draft_steps=DRAFT_STEPS,
                    revision_steps=BSR_CONTRACT["max_revision_steps"]["steps"],
                    batch_seed=SEED + chunk_start,
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
            for sample, result, text in zip(chunk_rows, outputs, generated_texts, strict=True):
                generated_text = text[:4000]
                predicted_answer = extract_gsm8k_answer(generated_text)
                correct = check_gsm8k_correct(generated_text, sample["reference"])
                score_row = {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "source_index": sample["source_index"],
                    "prompt_length": sample["prompt_length"],
                    "difficulty_bucket": sample["difficulty_bucket"],
                    "nfe": int(result["nfe"]),
                    "tokens_changed": int(result.get("tokens_changed", 0)),
                    "generated_length": int(result["gen_end"] - result["gen_start"]),
                    "correct": bool(correct),
                    "predicted_answer": predicted_answer,
                    "reference_answer": sample["reference"],
                    "entropy_stats": result.get("entropy_stats", {}) or {},
                    "bsr_stats": result.get("bsr_stats", {}) or {},
                }
                prediction_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "generated_text": generated_text,
                    }
                )
                revision_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "arm_name": arm_name,
                        "method": "bsr_v1",
                        "draft_steps": DRAFT_STEPS,
                        "max_revision_steps": BSR_CONTRACT["max_revision_steps"]["steps"],
                        "contract": BSR_CONTRACT,
                        "nfe": int(result["nfe"]),
                        "tokens_changed": int(result.get("tokens_changed", 0)),
                        "bsr_stats": result.get("bsr_stats", {}) or {},
                        "entropy_stats": result.get("entropy_stats", {}) or {},
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
                "max_revision_steps": BSR_CONTRACT["max_revision_steps"]["steps"],
            }
        )

        baseline_maps = {arm: load_baseline_scores(arm) for arm in BASELINE_ARMS}
        comparisons = {
            f"bsr_v1_vs_{arm}": compare_against_baseline(candidate_scores, baseline_maps[arm])
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
            "candidate_id": "cand_bsr",
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
            "frozen_contract": BSR_CONTRACT,
            "comparisons": comparisons,
            "qualitative_examples": qualitative_examples(candidate_scores, baseline_maps),
            "evidence_files": {
                "manifest": str(manifest_path),
                "runtime_contract": str(SETUP_DIR / "runtime_contract.json"),
                "batch_probe": str(SETUP_DIR / "batch_probe.json"),
                "frozen_contract": str(contract_path),
            },
            "gpu_profile_path": str(RESULTS_DIR / f"{task_id}_gpu_profile.json"),
            "elapsed_min": round((time.time() - started_at) / 60, 2),
            "timestamp": now_iso(),
        }

        write_json(arm_dir / "metrics.json", metrics_payload)
        write_json(
            arm_dir / "gpu_progress.json",
            {
                "task_id": task_id,
                "arm_name": arm_name,
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
            f"vs_card84_net={comparisons['bsr_v1_vs_card84']['net_repaired']} "
            f"vs_rand84_net={comparisons['bsr_v1_vs_rand84']['net_repaired']} "
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
                "accepted_island_mean": round(
                    sum(row["bsr_stats"]["accepted_island_count"] for row in candidate_scores) / max(1, len(candidate_scores)),
                    2,
                ),
            },
        )
        mark_done(task_id, "success", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": task_id,
            "candidate_id": "cand_bsr",
            "arm_name": arm_name,
            "dataset": DATASET_NAME,
            "status": "failed",
            "error": repr(exc),
            "frozen_contract": BSR_CONTRACT,
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{task_id}.json", failure)
        report_progress(task_id, {"phase": "failed", "error": repr(exc), "dataset": DATASET_NAME})
        mark_done(task_id, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
