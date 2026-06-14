#!/usr/bin/env python3
"""Build the frozen BSR sham controls on the GSM8K audited slice."""

from __future__ import annotations

import argparse
import gc
import json
import random
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


TASK_ID = "build_bsr_sham_controls"
DATASET_NAME = "gsm8k"
GEN_LENGTH = dataset_gen_length(DATASET_NAME)
MASK_TOKEN_ID = 126336
DRAFT_STEPS = 64
REVISION_STEPS = 3
BASELINE_ARMS = ("card84", "rand84", "bsr_v1")

BSR_REFERENCE_CONTRACT = {
    "island_score": {
        "entropy_weight": 0.75,
        "boundary_weight": 0.25,
        "target_fraction": 0.10,
    },
    "span_merge_rule": {
        "gap_tolerance": 1,
    },
    "boundary_lock_width": {
        "width": 1,
    },
    "accept_reject_rule": {
        "min_entropy_drop": 0.02,
    },
}

CONTROL_SPECS: dict[str, dict[str, Any]] = {
    "randspan84": {
        "method": "RandSpan-84",
        "selection": "random_commit_spans",
        "boundary_lock": False,
        "description": "Match cand_bsr touched-token budget with random spans and no boundary protection.",
    },
    "entropyspan_no_boundary": {
        "method": "EntropySpan-NoBoundary",
        "selection": "entropy_spans",
        "boundary_lock": False,
        "description": "Use the same entropy-detected spans as cand_bsr, but revise the full span without boundary protection.",
    },
    "boundarylock_randomspan": {
        "method": "BoundaryLock-RandomSpan",
        "selection": "random_outer_spans",
        "boundary_lock": True,
        "description": "Keep boundary lock, but replace uncertainty-driven spans with random spans of matched length.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default=TASK_ID)
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


def build_commit_mask(
    gen_length: int,
    merged_islands: list[tuple[int, int]],
    boundary_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    revision_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
    commit_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
    for start, end in merged_islands:
        revision_mask[start:end + 1] = True
        interior_start = min(end, start + boundary_width)
        interior_end = max(start, end - boundary_width)
        if interior_start <= interior_end:
            commit_mask[interior_start:interior_end + 1] = True
    return revision_mask, commit_mask


def sample_nonoverlap_spans(gen_length: int, lengths: list[int], seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    occupied = [False] * gen_length
    spans: list[tuple[int, int]] = []

    for length in [length for length in lengths if length > 0]:
        candidates = [
            start
            for start in range(0, gen_length - length + 1)
            if not any(occupied[start:start + length])
        ]
        if not candidates:
            continue
        start = rng.choice(candidates)
        end = start + length - 1
        for idx in range(start, end + 1):
            occupied[idx] = True
        spans.append((start, end))

    return sorted(spans)


def build_entropy_reference(
    entropy: torch.Tensor,
    gen_length: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    left = torch.roll(entropy, shifts=1)
    right = torch.roll(entropy, shifts=-1)
    left[0] = entropy[0]
    right[-1] = entropy[-1]
    boundary_tension = (entropy - 0.5 * (left + right)).abs()

    entropy_norm = normalize_tensor(entropy)
    boundary_norm = normalize_tensor(boundary_tension)
    island_score = (
        BSR_REFERENCE_CONTRACT["island_score"]["entropy_weight"] * entropy_norm
        + BSR_REFERENCE_CONTRACT["island_score"]["boundary_weight"] * boundary_norm
    )
    target_fraction = float(BSR_REFERENCE_CONTRACT["island_score"]["target_fraction"])
    num_core = max(1, int(round(target_fraction * gen_length)))
    _, top_idx = island_score.topk(num_core)
    core_mask = torch.zeros(gen_length, dtype=torch.bool, device=entropy.device)
    core_mask[top_idx] = True

    raw_islands = _contiguous_islands(core_mask.detach().cpu())
    merged_islands = merge_islands(raw_islands, BSR_REFERENCE_CONTRACT["span_merge_rule"]["gap_tolerance"])
    boundary_width = int(BSR_REFERENCE_CONTRACT["boundary_lock_width"]["width"])
    outer_lengths = [end - start + 1 for start, end in merged_islands]
    commit_lengths = [
        max(0, length - (2 * boundary_width))
        for length in outer_lengths
    ]
    return merged_islands, outer_lengths, commit_lengths


def build_control_masks(
    control_name: str,
    entropy: torch.Tensor,
    gen_length: int,
    sample_index: int,
) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor, dict[str, Any]]:
    merged_islands, outer_lengths, commit_lengths = build_entropy_reference(entropy, gen_length)
    device = entropy.device
    boundary_width = int(BSR_REFERENCE_CONTRACT["boundary_lock_width"]["width"])

    if control_name == "entropyspan_no_boundary":
        spans = merged_islands
        revision_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
        for start, end in spans:
            revision_mask[start:end + 1] = True
        commit_mask = revision_mask.clone()
        span_lengths = outer_lengths
    elif control_name == "boundarylock_randomspan":
        spans = sample_nonoverlap_spans(gen_length, outer_lengths, SEED + sample_index)
        revision_mask, commit_mask = build_commit_mask(gen_length, spans, boundary_width, device)
        span_lengths = [end - start + 1 for start, end in spans]
    elif control_name == "randspan84":
        spans = sample_nonoverlap_spans(gen_length, commit_lengths, SEED + sample_index)
        revision_mask = torch.zeros(gen_length, dtype=torch.bool, device=device)
        for start, end in spans:
            revision_mask[start:end + 1] = True
        commit_mask = revision_mask.clone()
        span_lengths = [end - start + 1 for start, end in spans]
    else:  # pragma: no cover - guarded by CONTROL_SPECS
        raise ValueError(f"Unknown control_name: {control_name}")

    locked_mask = _expand_positions(revision_mask, boundary_width) & (~commit_mask)
    meta = {
        "reference_outer_span_lengths": outer_lengths,
        "reference_commit_span_lengths": [length for length in commit_lengths if length > 0],
        "selection_span_lengths": span_lengths,
        "boundary_lock_width": boundary_width if CONTROL_SPECS[control_name]["boundary_lock"] else 0,
        "selection": CONTROL_SPECS[control_name]["selection"],
        "description": CONTROL_SPECS[control_name]["description"],
        "selected_span_count": len(spans),
    }
    return spans, revision_mask, commit_mask, {"locked_mask": locked_mask, **meta}


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


def qualitative_examples(
    candidate_scores: list[dict[str, Any]],
    baseline_maps: dict[str, dict[str, dict[str, Any]]],
    control_key: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for target in (True, False):
        for row in candidate_scores:
            if bool(row.get("correct")) != target:
                continue
            sham_stats = row.get("control_stats", {}) or {}
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "difficulty_bucket": row.get("difficulty_bucket"),
                    "correct": bool(row["correct"]),
                    "predicted_answer": row.get("predicted_answer", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "tokens_changed": int(row.get("tokens_changed", 0)),
                    "control": CONTROL_SPECS[control_key]["method"],
                    "mean_span_length": sham_stats.get("mean_span_length"),
                    "selected_span_count": sham_stats.get("selected_span_count"),
                    "harmed_stable_token_proxy": sham_stats.get("harmed_stable_token_proxy"),
                    "card84_correct": bool(
                        baseline_maps["card84"].get(row["sample_id"], {}).get("correct", False)
                    ),
                    "rand84_correct": bool(
                        baseline_maps["rand84"].get(row["sample_id"], {}).get("correct", False)
                    ),
                    "bsr_v1_correct": bool(
                        baseline_maps["bsr_v1"].get(row["sample_id"], {}).get("correct", False)
                    ),
                }
            )
            if len(examples) >= 4:
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
        "draft_steps": DRAFT_STEPS,
        "revision_steps": REVISION_STEPS,
        "timestamp": now_iso(),
    }
    write_json(RESULTS_DIR / f"{task_id}_gpu_profile.json", profile)
    return profile


def run_control_batch(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    *,
    control_name: str,
    gen_length: int,
    draft_steps: int,
    revision_steps: int,
    batch_seed: int,
) -> list[dict[str, Any]]:
    del batch_seed  # deterministic once the draft is fixed

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
    selected_spans_per_sample: list[list[tuple[int, int]]] = []
    stats_list: list[dict[str, Any]] = []
    nfe_per_sample = torch.tensor([row["nfe"] + 1 for row in draft], dtype=torch.long, device=device)

    for i in range(batch_size):
        spans, revision_mask, commit_mask, meta = build_control_masks(
            control_name,
            draft_entropy[i],
            gen_length,
            sample_index=i,
        )
        revision_masks.append(revision_mask)
        commit_masks.append(commit_mask)
        selected_spans_per_sample.append(spans)
        revised_input[i, gen_start:gen_end][commit_mask] = MASK_TOKEN_ID

        selected_lengths = meta["selection_span_lengths"]
        stats_list.append(
            {
                "control_name": CONTROL_SPECS[control_name]["method"],
                "selection": meta["selection"],
                "description": meta["description"],
                "boundary_lock_width": meta["boundary_lock_width"],
                "mean_span_length": float(sum(selected_lengths) / max(1, len(selected_lengths))),
                "selected_span_count": meta["selected_span_count"],
                "touched_token_ratio": float(commit_mask.float().mean().item()),
                "locked_token_ratio": float(meta["locked_mask"].float().mean().item()),
                "reference_outer_span_lengths": meta["reference_outer_span_lengths"],
                "reference_commit_span_lengths": meta["reference_commit_span_lengths"],
                "selection_span_lengths": selected_lengths,
                "accepted_span_count": 0,
                "rejected_span_count": 0,
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
    entropy_drop_threshold = float(BSR_REFERENCE_CONTRACT["accept_reject_rule"]["min_entropy_drop"])
    for i in range(batch_size):
        baseline_tokens = draft_input[i, gen_start:gen_end].clone()
        candidate_tokens = revised_input[i, gen_start:gen_end]
        final_tokens = baseline_tokens.clone()
        commit_mask = commit_masks[i]
        revision_mask = revision_masks[i]
        selected_spans = selected_spans_per_sample[i]
        accepted = 0
        rejected = 0

        stable_mask = (~revision_mask).clone()
        harmed_stable = int((candidate_tokens[stable_mask] != baseline_tokens[stable_mask]).sum().item())

        for start, end in selected_spans:
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
        stats["accepted_span_count"] = accepted
        stats["rejected_span_count"] = rejected
        stats["harmed_stable_token_proxy"] = harmed_stable
        stats["revision_target_count"] = int(commit_mask.sum().item())
        stats["full_revision_region_count"] = int(revision_mask.sum().item())
        stats["entropy_drop_threshold"] = entropy_drop_threshold

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
                "control_stats": stats,
                "prompt_pad_offset": draft[i]["prompt_pad_offset"],
            }
        )

    return results


def summarize_control(
    control_key: str,
    candidate_scores: list[dict[str, Any]],
    baseline_maps: dict[str, dict[str, dict[str, Any]]],
    started_at: float,
    effective_batch_size: int,
) -> dict[str, Any]:
    metrics = aggregate_metrics(candidate_scores)
    metrics.update(
        {
            "dataset": DATASET_NAME,
            "latency_sec": round(time.time() - started_at, 2),
            "batch_size": effective_batch_size,
            "draft_steps": DRAFT_STEPS,
            "revision_steps": REVISION_STEPS,
            "mean_span_length": round(
                sum(float((row.get("control_stats") or {}).get("mean_span_length", 0.0)) for row in candidate_scores)
                / max(1, len(candidate_scores)),
                4,
            ),
            "mean_harmed_stable_tokens": round(
                sum(int((row.get("control_stats") or {}).get("harmed_stable_token_proxy", 0)) for row in candidate_scores)
                / max(1, len(candidate_scores)),
                4,
            ),
            "mean_touched_token_ratio": round(
                sum(float((row.get("control_stats") or {}).get("touched_token_ratio", 0.0)) for row in candidate_scores)
                / max(1, len(candidate_scores)),
                4,
            ),
        }
    )

    comparisons = {
        f"{control_key}_vs_{arm}": compare_against_baseline(candidate_scores, baseline_maps[arm])
        for arm in BASELINE_ARMS
    }
    return {
        "method": CONTROL_SPECS[control_key]["method"],
        "description": CONTROL_SPECS[control_key]["description"],
        "metrics": metrics,
        "comparisons": comparisons,
        "examples": qualitative_examples(candidate_scores, baseline_maps, control_key),
    }


def main() -> int:
    args = parse_args()
    task_id = args.task_id

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(__import__("os").getpid()), encoding="utf-8")

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

        report_progress(
            task_id,
            {
                "phase": "load_engine",
                "dataset": DATASET_NAME,
                "samples": len(manifest),
                "inherited_safe_batch_size": inherited_batch_size,
                "controls": [spec["method"] for spec in CONTROL_SPECS.values()],
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
        effective_batch_size = max(1, min(len(manifest), probed_batch_size))
        torch.cuda.reset_peak_memory_stats(engine.device)
        gc.collect()
        torch.cuda.empty_cache()

        baseline_maps = {
            arm: load_baseline_scores(arm)
            for arm in BASELINE_ARMS
        }

        control_payloads: dict[str, Any] = {}
        decomposition_table: list[dict[str, Any]] = []
        for control_key in CONTROL_SPECS:
            arm_dir = ARMS_DIR / control_key / f"seed{SEED}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = arm_dir / "predictions.jsonl"
            scores_path = arm_dir / "per_sample_scores.jsonl"
            revision_trace_path = arm_dir / "revision_trace.jsonl"
            for path in (predictions_path, scores_path, revision_trace_path):
                if path.exists():
                    path.unlink()

            candidate_scores: list[dict[str, Any]] = []
            control_started = time.time()
            processed = 0
            for chunk_start in range(0, len(manifest), effective_batch_size):
                chunk_end = min(chunk_start + effective_batch_size, len(manifest))
                chunk_rows = manifest[chunk_start:chunk_end]
                chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
                report_progress(
                    task_id,
                    {
                        "phase": f"{control_key}_running",
                        "control": CONTROL_SPECS[control_key]["method"],
                        "processed": processed,
                        "dataset_total": len(manifest),
                        "batch_size": effective_batch_size,
                    },
                )
                results = run_control_batch(
                    engine,
                    chunk_prompts,
                    control_name=control_key,
                    gen_length=GEN_LENGTH,
                    draft_steps=DRAFT_STEPS,
                    revision_steps=REVISION_STEPS,
                    batch_seed=SEED + chunk_start,
                )

                chunk_predictions: list[dict[str, Any]] = []
                chunk_scores: list[dict[str, Any]] = []
                chunk_trace: list[dict[str, Any]] = []
                for sample, result in zip(chunk_rows, results, strict=True):
                    generated_ids = result["input_ids"][result["gen_start"]:result["gen_end"]].tolist()
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    predicted_answer = extract_gsm8k_answer(generated_text)
                    correct = check_gsm8k_correct(generated_text, sample["reference"])
                    control_stats = result.get("control_stats", {}) or {}

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
                        "control_stats": control_stats,
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
                            "control": CONTROL_SPECS[control_key]["method"],
                            "draft_steps": DRAFT_STEPS,
                            "revision_steps": REVISION_STEPS,
                            "nfe": int(result["nfe"]),
                            "tokens_changed": int(result["tokens_changed"]),
                            "control_stats": control_stats,
                        }
                    )

                append_jsonl(predictions_path, chunk_predictions)
                append_jsonl(scores_path, chunk_scores)
                append_jsonl(revision_trace_path, chunk_trace)
                candidate_scores.extend(chunk_scores)
                processed += len(chunk_rows)
                gc.collect()

            control_result = summarize_control(
                control_key,
                candidate_scores,
                baseline_maps,
                started_at=control_started,
                effective_batch_size=effective_batch_size,
            )
            control_payloads[control_key] = control_result
            comparisons = control_result["comparisons"]
            decomposition_table.append(
                {
                    "method": control_result["method"],
                    "repair_count": comparisons[f"{control_key}_vs_rand84"]["fixed"],
                    "harm_count": comparisons[f"{control_key}_vs_rand84"]["harmed"],
                    "harmed_stable_tokens": control_result["metrics"]["mean_harmed_stable_tokens"],
                    "mean_span_length": control_result["metrics"]["mean_span_length"],
                    "mean_touched_token_ratio": control_result["metrics"]["mean_touched_token_ratio"],
                }
            )

        gpu_profile = write_gpu_profile(
            task_id,
            engine,
            prompt_max_len,
            inherited_batch_size,
            probed_batch_size,
            effective_batch_size,
            runtime_meta,
        )
        payload = {
            "task_id": task_id,
            "candidate_id": "shared",
            "dataset": DATASET_NAME,
            "status": "success",
            "sample_count": len(manifest),
            "summary": (
                "BSR sham controls completed on GSM8K: "
                + "; ".join(
                    f"{value['method']} acc={value['metrics']['accuracy']} vs_rand_net={value['comparisons'][f'{key}_vs_rand84']['net_repaired']}"
                    for key, value in control_payloads.items()
                )
            ),
            "frozen_reference_contract": BSR_REFERENCE_CONTRACT,
            "controls": control_payloads,
            "decomposition_table": decomposition_table,
            "runtime": {
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
                "requested_compile": runtime_meta["requested_compile"],
                "requested_flash_attention": runtime_meta["requested_flash_attention"],
                "compile_error": runtime_meta["compile_error"],
                "effective_batch_size": effective_batch_size,
                "peak_vram_mb": gpu_profile["peak_vram_mb"],
            },
            "evidence_files": {
                "manifest": str(manifest_path),
                "runtime_contract": str(SETUP_DIR / "runtime_contract.json"),
                "batch_probe": str(SETUP_DIR / "batch_probe.json"),
            },
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / "bsr_gsm8k_sham_controls.json", payload)
        write_json(RESULTS_DIR / f"{task_id}.json", payload)
        report_progress(
            task_id,
            {
                "phase": "done",
                "controls": len(control_payloads),
                "batch_size": effective_batch_size,
                "summary": payload["summary"],
            },
        )
        mark_done(task_id, "success", payload["summary"])
        return 0
    except Exception as exc:  # noqa: BLE001
        failure_payload = {
            "task_id": task_id,
            "candidate_id": "shared",
            "dataset": DATASET_NAME,
            "status": "failed",
            "error": repr(exc),
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
