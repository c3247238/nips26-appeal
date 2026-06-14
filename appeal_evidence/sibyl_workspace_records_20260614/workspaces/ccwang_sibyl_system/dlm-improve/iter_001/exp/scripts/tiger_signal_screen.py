#!/usr/bin/env python3
"""TIGER pilot screening: random vs entropy vs instability-guided revision."""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from batched_dlm_utils import BatchedDLMInference, MASK_TOKEN_ID, _left_pad_batch
from batched_gsm8k_baselines import (
    GEN_LENGTH,
    RESULTS_DIR,
    SEED,
    check_gsm8k_correct,
    extract_gsm8k_answer,
    load_samples,
    mark_done,
    read_safe_batch_size,
    report_progress,
    write_json,
)


TASK_ID = "tiger_signal_screen"
REVISION_FRACTION = 0.10
REVISION_STEPS = 3
PROMPT_PERTURB_FRACTION = 0.15
MAX_INITIAL_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile for this task. Off by default because TIGER hit CUBLAS errors on this stack.",
    )
    return parser.parse_args()


def engine_attention_backend(engine: BatchedDLMInference) -> str:
    return getattr(getattr(engine.model, "config", None), "_attn_implementation", None) or "eager_or_default"


def engine_compile_enabled(engine: BatchedDLMInference) -> bool:
    return hasattr(engine.model, "_orig_mod")


def is_retryable_runtime_error(exc: RuntimeError) -> bool:
    text = repr(exc).lower()
    markers = (
        "cuda error",
        "cublas_status_invalid_value",
        "cuda out of memory",
        "out of memory",
        "cudnn",
        "triton",
    )
    return any(marker in text for marker in markers)


def _stack_attention_inputs(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    gen_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    padded_prompts, prompt_mask, _ = _left_pad_batch(
        prompt_ids_list,
        engine.pad_token_id,
        engine.device,
    )
    max_prompt_len = padded_prompts.shape[1]
    gen_attn = torch.ones(
        (len(prompt_ids_list), gen_length),
        dtype=torch.bool,
        device=engine.device,
    )
    attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)
    return padded_prompts, attn_mask.long(), max_prompt_len, max_prompt_len + gen_length


def batched_instability_revise(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    gen_length: int,
    num_draft_steps: int,
    revision_fraction: float,
    revision_steps: int,
    batch_size: int,
    seed: int,
) -> list[dict]:
    all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]
    for chunk_start in range(0, len(prompt_ids_list), batch_size):
        chunk_prompts = prompt_ids_list[chunk_start:chunk_start + batch_size]
        draft_results = engine.batched_standard_denoising(
            prompt_ids_list=chunk_prompts,
            gen_length=gen_length,
            num_steps=num_draft_steps,
            batch_size=len(chunk_prompts),
            seed=seed + chunk_start,
        )
        draft_batch = torch.stack([item["input_ids"] for item in draft_results], dim=0)
        _, attn_mask, gen_start, gen_end = _stack_attention_inputs(engine, chunk_prompts, gen_length)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            base_logits = engine.model(
                input_ids=draft_batch,
                attention_mask=attn_mask,
            ).logits[:, gen_start:gen_end, :]

        base_probs = F.softmax(base_logits, dim=-1)
        base_conf, base_top = base_probs.max(dim=-1)

        perturbed = draft_batch.clone()
        for row_idx, prompt_ids in enumerate(chunk_prompts):
            prompt_len = len(prompt_ids)
            if prompt_len <= 0:
                continue
            prompt_start = gen_start - prompt_len
            prompt_positions = list(range(prompt_start, gen_start))
            mask_count = max(1, int(round(PROMPT_PERTURB_FRACTION * len(prompt_positions))))
            rng = random.Random(seed + chunk_start + row_idx)
            for pos in rng.sample(prompt_positions, k=min(mask_count, len(prompt_positions))):
                perturbed[row_idx, pos] = MASK_TOKEN_ID

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            perturbed_logits = engine.model(
                input_ids=perturbed,
                attention_mask=attn_mask,
            ).logits[:, gen_start:gen_end, :]

        perturbed_probs = F.softmax(perturbed_logits, dim=-1)
        perturbed_top = perturbed_probs.argmax(dim=-1)
        base_token_conf_under_perturb = perturbed_probs.gather(
            2,
            base_top.unsqueeze(-1),
        ).squeeze(-1)
        instability = (perturbed_top != base_top).float() + (
            base_conf - base_token_conf_under_perturb
        ).clamp(min=0.0)

        num_to_revise = max(1, int(round(revision_fraction * gen_length)))
        _, revision_targets = instability.topk(num_to_revise, dim=1)
        original_tokens = torch.gather(
            draft_batch[:, gen_start:gen_end],
            1,
            revision_targets,
        ).clone()

        for row_idx in range(draft_batch.shape[0]):
            draft_batch[row_idx, gen_start + revision_targets[row_idx]] = MASK_TOKEN_ID

        nfe_per_sample = torch.tensor(
            [int(item["nfe"]) + 2 for item in draft_results],
            device=engine.device,
        )
        sample_done = torch.zeros(
            draft_batch.shape[0],
            dtype=torch.bool,
            device=engine.device,
        )

        for rev_step in range(revision_steps):
            if sample_done.all():
                break

            gen_region = draft_batch[:, gen_start:gen_end]
            is_masked = gen_region == MASK_TOKEN_ID
            sample_done |= is_masked.sum(dim=1) == 0
            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = engine.model(
                    input_ids=draft_batch[active_indices],
                    attention_mask=attn_mask[active_indices],
                ).logits[:, gen_start:gen_end, :]

            nfe_per_sample[active_indices] += 1
            active_gen_region = gen_region[active_indices]
            active_is_masked = active_gen_region == MASK_TOKEN_ID

            for active_row, batch_idx in enumerate(active_indices):
                sample_masked = active_is_masked[active_row]
                n_masked = int(sample_masked.sum().item())
                if n_masked == 0:
                    sample_done[batch_idx] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = logits[active_row, masked_positions]
                probs = F.softmax(masked_logits, dim=-1)
                top_conf, top_tok = probs.max(dim=-1)
                target_unmasked = int(round((rev_step + 1) / revision_steps * num_to_revise))
                already_unmasked = num_to_revise - n_masked
                num_to_unmask = max(1, min(n_masked, target_unmasked - already_unmasked))
                _, topk_idx = top_conf.topk(min(num_to_unmask, len(top_conf)))
                fill_positions = masked_positions[topk_idx]
                draft_batch[batch_idx, gen_start + fill_positions] = top_tok[topk_idx]

        remaining_masked = draft_batch[:, gen_start:gen_end] == MASK_TOKEN_ID
        if remaining_masked.any():
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = engine.model(
                    input_ids=draft_batch,
                    attention_mask=attn_mask,
                ).logits[:, gen_start:gen_end, :]
            nfe_per_sample += 1
            for row_idx in range(draft_batch.shape[0]):
                masked_positions = torch.where(remaining_masked[row_idx])[0]
                if masked_positions.numel() == 0:
                    continue
                probs = F.softmax(logits[row_idx, masked_positions], dim=-1)
                _, top_tok = probs.max(dim=-1)
                draft_batch[row_idx, gen_start + masked_positions] = top_tok

        new_tokens = torch.gather(
            draft_batch[:, gen_start:gen_end],
            1,
            revision_targets,
        )
        tokens_changed = (new_tokens != original_tokens).sum(dim=1)

        for row_idx in range(draft_batch.shape[0]):
            scores = instability[row_idx]
            selected_scores = scores[revision_targets[row_idx]]
            all_results[chunk_start + row_idx] = {
                "input_ids": draft_batch[row_idx],
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": int(nfe_per_sample[row_idx].item()),
                "tokens_changed": int(tokens_changed[row_idx].item()),
                "instability_stats": {
                    "mean_instability": float(scores.mean().item()),
                    "max_instability": float(scores.max().item()),
                    "revision_mean_instability": float(selected_scores.mean().item()),
                    "num_revised": num_to_revise,
                    "tokens_changed": int(tokens_changed[row_idx].item()),
                },
            }
    return all_results


def evaluate_method(
    engine: BatchedDLMInference,
    method_name: str,
    mode: str,
    samples: list[dict],
    batch_size: int,
) -> dict:
    total_correct = 0
    total_nfe = 0
    total_tokens_changed = 0
    signal_scores = []
    per_sample = []
    method_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(engine.device)

    for chunk_start in range(0, len(samples), batch_size):
        chunk = samples[chunk_start:chunk_start + batch_size]
        prompts = [sample["prompt_ids"] for sample in chunk]

        if mode == "random":
            chunk_results = engine.batched_random_revise(
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=64,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=batch_size,
                seed=SEED,
            )
        elif mode == "entropy":
            chunk_results = engine.batched_entropy_revise(
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=64,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=batch_size,
                seed=SEED,
            )
        elif mode == "instability":
            chunk_results = batched_instability_revise(
                engine=engine,
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=64,
                revision_fraction=REVISION_FRACTION,
                revision_steps=REVISION_STEPS,
                batch_size=batch_size,
                seed=SEED,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        chunk_texts = engine.decode_results(chunk_results)
        for sample, result, generated_text in zip(chunk, chunk_results, chunk_texts, strict=True):
            is_correct = check_gsm8k_correct(generated_text, sample["final_answer"])
            total_correct += int(is_correct)
            total_nfe += int(result["nfe"])
            total_tokens_changed += int(result.get("tokens_changed", 0))
            stats = result.get("instability_stats") or result.get("entropy_stats") or {}
            if stats:
                signal_scores.append(
                    float(
                        stats.get(
                            "revision_mean_instability",
                            stats.get(
                                "revision_mean_entropy",
                                stats.get("mean_entropy", 0.0),
                            ),
                        ),
                    ),
                )
            per_sample.append(
                {
                    "idx": sample["idx"],
                    "correct": bool(is_correct),
                    "nfe": int(result["nfe"]),
                    "tokens_changed": int(result.get("tokens_changed", 0)),
                    "signal_stats": stats,
                    "predicted_answer": extract_gsm8k_answer(generated_text),
                    "reference_answer": sample["final_answer"],
                    "generated_text": generated_text[:500],
                }
            )

        processed = min(chunk_start + len(chunk), len(samples))
        elapsed = time.perf_counter() - method_start
        report_progress(
            TASK_ID,
            {"random": 1, "entropy": 2, "instability": 3}[mode],
            3,
            processed,
            len(samples),
            {
                "method": method_name,
                "accuracy_so_far": round(total_correct / max(1, processed), 4),
                "avg_nfe": round(total_nfe / max(1, processed), 2),
                "avg_tokens_changed": round(total_tokens_changed / max(1, processed), 2),
                "batch_size": batch_size,
                "elapsed_sec": round(elapsed, 2),
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
            },
        )

    latency_sec = time.perf_counter() - method_start
    accuracy = total_correct / max(1, len(samples))
    avg_nfe = total_nfe / max(1, len(samples))
    avg_tokens_changed = total_tokens_changed / max(1, len(samples))
    tokens_per_sec = (len(samples) * GEN_LENGTH) / max(latency_sec, 1e-6)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)

    return {
        "method": method_name,
        "num_samples": len(samples),
        "accuracy": round(accuracy, 4),
        "actual_nfe": round(avg_nfe, 2),
        "latency_sec": round(latency_sec, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "batch_size": batch_size,
        "peak_vram_mb": peak_vram_mb,
        "attention_backend": engine_attention_backend(engine),
        "compile_enabled": engine_compile_enabled(engine),
        "avg_tokens_changed": round(avg_tokens_changed, 2),
        "avg_signal_value": round(sum(signal_scores) / max(1, len(signal_scores)), 4),
        "per_sample": per_sample[:10],
    }


def run_method_with_backoff(
    engine: BatchedDLMInference,
    method_name: str,
    mode: str,
    samples: list[dict],
    initial_batch_size: int,
) -> tuple[dict, int]:
    current_batch_size = initial_batch_size
    last_error: RuntimeError | None = None
    method_epoch = {"random": 1, "entropy": 2, "instability": 3}[mode]

    while current_batch_size >= 1:
        try:
            return (
                evaluate_method(engine, method_name, mode, samples, current_batch_size),
                current_batch_size,
            )
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_runtime_error(exc) or current_batch_size == 1:
                raise
            next_batch_size = max(1, current_batch_size // 2)
            report_progress(
                TASK_ID,
                method_epoch,
                3,
                0,
                len(samples),
                {
                    "phase": "batch_backoff",
                    "method": method_name,
                    "batch_size": current_batch_size,
                    "retry_batch_size": next_batch_size,
                    "attention_backend": engine_attention_backend(engine),
                    "compile_enabled": engine_compile_enabled(engine),
                    "error": repr(exc)[:240],
                },
            )
            current_batch_size = next_batch_size
            torch.cuda.empty_cache()
            gc.collect()

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{method_name} batch backoff failed without a captured exception")


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    engine = None
    try:
        requested_batch_size = args.batch_size if args.batch_size > 0 else read_safe_batch_size()
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        engine = BatchedDLMInference(
            device="cuda",
            use_flash_attn=True,
            use_compile=args.use_compile,
        )
        samples = load_samples(engine.tokenizer, args.samples)
        batch_size = min(requested_batch_size, MAX_INITIAL_BATCH_SIZE, len(samples))
        report_progress(
            TASK_ID,
            0,
            3,
            0,
            len(samples),
            {
                "phase": "loaded",
                "batch_size": batch_size,
                "safe_batch_size_reference": requested_batch_size,
                "attention_backend": engine_attention_backend(engine),
                "compile_enabled": engine_compile_enabled(engine),
            },
        )

        methods = []
        current_batch_size = batch_size
        for method_name, mode in [
            ("Random-Revise-64+3", "random"),
            ("Entropy-Revise-64+3", "entropy"),
            ("TIGER-Instability-64+3", "instability"),
        ]:
            payload, current_batch_size = run_method_with_backoff(
                engine,
                method_name,
                mode,
                samples,
                current_batch_size,
            )
            methods.append(payload)

        by_name = {item["method"]: item for item in methods}
        entropy_acc = by_name["Entropy-Revise-64+3"]["accuracy"]
        tiger_acc = by_name["TIGER-Instability-64+3"]["accuracy"]
        random_acc = by_name["Random-Revise-64+3"]["accuracy"]
        attention_backend = engine_attention_backend(engine)
        compile_enabled = engine_compile_enabled(engine)
        output = {
            "task_id": TASK_ID,
            "summary_name": "TIGER signal screening",
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "gen_length": GEN_LENGTH,
            "config": {
                "batch_size": current_batch_size,
                "safe_batch_size_reference": requested_batch_size,
                "revision_fraction": REVISION_FRACTION,
                "revision_steps": REVISION_STEPS,
                "prompt_perturb_fraction": PROMPT_PERTURB_FRACTION,
                "attention_backend": attention_backend,
                "compile_enabled": compile_enabled,
                "seed": SEED,
            },
            "methods": methods,
            "decision": {
                "entropy_acc": entropy_acc,
                "random_acc": random_acc,
                "tiger_acc": tiger_acc,
                "delta_vs_entropy": round(tiger_acc - entropy_acc, 4),
                "delta_vs_random": round(tiger_acc - random_acc, 4),
                "passes_gate": bool(tiger_acc >= entropy_acc + 0.01),
            },
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", output)
        write_json(
            RESULTS_DIR / f"{TASK_ID}_gpu_profile.json",
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "used_batch_size": current_batch_size,
                "peak_vram_mb": max(item["peak_vram_mb"] for item in methods),
                "attention_backend": attention_backend,
                "compile_enabled": compile_enabled,
            },
        )
        mark_done(
            TASK_ID,
            "success",
            f"random={random_acc:.4f}, entropy={entropy_acc:.4f}, tiger={tiger_acc:.4f}",
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {
                "task_id": TASK_ID,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": repr(exc),
            },
        )
        mark_done(TASK_ID, "failed", repr(exc))
        return 1
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
