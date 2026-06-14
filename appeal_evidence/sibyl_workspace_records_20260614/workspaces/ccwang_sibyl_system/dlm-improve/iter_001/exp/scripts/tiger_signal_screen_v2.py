#!/usr/bin/env python3
"""Stable TIGER pilot runner.

This script intentionally avoids the legacy full_llada_gsm8k path and uses the
already validated batched_dlm_utils runtime only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from batched_dlm_utils import MASK_TOKEN_ID, BatchedDLMInference, _left_pad_batch, cosine_schedule

PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
SETUP_PATH = RESULTS_DIR / "setup_throughput_verification.json"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
GSM8K_TEST_PATH = "/home/ccwang/sibyl_system/shared/datasets/gsm8k"
TASK_ID = "tiger_signal_screen"
GEN_LENGTH = 256
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--draft-steps", type=int, default=64)
    parser.add_argument("--revision-fraction", type=float, default=0.10)
    parser.add_argument("--revision-steps", type=int, default=3)
    parser.add_argument("--perturb-frac", type=float, default=0.15)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def report_progress(epoch: int, total_epochs: int, step: int, total_steps: int, metric: dict) -> None:
    write_json(
        RESULTS_DIR / f"{TASK_ID}_PROGRESS.json",
        {
            "task_id": TASK_ID,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "loss": None,
            "metric": metric,
            "updated_at": datetime.now().isoformat(),
        },
    )


def mark_done(status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()
    final_progress = {}
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
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


def read_safe_batch_size() -> int:
    if not SETUP_PATH.exists():
        return 16
    try:
        payload = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
        compile_on = payload["checks"].get("throughput_compile_on") or {}
        compile_off = payload["checks"].get("throughput_compile_off") or {}
        safe_batch = compile_on.get("safe_batch_size") or compile_off.get("safe_batch_size") or 16
        return max(1, int(safe_batch))
    except (OSError, ValueError, KeyError, TypeError):
        return 16


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


def draft_fill(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    gen_length: int,
    num_draft_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, list[int], torch.Tensor]:
    device = engine.device
    batch_size = len(prompt_ids_list)
    padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(prompt_ids_list, engine.pad_token_id, device)
    max_prompt_len = padded_prompts.shape[1]
    gen_masks = torch.full((batch_size, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    input_ids = torch.cat([padded_prompts, gen_masks], dim=1)
    gen_attn = torch.ones((batch_size, gen_length), dtype=torch.bool, device=device)
    attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)
    gen_start = max_prompt_len
    gen_end = max_prompt_len + gen_length
    nfe_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
    sample_done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(num_draft_steps):
        if sample_done.all():
            break
        frac_prev = cosine_schedule(step, num_draft_steps) if step > 0 else 0.0
        frac_curr = cosine_schedule(step + 1, num_draft_steps)
        gen_region = input_ids[:, gen_start:gen_end]
        is_masked = gen_region == MASK_TOKEN_ID
        newly_done = (is_masked.sum(dim=1) == 0) & (~sample_done)
        sample_done |= newly_done
        active = ~sample_done
        if not active.any():
            break
        active_indices = torch.where(active)[0]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = engine.model(
                input_ids=input_ids[active_indices],
                attention_mask=attn_mask[active_indices].long(),
            ).logits[:, gen_start:gen_end, :]
        nfe_per_sample[active_indices] += 1
        active_gen_region = gen_region[active_indices]
        active_is_masked = active_gen_region == MASK_TOKEN_ID
        for ai, gi_t in enumerate(active_indices):
            gi = gi_t.item()
            sample_masked = active_is_masked[ai]
            n_masked = int(sample_masked.sum().item())
            if n_masked == 0:
                sample_done[gi] = True
                continue
            masked_positions = torch.where(sample_masked)[0]
            masked_logits = logits[ai, masked_positions]
            probs = F.softmax(masked_logits, dim=-1)
            top1_conf, top1_token = probs.max(dim=-1)
            num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
            num_to_unmask = min(num_to_unmask, n_masked)
            _, topk_idx = top1_conf.topk(num_to_unmask)
            pos_to_unmask = masked_positions[topk_idx]
            input_ids[gi, gen_start + pos_to_unmask] = top1_token[topk_idx]
    return input_ids, attn_mask, gen_start, gen_end, prompt_lens, nfe_per_sample


def revise_selected(
    engine: BatchedDLMInference,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    gen_start: int,
    gen_end: int,
    selected_positions: torch.Tensor,
    revision_steps: int,
    nfe_per_sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_revise = selected_positions.shape
    original_tokens = torch.gather(input_ids[:, gen_start:gen_end], 1, selected_positions).clone()
    for i in range(batch_size):
        input_ids[i, gen_start + selected_positions[i]] = MASK_TOKEN_ID

    sample_done = torch.zeros(batch_size, dtype=torch.bool, device=engine.device)
    for rev_step in range(revision_steps):
        if sample_done.all():
            break
        gen_region = input_ids[:, gen_start:gen_end]
        is_masked = gen_region == MASK_TOKEN_ID
        newly_done = (is_masked.sum(dim=1) == 0) & (~sample_done)
        sample_done |= newly_done
        active = ~sample_done
        if not active.any():
            break
        active_indices = torch.where(active)[0]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = engine.model(
                input_ids=input_ids[active_indices],
                attention_mask=attn_mask[active_indices].long(),
            ).logits[:, gen_start:gen_end, :]
        nfe_per_sample[active_indices] += 1
        active_gen_region = gen_region[active_indices]
        active_is_masked = active_gen_region == MASK_TOKEN_ID
        for ai, gi_t in enumerate(active_indices):
            gi = gi_t.item()
            sample_masked = active_is_masked[ai]
            n_masked = int(sample_masked.sum().item())
            if n_masked == 0:
                sample_done[gi] = True
                continue
            masked_positions = torch.where(sample_masked)[0]
            masked_logits = logits[ai, masked_positions]
            probs = F.softmax(masked_logits, dim=-1)
            top1_conf, top1_token = probs.max(dim=-1)
            target_unmasked = int(round((rev_step + 1) / revision_steps * num_revise))
            already_unmasked = num_revise - n_masked
            num_to_unmask = max(1, target_unmasked - already_unmasked)
            num_to_unmask = min(num_to_unmask, n_masked)
            k = min(num_to_unmask, len(top1_conf))
            if k <= 0:
                continue
            _, topk_idx = top1_conf.topk(k)
            pos_to_unmask = masked_positions[topk_idx]
            input_ids[gi, gen_start + pos_to_unmask] = top1_token[topk_idx]

    remaining_masked = input_ids[:, gen_start:gen_end] == MASK_TOKEN_ID
    if remaining_masked.any():
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = engine.model(input_ids=input_ids, attention_mask=attn_mask.long()).logits[:, gen_start:gen_end, :]
        nfe_per_sample += 1
        for i in range(batch_size):
            masked_positions = torch.where(remaining_masked[i])[0]
            if masked_positions.numel() == 0:
                continue
            probs = F.softmax(logits[i, masked_positions], dim=-1)
            input_ids[i, gen_start + masked_positions] = probs.argmax(dim=-1)

    new_tokens = torch.gather(input_ids[:, gen_start:gen_end], 1, selected_positions)
    return input_ids, (new_tokens != original_tokens).sum(dim=1)


def instability_revise(
    engine: BatchedDLMInference,
    prompt_ids_list: list[list[int]],
    batch_size: int,
    draft_steps: int,
    revision_fraction: float,
    revision_steps: int,
    perturb_frac: float,
) -> list[dict]:
    all_results = []
    for chunk_start in range(0, len(prompt_ids_list), batch_size):
        chunk_prompts = prompt_ids_list[chunk_start:chunk_start + batch_size]
        input_ids, attn_mask, gen_start, gen_end, prompt_lens, nfe_per_sample = draft_fill(
            engine, chunk_prompts, GEN_LENGTH, draft_steps
        )
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            full_logits = engine.model(input_ids=input_ids, attention_mask=attn_mask.long()).logits[:, gen_start:gen_end, :]
        nfe_per_sample += 1
        full_probs = F.softmax(full_logits, dim=-1)
        chosen_tokens = input_ids[:, gen_start:gen_end].unsqueeze(-1)
        chosen_prob_full = torch.gather(full_probs, 2, chosen_tokens).squeeze(-1)

        perturbed_ids = input_ids.clone()
        perturbed_attn = attn_mask.clone()
        max_prompt_len = gen_start
        for i, prompt_len in enumerate(prompt_lens):
            drop = max(1, min(16, int(round(prompt_len * perturb_frac))))
            start = max(max_prompt_len - prompt_len, max_prompt_len - drop)
            perturbed_ids[i, start:max_prompt_len] = engine.pad_token_id
            perturbed_attn[i, start:max_prompt_len] = False

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            perturbed_logits = engine.model(
                input_ids=perturbed_ids,
                attention_mask=perturbed_attn.long(),
            ).logits[:, gen_start:gen_end, :]
        nfe_per_sample += 1
        perturbed_probs = F.softmax(perturbed_logits, dim=-1)
        chosen_prob_perturbed = torch.gather(perturbed_probs, 2, chosen_tokens).squeeze(-1)
        switched = (full_probs.argmax(dim=-1) != perturbed_probs.argmax(dim=-1)).float()
        instability = (chosen_prob_full - chosen_prob_perturbed).abs() + 0.5 * switched

        num_to_revise = max(1, int(round(revision_fraction * GEN_LENGTH)))
        revision_targets = instability.topk(num_to_revise, dim=1).indices
        input_ids, tokens_changed = revise_selected(
            engine, input_ids, attn_mask, gen_start, gen_end, revision_targets, revision_steps, nfe_per_sample
        )
        for i in range(len(chunk_prompts)):
            selected_scores = instability[i, revision_targets[i]]
            all_results.append(
                {
                    "input_ids": input_ids[i].detach().cpu(),
                    "gen_start": gen_start,
                    "gen_end": gen_end,
                    "nfe": int(nfe_per_sample[i].item()),
                    "tokens_changed": int(tokens_changed[i].item()),
                    "signal_stats": {
                        "mean_instability": float(instability[i].mean().item()),
                        "selected_mean_instability": float(selected_scores.mean().item()),
                        "selected_switch_rate": float(switched[i, revision_targets[i]].mean().item()),
                    },
                }
            )
    return all_results


def run_with_backoff(runner, engine: BatchedDLMInference, batch_size: int) -> tuple[list[dict], int, float, int]:
    current = batch_size
    while current >= 1:
        try:
            torch.cuda.reset_peak_memory_stats(engine.device)
            started = time.perf_counter()
            results = runner(current)
            latency = time.perf_counter() - started
            peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
            for result in results:
                result["batch_size"] = current
                result["latency_sec"] = latency / max(1, len(results))
                result["peak_vram_mb"] = peak_vram_mb
            return results, current, latency, peak_vram_mb
        except RuntimeError:
            if current == 1:
                raise
            current = max(1, current // 2)
            torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError("batch backoff exhausted")


def evaluate_samples(samples: list[dict], texts: list[str], results: list[dict], method_name: str) -> dict:
    correct = 0
    total_nfe = 0
    total_changed = 0
    signal_vals = []
    for sample, text, result in zip(samples, texts, results, strict=True):
        correct += int(check_gsm8k_correct(text, sample["final_answer"]))
        total_nfe += int(result["nfe"])
        total_changed += int(result.get("tokens_changed", 0))
        if "signal_stats" in result:
            signal_vals.append(result["signal_stats"])
    latency_sec = sum(float(r.get("latency_sec", 0.0)) for r in results)
    payload = {
        "method": method_name,
        "accuracy": round(correct / max(1, len(samples)), 4),
        "actual_nfe": round(total_nfe / max(1, len(samples)), 2),
        "latency_sec": round(latency_sec, 2),
        "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(latency_sec, 1e-6), 2),
        "batch_size": int(results[0].get("batch_size", 0)),
        "peak_vram_mb": int(results[0].get("peak_vram_mb", 0)),
        "avg_tokens_changed": round(total_changed / max(1, len(samples)), 2),
    }
    if signal_vals:
        payload["signal_stats"] = {
            "mean_instability": round(sum(s["mean_instability"] for s in signal_vals) / len(signal_vals), 4),
            "selected_mean_instability": round(sum(s["selected_mean_instability"] for s in signal_vals) / len(signal_vals), 4),
            "selected_switch_rate": round(sum(s["selected_switch_rate"] for s in signal_vals) / len(signal_vals), 4),
        }
    return payload


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    engine = None
    try:
        requested = args.batch_size if args.batch_size > 0 else read_safe_batch_size()
        batch_size = min(requested, args.samples, 16)
        engine = BatchedDLMInference(model_path=MODEL_PATH, device="cuda", use_flash_attn=True, use_compile=False)
        samples = load_samples(engine.tokenizer, args.samples)
        prompts = [sample["prompt_ids"] for sample in samples]
        report_progress(0, 3, 0, len(samples), {"phase": "loaded", "batch_size": batch_size})

        random_results, random_bs, _, _ = run_with_backoff(
            lambda bs: engine.batched_random_revise(
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=args.draft_steps,
                revision_fraction=args.revision_fraction,
                revision_steps=args.revision_steps,
                batch_size=bs,
                seed=SEED,
            ),
            engine,
            batch_size,
        )
        random_texts = engine.decode_results(random_results)
        random_payload = evaluate_samples(samples, random_texts, random_results, "Random-Revise-64")
        report_progress(1, 3, len(samples), len(samples), random_payload)

        entropy_results, entropy_bs, _, _ = run_with_backoff(
            lambda bs: engine.batched_entropy_revise(
                prompt_ids_list=prompts,
                gen_length=GEN_LENGTH,
                num_draft_steps=args.draft_steps,
                revision_fraction=args.revision_fraction,
                revision_steps=args.revision_steps,
                batch_size=bs,
                seed=SEED,
            ),
            engine,
            min(random_bs, batch_size),
        )
        entropy_texts = engine.decode_results(entropy_results)
        entropy_payload = evaluate_samples(samples, entropy_texts, entropy_results, "Raw-Entropy-Revise")
        report_progress(2, 3, len(samples), len(samples), entropy_payload)

        tiger_results, tiger_bs, _, _ = run_with_backoff(
            lambda bs: instability_revise(
                engine,
                prompts,
                batch_size=bs,
                draft_steps=args.draft_steps,
                revision_fraction=args.revision_fraction,
                revision_steps=args.revision_steps,
                perturb_frac=args.perturb_frac,
            ),
            engine,
            min(entropy_bs, batch_size),
        )
        tiger_texts = engine.decode_results(tiger_results)
        tiger_payload = evaluate_samples(samples, tiger_texts, tiger_results, "Instability-Revise")
        report_progress(3, 3, len(samples), len(samples), tiger_payload)

        output = {
            "task_id": TASK_ID,
            "summary_name": "TIGER 信号筛选：instability vs entropy",
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "gen_length": GEN_LENGTH,
            "config": {
                "requested_batch_size": requested,
                "initial_batch_size": batch_size,
                "random_batch_size": random_bs,
                "entropy_batch_size": entropy_bs,
                "tiger_batch_size": tiger_bs,
                "draft_steps": args.draft_steps,
                "revision_fraction": args.revision_fraction,
                "revision_steps": args.revision_steps,
                "perturb_frac": args.perturb_frac,
                "compile_enabled": False,
            },
            "methods": [random_payload, entropy_payload, tiger_payload],
            "decision": {
                "delta_vs_entropy": round(tiger_payload["accuracy"] - entropy_payload["accuracy"], 4),
                "delta_vs_random": round(tiger_payload["accuracy"] - random_payload["accuracy"], 4),
                "passes_gate": bool(tiger_payload["accuracy"] >= entropy_payload["accuracy"] + 0.01),
            },
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", output)
        write_json(
            RESULTS_DIR / f"{TASK_ID}_gpu_profile.json",
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "peak_vram_mb": max(m["peak_vram_mb"] for m in output["methods"]),
                "used_batch_size": max(random_bs, entropy_bs, tiger_bs),
                "compile_enabled": False,
                "attention_backend": getattr(getattr(engine.model, "config", None), "_attn_implementation", "eager"),
            },
        )
        summary = ", ".join(f"{m['method']} acc={m['accuracy']:.4f} nfe={m['actual_nfe']:.2f}" for m in output["methods"])
        mark_done("success", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {"task_id": TASK_ID, "timestamp": datetime.now().isoformat(), "status": "failed", "error": repr(exc)},
        )
        mark_done("failed", repr(exc))
        return 1
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
