#!/usr/bin/env python3
"""A lightweight CORE-style proxy baseline for GSM8K pilot."""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import full_llada_gsm8k as base


TASK_ID = "baseline_core"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
NUM_SAMPLES = 100
GEN_LENGTH = 256
MASK_TOKEN_ID = 126336
SEED = 42
GPU_PROFILE_PATH = RESULTS_DIR / f"{TASK_ID}_gpu_profile.json"


def write_pid() -> None:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")


def report_progress(step: int, total_steps: int, metric: dict[str, Any]) -> None:
    payload = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "loss": None,
        "metric": metric,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mark_done(status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            final_progress = {}
    payload = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model() -> tuple[Any, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # LLaDA on this stack should avoid SDPA; flash-attn is unavailable, so force eager.
    attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to("cuda").eval()
    # Keep CORE proxy on eager mode only. torch.compile was unstable here and
    # triggered cublas invalid-value failures during long sequential decoding.
    compile_enabled = False
    return model, tokenizer, attn_impl, compile_enabled


def prompt_perturbation_scores(model, input_ids: torch.Tensor, prompt_len: int, gen_start: int, gen_end: int) -> torch.Tensor:
    """Measure token brittleness by prompt masking perturbation."""
    with torch.no_grad():
        original_logits = model(input_ids=input_ids).logits[0, gen_start:gen_end]
    original_probs = F.softmax(original_logits, dim=-1)
    original_top_token = original_probs.argmax(dim=-1)
    original_top_conf = original_probs.max(dim=-1).values

    perturbed = input_ids.clone()
    prompt_positions = list(range(prompt_len))
    if prompt_positions:
        rng = random.Random(SEED + prompt_len)
        mask_count = max(1, int(round(0.15 * len(prompt_positions))))
        for pos in rng.sample(prompt_positions, k=min(mask_count, len(prompt_positions))):
            perturbed[0, pos] = MASK_TOKEN_ID

    with torch.no_grad():
        perturbed_logits = model(input_ids=perturbed).logits[0, gen_start:gen_end]
    perturbed_probs = F.softmax(perturbed_logits, dim=-1)
    perturbed_top_token = perturbed_probs.argmax(dim=-1)
    perturbed_conf_on_original = perturbed_probs.gather(1, original_top_token.unsqueeze(-1)).squeeze(-1)

    disagreement = (perturbed_top_token != original_top_token).float()
    confidence_drop = (original_top_conf - perturbed_conf_on_original).clamp(min=0.0)
    return disagreement + confidence_drop


def revise_selected_positions(model, input_ids: torch.Tensor, gen_start: int, selected_positions: list[int], revision_steps: int = 3) -> tuple[torch.Tensor, int]:
    if not selected_positions:
        return input_ids, 0
    revised = input_ids.clone()
    revised[0, gen_start + torch.tensor(selected_positions, device=revised.device)] = MASK_TOKEN_ID
    nfe = 0
    for step in range(revision_steps):
        masked_positions = torch.where(revised[0, gen_start:] == MASK_TOKEN_ID)[0]
        if masked_positions.numel() == 0:
            break
        with torch.no_grad():
            logits = model(input_ids=revised).logits[0, gen_start:gen_start + revised.shape[1] - gen_start]
        nfe += 1
        masked_logits = logits[masked_positions]
        probs = F.softmax(masked_logits, dim=-1)
        top_conf, top_tok = probs.max(dim=-1)
        num_to_fill = max(1, math.ceil(masked_positions.numel() / (revision_steps - step)))
        num_to_fill = min(num_to_fill, masked_positions.numel())
        fill_idx = top_conf.topk(num_to_fill).indices
        fill_positions = masked_positions[fill_idx]
        revised[0, gen_start + fill_positions] = top_tok[fill_idx]
    return revised, nfe


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, tokenizer, attn_impl, compile_enabled = load_model()
    samples = base.get_gsm8k_samples(tokenizer, NUM_SAMPLES)
    results = []
    total_correct = 0
    total_nfe = 0
    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    for idx, sample in enumerate(samples, start=1):
        draft_input, gen_start, gen_end, draft_nfe = base.run_standard_denoising(
            model,
            sample["prompt_ids"],
            GEN_LENGTH,
            64,
        )
        brittleness = prompt_perturbation_scores(
            model,
            draft_input,
            len(sample["prompt_ids"]),
            gen_start,
            gen_end,
        )
        total_nfe += draft_nfe + 2
        k = max(1, int(round(0.10 * brittleness.numel())))
        selected = brittleness.topk(k).indices.tolist()
        revised, rev_nfe = revise_selected_positions(model, draft_input, gen_start, selected, revision_steps=3)
        total_nfe += rev_nfe

        gen_tokens = revised[0, gen_start:gen_end].cpu().tolist()
        gen_tokens = [token for token in gen_tokens if token != MASK_TOKEN_ID]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        correct = base.check_gsm8k_correct(generated_text, sample["final_answer"])
        total_correct += int(correct)
        results.append(
            {
                "idx": sample["idx"],
                "correct": bool(correct),
                "nfe": draft_nfe + 2 + rev_nfe,
                "selected_positions": selected[:10],
                "brittleness_mean": round(float(brittleness.mean().item()), 4),
                "predicted_answer": base.extract_gsm8k_answer(generated_text),
                "reference_answer": sample["final_answer"],
                "generated_text": generated_text[:400],
            }
        )
        if idx % 10 == 0 or idx == len(samples):
            report_progress(
                idx,
                len(samples),
                {
                    "samples_done": idx,
                    "total_samples": len(samples),
                    "accuracy_so_far": round(total_correct / idx, 4),
                    "avg_nfe": round(total_nfe / idx, 2),
                    "attention_backend": attn_impl,
                },
            )

    payload = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "method": "CORE-proxy-64",
        "proxy_note": "Prompt perturbation brittleness + selective 3-step revision; not the official CORE implementation.",
        "attention_backend": attn_impl,
        "compile_enabled": compile_enabled,
        "num_samples": len(samples),
        "accuracy": round(total_correct / max(len(samples), 1), 4),
        "correct": total_correct,
        "total": len(samples),
        "actual_nfe": round(total_nfe / max(len(samples), 1), 2),
        "latency_sec": round(time.time() - start, 3),
        "tokens_per_sec": round((len(samples) * GEN_LENGTH) / max(time.time() - start, 1e-6), 2),
        "batch_size": 1,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
        "per_sample": results[:10],
    }
    (RESULTS_DIR / f"{TASK_ID}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    GPU_PROFILE_PATH.write_text(
        json.dumps(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
                "used_batch_size": 1,
                "peak_vram_mb": payload["peak_vram_mb"],
                "attention_backend": attn_impl,
                "compile_enabled": compile_enabled,
                "proxy": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mark_done("success", f"CORE proxy accuracy={payload['accuracy']:.3f}, actual_nfe={payload['actual_nfe']:.2f}")
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
        (RESULTS_DIR / f"{TASK_ID}.json").write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        mark_done("failed", f"{type(exc).__name__}: {exc}")
        raise
