#!/usr/bin/env python3
"""Pilot task: diagnostic_calibration_heldout.

Runs a lightweight teacher-forced calibration diagnostic on the held-out GSM8K
calibration split created by protocol_repair.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_ID = "diagnostic_calibration_heldout"
PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
RESULTS_PATH = RESULTS_DIR / f"{TASK_ID}.json"
CALIBRATION_SPLIT_PATH = PROJECT_ROOT / "data" / "gsm8k_calibration_split_seed42_100.jsonl"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
SEED = 42
MASK_RATIOS = [0.9, 0.5, 0.1]
MAX_SAMPLES = 100


def report_progress(epoch: int, total_epochs: int, metric: dict | None = None) -> None:
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": 0,
                "total_steps": 0,
                "loss": None,
                "metric": metric or {},
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def mark_done(status: str, summary: str) -> None:
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress: dict = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            final_progress = {}
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": status,
                "summary": summary,
                "final_progress": final_progress,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    if len(confidences) == 0:
        return 0.0
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = boundaries[idx], boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        conf = float(confidences[mask].mean())
        acc = float(accuracies[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def load_samples(tokenizer) -> list[dict]:
    rows = []
    with CALIBRATION_SPLIT_PATH.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= MAX_SAMPLES:
                break
            row = json.loads(line)
            prompt = f"Solve the following math problem step by step.\n\nQuestion: {row['question']}\n\nAnswer:"
            answer = " " + row["answer"]
            rows.append(
                {
                    "idx": idx,
                    "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
                    "answer_ids": tokenizer.encode(answer, add_special_tokens=False)[:256],
                }
            )
    return rows


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    try:
        report_progress(0, 4, {"phase": "load_model"})
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda").eval()
        samples = load_samples(tokenizer)

        diagnostics = []
        for idx, mask_ratio in enumerate(MASK_RATIOS, start=1):
            report_progress(idx, 4, {"phase": "mask_ratio", "mask_ratio": mask_ratio})
            confidences = []
            correct_flags = []
            entropies = []

            for sample in samples:
                answer_ids = sample["answer_ids"]
                if not answer_ids:
                    continue
                answer_len = len(answer_ids)
                mask_count = max(1, int(round(answer_len * mask_ratio)))
                chosen = set(random.Random(SEED + sample["idx"] + int(mask_ratio * 100)).sample(range(answer_len), mask_count))
                corrupted = [MASK_TOKEN_ID if pos in chosen else token for pos, token in enumerate(answer_ids)]
                input_ids = torch.tensor([sample["prompt_ids"] + corrupted], dtype=torch.long, device="cuda")

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids).logits[0]

                prompt_len = len(sample["prompt_ids"])
                for pos in sorted(chosen):
                    logit = logits[prompt_len + pos]
                    probs = F.softmax(logit, dim=-1)
                    conf, pred = probs.max(dim=-1)
                    entropy = float(-(probs * (probs + 1e-10).log()).sum().item())
                    confidences.append(float(conf.item()))
                    correct_flags.append(int(int(pred.item()) == int(answer_ids[pos])))
                    entropies.append(entropy)

            conf_arr = np.array(confidences, dtype=np.float64)
            acc_arr = np.array(correct_flags, dtype=np.float64)
            ent_arr = np.array(entropies, dtype=np.float64)
            corr = 0.0
            if len(conf_arr) > 1 and np.std(ent_arr) > 0 and np.std(acc_arr) > 0:
                corr = float(np.corrcoef(ent_arr, acc_arr)[0, 1])
            diagnostics.append(
                {
                    "mask_ratio": mask_ratio,
                    "num_points": int(len(conf_arr)),
                    "ece": round(compute_ece(conf_arr, acc_arr), 4),
                    "mean_confidence": round(float(conf_arr.mean()) if len(conf_arr) else 0.0, 4),
                    "mean_accuracy": round(float(acc_arr.mean()) if len(acc_arr) else 0.0, 4),
                    "entropy_error_corr": round(corr, 4),
                }
            )

        result = {
            "task_id": TASK_ID,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "sample_count": len(samples),
            "mask_ratio_diagnostics": diagnostics,
            "summary": "Held-out teacher-forced calibration diagnostic completed",
        }
        result["pass"] = any(item["ece"] > 0.05 for item in diagnostics)
        RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_progress(4, 4, {"pass": result["pass"], "diagnostics": diagnostics})
        mark_done("success" if result["pass"] else "failed", result["summary"])
        return 0 if result["pass"] else 1
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": TASK_ID,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": repr(exc),
        }
        RESULTS_PATH.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        report_progress(4, 4, {"pass": False, "error": repr(exc)})
        mark_done("failed", f"diagnostic_calibration_heldout failed: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
