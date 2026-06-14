#!/usr/bin/env python3
"""Minimal multi-seed spot-check for the GSM8K headline pair."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

import cand_bucket_pilot as bucket


TASK_ID = "cand_seed_spotcheck"
PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
RESULTS_PATH = RESULTS_DIR / "seed_sensitivity_spotcheck.json"
SEEDS = [40, 41, 42]


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def report_progress(epoch: int, total_epochs: int, metric: dict | None = None) -> None:
    write_json(
        RESULTS_DIR / f"{TASK_ID}_PROGRESS.json",
        {
            "task_id": TASK_ID,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "metric": metric or {},
            "updated_at": now_iso(),
        },
    )


def mark_done(status: str, summary: str) -> None:
    pid_path = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_path.exists():
        pid_path.unlink()
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_path.exists():
        try:
            final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            final_progress = {}
    write_json(
        RESULTS_DIR / f"{TASK_ID}_DONE",
        {
            "task_id": TASK_ID,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": now_iso(),
        },
    )


def sign(delta: float) -> str:
    if delta > 0:
        return "entropy_better"
    if delta < 0:
        return "standard_better"
    return "tie"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    task_start = time.time()
    runtime_meta = bucket.runtime_contract()
    report_progress(0, len(SEEDS), {"phase": "init", **runtime_meta})

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    engine = bucket.BatchedDLMInference(
        model_path=bucket.MODEL_PATH,
        device="cuda",
        use_flash_attn=True,
        use_compile=bool(runtime_meta["compile_enabled"]),
    )
    samples = bucket.load_samples(engine.tokenizer, bucket.NUM_SAMPLES)

    per_seed = []
    for idx, seed in enumerate(SEEDS, start=1):
        bucket.SEED = seed
        report_progress(idx, len(SEEDS), {"phase": "running_seed", "seed": seed})
        standard = bucket.run_method(engine, "standard", samples, runtime_meta["requested_batch_size"])
        entropy = bucket.run_method(engine, "entropy", samples, runtime_meta["requested_batch_size"])
        delta = round(entropy["accuracy"] - standard["accuracy"], 4)
        per_seed.append(
            {
                "seed": seed,
                "standard_accuracy": standard["accuracy"],
                "entropy_accuracy": entropy["accuracy"],
                "delta": delta,
                "winner": sign(delta),
                "standard_batch_size": standard["batch_size"],
                "entropy_batch_size": entropy["batch_size"],
                "standard_actual_nfe": standard["actual_nfe"],
                "entropy_actual_nfe": entropy["actual_nfe"],
            }
        )

    directions = [row["winner"] for row in per_seed]
    non_tie = [item for item in directions if item != "tie"]
    stable = bool(non_tie) and len(set(non_tie)) == 1 and len(non_tie) == len(directions)
    verdict = "stable" if stable else "fragile"
    reference_seed = next((row for row in per_seed if row["seed"] == 42), per_seed[-1])
    payload = {
        "task_id": TASK_ID,
        "timestamp": now_iso(),
        "pair": {
            "baseline_method": "Standard-64",
            "target_method": "Entropy-Revise-64+3",
        },
        "runtime_contract": runtime_meta,
        "seeds": per_seed,
        "reference_seed": reference_seed,
        "direction_summary": {
            "directions": directions,
            "stable": stable,
            "verdict": verdict,
            "headline_direction": reference_seed["winner"],
        },
        "claim_guidance": {
            "allowed": "Report sign consistency only; do not claim full multi-seed robustness.",
            "if_fragile": "Tighten the headline claim and frame the result as seed-sensitive.",
        },
        "wall_clock_sec": round(time.time() - task_start, 2),
        "summary": (
            f"seed_count={len(per_seed)}, headline_direction={reference_seed['winner']}, "
            f"verdict={verdict}, deltas={[row['delta'] for row in per_seed]}"
        ),
    }
    write_json(RESULTS_PATH, payload)
    report_progress(len(SEEDS), len(SEEDS), {"phase": "done", "verdict": verdict, "headline_direction": reference_seed["winner"]})
    mark_done("success", payload["summary"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        write_json(
            RESULTS_DIR / f"{TASK_ID}.json",
            {
                "task_id": TASK_ID,
                "status": "error",
                "timestamp": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        mark_done("failed", f"{type(exc).__name__}: {exc}")
        raise
