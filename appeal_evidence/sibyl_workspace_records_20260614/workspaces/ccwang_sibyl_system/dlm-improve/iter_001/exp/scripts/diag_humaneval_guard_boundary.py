#!/usr/bin/env python3
"""Summarize existing HumanEval guard-boundary evidence for cand_diag."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_ID = "diag_humaneval_guard_boundary"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
SOURCE_PATH = RESULTS_DIR / "tiger_gating_boundary.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            "epoch": 1,
            "total_epochs": 1,
            "step": 1,
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
            final_progress = load_json(progress_path)
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


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()
    source = load_json(SOURCE_PATH)
    methods = source["humaneval_boundary"]
    standard = methods["standard"]
    ungated = methods["ungated_tiger"]
    gated = methods["gated_tiger"]
    syntax_improvement = gated["syntax_failure_rate"] - ungated["syntax_failure_rate"]
    runtime_improvement = gated["runtime_failure_rate"] - ungated["runtime_failure_rate"]
    pass_boundary = syntax_improvement <= -0.10 and gated["pass_at_1"] <= standard["pass_at_1"]
    reopen_alert = gated["pass_at_1"] >= standard["pass_at_1"] + 0.02 or (
        gated["syntax_failure_rate"] < standard["syntax_failure_rate"]
        and gated["runtime_failure_rate"] < standard["runtime_failure_rate"]
    )
    report_progress(
        {
            "syntax_delta_vs_ungated": round(syntax_improvement, 4),
            "runtime_delta_vs_ungated": round(runtime_improvement, 4),
            "boundary_pass": pass_boundary,
            "reopen_method_alert": reopen_alert,
        }
    )
    summary = (
        f"Gating changed syntax failure by {syntax_improvement:.2f} and runtime failure by {runtime_improvement:.2f}; "
        f"gated pass@1={gated['pass_at_1']:.2f} vs standard={standard['pass_at_1']:.2f}."
    )
    payload = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "source_asset": str(SOURCE_PATH),
        "num_samples": methods.get("num_samples"),
        "methods": [
            {
                "method": "Standard",
                "pass_at_1": standard["pass_at_1"],
                "syntax_failure_rate": standard["syntax_failure_rate"],
                "runtime_failure_rate": standard["runtime_failure_rate"],
            },
            {
                "method": "Entropy/TIGER Ungated Revision",
                "pass_at_1": ungated["pass_at_1"],
                "syntax_failure_rate": ungated["syntax_failure_rate"],
                "runtime_failure_rate": ungated["runtime_failure_rate"],
            },
            {
                "method": "Gated TIGER",
                "pass_at_1": gated["pass_at_1"],
                "syntax_failure_rate": gated["syntax_failure_rate"],
                "runtime_failure_rate": gated["runtime_failure_rate"],
                "gate_open_rate": source["gsm8k_reasoning"].get("gate_open_rate"),
                "syntax_guard_avg_ms": methods.get("syntax_guard_avg_ms"),
            },
        ],
        "headline_findings": [
            "cheap guard 明显降低 syntax failure，但没有恢复到 Standard 的 pass@1 水平。",
            "runtime failure 没有同步改善，说明 guard 更像浅层防护，而不是整体恢复机制。",
            "这条证据支持 appendix-only boundary positioning，而不是重开 method-forward 主线。",
        ],
        "deltas": {
            "gated_vs_ungated_syntax_failure": round(syntax_improvement, 4),
            "gated_vs_ungated_runtime_failure": round(runtime_improvement, 4),
            "gated_vs_standard_pass_at_1": round(gated["pass_at_1"] - standard["pass_at_1"], 4),
        },
        "boundary_pass": pass_boundary,
        "reopen_method_alert": reopen_alert,
        "pass": pass_boundary and not reopen_alert,
        "summary": summary,
    }
    write_json(RESULTS_DIR / f"{TASK_ID}.json", payload)
    mark_done("success" if payload["pass"] else "failed", summary)
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
