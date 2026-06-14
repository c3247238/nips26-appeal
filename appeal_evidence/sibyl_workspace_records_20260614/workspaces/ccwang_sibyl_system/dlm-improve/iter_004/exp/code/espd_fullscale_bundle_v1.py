#!/usr/bin/env python3
"""Assemble the ESPD full-scale GSM8K evidence bundle for result debate."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(
    os.environ.get(
        "SIBYL_RESULTS_DIR",
        str(Path(__file__).resolve().parents[1] / "results"),
    )
)


TASK_ID = "espd_fullscale_bundle_v1"
CANDIDATE_RESULT_JSON = os.environ.get("SIBYL_ESPD_RESULT_JSON", "espd_gsm8k_full_v1.json")
FIXED_FRONTIER_RESULT_JSON = os.environ.get(
    "SIBYL_ESPD_FIXED_FRONTIER_RESULT_JSON",
    "espd_fixed_frontier_gsm8k_full_v1.json",
)
CONTROLS_RESULT_JSON = os.environ.get(
    "SIBYL_GSM8K_CONTROLS_RESULT_JSON",
    "gsm8k_controls_full_v1.json",
)
QUALITY_TOLERANCE = float(os.environ.get("SIBYL_QUALITY_TOLERANCE", "0.01"))
SPEED_GAIN_FLOOR = float(os.environ.get("SIBYL_SPEED_GAIN_FLOOR", "1.0"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def report_progress(task_id: str, metric: dict[str, Any]) -> None:
    write_json(
        RESULTS_DIR / f"{task_id}_PROGRESS.json",
        {
            "task_id": task_id,
            "epoch": 0,
            "total_epochs": 1,
            "step": 0,
            "total_steps": 1,
            "loss": None,
            "metric": metric,
            "updated_at": now_iso(),
        },
    )


def mark_done(task_id: str, status: str, summary: str) -> None:
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    final_progress: dict[str, Any] = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            final_progress = {}
    write_json(
        RESULTS_DIR / f"{task_id}_DONE",
        {
            "task_id": task_id,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": now_iso(),
        },
    )


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_method_row(method: str, score: float, speed: float, wall_clock_sec: float, **extra: Any) -> dict[str, Any]:
    row = {
        "method": method,
        "quality_at_equal_compute": round(score, 4),
        "speed_at_equal_quality_band": round(speed, 2),
        "wall_clock_sec": round(wall_clock_sec, 2),
    }
    row.update(extra)
    return row


def pick_comparison(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    comparisons = payload.get("comparisons", {}) or {}
    for key in keys:
        value = comparisons.get(key)
        if isinstance(value, dict):
            return value
    return {}


def main() -> int:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    try:
        report_progress(TASK_ID, {"phase": "load_inputs"})
        candidate = load_json(RESULTS_DIR / CANDIDATE_RESULT_JSON)
        fixed_frontier = load_json(RESULTS_DIR / FIXED_FRONTIER_RESULT_JSON)
        controls = load_json(RESULTS_DIR / CONTROLS_RESULT_JSON)

        candidate_metrics = candidate.get("metrics", {}) or {}
        fixed_metrics = fixed_frontier.get("metrics", {}) or {}
        card_metrics = (controls.get("arms", {}) or {}).get("card84", {}) or {}
        rand_metrics = (controls.get("arms", {}) or {}).get("rand84", {}) or {}

        cand_quality = as_float(candidate_metrics.get("equal_compute_quality", candidate_metrics.get("accuracy")))
        cand_speed = as_float(candidate_metrics.get("equal_quality_speed"))
        cand_wall = as_float(candidate_metrics.get("latency_sec"))
        fixed_quality = as_float(fixed_metrics.get("equal_compute_quality", fixed_metrics.get("accuracy")))
        fixed_speed = as_float(fixed_metrics.get("equal_quality_speed"))
        fixed_wall = as_float(fixed_metrics.get("latency_sec"))
        card_quality = as_float(card_metrics.get("accuracy"))
        card_speed = as_float(card_metrics.get("tokens_per_sec"))
        card_wall = as_float(card_metrics.get("latency_sec"))
        rand_quality = as_float(rand_metrics.get("accuracy"))
        rand_speed = as_float(rand_metrics.get("tokens_per_sec"))
        rand_wall = as_float(rand_metrics.get("latency_sec"))

        speed_gain_vs_fixed = cand_speed - fixed_speed
        speed_gain_ratio_vs_fixed = (cand_speed / fixed_speed) if fixed_speed > 0 else None
        best_shared_quality = max(card_quality, rand_quality)
        quality_gap_vs_best_shared = cand_quality - best_shared_quality
        quality_gap_vs_card = cand_quality - card_quality
        quality_gap_vs_rand = cand_quality - rand_quality
        quality_gap_vs_fixed = cand_quality - fixed_quality

        gate_checks = {
            "candidate_beats_fixed_frontier_on_speed": speed_gain_vs_fixed > SPEED_GAIN_FLOOR,
            "candidate_preserves_quality_vs_fixed_frontier": cand_quality >= fixed_quality - QUALITY_TOLERANCE,
            "candidate_non_collapse_vs_best_shared_control": cand_quality >= best_shared_quality - QUALITY_TOLERANCE,
            "candidate_non_collapse_vs_card84": cand_quality >= card_quality - QUALITY_TOLERANCE,
            "routing_story_survives_sham_control": (
                speed_gain_vs_fixed > SPEED_GAIN_FLOOR
                and cand_quality >= fixed_quality - QUALITY_TOLERANCE
            ),
        }
        overall_recommendation = (
            "ADVANCE"
            if all(gate_checks.values())
            else "REFINE"
        )

        decision_matrix = [
            build_method_row(
                "cand_espd",
                cand_quality,
                cand_speed,
                cand_wall,
                active_frontier_ratio=round(
                    as_float(candidate_metrics.get("active_frontier_ratio")),
                    4,
                ),
                avg_nfe=round(as_float(candidate_metrics.get("avg_nfe")), 2),
                effective_batch_size=as_int((candidate.get("runtime") or {}).get("effective_batch_size")),
            ),
            build_method_row(
                "ESPD-FixedFrontier",
                fixed_quality,
                fixed_speed,
                fixed_wall,
                active_frontier_ratio=round(
                    as_float(fixed_metrics.get("active_frontier_ratio")),
                    4,
                ),
                avg_nfe=round(as_float(fixed_metrics.get("avg_nfe")), 2),
                effective_batch_size=as_int((fixed_frontier.get("runtime") or {}).get("effective_batch_size")),
            ),
            build_method_row(
                "CARD-84",
                card_quality,
                card_speed,
                card_wall,
                active_frontier_ratio=round(as_float(card_metrics.get("revision_fraction")), 4),
                avg_nfe=round(as_float(card_metrics.get("avg_nfe")), 2),
                effective_batch_size=as_int(card_metrics.get("batch_size")),
            ),
            build_method_row(
                "RAND-84",
                rand_quality,
                rand_speed,
                rand_wall,
                active_frontier_ratio=round(as_float(rand_metrics.get("revision_fraction")), 4),
                avg_nfe=round(as_float(rand_metrics.get("avg_nfe")), 2),
                effective_batch_size=as_int(rand_metrics.get("batch_size")),
            ),
        ]

        payload = {
            "task_id": TASK_ID,
            "status": "success",
            "candidate_id": "cand_espd",
            "selected_candidate_id": "cand_espd",
            "overall_recommendation": overall_recommendation,
            "dataset": "gsm8k",
            "sample_count": as_int(controls.get("sample_count", candidate_metrics.get("count"))),
            "quality_tolerance": QUALITY_TOLERANCE,
            "speed_gain_floor_tok_per_sec": SPEED_GAIN_FLOOR,
            "decision_matrix": decision_matrix,
            "gate_checks": gate_checks,
            "comparisons": {
                "cand_espd_vs_rand84": pick_comparison(candidate, "cand_espd_vs_rand84", "espd_vs_rand84"),
                "cand_espd_vs_card84": pick_comparison(candidate, "cand_espd_vs_card84", "espd_vs_card84"),
                "fixed_frontier_vs_cand_espd": pick_comparison(fixed_frontier, "fixed_frontier_vs_cand_espd"),
                "card84_vs_rand84": pick_comparison(controls, "card84_vs_rand84"),
                "speed_gain_vs_fixed_frontier": round(speed_gain_vs_fixed, 2),
                "speed_gain_ratio_vs_fixed_frontier": (
                    round(speed_gain_ratio_vs_fixed, 4) if speed_gain_ratio_vs_fixed is not None else None
                ),
                "quality_gap_vs_best_shared_control": round(quality_gap_vs_best_shared, 4),
                "quality_gap_vs_card84": round(quality_gap_vs_card, 4),
                "quality_gap_vs_rand84": round(quality_gap_vs_rand, 4),
                "quality_gap_vs_fixed_frontier": round(quality_gap_vs_fixed, 4),
            },
            "runtime": {
                "candidate_effective_batch_size": as_int((candidate.get("runtime") or {}).get("effective_batch_size")),
                "fixed_frontier_effective_batch_size": as_int(
                    (fixed_frontier.get("runtime") or {}).get("effective_batch_size")
                ),
                "shared_controls_effective_batch_size": as_int((controls.get("runtime") or {}).get("effective_batch_size")),
                "candidate_peak_vram_mb": as_int((candidate.get("runtime") or {}).get("peak_vram_mb")),
                "fixed_frontier_peak_vram_mb": as_int((fixed_frontier.get("runtime") or {}).get("peak_vram_mb")),
                "shared_controls_peak_vram_mb": as_int((controls.get("runtime") or {}).get("peak_vram_mb")),
            },
            "reasons": [
                "The full-scale bundle compares cand_espd against both shared controls and the fixed-frontier sham under the same retained runtime contract.",
                "The key test is whether cand_espd retains shared-control quality while preserving a speed advantage that the fixed-frontier sham cannot match.",
                "This artifact is intended to be the machine-readable handoff for result_debate, not the final publication claim.",
            ],
            "summary": (
                "ESPD full-scale bundle "
                f"verdict={overall_recommendation}; cand_espd quality={cand_quality:.4f}, "
                f"speed={cand_speed:.2f} tok/s, fixed-frontier speed={fixed_speed:.2f} tok/s, "
                f"best shared quality={best_shared_quality:.4f}."
            ),
            "artifact_paths": {
                "candidate_result": str(RESULTS_DIR / CANDIDATE_RESULT_JSON),
                "fixed_frontier_result": str(RESULTS_DIR / FIXED_FRONTIER_RESULT_JSON),
                "shared_controls_result": str(RESULTS_DIR / CONTROLS_RESULT_JSON),
            },
            "timestamp": now_iso(),
        }

        write_json(RESULTS_DIR / f"{TASK_ID}.json", payload)
        report_progress(
            TASK_ID,
            {
                "phase": "done",
                "overall_recommendation": overall_recommendation,
                "speed_gain_vs_fixed_frontier": round(speed_gain_vs_fixed, 2),
                "quality_gap_vs_best_shared_control": round(quality_gap_vs_best_shared, 4),
            },
        )
        mark_done(TASK_ID, "success", payload["summary"])
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": TASK_ID,
            "status": "failed",
            "error": repr(exc),
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / f"{TASK_ID}.json", failure)
        report_progress(TASK_ID, {"phase": "failed", "error": repr(exc)})
        mark_done(TASK_ID, "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
