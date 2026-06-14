#!/usr/bin/env python3
"""Aggregate the cand_espd GSM8K screening pilot from completed artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pilot_standard_arm import RESULTS_DIR, mark_done, now_iso, report_progress, write_json


TASK_ID = "espd_gsm8k_pilot_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    try:
        report_progress(TASK_ID, {"phase": "load_inputs"})
        espd = load_json(RESULTS_DIR / "implement_espd.json")
        refresh = load_json(RESULTS_DIR / "gsm8k_controls_refresh_v2.json")
        fixed = load_json(RESULTS_DIR / "espd_gsm8k_frontier_controls.json")

        speed_gain_vs_fixed = round(
            espd["metrics"]["equal_quality_speed"] - fixed["metrics"]["equal_quality_speed"],
            4,
        )
        speed_gain_ratio = round(
            espd["metrics"]["equal_quality_speed"] / max(1e-6, fixed["metrics"]["equal_quality_speed"]),
            4,
        )
        quality_gap_vs_rand = round(
            espd["metrics"]["accuracy"] - refresh["arms"]["rand84"]["accuracy"],
            4,
        )
        quality_gap_vs_fixed = round(
            espd["metrics"]["accuracy"] - fixed["metrics"]["accuracy"],
            4,
        )
        routing_signal_clean = (
            speed_gain_vs_fixed > 0
            and espd["comparisons"]["espd_vs_rand84"]["net_repaired"] >= 1
            and fixed["comparisons"]["fixed_frontier_vs_cand_espd"]["net_repaired"] == 0
        )
        if routing_signal_clean and quality_gap_vs_rand >= 0.0:
            verdict = "ADVANCE"
        elif speed_gain_vs_fixed > 0:
            verdict = "REFINE"
        else:
            verdict = "PIVOT"

        payload = {
            "task_id": TASK_ID,
            "candidate_id": "cand_espd",
            "dataset": "gsm8k",
            "status": "success",
            "screening_verdict": verdict,
            "quality_at_equal_compute": espd["metrics"]["accuracy"],
            "speed_at_equal_quality_band": espd["metrics"]["equal_quality_speed"],
            "active_frontier_ratio": espd["metrics"]["active_frontier_ratio"],
            "gate_checks": {
                "speed_gain_vs_fixed_frontier": speed_gain_vs_fixed > 0,
                "quality_non_collapse_vs_rand84": quality_gap_vs_rand >= 0.0,
                "fixed_frontier_fails_to_explain_speed": fixed["comparisons"]["fixed_frontier_vs_cand_espd"]["net_repaired"] == 0,
            },
            "two_d_report": [
                {
                    "method": "cand_espd",
                    "quality_at_equal_compute": espd["metrics"]["accuracy"],
                    "speed_at_equal_quality_band": espd["metrics"]["equal_quality_speed"],
                    "active_frontier_ratio": espd["metrics"]["active_frontier_ratio"],
                },
                {
                    "method": "ESPD-FixedFrontier",
                    "quality_at_equal_compute": fixed["metrics"]["accuracy"],
                    "speed_at_equal_quality_band": fixed["metrics"]["equal_quality_speed"],
                    "active_frontier_ratio": fixed["metrics"]["active_frontier_ratio"],
                },
                {
                    "method": "CARD-84",
                    "quality_at_equal_compute": refresh["arms"]["card84"]["accuracy"],
                    "speed_at_equal_quality_band": refresh["arms"]["card84"]["tokens_per_sec"],
                    "active_frontier_ratio": refresh["arms"]["card84"]["revision_fraction"],
                },
                {
                    "method": "RAND-84",
                    "quality_at_equal_compute": refresh["arms"]["rand84"]["accuracy"],
                    "speed_at_equal_quality_band": refresh["arms"]["rand84"]["tokens_per_sec"],
                    "active_frontier_ratio": refresh["arms"]["rand84"]["revision_fraction"],
                },
            ],
            "comparisons": {
                "cand_espd_vs_rand84": espd["comparisons"]["espd_vs_rand84"],
                "cand_espd_vs_card84": espd["comparisons"]["espd_vs_card84"],
                "fixed_frontier_vs_cand_espd": fixed["comparisons"]["fixed_frontier_vs_cand_espd"],
                "speed_gain_vs_fixed_frontier": speed_gain_vs_fixed,
                "speed_gain_ratio_vs_fixed_frontier": speed_gain_ratio,
                "quality_gap_vs_rand84": quality_gap_vs_rand,
                "quality_gap_vs_fixed_frontier": quality_gap_vs_fixed,
            },
            "stopped_step_histograms": {
                "cand_espd": espd["metrics"]["stopped_step_histogram"],
                "fixed_frontier": fixed["metrics"]["stopped_step_histogram"],
            },
            "summary": (
                f"cand_espd GSM8K screening verdict={verdict}; "
                f"quality={espd['metrics']['accuracy']} and speed={espd['metrics']['equal_quality_speed']} tok/s, "
                f"vs fixed-frontier speed={fixed['metrics']['equal_quality_speed']} tok/s."
            ),
            "artifact_paths": {
                "candidate_result": str(RESULTS_DIR / "implement_espd.json"),
                "shared_controls": str(RESULTS_DIR / "gsm8k_controls_refresh_v2.json"),
                "fixed_frontier_control": str(RESULTS_DIR / "espd_gsm8k_frontier_controls.json"),
            },
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / "espd_gsm8k_pilot.json", payload)
        write_json(RESULTS_DIR / f"{TASK_ID}.json", payload)
        report_progress(
            TASK_ID,
            {
                "phase": "done",
                "screening_verdict": verdict,
                "quality_at_equal_compute": payload["quality_at_equal_compute"],
                "speed_at_equal_quality_band": payload["speed_at_equal_quality_band"],
            },
        )
        mark_done(TASK_ID, "success", payload["summary"])
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": TASK_ID,
            "candidate_id": "cand_espd",
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
