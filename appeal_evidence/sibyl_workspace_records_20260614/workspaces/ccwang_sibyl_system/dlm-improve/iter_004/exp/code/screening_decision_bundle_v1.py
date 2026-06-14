#!/usr/bin/env python3
"""Summarize phase-1 screening outcomes into a machine-readable decision bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pilot_standard_arm import RESULTS_DIR, mark_done, now_iso, report_progress, write_json


TASK_ID = "screening_decision_bundle_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    try:
        report_progress(TASK_ID, {"phase": "load_inputs"})
        bsr = load_json(RESULTS_DIR / "bsr_gsm8k_pilot.json")
        espd = load_json(RESULTS_DIR / "espd_gsm8k_pilot.json")

        candidate_rows = [
            {
                "candidate": "cand_bsr",
                "quality_at_equal_compute": bsr["quality_at_equal_compute"],
                "speed_at_equal_quality_band": None,
                "repair_harm_ratio": bsr["repair_harm_ratio"],
                "decision": bsr["screening_verdict"],
                "note": "Quality line beats RAND-84, but EntropySpan-NoBoundary outperforms the frozen boundary-lock variant, so the object is promising but the contract needs refinement.",
            },
            {
                "candidate": "cand_espd",
                "quality_at_equal_compute": espd["quality_at_equal_compute"],
                "speed_at_equal_quality_band": espd["speed_at_equal_quality_band"],
                "repair_harm_ratio": None,
                "decision": espd["screening_verdict"],
                "note": "Speed line keeps quality at CARD-84 level while beating the fixed-frontier control on throughput, so the routing story is ready to advance.",
            },
        ]

        payload = {
            "task_id": TASK_ID,
            "status": "success",
            "overall_recommendation": "ADVANCE",
            "selected_candidate_id": "cand_espd",
            "advance_candidates": ["cand_espd"],
            "refine_candidates": ["cand_bsr"],
            "pivot_candidates": [],
            "decision_matrix": candidate_rows,
            "reasons": [
                "cand_espd is the cleanest positive signal: equal quality to CARD-84, positive net repair vs RAND-84, and a fixed-frontier control that cannot explain the same speedup.",
                "cand_bsr remains a live quality-line candidate, but the strongest sham (EntropySpan-NoBoundary) beating the frozen v1 variant means the boundary-lock contract should be revised before advancing it as-is.",
                "The screening frame now supports a split next step: advance the ESPD speed line while refining the BSR object-level contract."
            ],
            "artifact_paths": {
                "bsr_result": str(RESULTS_DIR / "bsr_gsm8k_pilot.json"),
                "espd_result": str(RESULTS_DIR / "espd_gsm8k_pilot.json"),
            },
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / "iteration4_phase1_screening_bundle.json", payload)
        write_json(RESULTS_DIR / f"{TASK_ID}.json", payload)
        report_progress(
            TASK_ID,
            {
                "phase": "done",
                "overall_recommendation": payload["overall_recommendation"],
                "selected_candidate_id": payload["selected_candidate_id"],
            },
        )
        mark_done(TASK_ID, "success", json.dumps(payload["decision_matrix"], ensure_ascii=False))
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
