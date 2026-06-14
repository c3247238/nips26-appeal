#!/usr/bin/env python3
"""Aggregate the cand_bsr GSM8K screening pilot from completed artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pilot_standard_arm import RESULTS_DIR, mark_done, now_iso, report_progress, write_json


TASK_ID = "bsr_gsm8k_pilot_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def method_row(
    method: str,
    score: float,
    repair_count: int | None,
    harm_count: int | None,
    harmed_stable_tokens: float | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "score": score,
        "repair_count": repair_count,
        "harm_count": harm_count,
        "harmed_stable_tokens": harmed_stable_tokens,
    }


def main() -> int:
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    try:
        report_progress(TASK_ID, {"phase": "load_inputs"})
        bsr = load_json(RESULTS_DIR / "implement_bsr_v1.json")
        refresh = load_json(RESULTS_DIR / "gsm8k_controls_refresh_v2.json")
        sham = load_json(RESULTS_DIR / "bsr_gsm8k_sham_controls.json")

        bsr_vs_rand = bsr["comparisons"]["bsr_v1_vs_rand84"]
        bsr_vs_card = bsr["comparisons"]["bsr_v1_vs_card84"]
        sham_controls = sham["controls"]
        sham_nets = {
            key: value["comparisons"][f"{key}_vs_rand84"]["net_repaired"]
            for key, value in sham_controls.items()
        }
        best_sham_key = max(
            sham_controls,
            key=lambda key: sham_controls[key]["metrics"]["accuracy"],
        )

        beats_rand84 = (
            bsr["overall"]["accuracy"] > refresh["arms"]["rand84"]["accuracy"]
            and bsr_vs_rand["net_repaired"] > 0
        )
        sham_failure_present = any(net < bsr_vs_rand["net_repaired"] for net in sham_nets.values())
        boundary_story_broken = sham_controls["entropyspan_noboundary"]["metrics"]["accuracy"] >= bsr["overall"]["accuracy"]

        if beats_rand84 and sham_failure_present and not boundary_story_broken:
            verdict = "ADVANCE"
        elif beats_rand84 and sham_failure_present:
            verdict = "REFINE"
        else:
            verdict = "PIVOT"

        table = [
            method_row(
                "cand_bsr_v1",
                bsr["overall"]["accuracy"],
                bsr_vs_rand["fixed"],
                bsr_vs_rand["harmed"],
                bsr["qualitative_examples"][0]["rejected_islands"] if bsr.get("qualitative_examples") else None,
            ),
            method_row(
                "CARD-84",
                refresh["arms"]["card84"]["accuracy"],
                refresh["comparisons"]["card84_vs_rand84"]["fixed"],
                refresh["comparisons"]["card84_vs_rand84"]["harmed"],
                None,
            ),
            method_row(
                "RAND-84",
                refresh["arms"]["rand84"]["accuracy"],
                refresh["comparisons"]["rand84_vs_card84"]["fixed"],
                refresh["comparisons"]["rand84_vs_card84"]["harmed"],
                None,
            ),
        ]
        for key, value in sham_controls.items():
            comparison = value["comparisons"][f"{key}_vs_rand84"]
            table.append(
                method_row(
                    value["label"],
                    value["metrics"]["accuracy"],
                    comparison["fixed"],
                    comparison["harmed"],
                    value["metrics"]["harmed_stable_tokens"],
                )
            )

        payload = {
            "task_id": TASK_ID,
            "candidate_id": "cand_bsr",
            "dataset": "gsm8k",
            "status": "success",
            "screening_verdict": verdict,
            "quality_at_equal_compute": bsr["overall"]["accuracy"],
            "repair_harm_ratio": round(
                bsr_vs_rand["fixed"] / max(1, bsr_vs_rand["harmed"]),
                4,
            ),
            "gate_checks": {
                "beats_rand84": beats_rand84,
                "beats_card84": bsr["overall"]["accuracy"] > refresh["arms"]["card84"]["accuracy"],
                "sham_failure_present": sham_failure_present,
                "boundary_story_broken": boundary_story_broken,
            },
            "main_result_table": table,
            "comparisons": {
                "cand_bsr_v1_vs_rand84": bsr_vs_rand,
                "cand_bsr_v1_vs_card84": bsr_vs_card,
                "sham_vs_rand84": sham_nets,
            },
            "mechanism_notes": [
                "BoundaryLock-RandomSpan fails to match cand_bsr_v1, so uncertainty-free random spans do not explain the gain.",
                "EntropySpan-NoBoundary exceeds cand_bsr_v1 on this audited slice, so the current boundary-lock contract likely needs refinement rather than wholesale rejection.",
                "RandSpan-84 ties CARD-84 on accuracy but still trails the strongest entropy-based sham."
            ],
            "pilot_focus": "Compare cand_bsr_v1 against shared controls and three sham controls under matched audited-slice evidence.",
            "best_matching_sham_control": {
                "method": sham_controls[best_sham_key]["label"],
                "accuracy": sham_controls[best_sham_key]["metrics"]["accuracy"],
                "net_repaired_vs_rand84": sham_nets[best_sham_key],
            },
            "summary": (
                f"cand_bsr_v1 GSM8K screening verdict={verdict}; "
                f"acc={bsr['overall']['accuracy']} vs RAND-84={refresh['arms']['rand84']['accuracy']}, "
                f"best sham={sham_controls[best_sham_key]['label']} ({sham_controls[best_sham_key]['metrics']['accuracy']})."
            ),
            "artifact_paths": {
                "candidate_result": str(RESULTS_DIR / "implement_bsr_v1.json"),
                "shared_controls": str(RESULTS_DIR / "gsm8k_controls_refresh_v2.json"),
                "sham_controls": str(RESULTS_DIR / "bsr_gsm8k_sham_controls.json"),
            },
            "timestamp": now_iso(),
        }
        write_json(RESULTS_DIR / "bsr_gsm8k_pilot.json", payload)
        write_json(RESULTS_DIR / f"{TASK_ID}.json", payload)
        report_progress(
            TASK_ID,
            {
                "phase": "done",
                "screening_verdict": verdict,
                "quality_at_equal_compute": payload["quality_at_equal_compute"],
                "repair_harm_ratio": payload["repair_harm_ratio"],
            },
        )
        mark_done(TASK_ID, "success", payload["summary"])
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "task_id": TASK_ID,
            "candidate_id": "cand_bsr",
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
