from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

TASK_ID = "freeze_iteration4_pivot_contract"
RESULT_FILE = "iteration4_pivot_contract.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _report_progress(results_dir: Path, *, step: int, total_steps: int, metric: dict) -> None:
    _write_json(
        results_dir / f"{TASK_ID}_PROGRESS.json",
        {
            "task_id": TASK_ID,
            "epoch": 1,
            "total_epochs": 1,
            "step": step,
            "total_steps": total_steps,
            "loss": None,
            "metric": metric,
            "updated_at": datetime.now().isoformat(),
        },
    )


def _mark_done(results_dir: Path, *, status: str, summary: str) -> None:
    pid_file = results_dir / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = results_dir / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            final_progress = {}
    _write_json(
        results_dir / f"{TASK_ID}_DONE",
        {
            "task_id": TASK_ID,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": datetime.now().isoformat(),
            "result_path": str(results_dir / RESULT_FILE),
        },
    )


def _candidate_status(candidates: list[dict], candidate_id: str) -> str:
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            return str(candidate.get("status", ""))
    return ""


def _id_set(values: list) -> set[str]:
    output: set[str] = set()
    for value in values:
        if isinstance(value, str):
            output.add(value)
        elif isinstance(value, dict):
            candidate_id = value.get("candidate_id")
            if candidate_id:
                output.add(str(candidate_id))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    results_dir = workspace_root / "exp" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    proposal_path = workspace_root / "idea" / "proposal.md"
    candidates_path = workspace_root / "idea" / "candidates.json"
    hypotheses_path = workspace_root / "idea" / "hypotheses.md"
    debate_path = workspace_root / "idea" / "debate" / "summary.md"
    task_plan_path = workspace_root / "plan" / "task_plan.json"
    decision_path = workspace_root / "supervisor" / "idea_validation_decision.json"
    pilot_summary_path = results_dir / "pilot_summary.json"

    proposal = _load_text(proposal_path)
    hypotheses = _load_text(hypotheses_path)
    debate = _load_text(debate_path)
    candidates_blob = _load_json(candidates_path)
    task_plan = _load_json(task_plan_path)
    decision = _load_json(decision_path)
    pilot_summary = _load_json(pilot_summary_path)

    candidates = candidates_blob.get("candidates", [])
    archived = _id_set(candidates_blob.get("archived_negative_controls", []))
    dropped = _id_set(candidates_blob.get("dropped_candidates", []))

    freeze_task = {}
    for task in task_plan.get("tasks", []):
        if task.get("id") == TASK_ID:
            freeze_task = task
            break

    serious_pool = {
        item.get("candidate_id", "")
        for item in candidates
        if item.get("candidate_id") in {"cand_bsr", "cand_espd", "cand_ugr"}
    }

    pilot_candidate_ids = {
        item.get("candidate_id", "") for item in pilot_summary.get("candidate_results", [])
    }

    checks = {
        "proposal_marks_bsr_front_runner": "front-runner" in proposal and "cand_bsr" in proposal,
        "proposal_archives_mgcd_dsg": "archived negative controls" in proposal and "cand_mgcd" in proposal and "cand_dsg" in proposal,
        "candidates_select_bsr": candidates_blob.get("selected_candidate_id") == "cand_bsr",
        "candidate_statuses_match_pivot": (
            _candidate_status(candidates, "cand_bsr") == "front_runner"
            and _candidate_status(candidates, "cand_espd") == "backup"
            and _candidate_status(candidates, "cand_ugr") == "backup"
        ),
        "archived_controls_match": archived == {"cand_mgcd", "cand_dsg"} and dropped == {"cand_mgcd", "cand_dsg"},
        "hypotheses_reframe_local_object_and_routing": "Local Repair Object Hypothesis" in hypotheses and "Entropy-As-Routing Hypothesis" in hypotheses,
        "debate_serious_pool_frozen": serious_pool == {"cand_bsr", "cand_espd", "cand_ugr"} and "唯一 quality front-runner" in debate and "并行 speed line" in debate,
        "decision_is_pivot_with_mgcd_dsg_drop": decision.get("decision") == "PIVOT" and set(decision.get("dropped_candidates", [])) == {"cand_mgcd", "cand_dsg"},
        "pilot_summary_records_negative_evidence": (
            pilot_summary.get("decision_recommendation") == "PIVOT"
            and pilot_summary.get("full_benchmark_allowed") is False
            and pilot_candidate_ids == {"cand_mgcd", "cand_dsg"}
        ),
        "task_plan_freezes_new_pool": (
            task_plan.get("policy", {}).get("mode") == "phase1_screening_only"
            and task_plan.get("selected_candidates") == ["cand_bsr", "cand_espd"]
            and freeze_task.get("expected_output") == f"exp/results/{RESULT_FILE}"
        ),
        "task_pass_criteria_match_contract": "cand_bsr" in str(freeze_task.get("pilot", {}).get("pass_criteria", "")) and "cand_espd" in str(freeze_task.get("pilot", {}).get("pass_criteria", "")),
    }

    _report_progress(
        results_dir,
        step=1,
        total_steps=3,
        metric={
            "phase": "alignment_checks_started",
            "checks_total": len(checks),
            "workspace_root": str(workspace_root),
        },
    )
    time.sleep(8)

    failed_checks = [name for name, ok in checks.items() if not ok]
    passed = not failed_checks

    payload = {
        "task_id": TASK_ID,
        "status": "success" if passed else "failed",
        "checked_at": datetime.now().isoformat(),
        "workspace_root": str(workspace_root),
        "summary": (
            "Iteration 4 pivot contract is frozen: cand_bsr is the only quality front-runner, cand_espd is the speed line, and MGCD/DSG remain archived negatives."
            if passed
            else "Iteration 4 pivot contract still has artifact drift; inspect failed_checks before dispatching screening tasks."
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "evidence_files": {
            "proposal": str(proposal_path),
            "candidates": str(candidates_path),
            "hypotheses": str(hypotheses_path),
            "debate_ledger": str(debate_path),
            "task_plan": str(task_plan_path),
            "pivot_decision": str(decision_path),
            "pilot_summary": str(pilot_summary_path),
        },
        "frozen_pool": {
            "front_runner": "cand_bsr",
            "speed_line": "cand_espd",
            "conditional_backup": "cand_ugr",
            "archived_negative_controls": sorted(archived),
        },
    }
    _write_json(results_dir / RESULT_FILE, payload)

    _report_progress(
        results_dir,
        step=2,
        total_steps=3,
        metric={
            "phase": "alignment_checks_complete",
            "checks_passed": sum(bool(v) for v in checks.values()),
            "checks_total": len(checks),
            "failed_checks": failed_checks,
        },
    )

    summary = payload["summary"]
    _report_progress(
        results_dir,
        step=3,
        total_steps=3,
        metric={"phase": "done", "summary": summary},
    )
    _mark_done(results_dir, status="success" if passed else "failed", summary=summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
