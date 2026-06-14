from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

TASK_ID = "repair_iteration4_contract"
RESULT_FILE = "iteration4_contract_repair.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        },
    )


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


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
    pilot_summary_path = results_dir / "pilot_summary.md"
    summary_path = results_dir / "summary.md"
    task_plan_path = workspace_root / "plan" / "task_plan.json"

    proposal = _load_text(proposal_path)
    pilot_summary = _load_text(pilot_summary_path)
    summary = _load_text(summary_path)
    candidates = _load_json(candidates_path)
    task_plan = _load_json(task_plan_path)

    checks = {
        "proposal_exists": proposal_path.exists(),
        "proposal_front_runner_hypothesis": "front-runner hypothesis" in proposal,
        "proposal_mentions_bsr": "cand_bsr" in proposal or "`BSR" in proposal or "BSR:" in proposal,
        "pilot_summary_exists": pilot_summary_path.exists(),
        "pilot_summary_declares_no_new_pilot_yet": "没有新的 pilot 结果" in pilot_summary,
        "pilot_summary_mentions_screening": "screening pilot" in pilot_summary,
        "summary_exists": summary_path.exists(),
        "summary_method_pivot_preparation": "method pivot + screening-pilot preparation" in summary,
        "candidates_include_bsr": any(
            candidate.get("candidate_id") == "cand_bsr"
            for candidate in candidates.get("candidates", [])
        ),
        "candidates_front_runner_mgcd": (
            candidates.get("selected_candidate_id") == "cand_mgcd"
            and any(
                candidate.get("candidate_id") == "cand_mgcd"
                and candidate.get("status") == "front_runner"
                for candidate in candidates.get("candidates", [])
            )
        ),
        "task_plan_exists": task_plan_path.exists(),
        "task_plan_pilot_only_policy": task_plan.get("policy", {}).get("mode") == "pilot_only_until_selection",
        "task_plan_has_repair_task": any(
            task.get("id") == TASK_ID for task in task_plan.get("tasks", [])
        ),
    }

    _report_progress(
        results_dir,
        step=1,
        total_steps=2,
        metric={"checks_passed": sum(checks.values()), "checks_total": len(checks)},
    )

    failed_checks = [name for name, ok in checks.items() if not ok]
    passed = not failed_checks
    payload = {
        "task_id": TASK_ID,
        "status": "success" if passed else "failed",
        "workspace_root": str(workspace_root),
        "checked_at": datetime.now().isoformat(),
        "checks": checks,
        "failed_checks": failed_checks,
        "summary": (
            "Iteration 4 artifacts consistently describe a pilot-first refinement round."
            if passed
            else "Iteration 4 artifact drift remains; see failed_checks."
        ),
    }
    _write_json(results_dir / RESULT_FILE, payload)

    _report_progress(
        results_dir,
        step=2,
        total_steps=2,
        metric={
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "failed_checks": failed_checks,
        },
    )
    _mark_done(
        results_dir,
        status="success" if passed else "failed",
        summary=payload["summary"],
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
