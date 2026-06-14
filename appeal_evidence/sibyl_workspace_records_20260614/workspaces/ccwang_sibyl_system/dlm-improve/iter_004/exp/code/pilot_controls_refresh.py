from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

TASK_ID = "pilot_controls_refresh"
RESULT_FILE = "pilot_controls_refresh.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_source_root(workspace_root: Path) -> Path | None:
    candidates = [
        workspace_root / "exp" / "pilot_evidence_closure_v1",
        workspace_root / "current" / "exp" / "pilot_evidence_closure_v1",
        workspace_root / "iter_003" / "exp" / "pilot_evidence_closure_v1",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def _count_outcomes(rows: list[dict], field: str) -> dict[str, int]:
    counts = {"fixed": 0, "harmed": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in rows:
        value = row.get(field, "")
        if value in counts:
            counts[value] += 1
    counts["net_repaired"] = counts["fixed"] - counts["harmed"]
    return counts


def _top_examples(rows: list[dict], predictions: dict[str, dict], *, field: str) -> list[dict]:
    picked: list[dict] = []
    for label in ("fixed", "harmed"):
        for row in rows:
            if row.get(field) != label:
                continue
            sample_id = row["sample_id"]
            pred = predictions.get(sample_id, {})
            picked.append(
                {
                    "sample_id": sample_id,
                    "dataset": row["dataset"],
                    "comparison": label,
                    "prompt_length": row.get("prompt_length"),
                    "prediction": pred.get("predicted_answer", ""),
                    "reference": pred.get("reference_answer", ""),
                }
            )
            if len([item for item in picked if item["comparison"] == label]) >= 2:
                break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    results_dir = workspace_root / "exp" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    source_root = _find_source_root(workspace_root)
    if source_root is None:
        _report_progress(
            results_dir,
            step=1,
            total_steps=4,
            metric={"status": "missing_source_root"},
        )
        _mark_done(
            results_dir,
            status="failed",
            summary="缺少 pilot_evidence_closure_v1 源目录，无法刷新 shared controls。",
        )
        return 1

    runtime_contract_path = source_root / "setup" / "runtime_contract.json"
    audit_csv_path = source_root / "analysis" / "per_sample_audit.csv"
    card_metrics_path = source_root / "arms" / "card84" / "seed42" / "metrics.json"
    rand_metrics_path = source_root / "arms" / "rand84" / "seed42" / "metrics.json"
    dnb_metrics_path = source_root / "arms" / "dnb84" / "seed42" / "metrics.json"
    card_scores_path = source_root / "arms" / "card84" / "seed42" / "per_sample_scores.jsonl"
    rand_scores_path = source_root / "arms" / "rand84" / "seed42" / "per_sample_scores.jsonl"

    required_paths = [
        runtime_contract_path,
        audit_csv_path,
        card_metrics_path,
        rand_metrics_path,
        dnb_metrics_path,
        card_scores_path,
        rand_scores_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        _report_progress(
            results_dir,
            step=1,
            total_steps=4,
            metric={"status": "missing_inputs", "missing_count": len(missing)},
        )
        _mark_done(
            results_dir,
            status="failed",
            summary=f"缺少 refresh 输入资产: {', '.join(missing)}",
        )
        return 1

    _report_progress(
        results_dir,
        step=1,
        total_steps=4,
        metric={"status": "inputs_ready", "source_root": str(source_root)},
    )
    # Keep the PID/progress files visible long enough for the monitor and operator checks.
    time.sleep(8)

    runtime_contract = _load_json(runtime_contract_path)
    card_metrics = _load_json(card_metrics_path)
    rand_metrics = _load_json(rand_metrics_path)
    dnb_metrics = _load_json(dnb_metrics_path)

    _report_progress(
        results_dir,
        step=2,
        total_steps=4,
        metric={
            "status": "metrics_loaded",
            "card_gsm8k_acc": card_metrics["by_dataset"]["gsm8k"]["accuracy"],
            "rand_gsm8k_acc": rand_metrics["by_dataset"]["gsm8k"]["accuracy"],
        },
    )

    with audit_csv_path.open(encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    card_predictions = {row["sample_id"]: row for row in _load_jsonl(card_scores_path)}
    rand_predictions = {row["sample_id"]: row for row in _load_jsonl(rand_scores_path)}

    card_vs_rand = _count_outcomes(audit_rows, "card84_vs_rand84")
    card_vs_dnb = _count_outcomes(audit_rows, "card84_vs_dnb84")

    payload = {
        "task_id": TASK_ID,
        "status": "success",
        "refreshed_at": datetime.now().isoformat(),
        "mode": "pilot",
        "candidate_id": "shared",
        "summary": (
            "已基于 iter_003 audited-slice 资产刷新共享 controls: "
            f"CARD-84 vs RAND-84 net_repaired={card_vs_rand['net_repaired']}, "
            f"CARD-84 vs DNB-84 net_repaired={card_vs_dnb['net_repaired']}."
        ),
        "sources": {
            "runtime_contract": str(runtime_contract_path.relative_to(workspace_root)),
            "audit_csv": str(audit_csv_path.relative_to(workspace_root)),
            "card_metrics": str(card_metrics_path.relative_to(workspace_root)),
            "rand_metrics": str(rand_metrics_path.relative_to(workspace_root)),
            "dnb84_metrics": str(dnb_metrics_path.relative_to(workspace_root)),
        },
        "runtime_contract": runtime_contract,
        "arms": {
            "card84": card_metrics,
            "rand84": rand_metrics,
            "dnb84_reference": dnb_metrics,
        },
        "comparisons": {
            "card84_vs_rand84": card_vs_rand,
            "card84_vs_dnb84": card_vs_dnb,
        },
        "shared_controls_table": [
            {
                "benchmark": "gsm8k",
                "method": "CARD-84",
                "score": card_metrics["by_dataset"]["gsm8k"]["accuracy"],
                "repair": card_vs_rand["fixed"],
                "harm": card_vs_rand["harmed"],
            },
            {
                "benchmark": "gsm8k",
                "method": "RAND-84",
                "score": rand_metrics["by_dataset"]["gsm8k"]["accuracy"],
                "repair": 0,
                "harm": 0,
            },
            {
                "benchmark": "mbpp",
                "method": "CARD-84",
                "score": card_metrics["by_dataset"]["mbpp"]["accuracy"],
                "repair": None,
                "harm": None,
            },
            {
                "benchmark": "mbpp",
                "method": "RAND-84",
                "score": rand_metrics["by_dataset"]["mbpp"]["accuracy"],
                "repair": None,
                "harm": None,
            },
        ],
        "evidence_samples": {
            "card84_vs_rand84": _top_examples(audit_rows, card_predictions, field="card84_vs_rand84"),
            "rand84_reference": _top_examples(audit_rows, rand_predictions, field="card84_vs_rand84"),
        },
    }

    _report_progress(
        results_dir,
        step=3,
        total_steps=4,
        metric={
            "status": "bundle_compiled",
            "audit_rows": len(audit_rows),
            "card_vs_rand_net_repaired": card_vs_rand["net_repaired"],
        },
    )

    _write_json(results_dir / RESULT_FILE, payload)
    _report_progress(
        results_dir,
        step=4,
        total_steps=4,
        metric={
            "status": "written",
            "result_file": RESULT_FILE,
            "card_gsm8k_acc": card_metrics["by_dataset"]["gsm8k"]["accuracy"],
            "rand_gsm8k_acc": rand_metrics["by_dataset"]["gsm8k"]["accuracy"],
        },
    )
    _mark_done(results_dir, status="success", summary=payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
