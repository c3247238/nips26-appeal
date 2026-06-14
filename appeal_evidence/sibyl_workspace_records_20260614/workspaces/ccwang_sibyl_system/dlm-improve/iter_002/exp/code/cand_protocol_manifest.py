#!/usr/bin/env python3
"""Build protocol-first manifest artifacts for iter_002.

Outputs:
- canonical_asset_manifest.json
- runtime_fairness_matrix.json
- observer_controller_protocol.json
- cand_protocol_manifest_PROGRESS.json
- cand_protocol_manifest_DONE
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


TASK_ID = "cand_protocol_manifest"
PROJECT_ROOT = Path(os.environ.get("SIBYL_REMOTE_PROJECT_ROOT", "/home/ccwang/sibyl_system/projects/dlm-improve"))
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
MANIFEST_PATH = RESULTS_DIR / "canonical_asset_manifest.json"
FAIRNESS_PATH = RESULTS_DIR / "runtime_fairness_matrix.json"
PROTOCOL_PATH = RESULTS_DIR / "observer_controller_protocol.json"


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict | list) -> None:
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
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def asset(path_str: str) -> Path:
    return PROJECT_ROOT / path_str


def runtime_row_from_diag(method: dict) -> dict:
    return {
        "method": method["method"],
        "source_artifact": str(asset("exp/results/diag_compute_curve_gsm8k.json")),
        "nominal_nfe": method.get("nominal_nfe"),
        "actual_nfe": method.get("actual_nfe"),
        "latency_sec": method.get("latency_sec"),
        "tokens_per_sec": method.get("tokens_per_sec"),
        "batch_size": method.get("batch_size"),
        "attention_backend": method.get("attention_backend"),
        "compile_enabled": method.get("compile_enabled"),
        "compute_gap_pct": method.get("compute_gap_pct"),
        "rank_shift": method.get("rank_shift"),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    try:
        report_progress(1, 4, {"phase": "load_assets"})
        diag_compute = load_json(asset("exp/results/diag_compute_curve_gsm8k.json"))
        signal_gap = load_json(asset("exp/results/diag_signal_gap_audit.json"))
        math_boundary = load_json(asset("exp/results/diag_math500_shortlist.json"))
        code_boundary = load_json(asset("exp/results/diag_humaneval_guard_boundary.json"))
        pareto = load_json(asset("exp/results/final_pareto_synthesis.json"))
        runtime_probe = load_json(asset("exp/results/runtime_probe_iter2.json"))

        report_progress(2, 4, {"phase": "runtime_fairness"})
        runtime_fairness = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "recommended_runtime_path": runtime_probe["checks"]["throughput_compile_on"]["attn_backend"] + "|compile=True",
            "recommended_batch_size": runtime_probe["checks"]["throughput_compile_on"]["safe_batch_size"],
            "probe_summary": runtime_probe.get("summary", ""),
            "rows": [runtime_row_from_diag(method) for method in diag_compute.get("methods", [])],
            "probe_runtime": {
                "source_artifact": str(asset("exp/results/runtime_probe_iter2.json")),
                "compile_off": runtime_probe["checks"]["throughput_compile_off"],
                "compile_on": runtime_probe["checks"]["throughput_compile_on"],
            },
            "notes": [
                "Runtime fairness is anchored to actual_nfe plus realized latency, not nominal method names.",
                "The iter_002 runtime probe recommends eager attention with torch.compile enabled; flash-attn is unavailable on the current host.",
                "Manifest rows preserve the exact JSON source so reviewer-facing claims can be traced back to machine-readable evidence.",
            ],
        }
        write_json(FAIRNESS_PATH, runtime_fairness)

        report_progress(3, 4, {"phase": "claim_manifest"})
        claim_manifest = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "paper_positioning": "compute-normalized diagnostic / protocol paper",
            "headline_claims": [
                {
                    "claim_id": "C1_compute_reorders_headlines",
                    "claim_text": "Nominal compute labels can misstate the realized compute ordering; at least one headline pair reorders under actual NFE.",
                    "paper_slot": "Table 1 / Results",
                    "figure_or_table": "compute-normalized headline comparison",
                    "primary_artifact": str(asset("exp/results/diag_compute_curve_gsm8k.json")),
                    "runtime_fields": ["nominal_nfe", "actual_nfe", "latency_sec", "batch_size", "attention_backend", "compile_enabled", "rank_shift"],
                    "supporting_fields": ["pairwise_reorders", "max_abs_compute_gap_pct", "headline_findings"],
                },
                {
                    "claim_id": "C2_observer_not_equal_controller_gain",
                    "claim_text": "Observer signal quality and realized controller gain should be reported separately rather than collapsed into one headline score.",
                    "paper_slot": "Discussion / Protocol framing",
                    "figure_or_table": "observer-controller split diagram",
                    "primary_artifact": str(asset("exp/results/diag_signal_gap_audit.json")),
                    "runtime_fields": ["actual_nfe", "latency_sec"],
                    "supporting_fields": ["signal_metrics", "gap_summary", "headline_findings"],
                },
                {
                    "claim_id": "C3_reasoning_boundary_is_positive_but_narrow",
                    "claim_text": "Reasoning-side benefits exist, but they are boundary evidence rather than a universal win across tasks.",
                    "paper_slot": "Boundary positioning",
                    "figure_or_table": "MATH500 boundary summary",
                    "primary_artifact": str(asset("exp/results/diag_math500_shortlist.json")),
                    "runtime_fields": ["actual_nfe", "latency_sec"],
                    "supporting_fields": ["headline_findings", "pairwise"],
                },
                {
                    "claim_id": "C4_code_boundary_is_negative",
                    "claim_text": "Code generation remains a harm boundary; the paper should not generalize the reasoning-side gains to code tasks.",
                    "paper_slot": "Boundary positioning",
                    "figure_or_table": "HumanEval guard boundary",
                    "primary_artifact": str(asset("exp/results/diag_humaneval_guard_boundary.json")),
                    "runtime_fields": ["actual_nfe", "latency_sec"],
                    "supporting_fields": ["headline_findings", "boundary_summary"],
                },
                {
                    "claim_id": "C5_runtime_contract_is_auditable",
                    "claim_text": "Runtime assumptions are explicit and auditable via a dedicated probe rather than hidden in prose.",
                    "paper_slot": "Appendix / Runtime fairness",
                    "figure_or_table": "runtime fairness appendix table",
                    "primary_artifact": str(asset("exp/results/runtime_probe_iter2.json")),
                    "runtime_fields": ["safe_batch_size", "attn_backend", "compile_enabled", "tokens_per_sec", "peak_vram_mb"],
                    "supporting_fields": ["checks", "summary"],
                },
                {
                    "claim_id": "C6_pareto_story_is_protocol_first",
                    "claim_text": "The paper's stable contribution is protocol-first and diagnostic-first, not a new hero controller.",
                    "paper_slot": "Abstract / Intro",
                    "figure_or_table": "integrated synthesis",
                    "primary_artifact": str(asset("exp/results/final_pareto_synthesis.json")),
                    "runtime_fields": ["actual_nfe", "latency_sec"],
                    "supporting_fields": ["preferred_story", "keep_claims", "drop_claims"],
                },
            ],
            "artifact_inventory": [
                {
                    "artifact_path": str(asset("exp/results/diag_compute_curve_gsm8k.json")),
                    "role": "headline compute-normalized comparison",
                    "status": diag_compute.get("status", "unknown"),
                },
                {
                    "artifact_path": str(asset("exp/results/diag_signal_gap_audit.json")),
                    "role": "observer vs controller audit",
                    "status": signal_gap.get("status", "unknown"),
                },
                {
                    "artifact_path": str(asset("exp/results/diag_math500_shortlist.json")),
                    "role": "reasoning boundary slice",
                    "status": math_boundary.get("status", "unknown"),
                },
                {
                    "artifact_path": str(asset("exp/results/diag_humaneval_guard_boundary.json")),
                    "role": "code harm boundary slice",
                    "status": code_boundary.get("status", "unknown"),
                },
                {
                    "artifact_path": str(asset("exp/results/final_pareto_synthesis.json")),
                    "role": "paper-level narrative synthesis",
                    "status": pareto.get("status", "unknown"),
                },
                {
                    "artifact_path": str(asset("exp/results/runtime_probe_iter2.json")),
                    "role": "iter_002 runtime probe",
                    "status": runtime_probe.get("status", "unknown"),
                },
            ],
        }
        write_json(MANIFEST_PATH, claim_manifest)

        protocol = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "framing": "Observer / controller / runtime should be reported as separate protocol objects.",
            "definitions": {
                "d_of_s": {
                    "name": "observer_diagnostic_signal",
                    "definition": "A sample-level observer score that estimates whether a draft is revision-worthy before any controller intervention.",
                    "artifact_source": str(asset("exp/results/diag_signal_gap_audit.json")),
                    "examples": ["entropy", "instability", "draft-side uncertainty proxies"],
                },
                "g_of_s": {
                    "name": "realized_controller_gain",
                    "definition": "The realized sample-level outcome delta after a concrete controller policy and runtime path are applied.",
                    "artifact_source": str(asset("exp/results/benefit_bucket_audit_pilot.json")),
                    "examples": ["fixed", "harmed", "no_effect"],
                },
            },
            "runtime_contract": {
                "runtime_probe_artifact": str(asset("exp/results/runtime_probe_iter2.json")),
                "recommended_path": runtime_fairness["recommended_runtime_path"],
                "recommended_batch_size": runtime_fairness["recommended_batch_size"],
                "flash_attention_available": runtime_probe["checks"]["gpu"].get("flash_attn_2_available", False),
            },
            "non_equivalence_caveats": [
                "A stronger observer signal d(s) does not imply a stronger realized gain g(s); controller policy and compute budget intervene.",
                "Even nominally matched methods can move on the Pareto frontier once actual NFE and realized latency are audited.",
                "Boundary slices from reasoning and code tasks are positioning evidence, not a license to claim universal cross-task behavior.",
            ],
        }
        write_json(PROTOCOL_PATH, protocol)

        summary = (
            f"manifest_claims={len(claim_manifest['headline_claims'])}, "
            f"runtime_rows={len(runtime_fairness['rows'])}, "
            f"recommended_path={runtime_fairness['recommended_runtime_path']}, "
            f"recommended_batch={runtime_fairness['recommended_batch_size']}"
        )
        report_progress(4, 4, {"phase": "done", "manifest_claims": len(claim_manifest["headline_claims"])})
        mark_done("success", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        report_progress(4, 4, {"phase": "failed", "error": repr(exc)})
        mark_done("failed", repr(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
