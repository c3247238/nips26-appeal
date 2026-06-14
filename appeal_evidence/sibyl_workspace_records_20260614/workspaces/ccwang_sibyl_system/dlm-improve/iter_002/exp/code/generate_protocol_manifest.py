#!/usr/bin/env python3
"""Generate protocol-lineage artifacts for cand_protocol_manifest.

Outputs:
  - exp/results/canonical_asset_manifest.json
  - exp/results/runtime_fairness_matrix.json
  - exp/results/observer_controller_protocol.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current")
ITER1_RESULTS = WORKSPACE.parent / "iter_001" / "exp" / "results"
RESULTS_DIR = WORKSPACE / "exp" / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_to_workspace(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def file_exists(path: Path) -> bool:
    return path.exists()


def build_runtime_fairness_matrix(
    compute_curve: dict,
    math500: dict,
    humaneval: dict,
    runtime_probe: dict,
    gpu_profile: dict,
) -> dict:
    headline_methods = []
    for method in compute_curve["methods"]:
        headline_methods.append(
            {
                "method": method["method"],
                "task_slice": "gsm8k_headline_shortlist",
                "accuracy": method["accuracy"],
                "nominal_nfe": method["nominal_nfe"],
                "actual_nfe": method["actual_nfe"],
                "latency_sec": method["latency_sec"],
                "tokens_per_sec": method["tokens_per_sec"],
                "batch_size": method["batch_size"],
                "attention_backend": method["attention_backend"],
                "compile_enabled": method["compile_enabled"],
                "compute_gap_pct": method["compute_gap_pct"],
                "actual_compute_rank": method["actual_compute_rank"],
                "rank_shift_vs_nominal": method["rank_shift"],
            }
        )

    boundary_methods = []
    for method in math500["methods"]:
        if method["method"] in {"Standard-64", "Entropy-Revise-64+3", "CORE-proxy-64"}:
            boundary_methods.append(
                {
                    "method": method["method"],
                    "task_slice": "math500_boundary_slice",
                    "accuracy": method["accuracy"],
                    "actual_nfe": method["actual_nfe"],
                    "latency_sec": method["latency_sec"],
                    "tokens_per_sec": method["tokens_per_sec"],
                    "batch_size": method["batch_size"],
                    "peak_vram_mb": method["peak_vram_mb"],
                    "attention_backend": method["attention_backend"],
                    "compile_enabled": method["compile_enabled"],
                }
            )
    for method in humaneval["methods"]:
        boundary_methods.append(
            {
                "method": method["method"],
                "task_slice": "humaneval_boundary_slice",
                "pass_at_1": method["pass_at_1"],
                "syntax_failure_rate": method["syntax_failure_rate"],
                "runtime_failure_rate": method["runtime_failure_rate"],
                "gate_open_rate": method.get("gate_open_rate"),
                "syntax_guard_avg_ms": method.get("syntax_guard_avg_ms"),
                "runtime_fields_available": False,
                "note": "Boundary asset focuses on structure outcomes rather than full runtime fairness axes.",
            }
        )

    return {
        "task_id": "cand_protocol_manifest",
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment_probe": {
            "runtime_probe_asset": rel_to_workspace(RESULTS_DIR / "runtime_probe_iter2.json"),
            "gpu_profile_asset": rel_to_workspace(RESULTS_DIR / "shared_runtime_probe_gpu_profile.json"),
            "gpu_name": gpu_profile["gpu_name"],
            "vram_total_mb": gpu_profile["vram_total_mb"],
            "safe_batch_size": runtime_probe["checks"]["throughput_compile_on"]["safe_batch_size"],
            "attention_backend": runtime_probe["checks"]["model"]["attn_implementation"],
            "flash_attn_2_available": runtime_probe["checks"]["gpu"]["flash_attn_2_available"],
            "compile_tested": gpu_profile["compile_tested"],
            "compile_tokens_per_sec": runtime_probe["checks"]["throughput_compile_on"]["throughput"]["tokens_per_sec"],
            "eager_tokens_per_sec": runtime_probe["checks"]["throughput_compile_off"]["throughput"]["tokens_per_sec"],
        },
        "fairness_axes": [
            "nominal_nfe",
            "actual_nfe",
            "latency_sec",
            "tokens_per_sec",
            "batch_size",
            "attention_backend",
            "compile_enabled",
        ],
        "headline_methods": headline_methods,
        "boundary_methods": boundary_methods,
        "pairwise_reorders": compute_curve["pairwise_reorders"],
        "headline_findings": [
            "Runtime fairness must be audited on realized pipeline fields, not nominal steps alone.",
            "CORE-proxy-64 remains the raw-accuracy winner on GSM8K shortlist but loses compute-rank position under realized latency/NFE.",
            "Compile and batch-size asymmetries are material to interpretation and must stay visible in appendix-grade reviewer artifacts.",
        ],
        "limitations": [
            "Boundary slices do not all expose the same runtime fields; HumanEval boundary results remain structure-focused evidence.",
            "actual_nfe is necessary but not sufficient; batchability and compile/backend choices still affect realized cost.",
        ],
    }


def build_observer_controller_protocol(
    signal_audit: dict,
    calibration: dict,
    tiger_screen: dict,
) -> dict:
    audit_rows = {row["signal"]: row for row in signal_audit["signal_audit"]}
    calibration_row = audit_rows["calibration"]
    entropy_row = audit_rows["entropy"]
    instability_row = audit_rows["instability"]
    strongest_calibration = max(
        calibration["mask_ratio_diagnostics"],
        key=lambda row: abs(row["entropy_error_corr"]),
    )

    return {
        "task_id": "cand_protocol_manifest",
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "paper_positioning": "compute-normalized diagnostic/protocol paper",
            "claim_style": "falsification-style framing under tested policies",
            "non_claims": [
                "This artifact does not claim a universal law across all DLMs or tasks.",
                "This artifact does not claim that observer quality alone can identify the best controller.",
            ],
        },
        "objects": {
            "observer_quality": {
                "symbol": "d(s)",
                "definition": "A signal's diagnostic usefulness for error risk under the current slice, measured by the artifact-specific correlation or held-out diagnostic statistic that is explicitly reported.",
                "interpretation": "Higher d(s) means the signal tracks error risk better; it does not imply intervention value.",
            },
            "controller_gain": {
                "symbol": "g(s)",
                "definition": "A signal's realized control usefulness under the deployed policy and compute budget, measured by accuracy delta or control-effectiveness field in the corresponding shortlist/screen asset.",
                "interpretation": "Higher g(s) means the deployed control policy helped more under the tested runtime path; it does not imply stronger diagnostic quality.",
            },
        },
        "signal_rows": [
            {
                "signal": "calibration",
                "diagnostic_score": calibration_row["diagnostic_score"],
                "diagnostic_measure": calibration_row["diagnostic_measure"],
                "diagnostic_source": rel_to_workspace(ITER1_RESULTS / "diagnostic_calibration_heldout.json"),
                "diagnostic_context": {
                    "best_mask_ratio": strongest_calibration["mask_ratio"],
                    "best_ece": strongest_calibration["ece"],
                    "mean_confidence": strongest_calibration["mean_confidence"],
                    "mean_accuracy": strongest_calibration["mean_accuracy"],
                },
                "control_effectiveness": calibration_row["control_effectiveness"],
                "control_measure": calibration_row["control_measure"],
                "control_source": rel_to_workspace(ITER1_RESULTS / "diag_signal_gap_audit.json"),
                "deployed_controller": False,
            },
            {
                "signal": "entropy",
                "diagnostic_score": entropy_row["diagnostic_score"],
                "diagnostic_measure": entropy_row["diagnostic_measure"],
                "diagnostic_source": rel_to_workspace(ITER1_RESULTS / "diag_signal_gap_audit.json"),
                "control_effectiveness": entropy_row["control_effectiveness"],
                "control_measure": entropy_row["control_measure"],
                "control_delta_vs_standard": entropy_row["control_delta_vs_standard"],
                "control_source": rel_to_workspace(ITER1_RESULTS / "gsm8k_main_shortlist.json"),
                "deployed_controller": True,
            },
            {
                "signal": "instability",
                "diagnostic_score": instability_row["diagnostic_score"],
                "diagnostic_measure": instability_row["diagnostic_measure"],
                "diagnostic_source": rel_to_workspace(ITER1_RESULTS / "tiger_signal_screen.json"),
                "control_effectiveness": instability_row["control_effectiveness"],
                "control_measure": instability_row["control_measure"],
                "control_delta_vs_standard": instability_row["control_delta_vs_standard"],
                "control_source": rel_to_workspace(ITER1_RESULTS / "gsm8k_main_shortlist.json"),
                "deployed_controller": True,
            },
        ],
        "interpretation_limits": [
            "Scores come from different but explicitly named artifacts; they are comparable only at the scoped protocol level, not as universal signal constants.",
            "Calibration is observational-only in the current shortlist and must not be described as a failed deployed controller.",
            "Observer/controller mismatch should be phrased as 'under the tested policies and runtime path' rather than as an unrestricted theorem.",
        ],
        "supporting_assets": {
            "signal_audit": rel_to_workspace(ITER1_RESULTS / "diag_signal_gap_audit.json"),
            "calibration_diagnostic": rel_to_workspace(ITER1_RESULTS / "diagnostic_calibration_heldout.json"),
            "tiger_signal_screen": rel_to_workspace(ITER1_RESULTS / "tiger_signal_screen.json"),
        },
    }


def build_canonical_asset_manifest(
    compute_curve: dict,
    signal_audit: dict,
    math500: dict,
    humaneval: dict,
    pareto: dict,
) -> dict:
    return {
        "task_id": "cand_protocol_manifest",
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_positioning": "compute-normalized diagnostic/protocol paper",
        "claim_to_asset": [
            {
                "claim_id": "C1",
                "claim": "Honest compute changes key GSM8K comparison orderings and therefore changes what headline comparisons are credible.",
                "scope": "headline_gsm8k_shortlist_only",
                "paper_objects": ["Table 1", "Figure honest_compute"],
                "primary_artifacts": [
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "diag_compute_curve_gsm8k.json"),
                        "role": "compute-normalized ranking and reorder evidence",
                        "runtime_fields": [
                            "nominal_nfe",
                            "actual_nfe",
                            "latency_sec",
                            "tokens_per_sec",
                            "batch_size",
                            "attention_backend",
                            "compile_enabled",
                        ],
                    },
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "gsm8k_main_shortlist.json"),
                        "role": "source shortlist metrics and per-method runtime metadata",
                        "runtime_fields": [
                            "actual_nfe",
                            "latency_sec",
                            "tokens_per_sec",
                            "batch_size",
                            "attention_backend",
                            "compile_enabled",
                        ],
                    },
                ],
                "supporting_artifacts": [
                    {
                        "path": rel_to_workspace(RESULTS_DIR / "runtime_probe_iter2.json"),
                        "role": "current-runtime probe documenting compile/eager throughput and safe batch size",
                    },
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "final_pareto_synthesis.json"),
                        "role": "decision-level synthesis that motivated pivot to cand_diag",
                    },
                ],
                "headline_evidence": compute_curve["headline_findings"],
            },
            {
                "claim_id": "C2",
                "claim": "Observer quality does not automatically become controller gain under the tested policies.",
                "scope": "tested_policies_only",
                "paper_objects": ["Figure signal_gap", "Appendix protocol note"],
                "primary_artifacts": [
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "diag_signal_gap_audit.json"),
                        "role": "diagnostic-control gap summary",
                        "runtime_fields": [],
                    },
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "diagnostic_calibration_heldout.json"),
                        "role": "held-out calibration diagnostic source for d(s)",
                        "runtime_fields": [],
                    },
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "tiger_signal_screen.json"),
                        "role": "screen-level signal statistics and lightweight control comparison",
                        "runtime_fields": [
                            "batch_size",
                            "attention_backend",
                            "compile_enabled",
                            "actual_nfe",
                            "latency_sec",
                        ],
                    },
                ],
                "supporting_artifacts": [
                    {
                        "path": rel_to_workspace(RESULTS_DIR / "observer_controller_protocol.json"),
                        "role": "scoped protocol definitions for d(s), g(s), and interpretation limits",
                    }
                ],
                "headline_evidence": signal_audit["headline_findings"],
            },
            {
                "claim_id": "C3",
                "claim": "Boundary slices support only boundary positioning, not a general regime law across tasks.",
                "scope": "math500_and_humaneval_boundary_only",
                "paper_objects": ["Boundary results table", "Discussion limits"],
                "primary_artifacts": [
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "diag_math500_shortlist.json"),
                        "role": "reasoning-side boundary slice",
                        "runtime_fields": [
                            "actual_nfe",
                            "latency_sec",
                            "tokens_per_sec",
                            "batch_size",
                            "attention_backend",
                            "compile_enabled",
                        ],
                    },
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "diag_humaneval_guard_boundary.json"),
                        "role": "code-side structural stress test boundary slice",
                        "runtime_fields": [
                            "pass_at_1",
                            "syntax_failure_rate",
                            "runtime_failure_rate",
                        ],
                    },
                ],
                "supporting_artifacts": [
                    {
                        "path": rel_to_workspace(ITER1_RESULTS / "final_pareto_synthesis.json"),
                        "role": "documents appendix-only / boundary-only positioning decision",
                    }
                ],
                "headline_evidence": humaneval["headline_findings"],
            },
        ],
        "required_outputs_present": {
            "canonical_asset_manifest": file_exists(RESULTS_DIR / "canonical_asset_manifest.json"),
            "runtime_fairness_matrix": file_exists(RESULTS_DIR / "runtime_fairness_matrix.json"),
            "observer_controller_protocol": file_exists(RESULTS_DIR / "observer_controller_protocol.json"),
        },
        "notes": [
            "Claim-to-asset mapping is intentionally scoped to the diagnostic/protocol paper framing, not to deprecated TIGER hero narratives.",
            "Each claim must remain aligned with the explicit evidence scope recorded here.",
        ],
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    compute_curve = load_json(ITER1_RESULTS / "diag_compute_curve_gsm8k.json")
    signal_audit = load_json(ITER1_RESULTS / "diag_signal_gap_audit.json")
    math500 = load_json(ITER1_RESULTS / "diag_math500_shortlist.json")
    humaneval = load_json(ITER1_RESULTS / "diag_humaneval_guard_boundary.json")
    pareto = load_json(ITER1_RESULTS / "final_pareto_synthesis.json")
    calibration = load_json(ITER1_RESULTS / "diagnostic_calibration_heldout.json")
    tiger_screen = load_json(ITER1_RESULTS / "tiger_signal_screen.json")
    runtime_probe = load_json(RESULTS_DIR / "runtime_probe_iter2.json")
    gpu_profile = load_json(RESULTS_DIR / "shared_runtime_probe_gpu_profile.json")

    runtime_matrix = build_runtime_fairness_matrix(
        compute_curve, math500, humaneval, runtime_probe, gpu_profile
    )
    protocol = build_observer_controller_protocol(
        signal_audit, calibration, tiger_screen
    )
    manifest = build_canonical_asset_manifest(
        compute_curve, signal_audit, math500, humaneval, pareto
    )

    (RESULTS_DIR / "runtime_fairness_matrix.json").write_text(
        json.dumps(runtime_matrix, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "observer_controller_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Recompute required_outputs_present after the companion artifacts are written.
    manifest["required_outputs_present"] = {
        "canonical_asset_manifest": True,
        "runtime_fairness_matrix": True,
        "observer_controller_protocol": True,
    }
    (RESULTS_DIR / "canonical_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
