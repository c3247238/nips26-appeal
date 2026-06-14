#!/usr/bin/env python3
"""Analyze iteration-3 pilot arms and decide the claim branch."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current")
PILOT_DIR = ROOT / "exp" / "pilot_evidence_closure_v1"
SETUP_DIR = PILOT_DIR / "setup"
ARMS_DIR = PILOT_DIR / "arms"
ANALYSIS_DIR = PILOT_DIR / "analysis"

ARM_META = {
    "dnb64": {"task_id": "run_dnb64_baseline", "label": "DNB-64"},
    "dnb84": {"task_id": "run_dnb84_matched_compute", "label": "DNB-84"},
    "card84": {"task_id": "run_card84_reference_controller", "label": "CARD-84"},
    "rand84": {"task_id": "run_rand84_sham_control", "label": "RAND-84"},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transition(base_correct: bool, arm_correct: bool) -> str:
    if not base_correct and arm_correct:
        return "fixed"
    if base_correct and not arm_correct:
        return "harmed"
    if base_correct and arm_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def metric_bundle(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["transition"] for row in rows)
    fixed = counts.get("fixed", 0)
    harmed = counts.get("harmed", 0)
    return {
        "count": len(rows),
        "fixed": fixed,
        "harmed": harmed,
        "unchanged_correct": counts.get("unchanged_correct", 0),
        "unchanged_wrong": counts.get("unchanged_wrong", 0),
        "net_repaired": fixed - harmed,
    }


def mbpp_failure_bucket(row: dict[str, Any]) -> str:
    error = (row.get("error") or "").strip()
    if not error:
        return "passed"
    prefix = error.split(":", 1)[0]
    if prefix in {"TimeoutError", "SyntaxError", "IndentationError", "AssertionError", "NameError", "TypeError"}:
        return prefix
    return "other"


def main() -> int:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_payload = load_json(SETUP_DIR / "sample_manifest.json")
    manifest_rows = manifest_payload if isinstance(manifest_payload, list) else manifest_payload["rows"]
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    runtime_contract = load_json(SETUP_DIR / "runtime_contract.json")

    scores_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    metrics_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARM_META:
        arm_dir = ARMS_DIR / arm / "seed42"
        scores = load_jsonl(arm_dir / "per_sample_scores.jsonl")
        metrics = load_json(arm_dir / "metrics.json")
        scores_by_arm[arm] = {row["sample_id"]: row for row in scores}
        metrics_by_arm[arm] = metrics

    join_ok = all(set(scores_by_arm[arm]) == set(manifest_by_id) for arm in ARM_META)
    if not join_ok:
        missing = {
            arm: sorted(set(manifest_by_id) - set(scores_by_arm[arm]))
            for arm in ARM_META
            if set(scores_by_arm[arm]) != set(manifest_by_id)
        }
        raise RuntimeError(f"Arm/sample join failure: {missing}")

    audit_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    comparison_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    comparisons = [
        ("card84", "dnb64"),
        ("card84", "dnb84"),
        ("card84", "rand84"),
        ("dnb84", "dnb64"),
        ("rand84", "dnb64"),
    ]

    for sample in manifest_rows:
        sid = sample["sample_id"]
        row: dict[str, Any] = {
            "sample_id": sid,
            "dataset": sample["dataset"],
            "difficulty_bucket": sample["difficulty_bucket"],
            "prompt_length": sample["prompt_length"],
            "selection_reason": sample.get("selection_reason", ""),
        }
        for arm in ARM_META:
            score = scores_by_arm[arm][sid]
            row[f"{arm}_correct"] = int(bool(score["correct"]))
            row[f"{arm}_nfe"] = int(score["nfe"])
            row[f"{arm}_tokens_changed"] = int(score.get("tokens_changed", 0))
            if sample["dataset"] == "mbpp":
                row[f"{arm}_failure_mode"] = mbpp_failure_bucket(score)
        for arm, base in comparisons:
            t = transition(bool(scores_by_arm[base][sid]["correct"]), bool(scores_by_arm[arm][sid]["correct"]))
            row[f"{arm}_vs_{base}"] = t
            entry = {
                "sample_id": sid,
                "dataset": sample["dataset"],
                "comparison": f"{arm}_vs_{base}",
                "transition": t,
            }
            comparison_buckets[(sample["dataset"], f"{arm}_vs_{base}")].append(entry)
            transition_rows.append(entry)
        audit_rows.append(row)

    audit_path = ANALYSIS_DIR / "per_sample_audit.csv"
    audit_fields = sorted({key for row in audit_rows for key in row.keys()})
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    transition_path = ANALYSIS_DIR / "transition_matrix.csv"
    transition_fieldnames = ["dataset", "comparison", "transition", "count"]
    with transition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=transition_fieldnames)
        writer.writeheader()
        for (dataset, comparison), rows in sorted(comparison_buckets.items()):
            counts = Counter(r["transition"] for r in rows)
            for name in ("fixed", "harmed", "unchanged_correct", "unchanged_wrong"):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "comparison": comparison,
                        "transition": name,
                        "count": counts.get(name, 0),
                    }
                )

    summary_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for (dataset, comparison), rows in comparison_buckets.items():
        summary_stats[dataset][comparison] = metric_bundle(rows)

    mbpp_harm_buckets: dict[str, Counter[str]] = {}
    for arm in ("dnb64", "dnb84", "card84", "rand84"):
        counter: Counter[str] = Counter()
        for sample in manifest_rows:
            if sample["dataset"] != "mbpp":
                continue
            score = scores_by_arm[arm][sample["sample_id"]]
            if score["correct"]:
                continue
            counter[mbpp_failure_bucket(score)] += 1
        mbpp_harm_buckets[arm] = counter

    card_vs_dnb84_gsm8k = summary_stats["gsm8k"]["card84_vs_dnb84"]
    card_vs_rand84_gsm8k = summary_stats["gsm8k"]["card84_vs_rand84"]
    card_vs_dnb64_mbpp = summary_stats["mbpp"]["card84_vs_dnb64"]
    dnb84_vs_dnb64_mbpp = summary_stats["mbpp"]["dnb84_vs_dnb64"]
    mbpp_harm_delta = card_vs_dnb64_mbpp["harmed"] - dnb84_vs_dnb64_mbpp["harmed"]

    gate_details = {
        "gsm8k_card_minus_dnb84_net_repaired": card_vs_dnb84_gsm8k["net_repaired"],
        "gsm8k_card_minus_rand84_net_repaired": card_vs_rand84_gsm8k["net_repaired"],
        "mbpp_card_harmed_vs_dnb84_harmed_delta_against_dnb64": mbpp_harm_delta,
        "join_ok": join_ok,
    }
    passes = {
        "gsm8k_card_beats_dnb84_by_2_net_repaired": card_vs_dnb84_gsm8k["net_repaired"] >= 2,
        "gsm8k_card_beats_rand84_by_2_net_repaired": card_vs_rand84_gsm8k["net_repaired"] >= 2,
        "mbpp_card_harm_not_worse_than_dnb84_plus_3": mbpp_harm_delta <= 3,
        "artifacts_joinable": join_ok,
    }

    selected_candidate = "cand_audit_mainline" if all(passes.values()) else "cand_negative_audit_pivot"
    decision_payload = {
        "pilot_id": "pilot_evidence_closure_v1",
        "selected_candidate": selected_candidate,
        "passes": passes,
        "gate_details": gate_details,
        "trajectory_addon_allowed": selected_candidate == "cand_audit_mainline"
        and (ARMS_DIR / "card84" / "seed42" / "revision_trace.jsonl").exists()
        and (ARMS_DIR / "rand84" / "seed42" / "revision_trace.jsonl").exists(),
        "arm_metrics": {arm: metrics_by_arm[arm]["overall"] for arm in ARM_META},
    }
    decision_path = ANALYSIS_DIR / "decision.json"
    decision_path.write_text(json.dumps(decision_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    claim_map = {
        "claims": [
            {
                "claim_id": "claim_runtime_contract",
                "decision": "supported",
                "supporting_artifact": str((SETUP_DIR / "runtime_contract.json").relative_to(ROOT)),
                "sample_join_key": "sample_id",
            },
            {
                "claim_id": "claim_card_beats_dnb84_on_gsm8k",
                "decision": "supported" if passes["gsm8k_card_beats_dnb84_by_2_net_repaired"] else "not_supported",
                "supporting_artifact": str(transition_path.relative_to(ROOT)),
                "sample_join_key": "sample_id",
            },
            {
                "claim_id": "claim_card_beats_rand84_on_gsm8k",
                "decision": "supported" if passes["gsm8k_card_beats_rand84_by_2_net_repaired"] else "not_supported",
                "supporting_artifact": str(transition_path.relative_to(ROOT)),
                "sample_join_key": "sample_id",
            },
            {
                "claim_id": "claim_mbpp_harm_is_bounded",
                "decision": "supported" if passes["mbpp_card_harm_not_worse_than_dnb84_plus_3"] else "not_supported",
                "supporting_artifact": str((ANALYSIS_DIR / "code_failure_modes.md").relative_to(ROOT)),
                "sample_join_key": "sample_id",
            },
        ]
    }
    claim_map_path = ANALYSIS_DIR / "claim_to_asset_map.json"
    claim_map_path.write_text(json.dumps(claim_map, indent=2, ensure_ascii=False), encoding="utf-8")

    code_failure_lines = [
        "# MBPP Failure Modes",
        "",
        "本文件统计 MBPP 上各 arm 的失败桶，以及关键 harm 对比。",
        "",
    ]
    for arm in ("dnb64", "dnb84", "card84", "rand84"):
        code_failure_lines.append(f"## {ARM_META[arm]['label']}")
        for bucket, count in sorted(mbpp_harm_buckets[arm].items()):
            code_failure_lines.append(f"- `{bucket}`: {count}")
        if not mbpp_harm_buckets[arm]:
            code_failure_lines.append("- `passed_only`: 0")
        code_failure_lines.append("")
    code_failure_lines.extend(
        [
            "## Harm Delta",
            f"- `card84_vs_dnb64 harmed`: {card_vs_dnb64_mbpp['harmed']}",
            f"- `dnb84_vs_dnb64 harmed`: {dnb84_vs_dnb64_mbpp['harmed']}",
            f"- `delta(card - dnb84)`: {mbpp_harm_delta}",
            "",
        ]
    )
    code_failure_path = ANALYSIS_DIR / "code_failure_modes.md"
    code_failure_path.write_text("\n".join(code_failure_lines), encoding="utf-8")

    summary_lines = [
        "# Pilot Claim Gate Summary",
        "",
        f"- 结论分支: `{selected_candidate}`",
        f"- `CARD-84` overall accuracy: {metrics_by_arm['card84']['overall']['accuracy']}",
        f"- `RAND-84` overall accuracy: {metrics_by_arm['rand84']['overall']['accuracy']}",
        f"- GSM8K `card84_vs_dnb84` net repaired: {card_vs_dnb84_gsm8k['net_repaired']}",
        f"- GSM8K `card84_vs_rand84` net repaired: {card_vs_rand84_gsm8k['net_repaired']}",
        f"- MBPP harm delta (`CARD-84` minus `DNB-84`, both against `DNB-64`): {mbpp_harm_delta}",
        "",
        "## Main Readout",
        "",
        f"- `DNB-64`: accuracy={metrics_by_arm['dnb64']['overall']['accuracy']}, gsm8k={metrics_by_arm['dnb64']['by_dataset']['gsm8k']['accuracy']}, mbpp={metrics_by_arm['dnb64']['by_dataset']['mbpp']['accuracy']}",
        f"- `DNB-84`: accuracy={metrics_by_arm['dnb84']['overall']['accuracy']}, gsm8k={metrics_by_arm['dnb84']['by_dataset']['gsm8k']['accuracy']}, mbpp={metrics_by_arm['dnb84']['by_dataset']['mbpp']['accuracy']}",
        f"- `CARD-84`: accuracy={metrics_by_arm['card84']['overall']['accuracy']}, gsm8k={metrics_by_arm['card84']['by_dataset']['gsm8k']['accuracy']}, mbpp={metrics_by_arm['card84']['by_dataset']['mbpp']['accuracy']}",
        f"- `RAND-84`: accuracy={metrics_by_arm['rand84']['overall']['accuracy']}, gsm8k={metrics_by_arm['rand84']['by_dataset']['gsm8k']['accuracy']}, mbpp={metrics_by_arm['rand84']['by_dataset']['mbpp']['accuracy']}",
        "",
        "## Gate Details",
        "",
    ]
    for key, value in passes.items():
        summary_lines.append(f"- `{key}`: {value}")
    summary_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `per_sample_audit.csv`: {audit_path.relative_to(ROOT)}",
            f"- `transition_matrix.csv`: {transition_path.relative_to(ROOT)}",
            f"- `decision.json`: {decision_path.relative_to(ROOT)}",
            f"- `claim_to_asset_map.json`: {claim_map_path.relative_to(ROOT)}",
            f"- `code_failure_modes.md`: {code_failure_path.relative_to(ROOT)}",
            f"- `runtime_contract.json`: {(SETUP_DIR / 'runtime_contract.json').relative_to(ROOT)}",
            "",
            "## Runtime Contract",
            "",
            f"- attention backend: `{runtime_contract.get('attention_backend') or runtime_contract.get('resolved_attn_implementation') or 'unknown'}`",
            f"- flash attention available: `{runtime_contract.get('flash_attention_2_available', runtime_contract.get('flash_attention_available'))}`",
            f"- compile enabled in production arms: `False`",
        ]
    )
    summary_path = ANALYSIS_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
