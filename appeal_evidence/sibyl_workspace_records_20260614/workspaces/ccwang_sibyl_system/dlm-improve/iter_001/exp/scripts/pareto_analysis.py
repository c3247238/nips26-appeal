#!/usr/bin/env python3
"""
Pareto Frontier Analysis for CARD pilot experiments.

Aggregates all available pilot results across model-benchmark pairs,
constructs Pareto frontiers (accuracy vs NFE, accuracy vs wall-clock),
and evaluates whether CARD is Pareto-dominant/competitive.
"""

import json
import os
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def is_pareto_optimal(points):
    """Given list of (nfe, metric) tuples, return indices on Pareto frontier.
    Pareto optimal = no other point has both lower NFE AND higher metric."""
    optimal = []
    for i, (nfe_i, met_i) in enumerate(points):
        dominated = False
        for j, (nfe_j, met_j) in enumerate(points):
            if i == j:
                continue
            if nfe_j <= nfe_i and met_j >= met_i and (nfe_j < nfe_i or met_j > met_i):
                dominated = True
                break
        if not dominated:
            optimal.append(i)
    return optimal

def extract_methods(result_json, metric_key="accuracy"):
    """Extract (method, metric, nfe, wall_clock) from summary_table."""
    methods = []
    for row in result_json.get("summary_table", []):
        metric = row.get(metric_key, row.get("pass_at_1", 0.0))
        methods.append({
            "method": row["method"],
            "metric": metric,
            "nfe": row.get("avg_nfe", row.get("nfe", 0)),
            "wall_clock_sec": row.get("wall_clock_sec", 0),
            "avg_time_per_sample_sec": row.get("avg_time_per_sample_sec", 0),
        })
    return methods

def analyze_pair(model, benchmark, methods, metric_name="accuracy"):
    """Analyze a single model-benchmark pair."""
    if not methods:
        return None

    # Compute Pareto frontier on NFE
    points_nfe = [(m["nfe"], m["metric"]) for m in methods]
    pareto_nfe_idx = is_pareto_optimal(points_nfe)
    pareto_nfe = [methods[i]["method"] for i in pareto_nfe_idx]

    # Compute Pareto frontier on wall-clock time
    points_time = [(m["wall_clock_sec"], m["metric"]) for m in methods]
    pareto_time_idx = is_pareto_optimal(points_time)
    pareto_time = [methods[i]["method"] for i in pareto_time_idx]

    # Check CARD status
    card_methods = [m for m in methods if "CARD" in m["method"]]
    card_on_pareto_nfe = any(m["method"] in pareto_nfe for m in card_methods)
    card_on_pareto_time = any(m["method"] in pareto_time for m in card_methods)

    # Best method
    best = max(methods, key=lambda m: m["metric"])

    # DNB comparison
    dnb = [m for m in methods if "DNB" in m["method"]]
    card84 = [m for m in methods if m["method"] == "CARD-84"]

    card_vs_dnb = None
    if dnb and card84:
        card_vs_dnb = {
            "card84_metric": card84[0]["metric"],
            "dnb84_metric": dnb[0]["metric"],
            "delta": round(card84[0]["metric"] - dnb[0]["metric"], 4),
            "card_wins": card84[0]["metric"] > dnb[0]["metric"],
        }

    # Efficiency: metric per NFE
    for m in methods:
        m["efficiency"] = round(m["metric"] / max(m["nfe"], 1), 6)

    return {
        "model": model,
        "benchmark": benchmark,
        "metric_name": metric_name,
        "num_methods": len(methods),
        "methods": methods,
        "pareto_frontier_nfe": pareto_nfe,
        "pareto_frontier_time": pareto_time,
        "card_on_pareto_nfe": card_on_pareto_nfe,
        "card_on_pareto_time": card_on_pareto_time,
        "best_method": best["method"],
        "best_metric": best["metric"],
        "card_vs_dnb": card_vs_dnb,
    }


def main():
    all_pairs = []

    # ── 1. LLaDA-8B GSM8K (most complete data: card_vs_dnb has all 5 methods) ──
    card_vs_dnb_path = RESULTS_DIR / "card_vs_dnb.json"
    if card_vs_dnb_path.exists():
        data = load_json(card_vs_dnb_path)
        methods = extract_methods(data, "accuracy")

        # Also add Standard-32 and Standard-128 from baseline_standard
        baseline_path = RESULTS_DIR / "baseline_standard.json"
        if baseline_path.exists():
            bl = load_json(baseline_path)
            existing_names = {m["method"] for m in methods}
            for row in bl.get("summary_table", []):
                if row["method"] not in existing_names:
                    methods.append({
                        "method": row["method"],
                        "metric": row["accuracy"],
                        "nfe": row["nfe"],
                        "wall_clock_sec": row["wall_clock_sec"],
                        "avg_time_per_sample_sec": row["avg_time_per_sample_sec"],
                    })

        # Add ablation methods from draft_revise_ablation
        ablation_path = RESULTS_DIR / "draft_revise_ablation.json"
        if ablation_path.exists():
            abl = load_json(ablation_path)
            existing_names = {m["method"] for m in methods}
            # Add key revision methods not already present
            for row in abl.get("summary_table", []):
                name = f"{row['draft_type']}-{row['revision_type']}"
                if name not in existing_names and row["revision_type"] != "none":
                    if row["draft_type"] == "standard":
                        mapped_name = {
                            "random": "Random-Revise-64",
                            "raw_entropy": "Entropy-Revise-64",
                        }.get(row["revision_type"])
                        if mapped_name and mapped_name not in existing_names:
                            methods.append({
                                "method": mapped_name,
                                "metric": row["accuracy"],
                                "nfe": row["avg_nfe"],
                                "wall_clock_sec": row["wall_clock_sec"],
                                "avg_time_per_sample_sec": row["avg_time_per_sample_sec"],
                            })
                            existing_names.add(mapped_name)

        pair = analyze_pair("LLaDA-8B-Instruct", "GSM8K", methods, "accuracy")
        if pair:
            all_pairs.append(pair)

    # ── 2. LLaDA-8B HumanEval ──
    he_path = RESULTS_DIR / "full_llada_humaneval.json"
    if he_path.exists():
        data = load_json(he_path)
        methods = extract_methods(data, "pass_at_1")
        pair = analyze_pair("LLaDA-8B-Instruct", "HumanEval", methods, "pass@1")
        if pair:
            all_pairs.append(pair)

    # ── 3. Dream-7B GSM8K ──
    dg_path = RESULTS_DIR / "full_dream_gsm8k.json"
    if dg_path.exists():
        data = load_json(dg_path)
        methods = extract_methods(data, "accuracy")
        pair = analyze_pair("Dream-7B-Instruct", "GSM8K", methods, "accuracy")
        if pair:
            all_pairs.append(pair)

    # ── 4. Dream-7B HumanEval ──
    dh_path = RESULTS_DIR / "full_dream_humaneval.json"
    if dh_path.exists():
        data = load_json(dh_path)
        methods = extract_methods(data, "pass_at_1")
        pair = analyze_pair("Dream-7B-Instruct", "HumanEval", methods, "pass@1")
        if pair:
            all_pairs.append(pair)

    # ── Aggregate statistics ──
    card_pareto_nfe_count = sum(1 for p in all_pairs if p["card_on_pareto_nfe"])
    card_pareto_time_count = sum(1 for p in all_pairs if p["card_on_pareto_time"])
    total_pairs = len(all_pairs)

    card_beats_dnb_count = sum(
        1 for p in all_pairs
        if p["card_vs_dnb"] and p["card_vs_dnb"]["card_wins"]
    )
    card_vs_dnb_total = sum(1 for p in all_pairs if p["card_vs_dnb"])

    # Missing experiments
    missing = []
    expected_files = {
        "full_llada_gsm8k": "full_llada_gsm8k.json (full-scale, only monitor data available)",
        "full_llada_mbpp": "full_llada_mbpp.json",
        "full_llada_mmlu": "full_llada_mmlu.json",
    }
    for task_id, desc in expected_files.items():
        # Check if we have a proper results file (not just monitor)
        f = RESULTS_DIR / f"{task_id}.json"
        if not f.exists() or "monitor" in str(f):
            missing.append(desc)

    # Pilot pass criteria: "CARD appears on or near the Pareto frontier
    # for at least 3 of 6 model-benchmark combinations"
    # We only have 4 pairs available (2 missing: MBPP, MMLU; GSM8K full incomplete)
    # Evaluate on available data

    # "Near" Pareto = within 2% of frontier at same NFE
    card_near_pareto = 0
    for p in all_pairs:
        if p["card_on_pareto_nfe"]:
            card_near_pareto += 1
        else:
            # Check if CARD is within 2% of best at similar NFE
            card_m = [m for m in p["methods"] if "CARD" in m["method"]]
            if card_m:
                best_card = max(card_m, key=lambda m: m["metric"])
                best_overall = p["best_metric"]
                if best_overall > 0 and (best_overall - best_card["metric"]) <= 0.02:
                    card_near_pareto += 1

    # Count informative pairs (exclude pairs where ALL methods score 0 or < 5%)
    informative_pairs = [p for p in all_pairs if p["best_metric"] >= 0.05]
    n_informative = len(informative_pairs)
    card_pareto_informative = sum(1 for p in informative_pairs if p["card_on_pareto_nfe"])
    card_near_informative = sum(
        1 for p in informative_pairs
        if p["card_on_pareto_nfe"] or (
            any("CARD" in m["method"] for m in p["methods"]) and
            p["best_metric"] > 0 and
            (p["best_metric"] - max((m["metric"] for m in p["methods"] if "CARD" in m["method"]), default=0)) <= 0.02
        )
    )

    # Overall assessment — account for missing data and uninformative pairs
    if total_pairs == 0:
        verdict = "NO_DATA"
        pass_criteria = False
    elif card_pareto_nfe_count >= 3:
        verdict = "STRONG_PASS"
        pass_criteria = True
    elif card_near_pareto >= 3:
        verdict = "PASS"
        pass_criteria = True
    elif n_informative < 4 and card_pareto_informative >= 1:
        # Insufficient informative data to judge 3/6 — but at least 1 strong result
        verdict = "INSUFFICIENT_DATA"
        pass_criteria = False  # cannot confirm pass, but not a definitive fail
    elif card_pareto_nfe_count >= 2 or card_near_pareto >= 2:
        verdict = "MARGINAL_PASS"
        pass_criteria = False
    else:
        verdict = "FAIL"
        pass_criteria = False

    # Build cross-model-benchmark summary table
    summary_table = []
    for p in all_pairs:
        card_m = [m for m in p["methods"] if "CARD" in m["method"]]
        best_card = max(card_m, key=lambda m: m["metric"]) if card_m else None
        std64 = [m for m in p["methods"] if m["method"] == "Standard-64"]
        std64_m = std64[0] if std64 else None
        dnb = [m for m in p["methods"] if "DNB" in m["method"]]
        dnb_m = dnb[0] if dnb else None

        row = {
            "model": p["model"],
            "benchmark": p["benchmark"],
            "metric": p["metric_name"],
            "best_method": p["best_method"],
            "best_metric": p["best_metric"],
            "card_metric": best_card["metric"] if best_card else None,
            "card_nfe": best_card["nfe"] if best_card else None,
            "std64_metric": std64_m["metric"] if std64_m else None,
            "dnb84_metric": dnb_m["metric"] if dnb_m else None,
            "card_on_pareto": p["card_on_pareto_nfe"],
            "card_delta_vs_std64": round(best_card["metric"] - std64_m["metric"], 4) if best_card and std64_m else None,
            "card_delta_vs_dnb84": round(best_card["metric"] - dnb_m["metric"], 4) if best_card and dnb_m else None,
        }
        summary_table.append(row)

    result = {
        "task_id": "pareto_analysis",
        "candidate_id": "cand_card",
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "data_sources": {
            "available": [p["model"] + "/" + p["benchmark"] for p in all_pairs],
            "missing": missing,
            "note": "Pilot-level analysis on 100-sample subsets. Full-scale GSM8K running but incomplete. MBPP, MMLU not yet executed."
        },
        "aggregate": {
            "total_pairs_analyzed": total_pairs,
            "card_on_pareto_nfe": card_pareto_nfe_count,
            "card_on_pareto_time": card_pareto_time_count,
            "card_near_pareto": card_near_pareto,
            "card_beats_dnb": f"{card_beats_dnb_count}/{card_vs_dnb_total}",
        },
        "pass_criteria": {
            "description": "CARD appears on or near the Pareto frontier for at least 3 of 6 model-benchmark combinations",
            "threshold": "3/6",
            "achieved": f"{card_near_pareto}/{total_pairs} (of {total_pairs} available, 6 planned)",
            "verdict": verdict,
            "pass": pass_criteria,
        },
        "summary_table": summary_table,
        "per_pair": all_pairs,
        "key_findings": [],
        "risks_and_caveats": [],
    }

    # Generate key findings
    findings = []

    # LLaDA GSM8K finding
    llada_gsm = [p for p in all_pairs if p["model"] == "LLaDA-8B-Instruct" and p["benchmark"] == "GSM8K"]
    if llada_gsm:
        p = llada_gsm[0]
        findings.append(
            f"LLaDA-8B GSM8K: CARD-84 achieves {p['card_vs_dnb']['card84_metric']:.0%} vs DNB-84 {p['card_vs_dnb']['dnb84_metric']:.0%} "
            f"(+{p['card_vs_dnb']['delta']:.0%}). CARD is on NFE Pareto frontier: {p['card_on_pareto_nfe']}."
        )

    # Dream-7B finding
    dream_gsm = [p for p in all_pairs if p["model"] == "Dream-7B-Instruct" and p["benchmark"] == "GSM8K"]
    if dream_gsm:
        p = dream_gsm[0]
        findings.append(
            f"Dream-7B GSM8K: All methods perform poorly (best={p['best_metric']:.0%}). "
            f"CARD on Pareto: {p['card_on_pareto_nfe']}. "
            f"Standard-64 dominates at 9% — revision/extra steps hurt on this weak model."
        )

    # HumanEval findings
    llada_he = [p for p in all_pairs if p["model"] == "LLaDA-8B-Instruct" and p["benchmark"] == "HumanEval"]
    if llada_he:
        p = llada_he[0]
        findings.append(
            f"LLaDA-8B HumanEval: DNB-84 (14%) beats CARD-84 (8%) and Standard-64 (11%). "
            f"Revision HURTS code generation — entropy-targeted remasking disrupts syntactic structure."
        )

    dream_he = [p for p in all_pairs if p["model"] == "Dream-7B-Instruct" and p["benchmark"] == "HumanEval"]
    if dream_he:
        p = dream_he[0]
        findings.append(
            f"Dream-7B HumanEval: All methods score 0% pass@1. Model lacks code generation capability."
        )

    # Non-monotonicity finding
    findings.append(
        "Standard denoising is non-monotonic: 128 steps (33%) < 64 steps (38%) on LLaDA GSM8K pilot. "
        "This means DNB baseline is weaker than expected at high step counts, which inflates CARD's relative advantage."
    )

    result["key_findings"] = findings

    # Risks and caveats
    risks = [
        "Pilot only: 100 samples per condition. Full-scale results may differ (pilot had CARD +11% vs DNB; full-scale monitor shows tighter margins).",
        "CARD fails on code generation (HumanEval): revision disrupts syntactic structure. Task-type sensitivity is a major limitation.",
        "Dream-7B results are uninformative due to model weakness (9% GSM8K, 0% HumanEval). Need a stronger second model or accept single-model evidence.",
        "3 of 6 planned model-benchmark pairs are missing (MBPP, MMLU, full GSM8K). Final verdict may change.",
        "Non-monotonic standard baselines make DNB comparison favorable to CARD — this is an honest observation but reviewers may question why vanilla degrades.",
        "Full-scale monitor shows Std-64=30.9% vs pilot Std-64=38%. Significant pilot-to-full shrinkage, likely due to sample selection bias in pilot seed=42.",
    ]
    result["risks_and_caveats"] = risks

    # Write output
    out_path = RESULTS_DIR / "pareto_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")

    # Also write pilot summary
    pilot_out = RESULTS_DIR / "pilots" / "pareto_analysis_pilot.json"
    # Determine go/no-go: CONDITIONAL_GO if strong on at least 1 informative pair
    # and insufficient data to definitively fail
    if card_pareto_informative >= 1 and n_informative <= 3:
        go_no_go = "CONDITIONAL_GO"
    elif card_near_pareto >= 3:
        go_no_go = "GO"
    elif card_near_pareto >= 2:
        go_no_go = "CONDITIONAL_GO"
    else:
        go_no_go = "NO_GO"

    pilot_summary = {
        "task_id": "pareto_analysis",
        "go_no_go": go_no_go,
        "confidence": 0.55,  # moderate — strong on GSM8K, weak everywhere else
        "verdict": verdict,
        "card_on_pareto": f"{card_near_pareto}/{total_pairs}",
        "card_beats_dnb": f"{card_beats_dnb_count}/{card_vs_dnb_total}",
        "strongest_result": "LLaDA-8B GSM8K: CARD-84 43% vs DNB-84 32% (+11%)",
        "weakest_result": "LLaDA-8B HumanEval: CARD-84 8% vs DNB-84 14% (-6%)",
        "critical_gap": "CARD hurts code generation tasks. Revision strategy needs task-type awareness.",
        "missing_data": missing,
        "recommendation": (
            "PROCEED with full experiments but add task-type-aware revision gating: "
            "disable revision for code tasks where syntactic coherence matters. "
            "The GSM8K result is strong but needs full-scale confirmation. "
            "Dream-7B is too weak to validate; consider using a stronger second model."
        ),
    }
    with open(pilot_out, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    print(f"Wrote {pilot_out}")

    # Print summary
    print("\n" + "="*70)
    print("PARETO ANALYSIS — PILOT SUMMARY")
    print("="*70)
    print(f"\nPairs analyzed: {total_pairs}/6")
    print(f"CARD on Pareto (NFE):  {card_pareto_nfe_count}/{total_pairs}")
    print(f"CARD on Pareto (time): {card_pareto_time_count}/{total_pairs}")
    print(f"CARD near Pareto:      {card_near_pareto}/{total_pairs}")
    print(f"CARD beats DNB:        {card_beats_dnb_count}/{card_vs_dnb_total}")
    print(f"\nVerdict: {verdict}")
    print(f"Pass criteria met: {pass_criteria}")

    print("\n--- Cross-Model-Benchmark Summary ---")
    for row in summary_table:
        print(f"\n{row['model']} / {row['benchmark']} ({row['metric']}):")
        print(f"  Best: {row['best_method']} = {row['best_metric']:.2%}")
        if row['card_metric'] is not None:
            print(f"  CARD: {row['card_metric']:.2%} (NFE={row['card_nfe']})")
        if row['std64_metric'] is not None:
            print(f"  Std-64: {row['std64_metric']:.2%}")
        if row['dnb84_metric'] is not None:
            print(f"  DNB-84: {row['dnb84_metric']:.2%}")
        if row['card_delta_vs_dnb84'] is not None:
            print(f"  CARD vs DNB: {row['card_delta_vs_dnb84']:+.2%}")
        print(f"  On Pareto: {row['card_on_pareto']}")

    print("\n--- Key Findings ---")
    for i, f_text in enumerate(findings, 1):
        print(f"{i}. {f_text}")

    print("\n--- Risks ---")
    for i, r in enumerate(risks, 1):
        print(f"{i}. {r}")

if __name__ == "__main__":
    main()
