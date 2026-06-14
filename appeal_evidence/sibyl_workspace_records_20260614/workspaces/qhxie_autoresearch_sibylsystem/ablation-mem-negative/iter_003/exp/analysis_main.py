#!/usr/bin/env python3
"""
Main Results Summary Analysis

Summarizes all experimental findings from Iteration 3:
- P1: Collision rate proxy validation (POSITIVE)
- P2: UAD reproducibility (NEGATIVE)
- P3: Random baseline (NEGATIVE)
- F2: UAD ablations (NEGATIVE)
- F4: Extended collision correlation (POSITIVE)
- F5: False positive analysis (NEGATIVE)

Produces a comprehensive summary for the negative result paper.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("exp/results/analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "analysis_main"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))


def main():
    start_time = time.time()
    PID_FILE.write_text(str(os.getpid()))

    report_progress(0, 3, metric={"stage": "loading_results"})

    # Load all result files
    p1 = json.loads(Path("exp/results/pilots/p1_collision_proxy_results.json").read_text())
    p2 = json.loads(Path("exp/results/pilots/p2_uad_reproduce_results.json").read_text())
    p3 = json.loads(Path("exp/results/pilots/p3_random_baseline_results.json").read_text())
    f2 = json.loads(Path("exp/results/full/f2_uad_ablations_results.json").read_text())
    f4 = json.loads(Path("exp/results/full/f4_collision_correlation_results.json").read_text())
    f5 = json.loads(Path("exp/results/full/f5_false_positive_results.json").read_text())

    report_progress(1, 3, metric={"stage": "synthesizing"})

    # ─── Key Findings ───

    findings = {
        "collision_proxy_valid": {
            "status": "CONFIRMED",
            "experiment": "P1 + F4",
            "finding": "Collision rate (top-k feature overlap) is a valid proxy for true absorption rate",
            "evidence": {
                "p1_spearman_r": p1["spearman_r"],
                "p1_bootstrap_ci": p1["bootstrap_ci_95"],
                "f4_spearman_r": f4["overall"]["spearman_r"],
                "f4_bootstrap_ci": f4["overall"]["bootstrap_ci_95"],
                "f4_n_pairs": f4["overall"]["n_valid_pairs"],
                "hierarchies_tested": ["numbers", "punctuation"],
            },
            "conclusion": "Collision rate correlates strongly with true absorption (r=0.71-0.87) across multiple hierarchy types",
        },
        "uad_fails": {
            "status": "CONFIRMED_NEGATIVE",
            "experiment": "P2 + P3 + F2",
            "finding": "UAD (co-occurrence clustering) cannot detect hierarchical absorption in pre-trained SAE",
            "evidence": {
                "p2_f1": p2["uad_results"]["f1"],
                "p2_precision": p2["uad_results"]["precision"],
                "p2_recall": p2["uad_results"]["recall"],
                "p2_true_positives": p2["uad_results"]["true_positives"],
                "p2_false_positives": p2["uad_results"]["false_positives"],
                "p3_uad_f1": p3["uad_results"]["f1"],
                "p3_same_cluster_random_f1": p3["baselines"]["same_cluster_random"]["mean_f1"],
                "p3_uad_equals_random": abs(p3["uad_results"]["f1"] - p3["baselines"]["same_cluster_random"]["mean_f1"]) < 1e-10,
                "f2_best_variant": f2["ablation_results"]["kmeans"],
                "f2_full_uad_f1": f2["ablation_results"]["full_uad"]["f1"],
            },
            "conclusion": "UAD F1 = 0.00048, identical to same-cluster random baseline. The clustering step provides zero value.",
        },
        "ablation_insight": {
            "status": "DISCOVERY",
            "experiment": "F2",
            "finding": "K-means clustering on phi vectors achieves 6/7 true positives (85.7% recall) but near-zero precision",
            "evidence": {
                "kmeans_f1": f2["ablation_results"]["kmeans"]["f1"],
                "kmeans_precision": f2["ablation_results"]["kmeans"]["precision"],
                "kmeans_recall": f2["ablation_results"]["kmeans"]["recall"],
                "kmeans_true_positives": f2["ablation_results"]["kmeans"]["true_positives"],
                "kmeans_false_positives": f2["ablation_results"]["kmeans"]["false_positives"],
            },
            "conclusion": "Clustering approach matters, but even the best variant has F1=0.0037 due to massive false positives",
        },
    }

    # ─── Root Cause Analysis ───

    root_cause = {
        "primary": "Absorption features are mutually exclusive at the token level",
        "explanation": "Features that absorb the same parent concept (e.g., 'four', 'five', 'six' all absorbed by feature 24189) fire on DIFFERENT tokens representing different child concepts. They never activate on the same token.",
        "why_uad_fails": "UAD uses co-occurrence clustering (phi coefficient) to find features that fire TOGETHER. But absorption features fire on mutually exclusive instances of a parent concept, so their co-occurrence is near zero.",
        "why_proxy_works": "Collision rate (top-k feature overlap) measures structural similarity of feature responses, not co-occurrence. Two child concepts may share the same absorbing feature in their top-k responses even though they never appear together.",
    }

    # ─── Paper Positioning ───

    paper_positioning = {
        "title_suggestion": "Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders",
        "contribution_type": "Negative result with constructive insight",
        "key_contributions": [
            "Empirical demonstration that UAD (co-occurrence clustering) fails to detect hierarchical absorption (F1=0.0005)",
            "Validation that collision rate IS a valid proxy for absorption (Spearman r=0.87, CI=[0.78, 0.94])",
            "Identification of root cause: token-level mutual exclusivity of absorption features",
            "Proposed alternative approaches: decoder weight similarity, causal intervention",
        ],
        "honest_limitations": [
            "Only tested on GPT-2 Small with gpt2-small-res-jb SAE",
            "Only 7 ground truth absorption pairs (small sample)",
            "Collision rate proxy tested on 56 pairs across 2 hierarchy types",
            "No causal validation of proposed alternatives",
        ],
    }

    # ─── Main Results Table ───

    results_table = [
        {
            "experiment": "P1: Collision Proxy (First Letters)",
            "n_pairs": 10,
            "spearman_r": round(p1["spearman_r"], 3),
            "status": "PASS",
            "notes": "Proxy metric validated",
        },
        {
            "experiment": "P2: UAD Reproduction",
            "f1": round(p2["uad_results"]["f1"], 4),
            "precision": round(p2["uad_results"]["precision"], 4),
            "recall": round(p2["uad_results"]["recall"], 4),
            "status": "FAIL",
            "notes": "F1=0, only 1/7 TP",
        },
        {
            "experiment": "P3: Random Baseline",
            "uad_f1": round(p3["uad_results"]["f1"], 4),
            "random_f1": round(p3["baselines"]["analytical_global_random"]["mean_f1"], 4),
            "same_cluster_f1": round(p3["baselines"]["same_cluster_random"]["mean_f1"], 4),
            "status": "FAIL",
            "notes": "UAD = same-cluster random",
        },
        {
            "experiment": "F2: UAD Ablations",
            "best_variant": "kmeans",
            "best_f1": round(f2["ablation_results"]["kmeans"]["f1"], 4),
            "best_recall": round(f2["ablation_results"]["kmeans"]["recall"], 4),
            "status": "FAIL",
            "notes": "Even best variant has F1=0.0037",
        },
        {
            "experiment": "F4: Extended Collision Correlation",
            "n_pairs": f4["overall"]["n_valid_pairs"],
            "spearman_r": round(f4["overall"]["spearman_r"], 3),
            "ci_95": [round(f4["overall"]["bootstrap_ci_95"][0], 3), round(f4["overall"]["bootstrap_ci_95"][1], 3)],
            "status": "PASS",
            "notes": "Strong correlation across hierarchies",
        },
        {
            "experiment": "F5: False Positive Analysis",
            "n_detected": f5["summary"]["n_detected"],
            "n_fp": f5["summary"]["n_false_positives"],
            "precision": round(f5["summary"]["precision"], 6),
            "status": "NEGATIVE",
            "notes": "99.98% false positive rate",
        },
    ]

    report_progress(2, 3, metric={"stage": "writing_output"})

    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "iteration": 3,
        "overall_assessment": {
            "uad_works": False,
            "proxy_valid": True,
            "recommendation": "PIVOT to negative result paper: 'Why co-occurrence clustering cannot detect feature absorption'",
        },
        "findings": findings,
        "root_cause": root_cause,
        "paper_positioning": paper_positioning,
        "results_table": results_table,
        "go_no_go": {
            "g1_collision_proxy": "PASS - Collision rate is a valid proxy (r=0.87)",
            "g2_uad_reproduce": "FAIL - UAD cannot detect absorption (F1=0)",
            "g3_random_baseline": "FAIL - UAD equals random baseline",
        },
        "pilot_decision": {
            "recommendation": "PIVOT",
            "reason": "Core UAD method fails. Collision rate proxy is valid but UAD cannot exploit it.",
            "new_direction": "Negative result paper on co-occurrence clustering limitations + constructive proposals for alternative approaches",
        },
        "runtime_seconds": time.time() - start_time,
    }

    results_file = RESULTS_DIR / "main_analysis.json"
    results_file.write_text(json.dumps(output, indent=2))

    # Also write a markdown summary
    md_content = f"""# Iteration 3: Main Results Summary

> Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> Analysis: sibyl-experimenter

## Overall Assessment

**UAD Status: FAILED**
**Proxy Metric Status: VALIDATED**
**Recommendation: PIVOT to negative result paper**

---

## Key Findings

### 1. Collision Rate Proxy: VALIDATED (Positive Result)

| Experiment | N Pairs | Spearman r | 95% CI | Status |
|-----------|---------|-----------|--------|--------|
| P1 (First Letters) | 10 | {p1['spearman_r']:.3f} | [{p1['bootstrap_ci_95'][0]:.3f}, {p1['bootstrap_ci_95'][1]:.3f}] | PASS |
| F4 (Extended) | {f4['overall']['n_valid_pairs']} | {f4['overall']['spearman_r']:.3f} | [{f4['overall']['bootstrap_ci_95'][0]:.3f}, {f4['overall']['bootstrap_ci_95'][1]:.3f}] | PASS |

**Conclusion**: Collision rate (top-k feature overlap) is a valid proxy for true absorption rate across multiple hierarchy types.

### 2. UAD Method: FAILED (Negative Result)

| Experiment | F1 | Precision | Recall | TP | FP | Status |
|-----------|-----|-----------|--------|----|----|--------|
| P2 (Reproduction) | {p2['uad_results']['f1']:.4f} | {p2['uad_results']['precision']:.4f} | {p2['uad_results']['recall']:.4f} | {p2['uad_results']['true_positives']} | {p2['uad_results']['false_positives']} | FAIL |
| P3 (vs Random) | {p3['uad_results']['f1']:.4f} | - | - | - | - | FAIL |
| F2 (Ablations) | {f2['ablation_results']['full_uad']['f1']:.4f} | {f2['ablation_results']['full_uad']['precision']:.4f} | {f2['ablation_results']['full_uad']['recall']:.4f} | {f2['ablation_results']['full_uad']['true_positives']} | {f2['ablation_results']['full_uad']['false_positives']} | FAIL |

**Key insight**: UAD F1 ({p3['uad_results']['f1']:.6f}) equals same-cluster random baseline ({p3['baselines']['same_cluster_random']['mean_f1']:.6f}). The clustering step provides zero value.

### 3. Best Ablation Variant: K-means

- F1: {f2['ablation_results']['kmeans']['f1']:.4f}
- Precision: {f2['ablation_results']['kmeans']['precision']:.4f}
- Recall: {f2['ablation_results']['kmeans']['recall']:.4f}
- TP: {f2['ablation_results']['kmeans']['true_positives']}/7

Even the best variant has near-zero precision due to massive false positives.

---

## Root Cause

**Primary**: Absorption features are mutually exclusive at the token level.

Features that absorb the same parent concept fire on DIFFERENT tokens representing different child concepts. They never activate on the same token.

**Why UAD fails**: UAD uses co-occurrence clustering (phi coefficient) to find features that fire TOGETHER. But absorption features fire on mutually exclusive instances, so their co-occurrence is near zero.

**Why proxy works**: Collision rate measures structural similarity of feature responses, not co-occurrence. Two child concepts may share the same absorbing feature in their top-k even though they never appear together.

---

## Paper Positioning

**Title**: "Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders"

**Type**: Negative result with constructive insight

**Contributions**:
1. Empirical demonstration that UAD fails (F1=0.0005)
2. Validation that collision rate IS valid (r=0.87)
3. Root cause identification: token-level mutual exclusivity
4. Proposed alternatives: decoder weight similarity, causal intervention

---

## Honest Limitations

- Only tested on GPT-2 Small with gpt2-small-res-jb SAE
- Only 7 ground truth absorption pairs
- Collision rate tested on 56 pairs across 2 hierarchy types
- No causal validation of proposed alternatives

---

## Decision

**PIVOT**: Abandon UAD as a detection method. Write negative result paper focusing on:
1. Why co-occurrence clustering is the wrong tool for absorption detection
2. Validation of collision rate as a proxy metric
3. Proposed alternative approaches for future work
"""

    md_file = RESULTS_DIR / "main_analysis.md"
    md_file.write_text(md_content)

    print(f"Results saved to {results_file}")
    print(f"Markdown summary saved to {md_file}")

    print("\n" + "=" * 60)
    print("MAIN ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nOverall: UAD FAILED, Proxy VALIDATED")
    print(f"Recommendation: PIVOT to negative result paper")
    print(f"\nCollision rate correlation: r={f4['overall']['spearman_r']:.3f} (VALID)")
    print(f"UAD F1: {p2['uad_results']['f1']:.4f} (FAILED)")
    print(f"UAD = Random: {abs(p3['uad_results']['f1'] - p3['baselines']['same_cluster_random']['mean_f1']) < 1e-10}")
    print("=" * 60)

    mark_done(
        status="success",
        summary="Main analysis complete: UAD fails, collision proxy valid. Recommend PIVOT to negative result paper."
    )
    return output


if __name__ == "__main__":
    main()
