#!/usr/bin/env python3
"""
Analysis: Exploratory Results (UAD + DFDA)
Aggregate UAD and DFDA results. Assess feasibility for publication.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR = RESULTS_DIR / "pilots"
TASK_ID = "analysis_exploratory"

pid_file = FULL_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = FULL_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if pid_file.exists():
        pid_file.unlink()
    progress_file = FULL_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = FULL_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def main():
    start_time = time.time()
    report_progress(0, 3, step=1, total_steps=3, metric={"phase": "init"})

    results = {
        "task_id": TASK_ID,
        "start_time": datetime.now().isoformat(),
    }

    # ── Load UAD results ───────────────────────────────────────────────────
    print("[1/3] Loading UAD results...")
    report_progress(1, 3, step=1, total_steps=3, metric={"phase": "load_uad"})

    uad_pilot = json.loads((PILOTS_DIR / "p3_uad_results.json").read_text())
    uad_full = json.loads((FULL_DIR / "f5_uad_results.json").read_text())

    results["uad"] = {
        "pilot": {
            "precision": uad_pilot["evaluation"]["precision"],
            "recall": uad_pilot["evaluation"]["recall"],
            "f1": uad_pilot["evaluation"]["f1"],
            "true_positives": uad_pilot["evaluation"]["true_positives"],
            "supervised_collisions": uad_pilot["evaluation"]["supervised_collisions"],
            "n_features": uad_pilot["top_indices_count"] if "top_indices_count" in uad_pilot else 100,
        },
        "full": {
            "precision": uad_full["evaluation"]["precision"],
            "recall": uad_full["evaluation"]["recall"],
            "f1": uad_full["evaluation"]["f1"],
            "true_positives": uad_full["evaluation"]["true_positives"],
            "supervised_collisions": uad_full["evaluation"]["supervised_collisions"],
            "n_features": uad_full["top_indices_count"],
        },
    }

    print(f"  Pilot UAD: P={results['uad']['pilot']['precision']:.3f}, R={results['uad']['pilot']['recall']:.3f}, F1={results['uad']['pilot']['f1']:.3f}")
    print(f"  Full UAD:  P={results['uad']['full']['precision']:.3f}, R={results['uad']['full']['recall']:.3f}, F1={results['uad']['full']['f1']:.3f}")

    # ── Load DFDA results ──────────────────────────────────────────────────
    print("[2/3] Loading DFDA results...")
    report_progress(2, 3, step=2, total_steps=3, metric={"phase": "load_dfda"})

    dfda_pilot = json.loads((PILOTS_DIR / "p4_dfda_results.json").read_text())
    dfda_full = json.loads((FULL_DIR / "f6_dfda_results.json").read_text())

    results["dfda"] = {
        "pilot": {
            "n_pairs": len(dfda_pilot.get("pair_results", [])),
            "mean_improvement": dfda_pilot.get("mean_improvement", 0),
            "total_params": dfda_pilot.get("total_params", 0),
        },
        "full": {
            "n_pairs": len(dfda_full.get("pair_results", [])),
            "mean_improvement": dfda_full.get("mean_improvement", 0),
            "median_improvement": dfda_full.get("median_improvement", 0),
            "total_params": dfda_full.get("total_params", 0),
        },
    }

    print(f"  Pilot DFDA: pairs={results['dfda']['pilot']['n_pairs']}, mean_imp={results['dfda']['pilot']['mean_improvement']:.2%}")
    print(f"  Full DFDA:  pairs={results['dfda']['full']['n_pairs']}, mean_imp={results['dfda']['full']['mean_improvement']:.2%}, median={results['dfda']['full']['median_improvement']:.2%}")

    # ── Assess feasibility ─────────────────────────────────────────────────
    print("[3/3] Assessing feasibility for publication...")
    report_progress(3, 3, step=3, total_steps=3, metric={"phase": "assess"})

    # UAD assessment
    uad_f1_full = results["uad"]["full"]["f1"]
    uad_recall_full = results["uad"]["full"]["recall"]

    uad_feasible = uad_f1_full >= 0.5 and uad_recall_full >= 0.8
    results["uad_feasible"] = uad_feasible
    results["uad_assessment"] = (
        "FEASIBLE for publication" if uad_feasible
        else "NEEDS IMPROVEMENT before publication"
    )

    # DFDA assessment
    dfda_mean_imp = results["dfda"]["full"]["mean_improvement"]
    dfda_feasible = dfda_mean_imp > 0.05  # >5% improvement
    results["dfda_feasible"] = dfda_feasible
    results["dfda_assessment"] = (
        "FEASIBLE for publication" if dfda_feasible
        else "NEEDS IMPROVEMENT before publication"
    )

    print(f"\n  UAD Assessment: {results['uad_assessment']} (F1={uad_f1_full:.3f})")
    print(f"  DFDA Assessment: {results['dfda_assessment']} (mean_imp={dfda_mean_imp:.2%})")

    # Overall
    overall_feasible = uad_feasible or dfda_feasible
    results["overall_feasible"] = overall_feasible
    results["overall_assessment"] = (
        "At least one exploratory direction is feasible for publication"
        if overall_feasible
        else "Both exploratory directions need significant improvement"
    )

    print(f"\n  Overall: {results['overall_assessment']}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = FULL_DIR / "exploratory_analysis.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    summary = (
        f"UAD: F1={uad_f1_full:.3f} ({results['uad_assessment']}) | "
        f"DFDA: mean_imp={dfda_mean_imp:.2%} ({results['dfda_assessment']}) | "
        f"elapsed={elapsed:.1f}s"
    )

    mark_done(status="success", summary=summary)
    print("\nExploratory analysis complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
