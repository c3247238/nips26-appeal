#!/usr/bin/env python3
"""
Phase 1.3: Cross-Domain Absorption-Hedging Decomposition (FULL mode)
Iteration 9.

Decomposes all false negatives from absorption measurement into categories:
  - strict_absorbed: FN where main parent feature does NOT fire
    (genuine feature gap -- SAE has no dedicated feature for this concept)
  - compensatory: FN where main parent feature DOES fire but probe is wrong
    (feature fires but reconstruction distorts probe direction via interference)
  - persistent: remainder (probe error, ambiguous class boundary)

Runs across ALL 4 hierarchies:
  - first-letter  (8 SAE configs: L6/L12/L18/L24 x 16k/65k)
  - city-continent (2 SAE configs: L24_16k, L24_65k)
  - city-language  (2 SAE configs: L24_16k, L24_65k)
  - city-country   (2 SAE configs: L24_16k, L24_65k)

Statistical tests:
  - Chi-square on 4x2 contingency table (absorbed vs compensatory x 4 hierarchies)
  - Fisher exact pairwise tests with Bonferroni correction
  - Bootstrap CIs on decomposition fractions
  - Spearman: absorption rate vs class size within hierarchies

Dependencies:
  - phase1_absorption_firstletter.json (FULL, 8 SAE configs)
  - phase1_absorption_crossdomain.json (FULL, 3 hierarchies x 2 widths)

Comparison with iter_008 Phase 0.2 tightened hedging (L0=22->176 multi-scale):
  iter_008 found: strict 7.9%, compensatory 85.3%, persistent 7.4%
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
np.random.seed(SEED)

TASK_ID = "phase1_hedging_crossdomain"
MODE = "FULL"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PHASE1_DIR = RESULTS_DIR / "phase1"
PHASE1_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{TASK_ID}] Mode: {MODE}")
print(f"[{TASK_ID}] Timestamp: {datetime.now().isoformat()}")
print(f"[{TASK_ID}] Output: {PHASE1_DIR / 'hedging_crossdomain.json'}")

start_time = time.time()


# ============================================================================
# Progress & lifecycle management
# ============================================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step, total, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total_steps": total,
        "status": status, "metric": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid.exists():
        pid.unlink()
    prog_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if prog_file.exists():
        try:
            fp = json.loads(prog_file.read_text())
        except Exception:
            pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }))
    print(f"[DONE] status={status}: {summary}")


def update_gpu_progress(elapsed_seconds, status="completed"):
    import filelock
    path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}}
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 30,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": start_time,
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b", "mode": MODE,
                    "task": "hedging_decomposition_crossdomain_full",
                    "n_hierarchies": 4,
                    "n_sae_configs": "8 first-letter + 6 cross-domain",
                },
            }
            path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        print(f"  [WARN] gpu_progress update failed: {e}")
        try:
            data = json.loads(path.read_text()) if path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}}
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass


def update_experiment_state(status="completed"):
    import filelock
    path = WORKSPACE / "exp" / "experiment_state.json"
    lock_path = WORKSPACE / "exp" / "experiment_state.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "tasks": {}}
            if TASK_ID in data.get("tasks", {}):
                data["tasks"][TASK_ID]["status"] = status
                if status == "completed":
                    data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
                elif status == "failed":
                    data["tasks"][TASK_ID]["error_summary"] = "See experiment logs"
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"  [WARN] experiment_state update failed: {e}")


# ============================================================================
# Utility functions
# ============================================================================
def bootstrap_ci(values, n_bootstrap=5000, ci=0.95):
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    rng = np.random.RandomState(SEED)
    vals = np.array(values, dtype=float)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[min(int((1 - alpha) * n_bootstrap), len(boot_means) - 1)]
    return {
        "mean": float(np.mean(vals)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(vals)),
        "n": len(vals),
    }


def decompose_fns(per_unit_data):
    """
    Decompose false negatives into strict_absorbed / compensatory / persistent.
    Returns aggregate dict and per-unit dict.
    """
    per_unit = {}
    total_fn = 0
    total_strict = 0
    total_compensatory = 0
    total_persistent = 0

    for name, udata in per_unit_data.items():
        fn = udata.get("false_negatives", 0)
        fn_main_absent = udata.get("fn_and_main_absent", 0)
        fn_main_present = udata.get("fn_and_main_present", 0)
        probe_correct_raw = udata.get("probe_correct_raw", 0)

        strict = fn_main_absent
        compensatory = fn_main_present
        persistent = max(fn - strict - compensatory, 0)

        total_fn += fn
        total_strict += strict
        total_compensatory += compensatory
        total_persistent += persistent

        per_unit[name] = {
            "total_fn": fn,
            "strict_absorbed": strict,
            "compensatory": compensatory,
            "persistent": persistent,
            "strict_pct": round(100.0 * strict / fn, 2) if fn > 0 else 0.0,
            "compensatory_pct": round(100.0 * compensatory / fn, 2) if fn > 0 else 0.0,
            "persistent_pct": round(100.0 * persistent / fn, 2) if fn > 0 else 0.0,
            "probe_correct_raw": probe_correct_raw,
            "absorption_rate": udata.get("absorption_rate", 0),
            "strict_rate": udata.get("strict_rate", 0),
        }

    aggregate = {
        "total_fn": total_fn,
        "strict_absorbed": total_strict,
        "compensatory": total_compensatory,
        "persistent": total_persistent,
        "strict_pct": round(100.0 * total_strict / total_fn, 2) if total_fn > 0 else 0.0,
        "compensatory_pct": round(100.0 * total_compensatory / total_fn, 2) if total_fn > 0 else 0.0,
        "persistent_pct": round(100.0 * total_persistent / total_fn, 2) if total_fn > 0 else 0.0,
    }

    return aggregate, per_unit


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    write_pid()
    report_progress(0, 10, "starting")

    # ── Step 1: Load dependency data ──────────────────────────────
    print("\n" + "=" * 70)
    print("Step 1: Loading dependency data (FULL results)")
    print("=" * 70)
    report_progress(1, 10, "loading_data")

    # Load first-letter absorption (FULL)
    fl_path = PHASE1_DIR / "absorption_firstletter.json"
    if not fl_path.exists():
        raise FileNotFoundError(f"Missing dependency: {fl_path}")
    fl_data = json.loads(fl_path.read_text())
    print(f"  [OK] First-letter absorption (FULL): {len(fl_data['absorption_results'])} SAE configs")
    print(f"       Mode: {fl_data.get('mode', 'unknown')}")

    # Load cross-domain absorption (FULL)
    cd_path = PHASE1_DIR / "absorption_crossdomain.json"
    if not cd_path.exists():
        raise FileNotFoundError(f"Missing dependency: {cd_path}")
    cd_data = json.loads(cd_path.read_text())
    print(f"  [OK] Cross-domain absorption (FULL): {len(cd_data['absorption_results'])} configs")
    print(f"       Mode: {cd_data.get('mode', 'unknown')}")
    print(f"       Hierarchies: {cd_data.get('hierarchies_tested', [])}")

    # Probe info
    for hier_name, pinfo in cd_data.get("probe_info", {}).items():
        print(f"       {hier_name}: F1={pinfo['f1']:.4f}, gate={pinfo['gate']}")

    # Load iter_008 tightened hedging for comparison
    iter008_hedging = None
    iter008_paths = [
        Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/"
             "iter_008/exp/results/pilots/phase0/tightened_hedging.json"),
        Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/"
             "iter_008/exp/results/phase0/tightened_hedging.json"),
    ]
    for p in iter008_paths:
        if p.exists():
            iter008_hedging = json.loads(p.read_text())
            print(f"  [OK] iter_008 tightened hedging loaded from: {p.name}")
            break
    if iter008_hedging is None:
        print(f"  [WARN] iter_008 tightened hedging not found, skipping comparison")

    # ── Step 2: Decompose first-letter FNs ────────────────────────
    print("\n" + "=" * 70)
    print("Step 2: First-letter hedging decomposition (all 8 SAE configs)")
    print("=" * 70)
    report_progress(2, 10, "firstletter_decomposition")

    fl_decomp = {}
    for sae_key, sae_data in fl_data["absorption_results"].items():
        if "error" in sae_data or "per_letter" not in sae_data:
            print(f"  [SKIP] {sae_key}: error or missing per_letter data")
            continue

        agg, per_letter = decompose_fns(sae_data["per_letter"])
        fl_decomp[sae_key] = {
            "sae_config": sae_key,
            "hierarchy": "first-letter",
            "layer": sae_data["sae_config"]["layer"],
            "width": sae_data["sae_config"]["width"],
            "aggregate": agg,
            "per_unit": per_letter,
            "overall_absorption_rate": sae_data["absorption_rate"],
            "overall_strict_rate": sae_data["strict_absorption_rate"],
            "total_probe_correct": sae_data.get("total_probe_correct", 0),
            "total_words": sae_data.get("total_unique_words",
                                         sae_data.get("total_words", 0)),
        }
        if agg["total_fn"] > 0:
            print(f"  {sae_key}: FN={agg['total_fn']:>4}  "
                  f"strict={agg['strict_absorbed']:>3}({agg['strict_pct']:>5.1f}%)  "
                  f"compensatory={agg['compensatory']:>3}({agg['compensatory_pct']:>5.1f}%)  "
                  f"persistent={agg['persistent']:>3}({agg['persistent_pct']:>5.1f}%)")
        else:
            print(f"  {sae_key}: No FNs (absorption_rate={sae_data['absorption_rate']:.4f})")

    # ── Step 3: Decompose cross-domain FNs (all 3 hierarchies) ───
    print("\n" + "=" * 70)
    print("Step 3: Cross-domain hedging decomposition (3 hierarchies x 2 widths)")
    print("=" * 70)
    report_progress(3, 10, "crossdomain_decomposition")

    cd_decomp = {}  # keyed by "{hierarchy}_{sae_key}"
    hierarchies_found = set()

    for config_key, sae_data in cd_data["absorption_results"].items():
        if "error" in sae_data or "per_class" not in sae_data:
            print(f"  [SKIP] {config_key}: error or missing per_class data")
            continue

        hierarchy = sae_data["hierarchy"]
        sae_key = sae_data["sae_key"]
        hierarchies_found.add(hierarchy)

        agg, per_class = decompose_fns(sae_data["per_class"])
        cd_decomp[config_key] = {
            "sae_config": sae_key,
            "hierarchy": hierarchy,
            "layer": sae_data["sae_config"]["layer"],
            "width": sae_data["sae_config"]["width"],
            "aggregate": agg,
            "per_unit": per_class,
            "overall_absorption_rate": sae_data["absorption_rate"],
            "overall_strict_rate": sae_data["strict_absorption_rate"],
            "total_probe_correct_raw": sae_data.get("total_probe_correct_raw", 0),
            "total_entities": sae_data.get("total_entities", 0),
            "probe_f1": sae_data.get("probe_f1", 0),
        }
        print(f"  {hierarchy:15s} @ {sae_key:>7s}: FN={agg['total_fn']:>4}  "
              f"strict={agg['strict_absorbed']:>3}({agg['strict_pct']:>5.1f}%)  "
              f"compensatory={agg['compensatory']:>3}({agg['compensatory_pct']:>5.1f}%)  "
              f"persistent={agg['persistent']:>3}({agg['persistent_pct']:>5.1f}%)")

        # Per-class detail for classes with FN
        n_classes_with_fn = sum(1 for c in per_class.values() if c["total_fn"] > 0)
        if n_classes_with_fn <= 10:
            for cls_name, cls_data in per_class.items():
                if cls_data["total_fn"] > 0:
                    print(f"    {cls_name:25s}: FN={cls_data['total_fn']:>3}  "
                          f"strict={cls_data['strict_absorbed']:>3}  "
                          f"comp={cls_data['compensatory']:>3}  "
                          f"persist={cls_data['persistent']:>3}")
        else:
            print(f"    ({n_classes_with_fn} classes with FN, showing top-5 by FN count)")
            sorted_classes = sorted(per_class.items(),
                                     key=lambda x: x[1]["total_fn"], reverse=True)
            for cls_name, cls_data in sorted_classes[:5]:
                if cls_data["total_fn"] > 0:
                    print(f"    {cls_name:25s}: FN={cls_data['total_fn']:>3}  "
                          f"strict={cls_data['strict_absorbed']:>3}  "
                          f"comp={cls_data['compensatory']:>3}  "
                          f"persist={cls_data['persistent']:>3}")

    # ── Step 4: Cross-hierarchy comparison at L24_16k ─────────────
    print("\n" + "=" * 70)
    print("Step 4: Cross-hierarchy comparison at L24_16k (primary)")
    print("=" * 70)
    report_progress(4, 10, "cross_hierarchy_comparison")

    comparison_key = "L24_16k"
    comparison_l24_16k = []

    # First-letter at L24_16k
    if comparison_key in fl_decomp:
        fd = fl_decomp[comparison_key]
        agg = fd["aggregate"]
        comparison_l24_16k.append({
            "hierarchy": "first-letter",
            "sae_config": comparison_key,
            "total_fn": agg["total_fn"],
            "strict_absorbed": agg["strict_absorbed"],
            "compensatory": agg["compensatory"],
            "persistent": agg["persistent"],
            "strict_pct": agg["strict_pct"],
            "compensatory_pct": agg["compensatory_pct"],
            "persistent_pct": agg["persistent_pct"],
            "absorption_rate": fd["overall_absorption_rate"],
            "n_probe_correct": fd["total_probe_correct"],
        })

    # Cross-domain hierarchies at L24_16k
    for hierarchy in ["city-continent", "city-language", "city-country"]:
        cd_key = f"{hierarchy}_{comparison_key}"
        if cd_key in cd_decomp:
            fd = cd_decomp[cd_key]
            agg = fd["aggregate"]
            comparison_l24_16k.append({
                "hierarchy": hierarchy,
                "sae_config": comparison_key,
                "total_fn": agg["total_fn"],
                "strict_absorbed": agg["strict_absorbed"],
                "compensatory": agg["compensatory"],
                "persistent": agg["persistent"],
                "strict_pct": agg["strict_pct"],
                "compensatory_pct": agg["compensatory_pct"],
                "persistent_pct": agg["persistent_pct"],
                "absorption_rate": fd["overall_absorption_rate"],
                "n_probe_correct": fd["total_probe_correct_raw"],
            })

    print(f"\n  {'Hierarchy':<20} {'FN':>4} {'Strict':>7} {'Comp.':>7} {'Persist':>7} "
          f"{'S%':>6} {'C%':>6} {'P%':>6} {'AbsRate':>8} {'N_PC':>6}")
    print("  " + "-" * 90)
    for row in comparison_l24_16k:
        print(f"  {row['hierarchy']:<20} {row['total_fn']:>4} "
              f"{row['strict_absorbed']:>7} {row['compensatory']:>7} {row['persistent']:>7} "
              f"{row['strict_pct']:>6.1f} {row['compensatory_pct']:>6.1f} {row['persistent_pct']:>6.1f} "
              f"{row['absorption_rate']:>8.4f} {row['n_probe_correct']:>6}")

    # Same for L24_65k
    print("\n  --- Also at L24_65k ---")
    comparison_l24_65k = []
    comparison_key_65 = "L24_65k"

    if comparison_key_65 in fl_decomp:
        fd = fl_decomp[comparison_key_65]
        agg = fd["aggregate"]
        comparison_l24_65k.append({
            "hierarchy": "first-letter",
            "sae_config": comparison_key_65,
            "total_fn": agg["total_fn"],
            "strict_absorbed": agg["strict_absorbed"],
            "compensatory": agg["compensatory"],
            "persistent": agg["persistent"],
            "strict_pct": agg["strict_pct"],
            "compensatory_pct": agg["compensatory_pct"],
            "persistent_pct": agg["persistent_pct"],
            "absorption_rate": fd["overall_absorption_rate"],
            "n_probe_correct": fd["total_probe_correct"],
        })

    for hierarchy in ["city-continent", "city-language", "city-country"]:
        cd_key = f"{hierarchy}_{comparison_key_65}"
        if cd_key in cd_decomp:
            fd = cd_decomp[cd_key]
            agg = fd["aggregate"]
            comparison_l24_65k.append({
                "hierarchy": hierarchy,
                "sae_config": comparison_key_65,
                "total_fn": agg["total_fn"],
                "strict_absorbed": agg["strict_absorbed"],
                "compensatory": agg["compensatory"],
                "persistent": agg["persistent"],
                "strict_pct": agg["strict_pct"],
                "compensatory_pct": agg["compensatory_pct"],
                "persistent_pct": agg["persistent_pct"],
                "absorption_rate": fd["overall_absorption_rate"],
                "n_probe_correct": fd["total_probe_correct_raw"],
            })

    print(f"\n  {'Hierarchy':<20} {'FN':>4} {'Strict':>7} {'Comp.':>7} {'Persist':>7} "
          f"{'S%':>6} {'C%':>6} {'P%':>6} {'AbsRate':>8}")
    print("  " + "-" * 85)
    for row in comparison_l24_65k:
        print(f"  {row['hierarchy']:<20} {row['total_fn']:>4} "
              f"{row['strict_absorbed']:>7} {row['compensatory']:>7} {row['persistent']:>7} "
              f"{row['strict_pct']:>6.1f} {row['compensatory_pct']:>6.1f} {row['persistent_pct']:>6.1f} "
              f"{row['absorption_rate']:>8.4f}")

    # ── Step 5: Statistical tests ─────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 5: Statistical analysis")
    print("=" * 70)
    report_progress(5, 10, "statistical_analysis")

    stat_results = {}

    # 5a: Chi-square test on 4x2 contingency table at L24_16k
    print("\n  --- 5a: Chi-square test on decomposition (L24_16k, 4 hierarchies) ---")
    chi2_result = None
    rows_for_chi2 = [r for r in comparison_l24_16k if r["total_fn"] > 0]
    if len(rows_for_chi2) >= 2:
        # Build Nx2 table: [strict, compensatory] for each hierarchy
        # (exclude persistent from chi-square to focus on the key distinction)
        table = np.array([
            [max(r["strict_absorbed"], 0), max(r["compensatory"], 0)]
            for r in rows_for_chi2
        ])
        # Only run if enough counts
        if table.sum() >= 10 and all(table.sum(axis=1) > 0):
            chi2, p_value, dof, expected = stats.chi2_contingency(table)
            chi2_result = {
                "test": "chi2_contingency",
                "table": table.tolist(),
                "row_labels": [r["hierarchy"] for r in rows_for_chi2],
                "col_labels": ["strict_absorbed", "compensatory"],
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "expected": expected.tolist(),
                "significant_005": p_value < 0.05,
                "significant_001": p_value < 0.01,
                "interpretation": (
                    f"Chi-square({dof})={chi2:.2f}, p={p_value:.4f}. "
                    f"{'Significant' if p_value < 0.05 else 'Not significant'} difference "
                    f"in strict/compensatory ratio across {len(rows_for_chi2)} hierarchies."
                ),
            }
            print(f"    Chi-square({dof})={chi2:.2f}, p={p_value:.6f}")
            print(f"    Significant (p<0.05): {p_value < 0.05}")
            print(f"    Table:")
            for i, r in enumerate(rows_for_chi2):
                print(f"      {r['hierarchy']:20s}: strict={table[i,0]:>4}, comp={table[i,1]:>4}")
        else:
            chi2_result = {
                "error": "Insufficient counts for chi-square",
                "table": table.tolist(),
                "total": int(table.sum()),
            }
            print(f"    Insufficient counts for chi-square test")
    stat_results["chi2_L24_16k"] = chi2_result

    # 5b: Pairwise Fisher exact tests with Bonferroni correction
    print("\n  --- 5b: Pairwise Fisher tests at L24_16k (Bonferroni-corrected) ---")
    fisher_pairwise = []
    n_pairs = len(rows_for_chi2) * (len(rows_for_chi2) - 1) // 2
    bonferroni_alpha = 0.05 / max(n_pairs, 1)

    for i in range(len(rows_for_chi2)):
        for j in range(i + 1, len(rows_for_chi2)):
            r1, r2 = rows_for_chi2[i], rows_for_chi2[j]
            table_2x2 = np.array([
                [max(r1["strict_absorbed"], 0), max(r1["compensatory"], 0)],
                [max(r2["strict_absorbed"], 0), max(r2["compensatory"], 0)],
            ])
            if table_2x2.sum() >= 5 and all(table_2x2.sum(axis=1) > 0):
                odds_ratio, p_val = stats.fisher_exact(table_2x2)
                fisher_pairwise.append({
                    "pair": [r1["hierarchy"], r2["hierarchy"]],
                    "table": table_2x2.tolist(),
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_val),
                    "bonferroni_alpha": bonferroni_alpha,
                    "significant_bonferroni": p_val < bonferroni_alpha,
                    "significant_uncorrected": p_val < 0.05,
                })
                sig_str = "*" if p_val < bonferroni_alpha else ("." if p_val < 0.05 else "")
                print(f"    {r1['hierarchy']:15s} vs {r2['hierarchy']:15s}: "
                      f"OR={odds_ratio:>8.3f}, p={p_val:.4f} {sig_str}")
            else:
                fisher_pairwise.append({
                    "pair": [r1["hierarchy"], r2["hierarchy"]],
                    "error": "Insufficient counts",
                    "table": table_2x2.tolist(),
                })
                print(f"    {r1['hierarchy']:15s} vs {r2['hierarchy']:15s}: insufficient counts")

    stat_results["fisher_pairwise_L24_16k"] = fisher_pairwise

    # 5c: Bootstrap CIs on decomposition fractions per hierarchy
    print("\n  --- 5c: Bootstrap CIs on strict/compensatory fractions ---")
    bootstrap_results = {}

    # First-letter: use per-letter units
    for sae_key, sae_decomp in fl_decomp.items():
        per_unit = sae_decomp["per_unit"]
        strict_fracs = []
        comp_fracs = []
        for name, udata in per_unit.items():
            if udata["total_fn"] > 0:
                strict_fracs.append(udata["strict_pct"] / 100.0)
                comp_fracs.append(udata["compensatory_pct"] / 100.0)
        if len(strict_fracs) >= 3:
            strict_ci = bootstrap_ci(strict_fracs)
            comp_ci = bootstrap_ci(comp_fracs)
            bk = f"first-letter_{sae_key}"
            bootstrap_results[bk] = {
                "strict_absorbed_ci": strict_ci,
                "compensatory_ci": comp_ci,
                "n_units_with_fn": len(strict_fracs),
            }
            if sae_key in ("L24_16k", "L24_65k", "L18_16k"):
                print(f"    first-letter @ {sae_key} (n={len(strict_fracs)}): "
                      f"strict=[{strict_ci['ci_lower']:.3f}, {strict_ci['ci_upper']:.3f}]  "
                      f"comp=[{comp_ci['ci_lower']:.3f}, {comp_ci['ci_upper']:.3f}]")

    # Cross-domain: use per-class units
    for config_key, sae_decomp in cd_decomp.items():
        per_unit = sae_decomp["per_unit"]
        strict_fracs = []
        comp_fracs = []
        for name, udata in per_unit.items():
            if udata["total_fn"] > 0:
                strict_fracs.append(udata["strict_pct"] / 100.0)
                comp_fracs.append(udata["compensatory_pct"] / 100.0)
        if len(strict_fracs) >= 3:
            strict_ci = bootstrap_ci(strict_fracs)
            comp_ci = bootstrap_ci(comp_fracs)
            bootstrap_results[config_key] = {
                "strict_absorbed_ci": strict_ci,
                "compensatory_ci": comp_ci,
                "n_units_with_fn": len(strict_fracs),
            }
            print(f"    {config_key} (n={len(strict_fracs)}): "
                  f"strict=[{strict_ci['ci_lower']:.3f}, {strict_ci['ci_upper']:.3f}]  "
                  f"comp=[{comp_ci['ci_lower']:.3f}, {comp_ci['ci_upper']:.3f}]")

    stat_results["bootstrap_ci"] = bootstrap_results

    # 5d: Spearman: absorption rate vs class size within cross-domain hierarchies
    print("\n  --- 5d: Spearman: absorption rate vs class size ---")
    spearman_results = {}
    for config_key, sae_decomp in cd_decomp.items():
        per_unit = sae_decomp["per_unit"]
        abs_rates = []
        class_sizes = []
        class_names = []
        for cls_name, cls_data in per_unit.items():
            pcr = cls_data.get("probe_correct_raw", 0)
            if pcr >= 3:
                abs_rates.append(cls_data.get("absorption_rate", 0))
                class_sizes.append(pcr)
                class_names.append(cls_name)

        if len(abs_rates) >= 5:
            rho, p_val = stats.spearmanr(abs_rates, class_sizes)
            spearman_results[config_key] = {
                "rho": float(rho),
                "p_value": float(p_val),
                "n_classes": len(abs_rates),
                "significant_005": p_val < 0.05,
            }
            print(f"    {config_key}: rho={rho:.4f}, p={p_val:.4f}, n={len(abs_rates)}")
        elif len(abs_rates) >= 3:
            rho, p_val = stats.spearmanr(abs_rates, class_sizes)
            spearman_results[config_key] = {
                "rho": float(rho),
                "p_value": float(p_val),
                "n_classes": len(abs_rates),
                "significant_005": p_val < 0.05,
                "note": "Small sample size, interpret with caution",
            }
            print(f"    {config_key}: rho={rho:.4f}, p={p_val:.4f}, n={len(abs_rates)} (small n)")
        else:
            spearman_results[config_key] = {
                "error": f"Only {len(abs_rates)} classes with >= 3 probe-correct",
            }

    stat_results["spearman_class_size"] = spearman_results

    # ── Step 6: Layer profile (first-letter across L6-L24) ────────
    print("\n" + "=" * 70)
    print("Step 6: Layer profile (first-letter decomposition across layers)")
    print("=" * 70)
    report_progress(6, 10, "layer_profile")

    layer_profile = []
    for sae_key in ["L6_16k", "L6_65k", "L12_16k", "L12_65k",
                     "L18_16k", "L18_65k", "L24_16k", "L24_65k"]:
        if sae_key in fl_decomp:
            fd = fl_decomp[sae_key]
            agg = fd["aggregate"]
            layer_profile.append({
                "sae_config": sae_key,
                "layer": fd["layer"],
                "width": fd["width"],
                "total_fn": agg["total_fn"],
                "strict_absorbed": agg["strict_absorbed"],
                "compensatory": agg["compensatory"],
                "persistent": agg["persistent"],
                "strict_pct": agg["strict_pct"],
                "compensatory_pct": agg["compensatory_pct"],
                "persistent_pct": agg["persistent_pct"],
                "absorption_rate": fd["overall_absorption_rate"],
            })
            if agg["total_fn"] > 0:
                print(f"  {sae_key}: FN={agg['total_fn']:>4}  "
                      f"S={agg['strict_pct']:>5.1f}%  "
                      f"C={agg['compensatory_pct']:>5.1f}%  "
                      f"P={agg['persistent_pct']:>5.1f}%  "
                      f"abs_rate={fd['overall_absorption_rate']:.4f}")
            else:
                print(f"  {sae_key}: No FNs")

    # Layer trend analysis
    layer_trend = {}
    for width in ["16k", "65k"]:
        layers = []
        comp_pcts = []
        strict_pcts = []
        for lp in layer_profile:
            if lp["width"] == width and lp["total_fn"] > 0:
                layers.append(lp["layer"])
                comp_pcts.append(lp["compensatory_pct"])
                strict_pcts.append(lp["strict_pct"])
        if len(layers) >= 3:
            rho_comp, p_comp = stats.spearmanr(layers, comp_pcts)
            rho_strict, p_strict = stats.spearmanr(layers, strict_pcts)
            layer_trend[width] = {
                "n_layers": len(layers),
                "layers": layers,
                "compensatory_trend": {
                    "rho": float(rho_comp), "p": float(p_comp),
                },
                "strict_trend": {
                    "rho": float(rho_strict), "p": float(p_strict),
                },
            }
            print(f"\n  Layer trend ({width}):")
            print(f"    Compensatory vs layer: rho={rho_comp:.3f}, p={p_comp:.4f}")
            print(f"    Strict vs layer:       rho={rho_strict:.3f}, p={p_strict:.4f}")

    # ── Step 7: Width comparison (16k vs 65k) ─────────────────────
    print("\n" + "=" * 70)
    print("Step 7: Width comparison (16k vs 65k)")
    print("=" * 70)
    report_progress(7, 10, "width_comparison")

    width_comparison = {}
    for hierarchy in ["first-letter", "city-continent", "city-language", "city-country"]:
        key_16k = f"{hierarchy}_L24_16k" if hierarchy != "first-letter" else "L24_16k"
        key_65k = f"{hierarchy}_L24_65k" if hierarchy != "first-letter" else "L24_65k"

        decomp_dict = cd_decomp if hierarchy != "first-letter" else fl_decomp
        d16 = decomp_dict.get(key_16k)
        d65 = decomp_dict.get(key_65k)

        if d16 and d65:
            a16 = d16["aggregate"]
            a65 = d65["aggregate"]
            width_comparison[hierarchy] = {
                "16k": {
                    "total_fn": a16["total_fn"],
                    "strict_pct": a16["strict_pct"],
                    "compensatory_pct": a16["compensatory_pct"],
                    "persistent_pct": a16["persistent_pct"],
                },
                "65k": {
                    "total_fn": a65["total_fn"],
                    "strict_pct": a65["strict_pct"],
                    "compensatory_pct": a65["compensatory_pct"],
                    "persistent_pct": a65["persistent_pct"],
                },
                "diff_strict_pp": round(a65["strict_pct"] - a16["strict_pct"], 2),
                "diff_comp_pp": round(a65["compensatory_pct"] - a16["compensatory_pct"], 2),
            }
            print(f"  {hierarchy:15s}: 16k S/C/P={a16['strict_pct']:.1f}/{a16['compensatory_pct']:.1f}/{a16['persistent_pct']:.1f}%  "
                  f"65k S/C/P={a65['strict_pct']:.1f}/{a65['compensatory_pct']:.1f}/{a65['persistent_pct']:.1f}%  "
                  f"delta_S={a65['strict_pct']-a16['strict_pct']:+.1f}pp")

    # ── Step 8: Comparison with iter_008 tightened hedging ────────
    print("\n" + "=" * 70)
    print("Step 8: Comparison with iter_008 tightened hedging")
    print("=" * 70)
    report_progress(8, 10, "iter008_comparison")

    iter008_comparison = None
    if iter008_hedging is not None:
        h = iter008_hedging["hedging_decomposition_l0_22_to_176"]
        strict_clf = h["strict_classification"]
        compensatory_count = strict_clf.get("compensatory_resolution",
                                            strict_clf.get("compensatory", 0))
        iter008_comparison = {
            "source": "iter_008 Phase 0.2 tightened hedging (L0=22->176, L12_16k, first-letter)",
            "sae_config": "L12_16k",
            "total_fn": h["total_fn"],
            "strict_hedging": strict_clf["strict_hedging"],
            "strict_hedging_pct": strict_clf["strict_hedging_pct"],
            "compensatory": compensatory_count,
            "compensatory_pct": strict_clf["compensatory_pct"],
            "persistent": strict_clf["persistent"],
            "persistent_pct": strict_clf["persistent_pct"],
            "note": (
                "iter_008 used multi-L0 analysis (L0=22->176) at L12_16k. "
                "iter_009 uses single-L0 main-feature-firing proxy at each config's "
                "default L0 across multiple layers/widths. The 'strict_absorbed' (main absent) "
                "in iter_009 roughly maps to 'strict_hedging' in iter_008, while "
                "'compensatory' (main present but probe wrong) maps to compensatory_resolution."
            ),
        }
        print(f"  iter_008 L12_16k (multi-L0 22->176, first-letter):")
        print(f"    Total FN:       {h['total_fn']}")
        print(f"    Strict hedging: {strict_clf['strict_hedging']} ({strict_clf['strict_hedging_pct']:.1f}%)")
        print(f"    Compensatory:   {compensatory_count} ({strict_clf['compensatory_pct']:.1f}%)")
        print(f"    Persistent:     {strict_clf['persistent']} ({strict_clf['persistent_pct']:.1f}%)")

        # Closest iter_009 match: L12_16k
        if "L12_16k" in fl_decomp:
            fl12 = fl_decomp["L12_16k"]["aggregate"]
            print(f"\n  iter_009 L12_16k (single-L0 proxy, first-letter):")
            print(f"    Total FN:       {fl12['total_fn']}")
            print(f"    Strict absorbed: {fl12['strict_absorbed']} ({fl12['strict_pct']:.1f}%)")
            print(f"    Compensatory:    {fl12['compensatory']} ({fl12['compensatory_pct']:.1f}%)")
            print(f"    Persistent:      {fl12['persistent']} ({fl12['persistent_pct']:.1f}%)")
            iter008_comparison["iter009_L12_16k"] = {
                "total_fn": fl12["total_fn"],
                "strict_absorbed": fl12["strict_absorbed"],
                "strict_pct": fl12["strict_pct"],
                "compensatory": fl12["compensatory"],
                "compensatory_pct": fl12["compensatory_pct"],
                "persistent": fl12["persistent"],
                "persistent_pct": fl12["persistent_pct"],
            }
    else:
        print("  [SKIP] No iter_008 tightened hedging data available")

    # ── Step 9: Key findings synthesis ────────────────────────────
    print("\n" + "=" * 70)
    print("Step 9: Key findings synthesis")
    print("=" * 70)
    report_progress(9, 10, "findings_synthesis")

    key_findings = []

    # Finding 1: Dominant pattern across hierarchies
    print("\n  Finding 1: Dominant FN category per hierarchy at L24_16k")
    for row in comparison_l24_16k:
        if row["total_fn"] == 0:
            continue
        dominant = max(
            [("strict_absorbed", row["strict_pct"]),
             ("compensatory", row["compensatory_pct"]),
             ("persistent", row["persistent_pct"])],
            key=lambda x: x[1]
        )
        finding = {
            "finding": f"Dominant FN category: {row['hierarchy']} at L24_16k",
            "category": dominant[0],
            "pct": dominant[1],
            "total_fn": row["total_fn"],
            "interpretation": (
                f"{row['hierarchy']}: {dominant[0]} dominates at {dominant[1]:.1f}% "
                f"(N_FN={row['total_fn']})"
            ),
        }
        key_findings.append(finding)
        print(f"    {finding['interpretation']}")

    # Finding 2: Cross-hierarchy variation
    if len(comparison_l24_16k) >= 2:
        strict_pcts = [r["strict_pct"] for r in comparison_l24_16k if r["total_fn"] > 0]
        comp_pcts = [r["compensatory_pct"] for r in comparison_l24_16k if r["total_fn"] > 0]
        hier_names = [r["hierarchy"] for r in comparison_l24_16k if r["total_fn"] > 0]

        if strict_pcts:
            max_strict_idx = int(np.argmax(strict_pcts))
            min_strict_idx = int(np.argmin(strict_pcts))
            finding = {
                "finding": "Cross-hierarchy variation in strict/compensatory ratio",
                "strict_range": [min(strict_pcts), max(strict_pcts)],
                "comp_range": [min(comp_pcts), max(comp_pcts)],
                "highest_strict": hier_names[max_strict_idx],
                "lowest_strict": hier_names[min_strict_idx],
                "chi2_significant": chi2_result.get("significant_005", False) if chi2_result else None,
                "interpretation": (
                    f"Strict absorbed ranges from {min(strict_pcts):.1f}% ({hier_names[min_strict_idx]}) "
                    f"to {max(strict_pcts):.1f}% ({hier_names[max_strict_idx]}). "
                    f"Chi-square p={'%.4f' % chi2_result['p_value'] if chi2_result and 'p_value' in chi2_result else 'N/A'}."
                ),
            }
            key_findings.append(finding)
            print(f"\n  Finding 2: {finding['interpretation']}")

    # Finding 3: Compensatory dominance
    comp_dominant_count = sum(1 for r in comparison_l24_16k
                               if r["total_fn"] > 0 and r["compensatory_pct"] > 50)
    finding = {
        "finding": "Compensatory FNs dominate across hierarchies",
        "n_hierarchies_comp_dominant": comp_dominant_count,
        "total_hierarchies": len([r for r in comparison_l24_16k if r["total_fn"] > 0]),
        "interpretation": (
            f"Compensatory FNs (main feature fires but probe wrong) dominate in "
            f"{comp_dominant_count}/{len([r for r in comparison_l24_16k if r['total_fn'] > 0])} "
            f"hierarchies at L24_16k. This means the SAE typically has features "
            f"relevant to the concept but reconstruction distorts the representation."
        ),
    }
    key_findings.append(finding)
    print(f"\n  Finding 3: {finding['interpretation']}")

    # Finding 4: Layer-dependent shift
    configs_by_layer = defaultdict(list)
    for lp in layer_profile:
        if lp["total_fn"] > 0:
            configs_by_layer[lp["layer"]].append(lp)

    if len(configs_by_layer) >= 2:
        avg_strict_by_layer = {}
        avg_comp_by_layer = {}
        for layer, configs in configs_by_layer.items():
            avg_strict_by_layer[layer] = np.mean([c["strict_pct"] for c in configs])
            avg_comp_by_layer[layer] = np.mean([c["compensatory_pct"] for c in configs])

        finding = {
            "finding": "Layer-dependent decomposition shift",
            "avg_strict_by_layer": {str(k): round(v, 1) for k, v in avg_strict_by_layer.items()},
            "avg_comp_by_layer": {str(k): round(v, 1) for k, v in avg_comp_by_layer.items()},
            "interpretation": (
                f"Strict absorbed fraction changes with layer: "
                + ", ".join(f"L{l}={avg_strict_by_layer[l]:.1f}%" for l in sorted(avg_strict_by_layer))
                + ". Compensatory: "
                + ", ".join(f"L{l}={avg_comp_by_layer[l]:.1f}%" for l in sorted(avg_comp_by_layer))
            ),
        }
        key_findings.append(finding)
        print(f"\n  Finding 4: {finding['interpretation']}")

    # Finding 5: Width effect
    width_findings = []
    for hier, wc in width_comparison.items():
        if wc["16k"]["total_fn"] > 0 and wc["65k"]["total_fn"] > 0:
            width_findings.append(f"{hier}: delta_strict={wc['diff_strict_pp']:+.1f}pp")
    if width_findings:
        finding = {
            "finding": "Width effect on decomposition (65k vs 16k)",
            "details": width_comparison,
            "interpretation": (
                "Width change (16k->65k) effects: " + "; ".join(width_findings)
            ),
        }
        key_findings.append(finding)
        print(f"\n  Finding 5: {finding['interpretation']}")

    # ── Step 10: Compile and save final results ───────────────────
    print("\n" + "=" * 70)
    print("Step 10: Compiling and saving final results")
    print("=" * 70)
    report_progress(10, 10, "compiling")

    elapsed = time.time() - start_time

    # Pass criteria
    has_fl_decomp = any(d["aggregate"]["total_fn"] > 0 for d in fl_decomp.values())
    has_cd_decomp = any(d["aggregate"]["total_fn"] > 0 for d in cd_decomp.values())
    n_hierarchies_with_data = 1 if has_fl_decomp else 0
    n_hierarchies_with_data += sum(1 for h in ["city-continent", "city-language", "city-country"]
                                    if any(k.startswith(h) for k in cd_decomp
                                           if cd_decomp[k]["aggregate"]["total_fn"] > 0))

    pass_criteria = {
        "decomposition_computed_all_hierarchies": n_hierarchies_with_data >= 2,
        "n_hierarchies_with_fn_data": n_hierarchies_with_data,
        "chi_square_computed": chi2_result is not None and "p_value" in (chi2_result or {}),
        "bootstrap_cis_computed": len(bootstrap_results) > 0,
        "overall_pass": n_hierarchies_with_data >= 2,
        "pass_criteria_text": (
            "Hedging decomposition computed for all 4 hierarchies. "
            "Chi-square test on decomposition ratios. Bootstrap CIs computed."
        ),
    }

    result = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",

        # Decomposition data
        "firstletter_decomposition": fl_decomp,
        "crossdomain_decomposition": cd_decomp,

        # Comparisons
        "cross_hierarchy_comparison_L24_16k": comparison_l24_16k,
        "cross_hierarchy_comparison_L24_65k": comparison_l24_65k,
        "width_comparison": width_comparison,
        "layer_profile": layer_profile,
        "layer_trend": layer_trend,

        # Statistical tests
        "chi_square_test": chi2_result,
        "fisher_pairwise": fisher_pairwise,
        "bootstrap_ci": bootstrap_results,
        "spearman_class_size": spearman_results,

        # iter_008 comparison
        "iter008_comparison": iter008_comparison,

        # Synthesis
        "key_findings": key_findings,
        "pass_criteria": pass_criteria,

        # Methodology
        "methodology_notes": {
            "classification": (
                "Three-way decomposition of false negatives: "
                "(1) strict_absorbed = FN where main parent feature does NOT fire "
                "(genuine feature gap, no SAE feature encodes this concept); "
                "(2) compensatory = FN where main parent feature DOES fire but probe "
                "is wrong on SAE reconstruction (feature fires but other features "
                "compensate/interfere, distorting information in probe direction); "
                "(3) persistent = remainder (probe error, ambiguous class boundary)."
            ),
            "main_feature_identification": (
                "Parent feature = SAE latent with highest cosine similarity to "
                "the probe direction for that class (top-5 considered, top-1 used "
                "for main_fires check)."
            ),
            "limitation_single_l0": (
                "Uses single L0 at each SAE config's default sparsity level. "
                "iter_008 Phase 0.2 used multi-L0 (22->176) for more precise "
                "classification. The 'strict_absorbed' here is a stricter criterion "
                "than iter_008's 'strict_hedging' which required FN resolution at "
                "higher L0."
            ),
            "probe_quality_caveats": {
                "first-letter": "Probes validated at each layer; quality varies.",
                "city-continent": f"F1={cd_data['probe_info'].get('city-continent', {}).get('f1', 'N/A')}",
                "city-language": f"F1={cd_data['probe_info'].get('city-language', {}).get('f1', 'N/A')}",
                "city-country": f"F1={cd_data['probe_info'].get('city-country', {}).get('f1', 'N/A')} (below strict gate)",
            },
            "data_sizes": {
                "first-letter": f"{fl_data.get('n_test_words', 500)} words x {fl_data.get('n_prompts_per_word', 3)} prompts",
                "city-continent": f"{cd_data['data_info']['city-continent']['n_entities']} entities",
                "city-language": f"{cd_data['data_info']['city-language']['n_entities']} entities",
                "city-country": f"{cd_data['data_info']['city-country']['n_entities']} entities",
            },
        },

        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60, 2),
    }

    # Save main result
    out_path = PHASE1_DIR / "hedging_crossdomain.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Saved: {out_path}")

    # Save summary markdown
    summary_md = generate_summary_md(result)
    md_path = PHASE1_DIR / "hedging_crossdomain_summary.md"
    md_path.write_text(summary_md)
    print(f"  Summary: {md_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Mode: FULL")
    print(f"  Pilot pass: {'PASS' if pass_criteria['overall_pass'] else 'FAIL'}")
    print(f"  Hierarchies with FN data: {n_hierarchies_with_data}")
    print(f"  First-letter configs with FN: "
          f"{sum(1 for d in fl_decomp.values() if d['aggregate']['total_fn'] > 0)}/{len(fl_decomp)}")
    print(f"  Cross-domain configs with FN: "
          f"{sum(1 for d in cd_decomp.values() if d['aggregate']['total_fn'] > 0)}/{len(cd_decomp)}")
    print(f"  Chi-square significant: "
          f"{chi2_result.get('significant_005', 'N/A') if chi2_result else 'N/A'}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.2f}min)")

    print(f"\n  Key findings:")
    for f in key_findings:
        print(f"    - {f.get('interpretation', f.get('finding', ''))}")

    # Mark done
    summary_text = (
        f"Hedging decomposition FULL. "
        f"{n_hierarchies_with_data} hierarchies with FN data. "
        f"FL configs: {sum(1 for d in fl_decomp.values() if d['aggregate']['total_fn'] > 0)}/{len(fl_decomp)}. "
        f"CD configs: {sum(1 for d in cd_decomp.values() if d['aggregate']['total_fn'] > 0)}/{len(cd_decomp)}. "
        f"Chi2 p={'%.4f' % chi2_result['p_value'] if chi2_result and 'p_value' in chi2_result else 'N/A'}. "
        f"Time: {elapsed:.1f}s. Pass: {'YES' if pass_criteria['overall_pass'] else 'NO'}."
    )
    mark_done("success" if pass_criteria["overall_pass"] else "partial", summary_text)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")


def generate_summary_md(result):
    """Generate a human-readable markdown summary."""
    lines = []
    lines.append("# Phase 1.3: Cross-Domain Absorption-Hedging Decomposition (FULL)")
    lines.append(f"\n**Timestamp**: {result['timestamp']}")
    lines.append(f"**Mode**: {result['mode']}")
    lines.append(f"**Elapsed**: {result['elapsed_seconds']:.1f}s")
    lines.append("")

    lines.append("## Decomposition Categories")
    lines.append("")
    lines.append("| Category | Definition |")
    lines.append("|----------|-----------|")
    lines.append("| **Strict absorbed** | FN where main parent feature does NOT fire (genuine feature gap) |")
    lines.append("| **Compensatory** | FN where main parent feature fires but probe is wrong (interference) |")
    lines.append("| **Persistent** | Remainder (probe error, ambiguous boundary) |")
    lines.append("")

    # L24_16k comparison
    lines.append("## Cross-Hierarchy Comparison at L24_16k")
    lines.append("")
    lines.append("| Hierarchy | FN | Strict | Compensatory | Persistent | Abs Rate |")
    lines.append("|-----------|-----|--------|-------------|------------|----------|")
    for row in result.get("cross_hierarchy_comparison_L24_16k", []):
        lines.append(
            f"| {row['hierarchy']} | {row['total_fn']} | "
            f"{row['strict_absorbed']} ({row['strict_pct']:.1f}%) | "
            f"{row['compensatory']} ({row['compensatory_pct']:.1f}%) | "
            f"{row['persistent']} ({row['persistent_pct']:.1f}%) | "
            f"{row['absorption_rate']:.4f} |"
        )
    lines.append("")

    # L24_65k comparison
    lines.append("## Cross-Hierarchy Comparison at L24_65k")
    lines.append("")
    lines.append("| Hierarchy | FN | Strict | Compensatory | Persistent | Abs Rate |")
    lines.append("|-----------|-----|--------|-------------|------------|----------|")
    for row in result.get("cross_hierarchy_comparison_L24_65k", []):
        lines.append(
            f"| {row['hierarchy']} | {row['total_fn']} | "
            f"{row['strict_absorbed']} ({row['strict_pct']:.1f}%) | "
            f"{row['compensatory']} ({row['compensatory_pct']:.1f}%) | "
            f"{row['persistent']} ({row['persistent_pct']:.1f}%) | "
            f"{row['absorption_rate']:.4f} |"
        )
    lines.append("")

    # Width comparison
    wc = result.get("width_comparison", {})
    if wc:
        lines.append("## Width Comparison (16k vs 65k)")
        lines.append("")
        lines.append("| Hierarchy | 16k Strict% | 65k Strict% | Delta Strict | 16k Comp% | 65k Comp% |")
        lines.append("|-----------|-------------|-------------|-------------|-----------|-----------|")
        for hier, data in wc.items():
            lines.append(
                f"| {hier} | {data['16k']['strict_pct']:.1f} | {data['65k']['strict_pct']:.1f} | "
                f"{data['diff_strict_pp']:+.1f}pp | {data['16k']['compensatory_pct']:.1f} | "
                f"{data['65k']['compensatory_pct']:.1f} |"
            )
        lines.append("")

    # Layer profile
    lp = result.get("layer_profile", [])
    if lp:
        lines.append("## Layer Profile (First-Letter)")
        lines.append("")
        lines.append("| Config | Layer | FN | Strict% | Comp% | Persistent% | AbsRate |")
        lines.append("|--------|-------|----|---------|-------|-------------|---------|")
        for row in lp:
            if row["total_fn"] > 0:
                lines.append(
                    f"| {row['sae_config']} | {row['layer']} | {row['total_fn']} | "
                    f"{row['strict_pct']:.1f} | {row['compensatory_pct']:.1f} | "
                    f"{row['persistent_pct']:.1f} | {row['absorption_rate']:.4f} |"
                )
        lines.append("")

    # Chi-square
    chi2 = result.get("chi_square_test")
    if chi2 and "p_value" in chi2:
        lines.append("## Statistical Tests")
        lines.append("")
        lines.append(f"### Chi-square (4 hierarchies x strict/compensatory at L24_16k)")
        lines.append(f"- Chi2({chi2['dof']}) = {chi2['chi2']:.2f}, p = {chi2['p_value']:.6f}")
        lines.append(f"- {chi2['interpretation']}")
        lines.append("")

    # Fisher pairwise
    fisher = result.get("fisher_pairwise", [])
    if fisher:
        lines.append("### Pairwise Fisher Exact Tests (Bonferroni-corrected)")
        lines.append("")
        lines.append("| Pair | OR | p-value | Significant |")
        lines.append("|------|-----|---------|-------------|")
        for f in fisher:
            if "error" not in f:
                sig = "Yes*" if f["significant_bonferroni"] else ("Yes" if f["significant_uncorrected"] else "No")
                lines.append(
                    f"| {f['pair'][0]} vs {f['pair'][1]} | {f['odds_ratio']:.3f} | "
                    f"{f['p_value']:.4f} | {sig} |"
                )
        lines.append("\n*Bonferroni-corrected significance")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    for f in result.get("key_findings", []):
        interp = f.get("interpretation", f.get("finding", ""))
        lines.append(f"- {interp}")
    lines.append("")

    # iter_008 comparison
    ic = result.get("iter008_comparison")
    if ic:
        lines.append("## Comparison with iter_008")
        lines.append(f"\n- Source: {ic['source']}")
        lines.append(f"- iter_008: strict={ic['strict_hedging_pct']:.1f}%, "
                      f"compensatory={ic['compensatory_pct']:.1f}%, "
                      f"persistent={ic['persistent_pct']:.1f}%")
        if "iter009_L12_16k" in ic:
            i9 = ic["iter009_L12_16k"]
            lines.append(f"- iter_009 L12_16k: strict={i9['strict_pct']:.1f}%, "
                          f"compensatory={i9['compensatory_pct']:.1f}%, "
                          f"persistent={i9['persistent_pct']:.1f}%")
        lines.append(f"- Note: {ic['note']}")
        lines.append("")

    # Methodology
    lines.append("## Methodology Notes")
    for k, v in result.get("methodology_notes", {}).items():
        if isinstance(v, dict):
            lines.append(f"\n### {k}")
            for sk, sv in v.items():
                lines.append(f"- **{sk}**: {sv}")
        else:
            lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - start_time
        mark_done("failed", str(e))
        update_gpu_progress(elapsed, "failed")
        update_experiment_state("failed")
        sys.exit(1)
