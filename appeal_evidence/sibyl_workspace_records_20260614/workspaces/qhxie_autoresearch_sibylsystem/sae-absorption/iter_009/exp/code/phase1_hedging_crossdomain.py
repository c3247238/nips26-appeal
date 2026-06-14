#!/usr/bin/env python3
"""
Phase 1: Cross-Domain Hedging Decomposition (Strict vs Compensatory vs Persistent)
Iteration 9, PILOT mode.

Decomposes false negatives (FNs) from absorption measurement into three categories:
  - strict_absorbed: FN where parent feature does NOT fire (true feature gap -- the SAE
    has no dedicated feature for this concept, so information is genuinely lost)
  - compensatory: FN where parent feature DOES fire but probe is still wrong (the SAE
    has the right feature but other features compensate/interfere in reconstruction,
    distorting the probe direction)
  - persistent: edge cases, likely probe error or ambiguous class boundary

This is a pure analysis task -- no GPU model loading needed. We reuse the already-
computed absorption results from:
  - phase1_absorption_firstletter.json (8 SAE configs, L6-L24 x 16k/65k)
  - phase1_absorption_crossdomain.json (city-continent at L24_16k)

Comparison with iter_008 Phase 0.2 tightened hedging (L0=22->176 multi-scale):
  - iter_008 found: strict 7.9%, compensatory 86.2%, persistent 5.9%
  - This decomposition uses single-L0 main-feature-firing proxy

Dependencies:
  - Phase 1.2a (phase1_absorption_firstletter) -> per-letter FN decomposition
  - Phase 1.3 (phase1_absorption_crossdomain) -> per-class FN decomposition
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
np.random.seed(SEED)

TASK_ID = "phase1_hedging_crossdomain"
MODE = "PILOT"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[{TASK_ID}] Mode: {MODE}")
print(f"[{TASK_ID}] Timestamp: {datetime.now().isoformat()}")

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
                "planned_min": 10,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b", "mode": MODE,
                    "task": "hedging_decomposition_crossdomain",
                },
            }
            path.write_text(json.dumps(data, indent=2))
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
            path.write_text(json.dumps(data, indent=2))
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
            path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"  [WARN] experiment_state update failed: {e}")


# ============================================================================
# Bootstrap CI utility
# ============================================================================
def bootstrap_ci(values, n_bootstrap=2000, ci=0.95):
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


# ============================================================================
# Decomposition function
# ============================================================================
def decompose_fns(per_unit_data, unit_key="class"):
    """
    Decompose false negatives into strict_absorbed / compensatory / persistent.

    For each unit (letter or class):
      - false_negatives: total FN count
      - fn_and_main_absent: FN where main parent feature does NOT fire
        -> strict_absorbed (genuine feature gap)
      - fn_and_main_present: FN where main parent feature DOES fire
        -> compensatory (feature fires but reconstruction distorts probe direction)
      - remainder = false_negatives - fn_and_main_absent - fn_and_main_present
        -> persistent (probe error, ambiguous boundary)

    Returns per-unit decomposition and aggregate.
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
        persistent = fn - strict - compensatory

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
    report_progress(0, 7, "starting")

    # ── Step 1: Load dependency data ──────────────────────────────
    print("\n" + "=" * 70)
    print("Step 1: Loading dependency data")
    print("=" * 70)
    report_progress(1, 7, "loading_data")

    # Load first-letter absorption
    fl_path = PILOT_DIR / "phase1_absorption_firstletter.json"
    if not fl_path.exists():
        raise FileNotFoundError(f"Missing dependency: {fl_path}")
    fl_data = json.loads(fl_path.read_text())
    print(f"  [OK] First-letter absorption: {len(fl_data['absorption_results'])} SAE configs")

    # Load cross-domain absorption
    cd_path = PILOT_DIR / "phase1_absorption_crossdomain.json"
    if not cd_path.exists():
        raise FileNotFoundError(f"Missing dependency: {cd_path}")
    cd_data = json.loads(cd_path.read_text())
    print(f"  [OK] Cross-domain absorption: {len(cd_data['absorption_results'])} SAE configs")
    print(f"       Hierarchy: {cd_data['probe_info']['hierarchy']}")
    print(f"       Probe F1: {cd_data['probe_info']['f1']:.4f}")

    # Load iter_008 tightened hedging for comparison (if available)
    iter008_hedging = None
    iter008_hedging_path = Path(
        "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/"
        "iter_008/exp/results/pilots/phase0/tightened_hedging.json"
    )
    if iter008_hedging_path.exists():
        iter008_hedging = json.loads(iter008_hedging_path.read_text())
        print(f"  [OK] iter_008 tightened hedging loaded for comparison")
    else:
        print(f"  [WARN] iter_008 tightened hedging not found, skipping comparison")

    # ── Step 2: Decompose first-letter FNs across all SAE configs ──
    print("\n" + "=" * 70)
    print("Step 2: First-letter hedging decomposition (all SAE configs)")
    print("=" * 70)
    report_progress(2, 7, "firstletter_decomposition")

    fl_decomp = {}
    for sae_key, sae_data in fl_data["absorption_results"].items():
        if "error" in sae_data or "per_letter" not in sae_data:
            print(f"  [SKIP] {sae_key}: error or missing per_letter data")
            continue

        agg, per_letter = decompose_fns(sae_data["per_letter"], unit_key="letter")
        fl_decomp[sae_key] = {
            "sae_config": sae_key,
            "layer": sae_data["sae_config"]["layer"],
            "width": sae_data["sae_config"]["width"],
            "aggregate": agg,
            "per_letter": per_letter,
            "overall_absorption_rate": sae_data["absorption_rate"],
            "overall_strict_rate": sae_data["strict_absorption_rate"],
            "total_probe_correct": sae_data.get("total_probe_correct", 0),
        }
        if agg["total_fn"] > 0:
            print(f"  {sae_key}: FN={agg['total_fn']}  "
                  f"strict={agg['strict_absorbed']}({agg['strict_pct']:.1f}%)  "
                  f"compensatory={agg['compensatory']}({agg['compensatory_pct']:.1f}%)  "
                  f"persistent={agg['persistent']}({agg['persistent_pct']:.1f}%)")
        else:
            print(f"  {sae_key}: No FNs (absorption_rate=0)")

    # ── Step 3: Decompose cross-domain (city-continent) FNs ───────
    print("\n" + "=" * 70)
    print("Step 3: Cross-domain (city-continent) hedging decomposition")
    print("=" * 70)
    report_progress(3, 7, "crossdomain_decomposition")

    cd_decomp = {}
    cd_hierarchy = cd_data["probe_info"]["hierarchy"]

    for sae_key, sae_data in cd_data["absorption_results"].items():
        if "error" in sae_data or "per_class" not in sae_data:
            print(f"  [SKIP] {sae_key}: error or missing per_class data")
            continue

        agg, per_class = decompose_fns(sae_data["per_class"], unit_key="class")
        cd_decomp[sae_key] = {
            "sae_config": sae_key,
            "hierarchy": cd_hierarchy,
            "aggregate": agg,
            "per_class": per_class,
            "overall_absorption_rate": sae_data["absorption_rate"],
            "overall_strict_rate": sae_data["strict_absorption_rate"],
            "total_probe_correct": sae_data.get("total_probe_correct_raw", 0),
        }
        print(f"  {cd_hierarchy} @ {sae_key}: FN={agg['total_fn']}  "
              f"strict={agg['strict_absorbed']}({agg['strict_pct']:.1f}%)  "
              f"compensatory={agg['compensatory']}({agg['compensatory_pct']:.1f}%)  "
              f"persistent={agg['persistent']}({agg['persistent_pct']:.1f}%)")

        # Per-class detail
        for cls_name, cls_data in per_class.items():
            if cls_data["total_fn"] > 0:
                print(f"    {cls_name:20s}: FN={cls_data['total_fn']}  "
                      f"strict={cls_data['strict_absorbed']}  "
                      f"compensatory={cls_data['compensatory']}  "
                      f"persistent={cls_data['persistent']}")

    # ── Step 4: Cross-hierarchy comparison at L24_16k ─────────────
    print("\n" + "=" * 70)
    print("Step 4: Cross-hierarchy comparison at L24_16k")
    print("=" * 70)
    report_progress(4, 7, "cross_hierarchy_comparison")

    comparison_key = "L24_16k"
    comparison = []

    # First-letter at L24_16k
    if comparison_key in fl_decomp:
        fl_agg = fl_decomp[comparison_key]["aggregate"]
        comparison.append({
            "hierarchy": "first-letter",
            "sae_config": comparison_key,
            "total_fn": fl_agg["total_fn"],
            "strict_absorbed": fl_agg["strict_absorbed"],
            "compensatory": fl_agg["compensatory"],
            "persistent": fl_agg["persistent"],
            "strict_pct": fl_agg["strict_pct"],
            "compensatory_pct": fl_agg["compensatory_pct"],
            "persistent_pct": fl_agg["persistent_pct"],
            "absorption_rate": fl_decomp[comparison_key]["overall_absorption_rate"],
        })

    # City-continent at L24_16k
    if comparison_key in cd_decomp:
        cd_agg = cd_decomp[comparison_key]["aggregate"]
        comparison.append({
            "hierarchy": cd_hierarchy,
            "sae_config": comparison_key,
            "total_fn": cd_agg["total_fn"],
            "strict_absorbed": cd_agg["strict_absorbed"],
            "compensatory": cd_agg["compensatory"],
            "persistent": cd_agg["persistent"],
            "strict_pct": cd_agg["strict_pct"],
            "compensatory_pct": cd_agg["compensatory_pct"],
            "persistent_pct": cd_agg["persistent_pct"],
            "absorption_rate": cd_decomp[comparison_key]["overall_absorption_rate"],
        })

    print(f"\n  {'Hierarchy':<20} {'FN':>4} {'Strict':>7} {'Comp.':>7} {'Persist':>7} "
          f"{'S%':>6} {'C%':>6} {'P%':>6} {'AbsRate':>8}")
    print("  " + "-" * 85)
    for row in comparison:
        print(f"  {row['hierarchy']:<20} {row['total_fn']:>4} "
              f"{row['strict_absorbed']:>7} {row['compensatory']:>7} {row['persistent']:>7} "
              f"{row['strict_pct']:>6.1f} {row['compensatory_pct']:>6.1f} {row['persistent_pct']:>6.1f} "
              f"{row['absorption_rate']:>8.4f}")

    # ── Step 5: Comparison with iter_008 tightened hedging ────────
    print("\n" + "=" * 70)
    print("Step 5: Comparison with iter_008 tightened hedging")
    print("=" * 70)
    report_progress(5, 7, "iter008_comparison")

    iter008_comparison = None
    if iter008_hedging is not None:
        h = iter008_hedging["hedging_decomposition_l0_22_to_176"]
        strict_clf = h["strict_classification"]
        # Note: iter_008 uses "compensatory_resolution" not "compensatory"
        compensatory_count = strict_clf.get("compensatory_resolution",
                                            strict_clf.get("compensatory", 0))
        iter008_comparison = {
            "source": "iter_008 Phase 0.2 tightened hedging (L0=22->176, L12_16k)",
            "total_fn": h["total_fn"],
            "strict_hedging": strict_clf["strict_hedging"],
            "strict_hedging_pct": strict_clf["strict_hedging_pct"],
            "compensatory": compensatory_count,
            "compensatory_pct": strict_clf["compensatory_pct"],
            "persistent": strict_clf["persistent"],
            "persistent_pct": strict_clf["persistent_pct"],
            "note": (
                "iter_008 used multi-L0 analysis (L0=22->176) for classification. "
                "iter_009 uses single-L0 main-feature-firing proxy at each config's "
                "default L0. The 'strict_absorbed' (main absent) in iter_009 roughly "
                "corresponds to 'strict_hedging' (FN resolves at higher L0 + feature fires) "
                "in iter_008, while 'compensatory' (main present but probe wrong) maps to "
                "the compensatory category."
            ),
        }

        print(f"  iter_008 (L0=22->176, L12_16k, first-letter only):")
        print(f"    Total FN: {h['total_fn']}")
        print(f"    Strict hedging: {strict_clf['strict_hedging']} ({strict_clf['strict_hedging_pct']:.1f}%)")
        print(f"    Compensatory:   {compensatory_count} ({strict_clf['compensatory_pct']:.1f}%)")
        print(f"    Persistent:     {strict_clf['persistent']} ({strict_clf['persistent_pct']:.1f}%)")

        # Now compare with iter_009 first-letter at L12_16k (closest match)
        if "L12_16k" in fl_decomp:
            fl12_agg = fl_decomp["L12_16k"]["aggregate"]
            print(f"\n  iter_009 first-letter at L12_16k (single-L0, main-feature proxy):")
            print(f"    Total FN: {fl12_agg['total_fn']}")
            print(f"    Strict absorbed (main absent): {fl12_agg['strict_absorbed']} ({fl12_agg['strict_pct']:.1f}%)")
            print(f"    Compensatory (main present):   {fl12_agg['compensatory']} ({fl12_agg['compensatory_pct']:.1f}%)")
            print(f"    Persistent:                    {fl12_agg['persistent']} ({fl12_agg['persistent_pct']:.1f}%)")
        else:
            print(f"  iter_009 L12_16k not available for direct comparison")
    else:
        print(f"  [SKIP] No iter_008 tightened hedging data available")

    # ── Step 6: Statistical analysis ──────────────────────────────
    print("\n" + "=" * 70)
    print("Step 6: Statistical analysis")
    print("=" * 70)
    report_progress(6, 7, "statistical_analysis")

    stat_results = {}

    # 6a: Bootstrap CI on decomposition fractions
    print("\n  --- Bootstrap CI on decomposition fractions ---")
    for label, decomp_dict, per_unit_key in [
        ("first-letter", fl_decomp, "per_letter"),
        (cd_hierarchy, cd_decomp, "per_class"),
    ]:
        for sae_key, sae_decomp in decomp_dict.items():
            per_unit = sae_decomp[per_unit_key]
            # Collect unit-level fractions for bootstrap
            strict_fracs = []
            comp_fracs = []
            for name, udata in per_unit.items():
                if udata["total_fn"] > 0:
                    strict_fracs.append(udata["strict_pct"] / 100.0)
                    comp_fracs.append(udata["compensatory_pct"] / 100.0)

            if len(strict_fracs) >= 3:
                strict_ci = bootstrap_ci(strict_fracs)
                comp_ci = bootstrap_ci(comp_fracs)
                stat_key = f"{label}_{sae_key}"
                stat_results[stat_key] = {
                    "strict_absorbed_ci": strict_ci,
                    "compensatory_ci": comp_ci,
                    "n_units_with_fn": len(strict_fracs),
                }
                print(f"  {label} @ {sae_key} (n_units={len(strict_fracs)}):")
                print(f"    Strict fraction CI: [{strict_ci['ci_lower']:.3f}, {strict_ci['ci_upper']:.3f}]")
                print(f"    Compensatory CI:    [{comp_ci['ci_lower']:.3f}, {comp_ci['ci_upper']:.3f}]")

    # 6b: Fisher's exact test on compensatory vs strict between hierarchies at L24_16k
    print("\n  --- Fisher's exact test: first-letter vs city-continent (L24_16k) ---")
    fisher_result = None
    if comparison_key in fl_decomp and comparison_key in cd_decomp:
        fl_agg = fl_decomp[comparison_key]["aggregate"]
        cd_agg = cd_decomp[comparison_key]["aggregate"]

        # 2x2 table: [strict, compensatory] x [first-letter, city-continent]
        table = np.array([
            [max(fl_agg["strict_absorbed"], 0), max(fl_agg["compensatory"], 0)],
            [max(cd_agg["strict_absorbed"], 0), max(cd_agg["compensatory"], 0)],
        ])

        # Only run Fisher if we have enough counts
        if table.sum() >= 5:
            odds_ratio, p_value = stats.fisher_exact(table)
            fisher_result = {
                "table": table.tolist(),
                "rows": ["first-letter", cd_hierarchy],
                "cols": ["strict_absorbed", "compensatory"],
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
                "significant_005": p_value < 0.05,
                "interpretation": (
                    f"OR={odds_ratio:.2f}, p={p_value:.4f}. "
                    f"{'Significant' if p_value < 0.05 else 'Not significant'} difference "
                    f"in strict/compensatory ratio between hierarchies."
                ),
            }
            print(f"    Table: {table.tolist()}")
            print(f"    OR={odds_ratio:.3f}, p={p_value:.4f}")
            print(f"    Significant (p<0.05): {p_value < 0.05}")
        else:
            fisher_result = {
                "error": "Insufficient counts for Fisher's exact test",
                "table": table.tolist(),
                "total_count": int(table.sum()),
            }
            print(f"    Insufficient counts ({table.sum()}) for Fisher's exact test")
    else:
        print(f"    Cannot compare: missing {comparison_key} in one or both hierarchies")

    # 6c: Per-class absorption rate vs class size (Spearman)
    print("\n  --- Spearman: absorption rate vs class size ---")
    spearman_result = None
    if comparison_key in cd_decomp:
        per_class = cd_decomp[comparison_key]["per_class"]
        abs_rates = []
        class_sizes = []
        class_names = []
        for cls_name, cls_data in per_class.items():
            pcr = cls_data.get("probe_correct_raw", 0)
            if pcr >= 3:
                abs_rates.append(cls_data.get("absorption_rate", 0))
                class_sizes.append(pcr)
                class_names.append(cls_name)

        if len(abs_rates) >= 3:
            rho, p_val = stats.spearmanr(abs_rates, class_sizes)
            spearman_result = {
                "rho": float(rho),
                "p_value": float(p_val),
                "n_classes": len(abs_rates),
                "classes": class_names,
                "absorption_rates": abs_rates,
                "class_sizes": class_sizes,
                "significant_005": p_val < 0.05,
            }
            print(f"    rho={rho:.4f}, p={p_val:.4f}, n={len(abs_rates)} classes")
        else:
            spearman_result = {"error": "Fewer than 3 classes with >= 3 probe-correct"}
            print(f"    Insufficient classes for Spearman test")

    # 6d: Layer-dependence of decomposition (first-letter across L6-L24)
    print("\n  --- Layer dependence of decomposition (first-letter) ---")
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
                "strict_pct": agg["strict_pct"],
                "compensatory_pct": agg["compensatory_pct"],
                "persistent_pct": agg["persistent_pct"],
                "absorption_rate": fd["overall_absorption_rate"],
            })
            if agg["total_fn"] > 0:
                print(f"    {sae_key}: FN={agg['total_fn']}  "
                      f"S={agg['strict_pct']:.1f}%  C={agg['compensatory_pct']:.1f}%  P={agg['persistent_pct']:.1f}%")

    # ── Step 7: Compile final results ─────────────────────────────
    print("\n" + "=" * 70)
    print("Step 7: Compiling results")
    print("=" * 70)
    report_progress(7, 7, "compiling")

    elapsed = time.time() - start_time

    # Key findings
    key_findings = []

    # Finding 1: Dominant category at L24_16k for each hierarchy
    for row in comparison:
        dominant = max(
            [("strict_absorbed", row["strict_pct"]),
             ("compensatory", row["compensatory_pct"]),
             ("persistent", row["persistent_pct"])],
            key=lambda x: x[1]
        )
        key_findings.append({
            "finding": f"Dominant FN category in {row['hierarchy']} at {comparison_key}",
            "category": dominant[0],
            "pct": dominant[1],
            "interpretation": (
                f"{row['hierarchy']}: most FNs are {dominant[0]} "
                f"({dominant[1]:.1f}%, N={row['total_fn']})"
            ),
        })

    # Finding 2: Cross-hierarchy difference
    if len(comparison) == 2:
        fl_row = comparison[0]
        cd_row = comparison[1]
        diff_comp = cd_row["compensatory_pct"] - fl_row["compensatory_pct"]
        diff_strict = cd_row["strict_pct"] - fl_row["strict_pct"]
        key_findings.append({
            "finding": f"Decomposition difference: {cd_hierarchy} vs first-letter at {comparison_key}",
            "compensatory_diff_pp": round(diff_comp, 1),
            "strict_diff_pp": round(diff_strict, 1),
            "interpretation": (
                f"At {comparison_key}: {cd_hierarchy} has "
                f"{diff_comp:+.1f}pp compensatory, {diff_strict:+.1f}pp strict "
                f"vs first-letter"
            ),
        })

    # Finding 3: Layer dependence pattern
    if layer_profile:
        l24_configs = [lp for lp in layer_profile if lp["layer"] == 24 and lp["total_fn"] > 0]
        early_configs = [lp for lp in layer_profile if lp["layer"] <= 12 and lp["total_fn"] > 0]
        if l24_configs and early_configs:
            avg_comp_l24 = np.mean([lp["compensatory_pct"] for lp in l24_configs])
            avg_comp_early = np.mean([lp["compensatory_pct"] for lp in early_configs])
            key_findings.append({
                "finding": "Layer-dependent decomposition shift",
                "avg_compensatory_l24": round(avg_comp_l24, 1),
                "avg_compensatory_early": round(avg_comp_early, 1),
                "interpretation": (
                    f"Compensatory fraction: L24 avg={avg_comp_l24:.1f}%, "
                    f"early layers (L6/L12) avg={avg_comp_early:.1f}%"
                ),
            })

    # Pass criteria
    has_fl_decomp = any(d["aggregate"]["total_fn"] > 0 for d in fl_decomp.values())
    has_cd_decomp = any(d["aggregate"]["total_fn"] > 0 for d in cd_decomp.values())
    pass_criteria = {
        "decomposition_computed": has_fl_decomp and has_cd_decomp,
        "absorbed_vs_hedged_fractions_reported": len(comparison) >= 2,
        "overall_pass": has_fl_decomp and has_cd_decomp,
        "pilot_pass_criteria": "Decomposition computed. Absorbed vs hedged fractions reported.",
    }

    # Full result
    result = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",

        "firstletter_decomposition": fl_decomp,
        "crossdomain_decomposition": cd_decomp,

        "cross_hierarchy_comparison": comparison,
        "layer_profile": layer_profile,

        "iter008_comparison": iter008_comparison,
        "fisher_test": fisher_result,
        "spearman_class_size": spearman_result,
        "bootstrap_ci": stat_results,

        "key_findings": key_findings,
        "pass_criteria": pass_criteria,

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
            "probe_quality_caveat": (
                f"Cross-domain probe F1={cd_data['probe_info']['f1']:.4f} "
                f"(below strict 0.90 gate). Some FNs may reflect probe error, "
                f"not SAE-induced information loss."
            ),
        },

        "elapsed_seconds": elapsed,
        "elapsed_minutes": round(elapsed / 60, 2),
    }

    # Save
    out_path = PILOT_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Saved: {out_path}")

    # Also save summary markdown
    summary_md = generate_summary_md(result, comparison, key_findings, layer_profile)
    md_path = PILOT_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    print(f"  Summary: {md_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    pilot_pass = pass_criteria["overall_pass"]
    print(f"  Pilot pass: {'PASS' if pilot_pass else 'FAIL'}")
    print(f"  First-letter configs with FN: {sum(1 for d in fl_decomp.values() if d['aggregate']['total_fn'] > 0)}/{len(fl_decomp)}")
    print(f"  Cross-domain configs with FN: {sum(1 for d in cd_decomp.values() if d['aggregate']['total_fn'] > 0)}/{len(cd_decomp)}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.2f}min)")

    print(f"\n  Key findings:")
    for f in key_findings:
        print(f"    - {f['interpretation']}")

    # Mark done
    summary_text = (
        f"Hedging decomposition PILOT. "
        f"First-letter: {sum(1 for d in fl_decomp.values() if d['aggregate']['total_fn'] > 0)} configs with FN. "
        f"Cross-domain ({cd_hierarchy}): {sum(1 for d in cd_decomp.values() if d['aggregate']['total_fn'] > 0)} configs. "
        f"Time: {elapsed:.1f}s. "
        f"Pass: {'YES' if pilot_pass else 'NO'}."
    )
    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")


def generate_summary_md(result, comparison, key_findings, layer_profile):
    """Generate a human-readable markdown summary."""
    lines = []
    lines.append("# Phase 1: Cross-Domain Hedging Decomposition (Pilot)")
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

    lines.append("## Cross-Hierarchy Comparison at L24_16k")
    lines.append("")
    lines.append("| Hierarchy | FN | Strict | Compensatory | Persistent | Abs Rate |")
    lines.append("|-----------|-----|--------|-------------|------------|----------|")
    for row in comparison:
        lines.append(
            f"| {row['hierarchy']} | {row['total_fn']} | "
            f"{row['strict_absorbed']} ({row['strict_pct']:.1f}%) | "
            f"{row['compensatory']} ({row['compensatory_pct']:.1f}%) | "
            f"{row['persistent']} ({row['persistent_pct']:.1f}%) | "
            f"{row['absorption_rate']:.4f} |"
        )
    lines.append("")

    if layer_profile:
        lines.append("## Layer Profile (First-Letter)")
        lines.append("")
        lines.append("| Config | FN | Strict% | Compensatory% | Persistent% | AbsRate |")
        lines.append("|--------|-----|---------|--------------|-------------|---------|")
        for lp in layer_profile:
            if lp["total_fn"] > 0:
                lines.append(
                    f"| {lp['sae_config']} | {lp['total_fn']} | "
                    f"{lp['strict_pct']:.1f} | {lp['compensatory_pct']:.1f} | "
                    f"{lp['persistent_pct']:.1f} | {lp['absorption_rate']:.4f} |"
                )
        lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    for f in key_findings:
        lines.append(f"- {f['interpretation']}")
    lines.append("")

    if result.get("fisher_test") and "error" not in result["fisher_test"]:
        ft = result["fisher_test"]
        lines.append("## Fisher's Exact Test (L24_16k)")
        lines.append(f"- OR = {ft['odds_ratio']:.3f}, p = {ft['p_value']:.4f}")
        lines.append(f"- {ft['interpretation']}")
        lines.append("")

    if result.get("iter008_comparison"):
        ic = result["iter008_comparison"]
        lines.append("## Comparison with iter_008")
        lines.append(f"- Source: {ic['source']}")
        lines.append(f"- iter_008 (multi-L0): strict={ic['strict_hedging_pct']:.1f}%, "
                      f"compensatory={ic['compensatory_pct']:.1f}%, "
                      f"persistent={ic['persistent_pct']:.1f}%")
        lines.append(f"- Note: {ic['note']}")
        lines.append("")

    lines.append("## Methodology Notes")
    for k, v in result["methodology_notes"].items():
        lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done("failed", str(e))
        update_gpu_progress(0, "failed")
        sys.exit(1)
