#!/usr/bin/env python3
"""E4 Full: Multi-objective Pareto analysis on Pythia-160M SAEBench SAEs.

Analyzes all 7 architecture families from the E2 full Pythia dataset,
stratified by hook point and dictionary width.
Computes normalized Pareto fronts and pairwise stochastic dominance tests.
"""

import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "e4_full"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = WORKSPACE / "exp" / "e4_full"
EXP_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

E2_RESULTS = WORKSPACE / "exp" / "e2_full_pythia" / "absorption_results.json"
CACHE_DIR = Path("/home/qhxie/.cache/huggingface/hub/datasets--adamkarvonen--sae_bench_results_0125/snapshots/df1730fc7bd3ba7dc707b238e5fe098dba674caf")
HF_DATASET = "adamkarvonen/sae_bench_results_0125"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Process tracking
# ---------------------------------------------------------------------------
PID_FILE.write_text(str(os.getpid()))
start_time_iso = datetime.now().isoformat()


def report_progress(epoch, total_epochs, step=0, total_steps=0, message=""):
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "message": message,
                "updated_at": datetime.now().isoformat(),
            }
        )
    )


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": status,
                "summary": summary,
                "final_progress": progress,
                "timestamp": datetime.now().isoformat(),
            }
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_sae_id_from_location(location: str) -> str:
    """Convert e2_full location to SAEBench SAE ID format."""
    parts = location.split("/")
    family_and_release = parts[0]
    hook_point = parts[1]
    trainer = parts[2]

    family, release = family_and_release.split("_pythia-160m-deduped__")
    date_code = release

    sid = (
        f"saebench_pythia-160m-deduped_width-2pow14_date-{date_code}_{family}_"
        f"pythia-160m-deduped__{release}_{hook_point}_{trainer}"
    )
    return sid


def parse_release_dir(sid):
    m = re.search(r"^(.*date-\d+)", sid)
    if m:
        return m.group(1)
    arch = None
    for p in ["Standard", "TopK", "BatchTopK", "JumpRelu", "GatedSAE",
              "MatryoshkaBatchTopK", "PAnneal", "OrtSAE", "Masked"]:
        if p in sid:
            arch = p
            break
    if arch and f"_{arch}_" in sid:
        return sid.split(f"_{arch}_")[0]
    return sid


def list_cached_json_files(eval_type):
    eval_dir = CACHE_DIR / eval_type
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.rglob("*.json"))


def find_eval_file(eval_type, sid):
    release_dir = parse_release_dir(sid)
    eval_dir = CACHE_DIR / eval_type
    if eval_dir.exists():
        candidate = eval_dir / release_dir / f"{sid}_eval_results.json"
        if candidate.exists():
            return candidate
    return None


def extract_sparse_probing(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", {})
    sae = results.get("sae", {})
    return {
        "sparse_probing_acc": sae.get("sae_test_accuracy"),
        "sparse_probing_top1_acc": sae.get("sae_top_1_test_accuracy"),
    }


def extract_ravel(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", {})
    ravel = results.get("ravel_metrics", {})
    return {
        "ravel_cause": ravel.get("cause"),
        "ravel_isolation": ravel.get("isolation"),
    }


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------
def is_pareto_dominated(point, others, objectives):
    """Return True if point is strictly dominated by at least one other point."""
    for other in others:
        if point is other:
            continue
        strictly_better = False
        at_least_as_good = True
        for metric, direction in objectives.items():
            p_val = point.get(metric)
            o_val = other.get(metric)
            if p_val is None or o_val is None:
                at_least_as_good = False
                break
            if direction == "max":
                if o_val < p_val:
                    at_least_as_good = False
                    break
                if o_val > p_val:
                    strictly_better = True
            else:  # min
                if o_val > p_val:
                    at_least_as_good = False
                    break
                if o_val < p_val:
                    strictly_better = True
        if at_least_as_good and strictly_better:
            return True
    return False


def compute_pareto_front(points, objectives):
    """Return list of points that are non-dominated."""
    return [p for p in points if not is_pareto_dominated(p, points, objectives)]


def mann_whitney_test(family_a_points, family_b_points, metric):
    vals_a = [p.get(metric) for p in family_a_points if p.get(metric) is not None]
    vals_b = [p.get(metric) for p in family_b_points if p.get(metric) is not None]
    if len(vals_a) < 2 or len(vals_b) < 2:
        return {"n_a": len(vals_a), "n_b": len(vals_b), "u_stat": None, "p_value": None, "significant": False}
    u_stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {
        "n_a": len(vals_a),
        "n_b": len(vals_b),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    report_progress(1, 6, message="Loading E2 full Pythia absorption data")
    print("Loading E2 full Pythia absorption data...")
    e2_data = json.loads(E2_RESULTS.read_text())
    checkpoints = e2_data["checkpoints"]

    # Enrich with SAEBench sparse probing metrics
    report_progress(2, 6, message="Extracting SAEBench sparse probing metrics")
    print("Extracting SAEBench sparse probing metrics...")

    enriched = []
    for cp in checkpoints:
        sid = build_sae_id_from_location(cp["location"])
        cp_out = {
            "location": cp["location"],
            "family": cp["family"],
            "trainer": cp["trainer"],
            "official_absorption_full": cp.get("official_absorption_full"),
            "official_absorption_fraction": cp.get("official_absorption_fraction"),
            "dead_neuron_fraction": cp.get("dead_neuron_fraction"),
            "l0": cp.get("l0"),
            "explained_variance": cp.get("explained_variance"),
            "sae_id": sid,
        }
        sp_file = find_eval_file("sparse_probing", sid)
        if sp_file:
            cp_out.update(extract_sparse_probing(sp_file))
        else:
            print(f"  Warning: sparse_probing not found for {sid}")
        enriched.append(cp_out)

    # Stratify by hook_point and width (extract from location)
    report_progress(3, 6, message="Stratifying by hook point and width")
    print("Stratifying by hook point and width...")

    from collections import defaultdict
    strata = defaultdict(list)
    for cp in enriched:
        parts = cp["location"].split("/")
        hook_point = parts[1]
        # Width is fixed at 16384 (2pow14) for this dataset
        width = 16384
        stratum_key = (hook_point, width)
        strata[stratum_key].append(cp)

    print(f"Found {len(strata)} stratum/strata: {list(strata.keys())}")

    # Define objectives for multi-objective Pareto analysis
    objectives_4d = {
        "official_absorption_full": "min",
        "explained_variance": "max",
        "dead_neuron_fraction": "min",
        "sparse_probing_acc": "max",
    }

    objectives_2d = {
        "official_absorption_full": "min",
        "explained_variance": "max",
    }

    metrics_to_normalize = ["official_absorption_full", "explained_variance", "dead_neuron_fraction", "sparse_probing_acc"]

    all_strata_results = []

    report_progress(4, 6, message="Computing Pareto fronts and normalizing metrics")
    print("Computing Pareto fronts and normalizing metrics...")

    for stratum_key, stratum_points in strata.items():
        hook_point, width = stratum_key
        print(f"\nStratum: {hook_point}, width={width}, n_checkpoints={len(stratum_points)}")

        # Normalize metrics within stratum
        for metric in metrics_to_normalize:
            vals = [p[metric] for p in stratum_points if p.get(metric) is not None]
            if len(vals) < 2:
                continue
            min_val, max_val = min(vals), max(vals)
            rng = max_val - min_val if max_val != min_val else 1.0
            for p in stratum_points:
                if p.get(metric) is not None:
                    p[f"{metric}_norm"] = (p[metric] - min_val) / rng
                else:
                    p[f"{metric}_norm"] = None

        # Create unified normalized objectives (higher = better)
        for p in stratum_points:
            p["norm_absorption"] = 1.0 - p.get("official_absorption_full_norm", 0.5)
            p["norm_recon"] = p.get("explained_variance_norm", 0.5)
            p["norm_dead"] = 1.0 - p.get("dead_neuron_fraction_norm", 0.5)
            p["norm_sparse"] = p.get("sparse_probing_acc_norm", 0.5)

        # Get families in this stratum
        families_in_stratum = sorted(set(p["family"] for p in stratum_points))
        print(f"  Families: {families_in_stratum}")

        # Compute Pareto fronts per family (2D: absorption vs explained_variance)
        family_fronts = {}
        for fam in families_in_stratum:
            fam_points = [p for p in stratum_points if p["family"] == fam]
            front = compute_pareto_front(fam_points, objectives_2d)
            family_fronts[fam] = front
            print(f"    {fam}: {len(front)}/{len(fam_points)} non-dominated points")

        # Compute 4D Pareto fronts per family
        family_fronts_4d = {}
        for fam in families_in_stratum:
            fam_points = [p for p in stratum_points if p["family"] == fam]
            front = compute_pareto_front(fam_points, objectives_4d)
            family_fronts_4d[fam] = front

        all_strata_results.append({
            "stratum": {"hook_point": hook_point, "width": width},
            "families": families_in_stratum,
            "n_checkpoints": len(stratum_points),
            "checkpoints": stratum_points,
            "pareto_fronts_2d": family_fronts,
            "pareto_fronts_4d": family_fronts_4d,
        })

    # Pairwise stochastic dominance tests across all families
    report_progress(5, 6, message="Running pairwise stochastic dominance tests")
    print("\nRunning pairwise stochastic dominance tests...")

    all_dominance_results = []
    metrics_for_test = ["official_absorption_full", "explained_variance", "dead_neuron_fraction", "sparse_probing_acc"]

    for stratum_result in all_strata_results:
        stratum_points = stratum_result["checkpoints"]
        families = stratum_result["families"]

        for fam_a, fam_b in combinations(families, 2):
            for metric in metrics_for_test:
                a_points = [p for p in stratum_points if p["family"] == fam_a]
                b_points = [p for p in stratum_points if p["family"] == fam_b]
                result = mann_whitney_test(a_points, b_points, metric)
                result["family_a"] = fam_a
                result["family_b"] = fam_b
                result["metric"] = metric
                result["stratum"] = stratum_result["stratum"]
                all_dominance_results.append(result)
                sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
                p_str = f"{result['p_value']:.3f}" if result['p_value'] is not None else "N/A"
                print(f"  {fam_a} vs {fam_b} on {metric}: p={p_str} ({sig_str})")

    total_time = time.time() - datetime.fromisoformat(start_time_iso).timestamp()

    report_progress(6, 6, message="Saving results")

    # Build output JSON
    output = {
        "task_id": TASK_ID,
        "strata": [
            {
                "stratum": sr["stratum"],
                "families": sr["families"],
                "n_checkpoints": sr["n_checkpoints"],
                "pareto_fronts_2d": {
                    fam: [
                        {
                            "location": p["location"],
                            "trainer": p["trainer"],
                            "official_absorption_full": p["official_absorption_full"],
                            "explained_variance": p["explained_variance"],
                            "dead_neuron_fraction": p["dead_neuron_fraction"],
                            "sparse_probing_acc": p.get("sparse_probing_acc"),
                        }
                        for p in front
                    ]
                    for fam, front in sr["pareto_fronts_2d"].items()
                },
                "pareto_fronts_4d": {
                    fam: [
                        {
                            "location": p["location"],
                            "trainer": p["trainer"],
                            "official_absorption_full": p["official_absorption_full"],
                            "explained_variance": p["explained_variance"],
                            "dead_neuron_fraction": p["dead_neuron_fraction"],
                            "sparse_probing_acc": p.get("sparse_probing_acc"),
                        }
                        for p in front
                    ]
                    for fam, front in sr["pareto_fronts_4d"].items()
                },
            }
            for sr in all_strata_results
        ],
        "dominance_tests": all_dominance_results,
        "summary": {
            "n_strata": len(all_strata_results),
            "n_families": len(set(f for sr in all_strata_results for f in sr["families"])),
            "n_checkpoints_total": sum(sr["n_checkpoints"] for sr in all_strata_results),
            "n_dominance_tests": len(all_dominance_results),
            "n_significant_dominance_tests": sum(1 for r in all_dominance_results if r["significant"]),
        },
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }

    out_json = EXP_DIR / "pareto_results.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"\nSaved results to {out_json}")

    # Summary markdown
    summary_md = EXP_DIR / "summary.md"
    md_lines = [
        f"# E4 Full: Multi-Objective Pareto Analysis (Pythia-160M)",
        "",
        f"**Task:** {TASK_ID}  ",
        f"**Total Time:** {total_time/60:.1f} min  ",
        f"**Checkpoints Evaluated:** {output['summary']['n_checkpoints_total']}  ",
        f"**Families:** {output['summary']['n_families']}  ",
        f"**Strata:** {output['summary']['n_strata']}  ",
        "",
    ]

    for sr in all_strata_results:
        st = sr["stratum"]
        md_lines.extend([
            f"## Stratum: {st['hook_point']}, width={st['width']}",
            "",
            f"**Checkpoints:** {sr['n_checkpoints']}  ",
            f"**Families:** {', '.join(sr['families'])}",
            "",
            "### Pareto Fronts (2D: absorption vs explained_variance)",
            "",
        ])
        for fam, front in sr["pareto_fronts_2d"].items():
            md_lines.append(f"#### {fam} ({len(front)} non-dominated)")
            md_lines.append("")
            md_lines.append("| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |")
            md_lines.append("|---------|------------|---------------|--------------|--------------------|")
            for p in front:
                spa = p.get('sparse_probing_acc')
                spa_str = "N/A" if spa is None else f"{spa:.4f}"
                md_lines.append(
                    f"| {p['trainer']} | {p['official_absorption_full']:.4f} | "
                    f"{p['explained_variance']:.4f} | {p['dead_neuron_fraction']:.4f} | "
                    f"{spa_str} |"
                )
            md_lines.append("")

    md_lines.extend([
        "## Pairwise Stochastic Dominance Tests (Mann-Whitney U)",
        "",
        "| Family A | Family B | Metric | U-stat | p-value | Significant |",
        "|----------|----------|--------|--------|---------|-------------|",
    ])
    for r in all_dominance_results:
        p_str = f"{r['p_value']:.3f}" if r['p_value'] is not None else "N/A"
        u_str = f"{r['u_stat']:.1f}" if r['u_stat'] is not None else "N/A"
        sig = "Yes" if r["significant"] else "No"
        md_lines.append(
            f"| {r['family_a']} | {r['family_b']} | {r['metric']} | {u_str} | {p_str} | {sig} |"
        )

    md_lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- Total dominance tests: {output['summary']['n_dominance_tests']}",
        f"- Significant dominance tests (p < 0.05): {output['summary']['n_significant_dominance_tests']}",
    ])

    summary_md.write_text("\n".join(md_lines))
    print(f"Saved summary to {summary_md}")

    # Update gpu_progress
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    if gpu_progress_path.exists():
        gpu_progress = json.loads(gpu_progress_path.read_text())
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID in gpu_progress.get("running", {}):
        del gpu_progress["running"][TASK_ID]
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "EleutherAI/pythia-160m-deduped",
            "n_checkpoints": output["summary"]["n_checkpoints_total"],
            "n_families": output["summary"]["n_families"],
            "n_strata": output["summary"]["n_strata"],
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Update experiment_state
    exp_state_path = WORKSPACE / "exp" / "experiment_state.json"
    if exp_state_path.exists():
        exp_state = json.loads(exp_state_path.read_text())
        if TASK_ID in exp_state.get("tasks", {}):
            exp_state["tasks"][TASK_ID]["status"] = "completed"
            exp_state["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
        exp_state_path.write_text(json.dumps(exp_state, indent=2))

    mark_done(
        status="success",
        summary=(
            f"E4 full completed. {output['summary']['n_checkpoints_total']} checkpoints across "
            f"{output['summary']['n_families']} families in {output['summary']['n_strata']} stratum. "
            f"Pareto fronts and {output['summary']['n_dominance_tests']} dominance tests saved."
        ),
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        raise
