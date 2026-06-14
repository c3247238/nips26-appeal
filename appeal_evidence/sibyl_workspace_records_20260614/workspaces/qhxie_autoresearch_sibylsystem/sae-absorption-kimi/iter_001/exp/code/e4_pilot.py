#!/usr/bin/env python3
"""E4 Pilot: Pareto tradeoff validation on a matched stratum of Pythia-160M SAEs.

Selects 2 families from the E2 full Pythia dataset (all resid_post_layer_8, width 2pow14),
computes Pareto fronts for absorption vs reconstruction/dead-neuron rate/sparse probing F1,
and tests stochastic dominance with Mann-Whitney U tests.
"""

import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "e4_pilot"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = WORKSPACE / "exp" / "e4_pilot"
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
    """Convert e2_full location to SAEBench SAE ID format.

    Actual SAEBench cache filename format:
      saebench_pythia-160m-deduped_width-2pow14_date-0108_BatchTopK_pythia-160m-deduped__0108_resid_post_layer_8_trainer_0

    Example location:
      BatchTopK_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0
    """
    # location format: {family}_pythia-160m-deduped__0108/resid_post_layer_8/{trainer}
    parts = location.split("/")
    family_and_release = parts[0]  # e.g. BatchTopK_pythia-160m-deduped__0108
    hook_point = parts[1]  # e.g. resid_post_layer_8
    trainer = parts[2]  # e.g. trainer_0

    family, release = family_and_release.split("_pythia-160m-deduped__")
    date_code = release  # e.g. 0108

    # Build SAEBench ID matching actual cache filenames
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
    """Return True if point is strictly dominated by at least one other point.

    objectives: dict mapping metric_name -> 'min' or 'max'
    """
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
    report_progress(1, 5, message="Loading E2 full Pythia absorption data")
    print("Loading E2 full Pythia absorption data...")
    e2_data = json.loads(E2_RESULTS.read_text())
    checkpoints = e2_data["checkpoints"]

    # Enrich with SAEBench sparse probing metrics
    report_progress(2, 5, message="Extracting SAEBench sparse probing metrics")
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

    # Select 2 families with the most checkpoints and meaningful absorption variance
    from collections import defaultdict
    fam_counts = defaultdict(int)
    for cp in enriched:
        fam_counts[cp["family"]] += 1

    # Compute absorption variance per family
    fam_stats = defaultdict(lambda: {"abs": []})
    for cp in enriched:
        if cp["official_absorption_full"] is not None:
            fam_stats[cp["family"]]["abs"].append(cp["official_absorption_full"])

    # Pick top 2 families by count, preferring those with variance > 0
    fam_variances = {
        fam: np.var(vals["abs"]) if len(vals["abs"]) > 1 else 0.0
        for fam, vals in fam_stats.items()
    }

    # Sort families: prefer higher variance, then higher count
    sorted_families = sorted(
        fam_counts.keys(),
        key=lambda f: (fam_variances.get(f, 0.0), fam_counts[f]),
        reverse=True,
    )
    selected_families = sorted_families[:2]
    print(f"Selected families for Pareto pilot: {selected_families}")

    selected_points = [p for p in enriched if p["family"] in selected_families]

    report_progress(3, 5, message="Computing Pareto fronts")
    print("Computing Pareto fronts...")

    # Define objectives for multi-objective Pareto analysis
    # We want: minimize absorption, maximize reconstruction (explained_variance),
    # minimize dead_neuron_fraction, maximize sparse_probing_acc
    objectives = {
        "official_absorption_full": "min",
        "explained_variance": "max",
        "dead_neuron_fraction": "min",
        "sparse_probing_acc": "max",
    }

    # Normalize metrics to [0,1] within the selected stratum (all same hook/width)
    metrics_to_normalize = ["official_absorption_full", "explained_variance", "dead_neuron_fraction", "sparse_probing_acc"]
    for metric in metrics_to_normalize:
        vals = [p[metric] for p in selected_points if p.get(metric) is not None]
        if len(vals) < 2:
            continue
        min_val, max_val = min(vals), max(vals)
        rng = max_val - min_val if max_val != min_val else 1.0
        for p in selected_points:
            if p.get(metric) is not None:
                p[f"{metric}_norm"] = (p[metric] - min_val) / rng
            else:
                p[f"{metric}_norm"] = None

    # For Pareto front, we need a unified direction. We'll create a composite where
    # higher is better for all normalized objectives by flipping min objectives.
    for p in selected_points:
        p["norm_absorption"] = 1.0 - p.get("official_absorption_full_norm", 0.5)
        p["norm_recon"] = p.get("explained_variance_norm", 0.5)
        p["norm_dead"] = 1.0 - p.get("dead_neuron_fraction_norm", 0.5)
        p["norm_sparse"] = p.get("sparse_probing_acc_norm", 0.5)

    # Compute Pareto fronts per family using absorption vs reconstruction as primary 2D view
    pareto_objectives_2d = {
        "official_absorption_full": "min",
        "explained_variance": "max",
    }

    family_fronts = {}
    for fam in selected_families:
        fam_points = [p for p in selected_points if p["family"] == fam]
        front = compute_pareto_front(fam_points, pareto_objectives_2d)
        family_fronts[fam] = front
        print(f"  {fam}: {len(front)}/{len(fam_points)} non-dominated points")

    # Stochastic dominance tests (Mann-Whitney U) on all 4 metrics
    report_progress(4, 5, message="Running stochastic dominance tests")
    print("Running stochastic dominance tests...")

    dominance_results = []
    metrics_for_test = ["official_absorption_full", "explained_variance", "dead_neuron_fraction", "sparse_probing_acc"]
    for i, fam_a in enumerate(selected_families):
        for fam_b in selected_families[i+1:]:
            for metric in metrics_for_test:
                a_points = [p for p in selected_points if p["family"] == fam_a]
                b_points = [p for p in selected_points if p["family"] == fam_b]
                result = mann_whitney_test(a_points, b_points, metric)
                result["family_a"] = fam_a
                result["family_b"] = fam_b
                result["metric"] = metric
                dominance_results.append(result)
                sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
                p_str = f"{result['p_value']:.3f}" if result['p_value'] is not None else "N/A"
                print(f"  {fam_a} vs {fam_b} on {metric}: p={p_str} ({sig_str})")

    total_time = time.time() - datetime.fromisoformat(start_time_iso).timestamp()

    report_progress(5, 5, message="Saving results")

    output = {
        "task_id": TASK_ID,
        "selected_families": selected_families,
        "stratum": {
            "model": "EleutherAI/pythia-160m-deduped",
            "hook_point": "resid_post_layer_8",
            "width": 16384,
        },
        "checkpoints": selected_points,
        "pareto_fronts": {
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
            for fam, front in family_fronts.items()
        },
        "dominance_tests": dominance_results,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }

    out_json = EXP_DIR / "pareto_pilot.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_json}")

    # Summary markdown
    summary_md = EXP_DIR / "summary.md"
    md_lines = [
        f"# E4 Pilot: Pareto Tradeoff Validation ({selected_families[0]} vs {selected_families[1]})",
        "",
        f"**Task:** {TASK_ID}  ",
        f"**Stratum:** Pythia-160M, resid_post_layer_8, width=16384  ",
        f"**Total Time:** {total_time/60:.1f} min  ",
        "",
        "## Selected Families",
        "",
        f"- {selected_families[0]}",
        f"- {selected_families[1]}",
        "",
        "## Pareto Fronts (absorption vs explained_variance)",
        "",
    ]
    for fam, front in family_fronts.items():
        md_lines.append(f"### {fam} ({len(front)} non-dominated)")
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
        "## Stochastic Dominance Tests (Mann-Whitney U)",
        "",
        "| Family A | Family B | Metric | U-stat | p-value | Significant |",
        "|----------|----------|--------|--------|---------|-------------|",
    ])
    for r in dominance_results:
        p_str = f"{r['p_value']:.3f}" if r['p_value'] is not None else "N/A"
        u_str = f"{r['u_stat']:.1f}" if r['u_stat'] is not None else "N/A"
        sig = "Yes" if r["significant"] else "No"
        md_lines.append(
            f"| {r['family_a']} | {r['family_b']} | {r['metric']} | {u_str} | {p_str} | {sig} |"
        )

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
        "planned_min": 15,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "selected_families": selected_families,
            "model": "EleutherAI/pythia-160m-deduped",
            "hook_point": "resid_post_layer_8",
            "width": 16384,
            "n_checkpoints": len(selected_points),
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
        summary=f"E4 pilot completed. Families: {selected_families}. Pareto fronts and dominance tests saved.",
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
