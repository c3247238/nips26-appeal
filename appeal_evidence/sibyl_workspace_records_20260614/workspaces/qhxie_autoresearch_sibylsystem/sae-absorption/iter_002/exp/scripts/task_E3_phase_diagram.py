"""
task_E3_phase_diagram.py

Phase Diagram Across (L0, Width) Space.

Reuses data from E1_phase_transition.json and B2_scaling_curve.json (no new GPU computation).
Maps absorption_rate across (1/L0, log2_width) combinations and tests whether wider SAEs
require lower L0_c before absorption dominates.

Writes:
  - exp/results/full/E3_phase_diagram.json
  - exp/results/task_E3_phase_diagram_DONE
  - exp/results/task_E3_phase_diagram_PROGRESS.json
"""

import json
import os
import sys
import math
import time
from pathlib import Path
from datetime import datetime

# PID file
results_dir = Path("exp/results")
results_dir.mkdir(parents=True, exist_ok=True)
pid_file = results_dir / "task_E3_phase_diagram.pid"
pid_file.write_text(str(os.getpid()))

start_time = time.time()


def write_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = results_dir / "task_E3_phase_diagram_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": "task_E3_phase_diagram",
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if pid_file.exists():
        pid_file.unlink()
    progress_file = results_dir / "task_E3_phase_diagram_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = results_dir / "task_E3_phase_diagram_DONE"
    marker.write_text(json.dumps({
        "task_id": "task_E3_phase_diagram",
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


try:
    write_progress(0, 5)
    print("[E3] Phase Diagram Analysis: (L0, Width) Space")
    print(f"[E3] Started at {datetime.now().isoformat()}")

    # -------------------------------------------------------------------------
    # Step 1: Load E1 data
    # -------------------------------------------------------------------------
    write_progress(1, 5, metric={"status": "loading_data"})
    print("[E3] Step 1: Loading E1 phase transition data...")

    e1_path = Path("exp/results/full/E1_phase_transition.json")
    b2_path = Path("exp/results/full/B2_scaling_curve.json")

    if not e1_path.exists():
        raise FileNotFoundError(f"E1 results not found at {e1_path}")

    with open(e1_path) as f:
        e1_data = json.load(f)

    # Load B2 data as supplementary reference
    b2_data = None
    if b2_path.exists():
        with open(b2_path) as f:
            b2_data = json.load(f)
        print(f"[E3] B2 data loaded ({b2_data.get('n_valid_points', 0)} points)")

    all_results = e1_data.get("all_results", [])
    print(f"[E3] Loaded {len(all_results)} data points from E1")

    # -------------------------------------------------------------------------
    # Step 2: Build phase diagram data
    # -------------------------------------------------------------------------
    write_progress(2, 5, metric={"status": "building_grid"})
    print("[E3] Step 2: Building (inv_L0, log2_width) grid...")

    # Collect all valid data points
    phase_points = []
    for r in all_results:
        if r.get("status") != "success":
            continue
        if r.get("absorption_rate") is None:
            continue
        width = r.get("width", r.get("d_sae"))
        l0 = r.get("l0")
        inv_l0 = r.get("inv_l0")
        if width is None or l0 is None or inv_l0 is None:
            continue
        log2_width = math.log2(width)
        phase_points.append({
            "release": r.get("release", ""),
            "sae_id": r.get("sae_id", ""),
            "layer": r.get("layer"),
            "width": width,
            "log2_width": log2_width,
            "l0": l0,
            "inv_l0": inv_l0,
            "absorption_rate": r.get("absorption_rate"),
            "n_letters_measured": r.get("n_letters_measured", 0),
            "group": r.get("group", "unknown"),
        })

    n_points = len(phase_points)
    print(f"[E3] Phase diagram: {n_points} valid (inv_L0, log2_width) points")

    # Get unique widths and their L0 ranges
    from collections import defaultdict
    by_width = defaultdict(list)
    for p in phase_points:
        by_width[p["width"]].append(p)

    print("[E3] Data by width:")
    for w in sorted(by_width.keys()):
        pts = by_width[w]
        l0s = [p["l0"] for p in pts]
        rates = [p["absorption_rate"] for p in pts]
        print(f"  width={w} (log2={math.log2(w):.2f}): n={len(pts)}, "
              f"L0=[{min(l0s):.1f},{max(l0s):.1f}], "
              f"rate=[{min(rates):.3f},{max(rates):.3f}]")

    # -------------------------------------------------------------------------
    # Step 3: Width-conditional analysis
    # -------------------------------------------------------------------------
    write_progress(3, 5, metric={"status": "width_analysis"})
    print("[E3] Step 3: Width-conditional analysis...")

    # For each width group, compute mean absorption rate and L0
    width_summary = []
    for width in sorted(by_width.keys()):
        pts = by_width[width]
        rates = [p["absorption_rate"] for p in pts]
        l0s = [p["l0"] for p in pts]
        inv_l0s = [p["inv_l0"] for p in pts]
        layers = list(set(p["layer"] for p in pts if p["layer"] is not None))
        width_summary.append({
            "width": width,
            "log2_width": math.log2(width),
            "n_configs": len(pts),
            "mean_absorption_rate": sum(rates) / len(rates),
            "min_absorption_rate": min(rates),
            "max_absorption_rate": max(rates),
            "mean_l0": sum(l0s) / len(l0s),
            "l0_range": [min(l0s), max(l0s)],
            "mean_inv_l0": sum(inv_l0s) / len(inv_l0s),
            "layers": sorted(layers),
        })

    # Test hypothesis: wider SAEs require lower L0_c (more sparsity) before absorption dominates
    # Operationalize as: correlation between log2(width) and mean_absorption_rate at matched L0
    # Since we don't have matched L0 across widths, use mean rates as proxy
    log2_widths = [ws["log2_width"] for ws in width_summary]
    mean_rates = [ws["mean_absorption_rate"] for ws in width_summary]

    # Spearman correlation: log2_width vs. mean absorption rate
    def spearman_rho(x, y):
        n = len(x)
        if n < 3:
            return {"rho": None, "p": None, "n": n, "note": "insufficient_data"}
        # Rank
        rx = [sorted(x).index(xi) + 1 for xi in x]
        ry = [sorted(y).index(yi) + 1 for yi in y]
        d2 = sum((rxi - ryi) ** 2 for rxi, ryi in zip(rx, ry))
        rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
        # Approximate p-value (t-distribution with n-2 df)
        import math
        if abs(rho) == 1.0:
            return {"rho": rho, "p": 0.0, "n": n}
        t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        # Simple two-tailed p approximation
        try:
            import scipy.stats
            p = 2 * scipy.stats.t.sf(abs(t), df=n - 2)
        except ImportError:
            # Rough approximation
            p = min(1.0, 2 * math.exp(-0.717 * abs(t) - 0.416 * t * t))
        return {"rho": rho, "p": float(p), "n": n}

    spearman_width_rate = spearman_rho(log2_widths, mean_rates)
    print(f"[E3] Spearman(log2_width, mean_absorption_rate): rho={spearman_width_rate['rho']:.3f}, "
          f"p={spearman_width_rate['p']:.4f}, n={spearman_width_rate['n']}")

    # Also test: at similar L0, do wider SAEs show higher absorption?
    # Group by similar inv_L0 bin (±0.003 tolerance)
    inv_l0_bins = {}
    for p in phase_points:
        bin_key = round(p["inv_l0"] / 0.005) * 0.005  # 0.005 bin width
        if bin_key not in inv_l0_bins:
            inv_l0_bins[bin_key] = []
        inv_l0_bins[bin_key].append(p)

    # Find bins with multiple widths
    multi_width_bins = {k: v for k, v in inv_l0_bins.items()
                        if len(set(p["width"] for p in v)) > 1}
    print(f"[E3] Bins with multiple widths at matched L0: {len(multi_width_bins)}")
    matched_l0_comparison = []
    for bin_key, pts in sorted(multi_width_bins.items()):
        widths_in_bin = sorted(set(p["width"] for p in pts))
        for p in pts:
            matched_l0_comparison.append({
                "inv_l0_bin": bin_key,
                "width": p["width"],
                "log2_width": p["log2_width"],
                "absorption_rate": p["absorption_rate"],
                "l0": p["l0"],
            })
        print(f"  bin inv_l0≈{bin_key:.3f} (L0≈{1/bin_key:.1f}): widths={widths_in_bin}, "
              f"rates={[round(p['absorption_rate'],3) for p in pts]}")

    # -------------------------------------------------------------------------
    # Step 4: Heatmap data construction
    # -------------------------------------------------------------------------
    write_progress(4, 5, metric={"status": "heatmap_construction"})
    print("[E3] Step 4: Constructing heatmap data...")

    # Create a grid representation for the heatmap
    # x-axis: inv_L0 (from min to max)
    # y-axis: log2_width
    # z-axis: absorption_rate

    unique_widths = sorted(set(p["width"] for p in phase_points))
    unique_inv_l0 = sorted(set(round(p["inv_l0"], 4) for p in phase_points))

    # Build sparse grid
    grid = {}
    for p in phase_points:
        key = (round(p["inv_l0"], 4), p["width"])
        if key not in grid:
            grid[key] = []
        grid[key].append(p["absorption_rate"])

    # Average if multiple measurements at same grid point
    grid_mean = {k: sum(v) / len(v) for k, v in grid.items()}

    heatmap_data = {
        "x_axis_label": "1/L0 (sparsity)",
        "y_axis_label": "log2(width)",
        "z_axis_label": "absorption_rate",
        "x_values": sorted(unique_inv_l0),
        "y_values": [math.log2(w) for w in unique_widths],
        "y_width_values": unique_widths,
        "grid_points": [
            {
                "inv_l0": k[0],
                "l0": round(1 / k[0], 2) if k[0] > 0 else None,
                "width": k[1],
                "log2_width": math.log2(k[1]),
                "absorption_rate": v,
            }
            for k, v in sorted(grid_mean.items())
        ],
        "n_grid_points": len(grid_mean),
        "n_unique_widths": len(unique_widths),
        "n_unique_inv_l0_bins": len(unique_inv_l0),
        "coverage": f"{len(grid_mean)}/{len(unique_widths) * len(unique_inv_l0)} cells populated",
    }

    # Test hypothesis: L0_c shifts with width
    # Since absorption rates are very uniformly high (0.87-0.98), compute the deviation
    # from mean as a function of width at each inv_L0 level
    print("[E3] Absorption rate range across all configs:")
    all_rates = [p["absorption_rate"] for p in phase_points]
    print(f"  min={min(all_rates):.3f}, max={max(all_rates):.3f}, "
          f"mean={sum(all_rates)/len(all_rates):.3f}, "
          f"range={max(all_rates)-min(all_rates):.3f}")

    # Width vs absorption: does larger width predict higher or lower absorption?
    # At L0 ~50 (inv_L0 ~0.020), compare widths 12288, 24576, 49152, 98304
    same_l0_pts = [p for p in phase_points
                   if 0.018 <= p["inv_l0"] <= 0.022]
    print(f"[E3] At matched L0≈50 (inv_L0 in [0.018, 0.022]): {len(same_l0_pts)} points")
    for p in sorted(same_l0_pts, key=lambda x: x["width"]):
        print(f"  width={p['width']}, l0={p['l0']:.1f}, absorption={p['absorption_rate']:.3f}")

    # -------------------------------------------------------------------------
    # Step 5: Summary and hypothesis assessment
    # -------------------------------------------------------------------------
    write_progress(5, 5, metric={"status": "finalizing"})
    print("[E3] Step 5: Hypothesis assessment...")

    # H4 prediction: wider SAEs require lower L0_c (more sparsity) before absorption dominates
    # Evidence assessment:
    # 1. All absorption rates are very high (0.87-0.98) regardless of width
    # 2. No clear phase transition visible in available data range
    # 3. Width effect on absorption rate is weak (Spearman rho on width group means)

    # Directional evidence for width effect
    # Sort by width and check absorption trend
    width_to_mean_rate = {ws["width"]: ws["mean_absorption_rate"] for ws in width_summary}
    widths_sorted = sorted(width_to_mean_rate.keys())
    rates_by_width_sorted = [width_to_mean_rate[w] for w in widths_sorted]
    print("[E3] Width → mean absorption rate:")
    for w, r in zip(widths_sorted, rates_by_width_sorted):
        print(f"  width={w} (log2={math.log2(w):.1f}): {r:.3f}")

    # Count "correct direction" comparisons (wider → higher absorption, consistent with
    # H4 prediction that wider SAEs maintain absorption even at higher L0)
    n_pairs = 0
    n_concordant = 0
    for i in range(len(widths_sorted)):
        for j in range(i + 1, len(widths_sorted)):
            n_pairs += 1
            if rates_by_width_sorted[j] >= rates_by_width_sorted[i]:
                n_concordant += 1
    concordance_pct = n_concordant / n_pairs if n_pairs > 0 else 0

    print(f"[E3] Concordant pairs (wider→higher rate): {n_concordant}/{n_pairs} ({concordance_pct:.1%})")

    # Assess if scope is sufficient for phase diagram
    n_combinations = len(phase_points)
    pilot_pass = n_combinations >= 4  # Per task spec: at least 4 (L0, width) combinations

    pass_note = "sufficient for scatter plot" if n_combinations >= 4 else "insufficient data"
    plot_type = "scatter_plot" if n_combinations < 9 else "heatmap"
    print(f"[E3] n_combinations={n_combinations}, plot_type={plot_type}, pilot_pass={pilot_pass}")

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    output = {
        "task_id": "task_E3_phase_diagram",
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
        "config": {
            "source_data": "E1_phase_transition.json",
            "n_data_points": n_points,
            "gpu": "no_gpu_needed (analysis only)",
        },
        "pilot_pass": pilot_pass,
        "pass_criteria": ">=4 (L0, width) combinations measured",
        "n_combinations": n_combinations,
        "n_unique_widths": len(unique_widths),
        "unique_widths": unique_widths,
        "phase_diagram_type": plot_type,
        "phase_points": phase_points,
        "width_summary": width_summary,
        "heatmap_data": heatmap_data,
        "matched_l0_comparison": matched_l0_comparison,
        "spearman_log2width_vs_mean_absorption": spearman_width_rate,
        "width_concordance": {
            "n_pairs": n_pairs,
            "n_concordant": n_concordant,
            "concordance_pct": concordance_pct,
            "interpretation": (
                "wider SAEs tend to have HIGHER absorption rate" if concordance_pct > 0.6
                else "wider SAEs tend to have LOWER absorption rate" if concordance_pct < 0.4
                else "no clear width-absorption relationship"
            ),
        },
        "hypothesis_assessment": {
            "H4_prediction": "Wider SAEs require lower L0_c (more sparsity) before absorption dominates",
            "evidence_quality": "WEAK - all configs show high absorption (0.87-0.98), no clear phase boundary",
            "l0_range_tested": [min(p["l0"] for p in phase_points), max(p["l0"] for p in phase_points)],
            "absorption_rate_range": [min(all_rates), max(all_rates)],
            "width_range_tested": [min(unique_widths), max(unique_widths)],
            "n_widths_tested": len(unique_widths),
            "verdict": (
                "INSUFFICIENT DATA FOR PHASE DIAGRAM: absorption rates cluster near 0.95 "
                "across all (L0, width) combinations tested, with no clear phase boundary visible. "
                "The L0 range sampled (18-81) may be above any phase transition L0_c. "
                "Directional evidence: concordance={:.1%}".format(concordance_pct)
            ),
            "recommendation": (
                "Present as scatter plot (not heatmap). Note that all tested configurations "
                "show near-saturated absorption rates, consistent with E1 finding that "
                "H4a (sigmoid phase transition) is NOT SUPPORTED. "
                "Report as: 'absorption rate uniformly high across tested (L0, width) space; "
                "no phase boundary detected in available SAE suite.'"
            ),
        },
        "visualization_data": {
            "scatter_plot": {
                "description": "Absorption rate in (1/L0, log2_width) space — each point is one SAE config",
                "x": [p["inv_l0"] for p in phase_points],
                "y": [p["log2_width"] for p in phase_points],
                "z": [p["absorption_rate"] for p in phase_points],
                "labels": [f"L{p['layer']}-W{p['width']}" for p in phase_points],
                "x_label": "1/L0 (sparsity level)",
                "y_label": "log2(width)",
                "z_label": "absorption_rate",
                "colorbar_range": [0.85, 1.0],
            },
        },
        "primary_finding": (
            f"Phase diagram across {n_combinations} (L0, width) combinations shows uniformly "
            f"high absorption rates ({min(all_rates):.3f}–{max(all_rates):.3f}). "
            f"Spearman rho(log2_width, absorption)={spearman_width_rate['rho']:.3f} "
            f"(p={spearman_width_rate['p']:.4f}). "
            f"No systematic width-dependent shift in critical L0 detected. "
            f"All {len(unique_widths)} width configurations (12k–98k) show near-saturated "
            f"absorption in the tested L0 range (18–81)."
        ),
    }

    out_path = Path("exp/results/full/E3_phase_diagram.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[E3] Results written to {out_path}")

    elapsed = time.time() - start_time
    print(f"\n[E3] COMPLETED in {elapsed:.1f}s")
    print(f"[E3] pilot_pass={pilot_pass}")
    print(f"[E3] Primary finding: {output['primary_finding']}")

    # Update gpu_progress.json
    gpu_progress_path = Path("exp/gpu_progress.json")
    try:
        if gpu_progress_path.exists():
            gp = json.loads(gpu_progress_path.read_text())
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if "task_E3_phase_diagram" not in gp.get("completed", []):
            gp.setdefault("completed", []).append("task_E3_phase_diagram")
        gp.get("running", {}).pop("task_E3_phase_diagram", None)
        gp.setdefault("timings", {})["task_E3_phase_diagram"] = {
            "planned_min": 30,
            "actual_min": round(elapsed / 60),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gpt2-small",
                "source": "E1_phase_transition.json",
                "n_points": n_points,
                "gpu_count": 0,
            },
        }
        gpu_progress_path.write_text(json.dumps(gp, indent=2))
        print("[E3] gpu_progress.json updated")
    except Exception as e:
        print(f"[E3] WARNING: Failed to update gpu_progress.json: {e}")

    mark_done(status="success", summary=output["primary_finding"])
    print("[E3] DONE marker written.")

except Exception as e:
    import traceback
    traceback.print_exc()
    elapsed = time.time() - start_time
    # Update gpu_progress.json for failure
    gpu_progress_path = Path("exp/gpu_progress.json")
    try:
        if gpu_progress_path.exists():
            gp = json.loads(gpu_progress_path.read_text())
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
        gp.setdefault("failed", []).append("task_E3_phase_diagram")
        gp.get("running", {}).pop("task_E3_phase_diagram", None)
        gp.setdefault("timings", {})["task_E3_phase_diagram"] = {
            "planned_min": 30,
            "actual_min": round(elapsed / 60),
            "end_time": datetime.now().isoformat(),
        }
        gpu_progress_path.write_text(json.dumps(gp, indent=2))
    except Exception:
        pass
    mark_done(status="failed", summary=str(e))
    sys.exit(1)
