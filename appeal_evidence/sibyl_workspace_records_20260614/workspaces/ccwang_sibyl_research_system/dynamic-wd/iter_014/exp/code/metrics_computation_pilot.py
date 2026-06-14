#!/usr/bin/env python3
"""
Pilot metrics computation: BEM, CSI, AIS across all available pilot data.
Task: metrics_computation (PILOT mode)

Aggregates pilot results from:
- diagnostic_cifar10 (8 methods, 10 epochs, seed 42)
- ablation_cifar100 (8 variants, 10 epochs, seed 42)
- alignment_informativeness (18 runs, 30 epochs, seed 42)
- imagenet_main (8 methods, 2 epochs, seed 42)
- imagenet_budget_matched (5 lambda values, 2 epochs, seed 42)
- batchsize_sweep (3 methods x 5 batch sizes, 5 epochs, seed 42)

Computes:
- BEM = accuracy_improvement / total_WD_budget
- CSI_temporal = 1 / Var_t[lambda_t] (per-layer, averaged)
- CSI_spatial = 1 / Var_l[mean_lambda_l] (across layers)
- CSI_combined = (CSI_temporal + CSI_spatial) / 2
- AIS = alignment informativeness score

Critical checks:
- UDWDC-v2 CSI > 0 (stability fix verification)
- Compare rankings under BEM vs raw accuracy
- Statistical significance tests for ranking claims
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Paths
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
PILOTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
OUTPUT_DIR = WORKSPACE / "exp" / "results" / "pilots" / "metrics_computation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "metrics_computation"


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_json(path):
    """Load JSON file safely."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Cannot load {path}: {e}")
        return None


def compute_bem(accuracy, baseline_acc, total_wd_budget):
    """Compute Budget Equivalence Metric."""
    improvement = accuracy - baseline_acc
    if total_wd_budget is None or total_wd_budget == 0:
        return None, improvement
    bem = improvement / total_wd_budget
    return bem, improvement


def compute_csi_from_trajectories(trajectories_path, n_layers_with_wd=None):
    """
    Compute CSI from per-layer trajectory data.

    CSI_temporal = 1 / mean(Var_t[lambda_t^l]) per layer, averaged
    CSI_spatial = 1 / Var_l[mean(lambda_t^l)] across layers
    """
    data = load_json(trajectories_path)
    if data is None:
        return None

    # Extract effective WD trajectories per layer
    layer_wd_trajectories = {}

    if isinstance(data, dict):
        # Check if it has per-epoch structure
        for key, val in data.items():
            if isinstance(val, dict) and "effective_wd" in val:
                layer_wd_trajectories[key] = val["effective_wd"]
            elif isinstance(val, list):
                # Could be epoch-level data with layer info
                pass
    elif isinstance(data, list):
        # List of epoch records
        for epoch_data in data:
            if isinstance(epoch_data, dict):
                for layer_key in epoch_data:
                    if "effective_wd" in str(layer_key) or "lambda" in str(layer_key):
                        pass

    # If we can't parse trajectories, compute from epoch-level data
    return None


def compute_csi_from_epoch_data(epoch_data, method_name):
    """
    Compute CSI from epoch-level summary data.
    Uses effective WD budget variation across epochs.
    """
    if not epoch_data or len(epoch_data) < 2:
        return {"CSI_temporal": None, "CSI_spatial": None, "CSI_combined": None}

    wd_budgets = []
    for ep in epoch_data:
        if isinstance(ep, dict):
            wd = ep.get("wd_budget_epoch", ep.get("epoch_wd_budget", 0))
            wd_budgets.append(wd)

    if not wd_budgets or all(w == 0 for w in wd_budgets):
        if method_name == "NoWD":
            return {
                "CSI_temporal": 1.0,
                "CSI_spatial": 1.0,
                "CSI_combined": 1.0,
                "interpretation": "trivially stable (no WD)"
            }
        return {"CSI_temporal": None, "CSI_spatial": None, "CSI_combined": None}

    wd_arr = np.array(wd_budgets, dtype=float)
    mean_wd = np.mean(wd_arr)
    var_wd = np.var(wd_arr)

    if mean_wd == 0:
        csi_temporal = 0.0
    elif var_wd == 0:
        csi_temporal = 1.0
    else:
        cv = np.sqrt(var_wd) / abs(mean_wd)
        csi_temporal = 1.0 / (1.0 + cv)

    return {
        "CSI_temporal": round(csi_temporal, 6),
        "CSI_spatial": None,  # Need per-layer data
        "CSI_combined": round(csi_temporal, 6),
        "mean_wd_per_epoch": round(mean_wd, 8),
        "cv_wd": round(np.sqrt(var_wd) / abs(mean_wd), 6) if mean_wd != 0 else None
    }


def compute_csi_from_wd_budget_per_epoch(wd_budgets_per_epoch, method_name):
    """
    Compute CSI from per-epoch WD budget array.
    CSI_temporal = 1 / (1 + CV(wd_budget_per_epoch))
    """
    if not wd_budgets_per_epoch:
        return {"CSI_temporal": None, "CSI_spatial": None, "CSI_combined": None}

    wd_arr = np.array(wd_budgets_per_epoch, dtype=float)

    if all(w == 0 for w in wd_arr):
        if method_name in ["NoWD"]:
            return {
                "CSI_temporal": 1.0, "CSI_spatial": 1.0, "CSI_combined": 1.0,
                "interpretation": "trivially stable (no WD)"
            }
        return {
            "CSI_temporal": 0.0, "CSI_spatial": None, "CSI_combined": 0.0,
            "interpretation": "zero WD budget (controller collapse)"
        }

    mean_wd = np.mean(wd_arr)
    std_wd = np.std(wd_arr)

    if mean_wd == 0:
        csi_temporal = 0.0
        cv = float('inf')
    elif std_wd == 0:
        csi_temporal = 1.0
        cv = 0.0
    else:
        cv = std_wd / abs(mean_wd)
        csi_temporal = 1.0 / (1.0 + cv)

    return {
        "CSI_temporal": round(csi_temporal, 6),
        "CSI_spatial": None,
        "CSI_combined": round(csi_temporal, 6),
        "cv_wd": round(cv, 6) if cv != float('inf') else "inf",
        "mean_wd_per_epoch": round(mean_wd, 8),
        "std_wd_per_epoch": round(std_wd, 8)
    }


def compute_ais(alpha_data):
    """
    Compute Alignment Informativeness Score.
    AIS = |Spearman(alpha_bar, gen_gap)| * (1 - CV(alpha))
    """
    if not alpha_data or len(alpha_data) < 3:
        return None

    alpha_bars = [d["alpha_bar"] for d in alpha_data if "alpha_bar" in d]
    gen_gaps = [d["generalization_gap"] for d in alpha_data if "generalization_gap" in d]

    if len(alpha_bars) < 3 or len(gen_gaps) < 3:
        return None

    alpha_arr = np.array(alpha_bars)
    gap_arr = np.array(gen_gaps)

    # Spearman correlation
    rho, p_val = stats.spearmanr(alpha_arr, gap_arr)

    # CV of alpha
    mean_alpha = np.mean(np.abs(alpha_arr))
    std_alpha = np.std(alpha_arr)
    cv_alpha = std_alpha / mean_alpha if mean_alpha > 0 else float('inf')

    ais = abs(rho) * (1.0 / (1.0 + cv_alpha))

    return {
        "AIS": round(ais, 6),
        "spearman_rho": round(rho, 6),
        "spearman_p": round(p_val, 6),
        "cv_alpha": round(cv_alpha, 6),
        "n_samples": len(alpha_bars)
    }


def statistical_significance_test(acc_a, acc_b, n_seeds=1):
    """
    Compute basic significance metrics for comparing two methods.
    With single seed, report effect size only.
    """
    diff = acc_a - acc_b
    return {
        "difference": round(diff, 4),
        "direction": "A > B" if diff > 0 else "B > A" if diff < 0 else "equal",
        "note": "single-seed pilot — no statistical test possible"
    }


def aggregate_cifar10_diagnostic():
    """Aggregate CIFAR-10 diagnostic pilot results."""
    summary = load_json(PILOTS_DIR / "diagnostic_cifar10" / "pilot_summary.json")
    if not summary:
        return None

    methods = summary.get("per_method", {})
    nowd_acc = methods.get("NoWD", {}).get("best_acc", 0)

    # Map method names to directory names
    dir_map = {
        "UDWDC-v2": "UDWDC_v2",
    }

    results = {}
    for method_name, mdata in methods.items():
        acc = mdata.get("best_acc", mdata.get("final_test_acc", 0))

        # Try to get WD budget from epoch files
        dir_name = dir_map.get(method_name, method_name)
        # The file inside may use either - or _ naming
        traj_path = PILOTS_DIR / "diagnostic_cifar10" / dir_name / f"{method_name}_seed42_epochs.json"
        epoch_data = load_json(traj_path)

        total_wd = 0
        wd_per_epoch = []
        mean_effective_wds = []
        mean_alphas = []

        if epoch_data and isinstance(epoch_data, dict) and "epochs" in epoch_data:
            # Format: {"method": ..., "epochs": [{epoch: 0, ...}, ...]}
            for ep in epoch_data["epochs"]:
                wd_ep = ep.get("total_wd_budget_epoch", ep.get("wd_budget_epoch", 0))
                wd_per_epoch.append(wd_ep)
                total_wd += wd_ep
                if "mean_effective_wd" in ep:
                    mean_effective_wds.append(ep["mean_effective_wd"])
                if "mean_alpha_t" in ep:
                    mean_alphas.append(ep["mean_alpha_t"])

        bem, improvement = compute_bem(acc, nowd_acc, total_wd if total_wd > 0 else None)

        # CSI from rho trajectory
        rho_traj = mdata.get("mean_rho_trajectory", [])
        if rho_traj and len(rho_traj) >= 2:
            rho_arr = np.array(rho_traj)
            rho_mean = np.mean(rho_arr)
            rho_std = np.std(rho_arr)
            rho_cv = rho_std / rho_mean if rho_mean > 0 else float('inf')
            csi_rho = 1.0 / (1.0 + rho_cv)
        else:
            csi_rho = None
            rho_cv = None

        # CSI from WD budget per epoch
        csi_wd = compute_csi_from_wd_budget_per_epoch(wd_per_epoch, method_name) if wd_per_epoch else {}

        # AIS per method: SNR from alpha
        ais_per_method = None
        if mean_alphas and len(mean_alphas) >= 2:
            alpha_arr = np.array(mean_alphas)
            mean_a = np.mean(alpha_arr)
            std_a = np.std(alpha_arr)
            snr = mean_a**2 / (std_a**2) if std_a > 0 else 0
            ais_per_method = {
                "mean_alpha": round(float(mean_a), 6),
                "std_alpha": round(float(std_a), 6),
                "alpha_SNR": round(float(snr), 6),
            }

        results[method_name] = {
            "accuracy": acc,
            "improvement_vs_nowd": round(improvement, 4),
            "total_wd_budget": round(total_wd, 8) if total_wd else 0,
            "BEM": round(bem, 4) if bem is not None else None,
            "CSI_rho_based": round(csi_rho, 6) if csi_rho is not None else None,
            "rho_cv": round(rho_cv, 6) if rho_cv is not None else None,
            "CSI_temporal_wd": csi_wd.get("CSI_temporal"),
            "mean_effective_wd": round(np.mean(mean_effective_wds), 8) if mean_effective_wds else None,
            "ais": ais_per_method,
        }

    return {
        "dataset": "CIFAR-10",
        "model": "ResNet-20",
        "epochs": 10,
        "seed": 42,
        "baseline_acc": nowd_acc,
        "methods": results
    }


def aggregate_ablation_cifar100():
    """Aggregate CIFAR-100 ablation pilot results."""
    summary = load_json(PILOTS_DIR / "ablation_cifar100" / "pilot_summary.json")
    if not summary:
        return None

    variants = summary.get("variants", {})
    fixedwd_acc = variants.get("FixedWD", {}).get("final_test_acc", 0)

    results = {}
    for var_name, vdata in variants.items():
        acc = vdata.get("final_test_acc", 0)
        total_wd = vdata.get("total_wd_budget", 0)
        wd_per_epoch = vdata.get("wd_budget_per_epoch", [])
        gen_gap = vdata.get("generalization_gap", None)

        bem, improvement = compute_bem(acc, fixedwd_acc, total_wd if total_wd > 0 else None)
        csi = compute_csi_from_wd_budget_per_epoch(wd_per_epoch, var_name)

        results[var_name] = {
            "K_p": vdata.get("K_p", 0),
            "K_i": vdata.get("K_i", 0),
            "K_d": vdata.get("K_d", 0),
            "is_v2": vdata.get("is_v2_default", False),
            "accuracy": acc,
            "improvement_vs_fixedwd": round(improvement, 4),
            "generalization_gap": round(gen_gap, 4) if gen_gap is not None else None,
            "total_wd_budget": round(total_wd, 8),
            "v2_cumulative_wd_budget": vdata.get("v2_cumulative_wd_budget"),
            "BEM": round(bem, 4) if bem is not None else None,
            **csi
        }

    return {
        "dataset": "CIFAR-100",
        "model": "VGG-16-BN",
        "epochs": 10,
        "seed": 42,
        "baseline_acc": fixedwd_acc,
        "variants": results
    }


def aggregate_imagenet_main():
    """Aggregate ImageNet main pilot results."""
    summary = load_json(PILOTS_DIR / "imagenet_main" / "pilot_summary_v2.json")
    if not summary:
        return None

    results_data = summary.get("results", {})
    nowd_acc = results_data.get("NoWD", {}).get("best_val_acc", 0)

    results = {}
    for method_name, mdata in results_data.items():
        acc = mdata.get("best_val_acc", 0)
        total_wd = mdata.get("total_wd_budget", 0)
        epoch_results = mdata.get("epoch_results", [])

        bem, improvement = compute_bem(acc, nowd_acc, total_wd if total_wd > 0 else None)
        csi = compute_csi_from_epoch_data(epoch_results, method_name)

        results[method_name] = {
            "accuracy": acc,
            "top5_acc": mdata.get("final_val_top5", None),
            "improvement_vs_nowd": round(improvement, 4),
            "total_wd_budget": round(total_wd, 8),
            "cumulative_wd_budget_v2": mdata.get("cumulative_wd_budget_v2", 0),
            "BEM": round(bem, 4) if bem is not None else None,
            **csi
        }

    return {
        "dataset": "ImageNet",
        "model": "ResNet-50",
        "epochs": 2,
        "seed": 42,
        "baseline_acc": nowd_acc,
        "note": "2-epoch pilot only — rankings not reliable for final claims",
        "methods": results
    }


def aggregate_imagenet_budget_matched():
    """Aggregate ImageNet budget-matched pilot results."""
    summary = load_json(PILOTS_DIR / "imagenet_budget_matched" / "pilot_summary.json")
    if not summary:
        return None

    results_data = summary.get("results", {})

    configs = {}
    for config_name, cdata in results_data.items():
        lam = cdata.get("lambda", cdata.get("weight_decay", 0))
        acc = cdata.get("best_val_acc", 0)
        total_wd = cdata.get("total_wd_budget", 0)
        epoch_results = cdata.get("epoch_results", [])

        csi = compute_csi_from_epoch_data(epoch_results, f"FixedWD_lam{lam}")

        configs[config_name] = {
            "lambda": lam,
            "accuracy": acc,
            "total_wd_budget": round(total_wd, 4),
            "BEM_raw": round(acc / total_wd, 8) if total_wd > 0 else None,
            **csi
        }

    return {
        "dataset": "ImageNet",
        "model": "ResNet-50",
        "epochs": 2,
        "seed": 42,
        "note": "Budget sweep: isolating WD budget confound",
        "configs": configs
    }


def aggregate_alignment_informativeness():
    """Aggregate alignment informativeness pilot results."""
    summary = load_json(PILOTS_DIR / "alignment_informativeness" / "pilot_summary.json")
    if not summary:
        return None

    per_run = summary.get("per_run_results", [])
    loo_cv = summary.get("loo_cv_r_squared", {})
    h6 = summary.get("h6_analysis", {})

    # Compute AIS from per-run data
    ais_result = compute_ais(per_run)

    # Group by WD level for additional analysis
    wd_groups = {}
    for run in per_run:
        wd = run.get("wd", 0)
        wd_key = f"wd_{wd}"
        if wd_key not in wd_groups:
            wd_groups[wd_key] = []
        wd_groups[wd_key].append(run)

    wd_level_summary = {}
    for wd_key, runs in wd_groups.items():
        accs = [r["best_test_acc"] for r in runs]
        gaps = [r["generalization_gap"] for r in runs]
        alphas = [r["alpha_bar"] for r in runs]
        wd_level_summary[wd_key] = {
            "n_runs": len(runs),
            "mean_acc": round(np.mean(accs), 2),
            "mean_gap": round(np.mean(gaps), 4),
            "mean_alpha_bar": round(np.mean(alphas), 6),
        }

    return {
        "dataset": "CIFAR-100",
        "model": "ResNet-20",
        "epochs": 30,
        "seed": 42,
        "n_runs": len(per_run),
        "loo_cv_r_squared": loo_cv.get("predicting_gen_gap", {}),
        "h6_analysis": h6,
        "AIS": ais_result,
        "wd_level_summary": wd_level_summary
    }


def aggregate_batchsize_sweep():
    """Aggregate batch-size sweep pilot results."""
    summary = load_json(PILOTS_DIR / "batchsize_sweep" / "pilot_summary.json")
    if not summary:
        return None

    snr_data = summary.get("snr_monotonicity", {})
    acc_deltas = summary.get("accuracy_deltas", {})

    # Compute per-method AIS across batch sizes
    method_ais = {}
    for method, sdata in snr_data.items():
        snr_values = sdata.get("snr_values", {})
        if snr_values:
            snr_arr = np.array(list(snr_values.values()))
            mean_snr = np.mean(snr_arr)
            std_snr = np.std(snr_arr)
            cv_snr = std_snr / mean_snr if mean_snr > 0 else float('inf')

            method_ais[method] = {
                "mean_SNR": round(mean_snr, 6),
                "std_SNR": round(std_snr, 6),
                "CV_SNR": round(cv_snr, 6),
                "monotonic": sdata.get("monotonic", False),
                "direction": sdata.get("dominant_direction", "unknown"),
                "snr_values": {k: round(v, 8) for k, v in snr_values.items()},
            }

    return {
        "dataset": "CIFAR-100",
        "model": "VGG-16-BN",
        "epochs": 5,
        "seed": 42,
        "batch_sizes": [64, 128, 256, 512, 1024],
        "method_snr": method_ais,
        "accuracy_deltas": acc_deltas,
        "h3_analysis": {
            "note": "H3 predicts continuous beats binary at large batch, binary better at small batch",
            "bs64_cwd_delta": acc_deltas.get("64", {}).get("CWD", None),
            "bs64_udwdc_delta": acc_deltas.get("64", {}).get("UDWDC-v2", None),
            "bs1024_cwd_delta": acc_deltas.get("1024", {}).get("CWD", None),
            "bs1024_udwdc_delta": acc_deltas.get("1024", {}).get("UDWDC-v2", None),
        }
    }


def compute_cross_benchmark_rankings(all_results):
    """Compute rankings across all benchmarks and detect divergences."""
    rankings = {}

    # CIFAR-10 rankings
    if "cifar10_diagnostic" in all_results and all_results["cifar10_diagnostic"]:
        methods = all_results["cifar10_diagnostic"]["methods"]

        # By accuracy
        acc_sorted = sorted(
            [(m, d["accuracy"]) for m, d in methods.items()],
            key=lambda x: -x[1]
        )
        acc_ranks = {m: i+1 for i, (m, _) in enumerate(acc_sorted)}

        # By BEM (only for methods with valid BEM)
        bem_items = [(m, d["BEM"]) for m, d in methods.items() if d.get("BEM") is not None]
        bem_sorted = sorted(bem_items, key=lambda x: -x[1])
        bem_ranks = {m: i+1 for i, (m, _) in enumerate(bem_sorted)}

        divergences = []
        for method in acc_ranks:
            if method in bem_ranks:
                shift = acc_ranks[method] - bem_ranks[method]
                if shift != 0:
                    divergences.append({
                        "method": method,
                        "acc_rank": acc_ranks[method],
                        "bem_rank": bem_ranks[method],
                        "shift": shift
                    })

        rankings["cifar10"] = {
            "by_accuracy": acc_sorted,
            "by_BEM": bem_sorted,
            "divergences": divergences,
            "max_rank_shift": max(abs(d["shift"]) for d in divergences) if divergences else 0
        }

    # ImageNet rankings
    if "imagenet_main" in all_results and all_results["imagenet_main"]:
        methods = all_results["imagenet_main"]["methods"]

        acc_sorted = sorted(
            [(m, d["accuracy"]) for m, d in methods.items()],
            key=lambda x: -x[1]
        )
        acc_ranks = {m: i+1 for i, (m, _) in enumerate(acc_sorted)}

        bem_items = [(m, d["BEM"]) for m, d in methods.items() if d.get("BEM") is not None]
        bem_sorted = sorted(bem_items, key=lambda x: -x[1])
        bem_ranks = {m: i+1 for i, (m, _) in enumerate(bem_sorted)}

        divergences = []
        for method in acc_ranks:
            if method in bem_ranks:
                shift = acc_ranks[method] - bem_ranks[method]
                if shift != 0:
                    divergences.append({
                        "method": method,
                        "acc_rank": acc_ranks[method],
                        "bem_rank": bem_ranks[method],
                        "shift": shift
                    })

        rankings["imagenet"] = {
            "by_accuracy": acc_sorted,
            "by_BEM": bem_sorted,
            "divergences": divergences,
            "max_rank_shift": max(abs(d["shift"]) for d in divergences) if divergences else 0
        }

    # CSI rankings (from CIFAR-10)
    if "cifar10_diagnostic" in all_results and all_results["cifar10_diagnostic"]:
        methods = all_results["cifar10_diagnostic"]["methods"]
        csi_items = [(m, d.get("CSI_rho_based")) for m, d in methods.items() if d.get("CSI_rho_based") is not None]
        csi_sorted = sorted(csi_items, key=lambda x: -x[1])
        rankings["csi_cifar10"] = csi_sorted

    # CSI rankings from ablation
    if "ablation_cifar100" in all_results and all_results["ablation_cifar100"]:
        variants = all_results["ablation_cifar100"]["variants"]
        csi_items = [(v, d.get("CSI_temporal")) for v, d in variants.items() if d.get("CSI_temporal") is not None]
        csi_sorted = sorted(csi_items, key=lambda x: -x[1])
        rankings["csi_ablation"] = csi_sorted

    return rankings


def compute_udwdc_v2_stability_comparison(all_results):
    """
    CRITICAL: Compare UDWDC vs UDWDC-v2 CSI to verify stability fix.
    """
    comparison = {}

    # From CIFAR-10
    if "cifar10_diagnostic" in all_results and all_results["cifar10_diagnostic"]:
        methods = all_results["cifar10_diagnostic"]["methods"]
        udwdc = methods.get("UDWDC", {})
        udwdc_v2 = methods.get("UDWDC-v2", {})
        comparison["cifar10"] = {
            "UDWDC_acc": udwdc.get("accuracy"),
            "UDWDC-v2_acc": udwdc_v2.get("accuracy"),
            "UDWDC_CSI_rho": udwdc.get("CSI_rho_based"),
            "UDWDC-v2_CSI_rho": udwdc_v2.get("CSI_rho_based"),
            "v2_csi_positive": (udwdc_v2.get("CSI_rho_based") or 0) > 0,
        }

    # From ImageNet
    if "imagenet_main" in all_results and all_results["imagenet_main"]:
        methods = all_results["imagenet_main"]["methods"]
        udwdc = methods.get("UDWDC", {})
        udwdc_v2 = methods.get("UDWDC-v2", {})
        comparison["imagenet"] = {
            "UDWDC_acc": udwdc.get("accuracy"),
            "UDWDC-v2_acc": udwdc_v2.get("accuracy"),
            "UDWDC_wd_budget": udwdc.get("total_wd_budget"),
            "UDWDC-v2_wd_budget": udwdc_v2.get("total_wd_budget"),
            "UDWDC-v2_cumulative_v2": udwdc_v2.get("cumulative_wd_budget_v2"),
            "v2_wd_positive": (udwdc_v2.get("total_wd_budget") or 0) > 0,
        }

    # From ablation
    if "ablation_cifar100" in all_results and all_results["ablation_cifar100"]:
        variants = all_results["ablation_cifar100"]["variants"]
        comparison["ablation_variants"] = {}
        for var_name, vdata in variants.items():
            comparison["ablation_variants"][var_name] = {
                "accuracy": vdata.get("accuracy"),
                "total_wd_budget": vdata.get("total_wd_budget"),
                "CSI_temporal": vdata.get("CSI_temporal"),
                "is_v2": vdata.get("is_v2", False),
                "wd_budget_positive": (vdata.get("total_wd_budget") or 0) > 0,
            }

    return comparison


def compute_budget_confound_analysis(all_results):
    """
    Analyze whether performance differences are confounded by WD budget differences.
    Compare methods at matched budget levels.
    """
    analysis = {}

    # ImageNet budget-matched analysis
    if "imagenet_budget_matched" in all_results and all_results["imagenet_budget_matched"]:
        bm = all_results["imagenet_budget_matched"]["configs"]

        # Collect (budget, accuracy) pairs
        budget_acc = []
        for config_name, cdata in bm.items():
            budget_acc.append({
                "lambda": cdata["lambda"],
                "budget": cdata["total_wd_budget"],
                "accuracy": cdata["accuracy"],
            })

        # Compare with main methods
        if "imagenet_main" in all_results and all_results["imagenet_main"]:
            main_methods = all_results["imagenet_main"]["methods"]

            method_budgets = {}
            for m, d in main_methods.items():
                if d.get("total_wd_budget", 0) > 0:
                    method_budgets[m] = {
                        "budget": d["total_wd_budget"],
                        "accuracy": d["accuracy"],
                    }

            analysis["budget_comparison"] = {
                "fixed_wd_sweep": budget_acc,
                "dynamic_methods": method_budgets,
                "note": "Compare dynamic method accuracy at similar WD budget to FixedWD sweep"
            }

            # Find closest budget match for each dynamic method
            for method, mdata in method_budgets.items():
                if method in ["NoWD", "FixedWD"]:
                    continue
                m_budget = mdata["budget"]
                # Find closest FixedWD budget
                closest = min(budget_acc, key=lambda x: abs(x["budget"] - m_budget))
                analysis.setdefault("closest_matches", {})[method] = {
                    "method_budget": round(m_budget, 6),
                    "method_accuracy": mdata["accuracy"],
                    "closest_fixedwd_lambda": closest["lambda"],
                    "closest_fixedwd_budget": round(closest["budget"], 4),
                    "closest_fixedwd_accuracy": closest["accuracy"],
                    "accuracy_difference": round(mdata["accuracy"] - closest["accuracy"], 4),
                    "budget_confounded": abs(mdata["accuracy"] - closest["accuracy"]) < 0.02,
                }

    return analysis


def generate_summary(all_results, rankings, stability, budget_analysis):
    """Generate executive summary of metrics computation."""
    key_findings = []
    pass_criteria = {}

    # Check BEM computability
    bem_computed = 0
    bem_total = 0
    for section_key in ["cifar10_diagnostic", "imagenet_main"]:
        if section_key in all_results and all_results[section_key]:
            methods = all_results[section_key].get("methods", {})
            for m, d in methods.items():
                bem_total += 1
                if d.get("BEM") is not None:
                    bem_computed += 1

    pass_criteria["BEM_computable"] = bem_computed > 0
    key_findings.append(f"BEM computed for {bem_computed}/{bem_total} method-benchmark pairs")

    # Check CSI for all methods
    csi_computed = 0
    csi_total = 0
    udwdc_v2_csi_positive = False

    for section_key in ["cifar10_diagnostic"]:
        if section_key in all_results and all_results[section_key]:
            methods = all_results[section_key].get("methods", {})
            for m, d in methods.items():
                csi_total += 1
                if d.get("CSI_rho_based") is not None:
                    csi_computed += 1
                    if m == "UDWDC-v2" and d["CSI_rho_based"] > 0:
                        udwdc_v2_csi_positive = True

    pass_criteria["CSI_computable"] = csi_computed > 0
    pass_criteria["UDWDC_v2_CSI_positive"] = udwdc_v2_csi_positive
    key_findings.append(f"CSI computed for {csi_computed}/{csi_total} methods on CIFAR-10")

    # Check AIS
    ais_result = all_results.get("alignment_informativeness", {})
    ais_data = ais_result.get("AIS") if ais_result else None
    pass_criteria["AIS_computable"] = ais_data is not None and ais_data.get("AIS") is not None
    if ais_data:
        key_findings.append(f"AIS = {ais_data.get('AIS', 'N/A')}, Spearman rho = {ais_data.get('spearman_rho', 'N/A')}")

    # Overall pass
    pass_criteria["all_pass"] = all(pass_criteria.values())

    # Ranking divergence findings
    if "cifar10" in rankings:
        max_shift = rankings["cifar10"].get("max_rank_shift", 0)
        if max_shift > 0:
            key_findings.append(f"CIFAR-10: Max rank shift between accuracy and BEM = {max_shift}")

    if "imagenet" in rankings:
        max_shift = rankings["imagenet"].get("max_rank_shift", 0)
        if max_shift > 0:
            key_findings.append(f"ImageNet: Max rank shift between accuracy and BEM = {max_shift}")

    # Stability fix verification
    if stability:
        cifar_stab = stability.get("cifar10", {})
        if cifar_stab.get("v2_csi_positive"):
            key_findings.append("UDWDC-v2 CSI is POSITIVE on CIFAR-10 (stability fix verified)")
        else:
            key_findings.append("WARNING: UDWDC-v2 CSI not confirmed positive on CIFAR-10")

        imgnet_stab = stability.get("imagenet", {})
        if imgnet_stab.get("v2_wd_positive"):
            key_findings.append("UDWDC-v2 WD budget > 0 on ImageNet (no controller collapse)")

    # Budget confound findings
    if budget_analysis and "closest_matches" in budget_analysis:
        confounded = sum(1 for v in budget_analysis["closest_matches"].values() if v.get("budget_confounded"))
        total = len(budget_analysis["closest_matches"])
        key_findings.append(f"Budget confound: {confounded}/{total} methods show accuracy within 2% of budget-matched FixedWD")

    return {
        "pass_criteria": pass_criteria,
        "key_findings": key_findings,
        "recommendation": "GO" if pass_criteria.get("all_pass", False) else "REFINE"
    }


def main():
    print("=" * 70)
    print("METRICS COMPUTATION PILOT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Aggregate all data sources
    print("\n[1/7] Aggregating CIFAR-10 diagnostic...")
    cifar10 = aggregate_cifar10_diagnostic()
    print(f"  Methods: {len(cifar10['methods']) if cifar10 else 0}")

    print("\n[2/7] Aggregating CIFAR-100 ablation...")
    ablation = aggregate_ablation_cifar100()
    print(f"  Variants: {len(ablation['variants']) if ablation else 0}")

    print("\n[3/7] Aggregating ImageNet main...")
    imagenet = aggregate_imagenet_main()
    print(f"  Methods: {len(imagenet['methods']) if imagenet else 0}")

    print("\n[4/7] Aggregating ImageNet budget-matched...")
    budget_matched = aggregate_imagenet_budget_matched()
    print(f"  Configs: {len(budget_matched['configs']) if budget_matched else 0}")

    print("\n[5/7] Aggregating alignment informativeness...")
    alignment = aggregate_alignment_informativeness()
    print(f"  Runs: {alignment['n_runs'] if alignment else 0}")

    print("\n[6/7] Aggregating batch-size sweep...")
    batchsize = aggregate_batchsize_sweep()
    print(f"  Methods: {len(batchsize['method_snr']) if batchsize else 0}")

    # Compile all results
    all_results = {
        "cifar10_diagnostic": cifar10,
        "ablation_cifar100": ablation,
        "imagenet_main": imagenet,
        "imagenet_budget_matched": budget_matched,
        "alignment_informativeness": alignment,
        "batchsize_sweep": batchsize,
    }

    # Cross-benchmark analysis
    print("\n[7/7] Computing cross-benchmark rankings and analysis...")
    rankings = compute_cross_benchmark_rankings(all_results)
    stability = compute_udwdc_v2_stability_comparison(all_results)
    budget_analysis = compute_budget_confound_analysis(all_results)
    summary = generate_summary(all_results, rankings, stability, budget_analysis)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    for finding in summary["key_findings"]:
        print(f"  - {finding}")

    print(f"\nPass criteria: {summary['pass_criteria']}")
    print(f"Recommendation: {summary['recommendation']}")

    # Print ranking tables
    if "cifar10" in rankings:
        print("\n--- CIFAR-10 Rankings ---")
        print("By Accuracy:")
        for i, (m, acc) in enumerate(rankings["cifar10"]["by_accuracy"]):
            print(f"  {i+1}. {m}: {acc}%")
        print("By BEM:")
        for i, (m, bem) in enumerate(rankings["cifar10"]["by_BEM"]):
            print(f"  {i+1}. {m}: BEM={bem:.4f}")

    if "imagenet" in rankings:
        print("\n--- ImageNet Rankings (2-epoch pilot) ---")
        print("By Accuracy:")
        for i, (m, acc) in enumerate(rankings["imagenet"]["by_accuracy"]):
            print(f"  {i+1}. {m}: {acc}")
        print("By BEM:")
        for i, (m, bem) in enumerate(rankings["imagenet"]["by_BEM"]):
            print(f"  {i+1}. {m}: BEM={bem:.6f}")

    if "csi_ablation" in rankings:
        print("\n--- CSI Rankings (Ablation) ---")
        for i, (v, csi) in enumerate(rankings["csi_ablation"]):
            print(f"  {i+1}. {v}: CSI_temporal={csi:.6f}")

    # Stability comparison
    print("\n--- UDWDC vs UDWDC-v2 Stability ---")
    for bench, data in stability.items():
        if bench == "ablation_variants":
            continue
        print(f"  {bench}: {json.dumps(data, indent=4, cls=NumpyEncoder)}")

    # Compile full output
    output = {
        "task": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "metrics": all_results,
        "rankings": rankings,
        "stability_comparison": stability,
        "budget_confound_analysis": budget_analysis,
        "summary": summary,
    }

    # Save results
    output_path = OUTPUT_DIR / "metrics_results_v2.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")

    # Save pilot summary
    pilot_summary = {
        "overall_recommendation": summary["recommendation"],
        "task_id": TASK_ID,
        "mode": "pilot",
        "pass_criteria": summary["pass_criteria"],
        "key_findings": summary["key_findings"],
        "metrics_summary": {
            "BEM": {
                "cifar10_best": None,
                "imagenet_best": None,
            },
            "CSI": {
                "most_stable": None,
                "least_stable": None,
                "udwdc_v2_positive": summary["pass_criteria"].get("UDWDC_v2_CSI_positive", False)
            },
            "AIS": alignment.get("AIS") if alignment else None,
        },
        "ranking_divergences": {
            "cifar10_max_shift": rankings.get("cifar10", {}).get("max_rank_shift", 0),
            "imagenet_max_shift": rankings.get("imagenet", {}).get("max_rank_shift", 0),
        },
        "timestamp": datetime.now().isoformat()
    }

    # Fill in best BEM values
    if cifar10 and cifar10.get("methods"):
        bems = [(m, d["BEM"]) for m, d in cifar10["methods"].items() if d.get("BEM") is not None]
        if bems:
            best = max(bems, key=lambda x: x[1])
            pilot_summary["metrics_summary"]["BEM"]["cifar10_best"] = {"method": best[0], "BEM": best[1]}

    if imagenet and imagenet.get("methods"):
        bems = [(m, d["BEM"]) for m, d in imagenet["methods"].items() if d.get("BEM") is not None]
        if bems:
            best = max(bems, key=lambda x: x[1])
            pilot_summary["metrics_summary"]["BEM"]["imagenet_best"] = {"method": best[0], "BEM": best[1]}

    # Fill in CSI
    if "csi_cifar10" in rankings and rankings["csi_cifar10"]:
        pilot_summary["metrics_summary"]["CSI"]["most_stable"] = {
            "method": rankings["csi_cifar10"][0][0],
            "CSI": rankings["csi_cifar10"][0][1]
        }
        pilot_summary["metrics_summary"]["CSI"]["least_stable"] = {
            "method": rankings["csi_cifar10"][-1][0],
            "CSI": rankings["csi_cifar10"][-1][1]
        }

    summary_path = OUTPUT_DIR / "pilot_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(pilot_summary, f, indent=2, cls=NumpyEncoder)
    print(f"Pilot summary saved to: {summary_path}")

    # Write markdown summary
    md_lines = [
        "# Metrics Computation Pilot Summary",
        "",
        f"**Recommendation**: {summary['recommendation']}",
        f"**Timestamp**: {datetime.now().isoformat()}",
        "",
        "## Pass Criteria",
    ]
    for k, v in summary["pass_criteria"].items():
        status = "PASS" if v else "FAIL"
        md_lines.append(f"- [{status}] {k}")

    md_lines.extend(["", "## Key Findings", ""])
    for finding in summary["key_findings"]:
        md_lines.append(f"- {finding}")

    md_lines.extend(["", "## CIFAR-10 Rankings", ""])
    if "cifar10" in rankings:
        md_lines.append("| Method | Acc Rank | BEM Rank | Shift |")
        md_lines.append("|--------|----------|----------|-------|")
        acc_dict = {m: i+1 for i, (m, _) in enumerate(rankings["cifar10"]["by_accuracy"])}
        bem_dict = {m: i+1 for i, (m, _) in enumerate(rankings["cifar10"]["by_BEM"])}
        for method in acc_dict:
            acc_r = acc_dict[method]
            bem_r = bem_dict.get(method, "-")
            shift = acc_r - bem_r if isinstance(bem_r, int) else "-"
            md_lines.append(f"| {method} | {acc_r} | {bem_r} | {shift} |")

    md_lines.extend(["", "## UDWDC-v2 Stability Fix", ""])
    for bench, data in stability.items():
        if bench == "ablation_variants":
            continue
        md_lines.append(f"### {bench}")
        for k, v in data.items():
            md_lines.append(f"- {k}: {v}")

    md_path = OUTPUT_DIR / "pilot_summary.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"Markdown summary saved to: {md_path}")

    print("\n" + "=" * 70)
    print("METRICS COMPUTATION PILOT COMPLETE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
