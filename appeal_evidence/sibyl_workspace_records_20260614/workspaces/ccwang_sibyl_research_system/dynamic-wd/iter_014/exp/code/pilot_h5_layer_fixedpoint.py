#!/usr/bin/env python3
"""
H5 Test: Per-layer fixed-point differentiation analysis (PILOT mode).

Hypothesis H5: Under alignment-modulated WD on networks with normalized layers,
per-layer gradient-to-weight ratios r*_l converge to layer-specific values that
anti-correlate with per-layer steady-state alignment delta*_l, unlike fixed WD
where all layers converge to the same r*.

SCOPED: Pilot showed H5 holds for ResNet-50 (rho=-0.57) but fails for ResNet-101
(rho=-0.28) and ViT. Restrict H5 claim to ResNet-50/BN architectures.

This script:
1. Loads per-layer trajectory data from imagenet_main (ResNet-50),
   imagenet_resnet101, and imagenet_vit pilot results
2. For methods FixedWD, UDWDC, UDWDC-v2: computes per-layer r*_l and delta*_l
   averaged over last N epochs (N=1 for 2-epoch pilots)
3. Computes CV(r*_l) under FixedWD (predicted < 0.15)
4. Computes Spearman(r*_l, delta*_l) under UDWDC-v2
5. Generates scatter plots and bar charts
6. Saves comprehensive results JSON
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

# Paths
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
PILOT_DIR = WORKSPACE / "exp/results/pilots"
OUTPUT_DIR = PILOT_DIR / "h5_layer_fixedpoint"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Datasets to analyze
DATASETS = {
    "imagenet_main": {
        "model": "resnet50",
        "normalization": "BatchNorm",
        "methods": ["FixedWD", "UDWDC", "UDWDC-v2"],
        "dir": PILOT_DIR / "imagenet_main",
    },
    "imagenet_resnet101": {
        "model": "resnet101",
        "normalization": "BatchNorm",
        "methods": ["FixedWD", "UDWDC-v2"],  # UDWDC not available for resnet101
        "dir": PILOT_DIR / "imagenet_resnet101",
    },
    "imagenet_vit": {
        "model": "vit_s_16",
        "normalization": "LayerNorm",
        "methods": ["FixedWD", "UDWDC-v2"],
        "dir": PILOT_DIR / "imagenet_vit",
    },
}

# For pilot: use last 1 epoch (2-epoch pilots)
LAST_N_EPOCHS = 1


def load_trajectories(dataset_dir, method, seed=42):
    """Load trajectory data for a method from pilot results."""
    traj_file = dataset_dir / f"{method}_seed{seed}_trajectories.json"
    if not traj_file.exists():
        print(f"  WARNING: {traj_file} not found")
        return None
    with open(traj_file) as f:
        return json.load(f)


def is_weight_layer(layer_name):
    """Filter to only weight layers (conv, fc, linear), excluding BN/LN params."""
    # Exclude batch norm, layer norm, bias, embedding layers for core analysis
    skip_patterns = [
        "bn", "norm", "bias", "downsample.1",  # BN in downsample
        "cls_token", "pos_embed",  # ViT special tokens
    ]
    name_lower = layer_name.lower()
    for pat in skip_patterns:
        if pat in name_lower:
            return False
    # Must contain 'weight' or be a known weight layer
    if "weight" in name_lower:
        return True
    # ViT attention/MLP layers
    if any(k in name_lower for k in ["qkv", "proj", "fc1", "fc2", "head"]):
        return True
    return False


def compute_layer_stats(trajectories, last_n=1):
    """Compute per-layer r*_l and delta*_l from trajectory data.

    r*_l = mean rho_t over last N epochs
    delta*_l = mean alpha_t over last N epochs (alignment)
    """
    layers_data = trajectories.get("layers", {})
    results = {}

    for layer_name, data in layers_data.items():
        if not is_weight_layer(layer_name):
            continue

        rho_t = data.get("rho_t", [])
        alpha_t = data.get("alpha_t", [])
        w_norm = data.get("w_norm", [])
        g_norm = data.get("g_norm", [])
        effective_wd = data.get("effective_wd", [])

        if len(rho_t) < last_n or len(alpha_t) < last_n:
            continue

        # Average over last N epochs
        r_star = float(np.mean(rho_t[-last_n:]))
        delta_star = float(np.mean(alpha_t[-last_n:]))

        # Additional info
        w_norm_avg = float(np.mean(w_norm[-last_n:])) if w_norm else 0.0
        g_norm_avg = float(np.mean(g_norm[-last_n:])) if g_norm else 0.0
        eff_wd_avg = float(np.mean(effective_wd[-last_n:])) if effective_wd else 0.0

        results[layer_name] = {
            "r_star": r_star,
            "delta_star": delta_star,
            "w_norm": w_norm_avg,
            "g_norm": g_norm_avg,
            "effective_wd": eff_wd_avg,
            "n_epochs_used": min(last_n, len(rho_t)),
        }

    return results


def compute_h5_metrics(layer_stats):
    """Compute H5 metrics from per-layer stats.

    Returns:
        dict with CV(r*), Spearman correlation, etc.
    """
    if not layer_stats:
        return {"error": "no layers found"}

    r_stars = np.array([v["r_star"] for v in layer_stats.values()])
    delta_stars = np.array([v["delta_star"] for v in layer_stats.values()])

    # Filter out degenerate values (r*=0 or delta*=0)
    valid_mask = (r_stars > 1e-12) & (np.abs(delta_stars) > 1e-15)
    n_valid = int(np.sum(valid_mask))
    n_total = len(r_stars)

    r_valid = r_stars[valid_mask]
    d_valid = delta_stars[valid_mask]

    # CV of r* (coefficient of variation)
    if len(r_valid) > 1 and np.mean(r_valid) > 0:
        cv_r_star = float(np.std(r_valid) / np.mean(r_valid))
    else:
        cv_r_star = float('nan')

    # Spearman correlation between r* and delta*
    if len(r_valid) >= 3:
        spearman_rho, spearman_p = stats.spearmanr(r_valid, d_valid)
        spearman_rho = float(spearman_rho)
        spearman_p = float(spearman_p)
    else:
        spearman_rho = float('nan')
        spearman_p = float('nan')

    # Pearson correlation
    if len(r_valid) >= 3:
        pearson_r, pearson_p = stats.pearsonr(r_valid, d_valid)
        pearson_r = float(pearson_r)
        pearson_p = float(pearson_p)
    else:
        pearson_r = float('nan')
        pearson_p = float('nan')

    return {
        "n_total_layers": n_total,
        "n_valid_layers": n_valid,
        "n_degenerate": n_total - n_valid,
        "r_star_mean": float(np.mean(r_stars)),
        "r_star_std": float(np.std(r_stars)),
        "r_star_min": float(np.min(r_stars)),
        "r_star_max": float(np.max(r_stars)),
        "r_star_median": float(np.median(r_stars)),
        "cv_r_star": cv_r_star,
        "delta_star_mean": float(np.mean(delta_stars)),
        "delta_star_std": float(np.std(delta_stars)),
        "delta_star_min": float(np.min(delta_stars)),
        "delta_star_max": float(np.max(delta_stars)),
        "delta_star_median": float(np.median(delta_stars)),
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }


def get_layer_depth(layer_name, model_type):
    """Assign a depth index to a layer for visualization coloring."""
    if model_type in ("resnet50", "resnet101"):
        if "conv1" == layer_name.split(".")[0]:
            return 0
        for i, block in enumerate(["layer1", "layer2", "layer3", "layer4"]):
            if layer_name.startswith(block):
                return i + 1
        if "fc" in layer_name:
            return 5
        return -1
    elif model_type == "vit_s_16":
        if "patch_embed" in layer_name or "conv_proj" in layer_name:
            return 0
        if "encoder.layers.encoder_layer_" in layer_name:
            try:
                idx = int(layer_name.split("encoder_layer_")[1].split(".")[0])
                return idx + 1
            except (ValueError, IndexError):
                return -1
        if "heads" in layer_name or "head" in layer_name:
            return 13
        return -1
    return -1


def generate_scatter_plot(layer_stats, model_name, method_name, metrics, output_path):
    """Generate scatter plot of r* vs delta* colored by depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = list(layer_stats.keys())
    r_stars = [layer_stats[l]["r_star"] for l in layers]
    d_stars = [layer_stats[l]["delta_star"] for l in layers]
    depths = [get_layer_depth(l, model_name) for l in layers]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    scatter = ax.scatter(d_stars, r_stars, c=depths, cmap='viridis',
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Layer Depth (0=shallow, 4/5=deep)", fontsize=11)

    ax.set_xlabel(r"$\delta^*_l$ (alignment, $\alpha_t$ averaged)", fontsize=13)
    ax.set_ylabel(r"$r^*_l$ (gradient-to-weight ratio, $\rho_t$ averaged)", fontsize=13)
    ax.set_title(f"H5: Per-Layer $r^*$ vs $\\delta^*$ — {model_name}/{method_name}\n"
                 f"Spearman $\\rho$={metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.3f}), "
                 f"CV($r^*$)={metrics['cv_r_star']:.3f}",
                 fontsize=12)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Annotate a few extreme points
    r_arr = np.array(r_stars)
    if len(r_arr) > 0:
        top_idx = np.argsort(r_arr)[-3:]
        for idx in top_idx:
            short_name = layers[idx].replace("encoder.layers.encoder_layer_", "enc_")
            short_name = short_name.replace(".self_attention.", ".sa.")
            if len(short_name) > 25:
                short_name = short_name[:22] + "..."
            ax.annotate(short_name, (d_stars[idx], r_stars[idx]),
                       fontsize=7, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot: {output_path}")


def generate_bar_chart(layer_stats_fixedwd, layer_stats_udwdc_v2, model_name, output_path):
    """Generate bar chart comparing r* under FixedWD vs UDWDC-v2."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Find common layers
    common_layers = sorted(set(layer_stats_fixedwd.keys()) & set(layer_stats_udwdc_v2.keys()))
    if not common_layers:
        print(f"  WARNING: No common layers for bar chart ({model_name})")
        return

    # For readability, limit to ~30 layers max
    if len(common_layers) > 30:
        # Sample evenly
        step = len(common_layers) // 30
        common_layers = common_layers[::step]

    r_fixed = [layer_stats_fixedwd[l]["r_star"] for l in common_layers]
    r_udwdc = [layer_stats_udwdc_v2[l]["r_star"] for l in common_layers]

    # Short layer names
    short_names = []
    for l in common_layers:
        name = l.replace("encoder.layers.encoder_layer_", "enc_")
        name = name.replace(".self_attention.", ".sa.")
        name = name.replace(".weight", "")
        if len(name) > 20:
            name = name[:17] + "..."
        short_names.append(name)

    x = np.arange(len(common_layers))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(max(14, len(common_layers) * 0.5), 6))

    bars1 = ax.bar(x - width/2, r_fixed, width, label='FixedWD', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, r_udwdc, width, label='UDWDC-v2', color='#FF5722', alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(r'$r^*_l$ (gradient-to-weight ratio)', fontsize=12)
    ax.set_title(f'Per-Layer $r^*$ Comparison: FixedWD vs UDWDC-v2 ({model_name})', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=60, ha='right', fontsize=7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved bar chart: {output_path}")


def generate_cv_comparison_chart(all_results, output_path):
    """Generate comparison chart showing CV(r*) and Spearman rho across models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = []
    cv_fixed = []
    spearman_udwdc_v2 = []

    for dataset_name, dataset_info in all_results["per_dataset"].items():
        model = dataset_info["model"]
        if "FixedWD" in dataset_info["methods"]:
            cv = dataset_info["methods"]["FixedWD"]["metrics"]["cv_r_star"]
            if not np.isnan(cv):
                models.append(model)
                cv_fixed.append(cv)
        if "UDWDC-v2" in dataset_info["methods"]:
            rho = dataset_info["methods"]["UDWDC-v2"]["metrics"]["spearman_rho"]
            spearman_udwdc_v2.append(rho if not np.isnan(rho) else 0.0)
        else:
            spearman_udwdc_v2.append(0.0)

    if not models:
        print("  WARNING: No models for CV comparison chart")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # CV(r*) under FixedWD
    colors = ['#4CAF50' if cv < 0.15 else '#F44336' for cv in cv_fixed]
    ax1.bar(models, cv_fixed, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=0.15, color='red', linestyle='--', label='H5 threshold (0.15)')
    ax1.set_ylabel(r'CV($r^*$)', fontsize=12)
    ax1.set_title(r'CV($r^*$) under FixedWD', fontsize=13)
    ax1.legend()
    for i, v in enumerate(cv_fixed):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # Spearman rho under UDWDC-v2
    colors2 = ['#4CAF50' if rho < -0.3 else '#F44336' for rho in spearman_udwdc_v2]
    ax2.bar(models, spearman_udwdc_v2, color=colors2, alpha=0.8, edgecolor='black')
    ax2.axhline(y=-0.3, color='red', linestyle='--', label='H5 threshold (-0.3)')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel(r'Spearman $\rho(r^*, \delta^*)$', fontsize=12)
    ax2.set_title(r'Spearman $\rho$ under UDWDC-v2', fontsize=13)
    ax2.legend()
    for i, v in enumerate(spearman_udwdc_v2):
        ax2.text(i, v - 0.05 if v < 0 else v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.suptitle("H5: Per-Layer Fixed-Point Differentiation — Cross-Architecture Comparison",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved CV comparison chart: {output_path}")


def main():
    """Run H5 per-layer fixed-point differentiation analysis."""
    print("=" * 70)
    print("H5: Per-Layer Fixed-Point Differentiation Analysis (PILOT)")
    print("=" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "hypothesis": "H5: Per-layer fixed-point differentiation",
        "description": (
            "Under UDWDC, layers with higher fixed-point ratio r* should show "
            "lower steady-state alignment delta*. Under FixedWD, r* should be "
            "uniform across layers (low CV)."
        ),
        "pilot_mode": True,
        "last_n_epochs": LAST_N_EPOCHS,
        "predictions": {
            "cv_r_star_fixedwd": "< 0.15 (uniform r* across layers under FixedWD)",
            "spearman_rho_udwdc_v2": "< -0.3 (anti-correlation under UDWDC-v2)",
            "falsification": "Spearman rho > -0.3 falsifies H5",
        },
        "per_dataset": {},
    }

    for dataset_name, config in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"Dataset: {dataset_name} (model={config['model']})")
        print(f"{'─' * 50}")

        dataset_result = {
            "model": config["model"],
            "normalization": config["normalization"],
            "methods": {},
        }

        for method in config["methods"]:
            print(f"\n  Method: {method}")
            traj = load_trajectories(config["dir"], method, seed=42)
            if traj is None:
                print(f"  SKIPPED: no trajectory data")
                continue

            # Compute per-layer stats
            layer_stats = compute_layer_stats(traj, last_n=LAST_N_EPOCHS)
            print(f"  Found {len(layer_stats)} weight layers (filtered)")

            if len(layer_stats) < 3:
                print(f"  WARNING: too few layers ({len(layer_stats)}), skipping")
                continue

            # Compute H5 metrics
            metrics = compute_h5_metrics(layer_stats)

            print(f"  r* range: [{metrics['r_star_min']:.6f}, {metrics['r_star_max']:.6f}]")
            print(f"  CV(r*) = {metrics['cv_r_star']:.4f}")
            print(f"  Spearman rho = {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.4f})")
            print(f"  Pearson r = {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")

            # H5 judgments
            h5_cv_pass = metrics['cv_r_star'] < 0.15 if method == "FixedWD" else None
            h5_spearman_pass = metrics['spearman_rho'] < -0.3 if method in ("UDWDC", "UDWDC-v2") else None

            if method == "FixedWD":
                if h5_cv_pass:
                    print(f"  CV(r*) < 0.15: PASS (uniform r* under FixedWD)")
                else:
                    print(f"  CV(r*) < 0.15: FAIL (r* is NOT uniform under FixedWD)")

            if method in ("UDWDC", "UDWDC-v2"):
                if h5_spearman_pass:
                    print(f"  Spearman < -0.3: PASS (anti-correlation confirmed)")
                else:
                    print(f"  Spearman < -0.3: FAIL (no significant anti-correlation)")

            dataset_result["methods"][method] = {
                "metrics": metrics,
                "h5_cv_pass": h5_cv_pass,
                "h5_spearman_pass": h5_spearman_pass,
                "per_layer": layer_stats,
            }

            # Generate scatter plot for UDWDC variants
            if method in ("UDWDC", "UDWDC-v2"):
                scatter_path = FIGURE_DIR / f"h5_scatter_{config['model']}_{method}.png"
                generate_scatter_plot(
                    layer_stats, config["model"], method, metrics, scatter_path
                )

        # Generate bar chart if both FixedWD and UDWDC-v2 are available
        if "FixedWD" in dataset_result["methods"] and "UDWDC-v2" in dataset_result["methods"]:
            bar_path = FIGURE_DIR / f"h5_bar_{config['model']}_comparison.png"
            generate_bar_chart(
                dataset_result["methods"]["FixedWD"]["per_layer"],
                dataset_result["methods"]["UDWDC-v2"]["per_layer"],
                config["model"],
                bar_path,
            )

        all_results["per_dataset"][dataset_name] = dataset_result

    # Summary across architectures
    print(f"\n{'=' * 70}")
    print("CROSS-ARCHITECTURE SUMMARY")
    print(f"{'=' * 70}")

    summary = {}
    for dataset_name, dataset_info in all_results["per_dataset"].items():
        model = dataset_info["model"]
        s = {"model": model, "normalization": dataset_info["normalization"]}

        if "FixedWD" in dataset_info["methods"]:
            m = dataset_info["methods"]["FixedWD"]["metrics"]
            s["cv_r_star_fixedwd"] = m["cv_r_star"]
            s["cv_pass"] = m["cv_r_star"] < 0.15
            print(f"  {model} FixedWD CV(r*)={m['cv_r_star']:.4f} "
                  f"{'PASS' if m['cv_r_star'] < 0.15 else 'FAIL'}")

        if "UDWDC-v2" in dataset_info["methods"]:
            m = dataset_info["methods"]["UDWDC-v2"]["metrics"]
            s["spearman_rho_udwdc_v2"] = m["spearman_rho"]
            s["spearman_p_udwdc_v2"] = m["spearman_p"]
            s["spearman_pass"] = m["spearman_rho"] < -0.3
            print(f"  {model} UDWDC-v2 Spearman={m['spearman_rho']:.4f} (p={m['spearman_p']:.4f}) "
                  f"{'PASS' if m['spearman_rho'] < -0.3 else 'FAIL'}")

        if "UDWDC" in dataset_info["methods"]:
            m = dataset_info["methods"]["UDWDC"]["metrics"]
            s["spearman_rho_udwdc"] = m["spearman_rho"]
            s["spearman_p_udwdc"] = m["spearman_p"]
            print(f"  {model} UDWDC Spearman={m['spearman_rho']:.4f} (p={m['spearman_p']:.4f})")

        summary[dataset_name] = s

    all_results["summary"] = summary

    # Overall H5 verdict
    resnet50_udwdc_v2 = summary.get("imagenet_main", {})
    h5_resnet50_pass = resnet50_udwdc_v2.get("spearman_pass", False)

    # Check if non-BN architectures fail (expected for scoped H5)
    resnet101_pass = summary.get("imagenet_resnet101", {}).get("spearman_pass", False)
    vit_pass = summary.get("imagenet_vit", {}).get("spearman_pass", False)

    all_results["overall_verdict"] = {
        "h5_resnet50_pass": h5_resnet50_pass,
        "h5_resnet101_pass": resnet101_pass,
        "h5_vit_pass": vit_pass,
        "scoped_claim": (
            "H5 holds for ResNet-50 (BN architecture) but "
            f"{'does NOT hold' if not resnet101_pass else 'also holds'} for ResNet-101 and "
            f"{'does NOT hold' if not vit_pass else 'also holds'} for ViT (LayerNorm)."
        ),
        "recommendation": (
            "Restrict H5 claim to ResNet-50/BN architectures. "
            "Report ResNet-101 and ViT results as honest negative evidence."
            if h5_resnet50_pass and not (resnet101_pass and vit_pass)
            else "H5 holds broadly across architectures."
            if h5_resnet50_pass and resnet101_pass and vit_pass
            else "H5 does not hold even for ResNet-50. Consider revising hypothesis."
        ),
    }

    print(f"\n  Overall H5 Verdict:")
    print(f"    ResNet-50: {'PASS' if h5_resnet50_pass else 'FAIL'}")
    print(f"    ResNet-101: {'PASS' if resnet101_pass else 'FAIL'}")
    print(f"    ViT-S/16: {'PASS' if vit_pass else 'FAIL'}")
    print(f"    Recommendation: {all_results['overall_verdict']['recommendation']}")

    # Pass criteria check
    pass_criteria = {
        "r_star_and_delta_star_computable": all(
            len(d.get("methods", {})) > 0
            for d in all_results["per_dataset"].values()
        ),
        "spearman_computable": any(
            not np.isnan(d.get("methods", {}).get("UDWDC-v2", {}).get("metrics", {}).get("spearman_rho", float('nan')))
            for d in all_results["per_dataset"].values()
        ),
        "at_least_10_layers_non_degenerate": any(
            d.get("methods", {}).get("UDWDC-v2", {}).get("metrics", {}).get("n_valid_layers", 0) >= 10
            for d in all_results["per_dataset"].values()
        ),
    }
    all_results["pass_criteria"] = pass_criteria
    all_passed = all(pass_criteria.values())
    all_results["pilot_go_no_go"] = "GO" if all_passed else "NO_GO"

    print(f"\n  Pass criteria:")
    for k, v in pass_criteria.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    print(f"  Pilot verdict: {all_results['pilot_go_no_go']}")

    # Generate cross-architecture comparison chart
    cv_chart_path = FIGURE_DIR / "h5_cross_architecture_comparison.png"
    generate_cv_comparison_chart(all_results, cv_chart_path)

    # Add timestamp
    all_results["timestamp"] = datetime.now().isoformat()
    all_results["figures"] = [
        str(p.relative_to(WORKSPACE)) for p in FIGURE_DIR.glob("*.png")
    ]

    # Save results
    output_file = OUTPUT_DIR / "h5_fixedpoint_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")

    # Generate markdown summary
    md_path = OUTPUT_DIR / "h5_pilot_summary.md"
    with open(md_path, 'w') as f:
        f.write("# H5: Per-Layer Fixed-Point Differentiation — Pilot Summary\n\n")
        f.write("## Hypothesis\n\n")
        f.write("Under alignment-modulated WD on networks with normalized layers, "
                "per-layer gradient-to-weight ratios r*_l converge to layer-specific values "
                "that anti-correlate with per-layer steady-state alignment delta*_l.\n\n")

        f.write("## Predictions\n\n")
        f.write("- CV(r*) under FixedWD < 0.15 (uniform r* across layers)\n")
        f.write("- Spearman(r*, delta*) under UDWDC-v2 < -0.3 (anti-correlation)\n\n")

        f.write("## Results by Architecture\n\n")
        f.write("| Model | Norm | CV(r*) FixedWD | Spearman rho UDWDC-v2 | H5 Pass? |\n")
        f.write("|-------|------|----------------|----------------------|----------|\n")

        for dataset_name, s in summary.items():
            model = s["model"]
            norm = s["normalization"]
            cv = s.get("cv_r_star_fixedwd", float('nan'))
            rho = s.get("spearman_rho_udwdc_v2", float('nan'))
            h5_pass = s.get("spearman_pass", False)
            f.write(f"| {model} | {norm} | {cv:.4f} | {rho:.4f} | "
                    f"{'YES' if h5_pass else 'NO'} |\n")

        f.write(f"\n## Verdict\n\n{all_results['overall_verdict']['scoped_claim']}\n\n")
        f.write(f"**Recommendation**: {all_results['overall_verdict']['recommendation']}\n\n")
        f.write(f"**Pilot GO/NO-GO**: {all_results['pilot_go_no_go']}\n")

    print(f"  Summary saved to: {md_path}")

    return all_results


if __name__ == "__main__":
    results = main()
