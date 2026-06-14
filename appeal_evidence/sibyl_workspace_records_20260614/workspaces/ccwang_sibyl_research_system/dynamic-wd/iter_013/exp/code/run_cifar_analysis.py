#!/usr/bin/env python3
"""
Comprehensive CIFAR analysis for Unified Dynamic Weight Decay paper.
Loads all experimental results, computes aggregate statistics,
generates publication-quality figures, and writes a summary report.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
RESULTS   = WORKSPACE / "exp" / "results"
FULL      = RESULTS / "full"
FIG_DIR   = FULL / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["NoWD", "FixedWD", "SWD", "CWD", "CPR", "CAWD", "EqWD"]
SEEDS   = [42, 123, 456]

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "NoWD":    "#999999",
    "FixedWD": "#4477AA",
    "SWD":     "#66CCEE",
    "CWD":     "#228833",
    "CPR":     "#CCBB44",
    "CAWD":    "#EE6677",
    "EqWD":    "#AA3377",
}

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def find_results_json(run_dir: Path) -> dict | None:
    """Find and load the results JSON in a run directory."""
    for f in sorted(run_dir.glob("*_results.json")):
        with open(f) as fh:
            return json.load(fh)
    for f in sorted(run_dir.glob("default_results.json")):
        with open(f) as fh:
            return json.load(fh)
    return None


def load_method_results(base_dir: Path, methods=METHODS, seeds=SEEDS):
    """Load results for all method x seed combos. Returns {method: [result_dicts]}."""
    data = defaultdict(list)
    for m in methods:
        for s in seeds:
            run_dir = base_dir / f"{m}_seed{s}"
            if not run_dir.is_dir():
                print(f"  [WARN] missing: {run_dir}")
                continue
            res = find_results_json(run_dir)
            if res is None:
                print(f"  [WARN] no results JSON in {run_dir}")
                continue
            data[m].append(res)
    return data


def compute_stats(data: dict, key="best_test_top1"):
    """Return {method: (mean, std, values)} for a given metric key."""
    stats = {}
    for m, runs in data.items():
        vals = [r[key] for r in runs if key in r]
        if vals:
            stats[m] = (np.mean(vals), np.std(vals), vals)
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading CIFAR-100 / ResNet-20 full results ...")
resnet20_data = load_method_results(FULL / "cifar100_resnet20")
resnet20_stats = compute_stats(resnet20_data)

print("Loading CIFAR-100 / VGG16-BN full results ...")
vgg16_data = load_method_results(RESULTS / "cifar100_vgg16bn_full")
vgg16_stats = compute_stats(vgg16_data)

print("Loading ablation_beta ...")
beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
beta_stats = {}
for b in beta_values:
    run_dir = RESULTS / "ablation_beta" / f"beta_{b}"
    res = find_results_json(run_dir)
    if res:
        beta_stats[b] = res.get("best_test_top1", res.get("final_test_top1"))
    else:
        print(f"  [WARN] no result for beta={b}")

print("Loading ablation_ema ...")
ema_values = [0.8, 0.9, 0.95, 0.99]
ema_stats = {}
for a in ema_values:
    run_dir = RESULTS / "ablation_ema" / f"ema_{a}"
    res = find_results_json(run_dir)
    if res:
        ema_stats[a] = res.get("best_test_top1", res.get("final_test_top1"))
    else:
        print(f"  [WARN] no result for ema={a}")

print("Loading ablation_layertype ...")
layer_type_data = {}
for lt in ["uniform", "layeraware"]:
    runs = []
    for s in SEEDS:
        run_dir = RESULTS / "ablation_layertype" / f"{lt}_seed{s}"
        res = find_results_json(run_dir)
        if res:
            runs.append(res)
    if runs:
        vals = [r["best_test_top1"] for r in runs]
        layer_type_data[lt] = (np.mean(vals), np.std(vals), vals)

print("Loading nobn_ablation ...")
nobn_data = load_method_results(RESULTS / "nobn_ablation")
nobn_stats = compute_stats(nobn_data)

print("Loading alignment_diagnostic.json ...")
align_diag = None
align_path = FULL / "alignment_diagnostic.json"
if align_path.exists():
    with open(align_path) as f:
        align_diag = json.load(f)

print("Loading EqWD diagnostics_layers.json (ResNet-20 seed42) ...")
diag_layers = None
diag_path = FULL / "cifar100_resnet20" / "EqWD_seed42" / "diagnostics" / "diagnostics_layers.json"
if diag_path.exists():
    with open(diag_path) as f:
        diag_layers = json.load(f)
    print(f"  Loaded {len(diag_layers)} layers, {len(list(diag_layers.values())[0])} steps each")

print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════
# 3.  FIGURE 1: Methods comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating cifar_methods_comparison.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
configs = [
    ("CIFAR-100 / ResNet-20", resnet20_stats),
    ("CIFAR-100 / VGG16-BN", vgg16_stats),
]

for ax, (title, stats) in zip(axes, configs):
    present = [m for m in METHODS if m in stats]
    means = [stats[m][0] for m in present]
    stds  = [stats[m][1] for m in present]
    colors = [METHOD_COLORS[m] for m in present]

    bars = ax.bar(present, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlabel("Method")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Annotate values
    for bar, m_val, s_val in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s_val + 0.15,
                f"{m_val:.2f}", ha="center", va="bottom", fontsize=9)

    # Set y-axis range for readability
    if means:
        y_min = min(means) - max(stds) - 2
        y_max = max(means) + max(stds) + 1.5
        ax.set_ylim(y_min, y_max)

plt.suptitle("CIFAR-100: Method Comparison (3 seeds, mean +/- std)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "cifar_methods_comparison.png")
plt.close()
print(f"  Saved: {FIG_DIR / 'cifar_methods_comparison.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  FIGURE 2: Beta sensitivity
# ═══════════════════════════════════════════════════════════════════════════
print("Generating ablation_beta.png ...")

fig, ax = plt.subplots(figsize=(7, 4.5))
betas = sorted(beta_stats.keys())
accs  = [beta_stats[b] for b in betas]

ax.plot(betas, accs, "o-", color=METHOD_COLORS["EqWD"], markersize=8, linewidth=2, zorder=3)
for b, a in zip(betas, accs):
    ax.annotate(f"{a:.2f}", (b, a), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=10)
ax.set_xlabel(r"$\beta$ (control strength)")
ax.set_ylabel("Best Test Accuracy (%)")
ax.set_title(r"EqWD Sensitivity to $\beta$ (CIFAR-100 / ResNet-20)", fontweight="bold")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_tick_params(which="minor", size=0)
ax.set_xticks(betas)
ax.set_xticklabels([str(b) for b in betas])
ax.grid(alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "ablation_beta.png")
plt.close()
print(f"  Saved: {FIG_DIR / 'ablation_beta.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  FIGURE 3: EMA alpha sensitivity
# ═══════════════════════════════════════════════════════════════════════════
print("Generating ablation_ema.png ...")

fig, ax = plt.subplots(figsize=(7, 4.5))
alphas = sorted(ema_stats.keys())
accs_e = [ema_stats[a] for a in alphas]

ax.plot(alphas, accs_e, "s-", color=METHOD_COLORS["CAWD"], markersize=8, linewidth=2, zorder=3)
for al, a in zip(alphas, accs_e):
    ax.annotate(f"{a:.2f}", (al, a), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=10)
ax.set_xlabel(r"EMA coefficient $\alpha$")
ax.set_ylabel("Best Test Accuracy (%)")
ax.set_title(r"EqWD Sensitivity to EMA $\alpha$ (CIFAR-100 / ResNet-20)", fontweight="bold")
ax.set_xticks(alphas)
ax.grid(alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "ablation_ema.png")
plt.close()
print(f"  Saved: {FIG_DIR / 'ablation_ema.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  FIGURE 4: Layer-type ablation
# ═══════════════════════════════════════════════════════════════════════════
print("Generating ablation_layertype.png ...")

fig, ax = plt.subplots(figsize=(6, 4.5))
labels = []
means_lt = []
stds_lt = []
colors_lt = []
for lt, label, c in [("uniform", "Uniform", "#4477AA"), ("layeraware", "Layer-Aware", "#AA3377")]:
    if lt in layer_type_data:
        labels.append(label)
        means_lt.append(layer_type_data[lt][0])
        stds_lt.append(layer_type_data[lt][1])
        colors_lt.append(c)

bars = ax.bar(labels, means_lt, yerr=stds_lt, capsize=5, color=colors_lt,
              edgecolor="black", linewidth=0.6, width=0.5, zorder=3)
for bar, mv, sv in zip(bars, means_lt, stds_lt):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sv + 0.1,
            f"{mv:.2f}", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Best Test Accuracy (%)")
ax.set_title("Uniform vs Layer-Aware WD (CIFAR-100 / ResNet-20)", fontweight="bold")
ax.grid(axis="y", alpha=0.3, zorder=0)
if means_lt:
    ax.set_ylim(min(means_lt) - 2, max(means_lt) + 1.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "ablation_layertype.png")
plt.close()
print(f"  Saved: {FIG_DIR / 'ablation_layertype.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  FIGURE 5: r_t trajectories
# ═══════════════════════════════════════════════════════════════════════════
print("Generating ratio_trajectories.png ...")

if diag_layers:
    fig, ax = plt.subplots(figsize=(10, 5))
    all_layer_names = list(diag_layers.keys())
    # Pick representative layers: first, 1/4, 1/2, 3/4, last
    n_layers = len(all_layer_names)
    indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    indices = sorted(set(indices))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))

    for idx, color in zip(indices, cmap):
        lname = all_layer_names[idx]
        entries = diag_layers[lname]
        # Subsample for plotting speed
        step_size = max(1, len(entries) // 500)
        sub = entries[::step_size]
        epochs = [e["epoch"] + e["step"] / max(1, entries[-1]["step"]) * entries[-1]["epoch"]
                  if "step" in e else e.get("epoch", 0) for e in sub]
        # Just use step index for x-axis
        steps = list(range(len(sub)))
        r_ts = [e["r_t"] for e in sub]
        short_name = lname.replace(".weight", "")
        ax.plot(steps, r_ts, label=short_name, color=color, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Training Step (subsampled)")
    ax.set_ylabel(r"$r_t = \|\nabla L\| / \|w\|$")
    ax.set_title(r"EqWD $r_t$ Trajectories Across Layers (CIFAR-100 / ResNet-20)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ratio_trajectories.png")
    plt.close()
    print(f"  Saved: {FIG_DIR / 'ratio_trajectories.png'}")
else:
    print("  [SKIP] No diagnostics_layers.json found")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  FIGURE 6: WD heatmap (lambda_t per layer per epoch)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating wd_heatmap.png ...")

if diag_layers:
    all_layer_names = list(diag_layers.keys())
    # Build matrix: rows = layers, cols = epochs
    # Aggregate lambda_t by epoch (mean over steps within each epoch)
    epoch_lambda = {}  # {layer: {epoch: [lambda_t values]}}
    for lname, entries in diag_layers.items():
        epoch_lambda[lname] = defaultdict(list)
        for e in entries:
            epoch_lambda[lname][e["epoch"]].append(e["lambda_t"])

    # Get all epochs
    all_epochs = sorted(set(e["epoch"] for e in list(diag_layers.values())[0]))

    matrix = np.zeros((len(all_layer_names), len(all_epochs)))
    for i, lname in enumerate(all_layer_names):
        for j, ep in enumerate(all_epochs):
            vals = epoch_lambda[lname].get(ep, [])
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(14, 6))
    # Subsample epochs for readability
    epoch_step = max(1, len(all_epochs) // 50)
    sub_epochs = all_epochs[::epoch_step]
    sub_matrix = matrix[:, ::epoch_step]

    im = ax.imshow(sub_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, label=r"$\lambda_t$ (weight decay)")

    short_names = [n.replace(".weight", "") for n in all_layer_names]
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8)

    # X tick labels (epoch numbers)
    n_xticks = min(15, len(sub_epochs))
    xtick_step = max(1, len(sub_epochs) // n_xticks)
    ax.set_xticks(range(0, len(sub_epochs), xtick_step))
    ax.set_xticklabels([str(sub_epochs[i]) for i in range(0, len(sub_epochs), xtick_step)], fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    ax.set_title(r"EqWD Per-Layer $\lambda_t$ Across Training (CIFAR-100 / ResNet-20)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wd_heatmap.png")
    plt.close()
    print(f"  Saved: {FIG_DIR / 'wd_heatmap.png'}")
else:
    print("  [SKIP] No diagnostics_layers.json found")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  SUMMARY TABLES & MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════
print("Writing cifar_analysis_summary.md ...")

lines = []
lines.append("# CIFAR Analysis Summary\n")
lines.append(f"Generated automatically.\n")

# Table 1: Main comparison
lines.append("## Main Results: Mean +/- Std Test Accuracy (%)\n")
lines.append("| Method | CIFAR-100 / ResNet-20 | CIFAR-100 / VGG16-BN |")
lines.append("|--------|----------------------|---------------------|")
for m in METHODS:
    r20 = resnet20_stats.get(m)
    vgg = vgg16_stats.get(m)
    r20_str = f"{r20[0]:.2f} +/- {r20[1]:.2f}" if r20 else "N/A"
    vgg_str = f"{vgg[0]:.2f} +/- {vgg[1]:.2f}" if vgg else "N/A"
    bold_r20 = f"**{r20_str}**" if r20 and m == "EqWD" else r20_str
    bold_vgg = f"**{vgg_str}**" if vgg and m == "EqWD" else vgg_str
    lines.append(f"| {m} | {bold_r20} | {bold_vgg} |")
lines.append("")

# Best method per arch
for label, stats in [("ResNet-20", resnet20_stats), ("VGG16-BN", vgg16_stats)]:
    if stats:
        best = max(stats.items(), key=lambda x: x[1][0])
        lines.append(f"**Best on {label}**: {best[0]} ({best[1][0]:.2f}%)\n")

# Table 2: Beta ablation
lines.append("## Ablation: Beta Sensitivity\n")
lines.append("| Beta | Best Test Acc (%) |")
lines.append("|------|-------------------|")
for b in sorted(beta_stats.keys()):
    lines.append(f"| {b} | {beta_stats[b]:.2f} |")
lines.append("")

# Table 3: EMA ablation
lines.append("## Ablation: EMA Alpha Sensitivity\n")
lines.append("| Alpha | Best Test Acc (%) |")
lines.append("|-------|-------------------|")
for a in sorted(ema_stats.keys()):
    lines.append(f"| {a} | {ema_stats[a]:.2f} |")
lines.append("")

# Table 4: Layer-type ablation
lines.append("## Ablation: Uniform vs Layer-Aware\n")
lines.append("| Type | Mean +/- Std (%) |")
lines.append("|------|-----------------|")
for lt in ["uniform", "layeraware"]:
    if lt in layer_type_data:
        m, s, _ = layer_type_data[lt]
        lines.append(f"| {lt} | {m:.2f} +/- {s:.2f} |")
lines.append("")

# Table 5: NoBN ablation
lines.append("## Ablation: VGG16 Without Batch Normalization\n")
lines.append("| Method | Mean +/- Std (%) |")
lines.append("|--------|-----------------|")
for m in METHODS:
    if m in nobn_stats:
        mean, std, _ = nobn_stats[m]
        lines.append(f"| {m} | {mean:.2f} +/- {std:.2f} |")
lines.append("")

# Figures listing
lines.append("## Figures\n")
for fname in ["cifar_methods_comparison.png", "ablation_beta.png", "ablation_ema.png",
              "ablation_layertype.png", "ratio_trajectories.png", "wd_heatmap.png"]:
    fpath = FIG_DIR / fname
    status = "OK" if fpath.exists() else "MISSING"
    lines.append(f"- `{fname}`: {status}")
lines.append("")

# Alignment diagnostic summary
if align_diag and "results" in align_diag:
    lines.append("## Alignment Diagnostic (H3: delta_hat informativeness)\n")
    for arch_key, layers in align_diag["results"].items():
        lines.append(f"### {arch_key}\n")
        lines.append("| Layer | Residual Var Ratio | MI(delta, g|w) |")
        lines.append("|-------|-------------------|----------------|")
        for lname, ldata in list(layers.items())[:10]:
            ais = ldata.get("ais", {})
            rvr = ais.get("residual_variance_ratio", "N/A")
            mi = ais.get("mi_delta_gw", "N/A")
            rvr_s = f"{rvr:.4f}" if isinstance(rvr, float) else str(rvr)
            mi_s = f"{mi:.4f}" if isinstance(mi, float) else str(mi)
            short = lname.replace(".weight", "")
            lines.append(f"| {short} | {rvr_s} | {mi_s} |")
        lines.append("")

summary_path = FULL / "cifar_analysis_summary.md"
with open(summary_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved: {summary_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 10.  DONE MARKER
# ═══════════════════════════════════════════════════════════════════════════
done_path = RESULTS / "cifar_analysis_DONE"
with open(done_path, "w") as f:
    f.write("DONE\n")
print(f"  Saved: {done_path}")

print("\n" + "=" * 60)
print("CIFAR ANALYSIS COMPLETE")
print("=" * 60)
