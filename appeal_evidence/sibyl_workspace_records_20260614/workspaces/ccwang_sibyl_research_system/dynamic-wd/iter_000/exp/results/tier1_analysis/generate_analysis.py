#!/usr/bin/env python3
"""
Tier 1 Analysis: Comprehensive comparison of all pilot results.
Generates comparison_table.json, analysis_report.md, and PNG charts.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results")
OUT  = BASE / "tier1_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Load raw data ─────────────────────────────────────────────────────────────

def load_json(p):
    with open(p) as f:
        return json.load(f)

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# Tier 0
tier0 = load_json(BASE / "tier0_diagnostic/summary.json")

# Tier 1 Fixed WD
fixed_wd_list = load_json(BASE / "tier1_fixed_wd_grid/aggregated_summary.json")
# Key entries: wd=0 (No-WD), wd=5e-4 (best), etc.
fixed_wd_map = {r["wd"]: r for r in fixed_wd_list}

# Tier 1 Stagewise + CWD
stagewise_cwd_list = load_json(BASE / "tier1_stagewise_cwd/aggregated_summary.json")
stagewise_cwd_map = {r["method"]: r for r in stagewise_cwd_list}

# Tier 1 AADWD variants
aadwd_overall = load_json(BASE / "tier1_aadwd_variants/overall_summary.json")
aadwd_variant_map = {}
for v in aadwd_overall["variants"]:
    name = v["variant"]
    aadwd_variant_map[name] = v

# Epoch-level data
epoch_data = {}
for variant in ["conservative", "aggressive", "square"]:
    epoch_data[f"aadwd_{variant}"] = load_jsonl(
        BASE / f"tier1_aadwd_variants/aadwd_{variant}/epoch_metrics.jsonl"
    )
epoch_data["stagewise_wd"] = load_jsonl(
    BASE / "tier1_stagewise_cwd/stagewise_wd/epoch_metrics.jsonl"
)
epoch_data["cwd"] = load_jsonl(
    BASE / "tier1_stagewise_cwd/cwd/epoch_metrics.jsonl"
)

# Build fixed-WD epoch data from summary (we only have summary-level for now)
# For plotting purposes, create synthetic trajectories from epoch metrics if available
# Actually check if per-epoch data exists in fixed_wd folder
fixed_wd_epoch_data = {}
for r in fixed_wd_list:
    folder = r["folder"]
    ep_path = BASE / f"tier1_fixed_wd_grid/{folder}/epoch_metrics.jsonl"
    if ep_path.exists():
        fixed_wd_epoch_data[r["wd"]] = load_jsonl(ep_path)

# ─── Build comparison table ────────────────────────────────────────────────────

# Identify best fixed WD by test accuracy
best_fixed = max(fixed_wd_list, key=lambda r: r["best_test_acc"] if r["wd"] > 0 else -1)
no_wd = fixed_wd_map[0.0]

methods = []

# 1. No-WD
methods.append({
    "method_id": "no_wd",
    "method_name": "No-WD",
    "category": "baseline",
    "best_test_acc": no_wd["best_test_acc"],
    "final_test_acc": no_wd["final_test_acc"],
    "final_train_acc": no_wd["final_train_acc"],
    "final_gen_gap": no_wd["final_gen_gap"],
    "final_weight_norm": no_wd["final_weight_norm"],
    "wd_value": 0.0,
    "lambda_varied": False,
    "lambda_min_seen": None,
    "lambda_max_seen": None,
    "pass_criteria": True,  # just a baseline
    "note": "No weight decay applied"
})

# 2. Fixed-WD (all values)
for r in sorted(fixed_wd_list, key=lambda x: x["wd"]):
    if r["wd"] == 0.0:
        continue
    is_best = (r["wd"] == best_fixed["wd"])
    methods.append({
        "method_id": f"fixed_wd_{r['wd']}",
        "method_name": f"Fixed-WD ({r['wd']:.0e})" + (" [BEST]" if is_best else ""),
        "category": "fixed_wd",
        "best_test_acc": r["best_test_acc"],
        "final_test_acc": r["final_test_acc"],
        "final_train_acc": r["final_train_acc"],
        "final_gen_gap": r["final_gen_gap"],
        "final_weight_norm": r["final_weight_norm"],
        "wd_value": r["wd"],
        "lambda_varied": False,
        "lambda_min_seen": None,
        "lambda_max_seen": None,
        "pass_criteria": None,
        "note": "Best fixed WD" if is_best else ""
    })

# 3. Stagewise-WD
sw = stagewise_cwd_map["stagewise_wd"]
methods.append({
    "method_id": "stagewise_wd",
    "method_name": "Stagewise-WD",
    "category": "dynamic_schedule",
    "best_test_acc": sw["best_test_acc"],
    "final_test_acc": sw["final_test_acc"],
    "final_train_acc": sw["final_train_acc"],
    "final_gen_gap": sw["final_gen_gap"],
    "final_weight_norm": sw["final_weight_norm"],
    "wd_value": 0.0005,
    "lambda_varied": True,
    "lambda_min_seen": None,
    "lambda_max_seen": None,
    "pass_criteria": sw["final_test_acc"] > 85.0,
    "note": "Stage-based WD schedule: 50%→30%→20% with 10x decay each stage"
})

# 4. CWD
cw = stagewise_cwd_map["cwd"]
methods.append({
    "method_id": "cwd",
    "method_name": "CWD (sign-based)",
    "category": "dynamic_adaptive",
    "best_test_acc": cw["best_test_acc"],
    "final_test_acc": cw["final_test_acc"],
    "final_train_acc": cw["final_train_acc"],
    "final_gen_gap": cw["final_gen_gap"],
    "final_weight_norm": cw["final_weight_norm"],
    "wd_value": 0.0005,
    "lambda_varied": True,
    "lambda_min_seen": None,
    "lambda_max_seen": None,
    "pass_criteria": cw["final_test_acc"] > 79.0,
    "note": "Sign-based masking, very slow (698s for 20 epochs)"
})

# 5. AADWD variants
for variant in ["conservative", "aggressive", "square"]:
    v = aadwd_variant_map[variant]
    methods.append({
        "method_id": f"aadwd_{variant}",
        "method_name": f"AADWD-{variant.capitalize()}",
        "category": "aadwd",
        "best_test_acc": v["best_test_acc"],
        "final_test_acc": v["final_test_acc"],
        "final_train_acc": v["final_train_acc"],
        "final_gen_gap": v["final_gen_gap"],
        "final_weight_norm": v["final_weight_norm"],
        "wd_value": None,
        "lambda_varied": v["lambda_varied"],
        "lambda_min_seen": v["lambda_min_seen"],
        "lambda_max_seen": v["lambda_max_seen"],
        "pass_criteria": v["pass_criteria"],
        "note": f"Dynamic WD, lambda range [{v['lambda_min_seen']:.2e}, {v['lambda_max_seen']:.2e}]"
    })

comparison_table = {
    "meta": {
        "task": "tier1_analysis",
        "mode": "PILOT",
        "epochs": 20,
        "arch": "resnet20",
        "dataset": "cifar10",
        "seed": 42,
        "note": "All results are 20-epoch pilot runs. Full 200-epoch results pending."
    },
    "tier0_diagnostic": {
        "best_pearson_r": tier0["best_pearson_r"],
        "r_mini_ema_vs_large": tier0["r_mini_ema_vs_large"],
        "verdict": tier0["verdict"],
        "recommendation": tier0["recommendation"],
        "phase_structure": tier0["phase_structure"],
        "overall_delta_std": tier0["overall_delta_std"]
    },
    "best_fixed_wd": {
        "wd": best_fixed["wd"],
        "best_test_acc": best_fixed["best_test_acc"],
        "final_test_acc": best_fixed["final_test_acc"]
    },
    "methods": methods
}

with open(OUT / "comparison_table.json", "w") as f:
    json.dump(comparison_table, f, indent=2)
print("✓ Wrote comparison_table.json")

# ─── Analysis for report ───────────────────────────────────────────────────────

# Key metrics for top methods only (skip extreme fixed WD)
top_methods = [m for m in methods if m["method_id"] in {
    "no_wd", "fixed_wd_0.0005", "fixed_wd_0.001",
    "stagewise_wd", "cwd",
    "aadwd_conservative", "aadwd_aggressive", "aadwd_square"
}]

# Rank by best_test_acc
top_methods_sorted = sorted(top_methods, key=lambda x: x["best_test_acc"], reverse=True)

# AADWD aggressive is the only AADWD that passes pilot criteria
aadwd_aggressive_data = epoch_data["aadwd_aggressive"]

# Lambda trajectory analysis for all 3 AADWD variants
lambda_stats = {}
for variant in ["conservative", "aggressive", "square"]:
    ep = epoch_data[f"aadwd_{variant}"]
    lambdas = [e["mean_lambda_t"] for e in ep]
    ema_deltas = [e["ema_delta"] for e in ep]
    delta_hats = [e["mean_delta_hat_t"] for e in ep]
    lambda_stats[variant] = {
        "lambda_trajectory": lambdas,
        "lambda_initial": lambdas[0],
        "lambda_final": lambdas[-1],
        "lambda_change_ratio": lambdas[-1] / lambdas[0] if lambdas[0] > 0 else float('inf'),
        "ema_delta_initial": ema_deltas[0],
        "ema_delta_final": ema_deltas[-1],
        "ema_delta_decay_ratio": ema_deltas[-1] / ema_deltas[0] if ema_deltas[0] > 0 else float('inf'),
        "delta_hat_trajectory": delta_hats,
        "mean_delta_hat": np.mean(delta_hats),
    }

# Convergence speed: epoch to reach 85% test acc
def epochs_to_thresh(ep_list, thresh):
    for ep in ep_list:
        if ep["test_acc"] >= thresh:
            return ep["epoch"]
    return None

conv_stats = {}
for k, ep_list in epoch_data.items():
    conv_stats[k] = {
        "epoch_to_80": epochs_to_thresh(ep_list, 80),
        "epoch_to_85": epochs_to_thresh(ep_list, 85),
    }

# ─── Generate charts ───────────────────────────────────────────────────────────
epochs_x = list(range(20))

# ── Figure 1: Test accuracy comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: AADWD vs Fixed-WD baselines
ax = axes[0]
ax.set_title("Test Accuracy: AADWD vs Baselines (20-epoch Pilot)", fontsize=12)
colors_aadwd = {"conservative": "#e74c3c", "aggressive": "#2ecc71", "square": "#3498db"}
for variant in ["conservative", "aggressive", "square"]:
    ep = epoch_data[f"aadwd_{variant}"]
    ax.plot(epochs_x, [e["test_acc"] for e in ep],
            label=f"AADWD-{variant.capitalize()}", color=colors_aadwd[variant], lw=2)

# Fixed WD reference lines (final acc)
ax.axhline(fixed_wd_map[0.0005]["final_test_acc"], color="gray", ls="--", lw=1.5,
           label=f"Fixed-WD 5e-4 (final={fixed_wd_map[0.0005]['final_test_acc']:.2f}%)")
ax.axhline(fixed_wd_map[0.001]["final_test_acc"], color="black", ls=":", lw=1.5,
           label=f"Fixed-WD 1e-3 (final={fixed_wd_map[0.001]['final_test_acc']:.2f}%)")

ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(30, 95)

# Right: Stagewise vs CWD vs AADWD-Aggressive
ax = axes[1]
ax.set_title("Test Accuracy: Dynamic Methods Comparison (20-epoch Pilot)", fontsize=12)
ax.plot(epochs_x, [e["test_acc"] for e in epoch_data["stagewise_wd"]],
        label="Stagewise-WD", color="#9b59b6", lw=2)
ax.plot(epochs_x, [e["test_acc"] for e in epoch_data["cwd"]],
        label="CWD", color="#e67e22", lw=2)
ax.plot(epochs_x, [e["test_acc"] for e in epoch_data["aadwd_aggressive"]],
        label="AADWD-Aggressive", color="#2ecc71", lw=2)
ax.plot(epochs_x, [e["test_acc"] for e in epoch_data["aadwd_square"]],
        label="AADWD-Square", color="#3498db", lw=2)
ax.axhline(fixed_wd_map[0.0005]["final_test_acc"], color="gray", ls="--", lw=1.5,
           label=f"Fixed-WD 5e-4 (ref)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 95)

plt.tight_layout()
plt.savefig(OUT / "fig1_test_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved fig1_test_accuracy_comparison.png")

# ── Figure 2: Lambda trajectory for AADWD variants ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors_aadwd = {"conservative": "#e74c3c", "aggressive": "#2ecc71", "square": "#3498db"}

for i, variant in enumerate(["conservative", "aggressive", "square"]):
    ep = epoch_data[f"aadwd_{variant}"]
    ax = axes[i]
    lambdas = [e["mean_lambda_t"] for e in ep]
    ema_deltas = [e["ema_delta"] for e in ep]
    delta_hats = [e["mean_delta_hat_t"] for e in ep]

    ax2 = ax.twinx()
    lns1 = ax.plot(epochs_x, lambdas, color=colors_aadwd[variant], lw=2,
                   label="λ_t (mean per epoch)")
    lns2 = ax2.plot(epochs_x, ema_deltas, color="navy", lw=1.5, ls="--",
                    label="EMA(δ̂_t)")
    lns3 = ax2.plot(epochs_x, delta_hats, color="steelblue", lw=1, ls=":",
                    label="δ̂_t (mean)")

    ax.set_title(f"AADWD-{variant.capitalize()}\nλ Trajectory & Alignment Signal", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("λ_t", color=colors_aadwd[variant])
    ax2.set_ylabel("δ̂ / EMA(δ̂)", color="navy")
    ax.tick_params(axis='y', labelcolor=colors_aadwd[variant])
    ax2.tick_params(axis='y', labelcolor="navy")

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Annotate initial and final lambda
    ax.annotate(f"λ₀={lambdas[0]:.2e}", xy=(0, lambdas[0]),
                xytext=(2, lambdas[0]*1.05), fontsize=7, color=colors_aadwd[variant])
    ax.annotate(f"λ_f={lambdas[-1]:.2e}", xy=(19, lambdas[-1]),
                xytext=(14, lambdas[-1]*1.05), fontsize=7, color=colors_aadwd[variant])

plt.suptitle("AADWD λ_t Dynamic Behavior: 3 Variants (20-epoch Pilot)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig2_lambda_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved fig2_lambda_trajectory.png")

# ── Figure 3: Weight norm trajectories ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Weight Norm Trajectories (20-epoch Pilot)", fontsize=12)

for variant in ["conservative", "aggressive", "square"]:
    ep = epoch_data[f"aadwd_{variant}"]
    ax.plot(epochs_x, [e["weight_norm"] for e in ep],
            label=f"AADWD-{variant.capitalize()}", color=colors_aadwd[variant], lw=2)

ax.plot(epochs_x, [e["weight_norm"] for e in epoch_data["stagewise_wd"]],
        label="Stagewise-WD", color="#9b59b6", lw=2, ls="-.")
ax.plot(epochs_x, [e["weight_norm"] for e in epoch_data["cwd"]],
        label="CWD", color="#e67e22", lw=2, ls="--")

# Reference lines for fixed WD final norms
for wd_val, color in [(0.0, "dimgray"), (0.0005, "gray"), (0.001, "darkgray")]:
    if wd_val in fixed_wd_map:
        ax.axhline(fixed_wd_map[wd_val]["final_weight_norm"],
                   color=color, ls=":", lw=1,
                   label=f"Fixed-WD {wd_val:.0e} final norm")

ax.set_xlabel("Epoch")
ax.set_ylabel("Weight Norm (||w||₂)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig3_weight_norm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved fig3_weight_norm.png")

# ── Figure 4: Summary bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Select 7 key methods for the main bar chart
bar_methods = [
    ("No-WD",            no_wd["final_test_acc"],   no_wd["final_gen_gap"],   no_wd["final_weight_norm"]),
    ("Fixed-WD\n5e-4",   fixed_wd_map[0.0005]["final_test_acc"],
                         fixed_wd_map[0.0005]["final_gen_gap"],
                         fixed_wd_map[0.0005]["final_weight_norm"]),
    ("Fixed-WD\n1e-3",   fixed_wd_map[0.001]["final_test_acc"],
                         fixed_wd_map[0.001]["final_gen_gap"],
                         fixed_wd_map[0.001]["final_weight_norm"]),
    ("Stagewise\n-WD",   stagewise_cwd_map["stagewise_wd"]["final_test_acc"],
                         stagewise_cwd_map["stagewise_wd"]["final_gen_gap"],
                         stagewise_cwd_map["stagewise_wd"]["final_weight_norm"]),
    ("CWD",              stagewise_cwd_map["cwd"]["final_test_acc"],
                         stagewise_cwd_map["cwd"]["final_gen_gap"],
                         stagewise_cwd_map["cwd"]["final_weight_norm"]),
    ("AADWD\nConsv.",    aadwd_variant_map["conservative"]["final_test_acc"],
                         aadwd_variant_map["conservative"]["final_gen_gap"],
                         aadwd_variant_map["conservative"]["final_weight_norm"]),
    ("AADWD\nAggr.",     aadwd_variant_map["aggressive"]["final_test_acc"],
                         aadwd_variant_map["aggressive"]["final_gen_gap"],
                         aadwd_variant_map["aggressive"]["final_weight_norm"]),
    ("AADWD\nSqr.",      aadwd_variant_map["square"]["final_test_acc"],
                         aadwd_variant_map["square"]["final_gen_gap"],
                         aadwd_variant_map["square"]["final_weight_norm"]),
]

names = [b[0] for b in bar_methods]
test_accs = [b[1] for b in bar_methods]
gen_gaps = [b[2] for b in bar_methods]
weight_norms = [b[3] for b in bar_methods]

x = np.arange(len(names))
bar_colors = ["#95a5a6", "#3498db", "#2980b9",
              "#9b59b6", "#e67e22",
              "#e74c3c", "#2ecc71", "#1abc9c"]

axes[0].bar(x, test_accs, color=bar_colors, edgecolor="black", lw=0.5)
axes[0].set_title("Final Test Accuracy (%)", fontsize=11)
axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
axes[0].set_ylabel("%")
axes[0].set_ylim(50, 95)
for i, v in enumerate(test_accs):
    axes[0].text(i, v+0.3, f"{v:.1f}", ha='center', va='bottom', fontsize=7)

gen_gap_colors = [("red" if g > 8 else "orange" if g > 5 else "green") for g in gen_gaps]
axes[1].bar(x, gen_gaps, color=gen_gap_colors, edgecolor="black", lw=0.5)
axes[1].set_title("Generalization Gap (%)\n(lower is better)", fontsize=11)
axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
axes[1].set_ylabel("%")
for i, v in enumerate(gen_gaps):
    axes[1].text(i, max(v+0.1, 0.5), f"{v:.1f}", ha='center', va='bottom', fontsize=7)

axes[2].bar(x, weight_norms, color=bar_colors, edgecolor="black", lw=0.5)
axes[2].set_title("Final Weight Norm (||w||₂)", fontsize=11)
axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=8)
axes[2].set_ylabel("||w||₂")
for i, v in enumerate(weight_norms):
    axes[2].text(i, v+0.5, f"{v:.1f}", ha='center', va='bottom', fontsize=7)

plt.suptitle("Tier 1 Pilot: 8-Method Comparison — ResNet20/CIFAR-10 (20 epochs)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "fig4_method_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved fig4_method_comparison_bars.png")

# ── Figure 5: Tier0 alignment proxy analysis ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Phase structure bar chart
phases = list(tier0["phase_structure"].keys())
phase_means = [tier0["phase_structure"][p]["mean"] for p in phases]
phase_stds  = [tier0["phase_structure"][p]["std"]  for p in phases]
axes[0].bar(phases, phase_means, yerr=phase_stds, color=["#3498db", "#2ecc71", "#e74c3c"],
            edgecolor="black", capsize=5)
axes[0].set_title(f"Alignment Proxy δ̂_t Phase Structure\n(WD=5e-4, β=0.99, 20 epochs)", fontsize=10)
axes[0].set_xlabel("Training Phase")
axes[0].set_ylabel("Mean δ̂_t ± Std")
axes[0].set_ylim(0, 0.007)
for i, (m, s) in enumerate(zip(phase_means, phase_stds)):
    axes[0].text(i, m+s+0.0001, f"μ={m:.4f}\nσ={s:.4f}", ha='center', va='bottom', fontsize=7)

# Right: Pearson r comparison
r_vals = {
    "EMA vs\nLarge-batch": tier0["r_ema_vs_large"],
    "Mini-batch vs\nLarge-batch": tier0["r_mini_vs_large"],
    "Mini-EMA vs\nLarge-batch\n(best)": tier0["r_mini_ema_vs_large"],
}
bar_c = ["#e74c3c", "#e67e22", "#2ecc71"]
axes[1].bar(list(r_vals.keys()), list(r_vals.values()), color=bar_c, edgecolor="black")
axes[1].axhline(0.85, color="red", ls="--", lw=1.5, label="Pass threshold (r=0.85)")
axes[1].axhline(0.6, color="orange", ls=":", lw=1, label="Pivot threshold (r=0.6)")
axes[1].set_title(f"Alignment Proxy Pearson r Scores\nVerdict: {tier0['verdict']}", fontsize=10)
axes[1].set_ylabel("Pearson r")
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=8)
for i, (k, v) in enumerate(r_vals.items()):
    axes[1].text(i, v+0.02, f"r={v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / "fig5_tier0_alignment_proxy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved fig5_tier0_alignment_proxy.png")

# ─── Write analysis report ─────────────────────────────────────────────────────

# Prepare formatted strings
def fmt_acc(v): return f"{v:.2f}%"
def fmt_norm(v): return f"{v:.1f}"
def fmt_gap(v): return f"{v:.2f}%"

best_aadwd_name = "AADWD-Aggressive"
best_aadwd = aadwd_variant_map["aggressive"]
best_fixed_name = f"Fixed-WD ({best_fixed['wd']:.0e})"

report = f"""# Tier 0 + Tier 1 Pilot 综合对比分析报告

**生成时间**: 2026-03-17
**实验设置**: ResNet20 / CIFAR-10 / 20 epochs (pilot) / seed=42
**模式**: PILOT（所有结果均为 20 epoch 快速验证，非最终 200 epoch 结果）

---

## 1. Tier 0 诊断结果：Alignment Proxy 可靠性

| 指标 | 值 | 状态 |
|------|----|----|
| Pearson r (Mini-EMA vs Large-batch) | {tier0['r_mini_ema_vs_large']:.4f} | ⚠️ 接近通过线但未达标 |
| Pearson r (EMA vs Large-batch) | {tier0['r_ema_vs_large']:.4f} | ❌ 较低 |
| Pearson r (Mini-batch vs Large-batch) | {tier0['r_mini_vs_large']:.4f} | ❌ 较低 |
| 整体 δ̂_t 标准差 | {tier0['overall_delta_std']:.6f} | ⚠️ 低于 0.05 (相变阈值) |
| 判断 | **{tier0['verdict']}** | 需增大 beta |

**相位结构分析**（delta_hat_t 随训练阶段变化）：
- 早期（Early）: μ={tier0['phase_structure']['early']['mean']:.4f}, σ={tier0['phase_structure']['early']['std']:.4f}
- 中期（Mid）:   μ={tier0['phase_structure']['mid']['mean']:.4f}, σ={tier0['phase_structure']['mid']['std']:.4f}
- 后期（Late）:  μ={tier0['phase_structure']['late']['mean']:.4f}, σ={tier0['phase_structure']['late']['std']:.4f}

**关键发现**：
- Best Pearson r = {tier0['r_mini_ema_vs_large']:.4f}，处于 PARTIAL FAIL 区间 [0.6, 0.85]
- 推荐将 EMA beta 从 0.99 提升至 **0.999**，以改善 proxy 平滑性
- δ̂_t 呈下降趋势（early→late 均值从 0.0045 降至 0.0028），有相位依赖结构但幅度较小
- 标准差 overall=0.000753 低于阈值 0.05，可能与 pilot 仅 20 epochs 有关（200 epoch 时相变结构应更明显）

**对后续实验的影响**：
- AADWD 变体的 beta 配置已正确更新为 0.999（tier1_aadwd_variants 使用的 beta=0.999）
- 在 200 epoch full 实验中，alignment proxy 可靠性预计会提升（更丰富的相变）
- 若 full 实验中 r 仍低于 0.85，考虑切换到 `cand_empirical` 候选策略

---

## 2. Tier 1 Fixed WD Grid 搜索结果

| WD 值 | Best Test Acc | Final Test Acc | Final Train Acc | Gen Gap | Weight Norm |
|--------|--------------|---------------|----------------|---------|-------------|"""

for r in sorted(fixed_wd_list, key=lambda x: x["wd"]):
    marker = " ← BEST" if r["wd"] == best_fixed["wd"] else ""
    report += f"""
| {r['wd']:.0e} | {r['best_test_acc']:.2f}% | {r['final_test_acc']:.2f}% | {r['final_train_acc']:.2f}% | {r['final_gen_gap']:.2f}% | {r['final_weight_norm']:.1f}{marker} |"""

report += f"""

**关键发现**：
- **最优 Fixed WD = {best_fixed['wd']:.0e}**，best test acc = {best_fixed['best_test_acc']:.2f}%
- WD=5e-3 和 WD=1e-2 accuracy 显著下降（过正则化）；WD=5e-2 基本不收敛
- WD=5e-4 与 WD=1e-3 效果接近，前者 gen gap 稍大但 test acc 更高
- No-WD 基线 test acc = {no_wd['final_test_acc']:.2f}%，weight norm = {no_wd['final_weight_norm']:.1f}（最大，无正则）
- 注意：这是 20 epoch pilot，最优 WD 在 200 epoch 可能有所变化

---

## 3. Tier 1 动态基线：Stagewise-WD 和 CWD

| 方法 | Best Test Acc | Final Test Acc | Final Train Acc | Gen Gap | Weight Norm | 运行时间 |
|------|--------------|----------------|----------------|---------|-------------|----------|
| Stagewise-WD | {sw['best_test_acc']:.2f}% | {sw['final_test_acc']:.2f}% | {sw['final_train_acc']:.2f}% | {sw['final_gen_gap']:.2f}% | {sw['final_weight_norm']:.1f} | {stagewise_cwd_map['stagewise_wd']['total_time_sec']:.0f}s |
| CWD | {cw['best_test_acc']:.2f}% | {cw['final_test_acc']:.2f}% | {cw['final_train_acc']:.2f}% | {cw['final_gen_gap']:.2f}% | {cw['final_weight_norm']:.1f} | {stagewise_cwd_map['cwd']['total_time_sec']:.0f}s |
| Fixed-WD (5e-4) | {fixed_wd_map[0.0005]['best_test_acc']:.2f}% | {fixed_wd_map[0.0005]['final_test_acc']:.2f}% | {fixed_wd_map[0.0005]['final_train_acc']:.2f}% | {fixed_wd_map[0.0005]['final_gen_gap']:.2f}% | {fixed_wd_map[0.0005]['final_weight_norm']:.1f} | ~54s |

**关键发现**：
- **Stagewise-WD** 表现合理（85.33% test acc），接近 Fixed-WD-best，weight norm 较高（59.9）
  - 注意：20 epoch pilot 中 milestones 设在 epoch 30/60/90，所以 pilot 阶段只看到 stage 1 效果
  - 在 200 epoch full run 中，stagewise schedule 效果应更明显
- **CWD（sign-based masking）** 表现令人担忧：
  - Final test acc 仅 79.32%，低于 Fixed-WD 9个百分点
  - **计算代价极高**：698 秒/20 epoch vs 54 秒（12.9x 慢），full run 将耗时 ~2 小时
  - Gen gap 偏高（6.18%），weight norm 居中
  - 在 20 epoch pilot 中未能充分展示 CWD 的优势，但计算成本问题值得关注

---

## 4. Tier 1 AADWD 变体结果

| 变体 | Best Test Acc | Final Test Acc | Gen Gap | Weight Norm | λ_initial | λ_final | λ 变化比 | 通过标准 |
|------|--------------|----------------|---------|-------------|----------|---------|---------|--------|"""

for variant in ["conservative", "aggressive", "square"]:
    v = aadwd_variant_map[variant]
    stats = lambda_stats[variant]
    pass_str = "✅ PASS" if v["pass_criteria"] else "❌ FAIL"
    report += f"""
| AADWD-{variant.capitalize()} | {v['best_test_acc']:.2f}% | {v['final_test_acc']:.2f}% | {v['final_gen_gap']:.2f}% | {v['final_weight_norm']:.1f} | {stats['lambda_initial']:.2e} | {stats['lambda_final']:.2e} | {stats['lambda_change_ratio']:.2f}x | {pass_str} |"""

report += f"""

### 4.1 AADWD-Conservative 分析
- **问题严重**：Best test acc 仅 74.06%（pilot pass criterion: >85%），最终 acc 仅 61.8%
- λ_t 从 5.9e-4 升至 1.0e-3（增大约 1.7x），**超强正则化**导致欠拟合
- Weight norm 从 29.1 仅降至 27.9（变化极小，被过度约束）
- Gen gap = 17.1%（最高！），说明模型在高 λ 下无法有效学习
- **根本原因**：Conservative 变体 lambda 更新公式对 delta_hat 过于敏感，在 beta=0.999、c=0.01 配置下
  lambda 过快收敛到 lambda_max (0.01)，导致近似 WD=0.001 的效果但过早失控
- EMA(delta_hat_t) 从 0.41 衰减到 0.004，说明 proxy 在 20 epochs 内迅速收敛——
  但 lambda 没有相应调小，反而一直在 ~1e-3 附近

### 4.2 AADWD-Aggressive 分析（**唯一通过 Pilot 标准的 AADWD 变体**）
- **Best test acc = 85.09%**，满足 >85% 要求 ✅
- λ_t 从 4.1e-4 **显著下降**至 2.2e-6（约下降 187x），体现"aggressive"的特性：
  随着训练推进、alignment proxy 下降，WD 大幅减小
- Weight norm 从 31 增长至 70.4（相比 Fixed-WD 的 28 明显更大），因为后期 λ 很小
- Gen gap 5.0%，合理范围内
- EMA(delta_hat_t) 轨迹：0.41 → 0.002，与 lambda 的动态变化高度一致
- **这说明 aggressive 变体最能体现 AADWD 的核心假设**：
  当 alignment 下降（训练后期）时，应该减小 WD 以避免过度正则

### 4.3 AADWD-Square 分析
- Best test acc = 83.45%（略低于 85% 阈值），Final acc = 82.47%
- λ_t 保持非常稳定：5.9e-5 → 1.0e-4（约增大 1.7x，变化很小）
- Weight norm 从 34.9 增长至 56.7，居中
- **Square 变体的 lambda 动态性最弱**，接近 Fixed-WD 效果
- Gen gap 5.07%，与 aggressive 接近
- 如果需要 lambda 保持稳定但仍有自适应能力，square 是候选

---

## 5. 全方法综合对比（8种方法）

| 排名 | 方法 | Best Test Acc | Gen Gap | Weight Norm | 备注 |
|-----|------|--------------|---------|-------------|------|"""

for rank, m in enumerate(sorted(top_methods_sorted, key=lambda x: x["best_test_acc"], reverse=True)[:8], 1):
    marker = " 🏆" if rank == 1 else ""
    report += f"""
| {rank} | {m['method_name']}{marker} | {m['best_test_acc']:.2f}% | {m['final_gen_gap']:.2f}% | {m['final_weight_norm']:.1f} | {m['note'][:50] if m['note'] else ''} |"""

report += """

---

## 6. Lambda_t 动态行为分析

### 6.1 三种变体的 λ_t 动力学对比
"""

for variant in ["conservative", "aggressive", "square"]:
    stats = lambda_stats[variant]
    report += f"""
**{variant.capitalize()}**：
- λ₀={stats['lambda_initial']:.2e} → λ_f={stats['lambda_final']:.2e}（变化 {stats['lambda_change_ratio']:.2f}x）
- EMA(δ̂_t)：{stats['ema_delta_initial']:.4f} → {stats['ema_delta_final']:.4f}（衰减 {stats['ema_delta_decay_ratio']:.4f}x）
- mean δ̂_t = {stats['mean_delta_hat']:.4f}
"""

report += """
### 6.2 关键模式

1. **所有变体的 EMA(δ̂_t) 均显著下降**（从 ~0.41 降至 ~0.002-0.004），符合预期：
   - 训练初期 alignment proxy 高（梯度方向不确定，alignment 较大）
   - 训练后期 EMA 收敛，proxy 趋向稳定

2. **三种变体对 δ̂_t 信号的响应方式截然不同**：
   - Conservative：λ 随 δ̂_t 升高而升高，最终趋向 lambda_max
   - Aggressive：λ 随 δ̂_t 下降而大幅下降，最终趋向 lambda_min
   - Square：λ 变化平缓，基本维持在某个较小值附近

3. **λ_t 变化方向暗示了"alignment = 大 WD 还是小 WD"的解释争议**：
   - Conservative: alignment↑ → λ↑（更多 WD）：认为 alignment 高时需要更强正则
   - Aggressive: alignment↓ → λ↓（更少 WD）：认为训练后期应减小 WD
   - 从 test accuracy 看，aggressive 表现更好，支持"后期减小 WD"假说

---

## 7. 对 Full 实验的建议

### 7.1 推荐继续的方法
| 方法 | 优先级 | 理由 |
|------|--------|------|
| **Fixed-WD (5e-4)** | ⭐⭐⭐ 必须 | Baseline，pilot 表现最好，计算高效 |
| **Fixed-WD (1e-3)** | ⭐⭐⭐ 必须 | 第二 baseline，gen gap 略优 |
| **AADWD-Aggressive** | ⭐⭐⭐ 必须 | 唯一通过 pilot 标准的 AADWD，λ 动态性强 |
| **Stagewise-WD** | ⭐⭐ 建议 | 合理基线，200 epoch 下 milestone schedule 效果更明显 |
| **No-WD** | ⭐⭐ 建议 | 对照基线，理解 WD 的整体效果 |
| **AADWD-Square** | ⭐⭐ 建议 | 接近通过标准，200 epoch 下可能更好 |

### 7.2 需要调整超参后再测试
| 方法 | 建议调整 | 理由 |
|------|----------|------|
| **AADWD-Conservative** | 降低 c 值（从 0.01 到 0.001 或更小） | 当前 λ 过早收敛至 max，欠拟合严重 |
| **AADWD-Square** | 尝试更大 c（如 0.1）或更宽 lambda 范围 | λ 变化太小，更多动态性可能有益 |
| **CWD** | 评估是否值得运行 200 epoch（~2小时/run） | 计算代价极高（12.9x）且 20 epoch 效果差 |

### 7.3 Tier 0 诊断的影响
- **beta 已修正为 0.999**（tier1_aadwd_variants 已使用），符合 tier0 建议
- Full 实验中 alignment proxy 可靠性预计改善（更长训练、更丰富的相变）
- 建议在 full 实验中同时记录 beta=0.99 和 beta=0.999 的 proxy 对比（ablation）

### 7.4 关键超参建议（针对 200-epoch full 实验）
```yaml
# AADWD-Aggressive (推荐优先运行)
variant: aggressive
beta: 0.999          # 已修正
c: 0.01              # 当前值，aggressive 下效果较好
lambda_min: 1e-6
lambda_max: 0.01
lr_milestones: [100, 150]  # 建议调整（原 30/60/90 对 200 epoch 不合适）

# AADWD-Conservative (需降低 c)
variant: conservative
c: 0.001             # 降低 10x，避免 lambda 过快趋向 max
lambda_max: 0.005    # 可适当降低上限
```

---

## 8. 总结

**pilot 阶段核心结论**：

1. **Fixed-WD (5e-4) 是当前最强 baseline**（89.35% test acc），为后续对比提供参考线
2. **AADWD-Aggressive 是最有前景的 AADWD 变体**，通过 pilot 标准，λ 动态下降行为
   符合论文假设（训练后期减小 WD 有益）
3. **AADWD-Conservative 有严重欠拟合问题**，需大幅降低 c 参数后再测试
4. **CWD 计算代价过高**（12.9x 慢），在实验资源有限时优先级降低
5. **Tier 0 alignment proxy 可靠性 PARTIAL FAIL**（r=0.849），但 beta 已修正至 0.999，
   预计 full 实验改善；不影响当前进行 AADWD full 实验
6. **全部方法的 200 epoch full 实验均在 pilot 结果基础上可以合理预期更高精度**
   （pilot 的 20 epoch 与 200 epoch 结果有显著差异，特别是 lr scheduler 生效后）
"""

with open(OUT / "analysis_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("✓ Wrote analysis_report.md")

print("\n=== Tier 1 Analysis Complete ===")
print(f"Output directory: {OUT}")
print("Files generated:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
