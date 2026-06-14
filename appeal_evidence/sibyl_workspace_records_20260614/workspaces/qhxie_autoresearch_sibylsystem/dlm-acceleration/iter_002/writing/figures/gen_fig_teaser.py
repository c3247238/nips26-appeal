"""
Figure 1: Speed-Quality Landscape Teaser
Scatter plot showing individual methods and compositions on speedup vs accuracy retention axes.
Pareto frontier overlay highlights non-dominated points.
Data source: iter_002 verified experiment results.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# --- Data from iter_002 verified results (GSM8K, combined metric where applicable) ---
# Format: (speedup, accuracy_retention, label, category)
# Categories: "baseline", "kv_cache", "step_sched", "ar_guide", "pairwise", "threeway"

data = [
    # Baseline
    (1.0, 1.0, "Baseline\n(64-step)", "baseline"),

    # M1 (Entropy-Based KV Caching)
    (1.16, 0.945, r"M1 ($\eta$=0.5)", "kv_cache"),
    (1.25, 0.88, r"M1 ($\eta$=1.0)", "kv_cache"),
    (1.37, 0.672, r"M1 ($\eta$=2.0)", "kv_cache"),

    # IGSD (Step Distillation)
    (1.71, 0.678, "IGSD\n(td=32)", "step_sched"),
    (2.81, 0.582, "IGSD\n(td=16)", "step_sched"),

    # M3 (AR-Guided)
    (1.65, 1.025, "M3\n(gw=0.3)", "ar_guide"),
    (1.65, 1.039, "M3\n(gw=0.7)", "ar_guide"),

    # Pairwise compositions
    (2.75, 0.589, "M1+IGSD", "pairwise"),
    (2.72, 0.582, "M3+IGSD", "pairwise"),
    (0.86, 1.025, "M1+M3", "pairwise_bad"),

    # Three-way compositions
    (1.71, 0.627, "M1+IGSD\n+M3$_{\\mathrm{off}}$", "threeway"),
    (1.68, 0.627, "M1+IGSD\n+M3 gw=0.3", "threeway_bad"),
]

# Visual config per category
style = {
    "baseline":      dict(color="#333333", marker="*",  s=200, zorder=10, edgecolors="k", linewidths=0.8),
    "kv_cache":      dict(color="#2166ac", marker="o",  s=80,  zorder=5,  edgecolors="k", linewidths=0.5),
    "step_sched":    dict(color="#e08214", marker="s",  s=80,  zorder=5,  edgecolors="k", linewidths=0.5),
    "ar_guide":      dict(color="#1b7837", marker="^",  s=80,  zorder=5,  edgecolors="k", linewidths=0.5),
    "pairwise":      dict(color="#b2182b", marker="D",  s=100, zorder=7,  edgecolors="k", linewidths=0.8),
    "pairwise_bad":  dict(color="#b2182b", marker="D",  s=100, zorder=7,  edgecolors="k", linewidths=0.8, alpha=0.5),
    "threeway":      dict(color="#7b3294", marker="P",  s=120, zorder=8,  edgecolors="k", linewidths=0.8),
    "threeway_bad":  dict(color="#7b3294", marker="P",  s=120, zorder=8,  edgecolors="k", linewidths=0.8, alpha=0.5),
}

# Legend labels
legend_labels = {
    "baseline":     "Baseline (64-step)",
    "kv_cache":     "M1: KV caching",
    "step_sched":   "IGSD: step distillation",
    "ar_guide":     "M3: AR-guided",
    "pairwise":     "Pairwise composition",
    "threeway":     "Three-way composition",
}

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot each category
plotted_cats = set()
for sx, sy, lbl, cat in data:
    base_cat = cat.replace("_bad", "")
    kw = {k: v for k, v in style[cat].items()}
    if base_cat not in plotted_cats and cat in legend_labels:
        kw["label"] = legend_labels[base_cat]
        plotted_cats.add(base_cat)
    elif base_cat not in plotted_cats and base_cat in legend_labels:
        kw["label"] = legend_labels[base_cat]
        plotted_cats.add(base_cat)
    ax.scatter(sx, sy * 100, **kw)

    # Annotation offsets
    offsets = {
        "Baseline\n(64-step)":        (-35, 8),
        r"M1 ($\eta$=0.5)":          (5, 5),
        r"M1 ($\eta$=1.0)":          (5, -12),
        r"M1 ($\eta$=2.0)":          (5, -12),
        "IGSD\n(td=32)":             (5, 5),
        "IGSD\n(td=16)":             (5, -12),
        "M3\n(gw=0.3)":              (-55, -15),
        "M3\n(gw=0.7)":              (5, 5),
        "M1+IGSD":                   (5, 5),
        "M3+IGSD":                   (5, -12),
        "M1+M3":                     (5, 5),
        "M1+IGSD\n+M3$_{\\mathrm{off}}$": (-10, -18),
        "M1+IGSD\n+M3 gw=0.3":      (-10, 8),
    }
    ofs = offsets.get(lbl, (5, 5))
    fontsize = 6.5
    ax.annotate(lbl, (sx, sy * 100), textcoords="offset points", xytext=ofs,
                fontsize=fontsize, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.7, lw=0.3))

# Pareto frontier (non-dominated points: higher speedup AND higher AccRet)
pareto_candidates = [(sx, sy) for sx, sy, _, cat in data if "bad" not in cat]
# Sort by speedup
pareto_candidates.sort(key=lambda p: p[0])
# Extract Pareto front
pareto = []
max_acc = -1
for sx, sy in sorted(pareto_candidates, key=lambda p: p[0]):
    if sy >= max_acc:
        pareto.append((sx, sy))
        max_acc = sy
# Also need to sort differently: we want non-dominated set
# Non-dominated: no other point has both higher speedup and higher accret
pts = [(sx, sy) for sx, sy, _, cat in data if "bad" not in cat]
pareto_front = []
for px, py in pts:
    dominated = False
    for qx, qy in pts:
        if qx >= px and qy >= py and (qx > px or qy > py):
            dominated = True
            break
    if not dominated:
        pareto_front.append((px, py))
pareto_front.sort(key=lambda p: p[0])

if pareto_front:
    px, py = zip(*pareto_front)
    ax.plot([p for p in px], [p * 100 for p in py], '--', color='#b2182b', alpha=0.6, linewidth=1.2, label="Pareto frontier")

# Reference lines
ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Shade interference zone
ax.axhspan(0, 65, alpha=0.03, color='red')
ax.text(0.55, 55, "Low quality\nretention zone", fontsize=7, color='#999', ha='left', style='italic')

ax.set_xlabel("Speedup (relative to 64-step baseline)")
ax.set_ylabel("Accuracy Retention (%)")
ax.set_xlim(0.5, 3.2)
ax.set_ylim(50, 112)

ax.legend(loc="upper right", framealpha=0.9, edgecolor='gray', fancybox=False,
          ncol=2, columnspacing=1.0, handletextpad=0.4)

plt.tight_layout()
out_path = Path(__file__).parent / "fig_teaser.pdf"
fig.savefig(out_path, format="pdf")
print(f"Saved: {out_path}")
plt.close()
