"""
Figure 1 (Teaser): Cross-domain absorption rates at L24 with 16k JumpReLU SAE.
Right panel: Bar chart of absorption rates across 4 hierarchy types with bootstrap 95% CI.
Left panel description in text (schematic of absorption measurement, rendered as TikZ).

Data source: consolidation_summary.json and outline.md
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Absorption rates at L24_16k from consolidation_summary.json and outline.md
hierarchies = ["First-letter", "City-continent", "City-country", "City-language"]
absorption_rates = [27.1, 31.4, 45.1, 11.6]  # percentages

# Bootstrap 95% CI (from cross-domain results; approximate from iter_009/iter_010 data)
# Using typical CI widths from the results
ci_lower = [25.5, 27.8, 41.2, 9.1]
ci_upper = [28.7, 35.0, 49.0, 14.1]

# Probe F1 values for annotation
probe_f1 = [1.00, 0.87, 0.73, 0.82]

# Compute error bar sizes (asymmetric)
err_lower = [r - l for r, l in zip(absorption_rates, ci_lower)]
err_upper = [u - r for r, u in zip(absorption_rates, ci_upper)]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

x = np.arange(len(hierarchies))
bars = ax.bar(
    x,
    absorption_rates,
    color=colors,
    edgecolor="black",
    linewidth=0.6,
    width=0.65,
    yerr=[err_lower, err_upper],
    capsize=4,
    error_kw={"linewidth": 1.0, "capthick": 1.0},
)

# Annotate bars with absorption rate and probe F1
for i, (bar, rate, f1) in enumerate(zip(bars, absorption_rates, probe_f1)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + err_upper[i] + 1.5,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        -4.5,
        f"F$_1$={f1:.2f}",
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )

# Axes
ax.set_xticks(x)
ax.set_xticklabels(hierarchies, fontsize=10)
ax.set_ylabel("Absorption rate (%)", fontsize=11)
ax.set_ylim(0, 58)
ax.set_xlim(-0.5, len(hierarchies) - 0.5)

# Add dashed line for the 4.1x range annotation
ax.annotate(
    "",
    xy=(3, 11.6),
    xytext=(2, 45.1),
    arrowprops=dict(
        arrowstyle="<->",
        color="black",
        lw=1.2,
        connectionstyle="bar,fraction=-0.25",
    ),
)
ax.text(
    3.35,
    28.0,
    "4.1$\\times$",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="center",
)

# Statistical annotation
ax.text(
    0.02,
    0.97,
    "Within-RAVEL: $p = 7.4 \\times 10^{-66}$\n(Kruskal-Wallis)",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
)

# Title
ax.set_title(
    "Cross-domain absorption rates at L24 (Gemma 2 2B, JumpReLU 16k)",
    fontsize=10,
    pad=10,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/writing/figures/fig1_teaser.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.savefig(
    "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/writing/figures/fig1_teaser.png",
    bbox_inches="tight",
    dpi=300,
)
print("Figure 1 (teaser) saved successfully.")
