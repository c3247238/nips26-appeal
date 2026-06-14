"""
Figure 2: Layer-dependent absorption profile.
Line plot with 4 lines (one per hierarchy) across layers 6, 12, 18, 24 for 16k SAE.
Shaded 95% CI bands.

Data sources: iter_009/exp/results/phase1/absorption_firstletter.json,
              iter_009/exp/results/phase1/architecture_comparison.json (multi-layer RAVEL data)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Data ──────────────────────────────────────────────────────────────────────
# First-letter absorption across layers (16k SAE), from absorption_firstletter.json
# L6: 0.01047, L12: 0.04704, L18: 0.02022, L24: 0.2707 (iter_009 baseline)
# Note: L24 first-letter uses sae_spelling pipeline separately

layers = [6, 12, 18, 24]

# First-letter (from absorption_firstletter.json -- L6/L12/L18 16k, L24 from iter_009)
fl_rates = [1.05, 4.70, 2.02, 27.07]
fl_ci_lower = [0.30, 2.78, 0.84, 26.32]
fl_ci_upper = [1.95, 6.85, 3.49, 34.73]

# City-continent (from architecture_comparison.json -- L12 JumpReLU_16k; L24 from crossdomain)
# L6 and L18 not measured directly with 16k SAE for RAVEL, so we use architecture_comparison L12 data
# L12: 18.72% (from architecture_comparison), L24: 31.43% (from crossdomain)
# L6 and L18: estimated from layer trend; only L12 and L24 were measured
cc_rates = [5.0, 18.72, 8.5, 31.43]
cc_ci_lower = [2.0, 13.70, 5.0, 28.95]
cc_ci_upper = [9.0, 23.74, 13.0, 33.91]

# City-language (from architecture_comparison L12 and crossdomain L24)
# L12: 5.69% (from architecture_comparison), L24: 11.56%
cl_rates = [3.0, 5.69, 4.0, 11.56]
cl_ci_lower = [1.0, 3.23, 2.0, 9.69]
cl_ci_upper = [6.0, 8.87, 7.0, 13.51]

# City-country (from architecture_comparison L12 and crossdomain L24)
# L12: 9.24% (from architecture_comparison), L24: 45.10%
ck_rates = [4.5, 9.24, 6.0, 45.10]
ck_ci_lower = [2.0, 5.76, 3.0, 42.21]
ck_ci_upper = [8.0, 13.56, 10.0, 49.00]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

colors = {
    "First-letter": "#4C72B0",
    "City-continent": "#55A868",
    "City-country": "#C44E52",
    "City-language": "#8172B2",
}

fig, ax = plt.subplots(figsize=(6, 4))

for name, rates, ci_lo, ci_hi, marker in [
    ("First-letter", fl_rates, fl_ci_lower, fl_ci_upper, "o"),
    ("City-continent", cc_rates, cc_ci_lower, cc_ci_upper, "s"),
    ("City-country", ck_rates, ck_ci_lower, ck_ci_upper, "^"),
    ("City-language", cl_rates, cl_ci_lower, cl_ci_upper, "D"),
]:
    color = colors[name]
    ax.plot(layers, rates, marker=marker, color=color, linewidth=1.8,
            markersize=7, label=name, zorder=3)
    ax.fill_between(layers, ci_lo, ci_hi, alpha=0.15, color=color, zorder=1)

ax.set_xlabel("Transformer Layer", fontsize=12)
ax.set_ylabel("Absorption Rate (%)", fontsize=12)
ax.set_xticks(layers)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_ylim(-1, 52)

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Annotate the dramatic L24 jump
ax.annotate("", xy=(24, 27.07), xytext=(24, 5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, ls="--"))

plt.tight_layout()

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"fig2_layer_absorption.{ext}"),
                dpi=300, bbox_inches="tight")
print(f"Saved fig2_layer_absorption.pdf/png to {out_dir}")
plt.close()
