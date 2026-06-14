"""
Figure 5: Probe degradation ablation curve.
X-axis: probe F1. Y-axis: absorption rate (%).
7 first-letter degradation points with linear regression (R^2=0.777).
Quadratic fit shown as dashed (R^2=0.942).
3 RAVEL hierarchy points overlaid at their actual F1 levels.

Data source: iter_010/exp/results/phase1/probe_degradation.json
             current/exp/results/phase0/table3_ci_correction_report.json (corrected CIs)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Data from probe_degradation.json ──────────────────────────────────────────
# First-letter probe degradation (7 levels, FULL mode, per-token)
fl_f1 = [0.685, 0.754, 0.789, 0.846, 0.904, 0.951, 0.999]
fl_abs = [36.1, 35.3, 34.4, 33.6, 32.4, 28.9, 21.6]

# Seed standard deviations from noise_sensitivity
fl_seed_std = [1.9, 1.1, 4.2, 5.5, 3.6, 4.3, 0.0]

# RAVEL reference points (from absorption_crossdomain.json)
ravel_points = {
    "City-continent": {"f1": 0.871, "abs": 31.4, "color": "#55A868", "marker": "s"},
    "City-country":   {"f1": 0.726, "abs": 45.1, "color": "#C44E52", "marker": "^"},
    "City-language":  {"f1": 0.818, "abs": 11.6, "color": "#8172B2", "marker": "D"},
}

# Linear regression from probe_degradation.json
slope = -0.3978
intercept = 0.6544
r2_lin = 0.777

# Quadratic fit
quad_coeffs = [-2.0215, 3.0158, -0.7648]
r2_quad = 0.942

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# First-letter degradation points with seed-std error bars
ax.errorbar(fl_f1, fl_abs, yerr=fl_seed_std, fmt="o", color="#4C72B0",
            markersize=8, capsize=4, capthick=1.2, linewidth=1.2,
            label="First-letter degradation", zorder=4)

# Linear regression line
x_line = np.linspace(0.65, 1.02, 100)
y_lin = (slope * x_line + intercept) * 100
ax.plot(x_line, y_lin, "--", color="#4C72B0", alpha=0.6, linewidth=1.5,
        label=f"Linear fit ($R^2={r2_lin:.3f}$, $p=0.009$)")

# Quadratic fit
y_quad = (quad_coeffs[0] * x_line**2 + quad_coeffs[1] * x_line + quad_coeffs[2]) * 100
ax.plot(x_line, y_quad, ":", color="#4C72B0", alpha=0.4, linewidth=1.5,
        label=f"Quadratic fit ($R^2={r2_quad:.3f}$)")

# RAVEL reference points
for name, d in ravel_points.items():
    ax.scatter(d["f1"], d["abs"], marker=d["marker"], color=d["color"],
               s=120, edgecolors="black", linewidths=0.8, zorder=5,
               label=name)

# Annotate deltas
ax.annotate(r"$\Delta$=+0.6 pp",
            xy=(0.871, 31.4), xytext=(0.91, 35),
            fontsize=8, color="#55A868",
            arrowprops=dict(arrowstyle="->", color="#55A868", lw=0.8))

ax.annotate("$\\Delta$=$-$21.3 pp\n(genuine outlier)",
            xy=(0.818, 11.6), xytext=(0.72, 8),
            fontsize=8, color="#8172B2", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8172B2", lw=0.8))

ax.annotate(r"$\Delta$=+8.5 pp",
            xy=(0.726, 45.1), xytext=(0.67, 48),
            fontsize=8, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8))

ax.set_xlabel("Probe $F_1$", fontsize=12)
ax.set_ylabel("Absorption Rate (%)", fontsize=12)
ax.set_xlim(0.65, 1.02)
ax.set_ylim(0, 55)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"fig5_probe_degradation.{ext}"),
                dpi=300, bbox_inches="tight")
print(f"Saved fig5_probe_degradation.pdf/png to {out_dir}")
plt.close()
