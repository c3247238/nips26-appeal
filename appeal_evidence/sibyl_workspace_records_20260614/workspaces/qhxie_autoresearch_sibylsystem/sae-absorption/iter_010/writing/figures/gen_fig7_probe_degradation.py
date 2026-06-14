"""
Figure 7: Probe degradation ablation curve.
X-axis: probe F1. Y-axis: absorption rate.
7 first-letter data points with linear regression line (R^2=0.777).
Quadratic fit shown as dashed curve (R^2=0.942).
3 RAVEL hierarchy points overlaid at their actual F1 levels.

Data source: current/exp/results/phase1/probe_degradation.json
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Data from probe_degradation.json ──────────────────────────────────────────
# First-letter probe degradation curve (7 levels, FULL mode)
fl_f1 = [0.6852, 0.7539, 0.7887, 0.8460, 0.9042, 0.9508, 0.9991]
fl_abs = [36.14, 35.29, 34.40, 33.59, 32.36, 28.89, 21.61]

# Seed std for error bars (from noise_sensitivity)
fl_seed_std = [1.95, 1.12, 4.18, 5.50, 3.59, 4.30, 0.0]

# RAVEL reference points
ravel_points = {
    "City-continent": {"f1": 0.871, "abs": 31.43, "color": "#55A868", "marker": "s"},
    "City-country": {"f1": 0.726, "abs": 45.10, "color": "#C44E52", "marker": "^"},
    "City-language": {"f1": 0.818, "abs": 11.56, "color": "#8172B2", "marker": "D"},
}

# Linear regression: slope=-0.3978, intercept=0.6544, R^2=0.777
slope = -0.3978
intercept = 0.6544
r2_lin = 0.777

# Quadratic fit: coefficients [-2.0215, 3.0158, -0.7648], R^2=0.942
quad_coeffs = [-2.0215, 3.0158, -0.7648]
r2_quad = 0.942

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Plot first-letter degradation points
ax.errorbar(fl_f1, fl_abs, yerr=fl_seed_std, fmt="o", color="#4C72B0",
            markersize=8, capsize=4, capthick=1.2, linewidth=1.2,
            label=f"First-letter degradation", zorder=4)

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

# Annotate deltas for RAVEL points
# City-continent: near curve
predicted_cc = (slope * 0.871 + intercept) * 100
ax.annotate(f"$\\Delta$=+0.6 pp",
            xy=(0.871, 31.43), xytext=(0.91, 35),
            fontsize=8, color="#55A868",
            arrowprops=dict(arrowstyle="->", color="#55A868", lw=0.8))

# City-language: genuine outlier
predicted_cl = (slope * 0.818 + intercept) * 100
ax.annotate(f"$\\Delta$=$-$21.3 pp\n(genuine outlier)",
            xy=(0.818, 11.56), xytext=(0.72, 8),
            fontsize=8, color="#8172B2", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8172B2", lw=0.8))

# City-country: modest excess
ax.annotate(f"$\\Delta$=+8.5 pp",
            xy=(0.726, 45.10), xytext=(0.67, 48),
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
    fig.savefig(os.path.join(out_dir, f"fig7_probe_degradation.{ext}"),
                dpi=300, bbox_inches="tight")
print(f"Saved fig7_probe_degradation.pdf/png to {out_dir}")
plt.close()
