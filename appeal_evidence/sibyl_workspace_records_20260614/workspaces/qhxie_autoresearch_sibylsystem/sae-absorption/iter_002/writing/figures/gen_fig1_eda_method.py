"""
Figure 1: Method Diagram — Absorbed vs. Non-Absorbed Feature Geometry

Two-panel conceptual illustration:
  Left:  Non-absorbed feature. enc_c ≈ dec_c. EDA ≈ 0.
  Right: Absorbed feature. enc_c pulled toward parent dec_p. Large EDA.

Output: writing/figures/fig1_eda_method.pdf
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(WORKSPACE, "figures", "fig1_eda_method.pdf")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
GRAY   = "#888888"
BLACK  = "#222222"

def draw_arrow(ax, start, end, color, lw=2.0, arrowstyle="-|>", label=None,
               label_offset=(0, 0.04), fontsize=8):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=12))
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, color=color, fontsize=fontsize,
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def arc_angle_label(ax, center, r, angle_start_deg, angle_end_deg, label, color,
                    label_r_factor=1.35):
    thetas = np.linspace(np.radians(angle_start_deg), np.radians(angle_end_deg), 60)
    xs = center[0] + r * np.cos(thetas)
    ys = center[1] + r * np.sin(thetas)
    ax.plot(xs, ys, color=color, lw=1.2, linestyle="--")
    mid_theta = np.radians((angle_start_deg + angle_end_deg) / 2)
    lx = center[0] + r * label_r_factor * np.cos(mid_theta)
    ly = center[1] + r * label_r_factor * np.sin(mid_theta)
    ax.text(lx, ly, label, color=color, fontsize=7.5, ha="center", va="center")


fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
fig.subplots_adjust(left=0.04, right=0.97, bottom=0.12, top=0.88, wspace=0.32)

# ── Panel A: Non-absorbed ──────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-0.15, 1.25)
ax.set_ylim(-0.15, 1.25)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Non-absorbed feature", fontsize=9, fontweight="bold", color=BLACK, pad=4)

origin = (0.1, 0.1)

# enc_c ≈ dec_c (both pointing ~same direction, child concept)
theta_enc = np.radians(70)
theta_dec = np.radians(75)  # nearly identical — small EDA
L = 0.85

enc_end = (origin[0] + L * np.cos(theta_enc), origin[1] + L * np.sin(theta_enc))
dec_end = (origin[0] + L * np.cos(theta_dec), origin[1] + L * np.sin(theta_dec))

draw_arrow(ax, origin, enc_end, BLUE, lw=2.2, label=r"$\hat{e}_c$",
           label_offset=(-0.10, 0.0))
draw_arrow(ax, origin, dec_end, RED, lw=2.2, label=r"$d_c$",
           label_offset=(0.10, 0.0))

# child concept label at arrowhead region
ax.text(enc_end[0] - 0.05, enc_end[1] + 0.07,
        "child\nconcept", ha="center", va="bottom", fontsize=7.5,
        color=GRAY, style="italic")

# small arc showing tiny angle
arc_angle_label(ax, origin, 0.28, np.degrees(theta_enc), np.degrees(theta_dec),
                "", GRAY, label_r_factor=1.5)

# EDA annotation
ax.text(0.60, 0.12, "EDA $\\approx 0$", ha="center", fontsize=8.5,
        color=GREEN, fontweight="bold")

ax.text(0.65, 0.01, r"$\hat{e}_c \approx d_c$", ha="center", fontsize=8,
        color=GRAY)

# ── Panel B: Absorbed ─────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(-0.15, 1.35)
ax.set_ylim(-0.15, 1.30)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Absorbed feature", fontsize=9, fontweight="bold", color=BLACK, pad=4)

origin = (0.1, 0.1)

# dec_c stays pointing at child concept (similar to panel A)
theta_dec_c = np.radians(72)
L = 0.85

# enc_c pulled toward parent direction (d_p, shown separately)
theta_enc_c = np.radians(30)   # encoder pulled low-right toward parent
theta_d_p   = np.radians(18)   # parent decoder direction (even lower)

dec_c_end = (origin[0] + L * np.cos(theta_dec_c), origin[1] + L * np.sin(theta_dec_c))
enc_c_end = (origin[0] + L * np.cos(theta_enc_c), origin[1] + L * np.sin(theta_enc_c))
d_p_end   = (origin[0] + (L + 0.0) * np.cos(theta_d_p),
             origin[1] + (L + 0.0) * np.sin(theta_d_p))

# parent decoder d_p (dashed, showing the "magnet")
draw_arrow(ax, origin, d_p_end, GREEN, lw=1.6, arrowstyle="-|>",
           label=r"$d_p$", label_offset=(0.09, -0.07))

# enc_c pulled toward d_p
draw_arrow(ax, origin, enc_c_end, BLUE, lw=2.2, label=r"$\hat{e}_c$",
           label_offset=(0.12, -0.06))

# dec_c anchored to child concept
draw_arrow(ax, origin, dec_c_end, RED, lw=2.2, label=r"$d_c$",
           label_offset=(-0.11, 0.05))

# Concept labels
ax.text(dec_c_end[0] - 0.03, dec_c_end[1] + 0.07,
        "child\nconcept", ha="center", va="bottom", fontsize=7.5,
        color=GRAY, style="italic")
ax.text(d_p_end[0] + 0.10, d_p_end[1] - 0.04,
        "parent\nconcept", ha="left", va="center", fontsize=7.5,
        color=GRAY, style="italic")

# Big arc showing large EDA angle between enc_c and dec_c
arc_angle_label(ax, origin, 0.32, np.degrees(theta_enc_c), np.degrees(theta_dec_c),
                "EDA", RED, label_r_factor=1.55)

# EDA annotation
ax.text(0.65, 0.12, "EDA $\\gg 0$", ha="center", fontsize=8.5,
        color="#c0392b", fontweight="bold")

# Pull arrow annotation (gradient pull)
ax.annotate("gradient\npulls $\\hat{e}_c$\ntoward $d_p$",
            xy=enc_c_end, xytext=(enc_c_end[0] + 0.25, enc_c_end[1] + 0.22),
            fontsize=7, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8,
                            connectionstyle="arc3,rad=-0.2"))

# ── Shared formula inset at bottom ────────────────────────────────────────
fig.text(0.50, 0.03,
         r"$\mathrm{EDA}(c)\ =\ 1 - \cos(\hat{e}_c,\, d_c)$"
         r"$\quad$"
         r"Absorbed when $\lambda > \sin^2(\theta_{p,c})$",
         ha="center", va="bottom", fontsize=8.5, color=BLACK,
         bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.8))

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=BLUE,  label=r"$\hat{e}_j$ (encoder direction)"),
    mpatches.Patch(facecolor=RED,   label=r"$d_c$ (child decoder)"),
    mpatches.Patch(facecolor=GREEN, label=r"$d_p$ (parent decoder)"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=3, fontsize=7.5, frameon=True,
           bbox_to_anchor=(0.50, 0.09))

fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT_PATH}")
plt.close()
