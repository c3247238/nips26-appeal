"""
Generate Figure 1: EDA Method Diagram
Geometric illustration of encoder-decoder dissociation as the fingerprint of absorption.

Output:
  exp/results/full/fig1_eda_method.pdf
  exp/results/full/fig1_eda_method.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

# -------------------------------------------------------
# Setup paths
# -------------------------------------------------------
script_dir = Path(__file__).parent
workspace = script_dir.parent
results_dir = workspace / "exp" / "results" / "full"
results_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Color palette (publication-quality)
# -------------------------------------------------------
C_ENCODER   = "#2166ac"   # blue
C_DECODER   = "#d73027"   # red
C_PARENT    = "#4dac26"   # green
C_ANGLE     = "#f4a582"   # light orange
C_BG        = "#f7f7f7"
C_AXIS      = "#555555"
C_ARROW_ALPHA = 0.85

# -------------------------------------------------------
# Helper: draw a labeled arrow from origin (0,0) to (dx,dy)
# -------------------------------------------------------
def draw_arrow(ax, dx, dy, color, label, lw=2.5, label_offset=(0.05, 0.05),
               fontsize=10, alpha=1.0, linestyle="-"):
    ax.annotate(
        "", xy=(dx, dy), xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->,head_width=0.18,head_length=0.15",
            color=color, lw=lw, linestyle=linestyle, alpha=alpha
        )
    )
    lx, ly = dx + label_offset[0], dy + label_offset[1]
    ax.text(lx, ly, label, color=color, fontsize=fontsize, fontweight="bold",
            ha="left", va="bottom")

# -------------------------------------------------------
# Helper: draw angle arc between two directions
# -------------------------------------------------------
def draw_angle_arc(ax, angle_start_deg, angle_end_deg, radius=0.3, color="#888888",
                   label=None, label_r=0.45, fontsize=9):
    theta = np.linspace(np.radians(angle_start_deg), np.radians(angle_end_deg), 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            color=color, lw=1.5, linestyle="--", alpha=0.7)
    if label is not None:
        mid_angle = np.radians((angle_start_deg + angle_end_deg) / 2)
        ax.text(label_r * np.cos(mid_angle), label_r * np.sin(mid_angle),
                label, ha="center", va="center", fontsize=fontsize,
                color=color, style="italic")

# -------------------------------------------------------
# Figure layout: 1 row, 3 columns
# Left: Non-absorbed (low EDA)
# Middle: Formula / threshold insets
# Right: Absorbed (high EDA)
# -------------------------------------------------------
fig = plt.figure(figsize=(13, 5.2))
fig.patch.set_facecolor("white")

# GridSpec: [left_panel | middle_text | right_panel]
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, figure=fig, width_ratios=[2, 1.2, 2],
              left=0.05, right=0.97, top=0.88, bottom=0.08, wspace=0.08)

ax_left  = fig.add_subplot(gs[0])
ax_mid   = fig.add_subplot(gs[1])
ax_right = fig.add_subplot(gs[2])

# -------------------------------------------------------
# Shared axis setup
# -------------------------------------------------------
def setup_vector_ax(ax, title, subtitle=""):
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_AXIS, lw=0.8, alpha=0.4)
    ax.axvline(0, color=C_AXIS, lw=0.8, alpha=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color="#222222")
    if subtitle:
        ax.text(0.5, -0.12, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#555555", style="italic")
    # Origin dot
    ax.plot(0, 0, "ko", ms=5, zorder=10)
    ax.text(0.04, -0.12, "origin\n(feature $j$)", ha="center", va="top", fontsize=7.5, color="#444444")
    # Residual stream space label
    ax.text(1.42, -0.15, r"$\mathbb{R}^{d}$", ha="right", va="bottom",
            fontsize=9, color=C_AXIS, style="italic")

# =========================================================
# LEFT PANEL: Non-absorbed feature (low EDA)
# =========================================================
setup_vector_ax(ax_left,
                "Non-Absorbed Feature",
                "EDA ≈ 0  (encoder ≈ decoder)")

# Child concept direction: same for encoder and decoder
enc_angle_noabs = 50   # degrees from horizontal
dec_angle_noabs = 50   # same → low EDA

enc_noabs = np.array([np.cos(np.radians(enc_angle_noabs)),
                      np.sin(np.radians(enc_angle_noabs))])
dec_noabs = np.array([np.cos(np.radians(dec_angle_noabs)),
                      np.sin(np.radians(dec_angle_noabs))])

# Draw encoder and decoder (nearly overlapping, offset slightly for visibility)
offset = 0.04
draw_arrow(ax_left, enc_noabs[0] * 1.1 + offset, enc_noabs[1] * 1.1 + offset,
           C_ENCODER, r"$e_c$ (encoder)", lw=3,
           label_offset=(0.04, 0.03), fontsize=10)
draw_arrow(ax_left, dec_noabs[0] * 1.0 - offset, dec_noabs[1] * 1.0 - offset,
           C_DECODER, r"$d_c$ (decoder)", lw=3,
           label_offset=(0.04, -0.15), fontsize=10)

# Small angle indicator
draw_angle_arc(ax_left, enc_angle_noabs - 4, dec_angle_noabs + 4,
               radius=0.22, color="#aaaaaa",
               label=r"$\approx 0°$", label_r=0.4)

# Child concept label
ax_left.text(0.80, 0.88, "child\nconcept\ndirection",
             ha="center", va="bottom", fontsize=8.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", fc="#eeeeee", ec="#cccccc", alpha=0.8))
ax_left.annotate("", xy=(enc_noabs[0]*0.88, enc_noabs[1]*0.88),
                 xytext=(0.80, 0.82),
                 arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.0))

# EDA box
ax_left.text(0.5, -0.08,
             r"EDA$(c)$ = 1 − cos$(e_c,\, d_c)$ $\approx$ 0",
             ha="center", va="top", fontsize=9.5,
             transform=ax_left.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", fc="#d1e5f0", ec="#2166ac", alpha=0.9),
             color="#1a3d6b")

# =========================================================
# RIGHT PANEL: Absorbed feature (high EDA)
# =========================================================
setup_vector_ax(ax_right,
                "Absorbed Feature",
                "EDA is large  (encoder pulled toward parent)")

# Parent decoder direction (at ~15 degrees — shallow, partly overlapping child)
parent_angle = 15
# Child decoder direction (specialized, ~60 degrees from horizontal)
dec_angle_abs = 60
# Encoder has been pulled toward parent (between parent and child decoder)
enc_angle_abs = 22   # pulled toward parent decoder

parent_dir = np.array([np.cos(np.radians(parent_angle)),
                       np.sin(np.radians(parent_angle))])
dec_abs    = np.array([np.cos(np.radians(dec_angle_abs)),
                       np.sin(np.radians(dec_angle_abs))])
enc_abs    = np.array([np.cos(np.radians(enc_angle_abs)),
                       np.sin(np.radians(enc_angle_abs))])

# Parent decoder direction (dashed, green)
ax_right.annotate(
    "", xy=(parent_dir[0]*1.1, parent_dir[1]*1.1), xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="->,head_width=0.15,head_length=0.14",
        color=C_PARENT, lw=2.2, linestyle="--", alpha=0.8
    )
)
ax_right.text(parent_dir[0]*1.13 + 0.02, parent_dir[1]*1.13,
              r"$d_p$ (parent decoder)", color=C_PARENT,
              fontsize=9.5, fontweight="bold", va="center")

# Child decoder direction (red — anchored to child concept)
draw_arrow(ax_right, dec_abs[0]*1.05, dec_abs[1]*1.05,
           C_DECODER, r"$d_c$ (decoder)", lw=3,
           label_offset=(0.03, 0.04), fontsize=10)

# Encoder direction (blue — pulled toward parent)
draw_arrow(ax_right, enc_abs[0]*1.1, enc_abs[1]*1.1,
           C_ENCODER, r"$e_c$ (encoder)", lw=3,
           label_offset=(0.03, -0.16), fontsize=10)

# Angle between encoder and decoder (the EDA angle)
draw_angle_arc(ax_right, enc_angle_abs, dec_angle_abs,
               radius=0.42, color=C_ANGLE,
               label=r"$\theta_{\rm EDA}$", label_r=0.62)

# Angle between encoder and parent (small — encoder pulled toward parent)
draw_angle_arc(ax_right, parent_angle, enc_angle_abs,
               radius=0.24, color="#aaddaa",
               label=r"small", label_r=0.37)

# Arrow showing "pull" direction
ax_right.annotate(
    "", xy=(enc_abs[0]*0.7, enc_abs[1]*0.7),
    xytext=(enc_abs[0]*0.7 + (parent_dir[0] - enc_abs[0])*0.25,
            enc_abs[1]*0.7 + (parent_dir[1] - enc_abs[1])*0.25),
    arrowprops=dict(
        arrowstyle="->,head_width=0.12,head_length=0.11",
        color="#888800", lw=1.4, linestyle="--", alpha=0.7
    )
)
ax_right.text(0.52, 0.50, "gradient\npull →", ha="center", va="center",
              fontsize=7.5, color="#888800", style="italic")

# Child concept label
ax_right.text(dec_abs[0]*0.80 + 0.12, dec_abs[1]*0.80 + 0.08,
              "child\nconcept\ndirection",
              ha="left", va="bottom", fontsize=8.5, color="#555555",
              bbox=dict(boxstyle="round,pad=0.3", fc="#eeeeee", ec="#cccccc", alpha=0.8))

# EDA box
ax_right.text(0.5, -0.08,
              r"EDA$(c)$ = 1 − cos$(e_c,\, d_c)$ $\gg$ 0",
              ha="center", va="top", fontsize=9.5,
              transform=ax_right.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", fc="#fddbc7", ec="#d73027", alpha=0.9),
              color="#8b0000")

# =========================================================
# MIDDLE PANEL: Formula insets
# =========================================================
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.axis("off")
ax_mid.set_facecolor("white")

# Title
ax_mid.text(0.5, 0.97, "Theory", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#222222",
            transform=ax_mid.transAxes)

# EDA definition box
eda_box_y = 0.80
ax_mid.text(0.5, eda_box_y,
            "EDA Metric\n"
            r"$\text{EDA}(c) = 1 - \cos(\hat{e}_c, d_c)$",
            ha="center", va="top", fontsize=9, color="#1a3d6b",
            transform=ax_mid.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#d1e5f0", ec="#2166ac", alpha=0.95),
            linespacing=1.6)

# Absorption threshold box
thresh_box_y = 0.52
ax_mid.text(0.5, thresh_box_y,
            "Absorption Threshold\n"
            r"(Proposition 1)" + "\n\n"
            r"$\lambda > \sin^2(\theta_{p,c})$" + "\n\n"
            r"$\Leftrightarrow$ absorbed $S_2$ has lower loss",
            ha="center", va="top", fontsize=9, color="#4d0000",
            transform=ax_mid.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fddbc7", ec="#d73027", alpha=0.95),
            linespacing=1.5)

# Mechanistic link box
mech_box_y = 0.18
ax_mid.text(0.5, mech_box_y,
            "Mechanistic Conjecture\n"
            r"(Proposition 2)" + "\n\n"
            r"Absorbed $e_c \rightarrow d_p$" + "\n"
            r"while $d_c$ stays anchored" + "\n"
            r"$\Rightarrow$ EDA grows",
            ha="center", va="top", fontsize=8.5, color="#1a3a1a",
            transform=ax_mid.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#e6f4e6", ec="#4dac26", alpha=0.95),
            linespacing=1.4)

# Arrows connecting boxes
ax_mid.annotate("", xy=(0.5, thresh_box_y + 0.01),
                xytext=(0.5, eda_box_y - 0.14),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->,head_width=0.12",
                                color="#888888", lw=1.2))
ax_mid.annotate("", xy=(0.5, mech_box_y + 0.01),
                xytext=(0.5, thresh_box_y - 0.18),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->,head_width=0.12",
                                color="#888888", lw=1.2))

# =========================================================
# Legend row at top
# =========================================================
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=C_ENCODER, lw=2.5, label=r"$e_c$: encoder direction"),
    Line2D([0], [0], color=C_DECODER, lw=2.5, label=r"$d_c$: child decoder"),
    Line2D([0], [0], color=C_PARENT, lw=2.2, linestyle="--",
           label=r"$d_p$: parent decoder"),
]
fig.legend(handles=legend_elements, loc="upper center",
           ncol=3, fontsize=9.5, frameon=True,
           bbox_to_anchor=(0.5, 1.01),
           handlelength=2.0, columnspacing=1.2)

# =========================================================
# Super title
# =========================================================
fig.suptitle(
    "Figure 1: Encoder-Decoder Dissociation as the Geometric Fingerprint of Feature Absorption",
    fontsize=11, fontweight="bold", y=1.07, color="#111111"
)

# =========================================================
# Save
# =========================================================
out_pdf = results_dir / "fig1_eda_method.pdf"
out_png = results_dir / "fig1_eda_method.png"

plt.savefig(str(out_pdf), dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig(str(out_png), dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# =========================================================
# Write progress / done markers
# =========================================================
task_id = "task_F1_theory_revised"
progress_file = results_dir.parent / "full" / f"{task_id}_PROGRESS.json"
done_file = results_dir.parent / "full" / f"{task_id}_DONE"

import datetime
progress = {
    "task_id": task_id,
    "epoch": 1, "total_epochs": 1,
    "step": 1, "total_steps": 1,
    "metric": {"figure_generated": True,
               "theory_doc_written": True},
    "updated_at": datetime.datetime.now().isoformat()
}
progress_file.write_text(json.dumps(progress, indent=2))

done_marker = {
    "task_id": task_id,
    "status": "success",
    "summary": ("F1 theory revised: EDA as geometric fingerprint. "
                "Figure 1 generated as PDF+PNG. "
                "Proposition 2 (encoder pull) labeled as mechanistic conjecture."),
    "outputs": [str(out_pdf), str(out_png), str(results_dir / "F1_theory_revised.md")],
    "pilot_pass": True,
    "timestamp": datetime.datetime.now().isoformat()
}
done_file.write_text(json.dumps(done_marker, indent=2))

print(f"Progress file: {progress_file}")
print(f"Done marker:   {done_file}")
