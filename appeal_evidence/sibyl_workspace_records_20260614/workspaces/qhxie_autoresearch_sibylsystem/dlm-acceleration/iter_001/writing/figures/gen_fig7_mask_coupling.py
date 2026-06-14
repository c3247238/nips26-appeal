"""
Figure 7: Mask-State Coupling and Composability — Why composability is binary.
Two-column conceptual diagram showing compatible vs. incompatible composition.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from style_config import apply_style, COLORS, FIGSIZE_WIDE, DPI

apply_style()

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))

# ────────────────────────────────────────────────────────────────────────────
# LEFT PANEL: M1 + IGSD — Compatible Composition
# ────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(5, 9.5, "M1 + CD-SSD: Compatible", ha="center", va="center",
        fontsize=9, fontweight="bold", color=COLORS["M1+IGSD"])

# ---- IGSD Draft Phase box ----
draft_box = FancyBboxPatch((0.3, 6.8), 4.0, 1.6,
                            boxstyle="round,pad=0.1",
                            linewidth=1.0, edgecolor=COLORS["IGSD"],
                            facecolor="#E8F5E9", zorder=3)
ax.add_patch(draft_box)
ax.text(2.3, 7.7, "CD-SSD Draft Phase", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=COLORS["IGSD"])
ax.text(2.3, 7.1, f"$T_{{\\rm draft}}=16$ steps\nMask trajectory: unchanged", ha="center",
        va="center", fontsize=6.5, color="#2E7D32")

# Confidence partitioning arrow
ax.annotate("", xy=(2.3, 6.5), xytext=(2.3, 6.8),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))
ax.text(2.3, 6.3, "Confidence partition ($\\tau=0.9$)", ha="center", va="center",
        fontsize=6.5, color="#555555")

# Frozen tokens box
frozen_box = FancyBboxPatch((0.3, 4.3), 1.8, 1.6,
                              boxstyle="round,pad=0.1",
                              linewidth=1.0, edgecolor="#66BB6A",
                              facecolor="#C8E6C9", zorder=3)
ax.add_patch(frozen_box)
ax.text(1.2, 5.3, "$S_{\\rm accept}$", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color="#2E7D32")
ax.text(1.2, 4.7, "52% frozen\n$H_i = 0$", ha="center", va="center",
        fontsize=6.5, color="#2E7D32")

# Refine tokens box
refine_box = FancyBboxPatch((2.5, 4.3), 1.8, 1.6,
                              boxstyle="round,pad=0.1",
                              linewidth=1.0, edgecolor=COLORS["IGSD"],
                              facecolor="#DCEDC8", zorder=3)
ax.add_patch(refine_box)
ax.text(3.4, 5.3, "$S_{\\rm refine}$", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color="#33691E")
ax.text(3.4, 4.7, "48% refine\nT_full=64", ha="center", va="center",
        fontsize=6.5, color="#33691E")

# M1 cache box — attaches to frozen tokens
cache_box = FancyBboxPatch((0.3, 2.3), 1.8, 1.6,
                             boxstyle="round,pad=0.1",
                             linewidth=1.0, edgecolor=COLORS["M1"],
                             facecolor="#DBEAFE", zorder=3)
ax.add_patch(cache_box)
ax.text(1.2, 3.3, "M1 Cache", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=COLORS["M1"])
ax.text(1.2, 2.7, "CHR = 96%\n$\\eta$-threshold", ha="center", va="center",
        fontsize=6.5, color="#1565C0")

# Arrows: frozen → cache
ax.annotate("", xy=(1.2, 3.9), xytext=(1.2, 4.3),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["M1"], lw=1.2))

# Refine → output
ax.annotate("", xy=(3.4, 3.9), xytext=(3.4, 4.3),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["IGSD"], lw=1.0))
ax.text(3.4, 3.6, "Full T_full\nsteps", ha="center", va="center",
        fontsize=6, color=COLORS["IGSD"])

# Synergy annotation box
synergy_box = FancyBboxPatch((0.3, 0.5), 4.0, 1.4,
                               boxstyle="round,pad=0.15",
                               linewidth=1.2, edgecolor=COLORS["M1+IGSD"],
                               facecolor="#FFF9C4", zorder=3)
ax.add_patch(synergy_box)
ax.text(2.3, 1.3, "SYNERGY: Ortho = 1.385", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#F57F17")
ax.text(2.3, 0.85, "5.13× combined (expected 4.69×)", ha="center", va="center",
        fontsize=6.5, color="#F57F17")

# Arrow from cache result to synergy
ax.annotate("", xy=(1.9, 1.9), xytext=(2.1, 2.3),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["M1+IGSD"], lw=1.0))

# Key insight text
ax.text(2.3, 6.65, "M1 does NOT modify\nmask trajectory", ha="center", va="center",
        fontsize=5.8, color="#1565C0",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["M1"],
                  linewidth=0.7, alpha=0.85))

# ────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL: M2/M3 + any — Incompatible Composition
# ────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")

ax2.text(5, 9.5, "M2 / M3+CD-SSD: Incompatible", ha="center", va="center",
         fontsize=9, fontweight="bold", color=COLORS["interference"])

# Original denoising trajectory box
traj_box = FancyBboxPatch((0.3, 7.3), 9.0, 1.4,
                            boxstyle="round,pad=0.1",
                            linewidth=1.0, edgecolor="#888888",
                            facecolor="#F5F5F5", zorder=3)
ax2.add_patch(traj_box)
ax2.text(4.8, 8.15, "MDM Denoising Trajectory (baseline)", ha="center", va="center",
         fontsize=7.5, fontweight="bold", color="#444444")
ax2.text(4.8, 7.6, "Sequential mask state: $\\tilde{x}_1 \\to \\tilde{x}_2 \\to \\cdots \\to \\tilde{x}_T$\n"
         "Global coupling: each $p(x_i \\mid \\tilde{x}_t)$ depends on ALL other mask states",
         ha="center", va="center", fontsize=6.5, color="#555555")

# Trajectory disruption arrow
ax2.annotate("", xy=(2.0, 6.7), xytext=(3.0, 7.3),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["M2"], lw=1.5))
ax2.annotate("", xy=(6.8, 6.7), xytext=(6.0, 7.3),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["M3"], lw=1.5))

# M2 disruption box
m2_box = FancyBboxPatch((0.3, 5.0), 3.5, 1.5,
                          boxstyle="round,pad=0.1",
                          linewidth=1.2, edgecolor=COLORS["M2"],
                          facecolor="#F5F5F5", zorder=3)
ax2.add_patch(m2_box)
ax2.text(2.05, 5.9, "M2: Step Skipping ($J>3$)", ha="center", va="center",
         fontsize=7, fontweight="bold", color="#555555")
ax2.text(2.05, 5.3, "Skips $\\tilde{x}_t \\to \\tilde{x}_{t+J}$\nViolates cumulative\nconditioning", ha="center", va="center",
         fontsize=6, color="#555555")

# M3 disruption box
m3_box = FancyBboxPatch((5.5, 5.0), 3.8, 1.5,
                          boxstyle="round,pad=0.1",
                          linewidth=1.2, edgecolor=COLORS["M3"],
                          facecolor="#FFF0F0", zorder=3)
ax2.add_patch(m3_box)
ax2.text(7.4, 5.9, "M3: AR-Guided Unmasking", ha="center", va="center",
         fontsize=7, fontweight="bold", color=COLORS["M3"])
ax2.text(7.4, 5.3, "Biases unmasking order\nAR logits alter token\ndistribution", ha="center", va="center",
         fontsize=6, color=COLORS["M3"])

# Cascading effects
ax2.annotate("", xy=(2.05, 3.8), xytext=(2.05, 5.0),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["interference"], lw=1.3))
ax2.annotate("", xy=(7.4, 3.8), xytext=(7.4, 5.0),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["interference"], lw=1.3))

# Cascading interference boxes
m2_fail = FancyBboxPatch((0.3, 2.5), 3.5, 1.2,
                           boxstyle="round,pad=0.1",
                           linewidth=1.2, edgecolor=COLORS["interference"],
                           facecolor="#FFEBEE", zorder=3)
ax2.add_patch(m2_fail)
ax2.text(2.05, 3.2, "Unresolvable mask inconsistencies", ha="center", va="center",
         fontsize=6.5, fontweight="bold", color=COLORS["interference"])
ax2.text(2.05, 2.8, "AccRet → 0.24 at $J=8$\nNO_GO verdict", ha="center", va="center",
         fontsize=6, color=COLORS["interference"])

m3_fail = FancyBboxPatch((5.5, 2.5), 3.8, 1.2,
                           boxstyle="round,pad=0.1",
                           linewidth=1.2, edgecolor=COLORS["interference"],
                           facecolor="#FFEBEE", zorder=3)
ax2.add_patch(m3_fail)
ax2.text(7.4, 3.2, "Corrupted CD-SSD draft trajectory", ha="center", va="center",
         fontsize=6.5, fontweight="bold", color=COLORS["interference"])
ax2.text(7.4, 2.8, "Speedup: 3.4× → 2.3×\nOrtho = 0.493", ha="center", va="center",
         fontsize=6, color=COLORS["interference"])

# Bottom interference result box
interfere_box = FancyBboxPatch((1.5, 0.5), 7.0, 1.5,
                                boxstyle="round,pad=0.15",
                                linewidth=1.2, edgecolor=COLORS["interference"],
                                facecolor="#FFCDD2", zorder=3)
ax2.add_patch(interfere_box)
ax2.text(5.0, 1.35, "Root Cause: Mask-State Coupling", ha="center", va="center",
         fontsize=7.5, fontweight="bold", color=COLORS["interference"])
ax2.text(5.0, 0.85, "Methods modifying trajectory conflict with methods assuming trajectory stability",
         ha="center", va="center", fontsize=6.3, color=COLORS["interference"])

ax2.annotate("", xy=(4.0, 2.0), xytext=(3.5, 2.5),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["interference"], lw=1.0))
ax2.annotate("", xy=(6.0, 2.0), xytext=(6.5, 2.5),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["interference"], lw=1.0))

plt.tight_layout(pad=0.5)
output_path = os.path.join(os.path.dirname(__file__), "fig7_mask_coupling.pdf")
plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")
