"""
Generate UDWDC feedback control loop block diagram.
Based on figures/udwdc_control_loop_desc.md specification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from style_config import apply_style, DPI

apply_style()

fig, ax = plt.subplots(figsize=(11.0, 5.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.2)
ax.axis('off')

# ─── color scheme ───────────────────────────────────────────────────────────
C_BLUE   = '#1565C0'   # target / reference path
C_RED    = '#C62828'   # controller path (UDWDC)
C_GRAY   = '#546E7A'   # feedback / measurement path
C_BLACK  = '#212121'   # plant / weight update
C_BG     = '#ECEFF1'   # block fill
C_CTRL   = '#FFCDD2'   # controller highlight
C_PLANT  = '#E8F5E9'   # plant highlight

def rounded_box(ax, x, y, w, h, color=C_BG, edgecolor='#455A64', lw=1.3, zorder=2):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=edgecolor, lw=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def circle_node(ax, x, y, r=0.28, color='white', ec='#455A64', lw=1.3, zorder=3):
    circ = plt.Circle((x, y), r, facecolor=color, edgecolor=ec, lw=lw, zorder=zorder)
    ax.add_patch(circ)

def arrow(ax, x0, y0, x1, y1, color='#455A64', lw=1.3, zorder=4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=zorder)

def label(ax, x, y, text, fontsize=9, color='#212121', ha='center', va='center', **kw):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            family='serif', **kw)

# ─── layout constants ────────────────────────────────────────────────────────
# Row 1 (main forward path): y = 3.4
# Row 2 (feedback path):     y = 1.4
Y_FWD = 3.4
Y_FB  = 1.4

# x positions for main blocks
X_TGT  = 1.1   # target generator
X_SUM  = 2.6   # summation node
X_KP   = 4.1   # proportional gain
X_CLAMP = 5.8  # clamp
X_WU   = 7.7   # weight update (plant)
X_MEAS = 7.7   # measurement (feedback, same x as weight update, lower row)
X_END  = 10.2  # output

# ─── UDWDC controller dashed box ────────────────────────────────────────────
ctrl_box = FancyBboxPatch((3.35, 2.65), 3.25, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor=C_CTRL, edgecolor=C_RED, lw=1.5,
                           linestyle='--', zorder=1)
ax.add_patch(ctrl_box)
label(ax, 4.97, 4.28, 'UDWDC Controller', fontsize=8.5, color=C_RED, style='italic')

# ─── open-loop arrow (FixedWD bypass) ───────────────────────────────────────
# A thin straight arrow from λ_base label to Weight Update, drawn behind
ax.annotate('', xy=(X_WU - 0.55, Y_FWD + 0.62), xytext=(X_SUM + 0.28, Y_FWD + 0.62),
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.1, linestyle='dashed'),
            zorder=1)
label(ax, 4.4, Y_FWD + 0.95, 'Open-loop (FixedWD): $\\lambda_\\mathrm{base}$ direct',
      fontsize=7.5, color='#78909C', ha='center')

# ─── Target Generator block ──────────────────────────────────────────────────
rounded_box(ax, X_TGT, Y_FWD, 1.6, 0.80, color='#E3F2FD', edgecolor=C_BLUE)
label(ax, X_TGT, Y_FWD + 0.15, r'Target Generator', fontsize=8.5, color=C_BLUE)
label(ax, X_TGT, Y_FWD - 0.18, r'$\rho^*(t)=\eta_t/\tau$', fontsize=8.0, color=C_BLUE)

# η_t input (from above)
arrow(ax, X_TGT, Y_FWD + 0.8, X_TGT, Y_FWD + 0.42, color=C_BLUE)
label(ax, X_TGT - 0.28, Y_FWD + 1.0, r'$\eta_t$', fontsize=9, color=C_BLUE)

# ─── Summation node ──────────────────────────────────────────────────────────
circle_node(ax, X_SUM, Y_FWD, r=0.30, color='white', ec=C_RED)
label(ax, X_SUM, Y_FWD, r'$\Sigma$', fontsize=12, color=C_RED)
# '+' and '-' signs
label(ax, X_SUM - 0.36, Y_FWD + 0.15, '+', fontsize=9, color=C_GRAY)
label(ax, X_SUM - 0.06, Y_FWD - 0.25, '−', fontsize=11, color=C_BLUE)

# Arrow from Target Generator to summation (negative input, below)
ax.annotate('', xy=(X_SUM - 0.02, Y_FWD - 0.31), xytext=(X_TGT + 0.81, Y_FWD - 0.31),
            arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2), zorder=4)
ax.plot([X_TGT + 0.81, X_TGT + 0.81], [Y_FWD - 0.42, Y_FWD - 0.31], color=C_BLUE, lw=1.2)
ax.plot([X_TGT + 0.81, X_TGT + 0.81], [Y_FWD - 0.42, Y_FWD - 0.31], color=C_BLUE, lw=1.2)

# Arrow: target generator → summation (top, ρ*(t) label)
arrow(ax, X_TGT + 0.81, Y_FWD, X_SUM - 0.30, Y_FWD, color=C_BLUE)
label(ax, (X_TGT + X_SUM) / 2, Y_FWD + 0.22, r'$\rho^*(t)$', fontsize=8.5, color=C_BLUE)

# ─── Proportional Gain block ─────────────────────────────────────────────────
rounded_box(ax, X_KP, Y_FWD, 1.2, 0.75, color=C_CTRL, edgecolor=C_RED)
label(ax, X_KP, Y_FWD + 0.13, r'Proportional', fontsize=8.5, color=C_RED)
label(ax, X_KP, Y_FWD - 0.17, r'$\lambda_\mathrm{base}\cdot(\rho_t^l/\rho^*)$', fontsize=8.0, color=C_RED)

# Arrow: summation → Kp
arrow(ax, X_SUM + 0.31, Y_FWD, X_KP - 0.61, Y_FWD, color=C_RED)
label(ax, (X_SUM + X_KP) / 2, Y_FWD + 0.22, r'$e_t^l$', fontsize=9, color=C_RED)

# ─── Clamp block ─────────────────────────────────────────────────────────────
rounded_box(ax, X_CLAMP, Y_FWD, 1.3, 0.75, color=C_CTRL, edgecolor=C_RED)
label(ax, X_CLAMP, Y_FWD + 0.13, r'Clamp', fontsize=8.5, color=C_RED)
label(ax, X_CLAMP, Y_FWD - 0.17, r'$[0.1,\,10]\times\lambda_\mathrm{base}$', fontsize=7.5, color=C_RED)

# Arrow: Kp → Clamp
arrow(ax, X_KP + 0.61, Y_FWD, X_CLAMP - 0.66, Y_FWD, color=C_RED)

# ─── Weight Update (Plant) block ─────────────────────────────────────────────
rounded_box(ax, X_WU, Y_FWD, 2.2, 0.90, color=C_PLANT, edgecolor=C_BLACK)
label(ax, X_WU, Y_FWD + 0.18, r'Weight Update (Plant)', fontsize=8.5, color=C_BLACK)
label(ax, X_WU, Y_FWD - 0.16,
      r'$w_{t+1}^l = (1{-}\eta_t\lambda_t^l)w_t^l - \eta_t g_t^l$',
      fontsize=7.5, color=C_BLACK)

# Arrow: Clamp → Weight Update
arrow(ax, X_CLAMP + 0.66, Y_FWD, X_WU - 1.11, Y_FWD, color=C_RED)
label(ax, (X_CLAMP + X_WU) / 2, Y_FWD + 0.22, r'$\lambda_t^l$', fontsize=9, color=C_RED)

# Output arrow (w_t, g_t)
arrow(ax, X_WU + 1.11, Y_FWD, X_END, Y_FWD, color=C_BLACK)
label(ax, X_END + 0.3, Y_FWD, r'$w_t^l,\,g_t^l$', fontsize=8.5, color=C_BLACK, ha='left')

# ─── Measurement block (feedback path) ───────────────────────────────────────
rounded_box(ax, X_MEAS, Y_FB, 2.2, 0.80, color='#EEEEEE', edgecolor=C_GRAY)
label(ax, X_MEAS, Y_FB + 0.15, r'Measurement', fontsize=8.5, color=C_GRAY)
label(ax, X_MEAS, Y_FB - 0.15, r'$\rho_t^l = \|g_t^l\|/\|w_t^l\|$', fontsize=8.0, color=C_GRAY)

# Vertical arrow: weight update → measurement
ax.annotate('', xy=(X_MEAS + 0.5, Y_FB + 0.41), xytext=(X_MEAS + 0.5, Y_FWD - 0.46),
            arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2), zorder=4)
label(ax, X_MEAS + 0.9, (Y_FWD + Y_FB) / 2, r'$w_t^l,\,g_t^l$',
      fontsize=8, color=C_GRAY, ha='left')

# Feedback horizontal arrow: measurement → summation node (bottom path)
ax.plot([X_MEAS - 1.11, X_SUM], [Y_FB, Y_FB], color=C_GRAY, lw=1.2)
ax.annotate('', xy=(X_SUM, Y_FWD - 0.31), xytext=(X_SUM, Y_FB),
            arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2), zorder=4)
label(ax, (X_MEAS + X_SUM) / 2 - 0.5, Y_FB - 0.3,
      r'$\hat\rho_t^l$ (feedback)', fontsize=8.5, color=C_GRAY)

# ─── Comparison annotations (dashed boxes below main diagram) ────────────────
ann_y = 0.6
ann_configs = [
    (X_KP,   'CWD: $K_d$ with\n$\\alpha_t^l$ here',   '#1976D2'),
    (X_CLAMP,'CPR: $K_i$ integral\naccumulation here', '#388E3C'),
    (X_SUM,  'SWD: $K_p$ on\nglobal $\\|\\nabla\\mathcal{L}\\|$', '#F57C00'),
]
for xc, txt, col in ann_configs:
    box_ann = FancyBboxPatch((xc - 0.72, ann_y - 0.30), 1.44, 0.60,
                             boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=col, lw=1.0,
                             linestyle='dotted', zorder=2)
    ax.add_patch(box_ann)
    label(ax, xc, ann_y, txt, fontsize=7.0, color=col)

label(ax, 0.3, ann_y, 'Other\nmethods:', fontsize=7.5, color='#455A64', ha='left')

# ─── save ────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(__file__)
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(out_dir, f'udwdc_control_loop.{ext}'),
                dpi=DPI, bbox_inches='tight')
print("Saved udwdc_control_loop.pdf and .png")
plt.close()
