"""
Generate Figure 1: Rate-Distortion Framing Diagram for Feature Absorption
Shows:
  (a) Two solutions comparison: both active vs. absorbed
  (b) Geometric illustration of theta_{p,c} and the threshold formula
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.gridspec import GridSpec
import os

# Output directory
results_dir = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/exp/results/full"
os.makedirs(results_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure setup
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('white')

gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.06, right=0.96,
              top=0.88, bottom=0.12)

ax_left  = fig.add_subplot(gs[0, 0])   # Panel (a): two-solution comparison
ax_right = fig.add_subplot(gs[0, 1])   # Panel (b): decoder geometry + threshold

# Color palette
C_PARENT  = '#2166AC'   # blue  – parent feature
C_CHILD   = '#D6604D'   # red   – child feature
C_ORTHO   = '#808080'   # grey  – orthogonal residual
C_ARROW   = '#1A1A1A'   # near-black arrows
C_THRESH  = '#4DAC26'   # green – threshold curve
C_REGION1 = '#D1E5F0'   # light blue – non-absorption zone
C_REGION2 = '#FDDBC7'   # light red  – absorption zone

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): Two-solution comparison (table / equation style)
# ─────────────────────────────────────────────────────────────────────────────
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 10)
ax_left.axis('off')
ax_left.set_title('(a) Two SAE Solutions for a Hierarchical Pair',
                  fontsize=12, fontweight='bold', pad=8)

# ── Solution S1: both features active ─────────────────────────────────────
ax_left.add_patch(mpatches.FancyBboxPatch(
    (0.3, 5.7), 9.4, 3.7,
    boxstyle="round,pad=0.15", linewidth=1.5,
    edgecolor=C_PARENT, facecolor='#EBF3FB'))
ax_left.text(5, 9.15, 'Solution $S_1$:  Both Parent and Child Active',
             ha='center', va='center', fontsize=11, fontweight='bold', color=C_PARENT)
ax_left.text(5, 8.5,
             r'Reconstruction: $\hat{x} \;=\; z_p \mathbf{d}_p \;+\; z_c \mathbf{d}_c \;+\; \cdots$',
             ha='center', va='center', fontsize=10)
ax_left.text(2.5, 7.75, r'• $L_0$ cost: $\mathbf{2}$ active latents',
             ha='left', va='center', fontsize=9.5)
ax_left.text(2.5, 7.1, r'• Reconstruction error: $\mathbf{0}$ additional loss',
             ha='left', va='center', fontsize=9.5)
ax_left.text(2.5, 6.45, r'• Total cost: $\;\lambda \cdot 2 \cdot p_\mathrm{co}$  (sparsity)',
             ha='left', va='center', fontsize=9.5)

# ── Solution S2: absorbed ─────────────────────────────────────────────────
ax_left.add_patch(mpatches.FancyBboxPatch(
    (0.3, 1.4), 9.4, 3.7,
    boxstyle="round,pad=0.15", linewidth=1.5,
    edgecolor=C_CHILD, facecolor='#FEF0EB'))
ax_left.text(5, 4.85, 'Solution $S_2$:  Only Child Active (Absorbed)',
             ha='center', va='center', fontsize=11, fontweight='bold', color=C_CHILD)
ax_left.text(5, 4.2,
             r'Reconstruction: $\hat{x} \;=\; z_c^\prime \mathbf{d}_c \;+\; \cdots$  '
             r'($\mathbf{d}_p$ direction partially lost)',
             ha='center', va='center', fontsize=10)
ax_left.text(2.5, 3.5, r'• $L_0$ cost: $\mathbf{1}$ active latent  $\Rightarrow$ saves $\lambda \cdot p_\mathrm{co}$',
             ha='left', va='center', fontsize=9.5)
ax_left.text(2.5, 2.85,
             r'• Extra recon. error: $\sin^2(\theta_{p,c})\cdot p_\mathrm{co}$',
             ha='left', va='center', fontsize=9.5)
ax_left.text(2.5, 2.2,
             r'• Net gain: $p_\mathrm{co}\,[\lambda - \sin^2(\theta_{p,c})]$',
             ha='left', va='center', fontsize=9.5)

# ── Threshold condition ────────────────────────────────────────────────────
ax_left.add_patch(mpatches.FancyBboxPatch(
    (1.5, 0.15), 7.0, 0.95,
    boxstyle="round,pad=0.1", linewidth=2,
    edgecolor=C_THRESH, facecolor='#E8F8E8'))
ax_left.text(5, 0.63,
             r'$S_2$ preferred $\Leftrightarrow$  $\lambda > \sin^2(\theta_{p,c})$',
             ha='center', va='center', fontsize=11.5, fontweight='bold', color='#2B7A0B')

# ── Arrow between S1 and S2 ───────────────────────────────────────────────
ax_left.annotate('', xy=(5, 5.5), xytext=(5, 5.85),
                 arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8))

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Decoder geometry + phase diagram
# ─────────────────────────────────────────────────────────────────────────────
ax_right.set_title('(b) Decoder Geometry and Absorption Threshold',
                   fontsize=12, fontweight='bold', pad=8)

# Sub-divide panel (b) into top half (geometry) and bottom half (phase plot)
# We draw them in the same axes with careful manual placement.
# ── Geometry illustration (top half, y ∈ [0.45, 1.0] in axes coords) ───────

# Use inset axes for geometry
ax_geo = ax_right.inset_axes([0.0, 0.44, 1.0, 0.56])   # [x0, y0, w, h] in axes fraction
ax_geo.set_xlim(-0.15, 1.3)
ax_geo.set_ylim(-0.15, 1.15)
ax_geo.axis('off')
ax_geo.set_facecolor('white')

# Origin
origin = np.array([0.0, 0.0])

# Decoder directions
theta_deg = 35
theta_rad = np.radians(theta_deg)
d_c = np.array([1.0, 0.0])                           # child decoder (along x)
d_p = np.array([np.cos(theta_rad), np.sin(theta_rad)])  # parent decoder at angle theta

# Projection of d_p onto d_c
proj = np.dot(d_p, d_c) * d_c  # = (cos theta, 0)
# Orthogonal component (the residual that absorption cannot recover)
orth = d_p - proj              # = (0, sin theta)

# Draw child decoder direction
ax_geo.annotate('', xy=d_c, xytext=origin,
                arrowprops=dict(arrowstyle='->', color=C_CHILD, lw=2.5,
                                mutation_scale=18))
ax_geo.text(d_c[0]+0.04, d_c[1]-0.08, r'$\mathbf{d}_c$', color=C_CHILD,
            fontsize=12, fontweight='bold')

# Draw parent decoder direction
ax_geo.annotate('', xy=d_p, xytext=origin,
                arrowprops=dict(arrowstyle='->', color=C_PARENT, lw=2.5,
                                mutation_scale=18))
ax_geo.text(d_p[0]+0.04, d_p[1]+0.02, r'$\mathbf{d}_p$', color=C_PARENT,
            fontsize=12, fontweight='bold')

# Draw the projection (dashed)
ax_geo.plot([0, proj[0]], [0, proj[1]], '--', color='#999999', lw=1.5)
ax_geo.plot([proj[0]], [proj[1]], 'o', color='#999999', markersize=5)
ax_geo.text(proj[0]*0.55, -0.1, r'$\cos\theta_{p,c}$', color='#666666',
            fontsize=9.5, ha='center')

# Draw orthogonal component (the sin term)
ax_geo.annotate('', xy=d_p, xytext=proj,
                arrowprops=dict(arrowstyle='->', color=C_ORTHO, lw=1.8,
                                linestyle='dashed', mutation_scale=14))
ax_geo.text(proj[0]+0.07, (d_p[1]+proj[1])/2, r'$\sin\theta_{p,c}$',
            color=C_ORTHO, fontsize=9.5)

# Angle arc
angle_arc = Arc(xy=(0,0), width=0.38, height=0.38,
                angle=0, theta1=0, theta2=theta_deg,
                color=C_ARROW, lw=1.5)
ax_geo.add_patch(angle_arc)
ax_geo.text(0.23, 0.075, r'$\theta_{p,c}$', fontsize=10, color=C_ARROW)

# Unit circle arc (just a quarter)
phi = np.linspace(0, np.pi/2, 80)
ax_geo.plot(0.9*np.cos(phi), 0.9*np.sin(phi), ':', color='#CCCCCC', lw=1)

# Annotation for what is lost in absorption
ax_geo.text(1.0, 0.52, 'Lost in\nabsorption', fontsize=8.5,
            ha='left', va='center', color=C_ORTHO,
            bbox=dict(facecolor='#F8F8F8', edgecolor=C_ORTHO, boxstyle='round,pad=0.2',
                      alpha=0.9))
ax_geo.annotate('', xy=(proj[0]+0.045, (d_p[1]+proj[1])/2 + 0.05),
                xytext=(1.0, 0.52),
                arrowprops=dict(arrowstyle='->', color=C_ORTHO, lw=1.2))

ax_geo.set_title('Decoder geometry', fontsize=9.5, style='italic',
                 color='#444444', pad=2)

# ── Phase / threshold plot (bottom half, y ∈ [0.0, 0.44] in axes coords) ──
ax_phase = ax_right.inset_axes([0.0, 0.0, 1.0, 0.41])
ax_phase.set_facecolor('white')
ax_phase.set_xlabel(r'Decoder angle $\theta_{p,c}$ (degrees)', fontsize=9.5)
ax_phase.set_ylabel(r'Sparsity penalty $\lambda$', fontsize=9.5)
ax_phase.set_title('Absorption threshold', fontsize=9.5, style='italic',
                   color='#444444')
ax_phase.set_xlim(0, 90)
ax_phase.set_ylim(0, 1.02)
ax_phase.tick_params(labelsize=8.5)

theta_arr = np.linspace(0, 90, 300)
sin2_arr  = np.sin(np.radians(theta_arr))**2

# Shaded regions
ax_phase.fill_between(theta_arr, 0, sin2_arr, color=C_REGION2, alpha=0.6,
                      label='Absorption preferred\n($\\lambda > \\sin^2\\theta_{p,c}$)')
ax_phase.fill_between(theta_arr, sin2_arr, 1.02, color=C_REGION1, alpha=0.6,
                      label='Non-absorption preferred\n($\\lambda < \\sin^2\\theta_{p,c}$)')

# Threshold curve
ax_phase.plot(theta_arr, sin2_arr, color=C_THRESH, lw=2.5,
              label=r'Threshold: $\lambda = \sin^2(\theta_{p,c})$')

# Mark a typical GPT-2 Small lambda (1/L0 ≈ 1/49 ≈ 0.020) and theta range
lambda_gpt2 = 1 / 49.0
ax_phase.axhline(lambda_gpt2, color='#B22222', linestyle=':', lw=1.5, alpha=0.7)
theta_c = np.degrees(np.arcsin(np.sqrt(lambda_gpt2)))
ax_phase.axvline(theta_c, color='#B22222', linestyle=':', lw=1.5, alpha=0.7)
ax_phase.text(theta_c + 1, lambda_gpt2 + 0.035,
              f'GPT-2 L6\n$\\lambda = 1/L_0 \\approx {lambda_gpt2:.3f}$\n$\\theta_c \\approx {theta_c:.1f}°$',
              fontsize=7.5, color='#B22222')

ax_phase.text(20, 0.75, 'Non-absorption\n(S1 preferred)',
              ha='center', va='center', fontsize=9, color='#1A6393',
              fontweight='bold')
ax_phase.text(65, 0.25, 'Absorption\n(S2 preferred)',
              ha='center', va='center', fontsize=9, color='#8B2500',
              fontweight='bold')

ax_phase.legend(fontsize=7.5, loc='center left', framealpha=0.9,
                bbox_to_anchor=(0.01, 0.45))

# Hide the outer right axes (we used insets)
ax_right.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# Overall title and caption
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Figure 1: Rate-Distortion Framework for Feature Absorption in Sparse Autoencoders',
    fontsize=12.5, fontweight='bold', y=0.97
)

fig.text(0.5, 0.01,
         r"$S_2$ (absorbed) is loss-optimal iff $\lambda > \sin^2(\theta_{p,c})$, "
         r"where $\lambda$ is the sparsity penalty and $\theta_{p,c}$ is the decoder angle "
         r"between parent and child feature directions. The threshold is independent of "
         r"co-occurrence frequency $p_\mathrm{co}$.",
         ha='center', fontsize=8.5, color='#333333', style='italic',
         wrap=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
pdf_path = os.path.join(results_dir, "fig1_method.pdf")
png_path = os.path.join(results_dir, "fig1_method.png")

fig.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close(fig)
print("Figure generation complete.")
