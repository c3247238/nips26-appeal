"""Generate Figure 2: Mediation Path Diagram (L0 -> Absorption -> Quality).

Three-box path diagram with standardized coefficients for SCR full mediation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Box positions
boxes = {
    'X': (1.0, 1.5, 'log($L_0$)'),
    'M': (5.0, 4.5, 'Absorption'),
    'Y': (9.0, 1.5, 'SCR Quality'),
}

box_style = dict(boxstyle='round,pad=0.5', facecolor='#E8EAF6', edgecolor='#3F51B5',
                 linewidth=2)

for key, (cx, cy, label) in boxes.items():
    ax.text(cx, cy, label, ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=box_style)

# Path a: X -> M
ax.annotate('', xy=(4.0, 4.3), xytext=(2.0, 2.0),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.5))
ax.text(2.5, 3.5, '$a = -0.543$***', fontsize=10, color='#1565C0',
        fontweight='bold', rotation=42)

# Path b: M -> Y
ax.annotate('', xy=(8.0, 2.0), xytext=(6.0, 4.3),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.5))
ax.text(7.2, 3.5, '$b = -0.457$***', fontsize=10, color='#1565C0',
        fontweight='bold', rotation=-42)

# Path c': X -> Y (direct, dashed)
ax.annotate('', xy=(7.8, 1.5), xytext=(2.2, 1.5),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=2.0,
                           linestyle='dashed'))
ax.text(5.0, 0.9, "$c' = -0.029$ (n.s., $p = 0.71$)", fontsize=10,
        color='#B71C1C', ha='center', fontweight='bold')

# Indirect effect box
indirect_box = mpatches.FancyBboxPatch((3.0, -0.2), 4.0, 0.9,
                                        boxstyle='round,pad=0.2',
                                        facecolor='#E8F5E9', edgecolor='#2E7D32',
                                        linewidth=1.5)
ax.add_patch(indirect_box)
ax.text(5.0, 0.15, 'Indirect $ab = 0.025$, 95% CI [0.007, 0.048]\nFull mediation (SCR)',
        ha='center', va='center', fontsize=9, color='#2E7D32', fontweight='bold')

plt.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'fig2_mediation_path.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(outdir, 'fig2_mediation_path.png'), bbox_inches='tight', dpi=300)
print(f"Saved fig2_mediation_path.pdf and .png to {outdir}")
plt.close()
