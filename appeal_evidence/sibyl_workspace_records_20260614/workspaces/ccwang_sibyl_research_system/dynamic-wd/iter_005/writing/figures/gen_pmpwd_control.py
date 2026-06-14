"""Generate Figure 3: PMP-WD Control Diagram."""
import sys
sys.path.insert(0, '.')
from style_config import setup_style, COLORS, FONT_SIZE, FIG_WIDTH, DPI

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

setup_style()
plt.rcParams['axes.grid'] = False

fig, ax = plt.subplots(figsize=(FIG_WIDTH + 1.5, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# ── Helpers ──────────────────────────────────────────────────
def draw_block(ax, x, y, w, h, text, color='#2196F3', text_color='white',
               fontsize=None):
    fs = fontsize or FONT_SIZE - 1
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor='#333333',
                         linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=text_color, fontweight='bold', zorder=4)
    return x, y, w, h

def draw_arrow(ax, x1, y1, x2, y2, color='#333333', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def draw_circle_node(ax, cx, cy, r, text, color='#333333'):
    circle = plt.Circle((cx, cy), r, fill=False, edgecolor=color,
                        linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=FONT_SIZE - 2, color=color, fontweight='bold', zorder=4)

# ── Row 1: PMP-WD (Closed-Loop) ─────────────────────────────
row1_y = 4.1
row_h = 0.7
c_pmp = '#1565C0'

ax.text(0.15, row1_y + row_h + 0.25,
        'PMP-WD  (Closed-Loop State-Feedback)',
        fontsize=FONT_SIZE + 1, fontweight='bold', color=c_pmp,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD',
                  edgecolor=c_pmp, alpha=0.9))

# Blocks
draw_block(ax, 0.3, row1_y, 1.6, row_h, r'Measure $\hat{\rho}_t$',
           color=c_pmp)
# Comparator
draw_circle_node(ax, 2.7, row1_y + row_h/2, 0.25, r'$\Sigma$', color=c_pmp)
draw_block(ax, 3.5, row1_y, 1.3, row_h, r'Gain $\kappa$', color=c_pmp)
draw_block(ax, 5.4, row1_y, 1.8, row_h, r'Clip $[0, \lambda_{\max}]$',
           color=c_pmp)
draw_block(ax, 7.8, row1_y, 1.5, row_h, r'$\lambda^*(t)$',
           color='#0D47A1', text_color='white')

# Arrows
draw_arrow(ax, 1.9, row1_y + row_h/2, 2.45, row1_y + row_h/2, color=c_pmp)
draw_arrow(ax, 2.95, row1_y + row_h/2, 3.5, row1_y + row_h/2, color=c_pmp)
draw_arrow(ax, 4.8, row1_y + row_h/2, 5.4, row1_y + row_h/2, color=c_pmp)
draw_arrow(ax, 7.2, row1_y + row_h/2, 7.8, row1_y + row_h/2, color=c_pmp)

# rho* reference input
ax.annotate(r'$\rho^*$', xy=(2.7, row1_y + row_h/2 + 0.25),
            xytext=(2.7, row1_y + row_h + 0.6),
            fontsize=FONT_SIZE, ha='center', color=c_pmp, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=c_pmp, lw=1.3))

# Error signal label
ax.text(3.1, row1_y + row_h + 0.05, r'$\rho^* - \hat{\rho}_t$',
        fontsize=FONT_SIZE - 2, ha='center', color=c_pmp, style='italic')

# Feedback loop (bottom)
fb_y = row1_y - 0.35
ax.annotate('', xy=(0.3, row1_y + row_h/2),
            xytext=(0.3, fb_y),
            arrowprops=dict(arrowstyle='->', color=c_pmp, lw=1.2,
                            connectionstyle='arc3,rad=0'))
ax.plot([0.3, 8.55], [fb_y, fb_y], color=c_pmp, lw=1.2, ls='--')
ax.plot([8.55, 8.55], [fb_y, row1_y], color=c_pmp, lw=1.2, ls='--')
ax.text(4.5, fb_y - 0.2, 'feedback: training state '
        r'$(\theta_t, g_t)$',
        fontsize=FONT_SIZE - 2, ha='center', color=c_pmp, style='italic')

# ── Row 2: CWD (Binary Mask) ────────────────────────────────
row2_y = 2.2
c_cwd = '#C62828'

ax.text(0.15, row2_y + row_h + 0.25,
        'CWD  (Open-Loop Binary Mask)',
        fontsize=FONT_SIZE + 1, fontweight='bold', color=c_cwd,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE',
                  edgecolor=c_cwd, alpha=0.9))

draw_block(ax, 0.3, row2_y, 2.3, row_h,
           r'$\mathrm{sign}(\theta)\!=\!\mathrm{sign}(u)$?',
           color=c_cwd)
draw_block(ax, 3.5, row2_y, 2.0, row_h, 'Binary mask\n1 or 0',
           color=c_cwd, fontsize=FONT_SIZE - 2)
draw_block(ax, 6.5, row2_y, 2.0, row_h, r'$\lambda \times \mathrm{mask}$',
           color='#B71C1C', text_color='white')

draw_arrow(ax, 2.6, row2_y + row_h/2, 3.5, row2_y + row_h/2, color=c_cwd)
draw_arrow(ax, 5.5, row2_y + row_h/2, 6.5, row2_y + row_h/2, color=c_cwd)

ax.text(9.0, row2_y + row_h/2, 'no feedback',
        fontsize=FONT_SIZE - 2, ha='left', va='center',
        color='#888888', style='italic')

# ── Row 3: Cosine Schedule (Feedforward) ─────────────────────
row3_y = 0.5
c_cos = '#616161'

ax.text(0.15, row3_y + row_h + 0.25,
        'Cosine Schedule  (Feedforward)',
        fontsize=FONT_SIZE + 1, fontweight='bold', color=c_cos,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5',
                  edgecolor=c_cos, alpha=0.9))

draw_block(ax, 0.3, row3_y, 1.5, row_h, 'Time $t$',
           color=c_cos)
draw_block(ax, 2.8, row3_y, 3.0, row_h,
           r'$\frac{1}{2}(1+\cos(\pi t / T))$',
           color=c_cos)
draw_block(ax, 6.5, row3_y, 2.4, row_h,
           r'$\lambda \times \mathrm{sched}(t)$',
           color='#424242', text_color='white')

draw_arrow(ax, 1.8, row3_y + row_h/2, 2.8, row3_y + row_h/2, color=c_cos)
draw_arrow(ax, 5.8, row3_y + row_h/2, 6.5, row3_y + row_h/2, color=c_cos)

ax.text(9.0, row3_y + row_h/2, 'ignores state',
        fontsize=FONT_SIZE - 2, ha='left', va='center',
        color='#888888', style='italic')

plt.tight_layout()
fig.savefig('pmpwd_control.pdf', bbox_inches='tight')
fig.savefig('pmpwd_control.png', bbox_inches='tight', dpi=DPI)
print("Saved pmpwd_control.pdf and .png")
plt.close()
