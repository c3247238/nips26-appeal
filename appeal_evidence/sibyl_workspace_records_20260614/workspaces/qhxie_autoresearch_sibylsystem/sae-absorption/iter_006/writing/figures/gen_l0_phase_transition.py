"""Figure 3: L0-Absorption Phase Transition."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

# L12-16k data from confound_decomposition_multi_l0.json
l0_vals = [22, 41, 82, 176]
l12_rates = [42.85, 37.49, 14.39, 0.84]
l12_ci_lo = [40.08, 34.81, 12.38, 0.33]
l12_ci_hi = [45.61, 40.17, 16.40, 1.42]

# Cross-layer at L0=82 from first_letter_improved.json
# L10, L12, L20 aggregate absorption rates
l10_rate_82 = 13.88
l12_rate_82 = 15.96
l20_rate_82 = 13.55

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# L12 full sweep
yerr_lo = [r - l for r, l in zip(l12_rates, l12_ci_lo)]
yerr_hi = [h - r for r, h in zip(l12_rates, l12_ci_hi)]
ax.errorbar(l0_vals, l12_rates, yerr=[yerr_lo, yerr_hi],
            fmt='o-', color=COLORS['l12'], linewidth=LINE_WIDTH * 1.2,
            markersize=8, capsize=4, label='L12-16k', zorder=5)

# Cross-layer points at L0=82
ax.plot(82, l10_rate_82, 's', color=COLORS['l10'], markersize=10, zorder=6, label='L10-16k')
ax.plot(82, l20_rate_82, '^', color=COLORS['l20'], markersize=10, zorder=6, label='L20-16k')

# Shade the phase transition region
ax.axvspan(40, 80, alpha=0.08, color='gray', label='Phase transition region')

ax.set_xscale('log')
ax.set_xticks(l0_vals)
ax.set_xticklabels([str(v) for v in l0_vals])
ax.set_xlabel('L0 Operating Point (log scale)')
ax.set_ylabel('Absorption Rate (%)')
ax.set_ylim(-2, 50)
ax.legend(loc='upper right', framealpha=0.9)

# Annotate key values
for l0, rate in zip(l0_vals, l12_rates):
    offset = (0, 12) if l0 != 176 else (0, 12)
    ax.annotate(f'{rate:.1f}%', xy=(l0, rate), xytext=offset,
                textcoords='offset points', ha='center', fontsize=9)

ax.annotate('CV < 10%', xy=(82, max(l10_rate_82, l20_rate_82, l12_rate_82) + 3),
            ha='center', fontsize=8, color='gray', style='italic')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'l0_phase_transition.pdf')
fig.savefig(outpath)
print(f"Saved: {outpath}")
plt.close(fig)
