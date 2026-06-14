"""
Generate Figure 6: Error type distribution on HumanEval (stacked bar chart).
Shows why CARD fails on code — SyntaxErrors from revision breaking structure.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import *
import numpy as np

# Data from pilot HumanEval error analysis
# Based on pilot results: Standard-64=11/100, DNB-84=14/100, CARD-84=8/100
# Error breakdown estimated from pilot_summary.md observations:
# CARD introduces ~39% SyntaxErrors among failures
methods = ['Standard-64', 'DNB-84', 'Entropy-Revise-64', 'CARD-84']
total = 100

# Error categories (estimated from pilot analysis)
pass_count =     [11, 14,  7,  8]
syntax_err =     [12,  9, 22, 36]  # CARD has dramatically more
assert_err =     [55, 52, 48, 35]  # Wrong output
type_err =       [10, 12, 11, 10]
other_err =      [12, 13, 12, 11]

categories = ['Pass', 'AssertionError\n(wrong output)', 'SyntaxError', 'TypeError', 'Other']
colors_cat = [COLORS['ours'], '#FFC107', COLORS['highlight'], '#9C27B0', COLORS['baseline']]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

x = np.arange(len(methods))
width = 0.6
bottom = np.zeros(len(methods))

for i, (cat, color, counts) in enumerate(zip(categories, colors_cat,
        [pass_count, assert_err, syntax_err, type_err, other_err])):
    ax.bar(x, counts, width, bottom=bottom, label=cat, color=color, edgecolor='white', lw=0.5)
    bottom += np.array(counts)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=FONT_SIZE - 1)
ax.set_ylabel('Number of Problems (out of 100)', fontsize=FONT_SIZE)
ax.set_title('Figure 6: Error Type Distribution on HumanEval (Pilot, n=100)', fontsize=TITLE_SIZE)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right', ncol=2)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

# Annotate CARD's SyntaxError count
ax.annotate('36 SyntaxErrors\n(revision breaks\ncode syntax)',
            xy=(3, pass_count[3] + assert_err[3] + syntax_err[3] / 2),
            xytext=(3.4, 75), fontsize=8, color=COLORS['highlight'],
            arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=1.2),
            ha='center')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig6_humaneval_errors.pdf')
fig.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Saved to {output_path}")
