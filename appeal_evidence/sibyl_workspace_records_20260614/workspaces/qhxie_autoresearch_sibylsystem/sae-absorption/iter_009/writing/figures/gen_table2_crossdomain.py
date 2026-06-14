"""Generate Table 2: Cross-domain absorption rates at L24 (data table as PDF)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

# Data from full/phase1_absorption_crossdomain.json + phase1_absorption_firstletter.json
table_data = [
    # hierarchy, SAE, rate, CI_low, CI_high, probe_F1, N, N_FN
    ['First-letter',   '16k', 0.2707, 0.245, 0.295, '1.00', 1138, 308],
    ['First-letter',   '65k', 0.1770, 0.155, 0.200, '1.00', 1130, 200],
    ['City-continent', '16k', 0.3143, 0.289, 0.339, '0.87', 1330, 418],
    ['City-continent', '65k', 0.3128, 0.287, 0.338, '0.87', 1330, 416],
    ['City-language',  '16k', 0.1156, 0.097, 0.135, '0.82', 1073, 124],
    ['City-language',  '65k', 0.0774, 0.062, 0.094, '0.82', 1073,  83],
    ['City-country',   '16k', 0.4510, 0.422, 0.479, '0.73*', 1142, 515],
    ['City-country',   '65k', 0.3292, 0.302, 0.356, '0.73*', 1142, 376],
]

col_labels = ['Hierarchy', 'Width', 'Abs. Rate', '95% CI', 'Probe $F_1$', '$N$', '$N_{FN}$']

fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL * 0.85, 3.2))
ax.axis('off')

cell_text = []
for row in table_data:
    h, w, rate, ci_lo, ci_hi, f1, n, nfn = row
    cell_text.append([
        h, w,
        f'{rate*100:.1f}%',
        f'[{ci_lo*100:.1f}, {ci_hi*100:.1f}]',
        f'{f1}',
        f'{n:,}',
        f'{nfn:,}'
    ])

table = ax.table(cellText=cell_text, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.18, 0.08, 0.12, 0.17, 0.12, 0.10, 0.10])

table.auto_set_font_size(False)
table.set_fontsize(FONT_SIZE - 1)
table.scale(1.0, 1.5)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#E0E0E0')
    cell.set_text_props(fontweight='bold')

# Bold the highest rate
for i, row in enumerate(table_data):
    if row[2] == 0.4510:  # highest rate
        table[i+1, 2].set_text_props(fontweight='bold')
    if row[2] == 0.0774:  # lowest rate
        table[i+1, 2].set_text_props(fontstyle='italic')

# Alternate row colors
for i in range(len(table_data)):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i+1, j].set_facecolor('#F5F5F5')

plt.title('Cross-Domain Absorption Rates at Layer 24 (Gemma 2 2B, JumpReLU SAEs)',
          fontsize=FONT_SIZE, pad=20)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'table2_crossdomain.pdf')
plt.savefig(outpath)
plt.close()
print(f'Saved: {outpath}')
