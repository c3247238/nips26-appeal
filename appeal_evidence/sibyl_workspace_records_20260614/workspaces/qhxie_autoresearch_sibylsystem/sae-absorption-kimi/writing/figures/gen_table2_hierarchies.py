"""Generate Table 2: WordNet Semantic Hierarchies."""
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Data: parent, children, probe AUROC
hierarchies = [
    ("building", "house, school, library", "1.00"),
    ("container", "box, bag, cup", "1.00"),
    ("tool", "fork, comb, rake", "1.00"),
    ("room", "cell, court, hall", "1.00"),
    ("device", "fan, key", "1.00"),
    ("book", "album, journal", "1.00"),
    ("tree", "ash, poon", "1.00"),
    ("food", "fish, water", "1.00"),
    ("fruit", "berry, seed", "1.00"),
    ("animal", "pet, male", "1.00"),
]

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.axis("off")

# Column headers
col_headers = ["Parent", "Children", "Probe AUROC"]
col_widths = [0.22, 0.50, 0.22]

# Table data
table_data = [[h[0], h[1], h[2]] for h in hierarchies]

table = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    loc="center",
    cellLoc="left",
    colColours=["#e8e8e8"] * 3,
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Style header
for j in range(3):
    cell = table[(0, j)]
    cell.set_text_props(fontweight="bold")
    cell.set_facecolor("#d0d0d0")

# Alternate row colors
for i in range(1, len(hierarchies) + 1):
    for j in range(3):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        else:
            cell.set_facecolor("#ffffff")

# Center the AUROC column
for i in range(len(hierarchies) + 1):
    table[(i, 2)].set_text_props(ha="center")

plt.title("Table 2: WordNet Semantic Hierarchies", fontweight="bold", fontsize=11, pad=10)
fig.text(0.5, 0.02, "All hierarchies achieved probe AUROC = 1.0 on the ground-truth residual-stream probe.",
         ha="center", fontsize=8, style="italic")

plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/table2_hierarchies.pdf", format="pdf")
plt.close()
print("Saved table2_hierarchies.pdf")
