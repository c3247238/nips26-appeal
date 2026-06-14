"""Generate Table 3: Non-Hierarchy Control Pairs."""
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

# Data: word pair, relationship type
pairs = [
    ("big -- large", "Synonymy"),
    ("fast -- quick", "Synonymy"),
    ("begin -- start", "Synonymy"),
    ("doctor -- hospital", "Co-occurrence"),
    ("student -- school", "Co-occurrence"),
    ("river -- water", "Thematic"),
    ("fire -- heat", "Thematic"),
    ("sun -- light", "Thematic"),
    ("tree -- wood", "Compositional"),
    ("stone -- rock", "Synonymy"),
]

fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.axis("off")

col_headers = ["Word Pair", "Relationship Type"]
table_data = [[p[0], p[1]] for p in pairs]

table = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    loc="center",
    cellLoc="left",
    colColours=["#e8e8e8"] * 2,
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Style header
for j in range(2):
    cell = table[(0, j)]
    cell.set_text_props(fontweight="bold")
    cell.set_facecolor("#d0d0d0")

# Alternate row colors
for i in range(1, len(pairs) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        else:
            cell.set_facecolor("#ffffff")

plt.title("Table 3: Non-Hierarchy Control Pairs", fontweight="bold", fontsize=11, pad=10)
fig.text(0.5, 0.02, "Semantically correlated pairs without parent-child containment structure.",
         ha="center", fontsize=8, style="italic")

plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/table3_nonhierarchy.pdf", format="pdf")
plt.close()
print("Saved table3_nonhierarchy.pdf")
