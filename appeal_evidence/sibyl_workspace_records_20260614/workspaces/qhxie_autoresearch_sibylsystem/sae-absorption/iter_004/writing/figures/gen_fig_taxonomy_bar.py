"""Figure 4: Absorption taxonomy stacked bar chart across widths."""
import sys, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style_config import setup_style, COLORS, SINGLE_COL, FIG_HEIGHT
import matplotlib.pyplot as plt
import numpy as np

setup_style()

data_path = pathlib.Path(__file__).parent.parent.parent / "exp" / "results" / "full" / "C2D_taxonomy.json"
with open(data_path) as f:
    data = json.load(f)

cw = data["cross_width_analysis"]
widths = ["24k", "49k", "98k"]
type_i = [cw[w]["fractions"]["Type_I"] for w in widths]
type_ii = [cw[w]["fractions"]["Type_II"] for w in widths]
type_iii = [cw[w]["fractions"]["Type_III"] for w in widths]
none_f = [cw[w]["fractions"]["None"] for w in widths]

x = np.arange(len(widths))
bar_w = 0.5

fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.8, FIG_HEIGHT))

ax.bar(x, type_i, bar_w, color=COLORS["type_i"], label="Type I (full)")
ax.bar(x, type_ii, bar_w, bottom=type_i, color=COLORS["type_ii"], label="Type II (partial)")
bottom_iii = [a + b for a, b in zip(type_i, type_ii)]
ax.bar(x, type_iii, bar_w, bottom=bottom_iii, color=COLORS["type_iii"], label="Type III (distributed)")
bottom_none = [a + b for a, b in zip(bottom_iii, type_iii)]
ax.bar(x, none_f, bar_w, bottom=bottom_none, color=COLORS["none"], label="None")

# Canonical rate reference
ax.axhline(y=0.354, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
ax.text(2.3, 0.37, "Chanin rate\n(35.4%)", fontsize=7, ha="left", color="black", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(widths)
ax.set_xlabel("SAE width (latents)")
ax.set_ylabel("Fraction of 26 letters")
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
ax.set_title("Absorption Taxonomy Across SAE Widths")

# Annotate comprehensive rates
for i, w in enumerate(widths):
    comp = cw[w]["comprehensive_rate"]
    ax.text(i, comp + 0.02, f"{comp:.1%}", ha="center", fontsize=7, fontweight="bold")

out = pathlib.Path(__file__).parent / "fig_taxonomy_bar.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
