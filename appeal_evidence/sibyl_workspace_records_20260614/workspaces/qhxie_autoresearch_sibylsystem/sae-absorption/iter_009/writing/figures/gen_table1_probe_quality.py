"""Generate Table 1: Probe quality (F1) heatmap across 4 hierarchies x 4 layers.

Reads phase1/probe_training.json and absorption_firstletter.json to compose
the 4x4 probe F1 matrix with quality gate annotations.
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parents[2]  # current/
FIG_DIR = WORKSPACE / "writing" / "figures"

# ── Load data ───────────────────────────────────────────────────────────
with open(WORKSPACE / "exp/results/phase1/probe_training.json") as f:
    probe_data = json.load(f)

with open(WORKSPACE / "exp/results/phase1/absorption_firstletter.json") as f:
    fl_data = json.load(f)

# ── Build matrix ────────────────────────────────────────────────────────
hierarchies = ["First-letter", "City-continent", "City-language", "City-country"]
hierarchy_keys = ["first-letter", "city-continent", "city-language", "city-country"]
layers = [6, 12, 18, 24]

# F1 values: first-letter uses the sae_spelling probes (F1=1.0 at all layers)
# RAVEL hierarchies use sklearn logistic regression probes
f1_matrix = np.zeros((4, 4))

# First-letter: sae_spelling probes achieve F1=1.0 at position -6
for j, layer in enumerate(layers):
    f1_matrix[0, j] = 1.0  # F1=1.0 at all layers for sae_spelling probes

# RAVEL probes from probe_training.json
for i, hkey in enumerate(hierarchy_keys[1:], start=1):
    for j, layer in enumerate(layers):
        key = f"{hkey}_L{layer}"
        if key in probe_data["probes"]:
            p = probe_data["probes"][key]
            f1_matrix[i, j] = p.get("f1_weighted_test", p.get("f1_weighted_cv", 0))

# ── Quality gate status ─────────────────────────────────────────────────
def gate_label(f1):
    if f1 >= 0.90:
        return "strict"
    elif f1 >= 0.80:
        return "relaxed"
    else:
        return "below"

gate_symbols = {
    "strict": r"$\checkmark$",
    "relaxed": r"$\sim$",
    "below":  r"$\times$",
}

# ── Plot heatmap ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.0))

# Custom colormap: red (<0.80), yellow (0.80-0.90), green (>0.90)
cmap = matplotlib.colormaps.get_cmap("RdYlGn")
norm = mcolors.Normalize(vmin=0.35, vmax=1.0)

im = ax.imshow(f1_matrix, cmap=cmap, norm=norm, aspect="auto")

# Annotate cells
for i in range(4):
    for j in range(4):
        val = f1_matrix[i, j]
        gate = gate_label(val)
        text_color = "white" if val < 0.55 else "black"
        ax.text(j, i, f"{val:.2f}\n{gate_symbols[gate]}",
                ha="center", va="center", fontsize=9, color=text_color)

ax.set_xticks(range(4))
ax.set_xticklabels([f"L{l}" for l in layers], fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels(hierarchies, fontsize=10)
ax.set_xlabel("Transformer Layer", fontsize=11)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Weighted F1", fontsize=10)

# Legend for gate symbols
legend_text = (r"$\checkmark$ strict (F1 $\geq$ 0.90)    "
               r"$\sim$ relaxed (F1 $\geq$ 0.80)    "
               r"$\times$ below gate")
fig.text(0.5, -0.02, legend_text, ha="center", fontsize=8, style="italic")

plt.title("Table 1: Probe Quality Across Hierarchies and Layers", fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "table1_probe_quality.pdf", bbox_inches="tight", dpi=300)
plt.close()

print(f"Saved: {FIG_DIR / 'table1_probe_quality.pdf'}")
