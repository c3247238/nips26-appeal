"""
Generate Figure 1: Layer x Hierarchy Absorption Heatmap.

Shows absorption rate across 4 layers x 4 hierarchies x 2 SAE widths,
communicating the paper's central finding: absorption varies dramatically
by both layer and hierarchy type.

Data source: phase1 absorption results (first-letter + cross-domain).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data from experimental results ---
# First-letter absorption rates (all layers, both widths)
# Source: pilots/phase1_absorption_firstletter.json
fl_data = {
    "L6":  {"16k": 2.4, "65k": 2.4},
    "L12": {"16k": 5.7, "65k": 9.2},
    "L18": {"16k": 2.2, "65k": 4.5},
    "L24": {"16k": 34.5, "65k": 25.5},
}

# Cross-domain absorption rates at L24 only (best probe layer for RAVEL)
# Source: pilots/phase1_absorption_crossdomain.json
cd_data_l24 = {
    "city-continent": {"16k": 35.8, "65k": 26.0},
    "city-country":   {"16k": 18.5, "65k": 12.7},
    "city-language":   {"16k": 13.6, "65k": 13.6},
}

# Probe F1 per hierarchy at L24
probe_f1 = {
    "first-letter":   0.971,
    "city-continent":  0.843,
    "city-country":    0.789,
    "city-language":   0.823,
}

# Build matrices for 16k and 65k heatmaps
layers = ["L6", "L12", "L18", "L24"]
hierarchies = ["first-letter", "city-continent", "city-country", "city-language"]

# RAVEL hierarchies only measured at L24; use NaN for other layers
def build_matrix(width):
    mat = np.full((len(hierarchies), len(layers)), np.nan)
    for j, layer in enumerate(layers):
        # First-letter: measured at all layers
        mat[0, j] = fl_data[layer][width]
    # RAVEL: only L24
    for i, h in enumerate(hierarchies[1:], start=1):
        mat[i, 3] = cd_data_l24[h][width]
    return mat

mat_16k = build_matrix("16k")
mat_65k = build_matrix("65k")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

for ax, mat, title in zip(axes, [mat_16k, mat_65k], ["SAE width 16k", "SAE width 65k"]):
    # Mask NaN cells
    masked = np.ma.masked_where(np.isnan(mat), mat)
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=40, aspect="auto")

    # Annotate cells
    for i in range(len(hierarchies)):
        for j in range(len(layers)):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if val > 25 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
            else:
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=9, color="#999999")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_yticks(range(len(hierarchies)))
    hierarchy_labels = [
        f"first-letter\n(F1={probe_f1['first-letter']:.2f})",
        f"city-continent\n(F1={probe_f1['city-continent']:.2f})",
        f"city-country\n(F1={probe_f1['city-country']:.2f})",
        f"city-language\n(F1={probe_f1['city-language']:.2f})",
    ]
    if ax == axes[0]:
        ax.set_yticklabels(hierarchy_labels, fontsize=9)
    ax.set_xlabel("Model Layer", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hierarchies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

# Colorbar
cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.04)
cbar.set_label("Absorption Rate (%)", fontsize=10)

fig.suptitle("Feature Absorption Varies by Layer and Hierarchy Type",
             fontsize=12, fontweight="bold", y=1.02)

# Note about missing data
fig.text(0.5, -0.06,
         "RAVEL hierarchies (city-*) measured at L24 only (best probe layer). "
         "Dashes (--) indicate layers without RAVEL probes of sufficient quality.",
         ha="center", fontsize=8, color="#666666", style="italic")

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), "fig1_heatmap.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300)
print(f"Saved: {outpath}")
plt.close()
