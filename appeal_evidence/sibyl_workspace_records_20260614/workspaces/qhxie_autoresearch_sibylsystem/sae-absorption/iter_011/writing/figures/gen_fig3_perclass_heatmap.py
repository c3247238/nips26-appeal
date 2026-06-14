"""
Figure 3: Per-class absorption heatmap for city-continent.
6 continents x 2 SAE widths (16k and 65k). Color intensity = absorption rate,
annotated with n per cell.

Data source: iter_009/exp/results/phase1/absorption_crossdomain.json (per_class data)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import seaborn as sns
except ImportError:
    print("seaborn not available; falling back to matplotlib imshow")
    sns = None

# ── Data from absorption_crossdomain.json ─────────────────────────────────────
continents_order = ["Europe", "Oceania", "Asia", "North America", "Africa", "South America"]

rates_16k = {
    "Europe": 90.2, "Oceania": 52.9, "Asia": 24.4,
    "North America": 19.1, "Africa": 3.9, "South America": 3.9,
}
n_16k = {
    "Europe": 276, "Oceania": 51, "Asia": 324,
    "North America": 241, "Africa": 231, "South America": 207,
}
rates_65k = {
    "Europe": 89.5, "Oceania": 54.9, "Asia": 23.1,
    "North America": 17.8, "Africa": 4.3, "South America": 3.4,
}
n_65k = {
    "Europe": 276, "Oceania": 51, "Asia": 324,
    "North America": 241, "Africa": 231, "South America": 207,
}

data = np.array([[rates_16k[c] for c in continents_order],
                 [rates_65k[c] for c in continents_order]])
n_data = np.array([[n_16k[c] for c in continents_order],
                   [n_65k[c] for c in continents_order]])

annot = np.empty_like(data, dtype=object)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        annot[i, j] = f"{data[i, j]:.1f}%\n(n={n_data[i, j]})"

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 11, "font.family": "serif"})

fig, ax = plt.subplots(figsize=(8, 2.8))

if sns is not None:
    sns.heatmap(
        data, annot=annot, fmt="", cmap="YlOrRd",
        vmin=0, vmax=100, linewidths=0.8, linecolor="white",
        xticklabels=continents_order, yticklabels=["16k", "65k"],
        ax=ax, cbar_kws={"label": "Absorption Rate (%)", "shrink": 0.8},
        annot_kws={"fontsize": 9},
    )
else:
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(continents_order)))
    ax.set_xticklabels(continents_order)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["16k", "65k"])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Absorption Rate (%)", shrink=0.8)

ax.set_xlabel("")
ax.set_ylabel("SAE Width", fontsize=11)
ax.set_title("Per-Continent Absorption Rates at Layer 24", fontsize=12, pad=10)
ax.tick_params(axis="x", rotation=25)

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"fig3_perclass_heatmap.{ext}"),
                dpi=300, bbox_inches="tight")
print(f"Saved fig3_perclass_heatmap.pdf/png to {out_dir}")
plt.close()
