"""Figure 6: DAS(k=1) and DAS(k=3) vs SAE width."""
import sys, json, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style_config import setup_style, COLORS, SINGLE_COL, FIG_HEIGHT
import matplotlib.pyplot as plt

setup_style()

data_path = pathlib.Path(__file__).parent.parent.parent / "exp" / "results" / "full" / "C1D_das_vs_width.json"
with open(data_path) as f:
    data = json.load(f)

mdw = data["mean_das_by_width"]
widths = ["24k", "49k", "98k"]
x = np.arange(len(widths))

k1_means = [mdw[w]["mean_das_k1"] for w in widths]
k1_stds = [mdw[w]["std_das_k1"] for w in widths]
k3_means = [mdw[w]["mean_das_k3"] for w in widths]
k3_stds = [mdw[w]["std_das_k3"] for w in widths]

fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.8, FIG_HEIGHT))

ax.errorbar(x - 0.1, k1_means, yerr=k1_stds, fmt="o-", color=COLORS["das_k1"],
            capsize=4, linewidth=1.5, markersize=6, label="DAS($k$=1)")
ax.errorbar(x + 0.1, k3_means, yerr=k3_stds, fmt="s-", color=COLORS["das_k3"],
            capsize=4, linewidth=1.5, markersize=6, label="DAS($k$=3)")

ax.set_xticks(x)
ax.set_xticklabels(widths)
ax.set_xlabel("SAE width (latents)")
ax.set_ylabel("Mean DAS score")
ax.set_ylim(0, 0.65)
ax.legend(framealpha=0.9)
ax.set_title("Width Paradox: DAS vs. SAE Width")

out = pathlib.Path(__file__).parent / "fig_das_width.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
