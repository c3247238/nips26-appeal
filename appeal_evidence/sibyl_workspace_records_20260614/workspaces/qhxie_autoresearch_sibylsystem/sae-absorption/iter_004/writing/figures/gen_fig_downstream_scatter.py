"""Figure 5: Absorption vs downstream scatter (sparse probing F1 + RAVEL TPP)."""
import sys, json, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style_config import setup_style, COLORS, DOUBLE_COL, FIG_HEIGHT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

setup_style()

data_path = pathlib.Path(__file__).parent.parent.parent / "exp" / "results" / "full" / "C3A_saebench_corr.json"
with open(data_path) as f:
    data = json.load(f)

records = data["raw_records"]
corr = data["correlations"]

abs_scores = [r["absorption_score"] for r in records]
sp_f1 = [r["sparse_probing_f1"] for r in records]
tpp = [r["tpp_score"] for r in records]
widths = [r["width_str"] for r in records]

width_colors = {"16k": "#4CAF50", "65k": "#FF9800", "1m": "#D32F2F"}
width_markers = {"16k": "o", "65k": "s", "1m": "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.85, FIG_HEIGHT))

# Panel A: absorption vs sparse probing F1
for w in ["16k", "65k", "1m"]:
    idx = [i for i, wd in enumerate(widths) if wd == w]
    ax1.scatter([abs_scores[i] for i in idx], [sp_f1[i] for i in idx],
                c=width_colors[w], marker=width_markers[w], s=30, alpha=0.7, label=w, edgecolors="white", linewidths=0.5)

# Regression line
x_arr = np.array(abs_scores)
y_arr = np.array(sp_f1)
mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
z = np.polyfit(x_arr[mask], y_arr[mask], 1)
x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
ax1.plot(x_line, np.polyval(z, x_line), color="black", linewidth=1.5, linestyle="--", alpha=0.7)

r_sp = corr["sparse_probing_f1"]["pearson_r"]
ax1.set_xlabel("Absorption score")
ax1.set_ylabel("Sparse probing F1")
ax1.set_title(f"(a) Sparse Probing ($r = {r_sp:.3f}$)")
ax1.legend(title="Width", framealpha=0.9)

# Panel B: absorption vs RAVEL TPP
for w in ["16k", "65k", "1m"]:
    idx = [i for i, wd in enumerate(widths) if wd == w]
    ax2.scatter([abs_scores[i] for i in idx], [tpp[i] for i in idx],
                c=width_colors[w], marker=width_markers[w], s=30, alpha=0.7, label=w, edgecolors="white", linewidths=0.5)

y_arr2 = np.array(tpp)
mask2 = ~np.isnan(x_arr) & ~np.isnan(y_arr2)
z2 = np.polyfit(x_arr[mask2], y_arr2[mask2], 1)
ax2.plot(x_line, np.polyval(z2, x_line), color="black", linewidth=1.5, linestyle="--", alpha=0.7)

r_tpp = corr["ravel_proxy_tpp"]["pearson_r"]
ax2.set_xlabel("Absorption score")
ax2.set_ylabel("RAVEL TPP")
ax2.set_title(f"(b) RAVEL TPP ($r = {r_tpp:.3f}$)")
ax2.legend(title="Width", framealpha=0.9)

fig.tight_layout()
out = pathlib.Path(__file__).parent / "fig_downstream_scatter.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
