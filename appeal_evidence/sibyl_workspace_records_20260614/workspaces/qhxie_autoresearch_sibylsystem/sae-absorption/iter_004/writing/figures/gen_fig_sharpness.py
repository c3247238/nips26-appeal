"""Figure 2: Sharpness test — absorption rate vs alpha_ij bins."""
import sys, json, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style_config import setup_style, COLORS, SINGLE_COL, FIG_HEIGHT
import matplotlib.pyplot as plt

setup_style()

data_path = pathlib.Path(__file__).parent.parent.parent / "exp" / "results" / "full" / "C1B_lv_validation.json"
with open(data_path) as f:
    data = json.load(f)

sharpness = data["lv_sharpness"]
centers = sharpness["bin_data"]["bin_centers"]
rates = sharpness["bin_data"]["bin_rates"]
counts = sharpness["bin_data"]["bin_counts"]

# Filter bins with data (non-null rates and >= 5 observations for reliability)
valid = [(c, r, n) for c, r, n in zip(centers, rates, counts) if r is not None and n >= 5]
x_valid = np.array([v[0] for v in valid])
y_valid = np.array([v[1] for v in valid])
n_valid = np.array([v[2] for v in valid])

# Sigmoid and linear fits from data
sig_k = sharpness["sigmoid_params"]["k"]
sig_x0 = sharpness["sigmoid_params"]["x0"]
lin_a = sharpness["linear_params"]["a"]
lin_b = sharpness["linear_params"]["b"]

x_fit = np.linspace(0, 2.0, 200)
y_sigmoid = 1.0 / (1.0 + np.exp(-sig_k * (x_fit - sig_x0)))
y_linear = lin_a * x_fit + lin_b

fig, ax = plt.subplots(figsize=(SINGLE_COL, FIG_HEIGHT))

# Bar chart for empirical rates
bar_width = 0.08
ax.bar(x_valid, y_valid, width=bar_width, alpha=0.6, color="#78909C", label="Empirical rate", edgecolor="white")

# Overlay fits
ax.plot(x_fit, y_linear, color=COLORS["cosine"], linewidth=2, linestyle="--", label=f"Linear (AIC = {sharpness['aic_linear']:.1f})")
ax.plot(x_fit, y_sigmoid, color=COLORS["lv"], linewidth=2, label=f"Sigmoid (AIC = {sharpness['aic_sigmoid']:.1f})")

# Mark alpha = 1 threshold
ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ax.text(1.02, 0.85, r"$\alpha_{ij}=1$", fontsize=8, color="gray")

ax.set_xlabel(r"Competition coefficient $\alpha_{ij}$")
ax.set_ylabel("Absorption rate (Chanin metric)")
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", framealpha=0.9)
ax.set_title("Sharpness Test: No Threshold Transition at $\\alpha_{ij} \\approx 1$")

out = pathlib.Path(__file__).parent / "fig_sharpness.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
