"""Figure 3: Task-agnostic vs. First-Letter Absorption Correlation."""
import json
import matplotlib.pyplot as plt
import numpy as np
from style_config import set_paper_style, COLORS

set_paper_style()

with open("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/results/e3_validation_correlation_results.json") as f:
    data = json.load(f)

pts = data["results"]
x = np.array([p["first_letter_absorption"] for p in pts])
y = np.array([p["task_agnostic_absorption"] for p in pts])
families = [p["family"] for p in pts]
labels = [p["sae_id"].replace("blocks.", "L").replace(".hook_", " ") for p in pts]

fig, ax = plt.subplots(figsize=(5.5, 4.2))

unique_fams = sorted(set(families))
for fam in unique_fams:
    mask = [f == fam for f in families]
    xm = x[mask]
    ym = y[mask]
    color = COLORS.get(fam.lower(), "#333333")
    ax.scatter(xm, ym, s=80, c=color, alpha=0.8, edgecolors="white", linewidth=0.5, label=fam.replace("_", " ").title(), zorder=3)

# Annotate each point
for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)

ax.set_xlabel("First-letter absorption rate ($\\alpha_{\\mathrm{FL}}$")
ax.set_ylabel("Task-agnostic absorption rate ($\\alpha_{\\mathrm{TA}}$")
ax.set_title("Weak negative correlation between absorption metrics")
ax.legend(loc="upper right", frameon=False)

pr = data["correlation"]["pearson_r"]
pv = data["correlation"]["p_value"]
ax.text(0.05, 0.95, f"Pearson $r = {pr:.2f}$, $p = {pv:.3f}$", transform=ax.transAxes, fontsize=10, verticalalignment="top")

plt.tight_layout()
fig.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/writing/figures/fig3_task_agnostic.pdf")
print("Saved fig3_task_agnostic.pdf")
