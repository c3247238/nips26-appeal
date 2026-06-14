"""
Generate Figure 4: code-boundary failure breakdown on HumanEval.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import (  # noqa: E402
    COLORS,
    FIG_HEIGHT,
    FIG_WIDTH,
    LEGEND_SIZE,
    TITLE_SIZE,
)


FIG_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(FIG_DIR))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "exp", "results")


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


data = load_json("diag_humaneval_guard_boundary.json")
methods = data["methods"]
method_colors = {
    "Standard": COLORS["standard"],
    "Entropy/TIGER Ungated Revision": "#A1887F",
    "Gated TIGER": COLORS["highlight"],
}
metrics = [
    ("pass_at_1", "pass@1"),
    ("syntax_failure_rate", "syntax failure"),
    ("runtime_failure_rate", "runtime failure"),
]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * 1.08))
x = np.arange(len(metrics))
width = 0.24

for offset, method in zip(np.linspace(-width, width, len(methods)), methods):
    values = [method[key] for key, _ in metrics]
    ax.bar(
        x + offset,
        values,
        width=width,
        color=method_colors[method["method"]],
        edgecolor="white",
        linewidth=0.8,
        label=method["method"],
    )

ax.set_xticks(x, [label for _, label in metrics])
ax.set_ylabel("Rate")
ax.set_ylim(0, 0.8)
ax.set_title("Figure 4: Code boundary failure breakdown", fontsize=TITLE_SIZE)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(
    frameon=False,
    fontsize=LEGEND_SIZE - 0.2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
)

delta = data["deltas"]
gated = next(item for item in methods if item["method"] == "Gated TIGER")
fig.text(
    0.11,
    0.02,
    (
        f"Guard open rate = {gated['gate_open_rate']:.2f}; avg syntax guard cost = "
        f"{gated['syntax_guard_avg_ms']:.3f} ms. "
        f"Syntax improves by {-delta['gated_vs_ungated_syntax_failure']:.2f}, "
        f"but runtime failure worsens by {delta['gated_vs_ungated_runtime_failure']:.2f}."
    ),
    ha="left",
    va="bottom",
    fontsize=LEGEND_SIZE - 0.1,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0"),
)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
output_path = os.path.join(FIG_DIR, "fig4_code_boundary.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"Saved to {output_path}")
