"""
Generate Figure 2: Pareto view under honest compute on GSM8K.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import (  # noqa: E402
    COLORS,
    FIG_HEIGHT,
    FIG_WIDTH,
    FONT_SIZE,
    LEGEND_SIZE,
    TITLE_SIZE,
)


FIG_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(FIG_DIR))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "exp", "results")

METHOD_COLORS = {
    "Standard-64": COLORS["standard"],
    "DNB-84": COLORS["dnb"],
    "Prophet-64": "#5C6BC0",
    "CORE-proxy-64": COLORS["highlight"],
    "Entropy-Revise-64+3": COLORS["entropy_revise"],
    "TIGER-Instability-64+3": "#8E24AA",
}


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def short_method_name(name: str) -> str:
    return (
        name.replace("-64+3", "")
        .replace("-64", "")
        .replace("Entropy-Revise", "Entropy")
        .replace("TIGER-Instability", "TIGER")
        .replace("CORE-proxy", "CORE")
    )


data = load_json("diag_compute_curve_gsm8k.json")
methods = sorted(data["methods"], key=lambda item: item["latency_sec"])
frontier_names = set(data["pareto_frontier"])

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * 1.12))

frontier = [item for item in methods if item["method"] in frontier_names]
frontier = sorted(frontier, key=lambda item: item["latency_sec"])
ax.plot(
    [item["latency_sec"] for item in frontier],
    [item["accuracy"] for item in frontier],
    color="#455A64",
    lw=1.8,
    ls="--",
    alpha=0.8,
    zorder=1,
    label="Actual-compute frontier",
)

for item in methods:
    color = METHOD_COLORS[item["method"]]
    marker = "o" if item["method"] in frontier_names else "s"
    size = 170 if item["method"] == "CORE-proxy-64" else 120
    ax.scatter(
        item["latency_sec"],
        item["accuracy"],
        s=size,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    offsets = {
        "Prophet-64": (6, 6),
        "Standard-64": (6, 4),
        "DNB-84": (18, -4),
        "Entropy-Revise-64+3": (8, 18),
        "TIGER-Instability-64+3": (12, -8),
        "CORE-proxy-64": (-58, -46),
    }
    ax.annotate(
        f"{short_method_name(item['method'])}\nNFE={item['actual_nfe']:.0f}",
        (item["latency_sec"], item["accuracy"]),
        textcoords="offset points",
        xytext=offsets[item["method"]],
        fontsize=LEGEND_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )

ax.set_xlabel("Latency (sec)")
ax.set_ylabel("GSM8K accuracy")
ax.set_title("Figure 2: Pareto view under honest compute", fontsize=TITLE_SIZE)
ax.set_xlim(145, 498)
ax.set_ylim(0.334, 0.466)
ax.grid(True, alpha=0.25)

core = next(item for item in methods if item["method"] == "CORE-proxy-64")
entropy = next(item for item in methods if item["method"] == "Entropy-Revise-64+3")
ax.annotate(
    "",
    xy=(core["latency_sec"], core["accuracy"]),
    xytext=(entropy["latency_sec"], entropy["accuracy"]),
    arrowprops=dict(arrowstyle="->", color="#616161", lw=1.2),
)
ax.text(
    0.03,
    0.97,
    (
        f"CORE keeps the best raw accuracy ({core['accuracy']:.2f}) but needs "
        f"{core['latency_sec']:.1f}s and actual NFE={core['actual_nfe']:.0f}.\n"
        f"Entropy stays {data['key_comparisons']['core_vs_entropy']['accuracy_delta']:.2f} lower "
        f"while saving {data['key_comparisons']['core_vs_entropy']['latency_delta_sec']:.1f}s."
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=LEGEND_SIZE,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D0D0"),
)

legend_handles = []
for method_name in ["Standard-64", "Prophet-64", "Entropy-Revise-64+3", "TIGER-Instability-64+3", "CORE-proxy-64", "DNB-84"]:
    handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=short_method_name(method_name),
        markerfacecolor=METHOD_COLORS[method_name],
        markersize=8,
    )
    legend_handles.append(handle)
ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_SIZE, loc="lower right")

plt.tight_layout()
output_path = os.path.join(FIG_DIR, "fig2_honest_compute.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"Saved to {output_path}")
