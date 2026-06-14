"""
Generate Figure 3: observer quality vs controller gain.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import FIG_HEIGHT, FIG_WIDTH, LEGEND_SIZE, TITLE_SIZE  # noqa: E402


FIG_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(FIG_DIR))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "exp", "results")

SIGNAL_COLORS = {
    "calibration": "#1F77B4",
    "entropy": "#43A047",
    "instability": "#8E24AA",
}


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


data = load_json("diag_signal_gap_audit.json")
signals = data["signal_audit"]
screen = data["screen_metrics"]
shortlist = data["shortlist_metrics"]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * 1.08))
label_positions = {
    "instability": ((8, 8), "left"),
    "entropy": ((-80, 10), "left"),
    "calibration": ((-78, -10), "left"),
}

for entry in signals:
    color = SIGNAL_COLORS[entry["signal"]]
    ax.scatter(
        entry["diagnostic_score"],
        entry["control_effectiveness"],
        s=450 * max(entry["gap"], 0.08),
        color=color,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    ax.annotate(
        (
            f"{entry['signal'].capitalize()}\n"
            f"diag={entry['diagnostic_score']:.4f}\n"
            f"ctrl={entry['control_effectiveness']:.2f}"
        ),
        (entry["diagnostic_score"], entry["control_effectiveness"]),
        textcoords="offset points",
        xytext=label_positions[entry["signal"]][0],
        fontsize=LEGEND_SIZE,
        ha=label_positions[entry["signal"]][1],
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )

ax.axhline(0.0, color="#616161", lw=1, ls="--", alpha=0.7)
ax.axvline(0.1, color="#BDBDBD", lw=1, ls=":", alpha=0.7)
ax.set_xlim(-0.02, 0.72)
ax.set_ylim(-0.005, 0.03)
ax.set_xlabel("Diagnostic score")
ax.set_ylabel("Control effectiveness")
ax.set_title("Figure 3: Observer quality vs. controller gain", fontsize=TITLE_SIZE)
ax.grid(True, alpha=0.25)

ax.text(
    0.03,
    0.97,
    (
        f"Screen: random={screen['random_accuracy']:.2f}, entropy={screen['entropy_accuracy']:.2f}, "
        f"instability={screen['instability_accuracy']:.2f}\n"
        f"Shortlist: standard={shortlist['standard_accuracy']:.2f}, entropy={shortlist['entropy_accuracy']:.2f}, "
        f"TIGER={shortlist['instability_accuracy']:.2f}"
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=LEGEND_SIZE,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D0D0"),
)

plt.tight_layout()
output_path = os.path.join(FIG_DIR, "fig3_signal_gap.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"Saved to {output_path}")
