"""
Generate Figure 1 teaser for the diagnostic-study outline.

Panels:
1. Honest-compute ordering drift on GSM8K.
2. Observer-controller mismatch across signals.
3. HumanEval code-boundary failure trade-offs.
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
    FIG_WIDTH_FULL,
    FONT_SIZE,
    LEGEND_SIZE,
    LINE_WIDTH,
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

SIGNAL_COLORS = {
    "calibration": "#1F77B4",
    "entropy": COLORS["entropy_revise"],
    "instability": "#8E24AA",
}

CODE_METHOD_COLORS = {
    "Standard": COLORS["standard"],
    "Entropy/TIGER Ungated Revision": "#A1887F",
    "Gated TIGER": COLORS["highlight"],
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


compute = load_json("diag_compute_curve_gsm8k.json")
signal_gap = load_json("diag_signal_gap_audit.json")
code_boundary = load_json("diag_humaneval_guard_boundary.json")


fig, axes = plt.subplots(
    1,
    3,
    figsize=(FIG_WIDTH_FULL, FIG_HEIGHT * 1.18),
    gridspec_kw={"wspace": 0.42},
)
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.80, wspace=0.42)


# Panel A: nominal vs actual compute ordering
ax = axes[0]
methods = sorted(compute["methods"], key=lambda item: item["actual_nfe"])
y_pos = np.arange(len(methods))
for idx, method in enumerate(methods):
    color = METHOD_COLORS.get(method["method"], COLORS["baseline"])
    ax.plot(
        [method["nominal_nfe"], method["actual_nfe"]],
        [idx, idx],
        color=color,
        lw=3,
        alpha=0.9,
        solid_capstyle="round",
    )
    ax.scatter(
        method["nominal_nfe"],
        idx,
        s=50,
        marker="o",
        color="white",
        edgecolor=color,
        linewidth=1.2,
        zorder=3,
    )
    ax.scatter(
        method["actual_nfe"],
        idx,
        s=68,
        marker="D",
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.annotate(
        short_method_name(method["method"]),
        (method["actual_nfe"], idx),
        textcoords="offset points",
        xytext=(6, -3 if method["method"] == "DNB-84" else 1),
        va="center",
        fontsize=LEGEND_SIZE,
    )

ax.set_yticks([])
ax.set_xlim(62.0, 86.5)
ax.set_xlabel("Forward evaluations\n(circle nominal, diamond actual)")
ax.set_title("a) Honest compute\nreshuffles the shortlist", loc="left")
ax.grid(True, axis="x", alpha=0.25)
ax.axvline(67, color="#BDBDBD", lw=1, ls="--", alpha=0.7)
reorder = compute["pairwise_reorders"][0]
ax.text(
    0.02,
    0.06,
    (
        f"{short_method_name(reorder['left'])} falls behind "
        f"{short_method_name(reorder['right'])}\n"
        f"once actual NFE is counted; gap = {compute['max_abs_compute_gap_pct']:.2f}%."
    ),
    transform=ax.transAxes,
    fontsize=LEGEND_SIZE,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0"),
)


# Panel B: observer-controller mismatch
ax = axes[1]
signals = signal_gap["signal_audit"]
signal_offsets = {
    "instability": (6, 8),
    "entropy": (-18, 10),
    "calibration": (-92, -18),
}
for entry in signals:
    color = SIGNAL_COLORS[entry["signal"]]
    ax.plot(
        [entry["diagnostic_score"], entry["diagnostic_score"]],
        [0.0, entry["control_effectiveness"]],
        color=color,
        lw=2,
        alpha=0.75,
    )
    ax.scatter(
        entry["diagnostic_score"],
        entry["control_effectiveness"],
        s=230,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    ax.annotate(
        entry["signal"].capitalize(),
        (entry["diagnostic_score"], entry["control_effectiveness"]),
        textcoords="offset points",
        xytext=signal_offsets[entry["signal"]],
        fontsize=LEGEND_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )

ax.axhline(0.0, color="#616161", lw=1, ls="--", alpha=0.6)
ax.set_xlim(-0.02, 0.68)
ax.set_ylim(-0.005, 0.03)
ax.set_xlabel("Diagnostic score")
ax.set_ylabel("Controller gain vs random revise")
ax.set_title("b) Strong observers\nare not strong controllers", loc="left")
ax.grid(True, alpha=0.25)
ax.text(
    0.03,
    0.92,
    (
        f"Calibration wins the audit ({signals[0]['diagnostic_score']:.4f}),\n"
        "but deployed gain still stays at 0.00."
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=LEGEND_SIZE,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0"),
)


# Panel C: task-dependent failure at the code boundary
ax = axes[2]
methods = code_boundary["methods"]
metrics = [
    ("pass_at_1", "pass@1"),
    ("syntax_failure_rate", "syntax fail"),
    ("runtime_failure_rate", "runtime fail"),
]
x = np.arange(len(metrics))
width = 0.22
for offset, method in zip(np.linspace(-width, width, len(methods)), methods):
    color = CODE_METHOD_COLORS[method["method"]]
    values = [method[key] for key, _ in metrics]
    ax.bar(
        x + offset,
        values,
        width=width,
        color=color,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.8,
        label=method["method"],
    )

ax.set_xticks(x, [label for _, label in metrics])
ax.set_ylim(0, 0.8)
ax.set_ylabel("Rate")
ax.set_title("c) Syntax repair\ndoes not restore execution", loc="left")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=LEGEND_SIZE - 0.4, loc="lower left")
ax.text(
    0.03,
    0.93,
    (
        f"Gating cuts syntax failure by {-code_boundary['deltas']['gated_vs_ungated_syntax_failure']:.2f},\n"
        f"but pass@1 stays {abs(code_boundary['deltas']['gated_vs_standard_pass_at_1']):.2f} "
        "below Standard."
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=LEGEND_SIZE,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0"),
)


fig.suptitle(
    "Figure 1: The diagnostic story appears before any method claim",
    fontsize=TITLE_SIZE - 1,
    y=0.96,
)

output_path = os.path.join(FIG_DIR, "fig1_teaser.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"Saved to {output_path}")
