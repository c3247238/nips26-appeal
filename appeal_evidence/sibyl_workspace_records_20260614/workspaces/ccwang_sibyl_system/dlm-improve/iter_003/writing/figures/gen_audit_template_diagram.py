from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR))

from style_config import COLORS, FIG_WIDTH_FULL, apply_global_style  # noqa: E402


def add_box(ax, xy, width, height, text, facecolor, edgecolor="#263238"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color="#37474F",
    )
    ax.add_patch(arrow)


def main() -> None:
    apply_global_style(plt)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.28
    w = 0.16
    h = 0.46
    xs = [0.02, 0.23, 0.44, 0.65, 0.86]

    add_box(
        ax,
        (xs[0], y),
        w,
        h,
        "Observed\nsmall gain\n\nCARD-84 > DNB-84\non GSM8K",
        COLORS["neutral_light"],
    )
    add_box(
        ax,
        (xs[1], y),
        w,
        h,
        "Compute-matched\nactive control\n\nDNB-84\nrules out\n\"more budget only\"",
        "#D6EAF8",
    )
    add_box(
        ax,
        (xs[2], y),
        w,
        h,
        "Budget-matched\nsham control\n\nCARD-84 vs RAND-84\nnet repaired = +1",
        "#FDEBD0",
    )
    add_box(
        ax,
        (xs[3], y),
        w,
        h,
        "Sample-level audit\n+ current-only\nartifact closure\n\nper-sample audit\ntransition matrix",
        COLORS["neutral_light"],
    )
    add_box(
        ax,
        (xs[4], y),
        w,
        h,
        "Claim ceiling\n\nRisk marker,\nnot validated rule\nNo winning-method\nclaim",
        "#FDEDEC",
        edgecolor=COLORS["harm"],
    )

    for left, right in zip(xs, xs[1:]):
        add_arrow(ax, (left + w, y + h / 2), (right, y + h / 2))

    ax.set_title("Minimal audit template for small-gain DLM revision claims")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "audit_template.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
