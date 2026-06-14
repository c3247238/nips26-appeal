from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from style_config import COLORS, FIG_WIDTH_FULL, FONT_SIZE, LINE_WIDTH


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "writing" / "figures" / "fig1_entropy_routed_compute.pdf"


def add_box(ax, xy, width, height, text, edgecolor, facecolor="#ffffff", fontsize=10):
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=LINE_WIDTH,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=3,
    )


def add_arrow(ax, start, end, color="#444444", style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=LINE_WIDTH,
            color=color,
            zorder=1,
        )
    )


def main() -> None:
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    add_box(ax, (0.4, 2.0), 2.2, 0.95, "Shared 64-step\ndraft", "#333333", "#f5f5f5")
    add_box(
        ax,
        (3.1, 2.0),
        2.3,
        0.95,
        "Draft-time entropy\nscores",
        COLORS["baseline"],
        "#f2f2f2",
    )
    add_box(
        ax,
        (5.9, 2.0),
        2.4,
        0.95,
        "Top-$\\rho$ frontier\n($\\rho \\approx 0.1211$)",
        "#333333",
        "#f5f5f5",
    )

    add_box(
        ax,
        (9.0, 3.0),
        2.2,
        0.85,
        "Candidate frontier\nfrom entropy",
        COLORS["ours"],
        "#eaf3fb",
    )
    add_box(
        ax,
        (11.5, 3.0),
        2.1,
        0.85,
        "Stopping check\n$\\tau = 0.85$",
        COLORS["ours"],
        "#eaf3fb",
    )
    add_box(
        ax,
        (9.0, 1.1),
        2.2,
        0.85,
        "Fixed frontier\nsame budget",
        COLORS["ablation"],
        "#fff1e6",
    )
    add_box(
        ax,
        (11.5, 1.1),
        2.1,
        0.85,
        "Same stopping\nfamily",
        COLORS["ablation"],
        "#fff1e6",
    )

    add_arrow(ax, (2.6, 2.48), (3.1, 2.48))
    add_arrow(ax, (5.4, 2.48), (5.9, 2.48))
    add_arrow(ax, (8.3, 2.48), (9.0, 3.42), color=COLORS["ours"])
    add_arrow(ax, (8.3, 2.48), (9.0, 1.52), color=COLORS["ablation"])
    add_arrow(ax, (11.2, 3.42), (11.5, 3.42), color=COLORS["ours"])
    add_arrow(ax, (11.2, 1.52), (11.5, 1.52), color=COLORS["ablation"])

    ax.text(12.55, 4.25, "Frontier-only revision\n$T_{max} = 3$", color=COLORS["ours"], ha="center")
    ax.text(12.55, 0.35, "Frontier-only revision\n$T_{max} = 3$", color=COLORS["ablation"], ha="center")
    add_arrow(ax, (13.6, 3.42), (13.6, 4.0), color=COLORS["ours"])
    add_arrow(ax, (13.6, 1.52), (13.6, 0.95), color=COLORS["ablation"])

    ax.text(1.5, 3.35, "Identical draft for both branches", color="#444444", ha="center", fontsize=10)
    ax.text(4.25, 3.45, "High-entropy positions act as an observer signal", color="#444444", ha="center", fontsize=10)

    callout = (
        "Tested claim: not whether frontier compute helps in general,\n"
        "but whether entropy-based frontier placement yields a better\n"
        "trade-off than a matched fixed-frontier control."
    )
    ax.text(
        7.1,
        4.55,
        callout,
        ha="center",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8f8f8", "edgecolor": "#bbbbbb"},
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH)


if __name__ == "__main__":
    main()
