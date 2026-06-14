"""Unified figure style for iteration-3 paper assets."""

from __future__ import annotations

COLORS = {
    "card": "#2196F3",
    "baseline": "#9E9E9E",
    "sham": "#FF9800",
    "harm": "#F44336",
    "success": "#4CAF50",
    "neutral_dark": "#455A64",
    "neutral_light": "#CFD8DC",
}

FONT_SIZE = 11
LINE_WIDTH = 1.5
FIG_WIDTH = 6.0
FIG_WIDTH_FULL = 12.0


def apply_global_style(plt) -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 1,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "axes.linewidth": LINE_WIDTH,
            "lines.linewidth": LINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
