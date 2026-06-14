"""
Shared style configuration for all ComposeAccel paper figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color palette (colorblind-safe)
COLORS = {
    "M1": "#4477AA",        # Blue
    "M2": "#999999",        # Grey (NO_GO)
    "M3": "#EE6677",        # Red
    "IGSD": "#228833",      # Green
    "M1+IGSD": "#CCBB44",   # Yellow-gold (SYNERGY)
    "M3+IGSD": "#AA3377",   # Purple
    "M1+M3": "#66CCEE",     # Cyan
    "baseline": "#000000",  # Black
    "synergy": "#228833",   # Green
    "interference": "#CC3311",  # Dark red
}

# Method markers
MARKERS = {
    "M1": "o",
    "M2": "X",
    "M3": "s",
    "IGSD": "D",
    "M1+IGSD": "*",
    "M3+IGSD": "^",
    "M1+M3": "v",
    "baseline": "P",
}

# Figure defaults
FIGSIZE_SINGLE = (4.5, 3.5)
FIGSIZE_WIDE = (7, 3.5)
FIGSIZE_TALL = (4.5, 5)
DPI = 300
FONT_SIZE = 9
LABEL_SIZE = 8

def apply_style():
    """Apply publication-ready style to matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": LABEL_SIZE,
        "ytick.labelsize": LABEL_SIZE,
        "legend.fontsize": LABEL_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
        "lines.markersize": 6,
    })
