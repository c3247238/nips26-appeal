"""Shared visual style for all paper figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NeurIPS-style: single column = 5.5in, double = 11in
SINGLE_COL = 5.5
DOUBLE_COL = 11.0
FIG_HEIGHT = 3.5

COLORS = {
    "lv": "#2166AC",
    "cosine": "#B2182B",
    "type_i": "#D32F2F",
    "type_ii": "#FF9800",
    "type_iii": "#FDD835",
    "none": "#BDBDBD",
    "low_abs": "#2166AC",
    "high_abs": "#B2182B",
    "sparse_probing": "#4CAF50",
    "ravel": "#9C27B0",
    "scr": "#FF5722",
    "unlearning": "#607D8B",
    "das_k1": "#2166AC",
    "das_k3": "#B2182B",
}

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
