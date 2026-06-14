"""Publication-quality figure style for ML papers."""
import matplotlib.pyplot as plt

COLORS = {
    "standard": "#1f77b4",
    "topk": "#ff7f0e",
    "topk_mlp": "#2ca02c",
    "topk_attn": "#d62728",
    "feature_splitting": "#9467bd",
    "jumprelu": "#8c564b",
    "gatedsae": "#e377c2",
    "matryoshka": "#7f7f7f",
    "pannael": "#bcbd22",
    "batchtopk": "#17becf",
}

ARCH_LABELS = {
    "standard": "Standard",
    "topk": "TopK",
    "topk_mlp": "TopK MLP",
    "topk_attn": "TopK Attn",
    "feature_splitting": "Feature Splitting",
    "jumprelu": "JumpReLU",
    "gatedsae": "GatedSAE",
    "matryoshka": "Matryoshka",
    "pannael": "PAnneal",
    "batchtopk": "BatchTopK",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

def set_paper_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
