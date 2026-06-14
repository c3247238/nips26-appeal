# Unified visual style for all figures
# Paper: Beyond Competitive Exclusion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    'ours': '#2196F3',       # Blue for our method / measured rates
    'baseline': '#9E9E9E',   # Gray for baselines
    'ablation': '#FF9800',   # Orange for ablations
    'highlight': '#F44336',  # Red for highlighting
    'shuffled': '#E91E63',   # Pink for shuffled controls
    'random': '#9C27B0',     # Purple for random probe controls
    'hedging': '#2196F3',    # Blue for hedging
    'hierarchy': '#F44336',  # Red for hierarchy-driven
    'recon_error': '#9E9E9E',# Gray for reconstruction error
    'absorbed': '#F44336',   # Red for absorbed letters
    'non_absorbed': '#2196F3',# Blue for non-absorbed letters
    'l10': '#FF9800',        # Orange for layer 10
    'l12': '#2196F3',        # Blue for layer 12
    'l20': '#4CAF50',        # Green for layer 20
}

FONT_SIZE = 11
LINE_WIDTH = 1.5
FIG_WIDTH = 6.0     # inches, single column
FIG_WIDTH_FULL = 12.0  # inches, full width
FIG_HEIGHT = 4.0    # inches, default

# Apply global style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE + 1,
    'axes.titlesize': FONT_SIZE + 2,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'lines.linewidth': LINE_WIDTH,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
