# Unified visual style for all figures
# Paper: When Does Adaptive Weight Decay Help?

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Color palette
COLORS = {
    'constant': '#2196F3',       # Blue - baseline
    'cosine_schedule': '#4CAF50', # Green
    'cwd_hard': '#F44336',       # Red
    'swd': '#FF9800',            # Orange
    'half_lambda': '#9C27B0',    # Purple
    'random_mask': '#795548',    # Brown
    'no_wd': '#607D8B',          # Blue-gray
    'pmp_wd': '#E91E63',         # Pink - proposed method
    'highlight': '#F44336',      # Red for highlighting
    'adamw': '#2196F3',          # Blue for AdamW
    'sgd': '#FF9800',            # Orange for SGD
}

# Method display names
METHOD_NAMES = {
    'constant': 'Constant',
    'cosine_schedule': 'Cosine',
    'cwd_hard': 'CWD',
    'swd': 'SWD',
    'half_lambda': 'Half-$\\lambda$',
    'random_mask': 'Random',
    'no_wd': 'No WD',
}

# Figure dimensions
FONT_SIZE = 11
LINE_WIDTH = 1.5
FIG_WIDTH = 6.0       # inches, single column
FIG_WIDTH_FULL = 12.0  # inches, full width
FIG_HEIGHT = 4.0       # inches, default height
DPI = 300

# Method ordering for consistent plot legends
METHOD_ORDER = ['constant', 'cosine_schedule', 'cwd_hard', 'swd',
                'half_lambda', 'random_mask', 'no_wd']

def setup_style():
    """Apply unified style to all figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE + 1,
        'axes.titlesize': FONT_SIZE + 2,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'lines.linewidth': LINE_WIDTH,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
