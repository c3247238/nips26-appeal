# Unified visual style for all figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {
    'no_wd': '#78909C',         # Blue-gray
    'constant': '#424242',      # Dark gray (reference)
    'cosine_schedule': '#2196F3', # Blue
    'cwd_hard': '#4CAF50',      # Green
    'swd': '#FF9800',           # Orange
    'pmpwd': '#F44336',         # Red (ours)
    'half_lambda': '#9C27B0',   # Purple
    'random_mask': '#795548',   # Brown
    'highlight': '#E91E63',     # Pink for highlighting
    'band_fill': '#BBDEFB',     # Light blue for certified band
}

METHOD_LABELS = {
    'no_wd': 'No WD',
    'constant': 'Constant',
    'cosine_schedule': 'Cosine',
    'cwd_hard': 'CWD',
    'swd': 'SWD',
    'pmpwd': 'PMP-WD (Ours)',
    'half_lambda': 'Half-$\\lambda$',
    'random_mask': 'Random Mask',
}

METHOD_ORDER = ['no_wd', 'constant', 'cosine_schedule', 'cwd_hard', 'swd', 'pmpwd']

FONT_SIZE = 11
TITLE_SIZE = 13
LINE_WIDTH = 1.8
FIG_WIDTH = 6.0       # inches, single column
FIG_WIDTH_FULL = 12.0  # inches, full width
FIG_HEIGHT = 4.0       # inches, default height
MARKER_SIZE = 6

def setup_style():
    """Apply unified paper style."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
    })
