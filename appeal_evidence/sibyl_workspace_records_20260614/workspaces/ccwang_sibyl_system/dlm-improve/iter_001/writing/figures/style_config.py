# Unified visual style for all figures
COLORS = {
    'ours': '#2196F3',      # Blue for our method (CARD)
    'baseline': '#9E9E9E',  # Gray for baselines
    'standard': '#607D8B',  # Blue-gray for standard methods
    'dnb': '#795548',       # Brown for DNB
    'entropy_revise': '#4CAF50',  # Green for entropy-revise
    'ablation': '#FF9800',  # Orange for ablations
    'highlight': '#F44336', # Red for highlighting / failures
    'oracle': '#8BC34A',    # Light green for oracle calibration
    'sc': '#E91E63',        # Pink for self-consistency calibration
    'diagonal': '#000000',  # Black for perfect calibration line
}

FONT_SIZE = 11
TITLE_SIZE = 13
TICK_SIZE = 9
LEGEND_SIZE = 9
LINE_WIDTH = 1.5
MARKER_SIZE = 8
FIG_WIDTH = 6.0        # inches, single column
FIG_WIDTH_FULL = 12.0  # inches, full width
FIG_HEIGHT = 4.0       # inches, default height
DPI = 300

# Matplotlib style
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'lines.linewidth': LINE_WIDTH,
    'lines.markersize': MARKER_SIZE,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
