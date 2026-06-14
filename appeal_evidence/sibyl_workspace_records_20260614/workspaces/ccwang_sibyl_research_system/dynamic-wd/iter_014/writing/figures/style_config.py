# Unified visual style for all figures
# All figure scripts must import this module for consistent appearance.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# Method color palette
COLORS = {
    'FixedWD':           '#616161',   # Dark gray (baseline)
    'CWD':               '#1976D2',   # Blue
    'SWD':               '#F57C00',   # Orange
    'CPR':               '#388E3C',   # Green
    'DefazioCorrective': '#7B1FA2',   # Purple
    'NoWD':              '#BDBDBD',   # Light gray (null)
    'UDWDC':             '#D32F2F',   # Red (ours)
    'UDWDC-v2':          '#C62828',   # Dark red (ours, fix)
    # Ablation variants
    'Kp_only':           '#EF9A9A',
    'Ki_only':           '#CE93D8',
    'Kd_only':           '#90CAF9',
    'PI_control':        '#A5D6A7',
    'PD_control':        '#FFCC80',
    'Full_PID':          '#80CBC4',
    # Generic
    'ours':              '#D32F2F',
    'baseline':          '#616161',
    'ablation':          '#FF9800',
    'highlight':         '#F44336',
}

# Method display names (for legends)
NAMES = {
    'FixedWD':           'FixedWD',
    'CWD':               'CWD',
    'SWD':               'SWD',
    'CPR':               'CPR',
    'DefazioCorrective': 'Defazio',
    'NoWD':              'NoWD',
    'UDWDC':             'UDWDC',
    'UDWDC-v2':          'UDWDC-v2',
}

# Method markers
MARKERS = {
    'FixedWD': 's', 'CWD': 'D', 'SWD': '^', 'CPR': 'v',
    'DefazioCorrective': 'P', 'NoWD': 'x', 'UDWDC': 'o', 'UDWDC-v2': '*',
}

# Typography
FONT_SIZE = 11
TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 9.5

# Line styling
LINE_WIDTH = 1.8
MARKER_SIZE = 5

# Figure dimensions (inches)
FIG_WIDTH = 6.0        # single column
FIG_WIDTH_FULL = 12.0  # full width (two-column)
FIG_HEIGHT = 4.0       # default height

# DPI
DPI = 300


def apply_style():
    """Apply the unified paper style to matplotlib."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })


def get_color(method):
    return COLORS.get(method, '#000000')


def get_name(method):
    return NAMES.get(method, method)


def get_marker(method):
    return MARKERS.get(method, 'o')
