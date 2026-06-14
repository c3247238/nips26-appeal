"""
Generate publication-quality figures for the Dynamic Weight Decay paper.
All figures saved as 300 DPI PNG files.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Colorblind-friendly palette (tab10 subset, then paired)
PALETTE = {
    'constant':        '#1f77b4',   # blue
    'cosine_schedule': '#ff7f0e',   # orange
    'random_mask':     '#2ca02c',   # green
    'half_lambda':     '#d62728',   # red
    'no_wd':           '#9467bd',   # purple
    'cwd_hard':        '#8c564b',   # brown
    'swd':             '#e377c2',   # pink
}

METHOD_LABELS = {
    'constant':        r'Constant ($\Phi$=1)',
    'cosine_schedule': r'Cosine ($\Phi_\mathrm{cos}$)',
    'random_mask':     r'Random Mask ($\Phi_\mathrm{rand}$)',
    'half_lambda':     r'Half-$\lambda$ ($\Phi$=0.5)',
    'no_wd':           r'No WD ($\Phi$=0)',
    'cwd_hard':        r'CWD-Hard ($\Phi_\mathrm{align}$)',
    'swd':             r'SWD ($\Phi_\mathrm{sparse}$)',
}

OUT_DIR = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/iter_003/writing/figures'


# ===========================================================================
# Figure 1: Phi Framework Taxonomy Diagram
# ===========================================================================
def make_fig1_taxonomy():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # ---- Axis lines (horizontal) ----
    axes_data = [
        # (y, label, ticks_labels_positions, methods)
        (4.8, 'Temporal\n(schedule)', [0.5, 5.0, 9.5], ['Static', 'Scheduled', 'Annealed'],
         [('Constant', 0.5), ('Half-\u03bb', 2.2), ('Cosine', 5.0)]),
        (3.5, 'Directional\n(alignment)', [0.5, 5.0, 9.5], ['Blind', 'Soft-align', 'Hard-align'],
         [('CWD-Hard', 9.5), ('SWD', 7.0)]),
        (2.2, 'Spatial\n(coverage)', [0.5, 5.0, 9.5], ['All params', 'Selective', 'Sparse'],
         [('Constant', 0.5), ('Random Mask', 5.0), ('SWD', 8.5)]),
        (0.9, 'Budget\n(total \u03bb)', [0.5, 5.0, 9.5], ['Low (\u03a6=0)', 'Medium (\u03a6=0.5)', 'High (\u03a6=1)'],
         [('No WD', 0.5), ('Cosine', 5.3), ('Constant', 9.5)]),
    ]

    tick_h = 0.12
    for y, label, positions, tick_labels, methods in axes_data:
        # Axis backbone
        ax.annotate('', xy=(9.8, y), xytext=(0.2, y),
                    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.4))
        # Axis label (left side)
        ax.text(-0.05, y, label, ha='right', va='center', fontsize=10,
                fontweight='bold', color='#222222',
                bbox=dict(boxstyle='round,pad=0.2', fc='#f0f4ff', ec='none'))

        # Tick marks + labels
        for pos, tlabel in zip(positions, tick_labels):
            ax.plot([pos, pos], [y - tick_h, y + tick_h], color='#666666', lw=1.0)
            ax.text(pos, y - 0.30, tlabel, ha='center', va='top', fontsize=8.5,
                    color='#555555', style='italic')

        # Method dots
        for mname, mx in methods:
            # small colored dot
            col = '#e74c3c'
            ax.plot(mx, y, 'o', ms=7, color=col, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.8)
            # label above dot
            ax.text(mx, y + 0.30, mname, ha='center', va='bottom',
                    fontsize=8, color='#222222',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#fff8f0',
                              ec='#e74c3c', lw=0.6))

    # ---- Title block ----
    ax.text(5.0, 5.7,
            '\u03a6 Modulator Framework \u2014 Taxonomy of Dynamic Weight Decay Methods',
            ha='center', va='top', fontsize=12, fontweight='bold', color='#1a1a2e')

    # ---- Legend box ----
    legend_x, legend_y = 7.8, 3.2
    ax.add_patch(FancyBboxPatch((legend_x - 0.1, legend_y - 0.35), 2.2, 0.75,
                                boxstyle='round,pad=0.1', fc='#f9f9f9',
                                ec='#aaaaaa', lw=0.7, zorder=3))
    ax.plot(legend_x + 0.15, legend_y + 0.15, 'o', ms=7, color='#e74c3c',
            markeredgecolor='white', markeredgewidth=0.8, zorder=4)
    ax.text(legend_x + 0.4, legend_y + 0.15, '= Method placement',
            va='center', fontsize=8.5, color='#333333', zorder=4)

    plt.tight_layout(pad=0.5)
    path = f'{OUT_DIR}/fig1_taxonomy.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ===========================================================================
# Figure 2: Main Accuracy Comparison (Bar Chart)
# ===========================================================================
def make_fig2_accuracy():
    # Data ordered by CIFAR-10 mean acc (descending)
    methods_order = ['constant', 'cosine_schedule', 'random_mask',
                     'half_lambda', 'no_wd', 'cwd_hard', 'swd']

    cifar10 = {
        'constant':        (90.13, 0.31),
        'cosine_schedule': (90.12, 0.07),
        'random_mask':     (90.12, 0.30),
        'half_lambda':     (90.09, 0.29),
        'no_wd':           (90.08, 0.32),
        'cwd_hard':        (90.06, 0.24),
        'swd':             (89.88, 0.25),
    }
    cifar100 = {
        'cosine_schedule': (63.42, 0.42),
        'constant':        (63.15, 0.30),
        'swd':             (63.06, 0.29),
        'half_lambda':     (62.91, 0.47),
        'random_mask':     (62.87, 0.38),
        'cwd_hard':        (62.84, 0.30),
        'no_wd':           (62.66, 0.38),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bar_width = 0.55

    datasets = [
        ('CIFAR-10', cifar10, methods_order, 89.5, 90.5, axes[0]),
        ('CIFAR-100', cifar100, methods_order, 62.0, 64.0, axes[1]),
    ]

    for dname, data, order, ymin, ymax, ax in datasets:
        xs = np.arange(len(order))
        means = [data[m][0] for m in order]
        stds  = [data[m][1] for m in order]
        colors = [PALETTE[m] for m in order]
        short_labels = [METHOD_LABELS[m].split('(')[0].strip() for m in order]

        bars = ax.bar(xs, means, width=bar_width, color=colors,
                      yerr=stds, capsize=4, error_kw=dict(elinewidth=1.2, ecolor='#333333'),
                      zorder=3, edgecolor='white', linewidth=0.5)

        # Baseline dashed line (constant)
        baseline = data['constant'][0]
        ax.axhline(baseline, color='#1f77b4', linestyle='--', linewidth=1.2,
                   alpha=0.7, zorder=2, label=f'Constant baseline ({baseline:.2f}%)')

        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xs)
        ax.set_xticklabels(short_labels, rotation=28, ha='right', fontsize=9.5)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_xlabel('')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        # Dataset label
        ax.text(0.97, 0.97, dname, transform=ax.transAxes,
                ha='right', va='top', fontsize=12, fontweight='bold',
                color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#cccccc', lw=0.7))

        ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

        # Value labels on bars (placed above error bar caps)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.03,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=7.5,
                    color='#222222', rotation=0)

    plt.tight_layout(pad=1.5, w_pad=2.5)
    path = f'{OUT_DIR}/fig2_accuracy_comparison.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ===========================================================================
# Figure 3: BEM vs Accuracy Scatter Plot
# ===========================================================================
def make_fig3_bem_scatter():
    # BEM values and accuracy with std
    cifar10_pts = {
        'constant':        (0.000, 90.13, 0.31),
        'half_lambda':     (0.000, 90.09, 0.29),
        'cosine_schedule': (0.503, 90.12, 0.07),
        'cwd_hard':        (0.503, 90.06, 0.24),
        'random_mask':     (0.500, 90.12, 0.30),
        'swd':             (0.900, 89.88, 0.25),
        'no_wd':           (1.000, 90.08, 0.32),
    }
    cifar100_pts = {
        'constant':        (0.000, 63.15, 0.30),
        'half_lambda':     (0.000, 62.91, 0.47),
        'cosine_schedule': (0.503, 63.42, 0.42),
        'cwd_hard':        (0.500, 62.84, 0.30),
        'random_mask':     (0.501, 62.87, 0.38),
        'swd':             (0.900, 63.06, 0.29),
        'no_wd':           (1.000, 62.66, 0.38),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pts, dname, ymin, ymax in [
        (axes[0], cifar10_pts,  'CIFAR-10',  89.4, 90.7),
        (axes[1], cifar100_pts, 'CIFAR-100', 62.0, 64.2),
    ]:
        bems = [v[0] for v in pts.values()]
        accs = [v[1] for v in pts.values()]
        stds = [v[2] for v in pts.values()]

        # Flat trend line (linear regression)
        bem_arr = np.array(bems)
        acc_arr = np.array(accs)
        m, b = np.polyfit(bem_arr, acc_arr, 1)
        x_line = np.linspace(-0.05, 1.05, 100)
        ax.plot(x_line, m * x_line + b, '--', color='#888888', lw=1.2,
                alpha=0.7, zorder=1, label=f'Trend (slope={m:+.3f}%)')

        # Scatter points
        for method, (bem, acc, std) in pts.items():
            ax.errorbar(bem, acc, yerr=std, fmt='o', ms=9,
                        color=PALETTE[method], ecolor=PALETTE[method],
                        elinewidth=1.5, capsize=4, capthick=1.5,
                        markeredgecolor='white', markeredgewidth=0.8,
                        zorder=5, label=METHOD_LABELS[method])

        # Annotate methods — offset to avoid overlap
        offsets = {
            'constant':        (-0.04, +0.04),
            'half_lambda':     (-0.04, -0.06),
            'cosine_schedule': (+0.02, +0.04),
            'cwd_hard':        (+0.02, -0.07),
            'random_mask':     (-0.08, +0.04),
            'swd':             (+0.02, +0.04),
            'no_wd':           (-0.08, +0.04),
        }
        for method, (bem, acc, std) in pts.items():
            dx, dy = offsets.get(method, (0.02, 0.03))
            short = METHOD_LABELS[method].split('(')[0].strip()
            ax.annotate(short, (bem + dx, acc + dy), fontsize=8,
                        color='#333333', ha='center',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                  ec='none', alpha=0.7))

        ax.set_xlim(-0.12, 1.15)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('BEM (Budget Equivalence Metric)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.xaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.grid(True, linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)

        ax.text(0.97, 0.97, dname, transform=ax.transAxes,
                ha='right', va='top', fontsize=12, fontweight='bold',
                color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#cccccc', lw=0.7))

        ax.legend(loc='lower left', fontsize=8.5, framealpha=0.85,
                  ncol=1, handlelength=1.2)

    plt.tight_layout(pad=1.5, w_pad=2.5)
    path = f'{OUT_DIR}/fig3_bem_vs_accuracy.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ===========================================================================
# Figure 4: Diagnostic Metrics Heatmap (CIFAR-10)
# ===========================================================================
def make_fig4_heatmap():
    # Raw metric values from analysis.md (CIFAR-10)
    methods = ['constant', 'cosine_schedule', 'random_mask',
               'half_lambda', 'no_wd', 'cwd_hard', 'swd']
    # CSI, AIS, BEM
    raw = np.array([
        [0.841, 0.336, 0.000],  # constant
        [0.936, 0.352, 0.503],  # cosine_schedule
        [0.892, 0.359, 0.500],  # random_mask
        [0.853, 0.410, 0.000],  # half_lambda
        [0.964, 0.343, 1.000],  # no_wd
        [0.851, 0.368, 0.503],  # cwd_hard
        [0.838, 0.360, 0.900],  # swd
    ])
    metrics = ['CSI\n(Coupling Stability)', 'AIS\n(Alignment Info.)', 'BEM\n(Budget Equiv.)']

    # Normalize each column to [0, 1]
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    normalized = (raw - col_min) / (col_max - col_min + 1e-9)

    method_labels = [METHOD_LABELS[m].replace(r'\Phi', 'Φ') for m in methods]
    # Simplify labels for heatmap
    hm_labels = [
        'Constant',
        'Cosine',
        'Random Mask',
        'Half-λ',
        'No WD',
        'CWD-Hard',
        'SWD',
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Axes ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticklabels(hm_labels, fontsize=10)

    # Annotate cells with raw values
    for i in range(len(methods)):
        for j in range(len(metrics)):
            val = raw[i, j]
            norm_val = normalized[i, j]
            text_color = 'white' if norm_val > 0.65 else '#222222'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Normalized Value [0–1]', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Grid lines between cells
    ax.set_xticks(np.arange(len(metrics)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(methods)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Remove outer box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Dataset label
    ax.text(0.98, 1.02, 'CIFAR-10 / ResNet-20', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='#555555', style='italic')

    plt.tight_layout(pad=1.0)
    path = f'{OUT_DIR}/fig4_diagnostic_heatmap.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print('Generating Figure 1: Phi Framework Taxonomy ...')
    make_fig1_taxonomy()

    print('Generating Figure 2: Accuracy Comparison Bar Chart ...')
    make_fig2_accuracy()

    print('Generating Figure 3: BEM vs Accuracy Scatter ...')
    make_fig3_bem_scatter()

    print('Generating Figure 4: Diagnostic Heatmap ...')
    make_fig4_heatmap()

    print('\nAll figures generated successfully.')
