#!/usr/bin/env python3
"""
Generate all paper-ready visualization panels for the
Unified Dynamic Weight Decay Control (UDWDC) paper.

Reads pilot experiment data from multiple phases and produces
publication-quality PNG (300 DPI) + PDF figures.

Figures:
  1. Mean rho_t trajectories (CIFAR-10 diagnostic, 8 methods)
  2. Per-layer rho_t box plots (selected methods)
  3. H1 unification: lambda_t trajectories original vs UDWDC fit
  4. Ablation bar chart (UDWDC gain ablation, CIFAR-100)
  5. Batch-size vs alignment SNR (H3)
  6. Budget efficiency curves (H4) — ImageNet accuracy vs WD budget
  7. H5 per-layer r* vs delta* scatter (placeholder from pilot data)
  8. H7 temporal predictability R^2 distribution
  9. ImageNet training curves and method comparison
 10. CSI comparison: method-level stability heatmap
 11. BEM vs Accuracy ranking divergence
 12. Comprehensive metrics table
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
WORKSPACE = BASE.parents[3]  # workspaces/dynamic-wd/current
PILOTS_DIR = WORKSPACE / 'exp' / 'results' / 'pilots'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
OUT_DIR = BASE
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.85,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (Wong 2011 + extended)
METHOD_COLORS = {
    'FixedWD':           '#0072B2',  # blue
    'CWD':               '#E69F00',  # orange
    'SWD':               '#009E73',  # green
    'CPR':               '#D55E00',  # vermilion
    'DefazioCorrective': '#CC79A7',  # pink
    'NoWD':              '#999999',  # grey
    'UDWDC':             '#56B4E9',  # sky blue
    'UDWDC-v2':          '#F0E442',  # yellow
}

METHOD_MARKERS = {
    'FixedWD': 'o', 'CWD': 's', 'SWD': '^', 'CPR': 'D',
    'DefazioCorrective': 'v', 'NoWD': 'x', 'UDWDC': '*', 'UDWDC-v2': 'P',
}

VARIANT_COLORS = {
    'FixedWD':    '#0072B2',
    'Kp_only':    '#E69F00',
    'Ki_only':    '#009E73',
    'Kd_only':    '#D55E00',
    'PI_control': '#CC79A7',
    'PD_control': '#56B4E9',
    'Full_PID':   '#F0E442',
    'UDWDC_v2':   '#000000',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name):
    """Save as both PNG and PDF."""
    png_path = OUT_DIR / f'{name}.png'
    pdf_path = OUT_DIR / f'{name}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {png_path.name} + {pdf_path.name}')
    return str(png_path)


generated = []
errors = []


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Mean rho_t trajectories
# ═══════════════════════════════════════════════════════════════════════════
def fig01_rho_trajectories():
    print('[Fig 1] Mean rho_t trajectories (CIFAR-10, 8 methods)...')
    data = load_json(PILOTS_DIR / 'diagnostic_cifar10' / 'pilot_summary.json')
    methods = list(data['per_method'].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Mean rho_t across layers
    ax = axes[0]
    for method in methods:
        traj = data['per_method'][method]['mean_rho_trajectory']
        epochs = list(range(1, len(traj) + 1))
        ax.plot(epochs, traj,
                color=METHOD_COLORS.get(method, '#666'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=5, linewidth=1.8, label=method)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Mean $\rho_t$ (across layers)')
    ax.set_title(r'(a) Layer-Averaged $\rho_t$ Trajectories')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    # (b) Per-layer rho_t distribution at final epoch
    ax = axes[1]
    select_methods = ['FixedWD', 'CWD', 'UDWDC', 'UDWDC-v2', 'NoWD']
    layer_data = {}
    for method in select_methods:
        traj_path = PILOTS_DIR / 'diagnostic_cifar10' / method / f'{method}_seed42_trajectories.json'
        if not traj_path.exists():
            # Try alternate naming
            alt = method.replace('-', '_')
            traj_path = PILOTS_DIR / 'diagnostic_cifar10' / alt / f'{alt}_seed42_trajectories.json'
        if traj_path.exists():
            traj = load_json(traj_path)
            rho_final = []
            for lname, linfo in traj['layers'].items():
                if 'weight' in lname and ('conv' in lname or 'fc' in lname or 'linear' in lname):
                    rho_final.append(linfo['rho_t'][-1])
            layer_data[method] = rho_final
        else:
            layer_data[method] = []

    positions = np.arange(len(select_methods))
    bp_data = [layer_data.get(m, [0]) for m in select_methods]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})
    for i, method in enumerate(select_methods):
        bp['boxes'][i].set_facecolor(METHOD_COLORS.get(method, '#666'))
        bp['boxes'][i].set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(select_methods, rotation=15)
    ax.set_ylabel(r'$\rho_t$ at final epoch')
    ax.set_title(r'(b) Per-Layer $\rho_t$ Distribution (Epoch 10)')

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig01_rho_trajectories'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: H1 Unification — lambda_t fit
# ═══════════════════════════════════════════════════════════════════════════
def fig02_h1_unification():
    print('[Fig 2] H1 unification: lambda_t trajectories...')
    fit_path = PILOTS_DIR / 'h1_unification' / 'fitting_results.json'
    data = load_json(fit_path)

    results = data['results']
    n_methods = len(results)
    fig, axes = plt.subplots(1, min(n_methods, 5), figsize=(4 * min(n_methods, 5), 4))
    if n_methods == 1:
        axes = [axes]

    for idx, res in enumerate(results[:5]):
        ax = axes[idx]
        method = res['method']
        rel_err = res['relative_error_pct']

        # Extract per-layer errors as a proxy for quality
        per_layer = res.get('per_layer_errors', [])
        if per_layer:
            layer_names = [p['layer'].split('.')[-2] if '.' in p['layer'] else p['layer'][:10]
                           for p in per_layer[:10]]
            layer_errs = [p['relative_error'] * 100 for p in per_layer[:10]]
            colors_bar = ['#009E73' if e < 10 else '#E69F00' if e < 20 else '#D55E00'
                          for e in layer_errs]
            ax.barh(range(len(layer_names)), layer_errs, color=colors_bar,
                    edgecolor='black', linewidth=0.3, height=0.6)
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names, fontsize=7)
            ax.set_xlabel('Relative Error (%)')
            ax.axvline(x=15, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{method}\n(Avg err: {rel_err:.1f}%)', fontsize=10)
        ax.invert_yaxis()

    fig.suptitle('H1 Unification: UDWDC Fitting Error per Layer', fontsize=13, y=1.02)
    plt.tight_layout()
    generated.append(save_fig(fig, 'fig02_h1_unification_fit'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig03_ablation():
    print('[Fig 3] Ablation bar chart (CIFAR-100, VGG-16-BN)...')
    data = load_json(PILOTS_DIR / 'ablation_cifar100' / 'pilot_detailed_results.json')

    variants = list(data.keys())
    accs = [data[v]['final_test_acc'] for v in variants]
    gains_labels = []
    for v in variants:
        kp, ki, kd = data[v]['K_p'], data[v]['K_i'], data[v]['K_d']
        gains_labels.append(f'Kp={kp}\nKi={ki}\nKd={kd}')

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(variants))
    colors = [VARIANT_COLORS.get(v, '#666') for v in variants]

    bars = ax.bar(x, accs, color=colors, edgecolor='black', linewidth=0.5, width=0.65)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    baseline_acc = data['FixedWD']['final_test_acc']
    ax.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'FixedWD baseline ({baseline_acc:.1f}%)')

    display_names = [v.replace('_', '\n') for v in variants]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=9)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('UDWDC Gain Ablation Study\n(CIFAR-100, VGG-16-BN, 10 epochs, seed 42)')
    ax.legend(loc='lower right')
    ax.set_ylim(bottom=min(accs) - 2)

    # Add gain configuration as secondary x-labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(gains_labels, fontsize=7, color='#555')
    ax2.set_xlabel('Controller Gains', fontsize=9, color='#555')

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig03_ablation_cifar100'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Batch-size vs Alignment SNR (H3)
# ═══════════════════════════════════════════════════════════════════════════
def fig04_batchsize_snr():
    print('[Fig 4] Batch-size vs alignment SNR (H3)...')
    data = load_json(PILOTS_DIR / 'batchsize_sweep' / 'pilot_detailed_results.json')

    batch_sizes = [64, 128, 256, 512, 1024]
    methods = ['FixedWD', 'CWD', 'UDWDC-v2']
    method_keys = {
        'FixedWD': 'FixedWD', 'CWD': 'CWD', 'UDWDC-v2': 'UDWDCvv2'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Alignment SNR vs batch size
    ax = axes[0]
    for method in methods:
        snr_vals = []
        for bs in batch_sizes:
            key = f'{method_keys[method]}_bs{bs}'
            if key in data:
                snr_vals.append(data[key]['alignment_snr'])
            else:
                snr_vals.append(np.nan)
        ax.plot(batch_sizes, snr_vals,
                color=METHOD_COLORS.get(method, '#666'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=8, linewidth=2, label=method)

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Alignment SNR')
    ax.set_title(r'(a) Alignment SNR vs Batch Size (H3)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend()

    # (b) Accuracy delta vs batch size
    ax = axes[1]
    for method in ['CWD', 'UDWDC-v2']:
        deltas = []
        for bs in batch_sizes:
            fixed_key = f'FixedWD_bs{bs}'
            meth_key = f'{method_keys[method]}_bs{bs}'
            if fixed_key in data and meth_key in data:
                d = data[meth_key]['final_test_acc'] - data[fixed_key]['final_test_acc']
                deltas.append(d)
            else:
                deltas.append(np.nan)
        ax.plot(batch_sizes, deltas,
                color=METHOD_COLORS.get(method, '#666'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=8, linewidth=2, label=f'{method} vs FixedWD')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel(r'$\Delta$ Accuracy vs FixedWD (%)')
    ax.set_title('(b) Accuracy Improvement over FixedWD')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend()

    # Annotate H3 test
    ax.annotate('H3: binary > continuous\nat small batch',
                xy=(64, 3.67), xytext=(200, 3.0),
                fontsize=8, color='#D55E00',
                arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.2))

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig04_batchsize_snr'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Budget efficiency curves (H4) — ImageNet
# ═══════════════════════════════════════════════════════════════════════════
def fig05_budget_efficiency():
    print('[Fig 5] Budget efficiency curves (H4, ImageNet)...')
    metrics = load_json(PILOTS_DIR / 'metrics_computation' / 'metrics_results_v2.json')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) ImageNet: Accuracy vs WD budget (log scale)
    ax = axes[0]
    im_data = metrics['metrics']['imagenet_main']['methods']
    budget_matched = metrics['metrics']['imagenet_budget_matched']['configs']

    # Plot dynamic methods
    for method, mdata in im_data.items():
        budget = mdata.get('total_wd_budget', 0)
        acc = mdata.get('accuracy', 0)
        if budget > 0:
            ax.scatter(budget, acc * 100,
                       c=METHOD_COLORS.get(method, '#666'),
                       marker=METHOD_MARKERS.get(method, 'o'),
                       s=120, zorder=5, edgecolors='black', linewidth=0.5,
                       label=method)

    # Plot budget-matched FixedWD sweep
    bm_budgets, bm_accs, bm_labels = [], [], []
    for config_name, config_data in budget_matched.items():
        bm_budgets.append(config_data['total_wd_budget'])
        bm_accs.append(config_data['accuracy'] * 100)
        bm_labels.append(f"$\\lambda$={config_data['lambda']}")

    ax.plot(bm_budgets, bm_accs, 'k--', linewidth=1.5, alpha=0.7, zorder=3)
    ax.scatter(bm_budgets, bm_accs, c='black', marker='x', s=60, zorder=4,
               label='FixedWD sweep')
    for b, a, lbl in zip(bm_budgets, bm_accs, bm_labels):
        ax.annotate(lbl, (b, a), textcoords="offset points",
                    xytext=(5, -10), fontsize=7, color='#555')

    ax.set_xlabel('Total WD Budget')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('(a) Accuracy vs WD Budget (ImageNet, 2 epochs)')
    ax.set_xscale('log')
    ax.legend(fontsize=7, ncol=2, loc='upper right')

    # (b) BEM comparison bar chart
    ax = axes[1]
    # Use CIFAR-10 data where BEM is more meaningful
    cifar_methods_data = metrics['rankings']['cifar10']['by_BEM']
    methods_list = [m[0] for m in cifar_methods_data]
    bem_vals = [m[1] for m in cifar_methods_data]
    colors = [METHOD_COLORS.get(m, '#666') for m in methods_list]

    bars = ax.barh(range(len(methods_list)), bem_vals, color=colors,
                   edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(methods_list)))
    ax.set_yticklabels(methods_list)
    ax.set_xlabel('BEM (Budget Equivalence Metric)')
    ax.set_title('(b) BEM Ranking (CIFAR-10, ResNet-20)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, bem_vals)):
        if val >= 0:
            ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=8)
        else:
            ax.text(val - 1, i, f'{val:.1f}', va='center', ha='right', fontsize=8)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig05_budget_efficiency'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: H7 Temporal Predictability Gate — R^2 Distribution
# ═══════════════════════════════════════════════════════════════════════════
def fig06_h7_r2_distribution():
    print('[Fig 6] H7 temporal predictability R^2 distribution...')
    data = load_json(PILOTS_DIR / 'h7_temporal_gate' / 'h7_gate_result.json')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Histogram of R^2 values
    ax = axes[0]
    hist_counts = data['histogram']['counts']
    bin_edges = data['histogram']['bin_edges']
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

    # Filter to positive range (where data exists)
    valid = [(c, b) for c, b in zip(hist_counts, bin_centers) if c > 0]
    if valid:
        counts_v, centers_v = zip(*valid)
        widths = [bin_edges[1] - bin_edges[0]] * len(centers_v)
        colors_h = ['#D55E00' if c >= 0.85 else '#0072B2' for c in centers_v]
        ax.bar(centers_v, counts_v, width=widths[0] * 0.9, color=colors_h,
               edgecolor='black', linewidth=0.3)

    ax.axvline(x=0.85, color='red', linestyle='--', linewidth=1.5,
               label=f'Gate threshold ($R^2$ = 0.85)')
    pct_above = data['statistics']['pct_r2_above_085']
    ax.annotate(f'{pct_above:.1f}% above threshold',
                xy=(0.87, max(hist_counts) * 0.8),
                fontsize=10, color='#D55E00', fontweight='bold')
    ax.set_xlabel(r'$R^2$ (degree-4 polynomial fit)')
    ax.set_ylabel('Count (layer-method combinations)')
    ax.set_title(r'(a) Temporal Predictability: $R^2$ Distribution')
    ax.legend(fontsize=9)

    # (b) Per-method mean R^2
    ax = axes[1]
    core_methods = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective',
                    'NoWD', 'UDWDC', 'UDWDC-v2']
    per_method = data['per_method_stats']
    methods_plot = [m for m in core_methods if m in per_method]
    mean_r2s = [per_method[m]['mean_r2'] for m in methods_plot]
    pct_highs = [per_method[m]['pct_high_r2'] for m in methods_plot]

    x_pos = np.arange(len(methods_plot))
    colors_bar = [METHOD_COLORS.get(m, '#666') for m in methods_plot]

    bars = ax.bar(x_pos, mean_r2s, color=colors_bar, edgecolor='black',
                  linewidth=0.5, width=0.6)
    ax.axhline(y=0.85, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_plot, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(r'Mean $R^2$')
    ax.set_title(r'(b) Per-Method Mean $R^2$ (H7 Gate)')
    ax.set_ylim(0, 1.0)

    for bar, r2, pct in zip(bars, mean_r2s, pct_highs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{r2:.2f}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig06_h7_temporal_gate'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: ImageNet Training Curves + Final Accuracy Comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig07_imagenet_training():
    print('[Fig 7] ImageNet training curves and comparison...')
    data = load_json(PILOTS_DIR / 'imagenet_main' / 'pilot_summary_v2.json')
    results = data['results']
    methods = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Training loss curves
    ax = axes[0]
    for method in methods:
        epoch_data = results[method]['epoch_results']
        epochs = [e['epoch'] for e in epoch_data]
        losses = [e['train_loss'] for e in epoch_data]
        ax.plot(epochs, losses,
                color=METHOD_COLORS.get(method, '#666'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=7, linewidth=2, label=method)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) Training Loss (ResNet-50/ImageNet)')
    ax.legend(fontsize=7, ncol=2)

    # (b) Val accuracy bar chart
    ax = axes[1]
    val_accs = [(m, results[m]['final_val_acc'] * 100) for m in methods]
    val_accs.sort(key=lambda x: x[1], reverse=True)
    names_s = [v[0] for v in val_accs]
    accs_s = [v[1] for v in val_accs]
    colors = [METHOD_COLORS.get(n, '#666') for n in names_s]

    bars = ax.barh(range(len(names_s)), accs_s, color=colors,
                   edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s)
    ax.set_xlabel('Validation Accuracy (%)')
    ax.set_title('(b) Final Val Accuracy (2 epochs)')
    ax.invert_yaxis()
    for i, (bar, acc) in enumerate(zip(bars, accs_s)):
        ax.text(bar.get_width() + 0.3, i, f'{acc:.0f}%', va='center', fontsize=9)

    # (c) Top-5 accuracy comparison
    ax = axes[2]
    top5_accs = [(m, results[m].get('final_val_top5', 0) * 100) for m in methods]
    top5_accs.sort(key=lambda x: x[1], reverse=True)
    names_t5 = [v[0] for v in top5_accs]
    accs_t5 = [v[1] for v in top5_accs]
    colors_t5 = [METHOD_COLORS.get(n, '#666') for n in names_t5]

    bars = ax.barh(range(len(names_t5)), accs_t5, color=colors_t5,
                   edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(names_t5)))
    ax.set_yticklabels(names_t5)
    ax.set_xlabel('Top-5 Accuracy (%)')
    ax.set_title('(c) Final Top-5 Accuracy (2 epochs)')
    ax.invert_yaxis()
    for i, (bar, acc) in enumerate(zip(bars, accs_t5)):
        ax.text(bar.get_width() + 0.3, i, f'{acc:.0f}%', va='center', fontsize=9)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig07_imagenet_training'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: CSI Stability Comparison Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig08_csi_heatmap():
    print('[Fig 8] CSI stability comparison...')
    metrics = load_json(PILOTS_DIR / 'metrics_computation' / 'metrics_results_v2.json')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) CIFAR-10 CSI comparison
    ax = axes[0]
    cifar_data = metrics['metrics']['cifar10_diagnostic']['methods']
    methods_list = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective',
                    'NoWD', 'UDWDC', 'UDWDC-v2']

    metrics_names = ['CSI_rho_based', 'CSI_temporal_wd']
    display_labels = [r'CSI$_\rho$', r'CSI$_{WD}$']

    data_matrix = np.zeros((len(methods_list), len(metrics_names)))
    for i, m in enumerate(methods_list):
        if m in cifar_data:
            for j, mn in enumerate(metrics_names):
                data_matrix[i, j] = cifar_data[m].get(mn, 0)

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_yticks(range(len(methods_list)))
    ax.set_yticklabels(methods_list, fontsize=9)

    for i in range(len(methods_list)):
        for j in range(len(metrics_names)):
            val = data_matrix[i, j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='CSI Score', shrink=0.8)
    ax.set_title('(a) CSI (CIFAR-10, ResNet-20)')

    # (b) Ablation CSI comparison
    ax = axes[1]
    abl_data = metrics['metrics']['ablation_cifar100']['variants']
    abl_variants = ['FixedWD', 'Kp_only', 'Ki_only', 'Kd_only',
                    'PI_control', 'PD_control', 'Full_PID', 'UDWDC_v2']

    abl_csi = [abl_data[v]['CSI_temporal'] for v in abl_variants if v in abl_data]
    abl_names_display = [v.replace('_', '\n') for v in abl_variants if v in abl_data]
    colors_abl = [VARIANT_COLORS.get(v, '#666') for v in abl_variants if v in abl_data]

    bars = ax.bar(range(len(abl_csi)), abl_csi, color=colors_abl,
                  edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(abl_names_display)))
    ax.set_xticklabels(abl_names_display, fontsize=8)
    ax.set_ylabel('CSI (temporal)')
    ax.set_title('(b) CSI Ablation (CIFAR-100, VGG-16-BN)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5,
               label='Stability threshold')
    ax.legend(fontsize=8)

    for bar, val in zip(bars, abl_csi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig08_csi_comparison'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Effective lambda_t Trajectories
# ═══════════════════════════════════════════════════════════════════════════
def fig09_effective_lambda():
    print('[Fig 9] Effective lambda_t trajectories...')
    data = load_json(PILOTS_DIR / 'ablation_cifar100' / 'pilot_detailed_results.json')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) WD budget per epoch for each variant
    ax = axes[0]
    variants_plot = ['FixedWD', 'Kp_only', 'Ki_only', 'Kd_only', 'Full_PID']
    for variant in variants_plot:
        if variant in data:
            wd_per_epoch = data[variant]['wd_budget_per_epoch']
            epochs = list(range(1, len(wd_per_epoch) + 1))
            ax.plot(epochs, wd_per_epoch,
                    color=VARIANT_COLORS.get(variant, '#666'),
                    marker='o', markersize=5, linewidth=1.8,
                    label=variant)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('WD Budget per Epoch')
    ax.set_title('(a) WD Budget Trajectory per Epoch')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # (b) Mean effective WD over epochs
    ax = axes[1]
    for variant in variants_plot:
        if variant in data:
            mean_wds = [e['mean_effective_wd'] for e in data[variant]['epoch_history']]
            epochs = list(range(1, len(mean_wds) + 1))
            ax.plot(epochs, mean_wds,
                    color=VARIANT_COLORS.get(variant, '#666'),
                    marker='o', markersize=5, linewidth=1.8,
                    label=variant)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Mean Effective $\lambda_t$')
    ax.set_title(r'(b) Effective $\lambda_t$ per Epoch')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig09_effective_lambda'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10: BEM vs Accuracy Ranking Divergence
# ═══════════════════════════════════════════════════════════════════════════
def fig10_bem_ranking():
    print('[Fig 10] BEM vs Accuracy ranking divergence...')
    metrics = load_json(PILOTS_DIR / 'metrics_computation' / 'metrics_results_v2.json')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (dataset, title) in enumerate([
        ('cifar10', 'CIFAR-10, ResNet-20'),
        ('imagenet', 'ImageNet, ResNet-50 (2 epochs)')
    ]):
        ax = axes[ax_idx]
        rankings = metrics['rankings'].get(dataset, {})
        acc_rank = rankings.get('by_accuracy', [])
        bem_rank = rankings.get('by_BEM', [])

        if not acc_rank or not bem_rank:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
            continue

        # Build rank maps
        acc_order = {m[0]: i+1 for i, m in enumerate(acc_rank)}
        bem_order = {m[0]: i+1 for i, m in enumerate(bem_rank)}

        methods_list = list(acc_order.keys())
        x_pos = np.arange(len(methods_list))
        width = 0.35

        acc_ranks = [acc_order[m] for m in methods_list]
        bem_ranks = [bem_order.get(m, len(methods_list)) for m in methods_list]

        ax.bar(x_pos - width/2, acc_ranks, width, label='Accuracy Rank',
               color='#0072B2', edgecolor='black', linewidth=0.5)
        ax.bar(x_pos + width/2, bem_ranks, width, label='BEM Rank',
               color='#E69F00', edgecolor='black', linewidth=0.5)

        # Draw arrows for rank shifts
        for i, m in enumerate(methods_list):
            shift = acc_ranks[i] - bem_ranks[i]
            if abs(shift) > 0:
                mid = x_pos[i]
                ax.annotate('', xy=(mid + width/2, bem_ranks[i]),
                            xytext=(mid - width/2, acc_ranks[i]),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.5))

        ax.set_xlabel('Method')
        ax.set_ylabel('Rank (lower is better)')
        ax.set_title(f'({chr(97+ax_idx)}) {title}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods_list, rotation=25, ha='right', fontsize=8)
        ax.legend()
        ax.invert_yaxis()

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig10_bem_ranking_divergence'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11: Comprehensive Metrics Table
# ═══════════════════════════════════════════════════════════════════════════
def fig11_metrics_table():
    print('[Fig 11] Comprehensive metrics table...')
    metrics = load_json(PILOTS_DIR / 'metrics_computation' / 'metrics_results_v2.json')

    cifar_data = metrics['metrics']['cifar10_diagnostic']['methods']
    methods_order = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective',
                     'NoWD', 'UDWDC', 'UDWDC-v2']

    rows = []
    for m in methods_order:
        if m not in cifar_data:
            continue
        md = cifar_data[m]
        acc = md['accuracy']
        bem = md.get('BEM')
        bem_str = f'{bem:.1f}' if bem is not None else '--'
        csi_rho = md.get('CSI_rho_based', 0)
        csi_wd = md.get('CSI_temporal_wd', 0)
        ais_data = md.get('ais', {})
        snr = ais_data.get('alpha_SNR', 0)
        budget = md.get('total_wd_budget', 0)
        budget_str = f'{budget:.4f}' if budget > 0 else '0'

        rows.append([m, f'{acc:.2f}', bem_str,
                     f'{csi_rho:.3f}', f'{csi_wd:.3f}',
                     f'{snr:.4f}', budget_str])

    col_labels = ['Method', 'Acc (%)', 'BEM',
                  r'CSI$_\rho$', r'CSI$_{WD}$',
                  r'$\alpha$-SNR', 'WD Budget']

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#0072B2')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors and highlight best
    best_acc = max(float(r[1]) for r in rows)
    best_bem_vals = [float(r[2]) for r in rows if r[2] != '--']
    best_bem = max(best_bem_vals) if best_bem_vals else None

    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#D6EAF8')
            else:
                cell.set_facecolor('#FFFFFF')
            # Highlight best accuracy
            if j == 1 and float(rows[i-1][1]) == best_acc:
                cell.set_text_props(fontweight='bold', color='#006400')
            # Highlight best BEM
            if j == 2 and rows[i-1][2] != '--' and best_bem and float(rows[i-1][2]) == best_bem:
                cell.set_text_props(fontweight='bold', color='#006400')

    ax.set_title('Unified Metrics Summary (CIFAR-10, ResNet-20, 10 epochs, seed 42)',
                 fontsize=13, fontweight='bold', pad=25)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig11_metrics_table'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 12: Alignment Informativeness (H6)
# ═══════════════════════════════════════════════════════════════════════════
def fig12_alignment_informativeness():
    print('[Fig 12] Alignment informativeness (H6)...')
    metrics = load_json(PILOTS_DIR / 'metrics_computation' / 'metrics_results_v2.json')

    ais_data = metrics['metrics']['alignment_informativeness']
    loo_cv = ais_data['loo_cv_r_squared']
    wd_summary = ais_data['wd_level_summary']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) LOO-CV R^2 comparison
    ax = axes[0]
    predictors = list(loo_cv.keys())
    r2_vals = [loo_cv[p]['r2'] for p in predictors]
    ci_vals = [loo_cv[p]['ci_half'] for p in predictors]

    # Sort by R^2
    sorted_pairs = sorted(zip(predictors, r2_vals, ci_vals), key=lambda x: x[1], reverse=True)
    predictors_s, r2_s, ci_s = zip(*sorted_pairs)

    colors_bar = ['#009E73' if r > 0 else '#D55E00' for r in r2_s]
    bars = ax.barh(range(len(predictors_s)), r2_s, xerr=ci_s,
                   color=colors_bar, edgecolor='black', linewidth=0.5,
                   height=0.6, capsize=3)
    ax.set_yticks(range(len(predictors_s)))
    ax.set_yticklabels(predictors_s, fontsize=9)
    ax.set_xlabel(r'LOO-CV $R^2$')
    ax.set_title(r'(a) Predictor Comparison: LOO-CV $R^2$')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()

    # (b) WD level summary: accuracy vs WD
    ax = axes[1]
    wd_levels = sorted(wd_summary.keys(), key=lambda x: float(x.split('_')[1]))
    wd_vals = [float(w.split('_')[1]) for w in wd_levels]
    mean_accs = [wd_summary[w]['mean_acc'] for w in wd_levels]
    mean_gaps = [wd_summary[w]['mean_gap'] for w in wd_levels]

    ax2 = ax.twinx()
    line1, = ax.plot(wd_vals, mean_accs, 'o-', color='#0072B2', linewidth=2,
                     markersize=8, label='Mean Accuracy')
    line2, = ax2.plot(wd_vals, mean_gaps, 's--', color='#D55E00', linewidth=2,
                      markersize=8, label='Mean Gen Gap')

    ax.set_xlabel('Weight Decay ($\\lambda$)')
    ax.set_ylabel('Mean Accuracy (%)', color='#0072B2')
    ax2.set_ylabel('Mean Generalization Gap (%)', color='#D55E00')
    ax.set_title('(b) Accuracy & Gen Gap vs WD (CIFAR-100)')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor='#0072B2')
    ax2.tick_params(axis='y', labelcolor='#D55E00')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig12_alignment_informativeness'))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 13: UDWDC vs UDWDC-v2 Stability Comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig13_udwdc_v2_comparison():
    print('[Fig 13] UDWDC vs UDWDC-v2 stability comparison...')
    metrics = load_json(PILOTS_DIR / 'metrics_computation' / 'metrics_results_v2.json')

    stab = metrics['stability_comparison']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) CIFAR-10: accuracy + CSI comparison
    ax = axes[0]
    labels = ['UDWDC', 'UDWDC-v2']
    acc_vals = [stab['cifar10']['UDWDC_acc'], stab['cifar10']['UDWDC-v2_acc']]
    csi_vals = [stab['cifar10']['UDWDC_CSI_rho'], stab['cifar10']['UDWDC-v2_CSI_rho']]

    x_pos = np.arange(2)
    width = 0.35
    ax.bar(x_pos - width/2, acc_vals, width, label='Accuracy',
           color='#0072B2', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width/2, [c * 100 for c in csi_vals], width,
           label=r'CSI$_\rho$ ($\times$100)',
           color='#009E73', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.set_title('(a) CIFAR-10: Accuracy + CSI')
    ax.legend()

    for i, (a, c) in enumerate(zip(acc_vals, csi_vals)):
        ax.text(i - width/2, a + 0.3, f'{a:.2f}%', ha='center', fontsize=8)
        ax.text(i + width/2, c * 100 + 0.3, f'{c:.3f}', ha='center', fontsize=8)

    # (b) ImageNet: accuracy + WD budget
    ax = axes[1]
    im_udwdc_acc = stab['imagenet']['UDWDC_acc'] * 100
    im_v2_acc = stab['imagenet']['UDWDC-v2_acc'] * 100
    im_udwdc_wd = stab['imagenet']['UDWDC_wd_budget']
    im_v2_wd = stab['imagenet']['UDWDC-v2_wd_budget']

    ax.bar(0, im_udwdc_acc, width=0.5, color='#56B4E9', edgecolor='black', linewidth=0.5,
           label='UDWDC')
    ax.bar(1, im_v2_acc, width=0.5, color='#F0E442', edgecolor='black', linewidth=0.5,
           label='UDWDC-v2')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['UDWDC', 'UDWDC-v2'])
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('(b) ImageNet: Val Accuracy')
    ax.legend()

    for i, a in enumerate([im_udwdc_acc, im_v2_acc]):
        ax.text(i, a + 0.3, f'{a:.0f}%', ha='center', fontsize=9, fontweight='bold')

    # (c) Ablation variant WD budget check
    ax = axes[2]
    abl_variants = stab['ablation_variants']
    variant_names = list(abl_variants.keys())
    wd_budgets = [abl_variants[v]['total_wd_budget'] for v in variant_names]
    wd_positive = [abl_variants[v]['wd_budget_positive'] for v in variant_names]

    colors_check = ['#009E73' if p else '#D55E00' for p in wd_positive]
    bars = ax.bar(range(len(variant_names)), wd_budgets, color=colors_check,
                  edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([v.replace('_', '\n') for v in variant_names], fontsize=7)
    ax.set_ylabel('Total WD Budget')
    ax.set_title('(c) WD Budget > 0 Check')
    ax.set_yscale('log')

    # Legend
    pos_patch = mpatches.Patch(color='#009E73', label='Budget > 0')
    neg_patch = mpatches.Patch(color='#D55E00', label='Budget = 0')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8)

    plt.tight_layout()
    generated.append(save_fig(fig, 'fig13_udwdc_v2_stability'))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('  Paper-Ready Visualization Panel Generation')
    print(f'  Output: {OUT_DIR}')
    print('=' * 70)

    # Write PID file
    pid_path = RESULTS_DIR / 'visualization_panels.pid'
    pid_path.write_text(str(os.getpid()))

    fig_funcs = [
        ('fig01_rho_trajectories',         fig01_rho_trajectories),
        ('fig02_h1_unification',           fig02_h1_unification),
        ('fig03_ablation',                 fig03_ablation),
        ('fig04_batchsize_snr',            fig04_batchsize_snr),
        ('fig05_budget_efficiency',        fig05_budget_efficiency),
        ('fig06_h7_r2_distribution',       fig06_h7_r2_distribution),
        ('fig07_imagenet_training',        fig07_imagenet_training),
        ('fig08_csi_heatmap',              fig08_csi_heatmap),
        ('fig09_effective_lambda',         fig09_effective_lambda),
        ('fig10_bem_ranking',              fig10_bem_ranking),
        ('fig11_metrics_table',            fig11_metrics_table),
        ('fig12_alignment_informativeness', fig12_alignment_informativeness),
        ('fig13_udwdc_v2_comparison',      fig13_udwdc_v2_comparison),
    ]

    progress = {'status': 'running', 'generated': [], 'errors': []}
    progress_path = RESULTS_DIR / 'visualization_panels_PROGRESS.json'

    for name, func in fig_funcs:
        try:
            func()
            progress['generated'].append(name)
        except Exception as e:
            import traceback
            err_msg = f'{name}: {e}'
            print(f'  ERROR: {err_msg}')
            traceback.print_exc()
            errors.append(err_msg)
            progress['errors'].append(err_msg)
        progress_path.write_text(json.dumps(progress, indent=2))

    # Summary
    print('\n' + '=' * 70)
    print(f'  Generated: {len(generated)} figures')
    if errors:
        print(f'  Errors: {len(errors)}')
        for e in errors:
            print(f'    - {e}')

    # DONE marker
    done_data = {
        'task_id': 'visualization_panels',
        'status': 'success' if len(generated) >= 5 else 'partial',
        'summary': f'{len(generated)} figures generated, {len(errors)} errors',
        'final_progress': {
            'total_figures': len(generated),
            'figures': generated,
            'errors': errors,
        },
        'timestamp': datetime.now().isoformat(),
    }
    (RESULTS_DIR / 'visualization_panels_DONE').write_text(json.dumps(done_data, indent=2))

    # Clean up PID
    if pid_path.exists():
        pid_path.unlink()

    progress['status'] = 'complete'
    progress_path.write_text(json.dumps(progress, indent=2))

    pass_criteria = len(generated) >= 5
    print(f'\n  Pass criteria (>=5 figures, DPI>=300): {"PASS" if pass_criteria else "FAIL"}')
    return 0 if pass_criteria else 1


if __name__ == '__main__':
    sys.exit(main())
