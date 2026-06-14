#!/usr/bin/env python3
"""Generate pilot visualization panels from experimental data."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PILOTS_DIR = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent
RESULTS_DIR = PILOTS_DIR.parent  # exp/results

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

METHOD_COLORS = {
    'FixedWD': '#1f77b4',
    'CWD': '#ff7f0e',
    'SWD': '#2ca02c',
    'CPR': '#d62728',
    'DefazioCorrective': '#9467bd',
    'NoWD': '#8c564b',
    'UDWDC': '#e377c2',
}

METHOD_MARKERS = {
    'FixedWD': 'o',
    'CWD': 's',
    'SWD': '^',
    'CPR': 'D',
    'DefazioCorrective': 'v',
    'NoWD': 'x',
    'UDWDC': '*',
}

generated = []
progress = {"status": "running", "generated": [], "errors": []}


def save_progress():
    with open(RESULTS_DIR / 'visualization_panels_PROGRESS.json', 'w') as f:
        json.dump(progress, f, indent=2)


def panel1_rho_trajectories():
    """Panel 1: Per-layer rho_t trajectories from diagnostic_cifar10."""
    print("[Panel 1] Generating rho_t trajectories...")
    diag_dir = PILOTS_DIR / 'diagnostic_cifar10'
    summary_path = diag_dir / 'pilot_summary.json'

    with open(summary_path) as f:
        summary = json.load(f)

    methods = list(summary['per_method'].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mean rho_t across all layers per method
    ax = axes[0]
    for method in methods:
        traj = summary['per_method'][method]['mean_rho_trajectory']
        epochs = list(range(1, len(traj) + 1))
        ax.plot(epochs, traj,
                color=METHOD_COLORS.get(method, 'gray'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=5, linewidth=1.8, label=method)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Mean $\rho_t$ (across layers)')
    ax.set_title('(a) Mean Layer-Averaged $\\rho_t$ Trajectories')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Per-layer rho_t distribution at final epoch for select methods
    ax = axes[1]
    select_methods = ['FixedWD', 'CWD', 'UDWDC', 'NoWD']
    layer_data = {}
    for method in select_methods:
        traj_path = diag_dir / method / f'{method}_seed42_trajectories.json'
        if traj_path.exists():
            with open(traj_path) as f:
                traj = json.load(f)
            # Get final epoch rho_t for all conv/linear layers
            rho_final = []
            for layer_name, layer_info in traj['layers'].items():
                if 'weight' in layer_name and ('conv' in layer_name or 'linear' in layer_name or 'fc' in layer_name):
                    rho_final.append(layer_info['rho_t'][-1])
            layer_data[method] = rho_final

    positions = np.arange(len(select_methods))
    bp = ax.boxplot([layer_data.get(m, []) for m in select_methods],
                    positions=positions, widths=0.5, patch_artist=True)
    for i, method in enumerate(select_methods):
        bp['boxes'][i].set_facecolor(METHOD_COLORS.get(method, 'gray'))
        bp['boxes'][i].set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(select_methods, rotation=15)
    ax.set_ylabel(r'$\rho_t$ at final epoch')
    ax.set_title('(b) Per-Layer $\\rho_t$ Distribution (Epoch 10)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = OUT_DIR / 'panel1_rho_trajectories.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def panel2_ablation_bar():
    """Panel 2: Ablation results bar chart from ablation_cifar100."""
    print("[Panel 2] Generating ablation bar chart...")
    abl_path = PILOTS_DIR / 'ablation_cifar100' / 'pilot_summary.json'

    with open(abl_path) as f:
        summary = json.load(f)

    variants = summary['variants']
    names = list(variants.keys())
    accs = [variants[n]['final_test_acc'] for n in names]
    colors = []
    for n in names:
        if n == 'FixedWD':
            colors.append('#8c564b')  # baseline
        elif n == 'Full_PID':
            colors.append('#e377c2')  # full
        else:
            colors.append('#1f77b4')  # components

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, accs, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    # Highlight baseline
    baseline_acc = variants['FixedWD']['final_test_acc']
    ax.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label=f'FixedWD baseline ({baseline_acc:.1f}%)')

    display_names = []
    for n in names:
        dn = n.replace('_', '\n')
        display_names.append(dn)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=9)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('UDWDC Ablation Study (CIFAR-100, VGG-16-BN, 10 epochs)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=26)

    plt.tight_layout()
    out_path = OUT_DIR / 'panel2_ablation_bar.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def panel3_batchsize_snr():
    """Panel 3: Batch-size vs SNR plot from batchsize_sweep."""
    print("[Panel 3] Generating batch-size vs SNR plot...")
    bs_path = PILOTS_DIR / 'batchsize_sweep' / 'pilot_summary.json'

    with open(bs_path) as f:
        summary = json.load(f)

    snr_data = summary['snr_monotonicity']
    batch_sizes = summary['batch_sizes']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: SNR vs batch size
    ax = axes[0]
    for method in ['FixedWD', 'CWD', 'UDWDC']:
        snr_vals = snr_data[method]['snr_values']
        bs_list = [int(b) for b in snr_vals.keys()]
        vals = list(snr_vals.values())
        ax.plot(bs_list, vals,
                color=METHOD_COLORS.get(method, 'gray'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=7, linewidth=2, label=method)

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Alignment SNR')
    ax.set_title('(a) Alignment SNR vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Accuracy delta vs batch size
    ax = axes[1]
    acc_deltas = summary['accuracy_deltas']
    for method in ['CWD', 'UDWDC']:
        bs_list = sorted([int(b) for b in acc_deltas.keys()])
        deltas = [acc_deltas[str(b)][method] for b in bs_list]
        ax.plot(bs_list, deltas,
                color=METHOD_COLORS.get(method, 'gray'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=7, linewidth=2, label=f'{method} vs FixedWD')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Accuracy Delta vs FixedWD (%)')
    ax.set_title('(b) Accuracy Improvement over FixedWD')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / 'panel3_batchsize_snr.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def panel4_imagenet_training():
    """Panel 4: ImageNet training comparison from imagenet_main."""
    print("[Panel 4] Generating ImageNet training comparison...")
    im_path = PILOTS_DIR / 'imagenet_main' / 'pilot_summary.json'

    with open(im_path) as f:
        summary = json.load(f)

    results = summary['results']
    methods = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Training loss curves
    ax = axes[0]
    for method in methods:
        epochs_data = results[method]['epoch_results']
        epochs = [e['epoch'] for e in epochs_data]
        train_loss = [e['train_loss'] for e in epochs_data]
        ax.plot(epochs, train_loss,
                color=METHOD_COLORS.get(method, 'gray'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=6, linewidth=2, label=method)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) ImageNet Training Loss (ResNet-50, 2 epochs)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: Final validation accuracy bar chart
    ax = axes[1]
    val_accs = [(m, results[m]['final_val_acc']) for m in methods]
    val_accs.sort(key=lambda x: x[1], reverse=True)
    names_sorted = [v[0] for v in val_accs]
    accs_sorted = [v[1] for v in val_accs]
    colors = [METHOD_COLORS.get(n, 'gray') for n in names_sorted]

    bars = ax.barh(range(len(names_sorted)), accs_sorted, color=colors,
                   edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('Validation Accuracy (%)')
    ax.set_title('(b) ImageNet Validation Accuracy (2 epochs)')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs_sorted)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{acc:.2f}%', ha='left', va='center', fontsize=9)

    ax.invert_yaxis()

    plt.tight_layout()
    out_path = OUT_DIR / 'panel4_imagenet_training.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def panel5_metrics_table():
    """Panel 5: BEM/CSI/AIS metrics table from metrics_computation."""
    print("[Panel 5] Generating BEM/CSI/AIS metrics table...")
    met_path = PILOTS_DIR / 'metrics_computation' / 'metrics_results.json'

    with open(met_path) as f:
        metrics = json.load(f)

    # Build table data from CIFAR-10 results
    bem_data = metrics['metrics']['BEM']['cifar10_resnet20']['methods']
    csi_data = metrics['metrics']['CSI']['cifar10_resnet20_per_method_refined']
    ais_data = metrics['metrics']['AIS']['per_method_cifar10']

    methods_order = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective', 'NoWD', 'UDWDC']

    # Gather table rows
    rows = []
    for m in methods_order:
        acc = bem_data[m]['accuracy']
        bem = bem_data[m]['BEM']
        bem_str = f'{bem:.1f}' if bem is not None else 'N/A'

        csi_combined = csi_data[m]['CSI_combined'] if m in csi_data else None
        csi_str = f'{csi_combined:.4f}' if csi_combined is not None else 'N/A'

        ais_val = ais_data[m]['AIS'] if m in ais_data else None
        ais_str = f'{ais_val:.3f}' if ais_val is not None else 'N/A'

        snr = ais_data[m]['alpha_SNR'] if m in ais_data else None
        snr_str = f'{snr:.4f}' if snr is not None else 'N/A'

        rows.append([m, f'{acc:.2f}', bem_str, csi_str, ais_str, snr_str])

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    col_labels = ['Method', 'Acc (%)', 'BEM', 'CSI\n(combined)', 'AIS', r'$\alpha$-SNR']

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#D9E2F3')
            else:
                cell.set_facecolor('#FFFFFF')
            # Highlight best values
            if j == 1 and rows[i-1][1] == '82.06':  # Best accuracy
                cell.set_text_props(fontweight='bold', color='#2d7d2d')
            if j == 2 and rows[i-1][2] == '55.9':  # Best BEM
                cell.set_text_props(fontweight='bold', color='#2d7d2d')

    ax.set_title('Unified Metrics Summary (CIFAR-10, ResNet-20, 10 epochs)',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    out_path = OUT_DIR / 'panel5_metrics_table.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def panel6_csi_heatmap():
    """Panel 6 (bonus): CSI refined comparison heatmap."""
    print("[Panel 6] Generating CSI refined heatmap...")
    met_path = PILOTS_DIR / 'metrics_computation' / 'metrics_results.json'

    with open(met_path) as f:
        metrics = json.load(f)

    csi_data = metrics['metrics']['CSI']['cifar10_resnet20_per_method_refined']
    methods = ['FixedWD', 'DefazioCorrective', 'CWD', 'CPR', 'SWD', 'NoWD', 'UDWDC']
    components = ['CSI_temporal', 'CSI_spatial', 'CSI_combined']
    comp_labels = ['Temporal', 'Spatial', 'Combined']

    data = np.zeros((len(methods), len(components)))
    for i, m in enumerate(methods):
        for j, c in enumerate(components):
            val = csi_data[m][c]
            data[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 5))

    # Clip for visualization (UDWDC has negative values)
    vmin = min(-1, data.min())
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=1.0)

    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(components)):
            val = data[i, j]
            color = 'white' if val < 0 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='CSI Score', shrink=0.8)
    ax.set_title('Coupling Stability Index (CSI) - Refined\n(CIFAR-10, ResNet-20)', fontsize=12)

    plt.tight_layout()
    out_path = OUT_DIR / 'panel6_csi_heatmap.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def panel7_bem_ranking_divergence():
    """Panel 7 (bonus): BEM vs Accuracy ranking divergence."""
    print("[Panel 7] Generating BEM ranking divergence...")
    met_path = PILOTS_DIR / 'metrics_computation' / 'metrics_results.json'

    with open(met_path) as f:
        metrics = json.load(f)

    rankings = metrics['rankings']['cifar10_resnet20']
    acc_rank = rankings['by_raw_accuracy']
    bem_rank = rankings['by_BEM']

    methods_by_acc = sorted(acc_rank.keys(), key=lambda m: acc_rank[m]['rank'])

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(methods_by_acc))
    width = 0.35

    acc_ranks = [acc_rank[m]['rank'] for m in methods_by_acc]
    bem_ranks = []
    for m in methods_by_acc:
        if m in bem_rank:
            bem_ranks.append(bem_rank[m]['rank'])
        else:
            bem_ranks.append(len(methods_by_acc))  # NoWD not in BEM

    bars1 = ax.bar(x - width/2, acc_ranks, width, label='Accuracy Rank',
                   color='#4472C4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, bem_ranks, width, label='BEM Rank',
                   color='#ED7D31', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Method')
    ax.set_ylabel('Rank (lower is better)')
    ax.set_title('BEM vs Accuracy Ranking (CIFAR-10, ResNet-20)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_by_acc, rotation=20, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = OUT_DIR / 'panel7_bem_ranking.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    generated.append(str(out_path))
    return True


def main():
    print("=" * 60)
    print("Pilot Visualization Panel Generation")
    print("=" * 60)

    save_progress()

    panel_funcs = [
        ('panel1_rho_trajectories', panel1_rho_trajectories),
        ('panel2_ablation_bar', panel2_ablation_bar),
        ('panel3_batchsize_snr', panel3_batchsize_snr),
        ('panel4_imagenet_training', panel4_imagenet_training),
        ('panel5_metrics_table', panel5_metrics_table),
        ('panel6_csi_heatmap', panel6_csi_heatmap),
        ('panel7_bem_ranking', panel7_bem_ranking_divergence),
    ]

    errors = []
    for name, func in panel_funcs:
        try:
            func()
            progress['generated'].append(name)
        except Exception as e:
            err_msg = f"{name}: {e}"
            print(f"  ERROR: {err_msg}")
            errors.append(err_msg)
            progress['errors'].append(err_msg)
        save_progress()

    # Summary
    print("\n" + "=" * 60)
    print(f"Generated: {len(generated)} panels")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors:
            print(f"  - {e}")

    # Write DONE marker
    done_data = {
        "status": "complete",
        "total_panels": len(generated),
        "panels": generated,
        "errors": errors,
        "pass_criteria_met": len(generated) >= 3,
    }
    with open(RESULTS_DIR / 'visualization_panels_DONE', 'w') as f:
        json.dump(done_data, f, indent=2)

    progress['status'] = 'complete'
    save_progress()

    print(f"\nPass criteria (>=3 panels): {'PASS' if len(generated) >= 3 else 'FAIL'}")
    return 0 if len(generated) >= 3 else 1


if __name__ == '__main__':
    sys.exit(main())
