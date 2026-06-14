"""Regenerate Figure 8 (alignment deviation analysis) with correct 7 methods, no PMP-WD."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy import stats

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

METHODS = ['constant', 'cosine_schedule', 'cwd_hard', 'swd', 'half_lambda', 'random_mask', 'no_wd']
LABELS = ['Constant', 'Cosine', 'CWD', 'SWD', r'Half-$\lambda$', 'Random', 'No WD']
COLORS = ['#1f77b4', '#ff7f0e', '#8c564b', '#e377c2', '#d62728', '#2ca02c', '#9467bd']
SEEDS = [42, 123, 456]

# Base path for AdamW results (iter_003)
base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '../../iter_003/exp/results/full/cifar10/resnet20')

# Load data
cumul_deltas = []  # cumulative alignment deviation
worst_deltas = []  # worst-case alignment deviation
gen_gaps = []      # generalization gap
method_indices = []
seed_indices = []

# First load constant AIS as baseline
constant_ais = {}
for s in SEEDS:
    path = os.path.join(base, 'constant', f'seed_{s}', 'epoch_metrics.jsonl')
    if os.path.exists(path):
        with open(path) as f:
            epochs = [json.loads(line) for line in f]
        constant_ais[s] = [e['ais'] for e in epochs]

# Now compute alignment deviation for each method
for mi, method in enumerate(METHODS):
    for si, seed in enumerate(SEEDS):
        path = os.path.join(base, method, f'seed_{seed}', 'epoch_metrics.jsonl')
        if not os.path.exists(path):
            print(f"WARNING: Missing {method}/seed_{seed}")
            continue

        with open(path) as f:
            epochs = [json.loads(line) for line in f]

        ais_vals = [e['ais'] for e in epochs]

        # Get gen gap from final epoch (train_acc - test_acc)
        final = epochs[-1]
        gen_gap = final['train_acc'] - final['test_acc'] if 'train_acc' in final else final.get('gen_gap', 0)

        # Compute alignment deviation relative to constant baseline
        if seed in constant_ais:
            T = min(len(ais_vals), len(constant_ais[seed]))
            deltas = [abs(ais_vals[t] - constant_ais[seed][t]) for t in range(T)]
            cumul_delta = np.mean(deltas)
            worst_delta = np.max(deltas)
        else:
            cumul_delta = 0.0
            worst_delta = 0.0

        cumul_deltas.append(cumul_delta)
        worst_deltas.append(worst_delta)
        gen_gaps.append(gen_gap)
        method_indices.append(mi)
        seed_indices.append(si)

cumul_deltas = np.array(cumul_deltas)
worst_deltas = np.array(worst_deltas)
gen_gaps = np.array(gen_gaps)
method_indices = np.array(method_indices)

# Compute Spearman correlations
rho_cumul, p_cumul = stats.spearmanr(cumul_deltas, gen_gaps)
rho_worst, p_worst = stats.spearmanr(worst_deltas, gen_gaps)

print(f"N = {len(gen_gaps)} method-seed combinations")
print(f"Cumulative: rho = {rho_cumul:.3f}, p = {p_cumul:.3f}")
print(f"Worst-case: rho = {rho_worst:.3f}, p = {p_worst:.3f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for mi, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
    mask = method_indices == mi
    ax1.scatter(cumul_deltas[mask], gen_gaps[mask], c=color, s=60,
                label=label, edgecolors='black', linewidth=0.5, zorder=3)
    ax2.scatter(worst_deltas[mask], gen_gaps[mask], c=color, s=60,
                label=label, edgecolors='black', linewidth=0.5, zorder=3)

ax1.set_xlabel(r'Cumulative Alignment $\bar{\delta}$', fontsize=12)
ax1.set_ylabel('Generalization Gap (%)', fontsize=12)
ax1.set_title(f'(a) $\\rho$ = {rho_cumul:.3f}', fontsize=13)

ax2.set_xlabel(r'Worst-Case Alignment sup$\delta$', fontsize=12)
ax2.set_ylabel('Generalization Gap (%)', fontsize=12)
ax2.set_title(f'(b) $\\rho$ = {rho_worst:.3f}', fontsize=13)
ax2.legend(loc='upper right', fontsize=8, ncol=1)

plt.tight_layout()
outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'theorem2_validation.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'theorem2_validation.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved theorem2_validation.png/pdf with {len(METHODS)} methods (no PMP-WD)")
