"""
Generate hyperparameter sensitivity plots for tier2_hyperparam_sensitivity.
Produces:
  1. c_sweep_test_acc.png   - test_acc vs c
  2. c_sweep_weight_norm.png - weight_norm vs c
  3. c_sweep_mean_lambda.png - mean_lambda_t vs c
  4. beta_sweep_test_acc.png  - test_acc vs beta
  5. beta_sweep_weight_norm.png - weight_norm vs beta
  6. beta_sweep_mean_lambda.png - mean_lambda_t vs beta
  7. combined_sensitivity.png  - 2x3 panel
  8. test_acc_vs_epoch_c_sweep.png - training curves for c sweep
  9. test_acc_vs_epoch_beta_sweep.png - training curves for beta sweep
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(
    "/home/ccwang/sibyl-research-system/workspaces/dynamic-wd"
    "/current/exp/results/tier2_hyperparam_sensitivity"
)


def load_summary(run_name):
    p = RESULTS_DIR / run_name / "summary.json"
    with open(p) as f:
        return json.load(f)


def load_epoch_metrics(run_name):
    p = RESULTS_DIR / run_name / "epoch_metrics.jsonl"
    records = []
    with open(p) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# === Data collection ===
c_values = [0.001, 0.005, 0.01, 0.05, 0.1]
beta_fixed = 0.999

beta_values = [0.9, 0.99, 0.999, 0.9999]
c_fixed = 0.01

# Sweep 1: c
c_test_acc = []
c_weight_norm = []
c_mean_lambda = []
for c in c_values:
    run = f"aadwd_agg_c{c}_beta{beta_fixed}"
    s = load_summary(run)
    c_test_acc.append(s['final_test_acc'])
    c_weight_norm.append(s['final_weight_norm'])
    c_mean_lambda.append(s['final_mean_lambda_t'])

# Sweep 2: beta
beta_test_acc = []
beta_weight_norm = []
beta_mean_lambda = []
for beta in beta_values:
    run = f"aadwd_agg_c{c_fixed}_beta{beta}"
    s = load_summary(run)
    beta_test_acc.append(s['final_test_acc'])
    beta_weight_norm.append(s['final_weight_norm'])
    beta_mean_lambda.append(s['final_mean_lambda_t'])

# Load epoch curves
c_curves = {}
for c in c_values:
    run = f"aadwd_agg_c{c}_beta{beta_fixed}"
    c_curves[c] = load_epoch_metrics(run)

beta_curves = {}
for beta in beta_values:
    run = f"aadwd_agg_c{c_fixed}_beta{beta}"
    beta_curves[beta] = load_epoch_metrics(run)


# --- Helper ---
def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# === 1. Individual c-sweep plots ===
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(c_values, c_test_acc, 'o-', color='steelblue', linewidth=2, markersize=7)
for x, y in zip(c_values, c_test_acc):
    ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=9)
ax.set_xscale('log')
style_ax(ax, 'Test Accuracy vs c (beta=0.999)', 'c', 'Test Accuracy (%)')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'c_sweep_test_acc.png', dpi=150)
plt.close(fig)
print("Saved c_sweep_test_acc.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(c_values, c_weight_norm, 's-', color='coral', linewidth=2, markersize=7)
ax.set_xscale('log')
style_ax(ax, 'Weight Norm vs c (beta=0.999)', 'c', 'Weight Norm')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'c_sweep_weight_norm.png', dpi=150)
plt.close(fig)
print("Saved c_sweep_weight_norm.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(c_values, c_mean_lambda, '^-', color='green', linewidth=2, markersize=7)
ax.set_xscale('log')
ax.set_yscale('log')
style_ax(ax, 'Mean Lambda_t vs c (beta=0.999)', 'c', 'Mean lambda_t')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'c_sweep_mean_lambda.png', dpi=150)
plt.close(fig)
print("Saved c_sweep_mean_lambda.png")

# === 2. Individual beta-sweep plots ===
beta_labels = ['0.9', '0.99', '0.999', '0.9999']
x_pos = np.arange(len(beta_values))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_pos, beta_test_acc, 'o-', color='steelblue', linewidth=2, markersize=7)
for xp, y in zip(x_pos, beta_test_acc):
    ax.annotate(f'{y:.2f}%', (xp, y), textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(beta_labels)
style_ax(ax, 'Test Accuracy vs beta (c=0.01)', 'beta', 'Test Accuracy (%)')
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'beta_sweep_test_acc.png', dpi=150)
plt.close(fig)
print("Saved beta_sweep_test_acc.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_pos, beta_weight_norm, 's-', color='coral', linewidth=2, markersize=7)
ax.set_xticks(x_pos)
ax.set_xticklabels(beta_labels)
style_ax(ax, 'Weight Norm vs beta (c=0.01)', 'beta', 'Weight Norm')
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'beta_sweep_weight_norm.png', dpi=150)
plt.close(fig)
print("Saved beta_sweep_weight_norm.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_pos, beta_mean_lambda, '^-', color='green', linewidth=2, markersize=7)
ax.set_yscale('log')
ax.set_xticks(x_pos)
ax.set_xticklabels(beta_labels)
style_ax(ax, 'Mean Lambda_t vs beta (c=0.01)', 'beta', 'Mean lambda_t')
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'beta_sweep_mean_lambda.png', dpi=150)
plt.close(fig)
print("Saved beta_sweep_mean_lambda.png")

# === 3. Combined 2x3 panel ===
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# Row 0: c sweep
ax = axes[0, 0]
ax.plot(c_values, c_test_acc, 'o-', color='steelblue', linewidth=2, markersize=7)
ax.set_xscale('log')
style_ax(ax, 'Test Accuracy vs c\n(beta=0.999)', 'c', 'Test Accuracy (%)')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax = axes[0, 1]
ax.plot(c_values, c_weight_norm, 's-', color='coral', linewidth=2, markersize=7)
ax.set_xscale('log')
style_ax(ax, 'Weight Norm vs c\n(beta=0.999)', 'c', 'Weight Norm')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax = axes[0, 2]
ax.plot(c_values, c_mean_lambda, '^-', color='green', linewidth=2, markersize=7)
ax.set_xscale('log')
ax.set_yscale('log')
style_ax(ax, 'Mean lambda_t vs c\n(beta=0.999)', 'c', 'Mean lambda_t')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

# Row 1: beta sweep
ax = axes[1, 0]
ax.plot(x_pos, beta_test_acc, 'o-', color='steelblue', linewidth=2, markersize=7)
ax.set_xticks(x_pos)
ax.set_xticklabels(beta_labels)
style_ax(ax, 'Test Accuracy vs beta\n(c=0.01)', 'beta', 'Test Accuracy (%)')

ax = axes[1, 1]
ax.plot(x_pos, beta_weight_norm, 's-', color='coral', linewidth=2, markersize=7)
ax.set_xticks(x_pos)
ax.set_xticklabels(beta_labels)
style_ax(ax, 'Weight Norm vs beta\n(c=0.01)', 'beta', 'Weight Norm')

ax = axes[1, 2]
ax.plot(x_pos, beta_mean_lambda, '^-', color='green', linewidth=2, markersize=7)
ax.set_yscale('log')
ax.set_xticks(x_pos)
ax.set_xticklabels(beta_labels)
style_ax(ax, 'Mean lambda_t vs beta\n(c=0.01)', 'beta', 'Mean lambda_t')

fig.suptitle('AADWD-aggressive: Hyperparameter Sensitivity (ResNet20/CIFAR-10, 20 epochs PILOT)',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'combined_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved combined_sensitivity.png")


# === 4. Training curves for c sweep ===
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(c_values)))

fig, ax = plt.subplots(figsize=(8, 5))
for (c, color) in zip(c_values, colors):
    epochs_data = c_curves[c]
    epochs = [r['epoch'] for r in epochs_data]
    test_accs = [r['test_acc'] for r in epochs_data]
    ax.plot(epochs, test_accs, '-', color=color, linewidth=2, label=f'c={c}')

style_ax(ax, 'Test Accuracy Training Curves (c sweep, beta=0.999)',
         'Epoch', 'Test Accuracy (%)')
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'test_acc_vs_epoch_c_sweep.png', dpi=150)
plt.close(fig)
print("Saved test_acc_vs_epoch_c_sweep.png")

# === 5. Training curves for beta sweep ===
colors2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_values)))

fig, ax = plt.subplots(figsize=(8, 5))
for (beta, color) in zip(beta_values, colors2):
    epochs_data = beta_curves[beta]
    epochs = [r['epoch'] for r in epochs_data]
    test_accs = [r['test_acc'] for r in epochs_data]
    ax.plot(epochs, test_accs, '-', color=color, linewidth=2, label=f'beta={beta}')

style_ax(ax, 'Test Accuracy Training Curves (beta sweep, c=0.01)',
         'Epoch', 'Test Accuracy (%)')
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'test_acc_vs_epoch_beta_sweep.png', dpi=150)
plt.close(fig)
print("Saved test_acc_vs_epoch_beta_sweep.png")

# === 6. Lambda trajectory for beta sweep (showing EMA convergence behavior) ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean lambda per epoch, c sweep
ax = axes[0]
for (c, color) in zip(c_values, colors):
    epochs_data = c_curves[c]
    epochs = [r['epoch'] for r in epochs_data]
    lambdas = [r['mean_lambda_t'] for r in epochs_data]
    ax.plot(epochs, lambdas, '-', color=color, linewidth=2, label=f'c={c}')
style_ax(ax, 'Mean lambda_t per Epoch (c sweep, beta=0.999)', 'Epoch', 'Mean lambda_t')
ax.legend(fontsize=9)

# Mean lambda per epoch, beta sweep
ax = axes[1]
for (beta, color) in zip(beta_values, colors2):
    epochs_data = beta_curves[beta]
    epochs = [r['epoch'] for r in epochs_data]
    lambdas = [r['mean_lambda_t'] for r in epochs_data]
    ax.plot(epochs, lambdas, '-', color=color, linewidth=2, label=f'beta={beta}')
style_ax(ax, 'Mean lambda_t per Epoch (beta sweep, c=0.01)', 'Epoch', 'Mean lambda_t')
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(RESULTS_DIR / 'lambda_trajectories.png', dpi=150)
plt.close(fig)
print("Saved lambda_trajectories.png")

print("\nAll plots saved successfully.")
