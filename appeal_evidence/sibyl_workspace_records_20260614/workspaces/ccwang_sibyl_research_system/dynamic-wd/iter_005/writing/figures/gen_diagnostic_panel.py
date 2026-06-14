"""Generate Figure 7: Diagnostic metric panel (CSI, AIS, BEM vs accuracy)."""
import sys
sys.path.insert(0, '.')
from style_config import *
import numpy as np
import json, glob, os

setup_style()

# Collect data from summary files
results_base = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/iter_003/exp/results/full/cifar10/resnet20'
vgg_base = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/full/vgg16bn/cifar10'

accs, csis, aiss, bems, method_ids = [], [], [], [], []

for method in METHOD_ORDER:
    for seed in [42, 123, 456]:
        # ResNet-20 CIFAR-10
        f = os.path.join(results_base, method, f'seed_{seed}', 'summary.json')
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)
            accs.append(d['best_test_acc'])
            csis.append(d.get('final_csi', 0))
            aiss.append(d.get('final_ais', 0))
            bems.append(abs(d.get('final_bem', 0)))
            method_ids.append(method)

        # VGG-16-BN CIFAR-10
        f2 = os.path.join(vgg_base, method, f'seed_{seed}', 'summary.json')
        if os.path.exists(f2):
            with open(f2) as fp:
                d = json.load(fp)
            accs.append(d['best_test_acc'])
            csis.append(d.get('final_csi', 0))
            aiss.append(d.get('final_ais', 0))
            bems.append(abs(d.get('final_bem', 0)))
            method_ids.append(method)

accs = np.array(accs)
csis = np.array(csis)
aiss = np.array(aiss)
bems = np.array(bems)
method_colors = [COLORS.get(m, 'gray') for m in method_ids]

fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_FULL * 0.9, FIG_HEIGHT * 0.85))

# (a) CSI vs accuracy
ax = axes[0]
ax.scatter(csis, accs, c=method_colors, s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
from scipy import stats
if len(csis) > 2:
    r, p = stats.spearmanr(csis, accs)
    ax.set_title(f'(a) CSI vs Accuracy\n$r_s = {r:.2f}$, $p = {p:.2f}$', fontsize=10)
ax.set_xlabel('CSI')
ax.set_ylabel('Best Test Accuracy (%)')

# (b) AIS vs accuracy
ax = axes[1]
ax.scatter(aiss, accs, c=method_colors, s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
if len(aiss) > 2:
    r, p = stats.spearmanr(aiss, accs)
    ax.set_title(f'(b) AIS vs Accuracy\n$r_s = {r:.2f}$, $p = {p:.2f}$', fontsize=10)
ax.set_xlabel('AIS')
ax.set_ylabel('')

# (c) BEM vs accuracy
ax = axes[2]
ax.scatter(bems, accs, c=method_colors, s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
if len(bems) > 2:
    r, p = stats.spearmanr(bems, accs)
    ax.set_title(f'(c) BEM vs Accuracy\n$r_s = {r:.2f}$, $p = {p:.2f}$', fontsize=10)
ax.set_xlabel('BEM')
ax.set_ylabel('')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[m], edgecolor='black', linewidth=0.5,
                         label=METHOD_NAMES[m]) for m in METHOD_ORDER]
fig.legend(handles=legend_elements, loc='lower center', ncol=7,
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('diagnostic_panel.pdf')
plt.savefig('diagnostic_panel.png')
print("Saved diagnostic_panel.pdf and .png")
