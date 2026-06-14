"""Generate weight norm trajectories for all methods."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

setup_style()

base = os.path.join(os.path.dirname(__file__), '..', '..', 'exp', 'results', 'full')
instrumented = os.path.join(base, 'instrumented', 'cifar10', 'resnet20')
pmpwd_dir = os.path.join(base, 'pmpwd', 'cifar10', 'resnet20')

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

for method in METHOD_ORDER:
    if method == 'pmpwd':
        mdir = pmpwd_dir
    else:
        mdir = os.path.join(instrumented, method)
    
    all_norms = []
    for seed_dir in sorted(glob.glob(os.path.join(mdir, 'seed_*'))):
        metrics_file = os.path.join(seed_dir, 'epoch_metrics.jsonl')
        if os.path.exists(metrics_file):
            norms = []
            with open(metrics_file) as f:
                for line in f:
                    d = json.loads(line)
                    norms.append(d.get('weight_norm', 0))
            all_norms.append(norms)
    
    if all_norms:
        min_len = min(len(v) for v in all_norms)
        arr = np.array([v[:min_len] for v in all_norms])
        mean_norm = arr.mean(axis=0)
        std_norm = arr.std(axis=0)
        epochs = np.arange(min_len)
        ax.plot(epochs, mean_norm, label=METHOD_LABELS[method], color=COLORS[method])
        ax.fill_between(epochs, mean_norm - std_norm, mean_norm + std_norm, alpha=0.1, color=COLORS[method])

ax.set_xlabel('Epoch')
ax.set_ylabel('Weight Norm $\\|w_t\\|$')
ax.set_title('Weight Norm Trajectories (CIFAR-10, ResNet-20)')
ax.legend(fontsize=FONT_SIZE - 2)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'weight_norm_trajectories.pdf'))
print("Saved weight_norm_trajectories.pdf")
