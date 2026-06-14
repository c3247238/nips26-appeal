"""Generate Lyapunov V_t decay curves for all methods on CIFAR-10/ResNet-20."""
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
    
    all_vt = []
    for seed_dir in sorted(glob.glob(os.path.join(mdir, 'seed_*'))):
        metrics_file = os.path.join(seed_dir, 'epoch_metrics.jsonl')
        if os.path.exists(metrics_file):
            vt = []
            with open(metrics_file) as f:
                for line in f:
                    d = json.loads(line)
                    vt.append(d.get('lyapunov_v', 0))
            all_vt.append(vt)
    
    if all_vt:
        min_len = min(len(v) for v in all_vt)
        arr = np.array([v[:min_len] for v in all_vt])
        mean_vt = arr.mean(axis=0)
        epochs = np.arange(min_len)
        ax.plot(epochs, mean_vt, label=METHOD_LABELS[method], color=COLORS[method])

ax.set_xlabel('Epoch')
ax.set_ylabel('$V_t$ (Lyapunov function)')
ax.set_title('Lyapunov Function Trajectories (CIFAR-10, ResNet-20)')
ax.legend(fontsize=FONT_SIZE - 2, loc='upper left')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'lyapunov_curves.pdf'))
print("Saved lyapunov_curves.pdf")
