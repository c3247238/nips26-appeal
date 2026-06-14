"""Generate PMP-WD bang-bang switching pattern visualization."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

setup_style()

# Read PMP-WD seed 42 epoch metrics
metrics_file = os.path.join(os.path.dirname(__file__), '..', '..', 
    'exp', 'results', 'full', 'pmpwd', 'cifar10', 'resnet20', 'seed_42', 'epoch_metrics.jsonl')

epochs, sigmas, indicators, switch_rates, effective_wds = [], [], [], [], []
with open(metrics_file) as f:
    for line in f:
        d = json.loads(line)
        epochs.append(d['epoch'])
        sigmas.append(d.get('pmpwd_sigma', 0))
        indicators.append(d.get('pmpwd_indicator', 0))
        switch_rates.append(d.get('pmpwd_switch_rate', 0))
        effective_wds.append(d.get('effective_wd', 0))

epochs = np.array(epochs)
sigmas = np.array(sigmas)
switch_rates = np.array(switch_rates)
effective_wds = np.array(effective_wds)

fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5), sharex=True)

# Top: switching function sigma(t)
ax1 = axes[0]
ax1.plot(epochs, sigmas * 1000, color=COLORS['pmpwd'], linewidth=1.5)
ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.fill_between(epochs, sigmas * 1000, 0, where=np.array(sigmas) > 0, 
                  alpha=0.2, color=COLORS['pmpwd'], label='$\\sigma > 0$ (decay ON)')
ax1.fill_between(epochs, sigmas * 1000, 0, where=np.array(sigmas) <= 0, 
                  alpha=0.2, color=COLORS['no_wd'], label='$\\sigma \\leq 0$ (decay OFF)')
ax1.set_ylabel('$\\sigma(t) = \\langle p(t), w(t) \\rangle$ ($\\times 10^{-3}$)')
ax1.set_title('PMP-WD Bang-Bang Switching (CIFAR-10, ResNet-20, Seed 42)')
ax1.legend(fontsize=FONT_SIZE - 2)

# Bottom: switch rate and effective WD
ax2 = axes[1]
ax2.plot(epochs, switch_rates, color=COLORS['pmpwd'], label='Switch rate')
ax2.axhline(y=0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Fraction of params with decay ON')
ax2.set_ylim(0, 1)
ax2.legend(fontsize=FONT_SIZE - 2)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'pmpwd_switching.pdf'))
print("Saved pmpwd_switching.pdf")
