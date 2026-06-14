# Methodology: Alignment-Aware Dynamic Weight Decay (AADWD)

## 1. Problem Setup

We study nonconvex SGD with time-varying weight decay (SGDW):

```
w_{t+1} = (1 - lambda_t) * w_t - gamma_t * g_t
```

where `g_t` is the stochastic gradient, `gamma_t` the learning rate, and `lambda_t` the dynamic weight decay coefficient determined by gradient-parameter alignment.

### Alignment Measure

The core quantity is the cosine alignment between gradient and parameter:

```
delta_t = |<nabla f_S(w_t), w_t>| / (||nabla f_S(w_t)|| * ||w_t||)
```

In practice, we use the mini-batch proxy `delta_hat_t` with EMA smoothing.

## 2. Algorithm Variants

### Conservative (Primary)
```
lambda_t = clip(c * gamma_t * (1 - EMA(delta_hat_t)), lambda_min, lambda_max)
```
- Low alignment → high decay (regularize freely)
- High alignment → low decay (gradient and weight aligned, preserve direction)

### Aggressive
```
lambda_t = clip(c * gamma_t * EMA(delta_hat_t), lambda_min, lambda_max)
```
- Opposite logic: high alignment → high decay (penalize when gradient reinforces current direction)

### Square (Theory-Matched)
```
lambda_t = clip(c * gamma_t^2 * (1 - EMA(delta_hat_t)), lambda_min, lambda_max)
```
- Satisfies the O(gamma_t^2) condition from Theorem 3.1

## 3. Baselines

| Method | Description |
|--------|-------------|
| No-WD | SGD without weight decay |
| Fixed-WD (grid) | Standard SGDW with lambda in {1e-4, 3e-4, 5e-4, 1e-3, 3e-3} |
| Stagewise-WD | Step-wise decay schedule: lambda_0 for first 50%, lambda_0/10 for next 30%, lambda_0/100 for last 20% |
| CWD | Cautious Weight Decay (ICLR 2026): binary sign-based masking |
| Random-Dynamic-WD | lambda_t = clip(c * gamma_t * U[0,1], lambda_min, lambda_max) — controls for mere time variation |
| Norm-Matched-WD | Fixed lambda adjusted per epoch to match ||w_t|| trajectory of AADWD — isolates alignment signal from norm control |

## 4. Architectures and Datasets

| Architecture | Dataset | Epochs | Batch Size | LR Schedule |
|-------------|---------|--------|------------|-------------|
| ResNet20 | CIFAR-10 | 200 | 128 | 0.1, cosine decay to 0 |
| ResNet20 | CIFAR-100 | 200 | 128 | 0.1, cosine decay to 0 |
| VGG16-BN | CIFAR-10 | 200 | 128 | 0.1, cosine decay to 0 |

Standard augmentation: random crop (32x32 with 4px padding), horizontal flip, normalize.

## 5. Metrics

### Primary
- **Test accuracy** (top-1): Main performance metric
- **Generalization gap**: Train accuracy - Test accuracy

### Secondary (Theory Verification)
- **Alignment trajectory** `delta_hat_t`: Per-step mini-batch alignment (raw + EMA)
- **Delta_bar_T vs delta_max**: Cumulative-weighted alignment vs worst-case
- **Implied stability bound ratio**: `exp(-sum lambda_t (1 - Delta_bar_T))` for dynamic vs `exp(-T lambda (1 - delta_max))` for fixed
- **Weight norm trajectory** `||w_t||`: Regularization effect
- **Lambda trajectory** `lambda_t`: Dynamic decay schedule visualization

### Diagnostic (Tier 0)
- **Pearson correlation**: Between mini-batch (B=128) and large-batch (B=8192) alignment
- **Phase structure**: Mean and std of `delta_hat_t` per training third (early/mid/late)

## 6. Hyperparameters

### AADWD-specific
| Param | Search Space | Default |
|-------|-------------|---------|
| c (base coefficient) | {0.25, 0.5, 1.0, 2.0} | 1.0 |
| beta (EMA decay) | {0.9, 0.99, 0.999} | 0.99 |
| lambda_min | 1e-6 | 1e-6 |
| lambda_max | 0.01 | 0.01 |
| epsilon (numerical stability) | 1e-8 | 1e-8 |

### Fixed-WD grid
| Param | Search Space |
|-------|-------------|
| lambda | {1e-4, 3e-4, 5e-4, 1e-3, 3e-3} |

### Shared
- Optimizer: SGD with momentum 0.9
- Batch size: 128
- Epochs: 200 (full), 20 (pilot)
- Seeds: [42] for pilots, [42] for full (single seed per config.yaml)
- LR: 0.1 with cosine annealing

## 7. Evaluation Protocol

### Tier 0: Diagnostic (Gate for H3)
1. Train ResNet20/CIFAR-10 with 4 fixed WD values for 200 epochs
2. At 9 checkpoints (epochs 20, 50, 80, 100, 120, 150, 170, 190, 200), compute:
   - Mini-batch alignment (B=128, 50 batches)
   - Large-batch alignment (B=8192, full training set)
3. **Pass criterion**: Pearson r > 0.85 between mini-batch EMA and large-batch alignment
4. **Fail action**: Increase beta to 0.999, try Kalman filter; if still r < 0.6, pivot to empirical candidate

### Tier 1: Core Comparison (H1, H2, H4)
1. 7-method comparison on ResNet20/CIFAR-10, seed=42
2. All methods share same LR schedule and augmentation
3. Record: test acc, train acc, gen gap, weight norm, alignment trajectory
4. For AADWD variants: record lambda_t trajectory and cumulative contraction product

### Tier 2: Robustness and Ablations (H4, H5)
1. Cross-architecture: ResNet20/CIFAR-100, VGG16-BN/CIFAR-10
2. Hyperparameter sensitivity: c in {0.25, 0.5, 1.0, 2.0} on ResNet20/CIFAR-10
3. EMA ablation: beta in {0.9, 0.99, 0.999}
4. Ablation controls: Random-Dynamic-WD, Norm-Matched-WD, Equivalent-Cumulative-WD

## 8. Implementation Notes

### AADWD Integration
- Implement as a custom PyTorch optimizer wrapper or optimizer step hook
- Core computation per step: one inner product `<g_t, w_t>` and two norms — O(d) overhead
- Log all metrics to JSON lines format for post-processing

### CWD Baseline
- Implement per the ICLR 2026 paper: apply weight decay only when `sign(update) == sign(param)` coordinate-wise
- Use the authors' recommended default settings

### Logging
- Per-epoch: train loss, train acc, test acc, weight norm, mean lambda_t, mean delta_hat_t
- Per-step (sampled every 100 steps): lambda_t, delta_hat_t, EMA(delta_hat_t), ||w_t||, ||g_t||

## 9. Expected Visualizations

- **Figure 1**: Architecture diagram — overall AADWD pipeline showing alignment computation and decay modulation
- **Figure 2**: Alignment trajectory — delta_hat_t (raw + EMA) across training phases for different WD settings (Tier 0)
- **Figure 3**: Scatter plot — mini-batch vs large-batch alignment with Pearson r annotation (Tier 0)
- **Table 1**: Main benchmark results — 7 methods × {test acc, gen gap, weight norm} on ResNet20/CIFAR-10 (Tier 1)
- **Figure 4**: Lambda trajectory — dynamic decay schedule for Conservative/Aggressive/Square variants (Tier 1)
- **Figure 5**: Cumulative contraction — Delta_bar_T vs delta_max and implied stability bounds (Tier 1)
- **Table 2**: Cross-architecture results — ResNet20/CIFAR-100, VGG16/CIFAR-10 (Tier 2)
- **Figure 6**: Hyperparameter sensitivity — accuracy vs. c (AADWD) and accuracy vs. lambda (fixed), showing plateau width (Tier 2)
- **Figure 7**: Ablation results — bar chart comparing AADWD, Random-Dynamic, Norm-Matched, Equiv-Cumulative (Tier 2)
- **Figure 8**: Training loss curves — convergence comparison across methods (H1 verification)

## 10. Shared Resources

- **Dataset**: CIFAR-10, CIFAR-100 (auto-download via torchvision)
- **Checkpoints**: None required (train from scratch)
- **Code dependencies**: PyTorch >= 2.0, torchvision, numpy, matplotlib, scipy (for Pearson r)
