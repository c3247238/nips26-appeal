# Experiment Methodology: Stability-Optimal Weight Decay

## Overview

This iteration validates the Lyapunov Control Framework for unifying dynamic WD methods.
We build on iter_003 data (7 methods × 3 seeds × 2 datasets on ResNet-20/SGD/CIFAR-10/100)
and extend to: (1) VGG-16-BN architecture, (2) ImageNet/ResNet-50, (3) non-BN ablation,
(4) PMP-derived schedule, (5) Hill-function cooperativity, and (6) SGD baselines to
close the theory–experiment optimizer gap.

## Prior Evidence (iter_003 Summary)

| Dataset | Arch | Best Method | Acc | Constant Acc | Spread |
|---------|------|------------|-----|-------------|--------|
| CIFAR-10 | ResNet-20 | constant | 91.30 | 91.30 | 0.90% |
| CIFAR-100 | ResNet-20 | random_mask | 65.40 | 65.19 | 1.81% |

Key observations:
- AIS range: 0.27–0.43, consistently < 0.5 → supports H3 (budget equivalence)
- SWD CSI ~1.15 vs constant ~0.83 → supports H6 (CSI instability)
- All experiments used SGD optimizer, 200 epochs, lr=0.1, wd=5e-4, batch_size=128

## Methods Under Test

| ID | Method | Phi_t Definition | Source |
|----|--------|-----------------|--------|
| constant | Fixed WD | Phi_t = 1 | Standard |
| cwd_hard | CWD (binary masking) | Phi_t = I[delta_t > 0] | ICLR 2026 |
| swd | Sensitivity-weighted WD | Phi_t = s_t / mean(s_t) | Gradient-sensitivity |
| cosine_schedule | Cosine-annealed WD | Phi_t = cosine(t/T) | Schedule-based |
| half_lambda | Half-lambda WD | Phi_t = 0.5 | Budget ablation |
| no_wd | No weight decay | Phi_t = 0 (lambda=0) | Ablation control |
| random_mask | Random binary mask | Phi_t = Bernoulli(0.5) | Ablation control |
| pmp_wd | PMP-optimal schedule | Phi_t ∝ gamma*(1-delta)/||w|| | New (H4) |
| hill_n2 | Hill function n=2 | Phi_t = (1-delta)^2/(delta*^2+(1-delta)^2) | New (H7) |
| hill_n4 | Hill function n=4 | Phi_t = (1-delta)^4/(delta*^4+(1-delta)^4) | New (H7) |

## Experimental Setup

### Phase 1: VGG-16-BN on CIFAR-10/100 (extends architecture coverage)
- **Architecture**: VGG-16 with Batch Normalization
- **Datasets**: CIFAR-10, CIFAR-100
- **Methods**: constant, cwd_hard, swd, cosine_schedule, half_lambda, no_wd, random_mask
- **Seeds**: 42, 123, 456
- **Training**: SGD, lr=0.1, wd=5e-4, batch_size=128, epochs=200, cosine LR schedule
- **Rationale**: Addresses lesson-learned gap — VGG-16-BN data existed in iter_005 but was never integrated

### Phase 2: Non-BN Ablation (H5: AIS architecture dependence)
- **Architecture**: ResNet-20 without BN (replace BN with identity + bias)
- **Dataset**: CIFAR-10
- **Methods**: constant, cwd_hard, pmp_wd
- **Seeds**: 42, 123, 456
- **Training**: SGD, lr=0.01 (lower due to no BN), wd=5e-4, batch_size=128, epochs=200
- **Rationale**: Tests whether AIS > 0.5 without BN and whether dynamic WD helps

### Phase 3: PMP-WD and Hill Function Variants (H4, H7)
- **Architecture**: ResNet-20
- **Datasets**: CIFAR-10, CIFAR-100
- **Methods**: pmp_wd, hill_n2, hill_n4
- **Seeds**: 42, 123, 456
- **Training**: SGD, lr=0.1, wd=5e-4, batch_size=128, epochs=200
- **Rationale**: Tests PMP-derived optimal schedule and Hill cooperativity

### Phase 4: ImageNet / ResNet-50 (large-scale validation)
- **Architecture**: ResNet-50
- **Dataset**: ImageNet (full)
- **Methods**: constant, cwd_hard, swd, cosine_schedule, pmp_wd
- **Seeds**: 42, 123, 456
- **Training**: SGD, lr=0.1, wd=1e-4, batch_size=256, epochs=90, cosine LR, warmup 5 epochs
- **Rationale**: Addresses critical lesson-learned gap — ImageNet required by project constraints

### Phase 5: Steady-State Formula Validation (H1)
- **Architecture**: ResNet-20
- **Dataset**: CIFAR-10
- **Methods**: constant, cwd_hard, cosine_schedule
- **Seeds**: 42
- **Per-layer tracking**: ||g||, ||w||, cos(theta) logged every epoch
- **Analysis**: Compare predicted r* = gamma * E[||g||cos(theta)] / (lambda * E[Phi_t]) vs observed ||w||

### Phase 6: Analysis & Visualization
- Statistical tests: paired t-tests with Bonferroni correction, TOST equivalence at ±0.5%
- CSI vs accuracy correlation analysis
- AIS trajectory comparison (BN vs non-BN)
- Steady-state formula prediction error

## Baselines

1. **Constant WD** (lambda=5e-4): Primary baseline for all comparisons
2. **No WD** (lambda=0): Lower bound
3. **Half-lambda** (lambda=2.5e-4): Budget-ablation control
4. **Random mask**: Noise control (same expected budget as CWD)

## Metrics

| Metric | Definition | Hypothesis |
|--------|-----------|------------|
| Test accuracy (best & final) | Standard | All |
| CSI (Coupling Stability Index) | Var(R_t) across layers and time | H2, H6 |
| AIS (Alignment Informativeness Score) | mean(|delta_hat_t|) across layers | H3, H5 |
| BEM (Budget Equivalence Metric) | |sum(lambda_t) - T*lambda_const| / (T*lambda_const) | H3 |
| Weight norm ||w|| | L2 norm at convergence | H1 |
| Generalization gap | train_acc - test_acc | All |

## Evaluation Criteria

### Pre-registered falsification thresholds:
- **H1 falsified** if steady-state formula error > 30% for any method
- **H2 falsified** if alignment-aware WD CSI reduction < 10% vs constant
- **H3 falsified** if any dynamic method achieves > 0.5% accuracy gain at p < 0.05 (n=3+ seeds) under AIS < 0.5
- **H5 falsified** if AIS(non-BN) ≈ AIS(BN) (difference < 0.05)
- **H6 falsified** if SWD CSI is NOT significantly higher than constant WD
- **H7 falsified** if Hill n=1 (linear) is optimal or n has no effect

## Expected Visualizations

- **Table 1**: Main results comparison (method × dataset × architecture)
- **Table 2**: Statistical significance tests (p-values, effect sizes)
- **Figure 1**: Architecture diagram — Phi-Modulator taxonomy with control-theoretic framing
- **Figure 2**: AIS trajectories across training (BN vs non-BN, per method)
- **Figure 3**: CSI vs accuracy scatter plot (all methods, all architectures)
- **Figure 4**: Steady-state formula validation (predicted vs observed ||w||)
- **Figure 5**: Weight norm evolution curves (per method)
- **Figure 6**: Hill function response curves and accuracy by n
- **Figure 7**: ImageNet results comparison bar chart
- **Figure 8**: BEM-controlled equivalence visualization

## Compute Budget

| Phase | Tasks | GPU-hours (est.) |
|-------|-------|-----------------|
| VGG-16-BN CIFAR | 7 methods × 3 seeds × 2 datasets = 42 | ~21 |
| Non-BN ablation | 3 methods × 3 seeds = 9 | ~4 |
| PMP/Hill variants | 3 methods × 3 seeds × 2 datasets = 18 | ~9 |
| ImageNet ResNet-50 | 5 methods × 3 seeds = 15 | ~75 |
| Steady-state validation | 3 methods × 1 seed (detailed logging) = 3 | ~2 |
| **Total** | **87 tasks** | **~111 GPU-hours** |

With 8 GPUs available, parallelism can reduce wall-clock time to ~14 hours.
