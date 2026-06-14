# Methodology: Lyapunov-Certified Dynamic Weight Decay

## Overview

This iteration addresses the critical experimental gaps identified by the supervisor and critic:
1. **PMP-WD has ZERO experimental evaluation** -- the paper's primary algorithmic contribution
2. **ImageNet experiments all FAILED** -- mandatory per project constraints
3. **Alignment/certificate diagnostic data not logged** -- theorems are unvalidated empirically
4. **NoBN ablation incomplete** -- no_wd missing 2 seeds
5. **VGG-16-BN on CIFAR-100 not run** -- needed for cross-architecture validation

The plan is structured in 5 phases with clear dependencies. Pilot experiments gate full runs.

## Phase 0: Infrastructure (PMP-WD Implementation + Instrumented Training)

### PMP-WD Implementation
Implement the PMP-WD optimizer as described in Theorem 4 of the proposal:
- `lambda*(t) = Lambda_max * I(<p(t), w(t)> > 0)` where `p(t) ~ EMA of past gradients` (momentum buffer)
- Per-parameter bang-bang control: WD is Lambda_max when costate-weight alignment is positive, 0 otherwise
- Practical approximation: use SGD momentum buffer as costate proxy (zero additional compute)
- Hyperparameters: Lambda_max (base WD rate), EMA decay (use SGD momentum coefficient)

### Instrumented Training Script
Extend `train_sgd.py` to log per-epoch diagnostic quantities:
- `delta_t`: gradient-weight cosine alignment (per-layer and global)
- `||w_t||`: weight norm (per-layer)
- `||g_t||`: gradient norm (per-layer)
- `r_t = ||g_t|| / ||w_t||`: gradient-to-weight ratio
- `V_t = f(w_t) + mu_t * ||w_t||^2`: Lyapunov function value (requires L estimate)
- `lambda_min(t), lambda_max(t)`: certified band bounds
- `effective_wd_t = lambda_t * ||w_t||^2`: effective regularization at each step
- Save to `diagnostics.jsonl` alongside `epoch_metrics.jsonl`

### Smoothness Constant Estimation
- Estimate L (smoothness) via power iteration on Hessian every 20 epochs
- Cache L estimates for certified band computation
- Fallback: use gradient Lipschitz proxy `L_hat = max_t ||g_{t+1} - g_t|| / (gamma_t * ||g_t||)`

## Phase 1: PMP-WD Validation (CIFAR-10/100, ResNet-20)

**Purpose**: Validate the paper's primary algorithmic contribution (Theorem 4).

### Pilot
- PMP-WD on CIFAR-10/ResNet-20, seed 42, 200 epochs
- Pass criteria: converges (no divergence), accuracy within 2% of constant WD baseline (88-92%)
- Verify bang-bang switching behavior in lambda(t) trajectory
- Estimated: 15 min

### Full Runs
- CIFAR-10/ResNet-20: PMP-WD x 3 seeds (42, 123, 456)
- CIFAR-100/ResNet-20: PMP-WD x 3 seeds (42, 123, 456)
- Compare against all 7 existing methods from iter_003
- Track diagnostics (V_t, certified band, alignment)

## Phase 2: Certificate Visualization & Subsumption Verification (H1, H3)

**Purpose**: Validate Theorems 1 and 3 -- the core theoretical contributions.

### Instrumented Reruns (CIFAR-10/ResNet-20)
- Rerun constant, cwd_hard, cosine_schedule, swd, PMP-WD with full diagnostics
- 3 seeds each, 200 epochs, instrument every epoch
- Key output: certified band [lambda_min(t), lambda_max(t)] overlay plots showing each method's lambda(t) within the band
- Compute subsumption fraction: % of steps each method lies within certified band (target >= 95%)

### Certified Band Analysis
- Compute band width `lambda_max(t) - lambda_min(t)` across training
- Verify band widens early, narrows late (H1 prediction)
- Compare band width: BN vs non-BN architectures (H5 connection)

## Phase 3: Architecture Expansion

### 3A: VGG-16-BN on CIFAR-100
- 7 methods (no_wd, constant, cosine_schedule, cwd_hard, swd, half_lambda, random_mask) + PMP-WD = 8 methods
- 3 seeds each = 24 runs
- Reuse VGG-16-BN CIFAR-10 data from iter_005

### 3B: Complete NoBN Ablation (H5)
- no_wd: run seeds 123, 456 (seed 42 complete from iter_005)
- Add: cosine_schedule, swd, half_lambda, random_mask, PMP-WD on NoBN (3 seeds each)
- Total: 17 new NoBN runs
- Key test: does alignment-aware WD differentiate by >1% without BN?

### 3C: ImageNet/ResNet-50 (4 methods)
- Methods: constant, cwd_hard, cosine_schedule, PMP-WD
- 3 seeds each = 12 runs
- 90 epochs, standard ImageNet training recipe (LR 0.1, cosine decay, warmup 5 epochs)
- Multi-GPU: 2 GPUs per run (DDP), batch size 256
- Estimated: 4-6 hours per run
- Track diagnostics every 5 epochs (full-batch alignment too expensive; use 10% subsample)

## Phase 4: Cumulative Alignment Grid (H2)

**Purpose**: Test whether cumulative alignment predicts generalization better than worst-case.

### Grid Design
- 4 WD strengths: {1e-4, 5e-4, 1e-3, 5e-3}
- 4 schedules: {constant, cosine, linear_decay, step_decay_0.1_at_100}
- 2 architectures: {ResNet-20, VGG-16-BN}
- 1 dataset: CIFAR-10 (sufficient for correlation analysis)
- 3 seeds each = 96 runs (but many overlap with existing data)

### Metrics
- Full-batch alignment delta_t computed every 10 epochs
- Cumulative alignment: bar_delta_T = (1/T) sum_t delta_t
- Worst-case alignment: sup_t delta_t
- Generalization gap: train_acc - test_acc
- Compute Spearman correlations and bootstrap 95% CIs

## Phase 5: Analysis & Visualization

### Planned Visualizations
- **Figure 1**: Certified band visualization -- lambda_min(t), lambda_max(t) with method trajectories overlaid (5 methods, CIFAR-10/ResNet-20)
- **Figure 2**: PMP-WD bang-bang switching trajectory vs CWD binary mask (side-by-side)
- **Figure 3**: Method spread bar chart (BN vs non-BN) -- all 8 methods
- **Figure 4**: Alignment trajectory heatmap (delta_t across training for each method)
- **Figure 5**: Cumulative vs worst-case alignment correlation scatter plot
- **Figure 6**: Lyapunov function V_t decay curves for all methods
- **Table 1**: Main results (CIFAR-10, CIFAR-100, ImageNet) x 8 methods with mean +/- std
- **Table 2**: Subsumption verification -- % steps in certified band per method
- **Table 3**: BN vs NoBN accuracy spread and AIS scores

## Baselines

All experiments share:
- **Optimizer**: SGD with momentum 0.9
- **LR schedule**: Cosine decay from 0.1 to 0 with 5-epoch warmup
- **Base WD**: 5e-4 (except no_wd and half_lambda)
- **Batch size**: 128 (CIFAR), 256 (ImageNet)
- **Epochs**: 200 (CIFAR), 90 (ImageNet)
- **Data augmentation**: Standard (random crop, horizontal flip, normalize)

## Evaluation Metrics

1. **Test accuracy** (mean +/- std across 3 seeds)
2. **Generalization gap** (train_acc - test_acc)
3. **Lyapunov function** V_T at convergence
4. **Subsumption fraction** (% steps in certified band)
5. **Alignment metrics**: cumulative alignment, worst-case alignment, AIS
6. **CSI** (Coupling Stability Index): variance of effective WD relative to constant baseline

## Expected Visualizations

- Architecture diagram: Lyapunov control framework pipeline (state-space -> certificate -> control law -> subsumption)
- Figure 1: Certified band with method trajectories (core theoretical contribution)
- Figure 2: PMP-WD bang-bang behavior vs CWD binary mask
- Figure 3: Method accuracy spread (BN vs non-BN bar chart)
- Figure 4: Alignment trajectory heatmap across methods
- Figure 5: Cumulative vs worst-case alignment scatter (H2 test)
- Figure 6: Lyapunov V_t convergence curves
- Table 1: Main benchmark results (dataset x architecture x method)
- Table 2: Subsumption verification (method x certified band compliance)
- Table 3: BN ablation results (BN vs NoBN x method)

## Falsification Criteria

1. If constant/CWD violate certified band >20% of steps => certificate needs relaxation
2. If |rho(bar_delta, gen_gap)| - |rho(sup delta, gen_gap)| < 0.05 => H2 weakened
3. If PMP-WD diverges or performs >2% below constant => implementation error or theory problem
4. If alignment-aware WD outperforms constant by >0.5% ON BN architectures (p<0.05) => H5 falsified
5. If both |rho| < 0.3 => alignment framework questionable
