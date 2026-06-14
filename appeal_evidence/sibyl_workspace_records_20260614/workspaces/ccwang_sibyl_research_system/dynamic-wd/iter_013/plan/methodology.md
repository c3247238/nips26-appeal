# Methodology: Unified Dynamic Weight Decay Framework

## Overview

This experiment plan validates the Equilibrium-Driven Weight Decay (EqWD) algorithm and the Cumulative Alignment Contraction Theory across three scales: CIFAR-10 (pilot), CIFAR-100 (core), and ImageNet-1K (primary evidence). The plan is structured as a four-phase pipeline with hard go/no-go gates between phases.

**Key lessons from prior iterations:**
- ImageNet experiments are MANDATORY (CIFAR-only is insufficient for 2026 top venues)
- Multi-architecture coverage (ResNet + VGG) must be IN the paper, not just in logs
- Statistical power: N=3 seeds minimum; null-result claims require TOST equivalence testing
- Proofs appendix must be written alongside experiments (not deferred)
- Figures must match text claims (no phantom narrowing/convergence claims)

## Phase 1: Pilot Validation (CIFAR-10, ~30 min)

### Purpose
Validate that EqWD produces non-trivial ratio deviation signal and does not crash/diverge before investing GPU-hours in full experiments.

### Setup
- **Dataset**: CIFAR-10 (50K train / 10K test, standard augmentation: RandomCrop 32 + RandomHorizontalFlip + Normalize)
- **Architecture**: VGG-16-BN (15M params)
- **Optimizer**: SGD with momentum 0.9, cosine LR schedule, initial LR=0.1
- **Batch size**: 128
- **Epochs**: 50 (pilot-scale)
- **Seed**: 42
- **WD methods**: Fixed (0.0005), SWD, CWD, EqWD (beta=0.5, 1.0, 2.0)

### Go/No-Go Criteria
1. EqWD ratio deviation signal variance > 0.01 across layers
2. EqWD does not diverge (final loss < 2x baseline)
3. EqWD accuracy within 2% of fixed WD (no catastrophic failure)
4. Per-layer ratio trajectories show distinct behavior across WD methods

### Diagnostics Logged
- Per-layer r_t = ||g_t|| / ||w_t|| every 10 steps
- Per-layer cosine similarity delta_hat_t every 10 steps
- Per-layer effective WD lambda_t^l every 10 steps
- Training loss, test accuracy every epoch

## Phase 2: Core CIFAR Experiments (CIFAR-100, ~4 hours wall-clock)

### Setup
- **Dataset**: CIFAR-100 (50K train / 10K test, standard augmentation)
- **Architectures**: ResNet-20 (0.27M params), VGG-16-BN (15M params)
- **Optimizer**: SGD with momentum 0.9, cosine LR schedule, LR=0.1
- **Batch size**: 128
- **Epochs**: 200
- **Seeds**: 42, 123, 456
- **WD methods** (7 total):
  1. No-WD SGD (baseline lower bound)
  2. Fixed SGDW (lambda=0.0005, baseline upper bound)
  3. SWD (gradient-norm scheduled, NeurIPS 2023)
  4. CWD (binary alignment mask, ICLR 2026)
  5. CPR (per-matrix constraint, NeurIPS 2024)
  6. CAWD (continuous alignment modulation, our variant)
  7. EqWD (ratio-deviation driven, our primary contribution)

### Metrics
- Test accuracy (primary): mean +/- std over 3 seeds
- Generalization gap: test_loss - train_loss
- BEM: Budget Equivalence Metric (accuracy per FLOP)
- CSI: Coupling Stability Index = Var(r_t) / E[r_t] per layer
- AIS: Alignment Informativeness Score (MI estimate)
- C_T: Cumulative alignment contraction index

### Ablation Studies
1. **Beta sensitivity** (EqWD): beta in {0.1, 0.5, 1.0, 2.0, 5.0}, ResNet-20/CIFAR-100, seed 42
2. **EMA decay rate**: alpha in {0.8, 0.9, 0.95, 0.99}, ResNet-20/CIFAR-100, seed 42
3. **Layer-type-aware vs. uniform**: EqWD with/without BN layer distinction, VGG-16-BN/CIFAR-100, 3 seeds
4. **NoBN ablation**: VGG-16 (no BN) vs. VGG-16-BN, all 7 methods, 3 seeds

### Alignment Diagnostic (H3)
- Single run per architecture-dataset pair
- Log delta_hat_t, ||g_t||, ||w_t|| per layer per step (every 10 steps)
- Compute k-NN mutual information I(delta_hat_t; test_acc | ||g_t||, ||w_t||) with bootstrap CI
- This is a PREREQUISITE diagnostic: if CI includes 0 for >=3/4 settings, alignment-aware WD is theoretically unjustified

## Phase 3: ImageNet Experiments (PRIMARY EVIDENCE, ~16-24 hours wall-clock)

### Setup
- **Dataset**: ImageNet-1K (1.28M train / 50K val, standard augmentation: RandomResizedCrop 224 + RandomHorizontalFlip + ColorJitter + Normalize)
- **Architectures**: ResNet-50 (25.6M params), ResNet-101 (44.5M params, if time permits)
- **Optimizer**: SGD with momentum 0.9, step LR schedule (decay 0.1 at epochs 30, 60, 80), initial LR=0.1
- **Batch size**: 256 per GPU (effective 512 with 2 GPUs via DDP)
- **Epochs**: 90
- **Seeds**: 42, 123, 456
- **WD methods** (6 total):
  1. Fixed SGDW (lambda=0.0001)
  2. SWD
  3. CWD
  4. CPR
  5. CAWD
  6. EqWD (beta=1.0, tuned from CIFAR results)

### GPU Strategy
- **ResNet-50**: 2 GPUs per run via DDP (batch 256/GPU = 512 effective)
- **ResNet-101**: 2 GPUs per run via DDP
- **Parallelism**: 4 concurrent runs on 8 GPUs (4 runs x 2 GPUs each)
- **Estimated time per run**: 4-5 hours for 90 epochs
- **Total wall-clock**: ~18 hours (6 methods x 3 seeds = 18 runs, 4 concurrent = 5 waves)

### Critical Design Decisions
- Use PyTorch DDP (not DataParallel) for multi-GPU — DataParallel has poor scaling
- Pre-download ImageNet to local NVMe to avoid I/O bottlenecks
- Use mixed precision (torch.cuda.amp) to fit larger batches and speed up training
- Save checkpoints at epoch 30, 60, 80, 90 for analysis
- Log per-layer diagnostics every 100 steps (not every 10, to reduce I/O at scale)

### Failure Diagnosis Protocol
Previous iterations had "two iterations of blind failure" on ImageNet with no root cause diagnosis. This time:
1. Run a 1-epoch sanity check first (5 min) to verify data loading, DDP init, and loss convergence
2. Monitor GPU utilization and memory during first 1000 steps
3. If OOM: reduce batch size to 128/GPU, then 64/GPU
4. If loss diverges: check LR schedule, gradient clipping, WD magnitude
5. Log errors to `exp/logs/imagenet_debug.log` with full stack traces

## Phase 4: Budget Equivalence & Control Experiments (~6-8 hours wall-clock)

### Budget Equivalence Test (H6)
- **Protocol**: 50 Optuna trials per method on CIFAR-100/ResNet-20
- **Search space**: LR in [0.01, 0.3], WD in [1e-5, 1e-2], beta in [0.1, 5.0] (for EqWD)
- **Fixed WD baseline**: 50 trials searching over LR and WD only
- **Metric**: Best test accuracy after 200 epochs, mean over 3 seeds
- **Decision criterion**: If no dynamic WD method achieves CI lower bound > 0 over tuned fixed WD, report as negative finding

### Control Experiments
1. **Phase-schedule control**: Replay average lambda_t trajectory from EqWD as a fixed schedule
2. **Gradient-norm-only control**: Replace ratio deviation with ||g_t|| normalized to [0,1]
3. **Noise injection**: Add Gaussian noise with sigma = std(delta_hat_t) to alignment signal
4. **Layer-type ablation**: CWD on only BN layers vs. only non-BN layers

## Expected Visualizations

### Tables
- **Table 1**: Main CIFAR-100 results (7 methods x 2 architectures x 3 seeds, mean +/- std)
- **Table 2**: ImageNet results (6 methods x ResNet-50 x 3 seeds, mean +/- std, Top-1 and Top-5)
- **Table 3**: Ablation results (beta, EMA alpha, layer-type-aware)
- **Table 4**: Budget equivalence comparison (tuned vs. default hyperparameters)
- **Table 5**: CSI and AIS metrics across all methods

### Figures
- **Figure 1**: Architecture diagram — unified Phi framework showing all WD methods as special cases
- **Figure 2**: Per-layer ratio trajectories r_t^l for representative layers under each method (line plot)
- **Figure 3**: Effective WD heatmap lambda_t^l across layers and training time (heatmap)
- **Figure 4**: Alignment cosine vs. ratio deviation scatter plot (correlation analysis)
- **Figure 5**: C_T vs. generalization gap across methods (scatter with regression line)
- **Figure 6**: Beta sensitivity curve (line plot, accuracy vs. beta)
- **Figure 7**: Multi-architecture comparison (grouped bar chart: ResNet-20, VGG-16-BN, ResNet-50)
- **Figure 8**: Training dynamics comparison (loss curves for all methods, CIFAR-100 and ImageNet)
- **Figure 9**: Budget equivalence results (box plot comparing tuned fixed vs. dynamic methods)
- **Figure 10**: Layer-type analysis — SNR of alignment signal in BN vs. non-BN layers (bar chart)

### Appendix Figures
- Per-layer weight norm trajectories under each method
- AIS heatmap by layer and training phase
- Full Optuna optimization curves for budget equivalence test
- NoBN ablation detailed results

## Reproducibility

### Software Requirements
- Python 3.12, PyTorch >= 2.2, torchvision
- CUDA 12.x with cuDNN
- numpy, scipy, scikit-learn (for MI estimation)
- optuna (for budget equivalence)
- matplotlib, seaborn (for visualization)

### Data
- CIFAR-10/100: torchvision.datasets.CIFAR10/CIFAR100 (auto-download)
- ImageNet-1K: must be pre-downloaded to local storage

### Compute
- 8x NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- Estimated total GPU-hours: ~120 (CIFAR: ~16, ImageNet: ~90, Budget equiv: ~16)
- Estimated wall-clock: ~3 days with 8 GPUs

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|-----------|----------|
| ImageNet training fails | 1-epoch sanity check; detailed error logging; batch size fallback | Scale down to ImageNet-100 subset |
| EqWD ratio deviation is flat | Increase beta; try multi-step smoothing | Report as negative finding; pivot to evaluation framework contribution |
| Alignment signal is uninformative (H3 fails) | Document rigorously; reframe paper | Drop CAWD, focus on ratio-based methods only |
| Budget equivalence shows no winner | This IS a contribution; frame as rigorous negative finding | Strengthen evaluation framework angle |
| Proof gap in Theorem 1 | Allocate dedicated time for proof writing alongside experiments | Weaken to conjecture with empirical support |
