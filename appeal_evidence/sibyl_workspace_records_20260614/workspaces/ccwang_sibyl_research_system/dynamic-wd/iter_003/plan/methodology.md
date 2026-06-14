# Methodology: Unified Dynamic Weight Decay Framework

## 1. Overview

This experiment plan tests 7 hypotheses (H1–H7) across 4 phases, progressing from pilot validation through CIFAR comprehensive experiments to ImageNet scale-up and architecture generalization. All methods share a single PyTorch codebase with identical training infrastructure.

## 2. Unified Codebase Design

### 2.1 Per-Parameter Modulation Interface

All dynamic WD methods are implemented through a unified `Phi(w, u, t)` interface:

```python
class WDModulator(ABC):
    def compute_phi(self, w: Tensor, u: Tensor, t: int, layer_info: dict) -> Tensor:
        """Return per-parameter modulation mask Phi in [0, inf)."""
        ...
```

| Method | Phi Implementation | Axis |
|--------|-------------------|------|
| AdamW (constant) | `torch.ones_like(w)` | -- |
| CWD | `(sign(w * u) >= 0).float()` | Directional |
| Soft-CWD (beta) | `torch.sigmoid(beta * w * u)` | Directional |
| SWD/AdamS | `h(grad_norm) * ones` | Temporal |
| Cosine WD | `cos_schedule(t, T) * ones` | Temporal |
| Linear WD | `linear_schedule(t, T) * ones` | Temporal |
| Inverse-sqrt WD | `isqrt_schedule(t, T) * ones` | Temporal |
| AdamWN (target tau) | `max(0, 1 - tau / norm(w)) * ones` | Target |
| AlphaDecay-style | `diag(alpha_per_layer)` | Spatial |
| CWD + Cosine | `cwd_mask * cos_schedule(t, T)` | Directional + Temporal |
| CWD + AdamWN | `cwd_mask * norm_match` | Directional + Target |
| Random mask | `bernoulli(p_match) * ones` | Control |
| Inverted mask | `(sign(w * u) < 0).float()` | Control (anti-alignment) |
| Effective-lambda matched | `ones * effective_ratio` | Control |

### 2.2 Training Infrastructure

- **Optimizer**: AdamW with decoupled WD (all experiments)
- **SGD baseline**: SGD + momentum (0.9) + constant WD (control)
- **LR schedule**: Cosine annealing with warmup (standard for all methods)
- **Data augmentation**: Standard (RandomCrop + RandomHorizontalFlip for CIFAR; RandAugment + MixUp for ImageNet)
- **Batch size**: 128 (CIFAR), 256 per GPU (ImageNet)
- **Seeds**: 42, 123, 456 for all experiments
- **Logging**: Per-epoch metrics + per-100-step diagnostic snapshots (weight norms, alignment cosines, effective LR, Phi values)

### 2.3 Hyperparameter Tuning Protocol

- **LR grid** (same budget per method): {1e-4, 3e-4, 1e-3, 3e-3} for AdamW variants; {0.01, 0.03, 0.1, 0.3} for SGD
- **WD grid**: {1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3} (6 values)
- **Method-specific HPs**: CWD beta ∈ {10, 100, 1000}; AdamWN tau ∈ {auto-norm, 0.5*init_norm, 2*init_norm}; SWD sensitivity ∈ {0.1, 1.0, 10.0}
- **Selection**: Best validation accuracy on seed 42; evaluate on seeds 42, 123, 456 with selected HPs

## 3. Metrics Computation

### 3.1 Budget Equivalence Metric (BEM)

```
BEM(method, baseline) = (acc_method - acc_baseline) / acc_baseline
```
where baseline = constant WD at mean(lambda_t) of the method's schedule. BEM ≈ 0 confirms budget equivalence.

### 3.2 Coupling Stability Index (CSI)

```
CSI = w1 * CV(||w||_trajectory) + w2 * log(kappa(H_final)) + w3 * CV(eff_LR_layers)
```
- CV = coefficient of variation
- kappa = spectral condition number (approximated via power iteration)
- eff_LR = eta / (1 + lambda * ||w||) per layer
- Weights: w1=0.4, w2=0.3, w3=0.3

### 3.3 Alignment Informativeness Score (AIS)

```
AIS = corr(cos(w_i, g_i), delta_loss_i)   over training steps i
```
Computed per layer and averaged. AIS > 0.2 = informative; AIS < 0.1 = uninformative.

## 4. Experimental Phases

### Phase 1: Pilot Validation (Target: ~1h wall clock, ~3 GPU-hours)
- **Scope**: CIFAR-10, ResNet-20 only
- **Methods**: AdamW, CWD, SWD, Cosine-WD (4 methods)
- **Purpose**: (a) Validate codebase correctness, (b) Verify CSI/AIS differentiate methods, (c) Quick CWD effective-lambda test
- **Config**: 100 epochs, seed 42 only, fixed HP (LR=1e-3, WD=5e-4)
- **Kill criteria**: AdamW baseline < 91% → codebase bug; CSI identical for all methods → metric is degenerate

### Phase 2: CIFAR Comprehensive (Target: ~3h wall clock, ~18 GPU-hours)
- **Scope**: CIFAR-10 + CIFAR-100, ResNet-20 + VGG-16-BN
- **Methods**: All 7 primary + 4 CWD ablations = 11 configs
- **Seeds**: 42, 123, 456
- **Config**: 200 epochs
- **Key experiments**:
  - Budget equivalence: 5 temporal schedules vs. mean-matched constant WD (H3)
  - CWD falsification battery: CWD vs. matched-lambda vs. random-mask vs. inverted-mask vs. continuous-cosine (H4)
  - WD Stability: Warmup K ∈ {1, 10, 50, 200, 1000} (H2)
  - Metrics validation: CSI/AIS across all methods (H5)
  - Composition test: CWD+Cosine vs. CWD alone vs. Cosine alone; CWD+AdamWN vs. both alone (H7)
  - Spatial modulation: AlphaDecay-style per-layer WD on VGG (H6)

### Phase 3: ImageNet Validation (Target: ~9h wall clock, ~72 GPU-hours)
- **Scope**: ImageNet-1K, ResNet-50
- **Methods**: Top 5 from CIFAR + constant AdamW baseline (6 methods)
- **Seeds**: 42, 123, 456
- **Config**: 90 epochs, multi-GPU DDP on 8x RTX PRO 6000
- **Purpose**: Verify CIFAR findings transfer to scale (H3, H4, H5, H6)

### Phase 4: Architecture Generalization (Target: ~3h wall clock, ~24 GPU-hours)
- **Scope**: CIFAR-100, ViT-Small/16
- **Methods**: AdamW, CWD, AlphaDecay, best temporal schedule (4 methods)
- **Seeds**: 42, 123, 456
- **Config**: 200 epochs
- **Purpose**: Test BN vs. LN architecture differences (H6)

## 5. Statistical Analysis

- **Primary test**: Paired t-test (3 seeds) with Bonferroni correction for multiple comparisons
- **Effect size**: Cohen's d for all pairwise comparisons
- **Bootstrap CI**: 95% confidence intervals via 1000 bootstrap resamples of seed-level results
- **Significance threshold**: p < 0.05 after Bonferroni; practical significance > 0.3% accuracy
- **Rank correlation**: Spearman rho for metric predictiveness (H5)

## 6. Baselines

| # | Method | Category | Key HP |
|---|--------|----------|--------|
| 1 | AdamW + constant WD | Standard baseline | LR, WD grid-searched |
| 2 | SGD + momentum + constant WD | Classical baseline | LR, WD grid-searched |
| 3 | CWD-AdamW | Alignment-aware | beta (hard mask) |
| 4 | SWD/AdamS | Temporal scheduling | sensitivity |
| 5 | Cosine WD schedule | Temporal scheduling | -- |
| 6 | AdamWN | Norm-matched | target tau |
| 7 | AlphaDecay-style (per-layer) | Spatial | alpha per layer |
| 8 | CWD + Cosine | Composition | -- |

### CWD Falsification Controls
| # | Control | Purpose |
|---|---------|---------|
| C1 | Effective-lambda-matched constant WD | Tests if CWD benefit = reduced WD |
| C2 | Random binary mask (matched sparsity) | Tests if any mask works equally |
| C3 | Inverted mask (anti-alignment) | Tests if alignment direction matters |
| C4 | Continuous cosine-similarity WD | Tests soft vs. hard alignment |

## 7. Expected Visualizations

### Main Paper Figures
- **Figure 1**: Unified Phi-framework taxonomy diagram (method × modulation axis)
- **Figure 2**: WD Stability Condition — loss variance vs. warmup duration K (H2)
- **Figure 3**: Budget equivalence panel — 5 temporal schedules vs. mean-matched constant WD across 4 benchmarks (H3)
- **Figure 4**: CWD falsification battery — CWD vs. 4 controls across benchmarks (H4)
- **Figure 5**: CSI/AIS scatter plots with accuracy correlation (H5)
- **Figure 6**: Spatial vs. temporal modulation comparison (H6)
- **Figure 7**: Composition orthogonality test results (H7)

### Diagnostic Panels (6 per method, supplementary)
1. Training loss + test accuracy curves
2. Per-layer weight norm trajectories
3. Per-layer gradient-weight cosine similarity over training
4. Effective learning rate per layer over training
5. CSI and AIS evolution over training
6. Phi modulation heatmap (layers × time steps)

### Tables
- **Table 1**: Main results — all methods × all benchmarks, mean ± std over 3 seeds
- **Table 2**: CWD falsification results — CWD vs. 4 controls with Cohen's d and p-values
- **Table 3**: Metric predictiveness — Spearman rho for CSI/AIS predicting accuracy rankings
- **Table 4**: Composition results — individual vs. composed methods
- **Table 5**: Soft CWD approximation quality — hard CWD vs. soft CWD at various beta

## 8. WD Stability Condition Verification Protocol (H2)

For ResNet-20 on CIFAR-10 with AdamW:
1. Fix LR=1e-3, target WD=5e-4
2. WD warmup from 0 to 5e-4 over K steps, K ∈ {1, 10, 50, 200, 1000}
3. Measure: training loss variance in steps 0-1000, max loss spike magnitude
4. Compute K_critical = 1/(eta * lambda) = 1/(1e-3 * 5e-4) = 2,000,000 (very large)
5. With smaller LR or larger WD: K_critical = 1/(1e-3 * 5e-3) = 200 → K=1,10,50 should violate
6. Therefore also test with WD=5e-3 to find practical violations

## 9. Soft CWD Proximal Verification (H1)

1. Train with hard CWD (binary sign mask) and soft CWD (sigmoid with beta ∈ {10, 50, 100, 500, 1000})
2. CIFAR-10, ResNet-20, seed 42, 100 epochs
3. Measure: accuracy gap between hard and soft CWD; track mask similarity over training
4. Pass criterion: |acc_hard - acc_soft(beta=100)| < 0.05%

## 10. Codebase Fork Strategy

Fork from `why-weight-decay` (D'Angelo et al., NeurIPS 2024) for:
- Training infrastructure (data loading, LR scheduling, logging)
- AdamW and SGD baselines

Custom implementation needed for:
- Unified Phi modulator interface
- CWD, SWD, AdamWN, AlphaDecay optimizers/wrappers
- Diagnostic logging (per-layer norms, alignment, effective LR)
- CSI/AIS/BEM metric computation
- Visualization toolkit (6-panel diagnostic + paper figures)

## 11. Reproducibility Checklist

- [ ] All random seeds explicitly set (torch, numpy, CUDA, Python hash)
- [ ] Exact package versions pinned in requirements.txt
- [ ] Training config saved as JSON alongside each run
- [ ] Checkpoints saved at best validation and final epoch
- [ ] Raw metrics logged to CSV/JSONL for independent analysis
- [ ] Diagnostic snapshots every 100 steps for visualization
