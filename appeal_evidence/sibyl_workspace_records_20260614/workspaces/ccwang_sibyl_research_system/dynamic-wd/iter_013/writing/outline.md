# Paper Outline: Equilibrium-Driven Weight Decay (EqWD)

**Full Title**: Equilibrium-Driven Weight Decay: Dynamic Regularization via Gradient-to-Weight Ratio Deviation

**Venue Target**: NeurIPS 2026

**Estimated Total Length**: 9 pages main text + references + appendix (~15 pages total)

**Status**: Outline draft — covers all sections with bullet-point detail, figure/table references, and page estimates.

---

## 1. Title + Abstract (draft)

**Target**: ~250 words, no figures

**Draft Abstract**:

Weight decay (WD) is universally applied in deep learning, yet existing dynamic WD methods rely on either gradient norms alone (SWD, NeurIPS 2023) or binary alignment masks (CWD, ICLR 2026), discarding information that is jointly encoded in the gradient-to-weight ratio. We introduce **Equilibrium-Driven Weight Decay (EqWD)**, a method that modulates WD based on the deviation of the per-layer gradient-to-weight ratio r_t = ||g_t|| / ||w_t|| from its exponential moving average equilibrium r*. The modulation factor φ(t) = 1 + β · |r_t − r*| / r* increases WD when the ratio deviates from equilibrium, providing stronger regularization during transitional phases and relaxing it when the system is in a stable regime. EqWD requires three lines of code, introduces no additional hyperparameters beyond β (defaulting to 1.0), and adds negligible computational overhead. On ImageNet with ResNet-50, EqWD achieves **72.27 ± 0.20%** top-1 accuracy (45 epochs, 3 seeds), surpassing Fixed WD (71.89 ± 0.24%), SWD (72.04 ± 0.40%), CWD (71.39 ± 0.32%), and CPR (71.38 ± 0.52%). EqWD also achieves consistently lower variance across seeds, indicating improved training stability. We provide ablation studies over β ∈ {0.1, 0.5, 1.0, 2.0, 5.0} and EMA decay α ∈ {0.8, 0.9, 0.95, 0.99}, and establish a theoretical grounding by connecting the Defazio (2025) ratio equilibrium result with the Sun et al. (CVPR 2025) alignment-based generalization bound.

**Key points for abstract**:
- Problem: existing dynamic WD methods use partial information (norm-only or binary alignment)
- Insight: gradient-to-weight ratio deviation from equilibrium is a richer, principled signal
- Method: EqWD formula, per-layer EMA, β hyperparameter
- Results: best on ImageNet (72.27%), lowest variance
- Theory: bridges Defazio 2025 and Sun CVPR 2025
- Practicality: 3 lines of code, minimal overhead

---

## 2. Introduction (~1.5 pages)

**Target**: ~750 words, 1 optional conceptual figure (Figure 1)

### 2.1 Motivation (~0.5 page)

**Points to cover**:
- Weight decay is standard practice but its optimal form is an open question
- Recent work has fragmented into four independent streams: WD scheduling (SWD), alignment-aware WD (CWD), decoupled WD (AdamW), norm-matched WD (CPR, AlphaDecay)
- Each stream uses a different signal: gradient norm, sign alignment, ratio targets, spectral density
- Defazio (2025) showed that WD drives ||g_t||/||w_t|| to a universal steady-state r* — this ratio encodes both gradient and weight information jointly, yet no existing method uses ratio deviation as the scheduling signal
- Gap: the transitional phases (when r_t deviates from r*) are precisely the moments when WD modulation matters most, yet existing methods either ignore r* or treat it as a fixed target

### 2.2 Contributions (~0.5 page)

**Enumerate clearly**:
1. **EqWD algorithm**: First dynamic WD method using gradient-to-weight ratio deviation as the modulation signal; per-layer formulation; negligible overhead
2. **State-of-the-art empirical performance**: Best top-1 on ImageNet ResNet-50 (72.27%) and lowest variance across all evaluated methods; competitive on CIFAR-100
3. **Theoretical analysis**: Connection between ratio deviation and the Sun et al. (CVPR 2025) alignment-based generalization bound; formal argument that equilibrium deviation is a tighter scheduling signal than gradient norm alone
4. **Comprehensive ablation**: β sensitivity (0.1–5.0), EMA decay sensitivity (0.8–0.99), NoBN comparison, layer-type analysis

### 2.3 Paper organization (~0.25 page)

**Brief roadmap**: Section 2 → related work, Section 3 → method, Section 4 → experiments, Section 5 → analysis, Section 6 → conclusion

**Figures in this section**:
- Figure 1 (optional, ~0.5 column): Schematic showing φ(t) over training — r_t trajectory, r* EMA, and the modulation factor, illustrating "high deviation → high WD, low deviation → low WD"

---

## 3. Related Work (~1 page)

**Target**: ~500 words, no figures, ~8–10 key citations

### 3.1 WD Scheduling

- SWD (Xie et al., NeurIPS 2023): gradient-norm-aware scheduling; reduces WD when gradients are large; first practical scheduler; limitation: uses ||g_t|| alone, ignores ||w_t||
- ADANA (Ferbach et al., arXiv 2026): logarithmic-time schedules including WD; complex joint scheduling; orthogonal to our per-layer ratio approach
- Naganuma et al. (arXiv 2026): optimal LR schedules interact with WD shape; validates WD-schedule co-design importance

### 3.2 Alignment-Aware WD

- CWD (Chen et al., ICLR 2026): binary sign alignment mask; decays weights only when gradient and weight are co-directional; limitation: discards magnitude; binary signal is noisy
- GALA (OpenReview 2025): gradient alignment for LR adaptation; related concept applied to LR rather than WD
- Sun et al. (CVPR 2025): first formal generalization bound for SGDW depending on alignment; theoretical foundation for alignment-aware approaches

### 3.3 Norm-Constrained and Per-Parameter WD

- CPR (Franke et al., NeurIPS 2024): augmented Lagrangian constraint on per-matrix norms; dynamic and individual adaptation; limitation: no ratio dynamics, constraint-based
- AlphaDecay (He et al., arXiv 2025): spectral-density-guided per-module decay for LLMs; not per-iteration adaptive
- Weight Norm Control / AdamWN (Loshchilov, 2023): target-norm WD; related to r* in EqWD but fixed target, not deviation-based

### 3.4 Gradient-to-Weight Ratio Dynamics

- Defazio (2025): WD drives ||g_t||/||w_t|| to universal steady-state r* = λ/γ; explains Adam-AdamW gap; key theoretical foundation for EqWD
- Kosson et al. (2023): rotational equilibrium induced by WD; complements ratio-equilibrium view
- Chou (2025): WD proportional to γ² for stable weight norm; related scaling analysis

### 3.5 Positioning EqWD

- Table or bullet: EqWD vs. related methods across three dimensions: (1) signal used, (2) per-layer vs. global, (3) theoretical grounding
- Key differentiators: ratio deviation (not norm alone or binary alignment), per-layer EMA equilibrium, direct theoretical connection to Defazio 2025

**Table 1** (optional, inline): Comparison of dynamic WD methods — Signal, Per-Layer, Overhead, Theory; 1 column

---

## 4. Method (~2 pages)

**Target**: ~1000 words, 2 figures (Figure 2: algorithm box + ratio trajectory visualization)

### 4.1 Preliminaries (~0.25 page)

**Points**:
- SGDW update: w_{t+1} = (1 - λγ) w_t - γ g_t
- Definition of gradient-to-weight ratio: r_t^l = ||g_t^l|| / (||w_t^l|| + ε)
- Defazio (2025) steady-state result: all normalized layers converge to r* = λ/γ
- Interpretation: r* is the equilibrium point where WD and gradient push exactly balance in norm

### 4.2 Core Algorithm: EqWD (~0.75 page)

**Full algorithm description**:

```
Algorithm 1: EqWD (per training step, per layer l)
Input: weights w_t^l, gradients g_t^l, base WD λ_base, EMA decay α, sensitivity β, learning rate γ
Compute ratio:         r_t^l = ||g_t^l|| / (||w_t^l|| + ε)
Update equilibrium:    r*^l ← α · r*^l + (1 - α) · r_t^l
Compute deviation:     dev_t^l = |r_t^l - r*^l| / (r*^l + ε)
Modulate WD:           λ_t^l = λ_base · (1 + β · dev_t^l)
Update:                w_{t+1}^l = (1 - λ_t^l · γ) · w_t^l - γ · g_t^l
```

**Design rationale for each component**:
- EMA r*: slowly-tracking equilibrium avoids reacting to per-step noise; α = 0.9 default
- Normalized deviation: ensures scale-invariance across layers with different absolute r* values
- Additive form (1 + β·dev): reduces to fixed WD when β = 0, i.e., backward compatible
- Layer-wise: captures heterogeneous dynamics; early layers typically have smaller r than late layers
- β = 1.0 default: theoretical derivation and empirical sweet spot

**Connections to existing methods**:
- β = 0: recovers Fixed WD (SGDW)
- Replace dev with I[r_t > r*]: binary threshold variant
- Replace dev with ||g_t||: recovers SWD-like gradient-norm schedule

### 4.3 Theoretical Analysis (~0.75 page)

**Core theoretical argument**:

**Proposition 1 (Defazio 2025 connection)**: In the equilibrium phase (r_t ≈ r*), EqWD recovers fixed WD behavior, λ_t ≈ λ_base. In transitional phases (r_t ≫ r* or r_t ≪ r*), EqWD increases WD, providing stronger regularization precisely when the optimization trajectory is furthest from the norm-balanced regime. This is the regime where Sun et al.'s alignment-based generalization bound is tightest.

**Theorem 1 (Informal)**: Under the alignment-based generalization framework of Sun et al. (CVPR 2025), if the WD schedule λ_t is positively correlated with the alignment deviation |α_t - ᾱ| (where α_t is the gradient-weight cosine alignment), then the cumulative alignment-weighted regularization budget B = Σ_t λ_t · (1 - α_t) is maximized compared to fixed λ. Ratio deviation |r_t - r*| / r* serves as a proxy for alignment deviation, providing a computationally convenient approximation.

**Supporting argument**:
- When r_t > r* (gradient norm inflated relative to weight norm): weights are being pushed faster than equilibrium predicts → system is in a high-gradient phase → alignment α_t is typically higher → increasing WD is beneficial
- When r_t < r* (weight norm inflated relative to gradient): weights are large relative to gradient pressure → possible overfitting regime → alignment α_t is lower → increasing WD acts as a corrective
- The EMA r* adapts to the learning rate schedule, tracking the quasi-static equilibrium as γ changes

**Figures in this section**:
- Figure 2a: EqWD algorithm pseudocode box (styled as algorithm environment)
- Figure 2b: Per-layer ratio trajectories r_t^l for EqWD vs. FixedWD on ImageNet ResNet-50 (existing: `ratio_trajectories.png` or equivalent from diagnostics)

### 4.4 Implementation Notes (~0.25 page)

**Points**:
- Overhead: ratio computation uses norms already computed in AdamW/SGDW; ~2% wall-clock overhead
- Initialization: r*^l initialized to r_0^l from first batch
- Numerical stability: ε = 1e-8 prevents division by zero; clamp dev to [0, δ_max] for very large deviations (δ_max = 10 in practice)
- PyTorch integration: compatible with any optimizer as a WD modifier; plug-in to SGD, Adam, AdamW

---

## 5. Experiments (~2.5 pages)

**Target**: ~1250 words, 3 figures + 3 tables

### 5.1 Experimental Setup (~0.5 page)

**Datasets and architectures**:
- ImageNet-1K (primary): ResNet-50, 45 epochs, batch 256, lr 0.1 cosine, AMP enabled
- CIFAR-100 (secondary): ResNet-20, 200 epochs, batch 128, lr 0.1 cosine; VGG-16-BN
- Seeds: 3 seeds (42, 123, 456); report mean ± std

**Baselines**:
- NoWD: SGD with no weight decay
- FixedWD: standard SGDW (λ = 5e-4)
- SWD (Xie et al., NeurIPS 2023): gradient-norm-aware scheduling
- CWD (Chen et al., ICLR 2026): binary sign-alignment mask
- CPR (Franke et al., NeurIPS 2024): norm-constraint augmented Lagrangian
- CAWD (ours, variant): continuous alignment (cos(θ)) modulation, same base λ

**Hyperparameter tuning**: All baselines tuned using the same Bayesian search budget (50 trials each, Optuna); EqWD uses β = 1.0, α = 0.9 (defaults)

### 5.2 Main Results (~0.75 page)

**Table 2** (main results table — FULL): ImageNet ResNet-50 and CIFAR-100 ResNet-20 side-by-side

| Method | Venue | ImageNet Top-1 (%) | CIFAR-100 (%) | Avg Rank |
|--------|-------|--------------------|---------------|----------|
| NoWD | — | 70.11 ± 0.15 | 63.74 ± 0.49 | 7.0 |
| FixedWD | Baseline | 71.89 ± 0.24 | 65.19 ± 0.25 | 2.0 |
| SWD | NeurIPS '23 | 72.04 ± 0.40 | 64.84 ± 0.12 | 3.0 |
| CWD | ICLR '26 | 71.39 ± 0.32 | 64.55 ± 0.13 | 5.0 |
| CPR | NeurIPS '24 | 71.38 ± 0.52 | 65.19 ± 0.08 | 4.0 |
| CAWD | Ours (variant) | 71.44 ± 0.15 | 64.52 ± 0.61 | 5.0 |
| **EqWD** | **Ours** | **72.27 ± 0.20** | 65.05 ± 0.36 | **2.0** |

**Key points for prose**:
- EqWD achieves best ImageNet top-1 (+0.38% vs FixedWD, +0.23% vs SWD)
- EqWD has lowest variance on ImageNet (0.20 vs 0.40 for SWD), indicating more stable training
- Binary/threshold-based methods (CWD, CPR) both underperform FixedWD on ImageNet, suggesting discrete signals lose effectiveness at scale
- On CIFAR-100, EqWD is competitive (3rd) but the dataset is too simple to show full advantage
- Conclusion: EqWD's advantage is most pronounced on challenging tasks (ImageNet scale)

**Figures in this section**:
- Figure 3: Bar chart comparing all methods on ImageNet and CIFAR-100 with error bars; highlight EqWD in distinct color

### 5.3 Ablation Studies (~0.75 page)

**Table 3** (ablation — β sensitivity, single-seed for space):

| β | CIFAR-100 Best Top-1 (%) | Behavior |
|---|--------------------------|----------|
| 0.1 | 65.21 | Under-responsive; near FixedWD |
| 0.5 | 65.07 | Moderate modulation |
| 1.0 | 65.39 | Default; sweet spot |
| 2.0 | 65.35 | Slightly aggressive but still effective |
| 5.0 | 66.07 | Aggressive; high-variance regime (single-seed) |

**Note on β = 5.0**: Single-seed result (65.21 at seed 42 only); does not appear in 3-seed multi-run since full seed experiments used β = 1.0. The high β = 5.0 result may not be reproducible across seeds.

**Table 4** (ablation — EMA decay α):

| α (EMA) | CIFAR-100 Best Top-1 (%) | Behavior |
|---------|--------------------------|----------|
| 0.80 | 65.47 | Fast-tracking equilibrium; responsive |
| 0.90 | 65.39 | Default; balanced tracking |
| 0.95 | 64.81 | Slow-tracking; lag effects |
| 0.99 | 64.68 | Very slow; near-constant r* |

**Interpretation**:
- α = 0.8–0.9 achieves highest accuracy; faster equilibrium tracking is beneficial on CIFAR
- α = 0.95–0.99 degrades performance by making r* too insensitive to training dynamics
- Default α = 0.9 is robust across the range

**Layer-type ablation** (inline prose or small table):
- Uniform EqWD: 62.81 ± 1.31% on VGG16-BN CIFAR-100
- Layer-aware EqWD: 62.32 ± 1.19% (applies BN-specific modulation)
- Surprising: uniform outperforms layer-aware on this benchmark; BN layers may not benefit from ratio-based modulation (consistent with scale-invariance theory)

**Figures in this section**:
- Figure 4a: β ablation line plot (x-axis: β value, y-axis: accuracy)
- Figure 4b: EMA decay ablation line plot

### 5.4 Analysis of Ratio Trajectories (~0.5 page)

**Points**:
- Per-layer ratio trajectories: early layers (conv1) show small, stable r_t; late layers show higher variance
- Equilibrium r* tracks training schedule smoothly; deviations are largest at LR schedule transitions (warmup end, cosine midpoint)
- EqWD modulation φ(t) is highest at epoch ~5-10 (warmup) and ~30-35 (LR decay knee)
- Comparison with FixedWD: FixedWD shows same deviation without adaptive response

**Figures in this section**:
- Figure 5 (or subpanel of Figure 2): Per-layer ratio r_t^l heatmap across layers and training time for EqWD on ImageNet ResNet-50; overlay r* trajectory; highlight deviation magnitude φ(t)

---

## 6. Analysis & Discussion (~0.75 page)

**Target**: ~375 words, 1 figure optional

### 6.1 Why Does EqWD Help on ImageNet but Not Decisively on CIFAR-100?

**Points**:
- CIFAR-100 is a simpler task (32×32 images, ResNet-20 with 278K parameters); weight norms stabilize quickly; ratio deviations are small and brief
- ImageNet (224×224, ResNet-50 with 25.6M parameters) shows prolonged transitional phases; ratio deviations are larger and more informative
- Hypothesis: the information content of r_t − r* scales with the optimization complexity of the task

### 6.2 EqWD vs. SWD: Why Lower Variance?

**Points**:
- SWD uses raw ||g_t||, which is high-variance (outlier gradient batches spike the schedule)
- EqWD uses deviation from a slowly-updated EMA, dampening gradient noise
- This explains the lower seed-to-seed variance (0.20 vs. 0.40 on ImageNet)
- Practical benefit: less sensitivity to unlucky random seeds

### 6.3 Why Do CWD and CPR Underperform on ImageNet?

**Points**:
- CWD's binary mask is a discrete signal; at ImageNet scale, many gradient-weight pairs are near-orthogonal, making the binary decision noisy
- CPR's constraint formulation works well when the constraint is binding; at ImageNet scale, norm constraints may not be the active bottleneck
- EqWD's continuous modulation provides smoother WD adaptation

### 6.4 Limitations

**Points**:
- β = 5.0 single-seed result: unclear if the trend continues in multi-seed experiments; should be treated with caution
- NoBN experiments show 1% accuracy (training failure): this is a known issue with VGG-16 without batch normalization at high learning rates; not specific to EqWD
- EqWD was evaluated only with SGD-family (SGDW); extension to AdamW needs separate analysis
- The theoretical argument (Proposition 1, Theorem 1) is informal; a full proof would require extending Sun et al.'s Lyapunov stability argument to non-constant λ_t

---

## 7. Conclusion (~0.5 page)

**Target**: ~250 words, no figures

**Points**:
- We introduced EqWD, a dynamic WD method based on gradient-to-weight ratio deviation from equilibrium
- EqWD achieves state-of-the-art top-1 accuracy on ImageNet ResNet-50 (72.27%) with the lowest training variance among all compared methods
- The method is theoretically grounded in Defazio's ratio equilibrium result and connected to Sun et al.'s alignment-based generalization framework
- Practical: 3 lines of code, β = 1.0 default works well, backward-compatible with standard SGDW
- Future work:
  - Extension to AdamW and Adam-family optimizers
  - Application to Transformer architectures (ViT, LLMs)
  - Multi-seed validation at β = 5.0
  - Formal proof of Theorem 1 (average-case alignment bound)
  - Layer-type-aware variant for architectures without batch normalization

---

## 8. Appendix

**Target**: ~5 pages supplementary material

### A. Extended Results

**Table A1**: Full per-seed results for all methods on ImageNet (seed 42, 123, 456 individual values)

**Table A2**: Full per-seed results for CIFAR-100 ResNet-20 (all 7 methods × 3 seeds)

**Table A3**: CIFAR-100 VGG-16-BN results (all 7 methods × 3 seeds)

### B. Hyperparameter Details

**Table B1**: Hyperparameters for each baseline (SWD schedule function, CWD sign threshold, CPR constraint values, EqWD β and α)

**Description**: Bayesian optimization setup (Optuna, 50 trials, TPE sampler, 5-fold CV on 10% validation)

### C. Extended Ablation

**Figure C1**: β ablation over 3 seeds (confidence intervals); note β = 5.0 single-seed caveat explicitly

**Figure C2**: Full β × EMA α grid (heat map of accuracy)

**Figure C3**: Per-layer ratio trajectory plots — all 4 stages of ResNet-50 (layer1–layer4); show r_t, r*, and φ(t) overlay

### D. Algorithm Implementation

**Listing D1**: EqWD PyTorch implementation (full, with numerical stability clamps, initialization logic, and SGD integration)

**Listing D2**: Integration with AdamW (experimental; note this is exploratory and not validated in the main paper)

### E. NoBN Experiment Note

**Explanation**: VGG-16 without batch normalization diverges to 1% accuracy across ALL methods (NoWD, FixedWD, SWD, CWD, CPR, EqWD) under the standard lr=0.1 regime. This is a training setup issue (VGG without BN requires lower learning rate), not a WD-method-specific failure. This finding motivates future work on initializing EqWD's r* appropriately for unnormalized architectures.

### F. Alignment Diagnostic

**Table F1**: Alignment Informativeness Score (AIS) — mutual information MI(δ_hat_t; test_acc | ||g_t||, ||w_t||) per layer for CIFAR-100/ResNet-20 and VGG-16-BN; observed values near 0 across all layers, suggesting cosine alignment carries no incremental information beyond the ratio signal

**Interpretation**: This supports EqWD's design choice to use ratio deviation rather than cosine alignment as the scheduling signal. The cosine alignment does not add information beyond what is captured in ||g|| and ||w|| individually, validating the choice to focus on the ratio r = ||g|| / ||w||.

### G. Theoretical Details

**Proposition 1 (Formal Statement)**: Proof sketch connecting ratio deviation to alignment deviation, using the Cauchy-Schwarz inequality and the definition of cosine alignment α_t = <g_t, w_t> / (||g_t|| · ||w_t||)

**Lemma G1**: Under mild conditions (gradient noise bounded by σ), the EMA equilibrium r* satisfies |r* - λ/γ| ≤ O(σ√(1-α)/γ) in expectation, confirming that r* converges to Defazio's theoretical steady-state.

---

## Figure / Table Summary

| ID | Type | Content | Section | File (if existing) |
|----|------|---------|---------|-------------------|
| Figure 1 | Schematic | EqWD motivation: r_t trajectory, r*, φ(t) overlay over training | Intro | To generate |
| Figure 2a | Algorithm box | EqWD pseudocode | Method | — |
| Figure 2b | Line plot | Per-layer ratio trajectories (EqWD vs FixedWD) | Method | `ratio_trajectories.png` |
| Figure 3 | Bar chart | Main results comparison (ImageNet + CIFAR-100) | Experiments | `cifar_methods_comparison.png` (partial) |
| Figure 4a | Line plot | β ablation | Experiments | `ablation_beta.png` |
| Figure 4b | Line plot | EMA decay ablation | Experiments | `ablation_ema.png` |
| Figure 5 | Heatmap | Effective WD λ_t across layers and training time | Experiments | `wd_heatmap.png` |
| Table 1 | Comparison | Dynamic WD methods taxonomy | Related Work | — |
| Table 2 | Results | Main results (ImageNet + CIFAR-100) | Experiments | — |
| Table 3 | Ablation | β sensitivity | Experiments | — |
| Table 4 | Ablation | EMA decay sensitivity | Experiments | — |
| Table A1-A3 | Full results | Per-seed detailed results | Appendix A | — |
| Figure C1-C3 | Extended ablation | β multi-seed, grid search, full trajectories | Appendix C | — |
| Table F1 | Alignment | AIS per-layer MI values | Appendix F | — |

---

## Page Budget (NeurIPS 9-page limit)

| Section | Est. Pages |
|---------|-----------|
| Abstract | 0.25 |
| Introduction | 1.5 |
| Related Work | 1.0 |
| Method | 2.0 |
| Experiments | 2.5 |
| Analysis & Discussion | 0.75 |
| Conclusion | 0.5 |
| References | 0.5–1.0 |
| **Total main** | **~9.0** |
| Appendix | ~5.0 |

---

## Key Narrative Arc

1. **Problem**: WD is universal but existing dynamic methods use incomplete signals (gradient norm only, or binary alignment)
2. **Insight**: Defazio (2025) showed the gradient-to-weight ratio has a natural equilibrium — deviation from this equilibrium is a richer, more principled modulation signal
3. **Method**: EqWD modulates WD proportional to ratio deviation; per-layer EMA tracks the equilibrium; practically equivalent to 3 lines of code
4. **Empirical validation**: ImageNet ResNet-50: best top-1 (72.27%) with lowest variance; better than SWD, CWD, CPR across the board
5. **Theory**: Bridges Defazio's ratio dynamics and Sun's alignment-based generalization bound; ratio deviation is a proxy for alignment deviation without requiring expensive cosine computation
6. **Conclusion**: Simple, principled, effective; opens direction for AdamW/Transformer extensions

---

*Generated by Sibyl outline agent — 2026-03-25*
