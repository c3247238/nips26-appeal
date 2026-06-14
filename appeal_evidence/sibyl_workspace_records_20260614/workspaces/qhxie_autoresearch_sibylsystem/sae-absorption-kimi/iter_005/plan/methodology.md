# Methodology: Component-Isolated Study of SAE Absorption Reduction

## Overview

This study trains 6 SAE variants on SynthSAEBench-16k ground-truth synthetic data, varying one architectural component at a time, to determine which component(s) causally drive absorption reduction. All absorption metrics are computed from known ground-truth parent-child relationships---no probes, no AUROCs, no ceiling effects.

## Experimental Design

### Dataset: SynthSAEBench-16k

- 16,384 ground-truth features (10,884 hierarchical)
- 128 root trees, depth 3, branching factor 4
- Synthetic activations generated from known feature structure
- Built into SAELens (Chanin & Garriga-Alonso, 2026)

### SAE Variants (6 conditions)

| Variant | Architecture | Key Difference | What It Tests |
|---------|-------------|----------------|---------------|
| Baseline | Standard ReLU + L1 sparsity | None | Baseline absorption rate |
| +TopK | TopK activation (k=50) | Replace L1 with explicit k-sparsity | Effect of k-sparsity |
| +MultiScale | Nested dictionaries (2 levels) | Hierarchical multi-scale decomposition | Effect of multi-scale structure |
| +Orthogonality | Decoder orthogonality penalty | Chunk-wise orthogonality constraint on W_dec | Effect of decoder incoherence |
| +Gating | Gated SAE architecture | Decoupled detection/magnitude paths | Effect of gating mechanism |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | All components combined | Replicates prior work; tests interactions |

### Random Control

- Random decoder initialization, no training
- Expected: MCC < 0.1, absorption > 0.8
- Validates that metrics discriminate structure from randomness

## Metrics (Ground-Truth, No Probes)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Absorption rate** | Fraction of parent features subsumed by child features (using known ground-truth parent-child relationships) | Lower is better |
| **Feature recovery MCC** | Matthews correlation between learned and ground-truth feature assignments | Higher is better |
| **Reconstruction MSE** | Mean squared error between input and reconstructed activations | Lower is better |
| **L0 sparsity** | Average number of active features per token | Reported for context |
| **Feature hedging score** | Fraction of latents that incorrectly mix correlated features (Chanin et al. 2025 protocol) | Lower is better |

## Statistical Analysis

1. **One-way ANOVA** across 6 variants (5 replicates each)
2. **Pre-registered primary comparisons**: MultiScale vs. Baseline, Full Matryoshka vs. Baseline
3. **Post-hoc Tukey HSD** for exploratory pairwise comparisons
4. **Effect sizes** (Cohen's d) for each component vs. baseline
5. **Holm-Bonferroni correction** across all pairwise comparisons
6. **Mixed-effects model**: variant (fixed effect) + replicate (random effect)

## Baselines and Controls

1. **Standard ReLU SAE** --- Expected: highest absorption, lowest MCC
2. **Full Matryoshka SAE** --- Expected: lowest absorption, highest MCC (replicates Bussmann et al. 2025)
3. **Random-feature control** --- Expected: near-zero MCC, high absorption (validates metrics)
4. **L0-matched comparison** --- Report absorption per unit L0 to control for sparsity differences
5. **Reconstruction-absorption Pareto** --- Multi-objective plot showing trade-offs

## Expected Visualizations

- **Figure 1 (Architecture diagram)**: Pipeline showing the 6 SAE variants and ground-truth measurement
- **Table 1 (Main results)**: 6 variants x 5 metrics (absorption, MCC, MSE, L0, hedging)
- **Figure 2 (Ablation bar chart)**: Absorption rate per variant with error bars (5 replicates)
- **Figure 3 (Pareto frontier)**: Absorption vs. reconstruction MSE, marking Pareto-optimal points
- **Figure 4 (Effect sizes)**: Cohen's d for each component vs. baseline with 95% CIs
- **Figure 5 (Trade-off scatter)**: Absorption vs. hedging across variants

## SAELens Configuration Notes

Based on SAELens skill guidance:

- Use `LanguageModelSAERunnerConfig` with `architecture` parameter set per variant
- For TopK: `architecture="topk"`, `activation_fn="topk"`, `activation_fn_kwargs={"k": 50}`
- For Gated: `architecture="gated"`
- For Standard: `architecture="standard"`, `activation_fn="relu"`
- MultiScale and Orthogonality require custom config or post-hoc penalty
- Set `l1_coefficient=8e-5` with `l1_warm_up_steps=1000` to prevent dead features
- Monitor `L0`, `CE Loss Score`, and `Dead Features` during training
- Use `use_ghost_grads=True` to revive dead features

## GPU Resource Plan

- Each training task: 1 GPU (synthetic data, small model)
- All 6 variants can run in parallel across 6 GPUs
- Estimated 8-12 minutes per variant for full 16k features
- Analysis tasks: CPU only

## Risk Mitigation

| Threat | Mitigation |
|--------|-----------|
| Training instability | 5 replicates; report variance; ghost grads enabled |
| Component interactions | Full Matryoshka variant tests combined effect |
| No significant effects | Report null result as valuable finding |
| Absorption-hedging trade-off | Measure both metrics on all variants |
| Synthetic-to-real gap | Acknowledge in Discussion; H4 tests transfer |
