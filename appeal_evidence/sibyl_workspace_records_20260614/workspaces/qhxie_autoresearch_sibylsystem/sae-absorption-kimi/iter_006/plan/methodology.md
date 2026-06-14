# Methodology: Absorption is a Sparsity Phenomenon

## Overview

This study performs the first component-isolated causal analysis of SAE architectural innovations for absorption reduction, using ground-truth synthetic hierarchies from SynthSAEBench. We train SAE variants varying one component at a time and measure ground-truth absorption rate, reconstruction MSE, and L0 sparsity.

## Experimental Design

### Dataset: SynthSAEBench-16k

We use the synthetic benchmark with 16,384 ground-truth features (10,884 hierarchical) from Chanin & Garriga-Alonso (2026). The hierarchy consists of 128 root trees, depth 3, branching factor 4. Absorption is measured directly using known parent-child relationships---no probes, no ambiguity, no ceiling effects.

**Key advantage over real-LLM evaluation**: Ground-truth features eliminate all construct-validity problems identified in our prior iterations (co-occurrence confound, ceiling effect, model dependence).

### SAE Variants (Component-Isolated)

| Variant | Components | What It Tests |
|---------|-----------|---------------|
| Baseline | Standard ReLU + L1 sparsity | Baseline absorption rate |
| +TopK | Replace L1 with TopK sparsity (k=50) | Effect of explicit k-sparsity |
| +MultiScale | Nested dictionaries (2 levels) | Effect of hierarchical decomposition |
| +Orthogonality | Chunk-wise decoder orthogonality penalty | Effect of decoder incoherence |
| +Gating | Decoupled detection/magnitude paths | Effect of gating mechanism |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | Combined effect (replicates prior) |

Each variant is trained with 5 replicates (seeds 42, 123, 456, 789, 1011) to estimate variance.

### Critical Control: L0-Matched Comparison

To disentangle sparsity from architecture, we train L1 SAEs with tuned lambda to achieve L0 = 50 (matching TopK) and L0 = 550 (matching Orthogonality). If L0-matched Baseline achieves comparable absorption to TopK/Orthogonality, absorption is a sparsity phenomenon, not an architectural one.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_in | 256 | Hidden dimension of synthetic model |
| d_sae | 2048 | 8x expansion factor (overcomplete) |
| Training samples | 2,000,000 | Sufficient for convergence on synthetic data |
| Batch size | 1024 | Fits comfortably on single GPU |
| Learning rate | 1e-3 | Standard for SAE training |
| L1 coefficient | 5e-3 | Pilot-validated |
| TopK k | 50 | Matches pilot configuration |
| Orthogonality lambda | 1e-3 | Standard orthogonality penalty strength |
| Seeds | 42, 123, 456, 789, 1011 | 5 replicates for variance estimation |

### Metrics

**Primary metric**: Absorption rate --- fraction of parent features subsumed by child features (using known ground-truth parent-child relationships).

**Secondary metrics**:
- Reconstruction MSE
- L0 sparsity (average active features per token)
- Dead latent rate (fraction of never-active latents)
- Feature recovery MCC (Hungarian matching)
- Hedging score (fraction of latents assigned to parent features)

## Baselines

1. **Standard ReLU SAE** --- Expected: highest absorption, highest L0
2. **Full Matryoshka SAE** --- Expected: low absorption, but effect may be mediated by TopK
3. **Random-feature control** --- Expected: high absorption, validates metric discrimination
4. **L0-matched L1 SAE** --- Critical control: tests whether sparsity alone explains TopK's effect

## Statistical Analysis

- One-way ANOVA across variants (5 replicates each)
- Pre-registered primary comparisons: TopK vs. Baseline, Orthogonality vs. Baseline
- Post-hoc Tukey HSD for exploratory pairwise comparisons
- Effect sizes (Cohen's d) for each component vs. baseline
- Holm-Bonferroni correction across comparisons
- Mixed-effects model: variant (fixed) + replicate (random)

## Evaluation Benchmarks

This is a synthetic benchmark study. No external evaluation benchmarks are used. The ground-truth hierarchy provides the evaluation protocol.

## Expected Visualizations

- **Figure 1 (Architecture diagram)**: SAE component pipeline showing how each variant modifies the baseline
- **Table 1 (Main results)**: Variant x metric comparison table (absorption rate, MSE, L0, dead latents)
- **Figure 2 (Ablation study)**: Bar chart showing absorption rate per component with Cohen's d vs. baseline
- **Figure 3 (L0-absorption correlation)**: Scatter plot of L0 vs. absorption rate across all variants with regression line
- **Figure 4 (Dose-response)**: Line plot of absorption rate vs. k for TopK sweep
- **Figure 5 (Reconstruction-absorption trade-off)**: Scatter plot showing Pareto frontier

## GPU Resource Plan

| Task | GPUs | Strategy | Est. Time |
|------|------|----------|-----------|
| Full experiment (6 variants x 5 replicates) | 1 per variant | Single GPU | ~60 min |
| L0-matched ablation | 1 | Single GPU | ~30 min |
| TopK k-sweep | 1 | Single GPU | ~30 min |
| Analysis | CPU only | N/A | ~5 min |

All tasks use single-GPU execution. The synthetic data and small model size (256-dim hidden, 2048-dim SAE) make multi-GPU unnecessary.

## Risk Mitigation

| Threat | Severity | Mitigation |
|--------|----------|------------|
| L0-matched ablation shows sparsity is sole driver | Medium | Reframe paper accordingly; still valuable finding |
| MultiScale full data overturns TopK dominance | Low | Unlikely given pilot consistency; would strengthen paper |
| Synthetic data doesn't match LLM feature structure | High | Phase 2 real-LLM validation; acknowledge limitation |
| Dead latent crisis undermines TopK recommendation | Medium | Report transparently; test absorption on active latents only |

## Reproducibility

- All code uses fixed seeds (42, 123, 456, 789, 1011)
- SAELens version pinned
- PyTorch deterministic mode enabled where possible
- All results saved as JSON with full configuration metadata
