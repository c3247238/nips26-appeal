# Methodology: Feature Absorption in SAEs -- Encoder-Driven Mechanism with Causal Consequences

## Overview

This study investigates feature absorption in Sparse Autoencoders (SAEs) through three validated hypotheses and one exploratory safety analysis. Our pilot experiments have revealed a major finding: **absorption is driven by encoder alignment with hierarchical features, not decoder geometry**. This overturns the prevailing narrative and frames our experimental design.

## Core Hypotheses

| ID | Hypothesis | Pilot Status | Falsification Criterion |
|----|-----------|--------------|------------------------|
| H1 | Trained SAEs show higher multi-child proportional absorption than random baselines | **PASSED** | p > 0.05 OR delta < 0.15 |
| H_Mech | Absorption is driven by encoder alignment, not decoder geometry | **PASSED** (pilot) | Condition B not approx D, OR C not approx A |
| H3 | Steering absorbed features toward parent directions improves sensitivity | **PASSED** (pilot) | p > 0.01 for improvement |
| H_Safe | Safety-critical features show elevated absorption vs non-safety | **NOT TESTED** (synthetic failed) | Mann-Whitney p > 0.05 |

## Experimental Setup

### Models and SAEs
- **Synthetic SAEs**: d_model=128, d_sae=4096, L0_target=32, trained on hierarchical synthetic data (for H1, H_Mech, H3)
- **Real SAEs**: Gemma Scope SAEs via SAELens (for H_Safe)
  - Release: `gemma-2b-res-jb` or `gemma-scope-2b-pt-res`
  - Layer 12, d_sae=16384 (for safety analysis)

### Datasets
- **Synthetic hierarchy**: Parent-child feature pairs with configurable co-occurrence and cosine similarity
- **Real prompts**: Standard first-letter classification prompts from sae-spelling methodology

### Metrics
- **Multi-child proportional absorption**: Primary metric measuring how much parent feature activation is routed through child latents
- **Steering sensitivity ratio**: Ratio of sensitivity change for absorbed vs non-absorbed features under steering intervention
- **Statistical tests**: t-tests for group comparisons, Mann-Whitney U for safety analysis

## Baselines

| Baseline | Description | Expected Performance |
|----------|-------------|---------------------|
| Random SAE | Untrained random weights | ~0.15 absorption (from pilot) |
| Random encoder + trained decoder | Decoder-only training | ~0.30 absorption (matches random/random) |
| Trained encoder + random decoder | Encoder-only training | ~0.49 absorption (matches full training) |

## Experiment Design

### Experiment 1: H_Mech Factorial Validation (H_Mech)
**Type**: confirmation | **GPU**: 1 | **Estimated**: 45 min

2x2 factorial design with 5 seeds and 3 L0 sparsity levels:
- Condition A: Random encoder + Random decoder
- Condition B: Trained encoder + Random decoder
- Condition C: Random encoder + Trained decoder
- Condition D: Trained encoder + Trained decoder

**Validation criteria**:
- B ≈ D (encoder alignment is sufficient for absorption)
- C ≈ A (decoder geometry contributes nothing)
- Statistical significance: t-test p < 1e-10 for C vs D

**Pilot evidence**: B=0.490, D=0.484, C=0.299, A=0.299. Encoder effect=0.191, decoder effect=0.0.

### Experiment 2: Multi-Seed Stability (H1)
**Type**: confirmation | **GPU**: 1 | **Estimated**: 30 min

Multi-seed replication across seeds {42, 43, 44, 45, 46} with stochastic hierarchy generation.

**Validation criteria**:
- Trained SAE absorption > 0.3 across all seeds
- Random baseline absorption < 0.2 for majority of seeds
- t-test p < 0.001 between trained and random

**Concern from pilot**: Zero variance in trained SAE (all seeds = 0.5). Adding stochastic noise to hierarchy generation to address this.

### Experiment 3: Steering Intervention (H3)
**Type**: confirmation | **GPU**: 1 | **Estimated**: 20 min

Steering absorbed features toward parent directions and measuring sensitivity change.

**Validation criteria**:
- Steering changes activations: ||steered - baseline|| > 0
- Absorbed features show higher sensitivity ratio than non-absorbed
- Ratio > 1.5x (pilot showed 1.62x)

**Pilot evidence**: Steering works; absorbed sensitivity=0.055, non-absorbed=0.034, ratio=1.62x.

### Experiment 4: Safety-Critical Feature Analysis (H_Safe)
**Type**: exploratory | **GPU**: 1 | **Estimated**: 45 min

Testing whether safety-critical features are disproportionately absorbed in real Gemma Scope SAEs.

**Method**:
1. Load Gemma Scope SAE (layer 12, 16k features) via SAELens
2. Select 20 safety-relevant features (from Neuronpedia annotations or heuristic selection)
3. Match with 20 non-safety features by activation frequency and layer
4. Measure absorption via multi-child proportional method
5. Mann-Whitney U test for group difference

**Validation criteria**:
- Mann-Whitney p < 0.05 for safety > non-safety
- Effect size > 0.3

**Risk**: Synthetic pilot failed (p=0.665). Real SAEs required.

### Experiment 5: Held-Out Generalization
**Type**: analysis | **GPU**: 1 | **Estimated**: 20 min

80/20 train/test split on synthetic data to verify absorption generalizes to unseen hierarchical patterns.

## Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| Vary hierarchy strength (cosine similarity) | Does absorption depend on parent-child similarity? | Higher similarity → higher absorption |
| Vary L0 sparsity (20, 32, 50) | Does sparsity level affect encoder-driven absorption? | Higher sparsity → higher absorption |
| Remove stochastic noise | Is zero variance an artifact? | With noise: variance > 0.05 |
| Different layer depths (Gemma) | Does absorption vary by layer? | Higher layers may show different patterns |

## GPU Resource Planning

| Task | GPU Count | Strategy | Estimated Minutes |
|------|-----------|----------|-------------------|
| H_Mech factorial | 1 | Single GPU | 45 |
| Multi-seed H1 | 1 | Single GPU | 30 |
| H3 steering | 1 | Single GPU | 20 |
| H_Safe Gemma | 1 | Single GPU | 45 |
| Held-out generalization | 1 | Single GPU | 20 |

All tasks fit within the 1-hour budget. Total GPU time: ~2.5 hours.

## Expected Visualizations

- **Figure 1**: H_Mech 2x2 factorial bar chart (conditions A-D with error bars)
- **Figure 2**: Multi-seed stability plot (trained vs random across seeds)
- **Figure 3**: Steering sensitivity by alpha value (line plot for absorbed vs non-absorbed)
- **Figure 4**: Safety vs non-safety absorption comparison (box plot)
- **Figure 5**: Summary table of all hypothesis results

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Encoder-driven result is synthetic artifact | Medium | High | Test on real Gemma Scope SAEs; vary hierarchy strength |
| Zero variance persists with stochasticity | Low | Medium | If variance remains near-zero, document as deterministic property |
| H3 steering fails on real models | Medium | High | Document as synthetic-only result; still contributes methodology |
| H_Safe shows no difference on real SAEs | Medium | Medium | Document as negative result; still contributes methodology |
| Gemma Scope loading issues | Low | Low | Fallback to GPT-2 SAEs via SAELens |

## Shared Resources

```json
{"shared_resources": [
  {"type": "dataset", "name": "synthetic_hierarchy", "path": "shared/datasets/synthetic_hierarchy"},
  {"type": "checkpoint", "name": "gemma-scope-2b", "path": "shared/checkpoints/gemma-scope-2b"}
]}
```

## Dependencies

- `sae-lens` (SAE loading and analysis)
- `transformer-lens` (activation extraction)
- `torch` (GPU computation)
- `numpy`, `scipy` (statistics)
- `matplotlib`, `seaborn` (visualization)

## Statistical Testing Framework

- **Primary tests**: Two-sample t-tests for continuous absorption metrics
- **Non-parametric**: Mann-Whitney U for safety analysis (small samples)
- **Effect sizes**: Cohen's d for t-tests, rank-biserial correlation for Mann-Whitney
- **Significance threshold**: alpha = 0.05, two-tailed
- **Multiple comparisons**: Bonferroni correction for family-wise error
