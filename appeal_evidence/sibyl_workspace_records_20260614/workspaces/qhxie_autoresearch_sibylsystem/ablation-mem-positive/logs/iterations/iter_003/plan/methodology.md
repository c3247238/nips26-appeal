# Methodology: CV-Based Actionability Decomposition

## Overview

This methodology tests whether coefficient of variation (CV) predicts steering effectiveness for absorbed SAE features. The front-runner hypothesis (H1) proposes that absorbed features decompose into "robust" (high-CV, steerable) and "fragile" (low-CV, non-steerable) subpopulations.

## Validation Status

| Experiment | Status | Key Evidence |
|------------|--------|--------------|
| pilot_activation_patching | PASSED | 67.3% mean recovery, all 9/9 words >10% |
| pilot_steering_cv | PASSED | High-CV=0.153 vs Low-CV=0.075 (2x difference) |
| full_cv_analysis | COMPLETED | CV difference confirmed across all layers |
| full_activation_patching | COMPLETED | Validated in experiment_state |

## Hypotheses to Test

### H1: CV Predicts Steering (Primary)
- **Prediction**: High-CV (CV > 1.0) absorbed features show steering effect > 0.10; Low-CV (CV <= 1.0) <= 0.08
- **Falsification**: No significant difference (p > 0.05)
- **Evidence base**: Pilot showed 0.153 vs 0.075 (2x ratio)

### H6: Decoder Orthogonality (Secondary)
- **Prediction**: Features with orthogonal decoders show higher steering effectiveness
- **Falsification**: No correlation between orthogonality and steering

### Cross-Architecture Generalization
- **Prediction**: CV-steering correlation replicates on Gemma-2-2B JumpReLU SAE
- **Falsification**: Gemma-2 shows no CV effect

## Experiment Design

### Setup
- **Model**: GPT-2 Small (86M params)
- **SAE**: gpt2-small-res-jb, layer 6 residual stream (~16k latents)
- **Dataset**: 1000 text samples from validation set
- **Seed**: 42

### Steering Protocol
1. Classify absorbed features (absorption_score > 0.5) into high-CV (CV > 1.0) and low-CV (CV <= 1.0)
2. Select top 30 features from each group by CV score
3. Apply steering at strengths +3, +5, +7
4. Measure logit change at semantically appropriate tokens
5. Statistical test: Welch's t-test (one-sided, α = 0.01)

### Ablation Design
- **Ablation 1 (H1)**: High-CV vs Low-CV steering comparison
- **Ablation 2 (H6)**: Decoder orthogonality as alternative predictor
- **Control**: Fano factor (CV²/mean) to control for magnitude

## Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Steering effect | High-CV > 0.10, Low-CV <= 0.08 | Logit change at target token |
| Effect ratio | High/Low > 1.5 | Ratio of mean effects |
| Statistical significance | p < 0.01 | Welch's t-test with BH correction |
| Decoder orthogonality correlation | r > 0.3 | Cosine similarity vs steering effect |

## Expected Visualizations

1. **Table 1**: Main results - steering effect by CV group and strength
2. **Figure 1**: Bar chart comparing high-CV vs low-CV steering effects
3. **Figure 2**: Scatter plot - CV vs steering effect with regression line
4. **Figure 3**: Decoder orthogonality correlation with steering
5. **Table 2**: Cross-architecture replication results

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV threshold (1.0) not predictive | Medium | Validate on held-out features |
| Steering effect too small | Medium | Compare to non-absorbed baseline |
| Gemma-2 shows no CV effect | Medium | Report as negative result |

## Resource Requirements

- **GPU**: 1 GPU (GTX 1080 or equivalent)
- **Runtime**: ~90 minutes total
- **Models**: GPT-2 Small + Gemma-2-2B
- **No new training**: Training-free analysis via SAELens

## Relationship to Prior Work

- **Basu et al. (2026)**: Actionability paradox (98.2% AUROC → 0% steering) - we show heterogeneity within absorbed features
- **Chanin et al. (2024)**: Absorption detection - we connect to steering outcomes
- **Cui et al. (2026)**: Information-theoretic limits - we work within these constraints