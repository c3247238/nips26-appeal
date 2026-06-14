# Full Experiment Summary: Feature Absorption and Downstream SAE Reliability

## Overview

This experiment tests whether feature absorption in Sparse Autoencoders (SAEs) degrades downstream interpretability tasks (feature steering and sparse probing). Due to gated model access (Gemma-2-2B requires HF authentication), we pivoted to GPT-2 Small with multiple layer configurations to maximize variance and test cross-layer consistency (H3).

## Model and SAE Configuration

- **Model**: GPT-2 Small (85M parameters)
- **SAE Release**: gpt2-small-res-jb (24,576 latents)
- **Layers Tested**: 0, 4, 8, 10 (hook_resid_pre)
- **Features**: 26 first-letter features (A-Z)
- **Samples per feature**: 100
- **Seed**: 42

## Experiments Completed

### 1. Absorption Detection (4 layer configs)

| Layer | Mean Absorption | Max Absorption | Medium (>10%) | Low (<10%) |
|-------|----------------|----------------|---------------|------------|
| 0     | 0.021          | 0.094          | 0             | 26         |
| 4     | 0.039          | 0.160          | 6             | 20         |
| 8     | 0.034          | 0.242          | 4             | 22         |
| 10    | 0.029          | 0.209          | 4             | 22         |

**Key finding**: Layer 4 shows the most absorption variance (6 features with >10% absorption), making it the best candidate for correlation analysis.

### 2. Feature Steering (layers 4, 8)

Steering strength tested: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

- Steering is generally effective (success rates 0.4-1.0 at strength=50)
- No clear pattern linking absorption to steering failure

### 3. Sparse Probing (layers 4, 8)

k-sparse probes tested: [1, 5, 10, 20]

- F1 scores range from 0.18 to 1.0 at k=5
- High variance across features, not clearly linked to absorption

### 4. Correlation Analysis

#### H1: Absorption vs Steering Effectiveness

| Layer | Pearson r | p-value | R^2   | Passes (r<0, p<0.05) |
|-------|-----------|---------|-------|----------------------|
| 4     | +0.008    | 0.970   | 0.000 | No                   |
| 8     | -0.301    | 0.136   | 0.090 | No                   |

#### H2: Absorption vs Probing F1

| Layer | Pearson r | p-value | R^2   | Passes (r<0, p<0.05) |
|-------|-----------|---------|-------|----------------------|
| 4     | -0.003    | 0.987   | 0.000 | No                   |
| 8     | -0.107    | 0.604   | 0.011 | No                   |

#### H3: Cross-Layer Consistency

- CV of H1 slopes: -1.079 (fails CV < 0.5 criterion due to opposite signs)
- CV of H2 slopes: -0.932 (fails CV < 0.5 criterion)
- Slopes have opposite signs across layers, indicating inconsistency

## Hypothesis Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1 (Absorption degrades steering) | **NOT SUPPORTED** | r=-0.30 at layer 8 (trending negative but p=0.14); r=+0.01 at layer 4 |
| H2 (Absorption degrades probing) | **NOT SUPPORTED** | r=-0.11 at layer 8 (p=0.60); r=-0.003 at layer 4 (p=0.99) |
| H3 (Consistent across layers) | **NOT SUPPORTED** | Slopes have opposite signs; no consistent degradation pattern |

## Interpretation

### Why the Null Result?

1. **Low absorption variance**: Most features (18-26/26) show near-zero absorption, creating insufficient variance for correlation
2. **Steering is robust**: Even absorbed features still steer successfully, suggesting SAE steering may be resilient to absorption
3. **Probing measures different thing**: k-sparse probing F1 depends more on feature selectivity than absorption
4. **Metric sensitivity**: The differential correlation absorption metric may not capture the phenomenon strongly in GPT-2 Small

### What This Means

The null result is itself a meaningful finding: **feature absorption, as measured by the Chanin et al. differential correlation method, does not significantly degrade steering effectiveness or sparse probing accuracy in GPT-2 Small SAEs**. This suggests either:

1. Absorption is not a critical failure mode for these specific downstream tasks
2. The absorption metric is not capturing the right phenomenon
3. GPT-2 Small SAEs are too small/simple to exhibit strong absorption effects
4. Steering and probing are robust to the types of absorption present in this model

## Comparison with Pilot

| Metric         | Pilot (layer 8) | Full (layer 8) | Full (layer 4) |
|----------------|-----------------|----------------|----------------|
| H1 r           | -0.153          | -0.301         | +0.008         |
| H1 p           | 0.456           | 0.136          | 0.970          |
| H2 r           | -0.098          | -0.107         | -0.003         |
| H2 p           | 0.632           | 0.604          | 0.987          |

The full experiment on layer 8 shows a stronger negative trend for H1 (r=-0.30) but still not significant. Layer 4 shows essentially zero correlation.

## Limitations

1. **Gemma-2-2B unavailable**: Gated access prevented testing the originally planned primary model
2. **Single SAE family**: Only res-jb SAEs tested; other architectures may show different patterns
3. **First-letter task narrow**: May not generalize to more complex semantic features
4. **Small model**: GPT-2 Small may not exhibit absorption as strongly as larger models
5. **Steering metric**: Relative probability lift may not be the most sensitive measure

## Recommendations for Future Work

1. **Test with authenticated Gemma/Pythia access** to compare across model families
2. **Use semantic hierarchy features** (WordNet) instead of first-letter for richer structure
3. **Try alternative absorption metrics** (e.g., ablation-based from SAEBench)
4. **Test with JumpReLU SAEs** which reportedly show stronger absorption
5. **Increase sample size** per feature for more stable estimates
6. **Test steering with feature-specific prompts** rather than generic contexts

## Files Generated

- `full_absorption_gpt2_l{0,4,8,10}_absorption_rates.json` - Absorption data per layer
- `full_steering_probing_gpt2_l{4,8}_results.json` - Steering and probing results
- `correlation_report_full.json` - Statistical analysis
- `correlation_report_full.md` - Human-readable report
- `summary.md` - This file
