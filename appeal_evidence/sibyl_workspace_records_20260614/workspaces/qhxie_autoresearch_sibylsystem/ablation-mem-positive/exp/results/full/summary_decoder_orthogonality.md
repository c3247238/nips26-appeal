# Decoder Orthogonality Analysis - Full Results

## Experiment: full_decoder_orthogonality

### Objective
Test whether decoder weight orthogonality (mean cosine similarity with other decoder weights) predicts steering effectiveness for absorbed SAE features.

### Hypothesis
H6: Features with orthogonal decoders show higher steering effectiveness.
- **Falsification**: No correlation between orthogonality and steering (r < 0.3)

### Configuration
- Model: GPT-2 Small (gpt2-small)
- SAE: gpt2-small-res-jb, layer 6 (~16k latents)
- Features tested: 60 absorbed features (30 high orthogonality, 30 low orthogonality)
- Steering strength: +5
- Prompts: ["The movie was very", "The food was extremely", "The weather today is"]

### Results

#### Group Comparison (High vs Low Orthogonality)
| Group | Mean Effect | Std | N samples |
|-------|-------------|-----|-----------|
| High Orthogonality | 0.131 | 0.090 | 90 |
| Low Orthogonality | 0.107 | 0.086 | 90 |

**Welch's t-test**: t = 1.77, p = 0.079 (not significant at α = 0.05)

#### Correlation Analysis
| Metric | Value | p-value |
|--------|-------|---------|
| Pearson r | -0.136 | 0.301 |
| Spearman rho | -0.204 | 0.117 |

### Pass Criterion
**Required**: Correlation r > 0.3 between orthogonality and steering
**Result**: NOT MET (|r| = 0.136 < 0.3 threshold)

### Key Finding
**NOT_SUPPORTED**: No significant correlation between decoder orthogonality and steering effectiveness.

The negative Pearson r (-0.136) suggests a weak *inverse* relationship - features with MORE orthogonal decoders actually showed slightly LOWER steering effects, but this trend is not statistically significant (p = 0.30).

### Interpretation
Decoder orthogonality does NOT predict which absorbed features are steerable. This suggests that the CV-actionability relationship discovered in prior experiments is not explained by decoder weight geometry.

### Files
- Full results: `exp/results/full/full_decoder_orthogonality.json`
- GPU progress: `exp/results/gpu_progress.json`