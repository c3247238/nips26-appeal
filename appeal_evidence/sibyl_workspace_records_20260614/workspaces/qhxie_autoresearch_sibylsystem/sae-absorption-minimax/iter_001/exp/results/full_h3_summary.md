# Full H3 Results: Causal Intervention Reliability Across Absorption Spectrum

## Experiment Summary

**Task**: full_h3 - Steering Sensitivity Across Absorption Spectrum
**Model**: GPT-2 Small, Layer 8
**SAE**: gpt2-small-res-jb (d_sae=24576)
**Date**: 2026-04-26

## Configuration

- **N High-Absorption Features**: 50 (UAS > 1.0)
- **N Low-Absorption Features**: 50 (UAS < 0.3)
- **Alpha Values**: [1, 3, 5, 10, 20]
- **N Test Prompts**: 10
- **Metric**: Max logit change at last position

## Results

| Metric | Value |
|--------|-------|
| **Spearman r (UAS vs Sensitivity)** | +0.3548 |
| **P-value** | 2.92e-04 |
| **H3 Pass (r < -0.3)** | **FALSE** |
| High-Absorption Mean Sensitivity | 0.1035 |
| Low-Absorption Mean Sensitivity | 0.0874 |
| Sensitivity Ratio (Low/High) | 0.84 |

## Key Finding: H3 FAILED

**Contrary to H3's prediction**, this experiment found:

1. **Positive correlation** between UAS and steering sensitivity (r = +0.35)
   - H3 predicted: negative correlation (r < -0.3)
   - Actual: significant positive correlation

2. **High-absorption features show HIGHER steering sensitivity** (0.1035) than low-absorption features (0.0874)
   - H3 predicted: absorbed features would show LOWER steering sensitivity
   - Actual: absorbed features show ~15% HIGHER sensitivity

## Interpretation

This is a **negative result for H3** but supports the **contrarian hypothesis**: absorbed features may actually be MORE causally manipulable than non-absorbed features.

Possible explanations:
1. Absorbed features are more "entangled" with model behavior, making them easier to steer
2. The UAS metric may not correctly capture absorption's effect on causal reliability
3. The steering protocol (adding feature directions) may interact differently with absorbed vs non-absorbed features

## Implications

- **H3 is not supported** by this experiment
- The contrarian view (absorption may be tolerable or even beneficial for steering) has some empirical support
- Future work should investigate WHY absorbed features show higher steering sensitivity
- The UAS metric's relationship to causal reliability should be revisited
