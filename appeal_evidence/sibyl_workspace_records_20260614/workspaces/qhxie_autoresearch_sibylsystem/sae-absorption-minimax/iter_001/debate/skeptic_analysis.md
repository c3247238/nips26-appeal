# Skeptic Analysis: Critical Examination of Findings

## Overview
While the results are interesting, several methodological concerns and interpretive issues must be addressed before publication.

## Critical Concerns

### 1. H3 REVERSED: Statistical vs Practical Significance
- Spearman r = +0.35 is statistically significant (p < 0.001)
- But r² = 0.12, meaning UAS explains only **12% of variance** in steering sensitivity
- Effect size difference: 0.1035 vs 0.0874 = **16% relative difference**
- Is this practically significant for steering applications?

### 2. pilot_h3_null Shows High ≈ Low
- High absorption: 0.7485
- Low absorption: 0.7543
- **Difference: 0.006 (0.8%)**—essentially identical
- This CONTRADICTS the full_h3 spearman correlation
- Why does the grouped comparison differ from the correlation?

### 3. Null Controls Failed
- Pass criteria required: random_mean < 0.05
- Actual random_mean: **0.6207**
- This is **12x higher** than the threshold
- The null control baseline is very high, suggesting:
  - Random directions still cause measurable effects
  - The steering protocol is sensitive to any direction
  - Effect sizes may be inflated

### 4. Limited Model Generalization
- Only GPT-2 Small tested
- Results may not generalize to larger models (GPT-J, Llama, Gemma)
- Layer 8 may not be representative of other layers

### 5. Steering Metric Concerns
- Using max abs logit difference is sensitive to outliers
- A more robust metric (e.g., median, or KL divergence) might yield different conclusions
- No consideration of direction-specific effects

### 6. Confounding Variables
- Feature frequency correlates with both UAS and steering sensitivity
- Activation frequency (act_freq) is included in UAS formula
- High-absorption features may simply be more active, causing larger effects

## Recommendations
1. **Replicate on at least one more model** (Llama or Gemma) before claiming broad applicability
2. **Resolve the pilot_h3_null vs full_h3 contradiction**
3. **Use a stricter null control** (e.g., orthogonal directions)
4. **Report confidence intervals**, not just point estimates

## Conclusion
The findings are suggestive but not conclusive. More rigorous experiments are needed before publication claims.
