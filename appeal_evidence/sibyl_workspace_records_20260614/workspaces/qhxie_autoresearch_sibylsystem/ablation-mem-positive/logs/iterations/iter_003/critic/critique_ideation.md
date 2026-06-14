# Critique: Ideation

## Overview
The research proposal is well-motivated and addresses a genuine gap in the SAE literature. The actionability paradox from Basu et al. is a meaningful problem, and the CV-based decomposition provides a plausible and practical predictor. However, several aspects of the ideation deserve critical scrutiny.

## Major Issues

### 1. H4 "Variance Paradox" Is Circular
**Problem**: The proposal claims absorbed features have CV ~7.33 vs non-absorbed ~0.01 (733x ratio). But the high/low CV classification threshold of 1.0 was set based on pilot experiments. The "discovery" that absorbed features have high CV is trivially true given that absorbed = high-CV in the classification scheme.

**Impact**: This undermines the framing of H4 as a genuine empirical discovery. The real finding is that CV predicts steering, not that absorbed features have high CV.

**Recommendation**: Reframe H4 as a descriptive observation, not a paradox. The genuine hypothesis is H1 (CV predicts steering).

### 2. Phase Transition Framework Is Detached
Section 3.4 and Section 12 ("Phase Transition as Supporting Context") acknowledge that phase transitions are not the primary novelty. However:
- The proposal mentions chi_ratio = 1.88 (below the "sharp transition" threshold of 3.0)
- H3 (layer depth as temperature) was falsified at lambda = 0.001
- The finite-size scaling exponent nu = 3 is presented with R^2 = 0.951, which is genuinely interesting but not clearly connected to the CV-based actionability decomposition

The phase transition framework appears to be remnants from a prior iteration that have not been cleanly removed.

### 3. H3 and H6 Framing as "Informative Negatives"
Both H3 and H6 are listed as "falsified" with the explanation that they need retesting at different lambda or using different methods. This is appropriate scientific practice, but the proposal does not clearly explain:
- What would constitute falsification for H3 at lambda_c = 5e-5
- Whether H6 (graph topology) has any remaining theoretical motivation after the falsification

### 4. The "Domain Specificity" Hypothesis Is Post-Hoc
The proposal suggests Basu et al.'s actionability paradox may reflect clinical features being predominantly low-CV. This is a plausible explanation but has no direct empirical support:
- No CV data from Basu et al. clinical features is provided
- The hypothesis is generated post-hoc to explain why their paradox doesn't apply in this work's domain

This is reasonable speculation but should be framed as such, not as established fact.

## Minor Issues

### 5. Falsification Criteria Are Vague
Section 5.3 states "Falsification: No significant difference (p > 0.05)." But:
- What is the threshold for "significant"? (BH-corrected? alpha = 0.05 or 0.01?)
- The full experiment uses alpha = 0.01 with BH correction, but pilot used different criteria

### 6. Backup Scenarios Are Over-Complex
The proposal has three backup candidates (projection-based cross-layer, steering effectiveness, cross-architecture phase transition). This complexity suggests the primary proposal may not be well-validated. Consider focusing on a single clear pivot criterion rather than three alternatives.

### 7. Resource Estimate May Be Optimistic
Section 8 estimates ~110 min total, but the actual experiment appears to have taken longer (multiple full_* experiments completed with separate DONE markers). This is common in research but worth acknowledging.

## What Works Well

1. **Clear problem statement**: The actionability paradox is well-motivated and addresses a genuine field concern
2. **Actionable predictor**: CV is computationally cheap and practically useful
3. **Appropriate venue positioning**: Mid-tier venue (AAAI/EMNLP/Workshop) with honest scope
4. **Good falsification awareness**: Explicitly acknowledges what was falsified and why
5. **Practical risk assessment**: Section 10 correctly identifies key risks and mitigations