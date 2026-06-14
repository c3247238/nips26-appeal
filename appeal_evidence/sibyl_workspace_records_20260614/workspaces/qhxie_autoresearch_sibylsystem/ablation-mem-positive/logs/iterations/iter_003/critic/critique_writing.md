# Critique: Paper Writing

## Overview
The paper is generally well-written with clear structure and good use of visualizations. However, there are several areas where claims are overreaching or internal consistency breaks down.

## Major Issues

### 1. Abstract Overclaims (Section 6.1 Summary)
The abstract and conclusion state: "absorbed high-CV features achieve steering effectiveness comparable to non-absorbed features."

**Problem**: This comparison is flawed because:
- Non-absorbed baseline used different prompts (3 vs 5) and different steering strengths (only +5)
- At the main experiment's +5 strength, high-CV absorbed features show mean effect of 0.5222, NOT 0.097
- The 0.097 figure in Section 4.5 appears to be from a different (pilot) comparison

**Impact**: The paper's most impactful claim - that absorption per se does not destroy steering potential - lacks proper experimental support.

### 2. Cross-Architecture Claims Are Aspirational
The abstract lists "cross-model replication" in contributions. Section 4.6 explicitly states detailed Gemma-2-2B analysis "remains future work." Yet the paper continues to claim cross-architecture generalization in multiple places.

**Recommendation**: Remove all cross-architecture generalization claims from abstract and conclusions until actual results are integrated.

### 3. Inconsistent Effect Size Reporting
The paper reports:
- Pilot (Section 4.2): 0.153 vs 0.075 (2.03x ratio)
- Full experiment (Section 4.3): 0.522 vs 0.355 (1.47x ratio)
- Non-absorbed baseline (Section 4.5): 0.097 vs 0.102 (comparable)

These three comparisons use different:
- Number of prompts (5 vs pilot's 5 vs baseline's 3)
- Steering strengths (only +5 reported for baseline)
- Feature selection criteria

This makes it impossible to directly compare across sections.

### 4. The "Variance Paradox" Is Misleading
Section 3.1 presents the "variance paradox" as a key discovery: absorbed features have CV ~7.33 vs non-absorbed ~0.01 (733x ratio).

**Problem**: The CV-based classification (high-CV > 1.0, low-CV <= 1.0) is derived from the same distribution. Finding that absorbed features have high CV is circular given the classification scheme.

The actual discovery is that CV predicts steering effectiveness. The "paradox" framing overstates what was found.

## Minor Issues

### 5. Section 5.1 Internal Contradiction
"Why do JumpReLU and ReLU SAEs produce projection absorption rates that differ by only 7.7 percentage points?" appears in Section 5.1 but seems to reference a different paper or analysis not presented in this work. This sentence is confusing and appears to be cut-and-paste from another context.

### 6. Mechanism Section Is Conjectural
Section 3.3 proposes "context-sensitive routing" vs "bypass routing" as the mechanism explaining why CV predicts steering. This is reasonable hypothesis but lacks direct experimental support. Activation patching confirms causal structure exists, not which routing regime operates.

### 7. H6 Falsification Underdiscussed
H6 (decoder orthogonality) was falsified but the paper doesn't explain:
- Why orthogonality was hypothesized as mechanism
- What the falsification implies for the bypass/mediated routing framework
- Whether this changes the theoretical interpretation

## Recommendations

1. **Qualify non-absorbed comparison**: "Preliminary comparison under pilot conditions suggests..." or run proper comparison with identical conditions.

2. **Remove cross-architecture claims**: Until Gemma-2-2B results are integrated, remove all generalization claims.

3. **Reframe "Variance Paradox"**: Simply report "absorbed features exhibit higher CV than non-absorbed" without paradox framing.

4. **Add confidence intervals**: Report CIs alongside p-values for more complete statistical picture.

5. **Standardize comparisons**: Use identical experimental conditions across all comparisons (same prompts, same strengths, same n).

## What Works Well

1. Clear logical structure from motivation through experiments to conclusions
2. Appropriate acknowledgment of limitations in Section 5.6
3. Transparent reporting of H6 falsification
4. Good use of tables for main results
5. Appropriate statistical correction (Benjamini-Hochberg)