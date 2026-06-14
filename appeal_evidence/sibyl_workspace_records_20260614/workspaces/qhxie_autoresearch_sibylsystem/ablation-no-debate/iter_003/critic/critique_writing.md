# Critique: Writing Quality

## Overview

The paper is generally well-written with clear structure and logical flow. However, there are several critical issues where claims are overclaimed, findings are mischaracterized, or important limitations are inadequately addressed.

---

## Critical Issues

### 1. H_Mech Interpretation is Self-Contradictory

**Location**: Abstract, Section 4.1

**Issue**: The paper claims "Condition B produces absorption rates of 0.076 versus 0.017 for full training (Condition D)" and frames this as "B ≈ D confirms encoder-driven; C ≈ A confirms decoder-irrelevant."

The math does not support "≈":
- |B - D| = 0.059 absolute difference
- B is 4.5× larger than D (0.076 vs 0.017)

**The Claim is Wrong**: If B ≈ D, then encoder alone produces the same absorption as full training, meaning encoder is sufficient and decoder adds nothing. But B is 4.5× LARGER than D, not approximately equal. The correct interpretation is:

1. Encoder alignment DOES drive absorption (B >> A confirms this)
2. BUT decoder training SUPPRESSES absorption in full training (D < B)

The paper's own data suggests the decoder plays a regulatory/suppressive role during joint training that reduces absorption below what encoder-only training produces.

**Impact**: This is the paper's central finding and it is mischaracterized.

---

### 2. H_Safe Claims Are Invalid (Placeholder Data Only)

**Location**: Abstract, Section 4.4, 5.3

**Issue**: The paper states "Preliminary experiments (H_Safe pilot) on Gemma Scope SAEs show null absorption for both safety and non-safety features" and discusses implications of this finding.

This is factually incorrect. The pilot used placeholder feature indices (1024, 2048, 3072, 4096, 5120) not actual validated safety-critical features from Neuronpedia. The "null result" means nothing because the features tested were arbitrary indices with no validation as safety-relevant.

**The Abstract Claims**: "Pilot experiments on Gemma Scope SAEs show null absorption for safety-critical features, though this result requires validation with curated feature sets."

This is backwards. The result doesn't "require validation"—the experiment itself WAS NOT DONE. The features were placeholders.

**Impact**: Section 5.3 "Safety Implications" is built on a foundation of no actual evidence.

---

## Major Issues

### 3. Sensitivity Metric Appears Invalid

**Location**: Section 4.3, H_Pareto Results

**Issue**: L0=16 shows sensitivity_mean=1.525 with std=0.1. The metric is defined as Var(Δlogits | Δa_f) — variance of logit changes given feature activation changes. Variance is always non-negative but not bounded at 1. However, for this to be a meaningful "sensitivity" metric in [0,1] normalized form, values >1 are suspicious.

If sensitivity >1, either:
- The formula is wrong (should be normalized somehow)
- The scale is not normalized to [0,1]
- The implementation has a bug

This undermines the entire Pareto frontier claim since the frontier shape depends on these sensitivity values.

---

### 4. Unfulfilled Statistical Promise

**Location**: Section 3.5 (promised ANOVA), Section 4 (not reported)

**Issue**: The paper promises "Section 3.5 promises a one-way ANOVA across all completed variants" but this test is never reported in the results. The paper focuses on pairwise comparisons, which is actually more appropriate given the small N per condition, but the unfulfilled promise creates a credibility gap.

**Fix**: Either remove the promise from Section 3.5 or actually run and report the ANOVA.

---

### 5. MCC Discussion Inadequate

**Location**: Section 5.5 Limitations

**Issue**: The paper briefly mentions MCC ~0.21 across variants including Random control in the evolution lessons but does not address this in the paper proper. If Hungarian matching on overcomplete dictionaries yields chance-level recovery regardless of training, the meaning of "absorption reduction" is unclear.

The paper does not discuss whether absorption reduction might reflect sparsity-induced suppression rather than genuine hierarchical feature recovery.

---

## Minor Issues

### 6. Cross-Validation Layer Mismatch

GPT-2 Small validation uses layer 8 while Gemma Scope uses layer 12. Cross-model claims should acknowledge this layer-specific nature.

### 7. Metrics Specification Inconsistent

The proposal mentions "multi-child proportional method" but k values, sample sizes, and exact procedures vary and are not systematically reported in one place.

---

## What the Paper Does Well

1. **Clear structure**: Introduction → Background → Method → Results → Discussion flows logically
2. **Honest negative results reporting**: H2 failure (wrong direction correlation) is properly documented
3. **Honest inconclusive result for H_Safe**: The paper does note "placeholder feature selection prevents firm conclusions" in Section 4.4
4. **Good theoretical framing**: Information geometry perspective in Section 5.2 is compelling
5. **Figures referenced well**: Each figure is referenced at the right moment

---

## Recommendations

### Must Fix
1. **Reword H_Mech interpretation**: State that encoder alignment creates absorption, but decoder training in joint training suppresses it—Condition D < Condition B demonstrates decoder's regulatory role
2. **Remove or rewrite H_Safe claims**: Either properly identify real Neuronpedia safety features or state the experiment awaits curated annotations

### Should Fix
3. **Verify sensitivity metric implementation**: Sensitivity > 1 suggests formula bug or missing normalization
4. **Remove ANOVA promise or actually run it**

### Consider Fixing
5. **Add MCC discussion to paper proper** (not just evolution lessons)
6. **Standardize metrics reporting** in unified methods subsection