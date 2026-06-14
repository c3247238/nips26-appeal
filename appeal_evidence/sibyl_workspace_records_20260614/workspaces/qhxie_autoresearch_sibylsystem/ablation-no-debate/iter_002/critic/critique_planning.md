# Critique: Planning and Methodology

## Overview
The planning documents (plan/methodology.md, idea/alternatives.md) present a structured experimental program with appropriate risk assessment. However, the planning underestimates several risks and makes overly optimistic assumptions about what the pilot results predict.

## Strengths

1. **Comprehensive experimental program**: The 5-hypothesis plan (H_Mech, H1, H3, H_Safe, held-out) plus 3 ablations is well-structured and covers the key questions.

2. **Resource planning**: The GPU resource table in plan/methodology.md (Section 5) correctly estimates experiment durations and the total 2.5-hour budget is within the 1-hour guideline per experiment (split across multiple runs).

3. **Risk mitigation**: The fallback from Gemma Scope to GPT-2 SAEs was pre-planned and executed when needed.

4. **Ablation strategy**: The two-pronged ablation (hierarchy strength + L0 sparsity) is the right approach to understand the boundary conditions of the encoder effect.

## Weaknesses

### 1. Pilot Was Not Predictive of Full Results for H_Mech
**Severity: Critical**

The pilot showed B=0.490 vs D=0.484 (passing the B approx D criterion), but the full experiment showed B=0.861 vs D=0.436. The pilot was not predictive of the full result for the decoder compensation effect. The planning did not anticipate that the decoder compensation (D < B) would emerge at full scale.

This is the central planning failure: the pilot gave false confidence in the original H_Mech pass criteria.

**Fix**: The planning should have specified what would happen if B >> D (decoder compensation observed). The revised criteria (encoder effect > 0.5, decoder effect < 0.1) should have been the original criteria.

### 2. H3 Steering Hypothesis Had No Clear Mechanism
**Severity: Major**

The planning documents claim absorbed features would be more steerable than non-absorbed features, but the mechanism is never clearly articulated. If absorption means children substitute for the parent (and the parent is inactive), then steering the parent direction should not differentially affect the absorbed feature -- it should affect it less since it is not active.

The planning should have articulated the theoretical link between absorption state and steering responsiveness, or should have identified this as an exploratory question rather than a hypothesis.

**Fix**: For H3, articulate the mechanism: "If absorption reflects encoder geometry that suppresses parent features, then absorbed features should respond to parent-direction steering." Or frame it as exploratory: "We test whether absorption state predicts steering responsiveness."

### 3. Variance Concerns Were Acknowledged But Not Addressed in Planning
**Severity: Major**

The planning acknowledges in H1 that "zero variance in trained SAE (all seeds = 0.5)" was a concern. The proposed mitigation was "adding stochastic noise to hierarchy generation." However, the planning did not anticipate that the random baselines would also show zero variance (or near-zero), which actually makes the variance problem worse (not better) because effect sizes become inflated.

**Fix**: The planning should have specified that random SAE baselines would also be monitored for variance, and effect sizes would be computed with appropriate variance estimates rather than assuming both distributions contribute variance.

### 4. H_Safe Hypothesis Had No Mechanism
**Severity: Major**

The planning does not articulate why safety-critical features would be disproportionately absorbed. There is no theoretical reason to expect safety features to be more absorbed than other features -- the paper's null result makes sense in retrospect, but the planning did not have a principled reason to generate this hypothesis.

**Fix**: Either articulate the mechanism ("safety features are higher-level abstractions with more subordinate concepts, leading to more absorption") or frame H_Safe as exploratory: "We test whether absorption is category-specific or universal."

### 5. Held-Out Generalization Plan Was Underpowered
**Severity: Major**

The held-out generalization plan (80/20 split, n=1 held-out hierarchy per seed) yields only 5 data points. The planning did not anticipate that n=1 would produce a mathematically fragile correlation estimate (r=0.998 with 3 df).

**Fix**: Plan for n=3 held-out hierarchies per seed (60% test split) to get 15 data points for the correlation.

### 6. The Plan Did Not Anticipate the Decoder Disentanglement Effect
**Severity: Major**

The planning focused on whether encoder or decoder dominates absorption, but did not consider the possibility that a trained decoder would *reduce* absorption compared to encoder-only (B > D). The decoder compensation observation (D < B) is one of the paper's interesting secondary findings, but it was not anticipated in planning.

**Fix**: The planning should have included a condition for decoder compensation: "If D < B significantly, this indicates the trained decoder partially offsets encoder-driven absorption, which would be a secondary finding of interest."

## Summary
The planning documents provide a solid structure but underestimate the pilot's unreliability (especially for H_Mech), fail to articulate mechanisms for H3 and H_Safe, and underpower the held-out generalization test. The core experimental program is sound, but the pass/fail criteria and secondary finding identification need more pre-registration.
