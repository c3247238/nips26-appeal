# Critique: Ideation and Proposal

## Overview
The proposal (idea/alternatives.md) presents a well-structured set of hypotheses with pilot evidence and clear falsification criteria. However, several discrepancies between the proposal and the final paper reveal the iterative nature of the research and the risks of post-hoc framing.

## Strengths

1. **Pre-registered falsification criteria**: Each hypothesis has explicit pass/fail criteria stated upfront, enabling clear evaluation of success.

2. **Risk assessment**: The risk table in Section 6 identifies the key failure modes (synthetic artifact, zero variance, real-model validation) before experiments run.

3. **Honest negative result handling**: The proposal notes H3 steering pilot showed ratio=1.62x (above the 1.5x threshold) but the paper shows full experiment ratio=0.97x, demonstrating the pilot was not reliable and the full result was a genuine negative.

4. **Fallback planning**: GPT-2 SAEs are identified as fallback if Gemma Scope loading fails.

## Weaknesses

### 1. H_Mech Original Pass Criteria Were Statistically Implausible
**Severity: Critical**

The original H_Mech pass criteria required B approx D AND C approx A. The pilot showed B=0.490 vs D=0.484 (pass) and C=0.299 vs A=0.299 (pass). However, the full experiment showed B=0.861 vs D=0.436, causing the original criteria to fail 14 of 15 runs. The proposal's pass criteria were essentially lucky in the pilot and wrong in principle -- they assumed the decoder would not contribute at all (C=A) and that encoder alone would match full training (B=D), but the decoder clearly does compensate (D < B).

The proposal should have anticipated this by considering what it would mean if B != D (i.e., decoder disentanglement).

**Fix**: The proposal should have specified: "If B > D significantly, this indicates decoder compensation, which we would interpret as evidence that the decoder partially offsets encoder-driven absorption."

### 2. H3 Steering Hypothesis Was Poorly Motivated
**Severity: Major**

The H3 hypothesis (absorbed features are more sensitive to steering) has weak theoretical grounding. If a parent feature is absorbed -- meaning children substitute for it in activation -- then steering toward the parent's direction should not differentially affect the absorbed feature because the absorbed feature is not active in the first place. The hypothesis seems to assume absorbed features are "suppressed but recoverable," which contradicts the absorption definition.

The paper's negative result is actually the theoretically expected outcome, not a surprising finding.

**Fix**: The proposal should have noted the theoretical prediction that absorption should make features LESS steerable (since they are not active), making the H3 result a confirmation of theory rather than a surprising negative.

### 3. H_Safe Hypothesis Has No Clear Theoretical Prediction
**Severity: Major**

The safety-critical feature hypothesis (safety features are disproportionately absorbed) has no clear theoretical mechanism. Why would safety features be more absorbed than other features? The paper never explains the mechanism -- only that the null result suggests absorption is universal.

This makes H_Safe an undirected exploration rather than a hypothesis test with a principled reason.

**Fix**: Either articulate a specific mechanism (e.g., "safety features are higher-level abstractions with more specific children, leading to more absorption") or frame H_Safe as exploratory analysis rather than hypothesis test.

### 4. The Proposal Uses Gemma Scope; Paper Uses GPT-2
**Severity: Minor**

The proposal specifies Gemma Scope SAEs for H_Safe (layer 12, d_sae=16k), but the paper uses GPT-2 Small SAEs (layer 8, d_sae=24,576). The proposal notes GPT-2 as a fallback, but the fallback became the primary model. This is not necessarily wrong but creates an asymmetry between proposal and paper.

**Fix**: Update the proposal to reflect the actual model used, or explain why the switch was made.

## Summary
The proposal provides a solid pre-registration framework with appropriate risk assessment, but the H_Mech original criteria were poorly specified (did not anticipate decoder compensation), and H3 and H_Safe lack clear theoretical motivation. The negative results are real contributions but the framing as "failed hypotheses" rather than "confirmations of theory" misses an opportunity.
