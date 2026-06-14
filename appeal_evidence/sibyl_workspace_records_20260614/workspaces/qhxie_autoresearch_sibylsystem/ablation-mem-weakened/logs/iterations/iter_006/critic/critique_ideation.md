# Critique: Ideation and Research Direction

## Overall Assessment

The research direction is reasonable but the framing is problematic. The project started as an investigation into whether feature absorption degrades downstream tasks, found predominantly null results, and pivoted to framing null results as contributions and the LCA connection as a mechanism. This pivot obscures rather than illuminates the findings.

## Issues with Research Framing

### 1. Confusing Research Questions
The paper has evolved through multiple research questions:
- Original: Does absorption degrade steering/probing?
- Intermediate: Do decoder correlations predict absorption?
- Current: Is absorption explained by LCA competitive suppression?

These questions are related but distinct. The paper does not clearly state which question it is answering and why it changed.

### 2. The "Contribution" Framing Problem
The paper presents null results as contributions:
- "Honest null-result reporting" - but null results are reported in many papers; this is not special
- "Metric validation insight" - showing a metric may be invalid is not validation
- "Rate-distortion optimal compression" - this is a post-hoc theoretical framing, not a finding

**The core problem**: If the experiments show no significant effects after MCP, the paper is a null-result paper, not a finding paper. This is fine - null results are valuable - but the framing should match.

### 3. Multiple Comparisons Problem (Critical)
The paper performs 12 tests (4 hypotheses x 2 layers x 2 metrics) but presents uncorrected p-values as evidence throughout. This is a fundamental statistical flaw:
- H1b at L8: p=0.028 uncorrected → Bonferroni p=0.334, BH-FDR q=0.107
- All other tests: p>>0.05

The abstract, introduction, and conclusion all present H1b as evidence of a real effect, which is incorrect after proper correction.

### 4. The H6 Falsification Problem
H6 (decoder correlation graph predicts absorption pairs) is the paper's central predictive claim:
- **Result**: Precision@20 = 0.0, Fisher p = 1.0 (no enrichment)
- This means the graph identifies ZERO correct absorption pairs among 520 predictions

The paper frames this as an "informative negative result" but then continues to rely on the LCA framework throughout. This is logically inconsistent.

## What Should Have Been Done

### Alternative 1: Pure Null-Result Paper
Focus on:
- Systematic null results after proper MCP
- Trained SAE vs random SAE comparison (this is actually interesting)
- Precision-recall asymmetry (H5 is the only robust finding)

Frame as: "We found no evidence that absorption degrades downstream tasks. However, we found that the Chanin metric may be sensitive to dictionary structure rather than genuine absorption."

### Alternative 2: Metric Validation Paper
Focus on:
- MCC~0.21 for random SAE (metric validity issue)
- The 8x absorption difference between trained and random SAEs
- Propose that Chanin metric measures dictionary structure, not absorption

Frame as: "We investigate whether the Chanin absorption metric is well-calibrated. Our results suggest the metric may be measuring dictionary geometry artifacts rather than genuine absorption differences."

### Alternative 3: Theoretical Paper
Focus on:
- LCA-SAE correspondence (theoretical, not empirical)
- Competitive suppression mechanism (theoretical, not validated)
- Homeostatic rebalancing (theoretical proposal)

Frame as: "We propose a theoretical framework for understanding absorption as competitive suppression. Empirical validation is left to future work."

## Recommendations

1. **Pick one framing and stick to it**: Either be a null-result paper, a metric validation paper, or a theoretical paper. Current mixed framing undermines credibility.

2. **Be honest about MCP results**: State clearly that no test survived multiple comparison correction. Do not present uncorrected p-values as evidence.

3. **Do not claim mechanistic explanation when predictions fail**: The LCA framework is interesting but H6 fails. Claiming the mechanism is "supported" while its primary prediction fails is misleading.

4. **Investigate metric validity**: Before interpreting trained<random absorption differences, establish that the Chanin metric actually measures absorption and not dictionary structure.