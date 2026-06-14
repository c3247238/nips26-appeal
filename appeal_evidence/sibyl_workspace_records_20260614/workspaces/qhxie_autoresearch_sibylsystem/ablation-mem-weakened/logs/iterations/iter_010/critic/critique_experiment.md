# Critique: Experiment (Iteration 9/10)

## Overview

The experiment program is comprehensive (10 major analyses across layers and models) and the statistical methods are generally sound, but there are critical issues with how results are interpreted and reported. Most critically: **zero hypotheses survive multiple comparison correction**, but the paper presents some as if they do.

## Critical Issues

### 1. Multiple Comparisons Problem (CRITICAL)

**Problem**: The paper tests 12 hypotheses (H1_L4_Pearson, H1_L4_Spearman, H1b_L4_Pearson, H1b_L4_Spearman, H2_L4_Pearson, H2_L4_Spearman, H1_L8_Pearson, H1_L8_Spearman, H1b_L8_Pearson, H1b_L8_Spearman, H2_L8_Pearson, H2_L8_Spearman) and applies Bonferroni correction with alpha=0.00417. **Zero hypotheses survive**.

**Evidence from correlation_report_full.json**:
```json
"n_rejected_bonferroni": 0,
"n_rejected_bh": 0,
"any_significant_bonferroni": false,
"any_significant_after_bh": false
```

Yet the paper says (writing/paper.md line 61):
> "layer 8 exhibits the strongest absorption-steering correlation (r=-0.431, p=0.028 for delta-corrected steering, H1b)"

This is presented as a finding. It is not. The uncorrected p=0.028 becomes p=0.334 after Bonferroni correction. The BH-FDR q=0.107 also fails the q<0.05 threshold.

**What should have happened**: The paper should prominently state "Zero hypotheses survived multiple comparison correction" and remove all uncorrected p-values as evidence of effect.

### 2. Post-Hoc Power Analysis (CRITICAL)

**Problem**: Section 4.5 (page 10) says "approximately 20% power to detect a medium effect size (r=0.5)". This is post-hoc power analysis—computing power after the experiment has completed.

**Why this is wrong**: Post-hoc power analysis is methodologically invalid (Cohen, 1994). Observed p-values already contain all the information about power that matters—there's no need to "retrofit" power analysis to explain non-significant results.

**Evidence of the problem**: With n=26 and medium effect r=0.5, achieved power is approximately 47% (pre-hoc). The 20% figure comes from the observed effect size being smaller than expected. But this is circular: you can't use the observed effect size to compute power to detect the effect you didn't detect.

**Recommendation**: Remove post-hoc power analysis entirely. Replace with pre-hoc power statement if needed: "A priori power analysis (alpha=0.05, n=26, medium effect r=0.5) indicates approximately 47% power."

### 3. Metric Validation Concern (CRITICAL)

**Problem**: The Chanin absorption metric shows 8x higher absorption in random SAEs (0.278) vs trained SAEs (0.034). This could mean:
1. The metric correctly detects that trained SAEs have reduced absorption (interpretation the paper uses)
2. The metric is sensitive to dictionary structure/artifacts, which random SAEs have more of (neutral interpretation)
3. The metric is fundamentally flawed and measures nothing meaningful (negative interpretation)

**The paper assumes interpretation 1** without acknowledging alternatives.

**Evidence from pilot_summary.json**:
```json
"h10_random_sae_baseline": {
  "key_finding": "Random SAE shows 8x higher absorption than trained SAE (0.278 vs 0.034), opposite to prediction",
  "recommendation": "Chanin metric is not specific to learned structure; report as methodological finding"
}
```

The pilot summary itself says "Chanin metric is not specific to learned structure" but the paper treats it as specific.

**Recommendation**: Add explicit discussion of the three interpretations. Conclude: "We cannot distinguish between these interpretations with current data. The metric may conflate structural artifacts with genuine absorption phenomena."

### 4. n=4 Layers Insufficient for H9 (MAJOR)

**Problem**: H9 (layer-dependent structure) uses 4 layers (L0, L4, L8, L10). Pearson r=+0.82 is descriptive, not inferential. With 4 data points, no statistical test is appropriate.

**Evidence from writing/paper.md line 265**:
> "Pearson correlation between mean edge weight and layer index is r=+0.82 (though with only 4 layers, this is descriptive rather than inferential)"

The paper acknowledges this but then says (line 267):
> "aligns with prior findings that deeper layers exhibit stronger absorption effects"

This "aligns with" language treats the descriptive trend as supporting evidence for a substantive claim. It's not.

**Recommendation**: Remove "aligns with prior findings" language. Report H9 as purely descriptive: "A descriptive trend was observed: mean edge weight ranged from 0.312 (L0) to 0.384 (L8) to 0.367 (L10). With n=4 layers, no inferential statistics are appropriate."

### 5. H6 Falsification Not Emphasized (MAJOR)

**Problem**: H6 is the primary predictive hypothesis for the LCA-SAE correspondence framework, and it failed completely (precision@20=0.0). But the paper treats this as a "nuance" rather than the central finding.

**Evidence from writing/paper.md line 226**:
> "Result: Precision@20 = 0.0. None of the 520 top-20 neighbor predictions (26 features × 20 neighbors) correspond to a known absorption pair."

This is correct but the framing is wrong. The failure of H6 means the local inhibition graph with k=20 does not identify absorption pairs. This is a genuine, informative negative result, but the paper treats it as a caveat.

**Recommendation**: Elevate H6 falsification to a first-class result. Frame it: "The local inhibition graph with fixed k=20 does not predict absorption pairs. This suggests inhibition operates at finer granularity than top-k neighbor relationships. Future work should explore larger k, adaptive neighborhoods, or context-dependent edge weighting."

### 6. Missing Cross-Model Validation (MAJOR)

**Problem**: All experiments use GPT-2 Small. The proposal mentions Gemma-2-2B pilot but it was "not completed due to resource constraints." Cross-model validation is essential for generalizability claims.

**What exists**:
- cross_model_pythia_combined.json (Pythia-70M, but "inconclusive; limited feature overlap")
- full_absorption_gemma_l8_16k_DONE (Gemma 2, incomplete)

**Recommendation**: Either (a) complete the Gemma 2 experiment, or (b) remove all generalizability claims and explicitly state the findings are specific to GPT-2 Small.

## Summary

The experiments are methodologically sound but the interpretation has serious problems:
1. H1b p=0.028 is cited as evidence despite failing MCP
2. Post-hoc power analysis is methodologically invalid
3. Metric validation concern is acknowledged in pilot but ignored in paper
4. n=4 layers cannot support inferential claims about layer dependence
5. H6 falsification is treated as a caveat, not a finding

**Action items**:
1. Remove all uncorrected p-values as evidence of effect
2. Remove post-hoc power analysis
3. Add explicit discussion of metric interpretation ambiguity
4. Remove "aligns with prior findings" language for H9
5. Elevate H6 falsification to first-class result
6. Complete or explicitly limit cross-model generalizability