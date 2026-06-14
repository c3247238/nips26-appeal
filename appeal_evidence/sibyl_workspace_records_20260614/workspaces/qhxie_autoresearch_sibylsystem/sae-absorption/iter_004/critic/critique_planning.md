# Planning Critique

## Executive Summary

The experimental plan was well-structured with clear decision gates, realistic time estimates, and appropriate pilot validation. The pivot from Gemma 2 2B to GPT-2 Small was unavoidable (HuggingFace gating) but created a model-mismatch problem that was not adequately planned for. The safety probe experiment should have been dropped at the planning stage due to obvious confounding issues.

---

## Strengths

1. **Pilot validation gate (Component 0).** The 15-minute pilot with explicit decision rules (absorption rate in [10%, 45%], L0 matching, top-10 alpha_ij sanity check) is excellent experimental design. It prevents wasting GPU-hours on a broken pipeline.

2. **Ablation schedule.** Five pre-specified ablations (A1-A5) with explicit success criteria. Ablation A2 (decoder cosine pre-filter) turned out to be diagnostic of a critical failure mode (34% coverage), vindicating the pre-specification.

3. **Time estimates.** The 14 GPU-hour total was realistic. All 11 tasks completed within the allocated time, suggesting the estimates were accurate.

4. **Bonferroni correction pre-specification.** The H3 analysis pre-specified 4 simultaneous tests with alpha_corrected = 0.0125 and a meaningful-effect threshold of |r| > 0.3. This is rigorous statistical planning.

---

## Weaknesses

### 1. No plan for model fallback consequences

The methodology targeted Gemma 2 2B throughout, but gating forced a switch to GPT-2 Small for H1/H2/H4/taxonomy. The risk assessment listed "Gemma Scope API/access failure" at 20% probability with fallback to GPT-2 SAEs. But the plan did not anticipate the consequence: H3 would remain on Gemma (via SAEBench pre-computed data) while everything else moved to GPT-2, creating a cross-model gap.

A better plan would have specified: "If Gemma access fails, run H3 correlation on GPT-2 SAEBench data as well, ensuring all results are on the same model." GPT-2 SAEs are included in SAEBench; this was feasible.

### 2. Safety probe was planning over-reach

The safety probe (Component 3, Stage 3) required matching 3 SAEs on absorption level while controlling for layer, width, and architecture. With the available GPT-2 SAE configurations spanning different families, this matching was impossible. The plan specified "select 3 SAEs (lowest, median, highest absorption from Gemma 2 2B SAEBench)" -- but this was for Gemma, not GPT-2. After the model switch, the experiment became unfeasible as specified.

The experiment should have been marked "contingent on Gemma access" and dropped when the model switched.

### 3. Missing mixed-effects specification for H2

The PMI regression plan specified OLS with HC3 robust SEs. For 806 observations from 31 configs x 26 letters, a mixed-effects model with random intercepts for letter and config was the appropriate specification. This should have been planned from the start, not discovered as a limitation in post-analysis.

### 4. DAS estimation sample size mismatch

The methodology specifies "10,000 tokens through the SAE" for DAS estimation (Section 3.2). The actual implementation used 40 word-activation samples per letter (visible in C1D results: n_word_act_samples = 40 for most letters). This discrepancy was never resolved in the plan. 40 samples is inadequate for logistic regression with 3 predictors -- the DAS estimates are likely unreliable.

### 5. No plan for within-width stratification of H3

The plan specified partial correlations controlling for width, but never specified within-width stratified analysis. Given that the 54 SAEs span three very different widths (16k, 65k, 1M) and width is the dominant confound, stratification should have been a primary analysis, not an afterthought.

---

## Assessment

The plan was competent but over-ambitious. A more focused plan would have: (1) targeted a single model from the start, (2) dropped the safety probe, (3) specified mixed-effects models, and (4) pre-specified within-width stratification. The pivot infrastructure worked well; the problem was that the post-pivot plan was not updated to reflect the new model constraints.
