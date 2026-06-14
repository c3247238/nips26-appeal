# Critique: Introduction

## Summary Assessment
The introduction presents a clear problem (feature absorption as a central pathology with an untested benchmark), motivates it well, and states three focused research questions with four concrete contributions. However, there is a significant framing mismatch between the paper's actual content (a construct-validity study of the absorption metric on semantic hierarchies) and the broader Goodhart's Law framing in the proposal. The intro also overstates the Random-SAE finding and contains a few terminology inconsistencies with the glossary.

## Score: 6/10
**Justification**: The section is readable, well-structured, and hits the standard introduction beats. But it has a critical framing issue (the paper does not actually test Goodhart's Law), overclaims on the Random-SAE evidence, and lacks precision in a few technical details. To reach 7-8, it needs tighter alignment between what the paper does and what it promises, plus cleaner terminology.

## Critical Issues

### Issue 1: Framing Mismatch — Paper Promises Goodhart's Law But Delivers Construct Validity
- **Location**: Section 1.1, paragraph 3; Section 1.4, contribution 3; closing paragraph
- **Quote**: "Our findings reveal a methodological blind spot in a widely adopted benchmark and suggest that domain-specific absorption metrics---validated for hierarchy specificity and sensitivity to training---are needed before absorption scores can guide architecture selection for real-world interpretability tasks."
- **Problem**: The proposal (idea/proposal.md) explicitly reframes the project around Goodhart's Law ("Does the metric matter?"), but the actual experiments in iter_003 are the same construct-validity study from iter_002 (first-letter vs. semantic-hierarchy correlation, hierarchy specificity, Random-SAE control). The paper does NOT test whether architectures are gaming the metric, does NOT compare absorption reduction to downstream utility, and does NOT demonstrate benchmark optimization. It tests whether the metric generalizes to semantic hierarchies---a construct-validity question, not a Goodhart's Law question.
- **Fix**: Either (a) reframe the intro around construct validity as the primary contribution and drop the Goodhart's Law framing entirely, or (b) add the actual Goodhart's Law experiments (utility prediction, random-baseline-corrected architecture comparison) to the results. Given the existing data, option (a) is more honest: frame this as "the first construct-validity study of the SAEBench absorption metric" and save Goodhart's Law for a follow-up paper that actually tests utility prediction.

### Issue 2: Random-SAE Claim Overstates the Evidence
- **Location**: Section 1.4, contribution 3
- **Quote**: "A randomized SAE with permuted decoder directions achieves identical semantic-hierarchy absorption (0.352) to the trained Standard SAE, indicating that the semantic adaptation of the metric captures artifacts unrelated to learned structure."
- **Problem**: The Random SAE matches the Standard SAE on semantic-hierarchy absorption (0.352), but this is only ONE architecture pair. The claim "the semantic adaptation of the metric captures artifacts unrelated to learned structure" is too strong. It could be that the Standard SAE happens to perform poorly on semantic hierarchies, not that the metric is universally degenerate. The Random SAE also does NOT match trained SAEs on first-letter absorption (0.030 vs. 0.026 for Standard, but 0.576 for TopK), so the metric is not degenerate for all tasks.
- **Fix**: Soften to: "On semantic hierarchies, a Random-SAE control achieves scores identical to the trained Standard SAE (0.352), suggesting that the semantic-hierarchy adaptation of the metric may capture geometric artifacts rather than learned structure." Add nuance: the first-letter task does distinguish trained from random SAEs, so the degeneracy is task-specific.

### Issue 3: Contribution 2 Overstates the Hierarchy Specificity Finding
- **Location**: Section 1.4, contribution 2
- **Quote**: "Non-hierarchy correlated features show significantly higher absorption (\bar{A}_{NH} = 0.331) than semantic hierarchies (\bar{A}_{SH} = 0.235; paired t-test: t = -4.748, p = 0.003), rejecting the hypothesis that the metric is specific to hierarchical structure."
- **Problem**: This is accurate for the data, but the intro presents it as a definitive rejection without acknowledging the synthetic sentence template as a potential confound. The method section (2.3) notes that sentences follow a fixed template ("The [concept] is [property]."), which could introduce spurious correlations. A reviewer could argue that the non-hierarchy pairs (big-large, fast-quick) are easier to distinguish in this template than hierarchy pairs (building-house), making the specificity failure an artifact of the experimental design rather than the metric itself.
- **Fix**: Add a brief qualifier: "This rejection holds under our experimental conditions; we discuss alternative explanations, including template-induced spurious correlations, in Section 4.2."

## Major Issues

### Issue 4: Missing Citation for "One of Eight Canonical Evaluations"
- **Location**: Section 1.2, paragraph 2
- **Quote**: "These virtues have made first-letter absorption one of eight canonical evaluations in SAEBench, and architecture papers routinely report it as a primary metric (Bussmann et al., 2025; Rajamanoharan et al., 2024)."
- **Problem**: The claim "one of eight canonical evaluations" is specific but unsourced. The reader cannot verify which eight evaluations exist or whether absorption is truly canonical. The citations (Bussmann et al., 2025; Rajamanoharan et al., 2024) support architecture papers reporting absorption, but not the "eight canonical evaluations" claim.
- **Fix**: Either cite SAEBench (Karvonen et al., 2025) for the specific list of eight evaluations, or soften to "one of several standardized evaluations in SAEBench."

### Issue 5: Absorption Formula Not Defined in Intro
- **Location**: Section 1.2, paragraph 1
- **Quote**: "The absorption score A_full quantifies the maximum relative accuracy drop across these three conditions."
- **Problem**: The formula for A_full is not given in the intro, only a vague verbal description. The reader must wait until Section 2.4 to see the actual equation. For a metric that is central to the paper, the intro should at least sketch the formula.
- **Fix**: Add the formula inline or in a footnote: A_full = max(0, (acc_resid - acc_sae)/acc_resid, (acc_resid - acc_ksparse)/acc_resid). This takes one line and greatly improves clarity.

### Issue 6: RQ3 Is Underdeveloped in the Paper
- **Location**: Section 1.3, RQ3
- **Quote**: "RQ3 (Robustness): How stable is the correlation across feature-splitting thresholds (tau_fs) in k-sparse probing?"
- **Problem**: RQ3 receives minimal treatment in the results (Section 3.4 is only a few sentences) and is not mentioned in the contributions or conclusion. It feels like a minor robustness check rather than a full research question. The outline lists it but the paper does not deliver on it.
- **Fix**: Either elevate RQ3 with more analysis (e.g., discuss why the correlation is stable but inconclusive, implications for threshold selection), or demote it to a secondary analysis and reframe the intro accordingly.

### Issue 7: Transition Sentence Is Weak
- **Location**: End of Section 1.4
- **Quote**: "To test these questions, we need a precise measurement protocol."
- **Problem**: This is a generic transition that does not build momentum. It also repeats the "To test these questions" formula from the outline without adapting it to the actual flow of the section.
- **Fix**: Something more specific: "We now describe the measurement protocol that adapts the SAEBench absorption evaluator to semantic hierarchies, non-hierarchical controls, and a randomized baseline."

## Minor Issues

- **Section 1.1, paragraph 2**: "Chanin et al. proved that this phenomenon is not an artifact of training dynamics but a structural consequence of the sparsity objective when features have hierarchical relationships." → The proposal notes that the metric may measure co-occurrence rather than hierarchy. The intro should hedge this claim slightly: "Chanin et al. proved that absorption is incentivized by sparsity loss for hierarchical features, though our results suggest the metric may also respond to non-hierarchical correlations."

- **Section 1.2, paragraph 1**: "Ground-truth logistic probes measure classification accuracy on base-model residual activations, full SAE latents, and top-k sparse SAE latents." → The glossary defines "ground-truth probe" as trained on base-model activations. The intro conflates the probe types. Fix: "A ground-truth logistic probe on base-model residual activations provides an upper-bound baseline; probes on full SAE latents and top-k sparse latents measure information loss."

- **Section 1.4, contribution 1**: "First construct-validity test." → This is accurate but could be more specific: "First construct-validity test on semantic hierarchies derived from WordNet."

- **Terminology**: The intro uses "first-letter" (correct per glossary) but also "first letter" (without hyphen) in the abstract outline. Ensure consistency.

- **Section 1.3**: The RQs are labeled "RQ1 (Construct Validity)", "RQ2 (Hierarchy Specificity)", "RQ3 (Robustness)" but the hypotheses in the method are H1, H2, H3. The mapping is clear but could be made explicit: "We test these questions via three hypotheses (H1-H3, Section 2.6)."

- **Missing visual**: The intro has no figures, which is fine, but given the complexity of the absorption concept, a small diagram illustrating parent-child absorption (even referenced as "Figure 1, inset") could help. Not critical for NeurIPS but worth considering.

## Visual Element Assessment
- [x] Figures/tables match outline plan (no figures planned for intro --- correct)
- [x] All visuals referenced before appearance (N/A --- no visuals)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support: Section 1.2 describes the first-letter benchmark in detail without any visual. A small schematic of the probe conditions (residual -> SAE -> k-sparse) would reduce cognitive load.

## What Works Well

1. **Strong opening paragraph (1.1, para 1)**: Leads with the concrete thing (SAEs decompose activations into interpretable features) and immediately states the promise and the problem. No generic "In recent years..." opening.

2. **Clear problem-solution structure**: The intro follows a classic and effective arc: (a) absorption is important, (b) the benchmark is widely used, (c) the benchmark has never been validated on real hierarchies, (d) we validate it. A reader can follow this without prior knowledge.

3. **Specific numbers in contributions**: Contributions 2 and 3 include exact statistics (t = -4.748, p = 0.003; 0.352), which is excellent practice. This grounds the claims and makes the contributions feel concrete rather than aspirational.
