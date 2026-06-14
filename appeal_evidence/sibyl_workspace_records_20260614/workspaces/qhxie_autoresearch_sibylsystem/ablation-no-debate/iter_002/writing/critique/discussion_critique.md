# Critique: Discussion

## Summary Assessment
The Discussion is thoughtful and integrates findings well, particularly the two-player dynamic interpretation (encoder compresses, decoder decompresss) and the capacity-pressure mechanism. The limitations section is honest and appropriately scoped. The primary weakness is repetition: many of the findings described in Discussion are already stated in the Experiments section, making some paragraphs read as a summary rather than an interpretation. Additionally, the practical implications could be more actionable for the target audience of SAE practitioners.

## Score: 7/10
**Justification**: Excellent interpretation of findings and honest limitations. Deducted points for: (1) significant repetition of results already stated in the Experiments section, (2) the two-player dynamic interpretation, while insightful, is not connected to any specific data figure or table in the Experiments section, (3) the implications for practitioners are vague ("encoder regularization" could be more concretely specified).

## Critical Issues

### Issue 1: Repetition of Experimental Results
- **Location**: Sections 7.1 through 7.4
- **Problem**: Most of Sections 7.1-7.4 restate findings that are already clearly presented in the Experiments section. For example, Section 7.1 re-states the encoder/decoder effect sizes (E_enc = 0.843, E_dec = 0.011) and statistical insignificance of decoder -- all of which appear in Section 6.1. The Discussion should *interpret* these findings, not re-report them. Compare to how Bricken et al. (2023) Structure section discusses implications of findings, not their numerical values.
- **Fix**: Replace re-statement with interpretation. For 7.1: focus on what the B < D comparison reveals about the decoder's compensatory role. For 7.2: focus on the practical implications of the capacity-pressure curve (is it monotonic? what's the inflection point?). For 7.3-7.4: focus on what the null results mean for the field's assumptions about absorption.

### Issue 2: Two-Player Dynamic Lacks Direct Empirical Support
- **Location**: Section 7.1, lines 6-9
- **Problem**: The claim that "Condition D shows lower absorption than Condition B, suggesting the trained decoder actively compensates" is an inference, not a measured result. The paper's factorial design does not include a Condition B vs D comparison as a primary analysis -- it emerges from observing that D < B, but this was not a pre-registered hypothesis and the comparison is not quantified or tested for statistical significance. The decoder compensation interpretation is plausible but unproven.
- **Fix**: Either (a) add a quantitative comparison of Condition B vs D absorption with a paired t-test or equivalent, or (b) soften the language to "suggests the possibility that the decoder compensates" rather than stating it as a conclusion. Note this as a post-hoc observation requiring future validation.

### Issue 3: Practical Mitigation Recommendations Are Vague
- **Location**: Section 7.6, lines 47-53
- **Problem**: The three implications for the field are stated at a high level: "encoder regularization," "training-free detection," "reframing absorption." For a practitioner reading this paper, it is unclear what "encoder regularization" specifically entails. Is it an L2 penalty on parent-child encoder dot products? A contrastive loss? A modified activation function? The paper has the data to make specific recommendations and should do so.
- **Fix**: Add a concrete pilot recommendation for each implication. For encoder regularization: "A promising pilot would add a penalty term $\lambda \cdot \sum_{(p,c) \in \text{parent-child}} (w_p^\top w_c)^2$ to the SAE loss, discouraging encoder columns for child features from aligning with parent feature directions." For training-free detection: specify what geometric analysis of the encoder matrix would predict absorption risk.

## Minor Issues

- **Structure**: The section ordering (findings → limitations → implications) is standard, but the implications could benefit from being more prominent. Consider moving the most actionable implications (7.6) earlier and combining with 7.1-7.3.
- **Figure/table references**: Section 7.1 should explicitly reference the data supporting the B < D observation (which table or figure shows this comparison?).
- **Limitation interaction**: Section 7.5 mentions metric inconsistency across experiments but Section 7.1 (which discusses encoder-decoder interaction) uses the overlap fraction metric. Is this consistent? The metric inconsistency could affect the two-player dynamic interpretation.
