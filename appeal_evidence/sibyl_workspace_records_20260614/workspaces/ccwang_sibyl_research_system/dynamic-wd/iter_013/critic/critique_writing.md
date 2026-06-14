# Writing Critique

## Executive Summary

The writing quality is above average for a first draft. The paper is well-structured, honestly discusses limitations, and avoids egregious overclaiming. However, several issues need attention.

## MAJOR: Scope Mismatch Between Proposal and Paper

The original proposal promised:
1. Cumulative Alignment Contraction Theory (Theorems 1-2)
2. Unified Phi Formulation across all 4 WD streams
3. Standardized metrics (BEM, CSI, AIS)
4. Four types of contributions (theoretical, algorithmic, empirical, framework)

The delivered paper provides:
1. EqWD algorithm (Contribution 1)
2. Competitive empirical performance (Contribution 2)
3. Ratio sufficiency analysis with AIS (Contribution 3)
4. Ablation studies (Contribution 4)

Only AIS survived from the original framework vision. The other theoretical contributions were dropped without explanation. This is fine as pragmatic scope reduction, but the paper should not reference the original framework ambitions.

## MAJOR: Proposition 2 is Circular

Proposition 2 states: "IF alignment deviation is a function of (||g||, ||w||), THEN ratio is sufficient."

The proof sketch simply restates the assumption. The proposition adds no formal content---it says "if X implies Y, then Y holds when X holds." The real contribution is the AIS diagnostic that empirically tests the assumption.

Recommendation: Downgrade to "Empirical Observation" or "Hypothesis" and lead with the AIS evidence.

## MAJOR: Missing Budget Equivalence Discussion

The paper does not mention the budget equivalence results anywhere. This is a planned core experiment (Phase 4 of methodology) whose results are available but unreported. Whether the result is positive or negative, omitting planned experiments is a red flag for reviewers.

## MINOR: Factual Error in Variance Claim

The paper states EqWD has "the lowest standard deviation (0.20%) among all methods" on ImageNet. But NoWD has std=0.153% and CAWD has std=0.154%. The claim should be qualified: "lowest among methods exceeding the FixedWD baseline."

## MINOR: Inconsistent Standard Deviations

CIFAR analysis summary shows FixedWD std=0.21, paper reports 0.25, aggregate JSON shows 0.253. These minor inconsistencies suggest different rounding or computation methods.

## POSITIVE: Strengths

1. **Intellectual honesty**: Section 5.6 (Limitations) is unusually thorough, explicitly listing 7 limitations
2. **Balanced claims**: The paper avoids superlative claims and appropriately hedges with "tends to", "directional trend"
3. **Clear method description**: Algorithm 1 and the design rationale section are well-written
4. **Good related work coverage**: Section 2 is comprehensive and fairly positions competitors
5. **Discussion quality**: Sections 5.1-5.4 provide thoughtful analysis of why certain methods underperform

## Structural Suggestions

1. Move the CAWD negative result and AIS finding to more prominent positions---these are arguably the paper's most novel contributions
2. Add a "Budget Equivalence" subsection in Section 4 or 5 reporting the Bayesian optimization results
3. Explicitly address the effective WD inflation confound in the main text, not just the limitations list
4. Consider whether the paper would be stronger reframed as "When does adaptive WD help?" rather than "EqWD: a new algorithm"
