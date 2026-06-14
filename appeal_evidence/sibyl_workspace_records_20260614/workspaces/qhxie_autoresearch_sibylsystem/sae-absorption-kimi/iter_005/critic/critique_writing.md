# Writing Critique: Component-Isolated Study of SAE Absorption Reduction

## Executive Summary

The paper is well-structured, internally consistent, and avoids common writing pitfalls. The abstract is concise and informative. Negative results are reported honestly. However, several writing issues create tension between the paper's honest limitation statements and its confident conclusions. The most serious issue is the mismatch between provisional data (3 of 6 variants) and definitive-sounding claims.

## Critical Issues

### 1. Definitive Claims from Provisional Data (CRITICAL)

The paper states in Section 1.5 (Scope note) that only 3 of 6 variants have full data, and the ranking is provisional. This is honest and well-placed. However, the Conclusion (Section 6.1) states:

> "The provisional component ranking, based on available data (3 of 6 variants with full replicates), is: **TopK ($d = 5.51$) $\gg$ MultiScale ($d \approx 1.1$, pilot) $\gg$ Orthogonality ($d = 0.14$)**."

This is followed immediately by Section 6.2:

> "If the provisional ranking holds when the full variant set is completed, the community's investment in more complex architectures may be misdirected."

The conditional "if" is appropriate, but the Abstract states the finding more definitively:

> "Our key finding is that TopK sparsity---not multi-scale decomposition or orthogonality---is the dominant driver of absorption reduction"

**Fix**: Add "(among tested components)" or "(provisional)" to the Abstract and Conclusion claims. The Abstract should not overstate the certainty of a finding based on incomplete data.

### 2. Repetitive Phrasing (MODERATE)

"An order of magnitude larger" appears at least 4 times:
- Abstract: "an order of magnitude larger than any other tested component"
- Section 4.2: "an order of magnitude larger than any other tested component"
- Section 4.7: "TopK ($d = 5.51$) is roughly five times larger than MultiScale ($d \approx 1.1$)"
- Section 6.1: "TopK achieves a far larger effect than any other tested component"

This is not just repetitive; it is inaccurate in some cases. TopK vs MultiScale is ~5x, not ~10x. Only TopK vs Orthogonality (~39x) is "an order of magnitude."

**Fix**: Vary phrasing and use accurate multipliers: "fivefold larger than MultiScale" and "fortyfold larger than Orthogonality."

### 3. Verbatim Repetition Across Sections (MODERATE)

"This redirects the research question" appears in both Section 4.6 and Section 5.1 with nearly identical wording:

- Section 4.6: "This finding redirects the research question: instead of asking 'which architecture reduces absorption?' we should ask 'why does sparsity control absorption, and what is the optimal sparsity level?'"
- Section 5.1: "Instead of asking 'which architecture reduces absorption?' the field should ask: (1) what is the causal mechanism linking sparsity to absorption? (2) what is the optimal sparsity level..."

**Fix**: Rephrase one instance. Section 4.6 should present the observation; Section 5.1 should develop the implications.

### 4. MSE Column Formatting Inconsistency (MODERATE)

Table 3 header says "MSE ($\times 10^{-3}$)" but the values (10.44, 7.68, 0.03) lack standard deviations. All other columns show "mean $\pm$ std." The caption says "Values are mean $\pm$ std across 5 replicates" but MSE does not follow this format.

**Fix**: Add std to MSE values: "10.44 $\pm$ 0.85", "7.68 $\pm$ 0.28", "0.03 $\pm$ 0.00" (or report raw MSE without scaling).

### 5. "Near-Perfect" Overstatement for n=4 Correlation (MODERATE)

Figure 3 caption says "near-perfect" for the r ~ -0.97 correlation. With n=4 points, "near-perfect" overstates certainty. The text is more careful ("strong"), but the caption is too strong.

**Fix**: Change caption to "strong negative correlation ($r \approx -0.97$, exploratory, $n = 4$ variants)" and add the bootstrap CI to the main text.

## Minor Issues

1. **"The practical implications are severe" (Section 1.1)**: Slightly overstated. Better: "The practical implications are concrete."

2. **"Hedging score ($H$)" defined but symbol never used**: Section 3.4 defines $H$ but Table 3 uses "Hedging" without the symbol. Either use $H$ in the text or remove the symbol from the definition.

3. **Section 1.3 "Our prior work (iterations 2--4)" not cited**: The specific statistics are presented without context. A brief footnote would help.

4. **Table 1 lists 6 variants but only 3 have full data**: A visual indicator (asterisks) would prevent reader confusion.

## What Works Well

1. **Honest negative results**: H3 is explicitly "NOT SUPPORTED." The incomplete variant set is flagged prominently.

2. **Clear mechanism explanation**: Section 5.1 explains *why* TopK dominates with a plausible mechanism.

3. **Well-crafted abstract**: Leads with problem, states method, reports key finding with numbers, states implication.

4. **Scope note (Section 1.5)**: A model of epistemic honesty.
