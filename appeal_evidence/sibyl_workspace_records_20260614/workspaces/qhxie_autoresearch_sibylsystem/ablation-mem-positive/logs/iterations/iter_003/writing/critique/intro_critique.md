# Critique: Introduction

## Summary Assessment
The Introduction establishes a clear narrative from actionability paradox to CV-based resolution, with specific pilot data and a well-structured contributions list. Main weaknesses: (1) full statistical table in the narrative flow is inappropriate for introduction level, (2) "variance paradox" framing is confusing given the paper's actual focus, (3) H6 falsification appears prematurely without proper context, (4) subsection headers are missing despite clear outline structure.

## Score: 7/10
**Justification**: Solid introduction with good evidence-based claims. The issues are structural (subsection headers missing, table placement) rather than fundamental. A revision addressing these structural issues and narrative flow would elevate the score to 8+.

## Critical Issues

### Issue 1: Full Statistical Table Disrupts Narrative Flow
- **Location**: Lines 23-29 (Table 1)
- **Quote**: "Table 1: Steering Effect by CV Group and Strength | Strength | High-CV Mean | High-CV Std | Low-CV Mean | Low-CV Std | t-statistic | p (BH-adj) | ..."
- **Problem**: The inline markdown table with full statistics (mean, std, t-statistic, p-value for all three strengths) interrupts the narrative. This level of detail is appropriate for the Experiments section. A reader skimming the introduction should learn the key finding ("47% larger effect, p < 0.01") without wading through a 7-column table.
- **Fix**: Replace table with: "Across all three steering strengths (+3, +5, +7), high-CV features showed significantly larger steering effects than low-CV features (p < 0.01 with Benjamini-Hochberg correction), with an aggregate effect ratio of 1.47 (0.5251 vs 0.3565 mean absolute logit change)."

### Issue 2: Missing Subsection Headers
- **Location**: Entire section (no subsection headers)
- **Problem**: The outline specifies clear subsections (1.1 The Interpretability Actionability Gap, 1.2 Research Gap, 1.3 Our Approach: CV-Based Decomposition, 1.4 Contributions). The current text has no headers, making it impossible to verify these are all covered and making the section harder to navigate.
- **Fix**: Add headers at appropriate points: "## 1.1 The Interpretability Actionability Gap" (before line 1), "## 1.2 Research Gap" (before line 11), "## 1.3 Our Approach" (before line 17), "## 1.4 Contributions" (before line 35).

## Major Issues

### Issue 3: H6 Falsification Prematurely Buried in Introduction
- **Location**: Lines 33-34
- **Quote**: "Our results falsify this hypothesis: Pearson correlation between orthogonality and steering effect is r = -0.136 (p = 0.301, not significant)."
- **Problem**: The introduction mentions H6 was falsified but (1) doesn't explain what H6 was or why it was hypothesized, (2) doesn't explain why this matters, (3) introduces a negative result without proper framing. This confuses readers about whether this is a central or peripheral finding.
- **Fix**: Move to Experiments section. In the intro, simply state: "We also tested decoder orthogonality as an alternative predictor (Section 4.4) and found no significant correlation."

### Issue 4: "Variance Paradox" Framing Misleads About Paper's Focus
- **Location**: Line 19
- **Quote**: "a 733x ratio we term the 'variance paradox.'"
- **Problem**: The paper's contribution is CV predicting steering actionability. "Variance paradox" suggests the paper's central puzzle is explaining why absorbed features have high CV. But the paper treats the 733x ratio as a useful discriminating variable, not a paradox to solve. The "paradox" framing may mislead readers about the paper's actual contribution.
- **Fix**: Change to: "This 733x ratio in CV between absorbed and non-absorbed features provides a discriminating variable for our analysis."

### Issue 5: "Variance Paradox" Not Explained as Paradoxical
- **Location**: Lines 18-19
- **Quote**: "absorbed features exhibit CV approximately 7.33, while non-absorbed features exhibit CV approximately 0.01—a 733x ratio we term the 'variance paradox.'"
- **Problem**: The term "paradox" implies something counterintuitive—the expectation that absorbed features would have lower or similar CV than non-absorbed. But no expectation is stated, so readers cannot understand why it's paradoxical.
- **Fix**: Either explain the paradox: "This is counterintuitive: absorption is thought to degrade feature specificity, yet absorbed features show dramatically higher activation variability than non-absorbed features" or remove "paradox" framing entirely.

## Minor Issues

### Issue 6: Line 3 - Citation year already present
- **Location**: Line 3
- **Quote**: "Basu et al. (2026)"
- **Note**: The existing critique incorrectly flagged "Basu et al." without year. The current text already includes (2026). No fix needed.

### Issue 7: Line 19 - Use LaTeX notation for CV
- **Location**: Line 19
- **Quote**: "CV = sigma / mu"
- **Problem**: Uses informal text math rather than proper notation from notation.md
- **Fix**: Change to "$CV = \sigma / \mu$"

### Issue 8: Lines 35-45 - "causal actionability" undefined
- **Location**: Contribution 3
- **Quote**: "First connection between coefficient of variation and causal actionability"
- **Problem**: "causal actionability" is not defined; use "steering actionability" which is defined in glossary
- **Fix**: Change to "First connection between coefficient of variation and steering actionability"

### Issue 9: "steerable" vs "steering potential" terminology
- **Location**: Line 41
- **Quote**: "CV > 1.0 indicates steerable absorbed features"
- **Problem**: Glossary prefers "steering potential" over "steerable"
- **Fix**: Change to "CV > 1.0 indicates absorbed features with steering potential"

## Visual Element Assessment
- [x] Figures/tables match outline plan - No figures planned for intro (outline: "<!-- FIGURES - None -->")
- [x] All visuals referenced before appearance - Table 1 appears after reference at line 21, within the same section
- [x] Captions are self-explanatory - N/A (no figures)
- [x] No text-heavy sections that need visual support - Introduction is appropriately textual

## What Works Well

1. **Precise problem framing (lines 1-5)**: "98.2% AUROC for identifying clinical concepts" and "zero measurable change in model outputs" - exact numbers make the actionability paradox concrete and credible.

2. **Pilot evidence integrated naturally (lines 7-9)**: "0.153 vs 0.075 logit change, pilot data" and "67.3% of their activation on average" - specific numbers with experimental context establish credibility before proposing the approach.

3. **Well-structured contributions (lines 35-45)**: Numbered list with specific scope limitations ("partial resolution," "non-clinical LLM domain") strengthens credibility. The ordering (empirical finding first, then method, then theory, then resolution) properly prioritizes novelty.

4. **No banned patterns detected**: No generic openings, no hollow self-praise, no vague "significantly improves" without numbers.

5. **Honest framing of scope**: "may explain why Basu et al. observe universal failure in clinical domain" - appropriately hedged about generalization.

## Cross-Section Consistency Check

**Terminology**:
- "actionability paradox" ✓ consistent with glossary
- "robust absorbed" / "fragile absorbed" ✓ consistent with method and glossary
- "variance paradox" appears but is confusingly framed (see Issue 4-5)
- "CV-based decomposition" ✓ matches glossary preference

**Notation**:
- Line 19: "sigma / mu" should be "$\sigma / \mu$" per notation.md

**Claim consistency**:
- Lines 7-8 pilot claims match proposal and pilot_summary.md ✓
- Table 1 data: need to verify numbers match experiments section (experiments.md lines 35-37 vs intro lines 27-29)

**Reference consistency**:
- Basu et al. (2026) properly cited on first mention (line 3) ✓