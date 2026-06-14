# Critique: Experiments

## Summary Assessment

The Experiments section is technically sound with numbers that accurately match the underlying data files. The structure is clear and the negative results are handled appropriately. However, there are critical issues: (1) all four figure references point to non-existent files that will break the final paper, (2) the H_Safe hypothesis is silently dropped without acknowledgment in the section itself, and (3) the steering methodology description conflicts with the proposal on alpha values. The section would score well on technical correctness but poorly on completeness and visual communication.

## Score: 6/10

**Justification**: The section is competent but incomplete. Numbers are accurate and the logical flow is good, but missing generated figures is a critical failure for a paper at a top venue. Additionally, the silent omission of H_Safe undermines the paper's promised scope. A score of 8+ would require: (1) all figures generated and referenced correctly, (2) explicit acknowledgment of H_Safe non-implementation, (3) resolution of the alpha value discrepancy with the proposal.

## Critical Issues

### Issue 1: All Figures Referenced Are Not Generated
- **Location**: Lines 41, 66, 86, 119
- **Quote**: `[...] ![Multi-child proportional absorption rates...](figures/fig1_h1_absorption_comparison.pdf) [...]`
- **Problem**: The `figures/` directory does not exist at `writing/sections/figures/`. The markdown references four PDF figures that have not been generated. The `<!-- FIGURES -->` marker at the bottom lists generation scripts but no actual figures exist.
- **Fix**: Generate all four figures before submission. Scripts are referenced in the FIGURES marker comment. Verify the figures appear BEFORE their first text reference (currently Figure 3 is referenced at line 66 but its caption suggests it should appear after Figure 1 results).

### Issue 2: H_Safe Hypothesis Silently Omitted
- **Location**: Throughout section (absent)
- **Quote**: N/A - the hypothesis is simply not discussed
- **Problem**: The outline (Section 6) promises four hypotheses including H_Safe (safety-critical features show elevated absorption). The proposal explicitly lists H_Safe as a primary hypothesis. However, the experiments section only covers H1, H2, and H3. The discussion mentions "H_Safe not implemented" but this acknowledgment should appear in the experiments section itself, not buried in discussion.
- **Fix**: Add a brief subsection 6.7 "Safety-Critical Feature Analysis (Not Implemented)" stating: "Due to time constraints, H_Safe (safety-critical feature absorption) was not evaluated in this study. This remains an important open question for future work." This prevents reviewers from wondering why the promised hypothesis is missing.

## Major Issues

### Issue 3: Alpha Value Discrepancy With Proposal
- **Location**: Section 6.4.1, line 115
- **Quote**: "with $\alpha \in \{0.0, 0.1, 0.2\}$"
- **Problem**: The proposal specifies alpha values as {0.05, 0.1, 0.15, 0.2, 0.25}. The experiments section only tests 0.0, 0.1, 0.2 (three values instead of five). The actual JSON data confirms only these three alphas were used. This is a methodological deviation that should be documented.
- **Fix**: Either (a) update text to match actual methodology {0.0, 0.1, 0.2} and note this as a reduced alpha set, or (b) regenerate results with full alpha range. Current text misleadingly implies the full range was tested.

### Issue 4: Figure 3 Caption Misleading
- **Location**: Line 66-68
- **Quote**: "Trained SAEs exhibit higher proportional variance (mean $= 0.1154$, std $= 0.0072$) than all baselines."
- **Problem**: The caption at line 68 says "Random decoder shows near-zero variance" but the actual variance is 0.0040, not zero. This is technically accurate (near-zero) but potentially misleading given the dramatic framing. More importantly, this subsection is labeled 6.2.3 "Proportional Variance Analysis" but it belongs more logically to H1 supplementary analysis rather than being a separate sub-hypothesis.
- **Fix**: Clarify in text that 0.0040 is the actual value (even if "near-zero" is acceptable shorthand). Consider moving this analysis into H1's results or explicitly noting it as supplementary.

### Issue 5: Steering Non-Response Warrants Deeper Analysis
- **Location**: Section 6.4.2, line 130
- **Quote**: "This holds despite testing multiple alpha values and a substantial sample of 1021 features total."
- **Problem**: The sample size claim is misleading. Only 7 absorbed features were tested, which is a very small sample for drawing conclusions about steering effectiveness. The non-absorbed group (n=1014) showing no response could indicate a baseline methodology problem rather than confirming absorption is epistemic. The current text does not adequately address this alternative interpretation.
- **Fix**: Add a sentence noting that the small n for absorbed features (n=7) limits power, but the null result for non-absorbed features suggests a broader methodological issue with the steering approach on synthetic activations. This is addressed in discussion but should be flagged in experiments as well.

## Minor Issues

- **Line 52-53**: The absorption rate values are repeated in prose after already being presented in Table 1. Either remove the prose repetition or make it add new interpretation rather than restating numbers.

- **Line 62**: "d=8.94" uses one decimal but the actual value is 8.939 (should be "d=8.94" is acceptable rounding; no fix needed).

- **Table 1 header alignment**: The markdown table headers have inconsistent alignment. Standardize to left-aligned or consistent formatting throughout.

- **Line 39**: "Figure 1 presents absorption rates by condition. Table 1 summarizes the key statistics." - This is circular since Table 1 IS the absorption rates. Rephrase to "Table 1 and Table 2 present the quantitative results; Figure 1 provides visual comparison."

- **Section 6.5**: The heading "Discussion of Negative Results" preemptively discusses interpretation before the reader reaches the actual Discussion section. This is acceptable but consider whether this content belongs here or in Discussion.

## Visual Element Assessment

- [ ] **Figures NOT generated**: The `figures/` directory does not exist. All four figures are broken references.
- [x] All tables referenced appear before they are summarized in text
- [x] Table captions are self-explanatory
- [ ] **Text-heavy explanations**: The proportional variance subsection (6.2.3) could benefit from a figure showing the distribution, not just reporting means
- [x] Tables include standard deviations and effect sizes
- [ ] **No visual redundancy**: The section has good tabular presentation but could use more visual variety

## What Works Well

1. **Paragraph at lines 29-35**: Excellent technical explanation of why multi-child ablation solves the saturation problem. The mathematical notation is clear and the intuition is well-communicated.

2. **Section 6.5 "Discussion of Negative Results"**: The two-subsection structure for interpreting H2 and H3 failures is well-organized. The competitive exclusion discussion (lines 138-140) provides multiple plausible interpretations without overclaiming.

3. **Table 5 Summary (lines 148-154)**: The hypothesis summary table is exactly what reviewers need to quickly grasp the outcomes. Predictions, results, and key findings in one place is effective.
