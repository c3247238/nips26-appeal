# Writing Quality Review

## Summary

This paper investigates whether coefficient of variation (CV) predicts steering effectiveness in absorbed SAE features. The authors show that high-CV absorbed features produce 1.47x larger steering effects than low-CV features (0.525 vs 0.357 mean logit change, p < 0.01 with BH correction across all strengths), and that absorbed high-CV features achieve steering comparable to non-absorbed features (0.097 vs 0.102 logit change). They argue the "actionability paradox" reported by Basu et al. (2026) is domain-specific rather than universal, proposing CV > 1.0 as a practical screening threshold. Activation patching confirms genuine causal structure (67.3% mean parent recovery, all 9/9 words above 10% threshold). The paper is clearly written with good logical structure, though some sections contain confusing sentences and the Figure 5 reference needs correction.

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a standard logical flow: problem (actionability paradox) → research gap → approach (CV-based decomposition) → theory → experiments → discussion → conclusion. Each section builds on the previous one.

**Strengths**:
- Introduction clearly establishes the paradox and research gap before presenting the approach
- Theoretical Framework (Section 3) provides a plausible mechanistic account before experiments test it
- Discussion honestly reports H6 falsified and discusses limitations
- Conclusion effectively synthesizes contributions

**Issues**:
- Section 4.6 (Cross-Architecture Validation) states detailed integration remains future work — honest but weak for a paper claiming generalization
- Section 5.1 contains a fragmented sentence (lines 2-3): "Why do JumpReLU and ReLU SAEs produce projection absorption rates that differ by only 7.7 percentage points? The small absolute gap suggests that projection-based absorption captures a structural property..." This appears to be remnant text from an earlier version. It references "projection absorption rates" with no citation or context for the 7.7 figure.

### Notation & Terminology Consistency: 9/10

**Correct**:
- CV properly defined as σ/μ in Section 3.1
- A_j absorption detector formula given in Section 2.2
- R_parent defined in Section 3.3 before use in Section 4.1
- Greek letters used consistently: λ, χ, ν, τ
- "robust absorbed" vs "fragile absorbed" used consistently throughout
- "actionability paradox" used consistently for Basu et al.

**Violations**:
- Section 3.1 uses $CV_{absorbed} \approx 7.33$ and $CV_{non-absorbed} \approx 0.01$ — subscript forms slightly inconsistent with notation.md convention (which uses curly braces: CV_{absorbed})
- No other significant deviations found

### Claim-Evidence Integrity: 8/10

**Verified claims**:
1. "98.2% AUROC" — correctly attributed to Basu et al. (2026) ✓
2. Table 1 numbers match full_steering_cv.json exactly: High-CV +3 = 0.3079, +5 = 0.5222, +7 = 0.7453 ✓
3. "p < 0.01 with BH correction at all strengths" — verified ✓
4. "67.3% mean parent recovery, all 9/9 words above 10% threshold" — verified against pilot_summary.md ✓
5. "733x ratio (CV ≈ 7.33 vs 0.01)" — stated in theory and pilot data ✓
6. Non-absorbed baseline: 0.102 (SD: 0.078), absorbed high-CV: 0.097 — verified from Section 4.5 ✓
7. Effect ratio 1.47 — verified from aggregate data ✓

**Issues**:
- Section 4.5: "Non-absorbed features showed mean absolute steering effect of 0.102 (SD: 0.078), compared to absorbed high-CV features: 0.097." This comparison at face value seems straightforward, but the earlier Table 1 at +5 strength shows absorbed high-CV = 0.5222. The 0.097 in Section 4.5 appears to be from a different measurement context (possibly a different normalization or the raw mean before absolute value). The paper does not clarify this discrepancy. **Fix**: Add a clarifying note explaining this is a different metric/comparison than Table 1.
- No statistical test is reported for the non-absorbed vs absorbed high-CV comparison in Section 4.5. The claim "the difference is 0.0045, which is not practically significant" lacks a p-value.

### Visual Communication: 7/10

**Present**:
- Figure 1: Steering effect by CV group and strength — referenced in text before appearing ✓
- Figure 2: Activation patching — referenced before appearing ✓
- Figure 3: CV distribution — referenced in Section 3.1 ✓
- Figure 4: Decoder orthogonality — referenced and shows H6 falsification ✓
- Figure 5: Mechanism diagram — referenced as markdown (.md) file

**Critical Issue**:
- Figure 5 is referenced as `figures/fig5_mechanism_desc.md` — a markdown file, not a rendered PDF/PNG diagram. This will not compile in a PDF submission. The figure reference in Section 3.2 reads: `![Figure 5: Architecture diagram describing robust vs fragile absorption routing mechanism](figures/fig5_mechanism_desc.md)`. This syntax (markdown image) is appropriate for markdown but not for a paper intended for PDF submission.

**Missing visual**:
- No actual rendered figure exists for the mechanism diagram. The outline plans this as a "manual_diagram (tikz or similar)" but it has not been generated.

**Positive**:
- Table captions are self-explanatory ✓
- Tables are referenced before appearing ✓
- Figure captions in the figure list (Section 7) describe the intended content clearly

### Writing Quality: 7/10

**No banned patterns found** ✓:
- No "In recent years..." openings ✓
- No "Furthermore" or "Moreover" transitions ✓
- No vague "significantly improves" without numbers ✓
- Specific numbers used throughout ✓

**Confusing sentences**:

1. **Section 5.1, lines 2-3**: "Why do JumpReLU and ReLU SAEs produce projection absorption rates that differ by only 7.7 percentage points? The small absolute gap suggests that projection-based absorption captures a structural property..." — This is a sentence fragment. It appears to be remnant text from a different section or version. The 7.7 figure has no citation, no context, and no connection to the surrounding discussion about clinical domain features.

2. **Section 4.5 interpretation**: "The difference is 0.0045, which is not practically significant" — Without a p-value or confidence interval, this statistical judgement is unsupported. The paper should either provide the statistical test or soften to "numerically similar."

3. **Section 1.3**: "This is paradoxical because absorption is thought to degrade feature quality, yet absorbed features show dramatically higher activation variability than non-absorbed features" — The paradox is not clearly explained. The logic connecting "degrades quality" to "should have low variability" is implied but not explicit.

**Positives**:
- Leads with concrete results in abstract and throughout ✓
- Short, direct sentences in most experimental sections ✓
- Dose-response relationship clearly presented ✓

## Issues for the Editor

1. **[Critical] Figure 5 is a Markdown file, Not a Figure**: Section 3.2 references `figures/fig5_mechanism_desc.md` which will not render in a PDF paper. Convert to a proper rendered diagram (TikZ, matplotlib, etc.) or remove the figure reference and describe the mechanism in text only.

2. **[Major] Section 5.1 Fragmented Sentence**: Lines 2-3 "Why do JumpReLU and ReLU SAEs produce projection absorption rates that differ by only 7.7 percentage points? The small absolute gap suggests..." are confusing and unsupported. Either remove this passage or rewrite it with proper context and citation for the 7.7 figure. The surrounding text (about clinical domain features and CV distribution) makes sense without this fragment.

3. **[Major] Section 4.5 Comparison Context**: The Section 4.5 comparison (0.097 vs 0.102) uses different numbers than Table 1 (+5 strength shows 0.5222). Add a clarifying note: "Unlike the logit-change measurements in Table 1, these values reflect [describe the metric — e.g., normalized effect per prompt, or raw mean before absolute value]." Without this clarification, readers will be confused.

4. **[Minor] Section 4.5 Lacks Statistical Test**: The claim that 0.097 vs 0.102 difference is "not practically significant" should be supported by a statistical test (t-test or Mann-Whitney U). Report p-value or soften to "numerically similar."

## What Works Well

1. **Abstract quality**: Leads with the actionability paradox, presents the 1.47x key finding, and states the practical implication (CV > 1.0 predictor) in 6 well-structured sentences. Complete narrative arc.

2. **Hypothesis falsification handling (Sections 4.4 and 5.4)**: H6 correctly reported as NOT_SUPPORTED with full statistical detail (Pearson r = -0.136, p = 0.301; Spearman rho = -0.204, p = 0.117), honest mechanistic implications discussed. Good scientific practice.

3. **Main results table (Table 1)**: Properly formatted with mean, standard deviation, t-statistic, and BH-adjusted p-values across all three steering strengths. Reproducible and transparent.

4. **Activation patching validation (Table 2)**: All 9/9 words pass 10% threshold with mean 67.3% recovery — compelling causal validation. Table is clear and well-documented.

SCORE: 7
