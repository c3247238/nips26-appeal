# Critique: 2. Background and Related Work

## Summary Assessment

The section provides solid conceptual grounding for SAE architecture, feature interaction phenomena, and causal analysis methodology. The prose is clear and well-structured, with accurate citations. However, the section has three critical issues: (1) cross-references to non-existent section numbers (Section 7 for H4, Section 8 for H5), (2) an unrequired Subsection 2.4 that disrupts the outline's three-pillar structure, and (3) the H4 test is described as direct when it is actually uninformative due to an experiment design flaw.

## Score: 6/10

**Justification**: The writing quality and conceptual organization are strong. The section fulfills its outline promise to cover three conceptual pillars (SAEs, feature interaction phenomena, causal analysis). However, two critical cross-reference errors make the section actively misleading about where content lives in the paper, and the unrequired 2.4 subsection creates structural inconsistency with the outline.

## Critical Issues

### Issue 1: Wrong Section Numbers for H4 and H5 Cross-References

- **Location**: Section 2.2, paragraph 2 ("Section 8 (H5)"); Section 2.3, paragraph 3 ("Section 7 (H4)")
- **Quote**: "Section 8 (H5) investigates the relationship between dictionary size and absorption rates." and "Section 7 (H4) tests this directly..."
- **Problem**: Section 7 and Section 8 do not exist in the current paper structure. The experiments section is Section 5, with H4 as Section 5.3 and H5 as Section 5.4. The outline explicitly specifies "Sections 5.1-5.5 present results for H1-H5" and the intro's roadmap correctly says "Sections 5 through 8 present results for H1, H3, H4, and H5 respectively." A reader following these cross-references in the related work section would land in the wrong place.
- **Fix**: Change "Section 8 (H5)" to "Section 5.4 (H5)" and "Section 7 (H4)" to "Section 5.3 (H4)".

### Issue 2: Unrequired Subsection 2.4 Disrupts Outline Structure

- **Location**: Section 2.4 ("Summary of Positioning"), lines 27-28
- **Quote**: "This paper contributes a reproducible absorption metric and the first multi-hypothesis empirical characterization..."
- **Problem**: The outline specifies three subsections for Section 2 (2.1, 2.2, 2.3). The draft adds a 2.4 "Summary of Positioning" that was not requested in the outline. This subsection reads like an intro abstract rather than a related work summary, and it partially conflicts with the intro's contribution bullets.
- **Fix**: Remove Section 2.4 entirely. The positioning summary belongs in the intro (as contribution bullets), not as a subsection of related work. The final paragraph of Section 2.3 already provides a natural closing for the section.

### Issue 3: H4 Description Implies Clean Test When It Is Uninformative

- **Location**: Section 2.3, paragraph 3
- **Quote**: "Section 7 (H4) tests this directly by comparing faithfulness across raw residual patching, full SAE patching, and patching restricted to high-absorption or low-absorption latent subsets."
- **Problem**: The word "tests" implies the experiment provides a clean test of the hypothesis. However, the experiments section (5.3) explicitly labels H4 as "uninformative" because both absorption subsets yielded 0.000, making the comparison impossible. The experiment design flaw (comparing dictionary subsets, not full SAEs at layers with different absorption profiles) is acknowledged in the proposal. The related work framing should signal this limitation upfront rather than implying a clean test.
- **Fix**: Change "tests this directly" to "investigates this" and add a parenthetical: "Section 5.3 (H4) attempts to investigate this comparison, but the experiment design flaw (both absorption subsets yield 0.0 faithfulness) prevents any conclusion."

## Major Issues

### Issue 4: Section 2.4 Conflates Pending and Falsified Hypotheses

- **Location**: Section 2.4, paragraph 1
- **Quote**: "no prior work has provided... or an evaluation of whether absorption degrades downstream causal analyses."
- **Problem**: This sentence implies H4 was evaluated and found not to degrade analyses. But H4 was uninformative (both subsets yielded 0.000, making comparison impossible) — not tested and found negative. The section overstates the evidence.
- **Fix**: Change "or an evaluation of whether absorption degrades downstream causal analyses" to "or a systematic evaluation of whether absorption degrades downstream causal analyses (H4 was inconclusive; see Section 5.3)."

### Issue 5: H2 Status Mentioned Inconsistently

- **Location**: Section 2.4; intro.md paragraph 3
- **Problem**: The intro says H2 "was not tested" because "the experimental design needed revision." But the proposal.md explicitly states H2 is "CRITICAL PATH -- MUST RUN AT LAYER 4" and has been deferred for 11 iterations. Section 2.4 does not mention H2 at all.
- **Fix**: If keeping 2.4, add one sentence: "H2 (token-frequency correlation) and H6 (perfect-score latent investigation) remain pending future work." Or remove 2.4 per Issue 2.

## Minor Issues

- **Section 2.1, paragraph 2**: "concept localization (mapping features to their preferred tokens)" — "preferred tokens" is vague. Change to "mapping features to the tokens where they activate most strongly."
- **Section 2.2, paragraph 2**: "structural clustering in activation space" — consider "geometric clustering" or "clustering in activation space" (remove "structural" as jargon-heavy).
- **SAELens first use**: The glossary preference states "SAELens" on first use should be spelled out as "Sparse Autoencoder Lens (SAELens)." Section 2.1 introduces "SAELens (Bloomfield et al., 2024)" without spelling out the acronym. Fix: "Sparse Autoencoder Lens (SAELens, Bloomfield et al., 2024)" on first use.
- **Citation formatting**: "Sharkey et al." and "Templeton" appear without year citations, while other references use "Elhage et al., 2022" format. Fix: "Sharkey et al. (2023)" and "Templeton (2025)".
- **Section 2.3, paragraph 2**: "if absorption is rare, most latents are approximately independent" — the logical connection is not tight. Absorption being rare does not directly imply independence (absorbed features can be rare but highly correlated with their co-firers). Suggest: "Conversely, if absorption is rare and latents are largely independent, SAE-based circuit analysis may be more reliable than feared."

## Visual Element Assessment

- [x] No figures or tables required for this section (per outline)
- [x] No text-heavy sections that need visual support
- [x] Section is text-only, which is standard for Related Work

## What Works Well

**Paragraph 2 of Section 2.2 (absorption vs. superposition distinction)**: The distinction is drawn precisely. "Absorption is the more operationally severe failure mode: if feature f is fully absorbed by co-firers c in top5(f), then f carries no additional information beyond what is recoverable from those co-firers alone." This operationalizes the concept clearly.

**Section 2.3 causal analysis framing (paragraphs 1-2)**: The explanation of activation patching and the faithfulness metric is clear and well-motivated. "A central open problem in mechanistic interpretability is whether SAE latents are causally meaningful" correctly positions H4 as an open inquiry.

**Section 2.1 coverage breadth**: The background on SAEs as tools for circuit analysis, feature probing, and concept localization is appropriate for readers who may not be SAE experts.

**Citation coverage**: Bricken et al., Anthropic 2023, Elhage et al., Bloomfield et al., Sharkey et al., Templeton 2025, Wang et al., Meng et al. — all relevant and appropriately cited.